
#!/usr/bin/env python3
"""
prepare_timeseries.py

Memory-efficient processing pipeline for large time-series CSV folders (acoustic, current_temp, vibration).
- Reads CSVs in chunks
- Normalizes / resamples to a stable sampling interval (optional)
- Labels data by filename or folder-based label
- Stores processed data partitioned by sensor-type and label as Parquet (preferred) or compressed CSV fallback
- Creates sliding windows suitable for time-series forecasting and saves them as .npz arrays or HDF5

Usage example:
    python3 prepare_timeseries.py --acoustic_dir ./acoustic_csv_data \
                                  --current_temp_dir ./current_temp \
                                  --vibration_dir ./vibration_csv_data \
                                  --out_dir ./processed_data \
                                  --window_size 2048 --step 512

Notes:
- This script prefers pyarrow and fastparquet for Parquet I/O. If unavailable it falls back to gzipped CSVs.
- Works chunkwise to handle very large files (up to 10M rows or more).
"""

import argparse
import os
from pathlib import Path
import re
import numpy as np
import pandas as pd

# --- Configuration defaults ---
DEFAULT_CHUNK_ROWS = 2000000  # tune depending on memory
DEFAULT_OUT_FORMAT = "parquet"  # "parquet" or "csv_gz"
DEFAULT_DATETIME = False  # Time_s is numeric seconds; keep as float for speed unless conversion requested

# --- Utilities ---
def safe_mkdir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def infer_label_from_filename(fp: Path):
    # try to infer error type from filename using a simple pattern:
    # e.g. normal.csv, inner_race.csv, outer_race.csv, misalignment_1.csv, etc.
    name = fp.stem.lower()
    # common tokens
    tokens = re.split(r'[_\-\s\.]+', name)
    # if filename contains 'normal' return normal else join tokens except digits
    if "normal" in tokens:
        return "normal"
    # drop pure numeric tokens
    tokens = [t for t in tokens if not t.isdigit() and len(t) > 0]
    return "_".join(tokens[:3]) or name

def choose_parquet_engine():
    try:
        import pyarrow  # noqa: F401
        return "pyarrow"
    except Exception:
        try:
            import fastparquet  # noqa: F401
            return "fastparquet"
        except Exception:
            return None

# --- Core processing functions ---
def read_csv_in_chunks(fp, usecols=None, parse_dates=None, chunksize=DEFAULT_CHUNK_ROWS):
    """
    Generator yielding chunks of CSV as DataFrame.
    """
    for chunk in pd.read_csv(fp, usecols=usecols, parse_dates=parse_dates, chunksize=chunksize):
        yield chunk

def canonicalize_current_temp_columns(df):
    """
    For current_temp files, columns may be named with slashes.
    Map them to readable names like current_0, temp_0, etc., preserving order.
    """
    new_cols = []
    for i, c in enumerate(df.columns):
        # Skip 'Time_s' mapping handled earlier
        if c == "Time_s":
            new_cols.append(c)
            continue
        # create short safe name
        safe = f"ch_{i-1}"
        # try to extract ai\d+ pattern
        m = re.search(r'ai(\d+)', c)
        if m:
            safe = f"ai{m.group(1)}"
        new_cols.append(safe)
    df.columns = new_cols
    return df

def resample_uniform_time(df, time_col='Time_s', target_dt=None, method='interpolate'):
    """
    Resample/interpolate a numeric-time indexed DataFrame to a uniform grid.
    - time_col: name of time column in seconds (float) or datetime
    - target_dt: desired spacing in seconds (e.g. 0.001 for 1 kHz). If None, uses median dt of df.
    - method: 'interpolate' or 'downsample' (nearest)
    Returns resampled DataFrame with Time_s as the first column (numeric seconds).
    """
    t = df[time_col].astype(float).to_numpy()
    if len(t) < 2:
        return df
    dt = np.median(np.diff(t))
    if target_dt is None:
        target_dt = dt
    t_uniform = np.arange(t[0], t[-1], target_dt)
    out = {"Time_s": t_uniform}
    for col in df.columns:
        if col == time_col:
            continue
        y = df[col].to_numpy(dtype=float)
        # linear interpolation; fill NaNs by nearest
        y_interp = np.interp(t_uniform, t, y, left=np.nan, right=np.nan)
        out[col] = y_interp
    return pd.DataFrame(out)

def append_df_to_parquet(df: pd.DataFrame, out_path: Path, engine=None, partition_cols=None):
    """
    Append DataFrame to a Parquet file (or create). If engine not available, fallback to gzipped CSV append.
    """
    safe_mkdir(out_path.parent)
    if engine:
        # use partitioning by label if requested
        if partition_cols:
            # pandas to_parquet with partition_cols requires pyarrow
            df.to_parquet(out_path, engine=engine, partition_cols=partition_cols, index=False)
        else:
            # append by reading existing file and concatenating (parquet append not always supported)
            if out_path.exists():
                # read small metadata then append using fastparquet or pyarrow writer
                existing = pd.read_parquet(out_path, engine=engine)
                combined = pd.concat([existing, df], ignore_index=True)
                combined.to_parquet(out_path, engine=engine, index=False)
            else:
                df.to_parquet(out_path, engine=engine, index=False)
    else:
        # fallback: gzipped CSV append (may be slower)
        mode = "at" if out_path.exists() else "wt"
        df.to_csv(out_path, mode=mode, index=False, compression="gzip", header=not out_path.exists())

def process_file_chunked(fp: Path, out_dir: Path, sensor_type: str,
                         label: str=None, chunksize=DEFAULT_CHUNK_ROWS,
                         resample=False, target_dt=None, parquet_engine=None):
    """
    Process a single CSV file in chunks, optionally resample each chunk and append to out file.
    The final stored file will be: out_dir/{sensor_type}/{label}.parquet (single large file per label+sensor_type)
    """
    print(f"Processing {fp} ...")
    if label is None:
        label = infer_label_from_filename(fp)
    out_dir = Path(out_dir) / sensor_type
    safe_mkdir(out_dir)
    out_path = out_dir / f"{label}.parquet" if parquet_engine else out_dir / f"{label}.csv.gz"

    # read header first to detect columns and decide parsing
    # We'll use pandas chunk reader
    reader = pd.read_csv(fp, nrows=5)  # small peek
    cols = list(reader.columns)
    # decide time column name; prefer 'Time_s' if present
    if "Time_s" in cols:
        time_col = "Time_s"
        usecols = None  # read all
    else:
        # attempt to find a time-like column
        candidates = [c for c in cols if 'time' in c.lower() or 't'==c.lower()]
        time_col = candidates[0] if candidates else cols[0]
        usecols = None

    # For current_temp, canonicalize column names if needed
    def process_chunk_df(df):
        # ensure Time_s column exists and is float seconds
        if time_col not in df.columns:
            df = df.rename(columns={df.columns[0]: 'Time_s'})
        if sensor_type == "current_temp":
            df = canonicalize_current_temp_columns(df)
        df = df.rename(columns={time_col: "Time_s"})
        # drop rows with NaN time
        df = df[pd.notnull(df["Time_s"])]
        # optionally resample to uniform grid (per chunk)
        if resample:
            df = resample_uniform_time(df, time_col='Time_s', target_dt=target_dt)
        # attach label column for easier downstream grouping
        df["label"] = label
        return df

    # iterate in chunks
    chunk_iter = pd.read_csv(fp, usecols=usecols, chunksize=chunksize)
    first = True
    for chunk in chunk_iter:
        df_chunk = process_chunk_df(chunk)
        # write/append
        append_df_to_parquet(df_chunk, out_path, engine=parquet_engine)
        first = False
    print(f"Saved processed data to {out_path}")

def create_sliding_windows_for_label(processed_file: Path, out_dir: Path, sensor_type: str,
                                     window_size: int = 2048, step: int = 512,
                                     cols_to_use: list = None, save_format="npz"):
    """
    Read processed Parquet/CSV and create sliding windows for forecasting.
    Each window will have shape (window_size, n_features).
    Saves windows in compressed .npz files per label and sensor_type to keep sizes manageable.
    """
    print(f"Creating sliding windows from {processed_file} ...")
    if processed_file.suffix == ".parquet":
        df = pd.read_parquet(processed_file)
    else:
        df = pd.read_csv(processed_file, compression="gzip")
    # ensure sorted by time
    df = df.sort_values("Time_s")
    if cols_to_use is None:
        cols_to_use = [c for c in df.columns if c not in ("Time_s", "label")]
    data = df[cols_to_use].to_numpy(dtype=float)
    n, d = data.shape
    windows = []
    starts = np.arange(0, n - window_size + 1, step, dtype=int)
    for s in starts:
        w = data[s:s + window_size]
        if w.shape[0] == window_size:
            windows.append(w)
    windows = np.stack(windows) if windows else np.zeros((0, window_size, d), dtype=float)
    # save
    safe_mkdir(Path(out_dir) / sensor_type)
    label = processed_file.stem
    out_path = Path(out_dir) / sensor_type / f"{label}_w{window_size}_s{step}.{save_format}"
    if save_format == "npz":
        np.savez_compressed(out_path, windows=windows, cols=cols_to_use, time_start=df["Time_s"].iloc[0] if len(df)>0 else None)
    else:
        # fallback to numpy save
        np.save(out_path.with_suffix(".npy"), windows)
    print(f"Saved {windows.shape[0]} windows to {out_path}")

# --- CLI and orchestration ---
def main(args):
    acoustic_dir = Path(args.acoustic_dir) if args.acoustic_dir else None
    current_temp_dir = Path(args.current_temp_dir) if args.current_temp_dir else None
    vibration_dir = Path(args.vibration_dir) if args.vibration_dir else None
    out_dir = Path(args.out_dir)
    safe_mkdir(out_dir)

    parquet_engine = choose_parquet_engine() if args.out_format == "parquet" else None
    if args.out_format == "parquet" and parquet_engine is None:
        print("Warning: parquet selected but no engine found (pyarrow/fastparquet). Falling back to gzipped CSV output.")
    # process each folder
    folder_map = [
        ("acoustic", acoustic_dir),
        #("current_temp", current_temp_dir),
        ("vibration", vibration_dir),
    ]
    for sensor_type, folder in folder_map:
        if not folder:
            continue
        if not folder.exists():
            print(f"Warning: folder {folder} does not exist. Skipping.")
            continue
        for fp in sorted(folder.glob("*.csv")):
            lbl = infer_label_from_filename(fp)
            process_file_chunked(fp, out_dir=out_dir, sensor_type=sensor_type,
                                 label=lbl, chunksize=args.chunksize,
                                 resample=args.resample, target_dt=args.target_dt,
                                 parquet_engine=parquet_engine)
            # after processing file into a per-label file, optionally create windows immediately
            if args.make_windows:
                processed_path = Path(out_dir) / sensor_type / (lbl + (".parquet" if parquet_engine else ".csv.gz"))
                create_sliding_windows_for_label(processed_path, out_dir=args.out_dir_windows or out_dir,
                                                 sensor_type=sensor_type,
                                                 window_size=args.window_size, step=args.step,
                                                 save_format=args.window_format)

if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Process large time-series CSV folders into labeled Parquet and sliding windows."
    )
    p.add_argument("--acoustic_dir", type=str, default="./acoustic_csv_data",
                   help="Path to acoustic CSV folder (default: ./acoustic_csv_data).")
    p.add_argument("--current_temp_dir", type=str, default="./current_temp",
                   help="Path to current/temp CSV folder (default: ./current_temp).")
    p.add_argument("--vibration_dir", type=str, default="./vibration_csv_data",
                   help="Path to vibration CSV folder (default: ./vibration_csv_data).")
    p.add_argument("--out_dir", type=str, default="./processed_data",
                   help="Output base directory for processed data (default: ./processed_data).")
    p.add_argument("--out_format", choices=["parquet", "csv_gz"], default="parquet",
                   help="Output format (default: parquet).")
    p.add_argument("--chunksize", type=int, default=2_000_000,
                   help="Number of rows per chunk read from CSV (default: 2,000,000).")
    p.add_argument("--resample", action="store_true",
                   help="Whether to resample each file to a uniform time grid (default: False).")
    p.add_argument("--target_dt", type=float, default=None,
                   help="Target dt (seconds) for resampling (default: None → median dt per file).")
    p.add_argument("--make_windows", action="store_true",
                   help="Whether to create sliding windows after processing files (default: False).")
    p.add_argument("--out_dir_windows", type=str, default=None,
                   help="Where to save generated windows. Defaults to --out_dir if omitted.")
    p.add_argument("--window_size", type=int, default=2048,
                   help="Window length (samples) for sliding windows (default: 2048).")
    p.add_argument("--step", type=int, default=512,
                   help="Step between window starts (default: 512).")
    p.add_argument("--window_format", choices=['npz', 'npy'], default='npz',
                   help="Save format for windows (default: npz).")

    args = p.parse_args()

    # Print configuration summary
    print("\n=== Configuration Summary ===")
    for k, v in vars(args).items():
        print(f"{k}: {v}")
    print("==============================\n")

    main(args)

