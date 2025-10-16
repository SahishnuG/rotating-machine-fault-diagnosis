#!/usr/bin/env python3
"""
prepare_and_train_lstm.py

Pipeline that:
- Reads sensor CSV folders: acoustic, vibration, current_temp
- Parses filenames of pattern: {load}Nm_{condition}_{severity}.csv  (severity optional)
- Processes CSVs chunkwise into labeled sliding windows (.npz files)
- Streams windows into a TensorFlow/Keras LSTM classifier/regressor and trains
- Designed to be memory-efficient: uses chunked reads and streaming generators

Usage:
    python3 prepare_and_train_lstm.py --base_dir ./ --out_dir ./processed --make_windows --train

Default folders (relative to --base_dir):
    ./acoustic_csv_data, ./vibration_csv_data, ./current_temp
"""

import argparse
import re
from pathlib import Path
import numpy as np
import pandas as pd
import os
import math
import json

# For model training
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout, Masking, Bidirectional
    from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
except Exception:
    tf = None

# ----------------- Utility functions -----------------
def parse_filename_meta(fname: str):
    """
    Parses filenames like '10Nm_inner_mild.csv' or '10Nm_normal.csv' into dict:
    {load_Nm: int, condition: str, severity: str or None}
    """
    name = Path(fname).stem
    # flexible pattern: digits + 'Nm' then _condition then optional _severity
    m = re.match(r'(\d+)Nm_(\w+)(?:_(\w+))?$', name)
    if not m:
        return {"load_Nm": None, "condition": name, "severity": None}
    load, cond, sev = m.groups()
    return {"load_Nm": int(load), "condition": cond.lower(), "severity": sev.lower() if sev else None}

def canonicalize_current_temp_columns(df: pd.DataFrame):
    """
    Convert weird LabVIEW channel headers into simple safe names.
    e.g. "/'Log'/'cDAQ.../ai0'" -> "ai0" or "ch0"
    """
    new_cols = []
    for i, c in enumerate(df.columns):
        s = str(c)
        # keep Time_s if already present
        if s.lower() == "time_s" or s.lower() == "time" or s.lower() == "time_s":
            new_cols.append("Time_s")
            continue
        m = re.search(r'ai(\d+)', s)
        if m:
            new_cols.append(f"ai{m.group(1)}")
        else:
            # fallback: short name
            new_cols.append(f"ch{i}")
    df.columns = new_cols
    return df

def ensure_time_column(df: pd.DataFrame):
    # if there's no Time_s, try to infer from index or add a sample index as Time_s
    if "Time_s" in df.columns:
        return df
    # else add a Time_s using row index (assume uniform sampling)
    df = df.copy()
    df.insert(0, "Time_s", np.arange(len(df)))
    return df

# ----------------- Processing & window creation -----------------
def process_and_save_windows(csv_path: Path, sensor_type: str, out_dir: Path,
                             chunksize: int = 2_000_000, resample=False, target_dt=None,
                             window_size: int = 2048, step: int = 512, save_format="npz"):
    """
    Read csv in chunks, canonicalize columns if needed, create sliding windows and save them.
    Each saved file will be: out_dir/{sensor_type}/{label}_w{window}_s{step}.npz
    The .npz will contain arrays: windows (N, window_size, n_features), metadata dict
    """
    meta = parse_filename_meta(csv_path.name)
    label = f"{meta.get('load_Nm') or 'NA'}Nm_{meta.get('condition')}_{meta.get('severity') or 'none'}"
    out_folder = out_dir / sensor_type
    out_folder.mkdir(parents=True, exist_ok=True)
    # read entire file in chunks but accumulate windows per file to keep windows contiguous
    # For simplicity and given potential large file sizes we will stream rows and build windows using a rolling buffer
    reader = pd.read_csv(csv_path, chunksize=chunksize, header=0)
    buffer = None
    windows_collected = []
    cols_used = None
    for chunk in reader:
        df = chunk.copy()
        # canonicalize current_temp columns
        if sensor_type == "current_temp":
            df = canonicalize_current_temp_columns(df)
        df = ensure_time_column(df)
        # drop any all-NaN cols
        df = df.dropna(axis=1, how='all')
        # choose feature columns (exclude Time_s)
        feature_cols = [c for c in df.columns if c != "Time_s"]
        cols_used = feature_cols if cols_used is None else cols_used
        arr = df[feature_cols].to_numpy(dtype=float)
        if buffer is None:
            buffer = arr
        else:
            # append new chunk to buffer
            buffer = np.vstack([buffer, arr])
        # create windows from buffer where possible (but keep overlap)
        start = 0
        max_start = buffer.shape[0] - window_size
        while max_start >= 0 and start <= max_start:
            w = buffer[start:start+window_size]
            windows_collected.append(w)
            start += step
            max_start = buffer.shape[0] - window_size
        # keep tail part of buffer for next chunk
        if buffer.shape[0] >= window_size:
            buffer = buffer[start:]
    # after reading all chunks, no more new data; windows collected ready
    windows = np.stack(windows_collected) if len(windows_collected) > 0 else np.zeros((0, window_size, len(cols_used)), dtype=float)
    out_name = out_folder / f"{label}_w{window_size}_s{step}.npz"
    meta_obj = {"source_file": csv_path.name, "label": label, "load_Nm": meta.get("load_Nm"),
                "condition": meta.get("condition"), "severity": meta.get("severity"),
                "sensor_type": sensor_type, "cols": cols_used}
    np.savez_compressed(out_name, windows=windows, meta=json.dumps(meta_obj))
    print(f"Saved {windows.shape[0]} windows to {out_name}")
    return out_name

def find_and_process_all(base_dir: Path, out_dir: Path, window_size=2048, step=512, chunksize=2_000_000):
    folders = {
        "acoustic": base_dir / "acoustic_csv_data",
        "vibration": base_dir / "vibration_csv_data",
        "current_temp": base_dir / "current_temp",
    }
    created_files = []
    for sensor, folder in folders.items():
        if not folder.exists():
            print(f"Folder {folder} not found; skipping {sensor}")
            continue
        for csv in sorted(folder.glob("*.csv")):
            print("Processing:", csv)
            npz = process_and_save_windows(csv, sensor, out_dir, chunksize=chunksize,
                                           window_size=window_size, step=step)
            created_files.append(npz)
    return created_files

# ----------------- Dataset generator & label handling -----------------
def load_npz_windows(npz_path: Path):
    """Load windows and metadata from an .npz created by process_and_save_windows"""
    with np.load(npz_path, allow_pickle=True) as d:
        windows = d["windows"]
        meta = json.loads(d["meta"].tolist())
    return windows, meta

def build_file_index(npz_paths):
    """
Build a pandas DataFrame indexing all windows files and labels.
Columns: path, sensor_type, load_Nm, condition, severity, n_windows, cols
"""
    rows = []
    for p in npz_paths:
        _, meta = load_npz_windows(Path(p))
        rows.append({
            "path": str(p),
            "sensor_type": meta["sensor_type"],
            "load_Nm": meta["load_Nm"],
            "condition": meta["condition"],
            "severity": meta["severity"],
            "n_windows": int(np.load(p)["windows"].shape[0]),
            "cols": meta["cols"]
        })
    return pd.DataFrame(rows)

class WindowGenerator:
    """
    Generator that yields batches (X, y) by streaming windows from .npz files.
    y_labels: user-defined mapping from ('condition','severity','load') to integer class label.
    If y is None, generator yields X only.
    """
    def __init__(self, file_index_df, class_map, batch_size=32, shuffle=True, mode="classification"):
        self.df = file_index_df.copy()
        self.class_map = class_map
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.mode = mode  # 'classification' or 'regression'
        # create list of (path, idx) pairs for windows
        self.samples = []
        for _, r in self.df.iterrows():
            p = Path(r["path"])
            n = int(np.load(p)["windows"].shape[0])
            for i in range(n):
                self.samples.append((p, i))
        self.on_epoch_end()

    def __len__(self):
        return math.ceil(len(self.samples) / self.batch_size)

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.samples)

    def __iter__(self):
        return self

    def __next__(self):
        if len(self.samples) == 0:
            self.on_epoch_end()
            raise StopIteration
        batch = self.samples[:self.batch_size]
        self.samples = self.samples[self.batch_size:]
        Xs = []
        ys = []
        for p, idx in batch:
            with np.load(p, allow_pickle=True) as d:
                w = d["windows"][idx].astype(np.float32)
                meta = json.loads(d["meta"].tolist())
            Xs.append(w)
            if self.mode == "classification":
                key = (meta["condition"], meta["severity"], meta["load_Nm"])
                y = self.class_map.get(f"{meta['condition']}_{meta.get('severity') or 'none'}_{meta.get('load_Nm')}", 0)
                ys.append(y)
            else:
                ys.append(meta.get("load_Nm") or 0.0)
        X = np.stack(Xs)
        y = np.array(ys)
        return X, y

# ----------------- Simple LSTM model builder -----------------
def build_lstm_model(input_shape, n_classes=None, embedding=False):
    if tf is None:
        raise RuntimeError("TensorFlow not available. Install tensorflow to train the model.")
    model = Sequential()
    # mask zero rows if present
    model.add(Masking(mask_value=0.0, input_shape=input_shape))
    model.add(Bidirectional(LSTM(128, return_sequences=False)))
    model.add(Dropout(0.3))
    if n_classes is None:
        # regression
        model.add(Dense(1, activation="linear"))
        model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    else:
        model.add(Dense(n_classes, activation="softmax"))
        model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model

# ----------------- Training orchestration -----------------
def train_from_npz_index(file_index_df, out_model_dir: Path, epochs=10, batch_size=32, val_split=0.2):
    # build class map from unique condition+severity+load combinations
    labels = []
    for _, r in file_index_df.iterrows():
        labels.append(f"{r['condition']}_{r['severity'] or 'none'}_{r['load_Nm']}")
    unique = sorted(list(set(labels)))
    class_map = {name: i for i, name in enumerate(unique)}
    print("Classes:", class_map)
    # create generator
    gen = WindowGenerator(file_index_df, class_map, batch_size=batch_size, shuffle=True, mode="classification")
    # compute input shape by loading first sample
    sample_p, _ = gen.samples[0]
    X0 = np.load(sample_p)["windows"][0]
    timesteps, features = X0.shape
    n_classes = len(unique)
    model = build_lstm_model((timesteps, features), n_classes=n_classes)
    out_model_dir.mkdir(parents=True, exist_ok=True)
    ckpt = out_model_dir / "best_model.h5"
    callbacks = [
        EarlyStopping(monitor="loss", patience=3, restore_best_weights=True),
        ModelCheckpoint(str(ckpt), save_best_only=True, monitor="loss")
    ]
    steps_per_epoch = len(gen)
    # Note: For simplicity we will not create a separate validation generator here.
    # Use model.fit with generator-like object by wrapping in tf.data if needed.
    print("Training model...")
    # convert generator to tf.data.Dataset for fit compatibility
    def gen_fn():
        while True:
            for Xb, yb in WindowGenerator(file_index_df, class_map, batch_size=batch_size, shuffle=True):
                yield Xb, yb
    dataset = tf.data.Dataset.from_generator(gen_fn, output_types=(tf.float32, tf.int32), output_shapes=([None, timesteps, features], [None,]))
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    # Fit for a small number of steps per epoch to prevent infinite generator issues
    model.fit(dataset, epochs=epochs, steps_per_epoch=steps_per_epoch, callbacks=callbacks)
    print("Training complete. Model saved to", ckpt)
    return model, class_map

# ----------------- CLI -----------------
def main(args):
    base_dir = Path(args.base_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    # 1) Process files -> create .npz windows for each file
    created = find_and_process_all(base_dir, out_dir, window_size=args.window_size, step=args.step, chunksize=args.chunksize)
    if len(created) == 0:
        print("No npz windows created; exiting.")
        return
    # 2) Build index DataFrame
    file_index = build_file_index(created)
    idx_csv = out_dir / "file_index.csv"
    file_index.to_csv(idx_csv, index=False)
    print("Wrote file index to", idx_csv)
    # 3) Train model if requested
    if args.train:
        if tf is None:
            print("TensorFlow not installed; cannot train. Install tensorflow and re-run with --train")
            return
        model_out = Path(args.model_out_dir)
        model, class_map = train_from_npz_index(file_index, model_out, epochs=args.epochs, batch_size=args.batch_size)
        # save class map for inference
        with open(model_out / "class_map.json", "w") as f:
            json.dump(class_map, f)
        print("Saved class map to", model_out / "class_map.json")

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Prepare windows from CSVs and train LSTM.")
    p.add_argument("--base_dir", type=str, default="./", help="Base dir containing folders acoustic_csv_data, vibration_csv_data, current_temp (default ./)")
    p.add_argument("--out_dir", type=str, default="./processed_data", help="Where to save .npz windows and index (default ./processed_data)")
    p.add_argument("--window_size", type=int, default=2048, help="Window size in samples (default 2048)")
    p.add_argument("--step", type=int, default=512, help="Step size between windows (default 512)")
    p.add_argument("--chunksize", type=int, default=2000000, help="CSV read chunksize (default 2,000,000)")
    p.add_argument("--make_windows", action="store_true", help="Create windows (default False)")
    p.add_argument("--train", action="store_true", help="Train LSTM after creating windows (default False)")
    p.add_argument("--epochs", type=int, default=10, help="Training epochs (default 10)")
    p.add_argument("--batch_size", type=int, default=32, help="Training batch size (default 32)")
    p.add_argument("--model_out_dir", type=str, default="./model_out", help="Where to save trained model (default ./model_out)")
    args = p.parse_args()

    # If user asked to make windows or train, run main; otherwise just print config
    print("Configuration:")
    for k, v in vars(args).items():
        print(f"{k}: {v}")
    if args.make_windows or args.train:
        main(args)
    else:
        print("No action requested. Use --make_windows to create windows and --train to train the model.")