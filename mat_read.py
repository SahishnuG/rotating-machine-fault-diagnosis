#!/usr/bin/env python3
"""
mat_to_csv_acoustic_vibration.py

Convert Test.Lab / MATLAB .mat files containing a `Signal` struct to CSV.

- Acoustic output CSV columns: Time_s, values
- Vibration output CSV columns: Time_s, x_direction_housing_A, y_direction_housing_A, x_direction_housing_B, y_direction_housing_B

Usage:
    python mat_to_csv_acoustic_vibration.py --dataset acoustic --base_dir /path/to/mat_acoustic --out_dir /path/to/csv_acoustic
    python mat_to_csv_acoustic_vibration.py --dataset vibration --base_dir /path/to/mat_vibration --out_dir /path/to/csv_vibration

Dependencies:
    pip install numpy pandas scipy
"""

from pathlib import Path
import argparse
import traceback
import numpy as np
import pandas as pd

# SciPy loadmat
try:
    from scipy.io import loadmat
except Exception as e:
    loadmat = None
    print("ERROR: scipy is required to read .mat files. Install with: pip install scipy")

# ----------------- Helpers -----------------
def safe_getattr(obj, name):
    try:
        return getattr(obj, name)
    except Exception:
        return None

def is_numeric_array(x):
    try:
        a = np.asarray(x)
        return np.issubdtype(a.dtype, np.number) and a.size > 0
    except Exception:
        return False

# ----------------- Time extraction -----------------
def mat_time_from_x_values(x_struct):
    """
    Attempt to build Time_s vector from x_struct.
    Preferred: use start_value + increment * arange(number_of_values)
    Fallback: if x_struct is a numeric array, return it; else None.
    """
    if x_struct is None:
        return None
    try:
        start = safe_getattr(x_struct, 'start_value')
        incr = safe_getattr(x_struct, 'increment')
        n = safe_getattr(x_struct, 'number_of_values')
        if start is not None and incr is not None and n is not None:
            start_v = float(np.asarray(start).flatten()[0])
            inc_v = float(np.asarray(incr).flatten()[0])
            n_v = int(np.asarray(n).flatten()[0])
            return start_v + inc_v * np.arange(n_v)
    except Exception:
        pass
    # fallback: numeric array
    try:
        arr = np.asarray(x_struct).flatten()
        if np.issubdtype(arr.dtype, np.number) and arr.size > 0:
            return arr.astype(float)
    except Exception:
        pass
    return None

# ----------------- Unit transform -----------------
def apply_unit_transform(arr, quantity_struct):
    """
    If quantity_struct has unit_transformation with 'factor' and/or 'offset',
    apply them: arr' = arr * factor + offset.
    If not present, return arr unchanged.
    """
    if quantity_struct is None:
        return arr
    try:
        ut = getattr(quantity_struct, 'unit_transformation', None) or getattr(quantity_struct, 'unit_transformation_', None)
        if ut is None:
            return arr
        # read factor & offset if present
        factor = None
        offset = None
        try:
            factor = float(np.asarray(getattr(ut, 'factor')).flatten()[0])
        except Exception:
            factor = None
        try:
            offset = float(np.asarray(getattr(ut, 'offset')).flatten()[0])
        except Exception:
            offset = None
        out = arr.astype(float).copy()
        if factor is not None:
            out = out * factor
        if offset is not None:
            out = out + offset
        return out
    except Exception:
        return arr

# ----------------- Channel extraction (handles 2D values arrays) -----------------
def extract_channels_from_y_enhanced(y_struct):
    """
    Return list of (label, ndarray, quantity_struct_or_None)
    - Handles y_struct.values being NxC numeric -> split into C channels
    - Handles single-channel y_struct.values (1D)
    - Handles quantity.quantity_terms -> multiple separate channels
    """
    out = []
    if y_struct is None:
        return out

    # 1) direct 'values' field (common in your samples)
    vals = None
    try:
        vals = getattr(y_struct, 'values', None) or getattr(y_struct, 'values_', None)
    except Exception:
        vals = None

    if vals is not None:
        arr = np.asarray(vals)
        if arr.ndim == 2:
            # split columns
            ncols = arr.shape[1]
            for i in range(ncols):
                label = f"ch{i}"
                channel_arr = arr[:, i].astype(float)
                # pass quantity (if exists) for potential unit transform
                q = getattr(y_struct, 'quantity', None)
                out.append((label, channel_arr, q))
            return out
        elif arr.ndim == 1:
            out.append(("y", arr.astype(float), getattr(y_struct, 'quantity', None)))
            return out

    # 2) quantity.quantity_terms
    try:
        q = getattr(y_struct, 'quantity', None) or getattr(y_struct, 'quantity_', None)
        if q is not None:
            qterms = getattr(q, 'quantity_terms', None) or getattr(q, 'quantity_terms_', None)
            if qterms is not None:
                qlist = np.atleast_1d(qterms)
                for i, term in enumerate(qlist):
                    # label
                    lbl = None
                    for label_field in ('label', 'name', 'long_name', 'info'):
                        v = getattr(term, label_field, None)
                        if v is not None:
                            try:
                                lbl = str(np.asarray(v).tolist())
                            except Exception:
                                lbl = str(v)
                            break
                    # numeric arrays
                    arr_cand = None
                    for df in ('data', 'values', 'value', 'y', 'samples'):
                        cand = getattr(term, df, None)
                        if cand is not None and is_numeric_array(cand):
                            arr_cand = np.asarray(cand).flatten().astype(float)
                            break
                    if arr_cand is None and is_numeric_array(term):
                        arr_cand = np.asarray(term).flatten().astype(float)
                    if arr_cand is not None:
                        if not lbl:
                            lbl = f"ch{i}"
                        out.append((lbl, arr_cand, getattr(term, 'quantity', None) or q))
                if len(out) > 0:
                    return out
    except Exception:
        pass

    # 3) fallback: scan attributes for numeric arrays
    try:
        for name in dir(y_struct):
            if name.startswith('_'):
                continue
            try:
                v = getattr(y_struct, name)
            except Exception:
                continue
            if is_numeric_array(v):
                arr = np.asarray(v).flatten().astype(float)
                out.append((name, arr, None))
    except Exception:
        pass

    return out

# ----------------- Converters -----------------
def convert_vibration_mat_to_csv(mat_path: Path, out_csv_path: Path):
    """Convert vibration .mat -> CSV with 4 vibration channels + Time_s"""
    if loadmat is None:
        raise RuntimeError("scipy is required to read .mat files. Install with: pip install scipy")

    data = loadmat(str(mat_path), squeeze_me=True, struct_as_record=False)
    if 'Signal' not in data:
        raise KeyError(f"No 'Signal' found in {mat_path}")

    S = data['Signal']
    T = mat_time_from_x_values(safe_getattr(S, 'x_values'))
    channels = extract_channels_from_y_enhanced(safe_getattr(S, 'y_values'))

    if len(channels) == 0:
        raise RuntimeError(f"No channel data found in y_values for {mat_path}")

    # If channels come from a Nx4 values array we'll have 4 entries; otherwise try to pick/Pad to 4
    if len(channels) >= 4:
        chosen = channels[:4]
    else:
        chosen = channels[:]
        # pad using zeros of length of first channel (if exists)
        ref_len = len(chosen[0][1]) if chosen else 1
        for i in range(4 - len(chosen)):
            chosen.append((f'pad_ch{i}', np.zeros(ref_len, dtype=float), None))

    # harmonize lengths
    lengths = [len(arr) for _, arr, _ in chosen]
    if T is not None:
        lengths.append(len(T))
    minlen = min(lengths) if lengths else 0
    if minlen == 0:
        raise RuntimeError(f"Zero-length data encountered in {mat_path}")

    if T is None:
        T = np.arange(minlen)
    T = T[:minlen]

    # apply unit transforms and build dataframe
    default_names = [
        'x_direction_housing_A', 'y_direction_housing_A',
        'x_direction_housing_B', 'y_direction_housing_B'
    ]
    df = pd.DataFrame({'Time_s': T})
    for i, (lbl, arr, q) in enumerate(chosen[:4]):
        arr_trim = arr[:minlen]
        arr_unit = apply_unit_transform(arr_trim, q)
        df[default_names[i]] = arr_unit

    out_csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv_path, index=False)
    return out_csv_path

def convert_acoustic_mat_to_csv(mat_path: Path, out_csv_path: Path):
    """Convert acoustic .mat -> CSV with Time_s and values (Pa)"""
    if loadmat is None:
        raise RuntimeError("scipy is required to read .mat files. Install with: pip install scipy")

    data = loadmat(str(mat_path), squeeze_me=True, struct_as_record=False)
    if 'Signal' not in data:
        raise KeyError(f"No 'Signal' found in {mat_path}")

    S = data['Signal']
    T = mat_time_from_x_values(safe_getattr(S, 'x_values'))
    channels = extract_channels_from_y_enhanced(safe_getattr(S, 'y_values'))

    if len(channels) == 0:
        raise RuntimeError(f"No acoustic channel found in {mat_path}")

    # pick the first channel
    lbl, arr, q = channels[0]
    n = len(arr)
    if T is None:
        T = np.arange(n)
    minlen = min(len(T), n)
    arr_trim = arr[:minlen]
    arr_unit = apply_unit_transform(arr_trim, q)
    df = pd.DataFrame({'Time_s': T[:minlen], 'values': arr_unit})
    out_csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv_path, index=False)
    return out_csv_path

# ----------------- Directory processing -----------------
def process_directory(dataset_type, base_dir, out_dir, pattern="*.mat"):
    base = Path(base_dir)
    out = Path(out_dir)
    if not base.exists():
        print(f"[ERROR] base_dir does not exist: {base}")
        return
    files = sorted(base.glob(pattern))
    if len(files) == 0:
        print(f"[WARN] No .mat files found in {base}")
        return

    print(f"Found {len(files)} files in {base}. Converting dataset='{dataset_type}' -> {out}")
    successes = []
    failures = []
    for f in files:
        try:
            out_file = out / (f.stem + ".csv")
            if dataset_type == 'vibration':
                convert_vibration_mat_to_csv(f, out_file)
            elif dataset_type == 'acoustic':
                convert_acoustic_mat_to_csv(f, out_file)
            else:
                raise RuntimeError("Unsupported dataset type: " + str(dataset_type))
            successes.append(f)
            print(f"[OK] {f.name} -> {out_file.name}")
        except Exception as e:
            failures.append((f, str(e)))
            print(f"[FAIL] {f.name}: {e}")
            traceback.print_exc(limit=1)
    print("----- Summary -----")
    print(f"Successes: {len(successes)}, Failures: {len(failures)}")
    if failures:
        print("Sample failures:")
        for ff, reason in failures[:10]:
            print(f" - {ff.name}: {reason}")

# ----------------- CLI -----------------
def main():
    """
    Automatically process both acoustic and vibration datasets
    without requiring command-line arguments.
    """

    # -------- Default directories --------
    base_root = Path.cwd()  # current working directory
    datasets = {
        "acoustic": {
            "base_dir": base_root / "acoustic",
            "out_dir": base_root / "csv_acoustic",
            "pattern": "*.mat",
        },
        "vibration": {
            "base_dir": base_root / "vibration",
            "out_dir": base_root / "csv_vibration",
            "pattern": "*.mat",
        },
    }

    print("Starting automatic conversion of acoustic and vibration datasets...\n")

    for dataset_type, cfg in datasets.items():
        base_dir = cfg["base_dir"]
        out_dir = cfg["out_dir"]
        pattern = cfg["pattern"]

        print(f"\n==== Processing {dataset_type.upper()} dataset ====")
        process_directory(dataset_type, base_dir, out_dir, pattern=pattern)

    print("\n✅ All conversions complete.")

if __name__ == "__main__": main()