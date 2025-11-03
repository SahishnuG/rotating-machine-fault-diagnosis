#!/usr/bin/env python3
"""
make_scalograms_fcwt_and_train_vibration.py

- Uses fCWT (Fast CWT) if available to compute Morlet scalograms (falls back to pywt.cwt).
- Processes ONLY vibration CSVs; by default only '0Nm_Unbalance_1169mg.csv'.
- Saves outputs in SAME format as before:
    scalograms/vibration/<base>__<channel>__win00000.npy
    scalograms/vibration/<base>__<channel>__win00000.npy.meta.json
- Then tries to train simple classifiers on the produced scalograms.
  If only one class is present, it will print a message and skip training.

Install (optional):
    pip install fcwt   # to enable Fast CWT
"""

import os, glob, json
from pathlib import Path
from typing import Tuple, List
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# signal processing
import pywt
from scipy import signal

# ---- try to import fCWT ----
_fcwt_ok = False
try:
    import fcwt  # PyPI: fcwt (Fast Continuous Wavelet Transform)
    _fcwt_ok = True
except Exception:
    _fcwt_ok = False

# ---- tiny training stack (only if >=2 classes are found) ----
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

# ----------------- CONFIG -----------------
VIBRATION_FOLDER = "vibration"
OUT_BASE = "scalograms"
OUT_PNG_SUBFOLDER = None  # set to "pngs" to also save PNGs
os.makedirs(OUT_BASE, exist_ok=True)

# Only process these vibration files (base filename). Keep list empty to process all vibration CSVs.
VIBRATION_FILENAME_FILTER = ["0Nm_Unbalance_1169mg.csv"]

# windowing
WINDOW_SEC = 0.5
HOP_SEC = 0.25
MAX_WINDOWS_PER_FILE = 2000

# CWT params (Morlet)
WAVELET = "morl"
N_SCALES = 128
FREQ_MIN = 20.0
FREQ_MAX_RATIO = 0.45     # cap at 0.45 * fs
FLOAT_DTYPE = np.float32
VERBOSE = True

TIME_COL_KEYWORDS = ("time", "stamp")

# training (simple baseline if ≥2 classes exist)
IMG_H, IMG_W = 128, 128
TEST_SIZE = 0.2
RANDOM_STATE = 42
USE_PCA = True
PCA_DIM = 128

# ----------------- HELPERS -----------------
def find_time_column(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    low = [c.lower() for c in cols]
    for kw in TIME_COL_KEYWORDS:
        for i, c in enumerate(low):
            if kw in c:
                return cols[i]
    for c in cols:
        if np.issubdtype(df[c].dtype, np.number):
            return c
    return cols[0]

def estimate_fs_from_times(times: np.ndarray) -> float:
    diffs = np.diff(times)
    diffs_pos = diffs[diffs > 0]
    if diffs_pos.size == 0:
        raise ValueError("Cannot estimate sampling rate — no positive time differences.")
    dt = float(np.median(diffs_pos))
    return 1.0 / dt

def prepare_freqs(fs: float, n_scales: int, fmin: float, fmax_ratio: float):
    fmax = min(fmax_ratio * fs, fs / 2.0)
    if fmax <= fmin:
        fmin = max(0.5, fmax * 0.01)
    freqs = np.logspace(np.log10(fmin), np.log10(fmax), num=n_scales)
    # for meta completeness, also compute scales for Morlet:
    fc = pywt.central_frequency(WAVELET)
    dt = 1.0 / fs
    scales = fc / (freqs * dt)
    return freqs.astype(np.float32), scales.astype(np.float32)

def clean_and_interpolate(arr):
    arr = np.array(arr, dtype=float, copy=True)
    arr[~np.isfinite(arr)] = np.nan
    if np.isnan(arr).all():
        return np.zeros_like(arr)
    n = len(arr)
    inds = np.arange(n)
    good = ~np.isnan(arr)
    arr_interp = np.interp(inds, inds[good], arr[good])
    return arr_interp

def save_scalogram_array(out_path, power, freqs, times_window, meta):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.save(out_path, power.astype(FLOAT_DTYPE), allow_pickle=False)
    with open(out_path + ".meta.json", "w") as fh:
        json.dump(meta, fh)

def save_scalogram_png(png_path, power, freqs, times_window, vmin=None, vmax=None, cmap="viridis"):
    os.makedirs(os.path.dirname(png_path), exist_ok=True)
    plt.figure(figsize=(4, 3), dpi=150)
    extent = [times_window[0], times_window[-1], freqs[0], freqs[-1]]
    if vmin is None or vmax is None:
        vmin, vmax = power.min(), power.max()
        if vmin == vmax:
            vmax = vmin + 1e-6
    plt.imshow(power, aspect="auto", origin="lower", extent=extent, vmin=vmin, vmax=vmax, cmap=cmap)
    plt.yscale("log")
    plt.ylabel("Frequency (Hz)")
    plt.xlabel("Time (s)")
    plt.tight_layout()
    plt.savefig(png_path, bbox_inches="tight")
    plt.close()

# ----------------- fCWT / CWT COMPUTE -----------------
def compute_power_fcwt(sig: np.ndarray, fs: float, freqs: np.ndarray) -> np.ndarray:
    """
    Try to compute CWT power via fCWT. If fcwt is not available, fall back to pywt.
    Returns |coef| with shape (len(freqs), len(sig)).
    """
    # Prefer fCWT
    if _fcwt_ok:
        # fCWT expects float32
        x = np.asarray(sig, dtype=np.float32)
        f = np.asarray(freqs, dtype=np.float32)
        # Use fcwt Morlet defaults; some builds expose a simple cwt function:
        #   fcwt.cwt(signal, fs, freqs, wavelet='morlet', sigma=6.0)
        # If your fcwt has a Wavelet object API, this call may differ slightly.
        try:
            coef = fcwt.cwt(x, fs, f, wavelet='morlet', sigma=6.0)
            return np.abs(coef).astype(np.float32)
        except Exception as e:
            print(f"[WARN] fcwt.cwt call failed ({e}); falling back to pywt.cwt.")

    # Fall back to pywt (keeps identical output format)
    try:
        # build scales that correspond to requested freqs (meta will still include freqs)
        fc = pywt.central_frequency('morl')
        dt = 1.0 / fs
        scales = fc / (freqs * dt)
        coef, _ = pywt.cwt(sig.astype(np.float32), scales, 'morl', sampling_period=1.0/fs)
        return np.abs(coef).astype(np.float32)
    except Exception as e:
        raise RuntimeError(f"CWT failed for this window: {e}")

# ----------------- SCALOGRAM GENERATION (VIBRATION ONLY) -----------------
VIBRATION_CHANNELS = [
    "x_direction_housing_A",
    "y_direction_housing_A",
    "x_direction_housing_B",
    "y_direction_housing_B",
]

def process_csv_vibration(csv_path: str, channel_names: List[str], out_dir: str):
    if VERBOSE:
        print(f"[PROCESS] vibration: {csv_path}")

    try:
        df_head = pd.read_csv(csv_path, nrows=2)
    except Exception as e:
        print(f"[ERROR] can't read {csv_path}: {e}")
        return

    # find time column
    time_col = find_time_column(df_head)
    df = pd.read_csv(csv_path, usecols=lambda c: True, low_memory=False)
    if time_col not in df.columns:
        time_col = find_time_column(df)
    times = df[time_col].to_numpy(dtype=float)

    # estimate fs
    try:
        fs = estimate_fs_from_times(times)
    except Exception as e:
        print(f"[ERROR] could not estimate fs for {csv_path}: {e}")
        return
    if VERBOSE:
        print(f"  estimated fs = {fs:.2f} Hz")

    # windowing
    win_len = max(2, int(round(WINDOW_SEC * fs)))
    hop_len = max(1, int(round(HOP_SEC * fs)))
    total_samples = len(times)
    starts = list(range(0, max(1, total_samples - win_len + 1), hop_len))
    if MAX_WINDOWS_PER_FILE and len(starts) > MAX_WINDOWS_PER_FILE:
        starts = starts[:MAX_WINDOWS_PER_FILE]

    base = Path(csv_path).stem
    out_mod_dir = os.path.join(out_dir, "vibration")
    os.makedirs(out_mod_dir, exist_ok=True)
    out_png_dir = None
    if OUT_PNG_SUBFOLDER:
        out_png_dir = os.path.join(out_mod_dir, OUT_PNG_SUBFOLDER)
        os.makedirs(out_png_dir, exist_ok=True)

    # freqs (and scales for meta)
    freqs, scales = prepare_freqs(fs, N_SCALES, FREQ_MIN, FREQ_MAX_RATIO)

    for ch in channel_names:
        if ch not in df.columns:
            if VERBOSE:
                print(f"  channel '{ch}' not in CSV; skipping")
            continue
        raw = clean_and_interpolate(df[ch].to_numpy(copy=True))

        for i, s in enumerate(starts):
            e = s + win_len
            if e > total_samples:
                break
            window_times = times[s:e]
            window_sig = signal.detrend(raw[s:e])

            try:
                power = compute_power_fcwt(window_sig, fs, freqs)
            except Exception as e:
                print(f"   [WARN] CWT failed for {base} ch={ch} win={i}: {e}")
                continue

            meta = {
                "source_csv": os.path.basename(csv_path),
                "channel": ch,
                "window_index": int(i),
                "sample_start": int(s),
                "sample_end": int(e),
                "fs": float(fs),
                "scales": scales.tolist(),  # included for parity with old format
                "freqs": freqs.tolist(),
                "wavelet": "morl",
                "window_sec": float(WINDOW_SEC),
                "hop_sec": float(HOP_SEC),
                "backend": "fcwt" if _fcwt_ok else "pywt_fallback"
            }

            out_name = f"{base}__{ch}__win{str(i).zfill(5)}.npy"
            out_path = os.path.join(out_mod_dir, out_name)
            save_scalogram_array(out_path, power, freqs, window_times, meta)

            if OUT_PNG_SUBFOLDER:
                png_path = os.path.join(out_png_dir, out_name.replace(".npy", ".png"))
                arr_disp = np.log10(power + 1e-12)
                vmin, vmax = arr_disp.min(), arr_disp.max()
                save_scalogram_png(png_path, arr_disp, freqs, window_times, vmin=vmin, vmax=vmax)

            if VERBOSE and (i % 20 == 0):
                print(f"    ch={ch} window {i+1}/{len(starts)} saved")

    del df
    if VERBOSE:
        print(f"[DONE] {csv_path} -> {out_mod_dir}")

def generate_vibration_scalograms():
    files = sorted(glob.glob(os.path.join(VIBRATION_FOLDER, "*.csv")))
    if VIBRATION_FILENAME_FILTER:
        files = [f for f in files if Path(f).name in set(VIBRATION_FILENAME_FILTER)]
    if not files:
        print(f"[WARN] No matching vibration CSVs in {VIBRATION_FOLDER} for filter {VIBRATION_FILENAME_FILTER}")
        return
    for f in files:
        process_csv_vibration(f, VIBRATION_CHANNELS, OUT_BASE)

# ----------------- SIMPLE TRAIN (if ≥2 classes) -----------------
def parse_labels_from_source_csv(source_csv_basename):
    """
    Parse aaaaNm_bbbb_cccc.csv into {'load','condition','severity'}
    Also fix 'Unbalalnce' -> 'Unbalance'
    """
    name = Path(source_csv_basename).stem
    parts = name.split("_")
    label = {"load": None, "condition": None, "severity": None}
    if len(parts) >= 3:
        label["load"] = parts[0]
        label["condition"] = parts[1]
        label["severity"] = parts[2]
    elif len(parts) == 2:
        label["load"] = parts[0]
        label["condition"] = parts[1]
        label["severity"] = "00"
    else:
        label["condition"] = parts[0] if parts else None
    if label["condition"] and label["condition"].lower() == "unbalalnce":
        label["condition"] = "Unbalance"
    return label

def resize_img(X, H=IMG_H, W=IMG_W):
    # bilinear-ish via 1D interps
    F, T = X.shape
    f_idx = np.linspace(0, F-1, H)
    t_idx = np.linspace(0, T-1, W)
    X_t = np.vstack([np.interp(t_idx, np.arange(T), X[i, :]) for i in range(F)])
    X_ft = np.vstack([np.interp(f_idx, np.arange(F), X_t[:, j]) for j in range(W)]).T
    return X_ft.astype(np.float32)

def load_dataset_from_scalograms():
    folder = os.path.join(OUT_BASE, "vibration")
    npys = sorted(glob.glob(os.path.join(folder, "*.npy")))
    X, y = [], []
    for p in npys:
        meta_path = p + ".meta.json"
        if not os.path.exists(meta_path):
            continue
        try:
            with open(meta_path, "r") as fh:
                meta = json.load(fh)
            src = meta.get("source_csv", "")
            labels = parse_labels_from_source_csv(src)
            cls = labels.get("condition")
            if cls is None:
                continue
            power = np.load(p, allow_pickle=False)  # (F,T)
            # log compress + 0..1 normalize
            Xlog = np.log10(power + 1e-12)
            mn, mx = Xlog.min(), Xlog.max()
            if mx > mn:
                Xn = (Xlog - mn) / (mx - mn)
            else:
                Xn = np.zeros_like(Xlog, dtype=np.float32)
            Xrs = resize_img(Xn, IMG_H, IMG_W)
            X.append(Xrs.flatten())  # classical small baseline
            y.append(cls)
        except Exception:
            continue
    return (np.array(X, dtype=np.float32), np.array(y, dtype=object))

def train_if_possible():
    X, y = load_dataset_from_scalograms()
    if X.shape[0] == 0:
        print("[TRAIN] No scalograms found to train on.")
        return
    classes = sorted(set(y.tolist()))
    if len(classes) < 2:
        print(f"[TRAIN] Only one class present: {classes}. Classification requires ≥ 2 classes. Skipping training.")
        return

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)
    scaler = StandardScaler(with_mean=True, with_std=True)
    Xtr_s = scaler.fit_transform(Xtr)
    Xte_s = scaler.transform(Xte)
    if USE_PCA:
        pca = PCA(n_components=min(PCA_DIM, Xtr_s.shape[1]), random_state=RANDOM_STATE)
        Xtr_s = pca.fit_transform(Xtr_s)
        Xte_s = pca.transform(Xte_s)

    clf = LogisticRegression(max_iter=2000)
    clf.fit(Xtr_s, ytr)
    yhat = clf.predict(Xte_s)
    acc = accuracy_score(yte, yhat)
    f1m = f1_score(yte, yhat, average="macro")
    print(f"[TRAIN] LogisticRegression -> acc={acc:.3f}  f1-macro={f1m:.3f}  classes={classes}")

# ----------------- MAIN -----------------
if __name__ == "__main__":
    print(f"Fast CWT available: {_fcwt_ok}")
    print("Generating vibration scalograms (fCWT if available)...")
    generate_vibration_scalograms()
    print("Attempting simple training on generated scalograms...")
    train_if_possible()
    print("✅ Done.")
