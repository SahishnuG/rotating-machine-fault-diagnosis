#!/usr/bin/env python3
"""
make_scalograms.py

Generates Morlet CWT scalograms for every CSV in:
  - acoustic/
  - vibration/
and stores results as .npy arrays (and optional PNGs).

Now includes: fs halved for acoustic modality.
"""

import os, glob, json, math
from pathlib import Path
from typing import Tuple, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pywt
from scipy import signal

# ----------------- CONFIG -----------------
ACOUSTIC_FOLDER = "acoustic"
VIBRATION_FOLDER = "vibration"
OUT_BASE = "scalograms"
OUT_PNG_SUBFOLDER = "pngs"  # None to skip saving PNGs
os.makedirs(OUT_BASE, exist_ok=True)

WINDOW_SEC = 0.5
HOP_SEC = 0.25
MAX_WINDOWS_PER_FILE = 2000

WAVELET = "morl"
N_SCALES = 128
FREQ_MIN = 20.0
FREQ_MAX_RATIO = 0.45
NORMALIZE_SCALOGRAM = True

TIME_COL_KEYWORDS = ("time", "stamp")
FLOAT_DTYPE = np.float32

ACOUSTIC_CHANNELS = ["values"]
VIBRATION_CHANNELS = [
    "x_direction_housing_A",
    "y_direction_housing_A",
    "x_direction_housing_B",
    "y_direction_housing_B",
]

VERBOSE = True

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


def prepare_scales_for_freqs(fs: float, n_scales: int, fmin: float, fmax_ratio: float):
    fmax = min(fmax_ratio * fs, fs / 2.0)
    if fmax <= fmin:
        fmin = max(0.5, fmax * 0.01)
    freqs = np.logspace(np.log10(fmin), np.log10(fmax), num=n_scales)
    fc = pywt.central_frequency(WAVELET)
    dt = 1.0 / fs
    scales = fc / (freqs * dt)
    return scales, freqs


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


# ----------------- MAIN -----------------
def process_csv_file(csv_path: str, modality: str, channel_names: List[str], out_dir: str):
    if VERBOSE:
        print(f"[PROCESS] {modality}: {csv_path}")

    try:
        df_head = pd.read_csv(csv_path, nrows=2)
    except Exception as e:
        print(f"[ERROR] can't read {csv_path}: {e}")
        return

    time_col = find_time_column(df_head)
    df = pd.read_csv(csv_path, usecols=lambda c: True, low_memory=False)
    if time_col not in df.columns:
        print(f"[WARN] time column not found reliably in {csv_path}; using first numeric column.")
        time_col = find_time_column(df)
    times = df[time_col].to_numpy(dtype=float)

    # --- Estimate sampling rate ---
    try:
        fs = estimate_fs_from_times(times)
    except Exception as e:
        print(f"[ERROR] could not estimate fs for {csv_path}: {e}")
        return

    # 🔹 Apply correction for acoustic modality
    if modality == "acoustic":
        fs /= 2.0

    if VERBOSE:
        print(f"  estimated fs = {fs:.2f} Hz")

    win_len = max(2, int(round(WINDOW_SEC * fs)))
    hop_len = max(1, int(round(HOP_SEC * fs)))
    total_samples = len(times)
    starts = list(range(0, max(1, total_samples - win_len + 1), hop_len))
    if len(starts) > MAX_WINDOWS_PER_FILE:
        starts = starts[:MAX_WINDOWS_PER_FILE]

    base = Path(csv_path).stem
    out_mod_dir = os.path.join(out_dir, modality)
    os.makedirs(out_mod_dir, exist_ok=True)
    out_png_dir = None
    if OUT_PNG_SUBFOLDER:
        out_png_dir = os.path.join(out_mod_dir, OUT_PNG_SUBFOLDER)
        os.makedirs(out_png_dir, exist_ok=True)

    scales, freqs = prepare_scales_for_freqs(fs, N_SCALES, FREQ_MIN, FREQ_MAX_RATIO)

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
                coef, _ = pywt.cwt(window_sig, scales, WAVELET, sampling_period=1.0/fs)
                power = np.abs(coef)
            except Exception as e:
                print(f"   [WARN] CWT failed for {base} ch={ch} win={i}: {e}")
                continue

            meta = {
                "source_csv": os.path.basename(csv_path),
                "channel": ch,
                "window_index": int(i),
                "fs": float(fs),
                "scales": scales.tolist(),
                "freqs": freqs.tolist(),
                "wavelet": WAVELET,
                "window_sec": WINDOW_SEC,
                "hop_sec": HOP_SEC,
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


def process_folder(folder: str, modality: str, channel_names: List[str], out_dir: str):
    files = sorted(glob.glob(os.path.join(folder, "*.csv")))
    if not files:
        print(f"[WARN] No CSV files in {folder}")
        return
    for f in files:
        process_csv_file(f, modality, channel_names, out_dir)


if __name__ == "__main__":
    print("Starting scalogram generation...")
    process_folder(ACOUSTIC_FOLDER, "acoustic", ACOUSTIC_CHANNELS, OUT_BASE)
    process_folder(VIBRATION_FOLDER, "vibration", VIBRATION_CHANNELS, OUT_BASE)
    print("✅ All done.")
