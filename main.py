#!/usr/bin/env python3
"""
Gradio inference app (Option A: 4-channel vibration fusion)

- Upload acoustic and/or vibration CSVs
- Convert to Morlet CWT scalograms (same assumptions as training)
  * Acoustic: single channel
  * Vibration: 4 channels (x_A, y_A, x_B, y_B) per window, stacked as 4xFxT
- Display scalograms in the UI
- Load trained multitask CNNs (per modality)
- Predict condition + severity per modality (avg over windows)
- Produce an ensemble prediction (avg across modalities) if both provided

Expected files:
    models/acoustic_multitask.pt
    models/acoustic_labels.json
    models/vibration_multitask.pt
    models/vibration_labels.json
"""

import io, os, json
from typing import Dict, List, Tuple, Optional, Union
from pathlib import Path

import gradio as gr
import numpy as np
import pandas as pd
import pywt
from scipy import signal
import matplotlib.pyplot as plt
from PIL import Image

import torch
import torch.nn as nn

# --------------------------- CONFIG ---------------------------
MODEL_DIR = "models"
ACOUSTIC_WEIGHTS = os.path.join(MODEL_DIR, "acoustic_multitask.pt")
ACOUSTIC_LABELS  = os.path.join(MODEL_DIR, "acoustic_labels.json")
VIB_WEIGHTS      = os.path.join(MODEL_DIR, "vibration_multitask.pt")
VIB_LABELS       = os.path.join(MODEL_DIR, "vibration_labels.json")

# CSV parsing
TIME_COL_KEYWORDS = ("time", "stamp")
ACOUSTIC_CHANNELS = ["values"]
VIBRATION_CHANNELS = [
    "x_direction_housing_A",
    "y_direction_housing_A",
    "x_direction_housing_B",
    "y_direction_housing_B",
]

# Signal → scalogram
WAVELET = "morl"
N_SCALES = 128
FREQ_MIN = 20.0
FREQ_MAX_RATIO = 0.45
WINDOW_SEC = 0.5
HOP_SEC = 0.25

# Inference windows
N_WINDOWS_PER_FILE = 4  # number of evenly spaced windows per file for inference

# Device / AMP
USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda" if USE_CUDA else "cpu")
AMP_ENABLED = USE_CUDA

# --------------------------- FILE READER (fixes NamedString .read error) ---------------------------
def read_file_like(x: Union[str, Path, bytes, io.BytesIO, dict]) -> bytes:
    """
    Accepts Gradio File outputs: NamedString (path-like), str, Path, dict with 'name', or bytes/BytesIO.
    Returns raw bytes.
    """
    if x is None:
        return b""
    # Older/newer Gradio may give dicts: {'name': '/tmp/...csv', ...}
    if isinstance(x, dict) and "name" in x:
        x = x["name"]
    # Path-like
    if isinstance(x, (str, Path)):
        p = Path(x)
        return p.read_bytes()
    # Bytes-like
    if isinstance(x, bytes):
        return x
    if isinstance(x, io.BytesIO):
        return x.getvalue()
    # Some builds give an object with .name property
    name = getattr(x, "name", None)
    if isinstance(name, (str, Path)) and os.path.exists(name):
        return Path(name).read_bytes()
    # Last resort: has .read()?
    read = getattr(x, "read", None)
    if callable(read):
        return x.read()
    raise TypeError(f"Unsupported file type: {type(x)}")

# --------------------------- UTILITIES ---------------------------
def find_time_column(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    low = [c.lower() for c in cols]
    for kw in TIME_COL_KEYWORDS:
        for i, c in enumerate(low):
            if kw in c:
                return cols[i]
    for c in cols:
        if pd.api.types.is_numeric_dtype(df[c]):
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
    fmax = min(fmax_ratio * fs, fs/2.0)
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
    return np.interp(inds, inds[good], arr[good])

def compute_cwt_scalogram(sig: np.ndarray, fs: float, scales: np.ndarray) -> np.ndarray:
    sig = signal.detrend(sig)
    coef, _ = pywt.cwt(sig, scales, WAVELET, sampling_period=1.0/fs)
    power = np.abs(coef).astype(np.float32)  # (F, T)
    return power

def select_even_windows(total_len: int, win_len: int, n: int) -> List[int]:
    if total_len < win_len:
        return [0]
    if n <= 1:
        return [(total_len - win_len)//2]
    last_start = total_len - win_len
    return [int(round(i * last_start / (n-1))) for i in range(n)]

def fig_to_image_array(fig) -> np.ndarray:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=150)
    plt.close(fig)
    buf.seek(0)
    img = Image.open(buf).convert("RGB")
    return np.array(img)

def render_scalogram_image(power: np.ndarray, freqs: np.ndarray, fs: float, title: str = "") -> np.ndarray:
    disp = np.log10(power + 1e-12)
    if np.isfinite(disp).any():
        vmin, vmax = np.percentile(disp, [1, 99])
        if vmin == vmax:
            vmax = vmin + 1e-6
    else:
        vmin, vmax = disp.min(), disp.max() + 1e-6
    fig = plt.figure(figsize=(5, 3))
    extent = [0, power.shape[1]/fs, freqs[0], freqs[-1]]
    plt.imshow(disp, aspect="auto", origin="lower", extent=extent, vmin=vmin, vmax=vmax, cmap="viridis")
    plt.yscale("log")
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")
    if title:
        plt.title(title)
    plt.tight_layout()
    return fig_to_image_array(fig)

# --------------------------- MODEL ---------------------------
class ConvBlock(nn.Module):
    def __init__(self, c_in, c_out, k=3, p=1, s=1):
        super().__init__()
        self.conv = nn.Conv2d(c_in, c_out, k, stride=s, padding=p)
        self.bn   = nn.BatchNorm2d(c_out)
        self.act  = nn.ReLU(inplace=True)
    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class MultiTaskCNN(nn.Module):
    """
    Matches training: in_ch=1 for acoustic, in_ch=4 for vibration (4-channel fusion).
    """
    def __init__(self, n_cond: int, n_sev: int, in_ch: int):
        super().__init__()
        self.backbone = nn.Sequential(
            ConvBlock(in_ch, 32, 5, 2),
            nn.MaxPool2d((2, 4)),          # ↓F x2, ↓T x4
            ConvBlock(32, 64, 3, 1),
            nn.MaxPool2d((2, 2)),
            ConvBlock(64, 128, 3, 1),
            nn.AdaptiveAvgPool2d((4, 8)),
        )
        feat_dim = 128 * 4 * 8
        self.drop = nn.Dropout(0.2)
        self.fc_cond = nn.Linear(feat_dim, n_cond)
        self.fc_sev  = nn.Linear(feat_dim, n_sev)
    def forward(self, x):
        z = self.backbone(x)
        z = z.flatten(1)
        z = self.drop(z)
        return self.fc_cond(z), self.fc_sev(z)

def load_model_and_labels(weights_path: str, labels_json: str, in_ch: int):
    with open(labels_json, "r") as fh:
        lab = json.load(fh)
    conds = lab["conditions"]
    sevs  = lab["severities"]
    model = MultiTaskCNN(n_cond=len(conds), n_sev=len(sevs), in_ch=in_ch).to(DEVICE)
    state = torch.load(weights_path, map_location=DEVICE)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model, conds, sevs

# --------------------------- INFERENCE HELPERS ---------------------------
@torch.inference_mode()
def predict_acoustic_scalos(model: nn.Module, scalos: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    ps_c, ps_s = [], []
    for arr in scalos:
        X = arr.astype(np.float32, copy=False)
        X = np.log1p(X)
        mu, sd = X.mean(), X.std() + 1e-6
        X = (X - mu) / sd
        X = np.expand_dims(X, 0)  # (1,F,T)
        X = np.expand_dims(X, 0)  # (B=1,1,F,T)
        xb = torch.from_numpy(X).to(DEVICE, non_blocking=True)
        if AMP_ENABLED:
            with torch.cuda.amp.autocast():
                pc, ps = model(xb)
        else:
            pc, ps = model(xb)
        ps_c.append(pc.softmax(1).squeeze(0).cpu().numpy())
        ps_s.append(ps.softmax(1).squeeze(0).cpu().numpy())
    p_c = np.mean(np.stack(ps_c, 0), 0)
    p_s = np.mean(np.stack(ps_s, 0), 0)
    return p_c, p_s

@torch.inference_mode()
def predict_vibration_4ch(model: nn.Module, scalos_4ch: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    """
    scalos_4ch: list of arrays with shape (C=4,F,T) or (C<=4,F,T)
    Per-channel normalization (log1p + z-score) to match training.
    """
    ps_c, ps_s = [], []
    for X in scalos_4ch:
        X = X.astype(np.float32, copy=False)
        X = np.log1p(X)
        mu = X.mean(axis=(1,2), keepdims=True)
        sd = X.std(axis=(1,2), keepdims=True) + 1e-6
        X = (X - mu) / sd
        X = np.expand_dims(X, 0)  # (B=1,C,F,T)
        xb = torch.from_numpy(X).to(DEVICE, non_blocking=True)
        if AMP_ENABLED:
            with torch.cuda.amp.autocast():
                pc, ps = model(xb)
        else:
            pc, ps = model(xb)
        ps_c.append(pc.softmax(1).squeeze(0).cpu().numpy())
        ps_s.append(ps.softmax(1).squeeze(0).cpu().numpy())
    p_c = np.mean(np.stack(ps_c, 0), 0)
    p_s = np.mean(np.stack(ps_s, 0), 0)
    return p_c, p_s

def topk_table_md(title: str, names: List[str], probs: np.ndarray, k: int = 5) -> str:
    order = np.argsort(-probs)[:k]
    hdr = f"### {title}\n\n| Class | Prob |\n|---|---|\n"
    rows = "".join([f"| {names[i]} | {probs[i]:.2%} |\n" for i in order])
    return hdr + rows

def top1(probs: np.ndarray, names: List[str]) -> Tuple[str, float]:
    i = int(np.argmax(probs))
    return names[i], float(probs[i])

def ensemble_union(preds: List[Tuple[np.ndarray, List[str]]]) -> Tuple[np.ndarray, List[str]]:
    if not preds:
        return np.array([]), []
    classes = sorted(set().union(*[set(names) for (_, names) in preds]))
    idx = {c:i for i,c in enumerate(classes)}
    acc = np.zeros(len(classes), dtype=np.float64)
    for p, names in preds:
        v = np.zeros(len(classes), dtype=np.float64)
        for i, n in enumerate(names):
            v[idx[n]] = p[i]
        acc += v
    acc /= len(preds)
    s = acc.sum()
    if s > 0: acc /= s
    return acc.astype(np.float32), classes

# --------------------------- CSV → SCALOGRAMS ---------------------------
def csv_to_acoustic_scalos(csv_bytes: bytes, max_windows: int = N_WINDOWS_PER_FILE):
    df = pd.read_csv(io.BytesIO(csv_bytes), low_memory=False)
    time_col = find_time_column(df)
    times = df[time_col].to_numpy(dtype=float)
    fs = estimate_fs_from_times(times)
    # acoustic fs correction as in training
    fs /= 2.0

    chans = [c for c in ACOUSTIC_CHANNELS if c in df.columns]
    if not chans:
        raise ValueError(f"Acoustic CSV missing expected channel. Found: {list(df.columns)}")

    win_len = max(2, int(round(WINDOW_SEC * fs)))
    total_samples = len(times)
    starts = select_even_windows(total_samples, win_len, max_windows)

    scales, freqs = prepare_scales_for_freqs(fs, N_SCALES, FREQ_MIN, FREQ_MAX_RATIO)

    scalos, images = [], []
    ch = chans[0]
    sig_full = clean_and_interpolate(df[ch].to_numpy(copy=True))
    for wi, s in enumerate(starts):
        e = min(total_samples, s + win_len)
        sig = sig_full[s:e]
        if len(sig) < win_len:
            sig = np.pad(sig, (0, win_len - len(sig)), mode="edge")
        P = compute_cwt_scalogram(sig, fs, scales)  # (F,T)
        scalos.append(P)
        if wi == 0:
            images.append(render_scalogram_image(P, freqs, fs, title=f"Acoustic:{ch}"))
    return scalos, images  # images: list of 1 preview image

def csv_to_vibration_4ch_scalos(csv_bytes: bytes, max_windows: int = N_WINDOWS_PER_FILE):
    df = pd.read_csv(io.BytesIO(csv_bytes), low_memory=False)
    time_col = find_time_column(df)
    times = df[time_col].to_numpy(dtype=float)
    fs = estimate_fs_from_times(times)

    chans = [c for c in VIBRATION_CHANNELS if c in df.columns]
    if not chans:
        raise ValueError(f"Vibration CSV missing expected channels. Need any of {VIBRATION_CHANNELS}; Found: {list(df.columns)}")

    win_len = max(2, int(round(WINDOW_SEC * fs)))
    total_samples = len(times)
    starts = select_even_windows(total_samples, win_len, max_windows)

    scales, freqs = prepare_scales_for_freqs(fs, N_SCALES, FREQ_MIN, FREQ_MAX_RATIO)

    # Pre-load/clean all available channels
    series: Dict[str, np.ndarray] = {}
    for ch in chans:
        series[ch] = clean_and_interpolate(df[ch].to_numpy(copy=True))

    groups_4ch: List[np.ndarray] = []        # each is (C<=4,F,T)
    preview_images: List[np.ndarray] = []    # show 4 images for first window if possible

    for wi, s in enumerate(starts):
        e = min(total_samples, s + win_len)
        per_ch_scalos: List[np.ndarray] = []
        per_ch_images: List[np.ndarray] = []
        present_order = [ch for ch in VIBRATION_CHANNELS if ch in series]
        for ch in present_order:
            sig = series[ch][s:e]
            if len(sig) < win_len:
                sig = np.pad(sig, (0, win_len - len(sig)), mode="edge")
            P = compute_cwt_scalogram(sig, fs, scales)  # (F,T)
            per_ch_scalos.append(P)
            if wi == 0:
                per_ch_images.append(render_scalogram_image(P, freqs, fs, title=f"Vibration:{ch}"))

        # align time across channels
        Tm = min(p.shape[1] for p in per_ch_scalos)
        per_ch_scalos = [p[:, :Tm] for p in per_ch_scalos]
        X = np.stack(per_ch_scalos, axis=0)  # (C,F,T)
        groups_4ch.append(X)

        if wi == 0 and per_ch_images:
            preview_images.extend(per_ch_images)

    return groups_4ch, preview_images  # images: 4 per first window (if available)

# --------------------------- LAZY MODEL LOAD ---------------------------
_acoustic_model = None
_acoustic_cond = []
_acoustic_sev = []
_vib_model = None
_vib_cond = []
_vib_sev = []

def lazy_load_models() -> str:
    global _acoustic_model, _acoustic_cond, _acoustic_sev, _vib_model, _vib_cond, _vib_sev
    msgs = []
    if _acoustic_model is None and os.path.exists(ACOUSTIC_WEIGHTS) and os.path.exists(ACOUSTIC_LABELS):
        _acoustic_model, _acoustic_cond, _acoustic_sev = load_model_and_labels(ACOUSTIC_WEIGHTS, ACOUSTIC_LABELS, in_ch=1)
        msgs.append(f"Loaded acoustic model ({len(_acoustic_cond)} conds / {len(_acoustic_sev)} sevs)")
    if _vib_model is None and os.path.exists(VIB_WEIGHTS) and os.path.exists(VIB_LABELS):
        _vib_model, _vib_cond, _vib_sev = load_model_and_labels(VIB_WEIGHTS, VIB_LABELS, in_ch=4)
        msgs.append(f"Loaded vibration model ({len(_vib_cond)} conds / {len(_vib_sev)} sevs)")
    return "\n".join(msgs) if msgs else "Models ready."

# --------------------------- GRADIO CALLBACK ---------------------------
def run_inference(acoustic_csv, vibration_csv):
    status = lazy_load_models()

    gallery_images = []
    details_md = []
    summaries = []

    # --- Acoustic ---
    acoustic_pred = None
    if acoustic_csv is not None and _acoustic_model is not None:
        try:
            a_bytes = read_file_like(acoustic_csv)
            a_scalos, a_imgs = csv_to_acoustic_scalos(a_bytes, max_windows=N_WINDOWS_PER_FILE)
            gallery_images.extend([(img, "Acoustic scalogram") for img in a_imgs])

            p_c, p_s = predict_acoustic_scalos(_acoustic_model, a_scalos)
            c1, c1p = top1(p_c, _acoustic_cond)
            s1, s1p = top1(p_s, _acoustic_sev)
            summaries.append(f"**Acoustic** → Condition: **{c1}** ({c1p:.2%}), Severity: **{s1}** ({s1p:.2%})")
            details_md.append(topk_table_md("Acoustic — Condition (Top 5)", _acoustic_cond, p_c, k=min(5,len(_acoustic_cond))))
            details_md.append(topk_table_md("Acoustic — Severity (Top 5)",  _acoustic_sev,  p_s, k=min(5,len(_acoustic_sev))))
            acoustic_pred = (p_c, _acoustic_cond, p_s, _acoustic_sev)
        except Exception as e:
            summaries.append(f"Acoustic error: {e}")

    # --- Vibration (4-channel fusion) ---
    vibration_pred = None
    if vibration_csv is not None and _vib_model is not None:
        try:
            v_bytes = read_file_like(vibration_csv)
            v_groups, v_imgs = csv_to_vibration_4ch_scalos(v_bytes, max_windows=N_WINDOWS_PER_FILE)
            gallery_images.extend([(img, "Vibration scalogram") for img in v_imgs])

            p_c, p_s = predict_vibration_4ch(_vib_model, v_groups)
            c1, c1p = top1(p_c, _vib_cond)
            s1, s1p = top1(p_s, _vib_sev)
            summaries.append(f"**Vibration** → Condition: **{c1}** ({c1p:.2%}), Severity: **{s1}** ({s1p:.2%})")
            details_md.append(topk_table_md("Vibration — Condition (Top 5)", _vib_cond, p_c, k=min(5,len(_vib_cond))))
            details_md.append(topk_table_md("Vibration — Severity (Top 5)",  _vib_sev,  p_s, k=min(5,len(_vib_sev))))
            vibration_pred = (p_c, _vib_cond, p_s, _vib_sev)
        except Exception as e:
            summaries.append(f"Vibration error: {e}")

    # --- Ensemble across modalities (if both present) ---
    ensemble_md = ""
    if acoustic_pred and vibration_pred:
        a_pc, a_cnames, a_ps, a_snames = acoustic_pred
        v_pc, v_cnames, v_ps, v_snames = vibration_pred
        ens_c, ens_cnames = ensemble_union([(a_pc, a_cnames), (v_pc, v_cnames)])
        ens_s, ens_snames = ensemble_union([(a_ps, a_snames), (v_ps, v_snames)])
        if ens_c.size:
            ec, ecp = top1(ens_c, ens_cnames)
            ensemble_md += f"**Ensemble** → Condition: **{ec}** ({ecp:.2%})\n\n"
            ensemble_md += topk_table_md("Ensemble — Condition (Top 5)", ens_cnames, ens_c, k=min(5,len(ens_cnames))) + "\n"
        if ens_s.size:
            es, esp = top1(ens_s, ens_snames)
            ensemble_md += f"**Ensemble** → Severity: **{es}** ({esp:.2%})\n\n"
            ensemble_md += topk_table_md("Ensemble — Severity (Top 5)", ens_snames, ens_s, k=min(5,len(ens_snames)))
    else:
        ensemble_md = "_Ensemble shown when both modalities are provided._"

    status_out = status if status else "Models ready."
    summary_out = "\n\n".join(summaries) if summaries else "Upload at least one CSV."
    details_out = ensemble_md + ("\n\n" if ensemble_md else "") + ("\n\n".join(details_md) if details_md else "—")
    return status_out, summary_out, details_out, gallery_images

# --------------------------- UI ---------------------------
with gr.Blocks(title="Acoustic + Vibration (4-Ch) Fault Inference") as demo:
    gr.Markdown(
        """
        # 🛠️ Acoustic + Vibration (4-Channel) — Fault Condition & Severity
        Upload **acoustic** and/or **vibration** CSVs.  
        We compute **Morlet CWT scalograms** on several windows, show them, and predict with your trained models.
        - Acoustic: single-channel
        - Vibration: fused **4-channel** (x_A, y_A, x_B, y_B) per window
        """
    )
    with gr.Row():
        acoustic_in = gr.File(label="Acoustic CSV (must have 'values' column)", file_types=[".csv"])
        vibration_in = gr.File(label="Vibration CSV (x/y A/B columns)", file_types=[".csv"])
    run_btn = gr.Button("Run Inference 🚀", variant="primary")

    status = gr.Markdown("Models will load on first run.")
    summary = gr.Markdown()
    details = gr.Markdown()

    gr.Markdown("### Preview Scalograms")
    gallery = gr.Gallery(label="Scalograms", show_label=False, height=360, columns=2)

    run_btn.click(
        run_inference,
        inputs=[acoustic_in, vibration_in],
        outputs=[status, summary, details, gallery],
    )

if __name__ == "__main__":
    print(f"Device: {DEVICE} | CUDA: {USE_CUDA} | AMP: {AMP_ENABLED}")
    demo.launch()
