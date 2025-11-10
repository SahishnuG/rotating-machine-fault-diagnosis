#!/usr/bin/env python3
import io, os, json, uuid, shutil
from typing import Dict, List, Tuple, Optional, Union
from pathlib import Path

from flask import (
    Flask, request, render_template, send_from_directory, redirect, url_for, flash
)
from werkzeug.utils import secure_filename

import numpy as np
import pandas as pd
import pywt
from scipy import signal
import matplotlib
matplotlib.use("Agg")  # headless rendering
import matplotlib.pyplot as plt
from PIL import Image

import torch
import torch.nn as nn

# --------------------------- FLASK CONFIG ---------------------------
BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
JOBS_DIR = BASE_DIR / "jobs"
STATIC_DIR = BASE_DIR / "static"

ALLOWED_EXTENSIONS = {"csv"}
MAX_CONTENT_LENGTH = 1024 * 1024 * 1024  # 1024 MB
  # 128MB

app = Flask(__name__, static_folder="static", template_folder="templates")
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "dev-secret")
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH

# Ensure jobs dir exists
JOBS_DIR.mkdir(exist_ok=True)

# --------------------------- ORIGINAL CONFIG ---------------------------
ACOUSTIC_WEIGHTS = MODEL_DIR / "acoustic_multitask.pt"
ACOUSTIC_LABELS  = MODEL_DIR / "acoustic_labels.json"
VIB_WEIGHTS      = MODEL_DIR / "vibration_multitask.pt"
VIB_LABELS       = MODEL_DIR / "vibration_labels.json"

TIME_COL_KEYWORDS = ("time", "stamp")
ACOUSTIC_CHANNELS = ["values"]
VIBRATION_CHANNELS = [
    "x_direction_housing_A",
    "y_direction_housing_A",
    "x_direction_housing_B",
    "y_direction_housing_B",
]

WAVELET = "morl"
N_SCALES = 128
FREQ_MIN = 20.0
FREQ_MAX_RATIO = 0.45
WINDOW_SEC = 0.5
HOP_SEC = 0.25

N_WINDOWS_PER_FILE = 4

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda" if USE_CUDA else "cpu")
AMP_ENABLED = USE_CUDA

# --------------------------- HELPERS ---------------------------
def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

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
    return np.interp(inds, inds[good], arr[good])

def compute_cwt_scalogram(sig: np.ndarray, fs: float, scales: np.ndarray) -> np.ndarray:
    sig = signal.detrend(sig)
    coef, _ = pywt.cwt(sig, scales, WAVELET, sampling_period=1.0 / fs)
    power = np.abs(coef).astype(np.float32)  # (F, T)
    return power

def select_even_windows(total_len: int, win_len: int, n: int) -> List[int]:
    if total_len < win_len:
        return [0]
    if n <= 1:
        return [(total_len - win_len) // 2]
    last_start = total_len - win_len
    return [int(round(i * last_start / (n - 1))) for i in range(n)]

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
        vmin, vmax = float(disp.min()), float(disp.max() + 1e-6)
    fig = plt.figure(figsize=(5.5, 3.3))
    extent = [0, power.shape[1] / fs, freqs[0], freqs[-1]]
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
            nn.MaxPool2d((2, 4)),
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

def load_model_and_labels(weights_path: Path, labels_json: Path, in_ch: int):
    with open(labels_json, "r") as fh:
        lab = json.load(fh)
    conds = lab["conditions"]
    sevs  = lab["severities"]
    model = MultiTaskCNN(n_cond=len(conds), n_sev=len(sevs), in_ch=in_ch).to(DEVICE)
    state = torch.load(weights_path, map_location=DEVICE)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model, conds, sevs

# Lazy singletons
_acoustic_model = None
_acoustic_cond = []
_acoustic_sev = []
_vib_model = None
_vib_cond = []
_vib_sev = []

def lazy_load_models() -> str:
    global _acoustic_model, _acoustic_cond, _acoustic_sev, _vib_model, _vib_cond, _vib_sev
    msgs = []
    if _acoustic_model is None and ACOUSTIC_WEIGHTS.exists() and ACOUSTIC_LABELS.exists():
        _acoustic_model, _acoustic_cond, _acoustic_sev = load_model_and_labels(ACOUSTIC_WEIGHTS, ACOUSTIC_LABELS, in_ch=1)
        msgs.append(f"Loaded acoustic model ({len(_acoustic_cond)} cond / {len(_acoustic_sev)} sev)")
    if _vib_model is None and VIB_WEIGHTS.exists() and VIB_LABELS.exists():
        _vib_model, _vib_cond, _vib_sev = load_model_and_labels(VIB_WEIGHTS, VIB_LABELS, in_ch=4)
        msgs.append(f"Loaded vibration model ({len(_vib_cond)} cond / {len(_vib_sev)} sev)")
    return "\n".join(msgs) if msgs else "Models ready."

# --------------------------- CSV → SCALOGRAMS ---------------------------
def csv_to_acoustic_scalos(csv_bytes: bytes, max_windows: int = N_WINDOWS_PER_FILE):
    df = pd.read_csv(io.BytesIO(csv_bytes), low_memory=False)
    time_col = find_time_column(df)
    times = df[time_col].to_numpy(dtype=float)
    fs = estimate_fs_from_times(times)
    fs /= 2.0  # training assumption

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
    return scalos, images  # preview: 1 image

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

    series: Dict[str, np.ndarray] = {}
    for ch in chans:
        series[ch] = clean_and_interpolate(df[ch].to_numpy(copy=True))

    groups_4ch: List[np.ndarray] = []
    preview_images: List[np.ndarray] = []

    for wi, s in enumerate(starts):
        e = min(total_samples, s + win_len)
        per_ch_scalos: List[np.ndarray] = []
        per_ch_images: List[np.ndarray] = []
        present_order = [ch for ch in VIBRATION_CHANNELS if ch in series]
        for ch in present_order:
            sig = series[ch][s:e]
            if len(sig) < win_len:
                sig = np.pad(sig, (0, win_len - len(sig)), mode="edge")
            P = compute_cwt_scalogram(sig, fs, scales)
            per_ch_scalos.append(P)
            if wi == 0:
                per_ch_images.append(render_scalogram_image(P, freqs, fs, title=f"Vibration:{ch}"))

        Tm = min(p.shape[1] for p in per_ch_scalos)
        per_ch_scalos = [p[:, :Tm] for p in per_ch_scalos]
        X = np.stack(per_ch_scalos, axis=0)  # (C,F,T)
        groups_4ch.append(X)

        if wi == 0 and per_ch_images:
            preview_images.extend(per_ch_images)

    return groups_4ch, preview_images

# --------------------------- INFERENCE ---------------------------
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

def topk_table(title: str, names: List[str], probs: np.ndarray, k: int = 5) -> List[Tuple[str, float]]:
    order = np.argsort(-probs)[:k]
    return [(names[i], float(probs[i])) for i in order]

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
    if s > 0:
        acc /= s
    return acc.astype(np.float32), classes

# --------------------------- JOB STORAGE ---------------------------
def new_job_dir() -> Path:
    jid = uuid.uuid4().hex[:12]
    d = JOBS_DIR / jid
    d.mkdir(parents=True, exist_ok=True)
    return d

def save_image(path: Path, img_array: np.ndarray):
    img = Image.fromarray(img_array)
    img.save(path, format="PNG")

# --------------------------- ROUTES ---------------------------
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "GET":
        return render_template("index.html")

    # POST: handle uploads, render scalograms only (no prediction yet)
    a_file = request.files.get("acoustic_file")
    v_file = request.files.get("vibration_file")

    if not a_file and not v_file:
        flash("Please upload at least one CSV.")
        return redirect(url_for("index"))

    job_dir = new_job_dir()
    job_id = job_dir.name

    preview_paths = []
    errors = []

    # Process Acoustic
    acoustic_present = False
    if a_file and a_file.filename and allowed_file(a_file.filename):
        fname = secure_filename(a_file.filename)
        raw_path = job_dir / f"acoustic_{fname}"
        a_file.save(raw_path)
        try:
            bytes_data = raw_path.read_bytes()
            a_scalos, a_imgs = csv_to_acoustic_scalos(bytes_data, max_windows=N_WINDOWS_PER_FILE)
            # save preview
            if a_imgs:
                p = job_dir / "preview_acoustic.png"
                save_image(p, a_imgs[0])
                preview_paths.append(("Acoustic scalogram", f"/jobs/{job_id}/{p.name}"))
            # cache scalos to npz
            np.savez_compressed(job_dir / "acoustic_scalos.npz", *a_scalos)
            acoustic_present = True
        except Exception as e:
            errors.append(f"Acoustic error: {e}")
    elif a_file and a_file.filename:
        errors.append("Acoustic file must be a .csv")

    # Process Vibration
    vibration_present = False
    if v_file and v_file.filename and allowed_file(v_file.filename):
        fname = secure_filename(v_file.filename)
        raw_path = job_dir / f"vibration_{fname}"
        v_file.save(raw_path)
        try:
            bytes_data = raw_path.read_bytes()
            v_groups, v_imgs = csv_to_vibration_4ch_scalos(bytes_data, max_windows=N_WINDOWS_PER_FILE)
            # save previews (up to 4)
            for i, img in enumerate(v_imgs[:4]):
                p = job_dir / f"preview_vibration_{i+1}.png"
                save_image(p, img)
                preview_paths.append((f"Vibration scalogram {i+1}", f"/jobs/{job_id}/{p.name}"))
            # cache groups to npz list
            # Store list-of-arrays safely: save each group separately
            group_dir = job_dir / "vibration_groups"
            group_dir.mkdir(exist_ok=True)
            for i, g in enumerate(v_groups):
                np.savez_compressed(group_dir / f"group_{i}.npz", arr=g)
            vibration_present = True
        except Exception as e:
            errors.append(f"Vibration error: {e}")
    elif v_file and v_file.filename:
        errors.append("Vibration file must be a .csv")

    if not acoustic_present and not vibration_present and not errors:
        errors.append("No valid data found in the uploaded CSV(s).")

    status_msg = lazy_load_models()

    return render_template(
        "preview.html",
        job_id=job_id,
        previews=preview_paths,
        errors=errors,
        status_msg=status_msg,
        acoustic_ready=acoustic_present and (_acoustic_model is not None),
        vibration_ready=vibration_present and (_vib_model is not None),
    )

@app.route("/predict", methods=["POST"])
def predict():
    job_id = request.form.get("job_id")
    if not job_id:
        flash("Missing job id.")
        return redirect(url_for("index"))

    job_dir = JOBS_DIR / job_id
    if not job_dir.exists():
        flash("Job not found or expired.")
        return redirect(url_for("index"))

    status_msg = lazy_load_models()

    summaries = []
    details = {}
    ensemble = {}

    acoustic_pred = None
    vibration_pred = None

    # Load cached acoustic scalos
    a_path = job_dir / "acoustic_scalos.npz"
    if a_path.exists() and _acoustic_model is not None:
        try:
            npz = np.load(a_path)
            a_scalos = [npz[k] for k in npz.files]
            p_c, p_s = predict_acoustic_scalos(_acoustic_model, a_scalos)
            c1, c1p = top1(p_c, _acoustic_cond)
            s1, s1p = top1(p_s, _acoustic_sev)
            summaries.append(("Acoustic", c1, c1p, s1, s1p))
            details["acoustic_cond_top"] = topk_table("Acoustic — Condition", _acoustic_cond, p_c, k=min(5, len(_acoustic_cond)))
            details["acoustic_sev_top"]  = topk_table("Acoustic — Severity",  _acoustic_sev,  p_s, k=min(5, len(_acoustic_sev)))
            acoustic_pred = (p_c, _acoustic_cond, p_s, _acoustic_sev)
        except Exception as e:
            summaries.append(("Acoustic error", str(e), 0.0, "", 0.0))

    # Load cached vibration groups
    vg_dir = job_dir / "vibration_groups"
    if vg_dir.exists() and _vib_model is not None:
        try:
            groups = []
            for f in sorted(vg_dir.glob("group_*.npz")):
                arr = np.load(f)["arr"]
                groups.append(arr)
            if groups:
                p_c, p_s = predict_vibration_4ch(_vib_model, groups)
                c1, c1p = top1(p_c, _vib_cond)
                s1, s1p = top1(p_s, _vib_sev)
                summaries.append(("Vibration", c1, c1p, s1, s1p))
                details["vibration_cond_top"] = topk_table("Vibration — Condition", _vib_cond, p_c, k=min(5, len(_vib_cond)))
                details["vibration_sev_top"]  = topk_table("Vibration — Severity",  _vib_sev,  p_s, k=min(5, len(_vib_sev)))
                vibration_pred = (p_c, _vib_cond, p_s, _vib_sev)
        except Exception as e:
            summaries.append(("Vibration error", str(e), 0.0, "", 0.0))

    # Ensemble
    if acoustic_pred and vibration_pred:
        a_pc, a_cnames, a_ps, a_snames = acoustic_pred
        v_pc, v_cnames, v_ps, v_snames = vibration_pred
        ens_c, ens_cnames = ensemble_union([(a_pc, a_cnames), (v_pc, v_cnames)])
        ens_s, ens_snames = ensemble_union([(a_ps, a_snames), (v_ps, v_snames)])
        if ens_c.size:
            ec, ecp = top1(ens_c, ens_cnames)
            ensemble["cond_top1"] = (ec, ecp)
            ensemble["cond_table"] = topk_table("Ensemble — Condition", ens_cnames, ens_c, k=min(5, len(ens_cnames)))
        if ens_s.size:
            es, esp = top1(ens_s, ens_snames)
            ensemble["sev_top1"] = (es, esp)
            ensemble["sev_table"] = topk_table("Ensemble — Severity", ens_snames, ens_s, k=min(5, len(ens_snames)))

    # Collect preview images to keep them visible
    previews = []
    for p in sorted(job_dir.glob("preview_*.png")):
        previews.append((p.stem.replace("_", " ").title(), f"/jobs/{job_id}/{p.name}"))
    if (job_dir / "preview_acoustic.png").exists():
        p = job_dir / "preview_acoustic.png"
        previews.insert(0, ("Acoustic Scalogram", f"/jobs/{job_id}/{p.name}"))

    return render_template(
        "results.html",
        job_id=job_id,
        previews=previews,
        status_msg=status_msg,
        summaries=summaries,
        details=details,
        ensemble=ensemble
    )

@app.route("/jobs/<job_id>/<path:filename>")
def jobs_files(job_id, filename):
    directory = JOBS_DIR / job_id
    return send_from_directory(directory, filename, as_attachment=False)

@app.route("/cleanup/<job_id>", methods=["POST"])
def cleanup(job_id):
    d = JOBS_DIR / job_id
    if d.exists():
        shutil.rmtree(d, ignore_errors=True)
    flash("Session cleaned up.")
    return redirect(url_for("index"))

# --------------------------- MAIN ---------------------------
if __name__ == "__main__":
    print(f"Device: {DEVICE} | CUDA: {USE_CUDA} | AMP: {AMP_ENABLED}")
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 7860)), debug=True)
