#!/usr/bin/env python3
"""
ts_forecast_from_scalograms_fast.py

Fast comparative forecasting from saved scalograms (.npy in scalograms/<modality>/).

- Balanced sampling: exactly K files per CONDITION
- Early thinning: optional frequency crop + cap time columns + stride-decimate columns
- PCA over frequency bins (fit on train-only), target = PC1
- Windows: X (seq_len × N_FEAT), y (horizon)
- Models: Naive, ARIMA (single-fit fast), LSTM, GRU, CNN-LSTM
- Prints a clear "[INFO] Starting neural model training..." line before NN training begins

Run:
  python ts_forecast_from_scalograms_fast.py
"""

import os, glob
from pathlib import Path
from typing import List, Tuple
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error, mean_absolute_error
from statsmodels.tsa.arima.model import ARIMA
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ==================== KNOBS (speed/size) ====================
SCALOGRAM_ROOT = "scalograms"
MODALITIES = ["acoustic", "vibration"]

# Balanced sampling
FILES_PER_CONDITION = 4      # exactly this many per condition

# Time axis throttling
MAX_TIME_COLS_PER_FILE = 20000   # cap T per file (None = keep full)
COL_STRIDE = 8                   # keep every Nth column (>=1)

# Frequency throttling (rows)
FREQ_SLICE = (None, 128)         # (start, end) on freq axis; (None, None) keeps all

# PCA features from frequency axis
N_FEAT = 8
EPS = 1e-12

# Supervised framing
SEQ_LEN  = 64
HORIZON  = 8
STEP     = 8

# Splits (chronological by time)
TRAIN_FRAC = 0.70
VAL_FRAC   = 0.15  # rest = test

# Deep training
EPOCHS       = 8
BATCH_SIZE   = 128
LR           = 1e-3
HIDDEN_SIZE  = 64
NUM_LAYERS   = 1
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"

# Repro & threading
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
if DEVICE == "cuda":
    torch.cuda.manual_seed_all(SEED)
torch.set_num_threads(max(1, os.cpu_count() // 2))

# ==================== HELPERS ====================
def _parse_condition_from_fname(npy_path: str) -> str:
    """
    Expect names like: <load>_<condition>_<severity>__... e.g. 0Nm_BPFI_03__values__win00010.npy
    Also fixes 'Unbalalnce' -> 'Unbalance'.
    """
    stem = Path(npy_path).stem
    main = stem.split("__")[0]  # "0Nm_BPFI_03"
    parts = main.split("_")
    cond = parts[1] if len(parts) >= 2 else "UNKNOWN"
    return "Unbalance" if cond.lower() == "unbalalnce" else cond

def scan_balanced_per_condition(modality: str, k_per_cond: int) -> List[str]:
    """
    Return up to k_per_cond scalogram files per condition for a given modality.
    """
    folder = os.path.join(SCALOGRAM_ROOT, modality)
    all_paths = sorted(glob.glob(os.path.join(folder, "*.npy")))
    if not all_paths:
        return []
    from collections import defaultdict
    groups = defaultdict(list)
    for p in all_paths:
        cond = _parse_condition_from_fname(p)
        groups[cond].append(p)
    rng = np.random.default_rng(SEED)
    selected = []
    for cond, paths in groups.items():
        rng.shuffle(paths)
        selected.extend(paths[:k_per_cond])
    rng.shuffle(selected)
    return selected

def load_power(path: str) -> np.ndarray:
    arr = np.load(path, allow_pickle=False)
    if arr.ndim != 2:
        raise ValueError(f"Scalogram must be 2D (F,T): {path}")
    return arr.astype(np.float32)

def normalize_log(power: np.ndarray) -> np.ndarray:
    X = np.log10(power + EPS).astype(np.float32)
    mn, mx = X.min(), X.max()
    if mx > mn:
        X = (X - mn) / (mx - mn)
    else:
        X.fill(0.0)
    return X

def concat_time_matrix(paths: List[str]) -> np.ndarray:
    """
    Load multiple scalograms (F,T), apply freq crop, time cap, time stride, normalize,
    and concatenate along time axis -> (F, sumT).
    """
    chunks = []
    fs, fe = FREQ_SLICE
    for p in paths:
        try:
            P = load_power(p)  # (F,T)
            if fs is not None or fe is not None:
                P = P[slice(fs, fe), :]
            if MAX_TIME_COLS_PER_FILE and P.shape[1] > MAX_TIME_COLS_PER_FILE:
                P = P[:, :MAX_TIME_COLS_PER_FILE]
            if COL_STRIDE and COL_STRIDE > 1:
                P = P[:, ::COL_STRIDE]
            P = normalize_log(P)
            chunks.append(P)
        except Exception as e:
            print(f"[WARN] skip {Path(p).name}: {e}")
    if not chunks:
        raise RuntimeError("No valid scalograms loaded.")
    F = chunks[0].shape[0]
    for c in chunks:
        if c.shape[0] != F:
            raise ValueError("Inconsistent frequency-bin count across files. Ensure consistent generation settings.")
    return np.concatenate(chunks, axis=1)

def chrono_split_time(P: np.ndarray):
    """
    Split P (F,T) chronologically by time axis into train/val/test.
    """
    T = P.shape[1]
    n_train = int(T * TRAIN_FRAC)
    n_val   = int(T * (TRAIN_FRAC + VAL_FRAC))
    return P[:, :n_train], P[:, n_train:n_val], P[:, n_val:], n_train, n_val, T - n_val

def build_supervised(X_feat: np.ndarray, y_uni: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    X_feat: (T, N_FEAT)
    y_uni:  (T,)
    Returns:
      X: (N, SEQ_LEN, N_FEAT)
      y: (N, HORIZON)
    """
    N = X_feat.shape[0]
    Xs, Ys = [], []
    end = N - (SEQ_LEN + HORIZON) + 1
    for i in range(0, max(0, end), STEP):
        Xs.append(X_feat[i:i+SEQ_LEN, :])
        Ys.append(y_uni[i+SEQ_LEN:i+SEQ_LEN+HORIZON])
    if not Xs:
        return np.zeros((0, SEQ_LEN, X_feat.shape[1]), np.float32), np.zeros((0, HORIZON), np.float32)
    return np.stack(Xs).astype(np.float32), np.stack(Ys).astype(np.float32)

def rmse(a, b): return float(np.sqrt(mean_squared_error(a, b)))
def mae(a, b):  return float(mean_absolute_error(a, b))
def mape(a, b):
    denom = np.maximum(np.abs(a), 1e-8)
    return float(np.mean(np.abs((a - b)/denom)))*100.0

# ==================== DATASETS ====================
class SeqDataset(Dataset):
    def __init__(self, X, y): self.X, self.y = X, y
    def __len__(self): return len(self.X)
    def __getitem__(self, i):
        return torch.from_numpy(self.X[i]), torch.from_numpy(self.y[i])

# ==================== MODELS ====================
class LSTMForecaster(nn.Module):
    def __init__(self, input_size, hidden=64, layers=1, horizon=1):
        super().__init__()
        self.rnn = nn.LSTM(input_size, hidden, num_layers=layers, batch_first=True)
        self.head = nn.Linear(hidden, horizon)
    def forward(self, x):
        out, _ = self.rnn(x)    # (B,T,H)
        return self.head(out[:, -1, :])

class GRUForecaster(nn.Module):
    def __init__(self, input_size, hidden=64, layers=1, horizon=1):
        super().__init__()
        self.rnn = nn.GRU(input_size, hidden, num_layers=layers, batch_first=True)
        self.head = nn.Linear(hidden, horizon)
    def forward(self, x):
        out, _ = self.rnn(x)
        return self.head(out[:, -1, :])

class CNNLSTMForecaster(nn.Module):
    def __init__(self, input_size, hidden=64, layers=1, horizon=1,
                 conv_channels=64, kernel_size=5, dropout=0.1):
        super().__init__()
        pad = kernel_size // 2
        self.conv = nn.Conv1d(input_size, conv_channels, kernel_size, padding=pad)
        self.bn   = nn.BatchNorm1d(conv_channels)
        self.act  = nn.ReLU()
        self.do   = nn.Dropout(dropout)
        self.rnn  = nn.LSTM(conv_channels, hidden, num_layers=layers, batch_first=True)
        self.head = nn.Linear(hidden, horizon)
    def forward(self, x):
        x = x.transpose(1, 2)           # (B,F,T)
        x = self.do(self.act(self.bn(self.conv(x))))
        x = x.transpose(1, 2)           # (B,T,Cc)
        out, _ = self.rnn(x)
        return self.head(out[:, -1, :])

# ==================== TRAIN / INFER ====================
def train_model(model, dl_tr, dl_va, epochs=8, lr=1e-3, device="cpu"):
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    crit = nn.MSELoss()
    best = None; best_val = float("inf"); patience=3; wait=0
    for ep in range(1, epochs+1):
        model.train(); run=0.0; n=0
        for xb, yb in dl_tr:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(); pred = model(xb)
            loss = crit(pred, yb); loss.backward(); opt.step()
            run += loss.item()*xb.size(0); n += xb.size(0)
        tr = run/max(1,n)

        model.eval(); run=0.0; n=0
        with torch.no_grad():
            for xb, yb in dl_va:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb); loss = crit(pred, yb)
                run += loss.item()*xb.size(0); n += xb.size(0)
        va = run/max(1,n)
        print(f"[{model.__class__.__name__}] ep {ep:02d}  train={tr:.5f}  val={va:.5f}")

        if va < best_val - 1e-6:
            best_val = va; best = {k:v.cpu().clone() for k,v in model.state_dict().items()}; wait=0
        else:
            wait += 1
            if wait >= patience:
                print("  early stop"); break
    if best is not None: model.load_state_dict(best)
    return model

def infer_model(model, X, device="cpu"):
    model.eval(); outs=[]
    with torch.no_grad():
        for i in range(0, len(X), 1024):
            xb = torch.from_numpy(X[i:i+1024]).to(device)
            outs.append(model(xb).cpu().numpy())
    return np.vstack(outs) if outs else np.zeros((0, X.shape[1]))

def naive_forecast(Xte, horizon):
    last_pc1 = Xte[:, -1, 0:1]
    return np.repeat(last_pc1, horizon, axis=1)

# ==================== FAST ARIMA (single fit) ====================
def arima_single_fit_forecasts(y_train: np.ndarray, len_val: int, len_test: int, total_extra: int) -> np.ndarray:
    """
    Fit ARIMA ONCE to y_train and forecast a long sequence:
      length = len_val + len_test + total_extra
    Returns array of that length. We will slice windows' horizons out of this vector.
    """
    order_candidates = [(1,0,0), (1,1,0), (2,0,0), (2,1,0)]
    best_fit = None; best_aic = float("inf")
    for order in order_candidates:
        try:
            r = ARIMA(y_train, order=order).fit(method_kwargs={"warn_convergence": False})
            if r.aic < best_aic:
                best_aic = r.aic; best_fit = r
        except Exception:
            continue
    if best_fit is None:
        best_fit = ARIMA(y_train, order=(1,0,0)).fit(method_kwargs={"warn_convergence": False})

    steps = len_val + len_test + total_extra
    long_fcst = best_fit.forecast(steps=steps)  # from end of train
    return np.asarray(long_fcst, dtype=np.float32)

def arima_slice_windows_from_long_fcst(long_fcst: np.ndarray, len_val: int, seq_len: int, horizon: int, n_test_windows: int) -> np.ndarray:
    """
    Map the precomputed long forecast to each test window's horizon:
      For test window i (0-indexed), the horizon starts at: offset = len_val + seq_len + i
    """
    preds = np.zeros((n_test_windows, horizon), dtype=np.float32)
    for i in range(n_test_windows):
        start = len_val + seq_len + i
        end   = start + horizon
        if end <= len(long_fcst):
            preds[i, :] = long_fcst[start:end]
        else:
            # pad by repeating last element if forecast too short
            tail = long_fcst[start:] if start < len(long_fcst) else np.array([long_fcst[-1]], dtype=np.float32)
            fill = np.full(horizon - len(tail), tail[-1], dtype=np.float32)
            preds[i, :] = np.concatenate([tail, fill])
    return preds

# ==================== CORE PIPELINE ====================
def run_modality(modality: str):
    print(f"\n=== {modality.upper()} from scalograms ===")
    paths = scan_balanced_per_condition(modality, FILES_PER_CONDITION)
    if not paths:
        print(f"[WARN] no scalograms in {SCALOGRAM_ROOT}/{modality}")
        return

    # Build (F,T) after early thinning
    P = concat_time_matrix(paths)          # (F, T_total)
    F, T = P.shape
    print(f"Loaded {len(paths)} files -> matrix {F}×{T} (F×T)")

    # Chrono split on time
    Ptr, Pva, Pte, ntr, nva, nte = chrono_split_time(P)

    # PCA on train-only (freq->features per time-slice)
    pca = PCA(n_components=min(N_FEAT, F), random_state=SEED)
    pca.fit(Ptr.T)

    Xtr = pca.transform(Ptr.T).astype(np.float32)
    Xva = pca.transform(Pva.T).astype(np.float32)
    Xte = pca.transform(Pte.T).astype(np.float32)

    # Target = PC1
    ytr = Xtr[:, 0]
    yva = Xva[:, 0]
    yte = Xte[:, 0]

    # Supervised windows
    Xtr_w, Ytr_w = build_supervised(Xtr, ytr)
    Xva_w, Yva_w = build_supervised(Xva, yva)
    Xte_w, Yte_w = build_supervised(Xte, yte)
    print(f"Windows -> train={len(Xtr_w)}  val={len(Xva_w)}  test={len(Xte_w)}")

    if len(Xte_w) == 0:
        print("Not enough test windows. Reduce SEQ_LEN/HORIZON or relax caps.")
        return

    results = []

    # Naive baseline
    yhat_naive = naive_forecast(Xte_w, HORIZON)
    results.append(("Naive", yhat_naive))

    # ---------- ARIMA (single-fit, fast) ----------
    # Fit once on ytr, forecast over (len(yva) + len(yte) + seq_len + horizon),
    # then slice horizons per test window.
    long_fcst = arima_single_fit_forecasts(ytr, len(yva), len(yte), total_extra=SEQ_LEN + HORIZON)
    yhat_arima = arima_slice_windows_from_long_fcst(long_fcst, len(yva), SEQ_LEN, HORIZON, n_test_windows=len(Xte_w))
    results.append(("ARIMA(single-fit)", yhat_arima))

    # ---------- Neural nets ----------
    print("[INFO] Starting neural model training (LSTM/GRU/CNN-LSTM)...")  # <- requested print

    dl_tr = DataLoader(SeqDataset(Xtr_w, Ytr_w), batch_size=BATCH_SIZE, shuffle=True)
    dl_va = DataLoader(SeqDataset(Xva_w, Yva_w), batch_size=BATCH_SIZE, shuffle=False)

    # LSTM
    lstm = LSTMForecaster(Xtr_w.shape[2], HIDDEN_SIZE, NUM_LAYERS, HORIZON)
    lstm = train_model(lstm, dl_tr, dl_va, EPOCHS, LR, DEVICE)
    yhat_lstm = infer_model(lstm, Xte_w, DEVICE)
    results.append(("LSTM", yhat_lstm))

    # GRU
    gru = GRUForecaster(Xtr_w.shape[2], HIDDEN_SIZE, NUM_LAYERS, HORIZON)
    gru = train_model(gru, dl_tr, dl_va, EPOCHS, LR, DEVICE)
    yhat_gru = infer_model(gru, Xte_w, DEVICE)
    results.append(("GRU", yhat_gru))

    # CNN-LSTM
    cnnlstm = CNNLSTMForecaster(Xtr_w.shape[2], HIDDEN_SIZE, NUM_LAYERS, HORIZON,
                                conv_channels=HIDDEN_SIZE, kernel_size=5, dropout=0.1)
    cnnlstm = train_model(cnnlstm, dl_tr, dl_va, EPOCHS, LR, DEVICE)
    yhat_cnnlstm = infer_model(cnnlstm, Xte_w, DEVICE)
    results.append(("CNN-LSTM", yhat_cnnlstm))

    # ---------- Evaluate ----------
    def eval_preds(name, pred, Y):
        y_true_flat = Y.reshape(-1); y_pred_flat = pred.reshape(-1)
        print(f"  {name:>15s} | H1  RMSE={rmse(Y[:,0], pred[:,0]):.4f}  MAE={mae(Y[:,0], pred[:,0]):.4f}  MAPE={mape(Y[:,0], pred[:,0]):.2f}%"
              f"   | ALL  RMSE={rmse(y_true_flat, y_pred_flat):.4f}  MAE={mae(y_true_flat, y_pred_flat):.4f}  MAPE={mape(y_true_flat, y_pred_flat):.2f}%")

    print("\nTEST performance (target=PC1):")
    for name, pred in results:
        eval_preds(name, pred, Yte_w)

    # ---------- Plot one test example ----------
    idx = len(Xte_w)//2
    t_base = np.arange(SEQ_LEN); t_fut = np.arange(SEQ_LEN, SEQ_LEN+HORIZON)
    plt.figure(figsize=(9,4), dpi=130)
    plt.plot(t_base, Xte_w[idx,:,0], label="PC1 input")
    plt.plot(t_fut,  Yte_w[idx],     label="true PC1 future", linewidth=2)
    for name, pred in results:
        plt.plot(t_fut, pred[idx], '--', label=name)
    plt.title(f"{modality} | one test example")
    plt.xlabel("Time steps (columns)")
    plt.legend(ncol=3); plt.tight_layout(); plt.show()

# ==================== MAIN ====================
def main():
    for mod in MODALITIES:
        try:
            run_modality(mod)
        except Exception as e:
            print(f"[WARN] Skipping {mod}: {e}")

if __name__ == "__main__":
    main()
