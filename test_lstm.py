# run_multimodal_inference.py
import os, joblib, glob
import numpy as np, pandas as pd
from scipy import interpolate
import tensorflow as tf
from tensorflow.keras.models import load_model

MODELS_DIR = "models"
ACOUSTIC_DIR = "acoustic"
VIBRATION_DIR = "vibration"
CURRENT_DIR = "current_temp"

# -------- load model + scalers + meta --------
model = load_model(os.path.join(MODELS_DIR, "multimodal_lstm_final.h5"), compile=False)
scalers = joblib.load(os.path.join(MODELS_DIR, "input_scalers.joblib"))
Y_scaler = joblib.load(os.path.join(MODELS_DIR, "target_scaler.joblib"))
meta = joblib.load(os.path.join(MODELS_DIR, "model_meta.joblib"))

SEQ_LEN = meta["SEQ_LEN"]
TARGET_FS = meta["TARGET_FS"]
AC_CH = meta["AC_CH"]
VB_CH = meta["VB_CH"]
CT_CH = meta["CT_CH"]
PRED_DIM = AC_CH + VB_CH + CT_CH

def basename(path): return os.path.splitext(os.path.basename(path))[0]

def load_csv_signal(path, expected_channels=None):
    df = pd.read_csv(path)
    t = df["Time Stamp"].values.astype(float)
    meta_cols = [c for c in ["load","condition","severity"] if c in df.columns]
    data_cols = [c for c in df.columns if c not in (["Time Stamp"] + meta_cols)]
    data = df[data_cols].to_numpy(dtype=float)
    if expected_channels is not None:
        if data.shape[1] < expected_channels:
            pad = np.full((data.shape[0], expected_channels - data.shape[1]), np.nan)
            data = np.concatenate([data, pad], axis=1)
        else:
            data = data[:, :expected_channels]
    return t, data, df

def resample_to_fs(t, x, target_fs, start=None, stop=None):
    if start is None: start = t[0]
    if stop is None: stop = t[-1]
    dt = 1.0/target_fs
    t_new = np.arange(start, stop, dt)
    x_clean = x.copy()
    for c in range(x_clean.shape[1]):
        col = x_clean[:,c]
        if np.all(np.isnan(col)):
            x_clean[:,c] = 0.0; continue
        mask = np.isnan(col)
        if np.any(mask):
            xp = np.where(~mask)[0]; fp = col[~mask]
            first, last = xp[0], xp[-1]
            col[:first] = col[first]; col[last+1:] = col[last]
            col[mask] = np.interp(np.where(mask)[0], xp, fp)
            x_clean[:,c] = col
    x_new = np.zeros((len(t_new), x_clean.shape[1]), dtype=float)
    for c in range(x_clean.shape[1]):
        f = interpolate.interp1d(t, x_clean[:,c], kind='linear', bounds_error=False, fill_value="extrapolate")
        x_new[:,c] = f(t_new)
    return t_new, x_new

# -------- choose an experiment ID to run inference on --------
EXAMPLE_ID = None
# auto pick one that exists in all three folders if not provided
ac_list = {basename(p): p for p in glob.glob(os.path.join(ACOUSTIC_DIR, "*.csv"))}
vb_list = {basename(p): p for p in glob.glob(os.path.join(VIBRATION_DIR, "*.csv"))}
ct_list = {basename(p): p for p in glob.glob(os.path.join(CURRENT_DIR, "*.csv"))}
common = sorted(set(ac_list.keys()) & set(vb_list.keys()) & set(ct_list.keys()))
if not common:
    raise SystemExit("No common experiment files found.")
EXAMPLE_ID = EXAMPLE_ID or common[0]
print("Running inference on:", EXAMPLE_ID)

# load CSVs
t_ac, ac, df_ac = load_csv_signal(ac_list[EXAMPLE_ID], expected_channels=AC_CH)
t_vb, vb, df_vb = load_csv_signal(vb_list[EXAMPLE_ID], expected_channels=VB_CH)
t_ct, ct, df_ct = load_csv_signal(ct_list[EXAMPLE_ID], expected_channels=CT_CH)

start = max(t_ac[0], t_vb[0], t_ct[0])
stop  = min(t_ac[-1], t_vb[-1], t_ct[-1])
t_new, ac_r = resample_to_fs(t_ac, ac, TARGET_FS, start, stop)
_, vb_r = resample_to_fs(t_vb, vb, TARGET_FS, start, stop)
_, ct_r = resample_to_fs(t_ct, ct, TARGET_FS, start, stop)
N = len(t_new)
if N < SEQ_LEN + 1:
    raise SystemExit("Example too short for a full window. Choose other file or reduce SEQ_LEN.")

# pick the last full window for inference
s = N - SEQ_LEN - 1
ac_win = ac_r[s:s+SEQ_LEN, :AC_CH]
vb_win = vb_r[s:s+SEQ_LEN, :VB_CH]
ct_win = ct_r[s:s+SEQ_LEN, :CT_CH]

# scale windows using loaded scalers
ac_s = scalers["acoustic"].transform(ac_win.reshape(-1, AC_CH)).reshape(1, SEQ_LEN, AC_CH)
vb_s = scalers["vibration"].transform(vb_win.reshape(-1, VB_CH)).reshape(1, SEQ_LEN, VB_CH)
ct_s = scalers["current"].transform(ct_win.reshape(-1, CT_CH)).reshape(1, SEQ_LEN, CT_CH)

# run model
pred_s = model.predict({"acoustic_input": ac_s, "vibration_input": vb_s, "current_input": ct_s})
# inverse scale
pred = Y_scaler.inverse_transform(pred_s).reshape(-1)

# split prediction back into modalities
ac_pred = pred[0:AC_CH]
vb_pred = pred[AC_CH:AC_CH+VB_CH]
ct_pred = pred[AC_CH+VB_CH:AC_CH+VB_CH+CT_CH]

print("\nPredicted next-sample values (for experiment {}):".format(EXAMPLE_ID))
print(f"Acoustic (Pa): {ac_pred}")
print(f"Vibration (g): {vb_pred}")
print(f"Current/Temp (°C,°C,A,A,A): {ct_pred}")

# Also print the actual next sample (ground-truth) if available
gt_idx = s + SEQ_LEN
if gt_idx < N:
    gt_ac = ac_r[gt_idx:gt_idx+1, :AC_CH].reshape(-1)
    gt_vb = vb_r[gt_idx:gt_idx+1, :VB_CH].reshape(-1)
    gt_ct = ct_r[gt_idx:gt_idx+1, :CT_CH].reshape(-1)
    print("\nGround-truth next-sample values:")
    print(f"Acoustic (Pa): {gt_ac}")
    print(f"Vibration (g): {gt_vb}")
    print(f"Current/Temp (°C,°C,A,A,A): {gt_ct}")
else:
    print("Ground-truth next sample not available.")
