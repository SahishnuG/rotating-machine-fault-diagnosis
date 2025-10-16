# train_multimodal_lstm.py
import os, glob, joblib
import numpy as np
import pandas as pd
from scipy import interpolate
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks

# -------- CONFIG --------
ACOUSTIC_DIR = "acoustic"
VIBRATION_DIR = "vibration"
CURRENT_DIR = "current_temp"
MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)

TARGET_FS = 200.0        # Hz - target sampling frequency for all modalities
WINDOW_SECS = 5.0        # seconds -> window length
SEQ_LEN = int(TARGET_FS * WINDOW_SECS)
HORIZON = 1              # predict 1 next sample
BATCH_SIZE = 32
EPOCHS = 30
TEST_SIZE = 0.2
RANDOM_STATE = 42

AC_CH = 1
VB_CH = 4
CT_CH = 5
PRED_DIM = AC_CH + VB_CH + CT_CH  # 10

# -------- helper functions --------
def basename(path): return os.path.splitext(os.path.basename(path))[0]

def load_csv_signal(path, expected_channels=None):
    df = pd.read_csv(path)
    if "Time Stamp" not in df.columns:
        raise ValueError(f"No 'Time Stamp' in {path}")
    t = df["Time Stamp"].values.astype(float)
    # drop metadata columns if present
    meta_cols = [c for c in ["load","condition","severity"] if c in df.columns]
    data_cols = [c for c in df.columns if c not in (["Time Stamp"] + meta_cols)]
    data = df[data_cols].to_numpy(dtype=float)
    if expected_channels is not None:
        if data.shape[1] < expected_channels:
            # pad with NaNs
            pad = np.full((data.shape[0], expected_channels - data.shape[1]), np.nan)
            data = np.concatenate([data, pad], axis=1)
        elif data.shape[1] > expected_channels:
            data = data[:, :expected_channels]
    return t, data, df

def resample_to_fs(t, x, target_fs, start=None, stop=None):
    if start is None: start = t[0]
    if stop is None:  stop = t[-1]
    dt = 1.0/target_fs
    t_new = np.arange(start, stop, dt)
    # prepare x: fill NaNs
    x_clean = x.copy()
    for c in range(x_clean.shape[1]):
        col = x_clean[:, c]
        if np.all(np.isnan(col)):
            x_clean[:,c] = 0.0
            continue
        mask = np.isnan(col)
        if np.any(mask):
            xp = np.where(~mask)[0]
            fp = col[~mask]
            # set edges
            first, last = xp[0], xp[-1]
            col[:first] = col[first]
            col[last+1:] = col[last]
            # interpolate internal
            nan_idx = np.where(mask)[0]
            col[mask] = np.interp(nan_idx, xp, fp)
            x_clean[:,c] = col
    x_new = np.zeros((len(t_new), x_clean.shape[1]), dtype=float)
    for c in range(x_clean.shape[1]):
        f = interpolate.interp1d(t, x_clean[:,c], kind='linear', bounds_error=False, fill_value="extrapolate")
        x_new[:,c] = f(t_new)
    return t_new, x_new

def parse_condition_from_df(dfs):
    # preference order: current temp df, vibration df, acoustic df
    for df in dfs:
        if df is None: continue
        if "condition" in df.columns:
            return str(df["condition"].iloc[0])
    return "unknown"

# -------- collect file lists & common IDs --------
ac_files = {basename(p): p for p in glob.glob(os.path.join(ACOUSTIC_DIR, "*.csv"))}
vb_files = {basename(p): p for p in glob.glob(os.path.join(VIBRATION_DIR, "*.csv"))}
ct_files = {basename(p): p for p in glob.glob(os.path.join(CURRENT_DIR, "*.csv"))}

common_ids = sorted(list(set(ac_files.keys()) & set(vb_files.keys()) & set(ct_files.keys())))
print("Common experiments:", len(common_ids))

# -------- build dataset windows --------
X_ac, X_vb, X_ct, Y, labels = [], [], [], [], []
for idx in common_ids:
    try:
        t_ac, ac, df_ac = load_csv_signal(ac_files[idx], expected_channels=AC_CH)
        t_vb, vb, df_vb = load_csv_signal(vb_files[idx], expected_channels=VB_CH)
        t_ct, ct, df_ct = load_csv_signal(ct_files[idx], expected_channels=CT_CH)
        start = max(t_ac[0], t_vb[0], t_ct[0])
        stop  = min(t_ac[-1], t_vb[-1], t_ct[-1])
        if stop <= start + 1e-9:
            print("Skip (no overlap):", idx); continue
        t_new, ac_r = resample_to_fs(t_ac, ac, TARGET_FS, start, stop)
        _, vb_r = resample_to_fs(t_vb, vb, TARGET_FS, start, stop)
        _, ct_r = resample_to_fs(t_ct, ct, TARGET_FS, start, stop)
        N = len(t_new)
        if N < SEQ_LEN + HORIZON: 
            print("Too short:", idx); continue
        step = SEQ_LEN // 2
        for s in range(0, N - SEQ_LEN - HORIZON + 1, step):
            ac_win = ac_r[s:s+SEQ_LEN, :AC_CH]
            vb_win = vb_r[s:s+SEQ_LEN, :VB_CH]
            ct_win = ct_r[s:s+SEQ_LEN, :CT_CH]
            y_ac = ac_r[s+SEQ_LEN:s+SEQ_LEN+HORIZON, :AC_CH]
            y_vb = vb_r[s+SEQ_LEN:s+SEQ_LEN+HORIZON, :VB_CH]
            y_ct = ct_r[s+SEQ_LEN:s+SEQ_LEN+HORIZON, :CT_CH]
            # flatten next-sample values (HORIZON=1 -> straightforward)
            y_vec = np.concatenate([y_ac.reshape(-1), y_vb.reshape(-1), y_ct.reshape(-1)], axis=0)
            X_ac.append(ac_win); X_vb.append(vb_win); X_ct.append(ct_win)
            Y.append(y_vec)
            labels.append(parse_condition_from_df([df_ct, df_vb, df_ac]))
    except Exception as e:
        print("Error", idx, e)

X_ac = np.array(X_ac); X_vb = np.array(X_vb); X_ct = np.array(X_ct); Y = np.array(Y)
print("Shapes:", X_ac.shape, X_vb.shape, X_ct.shape, Y.shape)
if X_ac.shape[0] == 0:
    raise SystemExit("No windows created. Check data or parameters.")

# -------- train/val split --------
idxs = np.arange(X_ac.shape[0])
tr_idx, val_idx = train_test_split(idxs, test_size=TEST_SIZE, random_state=RANDOM_STATE)
X_ac_tr, X_ac_val = X_ac[tr_idx], X_ac[val_idx]
X_vb_tr, X_vb_val = X_vb[tr_idx], X_vb[val_idx]
X_ct_tr, X_ct_val = X_ct[tr_idx], X_ct[val_idx]
Y_tr, Y_val = Y[tr_idx], Y[val_idx]

# -------- scale data per-modality (fit on training only) --------
def fit_scaler_and_transform(X):
    ns, sl, ch = X.shape
    flat = X.reshape(-1, ch)
    scaler = StandardScaler().fit(flat)
    flat_s = scaler.transform(flat)
    return scaler, flat_s.reshape(ns, sl, ch)

scaler_ac, X_ac_tr_s = fit_scaler_and_transform(X_ac_tr)
scaler_vb, X_vb_tr_s = fit_scaler_and_transform(X_vb_tr)
scaler_ct, X_ct_tr_s = fit_scaler_and_transform(X_ct_tr)

def transform_with(scaler, X):
    ns, sl, ch = X.shape
    return scaler.transform(X.reshape(-1, ch)).reshape(ns, sl, ch)

X_ac_val_s = transform_with(scaler_ac, X_ac_val)
X_vb_val_s = transform_with(scaler_vb, X_vb_val)
X_ct_val_s = transform_with(scaler_ct, X_ct_val)

# scale targets
Y_scaler = StandardScaler().fit(Y_tr.reshape(-1, Y_tr.shape[-1]))
Y_tr_s = Y_scaler.transform(Y_tr.reshape(-1, Y_tr.shape[-1])).reshape(Y_tr.shape)
Y_val_s = Y_scaler.transform(Y_val.reshape(-1, Y_val.shape[-1])).reshape(Y_val.shape)

# -------- build multimodal LSTM model --------
def build_model(seq_len, ac_ch, vb_ch, ct_ch, pred_dim):
    ac_in = layers.Input(shape=(seq_len, ac_ch), name="acoustic_input")
    x1 = layers.LSTM(128, return_sequences=True)(ac_in)
    x1 = layers.LSTM(64)(x1)
    x1 = layers.Dense(32, activation="relu")(x1)

    vb_in = layers.Input(shape=(seq_len, vb_ch), name="vibration_input")
    x2 = layers.LSTM(256, return_sequences=True)(vb_in)
    x2 = layers.LSTM(128)(x2)
    x2 = layers.Dense(64, activation="relu")(x2)

    ct_in = layers.Input(shape=(seq_len, ct_ch), name="current_input")
    x3 = layers.LSTM(256, return_sequences=True)(ct_in)
    x3 = layers.LSTM(128)(x3)
    x3 = layers.Dense(64, activation="relu")(x3)

    fused = layers.Concatenate()([x1, x2, x3])
    f = layers.Dense(128, activation="relu")(fused)
    f = layers.Dense(64, activation="relu")(f)
    out = layers.Dense(pred_dim, activation="linear", name="forecast")(f)

    model = models.Model(inputs=[ac_in, vb_in, ct_in], outputs=out)
    model.compile(optimizer="adam", loss="mse")
    return model

model = build_model(SEQ_LEN, AC_CH, VB_CH, CT_CH, PRED_DIM)
model.summary()

# -------- train --------
ckpt = callbacks.ModelCheckpoint(os.path.join(MODELS_DIR, "multimodal_lstm_best.h5"), save_best_only=True, monitor="val_loss")
es = callbacks.EarlyStopping(monitor="val_loss", patience=6, restore_best_weights=True)

history = model.fit(
    {"acoustic_input": X_ac_tr_s, "vibration_input": X_vb_tr_s, "current_input": X_ct_tr_s},
    Y_tr_s,
    validation_data=(
        {"acoustic_input": X_ac_val_s, "vibration_input": X_vb_val_s, "current_input": X_ct_val_s},
        Y_val_s
    ),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=[ckpt, es]
)

# -------- save model + scalers + metadata --------
model.save(os.path.join(MODELS_DIR, "multimodal_lstm_final.h5"))
joblib.dump({"acoustic": scaler_ac, "vibration": scaler_vb, "current": scaler_ct}, os.path.join(MODELS_DIR, "input_scalers.joblib"))
joblib.dump(Y_scaler, os.path.join(MODELS_DIR, "target_scaler.joblib"))
joblib.dump({"SEQ_LEN": SEQ_LEN, "TARGET_FS": TARGET_FS, "AC_CH": AC_CH, "VB_CH": VB_CH, "CT_CH": CT_CH}, os.path.join(MODELS_DIR, "model_meta.joblib"))

print("Training finished. Models and scalers saved in", MODELS_DIR)
