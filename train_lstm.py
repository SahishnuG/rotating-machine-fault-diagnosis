#!/usr/bin/env python3
"""
train_multimodal_lstm_forecast.py

Train a multimodal LSTM that forecasts FUTURE_SECS of all modalities (acoustic, vibration, current)
so those predicted signals can later be passed to the CNN classifier.

Outputs:
 - models/multimodal_lstm_forecast_final.h5
 - models/multimodal_lstm_forecast_artifacts.joblib

Requirements:
 pip install pandas numpy scipy scikit-learn tensorflow joblib
"""
import os, glob, math, random, joblib
import numpy as np
import pandas as pd
from scipy import interpolate
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks

# ------------- CONFIG -------------
ACOUSTIC_DIR = "acoustic"
VIBRATION_DIR = "vibration"
CURRENT_DIR = "current_temp"
MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)

TARGET_FS = 100.0      # Hz (lower if memory/CPU issues)
WINDOW_SECS = 2.0      # input window length (seconds)
FUTURE_SECS = 2.0      # how far in future to predict (seconds) — we predict same length as input
SEQ_LEN = int(TARGET_FS * WINDOW_SECS)
HORIZON = int(TARGET_FS * FUTURE_SECS)   # number of future timesteps predicted
STEP = SEQ_LEN // 2

AC_CH, VB_CH, CT_CH = 1, 4, 5
TOTAL_OUT_CH = AC_CH + VB_CH + CT_CH   # total channels predicted per timestep

MAX_WINDOWS_PER_EXP = 300
MAX_WINDOWS_TOTAL = 20000

BATCH_SIZE = 32
EPOCHS = 10
TEST_SIZE = 0.2
RANDOM_STATE = 42

TIME_COL = "Time Stamp"

np.random.seed(RANDOM_STATE)
random.seed(RANDOM_STATE)
tf.random.set_seed(RANDOM_STATE)

# ------------- helpers -------------
def basename_noext(p): return os.path.splitext(os.path.basename(p))[0]

def read_csv_signal(path, expected_channels=None, nrows=None):
    df = pd.read_csv(path, nrows=nrows)
    if TIME_COL not in df.columns:
        raise ValueError(f"No '{TIME_COL}' in {path}")
    t = df[TIME_COL].to_numpy(dtype=float)
    meta_cols = [c for c in ("load","condition","severity") if c in df.columns]
    data_cols = [c for c in df.columns if c not in ([TIME_COL] + meta_cols)]
    x = df[data_cols].to_numpy(dtype=float)
    if expected_channels is not None:
        if x.shape[1] < expected_channels:
            pad = np.full((x.shape[0], expected_channels - x.shape[1]), np.nan)
            x = np.concatenate([x, pad], axis=1)
        else:
            x = x[:, :expected_channels]
    return t, x, df

def resample_to_fs(t, x, target_fs, start=None, stop=None):
    if start is None: start = t[0]
    if stop is None: stop = t[-1]
    dt = 1.0 / target_fs
    if stop <= start:
        return np.array([]), np.zeros((0, x.shape[1]), dtype=float)
    t_new = np.arange(start, stop, dt)
    if t_new.size == 0:
        return np.array([]), np.zeros((0, x.shape[1]), dtype=float)
    x_clean = x.copy()
    for ch in range(x_clean.shape[1]):
        col = x_clean[:, ch].astype(float)
        mask = np.isnan(col)
        if np.all(mask):
            col[:] = 0.0
        else:
            if np.any(mask):
                idx = np.arange(len(col)); good = idx[~mask]; vals = col[~mask]
                col[:good[0]] = vals[0]; col[good[-1]+1:] = vals[-1]
                col[mask] = np.interp(idx[mask], good, vals)
        x_clean[:, ch] = col
    x_new = np.zeros((len(t_new), x_clean.shape[1]), dtype=float)
    for ch in range(x_clean.shape[1]):
        f = interpolate.interp1d(t, x_clean[:, ch], kind='linear', bounds_error=False, fill_value="extrapolate")
        x_new[:, ch] = f(t_new)
    return t_new, x_new

def windows_and_future_from_resampled(x_res, seq_len=SEQ_LEN, step=STEP, horizon=HORIZON):
    # returns X_windows shape (n, seq_len, ch), Y_futures shape (n, horizon, ch)
    N = x_res.shape[0]
    Xw, Yw = [], []
    for s in range(0, N - seq_len - horizon + 1, step):
        Xw.append(x_res[s:s+seq_len, :])
        Yw.append(x_res[s+seq_len:s+seq_len+horizon, :])
    if len(Xw) == 0:
        return np.zeros((0, seq_len, x_res.shape[1]), dtype=np.float32), np.zeros((0, horizon, x_res.shape[1]), dtype=np.float32)
    return np.array(Xw, dtype=np.float32), np.array(Yw, dtype=np.float32)

# ------------- build file maps -------------
ac_files = {basename_noext(p): p for p in glob.glob(os.path.join(ACOUSTIC_DIR, "*.csv"))}
vb_files = {basename_noext(p): p for p in glob.glob(os.path.join(VIBRATION_DIR, "*.csv"))}
ct_files = {basename_noext(p): p for p in glob.glob(os.path.join(CURRENT_DIR, "*.csv"))}
all_ids = sorted(list(set(list(ac_files.keys()) + list(vb_files.keys()) + list(ct_files.keys()))))
print("Found experiments:", len(all_ids))

# ------------- collect windows (allow missing modalities) -------------
Xac, Xvb, Xct, masks, Yfutures, exp_ids = [], [], [], [], [], []

def pad_repeat(Xw, desired, ch):
    if Xw.shape[0] == 0:
        return np.zeros((desired, SEQ_LEN, ch), dtype=np.float32)
    if Xw.shape[0] >= desired:
        return Xw[:desired]
    last = Xw[-1]
    pads = np.stack([last] * (desired - Xw.shape[0]), axis=0)
    return np.concatenate([Xw, pads], axis=0)

for eid in all_ids:
    ac_path = ac_files.get(eid)
    vb_path = vb_files.get(eid)
    ct_path = ct_files.get(eid)

    # find overlap (or union)
    starts, stops = [], []
    try:
        if ac_path:
            t_ac, _, _ = read_csv_signal(ac_path, expected_channels=AC_CH)
            starts.append(t_ac[0]); stops.append(t_ac[-1])
    except: pass
    try:
        if vb_path:
            t_vb, _, _ = read_csv_signal(vb_path, expected_channels=VB_CH)
            starts.append(t_vb[0]); stops.append(t_vb[-1])
    except: pass
    try:
        if ct_path:
            t_ct, _, _ = read_csv_signal(ct_path, expected_channels=CT_CH)
            starts.append(t_ct[0]); stops.append(t_ct[-1])
    except: pass

    if len(starts) == 0:
        continue

    start = max(starts)
    stop = min(stops) if len(stops) > 0 else max(stops)
    if stop <= start + 1e-9:
        # fallback to union
        start = min(starts); stop = max(stops)
    if stop - start < (WINDOW_SECS + FUTURE_SECS):
        # not enough room to form input+future
        continue

    # resample present modalities
    ac_r = vb_r = ct_r = None
    try:
        if ac_path:
            _, ac_r = resample_to_fs(*read_csv_signal(ac_path, expected_channels=AC_CH)[:2], TARGET_FS, start, stop)
        if vb_path:
            _, vb_r = resample_to_fs(*read_csv_signal(vb_path, expected_channels=VB_CH)[:2], TARGET_FS, start, stop)
        if ct_path:
            _, ct_r = resample_to_fs(*read_csv_signal(ct_path, expected_channels=CT_CH)[:2], TARGET_FS, start, stop)
    except Exception:
        continue

    # choose minimum length among present ones and trim
    lens = [arr.shape[0] for arr in (ac_r, vb_r, ct_r) if arr is not None]
    if len(lens) == 0: continue
    n = min(lens)
    if n < SEQ_LEN + HORIZON:
        continue
    if ac_r is not None: ac_r = ac_r[:n]
    if vb_r is not None: vb_r = vb_r[:n]
    if ct_r is not None: ct_r = ct_r[:n]

    # windows for each modality
    Xac_w, Yac_w = (windows_and_future_from_resampled(ac_r, SEQ_LEN, STEP, HORIZON) if ac_r is not None else (np.zeros((0,SEQ_LEN,AC_CH),dtype=np.float32), np.zeros((0,HORIZON,AC_CH))))
    Xvb_w, Yvb_w = (windows_and_future_from_resampled(vb_r, SEQ_LEN, STEP, HORIZON) if vb_r is not None else (np.zeros((0,SEQ_LEN,VB_CH),dtype=np.float32), np.zeros((0,HORIZON,VB_CH))))
    Xct_w, Yct_w = (windows_and_future_from_resampled(ct_r, SEQ_LEN, STEP, HORIZON) if ct_r is not None else (np.zeros((0,SEQ_LEN,CT_CH),dtype=np.float32), np.zeros((0,HORIZON,CT_CH))))

    # number of windows to use
    nwin = max(Xac_w.shape[0], Xvb_w.shape[0], Xct_w.shape[0])
    if nwin == 0: continue
    nwin = min(nwin, MAX_WINDOWS_PER_EXP)

    Xac_w = pad_repeat(Xac_w, nwin, AC_CH)
    Xvb_w = pad_repeat(Xvb_w, nwin, VB_CH)
    Xct_w = pad_repeat(Xct_w, nwin, CT_CH)

    # targets: ensure shape (nwin, HORIZON, ch)
    def pad_future(Yw, desired, ch):
        if Yw.shape[0] == 0:
            return np.zeros((desired, HORIZON, ch), dtype=np.float32)
        if Yw.shape[0] >= desired:
            return Yw[:desired]
        last = Yw[-1]
        pads = np.stack([last] * (desired - Yw.shape[0]), axis=0)
        return np.concatenate([Yw, pads], axis=0)

    Yac_w = pad_future(Yac_w, nwin, AC_CH)
    Yvb_w = pad_future(Yvb_w, nwin, VB_CH)
    Yct_w = pad_future(Yct_w, nwin, CT_CH)

    # append to global lists
    mask = np.array([1.0 if ac_path else 0.0, 1.0 if vb_path else 0.0, 1.0 if ct_path else 0.0], dtype=np.float32)
    for i in range(nwin):
        Xac.append(Xac_w[i])
        Xvb.append(Xvb_w[i])
        Xct.append(Xct_w[i])
        # Yfutures combined channels (horizon, total_ch)
        Yfutures.append(np.concatenate([Yac_w[i], Yvb_w[i], Yct_w[i]], axis=1))  # axis=1 concat channels per timestep
        masks.append(mask)
        exp_ids.append(eid)

    if len(Xac) >= MAX_WINDOWS_TOTAL:
        break

# convert to numpy arrays
total = len(Xac)
if total == 0:
    raise SystemExit("No training windows collected. Adjust parameters or check files.")

if total > MAX_WINDOWS_TOTAL:
    sel = np.random.choice(total, size=MAX_WINDOWS_TOTAL, replace=False)
    Xac = [Xac[i] for i in sel]; Xvb=[Xvb[i] for i in sel]; Xct=[Xct[i] for i in sel]; Yfutures=[Yfutures[i] for i in sel]; masks=[masks[i] for i in sel]

Xac = np.array(Xac, dtype=np.float32)
Xvb = np.array(Xvb, dtype=np.float32)
Xct = np.array(Xct, dtype=np.float32)
Yfutures = np.array(Yfutures, dtype=np.float32)    # shape (N, HORIZON, TOTAL_OUT_CH)
masks = np.array(masks, dtype=np.float32)
exp_ids = np.array(exp_ids)

print("Collected:", Xac.shape, Xvb.shape, Xct.shape, Yfutures.shape, masks.shape)

# ------------- split train/val by experiment (avoid leakage) -------------
unique_exps = sorted(list(set(exp_ids)))
train_exps, val_exps = train_test_split(unique_exps, test_size=TEST_SIZE, random_state=RANDOM_STATE)
train_mask = np.isin(exp_ids, train_exps)
val_mask = np.isin(exp_ids, val_exps)

Xac_tr, Xvb_tr, Xct_tr = Xac[train_mask], Xvb[train_mask], Xct[train_mask]
masks_tr = masks[train_mask]
Y_tr = Yfutures[train_mask]

Xac_val, Xvb_val, Xct_val = Xac[val_mask], Xvb[val_mask], Xct[val_mask]
masks_val = masks[val_mask]
Y_val = Yfutures[val_mask]

print("Train windows:", Xac_tr.shape[0], "Val windows:", Xac_val.shape[0])

# ------------- per-modality scaling (fit on training inputs) -------------
from sklearn.preprocessing import StandardScaler
def fit_scaler_and_transform(X):
    ns, sl, ch = X.shape
    flat = X.reshape(-1, ch)
    scaler = StandardScaler().fit(flat)
    Xs = scaler.transform(flat).astype(np.float32).reshape(ns, sl, ch)
    return scaler, Xs

sc_ac, Xac_tr_s = fit_scaler_and_transform(Xac_tr)
sc_vb, Xvb_tr_s = fit_scaler_and_transform(Xvb_tr)
sc_ct, Xct_tr_s = fit_scaler_and_transform(Xct_tr)

def apply_scaler(scaler, X):
    ns, sl, ch = X.shape
    return scaler.transform(X.reshape(-1, ch)).astype(np.float32).reshape(ns, sl, ch)

Xac_val_s = apply_scaler(sc_ac, Xac_val)
Xvb_val_s = apply_scaler(sc_vb, Xvb_val)
Xct_val_s = apply_scaler(sc_ct, Xct_val)

# Scale targets per-output-channel (fit on training targets)
ns, horizon, totch = Y_tr.shape
Y_flat = Y_tr.reshape(-1, totch)
sc_y = StandardScaler().fit(Y_flat)
Y_tr_s = sc_y.transform(Y_flat).reshape(ns, horizon, totch).astype(np.float32)
Y_val_s = sc_y.transform(Y_val.reshape(-1, totch)).reshape(Y_val.shape).astype(np.float32)

# ------------- build model (branches -> fusion -> dense -> reshape horizon x channels) -------------
def lstm_branch(seq_len, nch, name):
    inp = layers.Input(shape=(seq_len, nch))
    x = layers.LSTM(128, return_sequences=True)(inp)
    x = layers.LSTM(64)(x)
    x = layers.Dense(64, activation="relu")(x)
    return models.Model(inp, x, name=name)

ac_branch = lstm_branch(SEQ_LEN, AC_CH, "ac_branch")
vb_branch = lstm_branch(SEQ_LEN, VB_CH, "vb_branch")
ct_branch = lstm_branch(SEQ_LEN, CT_CH, "ct_branch")

ac_in = layers.Input(shape=(SEQ_LEN, AC_CH), name="acoustic_input")
vb_in = layers.Input(shape=(SEQ_LEN, VB_CH), name="vibration_input")
ct_in = layers.Input(shape=(SEQ_LEN, CT_CH), name="current_input")
mask_in = layers.Input(shape=(3,), name="mask_input")

ac_feat = ac_branch(ac_in)
vb_feat = vb_branch(vb_in)
ct_feat = ct_branch(ct_in)

# small custom mask extractor
class MaskScalarLayer(layers.Layer):
    def __init__(self, index, **kwargs):
        super().__init__(**kwargs)
        self.index = int(index)
    def call(self, x):
        return tf.expand_dims(x[:, self.index], axis=1)
    def get_config(self):
        cfg = super().get_config(); cfg.update({"index": self.index}); return cfg

ms0 = MaskScalarLayer(0)(mask_in)
ms1 = MaskScalarLayer(1)(mask_in)
ms2 = MaskScalarLayer(2)(mask_in)

ac_feat = layers.Multiply()([ac_feat, ms0])
vb_feat = layers.Multiply()([vb_feat, ms1])
ct_feat = layers.Multiply()([ct_feat, ms2])

fusion = layers.Concatenate()([ac_feat, vb_feat, ct_feat, mask_in])
x = layers.Dense(512, activation="relu")(fusion)
x = layers.Dropout(0.3)(x)
x = layers.Dense(256, activation="relu")(x)
# predict flattened horizon * channels
out_units = HORIZON * TOTAL_OUT_CH
out = layers.Dense(out_units, activation="linear", name="forecast_flat")(x)
# reshape to (horizon, tot_ch)
out_seq = layers.Reshape((HORIZON, TOTAL_OUT_CH), name="forecast_seq")(out)

model = models.Model([ac_in, vb_in, ct_in, mask_in], out_seq)
model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="mse")
model.summary()

# ------------- datasets -------------
def make_dataset(Xa, Xv, Xc, masks_in, y_in, batch_size=BATCH_SIZE, training=True):
    ds = tf.data.Dataset.from_tensor_slices(((Xa, Xv, Xc, masks_in), y_in))
    if training:
        ds = ds.shuffle(4096, seed=RANDOM_STATE)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds

train_ds = make_dataset(Xac_tr_s, Xvb_tr_s, Xct_tr_s, masks_tr, Y_tr_s, training=True)
val_ds = make_dataset(Xac_val_s, Xvb_val_s, Xct_val_s, masks_val, Y_val_s, training=False)

# ------------- callbacks & train -------------
ckpt_path = os.path.join(MODELS_DIR, "multimodal_lstm_forecast_best.h5")
ckpt = callbacks.ModelCheckpoint(ckpt_path, save_best_only=True, monitor="val_loss", mode="min")
es = callbacks.EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True)

history = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, callbacks=[ckpt, es])

# ------------- save model and artifacts -------------
model.save(os.path.join(MODELS_DIR, "multimodal_lstm_forecast_final.h5"))
joblib.dump({
    "acoustic_scaler": sc_ac,
    "vibration_scaler": sc_vb,
    "current_scaler": sc_ct,
    "y_scaler": sc_y,
    "meta": {"SEQ_LEN": SEQ_LEN, "TARGET_FS": TARGET_FS, "HORIZON": HORIZON, "AC_CH": AC_CH, "VB_CH": VB_CH, "CT_CH": CT_CH}
}, os.path.join(MODELS_DIR, "multimodal_lstm_forecast_artifacts.joblib"))

print("Saved forecast model and artifacts to", MODELS_DIR)
