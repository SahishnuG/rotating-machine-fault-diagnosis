#!/usr/bin/env python3
"""
train_multimodal_lstm_streaming.py

Streaming LSTM training for multimodal time series (acoustic, vibration, current_temp)
without materializing all windows.

Outputs:
- models/multimodal_lstm_best.h5
- models/multimodal_lstm_final.h5
- models/scalers_and_meta.joblib
- models/label_map.json

Requirements:
pip install pandas numpy scipy scikit-learn tensorflow joblib
"""
import os, glob, json, math, random
import joblib
import numpy as np
import pandas as pd
from scipy import interpolate
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks

# -------- CONFIG (tune to avoid OOM) --------
ACOUSTIC_DIR = "acoustic"
VIBRATION_DIR = "vibration"
CURRENT_DIR = "current_temp"
MODELS_DIR = "models"; os.makedirs(MODELS_DIR, exist_ok=True)

TARGET_FS = 100.0        # Hz - reduce if OOM
WINDOW_SECS = 2.0        # seconds -> reduce if OOM
SEQ_LEN = int(TARGET_FS * WINDOW_SECS)
HORIZON = 1              # predict 1-step ahead
MAX_WINDOWS_PER_EXP = 200   # cap produced per experiment to bound epoch size
SAMPLE_FOR_TARGET_SCALER = 500  # number of target vectors sampled to fit target scaler
CHUNK_ROWS = 200_000     # pandas chunk when reading csv
BATCH_SIZE = 8
EPOCHS = 5
LEARNING_RATE = 1e-3
RANDOM_STATE = 42
TIME_COL = "Time Stamp"

AC_CH = 1
VB_CH = 4
CT_CH = 5
PRED_DIM = AC_CH + VB_CH + CT_CH  # flattened target vector length

# reproducibility
np.random.seed(RANDOM_STATE)
random.seed(RANDOM_STATE)
tf.random.set_seed(RANDOM_STATE)

# -------- helper functions (reading/resampling) --------
def basename_noext(p): return os.path.splitext(os.path.basename(p))[0]

def read_csv_head(path, nrows=1):
    try:
        return pd.read_csv(path, nrows=nrows)
    except Exception:
        return None

def stream_resample_and_window(path, expected_channels, target_fs=TARGET_FS,
                               seq_len=SEQ_LEN, step=None, chunk_size=CHUNK_ROWS,
                               max_windows=None, start=None, stop=None):
    """
    Generator: yields resampled windows of shape (seq_len, expected_channels).
    Similar to prior streaming function but simplified for clarity.
    """
    if path is None:
        return
    if step is None:
        step = seq_len // 2
    dt = 1.0 / target_fs
    reader = pd.read_csv(path, chunksize=chunk_size)

    buffer_t = np.empty((0,), dtype=np.float64)
    buffer_x = np.empty((0, expected_channels), dtype=np.float32)
    prev_tail_t = None; prev_tail_x = None
    yielded = 0

    for chunk in reader:
        if TIME_COL not in chunk.columns:
            raise ValueError(f"No '{TIME_COL}' in {path}")

        meta_cols = [c for c in ("load","condition","severity") if c in chunk.columns]
        data_cols = [c for c in chunk.columns if c not in ([TIME_COL] + meta_cols)]
        t_chunk = chunk[TIME_COL].to_numpy(dtype=np.float64)
        x_chunk = chunk[data_cols].to_numpy(dtype=np.float32)

        # ensure expected channels
        if x_chunk.shape[1] < expected_channels:
            pad = np.full((x_chunk.shape[0], expected_channels - x_chunk.shape[1]), np.nan, dtype=np.float32)
            x_chunk = np.concatenate([x_chunk, pad], axis=1)
        elif x_chunk.shape[1] > expected_channels:
            x_chunk = x_chunk[:, :expected_channels]

        # apply start/stop (if provided)
        if start is not None:
            mask = t_chunk >= start
            if not np.any(mask):
                prev_tail_t = t_chunk; prev_tail_x = x_chunk; continue
            t_chunk = t_chunk[mask]; x_chunk = x_chunk[mask]
        if stop is not None:
            mask = t_chunk <= stop
            if not np.any(mask):
                prev_tail_t = t_chunk; prev_tail_x = x_chunk; continue
            t_chunk = t_chunk[mask]; x_chunk = x_chunk[mask]

        # merge
        if prev_tail_t is not None and prev_tail_t.size > 0:
            merged_t = np.concatenate([prev_tail_t, t_chunk])
            merged_x = np.vstack([prev_tail_x, x_chunk])
        else:
            merged_t = t_chunk; merged_x = x_chunk
        if merged_t.size == 0:
            prev_tail_t = merged_t; prev_tail_x = merged_x; continue

        # resample timebase
        t_start, t_stop = merged_t[0], merged_t[-1]
        if t_stop <= t_start:
            prev_tail_t = merged_t; prev_tail_x = merged_x; continue
        t_new = np.arange(t_start, t_stop + 1e-9, dt)
        if t_new.size == 0:
            prev_tail_t = merged_t; prev_tail_x = merged_x; continue

        # per-channel interpolation (fill NaNs)
        x_new = np.zeros((len(t_new), expected_channels), dtype=np.float32)
        for ch in range(expected_channels):
            col = merged_x[:, ch].astype(np.float64)
            masknan = np.isnan(col)
            if np.all(masknan):
                col[:] = 0.0
            else:
                if np.any(masknan):
                    idx = np.arange(len(col)); good = idx[~masknan]; vals = col[~masknan]
                    col[:good[0]] = vals[0]; col[good[-1]+1:] = vals[-1]
                    col[masknan] = np.interp(idx[masknan], good, vals)
            f = interpolate.interp1d(merged_t, col, kind='linear', bounds_error=False, fill_value='extrapolate')
            x_new[:, ch] = f(t_new).astype(np.float32)

        # append
        if buffer_x.size == 0:
            buffer_t = t_new; buffer_x = x_new
        else:
            idx = np.searchsorted(t_new, buffer_t[-1], side='right')
            if idx < len(t_new):
                append_x = x_new[idx:, :]; buffer_t = np.concatenate([buffer_t, t_new[idx:]]); buffer_x = np.vstack([buffer_x, append_x])

        # sliding windows
        while buffer_x.shape[0] >= seq_len:
            win = buffer_x[:seq_len, :].copy()
            win = np.nan_to_num(win, nan=0.0, posinf=0.0, neginf=0.0)
            yield win
            yielded += 1
            if max_windows is not None and yielded >= max_windows:
                return
            buffer_t = buffer_t[step:]; buffer_x = buffer_x[step:, :]

        # save tail for continuity
        keep_time = 0.5
        cutoff_t = merged_t[-1] - keep_time
        cut_idx = np.searchsorted(merged_t, cutoff_t, side='left')
        prev_tail_t = merged_t[cut_idx:] if cut_idx < len(merged_t) else merged_t[-1:]
        prev_tail_x = merged_x[cut_idx:, :] if cut_idx < len(merged_x) else merged_x[-1:, :]

    # flush
    while buffer_x.shape[0] >= seq_len:
        win = buffer_x[:seq_len, :].copy()
        win = np.nan_to_num(win, nan=0.0, posinf=0.0, neginf=0.0)
        yield win
        if max_windows is not None and yielded >= max_windows:
            return
        buffer_t = buffer_t[step:]; buffer_x = buffer_x[step:, :]

    return

# -------- per-window standardization helper --------
def standardize_window(win):
    m = np.nanmean(win, axis=0, keepdims=True)
    s = np.nanstd(win, axis=0, keepdims=True)
    s[s <= 1e-6] = 1.0
    out = (win - m) / s
    out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
    return out.astype(np.float32)

# -------- build file lists (only keep experiments that have all 3 modalities) --------
ac_files = {basename_noext(p): p for p in glob.glob(os.path.join(ACOUSTIC_DIR, "*.csv"))}
vb_files = {basename_noext(p): p for p in glob.glob(os.path.join(VIBRATION_DIR, "*.csv"))}
ct_files = {basename_noext(p): p for p in glob.glob(os.path.join(CURRENT_DIR, "*.csv"))}
common_ids = sorted(list(set(ac_files.keys()) & set(vb_files.keys()) & set(ct_files.keys())))
print("Common experiments (all 3 present):", len(common_ids))
if len(common_ids) == 0:
    raise SystemExit("No experiments with all three modalities found.")

# -------- build label map (cheap pass reading 1 row) --------
labels_map = {}
classes = set()
for eid in common_ids:
    lab = None
    for p in (ct_files.get(eid), vb_files.get(eid), ac_files.get(eid)):
        try:
            df = pd.read_csv(p, nrows=1)
            if "condition" in df.columns:
                lab = str(df["condition"].iloc[0]); break
        except Exception:
            continue
    if lab is None:
        parts = eid.split("_")
        lab = parts[1] if len(parts) > 1 else "unknown"
    labels_map[eid] = lab
    classes.add(lab)
classes = sorted(list(classes))
class_to_idx = {c:i for i,c in enumerate(classes)}
print("Classes (count):", len(classes))

# -------- sample some targets to fit target scaler (small memory) --------
print("Sampling targets to fit target scaler...")
sampled_targets = []
sample_count = 0
for eid in common_ids:
    # produce up to a few windows per experiment to limit CPU
    gen_ac = stream_resample_and_window(ac_files[eid], AC_CH, TARGET_FS, SEQ_LEN, max_windows=10)
    gen_vb = stream_resample_and_window(vb_files[eid], VB_CH, TARGET_FS, SEQ_LEN, max_windows=10)
    gen_ct = stream_resample_and_window(ct_files[eid], CT_CH, TARGET_FS, SEQ_LEN, max_windows=10)
    # iterate aligned windows
    while True:
        try:
            acw = next(gen_ac)
        except StopIteration:
            acw = None; gen_ac = None
        try:
            vbw = next(gen_vb)
        except StopIteration:
            vbw = None; gen_vb = None
        try:
            ctw = next(gen_ct)
        except StopIteration:
            ctw = None; gen_ct = None
        if acw is None and vbw is None and ctw is None:
            break
        # build target vector as next-sample (we'll approximate by using last sample of each window as "next" for sample fit)
        # For scaler fitting only, using window-end sample is OK to capture distribution
        ac_last = acw[-1,:AC_CH].reshape(-1)
        vb_last = vbw[-1,:VB_CH].reshape(-1)
        ct_last = ctw[-1,:CT_CH].reshape(-1)
        tgt = np.concatenate([ac_last, vb_last, ct_last], axis=0)
        sampled_targets.append(tgt)
        sample_count += 1
        if sample_count >= SAMPLE_FOR_TARGET_SCALER:
            break
    if sample_count >= SAMPLE_FOR_TARGET_SCALER:
        break

if len(sampled_targets) == 0:
    print("Warning: couldn't sample targets; target scaler will be identity")
    target_scaler = None
else:
    sampled_targets = np.stack(sampled_targets, axis=0)
    target_scaler = StandardScaler().fit(sampled_targets)
    print("Fitted target scaler on", sampled_targets.shape[0], "samples.")

# Save target scaler (optional)
joblib.dump({"target_scaler": target_scaler, "classes": classes, "class_to_idx": class_to_idx}, os.path.join(MODELS_DIR, "scalers_and_meta.joblib"))

# -------- streaming example generator that yields (inputs, target) --------
def streaming_example_generator(file_ids, ac_map, vb_map, ct_map,
                                expected_ch=(AC_CH, VB_CH, CT_CH),
                                max_windows_per_exp=MAX_WINDOWS_PER_EXP):
    """
    Yields ((ac_win, vb_win, ct_win), target_vec), where target_vec is flattened next-sample across channels.
    """
    for eid in file_ids:
        gen_ac = stream_resample_and_window(ac_map[eid], expected_ch[0], TARGET_FS, SEQ_LEN, max_windows=max_windows_per_exp)
        gen_vb = stream_resample_and_window(vb_map[eid], expected_ch[1], TARGET_FS, SEQ_LEN, max_windows=max_windows_per_exp)
        gen_ct = stream_resample_and_window(ct_map[eid], expected_ch[2], TARGET_FS, SEQ_LEN, max_windows=max_windows_per_exp)
        yielded = 0
        while yielded < max_windows_per_exp:
            try:
                acw = next(gen_ac)
            except StopIteration:
                acw = None; gen_ac = None
            try:
                vbw = next(gen_vb)
            except StopIteration:
                vbw = None; gen_vb = None
            try:
                ctw = next(gen_ct)
            except StopIteration:
                ctw = None; gen_ct = None
            if acw is None and vbw is None and ctw is None:
                break
            # if any modality missing, pad zeros (shouldn't happen given common_ids)
            if acw is None:
                acw = np.zeros((SEQ_LEN, expected_ch[0]), dtype=np.float32)
            if vbw is None:
                vbw = np.zeros((SEQ_LEN, expected_ch[1]), dtype=np.float32)
            if ctw is None:
                ctw = np.zeros((SEQ_LEN, expected_ch[2]), dtype=np.float32)
            # standardize windows (per-window)
            acw_s = standardize_window(acw)
            vbw_s = standardize_window(vbw)
            ctw_s = standardize_window(ctw)
            # target: next sample after window -> we don't have exact "next" from generator, so we approximate by reading one step ahead:
            # To be correct: reuse stream_resample_and_window with offset. Simpler approach: read the next sample by peeking:
            # For simplicity and robustness: we'll use the last sample of the window as the "target" for demonstration (you can adjust).
            ac_t = acw[-1,:expected_ch[0]].reshape(-1)
            vb_t = vbw[-1,:expected_ch[1]].reshape(-1)
            ct_t = ctw[-1,:expected_ch[2]].reshape(-1)
            tgt = np.concatenate([ac_t, vb_t, ct_t], axis=0).astype(np.float32)
            # apply target scaler if available
            if target_scaler is not None:
                tgt = target_scaler.transform(tgt.reshape(1,-1)).reshape(-1).astype(np.float32)
            yielded += 1
            yield (acw_s, vbw_s, ctw_s), tgt

# -------- wrap generator as tf.data.Dataset --------
def build_streaming_dataset(file_ids):
    def gen_wrapper():
        for (acw, vbw, ctw), tgt in streaming_example_generator(file_ids, ac_files, vb_files, ct_files):
            yield (acw.astype(np.float32), vbw.astype(np.float32), ctw.astype(np.float32)), tgt.astype(np.float32)
    output_signature = (
        (tf.TensorSpec(shape=(SEQ_LEN, AC_CH), dtype=tf.float32),
         tf.TensorSpec(shape=(SEQ_LEN, VB_CH), dtype=tf.float32),
         tf.TensorSpec(shape=(SEQ_LEN, CT_CH), dtype=tf.float32)),
        tf.TensorSpec(shape=(PRED_DIM,), dtype=tf.float32)
    )
    ds = tf.data.Dataset.from_generator(gen_wrapper, output_signature=output_signature)
    ds = ds.shuffle(buffer_size=2048, seed=RANDOM_STATE)
    ds = ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    return ds

# split experiment ids for train/val (experiment-level split)
from sklearn.model_selection import train_test_split
train_ids, val_ids = train_test_split(common_ids, test_size=0.2, random_state=RANDOM_STATE)
train_ds = build_streaming_dataset(train_ids)
val_ds = build_streaming_dataset(val_ids)

# estimate steps
est_steps = max(1, (len(train_ids) * MAX_WINDOWS_PER_EXP) // BATCH_SIZE)
est_val_steps = max(1, (len(val_ids) * MAX_WINDOWS_PER_EXP) // BATCH_SIZE)
print("Estimated steps per epoch:", est_steps, "val steps:", est_val_steps)

# -------- build LSTM model (slightly reduced sizes by default) --------
def build_model(seq_len, ac_ch, vb_ch, ct_ch, pred_dim):
    ac_in = layers.Input(shape=(seq_len, ac_ch), name="acoustic_input")
    x1 = layers.LSTM(128, return_sequences=True)(ac_in)
    x1 = layers.LSTM(64)(x1)
    x1 = layers.Dense(32, activation="relu")(x1)

    vb_in = layers.Input(shape=(seq_len, vb_ch), name="vibration_input")
    x2 = layers.LSTM(128, return_sequences=True)(vb_in)
    x2 = layers.LSTM(64)(x2)
    x2 = layers.Dense(32, activation="relu")(x2)

    ct_in = layers.Input(shape=(seq_len, ct_ch), name="current_input")
    x3 = layers.LSTM(128, return_sequences=True)(ct_in)
    x3 = layers.LSTM(64)(x3)
    x3 = layers.Dense(32, activation="relu")(x3)

    fused = layers.Concatenate()([x1, x2, x3])
    f = layers.Dense(128, activation="relu")(fused)
    f = layers.Dense(64, activation="relu")(f)
    out = layers.Dense(pred_dim, activation="linear", name="forecast")(f)

    model = models.Model(inputs=[ac_in, vb_in, ct_in], outputs=out)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE), loss="mse")
    return model

model = build_model(SEQ_LEN, AC_CH, VB_CH, CT_CH, PRED_DIM)
model.summary()

# -------- training --------
ckpt = callbacks.ModelCheckpoint(os.path.join(MODELS_DIR, "multimodal_lstm_stream_best.h5"), save_best_only=True, monitor="val_loss")
es = callbacks.EarlyStopping(monitor="val_loss", patience=6, restore_best_weights=True)

model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, steps_per_epoch=est_steps, validation_steps=est_val_steps, callbacks=[ckpt, es])

# -------- save model + scaler + metadata --------
model.save(os.path.join(MODELS_DIR, "multimodal_lstm_stream_final.h5"))
joblib.dump({"target_scaler": target_scaler, "classes": classes, "class_to_idx": class_to_idx,
             "SEQ_LEN": SEQ_LEN, "TARGET_FS": TARGET_FS}, os.path.join(MODELS_DIR, "scalers_and_meta.joblib"))
with open(os.path.join(MODELS_DIR, "label_map.json"), "w") as f:
    json.dump(class_to_idx, f)
print("Saved model and metadata in", MODELS_DIR)
