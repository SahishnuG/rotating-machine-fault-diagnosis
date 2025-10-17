#!/usr/bin/env python3
"""
train_cnn_fixed.py

Fixed streaming multimodal CNN training:
- Uses Lambda to extract mask scalars (no KerasTensor->TF misuse)
- Normalizes labels (fixes 'Unbalalnce' typo and NaNs)
- Streaming generator, train/val split, validation monitoring
"""
import os, glob, random, json
import numpy as np, pandas as pd
from scipy import interpolate
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, callbacks

# ---------------- CONFIG ----------------
ACOUSTIC_DIR = "acoustic"
VIBRATION_DIR = "vibration"
CURRENT_DIR = "current_temp"
MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)

TARGET_FS = 100.0
WINDOW_SECS = 2.0
SEQ_LEN = int(TARGET_FS * WINDOW_SECS)
STEP = SEQ_LEN // 2

CHUNK_ROWS = 200_000
MAX_WINDOWS_PER_EXP = 200
SHUFFLE_BUFFER = 2048

BATCH_SIZE = 8
EPOCHS = 10
LEARNING_RATE = 1e-3
RANDOM_STATE = 42

TIME_COL = "Time Stamp"
AC_CH, VB_CH, CT_CH = 1, 4, 5

MAX_RESAMPLE_LEN = 300_000

np.random.seed(RANDOM_STATE)
random.seed(RANDOM_STATE)
tf.random.set_seed(RANDOM_STATE)

# ---------------- streaming resample & window ----------------
def stream_resample_and_window(path, expected_channels, target_fs=TARGET_FS,
                               seq_len=SEQ_LEN, step=STEP, chunk_size=CHUNK_ROWS,
                               overlap_seconds=0.5, max_windows=None, max_resample_len=MAX_RESAMPLE_LEN):
    if path is None:
        return
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

        if x_chunk.shape[1] < expected_channels:
            pad = np.full((x_chunk.shape[0], expected_channels - x_chunk.shape[1]), np.nan, dtype=np.float32)
            x_chunk = np.concatenate([x_chunk, pad], axis=1)
        elif x_chunk.shape[1] > expected_channels:
            x_chunk = x_chunk[:, :expected_channels]

        if prev_tail_t is not None and prev_tail_t.size > 0:
            merged_t = np.concatenate([prev_tail_t, t_chunk])
            merged_x = np.vstack([prev_tail_x, x_chunk])
        else:
            merged_t = t_chunk; merged_x = x_chunk

        if merged_t.size == 0:
            prev_tail_t = merged_t; prev_tail_x = merged_x; continue

        t_start = merged_t[0]; t_stop = merged_t[-1]
        if t_stop <= t_start:
            prev_tail_t = merged_t; prev_tail_x = merged_x; continue

        t_new = np.arange(t_start, t_stop + 1e-9, dt)
        if t_new.size > max_resample_len:
            t_new = t_new[:max_resample_len]
        if t_new.size == 0:
            prev_tail_t = merged_t; prev_tail_x = merged_x; continue

        x_new = np.zeros((len(t_new), expected_channels), dtype=np.float32)
        for ch in range(expected_channels):
            col = merged_x[:, ch].astype(np.float64)
            mask = np.isnan(col)
            if np.all(mask):
                col[:] = 0.0
            else:
                if np.any(mask):
                    idx = np.arange(len(col)); good = idx[~mask]; vals = col[~mask]
                    col[:good[0]] = vals[0]; col[good[-1]+1:] = vals[-1]
                    col[mask] = np.interp(idx[mask], good, vals)
            f = interpolate.interp1d(merged_t, col, kind='linear', bounds_error=False, fill_value="extrapolate")
            x_new[:, ch] = f(t_new).astype(np.float32)

        if buffer_x.size == 0:
            buffer_t = t_new; buffer_x = x_new
        else:
            idx = np.searchsorted(t_new, buffer_t[-1], side="right")
            if idx < len(t_new):
                append_t = t_new[idx:]; append_x = x_new[idx:, :]
                buffer_t = np.concatenate([buffer_t, append_t]); buffer_x = np.vstack([buffer_x, append_x])

        while buffer_x.shape[0] >= seq_len:
            win = buffer_x[:seq_len, :].copy()
            win = np.nan_to_num(win, nan=0.0, posinf=0.0, neginf=0.0)
            yield win
            yielded += 1
            if (max_windows is not None) and (yielded >= max_windows): return
            buffer_t = buffer_t[step:]; buffer_x = buffer_x[step:, :]

        keep_time = overlap_seconds
        cutoff_t = merged_t[-1] - keep_time
        cut_idx = np.searchsorted(merged_t, cutoff_t, side="left")
        prev_tail_t = merged_t[cut_idx:] if cut_idx < len(merged_t) else merged_t[-1:]
        prev_tail_x = merged_x[cut_idx:, :] if cut_idx < len(merged_x) else merged_x[-1:, :]

    while buffer_x.shape[0] >= seq_len:
        win = buffer_x[:seq_len, :].copy()
        win = np.nan_to_num(win, nan=0.0, posinf=0.0, neginf=0.0)
        yield win
        if (max_windows is not None) and (yielded >= max_windows): return
        buffer_t = buffer_t[step:]; buffer_x = buffer_x[step:, :]
    return

# ---------------- per-window standardization ----------------
def standardize_window(win: np.ndarray) -> np.ndarray:
    m = np.nanmean(win, axis=0, keepdims=True)
    s = np.nanstd(win, axis=0, keepdims=True)
    s[s <= 1e-6] = 1.0
    out = (win - m) / s
    out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
    return out.astype(np.float32)

# ---------------- files & labels ----------------
def basename_noext(path): return os.path.splitext(os.path.basename(path))[0]

ac_files = {basename_noext(p): p for p in glob.glob(os.path.join(ACOUSTIC_DIR, "*.csv"))}
vb_files = {basename_noext(p): p for p in glob.glob(os.path.join(VIBRATION_DIR, "*.csv"))}
ct_files = {basename_noext(p): p for p in glob.glob(os.path.join(CURRENT_DIR, "*.csv"))}
all_ids = sorted(list(set(list(ac_files.keys()) + list(vb_files.keys()) + list(ct_files.keys()))))
print(f"Found {len(all_ids)} unique experiment IDs across modalities.")

# label map (normalize known typos and NaNs)
labels_map = {}
classes = set()
for eid in all_ids:
    lab = None
    for p in (ct_files.get(eid), vb_files.get(eid), ac_files.get(eid)):
        if p is None: continue
        try:
            df = pd.read_csv(p, nrows=1)
            if "condition" in df.columns:
                val = df["condition"].iloc[0]
                if pd.isna(val): 
                    lab = "unknown"
                else:
                    lab = str(val).strip()
                break
        except Exception:
            continue
    if lab is None:
        parts = eid.split("_")
        lab = parts[1] if len(parts) > 1 else "unknown"
    # normalize typo and common issues
    if lab.lower() == "unbalalnce":
        lab = "Unbalance"
    if lab.lower() == "nan" or lab == "" or lab.lower() in ("none", "unknown"):
        lab = "unknown"
    labels_map[eid] = lab
    classes.add(lab)

classes = sorted(list(classes))
class_to_idx = {c: i for i, c in enumerate(classes)}
print("Classes:", classes)

random.shuffle(all_ids)

# ---------------- streaming example generator (defensive) ----------------
def streaming_example_generator(file_ids, ac_map, vb_map, ct_map,
                                expected_ch=(AC_CH, VB_CH, CT_CH),
                                max_windows_per_exp=MAX_WINDOWS_PER_EXP):
    def _safe_next(gen):
        if gen is None:
            return None, None
        try:
            v = next(gen)
            return v, gen
        except StopIteration:
            return None, None

    for eid in file_ids:
        ac_path = ac_map.get(eid)
        vb_path = vb_map.get(eid)
        ct_path = ct_map.get(eid)

        gen_ac = stream_resample_and_window(ac_path, expected_ch[0], TARGET_FS, SEQ_LEN, STEP, max_windows=max_windows_per_exp) if ac_path else None
        gen_vb = stream_resample_and_window(vb_path, expected_ch[1], TARGET_FS, SEQ_LEN, STEP, max_windows=max_windows_per_exp) if vb_path else None
        gen_ct = stream_resample_and_window(ct_path, expected_ch[2], TARGET_FS, SEQ_LEN, STEP, max_windows=max_windows_per_exp) if ct_path else None

        label = labels_map.get(eid, "unknown")
        label_idx = class_to_idx.get(label, 0)

        yielded = 0
        while yielded < max_windows_per_exp:
            win_ac, gen_ac = _safe_next(gen_ac)
            win_vb, gen_vb = _safe_next(gen_vb)
            win_ct, gen_ct = _safe_next(gen_ct)

            if win_ac is None and win_vb is None and win_ct is None and gen_ac is None and gen_vb is None and gen_ct is None:
                break

            if win_ac is None:
                win_ac = np.zeros((SEQ_LEN, expected_ch[0]), dtype=np.float32)
            if win_vb is None:
                win_vb = np.zeros((SEQ_LEN, expected_ch[1]), dtype=np.float32)
            if win_ct is None:
                win_ct = np.zeros((SEQ_LEN, expected_ch[2]), dtype=np.float32)

            win_ac_s = standardize_window(win_ac)
            win_vb_s = standardize_window(win_vb)
            win_ct_s = standardize_window(win_ct)

            mask = np.array([1.0 if ac_path else 0.0, 1.0 if vb_path else 0.0, 1.0 if ct_path else 0.0], dtype=np.float32)

            yielded += 1
            yield (win_ac_s, win_vb_s, win_ct_s, mask), np.int32(label_idx)

# ---------------- build streaming datasets ----------------
def build_streaming_dataset(file_ids):
    def gen_wrapper():
        for (acw, vbw, ctw, mask), lab in streaming_example_generator(file_ids, ac_files, vb_files, ct_files):
            yield (acw.astype(np.float32), vbw.astype(np.float32), ctw.astype(np.float32), mask.astype(np.float32)), np.int32(lab)

    output_signature = (
        (tf.TensorSpec(shape=(SEQ_LEN, AC_CH), dtype=tf.float32),
         tf.TensorSpec(shape=(SEQ_LEN, VB_CH), dtype=tf.float32),
         tf.TensorSpec(shape=(SEQ_LEN, CT_CH), dtype=tf.float32),
         tf.TensorSpec(shape=(3,), dtype=tf.float32)),
        tf.TensorSpec(shape=(), dtype=tf.int32)
    )
    ds = tf.data.Dataset.from_generator(gen_wrapper, output_signature=output_signature)
    ds = ds.shuffle(SHUFFLE_BUFFER, seed=RANDOM_STATE).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    return ds

train_ids, val_ids = train_test_split(all_ids, test_size=0.15, random_state=RANDOM_STATE, shuffle=True)
train_ds = build_streaming_dataset(train_ids)
val_ds = build_streaming_dataset(val_ids)

steps_per_epoch = max(1, (len(train_ids) * min(50, MAX_WINDOWS_PER_EXP)) // BATCH_SIZE)
val_steps = max(1, (len(val_ids) * min(20, MAX_WINDOWS_PER_EXP)) // BATCH_SIZE)
print("Estimated steps_per_epoch:", steps_per_epoch, "val_steps:", val_steps)

# ---------------- model definition (use Lambda for masks) ----------------
def conv_branch(seq_len, nch):
    inp = layers.Input(shape=(seq_len, nch))
    x = layers.Conv1D(64, 7, strides=2, padding='same', activation='relu')(inp)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPool1D(2)(x)
    x = layers.Conv1D(128, 5, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPool1D(2)(x)
    x = layers.Conv1D(256, 3, padding='same', activation='relu')(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(128, activation='relu')(x)
    return models.Model(inp, x)

ac_branch = conv_branch(SEQ_LEN, AC_CH)
vb_branch = conv_branch(SEQ_LEN, VB_CH)
ct_branch = conv_branch(SEQ_LEN, CT_CH)

ac_in = layers.Input(shape=(SEQ_LEN, AC_CH))
vb_in = layers.Input(shape=(SEQ_LEN, VB_CH))
ct_in = layers.Input(shape=(SEQ_LEN, CT_CH))
mask_in = layers.Input(shape=(3,))

ac_feat = ac_branch(ac_in)
vb_feat = vb_branch(vb_in)
ct_feat = ct_branch(ct_in)

# Use Lambda to extract mask scalars safely within the Keras graph
ac_mask_scalar = layers.Lambda(lambda x: tf.expand_dims(x[:, 0], axis=1), name="mask0")(mask_in)
vb_mask_scalar = layers.Lambda(lambda x: tf.expand_dims(x[:, 1], axis=1), name="mask1")(mask_in)
ct_mask_scalar = layers.Lambda(lambda x: tf.expand_dims(x[:, 2], axis=1), name="mask2")(mask_in)

ac_feat = layers.Multiply()([ac_feat, ac_mask_scalar])
vb_feat = layers.Multiply()([vb_feat, vb_mask_scalar])
ct_feat = layers.Multiply()([ct_feat, ct_mask_scalar])

fusion = layers.Concatenate()([ac_feat, vb_feat, ct_feat, mask_in])
x = layers.Dense(256, activation='relu')(fusion)
x = layers.Dropout(0.3)(x)
x = layers.Dense(128, activation='relu')(x)
out = layers.Dense(len(classes), activation='softmax')(x)

model = models.Model([ac_in, vb_in, ct_in, mask_in], out)
model.compile(optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.summary()

# ---------------- train ----------------
ckpt_path = os.path.join(MODELS_DIR, "stream_multimodal_best.h5")
ckpt = callbacks.ModelCheckpoint(ckpt_path, save_best_only=True, monitor='val_loss', mode='min')
es = callbacks.EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True)

history = model.fit(train_ds,
                    validation_data=val_ds,
                    epochs=EPOCHS,
                    steps_per_epoch=steps_per_epoch,
                    validation_steps=val_steps,
                    callbacks=[ckpt, es])

# ---------------- save ----------------
final_model_path = os.path.join(MODELS_DIR, "stream_multimodal_final.h5")
model.save(final_model_path)
with open(os.path.join(MODELS_DIR, "label_map.json"), "w") as f:
    json.dump(class_to_idx, f)
print("Saved model to:", final_model_path)
print("Saved label map to:", os.path.join(MODELS_DIR, "label_map.json"))
