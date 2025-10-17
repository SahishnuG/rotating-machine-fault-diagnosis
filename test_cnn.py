#!/usr/bin/env python3
"""
test_cnn.py

Safer test loader for the multimodal CNN:
- Rebuilds a clean architecture (no Lambda).
- Loads weights from the original saved .h5 using load_weights(..., by_name=True).
- Performs a smoke-test inference on up to TEST_N experiments.

Usage: python test_cnn.py
"""
import os
import glob
import json
import numpy as np
import pandas as pd
from scipy import interpolate
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models

# ---------------- CONFIG (must match training) ----------------
MODELS_DIR = "models"
CLEAN_MODEL_PATH = os.path.join(MODELS_DIR, "stream_multimodal_clean.h5")
ORIG_MODEL_PATH  = os.path.join(MODELS_DIR, "stream_multimodal_final.h5")  # contains weights; may contain Lambda in config
LABEL_MAP_PATH   = os.path.join(MODELS_DIR, "label_map.json")

ACOUSTIC_DIR = "acoustic"
VIBRATION_DIR = "vibration"
CURRENT_DIR = "current_temp"

TARGET_FS  = 100.0
WINDOW_SECS = 2.0
SEQ_LEN = int(TARGET_FS * WINDOW_SECS)
AC_CH, VB_CH, CT_CH = 1, 4, 5

MAX_TEST_ROWS = 20000
TEST_N = 20

# ---------------- small custom MaskSliceLayer (no Lambda) ----------------
class MaskSliceLayer(keras.layers.Layer):
    def __init__(self, index, **kwargs):
        super().__init__(**kwargs)
        self.index = int(index)
    def call(self, x):
        return tf.expand_dims(x[:, self.index], axis=1)
    def get_config(self):
        cfg = super().get_config()
        cfg.update({"index": self.index})
        return cfg

# ---------------- conv branch & clean model builder (must match training topology) ----------------
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

def build_clean_model(seq_len=SEQ_LEN, ac_ch=AC_CH, vb_ch=VB_CH, ct_ch=CT_CH, n_classes=2, lr=1e-3):
    ac_branch = conv_branch(seq_len, ac_ch)
    vb_branch = conv_branch(seq_len, vb_ch)
    ct_branch = conv_branch(seq_len, ct_ch)

    ac_in = layers.Input(shape=(seq_len, ac_ch), name="acoustic_input")
    vb_in = layers.Input(shape=(seq_len, vb_ch), name="vibration_input")
    ct_in = layers.Input(shape=(seq_len, ct_ch), name="current_input")
    mask_in = layers.Input(shape=(3,), name="mask_input")

    ac_feat = ac_branch(ac_in)
    vb_feat = vb_branch(vb_in)
    ct_feat = ct_branch(ct_in)

    ac_mask_scalar = MaskSliceLayer(0, name="mask0")(mask_in)
    vb_mask_scalar = MaskSliceLayer(1, name="mask1")(mask_in)
    ct_mask_scalar = MaskSliceLayer(2, name="mask2")(mask_in)

    ac_feat = layers.Multiply()([ac_feat, ac_mask_scalar])
    vb_feat = layers.Multiply()([vb_feat, vb_mask_scalar])
    ct_feat = layers.Multiply()([ct_feat, ct_mask_scalar])

    fusion = layers.Concatenate()([ac_feat, vb_feat, ct_feat, mask_in])
    x = layers.Dense(256, activation='relu')(fusion)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(128, activation='relu')(x)
    out = layers.Dense(n_classes, activation='softmax')(x)

    model = models.Model([ac_in, vb_in, ct_in, mask_in], out)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr),
                  loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# ---------------- helper: conservative first-window reader ----------------
def first_window_from_csv(path, expected_channels, target_fs=TARGET_FS, window_secs=WINDOW_SECS, max_rows=MAX_TEST_ROWS):
    if path is None:
        return None
    try:
        df = pd.read_csv(path, nrows=max_rows)
    except Exception:
        return None
    if "Time Stamp" not in df.columns:
        return None
    meta_cols = [c for c in ("load","condition","severity") if c in df.columns]
    data_cols = [c for c in df.columns if c not in (["Time Stamp"] + meta_cols)]
    if len(data_cols) == 0:
        return None
    t = df["Time Stamp"].to_numpy(dtype=float)
    x = df[data_cols].to_numpy(dtype=float)

    if x.shape[1] < expected_channels:
        pad = np.full((x.shape[0], expected_channels - x.shape[1]), np.nan)
        x = np.concatenate([x, pad], axis=1)
    elif x.shape[1] > expected_channels:
        x = x[:, :expected_channels]

    start = t[0]
    stop = start + window_secs
    dt = 1.0 / target_fs
    t_new = np.arange(start, stop, dt)
    if t_new.size == 0:
        return None

    x_new = np.zeros((len(t_new), expected_channels), dtype=np.float32)
    for ch in range(expected_channels):
        col = x[:, ch].astype(float)
        mask = np.isnan(col)
        if np.all(mask):
            col[:] = 0.0
        else:
            if np.any(mask):
                idx = np.arange(len(col)); good = idx[~mask]; vals = col[~mask]
                col[:good[0]] = vals[0]; col[good[-1]+1:] = vals[-1]
                col[mask] = np.interp(idx[mask], good, vals)
        f = interpolate.interp1d(t, col, kind='linear', bounds_error=False, fill_value="extrapolate")
        x_new[:, ch] = f(t_new).astype(np.float32)

    if x_new.shape[0] < SEQ_LEN:
        pad_rows = SEQ_LEN - x_new.shape[0]
        x_new = np.vstack([x_new, np.zeros((pad_rows, expected_channels), dtype=np.float32)])
    elif x_new.shape[0] > SEQ_LEN:
        x_new = x_new[:SEQ_LEN, :]
    return x_new

def standardize_window(win):
    m = np.nanmean(win, axis=0, keepdims=True)
    s = np.nanstd(win, axis=0, keepdims=True)
    s[s <= 1e-6] = 1.0
    out = (win - m) / s
    out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
    return out.astype(np.float32)

# ---------------- helper: infer classes safely ----------------
def infer_classes_from_labelmap_or_files():
    # prefer label_map.json
    if os.path.exists(LABEL_MAP_PATH):
        with open(LABEL_MAP_PATH, "r") as f:
            class_to_idx = json.load(f)
        idx_to_class = {int(v): k for k, v in class_to_idx.items()}
        return idx_to_class
    # else scan first rows of csvs to collect 'condition' values
    def basename_noext(p): return os.path.splitext(os.path.basename(p))[0]
    files = []
    for d in (CURRENT_DIR, VIBRATION_DIR, ACOUSTIC_DIR):
        files += glob.glob(os.path.join(d, "*.csv"))
    classes = set()
    for p in files:
        try:
            df = pd.read_csv(p, nrows=1)
            if "condition" in df.columns:
                classes.add(str(df["condition"].iloc[0]))
        except Exception:
            pass
    if len(classes) == 0:
        # fallback
        return None
    classes = sorted(list(classes))
    idx_to_class = {i: c for i, c in enumerate(classes)}
    return idx_to_class

# ---------------- main ----------------
def main():
    # discover experiment IDs
    def basename_noext(p): return os.path.splitext(os.path.basename(p))[0]
    ac_files = {basename_noext(p): p for p in glob.glob(os.path.join(ACOUSTIC_DIR, "*.csv"))}
    vb_files = {basename_noext(p): p for p in glob.glob(os.path.join(VIBRATION_DIR, "*.csv"))}
    ct_files = {basename_noext(p): p for p in glob.glob(os.path.join(CURRENT_DIR, "*.csv"))}
    all_ids = sorted(list(set(list(ac_files.keys()) + list(vb_files.keys()) + list(ct_files.keys()))))
    print("Found experiment IDs:", len(all_ids))
    if len(all_ids) == 0:
        print("No files found. Exiting."); return

    # 1) try to load an already-clean model
    model = None
    if os.path.exists(CLEAN_MODEL_PATH):
        try:
            print("Loading clean model:", CLEAN_MODEL_PATH)
            model = keras.models.load_model(CLEAN_MODEL_PATH, custom_objects={"MaskSliceLayer": MaskSliceLayer})
        except Exception as e:
            print("Failed to load clean model (continuing):", e)

    # 2) if clean missing, rebuild and load weights from HDF5 by-name
    idx_to_class = infer_classes_from_labelmap_or_files()
    if idx_to_class is not None:
        n_classes = len(idx_to_class)
    else:
        n_classes = 6  # fallback; adjust if necessary
        idx_to_class = {i: str(i) for i in range(n_classes)}

    if model is None:
        print("Rebuilding clean model with n_classes =", n_classes)
        model = build_clean_model(n_classes=n_classes)
        # Try loading weights directly from original HDF5 (by_name)
        if os.path.exists(ORIG_MODEL_PATH):
            try:
                print("Attempting to load weights from original HDF5 (by_name=True, skip_mismatch=True).")
                model.load_weights(ORIG_MODEL_PATH, by_name=True, skip_mismatch=True)
                print("Weights loaded (by_name). Saving clean model for future runs:", CLEAN_MODEL_PATH)
                model.save(CLEAN_MODEL_PATH)
            except Exception as e:
                print("model.load_weights(by_name=True) failed:", e)
                print("At this point you can either: (A) export weights from the original model using a trusted environment,"
                      " or (B) allow unsafe deserialization once (not recommended).")
                raise SystemExit("Failed to load weights from HDF5. Aborting.")
        else:
            raise SystemExit(f"No original model file found at {ORIG_MODEL_PATH} to load weights from.")

    print("Model ready. Summary:")
    model.summary()

    # smoke test on the first TEST_N experiments
    test_ids = all_ids[:min(TEST_N, len(all_ids))]
    for i, eid in enumerate(test_ids, 1):
        ac_path = ac_files.get(eid)
        vb_path = vb_files.get(eid)
        ct_path = ct_files.get(eid)

        ac_win = first_window_from_csv(ac_path, AC_CH) if ac_path else None
        vb_win = first_window_from_csv(vb_path, VB_CH) if vb_path else None
        ct_win = first_window_from_csv(ct_path, CT_CH) if ct_path else None

        if ac_win is None and vb_win is None and ct_win is None:
            print(f"[{i}/{len(test_ids)}] {eid} - no usable data (skipped)")
            continue

        if ac_win is None:
            ac_win = np.zeros((SEQ_LEN, AC_CH), dtype=np.float32)
        if vb_win is None:
            vb_win = np.zeros((SEQ_LEN, VB_CH), dtype=np.float32)
        if ct_win is None:
            ct_win = np.zeros((SEQ_LEN, CT_CH), dtype=np.float32)

        ac_win = standardize_window(ac_win)
        vb_win = standardize_window(vb_win)
        ct_win = standardize_window(ct_win)

        mask = np.array([[1.0 if ac_path else 0.0, 1.0 if vb_path else 0.0, 1.0 if ct_path else 0.0]], dtype=np.float32)

        preds = model.predict([np.expand_dims(ac_win,0), np.expand_dims(vb_win,0), np.expand_dims(ct_win,0), mask], verbose=0)
        top_idx = int(np.argmax(preds, axis=1)[0])
        top_label = idx_to_class.get(top_idx, str(top_idx))
        top_prob = float(np.max(preds))
        print(f"[{i}/{len(test_ids)}] {eid} -> {top_label} (p={top_prob:.3f})")

    print("Done.")

if __name__ == "__main__":
    main()
