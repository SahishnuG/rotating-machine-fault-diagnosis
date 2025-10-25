#!/usr/bin/env python3
"""
gradio_multimodal_app.py

Gradio frontend for test_cnn.py: lets a user upload up to three CSV sensor files
(acoustic, vibration, current), runs the same conservative first-window preproc,
invokes the clean model (loads or rebuilds + loads weights) and returns a
predicted condition + probability. Also plots the standardized windows.

Run: python gradio_multimodal_app.py

"""
import os
import glob
import json
import tempfile
import numpy as np
import pandas as pd
from scipy import interpolate
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
import gradio as gr
import matplotlib.pyplot as plt

# ---------------- CONFIG (keep in sync with original script) ----------------
MODELS_DIR = "models"
CLEAN_MODEL_PATH = os.path.join(MODELS_DIR, "stream_multimodal_clean.h5")
ORIG_MODEL_PATH  = os.path.join(MODELS_DIR, "stream_multimodal_final.h5")
LABEL_MAP_PATH   = os.path.join(MODELS_DIR, "label_map.json")

ACOUSTIC_DIR = "acoustic"
VIBRATION_DIR = "vibration"
CURRENT_DIR = "current_temp"

TARGET_FS  = 100.0
WINDOW_SECS = 2.0
SEQ_LEN = int(TARGET_FS * WINDOW_SECS)
AC_CH, VB_CH, CT_CH = 1, 4, 5

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
def first_window_from_csv(path, expected_channels, target_fs=TARGET_FS, window_secs=WINDOW_SECS, max_rows=20000):
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

# ---------------- helper: infer classes ----------------
def infer_classes_from_labelmap_or_files():
    if os.path.exists(LABEL_MAP_PATH):
        with open(LABEL_MAP_PATH, "r") as f:
            class_to_idx = json.load(f)
        idx_to_class = {int(v): k for k, v in class_to_idx.items()}
        return idx_to_class
    # else fallback: try to scan sample dirs
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
        return None
    classes = sorted(list(classes))
    idx_to_class = {i: c for i, c in enumerate(classes)}
    return idx_to_class

# ---------------- model loader ----------------
_global_model = None
_idx_to_class = None

def get_model():
    global _global_model, _idx_to_class
    if _global_model is not None:
        return _global_model, _idx_to_class

    _idx_to_class = infer_classes_from_labelmap_or_files()
    if _idx_to_class is not None:
        n_classes = len(_idx_to_class)
    else:
        n_classes = 6
        _idx_to_class = {i: str(i) for i in range(n_classes)}

    if os.path.exists(CLEAN_MODEL_PATH):
        try:
            print(f"Loading clean model from {CLEAN_MODEL_PATH}")
            _global_model = keras.models.load_model(CLEAN_MODEL_PATH, custom_objects={"MaskSliceLayer": MaskSliceLayer})
            return _global_model, _idx_to_class
        except Exception as e:
            print("Failed to load clean model:", e)

    # rebuild and try load weights from original HDF5 by-name
    _global_model = build_clean_model(n_classes=n_classes)
    if os.path.exists(ORIG_MODEL_PATH):
        try:
            _global_model.load_weights(ORIG_MODEL_PATH, by_name=True, skip_mismatch=True)
            # save clean copy
            os.makedirs(MODELS_DIR, exist_ok=True)
            _global_model.save(CLEAN_MODEL_PATH)
            return _global_model, _idx_to_class
        except Exception as e:
            raise RuntimeError("Failed to load weights from ORIG_MODEL_PATH: " + str(e))
    else:
        raise FileNotFoundError(f"No model file found at {CLEAN_MODEL_PATH} or {ORIG_MODEL_PATH}")

# ---------------- prediction wrapper used by Gradio ----------------
def predict_from_upload(ac_file, vb_file, ct_file):
    # save uploaded files (they may be file-like objects)
    def save_tmp(f):
        if f is None:
            return None
        if hasattr(f, "name") and os.path.exists(f.name):
            return f.name
        fd, path = tempfile.mkstemp(suffix=".csv")
        os.close(fd)
        # gradio file-like objects may be bytes; handle both
        try:
            data = f.read()
        except Exception:
            # sometimes f is a dict with 'name'
            if isinstance(f, dict) and 'name' in f:
                return f['name']
            return None
        with open(path, "wb") as out:
            if isinstance(data, str):
                out.write(data.encode())
            else:
                out.write(data)
        return path

    ac_path = save_tmp(ac_file)
    vb_path = save_tmp(vb_file)
    ct_path = save_tmp(ct_file)

    ac_win = first_window_from_csv(ac_path, AC_CH) if ac_path else None
    vb_win = first_window_from_csv(vb_path, VB_CH) if vb_path else None
    ct_win = first_window_from_csv(ct_path, CT_CH) if ct_path else None

    if ac_win is None and vb_win is None and ct_win is None:
        # For Gradio Label component, return a dict mapping label->probability
        return {"no data": 0.0}, None

    if ac_win is None:
        ac_win = np.zeros((SEQ_LEN, AC_CH), dtype=np.float32)
    if vb_win is None:
        vb_win = np.zeros((SEQ_LEN, VB_CH), dtype=np.float32)
    if ct_win is None:
        ct_win = np.zeros((SEQ_LEN, CT_CH), dtype=np.float32)

    ac_s = standardize_window(ac_win)
    vb_s = standardize_window(vb_win)
    ct_s = standardize_window(ct_win)

    mask = np.array([[1.0 if ac_path else 0.0, 1.0 if vb_path else 0.0, 1.0 if ct_path else 0.0]], dtype=np.float32)

    model, idx_to_class = get_model()
    preds = model.predict([np.expand_dims(ac_s,0), np.expand_dims(vb_s,0), np.expand_dims(ct_s,0), mask], verbose=0)
    top_idx = int(np.argmax(preds, axis=1)[0])
    top_label = idx_to_class.get(top_idx, str(top_idx))
    top_prob = float(np.max(preds))

    # make a simple plot of the three inputs
    fig, axs = plt.subplots(3, 1, figsize=(6, 6))
    times = np.arange(SEQ_LEN) / TARGET_FS
    axs[0].plot(times, ac_s.squeeze())
    axs[0].set_title(f"Acoustic (channels={AC_CH})")
    axs[1].plot(times, vb_s)
    axs[1].set_title(f"Vibration (channels={VB_CH})")
    axs[2].plot(times, ct_s)
    axs[2].set_title(f"Current (channels={CT_CH})")
    plt.tight_layout()

    # Return dict mapping label->prob (Gradio label expects this format)
    label_out = {str(top_label): float(top_prob)}

    # close figure handling is left to gradio; return fig object
    return label_out, fig

# ---------------- Gradio UI ----------------
with gr.Blocks() as demo:
    gr.Markdown("# Multimodal Signal Predictor\nUpload CSVs for acoustic, vibration, and current signals (optional).\nFiles must contain a `Time Stamp` column and signal columns.")
    with gr.Row():
        with gr.Column():
            ac_in = gr.File(label="Acoustic CSV (1 channel)", file_types=['.csv'])
            vb_in = gr.File(label="Vibration CSV (up to 4 channels)", file_types=['.csv'])
            ct_in = gr.File(label="Current CSV (up to 5 channels)", file_types=['.csv'])
            predict_btn = gr.Button("Predict")
        with gr.Column():
            # Label expects a mapping of label->prob or a list of (label, prob)
            output_label = gr.Label(num_top_classes=3, label="Prediction")
            plot_out = gr.Plot(label="Standardized windows")

    predict_btn.click(fn=predict_from_upload, inputs=[ac_in, vb_in, ct_in], outputs=[output_label, plot_out])

if __name__ == '__main__':
    demo.launch()
