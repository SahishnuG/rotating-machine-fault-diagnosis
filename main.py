#!/usr/bin/env python3
"""
main.py  (updated)

Handles:
 - preprocess: stream vibration CSV in chunks, compute spectrogram windows, extract acoustic features, write TFRecord shards
 - train: load TFRecord shards, create non-overlapping sequences of SEQ_LEN windows, train CNN-LSTM model

Notes:
 - Replace the placeholder 'label' assignment in preprocess with your real RUL/fault label logic.
 - If acoustic file does not fit memory, request the streaming-acoustic fallback (not implemented here).
"""

import os
import argparse
import numpy as np
import pandas as pd
from scipy.signal import spectrogram
from tqdm import tqdm
import math
import tensorflow as tf

# ----------------------------
# CONFIG - tune for your system
# ----------------------------
VIB_PATH = "vibration_csv_data/0Nm_BPFI_03.csv"
AC_PATH  = "acoustic_csv_data/0Nm_BPFI_03.csv"

# Preprocessing params
CHUNKSIZE = 1_000_000  # read CSV rows per chunk
WIN_SIZE = 2048         # samples per spectrogram window (choose to match physical time)
STEP = 1024             # hop size between windows (50% overlap typical)
NPERSEG = 512
NOVERLAP = 256
SPEC_SHAPE = (128, 128)  # (freq_bins, time_bins)
SEQ_LEN = 8              # number of windows per LSTM sequence
TFRECORD_DIR = "tfrecord_shards"
SHARDSIZE = 2000         # examples per TFRecord shard

# Training params
BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 1e-3

# TFRecord keys
FEATURE_SPEC_KEY = "spec"      # float32 spectrogram (h,w,1)
FEATURE_AC_KEY = "ac_feats"    # float32 acoustic features vector
FEATURE_LABEL_KEY = "rul"      # float32 target

# ----------------------------
# UTIL: TFRecord helpers
# ----------------------------
def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value.flatten().tolist()))

def write_tfrecord_examples(shard_index, examples, out_dir):
    """
    Write a list of examples to a TFRecord file.
    Each example is a tuple: (spec_array, ac_feat_array, label_float)
    """
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"shard-{shard_index:04d}.tfrecord")
    with tf.io.TFRecordWriter(path) as writer:
        for spec, ac, label in examples:
            spec = np.asarray(spec, dtype=np.float32)  # shape (h,w,1)
            ac = np.asarray(ac, dtype=np.float32)      # shape (ac_dim,)
            label = np.asarray([label], dtype=np.float32)
            feature = {
                FEATURE_SPEC_KEY: _float_feature(spec),
                FEATURE_AC_KEY: _float_feature(ac),
                FEATURE_LABEL_KEY: _float_feature(label)
            }
            example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
            writer.write(example_proto.SerializeToString())
    return path

# ----------------------------
# PREPROCESS: spectrogram helper
# ----------------------------
def compute_spectrogram(window, fs, nperseg=NPERSEG, noverlap=NOVERLAP, spec_shape=SPEC_SHAPE):
    """
    Compute spectrogram for `window` and resize to spec_shape (freq_bins, time_bins).
    Returns a float32 array shaped (freq_bins, time_bins, 1).
    """
    # safety: ensure window length is sufficient for nperseg
    if len(window) < 2:
        h, w = spec_shape
        return np.zeros((h, w, 1), dtype=np.float32)

    try:
        f, t, Sxx = spectrogram(window, fs=fs, nperseg=nperseg, noverlap=noverlap)
    except Exception:
        # fallback if spectrogram fails for any reason
        h, w = spec_shape
        return np.zeros((h, w, 1), dtype=np.float32)

    if Sxx.size == 0:
        h, w = spec_shape
        return np.zeros((h, w, 1), dtype=np.float32)

    # log-power & normalize per-window
    S = np.log1p(Sxx)
    S_min, S_max = float(S.min()), float(S.max())
    if S_max - S_min > 1e-12:
        S = (S - S_min) / (S_max - S_min)
    else:
        S = np.zeros_like(S)

    # Interpolate along time axis (each row is a frequency bin)
    orig_time_bins = S.shape[1]
    target_time_bins = spec_shape[1]
    if orig_time_bins == target_time_bins:
        S_time = S.copy()
    else:
        time_xp = np.arange(orig_time_bins)
        time_target = np.linspace(0, orig_time_bins - 1, target_time_bins)
        # for each frequency row, interpolate across time
        S_time = np.vstack([np.interp(time_target, time_xp, S[i, :]) for i in range(S.shape[0])])

    # Now S_time shape is (orig_freq_bins, target_time_bins)
    # Interpolate along frequency axis to reach target freq bins
    orig_freq_bins = S_time.shape[0]
    target_freq_bins = spec_shape[0]
    if orig_freq_bins == target_freq_bins:
        S_resized = S_time
    else:
        freq_xp = np.arange(orig_freq_bins)
        freq_target = np.linspace(0, orig_freq_bins - 1, target_freq_bins)
        # for each time column, interpolate across frequency
        S_resized = np.vstack([np.interp(freq_target, freq_xp, S_time[:, j]) for j in range(S_time.shape[1])]).T

    S_resized = np.asarray(S_resized, dtype=np.float32)
    if S_resized.shape != spec_shape:
        S_resized = np.resize(S_resized, spec_shape).astype(np.float32)

    return S_resized[..., np.newaxis]

# ----------------------------
# PREPROCESS: streaming windows -> spectrograms -> TFRecord shards
# ----------------------------
def streaming_preprocess_to_tfrecord(vib_csv, ac_csv, out_dir=TFRECORD_DIR,
                                     chunksize=CHUNKSIZE, win_size=WIN_SIZE, step=STEP,
                                     seq_len=SEQ_LEN, spec_shape=SPEC_SHAPE, shardsize=SHARDSIZE):
    """
    Streams CSVs in chunks, builds sliding windows for vibration, computes spectrograms,
    extracts acoustic features aligned to window centers, and writes TFRecord shards.
    """

    # Try to load acoustic in memory (ok for a few million rows ~ tens of MB)
    try:
        ac_df = pd.read_csv(ac_csv, usecols=["Time_s", "Acoustic_Pa"])
        ac_times = ac_df["Time_s"].values
        ac_vals = ac_df["Acoustic_Pa"].values
        ac_loaded_in_memory = True
        print(f"Loaded acoustic into memory: {len(ac_vals)} samples")
    except Exception as e:
        print("Warning: acoustic file couldn't be loaded entirely; switching to fallback (zero features). Exception:", e)
        ac_df = None
        ac_times = None
        ac_vals = None
        ac_loaded_in_memory = False

    # Rolling buffer for vibration windows
    rolling = np.empty((0,), dtype=np.float32)
    rolling_times = np.empty((0,), dtype=np.float64)
    shard_idx = 0
    examples_buffer = []

    # Estimate sample_rate using first two rows
    sample_rate = None
    tmp = pd.read_csv(vib_csv, usecols=["Time_s"], nrows=2).reset_index(drop=True)
    if len(tmp) >= 2:
        dt = tmp["Time_s"].iloc[1] - tmp["Time_s"].iloc[0]
        sample_rate = 1.0 / dt if dt > 0 else 1.0
    if sample_rate is None:
        sample_rate = 1.0

    print(f"Estimated sample_rate = {sample_rate:.1f} Hz")

    chunk_no = 0
    # Stream vibration in chunks
    for vib_chunk in pd.read_csv(vib_csv, usecols=["Time_s", "Vibration"], iterator=True, chunksize=chunksize):
        chunk_no += 1
        vib_vals = vib_chunk["Vibration"].values.astype(np.float32)
        vib_times = vib_chunk["Time_s"].values.astype(np.float64)
        # append to rolling buffer
        rolling = np.concatenate((rolling, vib_vals))
        rolling_times = np.concatenate((rolling_times, vib_times))

        # produce windows while we have at least one full window
        max_start = len(rolling) - win_size
        start_idx = 0
        while start_idx <= max_start:
            window = rolling[start_idx:start_idx+win_size]
            spec = compute_spectrogram(window, fs=sample_rate, spec_shape=spec_shape)
            center_time = float(rolling_times[start_idx + win_size // 2])

            # acoustic features (fast path if acoustic loaded)
            if ac_loaded_in_memory and ac_times is not None:
                ac_window_half_secs = (win_size / sample_rate) / 10.0
                left = center_time - ac_window_half_secs
                right = center_time + ac_window_half_secs
                li = np.searchsorted(ac_times, left, side='left')
                ri = np.searchsorted(ac_times, right, side='right')
                if ri <= li:
                    li = max(0, li-1)
                    ri = min(li+2, len(ac_vals))
                ac_win = ac_vals[li:ri]
                if ac_win.size == 0:
                    ac_feats = np.zeros((4,), dtype=np.float32)
                else:
                    ac_feats = np.array([np.mean(ac_win), np.std(ac_win), np.max(ac_win), np.min(ac_win)], dtype=np.float32)
            else:
                # acoustic not in memory - currently fallback to zeros
                # If you want streaming acoustic, we can implement a ring-buffer similar to vibration.
                ac_feats = np.zeros((4,), dtype=np.float32)

            # TODO: Replace this placeholder with your actual label alignment logic (RUL or fault).
            label = 0.0
            examples_buffer.append((spec, ac_feats, label))

            if len(examples_buffer) >= shardsize:
                write_tfrecord_examples(shard_idx, examples_buffer, out_dir)
                print(f"Wrote shard {shard_idx} with {len(examples_buffer)} examples.")
                shard_idx += 1
                examples_buffer = []

            start_idx += step

        # keep only tail needed to cross chunk boundary
        keep = max(win_size - step, 0)
        if keep > 0:
            rolling = rolling[-keep:]
            rolling_times = rolling_times[-keep:]
        else:
            rolling = np.empty((0,), dtype=np.float32)
            rolling_times = np.empty((0,), dtype=np.float64)

        print(f"Processed vib chunk {chunk_no}, rolling length now {len(rolling)}")

    # flush residual examples
    if len(examples_buffer) > 0:
        write_tfrecord_examples(shard_idx, examples_buffer, out_dir)
        print(f"Wrote final shard {shard_idx} with {len(examples_buffer)} examples.")
        shard_idx += 1

    print(f"Preprocessing complete. Total shards: {shard_idx}")

# ----------------------------
# TRAIN: load TFRecords with tf.data and train model (non-overlapping sequences)
# ----------------------------
def parse_flat(example_proto):
    """
    Parse a TFRecord example into (spec, ac, label) triple.
    spec -> tensor shape SPEC_SHAPE + (1,)
    ac   -> tensor shape (4,)
    label -> scalar tensor
    """
    feature_description = {
        FEATURE_SPEC_KEY: tf.io.VarLenFeature(tf.float32),
        FEATURE_AC_KEY: tf.io.VarLenFeature(tf.float32),
        FEATURE_LABEL_KEY: tf.io.VarLenFeature(tf.float32),
    }
    parsed = tf.io.parse_single_example(example_proto, feature_description)
    spec_flat = tf.sparse.to_dense(parsed[FEATURE_SPEC_KEY])
    ac_flat = tf.sparse.to_dense(parsed[FEATURE_AC_KEY])
    label_flat = tf.sparse.to_dense(parsed[FEATURE_LABEL_KEY])
    spec = tf.reshape(spec_flat, SPEC_SHAPE + (1,))
    ac = tf.reshape(ac_flat, (4,))
    label = tf.reshape(label_flat, ())
    return spec, ac, label

def build_model(spec_shape=SPEC_SHAPE, ac_dim=4, seq_len=SEQ_LEN):
    frame_h, frame_w = spec_shape
    # CNN frame model
    cnn_input = tf.keras.Input(shape=(frame_h, frame_w, 1), name="frame_spec")
    x = tf.keras.layers.Conv2D(16, 3, activation='relu', padding='same')(cnn_input)
    x = tf.keras.layers.MaxPool2D(2)(x)
    x = tf.keras.layers.Conv2D(32, 3, activation='relu', padding='same')(x)
    x = tf.keras.layers.MaxPool2D(2)(x)
    x = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same')(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    cnn_out = tf.keras.layers.Dense(64, activation='relu')(x)
    cnn_model = tf.keras.Model(cnn_input, cnn_out, name="frame_cnn")

    # Sequence inputs
    seq_spec_input = tf.keras.Input(shape=(seq_len, frame_h, frame_w, 1), name="seq_spec")
    seq_ac_input = tf.keras.Input(shape=(seq_len, ac_dim), name="seq_ac")

    td = tf.keras.layers.TimeDistributed(cnn_model)(seq_spec_input)
    fused = tf.keras.layers.Concatenate(axis=-1)([td, seq_ac_input])
    fused = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(128, activation='relu'))(fused)
    lstm_out = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=False))(fused)
    out = tf.keras.layers.Dense(1, activation='linear', name='rul_out')(lstm_out)
    model = tf.keras.Model([seq_spec_input, seq_ac_input], out)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE), loss='mse', metrics=['mae'])
    return model

def train_from_tfrecords(tfrecord_dir=TFRECORD_DIR, epochs=EPOCHS, batch_size=BATCH_SIZE):
    """
    Train from TFRecord shards that contain *per-window* examples.
    This builds non-overlapping sequences of length SEQ_LEN by batching windows together.
    Output dataset elements: ((spec_seq, ac_seq), label_last).
    """
    files = tf.io.gfile.glob(os.path.join(tfrecord_dir, "shard-*.tfrecord"))
    if not files:
        raise ValueError("No TFRecord shards found in " + tfrecord_dir)

    # Dataset of per-window examples
    ds = tf.data.TFRecordDataset(files, num_parallel_reads=tf.data.AUTOTUNE)
    ds = ds.map(parse_flat, num_parallel_calls=tf.data.AUTOTUNE)

    # Create non-overlapping sequences of length SEQ_LEN
    ds_seq_windows = ds.batch(SEQ_LEN, drop_remainder=True)

    # Map to model inputs ((spec_seq, ac_seq), label_last)
    def to_model_input(specs_seq, acs_seq, labels_seq):
        label_last = labels_seq[-1]
        return (specs_seq, acs_seq), label_last

    ds_model = ds_seq_windows.map(to_model_input, num_parallel_calls=tf.data.AUTOTUNE)

    # Batch sequences into training batches
    ds_model = ds_model.batch(batch_size, drop_remainder=True)
    ds_model = ds_model.prefetch(tf.data.AUTOTUNE)

    # Build and train model
    model = build_model(spec_shape=SPEC_SHAPE, ac_dim=4, seq_len=SEQ_LEN)
    model.summary()

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint("cnn_lstm_best.h5", save_best_only=True, monitor="loss"),
        tf.keras.callbacks.EarlyStopping(monitor="loss", patience=5, restore_best_weights=True)
    ]

    model.fit(ds_model, epochs=epochs, callbacks=callbacks)

# ----------------------------
# MAIN CLI
# ----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["preprocess", "train"], help="Mode: preprocess or train")
    args = parser.parse_args()

    if args.mode == "preprocess":
        print("Starting preprocessing into TFRecord shards...")
        streaming_preprocess_to_tfrecord(VIB_PATH, AC_PATH, out_dir=TFRECORD_DIR,
                                        chunksize=CHUNKSIZE, win_size=WIN_SIZE, step=STEP,
                                        seq_len=SEQ_LEN, spec_shape=SPEC_SHAPE, shardsize=SHARDSIZE)
    elif args.mode == "train":
        print("Starting training from TFRecord shards...")
        train_from_tfrecords(tfrecord_dir=TFRECORD_DIR, epochs=EPOCHS, batch_size=BATCH_SIZE)
    else:
        print("Unknown mode")

if __name__ == "__main__":
    main()
