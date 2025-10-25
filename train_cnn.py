"""
Spectrogram pipeline + PyTorch CNNs for three modalities (acoustic, vibration, current_temp).

Features:
- Stream large CSVs in chunks to avoid memory blowup.
- Estimate sampling rate from timestamps and resample/interpolate windows to uniform sampling.
- Convert windows to spectrograms (log-power) per channel, stack channels for multi-channel modalities.
- Save spectrograms + labels as .npy batches or use an on-the-fly generator.
- Train three CNNs (one per modality) each with two heads: condition (multi-class) and severity (multi-class).
- Inference function that accepts three raw signals and returns predictions using trained models.

Notes before running:
- Tested with Python 3.8+. Requires: numpy, pandas, scipy, torch, torchvision. Optionally tqdm.
- Adjust CHUNK_SIZE, WINDOW_SEC, HOP_SEC, TARGET_FS, and spectrogram params to match your data and compute budget.
- For extremely large datasets, consider converting spectrograms to an LMDB/TFRecord store rather than lots of .npy files.

"""

import os
import re
import math
import glob
import json
from typing import List, Tuple, Dict, Generator

import numpy as np
import pandas as pd
from scipy import signal, ndimage

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

try:
    from tqdm import tqdm
except Exception:
    tqdm = lambda x: x

# -----------------------------
# Configuration
# -----------------------------
CHUNK_ROWS = 1_000_000  # pandas read_csv chunk size
MIN_ROWS_FOR_FS = 1000  # rows to estimate sampling frequency
TARGET_FS = {
    'acoustic': 44100,    # target sampling rate (Hz) for acoustic (adjust if necessary)
    'vibration': 5000,    # target fs for vibration
    'current_temp': 1     # temperature/current sampled at 1 Hz in example
}
WINDOW_SEC = 1.0   # spectrogram window length in seconds
HOP_SEC = 0.5      # hop length in seconds
NFFT = 1024        # nfft for spectrogram
SPEC_SIZE = (128, 128)  # final spectrogram height x width (freq x time)

# Model & training settings (examples)
BATCH_SIZE = 32
NUM_WORKERS = 4
EPOCHS = 20
LR = 1e-3

# Map typo Unbalalnce -> Unbalance
CONDITION_FIXES = {'Unbalalnce': 'Unbalance'}

# -----------------------------
# Utility functions
# -----------------------------

def parse_filename(fn: str) -> Dict[str,str]:
    """Parse filenames like '0Nm_BPFI_03.csv' into load, condition, severity."""
    base = os.path.basename(fn)
    m = re.match(r"(?P<load>[^_]+)_(?P<condition>[^_]+)_(?P<severity>[^.]+)\\.?", base)
    if not m:
        return {'load': '', 'condition': '', 'severity': ''}
    d = m.groupdict()
    # fix condition typos
    d['condition'] = CONDITION_FIXES.get(d['condition'], d['condition'])
    return d


def estimate_fs(timestamps: np.ndarray) -> float:
    """Estimate sampling frequency from timestamps (seconds)."""
    if len(timestamps) < 2:
        return 1.0
    diffs = np.diff(timestamps)
    # ignore zeros and negatives
    diffs = diffs[diffs > 0]
    if len(diffs) == 0:
        return 1.0
    median_dt = np.median(diffs)
    if median_dt <= 0:
        return 1.0
    return float(round(1.0 / median_dt))


def resample_signal(times: np.ndarray, values: np.ndarray, target_fs: float) -> Tuple[np.ndarray, np.ndarray]:
    """Interpolate values to a uniform grid at target_fs.
    Returns (new_times, new_values)
    """
    if len(times) < 2:
        # trivial
        new_times = np.arange(0, 1.0, 1.0 / target_fs)
        new_values = np.zeros_like(new_times)
        return new_times, new_values

    t0, t1 = times[0], times[-1]
    duration = t1 - t0
    if duration <= 0:
        # degenerate
        new_times = np.arange(0, 1.0, 1.0 / target_fs)
        new_values = np.zeros_like(new_times)
        return new_times, new_values

    num = max(2, int(np.floor(duration * target_fs)))
    new_times = np.linspace(times[0], times[-1], num)
    new_values = np.interp(new_times, times, values)
    return new_times, new_values


def compute_log_spectrogram(signal_1d: np.ndarray, fs: float, nfft: int = NFFT, window_sec: float = WINDOW_SEC, hop_sec: float = HOP_SEC) -> np.ndarray:
    nperseg = int(window_sec * fs)
    noverlap = int((window_sec - hop_sec) * fs)
    if nperseg < 8:
        nperseg = min(256, len(signal_1d))
        noverlap = int(nperseg * 0.5)
    freqs, times, Sxx = signal.spectrogram(signal_1d, fs=fs, window='hann', nperseg=nperseg, noverlap=noverlap, nfft=nfft, scaling='density', mode='magnitude')
    # convert to log-scale (dB)
    Sxx = np.where(Sxx <= 1e-12, 1e-12, Sxx)
    Sxx_db = 10.0 * np.log10(Sxx)
    # normalize to 0-1
    Sxx_db -= Sxx_db.min()
    Sxx_db /= (Sxx_db.max() + 1e-6)
    return Sxx_db


def resize_spectrogram(spec: np.ndarray, target_size: Tuple[int,int]=SPEC_SIZE) -> np.ndarray:
    """Resize spectrogram (freq x time) to target_size using zoom.
    Uses scipy.ndimage.zoom which is fast and avoids OpenCV dependency.
    """
    h, w = spec.shape
    th, tw = target_size
    zoom_h = th / float(h) if h>0 else 1.0
    zoom_w = tw / float(w) if w>0 else 1.0
    spec_resized = ndimage.zoom(spec, (zoom_h, zoom_w), order=1)
    return spec_resized

# -----------------------------
# Streaming generator: from CSV -> spectrogram windows
# -----------------------------

def spectrograms_from_csv(file_path: str, modality: str, channels: List[str], target_fs: int=None) -> Generator[Tuple[np.ndarray, Dict[str,str]], None, None]:
    """Yield spectrogram windows as numpy arrays and metadata labels for a single CSV file.
    For multi-channel modalities, stack per-channel spectrograms into a C x H x W array.

    Yields: (spec_array, labels) where spec_array shape = (C, H, W)
    labels: {'condition','severity','load'}
    """
    if target_fs is None:
        target_fs = TARGET_FS.get(modality, 1000)

    # read a small portion to estimate fs
    reader = pd.read_csv(file_path, nrows=MIN_ROWS_FOR_FS)
    if reader.shape[0] < 2:
        # fallback
        est_fs = target_fs
    else:
        ts = reader.iloc[:,0].to_numpy(dtype=float)
        est_fs = estimate_fs(ts)

    # create pandas iterator to stream full file
    col_names = None
    # We'll read full file but by chunks and assemble a rolling buffer per channel
    col_iter = pd.read_csv(file_path, chunksize=CHUNK_ROWS)

    # rolling buffer
    buffer_times = np.array([], dtype=float)
    buffer_channels = {ch: np.array([], dtype=float) for ch in channels}

    parse_info = parse_filename(file_path)

    for chunk in col_iter:
        # Ensure columns present; handle extra columns
        # Time Stamp may be first column; ensure it's named 'Time Stamp' or take first column
        if 'Time Stamp' in chunk.columns:
            times = chunk['Time Stamp'].to_numpy(dtype=float)
        else:
            times = chunk.iloc[:,0].to_numpy(dtype=float)

        # Gather channel columns
        for ch in channels:
            if ch in chunk.columns:
                vals = chunk[ch].to_numpy(dtype=float)
            else:
                # if not present, fill zeros
                vals = np.zeros_like(times, dtype=float)
            buffer_channels[ch] = np.concatenate([buffer_channels[ch], vals])
        buffer_times = np.concatenate([buffer_times, times])

        # Process buffer into uniform-spaced signal by resampling then windowing
        # we only proceed if buffer has at least some duration
        while True:
            if len(buffer_times) < 2:
                break
            duration = buffer_times[-1] - buffer_times[0]
            if duration < WINDOW_SEC:
                # not enough data yet
                break
            # take a window of WINDOW_SEC from start
            window_start_time = buffer_times[0]
            window_end_time = window_start_time + WINDOW_SEC
            # select indices within window
            idx = np.where((buffer_times >= window_start_time) & (buffer_times <= window_end_time))[0]
            if len(idx) < 2:
                # advance buffer by dropping first sample
                buffer_times = buffer_times[1:]
                for ch in channels:
                    buffer_channels[ch] = buffer_channels[ch][1:]
                continue

            # extract and resample each channel
            channel_specs = []
            for ch in channels:
                segment_times = buffer_times[idx]
                segment_vals = buffer_channels[ch][idx]
                # resample to uniform grid
                _, resampled = resample_signal(segment_times, segment_vals, target_fs)
                spec = compute_log_spectrogram(resampled, fs=target_fs)
                spec = resize_spectrogram(spec, SPEC_SIZE)
                channel_specs.append(spec)

            # stack to C x H x W
            spec_stack = np.stack(channel_specs, axis=0).astype(np.float32)

            yield spec_stack, parse_info

            # advance buffer by HOP_SEC (convert to number of original samples approx)
            advance_time = HOP_SEC
            # drop samples <= window_start_time + advance_time
            new_start = window_start_time + advance_time
            keep_idx = np.where(buffer_times >= new_start)[0]
            if len(keep_idx) == 0:
                buffer_times = np.array([], dtype=float)
                for ch in channels:
                    buffer_channels[ch] = np.array([], dtype=float)
            else:
                buffer_times = buffer_times[keep_idx]
                for ch in channels:
                    buffer_channels[ch] = buffer_channels[ch][keep_idx]

    # end for chunks

# -----------------------------
# Save spectrograms for entire folders (optional step)
# -----------------------------

def build_dataset_from_folder(folder: str, modality: str, channels: List[str], out_dir: str):
    """Iterate CSVs in folder, generate spectrograms and save .npy files + labels.json
    Each saved batch file contains N examples: dict with keys 'X' (numpy array N x C x H x W) and 'labels' (list of dicts)
    """
    os.makedirs(out_dir, exist_ok=True)
    files = sorted(glob.glob(os.path.join(folder, '*.csv')))
    batch = []
    labels = []
    batch_idx = 0
    saved_files = []
    for fp in tqdm(files):
        for spec, info in spectrograms_from_csv(fp, modality, channels):
            batch.append(spec)
            labels.append(info)
            if len(batch) >= 1024:
                arr = np.stack(batch, axis=0)
                outp = os.path.join(out_dir, f'{modality}_batch_{batch_idx:05d}.npz')
                np.savez_compressed(outp, X=arr, labels=json.dumps(labels))
                saved_files.append(outp)
                batch = []
                labels = []
                batch_idx += 1
    if len(batch) > 0:
        arr = np.stack(batch, axis=0)
        outp = os.path.join(out_dir, f'{modality}_batch_{batch_idx:05d}.npz')
        np.savez_compressed(outp, X=arr, labels=json.dumps(labels))
        saved_files.append(outp)
    return saved_files

# -----------------------------
# PyTorch Dataset - loads precomputed .npz or accepts generator
# -----------------------------
class SpectrogramDataset(Dataset):
    def __init__(self, npz_files: List[str], label_map: Dict[str,int]=None, severity_map: Dict[str,int]=None, transform=None):
        self.files = npz_files
        self.transform = transform
        self.samples = []  # list of tuples (file_idx, within_file_idx)
        self.data_index = []
        self.label_map = label_map or {}
        self.severity_map = severity_map or {}
        # build index
        for fidx, f in enumerate(self.files):
            meta = np.load(f, allow_pickle=True)
            X = meta['X']
            labels = json.loads(meta['labels'].tolist()) if isinstance(meta['labels'], np.ndarray) else json.loads(meta['labels'])
            n = X.shape[0]
            for i in range(n):
                self.data_index.append((f, i))

    def __len__(self):
        return len(self.data_index)

    def __getitem__(self, idx):
        f, i = self.data_index[idx]
        meta = np.load(f, allow_pickle=True)
        X = meta['X'][i]
        labels = json.loads(meta['labels'].tolist())[i]
        # parse labels
        condition = labels.get('condition', 'unknown')
        severity = labels.get('severity', '0')
        # map to int
        y_cond = self.label_map.get(condition, -1)
        y_sev = self.severity_map.get(severity, -1)
        x = X.astype(np.float32)
        if self.transform:
            x = self.transform(x)
        return torch.from_numpy(x), torch.tensor(y_cond, dtype=torch.long), torch.tensor(y_sev, dtype=torch.long)

# -----------------------------
# Simple CNN model with two heads
# -----------------------------
class SimpleCNNMultiHead(nn.Module):
    def __init__(self, in_channels: int, num_conditions: int, num_severity: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4,4)),
        )
        self.flatten = nn.Flatten()
        self.fc_shared = nn.Linear(128*4*4, 256)
        self.cond_head = nn.Linear(256, num_conditions)
        self.sev_head = nn.Linear(256, num_severity)

    def forward(self, x):
        # x: B x C x H x W
        x = self.features(x)
        x = self.flatten(x)
        x = F.relu(self.fc_shared(x))
        return self.cond_head(x), self.sev_head(x)

# -----------------------------
# Training loop (per modality)
# -----------------------------

def train_model(modality: str, train_files: List[str], val_files: List[str], in_channels: int, label_map: Dict[str,int], severity_map: Dict[str,int], save_path: str, epochs=EPOCHS):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_ds = SpectrogramDataset(train_files, label_map=label_map, severity_map=severity_map)
    val_ds = SpectrogramDataset(val_files, label_map=label_map, severity_map=severity_map)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    model = SimpleCNNMultiHead(in_channels, len(label_map), len(severity_map)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    best_val_loss = float('inf')
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for xb, ycond, ysev in tqdm(train_loader):
            xb = xb.to(device)
            ycond = ycond.to(device)
            ysev = ysev.to(device)
            opt.zero_grad()
            out_cond, out_sev = model(xb)
            loss = criterion(out_cond, ycond) + criterion(out_sev, ysev)
            loss.backward()
            opt.step()
            running_loss += loss.item() * xb.size(0)
        epoch_loss = running_loss / len(train_loader.dataset)

        # validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, ycond, ysev in val_loader:
                xb = xb.to(device)
                ycond = ycond.to(device)
                ysev = ysev.to(device)
                out_cond, out_sev = model(xb)
                loss = criterion(out_cond, ycond) + criterion(out_sev, ysev)
                val_loss += loss.item() * xb.size(0)
        val_loss /= len(val_loader.dataset)
        print(f"Epoch {epoch+1}/{epochs} - train loss {epoch_loss:.4f}, val loss {val_loss:.4f}")

        # checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save({'model_state_dict': model.state_dict(), 'label_map': label_map, 'severity_map': severity_map}, save_path)
            print(f"Saved best model to {save_path}")
    return save_path

# -----------------------------
# Helper to build label maps from folder or files
# -----------------------------

def collect_label_maps_from_npz(npz_files: List[str]) -> Tuple[Dict[str,int], Dict[str,int]]:
    conds = set()
    sevs = set()
    for f in npz_files:
        meta = np.load(f, allow_pickle=True)
        labels = json.loads(meta['labels'].tolist()) if isinstance(meta['labels'], np.ndarray) else json.loads(meta['labels'])
        for l in labels:
            cond = l.get('condition', 'unknown')
            sev = l.get('severity', '0')
            cond = CONDITION_FIXES.get(cond, cond)
            conds.add(cond)
            sevs.add(sev)
    cond_list = sorted(list(conds))
    sev_list = sorted(list(sevs))
    cond_map = {c:i for i,c in enumerate(cond_list)}
    sev_map = {s:i for i,s in enumerate(sev_list)}
    return cond_map, sev_map

# -----------------------------
# Inference for 3 new signals
# -----------------------------

def load_model_for_inference(model_path: str, in_channels: int) -> Tuple[nn.Module, Dict[str,int], Dict[str,int]]:
    ckpt = torch.load(model_path, map_location='cpu')
    label_map = ckpt.get('label_map', {})
    severity_map = ckpt.get('severity_map', {})
    # invert maps for readable output
    inv_label_map = {v:k for k,v in label_map.items()}
    inv_sev_map = {v:k for k,v in severity_map.items()}
    model = SimpleCNNMultiHead(in_channels, len(label_map), len(severity_map))
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    return model, inv_label_map, inv_sev_map


def predict_from_signals(acoustic_signal: Tuple[np.ndarray, np.ndarray], vibration_signal: Tuple[np.ndarray, Dict[str,np.ndarray]], current_temp_signal: Tuple[np.ndarray, Dict[str,np.ndarray]], model_paths: Dict[str,str], channel_lists: Dict[str,List[str]], target_fs_map: Dict[str,int]=TARGET_FS) -> Dict[str,Tuple[str,str, Dict[str,float]]]:
    """
    acoustic_signal: (times, values) for acoustic (1D)
    vibration_signal: (times, {'x_direction_housing_A': arr, ...})
    current_temp_signal: (times, {'Temperature_housing_A': arr, ...})
    model_paths: dict with keys 'acoustic','vibration','current_temp' pointing to saved .pth
    channel_lists: dict mapping modality to list of channel names (strings)

    Returns dict mapping modality -> (predicted_condition, predicted_severity, softmax_scores)
    """
    results = {}
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    for modality in ['acoustic','vibration','current_temp']:
        mp = model_paths[modality]
        channels = channel_lists[modality]
        in_ch = len(channels)
        model, inv_label_map, inv_sev_map = load_model_for_inference(mp, in_ch)
        model.to(device)

        # prepare spectrogram for given modality
        if modality == 'acoustic':
            times, vals = acoustic_signal
            _, resampled = resample_signal(times, vals, target_fs_map[modality])
            spec = compute_log_spectrogram(resampled, fs=target_fs_map[modality])
            spec = resize_spectrogram(spec, SPEC_SIZE)
            spec_stack = np.expand_dims(spec, 0)  # C=1
        elif modality == 'vibration':
            times, ch_dict = vibration_signal
            # build per-channel resampled signals
            specs = []
            for ch in channels:
                vals = ch_dict.get(ch, np.zeros_like(times))
                _, resampled = resample_signal(times, vals, target_fs_map[modality])
                s = compute_log_spectrogram(resampled, fs=target_fs_map[modality])
                s = resize_spectrogram(s, SPEC_SIZE)
                specs.append(s)
            spec_stack = np.stack(specs, axis=0)
        else:  # current_temp
            times, ch_dict = current_temp_signal
            specs = []
            for ch in channels:
                vals = ch_dict.get(ch, np.zeros_like(times))
                _, resampled = resample_signal(times, vals, target_fs_map[modality])
                s = compute_log_spectrogram(resampled, fs=target_fs_map[modality])
                s = resize_spectrogram(s, SPEC_SIZE)
                specs.append(s)
            spec_stack = np.stack(specs, axis=0)

        x = torch.from_numpy(spec_stack).unsqueeze(0).to(device)  # 1xCxHxW
        with torch.no_grad():
            out_cond, out_sev = model(x)
            pcond = F.softmax(out_cond, dim=1).cpu().numpy()[0]
            psev = F.softmax(out_sev, dim=1).cpu().numpy()[0]
            i_cond = int(np.argmax(pcond))
            i_sev = int(np.argmax(psev))
            cond_str = inv_label_map.get(i_cond, str(i_cond))
            sev_str = inv_sev_map.get(i_sev, str(i_sev))
            results[modality] = (cond_str, sev_str, {'condition_probs': pcond.tolist(), 'severity_probs': psev.tolist()})
    return results

# -----------------------------
# Example usage (not executed automatically) - instructions
# -----------------------------
EXAMPLE = '''
1) Precompute datasets (optional):
   build_dataset_from_folder('acoustic/', 'acoustic', channels=['values'], out_dir='data/acoustic_npz')
   build_dataset_from_folder('vibration/', 'vibration', channels=['x_direction_housing_A','y_direction_housing_A','x_direction_housing_B','y_direction_housing_B'], out_dir='data/vibration_npz')
   build_dataset_from_folder('current_temp/', 'current_temp', channels=['Temperature_housing_A','Temperature_housing_B','U-phase','V-phase','W-phase'], out_dir='data/current_npz')

2) Create label maps:
   cond_map, sev_map = collect_label_maps_from_npz(sorted(glob.glob('data/acoustic_npz/*.npz')))

3) Train:
   train_model('acoustic', train_files, val_files, in_channels=1, label_map=cond_map, severity_map=sev_map, save_path='models/acoustic_best.pth')

4) Predict on three signals:
   acoustic_sig = (times_ac, vals_ac)
   vibration_sig = (times_v, {'x_direction_housing_A': arrA, 'y_direction_housing_A': arrB, ...})
   current_sig = (times_c, {'Temperature_housing_A': tA, ...})
   results = predict_from_signals(acoustic_sig, vibration_sig, current_sig, model_paths={'acoustic':'models/acoustic_best.pth', 'vibration':'models/vibration_best.pth','current_temp':'models/current_best.pth'}, channel_lists=...) 
'''

if __name__ == '__main__':
    print('This file provides functions to preprocess, train and predict. See EXAMPLE string for usage.')
