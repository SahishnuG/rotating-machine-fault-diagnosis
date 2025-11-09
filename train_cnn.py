#!/usr/bin/env python3
"""
train_federated_multitask_cnn.py

Multitask Federated CNN per modality (condition + severity) with a
final full-coverage fine-tune that sees ALL time columns of ALL scalograms.

Update: Vibration now uses TRUE 4-channel fusion (x_A, y_A, x_B, y_B) per window.
Acoustic remains single-channel.

Key ideas:
- FL rounds: memory-safe training via (freq slice, time stride, fixed crop)
- Final pass: deterministic tiling so the model trains on the ENTIRE data
- AMP (version-compatible), AdamW + cosine, label smoothing, grad clipping
- Robust label parsing incl. Unbalalnce->Unbalance, normal->Normal, severity fallback.
"""

import os, re, gc, json, math, glob, time, argparse, random, contextlib
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ------------------------------
# AMP compatibility helpers
# ------------------------------
def make_grad_scaler(amp_enabled: bool):
    if not amp_enabled:
        return None
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        try:
            return torch.amp.GradScaler(enabled=True)
        except TypeError:
            return torch.cuda.amp.GradScaler(enabled=True)
    return torch.cuda.amp.GradScaler(enabled=True)

@contextlib.contextmanager
def autocast_ctx(amp_enabled: bool, device: torch.device):
    if not amp_enabled or device.type != "cuda":
        yield
        return
    if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
        try:
            with torch.amp.autocast(device_type="cuda", enabled=True):
                yield
                return
        except TypeError:
            try:
                with torch.amp.autocast("cuda"):
                    yield
                    return
            except TypeError:
                pass
    with torch.cuda.amp.autocast():
        yield

# ------------------------------
# Repro
# ------------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# ------------------------------
# Labels / parsing
# ------------------------------
CONDITION_ALIASES = {
    "normal": "Normal",
    "NORMAL": "Normal",
    "Unbalalnce": "Unbalance",  # dataset typo
}
ACOUSTIC_CONDITIONS  = ["BPFI", "BPFO", "Normal"]
VIBRATION_CONDITIONS = ["BPFI", "BPFO", "Misalign", "Normal", "Unbalance"]

def canonical_condition(name: str) -> str:
    name = (name or "").strip()
    if name in CONDITION_ALIASES:
        return CONDITION_ALIASES[name]
    if name.lower() == "normal":
        return "Normal"
    return name

def parse_labels_from_source(source_csv: str):
    """
    Accepts:
      - aaaaNm_bbbb_cccc.csv  -> load, condition, severity
      - aaaaNm_bbbb.csv       -> load, condition, severity='00'
    """
    base = os.path.splitext(os.path.basename(source_csv))[0]
    parts = [p for p in base.split("_") if p]
    if len(parts) >= 3:
        load = parts[0]
        condition = canonical_condition(parts[1])
        severity = parts[2]
    elif len(parts) == 2:
        load = parts[0]
        condition = canonical_condition(parts[1])
        severity = "00"
    else:
        load = parts[0] if parts else "UNK"
        condition = "Normal"
        severity = "00"
    return load, condition, severity

# ------------------------------
# Dataset (single-channel, used for acoustic)
# ------------------------------
class ScalogramDataset(Dataset):
    """
    Single-channel (acoustic) memory-safe dataset:
      - optional frequency slice
      - time stride
      - fixed time crop (random/center)
      - per-sample log1p + zscore
    Returns x -> (1, F, Tcrop)
    """
    def __init__(
        self,
        items: List[Tuple[str, str]],                 # (npy_path, meta_json_path)
        condition_to_idx: Dict[str, int],
        severity_to_idx: Dict[str, int],
        log_amplitude: bool = True,
        freq_start: Optional[int] = None,
        freq_end:   Optional[int] = 128,
        col_stride: int = 16,
        time_crop:  Optional[int] = 512,
        center_crop: bool = False,
    ):
        self.items = items
        self.condition_to_idx = condition_to_idx
        self.severity_to_idx  = severity_to_idx
        self.log_amplitude    = log_amplitude
        self.freq_start = freq_start
        self.freq_end   = freq_end
        self.col_stride = max(1, int(col_stride))
        self.time_crop  = time_crop
        self.center_crop = center_crop

    def __len__(self): return len(self.items)

    def __getitem__(self, idx):
        npy_path, meta_path = self.items[idx]
        with open(meta_path, "r") as fh:
            meta = json.load(fh)
        source_csv = meta.get("source_csv")
        _, cond, sev = parse_labels_from_source(source_csv)

        arr = np.load(npy_path, mmap_mode="r")  # (F,T)
        X = np.array(arr, dtype=np.float32, copy=False)

        # frequency slice
        f0 = 0 if self.freq_start is None else max(0, int(self.freq_start))
        f1 = X.shape[0] if self.freq_end is None else min(X.shape[0], int(self.freq_end))
        X = X[f0:f1, :]

        # time stride
        if self.col_stride > 1:
            X = X[:, ::self.col_stride]

        # time crop
        if self.time_crop is not None and X.shape[1] > self.time_crop:
            T = X.shape[1]
            if self.center_crop:
                s = (T - self.time_crop) // 2
            else:
                s = np.random.randint(0, T - self.time_crop + 1)
            X = X[:, s:s + self.time_crop]

        # normalization (per-sample)
        if self.log_amplitude:
            X = np.log1p(X)
        mu, sd = X.mean(), X.std() + 1e-6
        X = (X - mu) / sd
        X = np.expand_dims(X, axis=0)  # (1,F,T)

        y_cond = self.condition_to_idx[canonical_condition(cond)]
        if sev in self.severity_to_idx:
            y_sev = self.severity_to_idx[sev]
        else:
            def num(s):
                n = re.sub(r"\D","", s or "")
                return int(n) if n else 0
            target = num(sev)
            key = min(self.severity_to_idx.keys(), key=lambda k: abs(num(k) - target))
            y_sev = self.severity_to_idx[key]

        return torch.from_numpy(X), torch.tensor(y_cond, dtype=torch.long), torch.tensor(y_sev, dtype=torch.long)

# ------------------------------
# Vibration: 4-channel grouping & datasets
# ------------------------------
VIB_CH_ORDER = [
    "x_direction_housing_A",
    "y_direction_housing_A",
    "x_direction_housing_B",
    "y_direction_housing_B",
]

def discover_items(modality_dir: str) -> List[Tuple[str, str, str]]:
    """
    Returns list of (npy_path, meta_path, source_csv)
    """
    items = []
    for npy in sorted(glob.glob(os.path.join(modality_dir, "*.npy"))):
        meta = npy + ".meta.json"
        if not os.path.exists(meta):
            continue
        try:
            with open(meta, "r") as fh:
                m = json.load(fh)
            src = m.get("source_csv")
            if not src:
                continue
        except Exception:
            continue
        items.append((npy, meta, src))
    return items

def group_vibration_items(items: List[Tuple[str, str, str]]):
    """
    items: list of (npy_path, meta_path, source_csv)
    returns:
      - grouped: dict[(source_csv, window_index)] -> list[(npy, meta)] ordered per VIB_CH_ORDER (subset allowed)
      - by_source: dict[source_csv] -> List[List[(npy, meta)]]
    """
    buckets: Dict[Tuple[str,int], Dict[str, Tuple[str,str]]] = {}
    for npy, meta, src in items:
        with open(meta, "r") as fh:
            m = json.load(fh)
        win = int(m.get("window_index", 0))
        ch  = m.get("channel", "")
        key = (src, win)
        buckets.setdefault(key, {})[ch] = (npy, meta)

    grouped: Dict[Tuple[str,int], List[Tuple[str,str]]] = {}
    for key, chmap in buckets.items():
        lst = [chmap[ch] for ch in VIB_CH_ORDER if ch in chmap]
        if lst:  # keep groups with at least 1 channel (ideally 4)
            grouped[key] = lst

    by_source: Dict[str, List[List[Tuple[str,str]]]] = {}
    for (src, win), lst in grouped.items():
        by_source.setdefault(src, []).append(lst)

    return grouped, by_source

class ScalogramDataset4Ch(Dataset):
    """
    Vibration ONLY: stacks up to 4 channels for the same (source_csv, window_index).
    Returns x -> (C<=4, F, Tcrop)
    """
    def __init__(
        self,
        groups: List[List[Tuple[str, str]]],  # list of lists of (npy, meta) in VIB_CH_ORDER
        condition_to_idx: Dict[str, int],
        severity_to_idx: Dict[str, int],
        log_amplitude: bool = True,
        freq_start: Optional[int] = None,
        freq_end:   Optional[int] = 128,
        col_stride: int = 16,
        time_crop:  Optional[int] = 512,
        center_crop: bool = False,
    ):
        # sanity: groups should be List[List[(npy, meta)]]
        if not isinstance(groups, list) or (len(groups) > 0 and not isinstance(groups[0], list)):
            raise TypeError(
                "ScalogramDataset4Ch expects groups=List[List[(npy, meta)]]. "
                "It looks like a flat list was passed."
            )
        self.groups = groups
        self.condition_to_idx = condition_to_idx
        self.severity_to_idx  = severity_to_idx
        self.log_amplitude    = log_amplitude
        self.freq_start = freq_start
        self.freq_end   = freq_end
        self.col_stride = max(1, int(col_stride))
        self.time_crop  = time_crop
        self.center_crop = center_crop

    def __len__(self): return len(self.groups)

    def _load_label(self, meta_path):
        with open(meta_path, "r") as fh:
            m = json.load(fh)
        src = m.get("source_csv")
        _, cond, sev = parse_labels_from_source(src)
        return canonical_condition(cond), sev

    def __getitem__(self, idx):
        group = self.groups[idx]
        xs = []
        cond, sev = None, None
        for (npy_path, meta_path) in group:
            if cond is None:
                c, s = self._load_label(meta_path)
                cond, sev = c, s
            arr = np.load(npy_path, mmap_mode="r")  # (F,T)
            X = np.array(arr, dtype=np.float32, copy=False)
            # freq slice
            f0 = 0 if self.freq_start is None else max(0, int(self.freq_start))
            f1 = X.shape[0] if self.freq_end is None else min(X.shape[0], int(self.freq_end))
            X = X[f0:f1, :]
            # stride
            if self.col_stride > 1:
                X = X[:, ::self.col_stride]
            xs.append(X)

        # align T across channels (post-stride)
        Tm = min(x.shape[1] for x in xs)
        xs = [x[:, :Tm] for x in xs]
        X = np.stack(xs, axis=0)  # (C,F,T)

        # crop
        if self.time_crop is not None and X.shape[2] > self.time_crop:
            T = X.shape[2]
            s = (T - self.time_crop)//2 if self.center_crop else np.random.randint(0, T - self.time_crop + 1)
            X = X[:, :, s:s+self.time_crop]

        # per-channel normalize
        if self.log_amplitude:
            X = np.log1p(X)
        mu = X.mean(axis=(1,2), keepdims=True)
        sd = X.std(axis=(1,2), keepdims=True) + 1e-6
        X = (X - mu) / sd

        y_cond = self.condition_to_idx[cond]
        if sev in self.severity_to_idx:
            y_sev = self.severity_to_idx[sev]
        else:
            def num(s):
                n = re.sub(r"\D","", s or "")
                return int(n) if n else 0
            target = num(sev)
            key = min(self.severity_to_idx.keys(), key=lambda k: abs(num(k) - target))
            y_sev = self.severity_to_idx[key]

        return torch.from_numpy(X), torch.tensor(y_cond, dtype=torch.long), torch.tensor(y_sev, dtype=torch.long)

class FullCoverageScalogramDataset4Ch(Dataset):
    """
    Vibration ONLY: deterministic tiling after stride, stacking channels per window.
    Returns x -> (C<=4, F, tile_len)
    """
    def __init__(
        self,
        groups: List[List[Tuple[str, str]]],  # list of lists of (npy, meta)
        condition_to_idx: Dict[str, int],
        severity_to_idx: Dict[str, int],
        log_amplitude: bool = True,
        freq_start: Optional[int] = None,
        freq_end:   Optional[int] = 128,
        col_stride: int = 8,
        tile_len:   int = 1024,
        tile_overlap: int = 256,
    ):
        # sanity: groups should be List[List[(npy, meta)]]
        if not isinstance(groups, list) or (len(groups) > 0 and not isinstance(groups[0], list)):
            raise TypeError(
                "FullCoverageScalogramDataset4Ch expects groups=List[List[(npy, meta)]]. "
                "It looks like a flat list was passed."
            )
        self.groups = groups
        self.condition_to_idx = condition_to_idx
        self.severity_to_idx  = severity_to_idx
        self.log_amplitude    = log_amplitude
        self.freq_start = freq_start
        self.freq_end   = freq_end
        self.col_stride = max(1, int(col_stride))
        self.tile_len   = max(1, int(tile_len))
        self.tile_overlap = max(0, int(tile_overlap))

        # build index (group_idx, tile_start)
        self.index: List[Tuple[int,int]] = []
        for gi, g in enumerate(self.groups):
            if not g or not isinstance(g[0], (list, tuple)) or not isinstance(g[0][0], str):
                raise TypeError(f"Bad group at index {gi}: expected list of (npy_path, meta_path) tuples.")
            arr = np.load(g[0][0], mmap_mode="r")
            T = arr.shape[1]
            T_post = (T + self.col_stride - 1)//self.col_stride
            if T_post <= self.tile_len:
                self.index.append((gi, 0))
            else:
                step = max(1, self.tile_len - self.tile_overlap)
                s = 0
                while s + self.tile_len <= T_post:
                    self.index.append((gi, s))
                    s += step
                if s < T_post:
                    self.index.append((gi, max(0, T_post - self.tile_len)))

    def __len__(self): return len(self.index)

    def _load_label(self, meta_path):
        with open(meta_path, "r") as fh:
            m = json.load(fh)
        src = m.get("source_csv")
        _, cond, sev = parse_labels_from_source(src)
        return canonical_condition(cond), sev

    def __getitem__(self, j):
        gi, t0 = self.index[j]
        group = self.groups[gi]
        xs = []
        cond, sev = None, None
        for (npy_path, meta_path) in group:
            if cond is None:
                c, s = self._load_label(meta_path)
                cond, sev = c, s
            arr = np.load(npy_path, mmap_mode="r")
            X = np.array(arr, dtype=np.float32, copy=False)
            f0 = 0 if self.freq_start is None else max(0, int(self.freq_start))
            f1 = X.shape[0] if self.freq_end is None else min(X.shape[0], int(self.freq_end))
            X = X[f0:f1, :]
            if self.col_stride > 1:
                X = X[:, ::self.col_stride]
            T = X.shape[1]
            if self.tile_len >= T:
                X = X[:, :self.tile_len] if self.tile_len <= T else np.pad(X, ((0,0),(0,self.tile_len-T)))
            else:
                X = X[:, t0:t0+self.tile_len]
            xs.append(X)

        Tm = min(x.shape[1] for x in xs)
        xs = [x[:, :Tm] for x in xs]
        X = np.stack(xs, axis=0)  # (C,F,T)

        if self.log_amplitude:
            X = np.log1p(X)
        mu = X.mean(axis=(1,2), keepdims=True)
        sd = X.std(axis=(1,2), keepdims=True) + 1e-6
        X = (X - mu) / sd

        y_cond = self.condition_to_idx[cond]
        if sev in self.severity_to_idx:
            y_sev = self.severity_to_idx[sev]
        else:
            def num(s):
                n = re.sub(r"\D","", s or "")
                return int(n) if n else 0
            target = num(sev)
            key = min(self.severity_to_idx.keys(), key=lambda k: abs(num(k) - target))
            y_sev = self.severity_to_idx[key]

        return torch.from_numpy(X), torch.tensor(y_cond, dtype=torch.long), torch.tensor(y_sev, dtype=torch.long)

# ------------------------------
# Single-channel full coverage dataset (for acoustic)
# ------------------------------
class FullCoverageScalogramDataset(Dataset):
    """
    Single-channel: deterministic tiling over time to cover ENTIRE width.
    Returns x -> (1, F, tile_len)
    """
    def __init__(
        self,
        items: List[Tuple[str, str]],                  # (npy, meta)
        condition_to_idx: Dict[str, int],
        severity_to_idx: Dict[str, int],
        log_amplitude: bool = True,
        freq_start: Optional[int] = None,
        freq_end:   Optional[int] = 128,
        col_stride: int = 8,
        tile_len:   int = 1024,
        tile_overlap: int = 256,
    ):
        self.base = items
        self.condition_to_idx = condition_to_idx
        self.severity_to_idx  = severity_to_idx
        self.log_amplitude    = log_amplitude
        self.freq_start = freq_start
        self.freq_end   = freq_end
        self.col_stride = max(1, int(col_stride))
        self.tile_len   = max(1, int(tile_len))
        self.tile_overlap = max(0, int(tile_overlap))

        self.index: List[Tuple[int,int]] = []
        for i, (npy_path, _) in enumerate(self.base):
            arr = np.load(npy_path, mmap_mode="r")
            T = arr.shape[1]
            T_post = (T + self.col_stride - 1)//self.col_stride
            if T_post <= self.tile_len:
                self.index.append((i, 0))
            else:
                step = max(1, self.tile_len - self.tile_overlap)
                s = 0
                while s + self.tile_len <= T_post:
                    self.index.append((i, s))
                    s += step
                if s < T_post:
                    self.index.append((i, max(0, T_post - self.tile_len)))

    def __len__(self): return len(self.index)

    def __getitem__(self, j):
        i, t0 = self.index[j]
        npy_path, meta_path = self.base[i]
        with open(meta_path, "r") as fh:
            m = json.load(fh)
        src = m.get("source_csv")
        _, cond, sev = parse_labels_from_source(src)

        arr = np.load(npy_path, mmap_mode="r")
        X = np.array(arr, dtype=np.float32, copy=False)  # (F,T)

        f0 = 0 if self.freq_start is None else max(0, int(self.freq_start))
        f1 = X.shape[0] if self.freq_end is None else min(X.shape[0], int(self.freq_end))
        X = X[f0:f1, :]

        if self.col_stride > 1:
            X = X[:, ::self.col_stride]

        T = X.shape[1]
        if self.tile_len >= T:
            X = X[:, :self.tile_len] if self.tile_len <= T else np.pad(X, ((0,0),(0,self.tile_len-T)))
        else:
            X = X[:, t0:t0+self.tile_len]

        if self.log_amplitude:
            X = np.log1p(X)
        mu, sd = X.mean(), X.std() + 1e-6
        X = (X - mu) / sd
        X = np.expand_dims(X, axis=0)  # (1,F,T)

        y_cond = self.condition_to_idx[canonical_condition(cond)]
        if sev in self.severity_to_idx:
            y_sev = self.severity_to_idx[sev]
        else:
            def num(s):
                n = re.sub(r"\D","", s or "")
                return int(n) if n else 0
            target = num(sev)
            key = min(self.severity_to_idx.keys(), key=lambda k: abs(num(k) - target))
            y_sev = self.severity_to_idx[key]

        return torch.from_numpy(X), torch.tensor(y_cond, dtype=torch.long), torch.tensor(y_sev, dtype=torch.long)

# ------------------------------
# Model (multitask) with configurable input channels
# ------------------------------
class ConvBlock(nn.Module):
    def __init__(self, c_in, c_out, k=3, p=1, s=1):
        super().__init__()
        self.conv = nn.Conv2d(c_in, c_out, k, stride=s, padding=p)
        self.bn   = nn.BatchNorm2d(c_out)
        self.act  = nn.ReLU(inplace=True)
    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class MultiTaskCNN(nn.Module):
    """
    Early downsampling in time to reduce memory: first MaxPool2d((2,4)).
    """
    def __init__(self, n_cond: int, n_sev: int, in_ch: int = 1):
        super().__init__()
        self.backbone = nn.Sequential(
            ConvBlock(in_ch, 32, 5, 2),     # in_ch = 1 (acoustic) or 4 (vibration)
            nn.MaxPool2d((2, 4)),           # ↓F x2, ↓T x4
            ConvBlock(32, 64, 3, 1),
            nn.MaxPool2d((2, 2)),
            ConvBlock(64, 128, 3, 1),
            nn.AdaptiveAvgPool2d((4, 8)),
        )
        feat_dim = 128 * 4 * 8
        self.drop = nn.Dropout(0.2)
        self.fc_cond = nn.Linear(feat_dim, n_cond)
        self.fc_sev  = nn.Linear(feat_dim, n_sev)
    def forward(self, x):
        z = self.backbone(x)
        z = z.flatten(1)
        z = self.drop(z)
        return self.fc_cond(z), self.fc_sev(z)

# ------------------------------
# FedAvg helpers
# ------------------------------
def get_state_dict(model: nn.Module) -> Dict[str, torch.Tensor]:
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

def set_state_dict(model: nn.Module, state: Dict[str, torch.Tensor]):
    model.load_state_dict(state, strict=True)

def average_states(states: List[Tuple[Dict[str, torch.Tensor], int]]) -> Dict[str, torch.Tensor]:
    total = sum(n for _, n in states)
    out: Dict[str, torch.Tensor] = {}
    for k in states[0][0].keys():
        acc = None
        for state, n in states:
            w = state[k] * (n / total)
            acc = w if acc is None else acc + w
        out[k] = acc
    return out

# ------------------------------
# Train / Eval
# ------------------------------
LABEL_SMOOTH = 0.1
WEIGHT_DECAY = 1e-2
CLIP_NORM    = 1.0

def train_one_epoch(model, loader, optimizer, device, scaler=None):
    model.train()
    ce = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTH)
    total, loss_sum = 0, 0.0
    for xb, ycond, ysev in loader:
        xb, ycond, ysev = xb.to(device, non_blocking=True), ycond.to(device, non_blocking=True), ysev.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        use_amp = (scaler is not None)
        with autocast_ctx(use_amp, device):
            pc, ps = model(xb)
            loss = ce(pc, ycond) + ce(ps, ysev)

        if use_amp:
            scaler.scale(loss).backward()
            nn.utils.clip_grad_norm_(model.parameters(), CLIP_NORM)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), CLIP_NORM)
            optimizer.step()

        bsz = xb.size(0)
        loss_sum += float(loss.item()) * bsz
        total += bsz
    return loss_sum / max(1, total)

def evaluate(model, loader, device):
    model.eval()
    ce = nn.CrossEntropyLoss()
    total, loss_sum = 0, 0.0
    corr_c, corr_s = 0, 0
    with torch.no_grad():
        for xb, ycond, ysev in loader:
            xb, ycond, ysev = xb.to(device, non_blocking=True), ycond.to(device, non_blocking=True), ysev.to(device, non_blocking=True)
            pc, ps = model(xb)
            loss = ce(pc, ycond) + ce(ps, ysev)
            loss_sum += float(loss.item()) * xb.size(0)
            corr_c += int((pc.argmax(1) == ycond).sum().item())
            corr_s += int((ps.argmax(1) == ysev).sum().item())
            total  += xb.size(0)
    return (loss_sum / max(1, total),
            corr_c / max(1, total),
            corr_s / max(1, total))

# ------------------------------
# Generic per-source client builder (single-channel case)
# ------------------------------
def build_clients(items: List[Tuple[str, str, str]], max_clients: Optional[int] = None):
    by_src: Dict[str, List[Tuple[str, str]]] = {}
    for npy, meta, src in items:
        by_src.setdefault(src, []).append((npy, meta))
    keys = list(by_src.keys())
    random.shuffle(keys)
    if max_clients is not None:
        keys = keys[:max_clients]
    return {k: by_src[k] for k in keys}

# ------------------------------
# Federated loop (with vibration 4-ch fusion)
# ------------------------------
def run_federated_modality(
    modality: str,
    data_root: str,
    rounds: int,
    clients_per_round: int,
    local_epochs: int,
    batch_size: int,
    lr: float,
    num_workers: int,
    amp: bool,
    max_clients: Optional[int],
    device: torch.device,
    save_dir: str,
    # memory-safe FL params
    fl_freq_end: Optional[int],
    fl_col_stride: int,
    fl_time_crop: Optional[int],
    # final full-coverage params
    full_freq_end: Optional[int],
    full_col_stride: int,
    full_tile_len: int,
    full_tile_overlap: int,
    full_ft_epochs: int,
    full_ft_lr_scale: float,
):
    print(f"\n=== Modality: {modality} | Multitask CNN ===")
    mod_dir = os.path.join(data_root, modality)
    base_items = discover_items(mod_dir)
    if not base_items:
        print(f"[WARN] No .npy in {mod_dir}")
        return

    # label spaces
    cond_classes = ACOUSTIC_CONDITIONS if modality == "acoustic" else VIBRATION_CONDITIONS
    condition_to_idx = {c:i for i,c in enumerate(cond_classes)}
    sev_present = set()
    for _, meta, _ in base_items:
        try:
            with open(meta, "r") as fh:
                m = json.load(fh)
            _, _, sev = parse_labels_from_source(m.get("source_csv", ""))
            sev_present.add(sev)
        except Exception:
            pass

    def sev_num(s):
        n = re.sub(r"\D","", s or "")
        return int(n) if n else 0
    sev_sorted = sorted(sev_present, key=sev_num) or ["00","01","02","03"]
    severity_to_idx = {s:i for i,s in enumerate(sev_sorted)}
    n_cond, n_sev = len(condition_to_idx), len(severity_to_idx)

    # ----- build clients -----
    if modality == "vibration":
        # group 4-ch windows and assign to clients by source_csv
        grouped_dict, by_source = group_vibration_items(base_items)
        clients = by_source
        total_groups = sum(len(v) for v in by_source.values())
        print(f"Vibration groups (4-ch windows): {total_groups}")
    else:
        clients = build_clients(base_items, max_clients=max_clients)

    keys = list(clients.keys())
    random.shuffle(keys)
    if max_clients is not None:
        keys = keys[:max_clients]
    n_clients = len(keys)
    print(f"Total simulated clients: {n_clients}")

    # holdout
    val_key = keys[0]
    val_items = clients[val_key][: max(8, math.ceil(0.01*len(clients[val_key]))) ]

    # ----- model -----
    in_ch = 4 if modality == "vibration" else 1
    model = MultiTaskCNN(n_cond=n_cond, n_sev=n_sev, in_ch=in_ch).to(device)

    # ----- loader factory -----
    def make_loader(samples, shuffle, center, freq_end, col_stride, time_crop):
        if modality == "vibration":
            ds = ScalogramDataset4Ch(
                samples, condition_to_idx, severity_to_idx,
                log_amplitude=True,
                freq_start=None, freq_end=freq_end,
                col_stride=col_stride, time_crop=time_crop,
                center_crop=center,
            )
        else:
            ds = ScalogramDataset(
                samples, condition_to_idx, severity_to_idx,
                log_amplitude=True,
                freq_start=None, freq_end=freq_end,
                col_stride=col_stride, time_crop=time_crop,
                center_crop=center,
            )
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                          num_workers=num_workers, pin_memory=(device.type=='cuda'), drop_last=False)

    val_loader = make_loader(val_items, False, True, fl_freq_end, fl_col_stride, fl_time_crop)
    scaler_enabled = (amp and device.type == 'cuda')

    # ===== Federated rounds =====
    for rnd in range(1, rounds+1):
        t0 = time.time()
        selectable = [k for k in keys if k != val_key] or keys
        m = min(clients_per_round, len(selectable))
        chosen = random.sample(selectable, m)

        global_state = get_state_dict(model)
        updates: List[Tuple[Dict[str, torch.Tensor], int]] = []

        for ck in chosen:
            train_loader = make_loader(clients[ck], True, False, fl_freq_end, fl_col_stride, fl_time_crop)
            local = MultiTaskCNN(n_cond, n_sev, in_ch=in_ch).to(device)
            set_state_dict(local, global_state)
            opt = torch.optim.AdamW(local.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
            sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=max(1, local_epochs))
            scaler = make_grad_scaler(amp_enabled=scaler_enabled)

            for _ in range(local_epochs):
                train_one_epoch(local, train_loader, opt, device, scaler)
                sched.step()

            n_samples = len(train_loader.dataset)
            updates.append((get_state_dict(local), n_samples))

            del local, opt, sched, scaler, train_loader
            torch.cuda.empty_cache(); gc.collect()

        if updates:
            new_state = average_states(updates)
            set_state_dict(model, new_state)

        vloss, vacc_c, vacc_s = evaluate(model, val_loader, device)
        dt = time.time() - t0
        print(f"Round {rnd:03d} | val_loss {vloss:.4f} | acc(cond) {vacc_c:.4f} | acc(sev) {vacc_s:.4f} | {dt:.1f}s")

    # ===== Final full-coverage fine-tune =====
    print("\n[INFO] Starting full-coverage fine-tune (see ALL columns of ALL files)...")

    if modality == "vibration":
        # flatten all 4-ch groups from all clients
        all_groups = [g for lst in clients.values() for g in lst]
        full_ds = FullCoverageScalogramDataset4Ch(
            all_groups, condition_to_idx, severity_to_idx,
            log_amplitude=True,
            freq_start=None, freq_end=full_freq_end,
            col_stride=full_col_stride,
            tile_len=full_tile_len,
            tile_overlap=full_tile_overlap,
        )
    else:
        # acoustic: flatten back to list of (npy, meta)
        all_items = [pair for lst in clients.values() for pair in lst]
        full_ds = FullCoverageScalogramDataset(
            all_items, condition_to_idx, severity_to_idx,
            log_amplitude=True,
            freq_start=None, freq_end=full_freq_end,
            col_stride=full_col_stride,
            tile_len=full_tile_len,
            tile_overlap=full_tile_overlap,
        )

    full_loader = DataLoader(full_ds, batch_size=batch_size, shuffle=True,
                             num_workers=num_workers, pin_memory=(device.type=='cuda'), drop_last=False)

    # smaller LR for fine-tune
    ft_opt = torch.optim.AdamW(model.parameters(), lr=max(1e-6, lr * full_ft_lr_scale), weight_decay=WEIGHT_DECAY)
    ft_sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(ft_opt, T_0=max(1, full_ft_epochs))
    ft_scaler = make_grad_scaler(amp_enabled=scaler_enabled)

    for ep in range(1, full_ft_epochs+1):
        tr = train_one_epoch(model, full_loader, ft_opt, device, ft_scaler)
        ft_sched.step()
        vloss, vacc_c, vacc_s = evaluate(model, val_loader, device)
        print(f"[FT] epoch {ep:02d} | train_loss {tr:.4f} | val_loss {vloss:.4f} | acc(cond) {vacc_c:.4f} | acc(sev) {vacc_s:.4f}")

    # save
    os.makedirs(save_dir, exist_ok=True)
    out_path = os.path.join(save_dir, f"{modality}_multitask.pt")
    torch.save(model.state_dict(), out_path)
    with open(os.path.join(save_dir, f"{modality}_labels.json"), "w") as fh:
        json.dump({"conditions": list(condition_to_idx.keys()), "severities": list(severity_to_idx.keys())}, fh, indent=2)
    print(f"Saved: {out_path}")

# ------------------------------
# Entry
# ------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data-root', type=str, default='scalograms')
    p.add_argument('--save-dir', type=str, default='models')

    # Federated loop
    p.add_argument('--rounds', type=int, default=10)
    p.add_argument('--clients-per-round', type=int, default=10)
    p.add_argument('--local-epochs', type=int, default=1)
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--num-workers', type=int, default=4)
    p.add_argument('--max-clients', type=int, default=None)
    p.add_argument('--cuda', action='store_true')
    p.add_argument('--amp', action='store_true')

    # Memory-safe params for FL rounds
    p.add_argument('--fl-freq-end', type=int, default=128, help='keep lowest freq bins during FL (None=all)')
    p.add_argument('--fl-col-stride', type=int, default=16, help='time downsample during FL')
    p.add_argument('--fl-time-crop', type=int, default=512, help='fixed time crop during FL (None=keep all after stride)')

    # Final full-coverage fine-tune
    p.add_argument('--full-freq-end', type=int, default=128, help='keep lowest freq bins for final pass')
    p.add_argument('--full-col-stride', type=int, default=8, help='stride for final coverage (lower = more precise)')
    p.add_argument('--full-tile-len', type=int, default=1024, help='tile width after stride (covers all tiles)')
    p.add_argument('--full-tile-overlap', type=int, default=256, help='overlap between tiles (after stride)')
    p.add_argument('--full-ft-epochs', type=int, default=2, help='epochs for final fine-tune')
    p.add_argument('--full-ft-lr-scale', type=float, default=0.3, help='LR multiplier vs base lr during fine-tune')

    args = p.parse_args()
    device = torch.device('cuda' if (args.cuda and torch.cuda.is_available()) else 'cpu')
    print(f"Device: {device}")

    for modality in ['acoustic', 'vibration']:
        run_federated_modality(
            modality=modality,
            data_root=args.data_root,
            rounds=args.rounds,
            clients_per_round=args.clients_per_round,
            local_epochs=args.local_epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            num_workers=args.num_workers,
            amp=args.amp,
            max_clients=args.max_clients,
            device=device,
            save_dir=args.save_dir,
            # FL params
            fl_freq_end=args.fl_freq_end,
            fl_col_stride=args.fl_col_stride,
            fl_time_crop=args.fl_time_crop,
            # final pass params
            full_freq_end=args.full_freq_end,
            full_col_stride=args.full_col_stride,
            full_tile_len=args.full_tile_len,
            full_tile_overlap=args.full_tile_overlap,
            full_ft_epochs=args.full_ft_epochs,
            full_ft_lr_scale=args.full_ft_lr_scale,
        )

if __name__ == '__main__':
    main()
