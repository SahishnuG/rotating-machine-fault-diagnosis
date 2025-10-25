# analyze_sampling_rates.py
import os
import glob
import math
import json
import argparse
from typing import List, Dict, Any, Tuple

import numpy as np
import pandas as pd
from scipy import signal

# -------------------------
# Configuration defaults
# -------------------------
MAX_SAMPLE_ROWS = 200000      # max rows to read per sampled chunk (adjust for memory)
SEGMENT_ROWS = 100000         # rows per segment (start/middle/end)
GAP_MULTIPLIER = 5.0          # dt > GAP_MULTIPLIER * median_dt considered a 'big gap'
PSDF_NPERSEG = 65536          # nperseg for Welch (will be limited to data length)
RECOMMEND_MULTIPLIER = 3.0    # target_fs = RECOMMEND_MULTIPLIER * f99 (capped at raw_fs if known)
MIN_FS_FOR_CWT = 50          # if recommended fs < 50 Hz, raise to 50 (practical lower bound)
VERBOSE = True

# -------------------------
# Helpers
# -------------------------
def find_time_column(df: pd.DataFrame) -> str:
    # look for 'Time' and 'Stamp' in column name
    for c in df.columns:
        if 'time' in c.lower() and 'stamp' in c.lower():
            return c
    # fallback: first numeric column
    for c in df.columns:
        if np.issubdtype(df[c].dtype, np.number):
            return c
    return df.columns[0]

def load_segment(file_path: str, start_row: int, nrows: int, usecols: List[str]=None) -> pd.DataFrame:
    # pandas skiprows: skip header row count unknown -> use header=0 and skiprows
    # but skiprows expects line numbers; we convert start_row to skiprows=(1+start_row)
    skiprows = list(range(1, start_row+1)) if start_row>0 else None
    df = pd.read_csv(file_path, nrows=nrows, skiprows=skiprows, header=0, usecols=usecols)
    return df

def basic_dt_stats(times: np.ndarray) -> Dict[str, Any]:
    dt = np.diff(times)
    dt_pos = dt[dt > 0]
    if dt_pos.size == 0:
        return {'median_dt': None, 'fs_est': None, 'dt_percentiles': None, 'max_dt': None, 'big_gaps': 0}
    median_dt = float(np.median(dt_pos))
    fs_est = float(1.0 / median_dt) if median_dt > 0 else None
    percentiles = np.percentile(dt_pos, [0.1,1,5,50,95,99,99.9]).tolist()
    max_dt = float(dt_pos.max())
    big_gaps = int(np.sum(dt_pos > GAP_MULTIPLIER * median_dt))
    return {'median_dt': median_dt, 'fs_est': fs_est, 'dt_percentiles': percentiles, 'max_dt': max_dt, 'big_gaps': big_gaps}

def compute_welch_fperc(signal_vec: np.ndarray, fs: float, pcts=[0.95, 0.99]) -> Dict[str, Any]:
    # guard
    nperseg = min(PSDF_NPERSEG, max(256, len(signal_vec)//2))
    if len(signal_vec) < 64:
        return {'f_pcts': {p: None for p in pcts}, 'f_max': None}
    f, Pxx = signal.welch(signal_vec, fs=fs, nperseg=nperseg)
    Pxx = np.where(Pxx < 0, 0, Pxx)
    cum = np.cumsum(Pxx)
    if cum[-1] <= 0:
        return {'f_pcts': {p: None for p in pcts}, 'f_max': float(f.max())}
    cum = cum / cum[-1]
    f_pcts = {}
    for p in pcts:
        idx = np.searchsorted(cum, p)
        if idx >= len(f):
            f_pcts[p] = float(f[-1])
        else:
            f_pcts[p] = float(f[idx])
    return {'f_pcts': f_pcts, 'f_max': float(f.max())}

# -------------------------
# Analyze one file
# -------------------------
def analyze_file(file_path: str, channel_cols: List[str], max_rows: int = MAX_SAMPLE_ROWS) -> Dict[str, Any]:
    result = {'file': file_path}
    # Quick file size / line count (may be large but is informative)
    try:
        with open(file_path, 'rb') as fh:
            # count lines
            total_lines = 0
            for _ in fh:
                total_lines += 1
        # subtract header line
        total_rows = max(0, total_lines - 1)
    except Exception:
        total_rows = None

    result['total_rows'] = total_rows

    # Decide sample positions (start, middle, end)
    segments = []
    try:
        # read header to get columns
        df_header = pd.read_csv(file_path, nrows=2)
        time_col = find_time_column(df_header)
        # choose usecols to speed up
        usecols = [time_col] + [c for c in channel_cols if c in df_header.columns]
        # read start
        seg0 = pd.read_csv(file_path, nrows=min(SEGMENT_ROWS, max_rows), usecols=usecols)
        segments.append(('start', seg0))
        # mid
        if total_rows and total_rows > 3 * SEGMENT_ROWS:
            mid_row = max(SEGMENT_ROWS, total_rows//2 - SEGMENT_ROWS//2)
            seg_mid = load_segment(file_path, start_row=mid_row, nrows=min(SEGMENT_ROWS, max_rows), usecols=usecols)
            segments.append(('middle', seg_mid))
        # end
        if total_rows and total_rows > 2 * SEGMENT_ROWS:
            end_row = max(0, total_rows - SEGMENT_ROWS)
            seg_end = load_segment(file_path, start_row=end_row, nrows=min(SEGMENT_ROWS, max_rows), usecols=usecols)
            segments.append(('end', seg_end))
    except Exception as e:
        result['error'] = f'Could not sample file: {e}'
        return result

    # Analyze each segment
    seg_summaries = {}
    psd_agg_f95 = []
    psd_agg_f99 = []
    fs_candidates = []
    gap_flag = False

    for name, seg in segments:
        if seg.shape[0] < 3:
            seg_summaries[name] = {'rows': seg.shape[0]}
            continue
        time_col = find_time_column(seg)
        times = seg[time_col].to_numpy(dtype=float)
        # normalize times so they start at 0 (for relative dt)
        times_rel = times - times[0]
        dt_stats = basic_dt_stats(times_rel)
        seg_summaries[name] = {'rows': int(seg.shape[0]), 'time_col': time_col, **dt_stats}
        if dt_stats['fs_est'] is not None:
            fs_candidates.append(dt_stats['fs_est'])
        if dt_stats['big_gaps'] > 0:
            gap_flag = True

        # pick first available numeric channel from channel_cols present for PSD
        available_channels = [c for c in channel_cols if c in seg.columns]
        if not available_channels:
            continue
        ch = available_channels[0]
        sig = seg[ch].to_numpy(dtype=float)
        # remove mean and detrend small piece
        sig = sig - np.mean(sig)
        try:
            sig = signal.detrend(sig)
        except Exception:
            pass

        # choose fs for PSD: prefer dt_stats fs_est
        fs_for_psd = dt_stats['fs_est'] if dt_stats['fs_est'] is not None else 1.0
        psd_info = compute_welch_fperc(sig, fs_for_psd, pcts=[0.95, 0.99])
        seg_summaries[name]['psd'] = psd_info
        f95 = psd_info['f_pcts'].get(0.95) if psd_info['f_pcts'] else None
        f99 = psd_info['f_pcts'].get(0.99) if psd_info['f_pcts'] else None
        if f95:
            psd_agg_f95.append(f95)
        if f99:
            psd_agg_f99.append(f99)

    # Consolidate fs
    fs_raw = None
    if fs_candidates:
        # take median candidate
        fs_raw = float(np.median(fs_candidates))
    # if no candidate but total_rows and time field exists, estimate from entire file first few rows
    result['fs_raw_median_candidates'] = fs_candidates
    result['fs_raw'] = fs_raw

    # pick aggregated PSD values (median across segments)
    agg_f95 = float(np.median(psd_agg_f95)) if psd_agg_f95 else None
    agg_f99 = float(np.median(psd_agg_f99)) if psd_agg_f99 else None
    result['f95'] = agg_f95
    result['f99'] = agg_f99

    # recommended target fs
    if agg_f99:
        recommended = int(min(fs_raw if fs_raw else 1e9, math.ceil(RECOMMEND_MULTIPLIER * agg_f99)))
        # floor to MIN_FS_FOR_CWT
        recommended = max(recommended, MIN_FS_FOR_CWT)
    elif fs_raw:
        recommended = int(fs_raw)
    else:
        recommended = None

    result['recommended_target_fs'] = recommended
    result['has_big_gaps'] = gap_flag
    result['segments'] = seg_summaries
    return result

# -------------------------
# Main driver
# -------------------------
def analyze_folder(folder: str, channel_cols: List[str], out_csv: str):
    files = sorted(glob.glob(os.path.join(folder, '*.csv')))
    all_reports = []
    for f in files:
        print('Analyzing', f)
        rep = analyze_file(f, channel_cols)
        all_reports.append(rep)
    # flatten to CSV
    rows = []
    for r in all_reports:
        base = {'file': r.get('file'), 'total_rows': r.get('total_rows'), 'fs_raw': r.get('fs_raw'),
                'f95': r.get('f95'), 'f99': r.get('f99'), 'recommended_target_fs': r.get('recommended_target_fs'),
                'has_big_gaps': r.get('has_big_gaps')}
        # add segment summaries (start/middle/end)
        segs = r.get('segments', {})
        for segname in ('start','middle','end'):
            s = segs.get(segname)
            if s:
                base[f'{segname}_rows'] = s.get('rows')
                base[f'{segname}_fs_est'] = s.get('fs_est')
                base[f'{segname}_median_dt'] = s.get('median_dt')
                psd = s.get('psd')
                if psd and 'f_pcts' in psd:
                    base[f'{segname}_f95'] = psd['f_pcts'].get(0.95)
                    base[f'{segname}_f99'] = psd['f_pcts'].get(0.99)
            else:
                base[f'{segname}_rows'] = None
                base[f'{segname}_fs_est'] = None
                base[f'{segname}_median_dt'] = None
                base[f'{segname}_f95'] = None
                base[f'{segname}_f99'] = None
        rows.append(base)
    df_out = pd.DataFrame(rows)
    df_out.to_csv(out_csv, index=False)
    print('Wrote report to', out_csv)
    return df_out

# -------------------------
# CLI
# -------------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Analyze sampling rates for CSVs in a folder')
    parser.add_argument('--acoustic_folder', type=str, default='acoustic', help='Path to acoustic CSV folder')
    parser.add_argument('--vibration_folder', type=str, default='vibration', help='Path to vibration CSV folder')
    parser.add_argument('--out', type=str, default='sampling_report.csv', help='Output CSV report path')
    parser.add_argument('--acoustic_channels', type=str, default='values', help='Comma-separated acoustic channel column names')
    parser.add_argument('--vibration_channels', type=str, default='x_direction_housing_A,y_direction_housing_A,x_direction_housing_B,y_direction_housing_B', help='Comma-separated vibration channel names')
    args = parser.parse_args()

    acoustic_cols = [c.strip() for c in args.acoustic_channels.split(',') if c.strip()]
    vibration_cols = [c.strip() for c in args.vibration_channels.split(',') if c.strip()]

    out_acoustic = os.path.splitext(args.out)[0] + '_acoustic.csv'
    out_vibration = os.path.splitext(args.out)[0] + '_vibration.csv'

    if os.path.isdir(args.acoustic_folder):
        analyze_folder(args.acoustic_folder, acoustic_cols, out_acoustic)
    else:
        print('Acoustic folder not found:', args.acoustic_folder)

    if os.path.isdir(args.vibration_folder):
        analyze_folder(args.vibration_folder, vibration_cols, out_vibration)
    else:
        print('Vibration folder not found:', args.vibration_folder)
