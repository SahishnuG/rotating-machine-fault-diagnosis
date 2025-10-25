#!/usr/bin/env python3
"""
tdms_to_csv_subset.py

Process only the specified TDMS files:
  4Nm_BPFO_03.tdms, 4Nm_BPFO_10.tdms, 4Nm_BPFO_30.tdms

Writes CSV + JSON sidecar metadata for each file.

Usage:
    python tdms_to_csv_subset.py --tdms_dir "./current,temp" --out_dir "./current_temp"
"""

import os
import json
import argparse
from fractions import Fraction

import numpy as np
import pandas as pd
from nptdms import TdmsFile

# ----------------------------
# Config
# ----------------------------
DEFAULT_SAMPLING_RATE = 25600
DEFAULT_GROUP_NAME = "Log"
TIME_COL_NAME = "Time Stamp"
SIDECAR_SUFFIX = ".meta.json"

# Files to process (exact basenames)
TARGET_FILES = {"4Nm_BPFO_03.tdms", "4Nm_BPFO_10.tdms", "4Nm_BPFO_30.tdms"}

# Channel mapping for your TDMS group -> CSV column names (adjust if needed)
CHANNEL_MAPPING = {
    'cDAQ9185-1F486B5Mod1/ai0': 'Temperature_housing_A',
    'cDAQ9185-1F486B5Mod1/ai1': 'Temperature_housing_B',
    'cDAQ9185-1F486B5Mod2/ai0': 'U-phase',
    'cDAQ9185-1F486B5Mod2/ai2': 'V-phase',
    'cDAQ9185-1F486B5Mod2/ai3': 'W-phase'
}

# ----------------------------
# Helpers (robust extraction)
# ----------------------------
def _is_numeric_array(arr):
    try:
        if isinstance(arr, np.ndarray):
            return np.issubdtype(arr.dtype, np.number)
        if isinstance(arr, (list, tuple)):
            return all(isinstance(x, (int, float, np.floating, np.integer)) for x in arr)
    except Exception:
        return False
    return False

def get_channel_sampling_info(tdms_channel):
    # prefer time_track()
    try:
        times = tdms_channel.time_track()
        if times is not None and len(times) >= 2:
            diffs = np.diff(times)
            dt = np.median(diffs[diffs > 0]) if np.any(diffs > 0) else None
            fs = (1.0 / dt) if (dt and dt > 0) else None
            start = float(times[0]) if len(times) > 0 else None
            return fs, start, times
    except Exception:
        pass

    props = tdms_channel.properties or {}
    for key in ('wf_increment', 'wf_xincrement', 'wf_xinc', 'Time_Increment', 'sample_interval', 'wf_x0'):
        if key in props:
            try:
                increment = float(props[key])
                fs = 1.0 / increment if increment > 0 else None
                start = float(props.get('wf_start_time', props.get('wf_start', 0.0)))
                return fs, start, None
            except Exception:
                pass
    for key in ('sample_rate', 'SampleRate', 'sampleRate', 'rate'):
        if key in props:
            try:
                fs = float(props[key])
                start = float(props.get('wf_start_time', props.get('wf_start', 0.0)))
                return fs, start, None
            except Exception:
                pass
    return None, None, None

def safe_extract_group_v2(tdms_path, group_name, channel_mapping, fallback_fs=DEFAULT_SAMPLING_RATE):
    tdms = TdmsFile.read(tdms_path)
    group = next((g for g in tdms.groups() if g.name == group_name), None)
    if group is None:
        raise RuntimeError(f"Group '{group_name}' not found in {tdms_path}. Available: {[g.name for g in tdms.groups()]}")

    extracted = {}
    meta = {'file': os.path.basename(tdms_path), 'channels': {}, 'chosen_fs': None, 'times_source': None, 'skipped_channels': {}}

    for tdms_ch_name, csv_col in (channel_mapping or {}).items():
        try:
            ch = group[tdms_ch_name]
        except KeyError:
            meta['skipped_channels'][csv_col] = {'reason': 'channel_not_found'}
            print(f"⚠️ Channel '{tdms_ch_name}' not found in {tdms_path}. Skipping.")
            continue

        try:
            raw = ch[:]
        except Exception as e:
            try:
                props = dict(ch.properties)
            except Exception:
                props = "<could not read properties>"
            meta['skipped_channels'][csv_col] = {'reason': 'read_failed', 'error': repr(e), 'properties_sample': props}
            print(f"⚠️ Could not read samples for '{tdms_ch_name}': {e}. Skipping.")
            continue

        arr = np.asarray(raw)
        if arr.size == 0:
            meta['skipped_channels'][csv_col] = {'reason': 'zero_length'}
            print(f"⚠️ Channel '{tdms_ch_name}' returned zero-length data. Skipping.")
            continue

        if not _is_numeric_array(arr):
            if arr.dtype.names:
                pulled = None
                for nm in arr.dtype.names:
                    sub = arr[nm]
                    if np.issubdtype(sub.dtype, np.number):
                        pulled = np.asarray(sub)
                        break
                if pulled is None:
                    meta['skipped_channels'][csv_col] = {'reason': 'non_numeric_struct', 'dtype_names': list(arr.dtype.names)}
                    print(f"⚠️ Channel '{tdms_ch_name}' produced structured dtype without numeric field. Skipping.")
                    continue
                else:
                    arr = pulled
            else:
                meta['skipped_channels'][csv_col] = {'reason': 'non_numeric_dtype', 'dtype': str(arr.dtype)}
                print(f"⚠️ Channel '{tdms_ch_name}' has non-numeric dtype {arr.dtype}. Skipping.")
                continue

        try:
            fs, start_time, times_array = get_channel_sampling_info(ch)
        except Exception as e:
            fs, start_time, times_array = (None, None, None)
            print(f"⚠️ get_channel_sampling_info failed for {tdms_ch_name}: {e}")

        extracted[csv_col] = {'data': arr.astype(float), 'fs': fs, 'start_time': start_time, 'times': times_array}
        meta['channels'][csv_col] = {'fs': fs, 'start_time': start_time, 'has_times_array': bool(times_array is not None), 'n_samples': int(arr.size)}

    good_channels = [k for k, v in extracted.items() if v['data'].size > 0]
    if not good_channels:
        raise RuntimeError(f"No valid numeric channels found in {tdms_path}. See meta.skipped_channels for details.")

    min_len = min(int(extracted[k]['data'].size) for k in good_channels)
    for k in good_channels:
        if extracted[k]['data'].size > min_len:
            print(f"Trimming channel '{k}' from {extracted[k]['data'].size} -> {min_len} samples")
            extracted[k]['data'] = extracted[k]['data'][:min_len]
            if extracted[k]['times'] is not None and len(extracted[k]['times']) > min_len:
                extracted[k]['times'] = extracted[k]['times'][:min_len]

    times_candidates = [extracted[k]['times'] for k in good_channels if extracted[k]['times'] is not None]
    if times_candidates:
        final_times = times_candidates[0]
        meta['times_source'] = 'channel_time_track'
        meta['chosen_fs'] = None
    else:
        fs_list = [extracted[k]['fs'] for k in good_channels if extracted[k]['fs'] is not None]
        if fs_list:
            chosen_fs = float(np.median(fs_list))
            meta['chosen_fs'] = chosen_fs
            final_times = np.arange(min_len) / chosen_fs
            meta['times_source'] = 'median_channel_fs'
        else:
            chosen_fs = float(fallback_fs)
            meta['chosen_fs'] = chosen_fs
            final_times = np.arange(min_len) / chosen_fs
            meta['times_source'] = 'fallback_fs'

    df = pd.DataFrame({TIME_COL_NAME: final_times})
    for k in good_channels:
        df[k] = extracted[k]['data'][:min_len]

    for k in good_channels:
        if extracted[k]['times'] is not None:
            dt = np.diff(extracted[k]['times'])
            dt_pos = dt[dt > 0]
            fs_est = float(1.0 / np.median(dt_pos)) if dt_pos.size > 0 else None
            meta['channels'][k]['fs_est_from_times'] = fs_est
        if meta['channels'][k].get('n_samples') is None:
            meta['channels'][k]['n_samples'] = int(min_len)

    return df, meta

def write_csv_and_sidecar(df, out_csv_path, meta):
    out_dir = os.path.dirname(out_csv_path)
    os.makedirs(out_dir, exist_ok=True)
    df.to_csv(out_csv_path, index=False)
    print(f"✅ Saved CSV -> {out_csv_path} (rows={len(df)})")
    sidecar_path = out_csv_path + SIDECAR_SUFFIX
    with open(sidecar_path, 'w') as fh:
        json.dump(meta, fh, indent=2)
    print(f"   Sidecar metadata -> {sidecar_path}")

# ----------------------------
# Main driver: process subset
# ----------------------------
def main(tdms_dir, out_dir, group_name, fallback_fs):
    tdms_dir = os.path.abspath(tdms_dir)
    out_dir = os.path.abspath(out_dir)
    if not os.path.isdir(tdms_dir):
        raise FileNotFoundError(f"TDMS directory not found: {tdms_dir}")
    files = [f for f in os.listdir(tdms_dir) if f.lower().endswith('.tdms') and f in TARGET_FILES]
    if not files:
        print("No target TDMS files found in", tdms_dir)
        return
    for fn in sorted(files):
        tdms_path = os.path.join(tdms_dir, fn)
        print("Processing:", tdms_path)
        try:
            df, meta = safe_extract_group_v2(tdms_path, group_name, CHANNEL_MAPPING, fallback_fs=fallback_fs)
            base = os.path.splitext(fn)[0]
            csv_name = base + '.csv'
            out_csv = os.path.join(out_dir, csv_name)
            cols = [TIME_COL_NAME] + list(df.columns.drop(TIME_COL_NAME))
            df = df[cols]
            write_csv_and_sidecar(df, out_csv, meta)
        except Exception as e:
            print(f"❌ Error processing {fn}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tdms_dir", type=str, default="./current,temp")
    parser.add_argument("--out_dir", type=str, default="./current_temp")
    parser.add_argument("--group", type=str, default=DEFAULT_GROUP_NAME)
    parser.add_argument("--fallback_fs", type=float, default=DEFAULT_SAMPLING_RATE)
    args = parser.parse_args()

    print("TDMS dir:", os.path.abspath(args.tdms_dir))
    main(args.tdms_dir, args.out_dir, args.group, args.fallback_fs)
