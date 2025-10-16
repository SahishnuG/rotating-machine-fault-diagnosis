#!/usr/bin/env python3
"""
convert_tdms_to_standard_csv_automap.py

Read TDMS files from ./current,temp, automatically map channels (using TDMS channel/group names)
to standardized column names, and write standardized CSVs to ./current_temp.

Requires: nptdms, pandas
pip install nptdms pandas
"""
import os
import re
import math
import numpy as np
import pandas as pd
from nptdms import TdmsFile

# === CONFIG ===
SOURCE_FOLDER = os.path.join(os.getcwd(), "current,temp")
OUTPUT_FOLDER = os.path.join(os.getcwd(), "current_temp")
DEFAULT_DT = 1.0  # seconds per sample if no timestamp is present (adjust if you know the real sample rate)

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Target column order/names
TARGET_COLS = [
    "Time Stamp",
    "Temperature_housing_A",
    "Temperature_housing_B",
    "U-phase",
    "V-phase",
    "W-phase",
    "load",
    "condition",
    "severity",
]

# Regex mapping rules (checked in order). Each pattern maps to a target name.
# Key: compiled regex; Value: target column name
REGEX_TO_TARGET = [
    # Temperatures
    (re.compile(r"(temp|temperature).*a", re.I), "Temperature_housing_A"),
    (re.compile(r"(temp|temperature).*b", re.I), "Temperature_housing_B"),
    (re.compile(r"(temp|temperature).*1", re.I), "Temperature_housing_A"),
    (re.compile(r"(temp|temperature).*2", re.I), "Temperature_housing_B"),
    (re.compile(r"(ai0|ai_0|ai-0|chan0|ch0)\b", re.I), "Temperature_housing_A"),
    (re.compile(r"(ai1|ai_1|ai-1|chan1|ch1)\b", re.I), "Temperature_housing_B"),
    # Currents / phases
    (re.compile(r"(^|[^a-zA-Z])(u|phase.?u|u_phase|i_u|iu)\b", re.I), "U-phase"),
    (re.compile(r"(^|[^a-zA-Z])(v|phase.?v|v_phase|i_v|iv)\b", re.I), "V-phase"),
    (re.compile(r"(^|[^a-zA-Z])(w|phase.?w|w_phase|i_w|iw)\b", re.I), "W-phase"),
    (re.compile(r"(ai2|ai_2|ai-2|chan2|ch2)\b", re.I), "U-phase"),
    (re.compile(r"(ai3|ai_3|ai-3|chan3|ch3)\b", re.I), "V-phase"),
    (re.compile(r"(ai4|ai_4|ai-4|chan4|ch4)\b", re.I), "W-phase"),
    # Generic current / i-phase fallback
    (re.compile(r"(current|i_phase|i-phase|amp|amps|a_phase|phase)", re.I), "U-phase"),
]


def parse_filename_metadata(filename):
    """Parse aaaaNm_bbbb_cccc.tdms style filenames returning dict."""
    base = os.path.splitext(filename)[0]
    parts = base.split("_")
    if len(parts) >= 3:
        return {"load": parts[0], "condition": parts[1], "severity": parts[2]}
    else:
        return {"load": None, "condition": None, "severity": None}


def try_map_channel_name(channel_label):
    """
    Try to map a single channel label (string) to one of the target columns
    using REGEX_TO_TARGET rules. Returns target name or None.
    """
    if not isinstance(channel_label, str):
        return None
    s = channel_label.replace("/", " ").replace("'", " ").replace("_", " ").strip()
    for (rgx, target) in REGEX_TO_TARGET:
        if rgx.search(s):
            return target
    return None


def collapse_multiindex_col(col):
    """Given a pandas MultiIndex column (group, channel) return a single label string."""
    if isinstance(col, tuple):
        # join group and channel with a slash, e.g. "Group/Channel"
        return f"{col[0]}/{col[1]}"
    return str(col)


def extract_dataframe_from_tdms(tdms_path):
    """
    Read TDMS and return a DataFrame (flat columns).
    Use TdmsFile.as_dataframe() which returns MultiIndex columns (group, channel).
    We'll collapse them to "group/channel" strings as column labels.
    """
    tdms = TdmsFile.read(tdms_path)
    # as_dataframe may include a time channel if present
    df = tdms.as_dataframe()
    # If DataFrame has MultiIndex columns, collapse them to a single level
    if isinstance(df.columns, pd.MultiIndex):
        new_cols = [collapse_multiindex_col(c) for c in df.columns.to_list()]
        df.columns = new_cols
    else:
        # simple index: ensure string names
        df.columns = [str(c) for c in df.columns]
    return df


def find_time_column(df):
    """
    Find a time-like column in df based on name heuristics.
    Returns column name or None.
    """
    for col in df.columns:
        low = col.lower()
        if "time" in low or "timestamp" in low or "timest" in low:
            return col
        # some exports name it like "index" or "sample_time"
        if re.search(r"\b(index|sample|sample_time|timed)\b", low):
            return col
    return None


def standardize_tdms_dataframe(df_raw, filename):
    """
    Given the raw DataFrame from a TDMS file, create a standardized DataFrame
    with columns:
    Time Stamp, Temperature_housing_A, Temperature_housing_B, U-phase, V-phase, W-phase, load, condition, severity
    """
    df = df_raw.copy()

    # 1) detect time column (if any)
    time_col = find_time_column(df)
    if time_col is not None:
        time_values = df[time_col].to_numpy()
        # drop time column from data channels so mapping maps only to physical channels
        df = df.drop(columns=[time_col])
    else:
        # create time vector
        n = len(df)
        time_values = np.arange(n) * DEFAULT_DT

    # 2) attempt to map each available channel to a target using channel names
    mapping = {}  # column_name -> target_name
    assigned_targets = set()

    for col in df.columns:
        mapped = try_map_channel_name(col)
        if mapped and mapped not in assigned_targets:
            mapping[col] = mapped
            assigned_targets.add(mapped)

    # 3) If not enough targets found, try matching using column-order heuristics:
    #    first two unmatched -> temps, next three -> U,V,W
    remaining_cols = [c for c in df.columns if c not in mapping.keys()]
    # fill temperatures first
    temp_targets = ["Temperature_housing_A", "Temperature_housing_B"]
    curr_targets = ["U-phase", "V-phase", "W-phase"]

    # assign any missing temp targets by scanning remaining columns for 'temp' if not already assigned
    for t in temp_targets:
        if t not in assigned_targets:
            # try to find a remaining column whose name contains 'temp' explicitly
            found = None
            for c in remaining_cols:
                if re.search(r"temp|temperature", c, re.I):
                    found = c
                    break
            if found:
                mapping[found] = t
                assigned_targets.add(t)
                remaining_cols.remove(found)

    # Next: assign remaining by order (first 2 -> temps, next 3 -> currents)
    # Count which targets still missing
    missing_temps = [t for t in temp_targets if t not in assigned_targets]
    missing_currs = [t for t in curr_targets if t not in assigned_targets]

    idx = 0
    # fill missing temps
    while missing_temps and idx < len(remaining_cols):
        col = remaining_cols[idx]
        mapping[col] = missing_temps.pop(0)
        idx += 1
    # fill missing currents
    while missing_currs and idx < len(remaining_cols):
        col = remaining_cols[idx]
        mapping[col] = missing_currs.pop(0)
        idx += 1

    # If still missing targets, create NaN placeholders later

    # 4) Build standardized DataFrame
    nrows = len(df)
    out = pd.DataFrame(index=range(nrows))
    out["Time Stamp"] = time_values

    # Place mapped columns
    for col, target in mapping.items():
        out[target] = df[col].to_numpy()

    # Ensure all target channels exist; if not, fill with NaN
    for t in ["Temperature_housing_A", "Temperature_housing_B", "U-phase", "V-phase", "W-phase"]:
        if t not in out.columns:
            out[t] = np.nan

    # 5) Add filename metadata
    meta = parse_filename_metadata(filename)
    out["load"] = meta["load"]
    out["condition"] = meta["condition"]
    out["severity"] = meta["severity"]

    # Reorder columns exactly as desired
    col_order = [
        "Time Stamp",
        "Temperature_housing_A",
        "Temperature_housing_B",
        "U-phase",
        "V-phase",
        "W-phase",
        "load",
        "condition",
        "severity",
    ]
    out = out[col_order]

    return out, mapping


def summarize_mapping(mapping):
    """Nice printable summary of mapping per file for logging."""
    if not mapping:
        return "(no automatic mapping)"
    items = [f"{k} -> {v}" for k, v in mapping.items()]
    return "; ".join(items)


# === Main processing loop ===
def main():
    tdms_files = [f for f in os.listdir(SOURCE_FOLDER) if f.lower().endswith(".tdms")]
    if not tdms_files:
        print(f"No .tdms files found in {SOURCE_FOLDER}")
        return

    for fname in tdms_files:
        fpath = os.path.join(SOURCE_FOLDER, fname)
        try:
            print(f"\nReading: {fname}")
            df_raw = extract_dataframe_from_tdms(fpath)
            df_std, mapping = standardize_tdms_dataframe(df_raw, fname)

            out_name = os.path.splitext(fname)[0] + ".csv"
            out_path = os.path.join(OUTPUT_FOLDER, out_name)
            df_std.to_csv(out_path, index=False)
            print(f"Saved standardized CSV: {out_path}")
            print("Mapping used:", summarize_mapping(mapping))

        except Exception as e:
            print(f"Error processing {fname}: {e}")


if __name__ == "__main__":
    main()
