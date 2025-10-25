from nptdms import TdmsFile
import os

def print_tdms_structure(tdms_file_path):
    try:
        tdms_file = TdmsFile.read(tdms_file_path)
        print(f"Inspecting TDMS File: {tdms_file_path}\n")
        
        print("Groups and Channels:")
        for group in tdms_file.groups():
            print(f"\n🔹 Group: {group.name}")
            for channel in group.channels():
                print(f"    Channel: {channel.name}")
                data = channel[:]
                print(f"        - Data Sample (first 5 points): {data[:5]}")
                
    except Exception as e:
        print(f"An error occurred while reading the TDMS file: {e}")

if __name__ == "__main__":
    tdms_file_path ='/current,temp/0Nm_BPFI_03.tdms'
    
    if os.path.exists(tdms_file_path):
        print_tdms_structure(tdms_file_path)
    else:
        print(f"TDMS file not found at {tdms_file_path}. Please check the path and try again.")

from nptdms import TdmsFile
import pandas as pd
import numpy as np
import os

# Define the sampling rate (samples per second)
SAMPLING_RATE = 25600  # 25.6 kHz

def construct_time_stamp(num_samples, sampling_rate):
    increment = 1.0 / sampling_rate
    return np.round(np.arange(num_samples) * increment, decimals=6)
    
def extract_channels(group, channel_mapping):
    data = {}
    for tdms_channel, csv_column in channel_mapping.items():
        try:
            channel = group[tdms_channel]
            data[csv_column] = channel[:]
        except KeyError:
            print(f"Channel '{tdms_channel}' not found in group '{group.name}'. Skipping this channel.")
    return data

def convert_tdms_to_csv(tdms_directory, csv_directory, group_name, channel_mapping, sampling_rate):
    # Ensure the CSV directory exists
    os.makedirs(csv_directory, exist_ok=True)

    # Iterate through all TDMS files in the directory
    for filename in os.listdir(tdms_directory):
        if filename.lower().endswith('.tdms'):
            tdms_path = os.path.join(tdms_directory, filename)
            print(f"Processing TDMS File: {tdms_path}")
            try:
                tdms_file = TdmsFile.read(tdms_path)
                
                # Group Check
                group_found = False
                for group in tdms_file.groups():
                    if group.name == group_name:
                        selected_group = group
                        group_found = True
                        break

                if not group_found:
                    print(f"Group '{group_name}' not found in {filename}. Skipping.\n")
                    continue

                # Extract Channels
                extracted_data = extract_channels(selected_group, channel_mapping)

                if not extracted_data:
                    print(f"No channels extracted from {filename}. Skipping.\n")
                    continue

                # Determine the number of samples from one of the channels
                num_samples = len(next(iter(extracted_data.values())))

                # Construct Time Stamp
                time_stamp = construct_time_stamp(num_samples, sampling_rate)

                # Create DataFrame
                df = pd.DataFrame({
                    'Time Stamp (s)': time_stamp,
                    'Temperature_housing_A (°C)': extracted_data.get('Temperature_housing_A', pd.NA),
                    'Temperature_housing_B (°C)': extracted_data.get('Temperature_housing_B', pd.NA),
                    'U-phase (A)': extracted_data.get('U-phase', pd.NA),
                    'V-phase (A)': extracted_data.get('V-phase', pd.NA),
                    'W-phase (A)': extracted_data.get('W-phase', pd.NA)
                })

                # Define the CSV file path
                csv_filename = os.path.splitext(filename)[0] + '.csv'
                csv_path = os.path.join(csv_directory, csv_filename)

                # Save to CSV
                df.to_csv(csv_path, index=False)
                print(f"Saved CSV to {csv_path}\n")

            except Exception as e:
                print(f"An error occurred while processing {filename}: {e}\n")
                continue


tdms_directory = './current,temp'
csv_directory = './current_temp'

# Group name
group_name = 'Log'

# Define channel mapping: TDMS channel names to desired CSV column names
channel_mapping = {
    'cDAQ9185-1F486B5Mod1/ai0': 'Temperature_housing_A',
    'cDAQ9185-1F486B5Mod1/ai1': 'Temperature_housing_B',
    'cDAQ9185-1F486B5Mod2/ai0': 'U-phase',
    'cDAQ9185-1F486B5Mod2/ai2': 'V-phase',
    'cDAQ9185-1F486B5Mod2/ai3': 'W-phase'
}

# Sampling rate (Hz) - 25.6 kHz
SAMPLING_RATE = 25600  # 25.6 kHz

# Convert TDMS files to CSV
convert_tdms_to_csv(tdms_directory, csv_directory, group_name, channel_mapping, SAMPLING_RATE)

