from scipy.io import loadmat
import numpy as np
import pandas as pd
import os

# ------------------------------------------------------------------------
# 1. CONFIGURATION

# Base directories for both datasets
DATASETS = {
    'acoustic': {
        'BASE_PATH': 'D:/Sahishnu/VIT/TY/Sem 5/ML/CP/ML_CP/acoustic/',
        'OUTPUT_PATH': 'D:/Sahishnu/VIT/TY/Sem 5/ML/CP/ML_CP/acoustic_csv_data/'
    },
    'vibration': {
        'BASE_PATH': 'D:/Sahishnu/VIT/TY/Sem 5/ML/CP/ML_CP/vibration/',
        'OUTPUT_PATH': 'D:/Sahishnu/VIT/TY/Sem 5/ML/CP/ML_CP/vibration_csv_data/'
    }
}

# Sampling rate and time step (adjust if needed)
TIME_STEP_S = 1 / 51200  # 51.2 kHz sampling rate

# ------------------------------------------------------------------------
# 2. CORE EXTRACTION UTILITY

def get_true_numeric_array(wrapper):
    """
    Safely extracts the numeric array from MATLAB structs with varying nesting levels.
    """
    try:
        return wrapper[0][0][0].flatten()
    except:
        try:
            return wrapper[0][0].flatten()
        except:
            return wrapper[0].flatten()

def convert_mat_to_csv(file_path, output_dir, signal_label):
    """
    Loads a single MAT file, extracts x and y values, and saves them to CSV.
    """
    file_name = os.path.basename(file_path)
    csv_file_name = file_name.replace('.mat', '.csv')
    csv_file_path = os.path.join(output_dir, csv_file_name)

    try:
        # Load the MAT file
        data = loadmat(file_path)

        # Access the main Signal structure
        signal_struct = data['Signal'][0][0]

        # Extract x and y components
        x_struct = signal_struct['x_values']
        y_struct = signal_struct['y_values']

        x_values = get_true_numeric_array(x_struct).astype(np.float64)
        y_values = get_true_numeric_array(y_struct).astype(np.float64)

        # Handle cases where x_values might be empty (generate synthetic time)
        if x_values.size == 0:
            num_samples = y_values.size
            x_values = np.arange(num_samples) * TIME_STEP_S

        # Skip empty signals
        if y_values.size == 0:
            print(f"  -> Skipped (empty signal): {file_name}")
            return

        # Create DataFrame
        df = pd.DataFrame({
            'Time_s': x_values,
            signal_label: y_values
        })

        # Save to CSV
        df.to_csv(csv_file_path, index=False)
        print(f"  -> SUCCESS: Saved {csv_file_name} ({len(df)} samples)")

    except FileNotFoundError:
        print(f"  -> ERROR: File not found: {file_path}")
    except KeyError as e:
        print(f"  -> ERROR: Missing expected key in {file_name}: {e}")
    except Exception as e:
        print(f"  -> ERROR: Failed conversion for {file_name}. Reason: {e}")

# ------------------------------------------------------------------------
# 3. EXECUTION

if __name__ == '__main__':
    print("\nStarting conversion of Acoustic and Vibration MAT files to CSV:")
    print("--------------------------------------------------------------")

    for dataset_name, paths in DATASETS.items():
        base = paths['BASE_PATH']
        output = paths['OUTPUT_PATH']
        signal_label = 'Acoustic_Pa' if dataset_name == 'acoustic' else 'Vibration'

        # Ensure output folder exists
        os.makedirs(output, exist_ok=True)

        print(f"\nProcessing {dataset_name.upper()} dataset...")
        mat_files = [f for f in os.listdir(base) if f.endswith('.mat')]

        if not mat_files:
            print(f"  ⚠️  No .mat files found in {base}")
            continue

        for file_name in mat_files:
            file_path = os.path.join(base, file_name)
            convert_mat_to_csv(file_path, output, signal_label)

    print("--------------------------------------------------------------")
    print("✅ Conversion Complete. Check 'acoustic_csv_data/' and 'vibration_csv_data/' folders.")
