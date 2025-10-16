import os
from nptdms import TdmsFile
import pandas as pd

# Path to your folder
folder_path = os.path.join(os.getcwd(), "current,temp")

# Output folder (optional)
output_folder = os.path.join(folder_path, "csv_exports")
os.makedirs(output_folder, exist_ok=True)

# Iterate over all files in the folder
for filename in os.listdir(folder_path):
    if filename.endswith(".tdms"):
        file_path = os.path.join(folder_path, filename)
        print(f"Reading: {filename}")

        # Read the TDMS file
        try:
            tdms_file = TdmsFile.read(file_path)
            
            # Convert to DataFrame
            df = tdms_file.as_dataframe()

            # Save as CSV (same name, .csv extension)
            output_name = os.path.splitext(filename)[0] + ".csv"
            output_path = os.path.join(output_folder, output_name)
            df.to_csv(output_path, index=False)
            
            print(f"✅ Saved: {output_name} ({len(df)} rows)")
        
        except Exception as e:
            print(f"❌ Error reading {filename}: {e}")

print("\nAll TDMS files processed!")
