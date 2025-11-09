# --- add at top of CONFIG ---
CURRENT_TEMP_FOLDER = "current_temp"

CURRENT_TEMP_CHANNELS = [
    "Temperature_housing_A",
    "Temperature_housing_B",
    "U-phase",
    "V-phase",
    "W-phase",
]

# Optional: switch to STFT for current/temperature (often better for quasi-stationary signals)
USE_STFT_FOR_CURRENT_TEMP = True
STFT_NPERSEG = 1024
STFT_NOVERLAP = 768  # 25% hop

# Streamed reading (chunk size in rows)
CSV_CHUNK_ROWS = 1_000_000  # tune per machine
