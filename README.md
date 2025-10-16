1. Download [Vibration, Acoustic, Temperature, and Motor Current Dataset of Rotating Machine Under Varying Load Conditions for Fault Diagnosis](https://data.mendeley.com/datasets/ztmf3m7h5x/6) and extract it in your local machine
2. Extract acoustic, current,temp and vibration
3. clone this repo and add the extracted folders in the repo (same dir as main.py)
4. create venv and install requirements
5. run mat_read.m in matlab and tdms_read.py (change path in code according to your machine) (if you don't have matlab give mat_read.m to chatgpt and ask it to make it a python code)
6. this will create acoustic csv data in acoustic/ and vibration csv data in vibration/ and csv_exports in current,temp
7. rename csv_exports to current_temp and put it in root dir (same dir as acoustic/ and vibration/)
9. python train_lstm.py