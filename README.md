1. Download [Vibration, Acoustic, Temperature, and Motor Current Dataset of Rotating Machine Under Varying Load Conditions for Fault Diagnosis](https://data.mendeley.com/datasets/ztmf3m7h5x/6) and extract it in your local machine
2. Extract acoustic, current,temp and vibration
3. clone this repo and add the extracted folders in the repo (same dir as main.py)
4. create venv and install requirements
5. run mat_read.m in matlab (if you don't have matlab give mat_read.m to chatgpt and ask it to make it a python code)
6. run tdms_read.py and tdms_read1.py
7. this adds csv files to acoustic/ and vibration/ and creates a folder called current_temp/ with converted tdms to csv
8. python train_lstm.py
9. python train_cnn.py
10. python test_cnn.py
11. python main.py