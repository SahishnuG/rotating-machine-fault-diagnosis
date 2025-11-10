# How to run
### Note: the models are already trained and stored in models/ so you can directly run main.py
1. Download [Vibration, Acoustic, Temperature, and Motor Current Dataset of Rotating Machine Under Varying Load Conditions for Fault Diagnosis](https://data.mendeley.com/datasets/ztmf3m7h5x/6) and extract it in your local machine
2. Extract acoustic, current,temp and vibration
3. clone this repo and add the extracted folders in the repo (same dir as main.py)
4. create venv and install requirements
5. run mat_read.m in matlab (if you don't have matlab give mat_read.m to chatgpt and ask it to make it a python code)
6. run tdms_read.py and tdms_read1.py
7. this adds csv files to acoustic/ and vibration/ and creates a folder called current_temp/ with converted tdms to csv
8. python preprocess.py (this will take a long time, applies cwt on the acoustic and vibration data and saves them as morlet scalograms (.png and .npy) to train the models on)
8. Optional - run compare_models.ipynb and analyse_samplingrate.py to better understand the data and the methods we used
9. python train_cnn.py --cuda --amp --batch-size 16 --rounds 5 --clients-per-round 5 --fl-col-stride 16 --fl-time-crop 512 --fl-freq-end 128 --full-col-stride 8 --full-tile-len 1024 --full-tile-overlap 256 --full-ft-epochs 2  
(I have given some default args to prevent memory overflow while still training on most of the data)
10. Optional - python test_cnn.py
11. Optional (or if the flask app isnt working fsr) - python main.py to run the gradio app
12. python app.py to run the final flask app
13. Input acoustic and vibration sensor csv data in the app to predict condition (error or normal) and severity (how severe the error is, if any)

### System architecture:
```mermaid
flowchart TD

%% -------- Dataset --------
subgraph DS[Dataset]
  A1[Acoustic MAT files]
  V1[Vibration MAT files]
end

%% -------- Convert to CSV --------
subgraph CV[Conversion to CSV]
  M1[MAT to CSV using mat_read m]
  A2[Acoustic CSV]
  V2[Vibration CSV]
end

A1 --> M1
V1 --> M1
M1 --> A2
M1 --> V2

%% -------- Preprocessing --------
subgraph PP[Preprocessing to scalograms]
  P0[preprocess py]
  P1A[Acoustic CWT]
  P1V[Vibration CWT 4 channels]
  SGA[Acoustic scalograms NPY]
  SGV[Vibration scalograms NPY]
end

A2 --> P0
V2 --> P0
P0 --> P1A --> SGA
P0 --> P1V --> SGV

%% -------- Training --------
subgraph TR[Training]
  subgraph FL[Federated learning rounds]
    T0[train cnn py]
    T1[Clients grouped by source csv]
    T2A[Acoustic loader 1 channel]
    T2V[Vibration loader 4 channels]
    T3[Multitask CNN condition and severity]
    T4[FedAvg aggregation]
    TVAL[Validation holdout]
  end
  subgraph FT[Full coverage fine tune]
    F1[Deterministic tiling of time]
    F2[Stride 1 tile 1024 overlap 256]
    F3[Lower learning rate few epochs]
  end
  WTS[Saved model weights]
  LAB[Saved label json]
end

SGA --> T0
SGV --> T0
T0 --> T1
T1 --> T2A --> T3
T1 --> T2V --> T3
T3 --> T4
TVAL --- T3
T4 --> F1 --> F2 --> F3 --> WTS
T0 --> LAB

%% -------- Baselines optional --------
subgraph BL[Baselines and analysis]
  B0[Direct npy trainer]
  B1[PCA classical models]
  B2[CNN and GRU baselines]
  B3[Metrics and plots]
end

SGA --> B0
SGV --> B0
B0 --> B1
B0 --> B2
B1 --> B3
B2 --> B3

%% -------- Inference App --------
subgraph APP[Gradio inference app]
  G0[main py]
  G1[User uploads CSV]
  G2[On the fly CWT]
  G3A[Acoustic CNN]
  G3V[Vibration CNN]
  G4[Per modality predictions]
  G5[Ensemble across modalities]
  G6[Scalogram previews and tables]
end

WTS --> G0
LAB --> G0
G1 --> G2
G2 --> G3A --> G4
G2 --> G3V --> G4
G4 --> G5 --> G6

```

### Model architecture:
```mermaid
flowchart TB;
 subgraph Backbone["Backbone"]
    direction LR
        CB1["ConvBlock #1<br>Conv2d(in_ch-&gt;32, k=5, s=1, p=2)<br>BatchNorm2d(32) + ReLU"]
        MP1["MaxPool2d(2,4)"]
        CB2["ConvBlock #2<br>Conv2d(32-&gt;64, k=3, s=1, p=1)<br>BatchNorm2d(64) + ReLU"]
        MP2["MaxPool2d(2,2)"]
        CB3["ConvBlock #3<br>Conv2d(64-&gt;128, k=3, s=1, p=1)<br>BatchNorm2d(128) + ReLU"]
        GAP["AdaptiveAvgPool2d(4,8)"]
  end
    I["Input<br>[B, in_ch, F, T]"] --> CB1
    CB1 --> MP1
    MP1 --> CB2
    CB2 --> MP2
    MP2 --> CB3
    CB3 --> GAP
    GAP --> FLAT["Flatten -> 4096"]
    FLAT --> DROP["Dropout p=0.2"]
    DROP --> Hc["Head: Condition<br>Linear(4096-&gt;n_cond)"] & Hs["Head: Severity<br>Linear(4096-&gt;n_sev)"]
    Hc -.-> Lc["CrossEntropyLoss (label_smoothing=0.1)"]
    Hs -.-> Ls["CrossEntropyLoss (label_smoothing=0.1)"]
    I --- Note1["in_ch = 1 (acoustic) or 4 (vibration)"]
```

**See project_understanding.docx to better understand the project**