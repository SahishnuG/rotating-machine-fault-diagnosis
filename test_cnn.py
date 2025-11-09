import sys
import json
import torch
import numpy as np

# ---------------- Model Definition ----------------
import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, c_in, c_out, k=3, p=1, s=1):
        super().__init__()
        self.conv = nn.Conv2d(c_in, c_out, k, stride=s, padding=p)
        self.bn   = nn.BatchNorm2d(c_out)
        self.act  = nn.ReLU(inplace=True)
    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class MultiTaskCNN(nn.Module):
    def __init__(self, n_cond, n_sev):
        super().__init__()
        self.backbone = nn.Sequential(
            ConvBlock(1, 32, 5, 2),
            nn.MaxPool2d((2, 4)),
            ConvBlock(32, 64, 3, 1),
            nn.MaxPool2d((2, 2)),
            ConvBlock(64, 128, 3, 1),
            nn.AdaptiveAvgPool2d((4, 8)),
        )
        feat_dim = 128 * 4 * 8
        self.drop = nn.Dropout(0.2)
        self.fc_cond = nn.Linear(feat_dim, n_cond)
        self.fc_sev  = nn.Linear(feat_dim, n_sev)
    def forward(self, x):
        z = self.backbone(x)
        z = z.flatten(1)
        z = self.drop(z)
        return self.fc_cond(z), self.fc_sev(z)

# ---------------- Load Model ----------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model(model_path, labels_path):
    with open(labels_path, "r") as f:
        labels = json.load(f)
    conds = labels["conditions"]
    sevs  = labels["severities"]

    model = MultiTaskCNN(len(conds), len(sevs)).to(device)
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model, conds, sevs

# Load both models
acoustic_model, acoustic_conds, acoustic_sevs = load_model(
    "models/acoustic_multitask.pt", "models/acoustic_labels.json"
)
vib_model, vib_conds, vib_sevs = load_model(
    "models/vibration_multitask.pt", "models/vibration_labels.json"
)

# ---------------- Load scalogram ----------------
if len(sys.argv) < 2:
    print("Usage: python test_models.py <scalogram.npy>")
    sys.exit(1)

scalo_path = sys.argv[1]
X = np.load(scalo_path).astype(np.float32)  # (F,T)

# normalize like training
X = np.log1p(X)
X = (X - X.mean()) / (X.std() + 1e-6)
X = np.expand_dims(X, axis=0)  # (1,F,T)
X = np.expand_dims(X, axis=0)  # (1,1,F,T)
X = torch.from_numpy(X).to(device)

# ---------------- Predict acoustic model ----------------
with torch.no_grad():
    pc_a, ps_a = acoustic_model(X)
    pc_a = pc_a.softmax(dim=1).cpu().numpy()[0]
    ps_a = ps_a.softmax(dim=1).cpu().numpy()[0]

print("\n=== Acoustic Model Prediction ===")
print("Top Condition:", acoustic_conds[pc_a.argmax()], f"({pc_a.max():.2%})")
print("Top Severity :", acoustic_sevs[ps_a.argmax()],  f"({ps_a.max():.2%})")

# ---------------- Predict vibration model ----------------
with torch.no_grad():
    pc_v, ps_v = vib_model(X)
    pc_v = pc_v.softmax(dim=1).cpu().numpy()[0]
    ps_v = ps_v.softmax(dim=1).cpu().numpy()[0]

print("\n=== Vibration Model Prediction ===")
print("Top Condition:", vib_conds[pc_v.argmax()], f"({pc_v.max():.2%})")
print("Top Severity :", vib_sevs[ps_v.argmax()],  f"({ps_v.max():.2%})")
