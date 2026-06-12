"""
===========================================================
SYMPTOM RISK MODEL TRAINING  (SymCAT Only)
Health Buddy – Generative AI Project

Model:
    Input  = 1300-dim symptom vector
    Output = 400-dim one-hot disease/risk vector

Outputs per run:
    stored_models/SymptomRisk/run_<timestamp>/
        risk_model.pt
        training_log.csv
        metrics.json
        loss_curve.png
        training_report.pdf
===========================================================
"""

import os, json, datetime
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from fpdf import FPDF

# =========================================================
# 0) DEVICE
# =========================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("🔥 Using device:", device)

# =========================================================
# 1) RUN FOLDER
# =========================================================
timestamp = datetime.datetime.now().strftime("%Y_%m_%d_%H-%M-%S")
RUN_PATH = f"stored_models/SymptomRisk/run_{timestamp}"
os.makedirs(RUN_PATH, exist_ok=True)
print("📁 Saving outputs to:", RUN_PATH)

# =========================================================
# 2) LOAD SYMCAT  Input1300 symptoms   Output400 disease classes
# =========================================================
def load_symcat(path, NUM_SYM=1300, NUM_DIS=400):    
    df = pd.read_pickle(path)
    print("🔍", os.path.basename(path), "→ rows =", len(df))

    X, Y = [], []
    for _, row in df.iterrows():
        vec = np.zeros(NUM_SYM, dtype=np.float32)
        for s in row["implicit_symptoms"].get(True, []):
            vec[s] = 1
        for s in row["explicit_symptoms"].get(True, []):
            vec[s] = 1
        X.append(vec)

        yy = np.zeros(NUM_DIS)
        yy[row["disease_tag"]] = 1
        Y.append(yy)

    return np.array(X), np.array(Y)

X_train, y_train = load_symcat("datasets/symcat_400/symcat_400_train_df.pkl")
X_val, y_val     = load_symcat("datasets/symcat_400/symcat_400_val_df.pkl")
X_test, y_test   = load_symcat("datasets/symcat_400/symcat_400_test_df.pkl")

in_dim  = X_train.shape[1]
out_dim = y_train.shape[1]

# =========================================================
# 3) DATALOADER
# =========================================================
train_loader = DataLoader(
    TensorDataset(torch.tensor(X_train).float(), torch.tensor(y_train).float()),
    batch_size=32,
    shuffle=True
)

# =========================================================
# 4) MODEL
# =========================================================
class SymptomRiskNet(nn.Module):
    def __init__(self, i, o):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(i, 640), nn.ReLU(),
            nn.Linear(640, 512), nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, o), nn.Sigmoid()
        )
    def forward(self, x):
        return self.net(x)

model = SymptomRiskNet(in_dim, out_dim).to(device)
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# =========================================================
# 5) TRAINING LOOP
# =========================================================
EPOCHS = 15
train_losses, val_losses = [], []

print("🚀 Training...\n")
for epoch in range(EPOCHS):
    model.train()
    total = 0

    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        pred = model(xb)
        loss = criterion(pred, yb)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total += loss.item()

    train_losses.append(total)

    # validation
    model.eval()
    with torch.no_grad():
        vpred = model(torch.tensor(X_val).float().to(device))
        vloss = criterion(vpred, torch.tensor(y_val).float().to(device)).item()
    val_losses.append(vloss)

    print(f"Epoch {epoch+1}/{EPOCHS} | Train={total:.4f} | Val={vloss:.4f}")

# =========================================================
# 6) TEST EVALUATION (FIXED!)
# =========================================================
model.eval()
with torch.no_grad():
    tpred = model(torch.tensor(X_test).float().to(device))
    test_loss = criterion(tpred, torch.tensor(y_test).float().to(device)).item()

    pred_lbl = torch.argmax(tpred, 1).cpu().numpy()

    # FIXED ERROR → convert numpy → tensor first
    true_lbl = torch.argmax(torch.tensor(y_test), dim=1).numpy()

    acc = (pred_lbl == true_lbl).mean()

# =========================================================
# 7) SAVE ARTIFACTS
# =========================================================
torch.save(model.state_dict(), f"{RUN_PATH}/risk_model.pt")

pd.DataFrame({
    "epoch": range(1, EPOCHS+1),
    "train_loss": train_losses,
    "val_loss": val_losses
}).to_csv(f"{RUN_PATH}/training_log.csv", index=False)

with open(f"{RUN_PATH}/metrics.json", "w") as f:
    json.dump({
        "timestamp": timestamp,
        "device": str(device),
        "input_dim": in_dim,
        "output_dim": out_dim,
        "epochs": EPOCHS,
        "final_train_loss": train_losses[-1],
        "final_val_loss": val_losses[-1],
        "test_loss": float(test_loss),
        "test_top1_acc": float(acc)
    }, f, indent=4)

# =========================================================
# 8) PLOT
# =========================================================
plt.figure(figsize=(8,5))
plt.plot(train_losses, label="Train")
plt.plot(val_losses, label="Val")
plt.legend()
plt.tight_layout()
plt.savefig(f"{RUN_PATH}/loss_curve.png")

# =========================================================
# 9) PDF REPORT
# =========================================================
pdf = FPDF()
pdf.add_page()
try:
    pdf.add_font("DejaVu", "", "FONTS/DejaVuSans.ttf", uni=True)
    pdf.set_font("DejaVu", size=12)
except Exception:
    pdf.set_font("Arial", size=12)

pdf.multi_cell(0, 8, f"""
HealthBuddy – Symptom Risk Model Report

Run: {timestamp}
Device: {device}

Input Dim: {in_dim}
Output Dim: {out_dim}

Final Train Loss: {train_losses[-1]:.4f}
Final Val Loss: {val_losses[-1]:.4f}
Test BCE Loss: {test_loss:.4f}
Test Top-1 Accuracy: {acc*100:.2f}%
""")

pdf.image(f"{RUN_PATH}/loss_curve.png", w=180)
pdf.output(f"{RUN_PATH}/training_report.pdf")

print("\n📄 Report saved!")
print("🎉 All outputs saved to:", RUN_PATH)
