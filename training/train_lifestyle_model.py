"""
===========================================================
LIFESTYLE RISK MODEL TRAINING  (UCI BRFSS)
Health Buddy – Generative AI Project

Classification:
   Diabetes_012 → {0 = No, 1 = Prediabetes, 2 = Diabetes}

Model:
   Input  = lifestyle features (BMI, Sleep, Smoking, etc.)
   Output = 3-class softmax (multi-class classification)
    Input: a binary symptom vector (size = 1300)
    Output: a 400-dimensional disease probability vector (multi-label)
Outputs saved into:
    stored_models/LifestyleRisk/run_<timestamp>/
===========================================================
"""

import os, json, datetime
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from fpdf import FPDF

# =========================================================
# 0) DEVICE
# =========================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

timestamp = datetime.datetime.now().strftime("%Y_%m_%d_%H-%M-%S")
RUN_PATH = f"stored_models/LifestyleRisk/run_{timestamp}"
os.makedirs(RUN_PATH, exist_ok=True)

print("🔥 Using device:", device)
print("📁 Saving outputs to:", RUN_PATH)

# =========================================================
# 1) LOAD DATASET
# =========================================================
csv = "datasets/Diabetes Health Indicators/diabetes_012_health_indicators_BRFSS2015.csv"
df = pd.read_csv(csv)

y = df["Diabetes_012"].values        # values = {0,1,2}
df = df.drop(columns=["Diabetes_012"])

# scale input features
scaler = StandardScaler()
X = scaler.fit_transform(df.values)

# split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.10, random_state=42
)

X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.10, random_state=42
)

train_loader = DataLoader(
    TensorDataset(
        torch.tensor(X_train).float(),
        torch.tensor(y_train).long()       # LONG for multi-class CE loss
    ),
    batch_size=32, shuffle=True
)

# =========================================================
# 2) MODEL
# =========================================================
class LifestyleNet(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_dim, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Dropout(0.2), # overfititng
            nn.Linear(64, 3)               # 3 classes
        )
    def forward(self, x): return self.layers(x)

model = LifestyleNet(X.shape[1]).to(device)

criterion = nn.CrossEntropyLoss()     # <-- Loss + Optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# =========================================================
# 3) TRAINING
# =========================================================
EPOCHS = 12
train_losses, val_losses = [], []

print("🚀 Training Lifestyle Model...\n")

for epoch in range(EPOCHS):
    model.train()
    total = 0

    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)

        pred = model(xb)                # shape: [B,3]
        loss = criterion(pred, yb)      # CE expects raw logits + long labels

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total += loss.item()

    train_losses.append(total)

    # validation on unseen data
    with torch.no_grad():
        vpred = model(torch.tensor(X_val).float().to(device))
        vloss = criterion(vpred, torch.tensor(y_val).long().to(device)).item()

    val_losses.append(vloss)

    print(f"Epoch {epoch+1} | Train={total:.4f} | Val={vloss:.4f}")

# =========================================================
# 4) TEST
# =========================================================
model.eval()
with torch.no_grad():
    logits = model(torch.tensor(X_test).float().to(device))
    test_loss = criterion(logits, torch.tensor(y_test).long().to(device)).item()

pred_labels = torch.argmax(logits, dim=1).cpu().numpy()
acc = (pred_labels == y_test).mean()

# =========================================================
# 5) SAVE EVERYTHING
# =========================================================
torch.save(model.state_dict(), f"{RUN_PATH}/lifestyle_risk_model.pt")

pd.DataFrame({
    "epoch": list(range(1, EPOCHS+1)),
    "train_loss": train_losses,
    "val_loss": val_losses
}).to_csv(f"{RUN_PATH}/training_log.csv", index=False)

with open(f"{RUN_PATH}/metrics.json", "w") as f:
    json.dump({
        "device": str(device),
        "in_features": X.shape[1],
        "epochs": EPOCHS,
        "final_train_loss": train_losses[-1],
        "final_val_loss": val_losses[-1],
        "test_loss": test_loss,
        "test_accuracy": float(acc)
    }, f, indent=4)

# =========================================================
# 6) LOSS CURVE
# =========================================================
plt.figure(figsize=(8,5))
plt.plot(train_losses, label="Train")
plt.plot(val_losses, label="Val")
plt.legend()
plt.tight_layout()
plt.savefig(f"{RUN_PATH}/loss_curve.png")

# =========================================================
# 7) PDF REPORT
# =========================================================
pdf = FPDF()
pdf.add_page()
try:
    pdf.add_font("DejaVu", "", "FONTS/DejaVuSans.ttf", uni=True)
    pdf.set_font("DejaVu", size=12)
except Exception:
    pdf.set_font("Arial", size=12)

pdf.multi_cell(0, 8, f"""
HealthBuddy – Lifestyle Risk Model Report

Run: {timestamp}
Device: {device}

Features: {X.shape[1]}
Classes: 3 (No / PreDiabetes / Diabetes)
Epochs: {EPOCHS}

Final Train Loss: {train_losses[-1]:.4f}
Final Validation Loss: {val_losses[-1]:.4f}

Test Loss: {test_loss:.4f}
Test Accuracy: {acc*100:.2f}% 
""")

pdf.image(f"{RUN_PATH}/loss_curve.png", w=180)
pdf.output(f"{RUN_PATH}/training_report.pdf")

print("\n🎉 Lifestyle Model Ready!")
