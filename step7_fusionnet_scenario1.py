# -*- coding: utf-8 -*-
"""
STEP 7 — Early Fusion + SE-Gating (FusionNet), SCENARIO 1
Combines scaled radiomic features (Step 5) with CNN embeddings (Step 6)
into a single fused vector, then trains a Squeeze-and-Excitation gating
network (FusionNet) to produce a per-case PDAC probability.

Architecture — FusionNet
------------------------
Input: concatenation of radiomic (R) and CNN embedding (C) features
       X_fused = [X_rad | X_cnn]  shape: (N, R+C)

  1. SE-Gating block
       Linear(R+C → squeeze_dim)   squeeze_dim ≈ 40% of (R+C)
       LayerNorm → ReLU
       Linear(squeeze_dim → R+C)
       Sigmoid
       gate_weights ∈ (0,1)^(R+C)      ← interpretability output
       x_gated = x * gate_weights       ← elementwise multiplication

  2. Classifier head
       Linear(R+C → 128)
       LayerNorm → ReLU → Dropout(0.3)
       Linear(128 → 1)                  ← logit

Training
--------
  Optimizer : AdamW  lr=1e-4  weight_decay=1e-4
  Loss      : BCEWithLogitsLoss  pos_weight = n_neg/n_pos
  Scheduler : LinearLR  1e-4 → 1e-5 over N_EPOCHS
  Early stop: patience=10 on val AUC

Interpretability
----------------
SE gate weights are extracted for every test case.
Mean gate weight per feature is plotted as a bar chart:
  blue  = radiomic features  (columns 0 … R-1)
  red   = CNN embedding dims (columns R … R+C-1)
Features with mean gate > 0.5 are amplified by the model.
Features with mean gate < 0.5 are suppressed.
"""

# ── Imports ───────────────────────────────────────────────────────────────────
import os, warnings
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from sklearn.metrics import (
    roc_auc_score, average_precision_score, brier_score_loss,
    accuracy_score, confusion_matrix, f1_score, matthews_corrcoef,
    roc_curve, precision_recall_curve,
)
from sklearn.calibration import calibration_curve

warnings.filterwarnings("ignore")

from google.colab import drive
drive.mount('/content/drive', force_remount=True)

# ── CONFIG ────────────────────────────────────────────────────────────────────
STEP2_DIR   = "/content/drive/MyDrive/classif_model_AYTO/steps_scenarioA/step2_scenarioA"
STEP5_DIR   = "/content/drive/MyDrive/classif_model_AYTO/steps_scenarioA/step5_scenarioA"
STEP5B_DIR  = "/content/drive/MyDrive/classif_model_AYTO/steps_scenarioA/step5_B_scenarioA" # Radiomic branch -- for ablation table
STEP6_DIR   = '/content/drive/MyDrive/classif_model_AYTO/steps_scenarioA/step6_scenarioA_results' # CNN branch
STEP6B_DIR  = "/content/drive/MyDrive/classif_model_AYTO/steps_scenarioA/step6_B_scenarioA" # CNN branch evaluation -- for ablation table
STEP7B_DIR  = "/content/drive/MyDrive/classif_model_AYTO/steps_scenarioA/step7b_Late_Fusion"   # for ablation table

OUTPUT_DIR  = "/content/drive/MyDrive/classif_model_AYTO/steps_scenarioA/step7_EarlyFusion"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Training hyperparameters
N_EPOCHS     = 100
LR           = 1e-4
WEIGHT_DECAY = 1e-4
BATCH_SIZE   = 32
PATIENCE     = 10
RANDOM_STATE = 42
DPI          = 150

torch.manual_seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device : {device}")

# Load data
X_rad_train = np.load(os.path.join(STEP5_DIR, "X_train_scaled_scenarioA.npy"))
X_rad_val   = np.load(os.path.join(STEP5_DIR, "X_val_scaled_scenarioA.npy"))
X_rad_test  = np.load(os.path.join(STEP5_DIR, "X_test_scaled_scenarioA.npy"))

X_cnn_train = np.load(os.path.join(STEP6_DIR, "deep_embeddings_train.npy"))
X_cnn_val   = np.load(os.path.join(STEP6_DIR, "deep_embeddings_val.npy"))
X_cnn_test  = np.load(os.path.join(STEP6_DIR, "deep_embeddings_test.npy"))

y_train = np.load(os.path.join(STEP2_DIR, "y_train_scenarioA.npy")).astype(np.float32)
y_val   = np.load(os.path.join(STEP2_DIR, "y_val_scenarioA.npy")).astype(np.float32)
y_test  = np.load(os.path.join(STEP2_DIR, "y_test_scenarioA.npy")).astype(np.float32)

feat_names_rad = np.load(
    os.path.join(STEP5_DIR, "selected_feature_names_scenarioA.npy"), allow_pickle=True
).tolist()

# Add immediately after loading X_cnn_* arrays, before concatenation:
from sklearn.preprocessing import StandardScaler

cnn_scaler = StandardScaler()
X_cnn_train = cnn_scaler.fit_transform(X_cnn_train)   # fit on train only
X_cnn_val   = cnn_scaler.transform(X_cnn_val)
X_cnn_test  = cnn_scaler.transform(X_cnn_test)

import joblib
joblib.dump(cnn_scaler, os.path.join(OUTPUT_DIR, "cnn_scaler.pkl"))

# ── Alignment assertions ──────────────────────────────────────────────────────
# Load patient IDs to verify row alignment across all three sources
ids_train = np.load(os.path.join(STEP2_DIR, "ids_train_scenarioA.npy"), allow_pickle=True)
ids_val   = np.load(os.path.join(STEP2_DIR, "ids_val_scenarioA.npy"),   allow_pickle=True)
ids_test  = np.load(os.path.join(STEP2_DIR, "ids_test_scenarioA.npy"),  allow_pickle=True)

for split, Xr, Xc, y, ids in [
    ("train", X_rad_train, X_cnn_train, y_train, ids_train),
    ("val",   X_rad_val,   X_cnn_val,   y_val,   ids_val),
    ("test",  X_rad_test,  X_cnn_test,  y_test,  ids_test),
]:
    assert Xr.shape[0] == Xc.shape[0] == y.shape[0] == len(ids), \
        f"{split}: row count mismatch! rad={Xr.shape[0]}, cnn={Xc.shape[0]}, " \
        f"y={y.shape[0]}, ids={len(ids)}"
    print(f"{split}: {Xr.shape[0]} cases aligned ✓")

"""# ── Concatenate radiomic + CNN features"""

# ── Concatenate radiomic + CNN features ──────────────────────────────────────
X_train = np.concatenate([X_rad_train, X_cnn_train], axis=1).astype(np.float32)
X_val   = np.concatenate([X_rad_val,   X_cnn_val],   axis=1).astype(np.float32)
X_test  = np.concatenate([X_rad_test,  X_cnn_test],  axis=1).astype(np.float32)

n_rad    = X_rad_train.shape[1] # 14
n_cnn    = X_cnn_train.shape[1] # e.g. 128
n_fused  = X_train.shape[1] # 142

feat_names_fused = (
    feat_names_rad +
    [f"emb_{i}" for i in range(n_cnn)]
)

n_pos = int(y_train.sum())
n_neg = int((y_train == 0).sum())

print(f"Fused input dim : {n_fused}  ({n_rad} radiomic + {n_cnn} CNN)")
print(f"Train : {X_train.shape}   PDAC={n_pos}  non-PDAC={n_neg}")
print(f"Val   : {X_val.shape}")
print(f"Test  : {X_test.shape}")

"""# FusionNet"""

class FusionNet(nn.Module):
    """
    SE-gating fusion network.

    forward(x, return_gate=False)
      return_gate=False  →  (batch,)        logit
      return_gate=True   →  (batch, input_dim)  gate weights in (0,1)
    """
    def __init__(self, input_dim: int, squeeze_ratio: float = 0.4):
        super().__init__()
        squeeze_dim = max(8, int(input_dim * squeeze_ratio))

        # SE-gating block
        self.se = nn.Sequential(
            nn.Linear(input_dim, squeeze_dim),
            nn.LayerNorm(squeeze_dim),
            nn.ReLU(inplace=True),
            nn.Linear(squeeze_dim, input_dim),
            nn.Sigmoid(),
        )

        # Classifier head
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
        )

    def forward(self, x: torch.Tensor,
                return_gate: bool = False) -> torch.Tensor:
        gate    = self.se(x)          # (batch, input_dim) — values in (0,1)
        x_gated = x * gate            # elementwise multiply
        if return_gate:
            return gate
        logit = self.classifier(x_gated)
        return logit.squeeze(1)       # (batch,)

"""# DataLoaders"""

def make_loader(X, y, shuffle=False):
    ds = TensorDataset(
        torch.tensor(X, dtype=torch.float32),
        torch.tensor(y, dtype=torch.float32),
    )
    return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=shuffle)

train_loader = make_loader(X_train, y_train, shuffle=True)
val_loader   = make_loader(X_val,   y_val)
test_loader  = make_loader(X_test,  y_test)

"""# Model, optimizer, loss, scheduler"""

model     = FusionNet(input_dim=n_fused).to(device)
optimizer = torch.optim.AdamW(model.parameters(),
                               lr=LR, weight_decay=WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.LinearLR(
    optimizer, start_factor=1.0, end_factor=0.1, total_iters=N_EPOCHS
)
pos_weight = torch.tensor([n_neg / n_pos], dtype=torch.float32).to(device)
criterion  = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"FusionNet parameters : {n_params:,}")
print(f"Positive weight      : {pos_weight.item():.3f}")

"""# Training loop"""

# @title
ckpt_path    = os.path.join(OUTPUT_DIR, "best_fusion_model.pt")
best_val_auc = 0.0
no_improve   = 0
history      = {"train_loss": [], "val_auc": []}

print("── Training " + "─" * 52)
for epoch in range(N_EPOCHS):

    # Train
    model.train()
    epoch_loss = 0.0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        loss = criterion(model(X_batch), y_batch)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    scheduler.step()
    avg_loss = epoch_loss / len(train_loader)

    # Validate
    model.eval()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            probs = torch.sigmoid(model(X_batch.to(device))).cpu().numpy()
            all_probs.extend(probs.tolist())
            all_labels.extend(y_batch.numpy().tolist())

    val_auc = roc_auc_score(all_labels, all_probs)
    history["train_loss"].append(avg_loss)
    history["val_auc"].append(val_auc)

    flag = ""
    if val_auc > best_val_auc + 1e-4:
        best_val_auc = val_auc
        torch.save(model.state_dict(), ckpt_path)
        no_improve   = 0
        flag         = "  ✓ saved"
    else:
        no_improve  += 1

    print(f"Epoch {epoch+1:03d}/{N_EPOCHS} | "
          f"Loss: {avg_loss:.4f} | Val AUC: {val_auc:.4f}{flag}")

    if no_improve >= PATIENCE:
        print(f"\nEarly stopping at epoch {epoch+1}  "
              f"(best val AUC: {best_val_auc:.4f})")
        break

# Training curves
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(history["train_loss"], color="steelblue")
axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("BCE Loss")
axes[0].set_title("FusionNet — Training Loss"); axes[0].grid(alpha=0.3)
axes[1].plot(history["val_auc"], color="tomato")
axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("ROC-AUC")
axes[1].set_title("FusionNet — Validation AUC"); axes[1].grid(alpha=0.3)
plt.tight_layout()
plt.show()

plt.savefig(os.path.join(OUTPUT_DIR, "fusion_training_curves.png"), dpi=DPI)
plt.close()
print("Training curves saved.")

"""# Evaluation helpers"""

def compute_metrics(y_true, y_prob, threshold=0.5):
    y_pred         = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0,1]).ravel()
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    return {
        "ROC-AUC"    : round(roc_auc_score(y_true, y_prob), 4),
        "Avg-Prec"   : round(average_precision_score(y_true, y_prob), 4),
        "Brier"      : round(brier_score_loss(y_true, y_prob), 4),
        "Accuracy"   : round(accuracy_score(y_true, y_pred), 4),
        "Sensitivity": round(sens, 4),
        "Specificity": round(spec, 4),
        "F1"         : round(f1_score(y_true, y_pred, zero_division=0), 4),
        "MCC"        : round(matthews_corrcoef(y_true, y_pred), 4),
        "TP": tp, "TN": tn, "FP": fp, "FN": fn,
        "threshold"  : round(threshold, 4),
    }

def best_threshold_youden(y_true, y_prob):
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    return float(thresholds[np.argmax(tpr - fpr)])

def get_probs(loader, mdl):
    mdl.eval()
    probs, labels = [], []
    with torch.no_grad():
        for X_batch, y_batch in loader:
            p = torch.sigmoid(mdl(X_batch.to(device))).cpu().numpy()
            probs.extend(p.tolist())
            labels.extend(y_batch.numpy().tolist())
    return np.array(probs), np.array(labels)

"""# Load best checkpoint and compute metrics"""

model.load_state_dict(torch.load(ckpt_path, map_location=device))

P_val,  y_val_np  = get_probs(val_loader,  model)
P_test, y_test_np = get_probs(test_loader, model)

thr       = best_threshold_youden(y_val_np, P_val)
m_val     = compute_metrics(y_val_np,  P_val,  thr)
m_test    = compute_metrics(y_test_np, P_test, thr)

METRIC_DISPLAY = ["ROC-AUC","Avg-Prec","Brier","Accuracy",
                  "Sensitivity","Specificity","F1","MCC"]

print(f"{'='*65}")
print("FUSIONNET — VALIDATION METRICS")
print(f"{'='*65}")
for k in METRIC_DISPLAY:
    print(f"  {k:<15}: {m_val[k]}")

print(f"\n{'='*65}")
print("FUSIONNET — TEST METRICS")
print(f"{'='*65}")
for k in METRIC_DISPLAY:
    print(f"  {k:<15}: {m_test[k]}")

pd.DataFrame([{"Split":"val",  **m_val}]).to_csv(
    os.path.join(OUTPUT_DIR, "results_val.csv"),  index=False)
pd.DataFrame([{"Split":"test", **m_test}]).to_csv(
    os.path.join(OUTPUT_DIR, "results_test.csv"), index=False)

"""# Plots — ROC, PR, Confusion matrix, Calibration"""

# ROC
fpr, tpr, _ = roc_curve(y_test_np, P_test)
fig, ax = plt.subplots(figsize=(6, 5))
ax.plot(fpr, tpr, color="#E91E63", lw=2,
        label=f"FusionNet  AUC={m_test['ROC-AUC']:.4f}")
ax.plot([0,1],[0,1],"k--",lw=1)
ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curve — FusionNet (Test)", fontweight="bold")
ax.legend(); ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "roc_curve.png"), dpi=DPI)
plt.close()

# PR
prec, rec, _ = precision_recall_curve(y_test_np, P_test)
ap = average_precision_score(y_test_np, P_test)
fig, ax = plt.subplots(figsize=(6, 5))
ax.plot(rec, prec, color="#E91E63", lw=2, label=f"FusionNet  AP={ap:.4f}")
ax.axhline(y_test_np.mean(), color="gray", linestyle="--", lw=1,
           label=f"Baseline (prevalence={y_test_np.mean():.2f})")
ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
ax.set_title("Precision-Recall — FusionNet (Test)", fontweight="bold")
ax.legend(); ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "pr_curve.png"), dpi=DPI)
plt.close()

# Confusion matrix
cm = confusion_matrix(y_test_np, (P_test >= thr).astype(int), labels=[0,1])
fig, ax = plt.subplots(figsize=(5, 4))
im = ax.imshow(cm, cmap="Blues")
for i in range(2):
    for j in range(2):
        ax.text(j, i, str(cm[i,j]), ha="center", va="center",
                fontsize=16, fontweight="bold",
                color="white" if cm[i,j] > cm.max()/2 else "black")
ax.set_xticks([0,1]); ax.set_xticklabels(["non-PDAC","PDAC"])
ax.set_yticks([0,1]); ax.set_yticklabels(["non-PDAC","PDAC"])
ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
ax.set_title(f"Confusion Matrix — FusionNet (Test)\n"
             f"threshold={thr:.3f}", fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "confusion_matrix.png"), dpi=DPI)
plt.close()

# Calibration
frac_pos, mean_pred = calibration_curve(y_test_np, P_test, n_bins=8, strategy="uniform")
fig, ax = plt.subplots(figsize=(6, 5))
ax.plot([0,1],[0,1],"k--",lw=1,label="Perfect calibration")
ax.plot(mean_pred, frac_pos, "o-", color="#E91E63", lw=2,
        label=f"FusionNet  Brier={m_test['Brier']:.4f}")
ax.set_xlabel("Mean predicted probability")
ax.set_ylabel("Fraction of positives (PDAC)")
ax.set_title("Calibration — FusionNet (Test)", fontweight="bold")
ax.legend(); ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "calibration.png"), dpi=DPI)
plt.close()

print("Saved: roc_curve.png  pr_curve.png  confusion_matrix.png  calibration.png")

"""# SE gate weight interpretability"""

# ── After extracting gate weights ─────────────────────────────────────────────
# Assuming gate_weights has shape (n_test_cases, n_features)
# and feature_names is a list of n_features strings
model.eval()
all_gates = []
with torch.no_grad():
    for X_batch, _ in test_loader:
        gates = model(X_batch.to(device), return_gate=True).cpu().numpy()
        all_gates.append(gates)

gate_matrix = np.concatenate(all_gates, axis=0)   # (N_test, n_fused)
np.save(os.path.join(OUTPUT_DIR, "test_gate_weights.npy"), gate_matrix)

mean_gate = gate_matrix.mean(axis=0)              # (n_fused,)
# Combine names so everything aligns (giving default names to CNN features)
cnn_names = [f"CNN_feat_{i}" for i in range(n_cnn)]
all_feature_names = np.array(feat_names_rad + cnn_names)

amplified  = [(name, w) for name, w in zip(all_feature_names, mean_gate) if w >= 0.5]
suppressed = [(name, w) for name, w in zip(all_feature_names, mean_gate) if w <  0.5]

amplified  = sorted(amplified,  key=lambda x: x[1], reverse=True)
suppressed = sorted(suppressed, key=lambda x: x[1], reverse=False)

print(f"\n── SE Gate Weight Summary ───────────────────────────────────────")
print(f"Total features     : {len(mean_gate)}")
print(f"Amplified (≥ 0.5)  : {len(amplified)}")
print(f"Suppressed (< 0.5) : {len(suppressed)}")

print(f"\nTop amplified features:")
for name, w in amplified:
    print(f"  {name:<35s}  gate = {w:.4f}")

print(f"\nMost suppressed features (bottom 5):")
for name, w in suppressed[-5:]:
    print(f"  {name:<35s}  gate = {w:.4f}")

print(f"Amplified features:")
for name, w in amplified:
    print(f"('{name}', {w:.4f})")

print(f"suppressed features:")
for name, w in suppressed:
    print(f"('{name}', {w:.4f})")

# Commented out IPython magic to ensure Python compatibility.
# %matplotlib inline
import os
import numpy as np
import torch
import matplotlib
#matplotlib.use('QtAgg')  # Or use 'QtAgg' if you have PyQt installed
import matplotlib.pyplot as plt

# 1. Parse your exact summary statistics and top feature text output
total_features = len(mean_gate)
amplified_count = len(amplified)
suppressed_count = len(suppressed)

# The top 15 highest-ranking features extracted directly from your list
top_features = [
    ('CNN_feat_0', 0.6715),
    ('haralick_10_std', 0.5848),
('haralick_6_std', 0.5708),
('haralick_13_q25', 0.5416),
('haralick_15_std', 0.5261),
('haralick_11_q25', 0.2741),
('haralick_5_std', 0.3612),
('haralick_12_q75', 0.3652),
('haralick_12_mean', 0.3653),
('haralick_7_q25', 0.3827),
('haralick_6_q25', 0.3896),
('haralick_3_q75', 0.3941),
('haralick_11_q75', 0.4005),
('haralick_13_mean', 0.4643),
('haralick_7_mean', 0.4643),
        ]

# Unpack data
names = [x[0] for x in top_features]
weights = [x[1] for x in top_features]

# Map your specific color palette dynamically: Haralick = Blue, CNN = Red
colors = []
for name in names:
    if "haralick" in name or "radiomic" in name:
        colors.append("#2196F3")  # Blue
    else:
        colors.append("#E53935")  # Red

# Reverse everything for horizontal plotting (largest weight at the top of the chart)
names = names[::-1]
weights = weights[::-1]
colors = colors[::-1]

# 2. Setup Figure Layout: Two side-by-side subplots tailored for thesis margin widths
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 5))

# ─── LEFT PANEL: GLOBAL DISTRIBUTION PIE CHART ───
pie_labels = [f'Suppressed\n({suppressed_count} Feat.)', f'Amplified\n({amplified_count} Feat.)']
pie_colors = ['#B0BEC5', '#4CAF50']  # Neutral grey vs Active green marker

wedges, texts, autotexts = ax1.pie(
    [suppressed_count, amplified_count],
    labels=pie_labels,
    colors=pie_colors,
    autopct='%1.1f%%',
    startangle=140,
    textprops=dict(fontsize=11, fontweight='medium'),
    pctdistance=0.55,
    wedgeprops=dict(width=0.4, edgecolor='w', linewidth=2) # Creates an elegant donut style
)
# Make percent text inside the donut bold and clear
for autotext in autotexts:
    autotext.set_fontsize(11)
    autotext.set_weight('bold')

ax1.set_title(f"A. Total Feature Overview\n(N = {total_features} total features)", fontsize=13, fontweight='bold', pad=15)

# ─── RIGHT PANEL: TOP SELECTION HIGHLIGHTS ───
y_pos = np.arange(len(names))
bars = ax2.barh(y_pos, weights, color=colors, edgecolor='none', height=0.65)

# Style references & cutoffs
ax2.axvline(0.5, color='black', linestyle=':', lw=1.5) # Removed label parameter

# Annotation labels & scales
ax2.set_title("B. Top 15 Highest-Weighted Features\n Red: CNN embed. | Blue: Rad. Feature | Threshold (0.5)", fontsize=13, fontweight='bold', pad=15)
ax2.set_xlabel("Mean Gate Weight", fontsize=11, fontweight='medium')
ax2.set_yticks(y_pos)
ax2.set_yticklabels(names, fontsize=10, family='monospace')
ax2.set_xlim(0, 0.8)
ax2.grid(axis='x', alpha=0.3, linestyle='--')

# --- THE LEGEND CODE BLOCK HAS BEEN COMPLETELY REMOVED FROM HERE ---

# Custom clean legend construction for structural colors
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='#E53935', label='CNN Embedding'),
    Patch(facecolor='#2196F3', label='Radiomic Feature'),
    Patch(facecolor='none', edgecolor='black', linestyle=':', label='Threshold (0.5)')
]
#ax2.legend(handles=legend_elements, loc='lower right', fontsize=10, framealpha=0.9)

# Adjust layouts and export at print-standard DPI
plt.tight_layout()

# Force save high-quality output to directory for safe keeping
#output_name = "thesis_feature_importance.png"
#plt.savefig(output_name, dpi=300, bbox_inches="tight")
#print(f"Success! High resolution plot saved to your workspace as: '{output_name}'")

plt.show()

"""### Mean SE Gate Weights Visualization and Feature Analysis"""

print("── Extracting SE gate weights (test set) ────────────────────")

model.eval()
all_gates = []
with torch.no_grad():
    for X_batch, _ in test_loader:
        gates = model(X_batch.to(device), return_gate=True).cpu().numpy()
        all_gates.append(gates)

gate_matrix = np.concatenate(all_gates, axis=0)   # (N_test, n_fused)
np.save(os.path.join(OUTPUT_DIR, "test_gate_weights.npy"), gate_matrix)

mean_gates = gate_matrix.mean(axis=0)              # (n_fused,)

# Bar chart: radiomic = blue, CNN = red
colors = ["#2196F3"] * n_rad + ["#E53935"] * n_cnn
fig, ax = plt.subplots(figsize=(max(12, n_fused // 4), 5))
bars = ax.bar(range(n_fused), mean_gates, color=colors, edgecolor="none")
ax.axhline(0.5, color="black", linestyle="--", lw=1,
           label="Gate=0.5  (amplify above, suppress below)")
ax.set_xlabel("Feature index")
ax.set_ylabel("Mean gate weight")
ax.set_title("SE Gate Weights — FusionNet (Test Set)\n"
             "Blue = radiomic  |  Red = CNN embedding", fontweight="bold")
ax.set_ylim(0, 1.05)
ax.legend(fontsize=9); ax.grid(axis="y", alpha=0.3)

# Label the radiomic features by name (they are few enough)
ax.set_xticks(range(n_rad))
ax.set_xticklabels(feat_names_rad, rotation=45, ha="right", fontsize=7)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "gate_weights.png"),
            dpi=DPI, bbox_inches="tight")
plt.close()
print("Saved: gate_weights.png  test_gate_weights.npy")

# Print top amplified and suppressed features
gate_df = pd.DataFrame({
    "feature"   : feat_names_fused,
    "type"      : ["radiomic"]*n_rad + ["CNN"]*n_cnn,
    "mean_gate" : mean_gates,
}).sort_values("mean_gate", ascending=False)

print("\nTop 10 amplified features (gate > 0.5):")
print(gate_df[gate_df["mean_gate"] > 0.5].head(10).to_string(index=False))
print("\nTop 10 suppressed features (gate < 0.5):")
print(gate_df[gate_df["mean_gate"] < 0.5].tail(10).to_string(index=False))

print("Top 10 Suppressed Features (gate < 0.5) with Rank:")
display(gate_df[gate_df["mean_gate"] < 0.5].tail(10))

from IPython.display import Image
Image(filename=os.path.join(OUTPUT_DIR, "gate_weights.png"))

"""# Complete the thesis ablation table"""

print("\n── Completing ablation table ────────────────────────────────")

ablation_path = os.path.join(STEP7B_DIR, "ablation_table.csv")

if os.path.exists(ablation_path):
    ablation_df = pd.read_csv(ablation_path,
                               index_col="Branch / Strategy")

    ablation_df.loc["Early fusion + SE-gating  (Step 7)", "Val AUC"]  = \
        m_val["ROC-AUC"]
    ablation_df.loc["Early fusion + SE-gating  (Step 7)", "Test AUC"] = \
        m_test["ROC-AUC"]
    ablation_df.loc["Early fusion + SE-gating  (Step 7)",
                    "Test Sensitivity"]  = m_test["Sensitivity"]
    ablation_df.loc["Early fusion + SE-gating  (Step 7)",
                    "Test Specificity"]  = m_test["Specificity"]
    ablation_df.loc["Early fusion + SE-gating  (Step 7)",
                    "Test F1"]           = m_test["F1"]
    ablation_df.loc["Early fusion + SE-gating  (Step 7)",
                    "Test MCC"]          = m_test["MCC"]

    ablation_df.to_csv(ablation_path)
    ablation_df.to_csv(os.path.join(OUTPUT_DIR, "ablation_table.csv"))

    print(f"\n{'='*65}")
    print("THESIS ABLATION TABLE — COMPLETE")
    print(f"{'='*65}")
    print(ablation_df.to_string())
else:
    print("  ablation_table.csv not found in step7b_outputs.")
    print("  Run step7b_late_fusion.py first, then re-run this script.")
    # Save a standalone result so nothing is lost
    pd.DataFrame([{
        "Branch / Strategy"  : "Early fusion + SE-gating (Step 7)",
        "Val AUC"            : m_val["ROC-AUC"],
        "Test AUC"           : m_test["ROC-AUC"],
        "Test Sensitivity"   : m_test["Sensitivity"],
        "Test Specificity"   : m_test["Specificity"],
        "Test F1"            : m_test["F1"],
        "Test MCC"           : m_test["MCC"],
    }]).to_csv(os.path.join(OUTPUT_DIR, "fusion_result_only.csv"), index=False)

"""# Final summary"""

print(f"{'='*65}")
print("STEP 7 COMPLETE")
print(f"{'='*65}")
print(f"  Val  AUC : {m_val['ROC-AUC']:.4f}")
print(f"  Test AUC : {m_test['ROC-AUC']:.4f}")
print(f"  Test Sensitivity : {m_test['Sensitivity']:.4f}")
print(f"  Test Specificity : {m_test['Specificity']:.4f}")
print(f"  Test F1          : {m_test['F1']:.4f}")
print(f"  Test MCC         : {m_test['MCC']:.4f}")
print(f"\nAll outputs saved to : {OUTPUT_DIR}")

np.save(os.path.join(OUTPUT_DIR, "P_test.npy"), P_test)
print(f"P_test saved to {os.path.join(OUTPUT_DIR, 'P_test.npy')}")
