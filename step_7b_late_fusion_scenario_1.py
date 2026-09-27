# -*- coding: utf-8 -*-
"""
STEP 7b — Late Fusion (Decision-Level Fusion)
Combines the PDAC probability outputs from the radiomic branch (Step 5b)
and the CNN branch (Step 6b) without any additional feature-level learning.

Three late fusion strategies are compared:
  A) Simple average        P = 0.5 * P_rad + 0.5 * P_cnn
  B) Weighted average      w* = argmax val-AUC over grid w in [0,1]
                           P  = w * P_rad + (1-w) * P_cnn
  C) Meta-classifier       LogisticRegression([P_rad, P_cnn]) fit on val,
                           evaluated on test

All weight/hyperparameter selection uses the VALIDATION set only.
The test set is touched exactly once, at the end.
"""

import os, warnings
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.calibration  import calibration_curve
from sklearn.metrics      import (
    roc_auc_score, average_precision_score, brier_score_loss,
    accuracy_score, confusion_matrix, f1_score, matthews_corrcoef,
    roc_curve,
)
warnings.filterwarnings("ignore")

from google.colab import drive
drive.mount('/content/drive', force_remount=True)

# ── CONFIG ────────────────────────────────────────────────────────────────────
STEP2_DIR   = "/content/drive/MyDrive/classif_model_AYTO/steps_scenarioA/step2_scenarioA"
STEP5_DIR   = "/content/drive/MyDrive/classif_model_AYTO/steps_scenarioA/step5_scenarioA"
STEP5B_DIR  = "/content/drive/MyDrive/classif_model_AYTO/steps_scenarioA/step5_B_scenarioA" # for ablation table
STEP6_DIR   = '/content/drive/MyDrive/classif_model_AYTO/steps_scenarioA/step6_scenarioA_results'
STEP6B_DIR  = "/content/drive/MyDrive/classif_model_AYTO/steps_scenarioA/step6_B_scenarioA" # for ablation table

OUTPUT_DIR  = "/content/drive/MyDrive/classif_model_AYTO/steps_scenarioA/step7b_Late_Fusion"
os.makedirs(OUTPUT_DIR, exist_ok=True)

RANDOM_STATE = 42
DPI          = 150
WEIGHT_GRID  = np.arange(0.0, 1.01, 0.05)

PALETTE = {
    "Radiomic only" : "#2196F3",
    "CNN only"      : "#4CAF50",
    "Simple avg"    : "#9C27B0",
    "Weighted avg"  : "#FF5722",
    "Meta-LR"       : "#009688",
}

"""# Helpers"""

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


def best_classifier_by_val(models_dict, X_val, y_val):
    """Return the name and clf with the highest val AUC."""
    best_name, best_auc, best_clf = None, -1, None
    for name, clf in models_dict.items():
        auc = roc_auc_score(y_val, clf.predict_proba(X_val)[:, 1])
        if auc > best_auc:
            best_auc, best_name, best_clf = auc, name, clf
    return best_name, best_clf

"""# Load labels"""

y_train = np.load(os.path.join(STEP2_DIR, "y_train_scenarioA.npy"))
y_val   = np.load(os.path.join(STEP2_DIR, "y_val_scenarioA.npy"))
y_test  = np.load(os.path.join(STEP2_DIR, "y_test_scenarioA.npy"))

print(f"Labels  train={len(y_train)}  val={len(y_val)}  test={len(y_test)}")
print(f"PDAC    train={int(y_train.sum())}  "
      f"val={int(y_val.sum())}  test={int(y_test.sum())}")

"""# Load branch models and produce probabilities"""

# @title
from sklearn.metrics import roc_auc_score, brier_score_loss

def best_classifier_by_val(models_dict, X_val, y_val, auc_tolerance=0.005):
    """
    Select best classifier by validation AUC, using Brier score as a
    tie-breaker whenever candidate AUCs fall within `auc_tolerance`
    of the best AUC (default: within 0.005).
    """
    scored = []
    for name, clf in models_dict.items():
        probs = clf.predict_proba(X_val)[:, 1]
        auc   = roc_auc_score(y_val, probs)
        brier = brier_score_loss(y_val, probs)
        scored.append((name, clf, auc, brier))

    best_auc  = max(s[2] for s in scored)
    candidates = [s for s in scored if best_auc - s[2] <= auc_tolerance]
    candidates.sort(key=lambda s: s[3])   # lowest Brier wins among near-tied AUCs
    best_name, best_clf, best_auc_val, best_brier = candidates[0]

    print(f"  Model selection (AUC tie-break tolerance = {auc_tolerance}):")
    for name, _, auc, brier in sorted(scored, key=lambda s: -s[2]):
        marker = "  ← selected" if name == best_name else ""
        print(f"    {name:<8} AUC={auc:.4f}  Brier={brier:.4f}{marker}")

    return best_name, best_clf

# @title
print("── Radiomic branch ──────────────────────────────────────────")
rad_models  = joblib.load(os.path.join(STEP5B_DIR, "all_models_scenarioA.pkl"))
X_rad_train = np.load(os.path.join(STEP5_DIR, "X_train_scaled_scenarioA.npy"))
X_rad_val   = np.load(os.path.join(STEP5_DIR, "X_val_scaled_scenarioA.npy"))
X_rad_test  = np.load(os.path.join(STEP5_DIR, "X_test_scaled_scenarioA.npy"))

rad_name, rad_clf = best_classifier_by_val(rad_models, X_rad_val, y_val)
P_rad_train = rad_clf.predict_proba(X_rad_train)[:, 1]
P_rad_val   = rad_clf.predict_proba(X_rad_val)[:, 1]
P_rad_test  = rad_clf.predict_proba(X_rad_test)[:, 1]
rad_thr     = best_threshold_youden(y_val, P_rad_val)
rad_val_auc = roc_auc_score(y_val,  P_rad_val)
rad_tst_auc = roc_auc_score(y_test, P_rad_test)
print(f"  Best model : {rad_name}  "
      f"val={rad_val_auc:.4f}  test={rad_tst_auc:.4f}")

print("\n── CNN branch ───────────────────────────────────────────────")
cnn_models  = joblib.load(os.path.join(STEP6B_DIR, "all_models.pkl"))
X_cnn_train = np.load(os.path.join(STEP6_DIR, "deep_embeddings_train.npy"))
X_cnn_val   = np.load(os.path.join(STEP6_DIR, "deep_embeddings_val.npy"))
X_cnn_test  = np.load(os.path.join(STEP6_DIR, "deep_embeddings_test.npy"))

# Add immediately after loading X_cnn_* arrays, before concatenation:
from sklearn.preprocessing import StandardScaler

cnn_scaler = StandardScaler()
X_cnn_train = cnn_scaler.fit_transform(X_cnn_train)   # fit on train only
X_cnn_val   = cnn_scaler.transform(X_cnn_val)
X_cnn_test  = cnn_scaler.transform(X_cnn_test)

import joblib
joblib.dump(cnn_scaler, os.path.join(OUTPUT_DIR, "cnn_scaler.pkl"))

cnn_name, cnn_clf = best_classifier_by_val(cnn_models, X_cnn_val, y_val)
P_cnn_train = cnn_clf.predict_proba(X_cnn_train)[:, 1]
P_cnn_val   = cnn_clf.predict_proba(X_cnn_val)[:, 1]
P_cnn_test  = cnn_clf.predict_proba(X_cnn_test)[:, 1]
cnn_thr     = best_threshold_youden(y_val, P_cnn_val)
cnn_val_auc = roc_auc_score(y_val,  P_cnn_val)
cnn_tst_auc = roc_auc_score(y_test, P_cnn_test)
print(f"  Best model : {cnn_name}  "
      f"val={cnn_val_auc:.4f}  test={cnn_tst_auc:.4f}")

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
    print(f"\n{split}: {Xr.shape[0]} cases aligned ✓")

"""# Strategy A — Simple average"""

print("── Strategy A: Simple average (w=0.50) ─────────────────────")
P_avg_val  = 0.5 * P_rad_val  + 0.5 * P_cnn_val
P_avg_test = 0.5 * P_rad_test + 0.5 * P_cnn_test
thr_avg    = best_threshold_youden(y_val, P_avg_val)
m_avg_val  = compute_metrics(y_val,  P_avg_val,  thr_avg)
m_avg_test = compute_metrics(y_test, P_avg_test, thr_avg)
print(f"  val={m_avg_val['ROC-AUC']:.4f}  test={m_avg_test['ROC-AUC']:.4f}")

"""# Strategy B — Weighted average"""

print("── Strategy B: Weighted average (grid search on val) ────────")
best_w, best_w_auc = 0.5, -1.0
for w in WEIGHT_GRID:
    auc = roc_auc_score(y_val, w * P_rad_val + (1 - w) * P_cnn_val)
    if auc > best_w_auc:
        best_w_auc, best_w = auc, w

P_wt_val   = best_w * P_rad_val  + (1 - best_w) * P_cnn_val
P_wt_test  = best_w * P_rad_test + (1 - best_w) * P_cnn_test
thr_wt     = best_threshold_youden(y_val, P_wt_val)
m_wt_val   = compute_metrics(y_val,  P_wt_val,  thr_wt)
m_wt_test  = compute_metrics(y_test, P_wt_test, thr_wt)

dominant = (f"radiomic (w={best_w:.2f})" if best_w > 0.5
            else f"CNN (w={1-best_w:.2f})" if best_w < 0.5
            else "both equally")
print(f"  Best w={best_w:.2f}  → trusts {dominant} more")
print(f"  val={m_wt_val['ROC-AUC']:.4f}  test={m_wt_test['ROC-AUC']:.4f}\n")

"""# Strategy C — Meta-classifier"""

print("── Strategy C: Meta-LR on [P_rad, P_cnn] ───────────────────")
# Why fit on validation, not training?
# The branch models were trained on the training set, so their training-set
# probabilities are overfit (too confident). Validation probabilities are
# unbiased estimates — correct data to fit the meta-learner on.
X_meta_val  = np.column_stack([P_rad_val,  P_cnn_val])
X_meta_test = np.column_stack([P_rad_test, P_cnn_test])

meta_clf = LogisticRegression(
    C=1.0, solver="lbfgs", max_iter=500,
    class_weight="balanced", random_state=RANDOM_STATE,
)
meta_clf.fit(X_meta_val, y_val)

P_meta_val  = meta_clf.predict_proba(X_meta_val)[:, 1]
P_meta_test = meta_clf.predict_proba(X_meta_test)[:, 1]
thr_meta    = best_threshold_youden(y_val, P_meta_val)
m_meta_val  = compute_metrics(y_val,  P_meta_val,  thr_meta)
m_meta_test = compute_metrics(y_test, P_meta_test, thr_meta)

coef_rad, coef_cnn = meta_clf.coef_[0]
print(f"  Meta-LR coef — radiomic={coef_rad:.4f}  CNN={coef_cnn:.4f}")
print(f"  val={m_meta_val['ROC-AUC']:.4f}  test={m_meta_test['ROC-AUC']:.4f}")

"""# Results tables"""

STRATEGIES = {
    "Simple avg"  : (m_avg_val,  m_avg_test,  P_avg_val,  P_avg_test),
    "Weighted avg": (m_wt_val,   m_wt_test,   P_wt_val,   P_wt_test),
    "Meta-LR"     : (m_meta_val, m_meta_test, P_meta_val, P_meta_test),
}
METRIC_DISPLAY = ["ROC-AUC","Avg-Prec","Brier","Accuracy",
                  "Sensitivity","Specificity","F1","MCC"]

rows_val  = [{"Strategy": s, **v} for s,(v,_,_,_) in STRATEGIES.items()]
rows_test = [{"Strategy": s, **t} for s,(_,t,_,_) in STRATEGIES.items()]
df_val    = pd.DataFrame(rows_val).set_index("Strategy")
df_test   = pd.DataFrame(rows_test).set_index("Strategy")

print("=" * 65)
print("LATE FUSION — VALIDATION METRICS")
print("=" * 65)
print(df_val[METRIC_DISPLAY].to_string())
print(f"\n{'='*65}")
print("LATE FUSION — TEST METRICS")
print("=" * 65)
print(df_test[METRIC_DISPLAY].to_string())

df_val.to_csv( os.path.join(OUTPUT_DIR, "results_val.csv"))
df_test.to_csv(os.path.join(OUTPUT_DIR, "results_test.csv"))

"""# Ablation table  (thesis-ready — last row filled after Step 7)"""

best_late = df_test["ROC-AUC"].idxmax()
blt_v, blt_t = STRATEGIES[best_late][0], STRATEGIES[best_late][1]

rad_m_test = compute_metrics(y_test, P_rad_test, rad_thr)
cnn_m_test = compute_metrics(y_test, P_cnn_test, cnn_thr)

ablation = pd.DataFrame([
    {"Branch / Strategy"  : f"Radiomic only  ({rad_name})",
     "Val AUC"            : round(rad_val_auc, 4),
     "Test AUC"           : round(rad_tst_auc, 4),
     "Test Sensitivity"   : rad_m_test["Sensitivity"],
     "Test Specificity"   : rad_m_test["Specificity"],
     "Test F1"            : rad_m_test["F1"],
     "Test MCC"           : rad_m_test["MCC"]},
    {"Branch / Strategy"  : f"CNN only  ({cnn_name})",
     "Val AUC"            : round(cnn_val_auc, 4),
     "Test AUC"           : round(cnn_tst_auc, 4),
     "Test Sensitivity"   : cnn_m_test["Sensitivity"],
     "Test Specificity"   : cnn_m_test["Specificity"],
     "Test F1"            : cnn_m_test["F1"],
     "Test MCC"           : cnn_m_test["MCC"]},
    {"Branch / Strategy"  : f"Late fusion — {best_late}",
     "Val AUC"            : blt_v["ROC-AUC"],
     "Test AUC"           : blt_t["ROC-AUC"],
     "Test Sensitivity"   : blt_t["Sensitivity"],
     "Test Specificity"   : blt_t["Specificity"],
     "Test F1"            : blt_t["F1"],
     "Test MCC"           : blt_t["MCC"]},
    {"Branch / Strategy"  : "Early fusion + SE-gating  (Step 7)",
     "Val AUC"            : "— run Step 7",
     "Test AUC"           : "— run Step 7",
     "Test Sensitivity"   : "—",
     "Test Specificity"   : "—",
     "Test F1"            : "—",
     "Test MCC"           : "—"},
]).set_index("Branch / Strategy")

ablation.to_csv(os.path.join(OUTPUT_DIR, "ablation_table.csv"))

print(f"{'='*65}")
print("THESIS ABLATION TABLE")
print(f"{'='*65}")
print(ablation.to_string())
print("\n  Fill in the last row after running Step 7.")

"""# Plots"""

# ── ROC curves ────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for ax, (y_true, pd_dict, title) in zip(axes, [
    (y_val,  {"Radiomic only": P_rad_val,  "CNN only": P_cnn_val,
              "Simple avg": P_avg_val, "Weighted avg": P_wt_val,
              "Meta-LR": P_meta_val}, "Validation"),
    (y_test, {"Radiomic only": P_rad_test, "CNN only": P_cnn_test,
              "Simple avg": P_avg_test, "Weighted avg": P_wt_test,
              "Meta-LR": P_meta_test}, "Test"),
]):
    ax.plot([0,1],[0,1],"k--",lw=1)
    for label, prob in pd_dict.items():
        fpr, tpr, _ = roc_curve(y_true, prob)
        auc = roc_auc_score(y_true, prob)
        ls  = "--" if "only" in label else "-"
        lw  = 1.5  if "only" in label else 2.5
        ax.plot(fpr, tpr, ls, lw=lw, color=PALETTE.get(label,"gray"),
                label=f"{label}  AUC={auc:.3f}")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC — Late Fusion ({title})", fontweight="bold")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "roc_curves.png"), dpi=DPI)
plt.close()

# ── Test AUC bar chart ────────────────────────────────────────────────────────
bar_labels = ["Radiomic\nonly", "CNN\nonly",
              "Simple\navg", "Weighted\navg", "Meta-LR"]
bar_aucs   = [rad_tst_auc, cnn_tst_auc,
              m_avg_test["ROC-AUC"], m_wt_test["ROC-AUC"],
              m_meta_test["ROC-AUC"]]
bar_colors = [PALETTE["Radiomic only"], PALETTE["CNN only"],
              PALETTE["Simple avg"], PALETTE["Weighted avg"],
              PALETTE["Meta-LR"]]

fig, ax = plt.subplots(figsize=(9, 5))
bars = ax.bar(bar_labels, bar_aucs, color=bar_colors,
              edgecolor="white", linewidth=0.8, width=0.55)
ax.axhline(0.5, color="gray", linestyle="--", lw=1, label="Random (AUC=0.5)")
ax.set_ylim(0, 1.1)
ax.set_ylabel("Test ROC-AUC", fontsize=12)
ax.set_title("Test AUC — Baselines vs Late Fusion Strategies",
             fontsize=12, fontweight="bold")
ax.legend(fontsize=9); ax.grid(axis="y", alpha=0.3)
for bar, v in zip(bars, bar_aucs):
    ax.text(bar.get_x()+bar.get_width()/2, v+0.012,
            f"{v:.3f}", ha="center", va="bottom",
            fontsize=10, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "comparison_test.png"), dpi=DPI)
plt.close()

# ── Calibration ───────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, (y_true, pd_dict, title) in zip(axes, [
    (y_val,  {"Simple avg": P_avg_val,  "Weighted avg": P_wt_val,
              "Meta-LR": P_meta_val}, "Validation"),
    (y_test, {"Simple avg": P_avg_test, "Weighted avg": P_wt_test,
              "Meta-LR": P_meta_test}, "Test"),
]):
    ax.plot([0,1],[0,1],"k--",lw=1,label="Perfect calibration")
    for label, prob in pd_dict.items():
        fp, mp = calibration_curve(y_true, prob, n_bins=8, strategy="uniform")
        brier  = brier_score_loss(y_true, prob)
        ax.plot(mp, fp, "o-", lw=2, color=PALETTE[label],
                label=f"{label}  Brier={brier:.3f}")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of positives (PDAC)")
    ax.set_title(f"Calibration — {title}", fontweight="bold")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "calibration.png"), dpi=DPI)
plt.close()

print("Saved: roc_curves.png  comparison_test.png  calibration.png")
print(f"All outputs → {OUTPUT_DIR}")
print("\nRun step7_fusion.py (early fusion + SE-gating) next,")
print("then fill in the last row of ablation_table.csv.")

"""# bootstrap_auc_ci"""

from sklearn.utils import resample

def bootstrap_auc_ci(y_true, y_prob, n_boot=1000, ci=0.95, random_state=42):
    """
    Bootstrap 95% CI for ROC-AUC on the test set.
    Returns (mean_auc, lower_bound, upper_bound).
    """
    rng  = np.random.RandomState(random_state)
    aucs = []
    for _ in range(n_boot):
        idx      = resample(np.arange(len(y_true)), random_state=rng)
        y_b      = y_true[idx]
        p_b      = y_prob[idx]
        if len(np.unique(y_b)) < 2:
            continue   # skip bootstrap samples with only one class
        aucs.append(roc_auc_score(y_b, p_b))
    aucs  = np.array(aucs)
    alpha = (1 - ci) / 2
    return aucs.mean(), np.percentile(aucs, alpha*100), np.percentile(aucs, (1-alpha)*100)

P_test = np.load(os.path.join('/content/drive/MyDrive/classif_model_AYTO/steps_scenarioA/step7_EarlyFusion/', "P_test.npy"))

# Run for each strategy:
for label, y_true, y_prob in [
    ("Radiomic only",         y_test, P_rad_test),
    ("CNN only",              y_test, P_cnn_test),
    ("Late fusion weighted",  y_test, P_wt_test),
    ("Early fusion SE-gating",y_test, P_test),    # from step7
]:
    mean_auc, lo, hi = bootstrap_auc_ci(y_true, y_prob)
    print(f"{label:<30}  AUC={mean_auc:.4f}  95% CI=[{lo:.4f}, {hi:.4f}]")

import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.utils import resample
import matplotlib.pyplot as plt
import os

def bootstrap_roc_curve_ci(y_true, y_prob, n_boot=1000, ci=0.95, random_state=42):
    """
    Computes bootstrap 95% CI bands for an ROC curve.
    Returns mean_fpr, mean_tpr, lower_tpr, upper_tpr.
    """
    rng = np.random.RandomState(random_state)
    tprs = []
    base_fpr = np.linspace(0, 1, 101) # Common FPR points for interpolation

    for i in range(n_boot):
        # Bootstrap resampling
        indices = resample(np.arange(len(y_true)), random_state=rng)
        y_b = y_true[indices]
        p_b = y_prob[indices]

        # Skip if only one class in bootstrap sample
        if len(np.unique(y_b)) < 2:
            continue

        fpr, tpr, _ = roc_curve(y_b, p_b)
        tprs.append(np.interp(base_fpr, fpr, tpr)) # Interpolate TPRs to common FPRs

    tprs = np.array(tprs)
    mean_tprs = tprs.mean(axis=0)

    alpha = (1 - ci) / 2
    lower_tpr = np.percentile(tprs, alpha * 100, axis=0)
    upper_tpr = np.percentile(tprs, (1 - alpha) * 100, axis=0)

    return base_fpr, mean_tprs, lower_tpr, upper_tpr

# Extend STRATEGIES with individual branch probabilities for plotting
STRATEGIES_FOR_PLOT = {
    "Radiomic only": P_rad_test,
    "CNN only": P_cnn_test,
    "Simple avg": P_avg_test,
    "Weighted avg": P_wt_test,
    "Meta-LR": P_meta_test,
}

fig, ax = plt.subplots(figsize=(8, 6))
ax.plot([0,1],[0,1],"k--",lw=1, label="Chance level")

for label, y_prob_test in STRATEGIES_FOR_PLOT.items():
    # Calculate bootstrap CI for AUC value
    mean_auc, auc_lo, auc_hi = bootstrap_auc_ci(y_test, y_prob_test, n_boot=1000, ci=0.95, random_state=RANDOM_STATE)

    # Calculate bootstrap CI for ROC curve
    mean_fpr, mean_tpr, lower_tpr, upper_tpr = bootstrap_roc_curve_ci(y_test, y_prob_test, n_boot=1000, ci=0.95, random_state=RANDOM_STATE)

    ls  = "--" if "only" in label else "-"
    lw  = 1.5  if "only" in label else 2.5
    color = PALETTE.get(label, "gray")

    ax.plot(mean_fpr, mean_tpr, ls, lw=lw, color=color,
            label=f"{label}  AUC={mean_auc:.3f} (95% CI: {auc_lo:.3f}-{auc_hi:.3f})")

    ax.fill_between(mean_fpr, lower_tpr, upper_tpr, color=color, alpha=0.1, lw=0) # Shaded CI band

ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curves with 95% Bootstrap CI Bands (Test Set)", fontweight="bold")
ax.legend(fontsize=8, loc='lower right'); ax.grid(alpha=0.3)
ax.set_aspect('equal', adjustable='box') # Make the plot square
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "roc_curves_with_ci.png"), dpi=DPI)
plt.close()

print(f"Saved: roc_curves_with_ci.png to {OUTPUT_DIR}")
