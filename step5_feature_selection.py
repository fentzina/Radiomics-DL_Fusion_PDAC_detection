# -*- coding: utf-8 -*-
"""
STEP 5 — Feature Selection Pipeline
Five-stage pipeline
-------------------
*  Stage 1 — Variance filter      remove near-constant features (thresh 1e-4)
*  Stage 2 — Univariate ANOVA     keep top-K by F-statistic (K=20)
*  Stage 3 — Correlation pruning  remove |r|>0.90, keep higher-F-ranked feature
*  Stage 4 — Multi-method bootstrap stability (50 rounds each):

               a) LASSO / L1 logistic regression
               b) Recursive Feature Elimination (RFE) with SVM
               c) Random Forest importance
               d) XGBoost importance
               e) mRMR  (Minimum Redundancy Maximum Relevance)
*  Stage 5 — Consensus & fallback  majority vote (≥3/5 methods) + fallback

Notes on mRMR
-------------
mRMR ranks features by maximising relevance to the class label while
minimising redundancy with already-selected features.  We use a
mutual-information formulation (MRMR_METHOD = "MID"):

  score_i = MI(feature_i, y) - (1/|S|) * Σ_{j in S} MI(feature_i, feature_j)

where S is the current selected set.  Each bootstrap round re-runs mRMR on the
resampled data and selects the top MRMR_TOP_K features.

**StandardScaler** is fit on training data ONLY, then applied identically to
validation and test splits.  The scaler is saved for inference.
"""

import os
import warnings
import numpy as np
import pandas as pd
import joblib

from sklearn.feature_selection import (
    VarianceThreshold, f_classif,
    RFE, RFECV,
)
from sklearn.linear_model  import LogisticRegression
from sklearn.svm           import SVC
from sklearn.ensemble      import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics       import roc_auc_score
from xgboost               import XGBClassifier
from scipy.stats           import rankdata

warnings.filterwarnings("ignore")

# ── Google Colab / Drive setup ────────────────────────────────────────────────
from google.colab import drive
drive.mount('/content/drive', force_remount=True)

STEP2_DIR       = "/content/drive/MyDrive/classif_model_AYTO/steps_scenarioA/step2_scenarioA"
HARMONIZED_DIR  = "/content/drive/MyDrive/classif_model_AYTO/steps_scenarioA/step3_scenarioA"
OUTPUT_DIR      = "/content/drive/MyDrive/classif_model_AYTO/steps_scenarioA/step5_scenarioA"
os.makedirs(OUTPUT_DIR, exist_ok=True)

N_CHANNELS      = 26
STATS           = ["mean", "std", "q25", "q75"]
FEATURE_NAMES   = [
    f"haralick_{c+1}_{s}"
    for c in range(N_CHANNELS)
    for s in STATS
] # 104 names

# Stage 1–3 thresholds
VARIANCE_THRESH = 1e-4
CORR_THRESH     = 0.90

# In step5_feature_selection.py, change:
TOP_K_ANOVA      = 30     # was 20 — keep more candidates --- #TOP_K_ANOVA = 20 # candidates entering stage 4
STABILITY_THRESH = 0.30   # was 0.40 — easier consensus --- #STABILITY_THRESH = fraction of rounds to count as stable
MIN_METHODS      = 2      # was 3 — only 2/5 methods need to agree ---  ≥ MIN_METHODS/5 must agree for hard select

# Stage 4 bootstrap parameters
N_BOOTSTRAP     = 50

# Method-specific parameters
LASSO_C         = 0.1 # inverse regularisation strength
RFE_N_FEATURES  = 10 # features RFE targets per round
RFE_STEP        = 1 # features dropped per RFE step
RF_N_ESTIMATORS = 200
XGB_N_ESTIMATORS = 100
MRMR_TOP_K      = 10 # features mRMR selects per round
MRMR_METHOD     = "MID" # "MID" (mutual-info difference) only

# Consensus
FALLBACK_N      = 5 # fallback: take top-N by vote sum if zero survive

"""# Functions Definition"""

# @title
# ── Utility: mutual information (discrete-continuous, binned) ─────────────────
def _mutual_info(x: np.ndarray, y: np.ndarray, n_bins: int = 10) -> float:
    """Estimate MI between a continuous feature x and binary label y via binning."""
    bins   = np.linspace(x.min() - 1e-9, x.max() + 1e-9, n_bins + 1)
    x_disc = np.digitize(x, bins) - 1          # 0 … n_bins-1
    n      = len(y)
    mi     = 0.0
    y_vals = np.unique(y)
    x_vals = np.arange(n_bins)
    p_y    = {yv: np.sum(y == yv) / n for yv in y_vals}
    for xv in x_vals:
        mask_x = x_disc == xv
        p_x    = mask_x.sum() / n
        if p_x == 0:
            continue
        for yv in y_vals:
            p_xy = np.sum(mask_x & (y == yv)) / n
            if p_xy == 0:
                continue
            mi += p_xy * np.log(p_xy / (p_x * p_y[yv]) + 1e-12)
    return max(mi, 0.0)


def _mi_feature_feature(x_i: np.ndarray, x_j: np.ndarray, n_bins: int = 10) -> float:
    """Estimate MI between two continuous features via joint binning."""
    bins_i = np.linspace(x_i.min() - 1e-9, x_i.max() + 1e-9, n_bins + 1)
    bins_j = np.linspace(x_j.min() - 1e-9, x_j.max() + 1e-9, n_bins + 1)
    di     = np.digitize(x_i, bins_i) - 1
    dj     = np.digitize(x_j, bins_j) - 1
    n      = len(x_i)
    mi     = 0.0
    p_i    = {v: np.sum(di == v) / n for v in range(n_bins)}
    p_j    = {v: np.sum(dj == v) / n for v in range(n_bins)}
    for vi in range(n_bins):
        for vj in range(n_bins):
            p_ij = np.sum((di == vi) & (dj == vj)) / n
            if p_ij == 0:
                continue
            pi, pj = p_i[vi], p_j[vj]
            if pi == 0 or pj == 0:
                continue
            mi += p_ij * np.log(p_ij / (pi * pj) + 1e-12)
    return max(mi, 0.0)


def mrmr_select(X: np.ndarray, y: np.ndarray, k: int, n_bins: int = 10) -> list:
    """
    mRMR feature selection (MID variant).

    Returns indices (into X's columns) of the top-k selected features.
    Selection rule at each step:
        score_i = MI(x_i, y)  -  mean_{j in S} MI(x_i, x_j)
    where S is the already-selected set.
    """
    n_feat = X.shape[1]
    k      = min(k, n_feat)

    # Pre-compute relevance (MI with class) for all features
    relevance = np.array([_mutual_info(X[:, i], y, n_bins) for i in range(n_feat)])

    selected  = []
    remaining = list(range(n_feat))

    for _ in range(k):
        if not remaining:
            break
        if not selected:
            # First feature: highest relevance
            best = remaining[int(np.argmax(relevance[remaining]))]
        else:
            scores = []
            for i in remaining:
                redundancy = np.mean([
                    _mi_feature_feature(X[:, i], X[:, j], n_bins)
                    for j in selected
                ])
                scores.append(relevance[i] - redundancy)
            best = remaining[int(np.argmax(scores))]
        selected.append(best)
        remaining.remove(best)

    return selected

"""# LOAD DATA"""

X_tr = np.load(os.path.join(HARMONIZED_DIR, "X_train_harmonized_scenarioA.npy"))
y_tr = np.load(os.path.join(STEP2_DIR,      "y_train_scenarioA.npy"))

print(f"Loaded: X_train={X_tr.shape}  y_train={y_tr.shape}")
print(f"{int(y_tr.sum())} PDAC / {int((1-y_tr).sum())} non-PDAC")

"""# STAGE 1 — Variance filter"""

print(f"Stage 1: Variance filter  (threshold > {VARIANCE_THRESH})")
vt      = VarianceThreshold(threshold=VARIANCE_THRESH)
vt.fit(X_tr)
mask_v  = vt.get_support()
X_tr_v  = X_tr[:, mask_v]
names_v = [FEATURE_NAMES[i] for i, keep in enumerate(mask_v) if keep]

print(f"  Removed {mask_v.size - mask_v.sum()} near-constant features "
      f"→ {len(names_v)} remain")

"""# STAGE 2 — Univariate ANOVA (top-K)"""

print(f"Stage 2: Univariate ANOVA  (top-{TOP_K_ANOVA})")
f_scores, p_values = f_classif(X_tr_v, y_tr)
top_k_actual       = min(TOP_K_ANOVA, len(names_v))
top_k_idx          = np.argsort(f_scores)[::-1][:top_k_actual]

X_tr_a  = X_tr_v[:, top_k_idx]
names_a  = [names_v[i] for i in top_k_idx]
f_rank   = {names_v[i]: f_scores[i] for i in top_k_idx}  # kept for pruning below

print(f"  Top-{top_k_actual} features retained:")
for rank, idx in enumerate(top_k_idx):
    print(f"    {rank+1:2d}. {names_v[idx]:35s}  F={f_scores[idx]:.2f}  "
          f"p={p_values[idx]:.4f}")
print()

"""# STAGE 3 — Correlation pruning (|r| > CORR_THRESH)"""

print(f"Stage 3: Correlation pruning  (|r| > {CORR_THRESH})")

corr_matrix = np.corrcoef(X_tr_a.T)
np.fill_diagonal(corr_matrix, 0.0)

to_remove = set()
for i in range(len(names_a)):
    if i in to_remove:
        continue
    for j in range(i + 1, len(names_a)):
        if j in to_remove:
            continue
        if abs(corr_matrix[i, j]) > CORR_THRESH:
            # Remove the lower-ranked feature (larger index = lower F-rank)
            to_remove.add(j)

keep_idx_corr = [i for i in range(len(names_a)) if i not in to_remove]
X_tr_c        = X_tr_a[:, keep_idx_corr]
names_c       = [names_a[i] for i in keep_idx_corr]

print(f"  Removed {len(to_remove)} correlated features → {len(names_c)} remain")
print(f"  Surviving: {names_c}")

"""# STAGE 4 — Multi-method bootstrap stability"""

print(f"Stage 4: Multi-method bootstrap stability  ({N_BOOTSTRAP} rounds × 5 methods)")
print(f"  Methods: LASSO | RFE-SVM | RandomForest | XGBoost | mRMR")

METHODS = ["lasso", "rfe", "rf", "xgb", "mrmr"]
counts  = {name: {m: 0 for m in METHODS} for name in names_c}
n_feat  = len(names_c)

for b in range(N_BOOTSTRAP):
    rng      = np.random.RandomState(b)
    boot_idx = rng.choice(len(X_tr_c), size=len(X_tr_c), replace=True)
    X_b      = X_tr_c[boot_idx]
    y_b      = y_tr[boot_idx]
    X_b_sc   = StandardScaler().fit_transform(X_b)

    # ── a) LASSO (L1 logistic regression) ────────────────────────────────────
    # Selects by driving uninformative coefficients to exactly zero.
    lasso = LogisticRegression(
        penalty="l1", solver="liblinear",
        C=LASSO_C,
        class_weight="balanced",
        random_state=b, max_iter=500
    )
    lasso.fit(X_b_sc, y_b)
    selected_lasso = set(np.where(np.abs(lasso.coef_[0]) > 0)[0])
    for i in selected_lasso:
        counts[names_c[i]]["lasso"] += 1

    # ── b) RFE with linear SVM ────────────────────────────────────────────────
    # Iteratively removes the feature with the smallest absolute SVM weight,
    # making selection robust to the choice of decision boundary.
    n_rfe = min(RFE_N_FEATURES, n_feat)
    svc   = SVC(kernel="linear", C=1.0, class_weight="balanced", random_state=b)
    rfe   = RFE(estimator=svc, n_features_to_select=n_rfe, step=RFE_STEP)
    rfe.fit(X_b_sc, y_b)
    selected_rfe = set(np.where(rfe.support_)[0])
    for i in selected_rfe:
        counts[names_c[i]]["rfe"] += 1

    # ── c) Random Forest importance ───────────────────────────────────────────
    # Mean decrease in impurity across all trees; threshold = mean importance.
    rf = RandomForestClassifier(
        n_estimators=RF_N_ESTIMATORS, random_state=b,
        class_weight="balanced",
        n_jobs=-1
    )
    rf.fit(X_b, y_b)
    rf_thresh     = np.mean(rf.feature_importances_)
    selected_rf   = set(np.where(rf.feature_importances_ >= rf_thresh)[0])
    for i in selected_rf:
        counts[names_c[i]]["rf"] += 1

    # ── d) XGBoost importance ─────────────────────────────────────────────────
    # Gradient-boosted tree gain; threshold = mean importance.
    xgb = XGBClassifier(
        n_estimators=XGB_N_ESTIMATORS,
        eval_metric="logloss",
        class_weight="balanced",
        random_state=b, verbosity=0, n_jobs=-1
    )
    xgb.fit(X_b, y_b)
    xgb_thresh    = np.mean(xgb.feature_importances_)
    selected_xgb  = set(np.where(xgb.feature_importances_ >= xgb_thresh)[0])
    for i in selected_xgb:
        counts[names_c[i]]["xgb"] += 1

    # ── e) mRMR ──────────────────────────────────────────────────────────────
    # Selects features with high MI to the class and low MI among themselves.
    # Particularly well-suited for radiomics: many correlated texture features.
    mrmr_k        = min(MRMR_TOP_K, n_feat)
    selected_mrmr = set(mrmr_select(X_b, y_b, k=mrmr_k))
    for i in selected_mrmr:
        counts[names_c[i]]["mrmr"] += 1

    if (b + 1) % 10 == 0:
        print(f"  Bootstrap round {b+1:3d}/{N_BOOTSTRAP} done")

print()

"""# STAGE 5 — Consensus + fallback"""

print(f"Stage 5: Consensus  (majority = ≥{MIN_METHODS}/5 methods agree)")
threshold    = N_BOOTSTRAP * STABILITY_THRESH
vote_sum     = {}   # total number of methods that passed threshold per feature

for name in names_c:
    votes = sum(
        1 for m in METHODS
        if counts[name][m] > threshold
    )
    vote_sum[name] = votes

# Hard-select: feature must pass threshold in ≥ MIN_METHODS methods
final_names = [n for n in names_c if vote_sum[n] >= MIN_METHODS]

# Fallback: if zero (or very few) features survive hard consensus, take top-N
# by total vote count so the pipeline always produces some output.
if len(final_names) < 2:
    print(f"  WARNING: only {len(final_names)} feature(s) passed hard consensus "
          f"(≥{MIN_METHODS}/5).  Applying fallback: top-{FALLBACK_N} by vote sum.")
    sorted_by_votes = sorted(names_c, key=lambda n: vote_sum[n], reverse=True)
    final_names     = sorted_by_votes[:FALLBACK_N]

"""# Save per-method stability table"""

rows = []
for name in names_c:
    row = {"feature": name}
    for m in METHODS:
        row[f"{m}_count"] = counts[name][m]
        row[f"{m}_frac"]  = round(counts[name][m] / N_BOOTSTRAP, 3)
    row["vote_sum"]  = vote_sum[name]
    row["selected"]  = name in final_names
    rows.append(row)

stab_df = pd.DataFrame(rows).sort_values("vote_sum", ascending=False)
stab_df.to_csv(os.path.join(OUTPUT_DIR, "stability_counts_scenarioA.csv"), index=False)

# Selection summary: one row per method + union/intersection
summary_rows = []
for m in METHODS:
    selected_by_m = [n for n in names_c if counts[n][m] > threshold]
    summary_rows.append({
        "method"          : m,
        "n_selected"      : len(selected_by_m),
        "features"        : ", ".join(selected_by_m),
    })
summary_rows.append({
    "method"   : f"consensus (≥{MIN_METHODS}/5)",
    "n_selected": len(final_names),
    "features" : ", ".join(final_names),
})
pd.DataFrame(summary_rows).to_csv(
    os.path.join(OUTPUT_DIR, "selection_summary_scenarioA.csv"), index=False
)

print("Per-method stability fractions (sorted by vote sum):")
print(stab_df[["feature"] + [f"{m}_frac" for m in METHODS] + ["vote_sum", "selected"]]
      .to_string(index=False))
print()
print(f"  LASSO stable      : {sum(1 for n in names_c if counts[n]['lasso'] > threshold)}")
print(f"  RFE-SVM stable    : {sum(1 for n in names_c if counts[n]['rfe']   > threshold)}")
print(f"  RF stable         : {sum(1 for n in names_c if counts[n]['rf']    > threshold)}")
print(f"  XGBoost stable    : {sum(1 for n in names_c if counts[n]['xgb']   > threshold)}")
print(f"  mRMR stable       : {sum(1 for n in names_c if counts[n]['mrmr']  > threshold)}")
print(f"\n  ── Final selected : {len(final_names)} features ──")
for i, n in enumerate(final_names, 1):
    print(f"    {i:2d}. {n}  (votes={vote_sum[n]}/5)")

"""# Map final names back to original column indices in master_X"""

original_idx_v    = np.where(mask_v)[0]          # indices surviving stage 1
original_idx_anova = original_idx_v[top_k_idx]   # surviving stage 2
original_idx_corr  = original_idx_anova[keep_idx_corr]  # surviving stage 3

final_names_set = set(final_names)
final_local_idx = [i for i, n in enumerate(names_c) if n in final_names_set]
# Preserve the order of final_names (not names_c order)
order_map       = {n: i for i, n in enumerate(final_names)}
final_local_idx = sorted(
    [i for i, n in enumerate(names_c) if n in final_names_set],
    key=lambda i: order_map[names_c[i]]
)
final_orig_idx  = original_idx_corr[final_local_idx]

print(f"\nOriginal column indices of selected features: {final_orig_idx.tolist()}")

"""# Extract selected features from all three splits"""

X_train_h = np.load(os.path.join(HARMONIZED_DIR, "X_train_harmonized_scenarioA.npy"))
X_val_h   = np.load(os.path.join(HARMONIZED_DIR, "X_val_harmonized_scenarioA.npy"))
X_test_h  = np.load(os.path.join(HARMONIZED_DIR, "X_test_harmonized_scenarioA.npy"))

X_train_sel = X_train_h[:, final_orig_idx]
X_val_sel   = X_val_h[:,   final_orig_idx]
X_test_sel  = X_test_h[:,  final_orig_idx]

np.save(os.path.join(OUTPUT_DIR, "X_train_selected_scenarioA.npy"), X_train_sel.astype(np.float32))
np.save(os.path.join(OUTPUT_DIR, "X_val_selected_scenarioA.npy"),   X_val_sel.astype(np.float32))
np.save(os.path.join(OUTPUT_DIR, "X_test_selected_scenarioA.npy"),  X_test_sel.astype(np.float32))
np.save(os.path.join(OUTPUT_DIR, "selected_feature_idx_scenarioA.npy"),
        final_orig_idx.astype(np.int32))
np.save(os.path.join(OUTPUT_DIR, "selected_feature_names_scenarioA.npy"),
        np.array(final_names, dtype=str))

"""# StandardScaler — fit on train ONLY, apply to all splits"""

scaler      = StandardScaler()
X_train_sc  = scaler.fit_transform(X_train_sel)
X_val_sc    = scaler.transform(X_val_sel)
X_test_sc   = scaler.transform(X_test_sel)

np.save(os.path.join(OUTPUT_DIR, "X_train_scaled_scenarioA.npy"), X_train_sc.astype(np.float32))
np.save(os.path.join(OUTPUT_DIR, "X_val_scaled_scenarioA.npy"),   X_val_sc.astype(np.float32))
np.save(os.path.join(OUTPUT_DIR, "X_test_scaled_scenarioA.npy"),  X_test_sc.astype(np.float32))
joblib.dump(scaler, os.path.join(OUTPUT_DIR, "radiomics_scaler_scenarioA.pkl"))

"""# Final summary"""

print("Step 5 complete — files saved to:", OUTPUT_DIR)
print(f"  X_train_selected.npy   shape: {X_train_sel.shape}")
print(f"  X_val_selected.npy     shape: {X_val_sel.shape}")
print(f"  X_test_selected.npy    shape: {X_test_sel.shape}")
print(f"  X_train_scaled.npy     shape: {X_train_sc.shape}")
print(f"  X_val_scaled.npy       shape: {X_val_sc.shape}")
print(f"  X_test_scaled.npy      shape: {X_test_sc.shape}")
print(f"  selected_feature_names.npy   → {final_names}")
print(f"  selected_feature_idx.npy     → {final_orig_idx.tolist()}")
print(f"  radiomics_scaler.pkl")
print(f"  stability_counts.csv")
print(f"  selection_summary.csv")
print("\nRun step6_cnn_patch_training.py next.")

# Return shape for notebook inspection
X_train_sc.shape

"""SKONAK
------

AngularSecondMoment = 0
    Contrast = 1
    Correlation = 2
    SumOfSquareVariance = 3
    SumAverage = 4
    SumVariance = 5
    SumEntropy = 6
    Entropy = 7
    DifferenceVariance = 8
    DifferenceEntropy = 9
    InformationMeasureOfCorrelation1 = 10
    InformationMeasureOfCorrelation2 = 11
    MaximalCorrelationCoefficient = 12
"""
