#!/usr/bin/env python3
"""
analyze_heterogeneity.py
========================
Descriptive statistical analysis of central vs. peripheral CoLIAGe
radiomic features across PDAC cases.

Input  : aggregated arrays from the aggregation step
           central_vectors.npy       (N, 26)  float32
           peripheral_vectors.npy    (N, 26)  float32
           heterogeneity_vectors.npy (N, 26)  float32  (central - peripheral)
           case_ids.npy              (N,)     str
         optionally:
           tumor_volumes.npy         (N,)     float32  voxel count per case

Output : results/
           heterogeneity_paired_ttests.csv
           heterogeneity_magnitude_stats.csv
           heterogeneity_correlation_with_volume.csv  (if tumor_volumes.npy present)
           heterogeneity_pca_components.csv
           plots/
             01_paired_ttest_pvalues.png
             02_feature_distributions_central_vs_peripheral.png
             03_heterogeneity_magnitude_distribution.png
             04_pca_heterogeneity_space.png
             05_correlation_with_volume.png  (if tumor_volumes.npy present)
             06_feature_heatmap_central_vs_peripheral.png

Run:
    python analyze_heterogeneity.py \
        --aggregated_dir /path/to/aggregated \
        --output_dir     /path/to/results
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")   # non-interactive backend, safe for server runs
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import ttest_rel, wilcoxon, pearsonr, spearmanr
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# ── Feature names — 13 Haralick × 2 angles = 26 channels ────────────────────
HARALICK_BASE = [
    "AngularSecondMoment",
    "Contrast",
    "Correlation",
    "SumOfSquaresVariance",
    "SumAverage",
    "SumVariance",
    "SumEntropy",
    "Entropy",
    "DifferenceVariance",
    "DifferenceEntropy",
    "InfoMeasureCorr1",
    "InfoMeasureCorr2",
    "MaximalCorrelationCoeff",
]
FEATURE_NAMES = (
    [f"{f}_Primary"   for f in HARALICK_BASE] +
    [f"{f}_Secondary" for f in HARALICK_BASE]
)
assert len(FEATURE_NAMES) == 26


# ── Helpers ───────────────────────────────────────────────────────────────────
def apply_bonferroni(p_values, n_tests=None):
    n = n_tests or len(p_values)
    return np.clip(np.array(p_values) * n, 0, 1)


def cohens_d_paired(a, b):
    """Cohen's d for paired samples."""
    diff = a - b
    return diff.mean() / (diff.std(ddof=1) + 1e-12)


def save_fig(fig, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ── Analysis functions ────────────────────────────────────────────────────────

def run_paired_ttests(central, peripheral, feature_names, output_dir):
    """
    Paired t-test and Wilcoxon signed-rank test per feature.
    Tests H0: no difference between central and peripheral mean texture.
    Bonferroni correction applied for multiple comparisons.
    """
    print("\n── 1. Paired statistical tests (central vs. peripheral) ─────────────")
    n_features = len(feature_names)
    results = []

    for i, fname in enumerate(feature_names):
        c = central[:, i]
        p = peripheral[:, i]

        t_stat, p_ttest  = ttest_rel(c, p)
        try:
            w_stat, p_wilcox = wilcoxon(c - p)
        except ValueError:
            w_stat, p_wilcox = np.nan, np.nan   # all differences zero — edge case

        d = cohens_d_paired(c, p)
        mean_diff = (c - p).mean()
        std_diff  = (c - p).std(ddof=1)

        results.append({
            "feature"        : fname,
            "mean_central"   : c.mean(),
            "mean_peripheral": p.mean(),
            "mean_diff"      : mean_diff,
            "std_diff"       : std_diff,
            "cohens_d"       : d,
            "t_statistic"    : t_stat,
            "p_ttest_raw"    : p_ttest,
            "w_statistic"    : w_stat,
            "p_wilcox_raw"   : p_wilcox,
        })

    df = pd.DataFrame(results)
    df["p_ttest_bonferroni"] = apply_bonferroni(df["p_ttest_raw"].values, n_features)
    df["p_wilcox_bonferroni"] = apply_bonferroni(df["p_wilcox_raw"].fillna(1).values, n_features)
    df["significant_ttest"]  = df["p_ttest_bonferroni"]  < 0.05
    df["significant_wilcox"] = df["p_wilcox_bonferroni"] < 0.05
    df = df.sort_values("p_ttest_bonferroni")

    n_sig = df["significant_ttest"].sum()
    print(f"  Features significantly different (Bonferroni t-test p<0.05): {n_sig}/{n_features}")

    out_csv = os.path.join(output_dir, "heterogeneity_paired_ttests.csv")
    df.to_csv(out_csv, index=False)
    print(f"  Saved: {out_csv}")

    # ── Plot: -log10(p) bar chart ─────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(14, 7))
    log_p = -np.log10(df["p_ttest_bonferroni"].clip(lower=1e-10))
    colors = ["tomato" if s else "steelblue" for s in df["significant_ttest"]]
    ax.barh(df["feature"], log_p, color=colors, edgecolor="none")
    ax.axvline(-np.log10(0.05), color="black", linestyle="--",
               linewidth=1.2, label="p=0.05 (Bonferroni)")
    ax.set_xlabel("-log₁₀(Bonferroni-corrected p-value)", fontsize=11)
    ax.set_title("Central vs. Peripheral: Feature Significance\n"
                 "(red = significant at α=0.05 after Bonferroni correction)",
                 fontsize=12, fontweight="bold")
    ax.tick_params(axis="y", labelsize=8)
    ax.legend(fontsize=9)
    plt.tight_layout()
    save_fig(fig, os.path.join(output_dir, "plots", "01_paired_ttest_pvalues.png"))

    return df


def plot_feature_distributions(central, peripheral, feature_names, output_dir):
    """
    Per-feature overlaid distributions: central (blue) vs. peripheral (orange).
    """
    print("\n── 2. Per-feature distribution plots ───────────────────────────────")
    n_feats = len(feature_names)
    n_cols  = 4
    n_rows  = (n_feats + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 3))
    axes = axes.flatten()

    for i, fname in enumerate(feature_names):
        ax = axes[i]
        ax.hist(central[:, i],    bins=25, alpha=0.6,
                color="steelblue", label="Central",    density=True)
        ax.hist(peripheral[:, i], bins=25, alpha=0.6,
                color="darkorange", label="Peripheral", density=True)
        ax.set_title(fname, fontsize=7, fontweight="bold")
        ax.tick_params(labelsize=6)
        if i == 0:
            ax.legend(fontsize=6)

    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    plt.suptitle("CoLIAGe Feature Distributions — Central vs. Peripheral (PDAC cases)",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    save_fig(fig, os.path.join(output_dir, "plots",
                               "02_feature_distributions_central_vs_peripheral.png"))


def analyze_heterogeneity_magnitude(heterogeneity_vectors, output_dir):
    """
    Distribution of the L2 norm of the heterogeneity vector across cases.
    Characterises how heterogeneous tumors are overall.
    """
    print("\n── 3. Heterogeneity magnitude analysis ─────────────────────────────")
    magnitudes = np.linalg.norm(heterogeneity_vectors, axis=1)

    stats_dict = {
        "n_cases"  : len(magnitudes),
        "mean"     : magnitudes.mean(),
        "std"      : magnitudes.std(ddof=1),
        "median"   : np.median(magnitudes),
        "q25"      : np.percentile(magnitudes, 25),
        "q75"      : np.percentile(magnitudes, 75),
        "min"      : magnitudes.min(),
        "max"      : magnitudes.max(),
    }
    df_stats = pd.DataFrame([stats_dict])
    out_csv  = os.path.join(output_dir, "heterogeneity_magnitude_stats.csv")
    df_stats.to_csv(out_csv, index=False)
    print(f"  Mean magnitude: {stats_dict['mean']:.4f} ± {stats_dict['std']:.4f}")
    print(f"  Median: {stats_dict['median']:.4f}  IQR: [{stats_dict['q25']:.4f}, {stats_dict['q75']:.4f}]")
    print(f"  Saved: {out_csv}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].hist(magnitudes, bins=30, color="mediumpurple", edgecolor="white", alpha=0.85)
    axes[0].axvline(magnitudes.mean(),   color="tomato", linestyle="--",
                    linewidth=1.5, label=f"Mean={magnitudes.mean():.3f}")
    axes[0].axvline(np.median(magnitudes), color="navy", linestyle=":",
                    linewidth=1.5, label=f"Median={np.median(magnitudes):.3f}")
    axes[0].set_xlabel("Heterogeneity Magnitude (L2 norm)", fontsize=10)
    axes[0].set_ylabel("Count", fontsize=10)
    axes[0].set_title("Distribution of Intra-Tumor Heterogeneity\nAcross PDAC Cases", fontsize=11)
    axes[0].legend(fontsize=9)

    # Sorted case plot — shows spread across the cohort
    sorted_mag = np.sort(magnitudes)
    axes[1].plot(sorted_mag, color="mediumpurple", linewidth=1.5)
    axes[1].fill_between(range(len(sorted_mag)), sorted_mag, alpha=0.2, color="mediumpurple")
    axes[1].set_xlabel("Case rank (sorted by magnitude)", fontsize=10)
    axes[1].set_ylabel("Heterogeneity Magnitude", fontsize=10)
    axes[1].set_title("Sorted Heterogeneity Magnitudes\n(each point = one PDAC case)", fontsize=11)

    plt.suptitle("Intra-Tumor Heterogeneity Magnitude Analysis", fontsize=12, fontweight="bold")
    plt.tight_layout()
    save_fig(fig, os.path.join(output_dir, "plots",
                               "03_heterogeneity_magnitude_distribution.png"))

    return magnitudes


def run_pca(heterogeneity_vectors, feature_names, output_dir):
    """
    PCA on the heterogeneity vectors — shows the main axes of
    texture variation across the PDAC cohort.
    """
    print("\n── 4. PCA on heterogeneity vectors ─────────────────────────────────")
    scaler = StandardScaler()
    H_scaled = scaler.fit_transform(heterogeneity_vectors)

    pca = PCA(n_components=min(10, H_scaled.shape[1]))
    coords = pca.fit_transform(H_scaled)

    explained = pca.explained_variance_ratio_
    cumulative = np.cumsum(explained)
    print(f"  PC1 explains {explained[0]*100:.1f}% variance")
    print(f"  PC1+PC2 explain {cumulative[1]*100:.1f}% variance")
    print(f"  Components for 80% variance: {np.searchsorted(cumulative, 0.80)+1}")

    # Save loadings
    loadings_df = pd.DataFrame(
        pca.components_.T,
        index=feature_names,
        columns=[f"PC{i+1}" for i in range(pca.n_components_)]
    )
    loadings_df.to_csv(os.path.join(output_dir, "heterogeneity_pca_components.csv"))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Scree plot
    axes[0].bar(range(1, len(explained)+1), explained*100,
                color="steelblue", alpha=0.8, label="Individual")
    axes[0].step(range(1, len(cumulative)+1), cumulative*100,
                 color="tomato", linewidth=2, label="Cumulative")
    axes[0].axhline(80, color="gray", linestyle="--", linewidth=1, alpha=0.7)
    axes[0].set_xlabel("Principal Component", fontsize=10)
    axes[0].set_ylabel("Explained Variance (%)", fontsize=10)
    axes[0].set_title("PCA Scree Plot — Heterogeneity Space", fontsize=11)
    axes[0].legend(fontsize=9)
    axes[0].set_xticks(range(1, len(explained)+1))

    # PC1 vs PC2 scatter
    axes[1].scatter(coords[:, 0], coords[:, 1],
                    alpha=0.5, s=20, color="mediumpurple", edgecolors="none")
    axes[1].set_xlabel(f"PC1 ({explained[0]*100:.1f}%)", fontsize=10)
    axes[1].set_ylabel(f"PC2 ({explained[1]*100:.1f}%)", fontsize=10)
    axes[1].set_title("PDAC Cases in Heterogeneity PCA Space\n(each point = one tumor)", fontsize=11)

    plt.suptitle("PCA of Intra-Tumor Heterogeneity Vectors", fontsize=12, fontweight="bold")
    plt.tight_layout()
    save_fig(fig, os.path.join(output_dir, "plots", "04_pca_heterogeneity_space.png"))

    return pca, coords


def correlate_with_volume(magnitudes, tumor_volumes, case_ids, output_dir):
    """
    Tests whether heterogeneity magnitude correlates with tumor size.
    Tumor volume = voxel count × (1mm)^3 = voxel count mm^3.
    """
    print("\n── 5. Correlation: heterogeneity magnitude vs. tumor volume ─────────")
    r_pearson,  p_pearson  = pearsonr(magnitudes,  tumor_volumes)
    r_spearman, p_spearman = spearmanr(magnitudes, tumor_volumes)

    print(f"  Pearson  r={r_pearson:.3f}  p={p_pearson:.4f}")
    print(f"  Spearman r={r_spearman:.3f}  p={p_spearman:.4f}")

    df_corr = pd.DataFrame({
        "metric"   : ["Pearson", "Spearman"],
        "r"        : [r_pearson, r_spearman],
        "p_value"  : [p_pearson, p_spearman],
        "n_cases"  : [len(magnitudes), len(magnitudes)],
    })
    df_corr.to_csv(os.path.join(output_dir,
                                "heterogeneity_correlation_with_volume.csv"), index=False)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(tumor_volumes / 1000, magnitudes,     # convert mm³ → cm³ for readability
               alpha=0.45, s=18, color="steelblue", edgecolors="none")

    # Regression line
    m, b = np.polyfit(tumor_volumes / 1000, magnitudes, 1)
    x_line = np.linspace((tumor_volumes/1000).min(), (tumor_volumes/1000).max(), 200)
    ax.plot(x_line, m * x_line + b, color="tomato", linewidth=2,
            label=f"Pearson r={r_pearson:.3f}, p={p_pearson:.3f}\n"
                  f"Spearman r={r_spearman:.3f}, p={p_spearman:.3f}")

    ax.set_xlabel("Tumor Volume (cm³)", fontsize=11)
    ax.set_ylabel("Heterogeneity Magnitude (L2 norm)", fontsize=11)
    ax.set_title("Intra-Tumor Heterogeneity vs. Tumor Size\n(PDAC cases)", fontsize=12)
    ax.legend(fontsize=9)
    plt.tight_layout()
    save_fig(fig, os.path.join(output_dir, "plots",
                               "05_correlation_with_volume.png"))


def plot_feature_heatmap(central, peripheral, feature_names, output_dir):
    """
    Heatmap of mean central and mean peripheral feature values side by side,
    making it easy to see which features show the largest spatial shift.
    """
    print("\n── 6. Feature mean heatmap ──────────────────────────────────────────")
    mean_central    = central.mean(axis=0)
    mean_peripheral = peripheral.mean(axis=0)
    mean_diff       = mean_central - mean_peripheral

    df_heat = pd.DataFrame({
        "Central"    : mean_central,
        "Peripheral" : mean_peripheral,
        "Difference" : mean_diff,
    }, index=feature_names)

    fig, axes = plt.subplots(1, 3, figsize=(16, 10))
    titles = ["Mean Central", "Mean Peripheral", "Difference (C − P)"]
    cmaps  = ["Blues", "Oranges", "coolwarm"]

    for ax, col, title, cmap in zip(axes, ["Central", "Peripheral", "Difference"],
                                     titles, cmaps):
        vals = df_heat[[col]].values
        center = 0 if col == "Difference" else None
        sns.heatmap(vals, ax=ax, cmap=cmap, center=center,
                    yticklabels=feature_names, xticklabels=[col],
                    annot=True, fmt=".3f", annot_kws={"size": 7},
                    cbar_kws={"shrink": 0.6})
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.tick_params(axis="y", labelsize=7)

    plt.suptitle("Mean CoLIAGe Feature Values: Central vs. Peripheral\n(PDAC cohort)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    save_fig(fig, os.path.join(output_dir, "plots",
                               "06_feature_heatmap_central_vs_peripheral.png"))


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Descriptive heterogeneity analysis of central vs. peripheral "
                    "CoLIAGe features across PDAC cases."
    )
    parser.add_argument("--aggregated_dir", required=True,
                        help="Directory containing central_vectors.npy, "
                             "peripheral_vectors.npy, heterogeneity_vectors.npy, "
                             "case_ids.npy (and optionally tumor_volumes.npy).")
    parser.add_argument("--output_dir", required=True,
                        help="Where to save CSVs and plots.")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "plots"), exist_ok=True)

    # ── Load aggregated arrays ────────────────────────────────────────────────
    print("Loading aggregated arrays...")
    central    = np.load(os.path.join(args.aggregated_dir, "central_vectors.npy"))
    peripheral = np.load(os.path.join(args.aggregated_dir, "peripheral_vectors.npy"))
    heterog    = np.load(os.path.join(args.aggregated_dir, "heterogeneity_vectors.npy"))
    case_ids   = np.load(os.path.join(args.aggregated_dir, "case_ids.npy"),
                         allow_pickle=True)

    assert central.shape == peripheral.shape == heterog.shape, \
        "central / peripheral / heterogeneity arrays must have the same shape."

    # Drop CT intensity channel (index 0) if present — keep only the 26 Haralick channels.
    # The pipeline saves channel 0 as CT intensity; heterogeneity analysis uses texture only.
    if central.shape[1] == 27:
        print("  Detected 27 features — dropping CT intensity channel (index 0), "
              "keeping 26 Haralick channels.")
        central    = central[:, 1:]
        peripheral = peripheral[:, 1:]
        heterog    = heterog[:, 1:]

    assert central.shape[1] == 26, \
        f"Expected 26 Haralick features, got {central.shape[1]}."
    assert len(case_ids) == central.shape[0], \
        "case_ids length must match number of cases."

    n_cases = central.shape[0]
    print(f"  {n_cases} PDAC cases, {central.shape[1]} features each.")

    # NaN check
    for name, arr in [("central", central), ("peripheral", peripheral),
                      ("heterogeneity", heterog)]:
        n_nan = int(np.isnan(arr).sum())
        if n_nan > 0:
            print(f"  WARNING: {n_nan} NaN values in {name} array — "
                  f"these cases will be excluded from affected tests.")

    # Optional tumor volumes
    vol_path = os.path.join(args.aggregated_dir, "tumor_volumes.npy")
    tumor_volumes = None
    if os.path.exists(vol_path):
        tumor_volumes = np.load(vol_path).astype(np.float32)
        assert len(tumor_volumes) == n_cases, \
            "tumor_volumes.npy must have the same number of entries as case_ids."
        print(f"  Tumor volumes loaded: mean={tumor_volumes.mean():.0f} mm³ "
              f"({tumor_volumes.mean()/1000:.1f} cm³)")
    else:
        print("  tumor_volumes.npy not found — skipping volume correlation analysis.")
        print("  To enable: save an (N,) float32 array of per-case mask voxel counts "
              "to aggregated_dir/tumor_volumes.npy before running this script.")

    # ── Run all analyses ──────────────────────────────────────────────────────
    df_tests = run_paired_ttests(central, peripheral, FEATURE_NAMES, args.output_dir)
    plot_feature_distributions(central, peripheral, FEATURE_NAMES, args.output_dir)
    magnitudes = analyze_heterogeneity_magnitude(heterog, args.output_dir)
    pca, coords = run_pca(heterog, FEATURE_NAMES, args.output_dir)
    if tumor_volumes is not None:
        correlate_with_volume(magnitudes, tumor_volumes, case_ids, args.output_dir)
    plot_feature_heatmap(central, peripheral, FEATURE_NAMES, args.output_dir)

    # ── Final summary ─────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("HETEROGENEITY ANALYSIS COMPLETE")
    print("=" * 60)
    print(f"  Cases analyzed          : {n_cases}")
    n_sig = int(df_tests["significant_ttest"].sum())
    print(f"  Significantly different features (Bonferroni, α=0.05): {n_sig}/26")
    print(f"  Mean heterogeneity magnitude : {magnitudes.mean():.4f}")
    print(f"  Results saved to: {args.output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
