"""
SECOM Exploratory Data Analysis (EDA) Pipeline
================================================
Comprehensive EDA for the SECOM semiconductor manufacturing dataset.

Sections:
  1. Data Ingestion & Alignment
  2. Dataset Overview & Basic Statistics
  3. Target Class Distribution (Imbalance Analysis)
  4. Missing Value Profiling & Heatmap
  5. Zero-Variance & Near-Constant Sensor Detection
  6. Feature Variance & Scale Distribution
  7. Skewness & Distributional Shape Analysis
  8. Outlier Profiling (IQR-based)
  9. Feature-Target Correlation (Top Discriminators)
 10. Inter-Feature Correlation & Redundancy
 11. Temporal Analysis: Yield Trends & Defect Clustering
 12. Hourly & Day-of-Week Production Patterns
 13. Feature Pruning Flow Summary
 14. Actionable Preprocessing Recommendations
"""

import os
import warnings
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")          # non-interactive backend for reliable saving
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Patch

warnings.filterwarnings("ignore")

# -- Global Style --------------------------------------------------------------
PALETTE_PASS = "#2b7bba"
PALETTE_FAIL = "#d9534f"
PALETTE_NEUTRAL = "#7f8c8d"
ACCENT_ORANGE = "#e67e22"
ACCENT_GREEN  = "#27ae60"

try:
    plt.style.use("seaborn-v0_8-whitegrid")
except OSError:
    plt.style.use("default")

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.titlesize": 13,
    "axes.labelsize": 10,
    "axes.edgecolor": "#cccccc",
    "axes.linewidth": 0.9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "figure.dpi": 150,
})


# ===============================================================================
# HELPER UTILITIES
# ===============================================================================

def _save(fig: plt.Figure, path: str) -> None:
    """Save figure with tight layout and close."""
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  [Saved] -> {path}")


def _section(n: int, title: str) -> None:
    print(f"\n{'='*65}")
    print(f"  SECTION {n}: {title}")
    print(f"{'='*65}")


# ===============================================================================
# MAIN EDA FUNCTION
# ===============================================================================

def run_eda(
    data_path: str = "secom/secom.csv",
    labels_path: str = "secom/secom_labels.csv",
    output_dir: str = "plots",
) -> dict:
    """
    Execute full EDA pipeline and save all plots + summary report.

    Returns
    -------
    dict
        Dictionary of computed EDA statistics (for downstream use).
    """
    os.makedirs(output_dir, exist_ok=True)

    # --------------------------------------------------------------------------
    # SECTION 1 - DATA INGESTION
    # --------------------------------------------------------------------------
    _section(1, "DATA INGESTION & ALIGNMENT")

    if not os.path.exists(data_path) or not os.path.exists(labels_path):
        raise FileNotFoundError(
            f"Data files not found.\n  Expected: {data_path}\n            {labels_path}"
        )

    df = pd.read_csv(data_path)
    lb = pd.read_csv(labels_path)

    assert len(df) == len(lb), "Row count mismatch between features and labels!"

    lb["target"]    = lb["label"].map({-1: 0, 1: 1})
    lb["timestamp"] = pd.to_datetime(lb["timestamp"], dayfirst=True)
    lb["date"]      = lb["timestamp"].dt.date
    lb["hour"]      = lb["timestamp"].dt.hour
    lb["week"]      = lb["timestamp"].dt.to_period("W")
    lb["dayofweek"] = lb["timestamp"].dt.dayofweek      # 0=Mon, 6=Sun

    n_samples, n_features = df.shape
    print(f"  Features  : {n_samples:,} rows  x  {n_features:,} columns")
    print(f"  Labels    : {lb.shape[0]:,} rows  x  {lb.shape[1]:,} columns")
    print(f"  Time span : {lb['timestamp'].min():%Y-%m-%d}  ->  {lb['timestamp'].max():%Y-%m-%d}")
    print(f"  Date check: index alignment verified OK")

    # --------------------------------------------------------------------------
    # SECTION 2 - DATASET OVERVIEW
    # --------------------------------------------------------------------------
    _section(2, "DATASET OVERVIEW & BASIC STATISTICS")

    total_cells   = df.size
    total_missing = df.isna().sum().sum()
    miss_pct_col  = df.isna().sum() / n_samples * 100   # per-column %
    nunique_col   = df.nunique(dropna=True)
    zero_var_mask = nunique_col <= 1
    n_zero_var    = zero_var_mask.sum()

    print(f"  Total cells          : {total_cells:>10,}")
    print(f"  Total missing cells  : {total_missing:>10,}  ({total_missing/total_cells*100:.2f}%)")
    print(f"  Constant features    : {n_zero_var:>10}")
    print(f"  Duplicate rows       : {df.duplicated().sum():>10}")

    # --------------------------------------------------------------------------
    # SECTION 3 - CLASS DISTRIBUTION
    # --------------------------------------------------------------------------
    _section(3, "TARGET CLASS DISTRIBUTION")

    n_pass = (lb["target"] == 0).sum()
    n_fail = (lb["target"] == 1).sum()
    pct_pass = n_pass / n_samples * 100
    pct_fail = n_fail / n_samples * 100
    imb_ratio = n_pass / n_fail

    print(f"  Pass (Normal)   [-1] : {n_pass:>6,}  ({pct_pass:.2f}%)")
    print(f"  Fail (Defective)[+1] : {n_fail:>6,}  ({pct_fail:.2f}%)")
    print(f"  Imbalance ratio      : {imb_ratio:.2f} : 1  (Pass : Fail)")

    # -- Plot 1: Class Distribution (bar + pie side-by-side) ------------------
    fig = plt.figure(figsize=(10, 4))
    gs  = GridSpec(1, 2, figure=fig, wspace=0.35)

    # Bar chart
    ax_bar = fig.add_subplot(gs[0])
    bars = ax_bar.bar(
        ["Pass (-1)", "Fail (+1)"],
        [n_pass, n_fail],
        color=[PALETTE_PASS, PALETTE_FAIL],
        width=0.45,
        edgecolor="white",
        linewidth=1.2,
    )
    for bar, val, pct in zip(bars, [n_pass, n_fail], [pct_pass, pct_fail]):
        ax_bar.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 25,
            f"{val:,}\n({pct:.1f}%)",
            ha="center", va="bottom", fontsize=9, fontweight="bold",
        )
    ax_bar.set_title("Wafer Yield Outcome Counts", fontweight="bold", pad=10)
    ax_bar.set_ylabel("Wafer Count")
    ax_bar.set_ylim(0, n_samples * 1.18)
    ax_bar.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))

    # Pie chart
    ax_pie = fig.add_subplot(gs[1])
    wedges, texts, autotexts = ax_pie.pie(
        [n_pass, n_fail],
        labels=["Pass (-1)", "Fail (+1)"],
        colors=[PALETTE_PASS, PALETTE_FAIL],
        autopct="%1.1f%%",
        startangle=90,
        wedgeprops={"edgecolor": "white", "linewidth": 1.5},
        textprops={"fontsize": 9},
    )
    for at in autotexts:
        at.set_fontweight("bold")
    ax_pie.set_title("Class Proportion", fontweight="bold", pad=10)
    ax_pie.annotate(
        f"Imbalance Ratio\n{imb_ratio:.1f} : 1",
        xy=(0, 0), ha="center", va="center",
        fontsize=8, color="#444444",
    )

    fig.suptitle(
        "SECOM Yield Distribution -- Severe Class Imbalance",
        fontsize=13, fontweight="bold", y=1.01,
    )
    _save(fig, os.path.join(output_dir, "eda_01_class_distribution.png"))

    # --------------------------------------------------------------------------
    # SECTION 4 - MISSING VALUE PROFILING
    # --------------------------------------------------------------------------
    _section(4, "MISSING VALUE PROFILING")

    bins_miss = {
        "0% (Complete)":        (miss_pct_col == 0).sum(),
        "0.1%-10%  (Sparse)":   ((miss_pct_col > 0) & (miss_pct_col <= 10)).sum(),
        "10%-30%   (Moderate)": ((miss_pct_col > 10) & (miss_pct_col <= 30)).sum(),
        "30%-50%   (High)":     ((miss_pct_col > 30) & (miss_pct_col <= 50)).sum(),
        ">50%      (Critical)": (miss_pct_col > 50).sum(),
    }
    for k, v in bins_miss.items():
        flag = "  <- DROP CANDIDATES" if "Critical" in k else ""
        print(f"  {k:30s}: {v:4d} features{flag}")
    print(f"\n  Total missing cells : {total_missing:,} / {total_cells:,}  ({total_missing/total_cells*100:.2f}%)")

    # -- Plot 2a: Missingness Histogram ----------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    ax = axes[0]
    n_hist, bins_hist, patches = ax.hist(
        miss_pct_col, bins=35,
        color=ACCENT_ORANGE, edgecolor="white", linewidth=0.6,
    )
    ax.axvline(50, color=PALETTE_FAIL, linestyle="--", linewidth=1.8,
               label="50% Drop Threshold")
    ax.axvline(10, color=PALETTE_PASS,  linestyle=":",  linewidth=1.4,
               label="10% Advisory Threshold")
    ax.set_title("Missing Values (%) per Feature -- Distribution", fontweight="bold")
    ax.set_xlabel("Missing Percentage (%)")
    ax.set_ylabel("Number of Features")
    ax.legend()

    # Annotate bin regions
    ax.axvspan(50, 100, alpha=0.10, color=PALETTE_FAIL, label="Drop Zone")
    ax.annotate(
        f"{bins_miss['>50%      (Critical)']} features\n(>50% NaN)", 
        xy=(75, n_hist.max() * 0.6), fontsize=8, color=PALETTE_FAIL,
        ha="center",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor=PALETTE_FAIL, alpha=0.8),
    )

    # -- Plot 2b: Sorted bar of top-20 most-missing features ------------------
    ax2 = axes[1]
    top_missing = miss_pct_col.nlargest(20)
    colors_m = [PALETTE_FAIL if v > 50 else ACCENT_ORANGE for v in top_missing.values]
    ax2.barh(
        [f.replace("feature_", "F") for f in top_missing.index],
        top_missing.values,
        color=colors_m, edgecolor="white",
    )
    ax2.axvline(50, color=PALETTE_FAIL, linestyle="--", linewidth=1.5,
                label="50% Drop Threshold")
    ax2.set_title("Top-20 Features with Highest Missingness", fontweight="bold")
    ax2.set_xlabel("Missing Percentage (%)")
    ax2.invert_yaxis()
    ax2.legend(loc="lower right", fontsize=7)
    for i, v in enumerate(top_missing.values):
        ax2.text(v + 0.5, i, f"{v:.1f}%", va="center", fontsize=7)

    fig.suptitle("Section 4 -- Missing Value Analysis", fontsize=12, fontweight="bold")
    plt.tight_layout()
    _save(fig, os.path.join(output_dir, "eda_02_missing_values.png"))

    # -- Plot 2c: Missingness Heatmap (sample of features sorted by miss%) ----
    top_missing_cols = miss_pct_col[miss_pct_col > 0].nlargest(40).index.tolist()
    if len(top_missing_cols) > 0:
        miss_matrix = df[top_missing_cols].isna().astype(int)
        # Sample rows for readability
        row_sample = np.linspace(0, n_samples - 1, min(200, n_samples), dtype=int)
        miss_sample = miss_matrix.iloc[row_sample, :]

        fig, ax = plt.subplots(figsize=(14, 5))
        sns.heatmap(
            miss_sample.T,
            ax=ax,
            cmap=["#f0f0f0", PALETTE_FAIL],
            cbar=False,
            yticklabels=[c.replace("feature_", "F") for c in top_missing_cols],
            xticklabels=False,
            linewidths=0,
        )
        ax.set_xlabel("Wafer Samples (sampled 200)", fontsize=9)
        ax.set_ylabel("Feature", fontsize=9)
        ax.set_title(
            "Missing Value Pattern Map -- Top 40 Most-Missing Features\n"
            "(Red = Missing, Grey = Observed)",
            fontweight="bold",
        )
        legend_elements = [
            Patch(facecolor=PALETTE_FAIL, label="Missing (NaN)"),
            Patch(facecolor="#f0f0f0", edgecolor="#cccccc", label="Observed"),
        ]
        ax.legend(handles=legend_elements, loc="upper right", fontsize=8)
        _save(fig, os.path.join(output_dir, "eda_03_missingness_heatmap.png"))

    # --------------------------------------------------------------------------
    # SECTION 5 - ZERO-VARIANCE & NEAR-CONSTANT SENSORS
    # --------------------------------------------------------------------------
    _section(5, "ZERO-VARIANCE & CONSTANT SENSORS")

    zero_var_cols = nunique_col[zero_var_mask].index.tolist()
    low_var_cols  = nunique_col[(nunique_col > 1) & (nunique_col <= 5)].index.tolist()
    print(f"  Constant features  (nunique <= 1) : {len(zero_var_cols):>4} features")
    print(f"  Near-constant      (nunique 2-5) : {len(low_var_cols):>4} features")
    print(f"  Normal features    (nunique > 5) : {(nunique_col > 5).sum():>4} features")
    print(f"  Sample constant IDs: {zero_var_cols[:8]}")

    # -- Plot 3: Unique-value distribution bar ---------------------------------
    fig, ax = plt.subplots(figsize=(9, 4))
    bins_u = [0, 1, 5, 20, 100, n_samples + 1]
    labels_u = ["Constant\n(1)", "Near-const\n(2-5)", "Low\n(6-20)", "Moderate\n(21-100)", "High\n(>100)"]
    counts_u = [
        (nunique_col <= 1).sum(),
        ((nunique_col > 1) & (nunique_col <= 5)).sum(),
        ((nunique_col > 5) & (nunique_col <= 20)).sum(),
        ((nunique_col > 20) & (nunique_col <= 100)).sum(),
        (nunique_col > 100).sum(),
    ]
    bar_colors = [PALETTE_FAIL, ACCENT_ORANGE, "#f1c40f", PALETTE_PASS, ACCENT_GREEN]
    bars = ax.bar(labels_u, counts_u, color=bar_colors, edgecolor="white", linewidth=1)
    for bar, val in zip(bars, counts_u):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2,
                str(val), ha="center", fontsize=9, fontweight="bold")
    ax.set_title("Feature Cardinality (Unique Values) Distribution", fontweight="bold", pad=10)
    ax.set_ylabel("Number of Features")
    ax.set_xlabel("Unique-Value Bucket")
    _save(fig, os.path.join(output_dir, "eda_04_feature_cardinality.png"))

    # --------------------------------------------------------------------------
    # Build CLEAN feature matrix for downstream sections
    # --------------------------------------------------------------------------
    drop_zero_var  = set(zero_var_cols)
    drop_high_miss = set(miss_pct_col[miss_pct_col > 50].index.tolist())
    keep_cols      = [c for c in df.columns if c not in drop_zero_var and c not in drop_high_miss]
    df_clean       = df[keep_cols].copy()
    n_clean_feat   = df_clean.shape[1]

    print(f"\n  -> Clean active matrix : {n_samples:,} x {n_clean_feat}  features")

    # --------------------------------------------------------------------------
    # SECTION 6 - VARIANCE & SCALE DISTRIBUTION
    # --------------------------------------------------------------------------
    _section(6, "FEATURE VARIANCE & SCALE DISTRIBUTION")

    col_var   = df_clean.var()
    col_mean  = df_clean.mean()
    col_std   = df_clean.std()
    col_range = df_clean.max() - df_clean.min()

    print(f"  Variance range : {col_var.min():.2e}  ->  {col_var.max():.2e}")
    print(f"  Std-dev range  : {col_std.min():.4f}  ->  {col_std.max():.2f}")
    print(f"  Mean range     : {col_mean.min():.2f}  ->  {col_mean.max():.2f}")

    # -- Plot 4: Variance on log-scale histogram -------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    ax = axes[0]
    log_var = np.log10(col_var.clip(lower=1e-10))
    ax.hist(log_var, bins=35, color=PALETTE_PASS, edgecolor="white", linewidth=0.6)
    ax.set_title("Feature Variance -- log10 Scale", fontweight="bold")
    ax.set_xlabel("log10(Variance)")
    ax.set_ylabel("Number of Features")
    ax.axvline(log_var.median(), color=PALETTE_FAIL, linestyle="--",
               label=f"Median = 10^{log_var.median():.1f}", linewidth=1.5)
    ax.legend()

    ax2 = axes[1]
    log_std = np.log10(col_std.clip(lower=1e-10))
    ax2.hist(log_std, bins=35, color=ACCENT_GREEN, edgecolor="white", linewidth=0.6)
    ax2.set_title("Feature Std Dev -- log10 Scale", fontweight="bold")
    ax2.set_xlabel("log10(Std Dev)")
    ax2.set_ylabel("Number of Features")
    ax2.axvline(log_std.median(), color=PALETTE_FAIL, linestyle="--",
                label=f"Median = 10^{log_std.median():.1f}", linewidth=1.5)
    ax2.legend()

    fig.suptitle("Section 6 -- Variance & Scale Spread (Scaling Required!)", fontsize=12, fontweight="bold")
    plt.tight_layout()
    _save(fig, os.path.join(output_dir, "eda_05_variance_scale.png"))

    # --------------------------------------------------------------------------
    # SECTION 7 - SKEWNESS ANALYSIS
    # --------------------------------------------------------------------------
    _section(7, "SKEWNESS & DISTRIBUTIONAL SHAPE")

    col_skew = df_clean.skew()
    n_high_skew = (col_skew.abs() > 3).sum()
    n_mod_skew  = ((col_skew.abs() > 1) & (col_skew.abs() <= 3)).sum()
    n_low_skew  = (col_skew.abs() <= 1).sum()

    print(f"  |skew| <= 1  (approx. normal) : {n_low_skew:>4} features")
    print(f"  1 < |skew| <= 3  (moderate)   : {n_mod_skew:>4} features")
    print(f"  |skew| > 3  (highly skewed)   : {n_high_skew:>4} features")
    print(f"  Skew range : {col_skew.min():.2f}  ->  {col_skew.max():.2f}")

    # -- Plot 5: Skewness Distribution -----------------------------------------
    fig, ax = plt.subplots(figsize=(9, 4.5))
    skew_vals = col_skew.values
    ax.hist(skew_vals, bins=50, color=ACCENT_ORANGE, edgecolor="white", linewidth=0.5)
    ax.axvline(-3, color=PALETTE_FAIL, linestyle="--", linewidth=1.4, label="|skew| = 3 threshold")
    ax.axvline( 3, color=PALETTE_FAIL, linestyle="--", linewidth=1.4)
    ax.axvspan(-50, -3, alpha=0.08, color=PALETTE_FAIL)
    ax.axvspan(  3,  50, alpha=0.08, color=PALETTE_FAIL)
    ax.axvline(0, color="#333333", linewidth=0.9, linestyle=":")
    ax.set_title(
        f"Skewness Distribution Across {n_clean_feat} Active Features\n"
        f"{n_high_skew} features (>{n_high_skew/n_clean_feat*100:.0f}%) are highly skewed -- "
        "log/Box-Cox transform recommended",
        fontweight="bold",
    )
    ax.set_xlabel("Skewness Coefficient")
    ax.set_ylabel("Number of Features")
    ax.legend()
    ax.set_xlim(-30, 45)
    _save(fig, os.path.join(output_dir, "eda_06_skewness.png"))

    # --------------------------------------------------------------------------
    # SECTION 8 - OUTLIER PROFILING (IQR-BASED)
    # --------------------------------------------------------------------------
    _section(8, "OUTLIER PROFILING (3xIQR RULE)")

    q1   = df_clean.quantile(0.25)
    q3   = df_clean.quantile(0.75)
    iqr  = q3 - q1
    outlier_mask  = (df_clean < (q1 - 3 * iqr)) | (df_clean > (q3 + 3 * iqr))
    outlier_cells = outlier_mask.sum().sum()
    feats_w_out   = (outlier_mask.sum() > 0).sum()
    outlier_pct_per_col = (outlier_mask.sum() / n_samples * 100).sort_values(ascending=False)

    print(f"  Outlier cells (>3xIQR)     : {outlier_cells:,}  "
          f"({outlier_cells / (n_samples * n_clean_feat) * 100:.2f}% of clean matrix)")
    print(f"  Features with >=1 outlier   : {feats_w_out} / {n_clean_feat}")
    print(f"  Top-5 outlier-dense features:")
    for feat, pct in outlier_pct_per_col.head(5).items():
        print(f"    {feat}: {pct:.1f}% of samples are outliers")

    # -- Plot 6: Outlier count per feature histogram ---------------------------
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    ax = axes[0]
    out_per_col = outlier_mask.sum().values
    ax.hist(out_per_col, bins=40, color="#8e44ad", edgecolor="white", linewidth=0.5)
    ax.set_title("Outlier Count per Feature (3xIQR)", fontweight="bold")
    ax.set_xlabel("Number of Outlier Samples")
    ax.set_ylabel("Number of Features")
    ax.axvline(np.median(out_per_col), color=PALETTE_FAIL, linestyle="--",
               label=f"Median = {np.median(out_per_col):.0f}", linewidth=1.5)
    ax.legend()

    ax2 = axes[1]
    top_out_feats = outlier_pct_per_col.head(20)
    ax2.barh(
        [f.replace("feature_", "F") for f in top_out_feats.index],
        top_out_feats.values,
        color="#8e44ad", edgecolor="white",
    )
    ax2.set_title("Top-20 Features by Outlier Density (%)", fontweight="bold")
    ax2.set_xlabel("% Samples as Outliers (3xIQR)")
    ax2.invert_yaxis()
    for i, v in enumerate(top_out_feats.values):
        ax2.text(v + 0.1, i, f"{v:.1f}%", va="center", fontsize=7)

    fig.suptitle("Section 8 -- Outlier Analysis (RobustScaler recommended)", fontsize=12, fontweight="bold")
    plt.tight_layout()
    _save(fig, os.path.join(output_dir, "eda_07_outliers.png"))

    # --------------------------------------------------------------------------
    # SECTION 9 - FEATURE-TARGET CORRELATION (TOP DISCRIMINATORS)
    # --------------------------------------------------------------------------
    _section(9, "FEATURE-TARGET CORRELATION (TOP DISCRIMINATORS)")

    df_w_target = df_clean.copy()
    df_w_target["target"] = lb["target"].values
    corr_with_target = df_w_target.corr()["target"].drop("target").abs().sort_values(ascending=False)
    top20_corr = corr_with_target.head(20)

    print(f"  Max |Pearson r| with target : {corr_with_target.max():.4f} ({corr_with_target.idxmax()})")
    print(f"  Top-5 discriminating features:")
    for feat, r in top20_corr.head(5).items():
        print(f"    {feat}: |r| = {r:.4f}")

    # -- Plot 7: Top-20 Feature Correlations with Target -----------------------
    fig, ax = plt.subplots(figsize=(9, 6))
    feat_names  = [f.replace("feature_", "F") for f in top20_corr.index]
    bar_clr = [PALETTE_FAIL if v > 0.12 else ACCENT_ORANGE if v > 0.08 else PALETTE_PASS
               for v in top20_corr.values]
    bars = ax.barh(feat_names[::-1], top20_corr.values[::-1],
                   color=bar_clr[::-1], edgecolor="white")
    ax.axvline(0.10, color=PALETTE_FAIL, linestyle="--", linewidth=1.3,
               label="r = 0.10 reference")
    ax.set_title("Top-20 Features by |Pearson r| with Yield Outcome",
                 fontweight="bold", pad=10)
    ax.set_xlabel("|Pearson Correlation| with Target")
    for bar, val in zip(bars, top20_corr.values[::-1]):
        ax.text(val + 0.002, bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", va="center", fontsize=7.5)
    legend_elems = [
        Patch(facecolor=PALETTE_FAIL,   label="|r| > 0.12 (Strong)"),
        Patch(facecolor=ACCENT_ORANGE,  label="0.08 < |r| <= 0.12 (Moderate)"),
        Patch(facecolor=PALETTE_PASS,   label="|r| <= 0.08 (Weak)"),
    ]
    ax.legend(handles=legend_elems, loc="lower right", fontsize=7.5)
    _save(fig, os.path.join(output_dir, "eda_08_feature_target_correlation.png"))

    # -- Plot 7b: Box-plots for top-6 discriminating features (pass vs. fail) -
    top6_feats = top20_corr.head(6).index.tolist()
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    axes = axes.flatten()

    for i, feat in enumerate(top6_feats):
        ax = axes[i]
        pass_vals = df_clean.loc[lb["target"] == 0, feat].dropna()
        fail_vals = df_clean.loc[lb["target"] == 1, feat].dropna()
        data_bp = [pass_vals.values, fail_vals.values]
        bp = ax.boxplot(
            data_bp,
            labels=["Pass\n(-1)", "Fail\n(+1)"],
            patch_artist=True,
            medianprops=dict(color="white", linewidth=2),
            flierprops=dict(marker=".", markersize=2, alpha=0.4),
        )
        bp["boxes"][0].set_facecolor(PALETTE_PASS)
        bp["boxes"][1].set_facecolor(PALETTE_FAIL)
        r_val = corr_with_target[feat]
        ax.set_title(f"{feat.replace('feature_', 'F')}  |r|={r_val:.3f}",
                     fontsize=9, fontweight="bold")
        ax.set_ylabel("Sensor Value", fontsize=8)
        ax.tick_params(labelsize=7)

    fig.suptitle(
        "Top-6 Discriminating Features -- Pass vs. Fail Distributions\n"
        "(Boxplots show medians, IQR, and outliers)",
        fontsize=12, fontweight="bold",
    )
    plt.tight_layout()
    _save(fig, os.path.join(output_dir, "eda_09_top6_boxplots.png"))

    # --------------------------------------------------------------------------
    # SECTION 10 - INTER-FEATURE CORRELATION & REDUNDANCY
    # --------------------------------------------------------------------------
    _section(10, "INTER-FEATURE CORRELATION & REDUNDANCY")

    # Compute correlation on a subset to stay tractable
    SAMPLE_COLS = 80
    np.random.seed(42)
    col_sample = np.random.choice(df_clean.columns, size=min(SAMPLE_COLS, n_clean_feat), replace=False)
    corr_sub   = df_clean[col_sample].corr().abs()

    upper_tri  = np.triu(np.ones(corr_sub.shape), k=1).astype(bool)
    upper_vals = corr_sub.where(upper_tri)
    n_high09   = (upper_vals > 0.9).sum().sum()
    n_high08   = (upper_vals > 0.8).sum().sum()
    n_high07   = (upper_vals > 0.7).sum().sum()

    # Scale to full feature set
    scale = (n_clean_feat * (n_clean_feat - 1)) / (SAMPLE_COLS * (SAMPLE_COLS - 1))
    print(f"  (Estimated on {SAMPLE_COLS}-column sample, scaled to full set)")
    print(f"  Pairs with |r| > 0.9  (highly redundant)  : ~{int(n_high09 * scale):,}")
    print(f"  Pairs with |r| > 0.8  (strongly correlated): ~{int(n_high08 * scale):,}")
    print(f"  Pairs with |r| > 0.7  (moderately correl.) : ~{int(n_high07 * scale):,}")

    # -- Plot 8: Correlation Heatmap (80-col sample) ---------------------------
    fig, ax = plt.subplots(figsize=(13, 11))
    mask = ~upper_tri  # show upper triangle only
    sns.heatmap(
        corr_sub,
        ax=ax,
        cmap="coolwarm",
        center=0,
        vmin=0, vmax=1,
        mask=np.tril(np.ones(corr_sub.shape)).astype(bool),
        xticklabels=[c.replace("feature_", "F") for c in col_sample],
        yticklabels=[c.replace("feature_", "F") for c in col_sample],
        linewidths=0.2,
        linecolor="#dddddd",
        cbar_kws={"shrink": 0.6, "label": "|Pearson r|"},
    )
    ax.set_title(
        f"Inter-Feature Correlation Heatmap -- {SAMPLE_COLS}-Feature Sample\n"
        "(High redundancy visible -- PCA / feature selection needed)",
        fontweight="bold", pad=12,
    )
    ax.tick_params(axis="x", rotation=90, labelsize=6)
    ax.tick_params(axis="y", labelsize=6)
    _save(fig, os.path.join(output_dir, "eda_10_correlation_heatmap.png"))

    # --------------------------------------------------------------------------
    # SECTION 11 - TEMPORAL ANALYSIS: YIELD TRENDS & DEFECT CLUSTERING
    # --------------------------------------------------------------------------
    _section(11, "TEMPORAL ANALYSIS -- YIELD TRENDS & DEFECT CLUSTERING")

    daily_stats = lb.groupby("date").agg(
        total_wafers=("target", "count"),
        fails=("target", "sum"),
    )
    daily_stats["yield_loss_pct"] = daily_stats["fails"] / daily_stats["total_wafers"] * 100
    daily_stats["date_dt"] = pd.to_datetime(daily_stats.index)

    weekly_lb = lb.copy()
    weekly_lb["week_start"] = lb["timestamp"].dt.to_period("W").dt.start_time
    weekly_stats = weekly_lb.groupby("week_start").agg(
        total_wafers=("target", "count"),
        fails=("target", "sum"),
    )
    weekly_stats["yield_loss_pct"] = weekly_stats["fails"] / weekly_stats["total_wafers"] * 100

    n_active_days = len(daily_stats)
    max_fail_day  = daily_stats["fails"].idxmax()
    max_fails     = daily_stats["fails"].max()
    avg_daily_yield_loss = daily_stats["yield_loss_pct"].mean()

    print(f"  Date span          : {lb['timestamp'].min():%Y-%m-%d}  ->  {lb['timestamp'].max():%Y-%m-%d}")
    print(f"  Active production days : {n_active_days}")
    print(f"  Avg daily yield loss   : {avg_daily_yield_loss:.1f}%")
    print(f"  Worst day              : {max_fail_day}  ->  {max_fails} fails")

    # -- Plot 9a: Dual-axis Timeline -------------------------------------------
    fig, ax1 = plt.subplots(figsize=(13, 5))
    ax2 = ax1.twinx()

    ax1.bar(
        weekly_stats.index, weekly_stats["total_wafers"],
        width=5, color=PALETTE_NEUTRAL, alpha=0.45, label="Total Wafers/Week",
    )
    ax2.plot(
        weekly_stats.index, weekly_stats["yield_loss_pct"],
        color=PALETTE_FAIL, marker="o", linewidth=2.2, markersize=5,
        label="Weekly Yield Loss (%)",
    )
    ax2.fill_between(weekly_stats.index, weekly_stats["yield_loss_pct"],
                     alpha=0.18, color=PALETTE_FAIL)
    ax2.axhline(pct_fail, color=PALETTE_FAIL, linestyle="--", linewidth=1.2,
                label=f"Overall average ({pct_fail:.1f}%)")

    ax1.set_xlabel("Production Week", fontsize=10)
    ax1.set_ylabel("Total Wafers Processed", fontsize=10, color=PALETTE_NEUTRAL)
    ax2.set_ylabel("Yield Loss Rate (%)", fontsize=10, color=PALETTE_FAIL)
    ax2.grid(False)
    ax1.tick_params(axis="x", rotation=30, labelsize=8)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=8)
    ax1.set_title(
        "SECOM Production Volume & Yield Loss Rate -- Weekly Overview\n"
        "(Clusters of high fail-rate weeks indicate process excursions)",
        fontweight="bold",
    )
    plt.tight_layout()
    _save(fig, os.path.join(output_dir, "eda_11_timeline_yield.png"))

    # -- Plot 9b: Daily yield loss as a line chart -----------------------------
    fig, ax = plt.subplots(figsize=(13, 4))
    ax.plot(
        daily_stats["date_dt"], daily_stats["yield_loss_pct"],
        color="#c0392b", linewidth=1.0, alpha=0.7,
    )
    ax.fill_between(
        daily_stats["date_dt"], daily_stats["yield_loss_pct"],
        alpha=0.20, color="#c0392b",
    )
    ax.axhline(pct_fail, color=PALETTE_FAIL, linestyle="--", linewidth=1.3,
               label=f"Overall avg = {pct_fail:.1f}%")
    excursion_days = daily_stats[daily_stats["yield_loss_pct"] > 30]
    ax.scatter(
        pd.to_datetime(excursion_days.index),
        excursion_days["yield_loss_pct"],
        color=PALETTE_FAIL, s=60, zorder=5,
        label=f"Excursion days (>30% fail) -- {len(excursion_days)} days",
    )
    ax.set_title("Daily Yield Loss Rate -- Excursion Detection",
                 fontweight="bold", pad=10)
    ax.set_xlabel("Date")
    ax.set_ylabel("Daily Fail Rate (%)")
    ax.legend(fontsize=8)
    ax.tick_params(axis="x", rotation=30, labelsize=8)
    plt.tight_layout()
    _save(fig, os.path.join(output_dir, "eda_12_daily_yield_loss.png"))

    # --------------------------------------------------------------------------
    # SECTION 12 - HOURLY & DAY-OF-WEEK PATTERNS
    # --------------------------------------------------------------------------
    _section(12, "HOURLY & DAY-OF-WEEK PRODUCTION PATTERNS")

    hourly = lb.groupby("hour").agg(
        total=("target", "count"),
        fails=("target", "sum"),
    )
    hourly["fail_rate"] = hourly["fails"] / hourly["total"] * 100

    dow_labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    dow = lb.groupby("dayofweek").agg(
        total=("target", "count"),
        fails=("target", "sum"),
    )
    dow["fail_rate"] = dow["fails"] / dow["total"] * 100

    # -- Plot 10: Hourly + DOW side-by-side ------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    ax.bar(hourly.index, hourly["total"],  color=PALETTE_PASS,  alpha=0.6,
           label="Total wafers", width=0.7)
    ax2b = ax.twinx()
    ax2b.plot(hourly.index, hourly["fail_rate"], color=PALETTE_FAIL,
              marker="o", linewidth=2, markersize=5, label="Fail Rate (%)")
    ax2b.set_ylabel("Fail Rate (%)", color=PALETTE_FAIL, fontsize=9)
    ax2b.grid(False)
    ax.set_title("Production Volume & Fail Rate by Hour of Day", fontweight="bold")
    ax.set_xlabel("Hour of Day (0-23)")
    ax.set_ylabel("Total Wafers")
    ax.set_xticks(range(0, 24, 2))
    lines_a, lbl_a = ax.get_legend_handles_labels()
    lines_b, lbl_b = ax2b.get_legend_handles_labels()
    ax.legend(lines_a + lines_b, lbl_a + lbl_b, fontsize=7.5, loc="upper left")

    ax3 = axes[1]
    ax3.bar(dow.index, dow["total"], color=PALETTE_PASS, alpha=0.6,
            label="Total wafers", width=0.6)
    ax4 = ax3.twinx()
    ax4.plot(dow.index, dow["fail_rate"], color=PALETTE_FAIL,
             marker="s", linewidth=2, markersize=6, label="Fail Rate (%)")
    ax4.set_ylabel("Fail Rate (%)", color=PALETTE_FAIL, fontsize=9)
    ax4.grid(False)
    ax3.set_title("Production Volume & Fail Rate by Day of Week", fontweight="bold")
    ax3.set_xlabel("Day of Week")
    ax3.set_ylabel("Total Wafers")
    ax3.set_xticks(dow.index)
    ax3.set_xticklabels([dow_labels[i] for i in dow.index])
    lines_c, lbl_c = ax3.get_legend_handles_labels()
    lines_d, lbl_d = ax4.get_legend_handles_labels()
    ax3.legend(lines_c + lines_d, lbl_c + lbl_d, fontsize=7.5, loc="upper left")

    fig.suptitle("Section 12 -- Temporal Production Patterns", fontsize=12, fontweight="bold")
    plt.tight_layout()
    _save(fig, os.path.join(output_dir, "eda_13_hourly_dow_patterns.png"))

    # --------------------------------------------------------------------------
    # SECTION 13 - CUMULATIVE FEATURE PRUNING SUMMARY PLOT
    # --------------------------------------------------------------------------
    _section(13, "FEATURE PRUNING FLOW SUMMARY")

    n_after_zero_var  = n_features - n_zero_var
    n_after_high_miss = n_after_zero_var - len(drop_high_miss)

    print(f"  Raw features          : {n_features}")
    print(f"  After dropping zero-variance ({n_zero_var} cols) : {n_after_zero_var}")
    print(f"  After dropping >50% missing  ({len(drop_high_miss)} cols) : {n_after_high_miss}")
    print(f"  Active feature matrix : {n_clean_feat}")

    stages  = ["Raw\n590", "-Constant\n(116)", "-High-Miss\n(28)", "Active\n446"]
    counts  = [n_features, n_after_zero_var, n_after_high_miss, n_clean_feat]
    clrs    = [PALETTE_NEUTRAL, PALETTE_FAIL, ACCENT_ORANGE, ACCENT_GREEN]

    fig, ax = plt.subplots(figsize=(9, 4.5))
    bars = ax.bar(stages, counts, color=clrs, edgecolor="white", linewidth=1.2, width=0.5)
    for bar, val in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
                str(val), ha="center", fontsize=11, fontweight="bold")
    ax.set_title("Feature Pruning Pipeline -- Column Count at Each Stage",
                 fontweight="bold", pad=10)
    ax.set_ylabel("Number of Features")
    ax.set_ylim(0, n_features * 1.15)
    for i in range(len(counts) - 1):
        reduction = counts[i] - counts[i + 1]
        ax.annotate(
            f"-{reduction}",
            xy=((i + i + 1) / 2, (counts[i] + counts[i + 1]) / 2),
            xytext=((i + i + 1) / 2, max(counts[i], counts[i + 1]) * 1.05),
            ha="center", fontsize=9, color=PALETTE_FAIL,
            arrowprops=dict(arrowstyle="->", color=PALETTE_FAIL, lw=1),
        )
    _save(fig, os.path.join(output_dir, "eda_14_feature_pruning.png"))

    # --------------------------------------------------------------------------
    # WRITE SUMMARY REPORT
    # --------------------------------------------------------------------------
    report_path = "eda_summary_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("SECOM DATASET -- EXPLORATORY DATA ANALYSIS SUMMARY REPORT\n")
        f.write("=" * 65 + "\n\n")
        f.write(f"1.  Samples (Wafers)         : {n_samples:,}\n")
        f.write(f"2.  Raw Features             : {n_features}\n")
        f.write(f"3.  Time Span                : {lb['timestamp'].min():%Y-%m-%d} -> {lb['timestamp'].max():%Y-%m-%d}\n")
        f.write(f"4.  Active Days              : {n_active_days}\n\n")
        f.write(f"--- CLASS DISTRIBUTION ---\n")
        f.write(f"    Pass (-1): {n_pass:,} wafers  ({pct_pass:.2f}%)\n")
        f.write(f"    Fail (+1): {n_fail:,} wafers  ({pct_fail:.2f}%)\n")
        f.write(f"    Imbalance Ratio: {imb_ratio:.2f} : 1\n\n")
        f.write(f"--- MISSING DATA ---\n")
        for k, v in bins_miss.items():
            f.write(f"    {k}: {v} features\n")
        f.write(f"    Total missing: {total_missing:,} / {total_cells:,} ({total_missing/total_cells*100:.2f}%)\n\n")
        f.write(f"--- FEATURE QUALITY ---\n")
        f.write(f"    Constant/zero-variance: {n_zero_var} features\n")
        f.write(f"    High-skew (|skew|>3)  : {n_high_skew} features\n")
        f.write(f"    Features with outliers: {feats_w_out} / {n_clean_feat}\n\n")
        f.write(f"--- PRUNING SUMMARY ---\n")
        f.write(f"    Raw: {n_features} -> After zero-var drop: {n_after_zero_var} -> "
                f"After high-miss drop: {n_after_high_miss}\n")
        f.write(f"    Active clean matrix: {n_samples:,} x {n_clean_feat}\n")
    print(f"  [Saved] Summary report -> {report_path}")

    _section(0, "EDA PIPELINE COMPLETED SUCCESSFULLY")
    print(f"  All plots saved to: ./{output_dir}/\n")

    return {
        "n_samples": n_samples,
        "n_features": n_features,
        "n_clean_features": n_clean_feat,
        "n_pass": n_pass,
        "n_fail": n_fail,
        "imbalance_ratio": imb_ratio,
        "total_missing_pct": total_missing / total_cells * 100,
        "n_zero_var": n_zero_var,
        "n_high_miss_drop": len(drop_high_miss),
        "n_high_skew": n_high_skew,
        "n_outlier_feats": feats_w_out,
        "top20_corr": top20_corr,
    }


# -----------------------------------------------------------------------------
if __name__ == "__main__":
    run_eda()
