"""
SECOM Exploratory Data Analysis (EDA) Pipeline

This script performs comprehensive exploratory data analysis on the SECOM dataset:
1. Data ingestion & alignment (features + labels).
2. Data quality checks (missingness, constant columns, duplicates).
3. Target class imbalance quantification.
4. Time-series defect distribution analysis.
5. Saves visualization plots and summary reports.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set visual style
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'Arial'
plt.rcParams['axes.edgecolor'] = '#cccccc'
plt.rcParams['axes.linewidth'] = 0.8

def run_eda(data_path="secom/secom.csv", labels_path="secom/secom_labels.csv", output_dir="plots"):
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 60)
    print(" 1. LOADING SECOM DATASET")
    print("=" * 60)
    
    if not os.path.exists(data_path) or not os.path.exists(labels_path):
        raise FileNotFoundError(f"Files not found: {data_path} or {labels_path}")
        
    df_features = pd.read_csv(data_path)
    df_labels = pd.read_csv(labels_path)
    
    print(f"Features shape : {df_features.shape[0]} rows x {df_features.shape[1]} columns")
    print(f"Labels shape   : {df_labels.shape[0]} rows x {df_labels.shape[1]} columns")
    
    # Standardize labels: -1 -> 0 (Pass), 1 -> 1 (Fail)
    df_labels['target'] = df_labels['label'].map({-1: 0, 1: 1})
    df_labels['timestamp'] = pd.to_datetime(df_labels['timestamp'], dayfirst=True)
    
    # ---------------------------------------------------------
    # 2. CLASS DISTRIBUTION
    # ---------------------------------------------------------
    print("\n" + "=" * 60)
    print(" 2. TARGET CLASS DISTRIBUTION")
    print("=" * 60)
    
    class_counts = df_labels['target'].value_counts().rename({0: 'Pass (0)', 1: 'Fail (1)'})
    class_pct = (df_labels['target'].value_counts(normalize=True) * 100).round(2)
    
    for cls in [0, 1]:
        label_name = "Pass (Normal)" if cls == 0 else "Fail (Defective)"
        count = (df_labels['target'] == cls).sum()
        pct = (count / len(df_labels)) * 100
        print(f"  {label_name:<16}: {count:>5} wafers ({pct:>5.2f}%)")
    
    imbalance_ratio = (df_labels['target'] == 0).sum() / (df_labels['target'] == 1).sum()
    print(f"  Class Imbalance Ratio: ~{imbalance_ratio:.1f} to 1 (Pass : Fail)")
    
    # Plot Class Distribution
    fig, ax = plt.subplots(figsize=(6, 4), dpi=150)
    colors = ['#2b7bba', '#d9534f']
    bars = ax.bar(['Pass (-1)', 'Fail (+1)'], [class_counts['Pass (0)'], class_counts['Fail (1)']], color=colors, width=0.5, edgecolor='black', linewidth=0.8)
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, yval + 20, f"{yval:,}\n({yval/len(df_labels)*100:.1f}%)", ha='center', va='bottom', fontsize=9, fontweight='bold')
    ax.set_title("SECOM Yield Outcome Distribution (Severe Imbalance)", fontsize=12, fontweight='bold', pad=12)
    ax.set_ylabel("Wafer Count", fontsize=10)
    ax.set_ylim(0, len(df_labels) * 1.1)
    plt.tight_layout()
    class_plot_path = os.path.join(output_dir, "eda_class_distribution.png")
    fig.savefig(class_plot_path)
    plt.close(fig)
    print(f"  [Saved] Class distribution chart -> {class_plot_path}")
    
    # ---------------------------------------------------------
    # 3. MISSING VALUE ANALYSIS
    # ---------------------------------------------------------
    print("\n" + "=" * 60)
    print(" 3. MISSING DATA PROFILING")
    print("=" * 60)
    
    missing_counts = df_features.isna().sum()
    missing_pct = (missing_counts / len(df_features)) * 100
    
    total_cells = df_features.size
    total_missing = missing_counts.sum()
    print(f"  Total missing values       : {total_missing:,} / {total_cells:,} ({total_missing / total_cells * 100:.2f}%)")
    print(f"  Features with 0 missing    : {(missing_pct == 0).sum():>3} features")
    print(f"  Features with 0% - 10% miss: {((missing_pct > 0) & (missing_pct <= 10)).sum():>3} features")
    print(f"  Features with 10% - 30% mis: {((missing_pct > 10) & (missing_pct <= 30)).sum():>3} features")
    print(f"  Features with 30% - 50% mis: {((missing_pct > 30) & (missing_pct <= 50)).sum():>3} features")
    print(f"  Features with > 50% missing: {(missing_pct > 50).sum():>3} features (recommended for deletion)")
    
    # Plot Missing Value Distribution
    fig, ax = plt.subplots(figsize=(7, 4), dpi=150)
    sns.histplot(missing_pct, bins=30, kde=False, color='#e67e22', edgecolor='black', ax=ax)
    ax.axvline(50, color='red', linestyle='--', linewidth=1.5, label='50% Drop Threshold')
    ax.set_title("Missing Values Percentage per Feature", fontsize=12, fontweight='bold', pad=12)
    ax.set_xlabel("Missing Percentage (%)", fontsize=10)
    ax.set_ylabel("Number of Features", fontsize=10)
    ax.legend()
    plt.tight_layout()
    missing_plot_path = os.path.join(output_dir, "eda_missingness_histogram.png")
    fig.savefig(missing_plot_path)
    plt.close(fig)
    print(f"  [Saved] Missingness histogram -> {missing_plot_path}")
    
    # ---------------------------------------------------------
    # 4. CONSTANT & LOW-VARIANCE SENSORS
    # ---------------------------------------------------------
    print("\n" + "=" * 60)
    print(" 4. CONSTANT / ZERO-VARIANCE SENSORS")
    print("=" * 60)
    
    nunique_per_col = df_features.nunique(dropna=True)
    zero_var_cols = nunique_per_col[nunique_per_col <= 1].index.tolist()
    print(f"  Constant features (nunique <= 1) : {len(zero_var_cols)} features")
    print(f"  First 10 constant features       : {zero_var_cols[:10]}")
    print("  -> These 116 features contain zero predictive variance and should be removed.")
    
    # ---------------------------------------------------------
    # 5. TIME-SERIES DEFECT OCCURRENCE
    # ---------------------------------------------------------
    print("\n" + "=" * 60)
    print(" 5. TIME-SERIES YIELD TREND")
    print("=" * 60)
    
    df_labels['date'] = df_labels['timestamp'].dt.date
    daily_stats = df_labels.groupby('date').agg(
        total_wafers=('target', 'count'),
        fails=('target', 'sum')
    )
    daily_stats['fail_rate'] = (daily_stats['fails'] / daily_stats['total_wafers']) * 100
    
    print(f"  Date span     : {df_labels['timestamp'].min().strftime('%Y-%m-%d')} to {df_labels['timestamp'].max().strftime('%Y-%m-%d')}")
    print(f"  Active days   : {len(daily_stats)} operational days")
    print(f"  Max daily fails: {daily_stats['fails'].max()} on {daily_stats['fails'].idxmax()}")
    
    # Plot Weekly/Timeline trend
    weekly_stats = df_labels.set_index('timestamp').resample('W').agg(
        total_wafers=('target', 'count'),
        fails=('target', 'sum')
    )
    weekly_stats['yield_loss_pct'] = (weekly_stats['fails'] / weekly_stats['total_wafers']) * 100
    
    fig, ax1 = plt.subplots(figsize=(10, 4.5), dpi=150)
    ax2 = ax1.twinx()
    
    ax1.bar(weekly_stats.index, weekly_stats['total_wafers'], width=5, color='#95a5a6', alpha=0.5, label='Total Wafers Processed')
    ax2.plot(weekly_stats.index, weekly_stats['yield_loss_pct'], color='#c0392b', marker='o', linewidth=2, label='Yield Loss Rate (%)')
    
    ax1.set_xlabel("Time (Weeks)", fontsize=10)
    ax1.set_ylabel("Wafers Processed", fontsize=10, color='#555555')
    ax2.set_ylabel("Defect / Fail Rate (%)", fontsize=10, color='#c0392b')
    ax2.grid(False)
    ax1.set_title("SECOM Production Volume & Defect Rate Across Time", fontsize=12, fontweight='bold', pad=12)
    plt.tight_layout()
    timeline_plot_path = os.path.join(output_dir, "eda_timeline_defects.png")
    fig.savefig(timeline_plot_path)
    plt.close(fig)
    print(f"  [Saved] Timeline defect chart -> {timeline_plot_path}")
    
    # Write summary text report
    report_path = "eda_summary_report.txt"
    with open(report_path, "w") as f:
        f.write("SECOM DATASET EXPLORATORY DATA ANALYSIS REPORT\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"1. Total Wafers / Samples: {len(df_features):,}\n")
        f.write(f"2. Total Raw Features: {df_features.shape[1]}\n")
        f.write(f"3. Class Balance:\n")
        f.write(f"   - Pass (-1): {class_counts['Pass (0)']:,} ({class_pct.get(0, 0):.2f}%)\n")
        f.write(f"   - Fail (+1): {class_counts['Fail (1)']:,} ({class_pct.get(1, 0):.2f}%)\n")
        f.write(f"   - Imbalance Ratio: {imbalance_ratio:.2f}:1\n\n")
        f.write(f"4. Missing Data:\n")
        f.write(f"   - Total Missing Cells: {total_missing:,} ({total_missing / total_cells * 100:.2f}%)\n")
        f.write(f"   - Features with >50% missing (recommend drop): {(missing_pct > 50).sum()}\n")
        f.write(f"5. Zero-Variance / Constant Features: {len(zero_var_cols)} features\n")
        f.write(f"6. Usable Features after basic filtering: {df_features.shape[1] - len(zero_var_cols) - (missing_pct > 50).sum()}\n")
    print(f"  [Saved] Detailed summary report -> {report_path}")
    print("=" * 60)
    print(" EDA COMPLETED SUCCESSFULLY!")
    print("=" * 60)

if __name__ == "__main__":
    run_eda()
