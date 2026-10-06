"""
SECOM Feature Ranking & Importance Pipeline

This script performs multi-perspective feature selection and ranking on the SECOM dataset:
1. Preprocessing:
   - Filters out zero-variance/constant features.
   - Filters out features with > 50% missing values.
   - Applies robust median imputation for remaining missing values.
   - Standardizes features.
2. Ranking Methods:
   - Method A: Univariate ANOVA F-test (f_classif) - Tests linear distribution shifts.
   - Method B: Mutual Information (mutual_info_classif) - Captures non-linear dependencies.
   - Method C: Random Forest Gini Importance (with balanced class weights).
3. Consensus Rank Aggregation:
   - Combines individual ranks into an ensemble Consensus Rank.
4. Exports:
   - 'top_features_ranked.csv': Full ranking spreadsheet.
   - 'plots/top20_feature_importance.png': Visualization of top features.
   - Comparative statistical profiles for Pass vs Fail wafers on top features.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import f_classif, mutual_info_classif
from sklearn.ensemble import RandomForestClassifier

# Set visual style
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'Arial'
plt.rcParams['axes.edgecolor'] = '#cccccc'
plt.rcParams['axes.linewidth'] = 0.8

def run_feature_ranking(
    data_path="secom/secom.csv",
    labels_path="secom/secom_labels.csv",
    output_dir="plots",
    missing_threshold=50.0,
    top_n=20
):
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 65)
    print(" 1. DATA PREPROCESSING FOR FEATURE SELECTION")
    print("=" * 65)
    
    X_raw = pd.read_csv(data_path)
    y_raw = pd.read_csv(labels_path)
    
    # Map target: Pass (-1) -> 0, Fail (1) -> 1
    y = y_raw['label'].map({-1: 0, 1: 1}).values
    
    # Step 1: Remove constant features
    nunique = X_raw.nunique(dropna=True)
    constant_cols = nunique[nunique <= 1].index.tolist()
    X_filtered = X_raw.drop(columns=constant_cols)
    print(f"  Dropped {len(constant_cols)} constant features (0 variance).")
    
    # Step 2: Remove high-missingness features (> threshold)
    missing_pct = (X_filtered.isna().sum() / len(X_filtered)) * 100
    high_missing_cols = missing_pct[missing_pct > missing_threshold].index.tolist()
    X_filtered = X_filtered.drop(columns=high_missing_cols)
    print(f"  Dropped {len(high_missing_cols)} features with > {missing_threshold}% missing values.")
    print(f"  Remaining active features: {X_filtered.shape[1]}")
    
    feature_names = X_filtered.columns.tolist()
    
    # Step 3: Median Imputation
    imputer = SimpleImputer(strategy='median')
    X_imputed = imputer.fit_transform(X_filtered)
    
    # Step 4: Standardization
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)
    
    # ---------------------------------------------------------
    # 2. RUNNING MULTI-PERSPECTIVE RANKING METHODS
    # ---------------------------------------------------------
    print("\n" + "=" * 65)
    print(" 2. COMPUTING FEATURE IMPORTANCES")
    print("=" * 65)
    
    # Method A: ANOVA F-statistic
    print("  [1/3] Calculating ANOVA F-test statistics (Univariate Linear)...")
    f_scores, p_values = f_classif(X_scaled, y)
    # Handle NaNs if any
    f_scores = np.nan_to_num(f_scores, nan=0.0)
    
    # Method B: Mutual Information
    print("  [2/3] Calculating Mutual Information (Non-Linear Dependency)...")
    mi_scores = mutual_info_classif(X_scaled, y, discrete_features=False, random_state=42)
    
    # Method C: Balanced Random Forest
    print("  [3/3] Training Balanced Random Forest Classifier (Tree Impurity)...")
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_scaled, y)
    rf_importances = rf.feature_importances_
    
    # ---------------------------------------------------------
    # 3. CONSENSUS RANK AGGREGATION
    # ---------------------------------------------------------
    print("\n" + "=" * 65)
    print(" 3. AGGREGATING CONSENSUS RANKS")
    print("=" * 65)
    
    df_ranks = pd.DataFrame({
        'Feature': feature_names,
        'ANOVA_F': f_scores,
        'ANOVA_pValue': p_values,
        'Mutual_Info': mi_scores,
        'RF_Importance': rf_importances
    })
    
    # Rank features: Higher score -> Better rank (1 is best)
    df_ranks['Rank_ANOVA'] = df_ranks['ANOVA_F'].rank(ascending=False, method='min')
    df_ranks['Rank_MI'] = df_ranks['Mutual_Info'].rank(ascending=False, method='min')
    df_ranks['Rank_RF'] = df_ranks['RF_Importance'].rank(ascending=False, method='min')
    
    # Consensus Rank: Average Rank across methods
    df_ranks['Avg_Rank'] = (df_ranks['Rank_ANOVA'] + df_ranks['Rank_MI'] + df_ranks['Rank_RF']) / 3.0
    df_ranks = df_ranks.sort_values(by='Avg_Rank').reset_index(drop=True)
    df_ranks['Consensus_Rank'] = range(1, len(df_ranks) + 1)
    
    # Save full ranking table
    ranked_csv_path = "top_features_ranked.csv"
    df_ranks.to_csv(ranked_csv_path, index=False)
    print(f"  [Saved] Full feature ranking table -> {ranked_csv_path}")
    
    # ---------------------------------------------------------
    # 4. STATISTICAL PROFILING OF TOP FEATURES
    # ---------------------------------------------------------
    print("\n" + "=" * 65)
    print(f" 4. TOP {min(15, len(df_ranks))} CONSENSUS FEATURES & SHIFT IN DEFECTIVE WAFERS")
    print("=" * 65)
    
    top_cols = df_ranks['Feature'].head(15).tolist()
    print(f"{'Rank':<5} {'Feature':<14} {'Pass Mean':<12} {'Fail Mean':<12} {'% Shift':<10} {'F-Score':<10} {'RF Imp':<10}")
    print("-" * 75)
    
    df_analyzed = X_raw.copy()
    df_analyzed['target'] = y
    
    for i, col in enumerate(top_cols, 1):
        pass_mean = df_analyzed[df_analyzed['target'] == 0][col].mean()
        fail_mean = df_analyzed[df_analyzed['target'] == 1][col].mean()
        shift = ((fail_mean - pass_mean) / (abs(pass_mean) + 1e-8)) * 100
        
        row_stat = df_ranks[df_ranks['Feature'] == col].iloc[0]
        f_val = row_stat['ANOVA_F']
        rf_val = row_stat['RF_Importance']
        
        print(f"{i:<5} {col:<14} {pass_mean:<12.4f} {fail_mean:<12.4f} {shift:>+7.2f}% {f_val:<10.2f} {rf_val:<10.4f}")
        
    # ---------------------------------------------------------
    # 5. VISUALIZATION OF TOP 20 FEATURES
    # ---------------------------------------------------------
    df_top = df_ranks.head(top_n).copy()
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7), dpi=150)
    
    # Left: Random Forest Importance
    sns.barplot(
        data=df_top,
        y='Feature',
        x='RF_Importance',
        palette='Blues_r',
        ax=ax1,
        edgecolor='black',
        linewidth=0.5
    )
    ax1.set_title(f"Top {top_n} Features by Random Forest Importance", fontsize=11, fontweight='bold')
    ax1.set_xlabel("Relative Importance Score", fontsize=10)
    ax1.set_ylabel("Sensor / Feature Code", fontsize=10)
    
    # Right: ANOVA F-Score
    sns.barplot(
        data=df_top,
        y='Feature',
        x='ANOVA_F',
        palette='Reds_r',
        ax=ax2,
        edgecolor='black',
        linewidth=0.5
    )
    ax2.set_title(f"Top {top_n} Features by ANOVA F-Statistic (Linear Separation)", fontsize=11, fontweight='bold')
    ax2.set_xlabel("ANOVA F-Score", fontsize=10)
    ax2.set_ylabel("")
    
    plt.suptitle("SECOM Top Sensor Feature Rankings", fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    plot_file = os.path.join(output_dir, "top20_feature_importance.png")
    fig.savefig(plot_file)
    plt.close(fig)
    print(f"\n  [Saved] Top features importance chart -> {plot_file}")
    
    print("=" * 65)
    print(" FEATURE RANKING COMPLETED SUCCESSFULLY!")
    print("=" * 65)

if __name__ == "__main__":
    run_feature_ranking()
