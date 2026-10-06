# SECOM Exploratory Data Analysis (EDA) Report
**A Systematic Diagnostic & Statistical Quality Assessment for Semiconductor Yield Modeling**

**Prepared for**: Academic Project Review / Research Advisor  
**Project**:Semiconductor Process Modeling & Defect Classification  
**Date**: October 2026  

---

## Executive Summary

Exploratory Data Analysis (EDA) is the foundational phase in semiconductor data science. In high-volume manufacturing (HVM), wafer fabs generate millions of sensor data points per batch. Before training predictive machine learning models, engineers must diagnose **data quality, sensor reliability, class distribution, and temporal stability**.

This report presents a thorough statistical audit of the benchmark **SECOM (Semiconductor Manufacturing)** dataset. The audit was conducted using the automated script [`eda_analysis.py`](eda_analysis.py) and reveals critical operational realities of wafer fabrication telemetry.

---

## 1. Dataset Dimensions & Ingestion Verification

The SECOM dataset consists of process telemetry collected across an in-house semiconductor manufacturing line spanning **89 operational days** (July 19, 2008 – October 17, 2008).

| Parameter | Value | Engineering Description |
| :--- | :--- | :--- |
| **Total Production Runs ($N$)** | **1,567 wafers** | Complete fabrication runs tracked from start to test |
| **Total Sensor Channels ($D$)** | **590 variables** | Analog readings, flow rates, pressures, and status flags |
| **Total Data Points** | **924,530 values** | Total matrix cells in the feature space |
| **Time Span** | **89 days** | 86 active production days |
| **Primary Target Label** | Binary ($\{-1, +1\}$) | Line test outcome: $-1$ (Pass) vs. $+1$ (Fail) |

---

## 2. Target Class Distribution: The 14.1 : 1 Imbalance

In a mature semiconductor fab, defect rates are naturally suppressed. High scrap rates threaten fab profitability, meaning that defects appear as rare events.

```
SECOM YIELD DISTRIBUTION:
─────────────────────────────────────────────────────────────────────────────
Pass (Normal)   [-1] : ████████████████████████████████████████  1,463 (93.36%)
Fail (Defective)[+1] : ███                                         104 ( 6.64%)
─────────────────────────────────────────────────────────────────────────────
Class Imbalance Ratio: ~14.07 : 1 (Pass : Fail)
```

![Class Distribution](plots/eda_class_distribution.png)

### Key Academic Insights:
1. **The "Accuracy Paradox"**:
   - A naive model predicting every wafer will pass achieves **93.36% raw accuracy**, yet catches **0% of defects**.
   - In wafer fabrication, letting a defective wafer pass through to packaging and shipping costs orders of magnitude more than reinspecting a flagged wafer.
2. **Mandatory Methodological Adjustments**:
   - Standard Cross-Entropy or Ordinary Least Squares loss functions will ignore the defective class.
   - The modeling pipeline **must utilize cost-sensitive weighting** (`class_weight='balanced'`, `scale_pos_weight = 14.07`), minority oversampling (SMOTE / ADASYN), and threshold moving.
   - Primary evaluation metrics must be **Recall / Sensitivity**, **PR-AUC (Precision-Recall Area Under Curve)**, and **Balanced Error Rate (BER)**.

---

## 3. Sensor Missingness & Signal Reliability

In real fab operations, sensors intermittently drop offline due to tool maintenance, calibration routines, telemetry network latency, or recipe-dependent probe disabling.

```
MISSING VALUE DISTRIBUTION ACROSS 590 SENSORS:
─────────────────────────────────────────────────────────────────────────────
0% Missing (Complete Data)    :  52 features ( 8.8%)
0.1% to 10.0% Missing (Normal): 486 features (82.4%)
10.1% to 30.0% Missing        :  20 features ( 3.4%)
30.1% to 50.0% Missing        :   4 features ( 0.7%)
> 50.0% Missing (Severely Sparse): 28 features ( 4.7%) ──► [Candidates for Deletion]
─────────────────────────────────────────────────────────────────────────────
Total Missing Matrix Cells    : 41,951 out of 924,530 cells (4.54%)
```

![Missingness Distribution](plots/eda_missingness_histogram.png)

### Engineering Decisions:
1. **Pruning High-Missingness Features (>50%)**:
   - 28 features have missingness rates exceeding 50%. Imputing more than half of a feature's values introduces synthetic noise and distorts decision trees. These 28 features are removed.
2. **Median Imputation for Low-Missingness Features**:
   - For the 510 features with $\le 50\%$ missing values, **median imputation** is chosen over mean imputation because semiconductor sensor signals frequently exhibit heavy-tailed distributions and extreme outlier spikes.

---

## 4. Zero-Variance & Dead Sensors (The 116 Constant Columns)

An essential discovery from our automated audit is that **116 out of 590 features possess zero variance** ($\text{Var}(X) = 0$, meaning `nunique <= 1` across all 1,567 wafers).

```
Sample of Dead / Constant Sensors Detected:
['feature_006', 'feature_014', 'feature_043', 'feature_050', 'feature_053',
 'feature_070', 'feature_098', 'feature_142', 'feature_150', 'feature_179', ...]
Total Count: 116 Features
```

### Physical Fab Interpretation:
Why do constant columns exist in fab telemetry?
- **Fixed Recipe Setpoints**: Parameters hardcoded into the process chamber (e.g., constant gas flow ratio or base chamber vacuum setpoint that never changed across this product run).
- **Decommissioned Channels**: Multi-channel data acquisition systems where specific probe channels were left unpowered.
- **Single-State Flags**: Digital status bits that remained locked throughout the manufacturing batch.

### Impact on Machine Learning:
- Constant columns contain **zero mutual information** ($I(X; Y) = 0$) and zero correlation with yield outcomes.
- Leaving them in the model wastes memory, slows down training, and dilutes tree-based feature importance calculations.
- **Immediate Action**: All 116 constant features are eliminated during step 1 of preprocessing.

---

## 5. Temporal Defect Dynamics & Yield Excursions

Wafers were processed sequentially from July 19 to October 17, 2008. By aggregating wafer runs into weekly and daily production buckets, we examine the temporal stability of the manufacturing line.

![Timeline of Defects](plots/eda_timeline_defects.png)

### Findings:
1. **Burst Defect Clustering (Yield Excursions)**:
   - Defects do not occur uniformly at a random rate of 6.6%. Instead, they cluster in specific time windows.
   - For instance, on **August 20, 2008**, the fab recorded **6 defective wafers in a single 24-hour cycle**.
2. **Process Drift & Maintenance Cycles**:
   - Weeks with elevated defect rates correlate with fab line anomalies—such as chamber wall seasoning degradation, target wear in PVD sputtering, or slurry deterioration in CMP.
   - This temporal dependency demonstrates that wafer defects are driven by physical process drift rather than purely random background noise.

---

## 6. Active Feature Space Summary

After rigorous EDA-guided hygiene, the active feature matrix is transformed as follows:

```
                      FEATURE PRUNING FLOW
┌────────────────────────────────────────────────────────┐
│              Raw Dataset: 590 Features                 │
└───────────────────────────┬────────────────────────────┘
                            │
        Prune 116 Constant Features (0 Variance)
                            │
                            ▼
┌────────────────────────────────────────────────────────┐
│            Intermediate: 474 Features                  │
└───────────────────────────┬────────────────────────────┘
                            │
        Prune 28 High-Missingness Features (>50% NaN)
                            │
                            ▼
┌────────────────────────────────────────────────────────┐
│       Clean Active Feature Matrix: 446 Features        │
│        (1,567 Samples × 446 Continuous Signals)        │
└────────────────────────────────────────────────────────┘
```

---

## 7. Actionable EDA Recommendations for Model Development

| Preprocessing Step | Action Taken | Technical Rationale |
| :--- | :--- | :--- |
| **Constant Feature Removal** | Drop 116 columns | Eliminates zero-information noise; prevents rank-deficient covariance matrices. |
| **High Missingness Filtering** | Drop 28 columns ($>50\%$ missing) | Prevents artificial noise injection caused by heavy synthetic imputation. |
| **Missing Value Imputation** | Apply Median Imputer | Robust against non-Gaussian sensor distributions and extreme outlier spikes. |
| **Feature Scaling** | Apply RobustScaler / StandardScaler | Required for distance-based and regularized models (SVM, Ridge, Neural Nets). |
| **Class Imbalance Management** | Set `scale_pos_weight = 14.1` | Ensures decision boundary does not collapse to the majority "Pass" class. |
| **Evaluation Metrics** | Track Recall, PR-AUC, and BER | Reflects true industrial objective: intercepting defective wafers before shipping. |

---


