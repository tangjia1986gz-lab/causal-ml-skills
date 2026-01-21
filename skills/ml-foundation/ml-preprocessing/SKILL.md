---
name: ml-preprocessing
description: Data preprocessing techniques for machine learning and causal inference
triggers:
  - missing values
  - outliers
  - feature engineering
  - data cleaning
  - preprocessing
  - imputation
  - standardization
  - normalization
  - PCA
  - clustering
  - bad controls
  - pre-treatment variables
---

# ML Preprocessing Skill

## Overview

Data preprocessing is a critical step in any machine learning or causal inference pipeline. This skill provides comprehensive tools and guidance for preparing data, including handling missing values, detecting outliers, engineering features, and reducing dimensionality. **Special attention is given to preprocessing considerations specific to causal inference**, particularly the selection of pre-treatment variables and avoiding bad controls.

## Quick Start

### CLI Tools

```bash
# Complete preprocessing pipeline
python scripts/run_preprocessing.py data.csv \
    --output processed.csv \
    --outcome Y --treatment D \
    --controls X1 X2 X3 \
    --missing-strategy mean \
    --outlier-method iqr \
    --standardize

# Data quality diagnostics
python scripts/diagnose_data_quality.py data.csv \
    --treatment D --outcome Y \
    --report-path quality_report.json

# Visualization
python scripts/visualize_distributions.py data.csv \
    --output-dir plots/ \
    --treatment D --outcome Y \
    --plot-all
```

### Python API

```python
from preprocessing import preprocess_for_causal, diagnose_missing

# Diagnose data quality
report = diagnose_missing(df)
print(report['recommendations'])

# Complete preprocessing pipeline
result = preprocess_for_causal(
    df=df,
    outcome='Y',
    treatment='D',
    controls=['X1', 'X2', 'X3'],
    missing_strategy='mean',
    outlier_method='iqr',
    standardize_controls=True
)
df_processed = result['data']
```

## Documentation Structure

### Reference Documents

| Document | Description |
|----------|-------------|
| [references/data_cleaning.md](references/data_cleaning.md) | Missing values (MCAR, MAR, MNAR), outliers, duplicates |
| [references/feature_engineering.md](references/feature_engineering.md) | Transformations, interactions, polynomial features |
| [references/scaling_encoding.md](references/scaling_encoding.md) | Standardization, normalization, one-hot, target encoding |
| [references/causal_considerations.md](references/causal_considerations.md) | Pre-treatment variables, bad controls, preserving identification |

### Scripts

| Script | Description |
|--------|-------------|
| [scripts/run_preprocessing.py](scripts/run_preprocessing.py) | Complete preprocessing pipeline CLI |
| [scripts/diagnose_data_quality.py](scripts/diagnose_data_quality.py) | Data quality assessment |
| [scripts/visualize_distributions.py](scripts/visualize_distributions.py) | Distribution and balance plots |

### Templates

| Template | Description |
|----------|-------------|
| [assets/markdown/data_report.md](assets/markdown/data_report.md) | Data quality report template |

---

## Causal Inference Preprocessing Principles

### The Pre-Treatment Rule

**Only include variables measured BEFORE treatment assignment as controls.**

```
Timeline:
    Confounders (Z)     Treatment (D)     Mediators (M)     Outcome (Y)
         |                    |                |                |
    INCLUDE as controls      ---        DO NOT INCLUDE         ---
```

See [references/causal_considerations.md](references/causal_considerations.md) for detailed guidance on:
- Identifying pre-treatment variables
- Avoiding bad controls (mediators, colliders, M-bias)
- Preserving causal identification through preprocessing

### Variable Selection Framework

For each potential control variable X, ask:

1. **Is X measured BEFORE treatment?** No --> EXCLUDE
2. **Does X affect BOTH treatment AND outcome?** Yes --> INCLUDE (confounder)
3. **Does X affect only outcome?** Yes --> MAY INCLUDE (precision)
4. **Does X affect only treatment?** Yes --> EXCLUDE (instrument)

---

## Missing Value Handling

### Detection and Visualization

```python
from preprocessing import diagnose_missing

report = diagnose_missing(df)
print(report['summary'])      # Overall statistics
print(report['by_column'])    # Per-column breakdown
print(report['patterns'])     # Missingness patterns
print(report['recommendations'])  # Suggested strategies
```

### Strategies by Context

| Strategy | When to Use | Causal Considerations |
|----------|-------------|----------------------|
| Listwise deletion | MCAR, <5% missing | May lose power but unbiased |
| Mean/median imputation | Quick baseline | Attenuates relationships |
| Multiple imputation | MAR, substantial missing | Properly propagates uncertainty |
| Missing indicator | Informative missingness | Allows missingness to predict outcome |

```python
from preprocessing import handle_missing

# Single imputation
df_imputed = handle_missing(df, strategy='mean', columns=['income', 'age'])

# Multiple imputation
imputed_dfs = handle_missing(df, strategy='multiple', n_imputations=5)
```

**Key Rules for Causal Inference:**
- Never impute treatment assignment
- Generally do not impute outcomes
- Create missing indicators only for pre-treatment variables

See [references/data_cleaning.md](references/data_cleaning.md) for comprehensive missing value guidance.

---

## Outlier Detection

### Methods

```python
from preprocessing import detect_outliers, remove_outliers

# Univariate methods
outliers_iqr = detect_outliers(df, columns=['income'], method='iqr', threshold=1.5)
outliers_zscore = detect_outliers(df, columns=['income'], method='zscore', threshold=3)

# Multivariate methods
outliers_multi = detect_outliers(df, columns=['X1', 'X2', 'X3'], method='isolation_forest')
outliers_mahal = detect_outliers(df, columns=['X1', 'X2', 'X3'], method='mahalanobis')
```

### Causal Considerations for Outliers

| Scenario | Recommendation |
|----------|----------------|
| Data entry errors | Remove after verification |
| Legitimate extreme values | Keep - they represent real variation |
| Different population | Consider separate analysis |
| Influential on treatment effect | Report sensitivity analysis |

**Important**: Removing outliers changes your estimand from effect on full population to effect on "typical" units.

See [references/data_cleaning.md](references/data_cleaning.md) for outlier handling guidance.

---

## Feature Engineering

### Standardization

```python
from preprocessing import standardize

# Standardize control variables (NOT outcome or treatment)
df_std, scaler = standardize(df, columns=['income', 'age', 'education'])

# Apply to new data
new_data_std = scaler.transform(new_data[['income', 'age', 'education']])
```

**When to use:**
- Regularized regression (LASSO, Ridge)
- Distance-based methods (matching)
- Comparing coefficient magnitudes

### Interaction Terms

```python
from preprocessing import create_interactions

# Create specific interactions
df_int = create_interactions(df, var_pairs=[('age', 'education'), ('income', 'treatment')])
```

**Causal applications:**
- Effect modification analysis
- Heterogeneous treatment effect estimation
- Flexible propensity score models

### Polynomial Features

```python
from sklearn.preprocessing import PolynomialFeatures

poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(df[['age', 'income']])
```

See [references/feature_engineering.md](references/feature_engineering.md) for detailed feature engineering guidance.
See [references/scaling_encoding.md](references/scaling_encoding.md) for encoding categorical variables.

---

## Dimensionality Reduction

### PCA for High-Dimensional Controls

```python
from preprocessing import run_pca

pca_result = run_pca(df[control_columns], n_components=10)
print(f"Variance explained: {pca_result['cumulative_variance'][-1]:.1%}")
df_reduced = pca_result['transformed']
```

**Causal Considerations:**
- PCA maximizes variance, not confounding control
- First principal component may capture outcome-irrelevant variation
- Components lack interpretability
- Consider supervised dimension reduction for causal applications

---

## Clustering for Heterogeneous Effects

```python
from preprocessing import cluster_analysis

# Cluster on PRE-TREATMENT characteristics
cluster_result = cluster_analysis(df[covariates], method='kmeans', n_clusters=4)
df['subgroup'] = cluster_result['labels']

# Estimate effects within subgroups
for group in df['subgroup'].unique():
    subset = df[df['subgroup'] == group]
    print(f"Subgroup {group}: n={len(subset)}")
```

**Use cases:**
- Discovering heterogeneity patterns
- Defining subgroups for policy targeting
- Identifying treatment-by-covariate interactions

---

## Full Preprocessing Pipeline

```python
from preprocessing import preprocess_for_causal

result = preprocess_for_causal(
    df=df,
    outcome='Y',
    treatment='D',
    controls=['X1', 'X2', 'X3', 'X4'],
    missing_strategy='mean',
    outlier_method='iqr',
    outlier_threshold=1.5,
    standardize_controls=True,
    create_missing_indicators=True
)

df_processed = result['data']
report = result['report']
print(f"Retained {report['pct_retained']:.1f}% of observations")
```

---

## Best Practices Summary

### 1. Pre-Treatment Only

```python
# CORRECT
pre_treatment_vars = ['age', 'baseline_income', 'education', 'pre_health']
df = preprocess(df, controls=pre_treatment_vars)

# WRONG - includes post-treatment variables
# post_treatment_vars = ['current_income', 'current_health']  # BAD!
```

### 2. Respect Train/Test Boundary

```python
# CORRECT: Split first, then preprocess
train, test = train_test_split(df, test_size=0.2)
train_scaled, scaler = standardize(train, columns)
test_scaled = scaler.transform(test[columns])  # Use train statistics!

# WRONG: Preprocess before split (data leakage)
# df_scaled = standardize(df, columns)  # Leaks test info!
# train, test = train_test_split(df_scaled)
```

### 3. Same Preprocessing for Both Groups

```python
# CORRECT: Single scaler for all
scaler = StandardScaler()
df[controls] = scaler.fit_transform(df[controls])

# WRONG: Different scaling by treatment
# treated_scaled = scaler1.fit_transform(df[df['D']==1][controls])  # BAD!
```

### 4. Document All Decisions

- Create preprocessing log
- Report sensitivity to alternative decisions
- Make preprocessing code reproducible

### 5. Report Sensitivity Analyses

```python
# Main analysis
effect_main = estimate_effect(df_processed)

# Sensitivity: different outlier thresholds
for threshold in [1.5, 2.0, 3.0]:
    df_alt = remove_outliers(df, method='iqr', threshold=threshold)
    effect_alt = estimate_effect(df_alt)
    print(f"IQR {threshold}: effect = {effect_alt:.3f}")
```

---

## Common Mistakes to Avoid

| Mistake | Why It's Problematic | Better Approach |
|---------|---------------------|-----------------|
| Imputing outcomes | Attenuates treatment effects | Complete case or bounds |
| Including post-treatment variables | Blocks causal pathway or opens collider path | Pre-treatment only |
| Preprocessing before train/test split | Data leakage | Split first |
| Different preprocessing by treatment | Violates exchangeability | Same rules for all |
| Removing "inconvenient" outliers | Selection bias | Pre-specified rules |
| Feature selection using outcome | Overfitting | Theory-driven selection |

---

## API Reference

### Core Functions

| Function | Description |
|----------|-------------|
| `diagnose_missing(df)` | Comprehensive missing value report |
| `handle_missing(df, strategy, columns)` | Handle missing values |
| `detect_outliers(df, columns, method)` | Detect outliers |
| `remove_outliers(df, columns, method)` | Remove outliers with report |
| `standardize(df, columns)` | Z-score standardization |
| `create_interactions(df, var_pairs)` | Create interaction terms |
| `run_pca(df, n_components)` | PCA dimensionality reduction |
| `cluster_analysis(df, method, n_clusters)` | K-means or DBSCAN clustering |
| `preprocess_for_causal(df, outcome, treatment, controls, ...)` | Complete pipeline |
| `validate_preprocessing(df_train, df_test, controls)` | Validate consistency |
| `quick_preprocess(df, outcome, treatment, controls)` | Quick preprocessing with defaults |

See `preprocessing.py` for full API documentation.

---

## References

- Rubin, D. B. (1987). Multiple Imputation for Nonresponse in Surveys. Wiley.
- Little, R. J., & Rubin, D. B. (2019). Statistical Analysis with Missing Data (3rd ed.). Wiley.
- Angrist, J. D., & Pischke, J. S. (2009). Mostly Harmless Econometrics. Princeton.
- Cinelli, C., Forney, A., & Pearl, J. (2022). A crash course in good and bad controls.
- van Buuren, S. (2018). Flexible Imputation of Missing Data (2nd ed.). CRC Press.
