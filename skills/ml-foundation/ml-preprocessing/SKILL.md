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
---

# ML Preprocessing Skill

## Overview

Data preprocessing is a critical step in any machine learning or causal inference pipeline. This skill provides comprehensive tools and guidance for preparing data, including handling missing values, detecting outliers, engineering features, and reducing dimensionality. Special attention is given to preprocessing considerations specific to causal inference.

## Missing Value Handling

### Detection and Visualization

Before handling missing values, understand the pattern and extent of missingness:

```python
from preprocessing import diagnose_missing

# Get comprehensive missing value report
report = diagnose_missing(df)
print(report['summary'])  # Overall summary
print(report['by_column'])  # Per-column statistics
print(report['patterns'])  # Missingness patterns
```

Key questions to answer:
- What percentage of data is missing?
- Is missingness random (MCAR), related to observed variables (MAR), or related to unobserved values (MNAR)?
- Are missing values concentrated in certain observations or variables?

### Strategies

#### 1. Listwise Deletion (Complete Case Analysis)

```python
from preprocessing import handle_missing

df_clean = handle_missing(df, strategy='drop', columns=None)
```

**When appropriate for causal inference**:
- Data is Missing Completely at Random (MCAR)
- Sample size is large relative to missingness
- Missingness is not related to treatment or outcome

**Caution**: Can introduce selection bias if missingness is related to treatment assignment or outcomes.

#### 2. Mean/Median Imputation

```python
# Mean imputation
df_imputed = handle_missing(df, strategy='mean', columns=['income', 'age'])

# Median imputation (robust to outliers)
df_imputed = handle_missing(df, strategy='median', columns=['income', 'age'])
```

**When appropriate for causal inference**:
- Missingness is limited (<5%)
- Variables are not strongly related to treatment or outcome
- Quick baseline analysis

**Limitations**: Reduces variance and can bias coefficient estimates.

#### 3. Multiple Imputation

```python
# Multiple imputation creates several imputed datasets
df_imputed = handle_missing(df, strategy='multiple', columns=None, n_imputations=5)
```

**When appropriate for causal inference**:
- Data is MAR (Missing at Random)
- Larger amounts of missingness
- More accurate uncertainty quantification needed

**Best practice**: Run causal analysis on each imputed dataset and pool results using Rubin's rules.

### Missingness and Causal Inference

The missing data mechanism affects causal validity:

| Mechanism | Description | Impact on Causal Estimates |
|-----------|-------------|---------------------------|
| MCAR | Completely random | Unbiased, reduced power |
| MAR | Depends on observed data | Can be handled with proper imputation |
| MNAR | Depends on unobserved data | Potentially biased; sensitivity analysis needed |

## Outlier Detection

### IQR Method

The Interquartile Range method identifies outliers as points beyond 1.5*IQR from the quartiles:

```python
from preprocessing import detect_outliers, remove_outliers

# Detect outliers using IQR
outlier_mask = detect_outliers(df, columns=['income', 'age'], method='iqr')

# Remove outliers
df_clean = remove_outliers(df, columns=['income', 'age'], method='iqr', threshold=1.5)
```

**Advantages**: Non-parametric, robust to non-normal distributions.

### Z-Score Method

Identifies outliers based on standard deviations from the mean:

```python
# Detect outliers beyond 3 standard deviations
outlier_mask = detect_outliers(df, columns=['income'], method='zscore')

# Remove with custom threshold
df_clean = remove_outliers(df, columns=['income'], method='zscore', threshold=2.5)
```

**Advantages**: Simple interpretation; works well for normally distributed data.

### Isolation Forest (Multivariate)

For detecting outliers across multiple dimensions simultaneously:

```python
# Multivariate outlier detection
outlier_mask = detect_outliers(df, columns=['income', 'age', 'education'], method='isolation_forest')
```

**Advantages**: Captures complex multivariate patterns; handles high-dimensional data.

### Mahalanobis Distance

Detects outliers considering variable correlations:

```python
outlier_mask = detect_outliers(df, columns=['X1', 'X2', 'X3'], method='mahalanobis')
```

**Advantages**: Accounts for correlation structure; statistically principled.

### Outliers in Causal Inference

**Important considerations**:
- Outliers may be legitimate extreme cases important for generalizability
- Removing outliers can change the estimand (e.g., ATT vs. effect for typical units)
- Report sensitivity of results to outlier handling decisions
- Consider winsorization as an alternative to removal

## Feature Engineering

### Standardization/Normalization

```python
from preprocessing import standardize

# Z-score standardization (mean=0, std=1)
df_std, scaler = standardize(df, columns=['income', 'age', 'education'])

# Access scaler for transforming new data
new_data_std = scaler.transform(new_data[['income', 'age', 'education']])
```

**When to use**:
- Variables on different scales
- Regularized regression (LASSO, Ridge)
- Distance-based methods (matching, clustering)

**Causal note**: Standardize covariates but typically not the outcome variable.

### Polynomial Features

```python
from sklearn.preprocessing import PolynomialFeatures

# Create polynomial features
poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(df[['age', 'income']])
```

**Use in causal inference**: Flexible functional forms for propensity scores or outcome models.

### Interaction Terms

```python
from preprocessing import create_interactions

# Create specific interaction terms
df_int = create_interactions(df, var_pairs=[('age', 'education'), ('income', 'treatment')])
```

**Causal applications**:
- Effect modification analysis
- Heterogeneous treatment effect estimation
- Flexible propensity score models

### One-Hot Encoding

```python
from sklearn.preprocessing import OneHotEncoder

encoder = OneHotEncoder(sparse=False, drop='first')
encoded = encoder.fit_transform(df[['region', 'education_level']])
```

**Causal note**: Drop one category (reference) to avoid collinearity in regression-based methods.

## Dimensionality Reduction

### PCA for High-Dimensional Controls

```python
from preprocessing import run_pca

# Reduce high-dimensional controls
pca_result = run_pca(df[control_columns], n_components=10)

print(f"Variance explained: {pca_result['variance_explained']}")
df_reduced = pca_result['transformed']
```

### When to Use in Causal ML

PCA can be useful for:
1. **High-dimensional confounders**: Reduce many potential confounders to principal components
2. **Proxy variables**: Create summary measures of latent constructs
3. **Regularization**: Reduce overfitting in propensity score models

**Cautions**:
- PCA components may not have causal interpretations
- First component may not capture all confounding
- Consider sparse PCA or supervised dimension reduction for causal applications

### Double Machine Learning with PCA

```python
# Example: Using PCA-reduced controls in DML
pca_controls = run_pca(df[high_dim_controls], n_components=20)['transformed']
df_for_dml = pd.concat([df[['Y', 'D']], pca_controls], axis=1)
```

## Clustering

### K-Means

```python
from preprocessing import cluster_analysis

# K-Means clustering
result = cluster_analysis(df[features], method='kmeans', n_clusters=5)

df['cluster'] = result['labels']
print(f"Inertia: {result['inertia']}")
```

**Choosing k**: Use elbow method or silhouette scores.

### DBSCAN

```python
# Density-based clustering (automatically determines clusters)
result = cluster_analysis(df[features], method='dbscan', eps=0.5, min_samples=5)

df['cluster'] = result['labels']
print(f"Number of clusters: {result['n_clusters']}")
print(f"Noise points: {result['n_noise']}")
```

**Advantages**: No need to specify number of clusters; handles non-spherical shapes.

### Applications in Heterogeneous Treatment Effects

Clustering can identify subgroups with different treatment effects:

```python
from preprocessing import cluster_analysis, preprocess_for_causal

# 1. Cluster on pre-treatment characteristics
cluster_result = cluster_analysis(df[covariates], method='kmeans', n_clusters=4)
df['subgroup'] = cluster_result['labels']

# 2. Estimate treatment effects within each subgroup
for group in df['subgroup'].unique():
    subset = df[df['subgroup'] == group]
    # Estimate effect for this subgroup
    print(f"Subgroup {group}: n={len(subset)}")
```

**Use cases**:
- Discovering heterogeneity patterns
- Defining subgroups for policy targeting
- Identifying treatment-by-covariate interactions

## Full Preprocessing Pipeline for Causal ML

```python
from preprocessing import preprocess_for_causal

# Complete preprocessing for causal inference
result = preprocess_for_causal(
    df=df,
    outcome='Y',
    treatment='D',
    controls=['X1', 'X2', 'X3', 'X4'],
    missing_strategy='multiple',
    outlier_method='iqr',
    standardize_controls=True
)

df_processed = result['data']
preprocessing_report = result['report']
```

## Best Practices for Causal Inference Preprocessing

### 1. Preserve Treatment/Control Comparability

- Apply the same preprocessing to treatment and control groups
- Document any group-specific handling

### 2. Avoid Post-Treatment Variable Issues

- Only include pre-treatment variables as controls
- Be cautious with variables that could be affected by treatment

### 3. Handle Missingness Thoughtfully

- Investigate whether missingness is related to treatment
- Consider creating missing indicators as additional controls
- Use multiple imputation for uncertainty quantification

### 4. Document All Preprocessing Decisions

- Create a preprocessing log
- Report sensitivity to alternative decisions
- Make preprocessing code reproducible

### 5. Use Proper Train/Test Splitting

```python
from sklearn.model_selection import train_test_split

# CORRECT: Split first, then preprocess
train, test = train_test_split(df, test_size=0.2, random_state=42)

# Fit preprocessors on training data only
scaler = StandardScaler().fit(train[controls])
train_scaled = scaler.transform(train[controls])
test_scaled = scaler.transform(test[controls])  # Use train statistics!
```

## Common Mistakes

### 1. Preprocessing Before Train/Test Split

**Wrong**:
```python
# DON'T DO THIS
df_scaled = standardize(df, columns)  # Leaks test information
train, test = train_test_split(df_scaled)
```

**Correct**:
```python
# DO THIS
train, test = train_test_split(df)
train_scaled, scaler = standardize(train, columns)
test_scaled = scaler.transform(test[columns])
```

### 2. Imputing Outcome Variables

**Wrong**: Imputing missing outcomes can bias treatment effect estimates.

**Correct**: Use analysis methods that handle missing outcomes appropriately, or carefully consider the missing data mechanism.

### 3. Removing "Inconvenient" Observations

**Wrong**: Dropping observations that don't fit expected patterns without justification.

**Correct**: Document outlier removal criteria a priori; report sensitivity analyses.

### 4. Over-Engineering Features

**Wrong**: Creating many features that may not have theoretical justification.

**Correct**: Use theory to guide feature engineering; regularize to prevent overfitting.

### 5. Ignoring Missingness Patterns

**Wrong**: Using default imputation without understanding missingness mechanism.

**Correct**: Investigate missingness patterns; consider whether missingness is informative.

## Summary

Preprocessing is foundational to causal inference quality. Key principles:

1. **Understand your data**: Diagnose missing values, outliers, and distributions before preprocessing
2. **Preserve causal validity**: Ensure preprocessing doesn't introduce bias or remove valid variation
3. **Document decisions**: Make all preprocessing choices explicit and reproducible
4. **Test sensitivity**: Report how results change under alternative preprocessing decisions
5. **Respect the train/test boundary**: Never use test data information when fitting preprocessors
