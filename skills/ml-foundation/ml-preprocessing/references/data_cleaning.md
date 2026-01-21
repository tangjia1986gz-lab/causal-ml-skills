# Data Cleaning for Causal Inference

## Overview

Data cleaning is a critical first step in any causal inference pipeline. Poor data quality can introduce bias, reduce statistical power, and invalidate causal conclusions. This document covers missing values, outliers, and duplicates with special attention to causal inference requirements.

## Missing Values

### Understanding Missingness Mechanisms

The missing data mechanism determines what methods are valid for handling missingness:

| Mechanism | Definition | Example | Causal Implications |
|-----------|------------|---------|---------------------|
| **MCAR** (Missing Completely at Random) | Missingness is independent of all data | Random sensor failure | Unbiased estimates with reduced power |
| **MAR** (Missing at Random) | Missingness depends on observed data | Higher earners skip income questions | Proper imputation can recover unbiased estimates |
| **MNAR** (Missing Not at Random) | Missingness depends on unobserved values | Depressed patients skip depression surveys | Potentially biased; requires sensitivity analysis |

### Diagnosing Missingness

```python
from preprocessing import diagnose_missing

# Comprehensive missing value report
report = diagnose_missing(df)

# Key outputs:
# - report['summary']: Overall statistics
# - report['by_column']: Per-column breakdown
# - report['patterns']: Common missingness patterns
# - report['is_monotone']: Whether pattern is monotone
# - report['recommendations']: Suggested strategies
```

### Testing for MCAR

Little's MCAR test can help determine if data is MCAR:

```python
import numpy as np
from scipy import stats

def littles_mcar_test(df):
    """
    Simplified MCAR test using chi-square statistic.
    Null hypothesis: Data is MCAR.
    """
    # Compare means of variables when other variables are missing vs observed
    # Low p-value suggests data is NOT MCAR
    pass  # Implementation requires careful handling of covariance structure
```

### Handling Strategies by Context

#### For Pre-Treatment Covariates

| Strategy | When to Use | Causal Considerations |
|----------|-------------|----------------------|
| Listwise deletion | MCAR, <5% missing | May lose power but preserves unbiasedness |
| Mean/median imputation | Quick baseline, limited missing | Attenuates relationships; use with caution |
| Multiple imputation | MAR, substantial missing | Properly propagates uncertainty |
| Missing indicator + imputation | Informative missingness suspected | Allows missingness to predict outcome |

#### For Treatment Variable

**Never impute treatment assignment.** Missing treatment typically means:
- Exclude observation from analysis
- Investigate why treatment status is unknown
- Consider as separate category if meaningful

#### For Outcome Variable

**Generally do not impute outcomes.** Options:
- Complete case analysis with selection bias assessment
- Bounds analysis (Manski bounds)
- Inverse probability weighting for missingness

### Multiple Imputation for Causal Analysis

```python
from preprocessing import handle_missing

# Generate multiple imputed datasets
imputed_dfs = handle_missing(df, strategy='multiple', n_imputations=20)

# Run causal analysis on each dataset
effects = []
variances = []

for imp_df in imputed_dfs:
    # Estimate effect on this imputed dataset
    effect, var = estimate_treatment_effect(imp_df)
    effects.append(effect)
    variances.append(var)

# Pool results using Rubin's rules
pooled_effect = np.mean(effects)
within_var = np.mean(variances)
between_var = np.var(effects, ddof=1)
total_var = within_var + (1 + 1/len(effects)) * between_var
pooled_se = np.sqrt(total_var)
```

### Missing Indicators as Controls

When missingness may be informative (correlated with potential outcomes):

```python
from preprocessing import preprocess_for_causal

result = preprocess_for_causal(
    df=df,
    outcome='Y',
    treatment='D',
    controls=['X1', 'X2', 'X3'],
    missing_strategy='mean',
    create_missing_indicators=True  # Creates X1_missing, etc.
)
```

**Caution**: Only include missing indicators for pre-treatment variables.

---

## Outliers

### Detection Methods

#### Univariate Methods

| Method | Formula | Best For |
|--------|---------|----------|
| IQR | Outside [Q1 - 1.5*IQR, Q3 + 1.5*IQR] | Non-normal distributions |
| Z-score | \|z\| > 3 | Normal distributions |
| Modified Z-score | Uses MAD instead of SD | Robust to outliers |

```python
from preprocessing import detect_outliers

# IQR method
outliers_iqr = detect_outliers(df, columns=['income'], method='iqr', threshold=1.5)

# Z-score method
outliers_zscore = detect_outliers(df, columns=['income'], method='zscore', threshold=3)
```

#### Multivariate Methods

| Method | Approach | Best For |
|--------|----------|----------|
| Mahalanobis distance | Distance from centroid considering correlations | Correlated variables |
| Isolation Forest | Tree-based anomaly detection | High-dimensional data |
| LOF (Local Outlier Factor) | Density-based detection | Non-uniform density |

```python
# Multivariate outlier detection
outliers_multi = detect_outliers(
    df,
    columns=['age', 'income', 'education'],
    method='isolation_forest',
    contamination=0.05
)
```

### Causal Considerations for Outliers

#### Should You Remove Outliers?

| Scenario | Recommendation |
|----------|----------------|
| Data entry errors | Remove after verification |
| Legitimate extreme values | Keep - they represent real variation |
| Different population | Consider separate analysis |
| Influential on treatment effect | Report sensitivity analysis |

#### Outliers and Treatment Effects

Removing outliers can change your estimand:
- **With outliers**: Effect on full population
- **Without outliers**: Effect on "typical" units (undefined population)

```python
# Sensitivity analysis for outlier removal
from preprocessing import remove_outliers

results = {}

# Analysis with all data
results['all_data'] = estimate_effect(df)

# Analysis without outliers (various thresholds)
for threshold in [1.5, 2.0, 2.5, 3.0]:
    df_clean, report = remove_outliers(df, controls, method='iqr', threshold=threshold)
    results[f'iqr_{threshold}'] = estimate_effect(df_clean)

# Report sensitivity
print("Sensitivity to outlier removal:")
for key, effect in results.items():
    print(f"  {key}: {effect:.3f}")
```

#### Winsorization as Alternative

Instead of removing outliers, cap extreme values:

```python
def winsorize(df, column, lower_pct=0.01, upper_pct=0.99):
    """Cap values at specified percentiles."""
    lower = df[column].quantile(lower_pct)
    upper = df[column].quantile(upper_pct)
    df[column] = df[column].clip(lower, upper)
    return df
```

**Advantage**: Retains observations while reducing influence of extreme values.

---

## Duplicates

### Types of Duplicates

| Type | Description | Action |
|------|-------------|--------|
| Exact duplicates | All columns identical | Usually remove |
| Key duplicates | Same ID, different values | Investigate data quality |
| Near duplicates | Very similar but not identical | Domain-specific decision |

### Detection

```python
# Exact duplicates
exact_dupes = df[df.duplicated(keep=False)]

# Key duplicates (same unit, different observations)
key_dupes = df[df.duplicated(subset=['unit_id', 'time'], keep=False)]

# Near duplicates (fuzzy matching)
# Requires domain-specific similarity metrics
```

### Causal Implications of Duplicates

- **Exact duplicates**: Can artificially inflate sample size and reduce standard errors
- **Key duplicates**: May indicate data quality issues or panel structure
- **Treatment duplicates**: Same unit treated multiple times - need panel/repeated measures methods

### Handling Strategy

```python
def handle_duplicates(df, key_cols, strategy='first'):
    """
    Handle duplicate observations.

    Parameters
    ----------
    strategy : str
        'first': Keep first occurrence
        'last': Keep last occurrence
        'mean': Average numeric values (for repeated measures)
        'flag': Add indicator for duplicated observations
    """
    if strategy in ['first', 'last']:
        return df.drop_duplicates(subset=key_cols, keep=strategy)
    elif strategy == 'mean':
        return df.groupby(key_cols).mean().reset_index()
    elif strategy == 'flag':
        df['is_duplicate'] = df.duplicated(subset=key_cols, keep=False)
        return df
```

---

## Best Practices Summary

### Pre-Analysis Checklist

1. **Document raw data quality**
   - Missing value rates by variable
   - Outlier prevalence
   - Duplicate counts

2. **Understand mechanisms**
   - Why are values missing?
   - Are outliers errors or real?
   - Why do duplicates exist?

3. **Create reproducible cleaning pipeline**
   - Version control cleaning code
   - Log all decisions
   - Make decisions before seeing results

4. **Report sensitivity**
   - Main analysis with chosen cleaning
   - Alternative specifications
   - Bounds under different assumptions

### Common Mistakes to Avoid

| Mistake | Why It's Problematic | Better Approach |
|---------|---------------------|-----------------|
| Imputing outcomes | Attenuates treatment effects | Complete case or bounds |
| Removing "inconvenient" outliers | Introduces selection bias | Pre-specified rules |
| Ignoring missingness patterns | May indicate bias | Test and document |
| Different cleaning by treatment group | Violates exchangeability | Same rules for all |

---

## References

- Rubin, D. B. (1987). Multiple Imputation for Nonresponse in Surveys. Wiley.
- Little, R. J., & Rubin, D. B. (2019). Statistical Analysis with Missing Data (3rd ed.). Wiley.
- van Buuren, S. (2018). Flexible Imputation of Missing Data (2nd ed.). CRC Press.
- Barnard, J., & Meng, X. L. (1999). Applications of multiple imputation in medical studies. Statistical Methods in Medical Research, 8(1), 17-36.
