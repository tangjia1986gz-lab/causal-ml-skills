# PSM Diagnostic Tests

> **Reference Document** | Propensity Score Matching
> Based on Stuart (2010), Imbens & Rubin (2015)

## Overview

Diagnostic tests are essential for validating PSM analysis. This document covers balance tests, overlap assessment, and covariate importance diagnostics.

---

## 1. Balance Tests

Balance tests verify that matching achieves its primary goal: creating comparable treatment and control groups.

### 1.1 Standardized Mean Difference (SMD)

**Definition**:
$$
SMD = \frac{\bar{X}_T - \bar{X}_C}{\sqrt{(S_T^2 + S_C^2)/2}}
$$

Where:
- $\bar{X}_T$: Mean of covariate for treated group
- $\bar{X}_C$: Mean of covariate for control group
- $S_T^2$, $S_C^2$: Variances for treated and control groups

**Thresholds**:

| |SMD| Value | Interpretation | Action |
|-------------|----------------|--------|
| < 0.1 | Excellent balance | Proceed with analysis |
| 0.1 - 0.25 | Acceptable balance | May proceed with caution |
| > 0.25 | Poor balance | Re-specify PS model or matching |

**Calculation**:

```python
def calculate_smd(x_treated, x_control, weights_t=None, weights_c=None):
    """
    Calculate Standardized Mean Difference.

    Parameters
    ----------
    x_treated : array-like
        Covariate values for treated units
    x_control : array-like
        Covariate values for control units
    weights_t : array-like, optional
        Weights for treated units
    weights_c : array-like, optional
        Weights for control units

    Returns
    -------
    float
        Standardized mean difference
    """
    import numpy as np

    if weights_t is None:
        weights_t = np.ones(len(x_treated))
    if weights_c is None:
        weights_c = np.ones(len(x_control))

    # Weighted means
    mean_t = np.average(x_treated, weights=weights_t)
    mean_c = np.average(x_control, weights=weights_c)

    # Weighted variances
    var_t = np.average((x_treated - mean_t)**2, weights=weights_t)
    var_c = np.average((x_control - mean_c)**2, weights=weights_c)

    # Pooled standard deviation
    pooled_std = np.sqrt((var_t + var_c) / 2)

    if pooled_std == 0:
        return 0.0

    return (mean_t - mean_c) / pooled_std
```

### 1.2 Variance Ratio

**Definition**:
$$
VR = \frac{S_T^2}{S_C^2}
$$

**Thresholds**:

| VR Value | Interpretation |
|----------|---------------|
| 0.5 - 2.0 | Acceptable |
| < 0.5 or > 2.0 | Variance imbalance |

**Why It Matters**:
- SMD only captures mean differences
- Variance differences can bias effect estimates
- Important for continuous outcomes

### 1.3 Kolmogorov-Smirnov Test

Tests whether two distributions are identical.

**Null Hypothesis**: Distributions are identical
**Alternative**: Distributions differ

**Interpretation**:
- p > 0.05: Fail to reject null, distributions similar
- p < 0.05: Reject null, distributions differ

**Limitation**: Very sensitive with large samples

### 1.4 Overall Balance Measures

**Mean Absolute SMD**:
$$
\text{Mean}|SMD| = \frac{1}{K}\sum_{k=1}^{K}|SMD_k|
$$

**Max Absolute SMD**:
$$
\text{Max}|SMD| = \max_k |SMD_k|
$$

**Percentage of Balanced Covariates**:
$$
\text{Pct Balanced} = \frac{\#\{|SMD_k| < 0.1\}}{K} \times 100\%
$$

---

## 2. Overlap Assessment

### 2.1 Propensity Score Distribution Comparison

**Visual Methods**:

1. **Histograms**: Side-by-side PS histograms for treatment groups
2. **Density Plots**: Overlaid kernel density estimates
3. **Box Plots**: Compare PS distributions
4. **Mirror Histograms**: Treatment above, control below axis

```
Propensity Score Distribution
+----------------------------------------+
|                                        |
|  Treated  |         *****              |
|           |     *********              |
|           | ****************           |
|  ---------|----------------------      |
|           | *********                  |
|           | **************             |
|  Control  |   ********                 |
|                                        |
|           0    0.25   0.5   0.75    1  |
+----------------------------------------+
```

### 2.2 Common Support Statistics

| Metric | Definition | Target |
|--------|------------|--------|
| Min PS (Treated) | Minimum PS among treated | > 0.05 |
| Max PS (Treated) | Maximum PS among treated | < 0.95 |
| Min PS (Control) | Minimum PS among control | > 0.05 |
| Max PS (Control) | Maximum PS among control | < 0.95 |
| Overlap Region | [max(min_T,min_C), min(max_T,max_C)] | Wide |
| Pct in Support | % of units in overlap region | > 90% |

### 2.3 Overlap Index

$$
\text{Overlap Index} = \frac{\text{Common Support Range}}{\text{Total PS Range}}
$$

**Interpretation**:
- 1.0: Perfect overlap
- 0.5: Moderate overlap
- < 0.3: Poor overlap

### 2.4 Overlap Visualization Code

```python
def plot_overlap_diagnostics(ps_treated, ps_control, figsize=(12, 8)):
    """
    Create comprehensive overlap diagnostic plots.

    Parameters
    ----------
    ps_treated : array-like
        Propensity scores for treated units
    ps_control : array-like
        Propensity scores for control units
    figsize : tuple
        Figure size

    Returns
    -------
    matplotlib.figure.Figure
        Diagnostic plots
    """
    import matplotlib.pyplot as plt
    import numpy as np

    fig, axes = plt.subplots(2, 2, figsize=figsize)

    # 1. Histograms
    ax = axes[0, 0]
    bins = np.linspace(0, 1, 50)
    ax.hist(ps_control, bins=bins, alpha=0.6, label='Control',
            color='blue', density=True)
    ax.hist(ps_treated, bins=bins, alpha=0.6, label='Treated',
            color='red', density=True)
    ax.set_xlabel('Propensity Score')
    ax.set_ylabel('Density')
    ax.set_title('PS Distribution')
    ax.legend()

    # 2. Mirror histogram
    ax = axes[0, 1]
    ax.hist(ps_treated, bins=bins, alpha=0.6, label='Treated',
            color='red', density=True)
    ax.hist(ps_control, bins=bins, alpha=0.6, label='Control',
            color='blue', density=True, bottom=-np.histogram(ps_control,
            bins=bins, density=True)[0])
    ax.axhline(y=0, color='black', linewidth=0.5)
    ax.set_xlabel('Propensity Score')
    ax.set_title('Mirror Histogram')

    # 3. Box plots
    ax = axes[1, 0]
    data = [ps_control, ps_treated]
    bp = ax.boxplot(data, labels=['Control', 'Treated'], patch_artist=True)
    bp['boxes'][0].set_facecolor('lightblue')
    bp['boxes'][1].set_facecolor('lightcoral')
    ax.set_ylabel('Propensity Score')
    ax.set_title('PS Box Plot')

    # 4. Cumulative distributions
    ax = axes[1, 1]
    sorted_c = np.sort(ps_control)
    sorted_t = np.sort(ps_treated)
    ax.plot(sorted_c, np.arange(len(sorted_c))/len(sorted_c),
            label='Control', color='blue')
    ax.plot(sorted_t, np.arange(len(sorted_t))/len(sorted_t),
            label='Treated', color='red')
    ax.set_xlabel('Propensity Score')
    ax.set_ylabel('Cumulative Probability')
    ax.set_title('Empirical CDF')
    ax.legend()

    plt.tight_layout()
    return fig
```

---

## 3. Covariate Importance Diagnostics

### 3.1 Feature Importance in PS Model

For machine learning PS estimation, examine feature importance:

| PS Method | Importance Measure |
|-----------|-------------------|
| Logistic Regression | Coefficient magnitude |
| LASSO | Non-zero coefficients |
| Random Forest | Gini importance |
| GBM | Split importance |

**Warning Signs**:
- Single covariate dominates importance
- Importance > 0.5 for any variable
- May indicate deterministic treatment assignment

### 3.2 Propensity Score Discrimination

**AUC (Area Under ROC Curve)**:

| AUC Value | Interpretation |
|-----------|---------------|
| 0.5 | Random (no discrimination) |
| 0.6-0.7 | Poor discrimination |
| 0.7-0.8 | Acceptable |
| 0.8-0.9 | Good |
| > 0.9 | Excellent (may indicate overlap issues) |

**Paradox**: Very high AUC suggests strong selection, which may indicate overlap problems or near-deterministic assignment.

### 3.3 Prognostic Score Comparison

Compare covariate importance for:
1. Predicting treatment (propensity score)
2. Predicting outcome (prognostic score)

**Key Insight**: Variables important for both are confounders that must be included.

---

## 4. Post-Matching Diagnostics

### 4.1 Love Plot

Visual comparison of balance before and after matching.

```
Love Plot: Covariate Balance
+------------------------------------------------+
|                                                |
| Variable                                       |
|                                                |
| age         o------------->*                   |
| income      o------->*                         |
| education   o--->*                             |
| employed    o------>*                          |
| married     o-->*                              |
|                                                |
|             |    |    |    |    |              |
|            0.0  0.1  0.2  0.3  0.4             |
|                |SMD|                           |
|                                                |
| o = Before matching   * = After matching       |
| Dashed line = 0.1 threshold                    |
+------------------------------------------------+
```

### 4.2 Effective Sample Size

For weighted analyses, the effective sample size accounts for weight variability:

$$
N_{eff} = \frac{(\sum_i w_i)^2}{\sum_i w_i^2}
$$

**Interpretation**:
- $N_{eff} = N$: All weights equal
- $N_{eff} << N$: Highly variable weights, less precision

### 4.3 Match Quality Statistics

| Metric | Definition | Target |
|--------|------------|--------|
| Mean PS Distance | Average |PS_T - PS_C| for matches | < 0.1 |
| Max PS Distance | Maximum |PS_T - PS_C| | < 0.25 |
| Percent Within Caliper | % matches within caliper | 100% |
| Match Ratio | Controls / Treated | Depends on method |

---

## 5. Diagnostic Checklist

```
PSM DIAGNOSTIC CHECKLIST
========================

BEFORE MATCHING:
[ ] Propensity score model fit (AUC, coefficients)
[ ] PS distribution overlap (visual inspection)
[ ] Common support region identified
[ ] Extreme PS values flagged

AFTER MATCHING:
[ ] Balance table (SMD for all covariates)
[ ] All |SMD| < 0.1 (or < 0.25 minimum)
[ ] Variance ratios between 0.5 and 2.0
[ ] Love plot created
[ ] Effective sample size calculated
[ ] Match quality (PS distance) reported

SENSITIVITY:
[ ] Rosenbaum bounds computed
[ ] Critical Gamma reported
[ ] Interpretation of sensitivity provided
```

---

## 6. Common Issues and Solutions

### Issue: Imbalance Persists After Matching

**Possible Causes**:
1. Poor PS model specification
2. Limited overlap
3. Insufficient matching caliper

**Solutions**:
1. Add interaction terms to PS model
2. Use machine learning for PS estimation
3. Try different matching methods
4. Exact match on key confounders
5. Trim sample to common support

### Issue: High Variance Weights

**Possible Causes**:
1. Extreme propensity scores
2. IPW with thin tails

**Solutions**:
1. Trim propensity scores
2. Use matching instead of IPW
3. Use doubly robust estimation

### Issue: Many Unmatched Units

**Possible Causes**:
1. Strict caliper
2. Matching without replacement
3. Poor overlap

**Solutions**:
1. Relax caliper (with caution)
2. Allow replacement
3. Use kernel matching
4. Report on unmatched population

---

## References

- Stuart, E. A. (2010). Matching Methods for Causal Inference: A Review and a Look Forward. *Statistical Science*, 25(1), 1-21.
- Imbens, G. W., & Rubin, D. B. (2015). *Causal Inference for Statistics, Social, and Biomedical Sciences*. Cambridge University Press.
- Austin, P. C. (2009). Balance Diagnostics for Comparing the Distribution of Baseline Covariates Between Treatment Groups. *Statistics in Medicine*, 28, 3083-3107.
- Ho, D. E., Imai, K., King, G., & Stuart, E. A. (2007). Matching as Nonparametric Preprocessing for Reducing Model Dependence. *Political Analysis*, 15(3), 199-236.
