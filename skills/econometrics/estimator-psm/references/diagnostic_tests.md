# PSM Diagnostic Tests

## 1. Covariate Balance Assessment

### Standardized Mean Difference (SMD)

**Definition:**
```
SMD = (X̄_treated - X̄_control) / √[(σ²_T + σ²_C) / 2]
```

**Thresholds:**
| SMD | Interpretation |
|:---:|----------------|
| < 0.1 | Good balance ✅ |
| 0.1 - 0.25 | Acceptable with caution ⚠️ |
| > 0.25 | Poor balance ❌ |

**Python:**
```python
def smd(treated, control, weights=None):
    if weights is not None:
        control_mean = np.average(control, weights=weights)
    else:
        control_mean = control.mean()
    pooled_std = np.sqrt((treated.var() + control.var()) / 2)
    return (treated.mean() - control_mean) / pooled_std
```

### Variance Ratio

**Definition:**
```
VR = σ²_treated / σ²_control
```

**Threshold:** 0.5 < VR < 2.0 indicates acceptable balance

### Kolmogorov-Smirnov Test

Tests whether distributions are identical:

```python
from scipy.stats import ks_2samp
stat, pval = ks_2samp(treated_var, control_var)
```

**Interpretation:** High p-value suggests similar distributions

## 2. Propensity Score Overlap

### Visual Check

```python
import matplotlib.pyplot as plt

plt.hist(ps[T==0], alpha=0.5, label='Control', bins=50)
plt.hist(ps[T==1], alpha=0.5, label='Treated', bins=50)
plt.legend()
plt.xlabel('Propensity Score')
```

**Look for:** Substantial overlap in distributions

### Numeric Check

```python
# Common support
ps_min = max(ps[T==1].min(), ps[T==0].min())
ps_max = min(ps[T==1].max(), ps[T==0].max())

# Observations outside common support
n_outside = ((ps < ps_min) | (ps > ps_max)).sum()
```

### Trimming Rules

| Rule | Threshold |
|------|-----------|
| Conservative | 0.1 < ps < 0.9 |
| Moderate | 0.05 < ps < 0.95 |
| Liberal | 0.01 < ps < 0.99 |

## 3. Balance Table

Standard output format:

| Variable | Mean (T) | Mean (C) | SMD (Raw) | SMD (Adj) | Balanced? |
|----------|:--------:|:--------:|:---------:|:---------:|:---------:|
| Age | 45.2 | 38.1 | 0.58 | 0.03 | ✅ |
| Income | 52,000 | 41,000 | 0.45 | 0.08 | ✅ |
| Education | 14.2 | 12.8 | 0.42 | 0.15 | ⚠️ |

## 4. Love Plot

Visualizes balance improvement:

```python
def love_plot(balance_df, save_path=None):
    fig, ax = plt.subplots(figsize=(10, 6))

    y_pos = range(len(balance_df))

    # Unadjusted
    ax.scatter(balance_df['SMD_raw'].abs(), y_pos,
               marker='o', color='red', label='Unadjusted')

    # Adjusted
    ax.scatter(balance_df['SMD_adj'].abs(), y_pos,
               marker='s', color='blue', label='Adjusted')

    # Reference line
    ax.axvline(x=0.1, color='green', linestyle='--')

    ax.set_xlabel('|SMD|')
    ax.legend()
```

## 5. Prognostic Score Balance

Balance on predicted outcome (more stringent):

```python
# Fit outcome model on control
outcome_model = LinearRegression()
outcome_model.fit(X[T==0], Y[T==0])

# Predict for all
prog_score = outcome_model.predict(X)

# Check balance on prognostic score
smd_prog = smd(prog_score[T==1], prog_score[T==0])
```

**Rationale:** Balance on prognostic score implies balance on all outcome-relevant covariates.

## 6. Sensitivity Analysis

### Rosenbaum Bounds

Tests robustness to unmeasured confounding.

**Gamma (Γ):** Odds ratio of differential treatment assignment due to hidden bias.

| Γ | Interpretation |
|---|----------------|
| 1.0 | No hidden bias |
| 1.5 | Moderate sensitivity |
| 2.0 | Substantial robustness |
| >2.5 | Very robust finding |

**Report:** "Effect remains significant for Γ up to [value]"

```python
def rosenbaum_bounds(treated_Y, control_Y, gamma_range=[1, 1.5, 2, 2.5, 3]):
    """
    Compute Rosenbaum bounds using Wilcoxon signed-rank test.
    """
    results = []
    diffs = treated_Y - control_Y
    n = len(diffs)

    for gamma in gamma_range:
        # Compute bounds on p-value under hidden bias
        p_upper = gamma / (1 + gamma)
        # ... (see full implementation in scripts/)
        results.append({'Gamma': gamma, 'p_value_upper': pval})

    return pd.DataFrame(results)
```

### Oster Coefficient Stability

Tests how much selection on unobservables would be needed:

```python
# Delta = ratio of selection on unobservables to observables
# needed to explain away the effect
delta = (beta_full - beta_restricted) / (beta_restricted - 0)
```

**Threshold:** δ > 1 suggests robustness

## 7. Specification Tests

### C-statistic (AUC)

Propensity score model discrimination:

```python
from sklearn.metrics import roc_auc_score
auc = roc_auc_score(T, ps)
```

**Guideline:**
| AUC | Interpretation |
|:---:|----------------|
| 0.5 | No discrimination (random) |
| 0.7 | Acceptable |
| 0.8 | Good |
| > 0.9 | May indicate lack of overlap |

### Hosmer-Lemeshow Test

Tests propensity score calibration:

```python
from scipy.stats import chi2

def hosmer_lemeshow(y, p, g=10):
    """
    Test if predicted probabilities match observed proportions.
    """
    # Bin predictions
    bins = pd.qcut(p, g, duplicates='drop')
    # Compare observed vs expected in each bin
    # ...
    return chi2_stat, p_value
```

## Diagnostic Checklist

Before reporting PSM results:

- [ ] SMD < 0.1 for all covariates
- [ ] Propensity score overlap adequate
- [ ] Love plot shows improvement
- [ ] No extreme weights (if IPW)
- [ ] Sensitivity analysis conducted
- [ ] Match rate reported (if matching)

## Key References

1. Austin (2009): "Balance diagnostics for comparing the distribution of baseline covariates"
2. Rosenbaum (2002): "Observational Studies" (sensitivity analysis)
3. Oster (2019): "Unobservable Selection and Coefficient Stability"
4. Stuart (2010): "Matching methods for causal inference: A review"
