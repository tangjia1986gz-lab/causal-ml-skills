---
name: estimator-psm
description: Propensity Score Matching (PSM) and related methods for causal inference under selection on observables. Use when treatment assignment depends on observed covariates. Provides matching, weighting (IPW, AIPW), and balance diagnostics via scikit-learn and causalinference.
license: MIT
metadata:
    skill-author: Causal-ML-Skills
---

# Propensity Score Methods: Causal Inference with Selection on Observables

## Overview

Propensity Score Methods are a family of techniques for estimating causal effects when treatment assignment is not random but depends on observed covariates. The core idea is to balance treated and control groups by matching or weighting on the probability of treatment (the propensity score), thereby mimicking a randomized experiment.

This skill provides comprehensive guidance for implementing rigorous propensity score analyses in Python, from basic nearest-neighbor matching to modern doubly robust estimators (AIPW), with proper balance diagnostics and sensitivity analysis.

## When to Use This Skill

This skill should be used when:

- Estimating causal effects from observational data
- Treatment assignment depends on observed covariates (selection on observables)
- You have rich covariate data to control for confounding
- Implementing Propensity Score Matching (PSM)
- Using Inverse Probability Weighting (IPW)
- Applying Doubly Robust / AIPW estimators
- Checking covariate balance after matching
- Conducting sensitivity analysis for unmeasured confounding

**Do NOT use this skill when:**
- Treatment depends on unobserved factors (use IV instead)
- You have a natural experiment or RCT (simpler methods suffice)
- You have panel data with staggered treatment (use DID)

## Quick Start Guide

### Basic Propensity Score Matching

```python
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
from scipy import stats

# Example: Effect of job training on earnings
# Treatment: participated in training program
# Outcome: post-training earnings
# Covariates: age, education, prior earnings

# Load data
df = pd.read_csv('data.csv')

# Define variables
treatment = 'training'
outcome = 'earnings'
covariates = ['age', 'education', 'prior_earnings', 'married', 'black']

# ============================================
# 1. ESTIMATE PROPENSITY SCORE
# ============================================
X = df[covariates]
y = df[treatment]

# Logistic regression for propensity score
ps_model = LogisticRegression(max_iter=1000, random_state=42)
ps_model.fit(X, y)
df['pscore'] = ps_model.predict_proba(X)[:, 1]

print("Propensity Score Summary:")
print(f"  Treated mean:   {df.loc[df[treatment]==1, 'pscore'].mean():.4f}")
print(f"  Control mean:   {df.loc[df[treatment]==0, 'pscore'].mean():.4f}")

# ============================================
# 2. CHECK OVERLAP (COMMON SUPPORT)
# ============================================
treated_ps = df.loc[df[treatment]==1, 'pscore']
control_ps = df.loc[df[treatment]==0, 'pscore']

# Trim to common support
ps_min = max(treated_ps.min(), control_ps.min())
ps_max = min(treated_ps.max(), control_ps.max())

df_trimmed = df[(df['pscore'] >= ps_min) & (df['pscore'] <= ps_max)]
print(f"\nCommon support: [{ps_min:.4f}, {ps_max:.4f}]")
print(f"Observations trimmed: {len(df) - len(df_trimmed)}")

# ============================================
# 3. NEAREST NEIGHBOR MATCHING
# ============================================
treated = df_trimmed[df_trimmed[treatment] == 1]
control = df_trimmed[df_trimmed[treatment] == 0]

# Fit nearest neighbors on control group
nn = NearestNeighbors(n_neighbors=1, metric='euclidean')
nn.fit(control[['pscore']])

# Find matches for treated
distances, indices = nn.kneighbors(treated[['pscore']])

# Get matched control observations
matched_control_idx = control.iloc[indices.flatten()].index
matched_control = df_trimmed.loc[matched_control_idx]

# ============================================
# 4. ESTIMATE ATT (Average Treatment Effect on Treated)
# ============================================
att = treated[outcome].mean() - matched_control[outcome].mean()

# Bootstrap standard error
n_boot = 1000
boot_atts = []
for _ in range(n_boot):
    boot_idx = np.random.choice(len(treated), size=len(treated), replace=True)
    boot_treated = treated.iloc[boot_idx]
    boot_control = matched_control.iloc[boot_idx]
    boot_att = boot_treated[outcome].mean() - boot_control[outcome].mean()
    boot_atts.append(boot_att)

att_se = np.std(boot_atts)
att_ci = (att - 1.96*att_se, att + 1.96*att_se)

print(f"\n{'='*50}")
print("PROPENSITY SCORE MATCHING RESULTS")
print(f"{'='*50}")
print(f"ATT estimate: {att:.4f}")
print(f"Standard error: {att_se:.4f}")
print(f"95% CI: [{att_ci[0]:.4f}, {att_ci[1]:.4f}]")
print(f"Matched pairs: {len(treated)}")
```

### Inverse Probability Weighting (IPW)

```python
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
import statsmodels.api as sm

# Load data and estimate propensity scores
df = pd.read_csv('data.csv')
covariates = ['age', 'education', 'prior_earnings', 'married']
treatment = 'training'
outcome = 'earnings'

# Propensity score
ps_model = LogisticRegression(max_iter=1000)
ps_model.fit(df[covariates], df[treatment])
df['pscore'] = ps_model.predict_proba(df[covariates])[:, 1]

# Trim extreme propensity scores (important for stability)
df = df[(df['pscore'] > 0.01) & (df['pscore'] < 0.99)]

# ============================================
# IPW WEIGHTS
# ============================================
# For ATT: weight control by ps/(1-ps), treated weight = 1
df['ipw_att'] = np.where(
    df[treatment] == 1,
    1,
    df['pscore'] / (1 - df['pscore'])
)

# For ATE: weight by 1/ps for treated, 1/(1-ps) for control
df['ipw_ate'] = np.where(
    df[treatment] == 1,
    1 / df['pscore'],
    1 / (1 - df['pscore'])
)

# ============================================
# IPW ESTIMATOR (ATT)
# ============================================
# Weighted mean difference
treated_mean = df.loc[df[treatment]==1, outcome].mean()
control_weighted_mean = np.average(
    df.loc[df[treatment]==0, outcome],
    weights=df.loc[df[treatment]==0, 'ipw_att']
)
att_ipw = treated_mean - control_weighted_mean

# ============================================
# IPW ESTIMATOR (ATE) - Horvitz-Thompson
# ============================================
n = len(df)
ate_ipw = (
    (df[treatment] * df[outcome] / df['pscore']).sum() / n -
    ((1 - df[treatment]) * df[outcome] / (1 - df['pscore'])).sum() / n
)

# ============================================
# VARIANCE ESTIMATION (Bootstrap)
# ============================================
def ipw_att_estimate(data):
    t = data[treatment]
    y = data[outcome]
    ps = data['pscore']
    w = np.where(t == 1, 1, ps / (1 - ps))
    return y[t==1].mean() - np.average(y[t==0], weights=w[t==0])

n_boot = 1000
boot_atts = [ipw_att_estimate(df.sample(frac=1, replace=True)) for _ in range(n_boot)]
att_se = np.std(boot_atts)

print(f"IPW ATT: {att_ipw:.4f} (SE: {att_se:.4f})")
print(f"IPW ATE: {ate_ipw:.4f}")
```

### Doubly Robust Estimator (AIPW)

```python
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import cross_val_predict

# Load data
df = pd.read_csv('data.csv')
covariates = ['age', 'education', 'prior_earnings', 'married']
treatment = 'training'
outcome = 'earnings'

X = df[covariates]
T = df[treatment]
Y = df[outcome]

# ============================================
# 1. PROPENSITY SCORE MODEL
# ============================================
ps_model = LogisticRegression(max_iter=1000)
# Cross-fitted propensity scores (reduces overfitting bias)
ps = cross_val_predict(ps_model, X, T, cv=5, method='predict_proba')[:, 1]
ps = np.clip(ps, 0.01, 0.99)  # Trim extreme values

# ============================================
# 2. OUTCOME MODELS
# ============================================
# Model for E[Y|X, T=1]
X_treated = X[T == 1]
Y_treated = Y[T == 1]
mu1_model = LinearRegression()
mu1_model.fit(X_treated, Y_treated)
mu1 = mu1_model.predict(X)  # Predict for all observations

# Model for E[Y|X, T=0]
X_control = X[T == 0]
Y_control = Y[T == 0]
mu0_model = LinearRegression()
mu0_model.fit(X_control, Y_control)
mu0 = mu0_model.predict(X)  # Predict for all observations

# ============================================
# 3. AIPW ESTIMATOR
# ============================================
n = len(df)

# AIPW for E[Y(1)]
aipw_y1 = (T * Y / ps + (1 - T / ps) * mu1).mean()

# AIPW for E[Y(0)]
aipw_y0 = ((1 - T) * Y / (1 - ps) + (1 - (1 - T) / (1 - ps)) * mu0).mean()

# ATE
ate_aipw = aipw_y1 - aipw_y0

# ============================================
# 4. ATT with AIPW
# ============================================
# ATT = E[Y(1) - Y(0) | T=1]
p_treated = T.mean()

att_aipw = (
    (T * (Y - mu0) / p_treated).mean() -
    ((1 - T) * ps * (Y - mu0) / ((1 - ps) * p_treated)).mean()
)

# ============================================
# 5. BOOTSTRAP INFERENCE
# ============================================
def aipw_ate(data, covars, treat, out):
    X = data[covars]
    T = data[treat]
    Y = data[out]

    # PS model
    ps_mod = LogisticRegression(max_iter=500)
    ps_mod.fit(X, T)
    ps = np.clip(ps_mod.predict_proba(X)[:, 1], 0.01, 0.99)

    # Outcome models
    mu1_mod = LinearRegression()
    mu1_mod.fit(X[T==1], Y[T==1])
    mu1 = mu1_mod.predict(X)

    mu0_mod = LinearRegression()
    mu0_mod.fit(X[T==0], Y[T==0])
    mu0 = mu0_mod.predict(X)

    # AIPW
    y1 = (T * Y / ps + (1 - T / ps) * mu1).mean()
    y0 = ((1 - T) * Y / (1 - ps) + (1 - (1 - T) / (1 - ps)) * mu0).mean()

    return y1 - y0

n_boot = 500
boot_ates = [aipw_ate(df.sample(frac=1, replace=True), covariates, treatment, outcome)
             for _ in range(n_boot)]
ate_se = np.std(boot_ates)

print(f"\n{'='*50}")
print("DOUBLY ROBUST (AIPW) RESULTS")
print(f"{'='*50}")
print(f"ATE estimate: {ate_aipw:.4f}")
print(f"Bootstrap SE: {ate_se:.4f}")
print(f"95% CI: [{ate_aipw - 1.96*ate_se:.4f}, {ate_aipw + 1.96*ate_se:.4f}]")
print(f"\nATT estimate: {att_aipw:.4f}")
```

### Covariate Balance Diagnostics

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def calculate_smd(treated, control, weights=None):
    """
    Calculate Standardized Mean Difference (SMD).
    SMD = (mean_treated - mean_control) / pooled_std
    Rule of thumb: |SMD| < 0.1 indicates good balance
    """
    if weights is not None:
        control_mean = np.average(control, weights=weights)
    else:
        control_mean = control.mean()

    treated_mean = treated.mean()
    pooled_std = np.sqrt((treated.var() + control.var()) / 2)

    if pooled_std == 0:
        return 0
    return (treated_mean - control_mean) / pooled_std

def balance_table(df, treatment, covariates, weights=None):
    """
    Generate balance table with SMD before and after matching/weighting.
    """
    results = []

    for var in covariates:
        treated = df.loc[df[treatment]==1, var]
        control = df.loc[df[treatment]==0, var]

        # Unadjusted SMD
        smd_unadj = calculate_smd(treated, control)

        # Adjusted SMD (with weights)
        if weights is not None:
            w = df.loc[df[treatment]==0, weights]
            smd_adj = calculate_smd(treated, control, weights=w)
        else:
            smd_adj = smd_unadj

        results.append({
            'Variable': var,
            'Treated Mean': treated.mean(),
            'Control Mean': control.mean(),
            'SMD (Unadj)': smd_unadj,
            'SMD (Adj)': smd_adj,
            'Balanced': abs(smd_adj) < 0.1
        })

    return pd.DataFrame(results)

def plot_balance(balance_df, save_path=None):
    """
    Love plot: visualize covariate balance.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    y_pos = range(len(balance_df))

    # Plot unadjusted SMD
    ax.scatter(balance_df['SMD (Unadj)'], y_pos, marker='o',
               label='Unadjusted', color='red', s=80)

    # Plot adjusted SMD
    ax.scatter(balance_df['SMD (Adj)'], y_pos, marker='s',
               label='Adjusted', color='blue', s=80)

    # Reference lines
    ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
    ax.axvline(x=-0.1, color='gray', linestyle='--', linewidth=0.5)
    ax.axvline(x=0.1, color='gray', linestyle='--', linewidth=0.5)

    # Fill balance region
    ax.axvspan(-0.1, 0.1, alpha=0.2, color='green', label='Balance zone')

    ax.set_yticks(y_pos)
    ax.set_yticklabels(balance_df['Variable'])
    ax.set_xlabel('Standardized Mean Difference')
    ax.set_title('Covariate Balance (Love Plot)')
    ax.legend(loc='upper right')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    return fig

# Usage example
balance = balance_table(df, treatment, covariates, weights='ipw_att')
print(balance.to_string(index=False))

# Check overall balance
n_balanced = (balance['Balanced']).sum()
print(f"\nBalanced covariates: {n_balanced}/{len(covariates)}")
```

### Sensitivity Analysis (Rosenbaum Bounds)

```python
import numpy as np
from scipy import stats

def rosenbaum_bounds(treated_outcomes, control_outcomes, gamma_range=None):
    """
    Compute Rosenbaum bounds for sensitivity to unmeasured confounding.

    Gamma: odds ratio of differential treatment assignment due to
           unmeasured confounding.
    Gamma = 1: no unmeasured confounding
    Gamma > 1: allows for unmeasured confounding

    Parameters
    ----------
    treated_outcomes : array-like
        Outcomes for matched treated units
    control_outcomes : array-like
        Outcomes for matched control units
    gamma_range : list
        Range of gamma values to test

    Returns
    -------
    dict
        Upper and lower bounds on p-values for each gamma
    """
    if gamma_range is None:
        gamma_range = [1.0, 1.1, 1.2, 1.3, 1.5, 2.0]

    # Matched pair differences
    diffs = np.array(treated_outcomes) - np.array(control_outcomes)
    n_pairs = len(diffs)

    # Signs of differences
    signs = np.sign(diffs)
    abs_diffs = np.abs(diffs)

    # Rank absolute differences
    ranks = stats.rankdata(abs_diffs)

    # Wilcoxon signed-rank statistic
    W = np.sum(ranks[signs > 0])

    results = []

    for gamma in gamma_range:
        # Under gamma, probability of positive sign ranges from
        # 1/(1+gamma) to gamma/(1+gamma)

        # Expected value and variance under H0 with gamma
        p_upper = gamma / (1 + gamma)
        p_lower = 1 / (1 + gamma)

        # Expected value of W under H0
        E_W = n_pairs * (n_pairs + 1) / 4

        # Variance bounds (approximation)
        V_W_upper = n_pairs * (n_pairs + 1) * (2*n_pairs + 1) / 24 * p_upper * (1 - p_upper) * 4
        V_W_lower = n_pairs * (n_pairs + 1) * (2*n_pairs + 1) / 24 * p_lower * (1 - p_lower) * 4

        # Z-scores
        if V_W_upper > 0:
            z_upper = (W - E_W) / np.sqrt(V_W_upper)
            p_upper_bound = 1 - stats.norm.cdf(z_upper)
        else:
            p_upper_bound = np.nan

        if V_W_lower > 0:
            z_lower = (W - E_W) / np.sqrt(V_W_lower)
            p_lower_bound = 1 - stats.norm.cdf(z_lower)
        else:
            p_lower_bound = np.nan

        results.append({
            'Gamma': gamma,
            'P-value (lower)': p_lower_bound,
            'P-value (upper)': p_upper_bound
        })

    return pd.DataFrame(results)

# Usage
bounds = rosenbaum_bounds(treated[outcome].values, matched_control[outcome].values)
print("\nRosenbaum Sensitivity Analysis:")
print(bounds.to_string(index=False))
print("\nInterpretation: Effect robust to unmeasured confounding up to Gamma where p-value > 0.05")
```

## Core Capabilities

### 1. Propensity Score Estimation

**Logistic Regression (default):**
```python
from sklearn.linear_model import LogisticRegression
ps_model = LogisticRegression(max_iter=1000)
ps = ps_model.fit(X, T).predict_proba(X)[:, 1]
```

**Gradient Boosting (flexible):**
```python
from sklearn.ensemble import GradientBoostingClassifier
ps_model = GradientBoostingClassifier(n_estimators=100, max_depth=3)
ps = ps_model.fit(X, T).predict_proba(X)[:, 1]
```

**Generalized Boosted Models (recommended):**
```python
# Use cross-validation to avoid overfitting
from sklearn.model_selection import cross_val_predict
ps = cross_val_predict(ps_model, X, T, cv=5, method='predict_proba')[:, 1]
```

### 2. Matching Methods

| Method | Description | When to Use |
|--------|-------------|-------------|
| **Nearest Neighbor** | Match to closest pscore | Simple, transparent |
| **Caliper Matching** | NN within caliper | Reduce bad matches |
| **Mahalanobis** | Match on covariate distance | When pscore alone insufficient |
| **CEM (Coarsened Exact)** | Exact match on binned covariates | Discrete covariates |

### 3. Weighting Methods

**IPW for ATT:**
```python
w = np.where(T == 1, 1, ps / (1 - ps))
```

**IPW for ATE:**
```python
w = np.where(T == 1, 1/ps, 1/(1-ps))
```

**Normalized weights (Hajek):**
```python
w_norm = w / w.sum()
```

### 4. Estimands

| Estimand | Definition | Weights |
|----------|------------|---------|
| **ATT** | E[Y(1)-Y(0)\|T=1] | Control weighted to treated |
| **ATE** | E[Y(1)-Y(0)] | Both groups weighted |
| **ATC** | E[Y(1)-Y(0)\|T=0] | Treated weighted to control |

## Common Workflows

### Workflow 1: Standard PSM Analysis

```
1. Data Preparation
   ├── Define treatment, outcome, covariates
   ├── Check for missing data
   └── Examine covariate distributions

2. Propensity Score Estimation
   ├── Fit logistic regression (or GBM)
   ├── Check propensity score overlap
   └── Trim extreme scores if needed

3. Matching
   ├── Choose matching method (NN, caliper)
   ├── Perform matching
   └── Check match quality

4. Balance Assessment
   ├── Calculate SMD for all covariates
   ├── Create Love plot
   └── If imbalanced, iterate on PS model

5. Effect Estimation
   ├── Calculate ATT/ATE
   ├── Bootstrap for SE
   └── Report confidence interval

6. Sensitivity Analysis
   ├── Rosenbaum bounds
   └── Interpret robustness to unmeasured confounding
```

### Workflow 2: Doubly Robust Analysis

```
1. Split sample (for honest inference)
   └── Or use cross-fitting

2. Estimate propensity scores
   ├── Cross-validated predictions
   └── Trim extreme values

3. Estimate outcome models
   ├── E[Y|X, T=1] model
   └── E[Y|X, T=0] model

4. Compute AIPW estimator
   ├── Combines PS and outcome models
   └── Consistent if either model correct

5. Bootstrap inference
   └── Account for estimation uncertainty
```

## Best Practices

### Propensity Score Estimation

1. **Include all confounders**: Variables affecting both treatment and outcome
2. **Don't include instruments**: Variables affecting only treatment
3. **Don't include colliders**: Variables affected by both treatment and outcome
4. **Check overlap**: Trim extreme propensity scores (< 0.01 or > 0.99)

### Matching

1. **Match without replacement** when possible (reduces variance)
2. **Use calipers**: 0.2 standard deviations of pscore is common
3. **Check balance**: SMD < 0.1 for all covariates
4. **Report match rate**: How many treated units are matched

### Inference

1. **Bootstrap SEs** for matching estimators
2. **Cluster bootstrap** if data is clustered
3. **Report both ATT and ATE** when meaningful
4. **Conduct sensitivity analysis** for unmeasured confounding

### Reporting

1. **Balance table**: SMD before and after
2. **Propensity score histogram**: Show overlap
3. **Love plot**: Visualize balance improvement
4. **Sample sizes**: N matched, N trimmed

## Reference Documentation

### references/identification_assumptions.md
- Conditional independence / Unconfoundedness
- Overlap / Common support
- SUTVA
- No anticipation

### references/estimation_methods.md
- Matching algorithms
- IPW and variants
- Doubly robust / AIPW
- Entropy balancing

### references/diagnostic_tests.md
- Standardized Mean Difference (SMD)
- Variance ratios
- KS test for distributions
- Prognostic score balance

### references/reporting_standards.md
- Balance tables (JAMA/AER format)
- Love plots
- Sensitivity analysis reporting

### references/common_errors.md
- Conditioning on post-treatment variables
- Ignoring overlap violations
- Not checking balance
- Using inappropriate estimand

## Common Pitfalls to Avoid

1. **Conditioning on post-treatment variables**: Only use pre-treatment covariates
2. **Ignoring overlap violations**: Check and address lack of common support
3. **Not checking balance**: Always verify SMD < 0.1 after matching
4. **Using wrong estimand**: ATT vs ATE depends on research question
5. **Overfitting propensity score**: Use cross-validation or regularization
6. **Ignoring model dependence**: Results shouldn't change with minor model changes
7. **Not conducting sensitivity analysis**: Always assess robustness to unmeasured confounding
8. **Forgetting to trim**: Extreme weights can cause instability
9. **Using propensity score as covariate**: Don't include pscore in outcome regression
10. **Matching then adjusting**: Don't double-adjust after matching
11. **Ignoring clustering**: Account for clustered data in inference
12. **Not reporting match rate**: High exclusion rates are concerning
13. **King & Nielsen critique**: PSM can increase imbalance - check!
14. **Assuming CI assumption**: Unconfoundedness is untestable
15. **Forgetting sample representativeness**: Matched sample may not be representative

## Troubleshooting

### Poor Overlap (Lack of Common Support)

**Issue:** Propensity score distributions don't overlap

**Solutions:**
```python
# 1. Trim extreme scores
df = df[(df['pscore'] > 0.05) & (df['pscore'] < 0.95)]

# 2. Use caliper matching
caliper = 0.2 * df['pscore'].std()

# 3. Consider different estimand (ATT may have better support than ATE)
```

### Imbalance After Matching

**Issue:** SMD > 0.1 for some covariates

**Solutions:**
```python
# 1. Add interaction terms to PS model
X['age_sq'] = X['age'] ** 2
X['age_edu'] = X['age'] * X['education']

# 2. Use more flexible PS model (GBM)
from sklearn.ensemble import GradientBoostingClassifier

# 3. Try different matching method (Mahalanobis, CEM)
```

### Extreme Weights in IPW

**Issue:** Some IPW weights are very large

**Solutions:**
```python
# 1. Trim weights
w = np.clip(w, 0.01, 100)

# 2. Normalize weights
w = w / w.sum() * len(w)

# 3. Use stabilized weights
sw = np.where(T == 1, T.mean() / ps, (1 - T.mean()) / (1 - ps))
```

### sklearn Import Error

**Solution:**
```bash
pip install scikit-learn pandas numpy scipy matplotlib
```

## Additional Resources

### Official Documentation
- scikit-learn: https://scikit-learn.org/
- causalinference: https://causalinferenceinpython.org/

### Key Papers
- Rosenbaum & Rubin (1983): "The Central Role of the Propensity Score"
- Hirano, Imbens & Ridder (2003): IPW estimator properties
- Bang & Robins (2005): Doubly Robust Estimation
- King & Nielsen (2019): "Why Propensity Scores Should Not Be Used for Matching"
- Imbens (2004): Nonparametric estimation of average treatment effects

### Textbooks
- Imbens & Rubin (2015): *Causal Inference for Statistics, Social, and Biomedical Sciences*
- Angrist & Pischke (2009): *Mostly Harmless Econometrics*, Ch. 3
- Morgan & Winship (2014): *Counterfactuals and Causal Inference*

## Installation

```bash
# Core packages
pip install scikit-learn pandas numpy scipy

# Visualization
pip install matplotlib seaborn

# Optional: specialized packages
pip install causalinference  # Dedicated causal inference package

# Full installation
pip install scikit-learn pandas numpy scipy matplotlib seaborn
```

## Related Skills

| Skill | When to Use Instead |
|-------|---------------------|
| `estimator-did` | Panel data with treatment timing |
| `estimator-iv` | Selection on unobservables |
| `estimator-rd` | Sharp cutoff in assignment variable |
| `causal-ddml` | High-dimensional covariates |
| `causal-forest` | Heterogeneous treatment effects |
