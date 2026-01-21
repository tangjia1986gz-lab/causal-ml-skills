---
name: panel-data-models
description: Panel data econometrics for longitudinal/cross-sectional data. Use when analyzing firm-year, person-wave, or country-time data. Provides Fixed Effects, Random Effects, Hausman test, dynamic panels (GMM), clustered standard errors via linearmodels package.
license: MIT
metadata:
    skill-author: Causal-ML-Skills
---

# Panel Data Models

## Overview

Panel data models analyze data with both cross-sectional (individual, firm, country) and time-series (year, quarter, wave) dimensions. These models exploit within-unit variation over time to control for unobserved heterogeneity, enabling more credible causal inference than cross-sectional data alone.

**Key advantage**: Panel data allows controlling for time-invariant unobserved confounders through fixed effects, addressing a major source of omitted variable bias.

**Primary packages**: `linearmodels` (recommended), `statsmodels`

## When to Use This Skill

This skill should be used when:

- Data has panel structure (repeated observations of same units over time)
- Need to control for unobserved time-invariant heterogeneity (firm culture, individual ability)
- Analyzing firm-year financial data
- Working with person-wave survey data (PSID, NLSY)
- Country-year macroeconomic analysis
- Estimating dynamic models with lagged dependent variables
- Need clustered standard errors for inference

**Do NOT use when:**
- Cross-sectional data only (single time period)
- Pure time series (single unit)
- Treatment effects estimation with staggered adoption (use `estimator-did`)
- Need instrumental variables with panel (use `estimator-iv` + panel)

## Quick Start Guide

### Basic Fixed Effects with linearmodels

```python
import pandas as pd
import numpy as np
from linearmodels.panel import PanelOLS, RandomEffects, compare

# Load panel data
df = pd.read_csv('firm_year_data.csv')

# CRITICAL: Set multi-index for panel structure
df = df.set_index(['firm_id', 'year'])

# Fixed Effects (within estimator)
# EntityEffects absorbs firm-specific intercepts
model_fe = PanelOLS.from_formula(
    'log_revenue ~ investment + rd_expense + employees + TimeEffects',
    data=df,
    drop_absorbed=True  # Needed when including TimeEffects
)

# Fit with clustered standard errors (cluster by firm)
result_fe = model_fe.fit(cov_type='clustered', cluster_entity=True)

print(result_fe.summary)

# Key outputs
print(f"\nR-squared (within): {result_fe.rsquared_within:.4f}")
print(f"R-squared (between): {result_fe.rsquared_between:.4f}")
print(f"R-squared (overall): {result_fe.rsquared_overall:.4f}")
```

### Random Effects Model

```python
from linearmodels.panel import RandomEffects

# Random Effects assumes unobserved effect uncorrelated with regressors
model_re = RandomEffects.from_formula(
    'log_revenue ~ investment + rd_expense + employees',
    data=df
)

result_re = model_re.fit()
print(result_re.summary)

# Random Effects variance decomposition
print(f"\nSigma_u (between variance): {result_re.variance_decomposition.Effects:.4f}")
print(f"Sigma_e (within variance): {result_re.variance_decomposition.Residual:.4f}")
```

### Hausman Test: FE vs RE

```python
from linearmodels.panel import PanelOLS, RandomEffects, compare

# Fit both models
fe_model = PanelOLS.from_formula('y ~ x1 + x2 + EntityEffects', data=df)
re_model = RandomEffects.from_formula('y ~ x1 + x2', data=df)

fe_result = fe_model.fit()
re_result = re_model.fit()

# Hausman test via coefficient comparison
# H0: RE is consistent (no correlation between effects and regressors)
# H1: FE is needed (effects correlated with regressors)

# Extract coefficients (exclude constant)
fe_coef = fe_result.params.drop('Intercept', errors='ignore')
re_coef = re_result.params.drop('Intercept', errors='ignore')

# Coefficient difference
diff = fe_coef - re_coef

# Variance of difference
fe_var = fe_result.cov.loc[diff.index, diff.index]
re_var = re_result.cov.loc[diff.index, diff.index]
var_diff = fe_var - re_var

# Hausman statistic
from scipy import stats
hausman_stat = float(diff.values @ np.linalg.inv(var_diff.values) @ diff.values)
df_hausman = len(diff)
p_value = 1 - stats.chi2.cdf(hausman_stat, df_hausman)

print(f"\nHausman Test:")
print(f"Chi-squared statistic: {hausman_stat:.4f}")
print(f"Degrees of freedom: {df_hausman}")
print(f"P-value: {p_value:.4f}")

if p_value < 0.05:
    print("Reject H0: Use Fixed Effects (effects correlated with regressors)")
else:
    print("Fail to reject H0: Random Effects is consistent")
```

### Two-Way Fixed Effects (Entity + Time)

```python
from linearmodels.panel import PanelOLS

# Two-way FE: control for both unit and time fixed effects
model_twfe = PanelOLS.from_formula(
    'outcome ~ treatment + control1 + control2 + EntityEffects + TimeEffects',
    data=df,
    drop_absorbed=True
)

result_twfe = model_twfe.fit(cov_type='clustered', cluster_entity=True)
print(result_twfe.summary)

# Check absorbed effects
print(f"\nAbsorbed effects: {result_twfe.absorbed_effects}")
```

### Dynamic Panel: Arellano-Bond GMM

```python
# For dynamic panels with lagged dependent variable
# Y_it = rho * Y_{i,t-1} + X_it * beta + alpha_i + epsilon_it

# Note: linearmodels doesn't have built-in Arellano-Bond
# Use pydynpd or manual implementation

# Manual first-difference approach with instruments
df_sorted = df.sort_index()

# Create lagged variables
df_sorted['y_lag1'] = df_sorted.groupby(level=0)['y'].shift(1)
df_sorted['y_lag2'] = df_sorted.groupby(level=0)['y'].shift(2)  # Instrument

# First difference to remove fixed effects
df_sorted['dy'] = df_sorted.groupby(level=0)['y'].diff()
df_sorted['dy_lag1'] = df_sorted.groupby(level=0)['y_lag1'].diff()
df_sorted['dx'] = df_sorted.groupby(level=0)['x'].diff()

# 2SLS on first differences (simple version)
from linearmodels.iv import IV2SLS

# Drop missing from differencing
df_diff = df_sorted.dropna(subset=['dy', 'dy_lag1', 'dx', 'y_lag2'])

# Arellano-Bond: use y_{t-2} as instrument for dy_{t-1}
model_ab = IV2SLS.from_formula(
    'dy ~ 1 + dx + [dy_lag1 ~ y_lag2]',
    data=df_diff.reset_index()
)

result_ab = model_ab.fit(cov_type='robust')
print(result_ab.summary)
```

### Clustered Standard Errors

```python
from linearmodels.panel import PanelOLS

model = PanelOLS.from_formula('y ~ x1 + x2 + EntityEffects', data=df)

# Different clustering options
# 1. Cluster by entity (firm)
result_cluster_entity = model.fit(cov_type='clustered', cluster_entity=True)

# 2. Cluster by time
result_cluster_time = model.fit(cov_type='clustered', cluster_time=True)

# 3. Two-way clustering (Cameron-Gelbach-Miller)
result_cluster_both = model.fit(cov_type='clustered', cluster_entity=True, cluster_time=True)

print("Standard Error Comparison:")
print(f"Clustered (entity):   {result_cluster_entity.std_errors['x1']:.4f}")
print(f"Clustered (time):     {result_cluster_time.std_errors['x1']:.4f}")
print(f"Two-way clustered:    {result_cluster_both.std_errors['x1']:.4f}")
```

## Core Capabilities

### 1. Fixed Effects Models

**Entity Fixed Effects**: Controls for time-invariant unobserved heterogeneity

| Aspect | Description |
|--------|-------------|
| Estimator | Within transformation: demean by entity |
| Controls for | Time-invariant confounders |
| Cannot estimate | Time-invariant regressors (absorbed) |
| Assumption | Strict exogeneity: E[epsilon_it \| X_i1,...,X_iT, alpha_i] = 0 |

**Time Fixed Effects**: Controls for common time shocks

| Aspect | Description |
|--------|-------------|
| Estimator | Include time dummies or demean by time |
| Controls for | Aggregate shocks affecting all units |
| Use when | Macroeconomic conditions, policy changes affect all |

### 2. Random Effects Models

| Aspect | Fixed Effects | Random Effects |
|--------|---------------|----------------|
| Assumption | alpha_i correlated with X | alpha_i uncorrelated with X |
| Estimator | Within | GLS (quasi-demeaning) |
| Efficiency | Less efficient | More efficient if valid |
| Time-invariant X | Cannot estimate | Can estimate |
| Test | - | Hausman test |

### 3. Standard Error Options

| Type | Use When | linearmodels syntax |
|------|----------|---------------------|
| Homoskedastic | Rare, baseline only | `cov_type='unadjusted'` |
| Heteroskedasticity-robust | Cross-sectional heteroskedasticity | `cov_type='robust'` |
| Clustered (entity) | Serial correlation within units | `cluster_entity=True` |
| Clustered (time) | Cross-sectional correlation | `cluster_time=True` |
| Two-way clustered | Both issues | Both `=True` |

### 4. Model Diagnostics

```python
def panel_diagnostics(result, df):
    """Comprehensive panel model diagnostics."""
    print("="*60)
    print("PANEL MODEL DIAGNOSTICS")
    print("="*60)

    # R-squared decomposition
    print("\n--- R-squared Decomposition ---")
    print(f"Within R²:  {result.rsquared_within:.4f}")
    print(f"Between R²: {result.rsquared_between:.4f}")
    print(f"Overall R²: {result.rsquared_overall:.4f}")

    # Panel structure
    n_entities = df.index.get_level_values(0).nunique()
    n_periods = df.index.get_level_values(1).nunique()
    print(f"\n--- Panel Structure ---")
    print(f"Entities: {n_entities}")
    print(f"Time periods: {n_periods}")
    print(f"Total observations: {len(df)}")
    print(f"Balanced: {len(df) == n_entities * n_periods}")

    # F-test for fixed effects
    if hasattr(result, 'f_statistic'):
        print(f"\n--- F-test for Fixed Effects ---")
        print(f"F-statistic: {result.f_statistic.stat:.4f}")
        print(f"P-value: {result.f_statistic.pval:.4f}")

    return result
```

## Common Workflows

### Workflow 1: Standard Panel Analysis

```
1. Data Preparation
   ├── Load data with entity and time identifiers
   ├── Set multi-index: df.set_index(['entity', 'time'])
   ├── Check for missing values
   └── Assess panel balance

2. Exploratory Analysis
   ├── Within vs between variation
   ├── Time trends by entity
   └── Summary statistics by group

3. Model Estimation
   ├── Start with Pooled OLS (baseline)
   ├── Estimate Fixed Effects
   ├── Estimate Random Effects
   └── Hausman test for model selection

4. Inference
   ├── Use clustered standard errors
   ├── Report multiple specifications
   └── Test joint significance

5. Robustness
   ├── Different clustering levels
   ├── Include/exclude time effects
   └── Subsamples (by time period, entity type)

6. Reporting
   └── Multi-column table with FE, RE, TWFE
```

### Workflow 2: Dynamic Panel Analysis

```
1. Check for Dynamics
   ├── Autocorrelation in residuals
   ├── Lagged dependent variable significance
   └── Nickell bias concern (short T)

2. Estimation Strategy
   ├── If T is large: FE with lagged Y
   ├── If T is small: Arellano-Bond GMM
   └── System GMM for persistent series

3. Diagnostics
   ├── Sargan/Hansen test (overidentification)
   ├── AR(1)/AR(2) in differences
   └── Instrument count vs observations

4. Reporting
   └── Dynamic coefficient + adjustment speed
```

## Best Practices

### Data Preparation

1. **Always set multi-index**: `df.set_index(['entity_id', 'time_id'])`
2. **Sort index**: `df = df.sort_index()`
3. **Check balance**: Unbalanced panels are fine but require care
4. **Handle missing**: Panel methods handle gaps, but document

### Model Selection

1. **Start with FE**: Safer assumption (allows correlation)
2. **Hausman test**: If p < 0.05, use FE; otherwise RE is more efficient
3. **Time effects**: Include if common shocks likely
4. **Dynamic terms**: Only if theory suggests persistence

### Inference

1. **Always cluster by entity**: Serial correlation is the norm
2. **Consider two-way clustering**: If cross-sectional correlation possible
3. **Report multiple SE types**: Shows robustness
4. **Check effective clusters**: Need ~50+ for reliable clustering

### Reporting

1. **Report both FE and RE**: Show Hausman test
2. **Include F-test**: Joint significance of fixed effects
3. **Report within R²**: More relevant than overall for FE
4. **Describe panel structure**: N entities, T periods, balance

## Reference Documentation

### references/fixed_effects.md
- Within transformation derivation
- First-difference vs within estimator
- Time fixed effects implementation
- Two-way fixed effects

### references/random_effects.md
- GLS derivation
- Quasi-demeaning parameter (theta)
- When RE is appropriate
- Mundlak approach

### references/dynamic_panels.md
- Nickell bias explanation
- Arellano-Bond GMM
- System GMM
- Instrument proliferation

### references/clustered_se.md
- Serial correlation in panels
- Cameron-Gelbach-Miller two-way clustering
- Finite sample corrections
- When to cluster

### references/diagnostic_tests.md
- Hausman test derivation
- F-test for fixed effects
- Serial correlation tests
- Cross-sectional dependence tests

## Common Pitfalls to Avoid

1. **Forgetting to set index**: linearmodels requires multi-index for panel
2. **Not clustering SEs**: Panel data almost always has serial correlation
3. **Including time-invariant X with FE**: Will be absorbed/collinear
4. **Ignoring Hausman test**: FE and RE can give very different results
5. **Using FE with small T, large N**: Incidental parameters problem
6. **Lagged Y with FE and small T**: Nickell bias (use GMM)
7. **Reporting only overall R²**: Within R² is more meaningful for FE
8. **Forgetting drop_absorbed=True**: Needed with both EntityEffects and TimeEffects
9. **Over-differencing**: First-difference loses information vs within
10. **Ignoring unbalanced structure**: May indicate selection issues

## Troubleshooting

### Error: "data must have a MultiIndex"

**Solution:**
```python
# Set the multi-index before estimation
df = df.set_index(['entity_id', 'time_id'])
```

### Singular Matrix Error

**Cause:** Perfect collinearity, often from time-invariant variables with FE

**Solution:**
```python
# Remove time-invariant variables when using EntityEffects
# OR use drop_absorbed=True
model = PanelOLS.from_formula(
    'y ~ x1 + x2 + EntityEffects',
    data=df,
    drop_absorbed=True
)
```

### Clustered SE Fails with Few Clusters

**Cause:** Need ~50+ clusters for reliable inference

**Solution:**
```python
# Use wild bootstrap for few clusters
# Or report both clustered and robust SE
# Or aggregate to higher level
```

### Negative Within R²

**Cause:** Model without intercept or technical issue

**Solution:**
```python
# Check model specification
# Within R² can be negative if model is very poor
# Usually indicates specification error
```

## Additional Resources

### Official Documentation
- linearmodels: https://bashtage.github.io/linearmodels/
- statsmodels Panel: https://www.statsmodels.org/stable/mixed_linear.html

### Key Papers
- Wooldridge, J. (2010). *Econometric Analysis of Cross Section and Panel Data*, 2nd ed.
- Arellano, M. & Bond, S. (1991). "Some Tests of Specification for Panel Data"
- Cameron, A.C., Gelbach, J.B., & Miller, D.L. (2011). "Robust Inference with Multiway Clustering"
- Bertrand, M., Duflo, E., & Mullainathan, S. (2004). "How Much Should We Trust Differences-in-Differences Estimates?"

### Textbooks
- Baltagi, B. (2021). *Econometric Analysis of Panel Data*, 6th ed.
- Hsiao, C. (2014). *Analysis of Panel Data*, 3rd ed.

## Installation

```bash
# Core packages
pip install linearmodels statsmodels pandas numpy

# For dynamic panels (optional)
pip install pydynpd

# Full installation
pip install linearmodels statsmodels pandas numpy scipy matplotlib seaborn
```

## Related Skills

| Skill | When to Use Instead |
|-------|---------------------|
| `estimator-did` | Treatment effects with staggered adoption |
| `estimator-iv` | Need instruments for endogeneity |
| `time-series-econometrics` | Single unit, many time periods |
| `causal-ddml` | High-dimensional controls with panel |
