---
name: estimator-did
description: Difference-in-Differences causal inference using Python. Use when estimating treatment effects with panel data, staggered adoption, event studies, or parallel trends testing. Provides TWFE, Callaway-Sant'Anna, and Sun-Abraham estimators via linearmodels, statsmodels, and did packages.
license: MIT
metadata:
    skill-author: Causal-ML-Skills
---

# Difference-in-Differences (DID): Causal Inference with Panel Data

## Overview

Difference-in-Differences (DID) is one of the most widely used methods for causal inference in economics, management, and policy research. This skill provides comprehensive guidance for implementing rigorous DID analyses in Python, from classic two-period designs to modern staggered adoption with heterogeneous treatment effects.

Apply this skill when you have panel data (repeated observations over time for multiple units) and want to estimate the causal effect of a treatment/intervention by comparing changes in outcomes between treated and control groups.

## When to Use This Skill

This skill should be used when:

- Estimating causal effects with panel/longitudinal data
- Analyzing policy interventions, treatment rollouts, or natural experiments
- Working with staggered treatment adoption (units treated at different times)
- Testing parallel trends assumptions with event studies
- Running Two-Way Fixed Effects (TWFE) regressions
- Implementing modern DID estimators (Callaway-Sant'Anna, Sun-Abraham)
- Diagnosing negative weights in staggered designs
- Creating publication-quality tables and event study plots

## Quick Start Guide

### Classic 2x2 DID (Two Groups, Two Periods)

```python
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf

# Example: Effect of minimum wage increase on employment
# Data structure: firm_id, year, treated (=1 if in treatment state),
#                 post (=1 if after policy), employment

# Load your data
df = pd.read_csv('panel_data.csv')

# Create DID interaction term
df['did'] = df['treated'] * df['post']

# Method 1: OLS with interaction (for 2x2 design)
model = smf.ols('employment ~ treated + post + did', data=df)
result = model.fit(cov_type='cluster', cov_kwds={'groups': df['firm_id']})

print(result.summary())
print(f"\nDID Estimate (ATT): {result.params['did']:.4f}")
print(f"Standard Error: {result.bse['did']:.4f}")
print(f"95% CI: [{result.conf_int().loc['did', 0]:.4f}, {result.conf_int().loc['did', 1]:.4f}]")
```

### Two-Way Fixed Effects (TWFE) with Panel Data

```python
import pandas as pd
import numpy as np
from linearmodels.panel import PanelOLS

# Load and prepare data
df = pd.read_csv('panel_data.csv')

# CRITICAL: Set MultiIndex for panel structure
df = df.set_index(['firm_id', 'year'])

# Define variables
# treatment: binary indicator (1 = treated in this period)
# outcome: dependent variable
# controls: list of control variables (optional)

# TWFE Regression
model = PanelOLS(
    dependent=df['outcome'],
    exog=df[['treatment']],  # Add controls: df[['treatment', 'size', 'age']]
    entity_effects=True,     # Firm fixed effects
    time_effects=True,       # Year fixed effects
    drop_absorbed=True       # Drop collinear variables
)

# IMPORTANT: Cluster standard errors at entity level
result = model.fit(cov_type='clustered', cluster_entity=True)

print(result.summary)

# Extract key results
coef = result.params['treatment']
se = result.std_errors['treatment']
pval = result.pvalues['treatment']
ci_low, ci_high = result.conf_int().loc['treatment']

print(f"\n{'='*50}")
print(f"Treatment Effect: {coef:.4f}")
print(f"Robust SE: {se:.4f}")
print(f"P-value: {pval:.4f}")
print(f"95% CI: [{ci_low:.4f}, {ci_high:.4f}]")
print(f"{'='*50}")
```

### Event Study (Pre-Trends Test)

```python
import pandas as pd
import numpy as np
from linearmodels.panel import PanelOLS
import matplotlib.pyplot as plt

# Load data with treatment timing
df = pd.read_csv('panel_data.csv')

# Create relative time variable (year relative to treatment)
# treatment_year: the year each unit was first treated (NaN if never treated)
df['rel_time'] = df['year'] - df['treatment_year']

# For never-treated units, set rel_time to a large negative (or use as control)
df['rel_time'] = df['rel_time'].fillna(-999)

# Create event time dummies (exclude t=-1 as reference)
# Typical window: [-4, -3, -2, -1(ref), 0, 1, 2, 3, 4+]
event_dummies = []
for t in range(-4, 5):
    if t == -1:  # Reference period
        continue
    col_name = f'D_{t}' if t >= 0 else f'D_m{abs(t)}'
    if t == 4:  # Bin post-treatment periods
        df[col_name] = (df['rel_time'] >= t).astype(int)
    elif t == -4:  # Bin pre-treatment periods
        df[col_name] = (df['rel_time'] <= t).astype(int)
    else:
        df[col_name] = (df['rel_time'] == t).astype(int)
    event_dummies.append(col_name)

# Set panel index
df = df.set_index(['firm_id', 'year'])

# Event study regression
model = PanelOLS(
    dependent=df['outcome'],
    exog=df[event_dummies],
    entity_effects=True,
    time_effects=True,
    drop_absorbed=True
)
result = model.fit(cov_type='clustered', cluster_entity=True)

# Extract coefficients for plotting
coefs = result.params[event_dummies]
ses = result.std_errors[event_dummies]

# Prepare plot data
times = [-4, -3, -2, 0, 1, 2, 3, 4]  # Excluding -1 (reference)
coef_vals = [coefs[f'D_m{abs(t)}'] if t < 0 else coefs[f'D_{t}'] for t in times]
se_vals = [ses[f'D_m{abs(t)}'] if t < 0 else ses[f'D_{t}'] for t in times]

# Add reference period (0 by construction)
times.insert(3, -1)
coef_vals.insert(3, 0)
se_vals.insert(3, 0)

# Plot event study
fig, ax = plt.subplots(figsize=(10, 6))
ax.errorbar(times, coef_vals, yerr=[1.96*s for s in se_vals],
            fmt='o-', capsize=4, capthick=2, markersize=8)
ax.axhline(y=0, color='r', linestyle='--', alpha=0.7)
ax.axvline(x=-0.5, color='gray', linestyle=':', alpha=0.7, label='Treatment')
ax.set_xlabel('Periods Relative to Treatment', fontsize=12)
ax.set_ylabel('Coefficient Estimate', fontsize=12)
ax.set_title('Event Study: Dynamic Treatment Effects', fontsize=14)
ax.legend()
plt.tight_layout()
plt.savefig('event_study.png', dpi=150)
plt.show()

# Test for pre-trends (joint F-test on pre-treatment dummies)
pre_treatment = [d for d in event_dummies if 'm' in d]  # D_m4, D_m3, D_m2
from scipy import stats
f_stat = result.wald_test(formula=' = '.join(pre_treatment) + ' = 0')
print(f"\nPre-trends test (joint F):")
print(f"F-statistic: {f_stat.stat:.4f}")
print(f"P-value: {f_stat.pval:.4f}")
```

### Staggered DID with Callaway-Sant'Anna (2021)

```python
import pandas as pd
import numpy as np

# For Callaway-Sant'Anna, use the 'differences' or 'csdid' package
# Install: pip install differences

from differences import ATTgt, Aggregation

# Prepare data
df = pd.read_csv('staggered_data.csv')

# Required columns:
# - unit_id: unique identifier for each unit
# - time: time period
# - outcome: dependent variable
# - first_treat: first treatment period (0 or NaN for never-treated)

# Estimate group-time ATT
att_gt = ATTgt(
    data=df,
    cohort_name='first_treat',      # Treatment timing variable
    time_name='time',                # Time variable
    unit_name='unit_id',             # Unit identifier
    outcome_name='outcome',          # Outcome variable
    control_type='nevertreated'      # Use never-treated as control
    # Alternative: control_type='notyettreated'
)

result = att_gt.fit()

# View group-time effects
print("Group-Time ATT Estimates:")
print(result.summary())

# Aggregate to overall ATT
agg = Aggregation(result)

# Simple weighted average
overall_att = agg.aggregate('simple')
print(f"\nOverall ATT: {overall_att['att']:.4f} (SE: {overall_att['se']:.4f})")

# Event study aggregation
event_study = agg.aggregate('event')
print("\nEvent Study Aggregation:")
print(event_study)

# Plot event study
agg.plot_event()
plt.savefig('cs_event_study.png', dpi=150)
```

### Alternative: Using DoubleML for DID

```python
import doubleml as dml
from doubleml import DoubleMLData, DoubleMLDID
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

# For DID with high-dimensional controls
df = pd.read_csv('panel_data.csv')

# Prepare DoubleML data structure
# Note: DoubleMLDID requires specific data format
dml_data = dml.DoubleMLData(
    df,
    y_col='outcome',
    d_cols='treatment',
    x_cols=['control1', 'control2', 'control3']  # High-dimensional controls
)

# Define ML learners for nuisance functions
ml_g = RandomForestRegressor(n_estimators=200, max_depth=5, random_state=42)
ml_m = RandomForestClassifier(n_estimators=200, max_depth=5, random_state=42)

# Fit DID with ML
dml_did = DoubleMLDID(dml_data, ml_g=ml_g, ml_m=ml_m, n_folds=5)
dml_did.fit()

print(dml_did.summary)
```

## Core Capabilities

### 1. Classic DID Estimators

**Two-Period DID (2x2)**:
- Simple comparison of means: `ATT = (Y_treat_post - Y_treat_pre) - (Y_control_post - Y_control_pre)`
- OLS with interaction term: `Y ~ Treat + Post + Treat*Post`
- Fixed effects formulation

**Multi-Period DID (TWFE)**:
- Entity (unit) fixed effects absorb time-invariant heterogeneity
- Time fixed effects absorb common shocks
- Treatment coefficient: average effect across all treated unit-periods

**When to use:**
- Single treatment timing (all units treated at same time)
- Moderate number of controls
- Standard inference requirements

**Reference:** See `references/estimation_methods.md` for formulas and implementation details.

### 2. Staggered DID (Modern Methods)

**Problem with TWFE**: When treatment timing varies across units, TWFE can produce biased estimates due to negative weights (Goodman-Bacon 2021).

**Solutions:**

| Method | Key Feature | Best For |
|--------|-------------|----------|
| **Callaway-Sant'Anna (2021)** | Group-time ATT, flexible aggregation | Most applications |
| **Sun-Abraham (2021)** | Interaction-weighted estimator | Continuous controls |
| **de Chaisemartin-D'Haultfoeuille (2020)** | Fuzzy DID, intensive margin | Heterogeneous timing |
| **Borusyak et al. (2024)** | Imputation estimator | Clean never-treated control |

**When to use:**
- Units treated at different times (staggered adoption)
- Treatment effects may be heterogeneous over time
- Need robust event study estimates

**Reference:** See `references/estimation_methods.md` Section 3 for detailed comparison.

### 3. Diagnostic Tests

**Pre-Trends Testing:**
- Event study: Visual inspection of pre-treatment coefficients
- Joint F-test: Test that all pre-treatment coefficients equal zero
- Rambachan-Roth (2023): Sensitivity to parallel trends violations

**Placebo Tests:**
- Fake treatment timing: Move treatment earlier in pre-period
- Fake outcomes: Use variables unaffected by treatment
- Permutation test: Randomly reassign treatment

**Negative Weights Check:**
- Bacon decomposition: Identify comparisons driving TWFE estimate
- Weight diagnostics: Detect problematic treated-vs-treated comparisons

**Reference:** See `references/diagnostic_tests.md` for implementation details.

### 4. Inference and Standard Errors

**Clustering:**
- ALWAYS cluster at level of treatment assignment
- If treatment varies at state level, cluster by state
- If treatment varies at firm level, cluster by firm

**Wild Bootstrap:**
- Use when number of clusters is small (<50)
- More accurate p-values than cluster-robust SEs

```python
# Cluster-robust standard errors
result = model.fit(cov_type='clustered', cluster_entity=True)

# Two-way clustering (entity and time)
result = model.fit(cov_type='clustered', clusters=df[['firm_id', 'year']])
```

**Reference:** See `references/identification_assumptions.md` for inference details.

## Common Workflows

### Workflow 1: Standard DID Analysis

```
1. Data Preparation
   ├── Load panel data
   ├── Verify balance (same units across time)
   └── Create treatment indicator

2. Descriptive Analysis
   ├── Summary statistics by treatment group
   ├── Outcome trends plot
   └── Treatment timing distribution

3. Pre-Trends Testing
   ├── Event study regression
   ├── Visual inspection (coefficients should be ~0 pre-treatment)
   └── Joint F-test for pre-trends

4. Main Estimation
   ├── TWFE regression
   ├── Cluster standard errors
   └── Report point estimate, SE, CI, p-value

5. Robustness Checks
   ├── Add/remove controls
   ├── Alternative control groups
   ├── Placebo tests
   └── Bacon decomposition (if staggered)

6. Reporting
   ├── Coefficient table (multiple specifications)
   ├── Event study plot
   └── Interpretation and limitations
```

### Workflow 2: Staggered DID Analysis

```
1. Data Preparation
   ├── Create cohort variable (first treatment year)
   ├── Identify never-treated units
   └── Calculate relative time

2. Diagnostics
   ├── Bacon decomposition (TWFE weights)
   ├── Check for negative weights
   └── Visualize treatment timing

3. Estimation
   ├── Callaway-Sant'Anna group-time ATT
   ├── Aggregate to overall ATT
   └── Event study aggregation

4. Sensitivity Analysis
   ├── Alternative control group (never vs not-yet treated)
   ├── Rambachan-Roth bounds
   └── Heterogeneous effects by cohort

5. Reporting
   ├── Group-time ATT table
   ├── Event study plot with CI
   └── Comparison with naive TWFE
```

## Best Practices

### Data Preparation

1. **Balance your panel**: Same units observed in all time periods
2. **Define treatment clearly**: Binary indicator = 1 when treatment is active
3. **Create first-treatment variable**: For staggered designs, record when each unit was first treated
4. **Handle never-treated**: Use 0 or missing for units never treated

### Estimation

1. **Start with TWFE**: Understand baseline before using modern methods
2. **Check for staggered timing**: If timing varies, use Callaway-Sant'Anna
3. **Always cluster**: At the level of treatment assignment
4. **Test pre-trends**: Event study + joint F-test

### Inference

1. **Report confidence intervals**: Not just point estimates
2. **Use wild bootstrap**: If clusters < 50
3. **Acknowledge limitations**: DID requires parallel trends (untestable)

### Reporting

1. **Show event study plot**: Visual evidence for parallel trends
2. **Multiple specifications**: Show robustness to different controls
3. **Bacon decomposition**: If using TWFE with staggered timing
4. **Callaway-Sant'Anna comparison**: Show both TWFE and CS estimates

## Reference Documentation

This skill includes comprehensive reference files for detailed guidance:

### references/identification_assumptions.md
- Parallel trends assumption (formal definition, implications)
- No anticipation assumption
- SUTVA (Stable Unit Treatment Value Assumption)
- Conditional parallel trends with covariates
- When assumptions fail and what to do

### references/estimation_methods.md
- Classic TWFE formulas
- Callaway-Sant'Anna estimator
- Sun-Abraham interaction-weighted estimator
- de Chaisemartin-D'Haultfoeuille estimator
- Borusyak et al. imputation estimator
- Comparison table and selection guide

### references/diagnostic_tests.md
- Pre-trends testing (event study, joint F-test)
- Bacon decomposition
- Placebo tests
- Sensitivity analysis (Rambachan-Roth)
- Balance tests

### references/reporting_standards.md
- AER/QJE/MS table formats
- LaTeX templates
- Event study plot standards
- Required robustness checks

### references/common_errors.md
- Using TWFE with staggered timing (negative weights)
- Incorrect clustering
- Confusing treatment indicator with post indicator
- Omitting entity or time fixed effects
- Testing parallel trends in post-period

### references/extensions.md
- Triple differences (DDD)
- Synthetic control
- DID with continuous treatment
- Fuzzy DID (imperfect compliance)

## Common Pitfalls to Avoid

1. **Using TWFE with staggered adoption without checking weights**: Run Bacon decomposition first
2. **Not clustering standard errors**: ALWAYS cluster at treatment assignment level
3. **Confusing `treatment` with `post`**: Treatment = 1 when unit is actually treated (not just eligible)
4. **Omitting fixed effects**: TWFE requires BOTH entity and time FE
5. **Testing parallel trends in post-period**: Pre-trends test uses PRE-treatment periods only
6. **Using `sm.OLS` instead of `PanelOLS`**: For panel data, use linearmodels
7. **Forgetting to set MultiIndex**: `PanelOLS` requires `df.set_index(['unit', 'time'])`
8. **Including time-varying confounders affected by treatment**: Bad controls problem
9. **Ignoring anticipation effects**: Check for effects before official treatment date
10. **Not showing event study**: Always include dynamic effects plot
11. **Using wrong control group**: For staggered DID, choose never-treated or not-yet-treated
12. **Interpreting TWFE as ATT with heterogeneous effects**: It's a weighted average that may be misleading
13. **Not discussing parallel trends assumption**: It's untestable - discuss plausibility
14. **Binning all post-periods together**: Estimate dynamic effects to see evolution
15. **Using wrong denominator for relative time**: Should be (current year - first treatment year)

## Troubleshooting

### "Singular matrix" Error

**Issue:** PanelOLS raises singular matrix error

**Solution:**
```python
# Drop collinear variables
model = PanelOLS(..., drop_absorbed=True)

# Or manually check for collinearity
from statsmodels.stats.outliers_influence import variance_inflation_factor
```

### Coefficients All NaN

**Issue:** Fixed effects absorb all variation

**Solution:** Check that treatment varies within entities over time:
```python
# Treatment must change within at least some entities
df.groupby('entity')['treatment'].nunique().value_counts()
```

### Event Study Coefficients Explode

**Issue:** Large coefficients at endpoints

**Solution:** Bin endpoint periods:
```python
# Bin t <= -4 and t >= 4
df['D_m4plus'] = (df['rel_time'] <= -4).astype(int)
df['D_4plus'] = (df['rel_time'] >= 4).astype(int)
```

### Clustering with Few Clusters

**Issue:** Cluster-robust SEs unreliable with < 50 clusters

**Solution:** Use wild cluster bootstrap:
```python
# Install: pip install wildboottest
from wildboottest import wildboottest
result = wildboottest(model, cluster='state', B=999)
```

### Staggered DID Package Not Found

**Issue:** `differences` package not available

**Solution:**
```bash
pip install differences

# Alternative packages:
pip install pydid          # Another Python implementation
pip install csdid          # Port of Stata csdid
```

## Additional Resources

### Official Documentation
- linearmodels: https://bashtage.github.io/linearmodels/
- statsmodels: https://www.statsmodels.org/stable/
- differences: https://pypi.org/project/differences/
- DoubleML: https://docs.doubleml.org/

### Key Papers
- Callaway & Sant'Anna (2021): "Difference-in-Differences with Multiple Time Periods"
- Goodman-Bacon (2021): "Difference-in-Differences with Variation in Treatment Timing"
- Sun & Abraham (2021): "Estimating Dynamic Treatment Effects in Event Studies"
- Roth (2022): "Pretest with Caution: Event-Study Estimates after Testing for Parallel Trends"
- Rambachan & Roth (2023): "A More Credible Approach to Parallel Trends"

### Tutorials
- Scott Cunningham's Causal Inference: The Mixtape: https://mixtape.scunning.com/
- Nick Huntington-Klein's The Effect: https://theeffectbook.net/

## Installation

```bash
# Core packages
pip install linearmodels statsmodels pandas numpy matplotlib

# For staggered DID
pip install differences

# For ML-based DID
pip install doubleml

# For wild bootstrap
pip install wildboottest

# Full installation
pip install linearmodels statsmodels differences doubleml wildboottest matplotlib seaborn
```

## Related Skills

| Skill | When to Use Instead |
|-------|---------------------|
| `estimator-rd` | Sharp cutoff in assignment variable |
| `estimator-iv` | Need instrument for endogenous treatment |
| `estimator-psm` | Cross-sectional data, propensity matching |
| `causal-ddml` | High-dimensional controls with ML |
| `panel-data-models` | General panel data analysis (not causal) |
