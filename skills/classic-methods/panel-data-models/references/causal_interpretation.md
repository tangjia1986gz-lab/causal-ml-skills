# Causal Interpretation of Panel Models

## Overview

Panel data is powerful for causal inference because it can control for time-invariant unobserved confounders. However, two-way fixed effects (TWFE) has important limitations when treatment timing varies.

## TWFE for Causal Inference

### The Standard Approach

$$y_{it} = \alpha_i + \gamma_t + \tau D_{it} + X_{it}\beta + \epsilon_{it}$$

Where:
- $\alpha_i$: entity fixed effects (absorb time-invariant confounders)
- $\gamma_t$: time fixed effects (absorb common shocks)
- $D_{it}$: treatment indicator
- $\tau$: treatment effect of interest

### Identifying Assumptions

1. **Parallel trends**: Absent treatment, treated and control would have same trends
2. **No anticipation**: Treatment doesn't affect outcomes before it occurs
3. **SUTVA**: No interference between units

### What TWFE Controls For

| Confounder Type | Controlled by FE? |
|-----------------|-------------------|
| Time-invariant entity characteristics | Yes (entity FE) |
| Common time shocks | Yes (time FE) |
| Time-varying entity-specific confounders | NO |
| Entity-specific trends | NO (need entity-specific trends) |

## The TWFE Problem with Staggered Adoption

### When TWFE Fails

With staggered treatment timing and heterogeneous treatment effects, TWFE can produce:
- **Biased estimates**: Even wrong sign
- **Negative weights**: Some comparisons receive negative weight
- **Misleading inference**: Conflation of treatment effect heterogeneity

### Why This Happens

TWFE is implicitly a weighted average of many 2x2 DiD comparisons:

1. **Clean comparisons**: Never-treated vs newly-treated
2. **Forbidden comparisons**: Already-treated vs newly-treated

The second type is problematic: already-treated units serve as "controls" but their outcomes are affected by treatment.

### Goodman-Bacon Decomposition

Decomposes TWFE into constituent 2x2 DiDs:

$$\hat{\tau}_{TWFE} = \sum_k w_k \hat{\tau}_k$$

Where weights $w_k$ depend on:
- Group sizes
- Variance of treatment within groups
- Timing of treatment

**Critical insight**: Weights can be negative when early-treated groups are used as controls.

```python
from panel_estimator import PanelEstimator

estimator = PanelEstimator(data, 'entity', 'time', 'y', ['treatment'])
decomp = estimator.goodman_bacon_decomposition('treatment')

print(f"TWFE estimate: {decomp['twfe_estimate']:.4f}")
print(f"Negative weight share: {decomp['negative_weight_share']:.2%}")
```

## When TWFE is OK

TWFE produces valid estimates when:

1. **Homogeneous treatment effects**: $\tau_{it} = \tau$ for all $i, t$
2. **Treatment effect constant over time**: No dynamic effects
3. **No already-treated controls**: All comparisons use never-treated

### Diagnostic: Check for Problems

```python
# Warning signs
if len(timing_groups) > 2:
    print("Multiple treatment cohorts - potential TWFE problems")

if decomp['negative_weight_share'] > 0.05:
    print(f"WARNING: {decomp['negative_weight_share']:.1%} of weight is negative")
    print("Consider robust DiD estimators")
```

## Modern Alternatives to TWFE

### 1. Callaway and Sant'Anna (2021)

Estimates group-time average treatment effects (ATT(g,t)):
- Treatment effect for group $g$ at time $t$
- Only uses never-treated (or not-yet-treated) as controls
- Aggregates to overall ATT without negative weights

**Key features**:
- Explicit about what "control group" means
- Allows for treatment effect heterogeneity
- Provides event-study estimates

```python
# Using the 'did' package in R or Python port
# ATT(g,t) for each group g at each time t
from did import ATTgt

att_gt = ATTgt(
    data=data,
    yname='y',
    gname='first_treated',  # First treatment period
    idname='entity',
    tname='time',
    control_group='nevertreated'  # or 'notyettreated'
)

# Aggregate to overall ATT
att_overall = att_gt.aggregate('simple')
```

### 2. Sun and Abraham (2021)

Interaction-weighted estimator:
- Saturate model with cohort x relative-time interactions
- Average with appropriate weights

$$y_{it} = \alpha_i + \gamma_t + \sum_{g} \sum_{l \neq -1} \tau_{g,l} \cdot \mathbf{1}[G_i = g] \cdot \mathbf{1}[t - G_i = l] + \epsilon_{it}$$

Where $G_i$ is the treatment date for entity $i$.

### 3. Borusyak, Jaravel, Spiess (2024)

Imputation approach:
1. Estimate counterfactual for treated observations using untreated data
2. Treatment effect = Actual - Imputed counterfactual
3. Robust to heterogeneous effects

### 4. de Chaisemartin and D'Haultfoeuille (2020)

- Computes "clean" DiD comparisons only
- Provides diagnostics for TWFE weights

### Comparison of Methods

| Method | Controls Used | Handles Heterogeneity | Event Study |
|--------|--------------|----------------------|-------------|
| TWFE | All untreated (including already-treated) | No | Problematic |
| Callaway-Sant'Anna | Never/not-yet treated | Yes | Yes |
| Sun-Abraham | Never/not-yet treated | Yes | Yes |
| Borusyak et al. | Imputation | Yes | Yes |

## Event Study Design

### Traditional Event Study

$$y_{it} = \alpha_i + \gamma_t + \sum_{k \neq -1} \tau_k \cdot \mathbf{1}[t - E_i = k] + \epsilon_{it}$$

Where $E_i$ is the event date for entity $i$, and $k = -1$ is the reference period.

### Problems with Traditional Event Study
- Same TWFE issues with staggered timing
- Coefficients can be contaminated by other periods' effects
- Pre-trends test may be misleading

### Robust Event Study

Use Callaway-Sant'Anna or Sun-Abraham for event study:

```python
# Event study with proper weighting
event_study = att_gt.aggregate('event')

# Plot
import matplotlib.pyplot as plt
plt.errorbar(event_study.relative_time,
             event_study.estimate,
             yerr=1.96 * event_study.std_error)
plt.axvline(x=-0.5, color='red', linestyle='--')
plt.axhline(y=0, color='black', linestyle='-')
plt.xlabel('Periods Relative to Treatment')
plt.ylabel('Treatment Effect')
plt.title('Event Study (Robust to Staggered Treatment)')
```

## Practical Recommendations

### Step 1: Diagnose the Problem

```python
# Check treatment timing variation
treatment_timing = data.groupby('entity')['treatment'].apply(
    lambda x: x.idxmax() if x.any() else None
)

n_timing_groups = treatment_timing.nunique()
has_never_treated = treatment_timing.isna().any()

print(f"Number of treatment timing groups: {n_timing_groups}")
print(f"Has never-treated group: {has_never_treated}")
```

### Step 2: Choose Appropriate Method

| Scenario | Recommendation |
|----------|---------------|
| Same treatment timing for all | TWFE is fine |
| Staggered timing, homogeneous effects | TWFE likely OK |
| Staggered timing, heterogeneous effects | Use robust estimator |
| No never-treated units | Callaway-Sant'Anna with not-yet-treated |
| Complex dynamics | Borusyak imputation |

### Step 3: Report Multiple Estimates

```python
results = {
    'TWFE': fe_result.coefficients['treatment'],
    'TWFE_clustered': fe_clustered.coefficients['treatment'],
    'Callaway_SantAnna': att_cs,
    'Sun_Abraham': att_sa,
}

print("Sensitivity Analysis:")
for method, est in results.items():
    print(f"  {method}: {est:.4f}")
```

### Step 4: Validate with Pre-Trends

- Test pre-treatment coefficients = 0
- Use robust method for event study
- Be cautious: Passing pre-trends test doesn't guarantee parallel trends

## Code Example: Complete Workflow

```python
import numpy as np
import pandas as pd
from panel_estimator import PanelEstimator

# Load data
data = pd.read_csv('panel_data.csv')

# 1. Setup
estimator = PanelEstimator(
    data, 'entity', 'time', 'outcome',
    ['treatment', 'control1', 'control2']
)

# 2. Standard TWFE
twfe = estimator.fit_fixed_effects(entity_effects=True, time_effects=True)
twfe_clustered = estimator.cluster_robust_inference(twfe, 'entity')

print("TWFE Results:")
print(twfe_clustered.summary())

# 3. Diagnose TWFE problems
bacon = estimator.goodman_bacon_decomposition('treatment')
print(f"\nGoodman-Bacon Decomposition:")
print(f"  TWFE estimate: {bacon['twfe_estimate']:.4f}")
print(f"  Negative weight share: {bacon['negative_weight_share']:.2%}")

if bacon['negative_weight_share'] > 0.05:
    print("\n  WARNING: Significant negative weights detected!")
    print("  Consider using Callaway-Sant'Anna or Sun-Abraham estimator")

# 4. Robustness (if using did package)
# from did import ATTgt
# att_gt = ATTgt(data, 'outcome', 'first_treated', 'entity', 'time')
# print(f"Callaway-Sant'Anna ATT: {att_gt.overall_att:.4f}")
```

## Summary

| Question | Answer |
|----------|--------|
| When is TWFE valid? | Homogeneous effects, no staggered timing |
| What's the problem with staggered DiD? | Already-treated used as controls, negative weights |
| How to diagnose? | Goodman-Bacon decomposition |
| Modern alternatives? | Callaway-Sant'Anna, Sun-Abraham, Borusyak |
| Always cluster? | Yes, at entity level minimum |

## References

- Goodman-Bacon, A. (2021). Difference-in-Differences with Variation in Treatment Timing. Journal of Econometrics.
- Callaway, B. & Sant'Anna, P.H.C. (2021). Difference-in-Differences with Multiple Time Periods. Journal of Econometrics.
- Sun, L. & Abraham, S. (2021). Estimating Dynamic Treatment Effects in Event Studies with Heterogeneous Treatment Effects. Journal of Econometrics.
- Borusyak, K., Jaravel, X., & Spiess, J. (2024). Revisiting Event Study Designs: Robust and Efficient Estimation. Review of Economic Studies.
- de Chaisemartin, C. & D'Haultfoeuille, X. (2020). Two-Way Fixed Effects Estimators with Heterogeneous Treatment Effects. American Economic Review.
- Roth, J. et al. (2023). What's Trending in Difference-in-Differences? A Synthesis of the Recent Econometrics Literature. Journal of Econometrics.
