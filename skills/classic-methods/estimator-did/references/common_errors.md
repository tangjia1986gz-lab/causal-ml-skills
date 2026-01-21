# Common Errors in Difference-in-Differences

> **Document Type**: Reference | **Last Updated**: 2025-01
> **Related**: [identification_assumptions.md](identification_assumptions.md), [diagnostic_tests.md](diagnostic_tests.md)

## Overview

This document catalogs common errors in DID analysis, explains why they are problematic, and provides correct approaches. Each error is illustrated with code examples.

---

## 1. Staggered Treatment Timing Errors

### Error 1.1: Using Standard TWFE with Staggered Adoption

**The Mistake**:
```python
# WRONG: Standard TWFE when treatment timing varies
from linearmodels.panel import PanelOLS

model = PanelOLS.from_formula(
    'y ~ treated + EntityEffects + TimeEffects',
    data=df.set_index(['id', 'year'])
)
result = model.fit()
```

**Why It's Wrong**:
When units are treated at different times, TWFE uses already-treated units as controls for later-treated units. If treatment effects are heterogeneous or dynamic, this produces biased estimates because:

1. Already-treated units are not valid counterfactuals
2. Implicit weights can be negative for some comparisons
3. The estimate is a weighted average of potentially different ATTs

**Correct Approach**:
```python
# CORRECT: Use heterogeneity-robust estimator
from did_estimator import estimate_did_staggered

result = estimate_did_staggered(
    data=df,
    outcome="y",
    treatment_time="first_treated",  # When unit was first treated
    unit_id="id",
    time_id="year",
    control_group="nevertreated"
)
```

**Detection**:
```python
from scripts.robustness_checks import bacon_decomposition

# Check if problematic comparisons have high weight
bacon = bacon_decomposition(df, "y", "treated", "id", "year")
if bacon['bad_comparison_weight'] > 0.1:
    print("WARNING: Significant weight on potentially biased comparisons")
```

---

### Error 1.2: Ignoring Negative Weights

**The Mistake**:
Proceeding with TWFE without checking for negative weights in the Goodman-Bacon decomposition.

**Why It's Wrong**:
Negative weights mean that some 2x2 DID comparisons are subtracted rather than added. With heterogeneous effects, the overall estimate can even have the wrong sign.

**Detection and Solution**:
```python
# Check for negative weights
bacon = bacon_decomposition(df, "y", "treated", "id", "year")

# Look at weight distribution
for comp_type, values in bacon['components'].items():
    if values['weight'] < 0:
        print(f"WARNING: Negative weight in {comp_type}: {values['weight']:.3f}")

# If negative weights exist, use robust estimator
if any(v['weight'] < 0 for v in bacon['components'].values()):
    result = estimate_did_staggered(...)
```

---

## 2. Parallel Trends Errors

### Error 2.1: Ignoring Pre-Trend Violations

**The Mistake**:
```python
# WRONG: Proceeding despite clear pre-trend violation
trends_test = test_parallel_trends(...)
print(f"Pre-trends test p-value: {trends_test.p_value}")  # p = 0.01

# Ignoring the warning and proceeding anyway
result = estimate_did_panel(...)  # This estimate is likely biased!
```

**Why It's Wrong**:
If parallel trends is violated, the DID estimate is biased. The bias equals the difference in counterfactual trends.

**Correct Approach**:
```python
# CORRECT: Address pre-trend violation
trends_test = test_parallel_trends(...)

if not trends_test.passed:
    print("Parallel trends assumption may be violated!")
    print("Consider one of the following approaches:")

    # Option 1: Group-specific linear trends
    result = estimate_did_with_trends(
        data=df,
        outcome="y",
        treatment="treated",
        unit_id="id",
        time_id="year",
        group_trends=True
    )

    # Option 2: Synthetic control
    from synthetic_control import estimate_synth
    result_synth = estimate_synth(...)

    # Option 3: Matching on pre-treatment outcomes
    df_matched = match_on_pretrends(df, ...)
    result_matched = estimate_did_panel(df_matched, ...)
```

---

### Error 2.2: Concluding Parallel Trends "Holds" from Pre-Trend Test

**The Mistake**:
```python
# WRONG: Over-interpreting pre-trends test
trends_test = test_parallel_trends(...)
if trends_test.passed:
    print("Parallel trends assumption HOLDS!")  # Too strong a claim
```

**Why It's Wrong**:
The pre-trends test only checks if trends were parallel in the **observed** pre-treatment period. It does NOT prove that counterfactual trends would be parallel. Failure to reject is not evidence of validity.

**Correct Approach**:
```python
# CORRECT: Appropriate language
if trends_test.passed:
    print("Pre-treatment trends appear parallel (we cannot reject H0).")
    print("This is consistent with, but does not prove, the parallel trends assumption.")
    print("The assumption fundamentally concerns unobserved counterfactual outcomes.")
```

---

### Error 2.3: Testing Pre-Trends on Few Periods

**The Mistake**:
```python
# WRONG: Only 1-2 pre-treatment periods
trends_test = test_parallel_trends(
    data=df,
    ...,
    n_pre_periods=2  # Insufficient power
)
```

**Why It's Wrong**:
With few pre-treatment periods:
1. Statistical power to detect violations is low
2. Cannot distinguish between noise and true trends
3. May miss non-linear pre-trends

**Correct Approach**:
```python
# CORRECT: Use multiple pre-treatment periods
trends_test = test_parallel_trends(
    data=df,
    ...,
    n_pre_periods=4  # Minimum recommended
)

# If few pre-periods available, be explicit about limitation
if n_pre_periods < 4:
    print("WARNING: Limited pre-treatment data constrains parallel trends test")
    print("Interpret results with caution")
```

---

## 3. Standard Error Errors

### Error 3.1: Not Clustering Standard Errors

**The Mistake**:
```python
# WRONG: Homoskedastic or heteroskedasticity-robust SEs only
result = model.fit()  # Default: not clustered
result = model.fit(cov_type='robust')  # Robust but not clustered
```

**Why It's Wrong**:
In panel data, observations within the same unit are correlated over time. Ignoring this correlation leads to:
1. Understated standard errors
2. Inflated t-statistics
3. Over-rejection of null hypotheses (false positives)

**Correct Approach**:
```python
# CORRECT: Cluster at the level of treatment assignment
result = estimate_did_panel(
    data=df,
    ...,
    cluster="id"  # Cluster at unit level (most common)
)

# Or if treatment varies at state level
result = estimate_did_panel(
    data=df,
    ...,
    cluster="state"
)
```

---

### Error 3.2: Clustering at Wrong Level

**The Mistake**:
```python
# WRONG: Clustering at individual level when treatment varies by state
result = estimate_did_panel(
    ...,
    cluster="individual_id"  # Treatment is at state level!
)
```

**Why It's Wrong**:
Clustering should be at the level of treatment variation. Clustering at a finer level ignores correlation induced by treatment assignment and understates standard errors.

**Decision Rule**:
```python
def choose_cluster_level(data, treatment_var, potential_clusters):
    """
    Choose appropriate clustering level based on treatment variation.
    """
    # Find level at which treatment varies
    for cluster in potential_clusters:
        treatment_within_cluster = data.groupby(cluster)[treatment_var].nunique()
        if all(treatment_within_cluster == 1):
            # Treatment constant within this cluster level
            print(f"Treatment varies at {cluster} level or higher")
            return cluster

    return potential_clusters[-1]  # Highest level as fallback
```

---

### Error 3.3: Too Few Clusters

**The Mistake**:
Clustering with small number of clusters (e.g., 10 states) and using standard asymptotic inference.

**Why It's Wrong**:
Clustered standard errors rely on asymptotic theory that requires many clusters. With few clusters:
1. Standard errors are biased downward
2. T-distribution approximation is poor
3. Inference is unreliable

**Correct Approach**:
```python
# Check number of clusters
n_clusters = df[cluster_var].nunique()

if n_clusters < 50:
    print(f"WARNING: Only {n_clusters} clusters. Using wild bootstrap.")

    from scripts.robustness_checks import wild_cluster_bootstrap

    boot_result = wild_cluster_bootstrap(
        data=df,
        outcome="y",
        treatment="treated",
        unit_id="id",
        time_id="year",
        cluster=cluster_var,
        n_bootstrap=999
    )

    print(f"Bootstrap SE: {boot_result['boot_se']:.4f}")
    print(f"Bootstrap CI: [{boot_result['ci_lower']:.4f}, {boot_result['ci_upper']:.4f}]")
```

---

## 4. Specification Errors

### Error 4.1: Including Bad Controls

**The Mistake**:
```python
# WRONG: Controlling for post-treatment variables
result = estimate_did_panel(
    data=df,
    outcome="y",
    treatment="treated",
    controls=["post_treatment_variable"]  # Affected by treatment!
)
```

**Why It's Wrong**:
Controlling for variables affected by treatment:
1. Absorbs part of the treatment effect (bias toward zero)
2. Opens new confounding paths
3. Confuses direct and indirect effects

**Correct Approach**:
```python
# CORRECT: Only control for pre-treatment or time-invariant variables
result = estimate_did_panel(
    data=df,
    outcome="y",
    treatment="treated",
    controls=["baseline_x1", "baseline_x2", "time_invariant_z"]
)

# If variable is time-varying, use pre-treatment values
df['x_baseline'] = df.groupby('id')['x'].transform(
    lambda x: x[df.loc[x.index, 'year'] < treatment_year].mean()
)
```

**Decision Tree for Controls**:
```
Is the variable affected by treatment?
├── YES → DO NOT CONTROL FOR IT
│   └── Exception: Mediation analysis (different estimand)
└── NO
    ├── Is it time-invariant?
    │   └── YES → Absorbed by unit FE; no need to include
    └── Is it time-varying?
        ├── Measured pre-treatment → OK to control
        └── Measured post-treatment → Use pre-treatment values
```

---

### Error 4.2: Functional Form Mistakes

**The Mistake**:
```python
# WRONG: Not considering appropriate functional form
# When outcome is bounded, non-negative, or skewed
result = estimate_did_panel(
    data=df,
    outcome="count_variable",  # Counts are non-negative!
    ...
)
```

**Why It's Wrong**:
Linear models with bounded outcomes can:
1. Predict impossible values
2. Give non-interpretable coefficients
3. Violate parallel trends in levels when satisfied in natural scale

**Correct Approach**:
```python
# CORRECT: Consider appropriate transformations
import numpy as np

# For positive continuous outcomes: log transformation
df['log_y'] = np.log(df['y'] + 1)
result_log = estimate_did_panel(df, "log_y", ...)

# For proportions: logit transformation
df['logit_p'] = np.log(df['p'] / (1 - df['p']))
result_logit = estimate_did_panel(df, "logit_p", ...)

# For counts: Poisson pseudo-maximum likelihood
from scripts.robustness_checks import poisson_did
result_poisson = poisson_did(df, "count_y", ...)

# Compare results across specifications
print("Linear:", result_linear.effect)
print("Log:", np.exp(result_log.effect) - 1, "(% change)")
print("Poisson:", result_poisson.effect)
```

---

### Error 4.3: Incorrect Treatment Definition

**The Mistake**:
```python
# WRONG: Treatment indicator doesn't match research design

# Case 1: Using treatment group indicator instead of treatment status
df['wrong_treatment'] = df['treatment_group']  # 1 for all treated, all periods

# Case 2: Not accounting for treatment intensity
df['wrong_treatment'] = (df['subsidy_amount'] > 0).astype(int)  # Ignores amount
```

**Why It's Wrong**:
1. Treatment indicator should be 1 only when treatment is actually received
2. Ignoring intensity treats partial and full treatment as identical
3. Wrong indicator biases estimates

**Correct Approach**:
```python
# CORRECT: Treatment = (in treatment group) AND (post-treatment period)
df['treatment'] = (
    (df['treatment_group'] == 1) &
    (df['year'] >= df['treatment_year'])
).astype(int)

# For intensity: Consider continuous treatment
df['treatment_intensity'] = df['subsidy_amount'] * df['post']
result = estimate_did_panel(
    data=df,
    outcome="y",
    treatment="treatment_intensity",
    ...
)
```

---

## 5. Sample Selection Errors

### Error 5.1: Conditioning on Post-Treatment Variables

**The Mistake**:
```python
# WRONG: Restricting sample based on post-treatment outcomes
df_sample = df[df['survived'] == 1]  # Only firms that survived
result = estimate_did_panel(df_sample, ...)
```

**Why It's Wrong**:
If survival is affected by treatment, conditioning on survival:
1. Creates selection bias
2. Compares non-comparable groups
3. Can reverse sign of true effect

**Correct Approach**:
```python
# CORRECT: Don't condition on post-treatment outcomes

# Option 1: Include all units (code non-survivors as 0 or missing)
result = estimate_did_panel(df, ...)

# Option 2: Lee bounds for selection
from scripts.robustness_checks import lee_bounds
bounds = lee_bounds(df, "y", "treated", "survived")
print(f"Bounds: [{bounds['lower']:.4f}, {bounds['upper']:.4f}]")

# Option 3: Principal stratification / always-survivors
from scripts.robustness_checks import principal_strata
result_ps = principal_strata(df, "y", "treated", "survived")
```

---

### Error 5.2: Using Unbalanced Panel Without Addressing It

**The Mistake**:
```python
# WRONG: Ignoring unbalanced panel
result = estimate_did_panel(df, ...)  # Some units have missing years
```

**Why It's Wrong**:
Unbalanced panels can cause:
1. Composition changes over time
2. Selection into sample correlated with treatment
3. Changing weights on different units

**Correct Approach**:
```python
# Check for balance
obs_per_unit = df.groupby('id')['year'].nunique()
n_complete = (obs_per_unit == obs_per_unit.max()).sum()
print(f"Balanced units: {n_complete}/{len(obs_per_unit)}")

if obs_per_unit.nunique() > 1:
    # Option 1: Restrict to balanced subsample
    balanced_ids = obs_per_unit[obs_per_unit == obs_per_unit.max()].index
    df_balanced = df[df['id'].isin(balanced_ids)]
    result_balanced = estimate_did_panel(df_balanced, ...)

    # Option 2: Explicitly model selection
    # (more complex, see Heckman selection models)

    # Compare results
    print(f"Full sample: {result_full.effect:.4f}")
    print(f"Balanced sample: {result_balanced.effect:.4f}")
    print(f"Difference: {result_full.effect - result_balanced.effect:.4f}")
```

---

## 6. Interpretation Errors

### Error 6.1: Extrapolating Beyond Data

**The Mistake**:
```python
# WRONG: Claiming universal effect from specific context
print(f"Our results show that minimum wage increases employment by {effect}")
# When: study was only in fast food restaurants in NJ/PA
```

**Why It's Wrong**:
DID identifies a **local** average treatment effect. The estimate may not generalize to:
1. Different populations
2. Different treatment intensities
3. Different time periods
4. Different institutional contexts

**Correct Approach**:
```markdown
# CORRECT: Acknowledge limitations
Our estimates represent the average effect of [specific treatment] on
[specific outcome] in [specific population] during [specific time period].

External validity considerations:
- Treatment intensity: Our treatment was [X]; effects of [Y] may differ
- Population: Our sample is [description]; generalization to [other groups] is uncertain
- Context: [Specific features] of our setting may limit generalizability
```

---

### Error 6.2: Confusing Statistical and Economic Significance

**The Mistake**:
```python
# WRONG: Focusing only on statistical significance
if result.p_value < 0.05:
    print("Treatment works!")  # But effect might be tiny
```

**Why It's Wrong**:
With large samples, even trivial effects can be statistically significant. Policy relevance depends on magnitude, not just p-values.

**Correct Approach**:
```python
# CORRECT: Report both statistical and economic significance
print(f"Treatment effect: {result.effect:.4f} (SE = {result.se:.4f})")
print(f"P-value: {result.p_value:.4f}")

# Economic significance
pre_mean = df[df['post'] == 0]['y'].mean()
pct_change = (result.effect / pre_mean) * 100
print(f"% change from baseline: {pct_change:.1f}%")

# Cost-effectiveness (if applicable)
cost_per_unit = 1000  # Example
effect_per_dollar = result.effect / cost_per_unit
print(f"Effect per $1000 spent: {effect_per_dollar:.4f}")
```

---

### Error 6.3: Causal Language Without Discussing Assumptions

**The Mistake**:
```python
# WRONG: Making causal claims without discussing assumptions
print(f"Policy X causes Y to increase by {effect}")
```

**Why It's Wrong**:
Causal interpretation requires assumptions to hold. Readers need to evaluate whether assumptions are plausible.

**Correct Approach**:
```markdown
# CORRECT: Discuss assumptions and limitations

We interpret our DID estimate as the causal effect of [treatment] on [outcome]
under the following assumptions:

1. **Parallel Trends**: We provide supporting evidence in Figure X and Table Y.
   The pre-treatment coefficients are [individually/jointly] insignificant.
   However, this test has [limitations].

2. **No Anticipation**: We find [no evidence / some evidence] of anticipation
   effects in periods t-1 and t-2.

3. **SUTVA**: Given the nature of treatment and geographic distance between
   units, we believe spillovers are [unlikely / a potential concern because...].

Threats to identification include: [list specific concerns for this application]
```

---

## 7. Quick Reference: Error Checklist

### Before Estimation

- [ ] Is treatment timing staggered? If yes, consider robust estimators
- [ ] Do I have enough pre-treatment periods (4+) to test parallel trends?
- [ ] Are my controls pre-treatment or time-invariant?
- [ ] Is my treatment indicator correctly defined?

### During Estimation

- [ ] Am I clustering at the correct level?
- [ ] Have I checked for parallel pre-trends?
- [ ] Have I run placebo tests?
- [ ] If staggered, have I checked the Bacon decomposition?

### After Estimation

- [ ] Do my standard errors make sense given clustering?
- [ ] Have I reported economic significance?
- [ ] Have I discussed assumptions and limitations?
- [ ] Have I run robustness checks?

---

## See Also

- [identification_assumptions.md](identification_assumptions.md) - Detailed assumption explanations
- [diagnostic_tests.md](diagnostic_tests.md) - How to test for these errors
- [estimation_methods.md](estimation_methods.md) - Correct estimation approaches
