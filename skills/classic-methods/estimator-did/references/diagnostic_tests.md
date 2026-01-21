# Diagnostic Tests for Difference-in-Differences

> **Document Type**: Reference | **Last Updated**: 2025-01
> **Related**: [identification_assumptions.md](identification_assumptions.md), [robustness_checks.py](../scripts/robustness_checks.py)

## Overview

This document provides comprehensive guidance on diagnostic tests for DID estimation. Each test is designed to assess the validity of identification assumptions and the robustness of estimates.

---

## 1. Parallel Trends Tests

### 1.1 Visual Inspection

**Purpose**: The most important first step - examine raw trends visually.

**Implementation**:
```python
from did_estimator import plot_parallel_trends

fig = plot_parallel_trends(
    data=df,
    outcome="y",
    treatment_group="treated_ever",
    time_id="year",
    treatment_time=2015,
    figsize=(12, 6)
)
fig.savefig("parallel_trends.png", dpi=300)
```

**What to Look For**:
| Pattern | Interpretation | Action |
|---------|---------------|--------|
| Parallel lines pre-treatment | Supports parallel trends | Proceed |
| Converging pre-trends | Treatment group catching up | Consider synthetic control |
| Diverging pre-trends | Groups trending differently | Critical violation |
| Level differences only | OK - DID allows this | Proceed |
| Different volatility | May indicate heterogeneity | Check robustness |

### 1.2 Event Study Regression

**Purpose**: Formally test for differential pre-trends using regression.

**Model Specification**:
$$
Y_{it} = \alpha_i + \lambda_t + \sum_{k \neq -1} \beta_k \cdot \mathbf{1}[K_{it} = k] + \varepsilon_{it}
$$

Where:
- $K_{it}$ = event time (periods relative to treatment)
- $k = -1$ is the reference period (normalized to zero)
- $\beta_k$ for $k < 0$ should be insignificant if parallel trends holds

**Implementation**:
```python
from scripts.test_parallel_trends import run_event_study_test

results = run_event_study_test(
    data=df,
    outcome="y",
    treatment_time="first_treated",
    unit_id="id",
    time_id="year",
    pre_periods=4,
    post_periods=4,
    reference_period=-1,
    cluster="id"
)

# Examine pre-treatment coefficients
for k, coef_info in results['coefficients'].items():
    if k < 0:
        print(f"Period {k}: {coef_info['estimate']:.4f} "
              f"(SE={coef_info['se']:.4f}, p={coef_info['pval']:.4f})")
```

**Interpretation Criteria**:
- **Individual coefficients**: Each pre-period coefficient should be insignificant (p > 0.05)
- **Joint F-test**: All pre-period coefficients should be jointly insignificant
- **Visual pattern**: No trending pattern in pre-period coefficients

### 1.3 Formal Parallel Trends Test

**Purpose**: Statistical test for differential pre-trends.

```python
from did_estimator import test_parallel_trends

result = test_parallel_trends(
    data=df,
    outcome="y",
    treatment_group="treated_ever",
    time_id="year",
    unit_id="id",
    treatment_time=2015,
    n_pre_periods=4
)

print(f"Test statistic: {result.statistic:.4f}")
print(f"P-value: {result.p_value:.4f}")
print(f"Conclusion: {'PASS' if result.passed else 'FAIL'}")
```

**Test Details**:
- Tests for significant slope in the difference between group trends
- H0: No differential trend (slope = 0)
- H1: Groups trending differently (slope != 0)
- Rejection (p < 0.05) suggests parallel trends violation

### 1.4 Rambachan-Roth Sensitivity Analysis

**Purpose**: Assess sensitivity to parallel trends violations.

**Concept**: Rather than testing if parallel trends holds, ask: "How large would the violation need to be to explain away the result?"

```python
from scripts.robustness_checks import rambachan_roth_bounds

bounds = rambachan_roth_bounds(
    data=df,
    outcome="y",
    treatment="treated",
    unit_id="id",
    time_id="year",
    treatment_time=2015,
    max_deviation_range=[-0.5, 0.5],  # Range of M values to explore
    confidence_level=0.95
)

# Plot sensitivity analysis
bounds.plot_sensitivity()
```

**Interpretation**:
- $M$ = maximum deviation from parallel trends
- If effect remains significant for plausible $M$, results are robust
- Breakdown point: smallest $M$ that makes effect insignificant

---

## 2. Placebo Tests

### 2.1 Fake Treatment Timing Placebo

**Purpose**: Test whether "effects" appear at times when no treatment occurred.

**Logic**: If we find significant "effects" before actual treatment, parallel trends is likely violated.

```python
from did_estimator import placebo_test

# Test with fake treatment 2 periods before actual
placebo_result = placebo_test(
    data=df,
    outcome="y",
    treatment_group="treated_ever",
    unit_id="id",
    time_id="year",
    actual_treatment_time=2015,
    placebo_treatment_time=2013
)

print(f"Placebo effect: {placebo_result.statistic:.4f}")
print(f"P-value: {placebo_result.p_value:.4f}")
# Should be INSIGNIFICANT (p > 0.1)
```

**Multiple Placebo Periods**:
```python
from scripts.robustness_checks import multi_placebo_test

# Test multiple fake treatment timings
results = multi_placebo_test(
    data=df,
    outcome="y",
    treatment_group="treated_ever",
    unit_id="id",
    time_id="year",
    actual_treatment_time=2015,
    placebo_times=[2011, 2012, 2013, 2014]
)

# Should have no significant effects at any placebo time
for time, res in results.items():
    status = "PASS" if res['p_value'] > 0.1 else "FAIL"
    print(f"Placebo {time}: effect={res['effect']:.4f}, p={res['p_value']:.4f} [{status}]")
```

### 2.2 Placebo Treatment Group

**Purpose**: Test whether "treatment effect" appears for a fake treatment group.

**Logic**: Randomly assign units to fake "treatment" and check for effects.

```python
from scripts.robustness_checks import placebo_treatment_group

results = placebo_treatment_group(
    data=df,
    outcome="y",
    unit_id="id",
    time_id="year",
    treatment_time=2015,
    n_permutations=500,  # Number of random assignments
    actual_effect=main_result.effect
)

# Compare actual effect to permutation distribution
print(f"Actual effect: {results['actual_effect']:.4f}")
print(f"Permutation mean: {results['permutation_mean']:.4f}")
print(f"Permutation SD: {results['permutation_sd']:.4f}")
print(f"Permutation p-value: {results['p_value']:.4f}")
```

### 2.3 Placebo Outcome

**Purpose**: Test whether treatment affects outcomes it should NOT affect.

**Logic**: If treatment is finding spurious effects, it might also "affect" unrelated outcomes.

```python
from scripts.robustness_checks import placebo_outcome_test

# Test effect on outcomes that should NOT be affected
placebo_outcomes = ["weather_temp", "national_gdp", "unrelated_metric"]

for outcome in placebo_outcomes:
    result = placebo_outcome_test(
        data=df,
        placebo_outcome=outcome,
        treatment="treated",
        unit_id="id",
        time_id="year",
        cluster="id"
    )
    status = "PASS" if result['p_value'] > 0.1 else "CONCERN"
    print(f"{outcome}: effect={result['effect']:.4f}, p={result['p_value']:.4f} [{status}]")
```

---

## 3. Balance Tests

### 3.1 Pre-Treatment Covariate Balance

**Purpose**: Check if treatment and control groups are similar on observables before treatment.

```python
from did_estimator import balance_test_did

balance = balance_test_did(
    data=df,
    treatment_group="treated_ever",
    covariates=["age", "income", "education", "firm_size"],
    time_id="year",
    pre_period=2014  # Last pre-treatment period
)

# Print balance table
for var, result in balance.items():
    diff = result['treat_mean'] - result['control_mean']
    std_diff = diff / result['pooled_sd']
    print(f"{var}:")
    print(f"  Treatment mean: {result['treat_mean']:.3f}")
    print(f"  Control mean: {result['control_mean']:.3f}")
    print(f"  Std. difference: {std_diff:.3f}")
    print(f"  P-value: {result['p_value']:.4f}")
```

**Interpretation**:
| Standardized Difference | Interpretation |
|------------------------|----------------|
| < 0.1 | Negligible imbalance |
| 0.1 - 0.25 | Small imbalance |
| 0.25 - 0.5 | Moderate imbalance |
| > 0.5 | Large imbalance - concern |

### 3.2 Normalized Difference

**Purpose**: More robust measure of imbalance than t-tests.

$$
\Delta = \frac{\bar{X}_T - \bar{X}_C}{\sqrt{(S_T^2 + S_C^2)/2}}
$$

```python
from scripts.robustness_checks import normalized_difference

for var in covariates:
    nd = normalized_difference(
        data=df,
        variable=var,
        treatment_group="treated_ever"
    )
    status = "OK" if abs(nd) < 0.25 else "CONCERN"
    print(f"{var}: {nd:.3f} [{status}]")
```

### 3.3 Joint Balance Test

**Purpose**: Test overall balance across all covariates simultaneously.

```python
from scripts.robustness_checks import joint_balance_test

# F-test that all covariates jointly predict treatment
result = joint_balance_test(
    data=df[df['year'] == 2014],  # Pre-treatment period
    treatment="treated_ever",
    covariates=["age", "income", "education", "firm_size"]
)

print(f"F-statistic: {result['f_stat']:.3f}")
print(f"P-value: {result['p_value']:.4f}")
# High p-value suggests good overall balance
```

---

## 4. Specification Tests

### 4.1 Goodman-Bacon Decomposition

**Purpose**: For staggered DID, decompose the TWFE estimate into 2x2 DID comparisons.

**Why It Matters**: TWFE with staggered treatment can be biased when:
- Treatment effects are heterogeneous across time
- Already-treated units serve as controls for later-treated units

```python
from scripts.robustness_checks import bacon_decomposition

decomp = bacon_decomposition(
    data=df,
    outcome="y",
    treatment="treated",
    unit_id="id",
    time_id="year"
)

# Print decomposition
print("Goodman-Bacon Decomposition:")
print(f"Overall TWFE estimate: {decomp['twfe_estimate']:.4f}")
print("\nComponent weights and estimates:")
for comp_type, values in decomp['components'].items():
    print(f"\n{comp_type}:")
    print(f"  Weight: {values['weight']:.3f}")
    print(f"  Estimate: {values['estimate']:.4f}")
```

**Key Concern**: "Earlier vs Later Treated" comparisons can have negative weights if effects are dynamic.

### 4.2 Heterogeneity-Robust Estimators

**Purpose**: When Bacon decomposition shows problematic comparisons, use robust methods.

```python
from did_estimator import estimate_did_staggered

# Callaway-Sant'Anna estimator
result_cs = estimate_did_staggered(
    data=df,
    outcome="y",
    treatment_time="first_treated",
    unit_id="id",
    time_id="year",
    control_group="nevertreated"  # or "notyettreated"
)

# Compare to TWFE
print(f"TWFE estimate: {result_twfe.effect:.4f}")
print(f"C-S estimate: {result_cs.effect:.4f}")
print(f"Difference: {result_twfe.effect - result_cs.effect:.4f}")
```

### 4.3 Functional Form Tests

**Purpose**: Test sensitivity to model specification.

```python
from scripts.robustness_checks import functional_form_tests

results = functional_form_tests(
    data=df,
    outcome="y",
    treatment="treated",
    unit_id="id",
    time_id="year",
    controls=["x1", "x2"]
)

print("Functional Form Sensitivity:")
for spec, res in results.items():
    print(f"{spec}: {res['effect']:.4f} (SE={res['se']:.4f})")
```

**Specifications to Test**:
- Linear levels
- Log outcome (if positive)
- Standardized outcome
- With/without controls
- Linear vs. quadratic controls

---

## 5. Inference Diagnostics

### 5.1 Clustering Level

**Purpose**: Test sensitivity of standard errors to different clustering levels.

```python
from scripts.robustness_checks import cluster_sensitivity

results = cluster_sensitivity(
    data=df,
    outcome="y",
    treatment="treated",
    unit_id="id",
    time_id="year",
    cluster_vars=["id", "state", "industry"]  # Different clustering levels
)

print("Standard Error Sensitivity to Clustering:")
print(f"{'Cluster Level':<15} {'Effect':>10} {'SE':>10} {'P-value':>10}")
for cluster, res in results.items():
    print(f"{cluster:<15} {res['effect']:>10.4f} {res['se']:>10.4f} {res['p_value']:>10.4f}")
```

**Guidelines**:
- Cluster at level of treatment variation
- Higher-level clustering generally more conservative
- Concern if results flip significance with different clustering

### 5.2 Wild Bootstrap

**Purpose**: More reliable inference with few clusters.

```python
from scripts.robustness_checks import wild_cluster_bootstrap

boot_result = wild_cluster_bootstrap(
    data=df,
    outcome="y",
    treatment="treated",
    unit_id="id",
    time_id="year",
    cluster="state",
    n_bootstrap=999,
    weight_type="rademacher"  # or "mammen"
)

print(f"Effect: {boot_result['effect']:.4f}")
print(f"Bootstrap SE: {boot_result['boot_se']:.4f}")
print(f"Bootstrap 95% CI: [{boot_result['ci_lower']:.4f}, {boot_result['ci_upper']:.4f}]")
print(f"Bootstrap p-value: {boot_result['p_value']:.4f}")
```

### 5.3 Randomization Inference

**Purpose**: Exact inference by permuting treatment assignment.

```python
from scripts.robustness_checks import randomization_inference

ri_result = randomization_inference(
    data=df,
    outcome="y",
    treatment_group="treated_ever",
    unit_id="id",
    time_id="year",
    treatment_time=2015,
    n_permutations=1000
)

print(f"Actual effect: {ri_result['actual_effect']:.4f}")
print(f"RI p-value: {ri_result['p_value']:.4f}")
print(f"Proportion larger: {ri_result['prop_larger']:.4f}")
```

---

## 6. Diagnostic Summary Report

### Generating a Complete Diagnostic Report

```python
from scripts.robustness_checks import run_full_diagnostics

report = run_full_diagnostics(
    data=df,
    outcome="y",
    treatment="treated",
    treatment_group="treated_ever",
    unit_id="id",
    time_id="year",
    treatment_time=2015,
    controls=["x1", "x2"],
    cluster="id"
)

# Print summary
report.print_summary()

# Export to markdown
report.to_markdown("diagnostics_report.md")

# Get LaTeX table
latex_table = report.to_latex()
```

### Diagnostic Checklist

| Test | Status | Interpretation |
|------|--------|----------------|
| Visual parallel trends | REQUIRED | Plot and inspect |
| Event study pre-coefficients | REQUIRED | All should be insignificant |
| Parallel trends test | REQUIRED | p > 0.05 |
| Placebo timing test | RECOMMENDED | No fake effects |
| Balance tests | RECOMMENDED | Std diff < 0.25 |
| Bacon decomposition | Required if staggered | Check for bad comparisons |
| Cluster robustness | RECOMMENDED | Results stable |
| Wild bootstrap | If few clusters | Check inference |

---

## References

1. Roth, J. (2022). "Pretest with Caution: Event-Study Estimates after Testing for Parallel Trends." *AER: Insights*.

2. Goodman-Bacon, A. (2021). "Difference-in-Differences with Variation in Treatment Timing." *Journal of Econometrics*.

3. Cameron, A. C., Gelbach, J. B., & Miller, D. L. (2008). "Bootstrap-Based Improvements for Inference with Clustered Errors." *REStat*.

4. MacKinnon, J. G., & Webb, M. D. (2018). "The Wild Bootstrap for Few (Treated) Clusters." *Econometrics Journal*.

---

## See Also

- [identification_assumptions.md](identification_assumptions.md) - What the tests are testing
- [common_errors.md](common_errors.md) - Mistakes in diagnostic testing
- [robustness_checks.py](../scripts/robustness_checks.py) - Implementation of tests
