# Common RD Errors and Pitfalls

> **Document Type**: Error Prevention Guide
> **Last Updated**: 2024-01
> **Purpose**: Help practitioners avoid frequent mistakes in RD analysis

## Overview

This document catalogs common errors in RD implementation and analysis, with guidance on how to detect and correct them.

---

## 1. Bandwidth Selection Errors

### Error 1.1: Arbitrary Bandwidth Choice

**Mistake**: Choosing bandwidth "by eye" or using round numbers without justification.

**Example**:
```python
# BAD: Arbitrary bandwidth
result = estimate_sharp_rd(data, running, outcome, cutoff, bandwidth=1.0)
```

**Why It's Wrong**:
- Ignores bias-variance tradeoff
- May introduce substantial bias (too large) or imprecision (too small)
- Not reproducible or defensible

**Correct Approach**:
```python
# GOOD: Data-driven optimal bandwidth
from rd_estimator import select_bandwidth

h_opt = select_bandwidth(
    running=df["score"],
    outcome=df["y"],
    cutoff=0.0,
    method="mserd"  # MSE-optimal
)

result = estimate_sharp_rd(data, running, outcome, cutoff, bandwidth=h_opt)
```

### Error 1.2: Using Full Sample Bandwidth

**Mistake**: Using all data rather than a local bandwidth.

**Why It's Wrong**:
- Global approach violates the local nature of RD
- Estimates depend on functional form far from cutoff
- Higher bias, especially with nonlinear relationships

**Detection**:
```python
# Check: Is bandwidth unreasonably large?
data_range = df["score"].max() - df["score"].min()
if bandwidth > data_range / 3:
    print("WARNING: Bandwidth may be too large for local estimation")
```

### Error 1.3: Not Checking Bandwidth Sensitivity

**Mistake**: Reporting only one bandwidth without sensitivity analysis.

**Correct Approach**:
```python
# Always check sensitivity
from rd_estimator import bandwidth_sensitivity

sens = bandwidth_sensitivity(
    data=df,
    running="score",
    outcome="y",
    cutoff=0.0,
    bandwidth_range=[0.5*h_opt, 0.75*h_opt, h_opt, 1.25*h_opt, 1.5*h_opt, 2*h_opt]
)

# Check stability
effects = [r['effect'] for r in sens['results']]
effect_range = max(effects) - min(effects)

if effect_range > 0.5 * abs(np.mean(effects)):
    print("WARNING: Results are sensitive to bandwidth choice")
```

---

## 2. Polynomial Order Errors

### Error 2.1: Using High-Order Global Polynomials

**Mistake**: Fitting polynomials of order 3, 4, or higher across the entire sample.

**Example of Bad Practice**:
```python
# BAD: Global high-order polynomial
import statsmodels.api as sm

# This approach is WRONG for RD
X = np.column_stack([
    df["score"],
    df["score"]**2,
    df["score"]**3,
    df["score"]**4,
    df["treated"]
])
model = sm.OLS(df["y"], sm.add_constant(X)).fit()
```

**Why It's Wrong** (Gelman & Imbens, 2019):
- Highly sensitive to outliers and data far from cutoff
- Polynomial behavior at boundaries is erratic
- Confidence intervals have poor coverage
- Results depend heavily on polynomial order

**Correct Approach**:
```python
# GOOD: Local linear or quadratic
result = estimate_sharp_rd(
    data=df,
    running="score",
    outcome="y",
    cutoff=0.0,
    bandwidth=h_opt,  # Local estimation
    order=1           # Linear (or 2 for quadratic)
)
```

### Error 2.2: Defaulting to Quadratic Without Justification

**Mistake**: Using quadratic specification as default without checking if linear is sufficient.

**Better Practice**:
```python
# Run both and compare
result_linear = estimate_sharp_rd(data, running, outcome, cutoff, bandwidth=h_opt, order=1)
result_quadratic = estimate_sharp_rd(data, running, outcome, cutoff, bandwidth=h_opt, order=2)

print(f"Linear effect: {result_linear.effect:.4f} (SE: {result_linear.se:.4f})")
print(f"Quadratic effect: {result_quadratic.effect:.4f} (SE: {result_quadratic.se:.4f})")

# Prefer linear unless there's strong reason for quadratic
# and results are similar
```

---

## 3. Data Heaping Errors

### Error 3.1: Ignoring Heaping at Round Numbers

**Mistake**: Not checking whether the running variable heaps at certain values.

**Examples of Heaping**:
- Test scores round to nearest 5 or 10
- Ages reported as round years
- Income reported in brackets

**Detection**:
```python
import matplotlib.pyplot as plt

# Check for heaping
score_counts = df["score"].value_counts()
top_values = score_counts.head(10)

# Plot distribution
fig, ax = plt.subplots(figsize=(12, 4))
df["score"].hist(bins=100, ax=ax, edgecolor='black', alpha=0.7)
ax.axvline(x=cutoff, color='red', linestyle='--')
ax.set_title("Check for Heaping at Round Numbers")
plt.show()

# Alert if heaping detected
if score_counts.max() > 0.05 * len(df):
    print("WARNING: Potential heaping detected")
    print(f"Most common values: {top_values.index.tolist()}")
```

### Error 3.2: Not Addressing Discrete Running Variable

**Mistake**: Treating a discrete running variable as continuous.

**Problem**: Standard local polynomial methods assume continuity.

**Solutions**:
```python
# Check discreteness
n_unique = df["score"].nunique()

if n_unique < 50:
    print(f"Running variable has only {n_unique} unique values")
    print("Consider:")
    print("1. Local randomization approach")
    print("2. Clustering standard errors at running variable values")
    print("3. Parametric specification with mass points")
```

---

## 4. Manipulation and Validity Errors

### Error 4.1: Skipping McCrary Test

**Mistake**: Not testing for manipulation before estimating.

**Why It Matters**: Manipulation violates the key identifying assumption.

**Correct Approach**:
```python
from rd_estimator import mccrary_test

# ALWAYS run McCrary test first
density_result = mccrary_test(
    running=df["score"],
    cutoff=0.0
)

if not density_result.passed:
    print("WARNING: Evidence of manipulation detected!")
    print(f"Log density difference: {density_result.statistic:.4f}")
    print(f"P-value: {density_result.p_value:.4f}")
    print("\nRemedies to consider:")
    print("1. Donut hole RD")
    print("2. Alternative identification strategy")
    print("3. Report as intent-to-treat with caveats")
```

### Error 4.2: Ignoring Covariate Imbalance

**Mistake**: Not checking whether covariates are balanced at the cutoff.

**Problem**: Imbalanced covariates suggest sorting or confounding.

**Correct Approach**:
```python
from rd_estimator import covariate_balance_rd

balance = covariate_balance_rd(
    data=df,
    running="score",
    cutoff=0.0,
    covariates=["age", "gender", "income", "prior_score"]
)

imbalanced = [cov for cov, r in balance.items() if not r.passed]

if imbalanced:
    print(f"WARNING: Imbalanced covariates: {imbalanced}")
    print("This may indicate:")
    print("1. Sorting around cutoff")
    print("2. Confounding policy changes at cutoff")
    print("3. Data quality issues")
```

### Error 4.3: Multiple Cutoffs at Same Threshold

**Mistake**: Not considering whether other policies share the same cutoff.

**Example**: Both a scholarship AND tutoring program use the same test score threshold.

**Detection Questions**:
- Are there other programs with the same eligibility threshold?
- Did any policy changes occur at this cutoff?
- Are there institutional reasons for this specific cutoff value?

**Mitigation**:
```python
# Document all policies at the cutoff
# Report as a bundled treatment effect if multiple policies
print("The estimated effect represents the combined impact of:")
print("1. [Policy A]")
print("2. [Policy B that shares the cutoff]")
```

---

## 5. Inference Errors

### Error 5.1: Using Conventional Standard Errors

**Mistake**: Not using robust bias-corrected inference.

**Problem**: Conventional CIs undercover when using MSE-optimal bandwidth.

**Correct Approach**:
```python
# Use robust bias-corrected CI
result = estimate_sharp_rd(data, running, outcome, cutoff, bandwidth=h_mse)

# Report robust CI, not conventional
print(f"Effect: {result.effect:.4f}")
print(f"Robust 95% CI: [{result.ci_lower:.4f}, {result.ci_upper:.4f}]")
# Note: result.ci_lower and result.ci_upper should be bias-corrected
```

### Error 5.2: Clustering Incorrectly

**Mistake**: Clustering at the wrong level in RD.

**When to Cluster**:
- If running variable is discrete with few values
- If there's group structure (e.g., schools)

**Common Error**: Clustering on the running variable when it's continuous.

```python
# Generally NOT needed for continuous running variable
# Only cluster if:
# 1. Running variable is discrete
# 2. There's a meaningful group structure

if running_is_discrete:
    # Use cluster-robust standard errors at running variable values
    pass
else:
    # Use heteroskedasticity-robust standard errors (default)
    pass
```

### Error 5.3: Multiple Hypothesis Testing Without Correction

**Mistake**: Running many specifications without adjusting for multiple testing.

**Correct Approach**:
```python
from scipy import stats

# If running many placebo tests, use Bonferroni or other correction
n_tests = len(placebo_cutoffs)
bonferroni_alpha = 0.05 / n_tests

print(f"Using Bonferroni-corrected alpha = {bonferroni_alpha:.4f}")

for cutoff, result in placebo_results.items():
    significant = result.p_value < bonferroni_alpha
    print(f"Cutoff {cutoff}: p = {result.p_value:.4f} {'*' if significant else ''}")
```

---

## 6. Sharp vs Fuzzy RD Errors

### Error 6.1: Treating Fuzzy as Sharp

**Mistake**: Using sharp RD when treatment compliance is imperfect.

**Detection**:
```python
# Check compliance rates
above_cutoff = df[df["score"] >= cutoff]
below_cutoff = df[df["score"] < cutoff]

treat_rate_above = above_cutoff["treated"].mean()
treat_rate_below = below_cutoff["treated"].mean()

print(f"Treatment rate above cutoff: {treat_rate_above:.1%}")
print(f"Treatment rate below cutoff: {treat_rate_below:.1%}")

if treat_rate_above < 0.99 or treat_rate_below > 0.01:
    print("WARNING: Imperfect compliance detected!")
    print("Use Fuzzy RD, not Sharp RD")
```

### Error 6.2: Weak First Stage in Fuzzy RD

**Mistake**: Proceeding with fuzzy RD when the first stage is weak.

**Problem**: Weak first stage leads to biased estimates and unreliable inference.

**Detection and Response**:
```python
result = estimate_fuzzy_rd(data, running, outcome, treatment, cutoff)

first_stage = result.diagnostics['first_stage']

if abs(first_stage) < 0.10:
    print(f"WARNING: Weak first stage ({first_stage:.4f})")
    print("Fuzzy RD estimates may be unreliable.")
    print("\nConsider:")
    print("1. Reporting reduced form (ITT) instead")
    print("2. Using weak-instrument robust inference")
    print("3. Investigating why compliance is so low")
```

---

## 7. Visualization Errors

### Error 7.1: Misleading RD Plots

**Mistake**: Creating plots that exaggerate or hide the discontinuity.

**Common Issues**:
- Y-axis doesn't start at zero when it should
- Too few bins hide the data pattern
- Lines extend beyond the data
- Missing error bars

**Correct Approach**:
```python
# Ensure informative, honest visualization
fig = rd_plot(
    data=df,
    running="score",
    outcome="y",
    cutoff=0.0,
    bandwidth=h_opt,
    n_bins=20,    # Enough bins to see pattern
    ci=True       # Always show uncertainty
)

# Consider whether y-axis should start at zero
# Depends on context and what's meaningful
```

### Error 7.2: Only Showing Fitted Lines

**Mistake**: RD plot shows only polynomial fits without data points.

**Problem**: Hides actual data variability and potential issues.

**Correct**: Always include binned scatter points alongside fitted lines.

---

## 8. Interpretation Errors

### Error 8.1: Generalizing Beyond the Cutoff

**Mistake**: Interpreting RD estimate as an average treatment effect for the whole population.

**Reality**: RD estimates the Local Average Treatment Effect (LATE) **at the cutoff**.

**Correct Interpretation**:
```python
interpretation = f"""
The RD estimate of {result.effect:.4f} represents the effect of treatment
for individuals at the margin of eligibility (score = {cutoff}).

This is a LOCAL effect and should not be extrapolated to:
- Individuals far from the cutoff
- The average treatment effect for all treated
- Settings with different cutoff values

External validity requires additional assumptions.
"""
print(interpretation)
```

### Error 8.2: Confusing Intent-to-Treat with Treatment Effect

**Mistake**: In fuzzy RD, confusing the reduced form (ITT) with the treatment effect (LATE).

**Clarification**:
```python
print("Fuzzy RD Estimates:")
print(f"Reduced Form (ITT): {result.diagnostics['reduced_form']:.4f}")
print(f"First Stage: {result.diagnostics['first_stage']:.4f}")
print(f"LATE (IV estimate): {result.effect:.4f}")

print("\nInterpretation:")
print("- Reduced Form: Effect of crossing cutoff (regardless of treatment)")
print("- LATE: Effect of actual treatment for compliers at cutoff")
```

---

## 9. Code Implementation Errors

### Error 9.1: Not Centering Running Variable

**Mistake**: Fitting polynomials without centering at cutoff.

**Problem**: Numerical instability and interpretation issues.

**Correct**:
```python
# The estimation should center at cutoff internally
# If implementing manually:
x_centered = df["score"] - cutoff

# Then fit polynomial on x_centered
# The intercept difference gives the RD effect
```

### Error 9.2: Including Post-Treatment Covariates

**Mistake**: Controlling for variables that could be affected by treatment.

**Example of Error**:
```python
# BAD: 'current_gpa' is affected by treatment (scholarship)
result = estimate_sharp_rd(
    data=df,
    running="score",
    outcome="graduation",
    cutoff=0.0,
    covariates=["current_gpa"]  # POST-TREATMENT! Don't include
)
```

**Correct**:
```python
# GOOD: Only pre-treatment covariates
result = estimate_sharp_rd(
    data=df,
    running="score",
    outcome="graduation",
    cutoff=0.0,
    covariates=["prior_gpa", "age_at_application"]  # Pre-treatment
)
```

---

## 10. Error Prevention Checklist

### Before Analysis
- [ ] Verify cutoff value is correct
- [ ] Check running variable for heaping/discreteness
- [ ] Identify all policies at this cutoff
- [ ] List all pre-treatment covariates

### During Analysis
- [ ] Run McCrary test first
- [ ] Use data-driven bandwidth selection
- [ ] Use local linear (not global polynomial)
- [ ] Apply robust bias-corrected inference
- [ ] Check for fuzzy vs sharp appropriately

### After Analysis
- [ ] Bandwidth sensitivity analysis
- [ ] Polynomial order sensitivity
- [ ] Covariate balance checks
- [ ] Placebo cutoff tests
- [ ] Interpret as LATE, not ATE

---

## References

- Gelman, A., & Imbens, G. (2019). Why high-order polynomials should not be used in regression discontinuity designs. *Journal of Business & Economic Statistics*, 37(3), 447-456.
- Lee, D. S., & Lemieux, T. (2010). Regression discontinuity designs in economics. *Journal of Economic Literature*, 48(2), 281-355.
- Cattaneo, M. D., Idrobo, N., & Titiunik, R. (2020). *A Practical Introduction to Regression Discontinuity Designs*. Cambridge University Press.
