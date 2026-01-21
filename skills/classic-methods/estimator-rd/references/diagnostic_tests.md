# RD Diagnostic Tests

> **Document Type**: Testing Reference
> **Last Updated**: 2024-01
> **Key References**: McCrary (2008), Cattaneo, Jansson & Ma (2020)

## Overview

Diagnostic tests are essential for validating the assumptions underlying RD designs. While the core continuity assumption cannot be directly tested, we can test observable implications that would be violated if the design is invalid.

---

## 1. McCrary Density Test

### Purpose
Tests for manipulation of the running variable by detecting discontinuities in its density at the cutoff.

### Intuition
If agents can precisely sort around the cutoff, we expect bunching (excess density) on the favorable side. A discontinuous density suggests manipulation.

### Methodology

**McCrary (2008) Approach**:
1. Construct a histogram of the running variable
2. Use local linear regression to smooth the histogram separately above and below cutoff
3. Test for a discontinuity in the log density at the cutoff

**Cattaneo, Jansson & Ma (2020) Refinements**:
- Robust bias-corrected inference
- Data-driven bandwidth selection
- More reliable in finite samples

### Implementation

```python
from rd_estimator import mccrary_test

# Basic usage
result = mccrary_test(
    running=df["score"],
    cutoff=0.0,
    bandwidth=None,  # Auto-select
    n_bins=None      # Auto-select
)

print(result)
# Output: McCrary Density Test
# Statistic: -0.0423
# P-value: 0.672
# Result: PASSED - No significant density discontinuity
```

### Interpretation Guide

| P-value | Interpretation | Action |
|---------|----------------|--------|
| > 0.10 | No evidence of manipulation | Proceed with RD |
| 0.05 - 0.10 | Marginal evidence | Report sensitivity, consider donut hole |
| 0.01 - 0.05 | Moderate evidence | Strong concern, use donut hole RD |
| < 0.01 | Strong evidence | Serious validity threat, reconsider design |

### Visual Inspection

Always complement the statistical test with visual inspection:

```python
import matplotlib.pyplot as plt
import numpy as np

# Histogram around cutoff
fig, ax = plt.subplots(figsize=(10, 6))

# Create bins
bins = np.linspace(df["score"].min(), df["score"].max(), 50)
ax.hist(df["score"], bins=bins, edgecolor='black', alpha=0.7)
ax.axvline(x=cutoff, color='red', linestyle='--', linewidth=2, label='Cutoff')
ax.set_xlabel("Running Variable")
ax.set_ylabel("Frequency")
ax.set_title("Density of Running Variable")
ax.legend()
plt.show()
```

**Look for**:
- Bunching just above or below the cutoff
- Unusual gaps in the distribution
- Heaping at round numbers

### Limitations

1. **Statistical power**: May not detect small amounts of manipulation
2. **Type of manipulation**: Only detects precise sorting, not imprecise influence
3. **Sample size**: Requires sufficient observations near cutoff
4. **Discrete running variables**: Less reliable with few unique values

---

## 2. Covariate Smoothness Tests

### Purpose
Test whether pre-treatment covariates show discontinuities at the cutoff. Under valid RD, covariates determined before treatment should be smooth.

### Methodology
Run an RD regression using each covariate as the outcome. Significant discontinuities suggest:
- Sorting based on covariate values
- Confounding factors that change at the cutoff
- Data quality issues

### Implementation

```python
from rd_estimator import covariate_balance_rd

# Test multiple covariates
balance_results = covariate_balance_rd(
    data=df,
    running="score",
    cutoff=0.0,
    covariates=["age", "gender", "income", "education", "prior_gpa"],
    bandwidth=optimal_bw
)

# Summary table
print("\n=== Covariate Balance at Cutoff ===\n")
print("Covariate      | Jump      | SE       | P-value  | Status")
print("-" * 60)

for cov, result in balance_results.items():
    status = "OK" if result.passed else "FAIL"
    print(f"{cov:<14} | {result.statistic:>9.4f} | {result.details['se']:>8.4f} | {result.p_value:>8.4f} | {status}")
```

### Interpretation

**Good Signs**:
- All covariates balanced (p > 0.05)
- Point estimates of discontinuities are small
- Pattern of small, mixed-sign discontinuities

**Warning Signs**:
- Multiple imbalanced covariates
- Systematic pattern (e.g., all positive discontinuities)
- Large discontinuities even if not significant

### Joint Test

With many covariates, use joint hypothesis testing:

```python
from scipy import stats

# Collect p-values
p_values = [r.p_value for r in balance_results.values()]

# Fisher's method for combining p-values
chi2_stat = -2 * sum(np.log(p) for p in p_values)
joint_p_value = 1 - stats.chi2.cdf(chi2_stat, df=2*len(p_values))

print(f"\nJoint test p-value (Fisher): {joint_p_value:.4f}")
```

---

## 3. Placebo Cutoff Tests

### Purpose
Test for effects at "fake" cutoffs where no effect should exist. Finding effects at placebo cutoffs suggests model misspecification.

### Methodology
1. Choose placebo cutoffs on each side of the true cutoff
2. Restrict sample to one side of the true cutoff
3. Estimate RD at each placebo cutoff
4. Effects should be null (not significant)

### Implementation

```python
from rd_estimator import placebo_cutoff_test

# Define placebo cutoffs
# Use quantiles of running variable away from true cutoff
placebo_cutoffs = [-0.5, -0.25, 0.25, 0.5]  # Example values

results = placebo_cutoff_test(
    data=df,
    running="score",
    outcome="y",
    true_cutoff=0.0,
    placebo_cutoffs=placebo_cutoffs,
    bandwidth=optimal_bw
)

print("\n=== Placebo Cutoff Tests ===\n")
for cutoff, result in results.items():
    status = "PASS" if result.passed else "FAIL"
    print(f"Cutoff {cutoff:>6.2f}: Effect = {result.statistic:>7.4f}, "
          f"p = {result.p_value:.4f} [{status}]")
```

### Choosing Placebo Cutoffs

**Good Choices**:
- Quartiles of the running variable (excluding area around true cutoff)
- Natural thresholds that are NOT treatment-relevant
- Points with sufficient data on both sides

**Bad Choices**:
- Too close to true cutoff (contaminated by true effect)
- Where sample size is very small
- At data artifacts (heaping points)

### Expected Results

| Outcome | Interpretation |
|---------|----------------|
| All pass | Supports RD specification |
| 1-2 marginal failures | Acceptable (Type I error) |
| Multiple failures | Model misspecification likely |
| Systematic pattern | Functional form issue |

---

## 4. Bandwidth Sensitivity Analysis

### Purpose
Assess whether results are robust to bandwidth choice. Estimates should be stable across reasonable bandwidths.

### Implementation

```python
from rd_estimator import bandwidth_sensitivity, select_bandwidth

# Get optimal bandwidth
h_opt = select_bandwidth(df["score"], df["y"], cutoff=0.0)

# Test range of bandwidths
sensitivity = bandwidth_sensitivity(
    data=df,
    running="score",
    outcome="y",
    cutoff=0.0,
    bandwidth_range=[0.5*h_opt, 0.75*h_opt, h_opt, 1.25*h_opt, 1.5*h_opt, 2*h_opt],
    kernel="triangular",
    order=1
)

print(sensitivity['summary_table'])
```

### Interpretation Guide

**Stable Results** (Good):
- Point estimates vary modestly across bandwidths
- All confidence intervals overlap
- Statistical significance consistent
- No systematic trend in estimates

**Unstable Results** (Concerning):
- Large variation in point estimates
- Sign changes across bandwidths
- Significance flips with small bandwidth changes
- Systematic increase/decrease with bandwidth

### Visualization

```python
import matplotlib.pyplot as plt

# Extract results
bws = [r['bandwidth'] for r in sensitivity['results']]
effects = [r['effect'] for r in sensitivity['results']]
ci_lower = [r['ci_lower'] for r in sensitivity['results']]
ci_upper = [r['ci_upper'] for r in sensitivity['results']]

fig, ax = plt.subplots(figsize=(10, 6))

ax.plot(bws, effects, 'o-', markersize=8, label='Point Estimate')
ax.fill_between(bws, ci_lower, ci_upper, alpha=0.3, label='95% CI')
ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
ax.axvline(x=h_opt, color='red', linestyle=':', label=f'Optimal h={h_opt:.3f}')

ax.set_xlabel('Bandwidth')
ax.set_ylabel('RD Effect Estimate')
ax.set_title('Bandwidth Sensitivity Analysis')
ax.legend()
plt.show()
```

---

## 5. Donut Hole Sensitivity

### Purpose
Test whether results are driven by observations very close to the cutoff, which may be most susceptible to manipulation.

### Methodology
Exclude observations within a small "donut hole" around the cutoff and re-estimate.

### Implementation

```python
from rd_estimator import donut_hole_rd

# Test different donut radii
donut_radii = [0.01, 0.02, 0.05, 0.10]

print("\n=== Donut Hole Sensitivity ===\n")
print("Radius    | Effect   | SE      | N Excluded | P-value")
print("-" * 55)

for radius in donut_radii:
    result = donut_hole_rd(
        data=df,
        running="score",
        outcome="y",
        cutoff=0.0,
        bandwidth=optimal_bw,
        donut_radius=radius
    )

    n_excluded = result.diagnostics['n_excluded']
    print(f"{radius:>8.2f}  | {result.effect:>8.4f} | {result.se:>7.4f} | "
          f"{n_excluded:>10d} | {result.p_value:.4f}")
```

### When to Use Donut Hole

**Appropriate**:
- McCrary test suggests manipulation
- Concern about heaping at exact cutoff
- Data quality issues right at threshold

**Caution**:
- Reduces effective sample size
- May discard most informative observations
- Trade-off between bias and precision

---

## 6. Polynomial Order Sensitivity

### Purpose
Check whether results depend on the polynomial specification.

### Implementation

```python
from rd_estimator import estimate_sharp_rd

print("\n=== Polynomial Order Sensitivity ===\n")
print("Order | Effect   | SE      | 95% CI              | P-value")
print("-" * 60)

for order in [1, 2, 3]:
    result = estimate_sharp_rd(
        data=df,
        running="score",
        outcome="y",
        cutoff=0.0,
        bandwidth=optimal_bw,
        order=order
    )

    ci_str = f"[{result.ci_lower:.4f}, {result.ci_upper:.4f}]"
    print(f"{order:>5d} | {result.effect:>8.4f} | {result.se:>7.4f} | "
          f"{ci_str:>19s} | {result.p_value:.4f}")
```

### Best Practices

1. **Default to linear (order=1)**: Most robust, recommended by Gelman & Imbens (2019)
2. **Compare with quadratic**: As sensitivity check
3. **Avoid order > 2**: High-order polynomials can be erratic near boundaries
4. **Report multiple specifications**: Transparent about robustness

---

## 7. Kernel Choice Sensitivity

### Purpose
Verify results are not sensitive to kernel function choice.

### Implementation

```python
kernels = ["triangular", "epanechnikov", "uniform"]

print("\n=== Kernel Sensitivity ===\n")
print("Kernel        | Effect   | SE      | 95% CI")
print("-" * 50)

for kernel in kernels:
    result = estimate_sharp_rd(
        data=df,
        running="score",
        outcome="y",
        cutoff=0.0,
        bandwidth=optimal_bw,
        kernel=kernel
    )

    ci_str = f"[{result.ci_lower:.4f}, {result.ci_upper:.4f}]"
    print(f"{kernel:<13s} | {result.effect:>8.4f} | {result.se:>7.4f} | {ci_str}")
```

### Expected Results

Results should be very similar across kernels. Large differences suggest:
- Insufficient data near cutoff
- Outliers influencing results
- Need for more careful bandwidth selection

---

## Diagnostic Summary Checklist

### Pre-Analysis
- [ ] McCrary density test
- [ ] Histogram visual inspection
- [ ] Covariate balance tests (all pre-treatment variables)
- [ ] Joint balance test

### Post-Estimation Robustness
- [ ] Bandwidth sensitivity (0.5h to 2h)
- [ ] Polynomial order (linear vs quadratic)
- [ ] Kernel choice (triangular vs uniform)
- [ ] Placebo cutoff tests
- [ ] Donut hole (if manipulation concern)

### Reporting Requirements

1. **Always Report**:
   - McCrary test statistic and p-value
   - Number of observations within bandwidth
   - Bandwidth choice and method

2. **Recommended**:
   - Covariate balance table
   - Bandwidth sensitivity plot/table
   - At least one alternative specification

3. **If Concerns Arise**:
   - Donut hole results
   - Multiple polynomial orders
   - Alternative bandwidth selection methods

---

## References

- McCrary, J. (2008). Manipulation of the running variable in the regression discontinuity design: A density test. *Journal of Econometrics*, 142(2), 698-714.
- Cattaneo, M. D., Jansson, M., & Ma, X. (2020). Simple local polynomial density estimators. *Journal of the American Statistical Association*, 115(531), 1449-1455.
- Gelman, A., & Imbens, G. (2019). Why high-order polynomials should not be used in regression discontinuity designs. *Journal of Business & Economic Statistics*, 37(3), 447-456.
- Lee, D. S., & Lemieux, T. (2010). Regression discontinuity designs in economics. *Journal of Economic Literature*, 48(2), 281-355.
