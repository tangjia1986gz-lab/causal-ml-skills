---
name: estimator-rd
description: Regression Discontinuity (RD) design estimation and inference. Use when treatment assignment depends on whether a running variable crosses a cutoff threshold. Best for sharp/fuzzy RD, bandwidth selection, manipulation testing, and local polynomial estimation. For panel data designs see panel-data-models; for IV approaches see estimator-iv.
license: MIT
metadata:
    skill-author: Causal-ML-Skills
    version: "2.0.0"
    category: econometrics
---

# RD: Causal Inference at Discontinuities

## Overview

Regression Discontinuity (RD) designs exploit discontinuous changes in treatment assignment at a threshold of a running variable to identify causal effects. This skill provides comprehensive guidance for implementing publication-quality RD analyses, including sharp and fuzzy designs, optimal bandwidth selection, manipulation testing, and robust inference using local polynomial methods.

## When to Use This Skill

This skill should be used when:
- Treatment assignment depends on whether a running variable crosses a cutoff
- You have a sharp discontinuity (deterministic assignment at threshold)
- You have a fuzzy discontinuity (probability of treatment jumps at threshold)
- You need to select optimal bandwidth for local estimation
- You want to test for manipulation of the running variable (McCrary test)
- You need covariate balance checks around the cutoff
- You're estimating local average treatment effects (LATE) at the threshold
- You want to perform sensitivity analysis across bandwidths
- You need publication-ready RD plots and tables

## Quick Start Guide

### Sharp RD with rdrobust

```python
import numpy as np
import pandas as pd
from rdrobust import rdrobust, rdbwselect, rdplot

# Generate simulated RD data
np.random.seed(42)
n = 1000
x = np.random.uniform(-1, 1, n)  # Running variable
cutoff = 0
treatment = (x >= cutoff).astype(int)
y = 0.5 + 2.0 * treatment + 0.8 * x + np.random.normal(0, 0.5, n)

# Optimal bandwidth selection (Calonico, Cattaneo, Titiunik 2014)
bw = rdbwselect(y, x, c=cutoff)
print(f"Optimal bandwidth (MSE): {bw.bws[0]:.4f}")
print(f"Optimal bandwidth (CER): {bw.bws[1]:.4f}")

# Sharp RD estimation with robust bias-corrected inference
rd_result = rdrobust(y, x, c=cutoff)
print(rd_result)

# Key outputs:
# - Conventional estimate: point estimate using local polynomial
# - Bias-corrected estimate: corrects for boundary bias
# - Robust CI: valid confidence intervals (recommended)

# RD plot with automatic binning
rdplot(y, x, c=cutoff, title="RD Plot: Effect at Cutoff")
```

### Fuzzy RD Design

```python
from rdrobust import rdrobust

# Fuzzy RD: treatment is not deterministic at cutoff
# Example: probability of treatment jumps from 0.3 to 0.8 at cutoff

np.random.seed(42)
n = 1000
x = np.random.uniform(-1, 1, n)
cutoff = 0

# Fuzzy assignment: treatment probability jumps at cutoff
prob_treat = np.where(x >= cutoff, 0.8, 0.3)
treatment = np.random.binomial(1, prob_treat)

# True effect = 2.0
y = 0.5 + 2.0 * treatment + 0.5 * x + np.random.normal(0, 0.5, n)

# Fuzzy RD estimation
# Estimates LATE: effect for compliers at the cutoff
fuzzy_result = rdrobust(y, x, c=cutoff, fuzzy=treatment)
print(fuzzy_result)

# The fuzzy RD estimate scales by the jump in treatment probability
# Effect / (P(T|X>=c) - P(T|X<c)) = LATE
```

### McCrary Manipulation Test

```python
from rddensity import rddensity

# Test for manipulation of running variable at cutoff
# H0: density is continuous at cutoff
# H1: density has a discontinuity (manipulation)

np.random.seed(42)
n = 1000
x_clean = np.random.normal(0, 1, n)  # No manipulation

# McCrary test
manip_test = rddensity(x_clean, c=0)
print(f"Test statistic: {manip_test.T:.4f}")
print(f"P-value: {manip_test.pval:.4f}")

# Interpretation:
# - p-value < 0.05: Evidence of manipulation, RD may be invalid
# - p-value >= 0.05: No evidence of manipulation

# Plot density around cutoff
from rddensity import rdplotdensity
rdplotdensity(rddensity(x_clean, c=0), x_clean)
```

### Covariate Balance Check

```python
from rdrobust import rdrobust
import pandas as pd

# Check if pre-determined covariates are smooth at the cutoff
# Discontinuity in covariates suggests manipulation or confounding

np.random.seed(42)
n = 1000
x = np.random.uniform(-1, 1, n)
cutoff = 0

# Pre-determined covariates (should not jump at cutoff)
age = 30 + 5 * np.random.randn(n) + 0.5 * x
education = 12 + 2 * np.random.randn(n) + 0.3 * x
income_pre = 50000 + 10000 * np.random.randn(n) + 5000 * x

covariates = {'age': age, 'education': education, 'income_pre': income_pre}

print("Covariate Balance Tests at Cutoff")
print("=" * 50)
for name, cov in covariates.items():
    result = rdrobust(cov, x, c=cutoff)
    coef = result.coef[0]
    ci_lower = result.ci[0, 0]
    ci_upper = result.ci[0, 1]
    pval = result.pv[0]

    sig = "***" if pval < 0.01 else ("**" if pval < 0.05 else ("*" if pval < 0.1 else ""))
    print(f"{name:15s}: {coef:8.3f} [{ci_lower:8.3f}, {ci_upper:8.3f}] p={pval:.3f}{sig}")

# All p-values should be > 0.05 for valid RD design
```

### Bandwidth Sensitivity Analysis

```python
from rdrobust import rdrobust
import numpy as np

# Test robustness to bandwidth choice
np.random.seed(42)
n = 1000
x = np.random.uniform(-1, 1, n)
treatment = (x >= 0).astype(int)
y = 0.5 + 2.0 * treatment + 0.8 * x + np.random.normal(0, 0.5, n)

# Get optimal bandwidth first
base_result = rdrobust(y, x, c=0)
h_opt = base_result.bws[0, 0]  # Optimal bandwidth

# Test at various bandwidths
bandwidths = [0.5 * h_opt, 0.75 * h_opt, h_opt, 1.25 * h_opt, 1.5 * h_opt, 2 * h_opt]

print("Bandwidth Sensitivity Analysis")
print("=" * 60)
print(f"{'Bandwidth':>12s} {'Estimate':>10s} {'SE':>10s} {'95% CI':>20s}")
print("-" * 60)

for h in bandwidths:
    result = rdrobust(y, x, c=0, h=h)
    coef = result.coef[0]
    se = result.se[0]
    ci_lower = result.ci[0, 0]
    ci_upper = result.ci[0, 1]

    label = f"{h/h_opt:.2f}x" if h != h_opt else "optimal"
    print(f"{h:12.4f} {coef:10.4f} {se:10.4f} [{ci_lower:8.4f}, {ci_upper:8.4f}] ({label})")

# Estimates should be stable across bandwidths for robust findings
```

## Core Capabilities

### 1. Sharp RD Design

Treatment is deterministically assigned based on running variable crossing cutoff.

**Identification:**
- E[Y(1) - Y(0) | X = c] identified if potential outcomes are continuous at c
- Assumes no manipulation of running variable
- Local identification: only identifies effect at the cutoff

**Estimation methods:**
- Local linear regression (recommended)
- Local polynomial regression (order p = 1 or 2)
- Bias-corrected robust inference (Calonico et al. 2014)

**When to use:** Treatment assignment is perfectly determined by crossing threshold

**Reference:** See `references/estimation_methods.md` for detailed algorithms.

### 2. Fuzzy RD Design

Treatment probability (not assignment) jumps at the cutoff.

**Identification:**
- Estimates LATE: effect for compliers at the cutoff
- Requires monotonicity: crossing cutoff only increases treatment probability
- Uses cutoff as instrument for treatment

**Estimation:**
- Two-stage: First stage regresses treatment on running variable
- Second stage uses predicted treatment
- Analogous to IV/2SLS at the cutoff

**When to use:** Treatment probability changes discontinuously but not from 0 to 1

**Reference:** See `references/identification_assumptions.md` for LATE interpretation.

### 3. Bandwidth Selection

Choosing the right bandwidth is crucial for bias-variance tradeoff.

**Methods:**
- **IK (Imbens-Kalyanaraman 2012):** MSE-optimal for point estimation
- **CCT (Calonico-Cattaneo-Titiunik 2014):** Coverage error rate optimal for inference
- **Cross-validation:** Data-driven selection

**Considerations:**
- Smaller bandwidth: less bias, more variance
- Larger bandwidth: more bias, less variance
- Different bandwidths for estimation vs inference

**Reference:** See `references/diagnostic_tests.md` for bandwidth selection guidance.

### 4. Manipulation Testing

Test whether units can precisely manipulate their running variable to receive treatment.

**McCrary (2008) Test:**
- Tests for discontinuity in density of running variable at cutoff
- Null: density is continuous (no manipulation)
- Significant result suggests RD may be invalid

**Alternative approaches:**
- Visual inspection of histogram
- Cattaneo-Jansson-Ma (2020) test (implemented in rddensity)

**Reference:** See `references/diagnostic_tests.md` for test procedures.

### 5. Local Polynomial Regression

Core estimation approach for RD designs.

**Polynomial orders:**
- **p = 0:** Local constant (Nadaraya-Watson)
- **p = 1:** Local linear (recommended, boundary bias correction)
- **p = 2:** Local quadratic (more flexible, higher variance)

**Kernel functions:**
- Triangular: (1 - |u|) for |u| ≤ 1 (recommended)
- Epanechnikov: (3/4)(1 - u²) for |u| ≤ 1
- Uniform: 1/2 for |u| ≤ 1

**Reference:** See `references/estimation_methods.md` for technical details.

## Best Practices

### Data Preparation

1. **Center running variable:** X_centered = X - cutoff
2. **Check for heaping:** Discrete running variables need special care
3. **Handle exactly-at-cutoff observations:** Document treatment rule
4. **Verify assignment mechanism:** Confirm how treatment is assigned

### Estimation

1. **Use local linear regression:** Preferred over global polynomials (Gelman & Imbens 2018)
2. **Report multiple bandwidths:** Show sensitivity to bandwidth choice
3. **Use robust bias-corrected inference:** rdrobust default
4. **Avoid high-order global polynomials:** Can fit noise, misleading extrapolation

### Inference

1. **Use robust confidence intervals:** Correct for bias in local polynomial
2. **Report both conventional and robust estimates:** For transparency
3. **Cluster standard errors if needed:** When observations within clusters are correlated
4. **Consider triangular kernel:** Gives more weight to observations near cutoff

### Reporting

1. **Always show RD plot:** Visual evidence is essential
2. **Report manipulation test:** McCrary or density test
3. **Show covariate balance:** Pre-determined covariates should be smooth
4. **Bandwidth sensitivity:** Multiple bandwidths in robustness section
5. **Report sample sizes:** Both sides of cutoff and within bandwidth

## Common Workflows

### Workflow 1: Standard Sharp RD Analysis

1. Visualize data: scatter plot of outcome vs running variable
2. Check for manipulation: McCrary density test
3. Check covariate balance: RD estimates for pre-determined variables
4. Select bandwidth: Use optimal selector (CCT recommended)
5. Estimate main effect: Local linear with robust inference
6. Robustness: Vary bandwidth, kernel, polynomial order
7. Create RD plot: Binscatter with fitted lines
8. Report: Table with estimates, diagnostics, visualization

### Workflow 2: Fuzzy RD with Validation

1. Document assignment mechanism: Why is it fuzzy?
2. First stage: Show treatment probability jumps at cutoff
3. Check instrument relevance: Large enough first-stage jump
4. Manipulation and balance tests: Same as sharp RD
5. Estimate fuzzy RD: Using fuzzy option in rdrobust
6. Interpret as LATE: Effect for compliers only
7. Report compliance rate: What fraction complies?

## Reference Documentation

This skill includes detailed reference files:

### references/identification_assumptions.md
- Continuity assumption formal definition
- LATE interpretation for RD
- Sharp vs Fuzzy RD identification
- Testable implications

### references/estimation_methods.md
- Local polynomial regression theory
- Bandwidth selection algorithms (IK, CCT)
- Variance estimation
- Bias correction methods

### references/diagnostic_tests.md
- McCrary manipulation test
- Covariate balance tests
- Placebo cutoffs
- Donut hole RD
- Bandwidth sensitivity

### references/reporting_standards.md
- AER/QJE table formats
- Required elements for RD papers
- LaTeX templates

### references/common_errors.md
- Global polynomial pitfalls
- Bandwidth selection mistakes
- Interpretation errors
- Visualization problems

## Common Pitfalls to Avoid

1. **Using global high-order polynomials:** Leads to overfitting, misleading extrapolation (Gelman & Imbens 2018)
2. **Ignoring manipulation test:** Always test for density discontinuity
3. **Not checking covariate balance:** Discontinuities in pre-determined variables invalidate RD
4. **Single bandwidth:** Must show robustness to bandwidth choice
5. **Wrong cutoff:** Verify the exact threshold where treatment changes
6. **Confusing sharp and fuzzy:** Different identification, different interpretation
7. **No RD plot:** Visual evidence is essential and expected
8. **Extrapolating beyond cutoff:** RD only identifies effect at the threshold
9. **Ignoring heaping:** Discrete running variables need care
10. **Symmetric bandwidth assumption:** May need different bandwidths on each side
11. **Not reporting sample sizes:** Readers need to assess precision
12. **Forgetting bias correction:** Use robust inference from rdrobust
13. **Including cutoff observations inconsistently:** Document handling rule
14. **Treating LATE as ATE:** Fuzzy RD only identifies complier effect
15. **Not considering placebo tests:** Test at fake cutoffs where nothing should happen

## Troubleshooting

### Problem: rdrobust not installed
```bash
pip install rdrobust
# or
conda install -c conda-forge rdrobust
```

### Problem: Manipulation test is significant
- Investigate source of bunching at cutoff
- Consider if bunching affects treatment assignment
- Report as limitation or abandon RD approach
- Try donut hole RD (exclude observations very close to cutoff)

### Problem: Estimates very sensitive to bandwidth
- May indicate RD assumption violations
- Check for outliers near cutoff
- Consider if running variable has measurement error
- Report sensitivity as limitation

### Problem: Very few observations near cutoff
- Consider larger bandwidth (more bias)
- Fuzzy RD may lack power
- Report as limitation
- Consider alternative identification strategies

### Problem: Covariates show discontinuity
- Suggests manipulation or confounding
- RD assumption likely violated
- Investigate mechanism
- May need to control for covariates (though controversial)

## Getting Help

For detailed documentation and examples:
- rdrobust: https://rdpackages.github.io/rdrobust/
- rddensity: https://rdpackages.github.io/rddensity/
- Cattaneo, Idrobo, Titiunik (2020): "A Practical Introduction to RD Designs"
- Lee & Lemieux (2010): "Regression Discontinuity Designs in Economics" (JEL)
- Imbens & Lemieux (2008): "Regression Discontinuity Designs: A Guide to Practice" (JoE)

## Installation

```bash
# Core RD packages
pip install rdrobust rddensity rdlocrand

# Supporting packages
pip install numpy pandas matplotlib statsmodels scipy
```

## Related Skills

- **estimator-iv**: For instrumental variables (Fuzzy RD is special case)
- **estimator-did**: For difference-in-differences designs
- **panel-data-models**: For panel data with fixed effects
- **causal-ddml**: For heterogeneous treatment effects with ML
