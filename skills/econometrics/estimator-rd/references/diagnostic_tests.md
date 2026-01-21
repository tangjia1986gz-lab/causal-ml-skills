# RD Diagnostic Tests

## 1. Manipulation Testing

### McCrary (2008) Density Test

**Purpose:** Test if units can precisely manipulate their running variable to receive treatment.

**Null Hypothesis:** The density of the running variable is continuous at the cutoff.

$$H_0: \lim_{x \downarrow c} f_X(x) = \lim_{x \uparrow c} f_X(x)$$

**Test Procedure:**
1. Estimate density on each side of cutoff using local polynomial
2. Compute discontinuity in log density: $\hat{\theta} = \log \hat{f}^+ - \log \hat{f}^-$
3. Compute test statistic: $T = \hat{\theta} / SE(\hat{\theta})$
4. Compare to standard normal distribution

**Interpretation:**
| p-value | Interpretation | Action |
|---------|----------------|--------|
| p > 0.10 | No evidence of manipulation | Proceed with RD |
| 0.05 < p < 0.10 | Weak evidence | Report as sensitivity |
| p < 0.05 | Evidence of manipulation | Question RD validity |
| p < 0.01 | Strong evidence | Likely invalid RD |

### Cattaneo-Jansson-Ma (2020) Test

**Improvements over McCrary:**
- Robust to bandwidth choice
- Better finite sample performance
- Implemented in `rddensity` package

```python
from rddensity import rddensity, rdplotdensity

# Run test
result = rddensity(X, c=cutoff)

# Key outputs
result.T      # Test statistic
result.pval   # P-value
result.h      # Bandwidths used

# Visualization
rdplotdensity(result, X)
```

---

## 2. Covariate Balance Tests

### Purpose

Pre-determined covariates should be continuous at the cutoff. Discontinuities suggest:
- Manipulation of running variable
- Confounding factors
- Violation of RD assumptions

### Test Procedure

Run RD estimation using each covariate as the "outcome":

$$\hat{\tau}_{W} = \lim_{x \downarrow c} E[W | X = x] - \lim_{x \uparrow c} E[W | X = x]$$

**Interpretation:**
- All coefficients should be statistically insignificant (p > 0.05)
- Point estimates should be small relative to covariate SD

### Which Covariates to Test

| Type | Include? | Rationale |
|------|----------|-----------|
| Pre-determined (before X known) | Yes | Should be balanced |
| Simultaneously determined with X | Maybe | Could reveal manipulation |
| Post-treatment outcomes | No | Expected to show discontinuity |
| Geographic/institutional | Yes | Important controls |

### Reporting Format

| Covariate | Estimate | SE | p-value | Significant? |
|-----------|----------|-------|---------|--------------|
| Age | 0.15 | 0.42 | 0.72 | No |
| Education | -0.08 | 0.18 | 0.66 | No |
| Income (pre) | 234 | 892 | 0.79 | No |
| Gender | 0.02 | 0.03 | 0.48 | No |

---

## 3. Placebo Cutoff Tests

### Purpose

Estimate RD effects at fake cutoffs where treatment does not change.

### Procedure

1. Choose placebo cutoffs $c' \neq c$ (e.g., median of X below or above c)
2. Restrict sample to one side of true cutoff
3. Estimate RD at placebo cutoff
4. Effects should be zero (insignificant)

### Interpretation

| Placebo Cutoff | Estimate | p-value | Interpretation |
|----------------|----------|---------|----------------|
| Below true c | ~0, p > 0.10 | Good: no confounding trend |
| Above true c | ~0, p > 0.10 | Good: no confounding trend |
| Either | Significant | Bad: underlying trend, not RD effect |

### Implementation

```python
# Placebo test below cutoff
subset_below = df[df['x'] < cutoff]
placebo_c_below = subset_below['x'].median()
placebo_result = rdrobust(subset_below['y'], subset_below['x'], c=placebo_c_below)

# Placebo test above cutoff
subset_above = df[df['x'] >= cutoff]
placebo_c_above = subset_above['x'].median()
placebo_result = rdrobust(subset_above['y'], subset_above['x'], c=placebo_c_above)
```

---

## 4. Bandwidth Sensitivity

### Purpose

Verify that results are not artifacts of bandwidth choice.

### Standard Protocol

Test at multiple bandwidths around the optimal:

| Multiplier | Description | Expected Result |
|------------|-------------|-----------------|
| 0.5 × h_opt | Half bandwidth | Consistent direction, wider CI |
| 0.75 × h_opt | 75% bandwidth | Similar estimate |
| 1.0 × h_opt | Optimal | Main specification |
| 1.25 × h_opt | 125% bandwidth | Similar estimate |
| 1.5 × h_opt | 150% bandwidth | Consistent direction |
| 2.0 × h_opt | Double bandwidth | May show more bias |

### Interpretation

**Good:** Estimates stable across bandwidths, all significant or all insignificant

**Concerning:** Estimates flip sign or significance varies dramatically

### Visualization

Create coefficient plot showing estimate and CI at each bandwidth.

---

## 5. Donut Hole RD

### Purpose

Test sensitivity to observations very close to the cutoff (most likely to be manipulated).

### Procedure

1. Define donut: exclude observations within $\pm \epsilon$ of cutoff
2. Re-estimate RD on remaining sample
3. Compare to main estimate

### Common Donut Sizes

| Epsilon | Description |
|---------|-------------|
| 1% of SD(X) | Small donut |
| 2.5% of SD(X) | Medium donut |
| 5% of SD(X) | Large donut |

### Interpretation

| Donut Result | Interpretation |
|--------------|----------------|
| Similar to main | Robust, manipulation unlikely |
| Very different | Observations near cutoff are different |
| Larger effect | May have attenuation from manipulation |

---

## 6. Polynomial Order Sensitivity

### Purpose

Verify results are robust to functional form assumptions.

### Standard Tests

| Order | Model | Use Case |
|-------|-------|----------|
| p = 1 | Local linear | Default (main spec) |
| p = 2 | Local quadratic | Robustness check |
| p = 3 | Local cubic | Rarely needed |

**Warning:** Avoid global high-order polynomials (Gelman & Imbens 2018)

---

## 7. Kernel Sensitivity

### Purpose

Verify results are not driven by kernel choice.

### Standard Tests

| Kernel | Properties |
|--------|------------|
| Triangular | Default, boundary-optimal |
| Epanechnikov | Smoother weights |
| Uniform | All equal weight within bandwidth |

Estimates should be qualitatively similar across kernels.

---

## 8. Summary Diagnostic Table

For publication, present diagnostics in a single table:

| Diagnostic | Test | Result | Implication |
|------------|------|--------|-------------|
| Manipulation | McCrary/rddensity | p = 0.45 | No evidence |
| Covariate balance | RD on covariates | 0/5 significant | Balanced |
| Placebo cutoff (below) | RD at median | p = 0.67 | No effect |
| Placebo cutoff (above) | RD at median | p = 0.82 | No effect |
| Bandwidth sensitivity | 0.5h to 2h | Stable | Robust |
| Donut hole (2.5%) | Exclude ±ε | Similar | Robust |

---

## Key References

- McCrary (2008): "Manipulation of the Running Variable in the Regression Discontinuity Design" - density test
- Cattaneo, Jansson, Ma (2020): "Simple Local Polynomial Density Estimators" - improved density test
- Lee & Lemieux (2010): "Regression Discontinuity Designs in Economics" - comprehensive diagnostics
- Imbens & Lemieux (2008): "Regression Discontinuity Designs: A Guide to Practice" - practical guidance
