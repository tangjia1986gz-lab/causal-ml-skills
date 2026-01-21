---
name: estimator-rd
description: Use when estimating causal effects using Regression Discontinuity designs. Triggers on RD, regression discontinuity, sharp RD, fuzzy RD, cutoff, threshold, running variable, bandwidth, discontinuity.
---

# Estimator: Regression Discontinuity (RD)

> **Version**: 1.0.0 | **Type**: Estimator
> **Aliases**: RD, RDD, Regression Discontinuity Design, Sharp RD, Fuzzy RD

## Overview

Regression Discontinuity (RD) estimates causal effects by exploiting threshold-based treatment assignment rules. When treatment is determined by whether a running variable crosses a known cutoff, we can estimate the causal effect by comparing outcomes just above and just below the threshold.

**Key Identification Assumption**: Potential outcomes are continuous functions of the running variable at the cutoff (no other discontinuity exists at the threshold).

## When to Use

### Ideal Scenarios
- Treatment assigned based on a test score cutoff (scholarships, program eligibility)
- Age-based eligibility thresholds (voting age, drinking age, retirement)
- Geographic boundaries (electoral districts, school zones)
- Income thresholds for program eligibility
- Time-based cutoffs (policy implementation dates)

### Data Requirements
- [ ] Running variable (score) that determines treatment assignment
- [ ] Known cutoff value
- [ ] Sufficient observations near the cutoff on both sides
- [ ] Treatment indicator (for fuzzy RD: actual treatment received)
- [ ] Outcome variable measured after treatment assignment

### When NOT to Use
- Agents can precisely manipulate their running variable -> Evidence of sorting violates RD
- Multiple discontinuities at the same cutoff -> Cannot isolate treatment effect
- Running variable is discrete with few values -> Insufficient variation near cutoff
- Cutoff is not binding -> Consider `estimator-iv` instead
- Effects far from cutoff needed -> RD provides only local estimates (LATE)

## Identification Assumptions

| Assumption | Description | Testable? |
|------------|-------------|-----------|
| **Continuity** | Potential outcomes E[Y(0)|X=c] and E[Y(1)|X=c] are continuous at cutoff | No (fundamentally untestable) |
| **No Manipulation** | Agents cannot precisely sort around the cutoff | Yes (McCrary density test) |
| **Local Randomization** | Near the cutoff, treatment is "as good as random" | Partially (covariate balance) |
| **LATE Interpretation** | Effect is local to units at the cutoff | N/A (by design) |

---

## Sharp vs Fuzzy RD

### Sharp RD
Treatment is a **deterministic** function of the running variable:

$$
D_i = \mathbf{1}(X_i \geq c)
$$

Everyone above the cutoff is treated; everyone below is not treated. The estimand is:

$$
\tau_{SRD} = \lim_{x \downarrow c} E[Y|X=x] - \lim_{x \uparrow c} E[Y|X=x]
$$

### Fuzzy RD
Treatment **probability** changes discontinuously at the cutoff, but compliance is imperfect:

$$
\lim_{x \downarrow c} P(D=1|X=x) \neq \lim_{x \uparrow c} P(D=1|X=x)
$$

The estimand is a ratio (similar to IV):

$$
\tau_{FRD} = \frac{\lim_{x \downarrow c} E[Y|X=x] - \lim_{x \uparrow c} E[Y|X=x]}{\lim_{x \downarrow c} E[D|X=x] - \lim_{x \uparrow c} E[D|X=x]}
$$

This is the Local Average Treatment Effect (LATE) for compliers at the cutoff.

---

## Workflow

```
+-------------------------------------------------------------+
|                    RD ESTIMATOR WORKFLOW                      |
+-------------------------------------------------------------+
|  1. SETUP          -> Define running variable, cutoff, Y, D   |
|  2. PRE-ESTIMATION -> McCrary test, covariate balance         |
|  3. ESTIMATION     -> Sharp/Fuzzy RD, bandwidth selection     |
|  4. DIAGNOSTICS    -> Placebo cutoffs, sensitivity, donut     |
|  5. REPORTING      -> RD plot, tables, interpretation         |
+-------------------------------------------------------------+
```

### Phase 1: Setup

**Objective**: Define the RD design and validate data structure

**Inputs Required**:
```python
# Standard RD input structure
running = "score"           # Running variable name
cutoff = 0.0                # Cutoff value
outcome = "y"               # Outcome variable name
treatment = "treated"       # Treatment indicator (for fuzzy RD)
covariates = ["x1", "x2"]   # Optional covariates for balance tests
```

**Data Validation Checklist**:
- [ ] Running variable is continuous (or has many discrete values)
- [ ] Cutoff value is known and well-defined
- [ ] Sufficient observations within bandwidth of cutoff (both sides)
- [ ] No missing values in running variable or outcome
- [ ] For fuzzy RD: treatment indicator available

**Data Structure Verification**:
```python
from rd_estimator import validate_rd_data

validation = validate_rd_data(
    data=df,
    running="score",
    outcome="y",
    cutoff=0.0,
    treatment="treated"  # Optional for sharp RD
)
print(validation.summary())
```

### Phase 2: Pre-Estimation Checks

**McCrary Density Test: Manipulation Check**

```python
from rd_estimator import mccrary_test

# Test for bunching/sorting at the cutoff
density_test = mccrary_test(
    running=df["score"],
    cutoff=0.0,
    bandwidth=None  # Auto-select
)

print(density_test)
```

**Interpretation**:
- PASS if: p-value > 0.05 (no significant density discontinuity)
- WARNING if: 0.01 < p-value < 0.05 (marginal evidence of sorting)
- FAIL if: p-value < 0.01 (strong evidence of manipulation - RD may be invalid)

---

**Covariate Balance at Cutoff**

```python
from rd_estimator import covariate_balance_rd

# Test if covariates are smooth at cutoff
balance = covariate_balance_rd(
    data=df,
    running="score",
    cutoff=0.0,
    covariates=["age", "gender", "income"],
    bandwidth=0.5
)

print(balance.summary_table)
```

**Interpretation**:
- Covariates should NOT show discontinuities at the cutoff
- Significant jumps in covariates suggest confounding or sorting
- This supports the "local randomization" interpretation

---

### Phase 3: Main Estimation

**Local Polynomial Regression**

The standard RD estimator uses local polynomial regression:

$$
\min_{\alpha, \beta} \sum_{i} K\left(\frac{X_i - c}{h}\right) \cdot \left(Y_i - \alpha - \beta(X_i - c) - ... \right)^2
$$

Where:
- $K(\cdot)$: Kernel function (typically triangular)
- $h$: Bandwidth
- Separate regressions above and below cutoff

**Kernel Functions**:
- **Triangular** (default): $K(u) = (1 - |u|) \cdot \mathbf{1}(|u| \leq 1)$
- **Epanechnikov**: $K(u) = \frac{3}{4}(1 - u^2) \cdot \mathbf{1}(|u| \leq 1)$
- **Uniform**: $K(u) = \frac{1}{2} \cdot \mathbf{1}(|u| \leq 1)$

**Polynomial Order**:
- **Linear (p=1)**: Default, most robust
- **Quadratic (p=2)**: More flexible, useful if relationship is curved
- Higher orders risk overfitting

**Bandwidth Selection**:
- **MSE-Optimal**: Minimizes mean squared error of the estimator
- **CER-Optimal**: Coverage Error Rate optimal for confidence intervals

```python
from rd_estimator import (
    estimate_sharp_rd,
    estimate_fuzzy_rd,
    select_bandwidth
)

# Step 1: Select bandwidth
bw = select_bandwidth(
    running=df["score"],
    outcome=df["y"],
    cutoff=0.0,
    method="mserd"  # or "cerrd" for CI-optimal
)
print(f"Selected bandwidth: {bw:.4f}")

# Step 2a: Sharp RD estimation
result_sharp = estimate_sharp_rd(
    data=df,
    running="score",
    outcome="y",
    cutoff=0.0,
    bandwidth=bw,
    kernel="triangular",
    order=1  # Linear
)

print(result_sharp.summary_table)

# Step 2b: Fuzzy RD estimation (if treatment is imperfect)
result_fuzzy = estimate_fuzzy_rd(
    data=df,
    running="score",
    outcome="y",
    treatment="treated",  # Actual treatment received
    cutoff=0.0,
    bandwidth=bw,
    kernel="triangular",
    order=1
)

print(result_fuzzy.summary_table)
```

**Returns**:
```python
CausalOutput(
    effect=0.52,           # Point estimate at cutoff
    se=0.15,               # Robust standard error
    ci_lower=0.23,         # Robust CI lower (bias-corrected)
    ci_upper=0.81,         # Robust CI upper
    p_value=0.0006,
    diagnostics={
        'bandwidth': 0.35,
        'n_left': 245,     # Observations below cutoff
        'n_right': 267,    # Observations above cutoff
        'n_effective': 512,# Effective sample size
        'kernel': 'triangular',
        'order': 1,
        'fuzzy': False
    },
    summary_table="...",
    interpretation="..."
)
```

### Phase 4: Robustness Checks

| Check | Purpose | Implementation |
|-------|---------|----------------|
| Placebo Cutoffs | Verify no effect at fake cutoffs | `placebo_cutoff_test()` |
| Bandwidth Sensitivity | Check stability across bandwidths | `bandwidth_sensitivity()` |
| Donut Hole | Exclude observations right at cutoff | `donut_hole_rd()` |
| Polynomial Order | Compare linear vs quadratic | Re-run with `order=2` |
| Alternative Kernels | Check kernel sensitivity | Re-run with different kernel |

**Placebo Cutoff Test**:

```python
from rd_estimator import placebo_cutoff_test

# Test at fake cutoffs where no effect should exist
placebo_results = placebo_cutoff_test(
    data=df,
    running="score",
    outcome="y",
    true_cutoff=0.0,
    placebo_cutoffs=[-0.5, -0.25, 0.25, 0.5],
    bandwidth=bw
)

for cutoff, result in placebo_results.items():
    status = "PASS" if result.passed else "FAIL"
    print(f"Cutoff {cutoff}: {status} (effect={result.effect:.3f}, p={result.p_value:.3f})")
```

**Bandwidth Sensitivity**:

```python
from rd_estimator import bandwidth_sensitivity

sensitivity = bandwidth_sensitivity(
    data=df,
    running="score",
    outcome="y",
    cutoff=0.0,
    bandwidth_range=[0.5*bw, 0.75*bw, bw, 1.25*bw, 1.5*bw, 2*bw]
)

# Results should be stable across reasonable bandwidth choices
print(sensitivity.summary_table)
```

**Donut Hole RD**:

```python
from rd_estimator import donut_hole_rd

# Exclude observations very close to cutoff (address manipulation concerns)
donut_result = donut_hole_rd(
    data=df,
    running="score",
    outcome="y",
    cutoff=0.0,
    bandwidth=bw,
    donut_radius=0.05  # Exclude |X - c| < 0.05
)

print(f"Donut RD effect: {donut_result.effect:.4f} (SE: {donut_result.se:.4f})")
```

### Phase 5: Reporting

**RD Plot (Essential Visualization)**:

```python
from rd_estimator import rd_plot

fig = rd_plot(
    data=df,
    running="score",
    outcome="y",
    cutoff=0.0,
    bandwidth=bw,
    n_bins=20,         # Number of bins for scatter
    poly_order=1,      # Polynomial order for fitted lines
    ci=True            # Show confidence intervals
)

fig.savefig("rd_plot.png", dpi=150)
```

**Standard Output Table Format**:

```
+----------------------------------------------------------+
|          Table X: Regression Discontinuity Estimates      |
+----------------------------------------------------------+
|                         (1)        (2)        (3)         |
|                      Linear    Quadratic   Half-BW        |
+----------------------------------------------------------+
| RD Effect            0.52***    0.48***    0.55***        |
|                     (0.15)     (0.18)     (0.22)          |
|                                                           |
| Bandwidth            0.35       0.35       0.175          |
| Kernel               Triangular Triangular Triangular     |
| Polynomial Order     1          2          1              |
|                                                           |
| N (left of cutoff)   245        245        118            |
| N (right of cutoff)  267        267        132            |
| Effective N          512        512        250            |
+----------------------------------------------------------+
| McCrary p-value      0.672                                |
| Placebo tests        Pass (4/4)                           |
+----------------------------------------------------------+
| Notes: Robust bias-corrected standard errors.             |
| *** p<0.01, ** p<0.05, * p<0.1                           |
+----------------------------------------------------------+
```

**Interpretation Template**:

```markdown
## Results Interpretation

Using a regression discontinuity design, we estimate that crossing the [threshold description]
leads to a [increase/decrease] of **[effect]** (SE = [se]) in [outcome description].

### Design Validity
1. **Manipulation Test**: McCrary density test shows [no evidence / some evidence]
   of sorting at the cutoff (p = [p-value]).
2. **Covariate Balance**: Covariates [are / are not] smooth at the cutoff,
   supporting the local randomization assumption.
3. **Placebo Tests**: [X/Y] placebo cutoffs show no significant effects,
   as expected under a valid RD design.

### Robustness
- The effect is [stable / sensitive] to bandwidth choice ([range] effect range).
- Results are [similar / different] with quadratic specification.
- Donut hole analysis [confirms / challenges] the main result.

### Interpretation
The RD estimate represents a **Local Average Treatment Effect (LATE)**
for units at the margin of the cutoff. Extrapolation to units far from
the cutoff should be done cautiously.

### Caveats
- [Any concerns about manipulation or sorting]
- [Bandwidth sensitivity issues]
- [External validity limitations]
```

---

## Common Mistakes

### 1. Using Global Polynomial Regression

**Mistake**: Fitting high-order polynomials across the entire support.

**Why it's wrong**: Global polynomials can produce biased estimates and misleading confidence intervals. The polynomial can bend unnaturally to fit observations far from the cutoff, distorting the estimate at the cutoff.

**Correct approach**:
```python
# WRONG: Global polynomial
model = OLS(y ~ running + running^2 + running^3 + treatment)

# CORRECT: Local polynomial within bandwidth
result = estimate_sharp_rd(
    data=df,
    running="score",
    outcome="y",
    cutoff=0.0,
    bandwidth=bw,  # Use only observations near cutoff
    order=1        # Linear is usually sufficient
)
```

### 2. Ignoring Bandwidth Selection

**Mistake**: Choosing bandwidth arbitrarily or using the full sample.

**Why it's wrong**: Bandwidth affects bias-variance tradeoff. Too large leads to bias; too small leads to high variance.

**Correct approach**:
```python
# WRONG: Arbitrary bandwidth
result = estimate_sharp_rd(data, running, outcome, cutoff, bandwidth=1.0)

# CORRECT: Data-driven bandwidth selection
bw = select_bandwidth(
    running=df["score"],
    outcome=df["y"],
    cutoff=0.0,
    method="mserd"  # MSE-optimal
)
result = estimate_sharp_rd(data, running, outcome, cutoff, bandwidth=bw)
```

### 3. Treating Fuzzy RD as Sharp RD

**Mistake**: Ignoring imperfect compliance when some units don't follow the assignment rule.

**Why it's wrong**: If not everyone above the cutoff is treated (or some below are), sharp RD gives an intent-to-treat effect, not the treatment effect.

**Correct approach**:
```python
# Check for imperfect compliance
compliance_above = df[df["score"] >= cutoff]["treated"].mean()
compliance_below = df[df["score"] < cutoff]["treated"].mean()

if compliance_above < 0.99 or compliance_below > 0.01:
    print("WARNING: Imperfect compliance detected. Use Fuzzy RD.")
    result = estimate_fuzzy_rd(
        data=df,
        running="score",
        outcome="y",
        treatment="treated",  # Actual treatment
        cutoff=0.0,
        bandwidth=bw
    )
```

### 4. Ignoring the McCrary Test

**Mistake**: Not testing for manipulation before estimating.

**Why it's wrong**: If agents sort around the cutoff, the continuity assumption is violated and RD estimates are biased.

**Correct approach**:
```python
# ALWAYS run McCrary test first
density_result = mccrary_test(df["score"], cutoff=0.0)

if not density_result.passed:
    print("WARNING: Evidence of manipulation at cutoff!")
    print("Consider:")
    print("1. Donut hole RD (exclude observations near cutoff)")
    print("2. Alternative identification strategies")
    print("3. Reporting intent-to-treat with caveats")
```

---

## Examples

### Example 1: Test Score Cutoff for Scholarship

**Research Question**: What is the effect of receiving a merit scholarship on college GPA?

**Setting**: Students with SAT scores >= 1200 receive a scholarship.

```python
import pandas as pd
from rd_estimator import (
    mccrary_test,
    select_bandwidth,
    estimate_sharp_rd,
    rd_plot,
    run_full_rd_analysis
)

# Load data
data = pd.read_csv("scholarship_data.csv")
# Columns: student_id, sat_score, scholarship, college_gpa, high_school_gpa

# Define RD setup
cutoff = 1200
running_var = "sat_score"
outcome_var = "college_gpa"

# Quick check: is this sharp or fuzzy?
above = data[data[running_var] >= cutoff]["scholarship"].mean()
below = data[data[running_var] < cutoff]["scholarship"].mean()
print(f"Scholarship rate above cutoff: {above:.2%}")
print(f"Scholarship rate below cutoff: {below:.2%}")
# If above ~100% and below ~0%, use sharp RD

# Run full analysis
result = run_full_rd_analysis(
    data=data,
    running=running_var,
    outcome=outcome_var,
    cutoff=cutoff,
    treatment="scholarship"  # For fuzzy RD, or None for sharp
)

# View results
print(result.summary_table)

# Create RD plot
fig = rd_plot(
    data=data,
    running=running_var,
    outcome=outcome_var,
    cutoff=cutoff,
    bandwidth=result.diagnostics['bandwidth']
)
fig.savefig("scholarship_rd_plot.png")
```

**Output**:
```
============================================================
         REGRESSION DISCONTINUITY ANALYSIS RESULTS
============================================================

RD Effect (LATE): 0.35
Standard Error: 0.12
95% CI: [0.12, 0.58]
P-value: 0.003

------------------------------------------------------------
DIAGNOSTICS
------------------------------------------------------------

McCrary Density Test: PASSED
  - Test statistic: 0.42
  - P-value: 0.67

Covariate Balance:
  - high_school_gpa: Balanced (p=0.45)
  - age: Balanced (p=0.72)

------------------------------------------------------------
BANDWIDTH AND SAMPLE
------------------------------------------------------------
Optimal Bandwidth: 85 SAT points
N (below cutoff): 312
N (above cutoff): 298
Effective N: 610

============================================================
```

**Interpretation**:
Students who just qualify for the merit scholarship (SAT >= 1200) have college GPAs approximately 0.35 points higher than students who just miss the cutoff. This effect is statistically significant at the 1% level. The McCrary test shows no evidence of manipulation, and pre-treatment covariates are balanced, supporting the validity of the RD design.

### Example 2: Electoral RD (Lee, 2008 Style)

**Research Question**: Does winning a close election affect party vote share in the next election?

```python
import pandas as pd
from rd_estimator import run_full_rd_analysis, bandwidth_sensitivity

# Load election data
data = pd.read_csv("house_elections.csv")
# Columns: district, year, dem_vote_share_t, dem_vote_share_t1, incumbent_party

# Running variable: Democratic vote margin (centered at 50%)
data["dem_margin"] = data["dem_vote_share_t"] - 0.5

# Outcome: Vote share in next election
result = run_full_rd_analysis(
    data=data,
    running="dem_margin",
    outcome="dem_vote_share_t1",
    cutoff=0.0,
    treatment=None  # Sharp RD: winning is deterministic
)

print(result.summary_table)

# Bandwidth sensitivity analysis
sens = bandwidth_sensitivity(
    data=data,
    running="dem_margin",
    outcome="dem_vote_share_t1",
    cutoff=0.0,
    bandwidth_range=[0.025, 0.05, 0.075, 0.10, 0.15, 0.20]
)

print("\nBandwidth Sensitivity:")
print(sens.summary_table)
```

**Output**:
```
RD Effect: 0.078*** (0.021)
95% CI: [0.037, 0.119]

This represents a 7.8 percentage point incumbency advantage.
```

---

## References

### Seminal Papers
- Thistlethwaite, D. L., & Campbell, D. T. (1960). Regression-discontinuity analysis: An alternative to the ex post facto experiment. *Journal of Educational Psychology*, 51(6), 309-317.
- Lee, D. S. (2008). Randomized experiments from non-random selection in US House elections. *Journal of Econometrics*, 142(2), 675-697.
- Imbens, G. W., & Lemieux, T. (2008). Regression discontinuity designs: A guide to practice. *Journal of Econometrics*, 142(2), 615-635.

### Methodological Advances
- Calonico, S., Cattaneo, M. D., & Titiunik, R. (2014). Robust nonparametric confidence intervals for regression-discontinuity designs. *Econometrica*, 82(6), 2295-2326.
- Cattaneo, M. D., Idrobo, N., & Titiunik, R. (2020). *A Practical Introduction to Regression Discontinuity Designs*. Cambridge University Press.
- McCrary, J. (2008). Manipulation of the running variable in the regression discontinuity design: A density test. *Journal of Econometrics*, 142(2), 698-714.

### Textbook Treatments
- Angrist, J. D., & Pischke, J. S. (2009). *Mostly Harmless Econometrics*. Princeton University Press. Chapter 6.
- Cunningham, S. (2021). *Causal Inference: The Mixtape*. Yale University Press. Chapter 6.
- Huntington-Klein, N. (2022). *The Effect*. CRC Press. Chapter 20.

### Software Documentation
- `rdrobust` (R/Stata): https://rdpackages.github.io/rdrobust/
- `rddensity` (R/Stata): https://rdpackages.github.io/rddensity/
- `rdmulti` (R/Stata): https://rdpackages.github.io/rdmulti/

---

## Related Estimators

| Estimator | When to Use Instead |
|-----------|---------------------|
| `estimator-did` | Treatment varies over time, not by threshold |
| `estimator-iv` | Fuzzy RD with very weak first stage |
| `estimator-rdid` | Combining RD with DID (repeated cross-sections at cutoff) |
| `estimator-bunching` | Running variable is continuous but bunching analysis preferred |
| `estimator-geographic-rd` | Cutoff is spatial (borders, boundaries) |

---

## Appendix: Mathematical Details

### Local Linear Regression Estimator

The sharp RD estimator using local linear regression minimizes:

$$
\min_{\alpha_-, \beta_-} \sum_{i: X_i < c} K_h(X_i - c) \cdot (Y_i - \alpha_- - \beta_-(X_i - c))^2
$$

$$
\min_{\alpha_+, \beta_+} \sum_{i: X_i \geq c} K_h(X_i - c) \cdot (Y_i - \alpha_+ - \beta_+(X_i - c))^2
$$

The RD estimate is: $\hat{\tau} = \hat{\alpha}_+ - \hat{\alpha}_-$

### Bandwidth Selection

**MSE-Optimal Bandwidth** (Imbens-Kalyanaraman / Calonico-Cattaneo-Titiunik):

$$
h_{MSE} = C_n \cdot \left( \frac{\sigma^2(c)}{\mu_2^{(2)}(c)^2 \cdot n} \right)^{1/5}
$$

Where:
- $\sigma^2(c)$: Variance of outcome at cutoff
- $\mu_2^{(2)}(c)$: Second derivative of conditional mean at cutoff
- $C_n$: Constant depending on kernel

**CER-Optimal Bandwidth** scales MSE-optimal bandwidth for coverage:

$$
h_{CER} = h_{MSE} \cdot n^{-1/20}
$$

### Fuzzy RD as Local Wald Estimator

The fuzzy RD estimand can be written as:

$$
\tau_{FRD} = \frac{E[Y|X=c^+] - E[Y|X=c^-]}{E[D|X=c^+] - E[D|X=c^-]}
$$

This is estimated using local polynomial regression for both numerator (reduced form) and denominator (first stage), then taking the ratio.

### Bias-Corrected Robust Inference

Following Calonico, Cattaneo, and Titiunik (2014), robust confidence intervals use:

$$
CI = \left[ \hat{\tau} - q_{1-\alpha/2} \cdot \hat{V}^{1/2}, \hat{\tau} + q_{1-\alpha/2} \cdot \hat{V}^{1/2} \right]
$$

Where $\hat{V}$ accounts for bias from using a smaller bandwidth than optimal.
