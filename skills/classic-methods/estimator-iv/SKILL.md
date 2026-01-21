---
name: estimator-iv
description: Use when estimating causal effects with Instrumental Variables. Triggers on IV, instrumental variable, 2SLS, two-stage least squares, LIML, endogeneity, weak instrument.
---

# Estimator: Instrumental Variables (IV)

> **Version**: 1.0.0 | **Type**: Estimator
> **Aliases**: IV, 2SLS, Two-Stage Least Squares, LIML, GMM-IV

## Overview

Instrumental Variables (IV) estimation identifies causal effects when the treatment variable is endogenous (correlated with the error term). IV uses an external source of variation - an instrument - that affects the outcome only through its effect on the treatment.

**Key Identification Assumption**: The instrument must be (1) relevant (correlated with treatment), (2) exogenous (uncorrelated with unobserved confounders), and (3) satisfy the exclusion restriction (affects outcome only through treatment).

## When to Use

### Ideal Scenarios
- Treatment is endogenous due to omitted variable bias, simultaneity, or measurement error
- A valid instrument is available that creates exogenous variation in treatment
- Randomized experiment is infeasible but "natural" quasi-experimental variation exists
- Estimating Local Average Treatment Effect (LATE) for compliers

### Data Requirements
- [ ] Outcome variable (Y)
- [ ] Endogenous treatment variable (D)
- [ ] One or more instruments (Z) that predict treatment
- [ ] (Optional) Control variables (X) that satisfy exogeneity
- [ ] Sufficient sample size (IV requires larger samples than OLS for precision)

### When NOT to Use
- Weak instruments (first-stage F < 10) -> Consider `estimator-liml` or find stronger instruments
- Exclusion restriction clearly violated -> No valid IV estimation possible
- Heterogeneous treatment effects with policy-relevant ATE needed -> LATE may not generalize
- Instrument affects outcome through multiple channels -> Consider `estimator-control-function`
- Better identification available -> Consider `estimator-did`, `estimator-rd` if applicable

## Identification Assumptions

| Assumption | Description | Testable? |
|------------|-------------|-----------|
| **Relevance** | Instrument Z is correlated with treatment D | Yes (First-stage F-test) |
| **Independence (Exogeneity)** | Instrument Z is uncorrelated with error term | No (requires domain knowledge) |
| **Exclusion Restriction** | Z affects Y only through D | No (requires domain knowledge) |
| **Monotonicity** (for LATE) | Z affects D in same direction for all units | No (requires domain knowledge) |

---

## Workflow

```
+-------------------------------------------------------------+
|                    IV ESTIMATOR WORKFLOW                      |
+-------------------------------------------------------------+
|  1. SETUP          -> Define outcome, treatment, instruments  |
|  2. PRE-ESTIMATION -> First-stage test, weak IV diagnostics   |
|  3. ESTIMATION     -> 2SLS, LIML, GMM                         |
|  4. DIAGNOSTICS    -> Overidentification, endogeneity tests   |
|  5. REPORTING      -> First-stage & second-stage tables       |
+-------------------------------------------------------------+
```

### Phase 1: Setup

**Objective**: Prepare data and define IV model specification

**Inputs Required**:
```python
# Standard CausalInput structure for IV
outcome = "y"                    # Outcome variable name
treatment = "d"                  # Endogenous treatment variable
instruments = ["z1", "z2"]       # Instrument(s)
controls = ["x1", "x2"]          # Exogenous control variables
```

**Data Validation Checklist**:
- [ ] Instruments are not constant (have variation)
- [ ] No missing values in key variables (or explicit handling strategy)
- [ ] Treatment has sufficient variation
- [ ] Sample size adequate (IV needs more observations than OLS)
- [ ] Instruments conceptually satisfy exclusion restriction

**Data Structure Verification**:
```python
from iv_estimator import validate_iv_data

# Check data structure
validation = validate_iv_data(
    data=df,
    outcome="y",
    treatment="d",
    instruments=["z1", "z2"],
    controls=["x1", "x2"]
)
print(validation)
```

### Phase 2: Pre-Estimation Checks

**First-Stage Test: Instrument Relevance**

The first stage regresses the endogenous treatment on instruments and controls:

$$
D_i = \pi_0 + \pi_1 Z_i + X_i'\gamma + v_i
$$

```python
from iv_estimator import first_stage_test, weak_iv_diagnostics

# Run first-stage regression
first_stage = first_stage_test(
    data=df,
    treatment="d",
    instruments=["z1", "z2"],
    controls=["x1", "x2"]
)

print(f"First-stage F-statistic: {first_stage['f_statistic']:.2f}")
print(f"Partial R-squared: {first_stage['partial_r2']:.4f}")
print(f"Instrument coefficients: {first_stage['coefficients']}")

# Check against Stock-Yogo critical values
weak_iv = weak_iv_diagnostics(
    first_stage_f=first_stage['f_statistic'],
    n_instruments=len(instruments)
)
print(weak_iv)
```

**Interpretation**:
- PASS if: First-stage F > 10 (Stock-Yogo rule of thumb)
- WARNING if: F between 5-10 (weak instruments, use LIML)
- FAIL if: F < 5 (very weak instruments, IV unreliable)

**Critical Values (Stock-Yogo 2005)**:

| # Instruments | 10% Bias | 15% Bias | 20% Bias | 25% Bias |
|---------------|----------|----------|----------|----------|
| 1 | 16.38 | 8.96 | 6.66 | 5.53 |
| 2 | 19.93 | 11.59 | 8.75 | 7.25 |
| 3 | 22.30 | 12.83 | 9.54 | 7.80 |
| 5 | 26.87 | 15.09 | 10.27 | 8.84 |

---

### Phase 3: Main Estimation

**Model Specification**:

**Structural Equation**:
$$
Y_i = \beta_0 + \beta_1 D_i + X_i'\gamma + \epsilon_i
$$

Where $D_i$ is endogenous: $Cov(D_i, \epsilon_i) \neq 0$

**Two-Stage Least Squares (2SLS)**:

Stage 1: $\hat{D}_i = \hat{\pi}_0 + \hat{\pi}_1 Z_i + X_i'\hat{\gamma}$

Stage 2: $Y_i = \beta_0 + \beta_1 \hat{D}_i + X_i'\gamma + \epsilon_i$

**Python Implementation**:

```python
from iv_estimator import (
    estimate_2sls,
    estimate_liml,
    estimate_gmm,
    run_full_iv_analysis
)

# Option 1: Standard 2SLS
result_2sls = estimate_2sls(
    data=df,
    outcome="y",
    treatment="d",
    instruments=["z1", "z2"],
    controls=["x1", "x2"]
)

# Option 2: LIML (robust to weak instruments)
result_liml = estimate_liml(
    data=df,
    outcome="y",
    treatment="d",
    instruments=["z1", "z2"],
    controls=["x1", "x2"]
)

# Option 3: GMM (efficient with heteroskedasticity)
result_gmm = estimate_gmm(
    data=df,
    outcome="y",
    treatment="d",
    instruments=["z1", "z2"],
    controls=["x1", "x2"]
)

# Option 4: Full workflow (recommended)
result = run_full_iv_analysis(
    data=df,
    outcome="y",
    treatment="d",
    instruments=["z1", "z2"],
    controls=["x1", "x2"]
)

print(result.summary_table)
print(result.diagnostics)
```

**Returns**:
```python
CausalOutput(
    effect=1.05,           # Point estimate of LATE
    se=0.23,               # Standard error
    ci_lower=0.60,         # 95% CI lower bound
    ci_upper=1.50,         # 95% CI upper bound
    p_value=0.0001,        # Two-sided p-value
    diagnostics={
        'first_stage': {...},
        'weak_iv_test': DiagnosticResult(...),
        'overidentification_test': DiagnosticResult(...),
        'endogeneity_test': DiagnosticResult(...)
    },
    summary_table="...",
    interpretation="..."
)
```

### Phase 4: Diagnostics

| Check | Purpose | Implementation |
|-------|---------|----------------|
| Weak IV Test | Ensure instruments are strong enough | `first_stage_test()` |
| Overidentification Test | Test instrument validity (if K > 1) | `overidentification_test()` |
| Endogeneity Test | Confirm OLS is biased | `endogeneity_test()` |
| Compare Estimators | Check robustness | Compare 2SLS, LIML, GMM |

**Overidentification Test (Sargan-Hansen/J-test)**:

When you have more instruments than endogenous variables (overidentified), you can test whether all instruments are valid:

```python
from iv_estimator import overidentification_test

# Only valid when K_instruments > K_endogenous
j_test = overidentification_test(model_result)

print(f"Sargan-Hansen J-statistic: {j_test.statistic:.4f}")
print(f"P-value: {j_test.p_value:.4f}")
print(j_test.interpretation)
```

**Interpretation**:
- PASS if: p > 0.05 (cannot reject that all instruments are valid)
- FAIL if: p < 0.05 (at least one instrument may be invalid)
- CAUTION: This test has low power and cannot validate exclusion restriction

**Endogeneity Test (Hausman/Wu-Hausman)**:

Tests whether OLS and IV estimates are significantly different:

```python
from iv_estimator import endogeneity_test

hausman = endogeneity_test(
    data=df,
    outcome="y",
    treatment="d",
    instruments=["z1", "z2"],
    controls=["x1", "x2"]
)

print(f"Hausman statistic: {hausman.statistic:.4f}")
print(f"P-value: {hausman.p_value:.4f}")
print(hausman.interpretation)
```

**Interpretation**:
- Significant (p < 0.05): Treatment is endogenous, IV is needed
- Not significant: OLS may be consistent (but IV still valid)

### Phase 5: Reporting

**Standard Output Table Format**:

```
+--------------------------------------------------------------+
|              Table X: Instrumental Variables Results          |
+--------------------------------------------------------------+
|                         (1)        (2)        (3)             |
|                        OLS       2SLS       LIML              |
+--------------------------------------------------------------+
| Treatment (d)         0.45***    1.05***    1.03***           |
|                      (0.08)     (0.23)     (0.24)             |
|                                                               |
| Controls               Yes        Yes        Yes              |
|                                                               |
| Observations          1,000      1,000      1,000             |
| R-squared             0.345        -          -               |
+--------------------------------------------------------------+
| First-Stage F           -        45.67      45.67             |
| Sargan p-value          -        0.456      0.456             |
| Hausman p-value         -        0.003        -               |
+--------------------------------------------------------------+
| Notes: Robust standard errors in parentheses.                 |
| *** p<0.01, ** p<0.05, * p<0.1                               |
+--------------------------------------------------------------+
```

**Interpretation Template**:

```markdown
## Results Interpretation

Using instrumental variables estimation with [instrument description] as
instrument(s) for [treatment], we estimate a causal effect of **[beta]**
(SE = [se]) on [outcome].

### First Stage
The instrument(s) are strongly correlated with [treatment]:
- First-stage F-statistic: [F] (> 10 threshold)
- Partial R-squared: [R2]

### Identification
- Relevance: [Supported/Not supported] by first-stage F-test
- Exclusion restriction: [Argument for why instrument only affects outcome through treatment]
- Exogeneity: [Argument for why instrument is as-good-as-randomly assigned]

### Diagnostics
- Overidentification test (Sargan-Hansen): p = [p-value]
- Endogeneity test (Hausman): p = [p-value]

### LATE Interpretation
This effect represents the Local Average Treatment Effect (LATE) for
**compliers** - units whose treatment status is affected by the instrument.
This may differ from the Average Treatment Effect (ATE) if treatment effects
are heterogeneous.

### Economic Significance
[Interpretation of magnitude in practical terms]

### Caveats
- [Potential threats to exclusion restriction]
- [Sample/external validity considerations]
```

---

## Weak Instruments Problem

### Why Weak Instruments Are Dangerous

1. **Bias toward OLS**: 2SLS bias approaches OLS bias as instruments weaken
2. **Unreliable inference**: Standard errors understated, CIs have wrong coverage
3. **Sensitivity to specification**: Small changes lead to large estimate changes

### Solutions for Weak Instruments

| Method | When to Use | Implementation |
|--------|-------------|----------------|
| **LIML** | Moderate weakness (F: 5-10) | `estimate_liml()` |
| **Anderson-Rubin CI** | Any weakness level | Confidence set that is valid |
| **Fuller's LIML** | Moderate weakness | Bias-corrected LIML |
| **Find better instruments** | Always preferred | Domain expertise |

```python
# Compare 2SLS and LIML when instruments may be weak
result_2sls = estimate_2sls(data, outcome, treatment, instruments, controls)
result_liml = estimate_liml(data, outcome, treatment, instruments, controls)

print(f"2SLS: {result_2sls.effect:.4f} (SE: {result_2sls.se:.4f})")
print(f"LIML: {result_liml.effect:.4f} (SE: {result_liml.se:.4f})")

# If estimates differ substantially, weak instruments are a concern
```

---

## Multiple Instruments: Overidentification

When you have more instruments (K) than endogenous variables (1), the model is overidentified.

### Benefits of Multiple Instruments
- **Efficiency gains**: More instruments can reduce variance
- **Testability**: Can test overidentifying restrictions
- **Robustness checks**: Different instrument subsets

### Dangers of Multiple Instruments
- **Bias amplification**: Many weak instruments can worsen bias
- **Many instruments bias**: As K grows, 2SLS becomes biased toward OLS

### Recommendation
- Use few strong instruments rather than many weak ones
- Test subsets of instruments for robustness
- Compare exactly-identified (K=1) to overidentified estimates

---

## LATE Interpretation with Heterogeneous Effects

### The LATE Framework (Imbens & Angrist 1994)

With heterogeneous treatment effects, IV identifies the **Local Average Treatment Effect (LATE)** for compliers:

$$
\beta_{IV} = E[Y_i(1) - Y_i(0) | \text{Complier}]
$$

### Types of Units

| Type | Definition | Example (Draft Lottery) |
|------|------------|-------------------------|
| **Compliers** | D changes with Z | Serve if drafted, don't if not |
| **Always-takers** | D=1 regardless of Z | Volunteer regardless of lottery |
| **Never-takers** | D=0 regardless of Z | Avoid service regardless |
| **Defiers** | D opposite of Z | Ruled out by monotonicity |

### When LATE = ATE
- Treatment effects are homogeneous
- Compliers are representative of population
- Instrument affects everyone's treatment decision

### Policy Implications
- LATE may differ from ATE if compliers are special
- Different instruments identify different LATEs
- Be explicit about which population your estimate applies to

---

## Common Mistakes

### 1. Ignoring Weak Instruments

**Mistake**: Proceeding with 2SLS despite first-stage F < 10.

**Why it's wrong**: 2SLS is biased toward OLS with weak instruments. Standard errors and confidence intervals are unreliable.

**Correct approach**:
```python
# WRONG: Proceed with 2SLS despite weak instruments
first_stage = first_stage_test(data, treatment, instruments, controls)
if first_stage['f_statistic'] < 10:
    print("Warning: weak instruments")
result = estimate_2sls(...)  # Still biased!

# CORRECT: Use LIML and report weak IV diagnostics
if first_stage['f_statistic'] < 10:
    print("Using LIML due to weak instruments")
    result = estimate_liml(data, outcome, treatment, instruments, controls)
    # Also compute Anderson-Rubin confidence interval
```

### 2. Poor Exclusion Restriction Arguments

**Mistake**: Claiming exclusion restriction holds without rigorous justification.

**Why it's wrong**: This assumption is untestable and crucial for validity.

**Correct approach**:
- Articulate the causal mechanism explicitly
- Consider all pathways from Z to Y
- Discuss potential violations and their direction of bias
- Run placebo tests if possible

### 3. Many Weak Instruments

**Mistake**: Using many instruments to "increase power."

**Why it's wrong**: Many weak instruments cause 2SLS to be biased toward OLS, defeating the purpose of IV.

**Correct approach**:
```python
# WRONG: Use many potential instruments
instruments = ["z1", "z2", "z3", "z4", "z5", "z6", "z7", "z8"]
result = estimate_2sls(data, outcome, treatment, instruments, controls)

# CORRECT: Use few strong instruments
# Select based on institutional knowledge, not statistical significance
instruments = ["z1", "z2"]  # Two strongest conceptually
result = estimate_2sls(data, outcome, treatment, instruments, controls)
```

### 4. Misinterpreting LATE as ATE

**Mistake**: Generalizing IV estimates to the entire population.

**Why it's wrong**: With heterogeneous effects, LATE applies only to compliers.

**Correct approach**:
```markdown
# WRONG interpretation:
"IV shows that education increases wages by $X for everyone."

# CORRECT interpretation:
"IV shows that education increases wages by $X for those whose
education was affected by the instrument (compliers). This may
differ from the effect for those who would attend college
regardless of the instrument."
```

---

## Examples

### Example 1: Returns to Education (Card 1995 - College Proximity)

**Research Question**: What is the causal effect of education on earnings?

**Problem**: Education is endogenous (ability bias).

**Instrument**: Geographic proximity to a four-year college.

**Exclusion Restriction Argument**: Growing up near a college affects college attendance (lower costs) but does not directly affect earnings except through education.

```python
import pandas as pd
from iv_estimator import run_full_iv_analysis

# Load Card (1995) data
data = pd.read_csv("card_proximity.csv")

# Data structure:
# - lwage: log hourly wage
# - educ: years of education
# - nearc4: 1 if grew up near 4-year college
# - exper, black, south, smsa: controls

# Run IV analysis
result = run_full_iv_analysis(
    data=data,
    outcome="lwage",
    treatment="educ",
    instruments=["nearc4"],
    controls=["exper", "expersq", "black", "south", "smsa"]
)

print(result.summary_table)

# Compare OLS and IV estimates
# OLS typically shows ~7% return to education
# IV typically shows ~10-13% return for compliers
```

**Interpretation**:
The IV estimate suggests a return to education of approximately 10-13% per year for compliers (those whose education was affected by college proximity). This exceeds the OLS estimate, suggesting:
1. OLS suffers from downward ability bias, OR
2. LATE > ATE because compliers (those on the margin of attending) have higher returns than average

### Example 2: Quarter of Birth (Angrist & Krueger 1991)

**Research Question**: What is the causal effect of education on earnings?

**Instrument**: Quarter of birth (affects years of schooling through compulsory schooling laws).

```python
import pandas as pd
from iv_estimator import run_full_iv_analysis

# Load Angrist-Krueger data
data = pd.read_csv("ak1991.csv")

# Instruments: quarter of birth dummies
instruments = ["qob1", "qob2", "qob3"]  # qob4 is reference

result = run_full_iv_analysis(
    data=data,
    outcome="lwage",
    treatment="educ",
    instruments=instruments,
    controls=["yob"]  # year of birth controls
)

print(result.summary_table)
print(f"\nFirst-stage F: {result.diagnostics['first_stage']['f_statistic']:.2f}")
```

**Caution**: This is a classic example where many weak instruments cause problems. The quarter of birth instruments are individually weak, leading to concerns about bias toward OLS.

### Example 3: Synthetic IV Data Validation

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / 'lib' / 'python'))
from data_loader import generate_synthetic_iv_data

# Generate synthetic data with known parameters
data, true_params = generate_synthetic_iv_data(
    n=2000,
    treatment_effect=1.0,
    first_stage_strength=0.5,
    noise_std=1.0,
    random_state=42
)

print(f"True effect: {true_params['true_effect']}")
print(f"OLS bias (approximate): {true_params['ols_bias']:.4f}")
print(f"Theoretical first-stage F: {true_params['theoretical_first_stage_f']:.2f}")

# Run IV analysis
from iv_estimator import run_full_iv_analysis

result = run_full_iv_analysis(
    data=data,
    outcome="y",
    treatment="d",
    instruments=["z"],
    controls=["x1", "x2"]
)

print(f"\nEstimated effect: {result.effect:.4f}")
print(f"Bias: {(result.effect - true_params['true_effect'])*100:.2f}%")
print(f"95% CI covers truth: {result.ci_lower <= true_params['true_effect'] <= result.ci_upper}")
```

---

## References

### Seminal Papers
- Angrist, J. D., & Krueger, A. B. (1991). Does Compulsory School Attendance Affect Schooling and Earnings? *Quarterly Journal of Economics*, 106(4), 979-1014.
- Card, D. (1995). Using Geographic Variation in College Proximity to Estimate the Return to Schooling. In *Aspects of Labour Market Behaviour*.
- Imbens, G. W., & Angrist, J. D. (1994). Identification and Estimation of Local Average Treatment Effects. *Econometrica*, 62(2), 467-475.
- Stock, J. H., & Yogo, M. (2005). Testing for Weak Instruments in Linear IV Regression. In *Identification and Inference for Econometric Models*.

### Methodological Papers
- Staiger, D., & Stock, J. H. (1997). Instrumental Variables Regression with Weak Instruments. *Econometrica*, 65(3), 557-586.
- Bound, J., Jaeger, D. A., & Baker, R. M. (1995). Problems with Instrumental Variables Estimation When the Correlation Between the Instruments and the Endogenous Explanatory Variable is Weak. *JASA*, 90(430), 443-450.
- Andrews, I., Stock, J. H., & Sun, L. (2019). Weak Instruments in IV Regression: Theory and Practice. *Annual Review of Economics*, 11, 727-753.

### Textbook Treatments
- Angrist, J. D., & Pischke, J. S. (2009). *Mostly Harmless Econometrics*. Princeton University Press. Chapter 4.
- Angrist, J. D., & Pischke, J. S. (2015). *Mastering 'Metrics*. Princeton University Press. Chapter 3.
- Cunningham, S. (2021). *Causal Inference: The Mixtape*. Yale University Press. Chapter 7.
- Wooldridge, J. M. (2010). *Econometric Analysis of Cross Section and Panel Data*. MIT Press. Chapter 5.

### Software Documentation
- `linearmodels`: https://bashtage.github.io/linearmodels/iv/index.html
- `ivreg` (R): https://cran.r-project.org/package=ivreg
- `ivregress` (Stata): https://www.stata.com/manuals/rivregress.pdf

---

## Related Estimators

| Estimator | When to Use Instead |
|-----------|---------------------|
| `estimator-ols` | Treatment is exogenous (no endogeneity concern) |
| `estimator-did` | Panel data with parallel trends |
| `estimator-rd` | Treatment assigned by threshold |
| `estimator-control-function` | Endogeneity with more flexibility |
| `estimator-heckman` | Selection on unobservables |

---

## Appendix: Mathematical Details

### Derivation of 2SLS Estimator

**Setup**: $Y = D\beta + X\gamma + \epsilon$ where $Cov(D, \epsilon) \neq 0$

**Instrument**: Z satisfies:
1. $Cov(Z, D) \neq 0$ (relevance)
2. $Cov(Z, \epsilon) = 0$ (exogeneity)

**First Stage**:
$$
D = Z\pi + X\delta + v
$$

**Reduced Form**:
$$
Y = Z\theta + X\lambda + u
$$

**2SLS Estimator**:
$$
\hat{\beta}_{2SLS} = (D'P_Z D)^{-1} D'P_Z Y
$$

Where $P_Z = Z(Z'Z)^{-1}Z'$ is the projection matrix.

**Alternative Form (Wald Estimator with single binary instrument)**:
$$
\hat{\beta}_{IV} = \frac{Cov(Y, Z)}{Cov(D, Z)} = \frac{E[Y|Z=1] - E[Y|Z=0]}{E[D|Z=1] - E[D|Z=0]}
$$

### Asymptotic Properties

Under standard regularity conditions:
$$
\sqrt{n}(\hat{\beta}_{2SLS} - \beta) \xrightarrow{d} N(0, \sigma^2_\epsilon (D'P_Z D)^{-1})
$$

### Finite Sample Bias

2SLS bias in finite samples:
$$
E[\hat{\beta}_{2SLS} - \beta] \approx \frac{\sigma_{\epsilon v}}{\sigma^2_v} \cdot \frac{1}{F+1}
$$

Where F is the concentration parameter (related to first-stage F-statistic).

**Implication**: As F decreases, 2SLS bias approaches OLS bias.

### LIML Estimator

The Limited Information Maximum Likelihood estimator minimizes:
$$
\hat{\beta}_{LIML} = (D'M_X D - \hat{\kappa} D'M_{[X,Z]} D)^{-1}(D'M_X Y - \hat{\kappa} D'M_{[X,Z]} Y)
$$

Where $\hat{\kappa}$ is the smallest eigenvalue of a certain matrix.

**Property**: LIML is median-unbiased even with weak instruments (unlike 2SLS).

### GMM Estimator

With heteroskedasticity, the efficient GMM estimator is:
$$
\hat{\beta}_{GMM} = (D'Z \hat{W} Z'D)^{-1} D'Z \hat{W} Z'Y
$$

Where $\hat{W}$ is the optimal weighting matrix.
