---
name: estimator-iv
description: Use when estimating causal effects with Instrumental Variables. Triggers on IV, instrumental variable, 2SLS, two-stage least squares, LIML, endogeneity, weak instrument.
---

# Estimator: Instrumental Variables (IV)

> **Version**: 2.0.0 | **Type**: Estimator
> **Aliases**: IV, 2SLS, Two-Stage Least Squares, LIML, GMM-IV

## Overview

Instrumental Variables (IV) estimation identifies causal effects when the treatment variable is endogenous (correlated with the error term). IV uses an external source of variation - an instrument - that affects the outcome only through its effect on the treatment.

**Key Identification Assumption**: The instrument must be (1) relevant (correlated with treatment), (2) exogenous (uncorrelated with unobserved confounders), and (3) satisfy the exclusion restriction (affects outcome only through treatment).

## Quick Start

```python
from iv_estimator import run_full_iv_analysis

# Run complete IV analysis
result = run_full_iv_analysis(
    data=df,
    outcome="y",
    treatment="d",
    instruments=["z1", "z2"],
    controls=["x1", "x2"]
)

print(result.summary_table)
print(result.interpretation)
```

**CLI Usage**:
```bash
# Complete analysis
python scripts/run_iv_analysis.py --data data.csv --outcome y --treatment d \
    --instruments z1 z2 --controls x1 x2 --output results/

# Test instruments
python scripts/test_instruments.py --data data.csv --outcome y --treatment d \
    --instruments z1 z2 --verbose

# Visualize first stage
python scripts/visualize_first_stage.py --data data.csv --outcome y \
    --treatment d --instruments z1 --output figures/

# Weak-IV robust inference
python scripts/weak_iv_robust.py --data data.csv --outcome y --treatment d \
    --instruments z1 --method all
```

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

| Assumption | Description | Testable? | Reference |
|------------|-------------|-----------|-----------|
| **Relevance** | Instrument Z is correlated with treatment D | Yes | [identification_assumptions.md](references/identification_assumptions.md) |
| **Independence (Exogeneity)** | Instrument Z is uncorrelated with error term | No | [identification_assumptions.md](references/identification_assumptions.md) |
| **Exclusion Restriction** | Z affects Y only through D | No | [identification_assumptions.md](references/identification_assumptions.md) |
| **Monotonicity** (for LATE) | Z affects D in same direction for all units | No | [identification_assumptions.md](references/identification_assumptions.md) |

**Detailed discussion**: See [references/identification_assumptions.md](references/identification_assumptions.md)

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

**Data Validation**:
```python
from iv_estimator import validate_iv_data

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

**Stock-Yogo Critical Values**: See [references/diagnostic_tests.md](references/diagnostic_tests.md)

### Phase 3: Main Estimation

**Model Specification**:

**Structural Equation**:
$$
Y_i = \beta_0 + \beta_1 D_i + X_i'\gamma + \epsilon_i
$$

Where $D_i$ is endogenous: $Cov(D_i, \epsilon_i) \neq 0$

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
```

**Estimation Methods Reference**: See [references/estimation_methods.md](references/estimation_methods.md)

### Phase 4: Diagnostics

| Check | Purpose | Implementation | Reference |
|-------|---------|----------------|-----------|
| Weak IV Test | Ensure instruments are strong enough | `weak_iv_diagnostics()` | [diagnostic_tests.md](references/diagnostic_tests.md) |
| Overidentification Test | Test instrument validity (if K > 1) | `overidentification_test()` | [diagnostic_tests.md](references/diagnostic_tests.md) |
| Endogeneity Test | Confirm OLS is biased | `endogeneity_test()` | [diagnostic_tests.md](references/diagnostic_tests.md) |
| Compare Estimators | Check robustness | Compare 2SLS, LIML, GMM | [estimation_methods.md](references/estimation_methods.md) |

**Comprehensive Testing**:
```bash
python scripts/test_instruments.py --data data.csv --outcome y --treatment d \
    --instruments z1 z2 --controls x1 x2 --verbose
```

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
| First-stage F           -        45.67      45.67             |
| Sargan p-value          -        0.456      0.456             |
| Hausman p-value         -        0.003        -               |
+--------------------------------------------------------------+
```

**Reporting Standards**: See [references/reporting_standards.md](references/reporting_standards.md)

**LaTeX Template**: See [assets/latex/iv_table.tex](assets/latex/iv_table.tex)

**Markdown Report Template**: See [assets/markdown/iv_report.md](assets/markdown/iv_report.md)

---

## Weak Instruments Problem

### Why Weak Instruments Are Dangerous

1. **Bias toward OLS**: 2SLS bias approaches OLS bias as instruments weaken
2. **Unreliable inference**: Standard errors understated, CIs have wrong coverage
3. **Sensitivity to specification**: Small changes lead to large estimate changes

### Solutions for Weak Instruments

| Method | When to Use | Script |
|--------|-------------|--------|
| **LIML** | Moderate weakness (F: 5-10) | `estimate_liml()` |
| **Anderson-Rubin CI** | Any weakness level | `scripts/weak_iv_robust.py` |
| **Fuller's LIML** | Moderate weakness | `scripts/weak_iv_robust.py --method fuller` |
| **Find better instruments** | Always preferred | Domain expertise |

```bash
# Weak-IV robust inference
python scripts/weak_iv_robust.py --data data.csv --outcome y --treatment d \
    --instruments z1 z2 --method all --verbose
```

**Detailed guidance**: See [references/estimation_methods.md](references/estimation_methods.md)

---

## Common Mistakes

| Mistake | Problem | Solution | Reference |
|---------|---------|----------|-----------|
| Ignoring weak instruments | Biased estimates | Use LIML, check F > 10 | [common_errors.md](references/common_errors.md) |
| Forbidden regressions | Inconsistent estimates | Use proper 2SLS | [common_errors.md](references/common_errors.md) |
| Many weak instruments | Bias toward OLS | Use few strong instruments | [common_errors.md](references/common_errors.md) |
| Misinterpreting LATE | Wrong policy conclusions | Discuss complier population | [common_errors.md](references/common_errors.md) |
| Poor exclusion restriction | Invalid identification | Rigorous justification | [common_errors.md](references/common_errors.md) |

**Full error catalog**: See [references/common_errors.md](references/common_errors.md)

---

## Visualization

Generate publication-ready figures:

```bash
python scripts/visualize_first_stage.py --data data.csv --outcome y \
    --treatment d --instruments z1 --controls x1 x2 --output figures/
```

**Outputs**:
- First-stage binned scatter plots
- Reduced form visualization
- IV vs OLS comparison
- Instrument strength diagnostics

---

## Examples

### Example 1: Returns to Education (Card 1995)

```python
import pandas as pd
from iv_estimator import run_full_iv_analysis

# Load Card (1995) data
data = pd.read_csv("card_proximity.csv")

# Run IV analysis
result = run_full_iv_analysis(
    data=data,
    outcome="lwage",
    treatment="educ",
    instruments=["nearc4"],
    controls=["exper", "expersq", "black", "south", "smsa"]
)

print(result.summary_table)
```

### Example 2: Synthetic Data Validation

```python
from iv_estimator import validate_estimator

# Validate estimator on synthetic data with known parameters
validation = validate_estimator(verbose=True)
print(f"Bias: {validation['iv_bias_pct']:.2f}%")
print(f"CI covers truth: {validation['ci_covers_truth']}")
```

---

## Directory Structure

```
estimator-iv/
├── SKILL.md                     # This file
├── iv_estimator.py              # Core implementation
├── references/
│   ├── identification_assumptions.md  # Relevance, exclusion, monotonicity
│   ├── diagnostic_tests.md            # Stock-Yogo, Sargan, Hausman
│   ├── estimation_methods.md          # 2SLS, LIML, GMM, JIVE
│   ├── reporting_standards.md         # Tables, AR CI, best practices
│   └── common_errors.md               # Weak IV, forbidden regressions
├── scripts/
│   ├── run_iv_analysis.py       # Complete IV analysis CLI
│   ├── test_instruments.py      # Instrument validity tests
│   ├── visualize_first_stage.py # First-stage plots
│   └── weak_iv_robust.py        # Anderson-Rubin, Fuller
└── assets/
    ├── latex/
    │   └── iv_table.tex         # LaTeX table template
    └── markdown/
        └── iv_report.md         # Analysis report template
```

---

## References

### Seminal Papers
- Angrist, J. D., & Krueger, A. B. (1991). Does Compulsory School Attendance Affect Schooling and Earnings? *Quarterly Journal of Economics*, 106(4), 979-1014.
- Card, D. (1995). Using Geographic Variation in College Proximity to Estimate the Return to Schooling.
- Imbens, G. W., & Angrist, J. D. (1994). Identification and Estimation of Local Average Treatment Effects. *Econometrica*, 62(2), 467-475.
- Stock, J. H., & Yogo, M. (2005). Testing for Weak Instruments in Linear IV Regression.

### Methodological Papers
- Staiger, D., & Stock, J. H. (1997). Instrumental Variables Regression with Weak Instruments. *Econometrica*, 65(3), 557-586.
- Andrews, I., Stock, J. H., & Sun, L. (2019). Weak Instruments in IV Regression: Theory and Practice. *Annual Review of Economics*, 11, 727-753.

### Textbook Treatments
- Angrist, J. D., & Pischke, J. S. (2009). *Mostly Harmless Econometrics*. Princeton University Press. Chapter 4.
- Angrist, J. D., & Pischke, J. S. (2015). *Mastering 'Metrics*. Princeton University Press. Chapter 3.
- Cunningham, S. (2021). *Causal Inference: The Mixtape*. Yale University Press. Chapter 7.

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

## Changelog

### v2.0.0 (2024)
- Added K-Dense structure with references/, scripts/, assets/ directories
- New CLI scripts for complete workflow
- Comprehensive documentation following Angrist-Pischke methodology
- Added weak-IV robust inference tools
- LaTeX and Markdown report templates

### v1.0.0
- Initial release with 2SLS, LIML, GMM estimators
