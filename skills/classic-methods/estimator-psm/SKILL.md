---
name: estimator-psm
description: Use when estimating causal effects with Propensity Score Matching. Triggers on PSM, propensity score, matching, ATT, ATE, selection on observables, nearest neighbor, caliper, kernel matching, Mahalanobis distance.
---

# Estimator: Propensity Score Matching (PSM)

> **Version**: 2.0.0 | **Type**: Estimator
> **Aliases**: PSM, Propensity Score Matching, Matching Estimator, Selection on Observables

## Overview

Propensity Score Matching (PSM) estimates causal effects by matching treated units to control units with similar propensity scores (predicted probability of receiving treatment given observed covariates). By comparing outcomes between matched treated and control units, PSM removes confounding bias due to observed covariates.

**Key Identification Assumption**: Conditional Independence Assumption (CIA) / Unconfoundedness - conditional on observed covariates X, treatment assignment D is independent of potential outcomes: $(Y_0, Y_1) \perp D | X$

## Quick Reference

| Resource | Location | Purpose |
|----------|----------|---------|
| **Main Implementation** | `psm_estimator.py` | Core PSM functions |
| **CLI Script** | `scripts/run_psm_analysis.py` | Complete analysis workflow |
| **Balance Tests** | `scripts/test_balance.py` | Comprehensive balance checking |
| **Overlap Visualization** | `scripts/visualize_overlap.py` | PS distribution plots |
| **Sensitivity Analysis** | `scripts/sensitivity_analysis.py` | Rosenbaum bounds |
| **LaTeX Tables** | `assets/latex/psm_table.tex` | Publication-ready tables |
| **Report Template** | `assets/markdown/psm_report.md` | Full analysis report |

## Reference Documents

| Document | Content |
|----------|---------|
| [Identification Assumptions](references/identification_assumptions.md) | CIA, overlap, SUTVA - detailed explanation and testability |
| [Diagnostic Tests](references/diagnostic_tests.md) | SMD, variance ratio, KS test, omnibus tests |
| [Estimation Methods](references/estimation_methods.md) | NN, caliper, kernel, Mahalanobis, IPW, doubly robust |
| [Reporting Standards](references/reporting_standards.md) | Balance tables, Love plots, write-up templates |
| [Common Errors](references/common_errors.md) | Post-treatment variables, trimming, balance checking |

---

## When to Use

### Ideal Scenarios
- Selection into treatment is based on observed characteristics (selection on observables)
- Rich set of pre-treatment covariates available
- Observational study where randomization is not feasible
- Want to create a matched comparison group that resembles the treatment group
- Need to check covariate balance explicitly

### Data Requirements
- [ ] Cross-sectional or panel data with clearly defined treatment and control groups
- [ ] Binary treatment indicator
- [ ] Pre-treatment covariates that predict treatment assignment
- [ ] Outcome variable measured post-treatment
- [ ] Sufficient overlap in covariate distributions between treatment and control

### When NOT to Use
- Unobserved confounding is likely present -> Consider `estimator-iv` (Instrumental Variables)
- Poor overlap / limited common support -> Consider `estimator-synthetic-control` or trimming
- Treatment effect varies by propensity score in complex ways -> Consider `estimator-causal-forest`
- Panel data with parallel trends -> Consider `estimator-did` (often stronger design)
- Regression discontinuity design available -> Consider `estimator-rd`

---

## Identification Assumptions

> **Reference**: [references/identification_assumptions.md](references/identification_assumptions.md)

| Assumption | Description | Testable? |
|------------|-------------|-----------|
| **Unconfoundedness (CIA)** | No unmeasured confounders; treatment assignment is ignorable given X | No (fundamentally untestable) |
| **Common Support (Overlap)** | For all X, probability of treatment is strictly between 0 and 1: $0 < P(D=1|X) < 1$ | Yes (propensity score distribution) |
| **SUTVA** | No interference between units; stable unit treatment values | No (domain knowledge) |
| **Correct PS Model** | Propensity score model is correctly specified | Partially (balance checks) |

---

## Workflow

```
+-------------------------------------------------------------+
|                    PSM ESTIMATOR WORKFLOW                     |
+-------------------------------------------------------------+
|  1. SETUP          -> Define outcome, treatment, covariates   |
|  2. PS ESTIMATION  -> Estimate propensity scores (logit/ML)   |
|  3. OVERLAP CHECK  -> Verify common support assumption        |
|  4. MATCHING       -> Match treated to controls               |
|  5. BALANCE CHECK  -> Verify covariate balance improved       |
|  6. ESTIMATION     -> Estimate ATT/ATE on matched sample      |
|  7. SENSITIVITY    -> Rosenbaum bounds for hidden bias        |
+-------------------------------------------------------------+
```

### Quick Start: CLI Script

```bash
# Basic analysis
python scripts/run_psm_analysis.py \
    --data data.csv \
    --outcome earnings \
    --treatment training \
    --covariates age education experience \
    --output results/ \
    --sensitivity \
    --verbose

# With specific matching method
python scripts/run_psm_analysis.py \
    --data data.csv \
    --outcome y --treatment d \
    --covariates x1 x2 x3 \
    --match-method kernel \
    --ps-method gbm \
    --save-plots
```

---

### Phase 1: Setup

**Objective**: Prepare data and define model specification

```python
from psm_estimator import validate_psm_data

# Define variables
outcome = "y"                    # Outcome variable name
treatment = "treated"            # Treatment indicator (0/1)
covariates = ["x1", "x2", "x3"]  # Pre-treatment covariates

# Validate data structure
validation = validate_psm_data(
    data=df,
    outcome=outcome,
    treatment=treatment,
    covariates=covariates
)
print(validation)
```

**Data Validation Checklist**:
- [ ] Treatment is binary (0/1)
- [ ] Covariates are measured PRE-treatment (no post-treatment variables!)
- [ ] No missing values in key variables (or explicit handling strategy)
- [ ] Sufficient treated and control observations
- [ ] Covariates should predict treatment assignment

---

### Phase 2: Propensity Score Estimation

> **Reference**: [references/estimation_methods.md](references/estimation_methods.md)

**Propensity Score Definition**:
$$
e(X) = P(D = 1 | X) = E[D | X]
$$

```python
from psm_estimator import estimate_propensity_score

# Option 1: Logistic regression (default)
ps_result = estimate_propensity_score(
    data=df,
    treatment="treated",
    covariates=["x1", "x2", "x3"],
    method="logit"
)

# Option 2: Machine learning methods
ps_result = estimate_propensity_score(
    data=df,
    treatment="treated",
    covariates=["x1", "x2", "x3"],
    method="gbm",  # or "random_forest", "lasso"
    cv_folds=5
)

print(ps_result.summary())
```

---

### Phase 3: Common Support / Overlap Check

> **Reference**: [references/diagnostic_tests.md](references/diagnostic_tests.md)

```python
from psm_estimator import check_common_support, plot_propensity_overlap

# Visual inspection
fig = plot_propensity_overlap(
    ps_treated=df.loc[df['treated']==1, 'propensity_score'],
    ps_control=df.loc[df['treated']==0, 'propensity_score']
)

# Statistical check
overlap_result = check_common_support(
    ps_treated=df.loc[df['treated']==1, 'propensity_score'],
    ps_control=df.loc[df['treated']==0, 'propensity_score'],
    method="minmax"
)
print(overlap_result)
```

**CLI Alternative**:
```bash
python scripts/visualize_overlap.py \
    --data data.csv \
    --treatment treated \
    --ps propensity_score \
    --output overlap_plot.png
```

---

### Phase 4: Matching

> **Reference**: [references/estimation_methods.md](references/estimation_methods.md)

**Matching Algorithms**:

| Method | Best For | Python Function |
|--------|----------|-----------------|
| **Nearest Neighbor** | Default choice | `match_nearest_neighbor()` |
| **Nearest Neighbor + Caliper** | Enforce match quality | `match_nearest_neighbor(caliper=0.1)` |
| **Kernel** | Use all controls, smooth | `match_kernel()` |
| **Mahalanobis** | Direct covariate matching | `match_mahalanobis()` |

```python
from psm_estimator import match_nearest_neighbor, match_kernel

# Option 1: Nearest Neighbor (1:1, with replacement)
matched = match_nearest_neighbor(
    data=df,
    propensity_score="ps",
    treatment="treated",
    n_neighbors=1,
    replacement=True,
    caliper=0.1,
    caliper_scale="ps_std"
)

# Option 2: Kernel Matching
matched = match_kernel(
    data=df,
    propensity_score="ps",
    treatment="treated",
    kernel="epanechnikov",
    bandwidth="optimal"
)

print(matched.summary())
```

---

### Phase 5: Balance Checking

> **Reference**: [references/diagnostic_tests.md](references/diagnostic_tests.md)

**Balance Metrics**:
- **Standardized Mean Difference (SMD)**: Target |SMD| < 0.1
- **Variance Ratio**: Target between 0.5 and 2.0

```python
from psm_estimator import check_balance, create_balance_table, plot_balance

# Calculate balance
balance_before = check_balance(df, "treated", covariates)
balance_after = check_balance(matched.matched_data, "treated", covariates,
                              weights="_match_weight")

# Create balance table
print(create_balance_table(df, matched.matched_data, "treated", covariates,
                           weights_after="_match_weight"))

# Love plot
fig = plot_balance(balance_before, balance_after, threshold=0.1)
```

**CLI Alternative**:
```bash
python scripts/test_balance.py \
    --data matched.csv \
    --treatment treated \
    --covariates age education income \
    --love-plot love_plot.png
```

---

### Phase 6: Effect Estimation

```python
from psm_estimator import estimate_att, estimate_ate

# ATT on matched sample
att_result = estimate_att(
    data=matched.matched_data,
    outcome="y",
    treatment="treated",
    weights="_match_weight",
    se_method="bootstrap"
)

print(f"ATT: {att_result.effect:.4f} (SE: {att_result.se:.4f})")
print(f"95% CI: [{att_result.ci_lower:.4f}, {att_result.ci_upper:.4f}]")

# ATE using IPW
ate_result = estimate_ate(
    data=df,
    outcome="y",
    treatment="treated",
    propensity_score="ps",
    estimator="doubly_robust"
)
```

---

### Phase 7: Sensitivity Analysis

> **Reference**: [references/identification_assumptions.md](references/identification_assumptions.md)

**Rosenbaum Bounds** examine how much hidden bias would be needed to invalidate results.

```python
from psm_estimator import rosenbaum_sensitivity

sensitivity = rosenbaum_sensitivity(
    data=matched.matched_data,
    outcome="y",
    treatment="treated",
    gamma_range=[1.0, 1.5, 2.0, 2.5, 3.0]
)

print(sensitivity.summary())
```

**CLI Alternative**:
```bash
python scripts/sensitivity_analysis.py \
    --data matched.csv \
    --outcome y \
    --treatment treated \
    --gamma-max 3.0 \
    --output sensitivity.png
```

---

## Full Analysis Example

```python
from psm_estimator import run_full_psm_analysis

# Run complete workflow
result = run_full_psm_analysis(
    data=df,
    outcome="earnings",
    treatment="training",
    covariates=["age", "education", "experience", "married"],
    match_method="nearest_neighbor",
    ps_method="logit",
    caliper=0.1,
    estimand="ATT",
    run_sensitivity=True
)

# View results
print(result.summary_table)
print(result.interpretation)
```

---

## Common Mistakes

> **Reference**: [references/common_errors.md](references/common_errors.md)

### 1. Including Post-Treatment Variables
```python
# WRONG
covariates = ["x1", "post_treatment_mediator"]

# CORRECT
covariates = ["x1_baseline", "x2_baseline"]
```

### 2. Ignoring Common Support
Always check overlap before matching. See [Overlap Assessment](references/diagnostic_tests.md#2-overlap-assessment).

### 3. Not Checking Balance
Matching doesn't guarantee balance. Always verify with SMD and variance ratios.

### 4. Wrong Standard Errors
Use bootstrap or Abadie-Imbens SEs, not naive analytical SEs.

### 5. Claiming Unconfoundedness Holds
This is untestable. Use sensitivity analysis to assess robustness.

---

## Reporting

> **Reference**: [references/reporting_standards.md](references/reporting_standards.md)

### Templates

- **LaTeX Tables**: [assets/latex/psm_table.tex](assets/latex/psm_table.tex)
- **Markdown Report**: [assets/markdown/psm_report.md](assets/markdown/psm_report.md)

### Required Elements

1. **Sample sizes** (treated, control, matched)
2. **PS model specification** and diagnostics (AUC)
3. **Overlap assessment** (PS distribution plot)
4. **Balance table** with SMD before/after
5. **Love plot** showing balance improvement
6. **Effect estimate** with CI and p-value
7. **Sensitivity analysis** (Rosenbaum bounds)

---

## Related Estimators

| Estimator | When to Use Instead |
|-----------|---------------------|
| `estimator-did` | Panel data with parallel trends assumption |
| `estimator-iv` | Unobserved confounding with valid instrument |
| `estimator-rd` | Treatment assigned by threshold rule |
| `estimator-synthetic-control` | Few treated units, aggregate data |
| `estimator-causal-forest` | Heterogeneous treatment effects |

---

## References

### Seminal Papers
- Rosenbaum, P. R., & Rubin, D. B. (1983). The Central Role of the Propensity Score in Observational Studies. *Biometrika*, 70(1), 41-55. [31,493 citations]
- Imbens, G. W., & Angrist, J. D. (1994). Identification and Estimation of Local Average Treatment Effects. *Econometrica*, 62(2), 467-475. [5,068 citations]
- Abadie, A., & Imbens, G. W. (2006). Large Sample Properties of Matching Estimators for Average Treatment Effects. *Econometrica*, 74(1), 235-267. [3,044 citations]

### Methodological Papers
- Hirano, K., Imbens, G. W., & Ridder, G. (2003). Efficient Estimation of Average Treatment Effects Using the Estimated Propensity Score. *Econometrica*, 71(4), 1161-1189. [1,999 citations]
- Abadie, A., & Imbens, G. W. (2011). Bias-Corrected Matching Estimators for Average Treatment Effects. *Journal of Business & Economic Statistics*, 29(1), 1-11. [1,903 citations]
- Abadie, A., & Imbens, G. W. (2008). On the Failure of the Bootstrap for Matching Estimators. *Econometrica*, 76(6), 1537-1557.

### Critical Perspectives
- King, G., & Nielsen, R. (2019). Why Propensity Scores Should Not Be Used for Matching. *Political Analysis*, 27(4), 435-454. [1,575 citations]

### Practical Guides
- Stuart, E. A. (2010). Matching Methods for Causal Inference: A Review and a Look Forward. *Statistical Science*, 25(1), 1-21. [942 citations]
- Caliendo, M., & Kopeinig, S. (2008). Some Practical Guidance for the Implementation of Propensity Score Matching. *Journal of Economic Surveys*, 22(1), 31-72. [6,819 citations]

### Textbooks
- Imbens, G. W., & Rubin, D. B. (2015). *Causal Inference for Statistics, Social, and Biomedical Sciences*. Cambridge University Press.
- Cunningham, S. (2021). *Causal Inference: The Mixtape*. Yale University Press.

---

## Directory Structure

```
estimator-psm/
├── SKILL.md                           # This file
├── psm_estimator.py                   # Main implementation
├── references/
│   ├── identification_assumptions.md # CIA, overlap, SUTVA
│   ├── diagnostic_tests.md           # Balance tests, overlap assessment
│   ├── estimation_methods.md         # Matching algorithms, IPW, DR
│   ├── reporting_standards.md        # Tables, figures, write-ups
│   └── common_errors.md              # Pitfalls and corrections
├── scripts/
│   ├── run_psm_analysis.py           # Full analysis CLI
│   ├── test_balance.py               # Balance testing
│   ├── visualize_overlap.py          # Overlap visualization
│   └── sensitivity_analysis.py       # Rosenbaum bounds
└── assets/
    ├── latex/
    │   └── psm_table.tex             # LaTeX table template
    └── markdown/
        └── psm_report.md             # Report template
```
