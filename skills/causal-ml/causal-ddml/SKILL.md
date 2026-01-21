---
name: causal-ddml
description: Use when estimating causal effects with Double/Debiased Machine Learning. Triggers on DDML, double machine learning, debiased, cross-fitting, Chernozhukov, high-dimensional, partially linear, PLR, IRM, orthogonal score.
---

# Estimator: Double/Debiased Machine Learning (DDML)

> **Version**: 2.0.0 | **Type**: Estimator | **Structure**: K-Dense
> **Aliases**: DDML, DML, Double ML, Debiased ML, Orthogonal ML

## Overview

Double/Debiased Machine Learning (DDML) estimates causal effects by combining machine learning methods for nuisance parameter estimation with debiased/orthogonal score functions that yield valid statistical inference. The method uses sample splitting (cross-fitting) to avoid overfitting bias while leveraging the flexibility of modern ML methods.

**Key Innovation**: DDML uses Neyman-orthogonal moment conditions that are insensitive (to first order) to estimation errors in nuisance functions, enabling the use of regularized/ML estimators without compromising valid inference.

**Primary Reference**: Chernozhukov, V., et al. (2018). Double/Debiased Machine Learning for Treatment and Structural Parameters. *The Econometrics Journal*, 21(1), C1-C68.

---

## Quick Reference

### When to Use

| Scenario | Recommendation |
|----------|----------------|
| High-dimensional controls (p >> n or many relevant controls) | Use DDML |
| Complex, nonlinear confounding relationships | Use DDML |
| Need valid inference with ML adjustment | Use DDML |
| Low-dimensional, simple relationships | Consider OLS instead |
| Small samples (n < 100) | Avoid - cross-fitting unreliable |
| Unconfoundedness clearly violated | Consider IV or DID instead |

### Model Selection

| Treatment Type | Recommended Model | Estimand |
|----------------|-------------------|----------|
| Continuous | PLR | Constant ATE |
| Binary (constant effect) | PLR | Constant ATE |
| Binary (heterogeneous) | IRM | ATE with heterogeneity |
| Endogenous + binary IV | IIVM | LATE |
| Endogenous + continuous IV | PLIV | Constant ATE |

### CLI Scripts

```bash
# Complete DDML analysis
python scripts/run_ddml_analysis.py --data data.csv --outcome y --treatment d

# Tune nuisance models
python scripts/tune_nuisance_models.py --data data.csv --outcome y --treatment d

# Generate diagnostic plots
python scripts/cross_fit_diagnostics.py --data data.csv --outcome y --treatment d

# Sensitivity analysis
python scripts/sensitivity_analysis.py --data data.csv --outcome y --treatment d

# Compare PLR vs IRM
python scripts/compare_estimators.py --data data.csv --outcome y --treatment d
```

---

## Directory Structure

```
causal-ddml/
├── SKILL.md                    # This file - main documentation
├── ddml_estimator.py           # Core estimation functions
├── references/                 # Detailed reference documentation
│   ├── identification_assumptions.md   # Neyman orthogonality, cross-fitting, rates
│   ├── diagnostic_tests.md             # Cross-fit diagnostics, nuisance quality
│   ├── estimation_methods.md           # PLR, IRM, IIVM, PLIV methods
│   ├── model_selection.md              # Learner selection, ensemble approaches
│   ├── reporting_standards.md          # Tables, CIs, robustness reporting
│   └── common_errors.md                # Pitfalls and how to avoid them
├── scripts/                    # Executable CLI tools
│   ├── run_ddml_analysis.py            # Complete analysis workflow
│   ├── tune_nuisance_models.py         # Automated hyperparameter tuning
│   ├── cross_fit_diagnostics.py        # Diagnostic visualization
│   ├── sensitivity_analysis.py         # Robustness checks
│   └── compare_estimators.py           # Model comparison
└── assets/                     # Templates and formatting
    ├── latex/
    │   └── ddml_table.tex              # LaTeX table templates
    └── markdown/
        └── ddml_report.md              # Analysis report template
```

---

## Identification Assumptions

> **Detailed Reference**: `references/identification_assumptions.md`

| Assumption | Description | Testable? |
|------------|-------------|-----------|
| **Unconfoundedness** | $(Y(0), Y(1)) \perp D \| X$ | No |
| **Overlap/Positivity** | $0 < P(D=1\|X) < 1$ | Yes |
| **Neyman Orthogonality** | Score insensitive to nuisance errors | By construction |
| **Rate Conditions** | $\|\hat{\ell} - \ell_0\| \cdot \|\hat{m} - m_0\| = o_P(n^{-1/2})$ | Partially |

### Key Insight: Product Rate Condition

DDML requires that the **product** of nuisance estimation errors decays faster than $n^{-1/2}$. This is weaker than requiring each to be $\sqrt{n}$-consistent, enabling use of regularized ML estimators.

---

## Workflow

```
+-------------------------------------------------------------+
|                    DDML ESTIMATOR WORKFLOW                    |
+-------------------------------------------------------------+
|  1. SETUP          -> Define Y, D, X (high-dimensional)       |
|  2. MODEL SELECTION-> Choose first-stage ML learners          |
|  3. CROSS-FITTING  -> K-fold sample splitting (K=5 typical)   |
|  4. ESTIMATION     -> PLR (Partially Linear) or IRM (Inter.)  |
|  5. INFERENCE      -> Debiased estimates + valid SEs          |
|  6. REPORTING      -> Tables with multiple ML specifications  |
+-------------------------------------------------------------+
```

### Phase 1: Setup

```python
from ddml_estimator import validate_ddml_setup, create_ddml_data

# Validate data structure
validation = validate_ddml_setup(
    data=df,
    outcome="y",
    treatment="d",
    controls=control_vars,
    n_folds=5
)

if not validation['is_valid']:
    raise ValueError(f"Validation failed: {validation['errors']}")
```

### Phase 2: Model Selection

> **Detailed Reference**: `references/model_selection.md`

```python
from ddml_estimator import select_first_stage_learners

# Auto-select best ML learners via cross-validation
best_learners = select_first_stage_learners(
    X=df[control_vars],
    y=df['y'],
    d=df['d'],
    cv_folds=5
)
```

**Learner Recommendations**:
| Scenario | Learner |
|----------|---------|
| Sparse, linear | Lasso, Elastic Net |
| Complex nonlinear | Random Forest, XGBoost |
| Very high-dimensional | Lasso + RF ensemble |
| Unknown structure | Compare multiple |

### Phase 3: Cross-Fitting

> **Detailed Reference**: `references/identification_assumptions.md` (Section 2)

```
K-Fold Cross-Fitting (K=5):
+--------------------------------------------------+
| Fold 1: Train on [2,3,4,5] -> Predict on [1]     |
| Fold 2: Train on [1,3,4,5] -> Predict on [2]     |
| ...                                               |
+--------------------------------------------------+
Result: Out-of-sample predictions for ALL observations
```

**Choosing K**:
| Sample Size | K | n/K |
|-------------|---|-----|
| n < 500 | 2-3 | ~150+ |
| 500-2000 | 5 | ~200+ |
| n > 2000 | 5-10 | ~200+ |

### Phase 4: Estimation

> **Detailed Reference**: `references/estimation_methods.md`

**PLR (Partially Linear Regression)**:
```python
from ddml_estimator import estimate_plr

result = estimate_plr(
    data=df,
    outcome="y",
    treatment="d",
    controls=control_vars,
    ml_l='lasso',      # E[Y|X] learner
    ml_m='lasso',      # E[D|X] learner
    n_folds=5
)
```

**IRM (Interactive Regression Model)**:
```python
from ddml_estimator import estimate_irm

result = estimate_irm(
    data=df,
    outcome="y",
    treatment="d",  # Must be binary
    controls=control_vars,
    ml_g='random_forest',     # E[Y|D,X] learner
    ml_m='logistic_lasso',    # P(D=1|X) learner
    n_folds=5,
    trimming_threshold=0.01   # Handle extreme propensities
)
```

### Phase 5: Inference

```python
print(f"Effect: {result.effect:.4f}")
print(f"SE: {result.se:.4f}")
print(f"95% CI: [{result.ci_lower:.4f}, {result.ci_upper:.4f}]")
print(f"P-value: {result.p_value:.4f}")
```

### Phase 6: Robustness

> **Detailed Reference**: `references/diagnostic_tests.md`

```python
from ddml_estimator import compare_learners

comparison = compare_learners(
    data=df,
    outcome="y",
    treatment="d",
    controls=control_vars,
    learner_list=['lasso', 'ridge', 'random_forest', 'xgboost']
)

print(comparison.summary_table)
print(f"Effect range: [{comparison.sensitivity['min_effect']:.4f}, "
      f"{comparison.sensitivity['max_effect']:.4f}]")
```

---

## PLR vs IRM Model Comparison

| Aspect | PLR | IRM |
|--------|-----|-----|
| Treatment Type | Any (continuous/binary) | Binary only |
| Effect Assumption | Constant | Allows heterogeneity |
| Estimand | ATE | ATE, ATTE |
| Nuisance Functions | E[Y\|X], E[D\|X] | E[Y\|D,X], P(D=1\|X) |
| Score | Partialling out | AIPW (doubly robust) |
| When to Use | Continuous D, constant effects | Binary D, heterogeneous effects |

---

## Diagnostic Tests

> **Detailed Reference**: `references/diagnostic_tests.md`

### Cross-Fitting Stability

```bash
python scripts/cross_fit_diagnostics.py --data data.csv --outcome y --treatment d --output plots/
```

Generates:
- `fold_variation.png` - Estimates by fold
- `repetition_stability.png` - Stability across repetitions
- `residuals.png` - Residual diagnostics
- `propensity_overlap.png` - Propensity distribution (IRM)
- `nuisance_performance.png` - Predicted vs actual

### Nuisance Model Quality

```python
# Check in result diagnostics
print(result.diagnostics['r2_y_given_x'])  # Outcome model R2
print(result.diagnostics['r2_d_given_x'])  # Treatment model R2
```

### Propensity Overlap (IRM)

```python
# Check propensity distribution
print(result.diagnostics['propensity_summary'])
# {min, max, mean, n_extreme_low, n_extreme_high}
```

---

## Common Errors

> **Detailed Reference**: `references/common_errors.md`

### 1. Not Using Cross-Fitting

```python
# WRONG: In-sample predictions
model.fit(X, y)
resid = y - model.predict(X)  # Overfitted!

# CORRECT: Cross-validated predictions
resid = y - cross_val_predict(model, X, y, cv=5)
```

### 2. IRM with Continuous Treatment

```python
# WRONG
result = estimate_irm(data, outcome, continuous_treatment, controls)

# CORRECT: Use PLR for continuous treatment
result = estimate_plr(data, outcome, continuous_treatment, controls)
```

### 3. Ignoring Propensity Overlap

```python
# ALWAYS check for extreme propensities with IRM
result = estimate_irm(..., trimming_threshold=0.01)
print(result.diagnostics['n_trimmed'])
```

### 4. Single Specification

```python
# WRONG: Report only one learner
result = estimate_plr(..., ml_l='lasso')

# CORRECT: Compare multiple specifications
comparison = compare_learners(..., learner_list=['lasso', 'rf', 'xgboost'])
```

### 5. Claiming Causality Without Justification

Always discuss:
1. Why unconfoundedness is plausible
2. What confounders are included
3. Potential omitted variables
4. Sensitivity to violations

---

## Reporting Standards

> **Detailed Reference**: `references/reporting_standards.md`
> **Template**: `assets/markdown/ddml_report.md`
> **LaTeX Table**: `assets/latex/ddml_table.tex`

### Minimum Reporting Requirements

1. **Model type**: PLR or IRM
2. **ML learners**: For each nuisance function
3. **Cross-fitting**: K folds, n repetitions
4. **Point estimate**: With SE and CI
5. **Sensitivity**: Range across specifications

### Example Table

```
| Specification | Effect | SE | 95% CI |
|---------------|--------|-----|--------|
| (1) Lasso | 0.082*** | 0.008 | [0.066, 0.098] |
| (2) RF | 0.079*** | 0.009 | [0.061, 0.097] |
| (3) XGBoost | 0.081*** | 0.008 | [0.065, 0.097] |
```

---

## DoubleML Package Integration

For production use, consider the `doubleml` package:

```python
import doubleml as dml
from doubleml import DoubleMLData, DoubleMLPLR

# Prepare data
dml_data = DoubleMLData(df, y_col='outcome', d_cols='treatment', x_cols=controls)

# Estimate
dml_plr = DoubleMLPLR(dml_data, ml_l=LassoCV(), ml_m=LassoCV(), n_folds=5)
dml_plr.fit()

print(dml_plr.summary)
```

**Documentation**: https://docs.doubleml.org/

---

## Examples

### Example 1: Returns to Education

```python
# High-dimensional controls for education-wage analysis
result = run_full_ddml_analysis(
    data=df,
    outcome="log_wage",
    treatment="years_education",
    controls=['age', 'age_sq', 'female', 'married', 'region_*', 'industry_*',
              'parents_education', 'test_score', 'family_income']
)

print(result.summary_table)
```

### Example 2: Job Training Program (Binary Treatment)

```python
# Compare PLR and IRM for binary treatment
result_plr = estimate_plr(data, 'earnings', 'training', controls)
result_irm = estimate_irm(data, 'earnings', 'training', controls)

print(f"PLR (constant effect): {result_plr.effect:.2f}")
print(f"IRM (heterogeneous): {result_irm.effect:.2f}")
```

---

## Mathematical Appendix

### Neyman-Orthogonal Score (PLR)

$$
\psi^{PLR}(W; \theta, \ell, m) = (Y - \ell(X) - \theta(D - m(X)))(D - m(X))
$$

Setting $E[\psi] = 0$:
$$
\hat{\theta} = \frac{\sum_i (Y_i - \hat{\ell}(X_i))(D_i - \hat{m}(X_i))}{\sum_i (D_i - \hat{m}(X_i))^2}
$$

### Asymptotic Distribution

Under regularity conditions:
$$
\sqrt{n}(\hat{\theta} - \theta_0) \xrightarrow{d} N(0, \sigma^2)
$$

Where:
$$
\sigma^2 = \frac{E[\psi^2]}{(E[\partial_\theta \psi])^2}
$$

---

## References

### Seminal Papers
1. Chernozhukov, V., et al. (2018). Double/Debiased Machine Learning for Treatment and Structural Parameters. *The Econometrics Journal*, 21(1), C1-C68.
2. Chernozhukov, V., et al. (2017). Double/Debiased/Neyman Machine Learning of Treatment Effects. *AER P&P*, 107(5), 261-265.

### Foundational High-Dimensional Papers
3. Belloni, A., Chernozhukov, V., & Hansen, C. (2011). Inference on Treatment Effects after Selection Among High-Dimensional Controls. *Review of Economic Studies*, 81(2), 608-650. [1,498 citations]
4. Belloni, A., et al. (2013). High-Dimensional Methods and Inference on Structural and Treatment Effects. *Journal of Economic Perspectives*, 28(2), 29-50. [678 citations]

### Extensions
5. Chernozhukov, V., et al. (2022). Locally Robust Semiparametric Estimation. *Econometrica*, 90(4), 1501-1535.
6. Semenova, V., & Chernozhukov, V. (2021). Debiased Machine Learning of CATE. *The Econometrics Journal*, 24(2), 264-289.
7. Chernozhukov, V., et al. (2017). Generic Machine Learning Inference on Heterogeneous Treatment Effects. *arXiv*. [210 citations]

### Software
5. DoubleML (Python/R): https://docs.doubleml.org/
6. EconML (Python): https://econml.azurewebsites.net/

---

## Related Skills

| Skill | When to Use Instead |
|-------|---------------------|
| `estimator-ols` | Low-dimensional, simple relationships |
| `estimator-psm` | Explicit propensity matching desired |
| `estimator-iv` | Unconfoundedness violated, instrument available |
| `estimator-did` | Panel data, staggered treatment |
| `causal-forest` | Focus on treatment effect heterogeneity |
