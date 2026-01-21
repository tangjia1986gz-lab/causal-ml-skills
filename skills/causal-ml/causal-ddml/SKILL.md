---
name: causal-ddml
description: Use when estimating causal effects with Double/Debiased Machine Learning. Triggers on DDML, double machine learning, debiased, cross-fitting, Chernozhukov, high-dimensional, partially linear, PLR, IRM, orthogonal score.
---

# Estimator: Double/Debiased Machine Learning (DDML)

> **Version**: 1.0.0 | **Type**: Estimator
> **Aliases**: DDML, DML, Double ML, Debiased ML, Orthogonal ML

## Overview

Double/Debiased Machine Learning (DDML) estimates causal effects by combining machine learning methods for nuisance parameter estimation with debiased/orthogonal score functions that yield valid statistical inference. The method uses sample splitting (cross-fitting) to avoid overfitting bias while leveraging the flexibility of modern ML methods.

**Key Innovation**: DDML uses Neyman-orthogonal moment conditions that are insensitive (to first order) to estimation errors in nuisance functions, enabling the use of regularized/ML estimators without compromising valid inference.

## When to Use

### Ideal Scenarios
- High-dimensional controls (more covariates than observations or many relevant controls)
- Complex, nonlinear relationships between treatment/outcome and confounders
- Need for valid confidence intervals despite using ML for covariate adjustment
- Partially linear models where treatment effect is of primary interest
- Settings where traditional parametric models may be misspecified

### Data Requirements
- [ ] Cross-sectional or panel data (with appropriate modifications)
- [ ] Sufficient sample size (typically n > 500 for stable cross-fitting)
- [ ] Treatment variable (binary or continuous)
- [ ] High-dimensional control variables (can include interactions, polynomials)
- [ ] No severe multicollinearity that breaks ML learners

### When NOT to Use
- Low-dimensional settings with few controls -> Consider `estimator-ols` or `estimator-psm`
- When ML flexibility is unnecessary -> Traditional regression suffices
- Very small samples (n < 100) -> Cross-fitting becomes unreliable
- Need for causal interpretation beyond treatment effect on treated
- Unconfoundedness assumption clearly violated -> Consider `estimator-iv` or `estimator-did`

## Identification Assumptions

| Assumption | Description | Testable? |
|------------|-------------|-----------|
| **Unconfoundedness** | Treatment is independent of potential outcomes conditional on X | No (fundamentally) |
| **Overlap/Positivity** | All units have positive probability of treatment | Yes (propensity scores) |
| **Correct Model Class** | True nuisance functions in model class or well-approximated | Partially (via cross-validation) |
| **Sufficient Sample Size** | n large enough for stable cross-fitting | Heuristic checks |

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

**Objective**: Prepare data and define DDML specification

**Inputs Required**:
```python
# Standard CausalInput structure for DDML
outcome = "y"                    # Outcome variable name
treatment = "d"                  # Treatment variable (binary or continuous)
controls = ["x1", "x2", ...,     # High-dimensional control variables
            "x1_sq", "x1_x2"]    # Can include polynomials/interactions
```

**Data Validation Checklist**:
- [ ] No missing values in key variables (DDML typically requires complete cases)
- [ ] Treatment has sufficient variation (for binary: not too imbalanced)
- [ ] Controls are scaled appropriately for ML learners
- [ ] Sample size adequate for K-fold splitting (n/K > 100 recommended)

**Data Structure Verification**:
```python
from ddml_estimator import create_ddml_data, validate_ddml_setup

# Check data structure
validation = validate_ddml_setup(
    data=df,
    outcome="y",
    treatment="d",
    controls=control_vars
)
print(validation)
```

### Phase 2: Model Selection (First-Stage Learners)

**Key Decision**: Choose ML methods for nuisance parameter estimation

| Nuisance Function | PLR Model | IRM Model |
|-------------------|-----------|-----------|
| $\ell_0(X) = E[Y|X]$ | Required | - |
| $m_0(X) = E[D|X]$ | Required | Required (propensity) |
| $g_0(d, X) = E[Y|D=d, X]$ | - | Required |

**Recommended Learners by Scenario**:

| Scenario | Recommended Learners | Reason |
|----------|---------------------|--------|
| Sparse effects | Lasso, Elastic Net | Variable selection |
| Complex nonlinear | Random Forest, XGBoost | Flexibility |
| Very high-dimensional | Lasso + RF ensemble | Robustness |
| Unknown structure | Multiple learners + comparison | Hedge against misspecification |

**Automatic Model Selection**:
```python
from ddml_estimator import select_first_stage_learners

# Auto-select best ML models via cross-validation
best_learners = select_first_stage_learners(
    X=df[control_vars],
    y=df['y'],
    d=df['d'],
    cv_folds=5
)

print(f"Best learner for E[Y|X]: {best_learners['ml_l']}")
print(f"Best learner for E[D|X]: {best_learners['ml_m']}")
```

### Phase 3: Cross-Fitting

**Why Cross-Fitting?**
Standard sample splitting wastes data. Cross-fitting (K-fold) uses all data for both training and prediction:

```
K-Fold Cross-Fitting (K=5):
+--------------------------------------------------+
| Fold 1: Train on [2,3,4,5] -> Predict on [1]     |
| Fold 2: Train on [1,3,4,5] -> Predict on [2]     |
| Fold 3: Train on [1,2,4,5] -> Predict on [3]     |
| Fold 4: Train on [1,2,3,5] -> Predict on [4]     |
| Fold 5: Train on [1,2,3,4] -> Predict on [5]     |
+--------------------------------------------------+
Result: Out-of-sample predictions for ALL observations
```

**Cross-Fitting Parameters**:
```python
n_folds = 5       # Number of folds (standard: 5)
n_rep = 1         # Number of repetitions (increase for stability)
```

### Phase 4: Main Estimation

**Model A: Partially Linear Regression (PLR)**

$$
Y = D \cdot \theta_0 + g_0(X) + \epsilon, \quad E[\epsilon|D,X] = 0
$$
$$
D = m_0(X) + V, \quad E[V|X] = 0
$$

Where:
- $\theta_0$: **Treatment effect of interest** (constant across X)
- $g_0(X)$: Nuisance function (flexible function of controls)
- $m_0(X)$: Propensity/conditional mean of treatment

**Neyman-Orthogonal Score for PLR**:
$$
\psi(W; \theta, \eta) = (Y - \ell_0(X) - \theta(D - m_0(X)))(D - m_0(X))
$$

```python
from ddml_estimator import estimate_plr

# Partially Linear Model
result_plr = estimate_plr(
    data=df,
    outcome="y",
    treatment="d",
    controls=control_vars,
    ml_l='lasso',      # Learner for E[Y|X]
    ml_m='lasso',      # Learner for E[D|X]
    n_folds=5
)

print(result_plr.summary_table)
```

**Model B: Interactive Regression Model (IRM)**

For binary treatment, allows heterogeneous effects:

$$
Y = g_0(D, X) + \epsilon, \quad E[\epsilon|D,X] = 0
$$
$$
D = m_0(X) + V, \quad E[V|X] = 0
$$

**ATE via IRM**:
$$
\theta_0 = E[g_0(1, X) - g_0(0, X)]
$$

```python
from ddml_estimator import estimate_irm

# Interactive Regression Model (binary treatment)
result_irm = estimate_irm(
    data=df,
    outcome="y",
    treatment="d",       # Must be binary (0/1)
    controls=control_vars,
    ml_g='random_forest',  # Learner for E[Y|D,X]
    ml_m='logistic_lasso', # Learner for P(D=1|X)
    n_folds=5
)

print(result_irm.summary_table)
```

### Phase 5: Inference

**DDML Provides Valid Inference Because**:
1. Neyman orthogonality removes first-order bias from nuisance estimation
2. Cross-fitting prevents overfitting bias
3. Under regularity conditions: $\sqrt{n}(\hat{\theta} - \theta_0) \to N(0, V)$

**Standard Errors**:
```python
# Results include valid standard errors
print(f"Effect: {result.effect:.4f}")
print(f"SE: {result.se:.4f}")
print(f"95% CI: [{result.ci_lower:.4f}, {result.ci_upper:.4f}]")
print(f"P-value: {result.p_value:.4f}")
```

**Multiple Cross-Fitting Iterations**:
For more stable inference, run multiple repetitions:
```python
result = estimate_plr(
    data=df, outcome="y", treatment="d", controls=controls,
    ml_l='lasso', ml_m='lasso',
    n_folds=5,
    n_rep=10  # Average over 10 random fold assignments
)
```

### Phase 6: Robustness Checks

| Check | Purpose | Implementation |
|-------|---------|----------------|
| Multiple Learners | Test sensitivity to ML choice | `compare_learners()` |
| Different K | Sensitivity to cross-fitting | Vary `n_folds` |
| Trimming | Address extreme propensities | `trimming_threshold` |
| Subgroup Analysis | Check heterogeneity | Run on subsamples |

**Compare Multiple Learner Specifications**:
```python
from ddml_estimator import compare_learners

# Compare different ML specifications
comparison = compare_learners(
    data=df,
    outcome="y",
    treatment="d",
    controls=control_vars,
    learner_list=['lasso', 'ridge', 'random_forest', 'xgboost']
)

print(comparison.summary_table)
```

**Sensitivity to ML Tuning**:
```python
from ddml_estimator import sensitivity_analysis_ddml

# Check robustness across learner choices
sensitivity = sensitivity_analysis_ddml(
    results=comparison,
    ml_specs=['lasso', 'ridge', 'random_forest', 'xgboost']
)

print(f"Effect range: [{sensitivity['min_effect']:.4f}, {sensitivity['max_effect']:.4f}]")
print(f"All estimates significant: {sensitivity['all_significant']}")
```

---

## PLR vs IRM Model Comparison

| Aspect | PLR | IRM |
|--------|-----|-----|
| Treatment Type | Any (continuous/binary) | Binary only |
| Effect Assumption | Constant treatment effect | Allows heterogeneity |
| Primary Estimand | ATE | ATE, ATT (with modifications) |
| Nuisance Functions | E[Y\|X], E[D\|X] | E[Y\|D,X], P(D=1\|X) |
| Computational Cost | Lower | Higher (more nuisance) |
| When to Use | Continuous treatment, constant effects | Binary treatment, potential heterogeneity |

---

## Learner Selection Guidance

### Available Learners

| Learner | Key | Good For | Tuning |
|---------|-----|----------|--------|
| Lasso | `'lasso'` | Sparse high-dimensional | CV for lambda |
| Ridge | `'ridge'` | Dense effects | CV for lambda |
| Elastic Net | `'elastic_net'` | Mix of sparse/dense | CV for alpha, lambda |
| Random Forest | `'random_forest'` | Nonlinear, interactions | n_estimators, max_depth |
| XGBoost | `'xgboost'` | Complex patterns | learning_rate, max_depth |
| LightGBM | `'lightgbm'` | Large data, speed | num_leaves, learning_rate |
| Neural Net | `'mlp'` | Very complex | architecture, regularization |

### Selection Strategy

```python
from ddml_estimator import select_first_stage_learners

# 1. Let cross-validation decide
auto_learners = select_first_stage_learners(X, y, d)

# 2. Use domain knowledge
# - Economic data with few key variables: Lasso
# - Survey data with many interactions: Random Forest
# - Very large n with complex patterns: XGBoost/LightGBM

# 3. Run multiple and compare (RECOMMENDED)
results = compare_learners(data, outcome, treatment, controls,
                          learner_list=['lasso', 'random_forest', 'xgboost'])
```

---

## Common Mistakes

### 1. Not Using Cross-Fitting

**Mistake**: Fitting nuisance functions on the same data used for inference.

**Why it's wrong**: Creates overfitting bias that invalidates standard errors.

**Correct approach**:
```python
# WRONG: Manual implementation without cross-fitting
from sklearn.linear_model import Lasso
model = Lasso().fit(X, y)
residuals = y - model.predict(X)  # Overfitted residuals!

# CORRECT: Use DDML with cross-fitting
result = estimate_plr(
    data=df, outcome="y", treatment="d", controls=controls,
    n_folds=5  # Cross-fitting ensures out-of-sample predictions
)
```

### 2. Wrong Model for Treatment Type

**Mistake**: Using IRM with continuous treatment.

**Why it's wrong**: IRM is designed for binary treatment only.

**Correct approach**:
```python
# WRONG: IRM with continuous treatment
result = estimate_irm(data, outcome, treatment_continuous, controls)

# CORRECT: Use PLR for continuous treatment
result = estimate_plr(data, outcome, treatment_continuous, controls)
```

### 3. Ignoring Propensity Score Overlap

**Mistake**: Not checking for extreme propensity scores that violate overlap.

**Why it's wrong**: Extreme propensities lead to unstable estimates.

**Correct approach**:
```python
from ddml_estimator import estimate_irm

# Check propensity distribution
result = estimate_irm(
    data=df, outcome="y", treatment="d", controls=controls,
    trimming_threshold=0.01  # Trim extreme propensities
)

# Review propensity distribution in diagnostics
print(result.diagnostics['propensity_summary'])
```

### 4. Too Few Folds with Small Sample

**Mistake**: Using K=10 folds with n=200, leaving only 20 observations per fold.

**Why it's wrong**: Small fold size leads to high variance in nuisance estimates.

**Correct approach**:
```python
# Adjust folds based on sample size
n = len(df)
if n < 500:
    n_folds = 2
elif n < 1000:
    n_folds = 3
else:
    n_folds = 5

result = estimate_plr(data=df, ..., n_folds=n_folds)
```

### 5. Claiming Causal Effect Without Unconfoundedness

**Mistake**: Interpreting DDML estimates as causal without justifying unconfoundedness.

**Why it's wrong**: DDML does not solve omitted variable bias - it only handles high-dimensional observed confounders.

**Correct approach**:
```python
# ALWAYS discuss identification
"""
Interpretation: Under the assumption that all relevant confounders
are included in X (unconfoundedness), the DDML estimate of {effect}
represents the causal effect. However, unobserved confounding could
bias these results. Consider:
1. DAG/causal diagram to justify controls
2. Sensitivity analysis for unmeasured confounding
3. Alternative identification strategies (IV, DID) if available
"""
```

---

## Examples

### Example 1: Returns to Education with High-Dimensional Controls

**Research Question**: What is the causal effect of years of education on wages?

**Challenge**: Many potential confounders (family background, ability, location, etc.)

```python
import pandas as pd
from ddml_estimator import run_full_ddml_analysis, compare_learners

# Load data (e.g., CPS or similar)
data = pd.read_csv("wages_education.csv")

# Define high-dimensional controls
# Include polynomials, interactions for flexibility
controls = [
    'age', 'age_sq', 'female', 'married',
    'black', 'hispanic', 'urban', 'south', 'west', 'midwest',
    'parents_education', 'n_siblings', 'birth_order',
    # Interactions
    'age_female', 'urban_education_region',
    # Additional proxies for ability
    'test_score', 'gpa_high_school'
]

# Run full DDML analysis
result = run_full_ddml_analysis(
    data=data,
    outcome="log_wage",
    treatment="years_education",
    controls=controls
)

# View main results
print(result.summary_table)

# Compare different ML specifications
comparison = compare_learners(
    data=data,
    outcome="log_wage",
    treatment="years_education",
    controls=controls,
    learner_list=['lasso', 'ridge', 'random_forest', 'xgboost']
)

print("\nSensitivity to ML Specification:")
print(comparison.summary_table)
```

**Output**:
```
### Double/Debiased ML Results (PLR Model)

| Variable | (1) Lasso | (2) RF | (3) XGBoost |
|:---------|:---------:|:------:|:-----------:|
| Treatment Effect | 0.082*** | 0.079*** | 0.081*** |
| | (0.008) | (0.009) | (0.008) |
| | | | |
| ML Learner (Y|X) | Lasso | RF | XGBoost |
| ML Learner (D|X) | Lasso | RF | XGBoost |
| Cross-fitting Folds | 5 | 5 | 5 |
| | | | |
| Observations | 15,000 | 15,000 | 15,000 |

*Notes: Robust standard errors. *** p<0.01, ** p<0.05, * p<0.1*

INTERPRETATION:
The DDML estimate suggests that each additional year of education
increases log wages by approximately 0.08 (8%). This estimate is
robust across different ML specifications (Lasso, RF, XGBoost),
providing evidence that the result is not sensitive to the choice
of first-stage learner.
```

### Example 2: Treatment Effect with Binary Treatment (IRM)

```python
from ddml_estimator import estimate_irm, estimate_plr

# Job training program evaluation
data = pd.read_csv("job_training.csv")

controls = [
    'age', 'age_sq', 'education', 'married', 'black', 'hispanic',
    'prior_earnings_1', 'prior_earnings_2', 'prior_earnings_3',
    'unemployed_prior', 'industry_dummies_*'
]

# Compare PLR (assumes constant effect) vs IRM (allows heterogeneity)
result_plr = estimate_plr(
    data=data,
    outcome="earnings",
    treatment="training",  # Binary: 0/1
    controls=controls,
    ml_l='random_forest',
    ml_m='logistic_lasso',
    n_folds=5
)

result_irm = estimate_irm(
    data=data,
    outcome="earnings",
    treatment="training",
    controls=controls,
    ml_g='random_forest',
    ml_m='logistic_lasso',
    n_folds=5
)

print("PLR (constant effect assumption):")
print(f"  ATE = {result_plr.effect:.2f} (SE = {result_plr.se:.2f})")

print("\nIRM (allows heterogeneous effects):")
print(f"  ATE = {result_irm.effect:.2f} (SE = {result_irm.se:.2f})")

# If estimates differ substantially, suggests treatment effect heterogeneity
```

---

## References

### Seminal Papers
- Chernozhukov, V., Chetverikov, D., Demirer, M., Duflo, E., Hansen, C., Newey, W., & Robins, J. (2018). Double/Debiased Machine Learning for Treatment and Structural Parameters. *The Econometrics Journal*, 21(1), C1-C68.
- Chernozhukov, V., et al. (2017). Double/Debiased/Neyman Machine Learning of Treatment Effects. *American Economic Review Papers & Proceedings*, 107(5), 261-265.

### Methodological Extensions
- Chernozhukov, V., et al. (2022). Locally Robust Semiparametric Estimation. *Econometrica*, 90(4), 1501-1535.
- Semenova, V., & Chernozhukov, V. (2021). Debiased Machine Learning of Conditional Average Treatment Effects and Other Causal Functions. *The Econometrics Journal*, 24(2), 264-289.
- Kennedy, E. H. (2022). Semiparametric Doubly Robust Targeted Double Machine Learning: A Review. arXiv:2203.06469.

### Textbook Treatments
- Chernozhukov, V. (2021). Applied Causal Inference Powered by ML and AI. MIT Course Notes.
- Athey, S., & Imbens, G. W. (2019). Machine Learning Methods Economists Should Know About. *Annual Review of Economics*, 11, 685-725.

### Software Documentation
- `doubleml` (Python/R): https://docs.doubleml.org/
- `econml` (Python): https://econml.azurewebsites.net/
- `DoubleML` (R): https://docs.doubleml.org/r/stable/

---

## Related Estimators

| Estimator | When to Use Instead |
|-----------|---------------------|
| `estimator-ols` | Low-dimensional settings, few controls |
| `estimator-psm` | Want explicit propensity score matching |
| `estimator-iv` | Unconfoundedness violated, instrument available |
| `causal-forest` | Focus on treatment effect heterogeneity |
| `estimator-did` | Panel data with time variation in treatment |

---

## Appendix: Mathematical Details

### Neyman Orthogonality

A score function $\psi(W; \theta, \eta)$ is Neyman-orthogonal if:
$$
\partial_\eta E[\psi(W; \theta_0, \eta)]|_{\eta=\eta_0} = 0
$$

This means the moment condition is locally insensitive to perturbations in nuisance parameters $\eta$.

### PLR Orthogonal Score

For the Partially Linear Model:
$$
\psi^{PLR}(W; \theta, \ell, m) = (Y - \ell(X) - \theta(D - m(X)))(D - m(X))
$$

Setting $E[\psi^{PLR}] = 0$ and solving:
$$
\theta_0 = \frac{E[(Y - \ell_0(X))(D - m_0(X))]}{E[(D - m_0(X))^2]}
$$

### Cross-Fitting Algorithm

```
Input: Data (Y, D, X), n_folds K, ML learners for ell, m
Output: Estimate theta_hat, standard error se

1. Split data into K folds: I_1, ..., I_K
2. For k = 1, ..., K:
   a. Train ell_k on data excluding I_k
   b. Train m_k on data excluding I_k
   c. For i in I_k:
      - Compute V_i = D_i - m_k(X_i)
      - Compute U_i = Y_i - ell_k(X_i)
3. Compute theta_hat = sum(U_i * V_i) / sum(V_i^2)
4. Compute residuals: psi_i = (U_i - theta_hat * V_i) * V_i
5. Estimate variance: V_hat = mean(psi_i^2) / mean(V_i^2)^2
6. Return theta_hat, se = sqrt(V_hat / n)
```

### Asymptotic Distribution

Under regularity conditions:
$$
\sqrt{n}(\hat{\theta} - \theta_0) \xrightarrow{d} N(0, \sigma^2)
$$

Where:
$$
\sigma^2 = \frac{E[\psi^2]}{(E[\partial_\theta \psi])^2} = \frac{E[(U - \theta_0 V)^2 V^2]}{(E[V^2])^2}
$$

### Rate Requirements

For valid inference, nuisance function estimates must satisfy:
$$
\|\hat{\ell} - \ell_0\|_2 \cdot \|\hat{m} - m_0\|_2 = o_P(n^{-1/2})
$$

This is achievable with many ML methods under appropriate sparsity or smoothness conditions.
