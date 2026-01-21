---
name: causal-ddml
description: Double/Debiased Machine Learning for causal inference with high-dimensional controls. Use when estimating causal effects with many covariates, nonlinear confounding, or need valid inference with ML. Provides PLR, IRM models via doubleml package.
license: MIT
metadata:
    skill-author: Causal-ML-Skills
---

# Double/Debiased Machine Learning (DDML)

## Overview

Double/Debiased Machine Learning (DDML) estimates causal effects by combining machine learning methods for nuisance parameter estimation with debiased/orthogonal score functions that yield valid statistical inference. The method uses sample splitting (cross-fitting) to avoid overfitting bias while leveraging the flexibility of modern ML methods.

**Key Innovation**: DDML uses Neyman-orthogonal moment conditions that are insensitive (to first order) to estimation errors in nuisance functions, enabling the use of regularized/ML estimators without compromising valid inference.

**Primary Reference**: Chernozhukov, V., et al. (2018). Double/Debiased Machine Learning for Treatment and Structural Parameters. *The Econometrics Journal*, 21(1), C1-C68.

## When to Use This Skill

This skill should be used when:

- High-dimensional controls (many covariates, p approaching n)
- Complex, nonlinear confounding relationships
- Need valid statistical inference with ML adjustment
- Partially Linear Regression (PLR) model is appropriate
- Interactive Regression Model (IRM) for binary treatment
- LATE estimation with instrumental variables (IIVM, PLIV)

**Do NOT use when:**
- Low-dimensional, simple relationships (use OLS)
- Small samples (n < 100) - cross-fitting unreliable
- Unconfoundedness clearly violated (use IV or DID)
- Focus is on heterogeneous treatment effects (use causal-forest)

## Quick Start Guide

### Basic PLR with doubleml

```python
import numpy as np
import pandas as pd
from sklearn.linear_model import LassoCV
from sklearn.ensemble import RandomForestRegressor
import doubleml as dml
from doubleml import DoubleMLData, DoubleMLPLR

# Example: Effect of education on wages with many controls
# Treatment: years_education (continuous)
# Outcome: log_wage
# Controls: age, experience, demographics, industry dummies, etc.

# Load data
df = pd.read_csv('data.csv')

# Define variables
outcome = 'log_wage'
treatment = 'years_education'
controls = ['age', 'age_sq', 'female', 'married', 'experience',
            'experience_sq', 'region_1', 'region_2', 'region_3']

# Create DoubleML data object
dml_data = DoubleMLData(
    df,
    y_col=outcome,
    d_cols=treatment,
    x_cols=controls
)

# Define ML learners for nuisance functions
# ml_l: E[Y|X] - outcome regression
# ml_m: E[D|X] - treatment regression
ml_l = LassoCV(cv=5, n_alphas=50, max_iter=10000)
ml_m = LassoCV(cv=5, n_alphas=50, max_iter=10000)

# Estimate PLR model
dml_plr = DoubleMLPLR(
    dml_data,
    ml_l=ml_l,
    ml_m=ml_m,
    n_folds=5,           # Number of cross-fitting folds
    n_rep=1,             # Number of repetitions
    score='partialling out'  # Score function
)

# Fit the model
dml_plr.fit()

# Results
print(dml_plr.summary)

# Access specific values
print(f"\nEffect estimate: {dml_plr.coef[0]:.4f}")
print(f"Standard error: {dml_plr.se[0]:.4f}")
print(f"t-statistic: {dml_plr.t_stat[0]:.4f}")
print(f"p-value: {dml_plr.pval[0]:.4f}")

# 95% Confidence interval
ci = dml_plr.confint(level=0.95)
print(f"95% CI: [{ci.iloc[0, 0]:.4f}, {ci.iloc[0, 1]:.4f}]")
```

### IRM for Binary Treatment

```python
from sklearn.linear_model import LogisticRegressionCV
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from doubleml import DoubleMLIRM

# Example: Effect of job training on earnings
# Treatment: training (binary 0/1)
# Outcome: earnings

df = pd.read_csv('data.csv')

outcome = 'earnings'
treatment = 'training'
controls = ['age', 'education', 'prior_earnings', 'married', 'children']

# Create DoubleML data
dml_data = DoubleMLData(
    df,
    y_col=outcome,
    d_cols=treatment,
    x_cols=controls
)

# For IRM:
# ml_g: E[Y|D,X] - outcome model (conditional on treatment)
# ml_m: P(D=1|X) - propensity score model
ml_g = RandomForestRegressor(n_estimators=200, max_depth=5, n_jobs=-1)
ml_m = LogisticRegressionCV(cv=5, max_iter=1000)

# Estimate IRM
dml_irm = DoubleMLIRM(
    dml_data,
    ml_g=ml_g,
    ml_m=ml_m,
    n_folds=5,
    n_rep=1,
    score='ATE',           # 'ATE' or 'ATTE'
    trimming_threshold=0.01  # Trim extreme propensities
)

dml_irm.fit()
print(dml_irm.summary)

# Check propensity score overlap
# Access cross-fitted propensity scores
prop_scores = dml_irm.nuisance_estimates['ml_m'][0].reshape(-1)
print(f"\nPropensity score range: [{prop_scores.min():.4f}, {prop_scores.max():.4f}]")
print(f"Observations trimmed: {(prop_scores < 0.01).sum() + (prop_scores > 0.99).sum()}")
```

### Comparing Multiple Learners (Sensitivity Analysis)

```python
from sklearn.linear_model import LassoCV, RidgeCV, ElasticNetCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from doubleml import DoubleMLPLR, DoubleMLData
import pandas as pd

# Define data
dml_data = DoubleMLData(df, y_col='outcome', d_cols='treatment', x_cols=controls)

# Learners to compare
learners = {
    'Lasso': (LassoCV(cv=5), LassoCV(cv=5)),
    'Ridge': (RidgeCV(cv=5), RidgeCV(cv=5)),
    'Elastic Net': (ElasticNetCV(cv=5, l1_ratio=0.5), ElasticNetCV(cv=5, l1_ratio=0.5)),
    'Random Forest': (RandomForestRegressor(n_estimators=100, max_depth=5),
                      RandomForestRegressor(n_estimators=100, max_depth=5)),
}

results = []
for name, (ml_l, ml_m) in learners.items():
    model = DoubleMLPLR(dml_data, ml_l=ml_l, ml_m=ml_m, n_folds=5)
    model.fit()

    results.append({
        'Learner': name,
        'Effect': model.coef[0],
        'SE': model.se[0],
        'CI_lower': model.confint().iloc[0, 0],
        'CI_upper': model.confint().iloc[0, 1],
        'p_value': model.pval[0]
    })

results_df = pd.DataFrame(results)
print("\n" + "="*70)
print("SENSITIVITY ANALYSIS: Comparing ML Learners")
print("="*70)
print(results_df.to_string(index=False))

# Check robustness
effect_range = results_df['Effect'].max() - results_df['Effect'].min()
mean_effect = results_df['Effect'].mean()
print(f"\nEffect range: {effect_range:.4f}")
print(f"Mean effect: {mean_effect:.4f}")
print(f"Coefficient of variation: {results_df['Effect'].std() / abs(mean_effect):.2%}")
```

### IV Estimation with DDML (PLIV)

```python
from doubleml import DoubleMLPLIV, DoubleMLData

# Example: Effect of education on wages using distance to college as IV
df = pd.read_csv('data.csv')

# Create data with instrument
dml_data = DoubleMLData(
    df,
    y_col='log_wage',
    d_cols='years_education',  # Endogenous
    x_cols=controls,
    z_cols='distance_college'  # Instrument
)

# PLIV model
ml_l = LassoCV(cv=5)
ml_m = LassoCV(cv=5)
ml_r = LassoCV(cv=5)  # For instrument regression

dml_pliv = DoubleMLPLIV(
    dml_data,
    ml_l=ml_l,
    ml_m=ml_m,
    ml_r=ml_r,
    n_folds=5
)

dml_pliv.fit()
print(dml_pliv.summary)
```

### Complete Analysis with Diagnostics

```python
import numpy as np
import pandas as pd
from sklearn.linear_model import LassoCV
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import r2_score
import doubleml as dml
from doubleml import DoubleMLData, DoubleMLPLR
import matplotlib.pyplot as plt

def run_ddml_analysis(df, outcome, treatment, controls, n_folds=5, n_rep=1):
    """
    Run complete DDML analysis with diagnostics.
    """
    # Create data object
    dml_data = DoubleMLData(df, y_col=outcome, d_cols=treatment, x_cols=controls)

    # Define learners
    ml_l = LassoCV(cv=5, n_alphas=50, max_iter=10000)
    ml_m = LassoCV(cv=5, n_alphas=50, max_iter=10000)

    # Estimate
    model = DoubleMLPLR(dml_data, ml_l=ml_l, ml_m=ml_m, n_folds=n_folds, n_rep=n_rep)
    model.fit()

    # Diagnostics: Nuisance model performance
    X = df[controls].values
    Y = df[outcome].values
    D = df[treatment].values

    # Cross-validated R2 for outcome model
    y_pred = cross_val_predict(LassoCV(cv=5), X, Y, cv=5)
    r2_outcome = r2_score(Y, y_pred)

    # Cross-validated R2 for treatment model
    d_pred = cross_val_predict(LassoCV(cv=5), X, D, cv=5)
    r2_treatment = r2_score(D, d_pred)

    print("="*60)
    print("DDML ANALYSIS RESULTS")
    print("="*60)
    print(f"\nModel: Partially Linear Regression (PLR)")
    print(f"Sample size: {len(df)}")
    print(f"Controls: {len(controls)}")
    print(f"Cross-fitting: {n_folds} folds, {n_rep} repetitions")

    print("\n--- Causal Effect ---")
    print(f"Effect: {model.coef[0]:.4f}")
    print(f"SE: {model.se[0]:.4f}")
    ci = model.confint()
    print(f"95% CI: [{ci.iloc[0,0]:.4f}, {ci.iloc[0,1]:.4f}]")
    print(f"t-stat: {model.t_stat[0]:.4f}")
    print(f"p-value: {model.pval[0]:.4f}")

    print("\n--- Nuisance Model Quality ---")
    print(f"R2 (outcome ~ X): {r2_outcome:.4f}")
    print(f"R2 (treatment ~ X): {r2_treatment:.4f}")

    if r2_outcome < 0.1:
        print("WARNING: Low outcome model fit - consider different learner")
    if r2_treatment < 0.1:
        print("WARNING: Low treatment model fit - check for weak selection")

    print("="*60)

    return model, {'r2_outcome': r2_outcome, 'r2_treatment': r2_treatment}

# Usage
model, diagnostics = run_ddml_analysis(df, 'log_wage', 'years_education', controls)
```

## Core Capabilities

### 1. DDML Models

| Model | Full Name | Treatment | Estimand | Use Case |
|-------|-----------|-----------|----------|----------|
| **PLR** | Partially Linear Regression | Any | ATE | Continuous or binary treatment, constant effect |
| **IRM** | Interactive Regression Model | Binary | ATE, ATTE | Binary treatment, heterogeneous effects |
| **IIVM** | Interactive IV Model | Binary + IV | LATE | Binary endogenous treatment |
| **PLIV** | Partially Linear IV | Continuous + IV | ATE | Continuous endogenous treatment |

### 2. Cross-Fitting

**K-Fold Sample Splitting:**
```
For each fold k:
  1. Train nuisance models on data excluding fold k
  2. Predict on fold k (out-of-sample)
  3. Compute orthogonal score on fold k
```

**Choosing K:**

| Sample Size | Recommended K | Min observations per fold |
|-------------|---------------|---------------------------|
| n < 500 | 2-3 | ~150 |
| 500-2000 | 5 | ~100 |
| n > 2000 | 5-10 | ~200 |

### 3. Nuisance Function Learners

| Learner | Best For | doubleml Syntax |
|---------|----------|-----------------|
| Lasso | Sparse, high-dimensional | `LassoCV()` |
| Ridge | Dense, correlated features | `RidgeCV()` |
| Elastic Net | Mixed sparsity | `ElasticNetCV()` |
| Random Forest | Nonlinear relationships | `RandomForestRegressor()` |
| Gradient Boosting | Complex interactions | `GradientBoostingRegressor()` |

## Common Workflows

### Workflow 1: Standard DDML Analysis

```
1. Data Preparation
   ├── Define outcome, treatment, controls
   ├── Check for missing data
   └── Consider variable transformations

2. Model Selection
   ├── PLR for continuous treatment
   ├── IRM for binary treatment
   └── PLIV/IIVM if instruments needed

3. Learner Selection
   ├── Start with Lasso (interpretable)
   ├── Try Random Forest (flexible)
   └── Compare multiple learners

4. Estimation
   ├── Set n_folds (typically 5)
   ├── Consider n_rep > 1 for stability
   └── Fit model

5. Diagnostics
   ├── Check nuisance model R2
   ├── Verify propensity overlap (IRM)
   └── Compare across learners

6. Reporting
   ├── Point estimate with CI
   ├── Sensitivity across specifications
   └── Discuss identification assumptions
```

### Workflow 2: Robustness Analysis

```
1. Baseline Specification
   └── Lasso for both nuisance functions

2. Learner Sensitivity
   ├── Ridge
   ├── Random Forest
   ├── Gradient Boosting
   └── Compare effect estimates

3. Sample Sensitivity
   ├── Vary trimming threshold (IRM)
   ├── Vary number of folds
   └── Multiple repetitions

4. Report
   └── Effect range across specifications
```

## Best Practices

### Model Selection

1. **Start simple**: Lasso provides interpretable baseline
2. **Compare learners**: Report sensitivity to ML specification
3. **Match model to treatment type**: PLR for continuous, IRM for binary
4. **Check rate conditions**: Both nuisance models need reasonable fit

### Cross-Fitting

1. **Use K >= 5 folds** for samples > 500
2. **Set n_rep > 1** for inference stability (e.g., n_rep=10)
3. **Don't peek**: Never use in-sample predictions

### Inference

1. **Report CIs** not just point estimates
2. **Compare across learners** to assess sensitivity
3. **Discuss identification**: Unconfoundedness is assumed, not tested

### Diagnostics

1. **Check nuisance R2**: Low R2 suggests weak selection
2. **Examine propensity overlap** (IRM): Trim extreme values
3. **Look for effect heterogeneity**: Consider causal forest if suspected

## Reference Documentation

### references/identification_assumptions.md
- Neyman orthogonality formal definition
- Cross-fitting theory
- Rate conditions for valid inference
- When assumptions fail

### references/estimation_methods.md
- PLR score function derivation
- IRM (AIPW) score
- PLIV and IIVM for IV settings
- Asymptotic theory

### references/diagnostic_tests.md
- Cross-fit stability checks
- Nuisance model quality metrics
- Propensity score diagnostics
- Sensitivity analysis

### references/model_selection.md
- Learner selection guidance
- Hyperparameter tuning
- Ensemble approaches

### references/reporting_standards.md
- Table formatting
- Required elements
- LaTeX templates

### references/common_errors.md
- Not using cross-fitting
- Wrong model for treatment type
- Ignoring propensity overlap
- Single specification reporting

## Common Pitfalls to Avoid

1. **Not using cross-fitting**: In-sample predictions cause overfitting bias
2. **Using IRM with continuous treatment**: IRM is for binary treatment only
3. **Ignoring propensity overlap**: Extreme propensities inflate variance
4. **Single ML specification**: Always compare multiple learners
5. **Claiming causality without justification**: Discuss unconfoundedness plausibility
6. **Too few folds**: Use K >= 5 for reasonable samples
7. **Ignoring nuisance model quality**: Low R2 signals problems
8. **Not reporting sensitivity**: Effect should be stable across specifications
9. **Forgetting rate conditions**: Product of errors must decay fast enough
10. **Using DDML for small samples**: Cross-fitting unreliable for n < 100

## Troubleshooting

### Effect Estimate Varies Widely Across Learners

**Issue:** Sensitivity analysis shows large effect range

**Solutions:**
```python
# 1. Increase repetitions for stability
model = DoubleMLPLR(..., n_rep=10)

# 2. Use ensemble learners
from sklearn.ensemble import StackingRegressor

# 3. Check if data supports identification
# Large variation may indicate weak identification
```

### Low Nuisance Model R2

**Issue:** Outcome or treatment model has poor fit

**Solutions:**
```python
# 1. Try more flexible learners
ml_l = RandomForestRegressor(n_estimators=500, max_depth=10)

# 2. Add polynomial/interaction terms
from sklearn.preprocessing import PolynomialFeatures

# 3. Check for missing important controls
```

### Extreme Propensity Scores (IRM)

**Issue:** Many observations with propensity near 0 or 1

**Solutions:**
```python
# 1. Increase trimming threshold
model = DoubleMLIRM(..., trimming_threshold=0.05)

# 2. Check overlap visually
import matplotlib.pyplot as plt
plt.hist(prop_scores[treatment==0], alpha=0.5, label='Control')
plt.hist(prop_scores[treatment==1], alpha=0.5, label='Treated')

# 3. Consider different estimand (ATTE instead of ATE)
```

### Import Error for doubleml

**Solution:**
```bash
pip install doubleml
```

## Additional Resources

### Official Documentation
- DoubleML: https://docs.doubleml.org/
- EconML: https://econml.azurewebsites.net/

### Key Papers
- Chernozhukov et al. (2018): "Double/Debiased Machine Learning for Treatment and Structural Parameters"
- Chernozhukov et al. (2017): "Double/Debiased/Neyman Machine Learning of Treatment Effects"
- Belloni et al. (2014): "Inference on Treatment Effects after Selection Among High-Dimensional Controls"

### Textbooks
- Chernozhukov et al. (2024): *Causal Inference for Statistics, Social, and Biomedical Sciences*
- Athey & Imbens (2019): "Machine Learning Methods That Economists Should Know About"

## Installation

```bash
# Core package
pip install doubleml

# ML learners
pip install scikit-learn xgboost lightgbm

# Full installation
pip install doubleml scikit-learn xgboost lightgbm pandas numpy matplotlib
```

## Related Skills

| Skill | When to Use Instead |
|-------|---------------------|
| `estimator-psm` | Explicit propensity matching desired |
| `estimator-iv` | Unconfoundedness violated, have instrument |
| `estimator-did` | Panel data, staggered treatment |
| `causal-forest` | Focus on heterogeneous treatment effects |
| `panel-data-models` | Panel structure without high-dimensional controls |
