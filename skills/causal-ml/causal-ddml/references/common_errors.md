# DDML Common Errors

> Pitfalls and mistakes to avoid in Double/Debiased Machine Learning

## Overview

This document catalogs common errors in DDML implementation, interpretation, and reporting, with corrective guidance for each.

---

## 1. Sample Splitting Errors

### 1.1 Not Using Cross-Fitting

**Error**: Fitting nuisance functions on the same data used for inference.

```python
# WRONG
from sklearn.linear_model import Lasso

model_y = Lasso().fit(X, y)
model_d = Lasso().fit(X, d)

y_resid = y - model_y.predict(X)  # Overfitted!
d_resid = d - model_d.predict(X)  # Overfitted!

theta = np.sum(y_resid * d_resid) / np.sum(d_resid**2)  # Biased!
```

**Why It's Wrong**:
- Residuals are systematically too small (overfitting)
- Creates bias in treatment effect estimate
- Invalidates standard errors

**Correct Approach**:
```python
# CORRECT: Use cross-fitting
from sklearn.model_selection import cross_val_predict

y_pred_cv = cross_val_predict(Lasso(), X, y, cv=5)
d_pred_cv = cross_val_predict(Lasso(), X, d, cv=5)

y_resid = y - y_pred_cv  # Out-of-sample residuals
d_resid = d - d_pred_cv  # Out-of-sample residuals

theta = np.sum(y_resid * d_resid) / np.sum(d_resid**2)  # Unbiased!
```

### 1.2 Data Leakage in Fold Construction

**Error**: Information leaking between training and prediction folds.

```python
# WRONG: Preprocessing before splitting
X_scaled = StandardScaler().fit_transform(X)  # Uses all data!
# Then doing cross-fitting on X_scaled leaks info

# WRONG: Feature selection before splitting
selected_features = SelectKBest(k=20).fit_transform(X, y)  # Uses all data!
```

**Correct Approach**:
```python
# CORRECT: Preprocessing within each fold
from sklearn.pipeline import Pipeline

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', LassoCV())
])

# Pipeline ensures scaling is fit only on training fold
y_pred_cv = cross_val_predict(pipeline, X, y, cv=5)
```

### 1.3 Too Few or Too Many Folds

**Error**: Using inappropriate number of folds for sample size.

```python
# WRONG: K=10 with n=200 -> only 20 obs per fold
result = estimate_plr(data, outcome, treatment, controls, n_folds=10)

# WRONG: K=2 with n=50000 -> wasting data efficiency
result = estimate_plr(data, outcome, treatment, controls, n_folds=2)
```

**Guidelines**:
| Sample Size | Recommended K |
|-------------|---------------|
| n < 500 | 2-3 |
| 500-2000 | 5 |
| n > 2000 | 5-10 |

**Rule of thumb**: n/K > 100 for stable nuisance estimation.

---

## 2. Nuisance Function Errors

### 2.1 Overfitting Nuisance Functions

**Error**: Using overly complex learners that overfit nuisance functions.

```python
# POTENTIALLY PROBLEMATIC: Very deep forest with small n
rf = RandomForestRegressor(
    n_estimators=1000,
    max_depth=None,  # No limit!
    min_samples_leaf=1  # Can overfit to single obs
)
```

**Correct Approach**:
```python
# BETTER: Regularized tree ensemble
rf = RandomForestRegressor(
    n_estimators=200,
    max_depth=10,        # Limit depth
    min_samples_leaf=5,  # Require more samples per leaf
    max_features='sqrt'  # Random feature subsets
)
```

### 2.2 Underfitting Nuisance Functions

**Error**: Using too simple models when relationship is complex.

```python
# WRONG: Linear model when true relationship is highly nonlinear
# If g(X) = X1^2 + sin(X2) + X1*X3, Lasso will underfit

model = Lasso()  # Assumes linearity in X
```

**Correct Approach**:
```python
# Compare multiple learners
learners = ['lasso', 'random_forest', 'xgboost']
comparison = compare_learners(data, outcome, treatment, controls, learners)

# Check CV performance
print(comparison.sensitivity)
```

### 2.3 Wrong Learner for Treatment Model

**Error**: Using regression for binary treatment propensity.

```python
# WRONG: Regression for binary treatment
from sklearn.linear_model import Lasso
propensity_model = Lasso().fit(X, d)  # d is binary!
propensity = propensity_model.predict(X)  # Can be negative or > 1!
```

**Correct Approach**:
```python
# CORRECT: Classification for binary treatment
from sklearn.linear_model import LogisticRegressionCV
propensity_model = LogisticRegressionCV().fit(X, d)
propensity = propensity_model.predict_proba(X)[:, 1]  # In [0, 1]
```

---

## 3. Score Function Errors

### 3.1 Using Wrong Score

**Error**: Using non-orthogonal score function.

```python
# WRONG: Naive OLS residual approach (not orthogonal)
y_resid = y - model_y.predict(X)
theta = np.mean(y_resid * d) / np.mean(d**2)  # Not Neyman-orthogonal!
```

**Correct Approach**:
```python
# CORRECT: Use partialling-out score
y_resid = y - model_y.predict(X)
d_resid = d - model_d.predict(X)

# Neyman-orthogonal score
theta = np.sum(y_resid * d_resid) / np.sum(d_resid**2)
```

### 3.2 IRM Score for Continuous Treatment

**Error**: Using IRM/AIPW score with continuous treatment.

```python
# WRONG: IRM requires binary treatment
result = estimate_irm(
    data=df,
    treatment='continuous_treatment',  # ERROR: Not binary!
    ...
)
```

**Correct Approach**:
```python
# Check treatment type
if df['treatment'].nunique() > 2:
    # Use PLR for continuous treatment
    result = estimate_plr(data=df, treatment='continuous_treatment', ...)
else:
    # IRM for binary treatment
    result = estimate_irm(data=df, treatment='binary_treatment', ...)
```

### 3.3 Ignoring Score Variance in SE Calculation

**Error**: Using wrong variance formula for standard errors.

```python
# WRONG: Naive variance (ignores nuisance estimation)
se_wrong = np.std(y_resid * d_resid) / np.sqrt(n) / np.mean(d_resid**2)

# CORRECT: Influence function based variance
psi = (y_resid - theta * d_resid) * d_resid
var_theta = np.mean(psi**2) / (np.mean(d_resid**2)**2)
se_correct = np.sqrt(var_theta / n)
```

---

## 4. Overlap/Positivity Errors

### 4.1 Ignoring Extreme Propensity Scores

**Error**: Not checking or handling extreme propensities.

```python
# WRONG: Ignoring overlap issues
result = estimate_irm(data, outcome, treatment, controls)
# No check for propensity distribution!
```

**Symptoms**:
- Unstable estimates across specifications
- Very large standard errors
- Estimates sensitive to individual observations

**Correct Approach**:
```python
# Check propensity distribution
def check_overlap(propensity_scores, threshold=0.01):
    n_extreme = np.sum((propensity_scores < threshold) |
                       (propensity_scores > 1 - threshold))
    if n_extreme > 0:
        print(f"WARNING: {n_extreme} observations with extreme propensities")
        print("Consider trimming or using overlap weights")

# Use trimming
result = estimate_irm(
    data, outcome, treatment, controls,
    trimming_threshold=0.05  # Trim propensities outside [0.05, 0.95]
)
```

### 4.2 Over-Trimming

**Error**: Trimming too aggressively, losing too many observations.

```python
# WRONG: Aggressive trimming
result = estimate_irm(
    data, outcome, treatment, controls,
    trimming_threshold=0.2  # Drops all p < 0.2 or p > 0.8!
)
# May lose large fraction of sample
```

**Guidelines**:
| Threshold | When to Use |
|-----------|-------------|
| 0.01 | Default, minimal trimming |
| 0.05 | Moderate overlap concerns |
| 0.10 | Severe overlap issues (use with caution) |

---

## 5. Interpretation Errors

### 5.1 Claiming Causality Without Justification

**Error**: Interpreting DDML estimates as causal without discussing assumptions.

```python
# WRONG INTERPRETATION:
"""
The DDML estimate shows that education CAUSES a 8% increase in wages.
"""
```

**Correct Approach**:
```python
"""
CORRECT INTERPRETATION:
Under the assumption that all relevant confounders are captured by our
control variables (selection on observables), the DDML estimate of 0.08
represents the causal effect of education on log wages.

However, this interpretation relies on the untestable assumption of
unconfoundedness. Unobserved factors such as innate ability could
bias our estimates if correlated with both education and wages.

We partially address this concern by including test scores and family
background as proxies for ability, but cannot rule out residual
confounding.
"""
```

### 5.2 Confusing PLR and IRM Estimands

**Error**: Not understanding what each model estimates.

```python
# PLR with heterogeneous effects
# What PLR estimates: weighted average effect
# theta_PLR = E[theta(X) * Var(D|X)] / E[Var(D|X)]

# IRM estimates: population ATE
# theta_IRM = E[Y(1) - Y(0)]

# These can differ when effects are heterogeneous!
```

**Correct Approach**:
```python
# Compare PLR and IRM when effects may be heterogeneous
result_plr = estimate_plr(data, outcome, treatment, controls)
result_irm = estimate_irm(data, outcome, treatment, controls)

if abs(result_plr.effect - result_irm.effect) > 0.1 * abs(result_plr.effect):
    print("Substantial difference suggests treatment effect heterogeneity")
    print("IRM gives population ATE; PLR gives variance-weighted average")
```

### 5.3 Over-Interpreting ML Variable Importance

**Error**: Using nuisance model feature importance for causal conclusions.

```python
# WRONG: Feature importance from nuisance model is NOT causal
rf = RandomForestRegressor().fit(X, y)
importances = rf.feature_importances_

# "X1 is the most important cause of Y" <- WRONG!
```

**Why It's Wrong**:
- Feature importance measures predictive association, not causation
- Correlated features share importance
- Nuisance model is for adjustment, not causal discovery

---

## 6. Implementation Errors

### 6.1 Not Setting Random Seeds

**Error**: Results not reproducible across runs.

```python
# WRONG: No random seed
result1 = estimate_plr(data, outcome, treatment, controls)  # Seed = ?
result2 = estimate_plr(data, outcome, treatment, controls)  # Different!
```

**Correct Approach**:
```python
# CORRECT: Set random seed for reproducibility
result = estimate_plr(
    data, outcome, treatment, controls,
    random_state=42  # Reproducible
)
```

### 6.2 Ignoring Missing Values

**Error**: Not handling missing data appropriately.

```python
# WRONG: Ignoring NAs (may cause silent errors)
result = estimate_plr(df, 'y', 'd', controls)  # df has NAs
```

**Correct Approach**:
```python
# Check for missing values
print(f"Missing values: {df[['y', 'd'] + controls].isna().sum().sum()}")

# Option 1: Complete case analysis
df_complete = df.dropna(subset=['y', 'd'] + controls)

# Option 2: Imputation (use with caution)
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy='median')
df[controls] = imputer.fit_transform(df[controls])
```

### 6.3 Not Scaling Features

**Error**: Using unscaled features with regularized learners.

```python
# PROBLEMATIC: Lasso with different scales
# If X1 in [0, 1] and X2 in [0, 1000], Lasso penalizes X2 less

# This can cause issues with convergence and interpretation
```

**Correct Approach**:
```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Include scaling in pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('lasso', LassoCV())
])
```

---

## 7. Reporting Errors

### 7.1 Only Reporting One Specification

**Error**: Not showing sensitivity to learner choice.

```python
# WRONG: Only Lasso results
result = estimate_plr(data, outcome, treatment, controls, ml_l='lasso')
print(f"Effect: {result.effect}")  # Single specification
```

**Correct Approach**:
```python
# CORRECT: Compare multiple learners
comparison = compare_learners(
    data, outcome, treatment, controls,
    learner_list=['lasso', 'ridge', 'random_forest', 'xgboost']
)

print(comparison.summary_table)
print(f"Effect range: [{comparison.sensitivity['min_effect']:.4f}, "
      f"{comparison.sensitivity['max_effect']:.4f}]")
```

### 7.2 Not Reporting Cross-Fitting Parameters

**Error**: Omitting methodological details.

```python
# WRONG: Incomplete reporting
"""
We use DDML to estimate the treatment effect, finding theta = 0.08.
"""
```

**Correct Approach**:
```python
"""
CORRECT REPORTING:
We estimate treatment effects using Double/Debiased Machine Learning
(Chernozhukov et al., 2018) with a Partially Linear Regression model.
For nuisance functions E[Y|X] and E[D|X], we use Lasso with
cross-validated penalty selection. We employ 5-fold cross-fitting
with 10 repetitions for stable inference. The sample includes n=15,000
observations and p=50 control variables.
"""
```

### 7.3 Ignoring Diagnostics

**Error**: Not reporting nuisance model quality or overlap diagnostics.

**Correct Approach**:
```python
# Report in appendix/supplementary materials
"""
Appendix: DDML Diagnostics

Nuisance Model Performance (5-fold CV):
- E[Y|X] R-squared: 0.42
- E[D|X] R-squared: 0.31

Propensity Score Distribution:
- Min: 0.05, Max: 0.92, Mean: 0.34
- 2.3% of observations trimmed due to extreme propensities

Cross-Fitting Stability:
- Estimate CV across 10 repetitions: 3.2%
"""
```

---

## 8. Error Checklist

### Before Estimation
- [ ] Data has no missing values in key variables
- [ ] Treatment type identified (binary vs continuous)
- [ ] Features appropriately scaled/preprocessed
- [ ] Sample size adequate for chosen K

### During Estimation
- [ ] Using cross-fitting (not in-sample predictions)
- [ ] Correct score function for model type
- [ ] Appropriate learner for treatment model
- [ ] Random seed set for reproducibility

### After Estimation
- [ ] Checked propensity overlap (for IRM)
- [ ] Compared multiple learner specifications
- [ ] Verified cross-fitting stability
- [ ] Standard errors calculated correctly

### Interpretation
- [ ] Discussed unconfoundedness assumption
- [ ] Distinguished association from causation
- [ ] Understood estimand (PLR vs IRM)
- [ ] Noted sensitivity to specifications

---

## References

1. Chernozhukov, V., et al. (2018). Double/Debiased Machine Learning. *Econometrics Journal*.

2. Bach, P., et al. (2022). DoubleML: An Object-Oriented Implementation. *JMLR*.

3. Kennedy, E. H. (2022). Semiparametric Doubly Robust Targeted Double Machine Learning.
