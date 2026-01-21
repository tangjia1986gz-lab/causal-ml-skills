# Feature Engineering for Causal Inference

## Overview

Feature engineering transforms raw variables into representations that improve model performance and interpretability. In causal inference, feature engineering serves specific purposes: creating flexible functional forms for propensity scores, capturing effect heterogeneity, and reducing dimensionality while preserving confounding control.

## Transformations

### Log Transformations

Useful for right-skewed variables common in economic data:

```python
import numpy as np

# Log transformation (add small constant for zeros)
df['log_income'] = np.log(df['income'] + 1)

# Log transformation only for positive values
df['log_income'] = np.where(
    df['income'] > 0,
    np.log(df['income']),
    np.nan
)
```

**Causal considerations**:
- Coefficient interpretation changes (elasticities, % changes)
- May improve propensity score model performance
- Consider whether treatment effect is additive or multiplicative

### Power Transformations

Box-Cox and Yeo-Johnson transformations for normality:

```python
from sklearn.preprocessing import PowerTransformer

# Yeo-Johnson (handles negative values)
pt = PowerTransformer(method='yeo-johnson')
df['income_transformed'] = pt.fit_transform(df[['income']])

# Box-Cox (positive values only)
pt_bc = PowerTransformer(method='box-cox')
df['income_bc'] = pt_bc.fit_transform(df[['income']][df['income'] > 0])
```

**When to use**:
- Propensity score models assuming normality
- Outcome models with non-normal residuals
- Improving linear model fit

### Binning/Discretization

Converting continuous variables to categories:

```python
# Equal-width bins
df['age_bins'] = pd.cut(df['age'], bins=5)

# Quantile bins (equal frequency)
df['income_quintiles'] = pd.qcut(df['income'], q=5, labels=['Q1', 'Q2', 'Q3', 'Q4', 'Q5'])

# Custom bins (domain-specific)
age_bins = [0, 18, 35, 50, 65, 100]
df['age_group'] = pd.cut(df['age'], bins=age_bins, labels=['<18', '18-34', '35-49', '50-64', '65+'])
```

**Causal considerations**:
- Allows flexible, non-linear relationships
- Each bin is essentially a dummy variable
- Lose information about within-bin variation
- Useful for stratified analysis and effect heterogeneity

---

## Interaction Terms

### Creating Interactions

Interactions capture how the effect of one variable depends on another:

```python
from preprocessing import create_interactions

# Specific interactions
df_int = create_interactions(df, var_pairs=[
    ('age', 'education'),
    ('income', 'region')
])

# Treatment-covariate interactions (for heterogeneity)
for covariate in ['age', 'income', 'education']:
    df[f'treat_x_{covariate}'] = df['treatment'] * df[covariate]
```

### Applications in Causal Inference

#### Effect Modification

When you believe treatment effects vary by a moderator:

```python
# Model: Y = b0 + b1*D + b2*X + b3*D*X + e
# b1: Effect when X=0
# b3: How effect changes with X

import statsmodels.api as sm

df['D_x_age'] = df['treatment'] * df['age']
X = sm.add_constant(df[['treatment', 'age', 'D_x_age']])
model = sm.OLS(df['outcome'], X).fit()

# Effect at different ages
effect_at_age_30 = model.params['treatment'] + model.params['D_x_age'] * 30
effect_at_age_50 = model.params['treatment'] + model.params['D_x_age'] * 50
```

#### Propensity Score Flexibility

Include interactions to capture complex selection:

```python
from sklearn.preprocessing import PolynomialFeatures

# Automatic polynomial + interaction features
poly = PolynomialFeatures(degree=2, interaction_only=False, include_bias=False)
X_expanded = poly.fit_transform(df[covariates])

# For propensity score model
from sklearn.linear_model import LogisticRegression
ps_model = LogisticRegression(penalty='l2', C=1.0)
ps_model.fit(X_expanded, df['treatment'])
```

---

## Polynomial Features

### Creating Polynomial Terms

```python
from sklearn.preprocessing import PolynomialFeatures

# Degree 2 polynomials (includes interactions)
poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(df[['age', 'income']])

# Get feature names
feature_names = poly.get_feature_names_out(['age', 'income'])
# ['age', 'income', 'age^2', 'age income', 'income^2']
```

### Causal Applications

#### Flexible Outcome Models

```python
# Partially linear model with flexible controls
# Y = D*tau + g(X) + e
# where g(X) is approximated by polynomials

from sklearn.linear_model import Ridge

# Create polynomial features for controls
poly = PolynomialFeatures(degree=3, include_bias=False)
X_poly = poly.fit_transform(df[controls])

# Partialling out approach
X_full = np.column_stack([df['treatment'].values.reshape(-1, 1), X_poly])
model = Ridge(alpha=1.0).fit(X_full, df['outcome'])
treatment_effect = model.coef_[0]  # First coefficient is treatment
```

#### Double/Debiased Machine Learning

Polynomial features can serve as basis for the ML estimators:

```python
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import cross_val_predict

# Flexible first stage
X_poly = PolynomialFeatures(degree=2).fit_transform(df[controls])

# Predict outcome
Y_hat = cross_val_predict(GradientBoostingRegressor(), X_poly, df['outcome'], cv=5)
Y_resid = df['outcome'] - Y_hat

# Predict treatment
D_hat = cross_val_predict(GradientBoostingRegressor(), X_poly, df['treatment'], cv=5)
D_resid = df['treatment'] - D_hat

# Second stage
treatment_effect = np.sum(D_resid * Y_resid) / np.sum(D_resid**2)
```

---

## Dimensionality Reduction

### PCA for High-Dimensional Controls

When you have many potential confounders:

```python
from preprocessing import run_pca

# Reduce high-dimensional controls
pca_result = run_pca(df[controls], n_components=10)

print(f"Components: {pca_result['n_components']}")
print(f"Variance explained: {pca_result['cumulative_variance'][-1]:.1%}")

# Use PCA components as controls
df_reduced = pd.concat([
    df[['outcome', 'treatment']],
    pca_result['transformed']
], axis=1)
```

### Causal Considerations for PCA

**Potential issues**:
- PCA maximizes variance, not confounding control
- First principal component may capture outcome-irrelevant variation
- Components lack interpretability

**Recommendations**:
1. Retain enough components to explain substantial variance (e.g., 95%)
2. Consider supervised dimension reduction (e.g., PLS)
3. Use regularized regression instead when possible
4. Report sensitivity to number of components

### Alternative: Sufficient Dimension Reduction

Methods designed to preserve regression relationships:

```python
# Sliced Inverse Regression (SIR) - conceptual example
# Finds directions that predict outcome

from sklearn.preprocessing import StandardScaler

# This is conceptual - actual implementation more complex
def sufficient_dimension_reduction(X, Y, n_directions=2):
    """
    Find sufficient directions for regression of Y on X.
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Slice Y into categories
    n_slices = 10
    Y_sliced = pd.qcut(Y, n_slices, labels=False)

    # Compute slice means
    slice_means = []
    for s in range(n_slices):
        mask = Y_sliced == s
        slice_means.append(X_scaled[mask].mean(axis=0))

    slice_means = np.array(slice_means)

    # SVD of slice means gives directions
    U, S, Vt = np.linalg.svd(slice_means, full_matrices=False)

    return Vt[:n_directions].T  # Projection matrix
```

---

## Encoding Categorical Variables

See `scaling_encoding.md` for detailed encoding methods. Key causal points:

### For Propensity Scores

- One-hot encoding with regularization
- Target encoding can leak outcome information - use with caution
- Ensure same encoding in treatment and control groups

### For Outcome Models

- Drop one category for regression (reference category)
- Consider effect coding for main effects interpretation
- Interactions with treatment for heterogeneity analysis

---

## Best Practices for Causal Feature Engineering

### 1. Only Use Pre-Treatment Variables

```python
# CORRECT: Features measured before treatment
pre_treatment_vars = ['age', 'baseline_income', 'education', 'pre_health_status']
df_features = engineer_features(df[pre_treatment_vars])

# WRONG: Including post-treatment variables
# post_treatment_vars = ['current_income', 'current_health']  # BAD!
```

### 2. Document All Transformations

```python
preprocessing_log = {
    'transformations': [
        {'variable': 'income', 'method': 'log', 'details': 'log(income + 1)'},
        {'variable': 'age', 'method': 'polynomial', 'details': 'degree 2'},
    ],
    'interactions': [
        ('age', 'education'),
        ('income', 'region'),
    ],
    'dimensionality_reduction': {
        'method': 'PCA',
        'n_components': 10,
        'variance_explained': 0.95
    }
}
```

### 3. Apply Same Transformations to Both Groups

```python
# CORRECT: Fit on full data, apply to all
scaler = StandardScaler()
df[controls] = scaler.fit_transform(df[controls])

# WRONG: Different scaling by treatment status
# treated_data = scaler1.fit_transform(df[df['D']==1][controls])  # BAD!
# control_data = scaler2.fit_transform(df[df['D']==0][controls])  # BAD!
```

### 4. Avoid Data Snooping

```python
# CORRECT: Define features before analysis
feature_spec = {
    'polynomials': ['age', 'income'],
    'interactions': [('age', 'education')],
    'log_transform': ['income', 'wealth']
}
# Then estimate effects

# WRONG: Trying different feature specs until significance
# for features in feature_combinations:
#     effect = estimate(features)
#     if effect.pvalue < 0.05: break  # BAD!
```

### 5. Report Sensitivity to Feature Choices

```python
# Sensitivity analysis
results = {}

# Baseline: No feature engineering
results['raw'] = estimate_effect(df, controls)

# With polynomials
df_poly = add_polynomials(df, controls, degree=2)
results['polynomial'] = estimate_effect(df_poly, expanded_controls)

# With interactions
df_int = add_interactions(df, controls)
results['interactions'] = estimate_effect(df_int, expanded_controls)

# With PCA
df_pca = add_pca(df, controls, n_components=10)
results['pca'] = estimate_effect(df_pca, pca_controls)

# Report all results
print("Sensitivity to feature engineering:")
for spec, effect in results.items():
    print(f"  {spec}: {effect:.3f}")
```

---

## Summary

Feature engineering in causal inference must balance flexibility with interpretability and validity:

| Goal | Approach | Causal Consideration |
|------|----------|---------------------|
| Flexible confounding control | Polynomials, interactions | More flexibility = better balance, but risk of overfitting |
| Effect heterogeneity | Treatment x covariate interactions | Pre-specify hypotheses |
| High-dimensional controls | PCA, regularization | May not capture all confounding |
| Non-linear relationships | Binning, splines | Allows flexible functional forms |

**Key principle**: Feature engineering choices should be made before examining treatment effect estimates, documented clearly, and subject to sensitivity analysis.
