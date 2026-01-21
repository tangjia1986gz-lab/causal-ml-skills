---
name: ml-model-linear
triggers:
  - lasso
  - ridge
  - elastic net
  - regularization
  - variable selection
  - high-dimensional
  - penalized regression
  - shrinkage
  - sparse regression
  - L1 penalty
  - L2 penalty
  - double selection
---

# Regularized Linear Models

Regularized regression methods for high-dimensional prediction and variable selection in causal inference applications.

## Overview

Regularized linear models add penalty terms to ordinary least squares (OLS) to prevent overfitting and enable variable selection. These methods are essential when:

- Number of features approaches or exceeds sample size (p ~ n or p > n)
- Multicollinearity exists among predictors
- Automatic variable selection is needed
- Prediction performance is prioritized over coefficient interpretation

## Ridge Regression (L2 Penalty)

Ridge regression adds an L2 penalty to the OLS objective:

$$\min_\beta \sum_{i=1}^n (y_i - x_i'\beta)^2 + \lambda \sum_{j=1}^p \beta_j^2$$

### Key Properties

- **Shrinkage**: Coefficients shrink toward zero but never exactly zero
- **No variable selection**: All variables remain in the model
- **Handles multicollinearity**: Stable estimates when predictors are correlated
- **Closed-form solution**: $\hat{\beta}_{ridge} = (X'X + \lambda I)^{-1}X'y$
- **Bias-variance tradeoff**: Introduces bias to reduce variance

### When to Use Ridge

- Multicollinearity is present
- All predictors are believed to be relevant
- Goal is prediction, not interpretation
- Grouped correlated variables should all be included

```python
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler

# Always standardize before regularization
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Ridge with cross-validation for alpha selection
ridge = RidgeCV(alphas=[0.001, 0.01, 0.1, 1, 10, 100])
ridge.fit(X_scaled, y)
print(f"Optimal alpha: {ridge.alpha_}")
```

## Lasso Regression (L1 Penalty)

Lasso (Least Absolute Shrinkage and Selection Operator) uses an L1 penalty:

$$\min_\beta \sum_{i=1}^n (y_i - x_i'\beta)^2 + \lambda \sum_{j=1}^p |\beta_j|$$

### Key Properties

- **Sparsity**: Sets some coefficients exactly to zero
- **Variable selection**: Automatically selects relevant features
- **No closed-form solution**: Requires iterative optimization
- **One-at-a-time selection**: Tends to select one variable from correlated groups
- **Oracle property**: Under certain conditions, selects true model asymptotically

### When to Use Lasso

- True model is believed to be sparse
- Variable selection is a primary goal
- Interpretability of selected variables matters
- High-dimensional settings (p > n possible)

```python
from sklearn.linear_model import LassoCV

# Lasso with cross-validation
lasso = LassoCV(alphas=None, cv=5)  # alphas=None uses automatic range
lasso.fit(X_scaled, y)

# Get selected variables
selected = np.where(lasso.coef_ != 0)[0]
print(f"Selected {len(selected)} out of {X.shape[1]} features")
```

## Elastic Net (L1 + L2 Combined)

Elastic Net combines both penalties:

$$\min_\beta \sum_{i=1}^n (y_i - x_i'\beta)^2 + \lambda_1 \sum_{j=1}^p |\beta_j| + \lambda_2 \sum_{j=1}^p \beta_j^2$$

Or equivalently with mixing parameter $\alpha$:

$$\min_\beta \|y - X\beta\|_2^2 + \lambda[\alpha\|\beta\|_1 + (1-\alpha)\|\beta\|_2^2]$$

### Key Properties

- **Grouped selection**: Selects groups of correlated variables together
- **Stability**: More stable than Lasso with correlated predictors
- **Two hyperparameters**: $\lambda$ (overall penalty) and $\alpha$ (L1/L2 mix)
- **Sparsity with grouping**: Combines advantages of Ridge and Lasso

### When to Use Elastic Net

- Correlated predictors exist and you want to keep groups together
- Number of features exceeds observations
- Both prediction and variable selection matter
- Lasso is unstable due to correlation

```python
from sklearn.linear_model import ElasticNetCV

# Elastic Net with CV for both alpha and l1_ratio
enet = ElasticNetCV(
    l1_ratio=[0.1, 0.5, 0.7, 0.9, 0.95, 0.99, 1.0],
    alphas=None,
    cv=5
)
enet.fit(X_scaled, y)
print(f"Optimal alpha: {enet.alpha_}, l1_ratio: {enet.l1_ratio_}")
```

## Cross-Validation for Hyperparameter Tuning

### Best Practices

1. **K-fold CV**: Use k=5 or k=10 for most applications
2. **Stratified splits**: For imbalanced outcomes
3. **Time series**: Use temporal CV (rolling window or expanding window)
4. **Nested CV**: For unbiased performance estimation

```python
from sklearn.model_selection import cross_val_score, KFold

# Manual cross-validation setup
cv = KFold(n_splits=5, shuffle=True, random_state=42)

# Get CV scores
scores = cross_val_score(lasso, X_scaled, y, cv=cv, scoring='neg_mean_squared_error')
rmse = np.sqrt(-scores.mean())
```

### Alpha Selection

- **1-SE Rule**: Choose largest alpha within 1 standard error of minimum CV error
- **More conservative**: Results in sparser models
- **Implementation**: Available in glmnet (R), manual in sklearn

## Feature Importance and Variable Selection

### Extracting Selected Variables (Lasso/ElasticNet)

```python
def get_selected_features(model, feature_names):
    """Extract non-zero coefficients and their importance."""
    coef = model.coef_
    nonzero_idx = np.where(coef != 0)[0]

    selected = {
        'features': [feature_names[i] for i in nonzero_idx],
        'coefficients': coef[nonzero_idx],
        'importance': np.abs(coef[nonzero_idx])
    }
    return selected
```

### Coefficient Interpretation

**IMPORTANT**: Regularized coefficients are NOT interpretable as marginal effects!

- Coefficients are shrunk toward zero (biased)
- Magnitude depends on penalty strength
- Use for variable selection, not effect estimation
- For causal inference, use selected variables in unpenalized regression

## Causal Inference Applications

### High-Dimensional Control Variable Selection

When estimating treatment effects with many potential controls:

1. Use Lasso/ElasticNet to select relevant controls
2. Include selected controls in causal model
3. Estimate treatment effect with standard methods

**Caution**: Naive Lasso selection can lead to omitted variable bias!

### Double Selection (Belloni, Chernozhukov, Hansen 2014)

The post-double-selection estimator addresses regularization bias in causal inference:

**Algorithm**:
1. **Step 1**: Lasso of Y on X (controls) → select controls predicting outcome
2. **Step 2**: Lasso of D on X (controls) → select controls predicting treatment
3. **Step 3**: OLS of Y on D and **union** of selected controls

**Why Double Selection?**
- Single selection may omit confounders that predict treatment but not outcome directly
- Union ensures all potential confounders are included
- Post-selection OLS provides valid inference

```python
def double_selection(X, y, d, alpha=None):
    """
    Belloni, Chernozhukov, Hansen (2014) double selection.

    Parameters
    ----------
    X : array-like of shape (n, p)
        Control variables (potential confounders)
    y : array-like of shape (n,)
        Outcome variable
    d : array-like of shape (n,)
        Treatment variable
    alpha : float, optional
        Lasso penalty parameter. If None, uses CV.

    Returns
    -------
    dict with treatment effect, standard error, and selected controls
    """
    from sklearn.linear_model import LassoCV
    import statsmodels.api as sm

    # Standardize X
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Step 1: Lasso Y ~ X
    lasso_y = LassoCV(cv=5)
    lasso_y.fit(X_scaled, y)
    selected_y = set(np.where(lasso_y.coef_ != 0)[0])

    # Step 2: Lasso D ~ X
    lasso_d = LassoCV(cv=5)
    lasso_d.fit(X_scaled, d)
    selected_d = set(np.where(lasso_d.coef_ != 0)[0])

    # Step 3: Union of selected variables
    selected_union = selected_y | selected_d

    # OLS with selected controls
    if selected_union:
        X_selected = X[:, list(selected_union)]
        X_ols = np.column_stack([d, X_selected])
    else:
        X_ols = d.reshape(-1, 1)

    X_ols = sm.add_constant(X_ols)
    model = sm.OLS(y, X_ols).fit(cov_type='HC1')

    return {
        'treatment_effect': model.params[1],
        'std_error': model.bse[1],
        'p_value': model.pvalues[1],
        'ci_lower': model.conf_int()[1, 0],
        'ci_upper': model.conf_int()[1, 1],
        'selected_controls_y': selected_y,
        'selected_controls_d': selected_d,
        'selected_union': selected_union,
        'n_selected': len(selected_union)
    }
```

### First-Stage Learners in Double/Debiased Machine Learning (DDML)

Regularized models serve as first-stage learners in DDML:

1. **Outcome model**: $\hat{g}(X) = E[Y|X]$ using Lasso/Ridge/ElasticNet
2. **Propensity model**: $\hat{m}(X) = E[D|X]$ using regularized logistic regression
3. **Cross-fitting**: Prevent overfitting bias with sample splitting

```python
from sklearn.linear_model import LassoCV, LogisticRegressionCV

def ddml_first_stage(X, y, d, cv=5):
    """First-stage learners for DDML."""

    # Outcome model: E[Y|X]
    outcome_model = LassoCV(cv=cv)
    outcome_model.fit(X, y)

    # Propensity model: E[D|X] (for binary treatment)
    propensity_model = LogisticRegressionCV(
        penalty='l1',
        solver='saga',
        cv=cv,
        max_iter=1000
    )
    propensity_model.fit(X, d)

    return outcome_model, propensity_model
```

## When to Use Each Method

| Method | Use When |
|--------|----------|
| **OLS** | p << n, no multicollinearity, all variables needed |
| **Ridge** | Multicollinearity, all predictors relevant, prediction focus |
| **Lasso** | Sparse true model, variable selection needed, p > n |
| **Elastic Net** | Correlated predictors, grouped selection, p >> n |
| **Double Selection** | Causal inference with high-dimensional controls |

## Interpretation of Coefficients

### Regularized vs. OLS Coefficients

| Aspect | OLS | Regularized |
|--------|-----|-------------|
| Unbiasedness | Unbiased | Biased (toward zero) |
| Variance | Higher | Lower |
| Interpretation | Marginal effects | Selection/prediction |
| Inference | Valid t-tests | Requires adjustment |

**Key Insight**: Use regularized methods for selection, then refit with OLS for interpretation.

## Common Mistakes

### 1. Not Standardizing Before Regularization

**Problem**: Penalties treat all coefficients equally; larger-scale variables get penalized less.

```python
# WRONG
lasso = Lasso(alpha=1.0)
lasso.fit(X, y)  # X not standardized

# CORRECT
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
lasso.fit(X_scaled, y)
```

### 2. Interpreting Regularized Coefficients as Causal Effects

**Problem**: Shrinkage biases coefficients; they are not valid marginal effects.

```python
# WRONG
print(f"Effect of X1: {lasso.coef_[0]}")  # Biased!

# CORRECT
# Use selected variables in unpenalized regression
selected = np.where(lasso.coef_ != 0)[0]
X_selected = X[:, selected]
ols = sm.OLS(y, sm.add_constant(X_selected)).fit()
print(f"Effect of X1: {ols.params[1]}")
```

### 3. Single Selection for Causal Inference

**Problem**: May omit confounders that affect treatment but not outcome directly.

```python
# WRONG (single selection)
lasso = LassoCV()
lasso.fit(X, y)
selected = np.where(lasso.coef_ != 0)[0]

# CORRECT (double selection)
result = double_selection(X, y, d)
```

### 4. Using Default Alpha Without Cross-Validation

**Problem**: Arbitrary penalty strength leads to poor performance.

```python
# WRONG
lasso = Lasso(alpha=1.0)  # Arbitrary alpha

# CORRECT
lasso = LassoCV(cv=5)  # CV-tuned alpha
```

### 5. Ignoring Cross-Fitting in DDML

**Problem**: Same-sample prediction leads to overfitting bias.

```python
# Use proper cross-fitting
from sklearn.model_selection import cross_val_predict

y_hat = cross_val_predict(lasso, X, y, cv=5)  # Out-of-fold predictions
```

## Regularization Path Visualization

```python
import matplotlib.pyplot as plt
from sklearn.linear_model import lasso_path

def plot_regularization_path(X, y, feature_names=None):
    """Plot coefficient paths as alpha varies."""
    alphas, coefs, _ = lasso_path(X, y)

    plt.figure(figsize=(10, 6))
    for i in range(coefs.shape[0]):
        label = feature_names[i] if feature_names else f"X{i}"
        plt.plot(np.log10(alphas), coefs[i], label=label)

    plt.xlabel('log10(alpha)')
    plt.ylabel('Coefficient')
    plt.title('Lasso Regularization Path')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    return plt.gcf()
```

## References

- Belloni, A., Chernozhukov, V., & Hansen, C. (2014). Inference on treatment effects after selection among high-dimensional controls. *Review of Economic Studies*, 81(2), 608-650.
- Tibshirani, R. (1996). Regression shrinkage and selection via the Lasso. *Journal of the Royal Statistical Society: Series B*, 58(1), 267-288.
- Zou, H., & Hastie, T. (2005). Regularization and variable selection via the elastic net. *Journal of the Royal Statistical Society: Series B*, 67(2), 301-320.
- Chernozhukov, V., et al. (2018). Double/debiased machine learning for treatment and structural parameters. *The Econometrics Journal*, 21(1), C1-C68.
