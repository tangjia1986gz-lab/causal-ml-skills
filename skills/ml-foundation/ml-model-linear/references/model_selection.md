# Linear Model Selection Guide

Comprehensive guide for choosing between OLS, Ridge, Lasso, and Elastic Net regression based on data characteristics and research objectives.

## Decision Framework

### Quick Selection Matrix

| Criterion | OLS | Ridge | Lasso | Elastic Net |
|-----------|-----|-------|-------|-------------|
| p << n (many more samples than features) | Yes | Optional | Optional | Optional |
| p ~ n (similar samples and features) | Risky | Yes | Yes | Yes |
| p > n (more features than samples) | No | Yes | Yes | Yes |
| Multicollinearity present | No | Yes | Partial | Yes |
| Expect sparse model | No | No | Yes | Yes |
| Correlated feature groups | N/A | Yes | One per group | Groups together |
| Need variable selection | No | No | Yes | Yes |
| Causal inference (marginal effects) | Yes | No | Post-Lasso | Post-Lasso |

## Ordinary Least Squares (OLS)

### When to Use OLS

1. **Classical regression settings**: p << n with no multicollinearity
2. **Causal inference**: When unbiased coefficient estimates are required
3. **Low-dimensional controls**: Established theoretical models with few variables
4. **Interpretability priority**: All coefficients are meaningful marginal effects

### OLS Assumptions

1. **Linearity**: E[Y|X] is linear in X
2. **Exogeneity**: E[epsilon|X] = 0 (no omitted variable bias)
3. **No perfect multicollinearity**: X'X is invertible
4. **Homoskedasticity**: Var(epsilon|X) = sigma^2 (for efficient OLS)
5. **Normality**: epsilon ~ N(0, sigma^2) (for valid inference)

### OLS Limitations

- **Variance explosion**: When p approaches n, (X'X)^{-1} becomes ill-conditioned
- **Multicollinearity**: Inflated standard errors, unstable coefficients
- **Overfitting**: High variance predictions with many predictors

```python
import statsmodels.api as sm

# OLS with robust standard errors
X_const = sm.add_constant(X)
ols_model = sm.OLS(y, X_const).fit(cov_type='HC1')

# Check for multicollinearity
from statsmodels.stats.outliers_influence import variance_inflation_factor
vif = [variance_inflation_factor(X_const, i) for i in range(X_const.shape[1])]
```

## Ridge Regression (L2 Penalty)

### Mathematical Formulation

$$\hat{\beta}_{ridge} = \arg\min_\beta \|y - X\beta\|_2^2 + \lambda \|\beta\|_2^2$$

Closed-form solution:
$$\hat{\beta}_{ridge} = (X'X + \lambda I)^{-1}X'y$$

### When to Use Ridge

1. **Multicollinearity**: Stabilizes coefficient estimates
2. **Prediction focus**: All variables believed relevant
3. **Grouped variables**: Want to keep correlated predictors together
4. **Shrinkage without selection**: Need all coefficients non-zero

### Ridge Properties

| Property | Behavior |
|----------|----------|
| Coefficient shrinkage | Toward zero, never exactly zero |
| Variable selection | No (all variables retained) |
| Bias-variance tradeoff | Increases bias, reduces variance |
| Correlated predictors | Shrinks together proportionally |
| Uniqueness | Always unique solution |

### Ridge Tuning

```python
from sklearn.linear_model import RidgeCV
import numpy as np

# Logarithmically spaced alphas
alphas = np.logspace(-4, 4, 50)

# Ridge with leave-one-out CV (efficient for Ridge)
ridge = RidgeCV(alphas=alphas, cv=None)  # cv=None uses efficient LOO
ridge.fit(X_scaled, y)

print(f"Optimal alpha: {ridge.alpha_}")
print(f"Effective degrees of freedom: {np.trace(X @ np.linalg.inv(X.T @ X + ridge.alpha_ * np.eye(X.shape[1])) @ X.T)}")
```

## Lasso Regression (L1 Penalty)

### Mathematical Formulation

$$\hat{\beta}_{lasso} = \arg\min_\beta \|y - X\beta\|_2^2 + \lambda \|\beta\|_1$$

No closed-form solution; requires coordinate descent or LARS algorithm.

### When to Use Lasso

1. **Sparse true model**: Many coefficients are truly zero
2. **Variable selection**: Need to identify important features
3. **High-dimensional settings**: p >> n possible
4. **Interpretability**: Want parsimonious model

### Lasso Properties

| Property | Behavior |
|----------|----------|
| Coefficient shrinkage | Toward zero, exactly zero for some |
| Variable selection | Yes (automatic sparsity) |
| Correlated predictors | Selects one arbitrarily |
| Maximum variables | At most min(n, p) non-zero |
| Solution uniqueness | May not be unique with correlated X |

### Lasso Selection Behavior

**Key insight**: Lasso tends to select one variable from a group of correlated predictors, which can be problematic:

1. **Good for prediction**: Redundant variables don't improve prediction
2. **Bad for interpretation**: May miss important correlated confounders
3. **Causal implication**: Can omit variables needed for causal identification

```python
from sklearn.linear_model import LassoCV

# Lasso with 1-SE rule implementation
lasso = LassoCV(cv=5, n_alphas=100)
lasso.fit(X_scaled, y)

# Manual 1-SE rule
mse_path = lasso.mse_path_
mse_mean = mse_path.mean(axis=1)
mse_std = mse_path.std(axis=1)
mse_se = mse_std / np.sqrt(mse_path.shape[1])

best_idx = np.argmin(mse_mean)
threshold = mse_mean[best_idx] + mse_se[best_idx]
valid_idx = np.where(mse_mean <= threshold)[0]
alpha_1se = lasso.alphas_[valid_idx[0]]  # Largest valid alpha

print(f"CV-optimal alpha: {lasso.alpha_}")
print(f"1-SE rule alpha: {alpha_1se}")
```

## Elastic Net (L1 + L2 Combined)

### Mathematical Formulation

$$\hat{\beta}_{enet} = \arg\min_\beta \|y - X\beta\|_2^2 + \lambda[\alpha\|\beta\|_1 + (1-\alpha)\|\beta\|_2^2]$$

Where:
- lambda: Overall penalty strength
- alpha (l1_ratio): Mixing parameter (1 = Lasso, 0 = Ridge)

### When to Use Elastic Net

1. **Correlated feature groups**: Want to select groups together
2. **p >> n**: More variables than observations
3. **Stability**: Lasso is too variable due to correlation
4. **Both selection and grouping**: Compromise between Ridge and Lasso

### Elastic Net Properties

| Property | Behavior |
|----------|----------|
| Coefficient shrinkage | Toward zero, some exactly zero |
| Variable selection | Yes, but keeps correlated groups |
| Correlated predictors | Tends to select all or none |
| Grouping effect | Correlated variables have similar coefficients |
| Computational | Slightly slower than Lasso |

### Elastic Net Tuning

Two hyperparameters require tuning:

```python
from sklearn.linear_model import ElasticNetCV

# Grid search over l1_ratio and alpha
l1_ratios = [0.1, 0.5, 0.7, 0.9, 0.95, 0.99, 1.0]

enet = ElasticNetCV(
    l1_ratio=l1_ratios,
    n_alphas=50,
    cv=5,
    max_iter=10000
)
enet.fit(X_scaled, y)

print(f"Optimal alpha: {enet.alpha_}")
print(f"Optimal l1_ratio: {enet.l1_ratio_}")

# Interpretation of l1_ratio
if enet.l1_ratio_ > 0.9:
    print("Model prefers Lasso-like sparsity")
elif enet.l1_ratio_ < 0.5:
    print("Model prefers Ridge-like grouping")
else:
    print("Model balances sparsity and grouping")
```

## Causal Inference Considerations

### Why Regularized Coefficients Are NOT Causal Effects

1. **Shrinkage bias**: Coefficients are systematically biased toward zero
2. **Selection bias**: Lasso may omit confounders
3. **Magnitude dependence**: Coefficient size depends on lambda, not effect size
4. **No standard errors**: Penalized coefficients lack valid inference

### Post-Selection Inference

For causal inference with high-dimensional controls:

1. **Use regularization for selection only**
2. **Refit with OLS on selected variables**
3. **Use double selection to avoid omitting confounders**
4. **Report post-selection standard errors (conservative)**

```python
# Correct approach for causal inference
from sklearn.linear_model import LassoCV
import statsmodels.api as sm

# Step 1: Select variables with Lasso
lasso = LassoCV(cv=5)
lasso.fit(X_scaled, y)
selected = np.where(lasso.coef_ != 0)[0]

# Step 2: Refit with OLS (unbiased coefficients)
X_selected = sm.add_constant(X[:, selected])
ols = sm.OLS(y, X_selected).fit(cov_type='HC1')

# Now coefficients are interpretable as marginal effects
print(ols.summary())
```

### Model Selection for Different Goals

| Goal | Recommended Approach |
|------|---------------------|
| Pure prediction | Ridge or Elastic Net with CV |
| Variable screening | Lasso with 1-SE rule |
| Causal effects | Post-double-selection Lasso |
| Feature importance | Lasso path stability selection |
| Interpretable model | Post-Lasso OLS |

## Practical Guidelines

### Data Preparation

1. **Always standardize** features before regularization
2. **Center outcome** for easier intercept interpretation
3. **Handle missing data** before fitting
4. **Check for outliers** that could dominate penalty

### Cross-Validation Best Practices

1. **K-fold CV**: Use k=5 or k=10 for stable estimates
2. **Stratified CV**: For imbalanced outcomes
3. **Time-series CV**: Rolling or expanding window for temporal data
4. **Nested CV**: For unbiased performance estimation

### Hyperparameter Selection

1. **CV-optimal**: Minimizes prediction error, may overfit
2. **1-SE rule**: More conservative, sparser model
3. **Theory-driven**: Prior knowledge about sparsity level
4. **Stability selection**: Bootstrap-based, most robust

## Common Pitfalls

### 1. Interpreting Regularized Coefficients as Effects

**Wrong**: "X1 increases Y by 0.5 units (Lasso coefficient)"
**Right**: "X1 was selected by Lasso; refit OLS to estimate effect"

### 2. Ignoring Standardization

**Problem**: Variables on different scales get unequal penalties
**Solution**: Always standardize before regularization

### 3. Using Default Lambda

**Problem**: Arbitrary penalty leads to poor model
**Solution**: Always use cross-validation for lambda selection

### 4. Single Selection for Causal Inference

**Problem**: May omit confounders that affect treatment but not outcome
**Solution**: Use double selection (Belloni et al., 2014)

## References

- Hastie, T., Tibshirani, R., & Wainwright, M. (2015). Statistical Learning with Sparsity: The Lasso and Generalizations. CRC Press.
- Zou, H., & Hastie, T. (2005). Regularization and variable selection via the elastic net. JRSS-B, 67(2), 301-320.
- Belloni, A., Chernozhukov, V., & Hansen, C. (2014). Inference on treatment effects after selection. Review of Economic Studies, 81(2), 608-650.
