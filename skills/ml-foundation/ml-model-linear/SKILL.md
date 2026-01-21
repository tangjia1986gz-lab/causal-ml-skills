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
  - post-lasso
  - partialling out
  - FWL theorem
---

# Regularized Linear Models

Regularized regression methods for high-dimensional prediction and variable selection in causal inference applications.

## Skill Resources

### References
- [Model Selection Guide](references/model_selection.md) - OLS, Ridge, Lasso, Elastic Net decision framework
- [Regularization Theory](references/regularization.md) - Theory, cross-validation, lambda selection, stability
- [Diagnostics](references/diagnostics.md) - Residual analysis, heteroskedasticity, multicollinearity (VIF)
- [Causal Applications](references/causal_applications.md) - Post-Lasso, double selection, partialling out, DDML

### Scripts
- `scripts/run_linear_model.py` - Linear model fitting CLI with cross-validation
- `scripts/diagnose_assumptions.py` - OLS assumption diagnostics
- `scripts/visualize_coefficients.py` - Coefficient plots and regularization paths

### Templates
- `assets/latex/linear_table.tex` - LaTeX regression table template
- `assets/markdown/linear_report.md` - Analysis report template

---

## Overview

Regularized linear models add penalty terms to ordinary least squares (OLS) to prevent overfitting and enable variable selection. These methods are essential when:

- Number of features approaches or exceeds sample size (p ~ n or p > n)
- Multicollinearity exists among predictors
- Automatic variable selection is needed
- Prediction performance is prioritized over coefficient interpretation

## Quick Start

### Fit a Lasso Model with Variable Selection

```python
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler

# Always standardize before regularization
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Lasso with cross-validation
lasso = LassoCV(cv=5, max_iter=10000)
lasso.fit(X_scaled, y)

# Get selected variables
selected = np.where(lasso.coef_ != 0)[0]
print(f"Selected {len(selected)} out of {X.shape[1]} features")
print(f"Optimal alpha: {lasso.alpha_}")
```

### Post-Double-Selection for Causal Inference

```python
from linear_models import double_selection

# Estimate treatment effect with high-dimensional controls
result = double_selection(X, y, d)

print(f"Treatment effect: {result['treatment_effect']:.3f}")
print(f"95% CI: [{result['ci_lower']:.3f}, {result['ci_upper']:.3f}]")
print(f"Selected {result['n_selected']} control variables")
```

### Command-Line Usage

```bash
# Fit Lasso with cross-validation
python scripts/run_linear_model.py data.csv --outcome y --model lasso

# Double selection for causal inference
python scripts/run_linear_model.py data.csv --outcome y --treatment d --model double-selection

# Run OLS assumption diagnostics
python scripts/diagnose_assumptions.py data.csv --outcome y --all-features --plot

# Visualize regularization path
python scripts/visualize_coefficients.py data.csv --outcome y --plot-type path
```

---

## Model Selection Guide

| Method | Use When |
|--------|----------|
| **OLS** | p << n, no multicollinearity, all variables needed |
| **Ridge** | Multicollinearity, all predictors relevant, prediction focus |
| **Lasso** | Sparse true model, variable selection needed, p > n |
| **Elastic Net** | Correlated predictors, grouped selection, p >> n |
| **Double Selection** | Causal inference with high-dimensional controls |

See [Model Selection Guide](references/model_selection.md) for detailed decision criteria.

---

## Ridge Regression (L2 Penalty)

Ridge regression adds an L2 penalty to the OLS objective:

$$\min_\beta \sum_{i=1}^n (y_i - x_i'\beta)^2 + \lambda \sum_{j=1}^p \beta_j^2$$

### Key Properties

- **Shrinkage**: Coefficients shrink toward zero but never exactly zero
- **No variable selection**: All variables remain in the model
- **Handles multicollinearity**: Stable estimates when predictors are correlated
- **Closed-form solution**: $\hat{\beta}_{ridge} = (X'X + \lambda I)^{-1}X'y$
- **Bias-variance tradeoff**: Introduces bias to reduce variance

```python
from sklearn.linear_model import RidgeCV

ridge = RidgeCV(alphas=[0.001, 0.01, 0.1, 1, 10, 100])
ridge.fit(X_scaled, y)
print(f"Optimal alpha: {ridge.alpha_}")
```

---

## Lasso Regression (L1 Penalty)

Lasso (Least Absolute Shrinkage and Selection Operator) uses an L1 penalty:

$$\min_\beta \sum_{i=1}^n (y_i - x_i'\beta)^2 + \lambda \sum_{j=1}^p |\beta_j|$$

### Key Properties

- **Sparsity**: Sets some coefficients exactly to zero
- **Variable selection**: Automatically selects relevant features
- **No closed-form solution**: Requires iterative optimization
- **One-at-a-time selection**: Tends to select one variable from correlated groups
- **Oracle property**: Under certain conditions, selects true model asymptotically

```python
from sklearn.linear_model import LassoCV

lasso = LassoCV(alphas=None, cv=5)  # alphas=None uses automatic range
lasso.fit(X_scaled, y)
selected = np.where(lasso.coef_ != 0)[0]
```

---

## Elastic Net (L1 + L2 Combined)

Elastic Net combines both penalties:

$$\min_\beta \|y - X\beta\|_2^2 + \lambda[\alpha\|\beta\|_1 + (1-\alpha)\|\beta\|_2^2]$$

### Key Properties

- **Grouped selection**: Selects groups of correlated variables together
- **Stability**: More stable than Lasso with correlated predictors
- **Two hyperparameters**: $\lambda$ (overall penalty) and $\alpha$ (L1/L2 mix)

```python
from sklearn.linear_model import ElasticNetCV

enet = ElasticNetCV(
    l1_ratio=[0.1, 0.5, 0.7, 0.9, 0.95, 0.99, 1.0],
    alphas=None, cv=5
)
enet.fit(X_scaled, y)
print(f"l1_ratio: {enet.l1_ratio_}")
```

---

## Causal Inference Applications

### The Partialling-Out Interpretation

The Frisch-Waugh-Lovell theorem connects regularization to causal inference:

1. Regress Y on X (controls), get residuals: $\tilde{Y}$
2. Regress D (treatment) on X, get residuals: $\tilde{D}$
3. The coefficient from regressing $\tilde{Y}$ on $\tilde{D}$ equals the treatment effect

**With ML first stages**: Use Lasso/Ridge to predict Y and D, then estimate effect from residuals.

### Post-Double-Selection (Belloni et al., 2014)

Double selection addresses regularization bias in causal inference:

**Algorithm**:
1. **Step 1**: Lasso of Y on X (controls) -> select controls predicting outcome
2. **Step 2**: Lasso of D on X (controls) -> select controls predicting treatment
3. **Step 3**: OLS of Y on D and **union** of selected controls

**Why Double Selection?**
- Single selection may omit confounders that predict treatment but not outcome directly
- Union ensures all potential confounders are included
- Post-selection OLS provides valid inference

```python
def double_selection(X, y, d, cv=5):
    """Belloni, Chernozhukov, Hansen (2014) double selection."""
    from sklearn.linear_model import LassoCV
    import statsmodels.api as sm

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Step 1: Lasso Y ~ X
    lasso_y = LassoCV(cv=cv)
    lasso_y.fit(X_scaled, y)
    selected_y = set(np.where(lasso_y.coef_ != 0)[0])

    # Step 2: Lasso D ~ X
    lasso_d = LassoCV(cv=cv)
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
        'selected_controls': selected_union,
        'n_selected': len(selected_union)
    }
```

See [Causal Applications](references/causal_applications.md) for detailed theory and implementation.

---

## Cross-Validation for Hyperparameter Tuning

### Alpha Selection Rules

- **CV-optimal**: Choose alpha minimizing cross-validation error
- **1-SE Rule**: Choose largest alpha within 1 standard error of minimum (more conservative, sparser)

```python
# 1-SE rule implementation
mse_path = lasso.mse_path_
mse_mean = mse_path.mean(axis=1)
mse_se = mse_path.std(axis=1) / np.sqrt(mse_path.shape[1])

best_idx = np.argmin(mse_mean)
threshold = mse_mean[best_idx] + mse_se[best_idx]
valid_idx = np.where(mse_mean <= threshold)[0]
alpha_1se = lasso.alphas_[valid_idx[0]]  # Largest valid alpha
```

See [Regularization Theory](references/regularization.md) for detailed CV strategies.

---

## Interpretation of Coefficients

### Regularized vs. OLS Coefficients

| Aspect | OLS | Regularized |
|--------|-----|-------------|
| Unbiasedness | Unbiased | Biased (toward zero) |
| Variance | Higher | Lower |
| Interpretation | Marginal effects | Selection/prediction |
| Inference | Valid t-tests | Requires adjustment |

**Key Insight**: Use regularized methods for selection, then refit with OLS for interpretation.

---

## Common Mistakes

### 1. Not Standardizing Before Regularization

```python
# WRONG
lasso.fit(X, y)  # X not standardized

# CORRECT
X_scaled = StandardScaler().fit_transform(X)
lasso.fit(X_scaled, y)
```

### 2. Interpreting Regularized Coefficients as Causal Effects

```python
# WRONG
print(f"Effect of X1: {lasso.coef_[0]}")  # Biased!

# CORRECT: Use selected variables in unpenalized regression
selected = np.where(lasso.coef_ != 0)[0]
X_selected = X[:, selected]
ols = sm.OLS(y, sm.add_constant(X_selected)).fit()
```

### 3. Single Selection for Causal Inference

```python
# WRONG (single selection)
lasso = LassoCV()
lasso.fit(X, y)
selected = np.where(lasso.coef_ != 0)[0]

# CORRECT (double selection)
result = double_selection(X, y, d)
```

### 4. Ignoring Cross-Fitting in ML-Based Causal Inference

```python
# WRONG: Same-sample fitting
y_hat = ml_model.fit(X, y).predict(X)  # Overfitted!

# CORRECT: Cross-fitting
from sklearn.model_selection import cross_val_predict
y_hat = cross_val_predict(ml_model, X, y, cv=5)  # Out-of-fold
```

See [Diagnostics](references/diagnostics.md) for assumption checking.

---

## Visualization

### Regularization Path

```python
import matplotlib.pyplot as plt
from sklearn.linear_model import lasso_path

alphas, coefs, _ = lasso_path(X_scaled, y)

plt.figure(figsize=(10, 6))
for i in range(coefs.shape[0]):
    plt.plot(np.log10(alphas), coefs[i])

plt.xlabel('log10(alpha)')
plt.ylabel('Coefficient')
plt.title('Lasso Regularization Path')
plt.axvline(np.log10(lasso.alpha_), color='red', linestyle='--')
```

Or use the CLI:
```bash
python scripts/visualize_coefficients.py data.csv --outcome y --plot-type path
```

---

## References

### Core Papers

- Belloni, A., Chernozhukov, V., & Hansen, C. (2014). Inference on treatment effects after selection among high-dimensional controls. *Review of Economic Studies*, 81(2), 608-650.
- Tibshirani, R. (1996). Regression shrinkage and selection via the Lasso. *JRSS-B*, 58(1), 267-288.
- Zou, H., & Hastie, T. (2005). Regularization and variable selection via the elastic net. *JRSS-B*, 67(2), 301-320.
- Chernozhukov, V., et al. (2018). Double/debiased machine learning for treatment and structural parameters. *The Econometrics Journal*, 21(1), C1-C68.

### Software

- scikit-learn: https://scikit-learn.org/stable/modules/linear_model.html
- DoubleML: https://docs.doubleml.org/
- hdm (R): https://cran.r-project.org/package=hdm
