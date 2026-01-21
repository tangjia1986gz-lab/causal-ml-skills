# Regularization Theory and Practice

Deep dive into regularization principles, cross-validation strategies, and hyperparameter selection for penalized regression.

## Theoretical Foundation

### The Bias-Variance Tradeoff

For any estimator, the expected prediction error decomposes:

$$\text{EPE} = \text{Bias}^2 + \text{Variance} + \text{Irreducible Error}$$

**OLS properties**:
- Unbiased: E[beta_hat] = beta
- High variance when p approaches n

**Regularized estimators**:
- Biased: E[beta_hat] != beta (shrinkage toward zero)
- Lower variance: More stable estimates
- Potentially lower EPE despite bias

### Ridge Regression Theory

#### Singular Value Decomposition Perspective

For X = U D V', the OLS solution is:
$$\hat{\beta}_{OLS} = V D^{-1} U' y$$

Ridge solution:
$$\hat{\beta}_{ridge} = V (D^2 + \lambda I)^{-1} D U' y = \sum_{j=1}^p \frac{d_j^2}{d_j^2 + \lambda} \frac{u_j' y}{d_j} v_j$$

**Interpretation**:
- Shrinks coefficients in directions of small singular values
- Large lambda -> coefficients in all directions shrink
- Small singular values (multicollinearity) get shrunk most

#### Effective Degrees of Freedom

Ridge regression has effective degrees of freedom:
$$\text{df}(\lambda) = \text{tr}(H_\lambda) = \sum_{j=1}^p \frac{d_j^2}{d_j^2 + \lambda}$$

Where H_lambda is the hat matrix for ridge. As lambda increases:
- df decreases from p toward 0
- Model becomes simpler/more constrained

### Lasso Theory

#### Soft Thresholding

For orthogonal X (X'X = I), Lasso has closed-form solution:
$$\hat{\beta}_j^{lasso} = \text{sign}(\hat{\beta}_j^{OLS})(|\hat{\beta}_j^{OLS}| - \lambda)_+$$

This is soft thresholding:
- Coefficients below lambda threshold set to exactly zero
- Remaining coefficients shrunk by lambda toward zero

#### Karush-Kuhn-Tucker (KKT) Conditions

Lasso optimum satisfies:
$$X_j'(y - X\hat{\beta}) = \lambda \cdot \text{sign}(\hat{\beta}_j) \quad \text{if } \hat{\beta}_j \neq 0$$
$$|X_j'(y - X\hat{\beta})| \leq \lambda \quad \text{if } \hat{\beta}_j = 0$$

**Intuition**: A variable enters the model when its correlation with residuals exceeds lambda.

#### Oracle Property

Under certain conditions (irrepresentable condition), Lasso achieves:
1. **Model selection consistency**: Correctly identifies true non-zero coefficients
2. **Estimation consistency**: Converges to true coefficients
3. **Rate**: Achieves minimax optimal rate sqrt(s log p / n)

Where s is the number of true non-zero coefficients.

### Elastic Net Theory

#### Grouping Effect

For correlated predictors X_i and X_j with correlation rho:
$$|\hat{\beta}_i^{enet} - \hat{\beta}_j^{enet}| \leq \frac{1}{\lambda_2} \sqrt{2(1-\rho)} \|y\|_1$$

**Implication**: As lambda_2 increases, correlated variables get more similar coefficients.

#### Naive Elastic Net Correction

The naive elastic net solution is:
$$\hat{\beta}^{naive} = \arg\min \|y - X\beta\|_2^2 + \lambda_1\|\beta\|_1 + \lambda_2\|\beta\|_2^2$$

Corrected elastic net:
$$\hat{\beta}^{enet} = (1 + \lambda_2) \hat{\beta}^{naive}$$

This debiases the double shrinkage from L1 and L2 penalties.

## Cross-Validation Strategies

### K-Fold Cross-Validation

Standard approach for hyperparameter selection:

```python
import numpy as np
from sklearn.model_selection import KFold

def kfold_cv(X, y, model_class, alphas, k=5, random_state=42):
    """
    K-fold cross-validation for regularization parameter.

    Returns MSE for each alpha value.
    """
    kf = KFold(n_splits=k, shuffle=True, random_state=random_state)
    cv_errors = np.zeros((len(alphas), k))

    for i, alpha in enumerate(alphas):
        for j, (train_idx, val_idx) in enumerate(kf.split(X)):
            model = model_class(alpha=alpha)
            model.fit(X[train_idx], y[train_idx])
            y_pred = model.predict(X[val_idx])
            cv_errors[i, j] = np.mean((y[val_idx] - y_pred)**2)

    return {
        'alphas': alphas,
        'cv_mean': cv_errors.mean(axis=1),
        'cv_std': cv_errors.std(axis=1),
        'cv_se': cv_errors.std(axis=1) / np.sqrt(k)
    }
```

### Leave-One-Out Cross-Validation (LOOCV)

Special case of K-fold where K = n:

**Advantages**:
- Uses maximum data for training
- No randomness in splits
- Efficient closed-form for Ridge

**Disadvantages**:
- Computationally expensive for Lasso
- High variance in error estimate
- May overfit hyperparameter

```python
# Efficient LOOCV for Ridge (closed-form)
from sklearn.linear_model import RidgeCV

# cv=None triggers efficient LOOCV
ridge_loocv = RidgeCV(alphas=np.logspace(-4, 4, 50), cv=None)
ridge_loocv.fit(X_scaled, y)
```

### Generalized Cross-Validation (GCV)

Approximates LOOCV with O(n) computation:

$$GCV(\lambda) = \frac{1}{n} \sum_{i=1}^n \left(\frac{y_i - \hat{y}_i}{1 - \text{df}(\lambda)/n}\right)^2$$

Where df(lambda) is the effective degrees of freedom.

```python
def gcv_ridge(X, y, alphas):
    """Generalized Cross-Validation for Ridge."""
    n, p = X.shape
    XtX = X.T @ X
    Xty = X.T @ y

    gcv_scores = []
    for alpha in alphas:
        # Ridge coefficients
        beta = np.linalg.solve(XtX + alpha * np.eye(p), Xty)
        y_hat = X @ beta

        # Effective degrees of freedom
        H = X @ np.linalg.solve(XtX + alpha * np.eye(p), X.T)
        df = np.trace(H)

        # GCV score
        resid = y - y_hat
        gcv = np.mean((resid / (1 - df/n))**2)
        gcv_scores.append(gcv)

    return np.array(gcv_scores)
```

### Time Series Cross-Validation

For temporal data, standard CV violates temporal ordering:

```python
from sklearn.model_selection import TimeSeriesSplit

def time_series_cv(X, y, model_class, alphas, n_splits=5):
    """Time series cross-validation with expanding window."""
    tscv = TimeSeriesSplit(n_splits=n_splits)
    cv_errors = np.zeros((len(alphas), n_splits))

    for i, alpha in enumerate(alphas):
        for j, (train_idx, val_idx) in enumerate(tscv.split(X)):
            model = model_class(alpha=alpha)
            model.fit(X[train_idx], y[train_idx])
            y_pred = model.predict(X[val_idx])
            cv_errors[i, j] = np.mean((y[val_idx] - y_pred)**2)

    return cv_errors.mean(axis=1)
```

### Nested Cross-Validation

For unbiased performance estimation:

```python
from sklearn.model_selection import cross_val_score, KFold, GridSearchCV

def nested_cv(X, y, model_class, param_grid, outer_cv=5, inner_cv=5):
    """
    Nested CV for unbiased performance estimation.

    Outer loop: Estimates generalization error
    Inner loop: Selects hyperparameters
    """
    outer_kf = KFold(n_splits=outer_cv, shuffle=True, random_state=42)
    outer_scores = []
    selected_alphas = []

    for train_idx, test_idx in outer_kf.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Inner CV for hyperparameter selection
        inner_cv_model = GridSearchCV(
            model_class(),
            param_grid,
            cv=inner_cv,
            scoring='neg_mean_squared_error'
        )
        inner_cv_model.fit(X_train, y_train)

        # Evaluate on outer test set
        y_pred = inner_cv_model.predict(X_test)
        outer_scores.append(np.mean((y_test - y_pred)**2))
        selected_alphas.append(inner_cv_model.best_params_['alpha'])

    return {
        'mean_mse': np.mean(outer_scores),
        'std_mse': np.std(outer_scores),
        'selected_alphas': selected_alphas
    }
```

## Lambda Selection Methods

### 1. Cross-Validation Minimum

Select lambda minimizing CV error:

```python
best_idx = np.argmin(cv_mean)
lambda_min = alphas[best_idx]
```

**Properties**:
- Optimizes prediction accuracy
- May select dense model
- Can overfit in small samples

### 2. One-Standard-Error Rule

Select largest lambda within 1 SE of minimum:

```python
def one_se_rule(alphas, cv_mean, cv_se):
    """
    1-SE rule: Most regularized model within 1 SE of best.

    Rationale: If two models have statistically similar CV error,
    prefer the simpler (more regularized) one.
    """
    best_idx = np.argmin(cv_mean)
    threshold = cv_mean[best_idx] + cv_se[best_idx]

    # Find largest alpha with CV error <= threshold
    valid_mask = cv_mean <= threshold
    valid_alphas = alphas[valid_mask]

    return valid_alphas.max()  # Largest valid alpha = most regularization
```

**Properties**:
- More conservative/parsimonious
- Sparser models (for Lasso)
- Better for variable selection
- Standard in glmnet

### 3. Information Criteria

BIC and AIC can guide lambda selection:

```python
def bic_lambda_selection(X, y, model_class, alphas):
    """Select lambda using BIC."""
    n = len(y)
    bic_scores = []

    for alpha in alphas:
        model = model_class(alpha=alpha)
        model.fit(X, y)

        # Residual sum of squares
        y_pred = model.predict(X)
        rss = np.sum((y - y_pred)**2)

        # Effective degrees of freedom (non-zero coefficients for Lasso)
        if hasattr(model, 'coef_'):
            df = np.sum(model.coef_ != 0) + 1  # +1 for intercept
        else:
            df = X.shape[1] + 1

        # BIC
        bic = n * np.log(rss/n) + df * np.log(n)
        bic_scores.append(bic)

    best_idx = np.argmin(bic_scores)
    return alphas[best_idx]
```

### 4. Stability Selection

Bootstrap-based selection for robust variable identification:

```python
def stability_selection(X, y, alphas, n_bootstrap=100, threshold=0.6):
    """
    Stability selection: Select variables that appear in >threshold
    fraction of bootstrap Lasso fits.
    """
    from sklearn.linear_model import LassoCV
    n, p = X.shape
    selection_counts = np.zeros(p)

    for b in range(n_bootstrap):
        # Bootstrap sample
        idx = np.random.choice(n, n, replace=True)
        X_boot, y_boot = X[idx], y[idx]

        # Fit Lasso with CV
        lasso = LassoCV(alphas=alphas, cv=5)
        lasso.fit(X_boot, y_boot)

        # Count selections
        selected = lasso.coef_ != 0
        selection_counts += selected

    # Selection probability
    selection_prob = selection_counts / n_bootstrap
    stable_features = np.where(selection_prob >= threshold)[0]

    return {
        'selection_probability': selection_prob,
        'stable_features': stable_features,
        'n_stable': len(stable_features)
    }
```

## Regularization Path

### Computing the Full Path

The regularization path shows coefficients as lambda varies:

```python
from sklearn.linear_model import lasso_path, enet_path

def compute_regularization_path(X, y, method='lasso', n_alphas=100):
    """
    Compute full regularization path.

    Returns alphas and coefficient matrix.
    """
    if method == 'lasso':
        alphas, coefs, _ = lasso_path(X, y, n_alphas=n_alphas)
    elif method == 'enet':
        alphas, coefs, _ = enet_path(X, y, l1_ratio=0.5, n_alphas=n_alphas)
    else:
        raise ValueError(f"Unknown method: {method}")

    return alphas, coefs
```

### LARS Algorithm

Least Angle Regression computes the full Lasso path efficiently:

1. Start with all coefficients at zero
2. Find predictor most correlated with residuals
3. Move coefficient in direction of correlation until another predictor equally correlated
4. Continue with both predictors, and so on

```python
from sklearn.linear_model import lars_path

alphas, _, coefs = lars_path(X, y, method='lasso')
```

### Path Visualization

```python
import matplotlib.pyplot as plt

def plot_path_with_cv(X, y, feature_names=None):
    """Plot regularization path with CV-optimal lambda."""
    from sklearn.linear_model import lasso_path, LassoCV

    # Compute path
    alphas, coefs, _ = lasso_path(X, y)

    # Get CV-optimal alpha
    lasso_cv = LassoCV(cv=5)
    lasso_cv.fit(X, y)

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))

    for i in range(coefs.shape[0]):
        label = feature_names[i] if feature_names else f"X{i}"
        ax.plot(np.log10(alphas), coefs[i], label=label)

    # Mark CV-optimal
    ax.axvline(np.log10(lasso_cv.alpha_), color='red', linestyle='--',
               label=f'CV-optimal (alpha={lasso_cv.alpha_:.4f})')

    ax.set_xlabel('log10(lambda)')
    ax.set_ylabel('Coefficient')
    ax.set_title('Lasso Regularization Path')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()

    return fig
```

## Practical Considerations

### Warm Starts

Fitting models along regularization path benefits from warm starts:

```python
from sklearn.linear_model import Lasso

def fit_path_warm_start(X, y, alphas):
    """Fit Lasso path with warm starts for efficiency."""
    # Sort alphas descending (start with most regularized)
    alphas_sorted = np.sort(alphas)[::-1]
    coefs = []

    # Initialize
    coef_warm = None

    for alpha in alphas_sorted:
        lasso = Lasso(alpha=alpha, warm_start=True, max_iter=10000)
        if coef_warm is not None:
            lasso.coef_ = coef_warm
        lasso.fit(X, y)
        coefs.append(lasso.coef_.copy())
        coef_warm = lasso.coef_

    return np.array(coefs)
```

### Convergence Diagnostics

Check that optimization has converged:

```python
from sklearn.linear_model import Lasso

lasso = Lasso(alpha=0.1, max_iter=10000)
lasso.fit(X, y)

# Check convergence
print(f"Number of iterations: {lasso.n_iter_}")
if lasso.n_iter_ == lasso.max_iter:
    print("WARNING: Did not converge. Increase max_iter.")
```

### Scaling Sensitivity

Regularization is sensitive to feature scaling:

```python
from sklearn.preprocessing import StandardScaler

# ALWAYS standardize before regularization
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Fit on scaled data
lasso = LassoCV(cv=5)
lasso.fit(X_scaled, y)

# Transform coefficients back to original scale (for interpretation)
coef_original_scale = lasso.coef_ / scaler.scale_
```

## References

- Hastie, T., Tibshirani, R., & Friedman, J. (2009). The Elements of Statistical Learning. Springer.
- Efron, B., Hastie, T., Johnstone, I., & Tibshirani, R. (2004). Least angle regression. Annals of Statistics, 32(2), 407-499.
- Zou, H. (2006). The adaptive lasso and its oracle properties. JASA, 101(476), 1418-1429.
- Meinshausen, N., & Buhlmann, P. (2010). Stability selection. JRSS-B, 72(4), 417-473.
