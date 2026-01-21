# Causal Inference Applications of Linear Models

This guide covers the use of regularized linear models in causal inference, with emphasis on partialling-out interpretation, post-double-selection, and high-dimensional control variable selection.

## The Causal Inference Challenge

### Why Standard Lasso Fails for Causal Inference

Consider estimating the effect of treatment D on outcome Y with controls X:

$$Y = \tau D + X'\beta + \epsilon$$

**Problem with naive Lasso**:
1. Lasso minimizes prediction error, not causal estimation error
2. May omit confounders that affect D but not Y directly
3. Shrinkage biases the treatment coefficient
4. Variable selection optimizes for Y prediction, not confounding control

**Example of bias**:
```
True model: Y = 2*D + 1.5*X1 + epsilon
            D = 0.5*X1 + 0.3*X2 + nu

If Lasso omits X1 (poor Y predictor with many controls):
- Omitted variable bias: E[tau_hat] = 2 + 1.5 * Cov(D,X1)/Var(D)
- Since X1 affects D, tau_hat is biased upward
```

## Partialling-Out / Frisch-Waugh-Lovell

### Theoretical Foundation

The Frisch-Waugh-Lovell (FWL) theorem states that in:
$$Y = \tau D + X'\beta + \epsilon$$

The OLS estimate of tau equals the coefficient from regressing residualized Y on residualized D:

1. Regress Y on X, get residuals: $\tilde{Y} = Y - X'\hat{\gamma}_Y$
2. Regress D on X, get residuals: $\tilde{D} = D - X'\hat{\gamma}_D$
3. Regress $\tilde{Y}$ on $\tilde{D}$: $\hat{\tau} = (\tilde{D}'\tilde{D})^{-1}\tilde{D}'\tilde{Y}$

### Partialling-Out with Machine Learning

This extends to ML first-stage estimators:

```python
import numpy as np
from sklearn.model_selection import cross_val_predict
from sklearn.linear_model import LassoCV, RidgeCV
import statsmodels.api as sm

def partialling_out_ml(X, y, d, ml_model_y=None, ml_model_d=None, cv=5):
    """
    Partialling-out estimator with ML first stages.

    Uses cross-fitting to avoid overfitting bias.

    Parameters
    ----------
    X : array-like of shape (n, p)
        Control variables
    y : array-like of shape (n,)
        Outcome variable
    d : array-like of shape (n,)
        Treatment variable
    ml_model_y : sklearn estimator, optional
        Model for E[Y|X]. Default: LassoCV
    ml_model_d : sklearn estimator, optional
        Model for E[D|X]. Default: LassoCV
    cv : int
        Number of cross-fitting folds

    Returns
    -------
    dict : treatment effect estimate with inference
    """
    if ml_model_y is None:
        ml_model_y = LassoCV(cv=5, max_iter=10000)
    if ml_model_d is None:
        ml_model_d = LassoCV(cv=5, max_iter=10000)

    # Cross-fitted predictions (out-of-fold)
    y_hat = cross_val_predict(ml_model_y, X, y, cv=cv)
    d_hat = cross_val_predict(ml_model_d, X, d, cv=cv)

    # Residualize
    y_tilde = y - y_hat
    d_tilde = d - d_hat

    # Second stage: OLS of residualized Y on residualized D
    d_tilde_const = sm.add_constant(d_tilde)
    model = sm.OLS(y_tilde, d_tilde_const).fit(cov_type='HC1')

    return {
        'treatment_effect': model.params[1],
        'std_error': model.bse[1],
        't_stat': model.tvalues[1],
        'p_value': model.pvalues[1],
        'ci_lower': model.conf_int()[1, 0],
        'ci_upper': model.conf_int()[1, 1],
        'y_residuals': y_tilde,
        'd_residuals': d_tilde,
        'first_stage_r2_y': 1 - np.var(y_tilde) / np.var(y),
        'first_stage_r2_d': 1 - np.var(d_tilde) / np.var(d)
    }
```

### Why Cross-Fitting Matters

Without cross-fitting, using in-sample predictions leads to:
1. **Overfitting bias**: First stage overestimates predictability
2. **Biased treatment effect**: Residuals are too small
3. **Invalid inference**: Standard errors are incorrect

```python
# WRONG: Same-sample fitting
ml_model.fit(X, y)
y_hat = ml_model.predict(X)  # Overfitted!

# CORRECT: Cross-fitting
y_hat = cross_val_predict(ml_model, X, y, cv=5)  # Out-of-fold
```

## Post-Double-Selection (Belloni et al., 2014)

### The Double Selection Algorithm

```
Algorithm: Post-Double-Selection Lasso
Input: Y (outcome), D (treatment), X (controls)
Output: Treatment effect estimate with valid inference

1. Lasso of Y on X
   - Fit Lasso_Y: Y ~ X
   - Selected_Y = {j : coef_j != 0}

2. Lasso of D on X
   - Fit Lasso_D: D ~ X
   - Selected_D = {j : coef_j != 0}

3. Union selection
   - Selected = Selected_Y UNION Selected_D

4. Post-selection OLS
   - Fit OLS: Y ~ D + X[Selected]
   - Use heteroskedasticity-robust standard errors

Output: coefficient on D from Step 4
```

### Why Union Selection?

**Single selection problem**: Lasso of Y on X may omit variables that:
- Strongly affect D (confounders)
- Have small direct effect on Y (but indirect through D)

**Double selection solution**:
- Step 1 captures variables predicting Y (direct effects)
- Step 2 captures variables predicting D (confounders)
- Union ensures no confounders are omitted

### Implementation

```python
def post_double_selection(
    X, y, d,
    alpha_y=None,
    alpha_d=None,
    cv=5,
    feature_names=None
):
    """
    Post-Double-Selection Lasso (Belloni, Chernozhukov, Hansen 2014).

    Provides valid inference for treatment effects with high-dimensional controls.

    Parameters
    ----------
    X : array-like of shape (n, p)
        Control variables
    y : array-like of shape (n,)
        Outcome variable
    d : array-like of shape (n,)
        Treatment variable
    alpha_y : float, optional
        Lasso penalty for Y~X. If None, uses CV
    alpha_d : float, optional
        Lasso penalty for D~X. If None, uses CV
    cv : int
        Cross-validation folds
    feature_names : list, optional
        Names of control variables

    Returns
    -------
    dict : results including treatment effect, CI, selected controls
    """
    from sklearn.linear_model import LassoCV, Lasso
    from sklearn.preprocessing import StandardScaler
    import statsmodels.api as sm

    n, p = X.shape
    if feature_names is None:
        feature_names = [f'X{i}' for i in range(p)]

    # Standardize X
    scaler = StandardScaler()
    X_std = scaler.fit_transform(X)

    # Step 1: Lasso Y ~ X
    if alpha_y is None:
        lasso_y = LassoCV(cv=cv, max_iter=10000)
    else:
        lasso_y = Lasso(alpha=alpha_y, max_iter=10000)
    lasso_y.fit(X_std, y)
    selected_y = set(np.where(lasso_y.coef_ != 0)[0])

    # Step 2: Lasso D ~ X
    if alpha_d is None:
        lasso_d = LassoCV(cv=cv, max_iter=10000)
    else:
        lasso_d = Lasso(alpha=alpha_d, max_iter=10000)
    lasso_d.fit(X_std, d)
    selected_d = set(np.where(lasso_d.coef_ != 0)[0])

    # Step 3: Union
    selected_union = sorted(selected_y | selected_d)

    # Step 4: Post-selection OLS
    if selected_union:
        X_selected = X[:, selected_union]
        X_ols = np.column_stack([d, X_selected])
    else:
        X_ols = d.reshape(-1, 1)

    X_ols = sm.add_constant(X_ols)
    ols = sm.OLS(y, X_ols).fit(cov_type='HC1')

    # Treatment effect is coefficient on D (index 1 after constant)
    tau = ols.params[1]
    se = ols.bse[1]
    ci = ols.conf_int()[1]

    return {
        'treatment_effect': tau,
        'std_error': se,
        't_stat': ols.tvalues[1],
        'p_value': ols.pvalues[1],
        'ci_lower': ci[0],
        'ci_upper': ci[1],
        'selected_by_y': selected_y,
        'selected_by_d': selected_d,
        'selected_union': selected_union,
        'selected_names': [feature_names[i] for i in selected_union],
        'n_selected_y': len(selected_y),
        'n_selected_d': len(selected_d),
        'n_selected_total': len(selected_union),
        'alpha_y': lasso_y.alpha_ if hasattr(lasso_y, 'alpha_') else alpha_y,
        'alpha_d': lasso_d.alpha_ if hasattr(lasso_d, 'alpha_') else alpha_d,
        'ols_model': ols
    }
```

### Theoretical Properties

Under regularity conditions, post-double-selection provides:

1. **Consistent estimation**: tau_hat -> tau as n -> infinity
2. **Asymptotic normality**: sqrt(n)(tau_hat - tau) -> N(0, V)
3. **Valid inference**: t-statistics and CIs have correct coverage
4. **Robustness**: Works when either Y or D model is misspecified

## Double/Debiased Machine Learning (DDML)

### Overview

DDML (Chernozhukov et al., 2018) extends partialling-out with:
1. **General ML first stages**: Any ML method, not just Lasso
2. **Cross-fitting**: Removes overfitting bias
3. **Neyman orthogonal scores**: Provides robustness to first-stage estimation error

### The Partially Linear Model

$$Y = \theta_0 D + g_0(X) + \epsilon, \quad E[\epsilon|D,X] = 0$$
$$D = m_0(X) + V, \quad E[V|X] = 0$$

Where:
- theta_0 is the causal parameter of interest
- g_0(X) = E[Y|X] (nuisance function)
- m_0(X) = E[D|X] (propensity/first stage)

### DDML Algorithm

```
Algorithm: Double/Debiased Machine Learning
Input: Y, D, X, ML methods for g and m, K folds

1. Split sample into K folds: I_1, ..., I_K

2. For each fold k = 1, ..., K:
   a. Train g_hat on data excluding fold k: g_hat_{-k}
   b. Train m_hat on data excluding fold k: m_hat_{-k}
   c. For observations in fold k:
      - Y_tilde_i = Y_i - g_hat_{-k}(X_i)
      - D_tilde_i = D_i - m_hat_{-k}(X_i)

3. Pool residuals across all folds

4. Estimate theta:
   theta_hat = (sum D_tilde_i^2)^{-1} * sum D_tilde_i * Y_tilde_i

5. Estimate variance:
   V_hat = (sum D_tilde_i^2)^{-2} * sum (Y_tilde_i - theta_hat * D_tilde_i)^2 * D_tilde_i^2
```

### Implementation

```python
def ddml_partially_linear(
    X, y, d,
    ml_g=None,
    ml_m=None,
    n_folds=5,
    random_state=42
):
    """
    Double/Debiased Machine Learning for Partially Linear Model.

    Implements Chernozhukov et al. (2018) estimator.

    Parameters
    ----------
    X : array-like of shape (n, p)
        Control variables
    y : array-like of shape (n,)
        Outcome variable
    d : array-like of shape (n,)
        Treatment variable
    ml_g : sklearn estimator
        ML method for E[Y|X]. Default: LassoCV
    ml_m : sklearn estimator
        ML method for E[D|X]. Default: LassoCV
    n_folds : int
        Number of cross-fitting folds
    random_state : int
        Random state for fold splitting

    Returns
    -------
    dict : DDML estimates with inference
    """
    from sklearn.model_selection import KFold
    from sklearn.base import clone

    if ml_g is None:
        ml_g = LassoCV(cv=5, max_iter=10000)
    if ml_m is None:
        ml_m = LassoCV(cv=5, max_iter=10000)

    n = len(y)
    y_tilde = np.zeros(n)
    d_tilde = np.zeros(n)

    # Cross-fitting
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)

    for train_idx, test_idx in kf.split(X):
        # Train on complement of fold
        g_model = clone(ml_g)
        m_model = clone(ml_m)

        g_model.fit(X[train_idx], y[train_idx])
        m_model.fit(X[train_idx], d[train_idx])

        # Predict on fold
        y_tilde[test_idx] = y[test_idx] - g_model.predict(X[test_idx])
        d_tilde[test_idx] = d[test_idx] - m_model.predict(X[test_idx])

    # Point estimate
    theta_hat = np.sum(d_tilde * y_tilde) / np.sum(d_tilde**2)

    # Variance estimate (heteroskedasticity-robust)
    psi = (y_tilde - theta_hat * d_tilde) * d_tilde
    J = np.mean(d_tilde**2)
    sigma2 = np.mean(psi**2)
    var_theta = sigma2 / (n * J**2)
    se_theta = np.sqrt(var_theta)

    # Inference
    t_stat = theta_hat / se_theta
    p_value = 2 * (1 - stats.norm.cdf(np.abs(t_stat)))
    ci_lower = theta_hat - 1.96 * se_theta
    ci_upper = theta_hat + 1.96 * se_theta

    return {
        'treatment_effect': theta_hat,
        'std_error': se_theta,
        't_stat': t_stat,
        'p_value': p_value,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'y_residuals': y_tilde,
        'd_residuals': d_tilde,
        'n_folds': n_folds,
        'sample_size': n
    }
```

## Choosing Between Methods

### Decision Matrix

| Scenario | Recommended Method |
|----------|-------------------|
| Low-dimensional X (p < n/10) | OLS with all controls |
| High-dimensional X, sparse | Post-Double-Selection |
| High-dimensional X, dense | DDML with Ridge |
| Complex nonlinear g, m | DDML with Random Forest/Boosting |
| Binary treatment | DDML with logistic propensity |
| Heterogeneous effects | Causal Forest, DDML with interactions |

### Practical Guidelines

1. **Always use cross-fitting** when using ML first stages
2. **Report both methods** when uncertain: Post-Lasso and DDML
3. **Check first-stage R-squared**: Low R^2 in D~X suggests weak identification
4. **Examine selected variables**: Do they make substantive sense?
5. **Sensitivity analysis**: Try different ML methods for robustness

## Common Pitfalls

### 1. Using Regularized Coefficients Directly

```python
# WRONG
lasso.fit(np.column_stack([d, X]), y)
treatment_effect = lasso.coef_[0]  # Biased!

# CORRECT
result = post_double_selection(X, y, d)
treatment_effect = result['treatment_effect']
```

### 2. Forgetting Cross-Fitting

```python
# WRONG: Overfitting bias
g_hat = ml_model.fit(X, y).predict(X)

# CORRECT: Cross-fitting
g_hat = cross_val_predict(ml_model, X, y, cv=5)
```

### 3. Single Selection

```python
# WRONG: May omit confounders
lasso_y = LassoCV().fit(X, y)
selected = np.where(lasso_y.coef_ != 0)[0]

# CORRECT: Double selection
result = post_double_selection(X, y, d)
selected = result['selected_union']
```

### 4. Ignoring Treatment Endogeneity

These methods control for **observable** confounders only. They do NOT solve:
- Unobserved confounding
- Reverse causality
- Measurement error in treatment

## Integration with DoubleML Package

```python
# Using the DoubleML package
from doubleml import DoubleMLPLR
from doubleml import DoubleMLData
from sklearn.ensemble import RandomForestRegressor

# Prepare data
dml_data = DoubleMLData.from_arrays(X, y, d)

# Define learners
ml_g = RandomForestRegressor(n_estimators=100, max_depth=5)
ml_m = RandomForestRegressor(n_estimators=100, max_depth=5)

# DDML estimator
dml_plr = DoubleMLPLR(dml_data, ml_g, ml_m, n_folds=5)
dml_plr.fit()

print(dml_plr.summary)
```

## References

### Core Papers

- Belloni, A., Chernozhukov, V., & Hansen, C. (2014). Inference on treatment effects after selection among high-dimensional controls. *Review of Economic Studies*, 81(2), 608-650.

- Chernozhukov, V., Chetverikov, D., Demirer, M., Duflo, E., Hansen, C., Newey, W., & Robins, J. (2018). Double/debiased machine learning for treatment and structural parameters. *The Econometrics Journal*, 21(1), C1-C68.

### Extensions

- Belloni, A., Chernozhukov, V., & Hansen, C. (2013). Inference for high-dimensional sparse econometric models. *Advances in Economics and Econometrics*.

- Athey, S., & Wager, S. (2019). Estimating treatment effects with causal forests. *JASA*.

### Software

- DoubleML: https://docs.doubleml.org/
- EconML: https://econml.azurewebsites.net/
- hdm (R): https://cran.r-project.org/package=hdm
