# Linear Model Diagnostics

Comprehensive guide to assessing OLS assumptions, detecting model misspecification, and validating regularized regression results.

## OLS Assumption Diagnostics

### Overview of Key Assumptions

| Assumption | Consequence if Violated | Diagnostic |
|------------|------------------------|------------|
| Linearity | Biased coefficients | Residual vs. fitted plot |
| Independence | Invalid inference | Durbin-Watson, ACF plot |
| Homoskedasticity | Inefficient estimates, wrong SEs | Breusch-Pagan, White test |
| Normality | Invalid small-sample inference | Q-Q plot, Shapiro-Wilk |
| No multicollinearity | Unstable estimates, inflated SEs | VIF, condition number |
| No influential outliers | Distorted estimates | Cook's D, leverage |

## Residual Analysis

### Basic Residual Diagnostics

```python
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy import stats

def residual_diagnostics(model, X, y, feature_names=None):
    """
    Comprehensive residual diagnostics for OLS model.

    Parameters
    ----------
    model : fitted statsmodels OLS
    X : feature matrix (without constant)
    y : outcome
    feature_names : list of feature names

    Returns
    -------
    dict : diagnostic results
    """
    # Get residuals
    residuals = model.resid
    fitted = model.fittedvalues
    standardized_resid = model.get_influence().resid_studentized_internal

    results = {}

    # 1. Normality test
    _, p_shapiro = stats.shapiro(residuals[:min(5000, len(residuals))])
    _, p_jarque_bera = stats.jarque_bera(residuals)
    results['normality'] = {
        'shapiro_wilk_p': p_shapiro,
        'jarque_bera_p': p_jarque_bera,
        'normal': p_shapiro > 0.05
    }

    # 2. Skewness and kurtosis
    results['distribution'] = {
        'skewness': stats.skew(residuals),
        'kurtosis': stats.kurtosis(residuals),
        'excess_kurtosis': stats.kurtosis(residuals)  # scipy uses excess
    }

    # 3. Residual statistics
    results['residual_stats'] = {
        'mean': np.mean(residuals),
        'std': np.std(residuals),
        'min': np.min(residuals),
        'max': np.max(residuals)
    }

    return results


def plot_residual_diagnostics(model, figsize=(12, 10)):
    """
    Create standard residual diagnostic plots.

    Four-panel plot:
    1. Residuals vs Fitted
    2. Q-Q Plot
    3. Scale-Location
    4. Residuals vs Leverage
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)

    residuals = model.resid
    fitted = model.fittedvalues
    influence = model.get_influence()
    standardized_resid = influence.resid_studentized_internal
    leverage = influence.hat_matrix_diag
    cooks_d = influence.cooks_distance[0]

    # 1. Residuals vs Fitted
    ax1 = axes[0, 0]
    ax1.scatter(fitted, residuals, alpha=0.5, edgecolors='none')
    ax1.axhline(y=0, color='red', linestyle='--')
    # Add lowess smoothing
    from statsmodels.nonparametric.smoothers_lowess import lowess
    smoothed = lowess(residuals, fitted, frac=0.3)
    ax1.plot(smoothed[:, 0], smoothed[:, 1], color='red', linewidth=2)
    ax1.set_xlabel('Fitted Values')
    ax1.set_ylabel('Residuals')
    ax1.set_title('Residuals vs Fitted')

    # 2. Q-Q Plot
    ax2 = axes[0, 1]
    stats.probplot(standardized_resid, dist="norm", plot=ax2)
    ax2.set_title('Normal Q-Q Plot')

    # 3. Scale-Location
    ax3 = axes[1, 0]
    sqrt_std_resid = np.sqrt(np.abs(standardized_resid))
    ax3.scatter(fitted, sqrt_std_resid, alpha=0.5, edgecolors='none')
    smoothed_scale = lowess(sqrt_std_resid, fitted, frac=0.3)
    ax3.plot(smoothed_scale[:, 0], smoothed_scale[:, 1], color='red', linewidth=2)
    ax3.set_xlabel('Fitted Values')
    ax3.set_ylabel('sqrt(|Standardized Residuals|)')
    ax3.set_title('Scale-Location')

    # 4. Residuals vs Leverage
    ax4 = axes[1, 1]
    ax4.scatter(leverage, standardized_resid, alpha=0.5, edgecolors='none')
    ax4.axhline(y=0, color='gray', linestyle='--')
    # Cook's distance contours
    p = model.df_model
    n = len(residuals)
    x_range = np.linspace(0.001, ax4.get_xlim()[1], 50)
    for cook_val in [0.5, 1.0]:
        y_cook = np.sqrt(cook_val * p * (1 - x_range) / x_range)
        ax4.plot(x_range, y_cook, 'r--', alpha=0.5)
        ax4.plot(x_range, -y_cook, 'r--', alpha=0.5)
    ax4.set_xlabel('Leverage')
    ax4.set_ylabel('Standardized Residuals')
    ax4.set_title("Residuals vs Leverage")

    plt.tight_layout()
    return fig
```

### Interpreting Residual Plots

**Residuals vs Fitted**:
- Should show random scatter around zero
- Patterns indicate: non-linearity, omitted variables
- Funnel shape indicates: heteroskedasticity

**Q-Q Plot**:
- Points should lie on diagonal line
- Deviations in tails: heavy/light tails
- S-shape: skewness

**Scale-Location**:
- Should show horizontal line
- Upward trend: increasing variance (heteroskedasticity)

**Residuals vs Leverage**:
- High leverage + large residual = influential point
- Cook's distance > 1 typically concerning

## Heteroskedasticity Tests

### Breusch-Pagan Test

Tests whether residual variance depends on fitted values:

```python
from statsmodels.stats.diagnostic import het_breuschpagan

def breusch_pagan_test(model):
    """
    Breusch-Pagan test for heteroskedasticity.

    H0: Homoskedasticity (constant variance)
    H1: Heteroskedasticity (variance depends on X)
    """
    resid = model.resid
    exog = model.model.exog

    bp_stat, bp_pvalue, f_stat, f_pvalue = het_breuschpagan(resid, exog)

    return {
        'lm_stat': bp_stat,
        'lm_pvalue': bp_pvalue,
        'f_stat': f_stat,
        'f_pvalue': f_pvalue,
        'heteroskedastic': bp_pvalue < 0.05
    }
```

### White Test

More general test that includes squared terms and interactions:

```python
from statsmodels.stats.diagnostic import het_white

def white_test(model):
    """
    White's test for heteroskedasticity.

    More powerful than Breusch-Pagan; tests for general heteroskedasticity.
    """
    resid = model.resid
    exog = model.model.exog

    white_stat, white_pvalue, f_stat, f_pvalue = het_white(resid, exog)

    return {
        'lm_stat': white_stat,
        'lm_pvalue': white_pvalue,
        'f_stat': f_stat,
        'f_pvalue': f_pvalue,
        'heteroskedastic': white_pvalue < 0.05
    }
```

### Goldfeld-Quandt Test

Tests for heteroskedasticity by comparing variances in subgroups:

```python
from statsmodels.stats.diagnostic import het_goldfeldquandt

def goldfeld_quandt_test(model, idx_split=None):
    """
    Goldfeld-Quandt test for heteroskedasticity.

    Splits sample and compares residual variances.
    """
    endog = model.model.endog
    exog = model.model.exog

    if idx_split is None:
        idx_split = len(endog) // 3

    gq_stat, gq_pvalue, ordering = het_goldfeldquandt(
        endog, exog, idx=idx_split, alternative='two-sided'
    )

    return {
        'f_stat': gq_stat,
        'p_value': gq_pvalue,
        'heteroskedastic': gq_pvalue < 0.05
    }
```

### Remedies for Heteroskedasticity

```python
# 1. Robust standard errors (most common)
model_robust = sm.OLS(y, X).fit(cov_type='HC1')  # Huber-White

# 2. Weighted Least Squares (if variance structure known)
weights = 1 / variance_function(X)
model_wls = sm.WLS(y, X, weights=weights).fit()

# 3. Feasible GLS (estimate variance function)
# First fit OLS to get residuals
ols = sm.OLS(y, X).fit()
# Estimate variance model
log_sq_resid = np.log(ols.resid**2)
var_model = sm.OLS(log_sq_resid, X).fit()
weights = 1 / np.exp(var_model.fittedvalues)
model_fgls = sm.WLS(y, X, weights=weights).fit()
```

## Multicollinearity Diagnostics

### Variance Inflation Factor (VIF)

VIF measures how much coefficient variance is inflated due to multicollinearity:

$$VIF_j = \frac{1}{1 - R_j^2}$$

Where R_j^2 is the R-squared from regressing X_j on all other X variables.

```python
from statsmodels.stats.outliers_influence import variance_inflation_factor

def calculate_vif(X, feature_names=None):
    """
    Calculate VIF for each feature.

    VIF > 5: Moderate multicollinearity
    VIF > 10: Severe multicollinearity
    """
    if feature_names is None:
        feature_names = [f'X{i}' for i in range(X.shape[1])]

    # Add constant if not present
    if not np.allclose(X[:, 0], 1):
        X = sm.add_constant(X)
        feature_names = ['const'] + list(feature_names)

    vif_data = []
    for i in range(X.shape[1]):
        vif = variance_inflation_factor(X, i)
        vif_data.append({
            'feature': feature_names[i],
            'vif': vif,
            'concern': 'severe' if vif > 10 else ('moderate' if vif > 5 else 'low')
        })

    return pd.DataFrame(vif_data)
```

### Condition Number

Condition number measures overall multicollinearity in the design matrix:

```python
def condition_number(X):
    """
    Calculate condition number of X'X.

    Condition number > 30: Multicollinearity concern
    Condition number > 100: Severe multicollinearity
    """
    # Standardize X (important for meaningful condition number)
    X_centered = X - X.mean(axis=0)
    X_scaled = X_centered / X_centered.std(axis=0)

    # Condition number via SVD
    _, s, _ = np.linalg.svd(X_scaled)
    cond = s.max() / s.min()

    return {
        'condition_number': cond,
        'singular_values': s,
        'concern': 'severe' if cond > 100 else ('moderate' if cond > 30 else 'low')
    }
```

### Correlation Matrix Analysis

```python
def correlation_analysis(X, feature_names=None, threshold=0.8):
    """
    Identify highly correlated feature pairs.
    """
    if feature_names is None:
        feature_names = [f'X{i}' for i in range(X.shape[1])]

    corr_matrix = np.corrcoef(X.T)

    high_corr_pairs = []
    for i in range(len(feature_names)):
        for j in range(i+1, len(feature_names)):
            if abs(corr_matrix[i, j]) >= threshold:
                high_corr_pairs.append({
                    'feature_1': feature_names[i],
                    'feature_2': feature_names[j],
                    'correlation': corr_matrix[i, j]
                })

    return {
        'correlation_matrix': pd.DataFrame(
            corr_matrix,
            index=feature_names,
            columns=feature_names
        ),
        'high_correlation_pairs': pd.DataFrame(high_corr_pairs)
    }
```

### Remedies for Multicollinearity

1. **Ridge regression**: Stabilizes estimates
2. **Drop variables**: Remove redundant predictors
3. **PCA**: Replace with principal components
4. **Domain knowledge**: Choose one representative from correlated group

## Influential Observation Diagnostics

### Cook's Distance

Measures influence of each observation on fitted values:

```python
def cooks_distance_analysis(model, threshold=None):
    """
    Identify influential observations using Cook's distance.

    Common thresholds:
    - 4/n (conservative)
    - 1 (less conservative)
    - 4/(n-p-1) (sample size adjusted)
    """
    influence = model.get_influence()
    cooks_d = influence.cooks_distance[0]

    n = len(cooks_d)
    p = model.df_model

    if threshold is None:
        threshold = 4 / n  # Conservative default

    influential = np.where(cooks_d > threshold)[0]

    return {
        'cooks_distance': cooks_d,
        'threshold': threshold,
        'influential_indices': influential,
        'n_influential': len(influential),
        'max_cooks_d': cooks_d.max(),
        'max_cooks_idx': np.argmax(cooks_d)
    }
```

### Leverage (Hat Values)

Measures how unusual each observation's X values are:

```python
def leverage_analysis(model, threshold_multiplier=2):
    """
    Identify high-leverage observations.

    Average leverage = p/n
    High leverage typically defined as > 2*p/n or 3*p/n
    """
    influence = model.get_influence()
    leverage = influence.hat_matrix_diag

    n = len(leverage)
    p = model.df_model + 1  # +1 for intercept
    avg_leverage = p / n
    threshold = threshold_multiplier * avg_leverage

    high_leverage = np.where(leverage > threshold)[0]

    return {
        'leverage': leverage,
        'average_leverage': avg_leverage,
        'threshold': threshold,
        'high_leverage_indices': high_leverage,
        'n_high_leverage': len(high_leverage)
    }
```

### DFFITS and DFBETAS

```python
def influence_measures(model):
    """
    Calculate DFFITS and DFBETAS for influence analysis.

    DFFITS: Change in fitted value when observation removed
    DFBETAS: Change in coefficients when observation removed
    """
    influence = model.get_influence()

    dffits = influence.dffits[0]
    dfbetas = influence.dfbetas

    n = len(dffits)
    p = model.df_model + 1

    # Thresholds
    dffits_threshold = 2 * np.sqrt(p / n)
    dfbetas_threshold = 2 / np.sqrt(n)

    return {
        'dffits': dffits,
        'dffits_threshold': dffits_threshold,
        'dfbetas': dfbetas,
        'dfbetas_threshold': dfbetas_threshold,
        'high_dffits': np.where(np.abs(dffits) > dffits_threshold)[0]
    }
```

## Comprehensive Diagnostic Report

```python
def full_diagnostic_report(X, y, feature_names=None):
    """
    Generate comprehensive OLS diagnostic report.
    """
    if feature_names is None:
        feature_names = [f'X{i}' for i in range(X.shape[1])]

    # Fit OLS
    X_const = sm.add_constant(X)
    model = sm.OLS(y, X_const).fit()

    report = {
        'model_summary': {
            'r_squared': model.rsquared,
            'adj_r_squared': model.rsquared_adj,
            'f_statistic': model.fvalue,
            'f_pvalue': model.f_pvalue,
            'aic': model.aic,
            'bic': model.bic
        },
        'residuals': residual_diagnostics(model, X, y, feature_names),
        'heteroskedasticity': {
            'breusch_pagan': breusch_pagan_test(model),
            'white': white_test(model)
        },
        'multicollinearity': {
            'vif': calculate_vif(X, feature_names),
            'condition_number': condition_number(X)
        },
        'influential_observations': {
            'cooks_distance': cooks_distance_analysis(model),
            'leverage': leverage_analysis(model)
        }
    }

    # Summary of concerns
    concerns = []

    if report['residuals']['normality']['p'] < 0.05:
        concerns.append('Non-normal residuals')
    if report['heteroskedasticity']['breusch_pagan']['heteroskedastic']:
        concerns.append('Heteroskedasticity detected')
    if any(report['multicollinearity']['vif']['vif'] > 10):
        concerns.append('Severe multicollinearity (VIF > 10)')
    if report['influential_observations']['cooks_distance']['n_influential'] > 0:
        concerns.append(f"{report['influential_observations']['cooks_distance']['n_influential']} influential observations")

    report['concerns'] = concerns if concerns else ['No major concerns']

    return report
```

## Regularized Model Diagnostics

### Cross-Validation Error Analysis

```python
def cv_error_diagnostics(cv_results, alphas):
    """
    Diagnose cross-validation results for regularized models.
    """
    cv_mean = cv_results['cv_mean']
    cv_std = cv_results['cv_std']

    # Check for overfitting/underfitting
    best_idx = np.argmin(cv_mean)

    diagnostics = {
        'best_alpha': alphas[best_idx],
        'best_cv_error': cv_mean[best_idx],
        'cv_std_at_best': cv_std[best_idx],
        'cv_coefficient_of_variation': cv_std[best_idx] / cv_mean[best_idx]
    }

    # Check if at boundary
    if best_idx == 0:
        diagnostics['warning'] = 'Best alpha at lower boundary; consider smaller alphas'
    elif best_idx == len(alphas) - 1:
        diagnostics['warning'] = 'Best alpha at upper boundary; consider larger alphas'

    return diagnostics
```

### Selection Stability

```python
def selection_stability_analysis(X, y, n_bootstrap=100, alphas=None):
    """
    Assess stability of Lasso variable selection across bootstrap samples.
    """
    from sklearn.linear_model import LassoCV

    n, p = X.shape
    selection_matrix = np.zeros((n_bootstrap, p))

    for b in range(n_bootstrap):
        # Bootstrap sample
        idx = np.random.choice(n, n, replace=True)
        X_boot, y_boot = X[idx], y[idx]

        # Fit Lasso
        lasso = LassoCV(alphas=alphas, cv=5)
        lasso.fit(X_boot, y_boot)

        # Record selection
        selection_matrix[b] = (lasso.coef_ != 0).astype(int)

    selection_prob = selection_matrix.mean(axis=0)

    return {
        'selection_probability': selection_prob,
        'always_selected': np.where(selection_prob == 1)[0],
        'never_selected': np.where(selection_prob == 0)[0],
        'unstable': np.where((selection_prob > 0) & (selection_prob < 1))[0],
        'stability_scores': selection_prob
    }
```

## References

- Belsley, D. A., Kuh, E., & Welsch, R. E. (1980). Regression Diagnostics. Wiley.
- Cook, R. D., & Weisberg, S. (1982). Residuals and Influence in Regression. Chapman & Hall.
- White, H. (1980). A heteroskedasticity-consistent covariance matrix estimator. Econometrica, 48(4), 817-838.
- Breusch, T. S., & Pagan, A. R. (1979). A simple test for heteroscedasticity. Econometrica, 47(5), 1287-1294.
