# Outlier Detection for Econometric Research

## Overview

Outliers can have substantial influence on econometric estimates, particularly in small samples and when using OLS. Proper outlier detection and handling is essential for robust causal inference.

## Univariate Outlier Detection

### Interquartile Range (IQR) Method

The classic non-parametric approach based on quartiles.

```python
import numpy as np
import pandas as pd

def iqr_outliers(series, k=1.5):
    """
    Detect outliers using IQR method.

    Parameters:
    -----------
    series : pd.Series
        Numeric data
    k : float
        Multiplier for IQR (1.5 = outlier, 3.0 = extreme outlier)

    Returns:
    --------
    dict with bounds and outlier mask
    """
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1

    lower_bound = q1 - k * iqr
    upper_bound = q3 + k * iqr

    outliers = (series < lower_bound) | (series > upper_bound)

    return {
        'lower_bound': lower_bound,
        'upper_bound': upper_bound,
        'n_outliers': outliers.sum(),
        'pct_outliers': (outliers.sum() / len(series)) * 100,
        'outlier_mask': outliers,
        'outlier_values': series[outliers].tolist()
    }
```

### Z-Score Method

Parametric method assuming normality.

```python
from scipy import stats

def zscore_outliers(series, threshold=3.0):
    """
    Detect outliers using Z-score method.

    Parameters:
    -----------
    series : pd.Series
        Numeric data
    threshold : float
        Z-score threshold (2.5, 3.0, or 3.5 common choices)
    """
    z_scores = np.abs(stats.zscore(series.dropna()))
    outliers = np.abs(z_scores) > threshold

    # Map back to original index
    outlier_mask = pd.Series(False, index=series.index)
    outlier_mask[series.dropna().index] = outliers

    return {
        'threshold': threshold,
        'n_outliers': outliers.sum(),
        'pct_outliers': (outliers.sum() / len(series.dropna())) * 100,
        'outlier_mask': outlier_mask,
        'z_scores': pd.Series(z_scores, index=series.dropna().index)
    }
```

### Modified Z-Score (MAD Method)

Robust alternative using Median Absolute Deviation.

```python
def mad_outliers(series, threshold=3.5):
    """
    Detect outliers using Modified Z-score (MAD method).
    More robust to outliers than standard Z-score.

    Parameters:
    -----------
    series : pd.Series
        Numeric data
    threshold : float
        Modified Z-score threshold (3.5 is common)
    """
    median = series.median()
    mad = np.median(np.abs(series - median))

    # Consistency constant for normal distribution
    k = 1.4826

    modified_z = 0.6745 * (series - median) / (k * mad) if mad > 0 else 0

    outliers = np.abs(modified_z) > threshold

    return {
        'median': median,
        'mad': mad,
        'threshold': threshold,
        'n_outliers': outliers.sum(),
        'outlier_mask': outliers,
        'modified_z_scores': modified_z
    }
```

### Percentile-Based Trimming

```python
def percentile_outliers(series, lower_pct=1, upper_pct=99):
    """
    Flag observations outside specified percentiles.
    """
    lower_bound = series.quantile(lower_pct / 100)
    upper_bound = series.quantile(upper_pct / 100)

    outliers = (series < lower_bound) | (series > upper_bound)

    return {
        'lower_bound': lower_bound,
        'upper_bound': upper_bound,
        'lower_pct': lower_pct,
        'upper_pct': upper_pct,
        'n_outliers': outliers.sum(),
        'outlier_mask': outliers
    }
```

## Multivariate Outlier Detection

### Mahalanobis Distance

Accounts for correlations between variables.

```python
from scipy.spatial.distance import mahalanobis
from scipy import stats

def mahalanobis_outliers(data, columns, threshold_pvalue=0.001):
    """
    Detect multivariate outliers using Mahalanobis distance.

    Parameters:
    -----------
    data : DataFrame
    columns : list
        Numeric columns to include
    threshold_pvalue : float
        Chi-square p-value threshold for outlier classification
    """
    subset = data[columns].dropna()

    # Calculate mean and covariance
    mean = subset.mean()
    cov = subset.cov()

    try:
        cov_inv = np.linalg.inv(cov)
    except np.linalg.LinAlgError:
        # Use pseudo-inverse if singular
        cov_inv = np.linalg.pinv(cov)

    # Calculate Mahalanobis distance for each observation
    distances = []
    for idx, row in subset.iterrows():
        d = mahalanobis(row, mean, cov_inv)
        distances.append(d)

    distances = np.array(distances)

    # Chi-square test for outliers
    # Mahalanobis^2 follows chi-square with p degrees of freedom
    p = len(columns)
    chi2_threshold = stats.chi2.ppf(1 - threshold_pvalue, p)
    outliers = distances**2 > chi2_threshold

    # P-values for each observation
    pvalues = 1 - stats.chi2.cdf(distances**2, p)

    results = pd.DataFrame({
        'mahalanobis_distance': distances,
        'mahalanobis_squared': distances**2,
        'pvalue': pvalues,
        'is_outlier': outliers
    }, index=subset.index)

    return {
        'chi2_threshold': chi2_threshold,
        'n_outliers': outliers.sum(),
        'pct_outliers': (outliers.sum() / len(subset)) * 100,
        'results': results
    }
```

### Isolation Forest

Machine learning approach for anomaly detection.

```python
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

def isolation_forest_outliers(data, columns, contamination=0.05, random_state=42):
    """
    Detect outliers using Isolation Forest algorithm.

    Parameters:
    -----------
    data : DataFrame
    columns : list
        Columns to use for detection
    contamination : float
        Expected proportion of outliers (0.01 to 0.5)
    """
    subset = data[columns].dropna()

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(subset)

    # Fit Isolation Forest
    iso_forest = IsolationForest(
        contamination=contamination,
        random_state=random_state,
        n_estimators=100
    )

    predictions = iso_forest.fit_predict(X_scaled)
    scores = iso_forest.decision_function(X_scaled)

    # -1 = outlier, 1 = inlier
    outliers = predictions == -1

    results = pd.DataFrame({
        'anomaly_score': scores,
        'is_outlier': outliers
    }, index=subset.index)

    return {
        'contamination': contamination,
        'n_outliers': outliers.sum(),
        'pct_outliers': (outliers.sum() / len(subset)) * 100,
        'results': results
    }
```

### Local Outlier Factor (LOF)

Density-based outlier detection.

```python
from sklearn.neighbors import LocalOutlierFactor

def lof_outliers(data, columns, n_neighbors=20, contamination=0.05):
    """
    Detect outliers using Local Outlier Factor.

    Parameters:
    -----------
    n_neighbors : int
        Number of neighbors for density estimation
    contamination : float
        Expected proportion of outliers
    """
    subset = data[columns].dropna()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(subset)

    lof = LocalOutlierFactor(
        n_neighbors=n_neighbors,
        contamination=contamination
    )

    predictions = lof.fit_predict(X_scaled)
    scores = lof.negative_outlier_factor_

    outliers = predictions == -1

    results = pd.DataFrame({
        'lof_score': -scores,  # Convert to positive (higher = more outlying)
        'is_outlier': outliers
    }, index=subset.index)

    return {
        'n_neighbors': n_neighbors,
        'n_outliers': outliers.sum(),
        'results': results
    }
```

## Influential Observations in Regression

### Cook's Distance

Measures the influence of each observation on all fitted values.

```python
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import OLSInfluence

def cooks_distance(y, X, threshold=None):
    """
    Calculate Cook's distance for regression influence.

    Parameters:
    -----------
    y : array-like
        Dependent variable
    X : DataFrame
        Independent variables (will add constant)
    threshold : float, optional
        Custom threshold (default: 4/n)
    """
    X_with_const = sm.add_constant(X)
    model = sm.OLS(y, X_with_const).fit()

    influence = OLSInfluence(model)
    cooks_d = influence.cooks_distance[0]

    n = len(y)
    if threshold is None:
        threshold = 4 / n

    influential = cooks_d > threshold

    results = pd.DataFrame({
        'cooks_distance': cooks_d,
        'is_influential': influential
    }, index=X.index)

    return {
        'threshold': threshold,
        'n_influential': influential.sum(),
        'pct_influential': (influential.sum() / n) * 100,
        'results': results,
        'model_summary': model.summary()
    }
```

### DFBETAS

Measures influence of each observation on each coefficient.

```python
def dfbetas_analysis(y, X, threshold=None):
    """
    Calculate DFBETAS for coefficient influence analysis.

    Parameters:
    -----------
    threshold : float, optional
        Custom threshold (default: 2/sqrt(n))
    """
    X_with_const = sm.add_constant(X)
    model = sm.OLS(y, X_with_const).fit()

    influence = OLSInfluence(model)
    dfbetas = influence.dfbetas

    n = len(y)
    if threshold is None:
        threshold = 2 / np.sqrt(n)

    # Check if any coefficient is influenced
    influential = np.abs(dfbetas).max(axis=1) > threshold

    results = pd.DataFrame(
        dfbetas,
        columns=['const'] + list(X.columns),
        index=X.index
    )
    results['max_dfbeta'] = np.abs(dfbetas).max(axis=1)
    results['is_influential'] = influential

    return {
        'threshold': threshold,
        'n_influential': influential.sum(),
        'dfbetas': results
    }
```

### DFFITS

Measures influence on fitted value of each observation.

```python
def dffits_analysis(y, X, threshold=None):
    """
    Calculate DFFITS for fitted value influence.

    Parameters:
    -----------
    threshold : float, optional
        Custom threshold (default: 2*sqrt(p/n))
    """
    X_with_const = sm.add_constant(X)
    model = sm.OLS(y, X_with_const).fit()

    influence = OLSInfluence(model)
    dffits = influence.dffits[0]

    n, p = X_with_const.shape
    if threshold is None:
        threshold = 2 * np.sqrt(p / n)

    influential = np.abs(dffits) > threshold

    results = pd.DataFrame({
        'dffits': dffits,
        'is_influential': influential
    }, index=X.index)

    return {
        'threshold': threshold,
        'n_influential': influential.sum(),
        'results': results
    }
```

### Leverage (Hat Values)

Measures how far observations are from the center of predictor space.

```python
def leverage_analysis(X, threshold=None):
    """
    Calculate leverage (hat) values.

    Parameters:
    -----------
    threshold : float, optional
        Custom threshold (default: 2*p/n)
    """
    X_with_const = sm.add_constant(X)
    n, p = X_with_const.shape

    # Hat matrix diagonal
    hat_matrix = X_with_const @ np.linalg.inv(X_with_const.T @ X_with_const) @ X_with_const.T
    leverage = np.diag(hat_matrix)

    if threshold is None:
        threshold = 2 * p / n

    high_leverage = leverage > threshold

    results = pd.DataFrame({
        'leverage': leverage,
        'is_high_leverage': high_leverage
    }, index=X.index)

    return {
        'threshold': threshold,
        'n_high_leverage': high_leverage.sum(),
        'mean_leverage': leverage.mean(),  # Should equal p/n
        'results': results
    }
```

## Comprehensive Influence Diagnostics

```python
def regression_influence_diagnostics(y, X, plot=True, output_dir=None):
    """
    Complete regression influence diagnostics.
    """
    X_with_const = sm.add_constant(X)
    model = sm.OLS(y, X_with_const).fit()
    influence = OLSInfluence(model)

    n, p = X_with_const.shape

    # Collect all diagnostics
    diagnostics = pd.DataFrame({
        'residual': model.resid,
        'standardized_resid': influence.resid_studentized_internal,
        'studentized_resid': influence.resid_studentized_external,
        'leverage': influence.hat_matrix_diag,
        'cooks_distance': influence.cooks_distance[0],
        'dffits': influence.dffits[0]
    }, index=X.index)

    # Add DFBETAS
    dfbetas_df = pd.DataFrame(
        influence.dfbetas,
        columns=[f'dfbeta_{c}' for c in ['const'] + list(X.columns)],
        index=X.index
    )
    diagnostics = pd.concat([diagnostics, dfbetas_df], axis=1)

    # Flag influential observations
    diagnostics['influential_cooks'] = diagnostics['cooks_distance'] > 4/n
    diagnostics['influential_dffits'] = np.abs(diagnostics['dffits']) > 2*np.sqrt(p/n)
    diagnostics['high_leverage'] = diagnostics['leverage'] > 2*p/n
    diagnostics['large_residual'] = np.abs(diagnostics['studentized_resid']) > 2

    diagnostics['any_influence_flag'] = (
        diagnostics['influential_cooks'] |
        diagnostics['influential_dffits'] |
        diagnostics['high_leverage']
    )

    if plot and output_dir:
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # Residuals vs Fitted
        axes[0, 0].scatter(model.fittedvalues, model.resid, alpha=0.5)
        axes[0, 0].axhline(y=0, color='r', linestyle='--')
        axes[0, 0].set_xlabel('Fitted Values')
        axes[0, 0].set_ylabel('Residuals')
        axes[0, 0].set_title('Residuals vs Fitted')

        # Cook's Distance
        axes[0, 1].stem(range(n), diagnostics['cooks_distance'], markerfmt='o')
        axes[0, 1].axhline(y=4/n, color='r', linestyle='--', label=f'Threshold (4/n)')
        axes[0, 1].set_xlabel('Observation')
        axes[0, 1].set_ylabel("Cook's Distance")
        axes[0, 1].set_title("Cook's Distance")
        axes[0, 1].legend()

        # Leverage vs Residuals
        axes[1, 0].scatter(diagnostics['leverage'], diagnostics['studentized_resid'], alpha=0.5)
        axes[1, 0].axhline(y=2, color='r', linestyle='--')
        axes[1, 0].axhline(y=-2, color='r', linestyle='--')
        axes[1, 0].axvline(x=2*p/n, color='r', linestyle='--')
        axes[1, 0].set_xlabel('Leverage')
        axes[1, 0].set_ylabel('Studentized Residuals')
        axes[1, 0].set_title('Influence Plot')

        # Q-Q plot of residuals
        sm.qqplot(model.resid, line='45', ax=axes[1, 1])
        axes[1, 1].set_title('Q-Q Plot of Residuals')

        plt.tight_layout()
        plt.savefig(f'{output_dir}/influence_diagnostics.png', dpi=150, bbox_inches='tight')
        plt.close()

    return {
        'diagnostics': diagnostics,
        'n_influential': diagnostics['any_influence_flag'].sum(),
        'influential_indices': diagnostics.index[diagnostics['any_influence_flag']].tolist(),
        'model_summary': model.summary()
    }
```

## Handling Outliers in Causal Inference

### Robustness Checks

1. **Compare estimates with and without outliers**
2. **Use robust regression methods** (MM-estimation, quantile regression)
3. **Winsorize** rather than remove (preserves sample size)
4. **Report sensitivity** of treatment effects to outlier handling

```python
def outlier_sensitivity_analysis(y, X, treatment_var, methods=['ols', 'robust', 'winsorized']):
    """
    Compare treatment effect estimates across outlier handling methods.
    """
    results = {}

    # OLS baseline
    model_ols = sm.OLS(y, sm.add_constant(X)).fit()
    results['ols'] = {
        'coef': model_ols.params[treatment_var],
        'se': model_ols.bse[treatment_var],
        'pvalue': model_ols.pvalues[treatment_var]
    }

    # Robust regression (MM-estimator)
    if 'robust' in methods:
        from statsmodels.robust.robust_linear_model import RLM
        model_robust = RLM(y, sm.add_constant(X), M=sm.robust.norms.TukeyBiweight()).fit()
        results['robust'] = {
            'coef': model_robust.params[treatment_var],
            'se': model_robust.bse[treatment_var],
            'pvalue': model_robust.pvalues[treatment_var]
        }

    # Winsorized regression
    if 'winsorized' in methods:
        from scipy.stats import mstats
        y_wins = mstats.winsorize(y, limits=[0.01, 0.01])
        X_wins = X.apply(lambda col: mstats.winsorize(col, limits=[0.01, 0.01]) if col.dtype in ['float64', 'int64'] else col)
        model_wins = sm.OLS(y_wins, sm.add_constant(X_wins)).fit()
        results['winsorized'] = {
            'coef': model_wins.params[treatment_var],
            'se': model_wins.bse[treatment_var],
            'pvalue': model_wins.pvalues[treatment_var]
        }

    return pd.DataFrame(results).T
```

## Causal Inference Considerations

| Issue | Impact on Causal Estimates | Recommendation |
|-------|---------------------------|----------------|
| Outliers in outcome | Can bias treatment effect | Winsorize or robust regression |
| Outliers in treatment | Usually binary, less concern | Check for measurement error |
| Outliers in covariates | Affects propensity scores | Separate analysis for extremes |
| Influential observations | Disproportionate effect on estimates | Report with/without sensitivity |
| High leverage points | Extrapolation beyond support | Check overlap, trim sample |
