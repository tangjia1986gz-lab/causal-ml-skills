# Vector Autoregression (VAR) Models

## Overview

Vector Autoregression (VAR) models jointly analyze multiple time series, capturing their dynamic interdependencies. VAR is a natural extension of univariate AR models to multivariate settings.

## Model Specification

### Basic VAR(p) Model

For a k-dimensional vector Y_t:
```
Y_t = c + A₁Y_{t-1} + A₂Y_{t-2} + ... + A_pY_{t-p} + ε_t
```

Where:
- Y_t = (y₁,t, y₂,t, ..., y_k,t)' is a k×1 vector
- c = (c₁, c₂, ..., c_k)' is a k×1 vector of constants
- A_i are k×k coefficient matrices
- ε_t ~ iid N(0, Σ) is the error term

### Matrix Representation

```
Y_t = c + A(L)Y_t + ε_t
```

Where A(L) = A₁L + A₂L² + ... + A_pL^p is a matrix polynomial in the lag operator.

## Estimation

### OLS Estimation

Each equation can be estimated by OLS:

```python
from statsmodels.tsa.api import VAR
import pandas as pd

def fit_var(data, maxlags=None, ic='aic', trend='c'):
    """
    Fit VAR model with automatic lag selection.

    Parameters
    ----------
    data : DataFrame
        Multivariate time series
    maxlags : int, optional
        Maximum lags to consider
    ic : str
        Information criterion for lag selection ('aic', 'bic', 'hqic', 'fpe')
    trend : str
        'c' = constant, 'ct' = constant + trend, 'n' = none

    Returns
    -------
    VARResults object
    """
    model = VAR(data)

    # Lag selection
    if maxlags is None:
        maxlags = int(12 * (len(data) / 100) ** 0.25)  # Rule of thumb

    lag_order = model.select_order(maxlags=maxlags)
    optimal_lag = lag_order.selected_orders[ic]

    # Fit model
    result = model.fit(maxlags=optimal_lag, trend=trend)

    return result, lag_order
```

### Lag Order Selection

```python
def var_lag_selection(data, maxlags=15):
    """
    Display lag order selection criteria.

    Returns DataFrame with AIC, BIC, HQIC, FPE for each lag.
    """
    model = VAR(data)
    lag_order = model.select_order(maxlags=maxlags)

    print(lag_order.summary())

    return lag_order
```

## Stability and Stationarity

### Stability Condition

VAR(p) is stable if all eigenvalues of the companion matrix lie inside the unit circle.

```python
def check_var_stability(result):
    """
    Check VAR stability condition.

    Returns
    -------
    dict with eigenvalues and stability indicator
    """
    import numpy as np

    # Get roots of characteristic polynomial
    roots = result.roots

    stability = {
        'eigenvalues': roots,
        'moduli': np.abs(roots),
        'is_stable': all(np.abs(roots) < 1),
        'max_modulus': np.max(np.abs(roots))
    }

    return stability
```

### VAR with Non-Stationary Data

Options for I(1) data:
1. **Difference and fit VAR in differences** - Loses long-run information
2. **Fit VAR in levels** - Valid if series cointegrated (VECM)
3. **Mixed approach** - Depends on analysis goals

## Granger Causality

### Definition

X "Granger-causes" Y if past values of X help predict Y beyond Y's own past values.

**CRITICAL**: Granger causality is about **prediction**, not causation!

### Testing Procedure

```python
def granger_causality_test(data, var_result, caused, causing, maxlag=None):
    """
    Test for Granger causality.

    Parameters
    ----------
    data : DataFrame
        Original data
    var_result : VARResults
        Fitted VAR model
    caused : str
        Name of potentially caused variable
    causing : str or list
        Name(s) of potentially causing variable(s)
    maxlag : int
        Number of lags to test

    Returns
    -------
    dict with test results
    """
    from statsmodels.tsa.stattools import grangercausalitytests

    if maxlag is None:
        maxlag = var_result.k_ar

    # Create bivariate series
    test_data = data[[caused, causing]]

    # Conduct test
    results = grangercausalitytests(test_data, maxlag=maxlag, verbose=False)

    # Extract results
    gc_results = {}
    for lag, result in results.items():
        gc_results[lag] = {
            'ssr_f_test': result[0]['ssr_ftest'],
            'ssr_chi2_test': result[0]['ssr_chi2test'],
            'params_f_test': result[0]['params_ftest'],
            'p_value_f': result[0]['ssr_ftest'][1],
            'p_value_chi2': result[0]['ssr_chi2test'][1]
        }

    return gc_results
```

### Block Granger Causality

Test whether a group of variables jointly Granger-causes another:

```python
def test_block_granger(var_result, caused, causing_list):
    """
    Test block Granger causality (multiple causing variables).

    Returns
    -------
    dict with F-statistic and p-value
    """
    # Use VAR's built-in test
    result = var_result.test_causality(caused, causing=causing_list, kind='f')

    return {
        'test_statistic': result.test_statistic,
        'p_value': result.pvalue,
        'df': result.df,
        'conclusion': 'Granger causal' if result.pvalue < 0.05 else 'Not Granger causal'
    }
```

### Granger Causality Summary

```python
def granger_causality_matrix(var_result, significance=0.05):
    """
    Create matrix of pairwise Granger causality results.

    Returns
    -------
    DataFrame with p-values for all pairs
    """
    import pandas as pd
    import numpy as np

    variables = var_result.names
    n_vars = len(variables)

    p_values = np.zeros((n_vars, n_vars))

    for i, caused in enumerate(variables):
        for j, causing in enumerate(variables):
            if i != j:
                try:
                    result = var_result.test_causality(caused, causing=[causing], kind='f')
                    p_values[i, j] = result.pvalue
                except:
                    p_values[i, j] = np.nan
            else:
                p_values[i, j] = np.nan

    df = pd.DataFrame(p_values, index=variables, columns=variables)
    df.index.name = 'Caused'
    df.columns.name = 'Causing'

    return df
```

## Impulse Response Functions (IRF)

### Orthogonalized IRF

Response of variable i to a one-standard-deviation shock in variable j:

```python
def compute_irf(var_result, periods=20, orthogonalized=True):
    """
    Compute impulse response functions.

    Parameters
    ----------
    var_result : VARResults
        Fitted VAR model
    periods : int
        Forecast horizon
    orthogonalized : bool
        If True, use Cholesky decomposition for orthogonalization

    Returns
    -------
    IRAnalysis object
    """
    if orthogonalized:
        irf = var_result.irf(periods=periods)
    else:
        irf = var_result.irf(periods=periods, ortho=False)

    return irf
```

### Plotting IRFs

```python
def plot_irf(irf, impulse=None, response=None, orth=True, figsize=(12, 8)):
    """
    Plot impulse response functions.

    Parameters
    ----------
    irf : IRAnalysis
        IRF object from compute_irf
    impulse : str, optional
        Variable giving the impulse (None = all)
    response : str, optional
        Variable responding (None = all)
    """
    import matplotlib.pyplot as plt

    fig = irf.plot(
        impulse=impulse,
        response=response,
        orth=orth,
        figsize=figsize
    )

    plt.tight_layout()
    return fig
```

### Cumulative IRF

```python
def plot_cumulative_irf(irf, impulse=None, response=None):
    """Plot cumulative impulse response functions."""
    fig = irf.plot_cum_effects(impulse=impulse, response=response)
    return fig
```

### IRF Confidence Intervals (Bootstrap)

```python
def irf_with_ci(var_result, periods=20, runs=100, seed=42):
    """
    Compute IRF with bootstrap confidence intervals.

    Returns
    -------
    IRAnalysis with confidence bands
    """
    import numpy as np

    np.random.seed(seed)
    irf = var_result.irf(periods=periods)

    # Bootstrap
    irf_stderr = var_result.irf_errband_mc(
        orth=True,
        repl=runs,
        steps=periods,
        seed=seed
    )

    return irf, irf_stderr
```

## Forecast Error Variance Decomposition (FEVD)

FEVD shows the proportion of forecast error variance attributable to each shock.

```python
def compute_fevd(var_result, periods=20):
    """
    Compute forecast error variance decomposition.

    Returns
    -------
    FEVD object
    """
    fevd = var_result.fevd(periods=periods)
    return fevd


def plot_fevd(fevd, figsize=(12, 8)):
    """Plot FEVD for all variables."""
    import matplotlib.pyplot as plt

    fig = fevd.plot(figsize=figsize)
    plt.tight_layout()
    return fig


def fevd_table(fevd, periods=[1, 5, 10, 20]):
    """
    Create FEVD summary table for selected horizons.

    Returns
    -------
    dict of DataFrames (one per variable)
    """
    import pandas as pd

    tables = {}
    decomp = fevd.decomp

    for i, var in enumerate(fevd.names):
        data = {h: decomp[h-1, :, i] * 100 for h in periods if h <= fevd.periods}
        df = pd.DataFrame(data, index=fevd.names).T
        df.index.name = 'Horizon'
        tables[var] = df

    return tables
```

## Model Diagnostics

### Residual Analysis

```python
def var_diagnostics(var_result):
    """
    Comprehensive VAR diagnostic tests.

    Returns
    -------
    dict with all diagnostic results
    """
    from statsmodels.stats.stattools import durbin_watson

    diagnostics = {}

    # Durbin-Watson for each equation
    dw = var_result.test_whiteness(nlags=var_result.k_ar + 1, signif=0.05)
    diagnostics['whiteness_test'] = {
        'statistic': dw.test_statistic,
        'p_value': dw.pvalue,
        'passed': dw.pvalue > 0.05
    }

    # Normality test (multivariate)
    norm = var_result.test_normality()
    diagnostics['normality_test'] = {
        'statistic': norm.test_statistic,
        'p_value': norm.pvalue,
        'passed': norm.pvalue > 0.05
    }

    # Stability
    diagnostics['stability'] = check_var_stability(var_result)

    return diagnostics
```

### Portmanteau Test

```python
def portmanteau_test(var_result, lags=10):
    """
    Portmanteau test for residual autocorrelation.
    """
    result = var_result.test_whiteness(nlags=lags)

    return {
        'statistic': result.test_statistic,
        'p_value': result.pvalue,
        'df': result.df,
        'conclusion': 'No autocorrelation' if result.pvalue > 0.05 else 'Autocorrelation detected'
    }
```

## Forecasting

```python
def var_forecast(var_result, steps=10, alpha=0.05):
    """
    Generate VAR forecasts with confidence intervals.

    Parameters
    ----------
    var_result : VARResults
        Fitted VAR model
    steps : int
        Forecast horizon
    alpha : float
        Significance level

    Returns
    -------
    dict with forecasts and intervals
    """
    import numpy as np

    # Point forecasts
    forecast = var_result.forecast(var_result.endog[-var_result.k_ar:], steps=steps)

    # Forecast error bands
    stderr = var_result.forecast_interval(
        var_result.endog[-var_result.k_ar:],
        steps=steps,
        alpha=alpha
    )

    return {
        'mean': forecast,
        'lower': stderr[1],
        'upper': stderr[2]
    }
```

## Structural VAR (SVAR)

Standard VAR cannot identify structural shocks. SVAR imposes restrictions for identification.

### Short-Run Restrictions (A-B Model)

```python
from statsmodels.tsa.vector_ar.svar_model import SVAR

def fit_svar(data, p, A_matrix=None, B_matrix=None):
    """
    Fit Structural VAR with short-run restrictions.

    Parameters
    ----------
    A_matrix : array-like
        A matrix (contemporaneous relationships)
        Use np.nan for parameters to estimate
    B_matrix : array-like
        B matrix (structural shock variance)
    """
    model = SVAR(data, svar_type='AB', A=A_matrix, B=B_matrix)
    result = model.fit(maxlags=p)

    return result
```

### Cholesky Identification

Recursive ordering assumes lower-triangular contemporaneous effects:

```python
def cholesky_identification(var_result):
    """
    Apply Cholesky decomposition for structural identification.

    Note: Results depend on variable ordering!
    """
    import numpy as np

    sigma = var_result.sigma_u
    P = np.linalg.cholesky(sigma)

    return P
```

## Causal Interpretation Warning

**Granger Causality Is NOT Causality**

Granger causality tests whether X helps predict Y. It does NOT establish:
1. X causes Y in any meaningful sense
2. Intervening on X would change Y
3. A causal mechanism exists

**Why Granger Causality Can Be Misleading:**
- Common confounders can create spurious Granger causality
- Timing issues (X might just react faster than Y to common cause)
- Omitted variables bias
- Feedback loops

**For True Causal Analysis:**
- Use difference-in-differences
- Consider synthetic control
- Apply instrumental variables
- Leverage natural experiments

## References

- Lutkepohl, H. (2005). New Introduction to Multiple Time Series Analysis. Springer.
- Sims, C. (1980). Macroeconomics and Reality. Econometrica.
- Stock, J.H. & Watson, M.W. (2001). Vector Autoregressions. Journal of Economic Perspectives.
