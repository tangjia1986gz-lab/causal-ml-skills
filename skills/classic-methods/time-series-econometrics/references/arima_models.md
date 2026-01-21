# ARIMA Models: Specification, Estimation, and Forecasting

## Model Components

### Autoregressive (AR) Process

An AR(p) process:
```
Y_t = c + φ₁Y_{t-1} + φ₂Y_{t-2} + ... + φ_pY_{t-p} + ε_t
```

**Properties**:
- Current value depends on past values
- ACF: Geometric decay
- PACF: Cutoff after lag p
- Stationarity requires roots outside unit circle

### Moving Average (MA) Process

An MA(q) process:
```
Y_t = μ + ε_t + θ₁ε_{t-1} + θ₂ε_{t-2} + ... + θ_qε_{t-q}
```

**Properties**:
- Current value depends on past shocks
- ACF: Cutoff after lag q
- PACF: Geometric decay
- Always stationary (finite order)
- Invertibility requires roots outside unit circle

### ARMA Process

Combined AR and MA: ARMA(p, q)
```
Y_t = c + Σφ_iY_{t-i} + ε_t + Σθ_jε_{t-j}
```

### ARIMA Process

ARIMA(p, d, q) = ARMA(p, q) on dth difference:
```
(1 - Σφ_iL^i)(1 - L)^d Y_t = c + (1 + Σθ_jL^j)ε_t
```

Where:
- p = AR order
- d = differencing order (integration)
- q = MA order
- L = lag operator

## Model Identification

### Box-Jenkins Methodology

1. **Identification**: Determine p, d, q from data
2. **Estimation**: Fit model parameters
3. **Diagnostic Checking**: Validate model adequacy
4. **Forecasting**: Generate predictions

### Identification from ACF/PACF

| Model | ACF Pattern | PACF Pattern |
|-------|-------------|--------------|
| AR(p) | Geometric decay | Cutoff after lag p |
| MA(q) | Cutoff after lag q | Geometric decay |
| ARMA(p,q) | Decay after lag q | Decay after lag p |

### Information Criteria

```python
def select_arima_order(series, max_p=5, max_d=2, max_q=5, criterion='aic'):
    """
    Select ARIMA order using information criteria.

    Parameters
    ----------
    series : array-like
        Time series data
    max_p, max_d, max_q : int
        Maximum orders to search
    criterion : str
        'aic', 'bic', or 'hqic'

    Returns
    -------
    tuple : (p, d, q) optimal order
    """
    from statsmodels.tsa.arima.model import ARIMA
    import numpy as np
    import warnings

    # First determine d
    d = determine_integration_order(series, max_d)

    best_score = np.inf
    best_order = (0, d, 0)

    for p in range(max_p + 1):
        for q in range(max_q + 1):
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    model = ARIMA(series, order=(p, d, q))
                    result = model.fit()

                    if criterion == 'aic':
                        score = result.aic
                    elif criterion == 'bic':
                        score = result.bic
                    else:
                        score = result.hqic

                    if score < best_score:
                        best_score = score
                        best_order = (p, d, q)
            except:
                continue

    return best_order, best_score
```

### Auto ARIMA (pmdarima)

```python
import pmdarima as pm

def auto_arima(series, seasonal=False, m=1, **kwargs):
    """
    Automatic ARIMA model selection.

    Parameters
    ----------
    series : array-like
        Time series data
    seasonal : bool
        Include seasonal component
    m : int
        Seasonal period (e.g., 12 for monthly)
    """
    model = pm.auto_arima(
        series,
        seasonal=seasonal,
        m=m,
        stepwise=True,
        suppress_warnings=True,
        error_action='ignore',
        **kwargs
    )

    return model
```

## Seasonal ARIMA (SARIMA)

### Model Specification

SARIMA(p, d, q)(P, D, Q)[m]:
```
Φ_P(L^m)φ_p(L)(1-L)^d(1-L^m)^D Y_t = c + Θ_Q(L^m)θ_q(L)ε_t
```

Where:
- (p, d, q) = Non-seasonal orders
- (P, D, Q) = Seasonal orders
- m = Seasonal period

### Example: Monthly Data with Annual Seasonality

```python
from statsmodels.tsa.statespace.sarimax import SARIMAX

def fit_sarima(series, order, seasonal_order, exog=None):
    """
    Fit SARIMA model.

    Parameters
    ----------
    order : tuple
        (p, d, q) non-seasonal orders
    seasonal_order : tuple
        (P, D, Q, m) seasonal orders

    Returns
    -------
    SARIMAXResults
    """
    model = SARIMAX(
        series,
        order=order,
        seasonal_order=seasonal_order,
        exog=exog,
        enforce_stationarity=True,
        enforce_invertibility=True
    )

    return model.fit(disp=False)
```

## Estimation

### Maximum Likelihood Estimation

```python
from statsmodels.tsa.arima.model import ARIMA

def fit_arima(series, order, trend='c'):
    """
    Fit ARIMA model via MLE.

    Parameters
    ----------
    series : array-like
        Time series data
    order : tuple
        (p, d, q) model order
    trend : str
        'n' = no trend
        'c' = constant only
        't' = linear trend
        'ct' = constant + trend

    Returns
    -------
    ARIMAResults object
    """
    model = ARIMA(series, order=order, trend=trend)
    result = model.fit()

    return result
```

### Estimation Output Interpretation

```python
def summarize_arima(result):
    """Extract key information from ARIMA results."""
    summary = {
        'order': result.specification['order'],
        'aic': result.aic,
        'bic': result.bic,
        'log_likelihood': result.llf,
        'coefficients': result.params.to_dict(),
        'std_errors': result.bse.to_dict(),
        'p_values': result.pvalues.to_dict(),
        'residual_variance': result.resid.var()
    }

    return summary
```

## Diagnostic Checking

### Residual Analysis

Good model residuals should be:
1. White noise (no autocorrelation)
2. Normally distributed (optional)
3. Homoskedastic

```python
def arima_diagnostics(result, lags=20):
    """
    Comprehensive ARIMA diagnostic tests.

    Returns
    -------
    dict with diagnostic results
    """
    from statsmodels.stats.diagnostic import acorr_ljungbox
    from scipy import stats

    resid = result.resid

    # Ljung-Box test for autocorrelation
    lb_result = acorr_ljungbox(resid, lags=lags, return_df=True)

    # Jarque-Bera test for normality
    jb_stat, jb_pvalue, skew, kurt = stats.jarque_bera(resid)

    # Heteroskedasticity (ARCH test)
    from statsmodels.stats.diagnostic import het_arch
    arch_stat, arch_pvalue, _, _ = het_arch(resid, nlags=5)

    diagnostics = {
        'ljung_box': {
            'statistic': lb_result['lb_stat'].iloc[-1],
            'p_value': lb_result['lb_pvalue'].iloc[-1],
            'passed': lb_result['lb_pvalue'].iloc[-1] > 0.05
        },
        'jarque_bera': {
            'statistic': jb_stat,
            'p_value': jb_pvalue,
            'passed': jb_pvalue > 0.05
        },
        'arch_test': {
            'statistic': arch_stat,
            'p_value': arch_pvalue,
            'passed': arch_pvalue > 0.05
        }
    }

    return diagnostics
```

### Visual Diagnostics

```python
def plot_arima_diagnostics(result, figsize=(12, 10)):
    """Plot ARIMA diagnostic plots."""
    import matplotlib.pyplot as plt
    from statsmodels.graphics.tsaplots import plot_acf
    from scipy import stats

    fig, axes = plt.subplots(2, 2, figsize=figsize)
    resid = result.resid

    # Standardized residuals
    axes[0, 0].plot(resid / resid.std())
    axes[0, 0].set_title('Standardized Residuals')
    axes[0, 0].axhline(y=0, color='r', linestyle='--')

    # Histogram + KDE
    resid.plot(kind='hist', density=True, ax=axes[0, 1], bins=30)
    x = np.linspace(resid.min(), resid.max(), 100)
    axes[0, 1].plot(x, stats.norm.pdf(x, resid.mean(), resid.std()), 'r-')
    axes[0, 1].set_title('Histogram + Normal')

    # Q-Q plot
    stats.probplot(resid, dist="norm", plot=axes[1, 0])
    axes[1, 0].set_title('Q-Q Plot')

    # ACF of residuals
    plot_acf(resid, lags=30, ax=axes[1, 1])
    axes[1, 1].set_title('Residual ACF')

    plt.tight_layout()
    return fig
```

## Forecasting

### Point Forecasts and Intervals

```python
def forecast_arima(result, steps, alpha=0.05):
    """
    Generate forecasts with confidence intervals.

    Parameters
    ----------
    result : ARIMAResults
        Fitted model
    steps : int
        Forecast horizon
    alpha : float
        Significance level for intervals

    Returns
    -------
    DataFrame with forecasts and intervals
    """
    import pandas as pd

    forecast = result.get_forecast(steps=steps)
    mean = forecast.predicted_mean
    conf_int = forecast.conf_int(alpha=alpha)

    df = pd.DataFrame({
        'forecast': mean,
        'lower': conf_int.iloc[:, 0],
        'upper': conf_int.iloc[:, 1]
    })

    return df
```

### Out-of-Sample Validation

```python
def rolling_forecast_validation(series, order, window, horizon=1):
    """
    Rolling window out-of-sample validation.

    Returns
    -------
    dict with MAE, RMSE, and forecasts
    """
    import numpy as np
    from statsmodels.tsa.arima.model import ARIMA

    n = len(series)
    forecasts = []
    actuals = []

    for i in range(window, n - horizon + 1):
        train = series[:i]
        actual = series[i:i + horizon]

        try:
            model = ARIMA(train, order=order)
            result = model.fit()
            pred = result.forecast(steps=horizon)

            forecasts.append(pred.values)
            actuals.append(actual.values)
        except:
            continue

    forecasts = np.array(forecasts).flatten()
    actuals = np.array(actuals).flatten()

    mae = np.mean(np.abs(forecasts - actuals))
    rmse = np.sqrt(np.mean((forecasts - actuals) ** 2))

    return {
        'mae': mae,
        'rmse': rmse,
        'forecasts': forecasts,
        'actuals': actuals
    }
```

## Practical Considerations

### Handling Missing Data
- Interpolate for small gaps
- Use state space models for complex patterns
- Consider multiple imputation

### Dealing with Outliers
- Identify with intervention analysis
- Model as temporary or permanent changes
- Robust estimation methods

### Model Comparison

```python
def compare_arima_models(series, orders_list, criterion='aic'):
    """
    Compare multiple ARIMA specifications.

    Parameters
    ----------
    series : array-like
        Time series data
    orders_list : list of tuples
        List of (p, d, q) orders to compare

    Returns
    -------
    DataFrame with comparison metrics
    """
    import pandas as pd
    from statsmodels.tsa.arima.model import ARIMA

    results = []

    for order in orders_list:
        try:
            model = ARIMA(series, order=order)
            fit = model.fit()

            results.append({
                'order': order,
                'aic': fit.aic,
                'bic': fit.bic,
                'hqic': fit.hqic,
                'log_likelihood': fit.llf,
                'n_params': fit.df_model
            })
        except:
            continue

    df = pd.DataFrame(results)
    df = df.sort_values(criterion)

    return df
```

## Common Pitfalls

1. **Over-differencing**: Check for negative ACF at lag 1 after differencing
2. **Over-fitting**: Use parsimony principle, prefer BIC over AIC
3. **Ignoring seasonality**: Always check for seasonal patterns
4. **Unit root in MA**: Ensure invertibility
5. **Short samples**: Need sufficient data for reliable estimation

## References

- Box, G.E.P., Jenkins, G.M., Reinsel, G.C., & Ljung, G.M. (2015). Time Series Analysis: Forecasting and Control. Wiley.
- Hyndman, R.J. & Athanasopoulos, G. (2021). Forecasting: Principles and Practice. OTexts.
