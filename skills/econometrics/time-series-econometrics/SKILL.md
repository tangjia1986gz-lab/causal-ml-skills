---
name: time-series-econometrics
description: Time series analysis for economic and financial data. Use when analyzing single-unit temporal data including forecasting, volatility modeling, and dynamic relationships. Provides ARIMA, VAR, GARCH, unit root tests, cointegration, and Granger causality via statsmodels.
license: MIT
metadata:
    skill-author: Causal-ML-Skills
---

# Time Series Econometrics

## Overview

Time series econometrics analyzes data observed sequentially over time for a single unit (country, firm, market). Unlike panel data (multiple units), time series focuses on temporal dynamics: trends, seasonality, autocorrelation, and cross-series relationships.

**Key applications:**
- Macroeconomic forecasting (GDP, inflation, unemployment)
- Financial volatility modeling (stock returns, exchange rates)
- Policy analysis (monetary policy transmission)
- Causal inference in time series (Granger causality)

**Primary package**: `statsmodels` (tsa module)

## When to Use This Skill

This skill should be used when:

- Analyzing single-unit data over many time periods (T > 50)
- Forecasting future values of economic/financial variables
- Modeling volatility clustering (GARCH)
- Testing for unit roots and stationarity
- Analyzing long-run relationships (cointegration)
- Testing for Granger causality between variables
- Building VAR models for impulse response analysis

**Do NOT use when:**
- Multiple cross-sectional units with time dimension (use `panel-data-models`)
- Treatment effects with panel data (use `estimator-did`)
- Very short time series (T < 30)
- Cross-sectional data only

## Quick Start Guide

### ARIMA Model

```python
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
import matplotlib.pyplot as plt

# Load time series data
# Example: quarterly GDP growth
np.random.seed(42)
n = 100
y = np.cumsum(np.random.randn(n) * 0.5) + 0.1 * np.arange(n)
dates = pd.date_range('2000-01-01', periods=n, freq='Q')
ts = pd.Series(y, index=dates)

# Step 1: Test for stationarity
adf_result = adfuller(ts, autolag='AIC')
print(f"ADF Statistic: {adf_result[0]:.4f}")
print(f"P-value: {adf_result[1]:.4f}")
print(f"Critical Values: {adf_result[4]}")

# If non-stationary, difference the series
if adf_result[1] > 0.05:
    ts_diff = ts.diff().dropna()
    print("Series is non-stationary, using first difference")
else:
    ts_diff = ts
    print("Series is stationary")

# Step 2: Fit ARIMA model
# ARIMA(p, d, q) where p=AR order, d=differencing, q=MA order
model = ARIMA(ts, order=(1, 1, 1))  # ARIMA(1,1,1)
result = model.fit()

print(result.summary())

# Step 3: Diagnostic checks
# Ljung-Box test for residual autocorrelation
from statsmodels.stats.diagnostic import acorr_ljungbox
lb_test = acorr_ljungbox(result.resid, lags=[10], return_df=True)
print(f"\nLjung-Box test (lag 10): stat={lb_test['lb_stat'].values[0]:.2f}, p={lb_test['lb_pvalue'].values[0]:.4f}")

# Step 4: Forecast
forecast = result.get_forecast(steps=8)
print(f"\nForecast (next 8 periods):")
print(forecast.predicted_mean)
```

### VAR Model (Vector Autoregression)

```python
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import grangercausalitytests

# Simulate bivariate system
np.random.seed(42)
n = 200
e1, e2 = np.random.randn(2, n) * 0.5

# y1 Granger-causes y2
y1 = np.zeros(n)
y2 = np.zeros(n)
for t in range(2, n):
    y1[t] = 0.5 * y1[t-1] + e1[t]
    y2[t] = 0.3 * y2[t-1] + 0.4 * y1[t-1] + e2[t]  # y1 causes y2

df = pd.DataFrame({'y1': y1, 'y2': y2})

# Fit VAR model
model = VAR(df)

# Select lag order using information criteria
lag_order = model.select_order(maxlags=10)
print("Lag Order Selection:")
print(lag_order.summary())

# Fit with optimal lag
result = model.fit(lag_order.aic)
print(f"\nVAR({result.k_ar}) fitted")
print(result.summary())

# Granger causality test
print("\n--- Granger Causality Tests ---")
# Test: does y1 Granger-cause y2?
gc_y1_to_y2 = result.test_causality('y2', ['y1'], kind='f')
print(f"y1 -> y2: F={gc_y1_to_y2.test_statistic:.4f}, p={gc_y1_to_y2.pvalue:.4f}")

# Test: does y2 Granger-cause y1?
gc_y2_to_y1 = result.test_causality('y1', ['y2'], kind='f')
print(f"y2 -> y1: F={gc_y2_to_y1.test_statistic:.4f}, p={gc_y2_to_y1.pvalue:.4f}")

# Impulse Response Function
irf = result.irf(periods=20)
# irf.plot()  # Uncomment to visualize
```

### GARCH Model (Volatility)

```python
from arch import arch_model
import numpy as np
import pandas as pd

# Simulate returns with volatility clustering
np.random.seed(42)
n = 1000
returns = np.zeros(n)
sigma = np.zeros(n)
sigma[0] = 0.01

for t in range(1, n):
    sigma[t] = np.sqrt(0.00001 + 0.1 * returns[t-1]**2 + 0.85 * sigma[t-1]**2)
    returns[t] = sigma[t] * np.random.randn()

returns = pd.Series(returns * 100)  # Scale to percentage

# Fit GARCH(1,1) model
model = arch_model(returns, vol='Garch', p=1, q=1, mean='Constant')
result = model.fit(disp='off')

print(result.summary())

# Extract conditional volatility
cond_vol = result.conditional_volatility

# Forecast volatility
forecast = result.forecast(horizon=5)
print(f"\nVolatility forecast (5 periods):")
print(np.sqrt(forecast.variance.dropna()))
```

### Unit Root Tests

```python
from statsmodels.tsa.stattools import adfuller, kpss

def unit_root_tests(series, name='Series'):
    """Comprehensive unit root testing."""
    print(f"\n{'='*50}")
    print(f"Unit Root Tests for: {name}")
    print('='*50)

    # ADF Test (H0: unit root exists)
    adf = adfuller(series, autolag='AIC')
    print(f"\nADF Test:")
    print(f"  Statistic: {adf[0]:.4f}")
    print(f"  P-value: {adf[1]:.4f}")
    print(f"  Lags used: {adf[2]}")
    print(f"  Critical values: 1%={adf[4]['1%']:.3f}, 5%={adf[4]['5%']:.3f}, 10%={adf[4]['10%']:.3f}")

    if adf[1] < 0.05:
        print("  -> Reject H0: Series is STATIONARY")
    else:
        print("  -> Fail to reject H0: Series has UNIT ROOT")

    # KPSS Test (H0: series is stationary)
    kpss_result = kpss(series, regression='c', nlags='auto')
    print(f"\nKPSS Test:")
    print(f"  Statistic: {kpss_result[0]:.4f}")
    print(f"  P-value: {kpss_result[1]:.4f}")
    print(f"  Critical values: 1%={kpss_result[3]['1%']:.3f}, 5%={kpss_result[3]['5%']:.3f}")

    if kpss_result[1] < 0.05:
        print("  -> Reject H0: Series is NON-STATIONARY")
    else:
        print("  -> Fail to reject H0: Series is STATIONARY")

    # Combined interpretation
    print(f"\nCombined Interpretation:")
    if adf[1] < 0.05 and kpss_result[1] >= 0.05:
        print("  Both tests agree: Series is STATIONARY")
    elif adf[1] >= 0.05 and kpss_result[1] < 0.05:
        print("  Both tests agree: Series has UNIT ROOT")
    else:
        print("  Tests disagree: Further investigation needed")

    return {'adf': adf, 'kpss': kpss_result}

# Example usage
np.random.seed(42)
stationary = np.random.randn(200)
random_walk = np.cumsum(np.random.randn(200))

unit_root_tests(stationary, 'Stationary Process')
unit_root_tests(random_walk, 'Random Walk')
```

### Cointegration Test

```python
from statsmodels.tsa.stattools import coint
from statsmodels.tsa.vector_ar.vecm import coint_johansen

# Simulate cointegrated series
np.random.seed(42)
n = 300

# Common stochastic trend
trend = np.cumsum(np.random.randn(n))

# Two series cointegrated with trend
y1 = trend + np.random.randn(n) * 0.5
y2 = 0.5 * trend + np.random.randn(n) * 0.5

# Engle-Granger two-step test
print("--- Engle-Granger Cointegration Test ---")
coint_stat, pvalue, crit = coint(y1, y2)
print(f"Test statistic: {coint_stat:.4f}")
print(f"P-value: {pvalue:.4f}")
print(f"Critical values: 1%={crit[0]:.3f}, 5%={crit[1]:.3f}, 10%={crit[2]:.3f}")

if pvalue < 0.05:
    print("-> Reject H0: Series ARE cointegrated")
else:
    print("-> Fail to reject H0: No cointegration")

# Johansen test (for multiple series)
print("\n--- Johansen Cointegration Test ---")
df = pd.DataFrame({'y1': y1, 'y2': y2})
johansen_result = coint_johansen(df, det_order=0, k_ar_diff=1)

print("Trace statistic:")
for i, (trace, cv) in enumerate(zip(johansen_result.lr1, johansen_result.cvt[:, 1])):
    print(f"  r <= {i}: trace={trace:.4f}, 5% CV={cv:.4f}, {'reject' if trace > cv else 'fail to reject'}")

print("\nMax eigenvalue statistic:")
for i, (max_eig, cv) in enumerate(zip(johansen_result.lr2, johansen_result.cvm[:, 1])):
    print(f"  r <= {i}: max_eig={max_eig:.4f}, 5% CV={cv:.4f}, {'reject' if max_eig > cv else 'fail to reject'}")
```

## Core Capabilities

### 1. Univariate Models

| Model | Use Case | statsmodels |
|-------|----------|-------------|
| AR(p) | Autoregressive patterns | `AutoReg` |
| MA(q) | Moving average errors | `ARIMA(0,0,q)` |
| ARMA(p,q) | Both AR and MA | `ARIMA(p,0,q)` |
| ARIMA(p,d,q) | Non-stationary data | `ARIMA` |
| SARIMA | Seasonal patterns | `SARIMAX` |

### 2. Multivariate Models

| Model | Use Case | statsmodels |
|-------|----------|-------------|
| VAR(p) | Multiple series dynamics | `VAR` |
| VECM | Cointegrated systems | `VECM` |
| Structural VAR | Identified shocks | `SVAR` |

### 3. Volatility Models

| Model | Use Case | Package |
|-------|----------|---------|
| GARCH(p,q) | Volatility clustering | `arch` |
| EGARCH | Asymmetric volatility | `arch` |
| GJR-GARCH | Leverage effects | `arch` |

### 4. Key Tests

| Test | H0 | Use Case |
|------|-----|----------|
| ADF | Unit root exists | Stationarity |
| KPSS | Series is stationary | Stationarity |
| Engle-Granger | No cointegration | Long-run relationship |
| Johansen | Cointegration rank | Multiple series |
| Ljung-Box | No autocorrelation | Residual diagnostics |
| ARCH-LM | No ARCH effects | Volatility |

## Common Workflows

### Workflow 1: Forecasting Pipeline

```
1. Data Preparation
   ├── Load and visualize series
   ├── Check for missing values
   └── Plot ACF/PACF

2. Stationarity Testing
   ├── ADF test
   ├── KPSS test
   └── Determine differencing order (d)

3. Model Identification
   ├── Examine ACF/PACF of stationary series
   ├── Use information criteria (AIC, BIC)
   └── Consider seasonal patterns

4. Estimation
   ├── Fit candidate models
   ├── Compare information criteria
   └── Select best model

5. Diagnostics
   ├── Ljung-Box test on residuals
   ├── Check residual normality
   └── Inspect residual ACF

6. Forecasting
   ├── Generate point forecasts
   ├── Compute confidence intervals
   └── Evaluate out-of-sample performance
```

### Workflow 2: VAR Analysis

```
1. Data Preparation
   ├── Check stationarity of all series
   ├── Transform if needed (log, difference)
   └── Align time indices

2. Lag Selection
   ├── Estimate VAR for different lags
   ├── Compare AIC, BIC, HQIC
   └── Consider theory

3. Estimation
   └── Fit VAR(p) with optimal lag

4. Specification Tests
   ├── Portmanteau test (residual autocorrelation)
   ├── Normality test
   └── Stability (eigenvalues inside unit circle)

5. Granger Causality
   ├── Test pairwise causality
   └── Interpret with caution (not true causality!)

6. Impulse Response
   ├── Compute IRFs
   ├── Bootstrap confidence bands
   └── Forecast error variance decomposition
```

## Best Practices

### Data Preparation

1. **Check frequency consistency**: Ensure regular time intervals
2. **Handle missing values**: Interpolation or model-based imputation
3. **Log transform**: For multiplicative series (GDP, prices)
4. **Seasonal adjustment**: Use X-13 or STL if needed

### Model Selection

1. **Parsimony**: Prefer simpler models (lower AIC/BIC)
2. **Theory-guided**: Let economics guide lag selection
3. **Stability check**: Ensure roots outside unit circle
4. **Out-of-sample**: Validate with holdout data

### Inference

1. **Stationarity first**: Non-stationary series → spurious regression
2. **HAC standard errors**: Newey-West for autocorrelation
3. **Bootstrap**: For small samples or non-standard distributions
4. **Cointegration**: If unit roots, test for cointegration

### Forecasting

1. **Rolling window**: Re-estimate as new data arrives
2. **Forecast combination**: Average multiple models
3. **Evaluate properly**: Use RMSE, MAE, MAPE on holdout
4. **Uncertainty**: Always report confidence intervals

## Reference Documentation

### references/arima_models.md
- Box-Jenkins methodology
- ACF/PACF interpretation
- Model selection criteria
- Seasonal ARIMA

### references/unit_roots.md
- ADF test details
- KPSS test details
- Phillips-Perron test
- Structural breaks

### references/var_models.md
- VAR specification
- Granger causality
- Impulse response functions
- Forecast error variance decomposition

### references/cointegration.md
- Engle-Granger two-step
- Johansen procedure
- VECM specification
- Error correction interpretation

### references/volatility_models.md
- ARCH/GARCH specification
- News impact curves
- Asymmetric GARCH
- Volatility forecasting

## Common Pitfalls to Avoid

1. **Spurious regression**: Regressing non-stationary series without cointegration
2. **Over-differencing**: Making stationary series non-invertible
3. **Ignoring seasonality**: Leads to poor forecasts and biased tests
4. **Too many lags**: Overfitting, poor out-of-sample performance
5. **Granger ≠ True causality**: Predictability, not causal effect
6. **Ignoring structural breaks**: Invalidates unit root tests
7. **Using levels when differences needed**: Biased inference
8. **Forgetting to check residuals**: Model misspecification
9. **Point forecasts only**: Always report uncertainty
10. **In-sample R² for evaluation**: Use out-of-sample metrics

## Troubleshooting

### ADF test gives conflicting results with KPSS

**Possible causes:**
- Near-unit-root process
- Structural break in series
- Trend-stationary vs difference-stationary

**Solution:**
- Use both tests with their proper null hypotheses
- Consider structural break tests (Zivot-Andrews)
- Examine the series visually

### ARIMA model has poor forecasts

**Possible causes:**
- Structural change in series
- Wrong model order
- Outliers affecting estimation

**Solution:**
- Use rolling window estimation
- Try auto_arima for order selection
- Check for outliers and level shifts

### VAR residuals are autocorrelated

**Possible causes:**
- Insufficient lag length
- Omitted variables
- Structural breaks

**Solution:**
- Increase lag order
- Add relevant exogenous variables
- Consider regime-switching models

## Additional Resources

### Official Documentation
- statsmodels TSA: https://www.statsmodels.org/stable/tsa.html
- arch package: https://arch.readthedocs.io/

### Key Textbooks
- Hamilton, J. (1994). *Time Series Analysis*
- Enders, W. (2014). *Applied Econometric Time Series*
- Lütkepohl, H. (2005). *New Introduction to Multiple Time Series Analysis*

### Key Papers
- Dickey, D.A. & Fuller, W.A. (1979). "Distribution of the Estimators for Autoregressive Time Series with a Unit Root."
- Engle, R.F. & Granger, C.W.J. (1987). "Co-Integration and Error Correction."
- Johansen, S. (1991). "Estimation and Hypothesis Testing of Cointegration Vectors."
- Bollerslev, T. (1986). "Generalized Autoregressive Conditional Heteroskedasticity."

## Installation

```bash
# Core packages
pip install statsmodels pandas numpy scipy

# For GARCH models
pip install arch

# For visualization
pip install matplotlib seaborn

# Full installation
pip install statsmodels arch pandas numpy scipy matplotlib seaborn
```

## Related Skills

| Skill | When to Use Instead |
|-------|---------------------|
| `panel-data-models` | Multiple units over time |
| `estimator-did` | Treatment effects with panel |
| `causal-ddml` | High-dimensional time series |
| `statistical-analysis` | Basic regression, not time series |
