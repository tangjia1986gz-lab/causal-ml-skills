# ARIMA Models

## Overview

ARIMA (AutoRegressive Integrated Moving Average) models combine three components:
- **AR(p)**: Autoregressive - past values predict current
- **I(d)**: Integrated - differencing for stationarity
- **MA(q)**: Moving Average - past errors predict current

## Model Specification

### ARIMA(p, d, q)

```
(1 - phi_1*L - ... - phi_p*L^p)(1-L)^d * Y_t = c + (1 + theta_1*L + ... + theta_q*L^q) * epsilon_t
```

Where:
- `L` is the lag operator: L*Y_t = Y_{t-1}
- `(1-L)^d` is the d-th difference operator
- `epsilon_t ~ WN(0, sigma^2)`

### Stationarity Conditions

**AR(1)**: |phi_1| < 1
**AR(p)**: Roots of 1 - phi_1*z - ... - phi_p*z^p outside unit circle

### Invertibility Conditions

**MA(1)**: |theta_1| < 1
**MA(q)**: Roots of 1 + theta_1*z + ... + theta_q*z^q outside unit circle

## Box-Jenkins Methodology

### 1. Identification

**ACF (Autocorrelation Function)**:
- AR(p): Decays exponentially or oscillates
- MA(q): Cuts off after lag q
- ARMA: Decays exponentially

**PACF (Partial Autocorrelation Function)**:
- AR(p): Cuts off after lag p
- MA(q): Decays exponentially
- ARMA: Decays exponentially

| Pattern | ACF | PACF |
|---------|-----|------|
| AR(p) | Tails off | Cuts off at p |
| MA(q) | Cuts off at q | Tails off |
| ARMA(p,q) | Tails off | Tails off |

### 2. Estimation

```python
from statsmodels.tsa.arima.model import ARIMA

model = ARIMA(y, order=(p, d, q))
result = model.fit()
```

### 3. Diagnostics

**Ljung-Box Test**:
```
Q = T(T+2) * sum_{k=1}^{m} (rho_k^2 / (T-k))

H0: No autocorrelation up to lag m
```

**Implementation**:
```python
from statsmodels.stats.diagnostic import acorr_ljungbox
lb = acorr_ljungbox(result.resid, lags=[10, 20])
```

## Model Selection Criteria

### AIC (Akaike Information Criterion)
```
AIC = -2*logL + 2*k
```

### BIC (Bayesian Information Criterion)
```
BIC = -2*logL + k*log(T)
```

**Rule**: Choose model with lowest AIC/BIC

## Seasonal ARIMA (SARIMA)

### SARIMA(p, d, q)(P, D, Q)_s

```
Phi(L^s) * phi(L) * (1-L^s)^D * (1-L)^d * Y_t = Theta(L^s) * theta(L) * epsilon_t
```

Where:
- `s` is the seasonal period (4 for quarterly, 12 for monthly)
- Capital letters denote seasonal components

```python
from statsmodels.tsa.statespace.sarimax import SARIMAX

model = SARIMAX(y, order=(1,1,1), seasonal_order=(1,1,1,12))
result = model.fit()
```

## Forecasting

### Point Forecast
```python
forecast = result.forecast(steps=h)
```

### Confidence Intervals
```python
pred = result.get_forecast(steps=h)
ci = pred.conf_int(alpha=0.05)
```

### Forecast Evaluation

| Metric | Formula |
|--------|---------|
| MSE | mean((y - y_hat)^2) |
| RMSE | sqrt(MSE) |
| MAE | mean(|y - y_hat|) |
| MAPE | mean(|y - y_hat|/|y|) * 100 |

## Common Pitfalls

1. **Over-differencing**: d too high → non-invertible
2. **Ignoring seasonality**: Poor forecasts
3. **In-sample fit obsession**: Use out-of-sample
4. **Not checking residuals**: Model misspecification

## References

- Box, G., Jenkins, G., Reinsel, G., & Ljung, G. (2015). *Time Series Analysis*.
- Hyndman, R. & Athanasopoulos, G. (2021). *Forecasting: Principles and Practice*, 3rd ed.
