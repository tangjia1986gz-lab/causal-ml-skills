# Common Errors in Time Series Analysis

## 1. Spurious Regression

**Error:**
```python
# Regressing one random walk on another
from statsmodels.api import OLS
result = OLS(y, X).fit()  # Both y and X are I(1)
print(f"R² = {result.rsquared:.4f}")  # Misleadingly high!
```

**Problem:** High R² and significant t-stats even when no true relationship exists.

**Correct:**
```python
# Test for cointegration first
from statsmodels.tsa.stattools import coint
stat, pvalue, _ = coint(y, X)
if pvalue < 0.05:
    # Cointegrated: estimate ECM
    pass
else:
    # Not cointegrated: use differences
    dy = y.diff().dropna()
    dX = X.diff().dropna()
    result = OLS(dy, dX).fit()
```

## 2. Ignoring Non-Stationarity

**Error:**
```python
# Fitting ARMA to non-stationary data
from statsmodels.tsa.arima.model import ARIMA
model = ARIMA(y, order=(1, 0, 1))  # d=0 but y is I(1)
```

**Correct:**
```python
# Test for unit root first
from statsmodels.tsa.stattools import adfuller
adf = adfuller(y)
if adf[1] > 0.05:
    # Non-stationary: difference or use ARIMA with d>0
    model = ARIMA(y, order=(1, 1, 1))
```

## 3. Over-Differencing

**Error:**
```python
# Differencing already stationary data
y_diff = y.diff().diff()  # Second difference of stationary series
# Creates non-invertible MA structure
```

**Signs:** MA coefficient close to -1, poor forecasts

**Correct:** Only difference until stationary. Use unit root tests to determine d.

## 4. Wrong Critical Values for Cointegration

**Error:**
```python
# Using ADF critical values for cointegration test
from statsmodels.tsa.stattools import adfuller
residuals = y - beta_hat * x
adf = adfuller(residuals)
if adf[0] < -2.86:  # Wrong! This is ADF critical value
    print("Cointegrated")
```

**Correct:**
```python
# Use cointegration-specific critical values
from statsmodels.tsa.stattools import coint
stat, pvalue, crit = coint(y, x)
# crit contains correct critical values: [-3.90, -3.34, -3.04] for 2 vars
```

## 5. Granger Causality Misinterpretation

**Error:**
```python
gc = result.test_causality('y2', ['y1'])
if gc.pvalue < 0.05:
    print("y1 CAUSES y2")  # Wrong interpretation!
```

**Correct:**
```python
if gc.pvalue < 0.05:
    print("y1 Granger-causes y2 (improves prediction)")
    # This is predictive causality, NOT structural causality!
    # Does not rule out: common cause, reverse causality, coincidence
```

## 6. Ignoring Seasonality

**Error:**
```python
# Monthly data without seasonal adjustment
model = ARIMA(monthly_sales, order=(1, 1, 1))
# Residuals show seasonal pattern
```

**Correct:**
```python
# Use SARIMA for seasonal data
from statsmodels.tsa.statespace.sarimax import SARIMAX
model = SARIMAX(monthly_sales, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
```

## 7. In-Sample Evaluation Only

**Error:**
```python
# Selecting model based only on in-sample fit
model1 = ARIMA(y, order=(5, 1, 5)).fit()
model2 = ARIMA(y, order=(1, 1, 1)).fit()
best = model1 if model1.aic < model2.aic else model2
# More complex model might overfit
```

**Correct:**
```python
# Use rolling out-of-sample evaluation
train = y[:-20]
test = y[-20:]
model = ARIMA(train, order=(1, 1, 1)).fit()
forecast = model.forecast(steps=20)
rmse = np.sqrt(((forecast - test)**2).mean())
```

## 8. Ignoring Structural Breaks

**Error:**
```python
# Unit root test without considering breaks
adf = adfuller(y)  # May not reject H0 due to break
```

**Problem:** Structural breaks can:
- Make stationary series appear non-stationary
- Make non-stationary series appear stationary
- Invalidate cointegration tests

**Correct:**
```python
# Use break-aware tests (Zivot-Andrews)
# Or split sample at known break points
```

## 9. Wrong VAR Ordering in IRF

**Error:**
```python
df = pd.DataFrame({'gdp': gdp, 'money': money})
model = VAR(df).fit()
irf = model.irf()
# Ordering assumes gdp affects money contemporaneously, not vice versa
```

**Correct:**
```python
# Choose ordering based on economic theory
# Variables affected contemporaneously by others should come later
df = pd.DataFrame({'money': money, 'gdp': gdp})  # If money affects GDP
```

## 10. Forecasting Without Uncertainty

**Error:**
```python
forecast = result.forecast(steps=12)
print(f"Forecast: {forecast}")  # Point forecasts only
```

**Correct:**
```python
pred = result.get_forecast(steps=12)
ci = pred.conf_int(alpha=0.05)
print(f"Forecast: {pred.predicted_mean}")
print(f"95% CI: [{ci.iloc[:,0]}, {ci.iloc[:,1]}]")
```

## 11. Too Many VAR Lags

**Error:**
```python
# Small sample with many lags
T = 100
k = 5  # variables
p = 10  # lags
# Parameters = k * k * p = 250, more than T!
```

**Rule of thumb:** T > k²p + 50

**Correct:**
```python
# Use information criteria with penalty for parameters
lag_order = model.select_order(maxlags=min(10, T//10))
```

## 12. GARCH on Returns in Levels

**Error:**
```python
# GARCH on price levels
from arch import arch_model
model = arch_model(prices)  # Wrong! Prices are non-stationary
```

**Correct:**
```python
# GARCH on returns (stationary)
returns = 100 * np.log(prices / prices.shift(1)).dropna()
model = arch_model(returns, vol='Garch', p=1, q=1)
```

## Summary Checklist

Before analysis:
- [ ] Checked stationarity of all series?
- [ ] Determined integration order?
- [ ] Checked for seasonality?
- [ ] Considered structural breaks?

During estimation:
- [ ] Appropriate model for data (ARIMA vs VAR vs VECM)?
- [ ] Correct critical values for tests?
- [ ] Residual diagnostics passed?

For inference:
- [ ] Granger causality interpreted correctly?
- [ ] IRF ordering justified?
- [ ] Cointegration tests valid?

For forecasting:
- [ ] Out-of-sample evaluation done?
- [ ] Confidence intervals provided?
- [ ] Model compared to alternatives?
