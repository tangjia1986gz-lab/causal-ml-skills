# Vector Autoregression (VAR) Models

## Overview

VAR models capture dynamic relationships among multiple time series. Each variable is regressed on lagged values of itself and all other variables.

## VAR(p) Specification

### Matrix Form
```
Y_t = c + A_1*Y_{t-1} + A_2*Y_{t-2} + ... + A_p*Y_{t-p} + u_t
```

Where:
- `Y_t` is k×1 vector of variables
- `A_i` are k×k coefficient matrices
- `u_t ~ N(0, Sigma)` is k×1 error vector

### Example: VAR(1) with 2 variables
```
[y1_t]   [c1]   [a11 a12] [y1_{t-1}]   [u1_t]
[y2_t] = [c2] + [a21 a22] [y2_{t-1}] + [u2_t]
```

## Estimation

### OLS Equation-by-Equation
Each equation can be estimated separately by OLS.

```python
from statsmodels.tsa.api import VAR

model = VAR(df)
result = model.fit(maxlags=p)
```

## Lag Order Selection

### Information Criteria

| Criterion | Formula | Tendency |
|-----------|---------|----------|
| AIC | -2*logL/T + 2*k/T | Over-fits |
| BIC (SIC) | -2*logL/T + k*log(T)/T | Parsimonious |
| HQIC | -2*logL/T + 2*k*log(log(T))/T | Intermediate |
| FPE | det(Sigma)*(T+np+1)/(T-np-1) | Like AIC |

```python
lag_order = model.select_order(maxlags=10)
print(lag_order.summary())
optimal_lag = lag_order.aic  # or .bic, .hqic
```

## Granger Causality

### Definition
X Granger-causes Y if past values of X help predict Y beyond past values of Y alone.

**Important**: Predictive causality, NOT true causality!

### Test
H0: Coefficients on X lags in Y equation are jointly zero

```python
gc_test = result.test_causality('y2', ['y1'], kind='f')
print(f"F-stat: {gc_test.test_statistic}, p-value: {gc_test.pvalue}")
```

### Interpretation Caution
- Granger causality ≠ structural causality
- May reflect common third factor
- Direction can flip with different variables included

## Impulse Response Functions (IRF)

### Definition
IRF shows the effect of a one-time shock to one variable on current and future values of all variables.

### Orthogonalized IRF
Uses Cholesky decomposition to orthogonalize shocks.

```python
irf = result.irf(periods=20)
irf.plot()  # Plot all IRFs
irf.plot(impulse='y1', response='y2')  # Specific pair
```

### Ordering Matters!
Cholesky decomposition assumes recursive ordering. First variable affects all contemporaneously; last variable affected by all.

## Forecast Error Variance Decomposition (FEVD)

Shows the proportion of forecast error variance of each variable attributable to shocks in each variable.

```python
fevd = result.fevd(periods=20)
fevd.plot()
```

## Structural VAR (SVAR)

### Motivation
Reduced-form VAR doesn't identify structural shocks. SVAR imposes restrictions to achieve identification.

### A-Model (Short-run restrictions)
```
A_0 * Y_t = c + A_1*Y_{t-1} + ... + epsilon_t
```

Need k(k-1)/2 restrictions on A_0 for identification.

### B-Model (Structural errors)
```
Y_t = c + A_1*Y_{t-1} + ... + B*epsilon_t
```

### Long-run Restrictions (Blanchard-Quah)
Identify shocks by their long-run effects (e.g., demand shocks have no long-run effect on output).

## Stability

### Condition
VAR(p) is stable if all eigenvalues of companion matrix lie inside unit circle.

```python
# Check stability
roots = result.roots
print(f"All roots inside unit circle: {all(np.abs(roots) < 1)}")
```

### Companion Form
```
Z_t = A*Z_{t-1} + v_t

where Z_t = [Y_t, Y_{t-1}, ..., Y_{t-p+1}]'
```

## Diagnostic Tests

### Portmanteau Test
Tests for residual autocorrelation.

```python
from statsmodels.stats.diagnostic import acorr_ljungbox
# Applied to each equation's residuals
```

### Normality Test
Jarque-Bera for each equation.

### ARCH Test
Tests for heteroskedasticity in residuals.

## Implementation Example

```python
from statsmodels.tsa.api import VAR
import pandas as pd

# Load data
df = pd.read_csv('data.csv', index_col=0, parse_dates=True)

# Fit VAR
model = VAR(df)
result = model.fit(maxlags=4, ic='aic')

# Summary
print(result.summary())

# Granger causality
gc = result.test_causality('gdp', ['money'], kind='f')

# IRF
irf = result.irf(periods=20)
irf.plot()

# Forecast
forecast = result.forecast(df.values[-result.k_ar:], steps=12)
```

## References

- Sims, C. (1980). "Macroeconomics and Reality." *Econometrica*.
- Lütkepohl, H. (2005). *New Introduction to Multiple Time Series Analysis*.
- Stock, J. & Watson, M. (2001). "Vector Autoregressions." *JEP*.
