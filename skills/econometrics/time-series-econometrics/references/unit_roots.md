# Unit Root Tests

## Overview

Unit root tests determine whether a time series is stationary or has a stochastic trend (unit root). Non-stationary series require differencing or other transformations before modeling.

## The Unit Root Problem

### Random Walk
```
Y_t = Y_{t-1} + epsilon_t
```

Properties:
- Variance grows with time: Var(Y_t) = t * sigma^2
- Shocks have permanent effects
- ACF decays very slowly

### Trend-Stationary vs Difference-Stationary

| Type | Model | Stationarize |
|------|-------|--------------|
| Trend-stationary | Y_t = alpha + beta*t + u_t | Detrend |
| Difference-stationary | Y_t = Y_{t-1} + epsilon_t | Difference |

## ADF Test (Augmented Dickey-Fuller)

### Test Equation
```
Delta_Y_t = alpha + beta*t + gamma*Y_{t-1} + sum_{j=1}^{p} delta_j*Delta_Y_{t-j} + epsilon_t
```

### Hypotheses
- H0: gamma = 0 (unit root)
- H1: gamma < 0 (stationary)

### Test Statistic
```
ADF = gamma_hat / SE(gamma_hat)
```

**Note**: Non-standard distribution (Dickey-Fuller distribution)

### Critical Values

| Significance | No constant | Constant | Constant + trend |
|-------------|-------------|----------|------------------|
| 1% | -2.58 | -3.43 | -3.96 |
| 5% | -1.95 | -2.86 | -3.41 |
| 10% | -1.62 | -2.57 | -3.13 |

### Implementation
```python
from statsmodels.tsa.stattools import adfuller

# With constant only
adf_c = adfuller(y, regression='c', autolag='AIC')

# With constant and trend
adf_ct = adfuller(y, regression='ct', autolag='AIC')

print(f"ADF stat: {adf_c[0]:.4f}, p-value: {adf_c[1]:.4f}")
```

## KPSS Test

### Key Difference from ADF
- H0: Series is stationary
- H1: Series has unit root

### Test Statistic
```
KPSS = (1/T^2) * sum_{t=1}^{T} S_t^2 / sigma^2_hat

where S_t = sum_{i=1}^{t} u_i (partial sum of residuals)
```

### Critical Values

| Significance | Level (c) | Trend (ct) |
|-------------|-----------|------------|
| 1% | 0.739 | 0.216 |
| 5% | 0.463 | 0.146 |
| 10% | 0.347 | 0.119 |

### Implementation
```python
from statsmodels.tsa.stattools import kpss

kpss_c = kpss(y, regression='c', nlags='auto')
print(f"KPSS stat: {kpss_c[0]:.4f}, p-value: {kpss_c[1]:.4f}")
```

## Phillips-Perron Test

Non-parametric correction for serial correlation (no lagged differences needed).

```python
from statsmodels.tsa.stattools import PhillipsPerron  # Not in statsmodels
# Alternative: use arch package
```

## Combined Testing Strategy

| ADF | KPSS | Conclusion |
|-----|------|------------|
| Reject H0 | Don't reject H0 | Stationary |
| Don't reject H0 | Reject H0 | Unit root |
| Reject H0 | Reject H0 | Trend-stationary |
| Don't reject H0 | Don't reject H0 | Inconclusive |

## Structural Breaks

Unit root tests can be misleading with structural breaks.

### Zivot-Andrews Test
Tests for unit root allowing one endogenous break.

### Perron Test
Tests with known break date.

## Determining Integration Order

**Procedure**:
1. Test level series for unit root
2. If unit root, test first difference
3. Continue until stationary
4. Integration order = number of differences needed

```python
def find_integration_order(y, max_d=3):
    """Find integration order."""
    from statsmodels.tsa.stattools import adfuller

    for d in range(max_d + 1):
        if d > 0:
            y = y.diff().dropna()

        adf = adfuller(y, autolag='AIC')
        if adf[1] < 0.05:
            return d

    return max_d
```

## Panel Unit Root Tests

For panel data:
- **LLC** (Levin-Lin-Chu): Common unit root
- **IPS** (Im-Pesaran-Shin): Heterogeneous unit roots
- **Fisher-ADF/PP**: Combines individual test p-values

## References

- Dickey, D.A. & Fuller, W.A. (1979). "Distribution of the Estimators for Autoregressive Time Series with a Unit Root." *JASA*.
- Kwiatkowski, D., Phillips, P.C.B., Schmidt, P., & Shin, Y. (1992). "Testing the Null Hypothesis of Stationarity." *Journal of Econometrics*.
- Zivot, E. & Andrews, D. (1992). "Further Evidence on the Great Crash, the Oil-Price Shock, and the Unit-Root Hypothesis." *JBES*.
