# Stationarity, Unit Roots, and Structural Breaks

## Stationarity Concepts

### Strict Stationarity
A time series {Y_t} is strictly stationary if the joint distribution of (Y_t1, Y_t2, ..., Y_tk) is identical to (Y_{t1+h}, Y_{t2+h}, ..., Y_{tk+h}) for all t, k, and h.

### Weak (Covariance) Stationarity
More practical conditions:
1. **Constant mean**: E[Y_t] = μ for all t
2. **Constant variance**: Var(Y_t) = σ² for all t
3. **Autocovariance depends only on lag**: Cov(Y_t, Y_{t-k}) = γ_k

### Why Stationarity Matters
- Non-stationary series can produce spurious regressions
- OLS assumptions break down with unit roots
- Forecasting requires stable relationships
- Most econometric theory assumes stationarity

## Unit Root Testing

### Augmented Dickey-Fuller (ADF) Test

**Model**: ΔY_t = α + βt + γY_{t-1} + Σδ_i ΔY_{t-i} + ε_t

**Hypotheses**:
- H0: γ = 0 (unit root, non-stationary)
- H1: γ < 0 (stationary)

**Test Variations**:
1. No constant, no trend: ΔY_t = γY_{t-1} + ε_t
2. Constant, no trend: ΔY_t = α + γY_{t-1} + ε_t
3. Constant and trend: ΔY_t = α + βt + γY_{t-1} + ε_t

**Lag Selection**:
- Use information criteria (AIC, BIC)
- Start with max lag, reduce until significant
- Ensure residuals are white noise

**Python Implementation**:
```python
from statsmodels.tsa.stattools import adfuller

def adf_test(series, maxlag=None, regression='c'):
    """
    Conduct ADF test for unit root.

    Parameters
    ----------
    series : array-like
        Time series data
    maxlag : int, optional
        Maximum lag for differenced terms
    regression : str
        'c' = constant only
        'ct' = constant and trend
        'n' = no constant, no trend

    Returns
    -------
    dict with test statistic, p-value, lags used, critical values
    """
    result = adfuller(series, maxlag=maxlag, regression=regression)

    return {
        'test_statistic': result[0],
        'p_value': result[1],
        'lags_used': result[2],
        'n_obs': result[3],
        'critical_values': result[4],
        'is_stationary': result[1] < 0.05
    }
```

### Phillips-Perron (PP) Test

Non-parametric correction for serial correlation:
- More robust to heteroskedasticity
- Uses Newey-West standard errors
- Same null hypothesis as ADF

```python
from statsmodels.tsa.stattools import PhillipsPerron

def pp_test(series, regression='c'):
    """Phillips-Perron unit root test."""
    result = PhillipsPerron(series, trend=regression)
    return {
        'test_statistic': result.stat,
        'p_value': result.pvalue,
        'is_stationary': result.pvalue < 0.05
    }
```

### KPSS Test

**Key Difference**: Null hypothesis is stationarity

**Model**: Y_t = ξt + r_t + ε_t, where r_t = r_{t-1} + u_t

**Hypotheses**:
- H0: Series is stationary (σ²_u = 0)
- H1: Series has unit root

**Interpretation**:
| ADF Result | KPSS Result | Conclusion |
|------------|-------------|------------|
| Reject H0  | Don't reject H0 | Stationary |
| Don't reject H0 | Reject H0 | Unit root |
| Reject H0  | Reject H0 | Trend stationary |
| Don't reject H0 | Don't reject H0 | Inconclusive |

```python
from statsmodels.tsa.stattools import kpss

def kpss_test(series, regression='c', nlags='auto'):
    """
    KPSS test for stationarity.

    Parameters
    ----------
    regression : str
        'c' = level stationarity
        'ct' = trend stationarity
    """
    stat, p_value, lags, crit = kpss(series, regression=regression, nlags=nlags)

    return {
        'test_statistic': stat,
        'p_value': p_value,
        'lags_used': lags,
        'critical_values': crit,
        'is_stationary': p_value > 0.05  # Note: reversed from ADF
    }
```

## Determining Integration Order

### Procedure
1. Test original series for unit root (ADF, PP)
2. If non-stationary, difference once and retest
3. Repeat until stationary
4. The number of differences = integration order d

```python
def determine_integration_order(series, max_d=2):
    """
    Determine integration order by successive differencing.

    Returns
    -------
    int : Integration order (0 = stationary, 1 = I(1), etc.)
    """
    import numpy as np

    current = series.copy()

    for d in range(max_d + 1):
        if d > 0:
            current = np.diff(current)

        adf_result = adf_test(current)

        if adf_result['is_stationary']:
            return d

    return max_d  # Warn: may need more differencing
```

## Structural Breaks

### Why Breaks Matter
- Unit root tests have low power with breaks
- Breaks can be mistaken for unit roots
- Policy changes, regime shifts affect series

### Zivot-Andrews Test

Tests for unit root with a single structural break:
- Break in intercept
- Break in trend
- Break in both

```python
from statsmodels.tsa.stattools import zivot_andrews

def za_test(series, regression='c', maxlag=None):
    """
    Zivot-Andrews test for unit root with structural break.

    Parameters
    ----------
    regression : str
        'c' = break in intercept
        't' = break in trend
        'ct' = break in both
    """
    result = zivot_andrews(series, regression=regression, maxlag=maxlag)

    return {
        'test_statistic': result[0],
        'p_value': result[1],
        'break_point': result[2],
        'lags_used': result[3],
        'critical_values': result[4]
    }
```

### CUSUM and CUSUMSQ Tests

Detect parameter instability over time:
- CUSUM: Tests for systematic changes
- CUSUMSQ: Tests for variance changes

### Bai-Perron Test

Multiple structural break detection:
- Identifies optimal number of breaks
- Estimates break dates
- Tests for break significance

## Visual Diagnostics

### Time Series Plot
Look for:
- Trends (upward/downward)
- Seasonality
- Variance changes
- Obvious breaks

### ACF and PACF Plots
```python
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

def plot_acf_pacf(series, lags=40, title=''):
    """Plot ACF and PACF for stationarity diagnosis."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    plot_acf(series, lags=lags, ax=axes[0])
    axes[0].set_title(f'ACF - {title}')

    plot_pacf(series, lags=lags, ax=axes[1])
    axes[1].set_title(f'PACF - {title}')

    plt.tight_layout()
    return fig
```

**ACF Patterns**:
- Slow decay: Non-stationary or AR process
- Cutoff after lag q: MA(q) process
- Alternating: Negative AR coefficient

**PACF Patterns**:
- Cutoff after lag p: AR(p) process
- Slow decay: MA process

## Best Practices

### Test Selection
1. Start with visual inspection
2. Run both ADF and KPSS for robustness
3. Use appropriate regression specification
4. Consider structural breaks if suspected

### Common Pitfalls
1. **Ignoring trends**: Include trend if present
2. **Wrong lag length**: Use information criteria
3. **Sample size**: Low power in small samples
4. **Near unit root**: Tests may be inconclusive
5. **Breaks as roots**: Test for breaks explicitly

### Reporting Results
```
Stationarity Test Results
=========================
Variable: GDP_growth
Sample: 1960Q1 - 2023Q4 (256 observations)

ADF Test (constant + trend):
  Test statistic: -3.42
  Critical values: -3.99 (1%), -3.43 (5%), -3.13 (10%)
  p-value: 0.048
  Lags used: 4
  Conclusion: Reject H0 at 5% (stationary)

KPSS Test (level):
  Test statistic: 0.32
  Critical values: 0.74 (1%), 0.46 (5%), 0.35 (10%)
  p-value: > 0.10
  Conclusion: Fail to reject H0 (stationary)

Combined Assessment: Series appears stationary
```

## References

- Hamilton, J.D. (1994). Time Series Analysis. Princeton University Press.
- Enders, W. (2014). Applied Econometric Time Series. Wiley.
- Perron, P. (1989). The Great Crash, the Oil Price Shock, and the Unit Root Hypothesis. Econometrica.
