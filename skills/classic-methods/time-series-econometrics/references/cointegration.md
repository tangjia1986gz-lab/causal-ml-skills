# Cointegration and Error Correction Models

## Overview

Cointegration addresses a fundamental problem: regressing one non-stationary series on another often produces spurious results. However, when non-stationary series share a common stochastic trend, their linear combination may be stationary, indicating a long-run equilibrium relationship.

## Spurious Regression Problem

### The Issue

When regressing I(1) series that are NOT cointegrated:
- R-squared tends toward 1
- t-statistics are inflated
- Durbin-Watson approaches 0
- Results are meaningless

### Granger-Newbold Simulations (1974)

Even independent random walks appear highly correlated:
```
Y_t = Y_{t-1} + ε_t
X_t = X_{t-1} + η_t  (independent of Y)

Regression: Y = α + βX + u
- High R², significant t-stats
- But no true relationship exists!
```

### Detection

Rule of thumb: If R² > Durbin-Watson, suspect spurious regression.

## Cointegration Concepts

### Definition

Variables X_t and Y_t (both I(1)) are cointegrated if there exists a linear combination:
```
Z_t = Y_t - βX_t  that is I(0)
```

The vector [1, -β]' is called the cointegrating vector.

### Intuition

- Both series have unit roots (permanent shocks)
- But they share a common stochastic trend
- Deviations from equilibrium are temporary
- Short-run dynamics with long-run equilibrium

### Examples

- **Purchasing Power Parity**: Exchange rate and price levels
- **Money Demand**: Money, income, and interest rates
- **Stock Prices**: Dividends and prices
- **Interest Rates**: Short and long rates

## Engle-Granger Two-Step Procedure

### Step 1: Estimate Cointegrating Regression

```python
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller, coint

def engle_granger_step1(y, x, add_constant=True):
    """
    Step 1: Estimate cointegrating regression.

    Y_t = α + βX_t + u_t

    Parameters
    ----------
    y : array-like
        Dependent variable (I(1))
    x : array-like
        Independent variable(s) (I(1))

    Returns
    -------
    dict with coefficients and residuals
    """
    import statsmodels.api as sm

    if add_constant:
        x_with_const = sm.add_constant(x)
    else:
        x_with_const = x

    model = sm.OLS(y, x_with_const)
    result = model.fit()

    return {
        'coefficients': result.params,
        'residuals': result.resid,
        'r_squared': result.rsquared,
        'model': result
    }
```

### Step 2: Test Residuals for Stationarity

```python
def engle_granger_step2(residuals, regression='c', maxlag=None):
    """
    Step 2: Test residuals for unit root.

    Uses special critical values (not standard ADF)!

    Returns
    -------
    dict with test results
    """
    result = adfuller(residuals, regression=regression, maxlag=maxlag)

    # Note: Critical values for cointegration test differ from standard ADF
    # Use MacKinnon (1990) critical values for n variables

    return {
        'test_statistic': result[0],
        'p_value': result[1],
        'lags_used': result[2],
        'critical_values': result[4],
        'is_cointegrated': result[1] < 0.05
    }
```

### Complete Engle-Granger Test

```python
def engle_granger_test(y, x, trend='c', maxlag=None):
    """
    Complete Engle-Granger cointegration test.

    Parameters
    ----------
    y : array-like
        Dependent variable
    x : array-like
        Independent variable(s)
    trend : str
        'c' = constant, 'ct' = constant + trend

    Returns
    -------
    dict with test results and cointegrating vector
    """
    # Use statsmodels' coint function
    stat, pvalue, crit = coint(y, x, trend=trend, maxlag=maxlag)

    return {
        'test_statistic': stat,
        'p_value': pvalue,
        'critical_values': dict(zip(['1%', '5%', '10%'], crit)),
        'is_cointegrated': pvalue < 0.05,
        'null_hypothesis': 'No cointegration',
        'method': 'Engle-Granger'
    }
```

## Johansen Procedure

### Advantages over Engle-Granger

1. Tests for multiple cointegrating vectors
2. Maximum likelihood estimation
3. Works with systems of variables
4. Does not require specifying dependent variable

### Model Setup

Consider a VAR(p) in k variables:
```
ΔY_t = ΠY_{t-1} + Σ Γ_i ΔY_{t-i} + ε_t
```

Where Π = αβ' contains:
- α = adjustment coefficients (k×r)
- β = cointegrating vectors (k×r)
- r = cointegration rank

### Rank Determination

The rank r (number of cointegrating relationships):
- r = 0: No cointegration, use VAR in differences
- r = k: All variables stationary, use VAR in levels
- 0 < r < k: r cointegrating vectors, use VECM

### Johansen Test Implementation

```python
from statsmodels.tsa.vector_ar.vecm import coint_johansen

def johansen_test(data, det_order=0, k_ar_diff=1):
    """
    Johansen cointegration test.

    Parameters
    ----------
    data : DataFrame
        Multivariate time series
    det_order : int
        Deterministic trend:
        -1 = no constant, no trend
        0 = constant only
        1 = constant + linear trend
    k_ar_diff : int
        Number of lagged differences

    Returns
    -------
    dict with trace and max eigenvalue test results
    """
    result = coint_johansen(data, det_order=det_order, k_ar_diff=k_ar_diff)

    k = data.shape[1]

    # Trace test
    trace_results = []
    for i in range(k):
        trace_results.append({
            'r': i,
            'eigenvalue': result.eig[i],
            'trace_statistic': result.lr1[i],
            'critical_values': {
                '10%': result.cvt[i, 0],
                '5%': result.cvt[i, 1],
                '1%': result.cvt[i, 2]
            },
            'reject_5pct': result.lr1[i] > result.cvt[i, 1]
        })

    # Max eigenvalue test
    max_eig_results = []
    for i in range(k):
        max_eig_results.append({
            'r': i,
            'eigenvalue': result.eig[i],
            'max_eig_statistic': result.lr2[i],
            'critical_values': {
                '10%': result.cvm[i, 0],
                '5%': result.cvm[i, 1],
                '1%': result.cvm[i, 2]
            },
            'reject_5pct': result.lr2[i] > result.cvm[i, 1]
        })

    # Determine rank
    rank = 0
    for i in range(k):
        if result.lr1[i] > result.cvt[i, 1]:  # 5% level
            rank = i + 1
        else:
            break

    return {
        'trace_test': trace_results,
        'max_eigenvalue_test': max_eig_results,
        'cointegration_rank': rank,
        'cointegrating_vectors': result.evec[:, :rank] if rank > 0 else None,
        'adjustment_coefficients': result.evec[:, :rank] if rank > 0 else None
    }
```

### Interpreting Johansen Results

```python
def interpret_johansen(result, variable_names):
    """
    Create readable interpretation of Johansen test.
    """
    print("Johansen Cointegration Test Results")
    print("=" * 50)

    # Trace test
    print("\nTrace Test (H0: r = r0 vs H1: r > r0)")
    print("-" * 50)
    print(f"{'r0':<5} {'Trace Stat':<15} {'5% CV':<15} {'Decision'}")
    print("-" * 50)

    for r in result['trace_test']:
        decision = "Reject" if r['reject_5pct'] else "Fail to reject"
        print(f"{r['r']:<5} {r['trace_statistic']:<15.4f} "
              f"{r['critical_values']['5%']:<15.4f} {decision}")

    print(f"\nConclusion: {result['cointegration_rank']} cointegrating relationship(s)")

    if result['cointegration_rank'] > 0:
        print("\nNormalized Cointegrating Vector(s):")
        vectors = result['cointegrating_vectors']
        for i in range(result['cointegration_rank']):
            vec = vectors[:, i] / vectors[0, i]
            print(f"  CE{i+1}: ", end="")
            for j, name in enumerate(variable_names):
                print(f"{vec[j]:.4f}*{name}", end=" ")
                if j < len(variable_names) - 1:
                    print("+ ", end="")
            print()
```

## Vector Error Correction Model (VECM)

### Model Specification

For cointegrated I(1) variables:
```
ΔY_t = αβ'Y_{t-1} + Σ Γ_i ΔY_{t-i} + ε_t
```

Where:
- β'Y_{t-1} = error correction term (deviations from equilibrium)
- α = speed of adjustment coefficients
- Γ_i = short-run dynamics

### VECM Estimation

```python
from statsmodels.tsa.vector_ar.vecm import VECM

def fit_vecm(data, coint_rank, k_ar_diff=1, deterministic='ci'):
    """
    Fit Vector Error Correction Model.

    Parameters
    ----------
    data : DataFrame
        Multivariate I(1) time series
    coint_rank : int
        Number of cointegrating relationships
    k_ar_diff : int
        Lagged differences in VECM
    deterministic : str
        'n' = no deterministic terms
        'co' = constant outside error correction
        'ci' = constant inside error correction
        'lo' = linear trend outside
        'li' = linear trend inside

    Returns
    -------
    VECMResults object
    """
    model = VECM(data, k_ar_diff=k_ar_diff, coint_rank=coint_rank,
                 deterministic=deterministic)
    result = model.fit()

    return result
```

### VECM Analysis

```python
def analyze_vecm(vecm_result, variable_names):
    """
    Extract and interpret VECM results.

    Returns
    -------
    dict with interpretable results
    """
    analysis = {}

    # Cointegrating relationships
    analysis['cointegrating_vectors'] = pd.DataFrame(
        vecm_result.beta,
        index=variable_names,
        columns=[f'CE{i+1}' for i in range(vecm_result.coint_rank)]
    )

    # Adjustment coefficients
    analysis['adjustment_coefficients'] = pd.DataFrame(
        vecm_result.alpha,
        index=variable_names,
        columns=[f'CE{i+1}' for i in range(vecm_result.coint_rank)]
    )

    # Short-run dynamics (Gamma matrices)
    analysis['short_run_dynamics'] = vecm_result.gamma

    # Summary statistics
    analysis['summary'] = {
        'log_likelihood': vecm_result.llf,
        'aic': vecm_result.aic,
        'bic': vecm_result.bic,
        'n_obs': vecm_result.nobs
    }

    return analysis
```

### Half-Life of Adjustment

How long for half the disequilibrium to be corrected:

```python
def half_life(alpha):
    """
    Calculate half-life of adjustment.

    Parameters
    ----------
    alpha : float
        Speed of adjustment coefficient (negative)

    Returns
    -------
    float : Half-life in periods
    """
    import numpy as np

    if alpha >= 0:
        return np.inf  # No mean reversion

    return -np.log(2) / np.log(1 + alpha)
```

## Forecasting with VECM

```python
def vecm_forecast(vecm_result, steps=10):
    """
    Generate forecasts from VECM.

    Returns
    -------
    array : Point forecasts
    """
    forecast = vecm_result.predict(steps=steps)
    return forecast
```

## Model Comparison

### VAR vs. VECM

| Feature | VAR in Differences | VECM |
|---------|-------------------|------|
| Long-run info | Lost | Preserved |
| Appropriate when | No cointegration | Cointegration exists |
| Forecasting | Short-horizon | Better long-horizon |
| Impulse responses | May be biased | More accurate |

### Decision Procedure

```python
def cointegration_decision(data, max_lags=10, significance=0.05):
    """
    Automated decision procedure for VAR vs VECM.

    Returns
    -------
    dict with recommendation
    """
    # 1. Test for unit roots
    unit_root_results = {}
    for col in data.columns:
        adf = adfuller(data[col])
        unit_root_results[col] = {
            'statistic': adf[0],
            'p_value': adf[1],
            'is_stationary': adf[1] < significance
        }

    all_stationary = all(r['is_stationary'] for r in unit_root_results.values())
    all_nonstationary = not any(r['is_stationary'] for r in unit_root_results.values())

    if all_stationary:
        return {
            'recommendation': 'VAR in levels',
            'reason': 'All series are stationary',
            'unit_root_tests': unit_root_results
        }

    # 2. If all I(1), test for cointegration
    if all_nonstationary:
        johansen = johansen_test(data, det_order=0, k_ar_diff=max_lags)

        if johansen['cointegration_rank'] > 0:
            return {
                'recommendation': 'VECM',
                'reason': f"Found {johansen['cointegration_rank']} cointegrating relationship(s)",
                'unit_root_tests': unit_root_results,
                'johansen_test': johansen
            }
        else:
            return {
                'recommendation': 'VAR in first differences',
                'reason': 'No cointegration found',
                'unit_root_tests': unit_root_results,
                'johansen_test': johansen
            }

    return {
        'recommendation': 'Further analysis needed',
        'reason': 'Mixed integration orders',
        'unit_root_tests': unit_root_results
    }
```

## Common Pitfalls

1. **Wrong critical values**: Engle-Granger requires special critical values
2. **Sample size**: Johansen test needs large samples
3. **Lag selection**: Incorrect lags distort results
4. **Deterministic terms**: Must match data characteristics
5. **Structural breaks**: Can create apparent cointegration
6. **Seasonal data**: May need seasonal cointegration tests

## Causal Interpretation

Cointegration establishes a **statistical equilibrium relationship**, not causation:
- Variables move together in the long run
- Error correction shows which variable "adjusts"
- But adjustment is not the same as causation
- Common factors might drive both variables

For causal claims, combine with:
- Economic theory
- Institutional knowledge
- Exogenous variation

## References

- Engle, R.F. & Granger, C.W.J. (1987). Co-integration and Error Correction. Econometrica.
- Johansen, S. (1991). Estimation and Hypothesis Testing of Cointegration Vectors. Econometrica.
- Murray, M.P. (1994). A Drunk and Her Dog: An Illustration of Cointegration and Error Correction. The American Statistician.
