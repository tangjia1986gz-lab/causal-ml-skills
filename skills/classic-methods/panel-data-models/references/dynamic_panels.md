# Dynamic Panel Models

## Overview

Dynamic panel models include lagged dependent variables as regressors:

$$y_{it} = \rho y_{i,t-1} + X_{it}\beta + \alpha_i + \epsilon_{it}$$

This creates endogeneity: $y_{i,t-1}$ is correlated with $\alpha_i$ by construction.

## The Problem with Standard Estimators

### Pooled OLS
- $y_{i,t-1}$ correlated with $\alpha_i$ (positive bias in $\hat{\rho}$)
- Inconsistent

### Fixed Effects (Within)
- Demeaning creates correlation between $\ddot{y}_{i,t-1}$ and $\ddot{\epsilon}_{it}$
- Nickell bias: $\hat{\rho}_{FE} \to \rho - \frac{1+\rho}{T-1}$ as $N \to \infty$
- Bias is $O(1/T)$, only vanishes as $T \to \infty$
- For typical panels (T = 5-10), bias is substantial

### First-Differencing
Same problem:
$$\Delta y_{it} = \rho \Delta y_{i,t-1} + \Delta X_{it}\beta + \Delta\epsilon_{it}$$

$\Delta y_{i,t-1} = y_{i,t-1} - y_{i,t-2}$ is correlated with $\Delta\epsilon_{it} = \epsilon_{it} - \epsilon_{i,t-1}$ through $y_{i,t-1}$ and $\epsilon_{i,t-1}$.

## Arellano-Bond (Difference GMM)

### Key Insight
Use lagged **levels** as instruments for first **differences**.

### Moment Conditions

Under sequential exogeneity $E[\epsilon_{is} | y_{i,s-1}, y_{i,s-2}, ..., X_{i1}, ..., X_{iT}] = 0$:

$$E[y_{i,t-s} \cdot \Delta\epsilon_{it}] = 0 \text{ for } s \geq 2$$

### Instrument Matrix

For $t = 3$ (first usable period after differencing):
- Instruments: $y_{i1}$

For $t = 4$:
- Instruments: $y_{i1}, y_{i2}$

For $t = T$:
- Instruments: $y_{i1}, y_{i2}, ..., y_{i,T-2}$

### GMM Estimation

**First Step**: Use identity or specific weight matrix
$$\hat{\beta}_1 = (W'ZA_1Z'W)^{-1}W'ZA_1Z'y$$

**Second Step**: Optimal weight matrix
$$A_2 = \left(\sum_i Z_i'\hat{u}_i\hat{u}_i'Z_i\right)^{-1}$$

$$\hat{\beta}_2 = (W'ZA_2Z'W)^{-1}W'ZA_2Z'y$$

### Implementation

```python
from panel_estimator import PanelEstimator

estimator = PanelEstimator(data, 'entity', 'time', 'y', ['x1', 'x2'])

# Arellano-Bond estimation
ab_result = estimator.fit_dynamic_panel(
    lags=1,
    method='arellano_bond',
    max_instruments=5  # Limit instrument count
)
print(ab_result.summary())
```

## Blundell-Bond (System GMM)

### Motivation
Difference GMM can be inefficient with:
- Persistent series ($\rho$ close to 1)
- Short panels
- Limited variation in first differences

### Additional Moment Conditions

Use lagged **differences** as instruments for **levels**:

$$E[\Delta y_{i,t-1} \cdot (\alpha_i + \epsilon_{it})] = 0$$

Requires initial conditions assumption:
$$E[\Delta y_{i2} \cdot \alpha_i] = 0$$

### Stacked System

Estimate both equations jointly:
1. First-differenced equation (instruments: lagged levels)
2. Levels equation (instruments: lagged differences)

### Implementation

```python
bb_result = estimator.fit_dynamic_panel(
    lags=1,
    method='blundell_bond'
)
```

## Instrument Proliferation

### Problem
With many time periods, instrument count explodes:
- $T = 10$ periods: potentially 36 instruments from $y$ alone
- Many weak instruments → biased estimates
- Overfitting the endogenous variables

### Solutions

1. **Collapse instruments**: Use one instrument per variable, not per period
2. **Limit lag depth**: Only use instruments up to lag $L$
3. **Principal components**: Reduce dimensionality

```python
# Limit instruments
ab_result = estimator.fit_dynamic_panel(
    lags=1,
    method='arellano_bond',
    max_instruments=3  # Only use y_{t-2}, y_{t-3}, y_{t-4}
)
```

## Specification Tests

### Arellano-Bond Serial Correlation Test

Test for serial correlation in differenced residuals:
- AR(1) in differences expected (by construction)
- AR(2) in differences should be zero if model is correct

```
H0: No second-order serial correlation in differenced residuals
```

If AR(2) is significant, instruments may be invalid.

### Sargan/Hansen Test of Overidentifying Restrictions

Tests validity of instruments:

$$S = \hat{u}'Z(Z'\hat{u}\hat{u}'Z)^{-1}Z'\hat{u} \sim \chi^2_{r-k}$$

Where:
- $r$ = number of instruments
- $k$ = number of parameters

- Low p-value: Reject validity of some instruments
- High p-value: Cannot reject validity (but weak power with many instruments)

### Difference-in-Hansen Test

Compare Sargan/Hansen statistics between nested models:
- Test subsets of instruments
- Useful for testing additional moment conditions in System GMM

## Practical Guidelines

### When to Use Dynamic Panels

1. **Theory suggests dynamics**: Adjustment costs, habit formation, learning
2. **AR coefficient is of interest**: Persistence of shocks
3. **T is small, N is large**: Nickell bias is severe

### Choosing Between AB and BB

| Criterion | Arellano-Bond | Blundell-Bond |
|-----------|---------------|---------------|
| Persistence | Low-moderate | High |
| Series variance | High | Low (levels have more info) |
| Moment conditions | Conservative | More assumptions |

### Robustness Checks

1. Compare OLS, FE, AB, BB estimates
   - OLS: upper bound on $\rho$
   - FE: lower bound on $\rho$
   - AB/BB: should be between

2. Vary instrument count
   - Estimates shouldn't change dramatically

3. Report AR(2) and Hansen tests
   - AR(2) p-value > 0.05
   - Hansen p-value > 0.05 (but not too high, suggesting weak instruments)

## Code Example: Full Dynamic Analysis

```python
import numpy as np
import pandas as pd
from panel_estimator import PanelEstimator

# Simulate dynamic panel
np.random.seed(42)
N, T = 200, 8
rho_true = 0.6
beta_true = 1.0

# Generate data
entities = np.repeat(range(N), T)
times = np.tile(range(T), N)
alpha = np.repeat(np.random.normal(0, 1, N), T)  # Entity effects
x = np.random.normal(0, 1, N * T)
epsilon = np.random.normal(0, 0.5, N * T)

# Generate y dynamically
y = np.zeros(N * T)
for t in range(T):
    idx = np.arange(N) * T + t
    if t == 0:
        y[idx] = alpha[idx] + beta_true * x[idx] + epsilon[idx]
    else:
        y[idx] = rho_true * y[idx - 1] + alpha[idx] + beta_true * x[idx] + epsilon[idx]

df = pd.DataFrame({
    'entity': entities,
    'time': times,
    'y': y,
    'x': x
})

# Estimate
estimator = PanelEstimator(df, 'entity', 'time', 'y', ['x'])

# OLS (biased upward)
# FE (biased downward with small T)
fe = estimator.fit_fixed_effects()

# Arellano-Bond
ab = estimator.fit_dynamic_panel(lags=1, method='arellano_bond')

print(f"True rho: {rho_true}")
print(f"True beta: {beta_true}")
print(f"FE estimate of x: {fe.coefficients['x']:.3f}")
print(f"AB estimate of L1.y: {ab.coefficients['L1.y']:.3f}")
print(f"AB estimate of x: {ab.coefficients['x']:.3f}")
```

## Alternative Approaches

### Bias-Corrected FE

For moderate T, apply analytical bias correction:

$$\hat{\rho}_{BC} = \hat{\rho}_{FE} + \frac{1+\hat{\rho}_{FE}}{T-1}$$

### Long-Difference Estimator

Use long differences (e.g., $y_{iT} - y_{i1}$) to reduce bias.

### Anderson-Hsiao Estimator

Simple IV using $y_{i,t-2}$ as instrument for $\Delta y_{i,t-1}$:
- Less efficient than GMM
- But simpler and more robust

## References

- Arellano, M. & Bond, S. (1991). Some Tests of Specification for Panel Data: Monte Carlo Evidence and an Application to Employment Equations. Review of Economic Studies.
- Blundell, R. & Bond, S. (1998). Initial Conditions and Moment Restrictions in Dynamic Panel Data Models. Journal of Econometrics.
- Roodman, D. (2009). How to Do xtabond2: An Introduction to Difference and System GMM in Stata. Stata Journal.
- Nickell, S. (1981). Biases in Dynamic Models with Fixed Effects. Econometrica.
