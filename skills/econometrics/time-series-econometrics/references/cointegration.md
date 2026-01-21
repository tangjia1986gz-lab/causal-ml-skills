# Cointegration

## Overview

Cointegration describes a long-run equilibrium relationship between non-stationary I(1) series. While individual series are non-stationary, their linear combination is stationary.

## Motivation

### Spurious Regression Problem
Regressing one random walk on another yields:
- High R²
- Significant t-statistics
- BUT relationship is meaningless

**Solution**: Test for cointegration. If cointegrated, regression is meaningful (long-run relationship).

## Definition

Two I(1) series Y_t and X_t are cointegrated if:
```
Z_t = Y_t - beta*X_t ~ I(0)
```

The vector [1, -beta] is the cointegrating vector.

## Engle-Granger Two-Step Procedure

### Step 1: Estimate Cointegrating Regression
```
Y_t = alpha + beta*X_t + u_t
```

Get residuals: `u_hat_t = Y_t - alpha_hat - beta_hat*X_t`

### Step 2: Test Residuals for Unit Root
```
Delta_u_hat_t = gamma*u_hat_{t-1} + sum_{j=1}^{p} delta_j*Delta_u_hat_{t-j} + epsilon_t
```

H0: gamma = 0 (no cointegration)
H1: gamma < 0 (cointegrated)

### Critical Values (Different from ADF!)

| Significance | 2 variables | 3 variables | 4 variables |
|-------------|-------------|-------------|-------------|
| 1% | -3.90 | -4.29 | -4.64 |
| 5% | -3.34 | -3.74 | -4.10 |
| 10% | -3.04 | -3.45 | -3.81 |

### Implementation
```python
from statsmodels.tsa.stattools import coint

stat, pvalue, crit = coint(y, x)
print(f"Test stat: {stat:.4f}, p-value: {pvalue:.4f}")
```

## Johansen Procedure

### Advantages over Engle-Granger
- Tests for multiple cointegrating vectors
- More efficient
- No normalization issue

### VECM Representation
```
Delta_Y_t = alpha*beta'*Y_{t-1} + sum_{j=1}^{p-1} Gamma_j*Delta_Y_{t-j} + epsilon_t

where:
- beta = cointegrating vectors (r×k)
- alpha = adjustment coefficients (k×r)
- Pi = alpha*beta' = long-run matrix (reduced rank)
```

### Rank of Pi (Number of Cointegrating Vectors)

| Rank(Pi) | Interpretation |
|----------|----------------|
| r = 0 | No cointegration, VAR in differences |
| 0 < r < k | r cointegrating vectors, VECM |
| r = k | All stationary, VAR in levels |

### Trace Test
H0: rank(Pi) <= r
H1: rank(Pi) > r

### Maximum Eigenvalue Test
H0: rank(Pi) = r
H1: rank(Pi) = r + 1

### Implementation
```python
from statsmodels.tsa.vector_ar.vecm import coint_johansen

result = coint_johansen(df, det_order=0, k_ar_diff=1)

# Trace test
print("Trace test:")
for i in range(len(result.lr1)):
    print(f"r <= {i}: stat={result.lr1[i]:.2f}, CV={result.cvt[i,1]:.2f}")

# Max eigenvalue test
print("Max eigenvalue test:")
for i in range(len(result.lr2)):
    print(f"r = {i}: stat={result.lr2[i]:.2f}, CV={result.cvm[i,1]:.2f}")
```

### Deterministic Terms

| det_order | Specification |
|-----------|---------------|
| -1 | No deterministic terms |
| 0 | Restricted constant (in cointegrating relation) |
| 1 | Unrestricted constant, restricted trend |

## Error Correction Model (ECM)

### Two-Variable Case
```
Delta_Y_t = alpha_1*(Y_{t-1} - beta*X_{t-1}) + lagged differences + epsilon_t
Delta_X_t = alpha_2*(Y_{t-1} - beta*X_{t-1}) + lagged differences + epsilon_t
```

### Interpretation
- `Y_{t-1} - beta*X_{t-1}`: Equilibrium error
- `alpha_1, alpha_2`: Speed of adjustment (should be negative for stable adjustment)
- `beta`: Long-run elasticity

### Estimation
```python
from statsmodels.tsa.vector_ar.vecm import VECM

model = VECM(df, k_ar_diff=1, coint_rank=1)
result = model.fit()
print(result.summary())
```

## Cointegration and Forecasting

### Cointegrated System
- Short-run: Variables may deviate from equilibrium
- Long-run: Variables return to equilibrium

### Implications for Forecasting
- ECM often forecasts better than unrestricted VAR
- Long-run constraints improve efficiency
- Error correction term provides mean reversion

## Testing Procedure

1. Test each series for unit root (ADF/KPSS)
2. If both I(1), test for cointegration
3. If cointegrated, estimate VECM
4. If not cointegrated, use VAR in differences

## Common Pitfalls

1. **Wrong critical values**: Engle-Granger uses different tables than ADF
2. **Multiple cointegrating vectors**: Engle-Granger finds only one
3. **Normalization**: Choice of dependent variable matters in E-G
4. **Small sample bias**: Johansen biased in small samples
5. **Structural breaks**: Can cause spurious cointegration or rejection

## References

- Engle, R.F. & Granger, C.W.J. (1987). "Co-Integration and Error Correction." *Econometrica*.
- Johansen, S. (1991). "Estimation and Hypothesis Testing of Cointegration Vectors." *Econometrica*.
- Johansen, S. (1995). *Likelihood-Based Inference in Cointegrated Vector Autoregressive Models*.
