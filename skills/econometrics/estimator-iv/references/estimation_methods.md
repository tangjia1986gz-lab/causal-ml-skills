# IV Estimation Methods

## 1. Two-Stage Least Squares (2SLS)

### Model

**Structural Equation:**
```
Y = X*beta + W*gamma + epsilon
```

**First Stage:**
```
X = Z*pi + W*delta + nu
```

Where:
- Y: outcome
- X: endogenous variable
- Z: instruments (excluded from structural equation)
- W: exogenous controls (included in both equations)

### Algorithm

1. **First Stage:** Regress X on Z and W, get fitted values X_hat
2. **Second Stage:** Regress Y on X_hat and W

### Properties

- **Consistent:** As n → infinity, converges to true beta
- **Biased in finite samples:** Bias = Cov(X_hat, epsilon) / Var(X_hat)
- **Bias increases with weak instruments:** When pi ≈ 0

### Python (linearmodels)

```python
from linearmodels.iv import IV2SLS

model = IV2SLS(
    dependent=df['Y'],
    exog=df[['const'] + controls],  # W
    endog=df[['X']],                 # Endogenous
    instruments=df[instruments]      # Z
)
result = model.fit(cov_type='robust')
```

## 2. Limited Information Maximum Likelihood (LIML)

### Motivation

LIML is more robust to weak instruments than 2SLS.

### Estimator

LIML minimizes:
```
(Y - X*beta)'M_Z(Y - X*beta) / (Y - X*beta)'M_W(Y - X*beta)
```

Where M_Z and M_W are residual maker matrices.

### Properties

- **Approximately median-unbiased** (unlike 2SLS which is biased toward OLS)
- **More dispersed** than 2SLS with strong instruments
- **k-class estimator** with k = lambda_min (minimum eigenvalue)

### Python (linearmodels)

```python
from linearmodels.iv import IVLIML

model = IVLIML(
    dependent=df['Y'],
    exog=df[['const'] + controls],
    endog=df[['X']],
    instruments=df[instruments]
)
result = model.fit(cov_type='robust')
```

### When to Use

| Condition | Recommended |
|-----------|-------------|
| F > 20 | 2SLS or LIML |
| 10 < F < 20 | LIML preferred |
| F < 10 | LIML + AR inference |

## 3. Generalized Method of Moments (GMM)

### Moment Conditions

```
E[Z' * epsilon] = 0
```

GMM chooses beta to minimize:
```
(Z'epsilon)' * W * (Z'epsilon)
```

Where W is a weighting matrix.

### Efficient GMM (Two-Step)

1. First step: W = (Z'Z)^(-1)
2. Second step: W = (Z' * diag(e^2) * Z)^(-1) where e = Y - X*beta_1

### Python (linearmodels)

```python
from linearmodels.iv import IVGMM

model = IVGMM(
    dependent=df['Y'],
    exog=df[['const'] + controls],
    endog=df[['X']],
    instruments=df[instruments]
)
result = model.fit(cov_type='robust')
```

### When to Use

- Heteroskedasticity present
- Multiple endogenous variables
- Need efficiency (but beware finite-sample bias)

## 4. Jackknife IV (JIVE)

### Motivation

Reduces finite-sample bias from many instruments.

### Estimator

Leave-one-out first stage:
```
X_hat_i = Z_i' * pi_{-i}
```

Where pi_{-i} is estimated excluding observation i.

### Properties

- Reduces bias from many instruments
- More robust to weak instruments
- Higher variance than 2SLS

## 5. Split-Sample IV

### Algorithm

1. Split sample randomly into two halves
2. Estimate first stage on first half
3. Compute X_hat for second half using first-half estimates
4. Run second stage on second half

### Properties

- Eliminates correlation between X_hat and epsilon
- Loses efficiency (uses only half the data)
- Robust to many weak instruments

## Comparison Summary

| Estimator | Bias (Weak IV) | Efficiency | Computation |
|-----------|----------------|------------|-------------|
| **2SLS** | High | Good (strong IV) | Simple |
| **LIML** | Low | Moderate | Simple |
| **GMM** | Moderate | Best (heterosk.) | Iterative |
| **JIVE** | Low | Low | N regressions |
| **Split-Sample** | None | Low | Simple |

## Standard Errors

### Robust (Heteroskedasticity-Consistent)

```python
result = model.fit(cov_type='robust')
```

### Clustered

```python
result = model.fit(cov_type='clustered', clusters=df['cluster_id'])
```

### HAC (Time Series)

```python
result = model.fit(cov_type='kernel', kernel='bartlett', bandwidth=4)
```

## Key References

1. **2SLS:** Theil (1953), Basmann (1957)
2. **LIML:** Anderson & Rubin (1949)
3. **GMM:** Hansen (1982)
4. **JIVE:** Angrist, Imbens & Krueger (1999)
5. **Weak IV:** Stock & Yogo (2005)
