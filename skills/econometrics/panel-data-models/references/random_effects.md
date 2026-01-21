# Random Effects Estimation

## Overview

Random Effects (RE) treats the unobserved entity effect as random, uncorrelated with regressors. This allows estimation of time-invariant covariates and improves efficiency when the assumption holds.

## Model Specification

```
Y_it = X_it * beta + alpha_i + epsilon_it

where:
- alpha_i ~ iid(0, sigma_u^2)  [entity effect]
- epsilon_it ~ iid(0, sigma_e^2)  [idiosyncratic error]
- Cov(alpha_i, X_it) = 0  [CRITICAL ASSUMPTION]
```

## GLS Estimation

### Composite Error
```
v_it = alpha_i + epsilon_it

Var(v_it) = sigma_u^2 + sigma_e^2
Cov(v_it, v_is) = sigma_u^2  for t != s (within-entity correlation)
```

### Quasi-Demeaning

Instead of full demeaning (FE), RE uses partial demeaning:
```
(Y_it - theta * Y_i_bar) = (1 - theta) * beta_0 + (X_it - theta * X_i_bar) * beta + error

where theta = 1 - sqrt(sigma_e^2 / (sigma_e^2 + T * sigma_u^2))
```

**Key insight:**
- theta = 0: Pooled OLS (ignores entity structure)
- theta = 1: Fixed Effects (full demeaning)
- 0 < theta < 1: Random Effects (partial demeaning)

## Variance Components Estimation

```python
from linearmodels.panel import RandomEffects

result = RandomEffects.from_formula('y ~ x1 + x2', data=df).fit()

# Variance decomposition
print(f"sigma_u (entity): {result.variance_decomposition.Effects}")
print(f"sigma_e (idiosyncratic): {result.variance_decomposition.Residual}")
print(f"theta: {result.theta}")
```

## Efficiency Comparison

| Estimator | Efficiency | Consistency |
|-----------|------------|-------------|
| Pooled OLS | Inefficient (ignores clustering) | Inconsistent if Cov(alpha, X) != 0 |
| Fixed Effects | Less efficient | Consistent under strict exogeneity |
| Random Effects | **Most efficient** (if valid) | Consistent only if Cov(alpha, X) = 0 |

## Hausman Test

Tests H0: Cov(alpha_i, X_it) = 0 (RE is consistent)

```python
import numpy as np
from scipy import stats

# Coefficient difference
b_fe = fe_result.params
b_re = re_result.params

diff = b_fe - b_re
var_diff = fe_result.cov - re_result.cov

# Hausman statistic
H = diff @ np.linalg.inv(var_diff) @ diff
p_value = 1 - stats.chi2.cdf(H, df=len(diff))

if p_value < 0.05:
    print("Use Fixed Effects")
else:
    print("Random Effects is consistent")
```

## Mundlak Approach

Compromise between FE and RE: include entity means as regressors.

```
Y_it = X_it * beta + X_i_bar * gamma + alpha_i + epsilon_it
```

**Interpretation:**
- beta: within-entity effect
- gamma: between-entity effect
- If gamma != 0: RE assumption violated

```python
# Mundlak specification
df['x1_mean'] = df.groupby(level=0)['x1'].transform('mean')
df['x2_mean'] = df.groupby(level=0)['x2'].transform('mean')

model_mundlak = RandomEffects.from_formula(
    'y ~ x1 + x2 + x1_mean + x2_mean',
    data=df
)
```

## When to Use Random Effects

**Appropriate when:**
- Entity effects truly random (random sample from population)
- Need to estimate time-invariant variables (gender, industry, location)
- Hausman test fails to reject

**Inappropriate when:**
- Endogeneity concerns (Hausman test rejects)
- Self-selected sample (effects correlated with X)
- Focus is on within-entity changes

## Correlated Random Effects

Extension: allow correlation through auxiliary equation
```
alpha_i = X_i_bar * pi + r_i

where r_i ~ iid(0, sigma_r^2), Cov(r_i, X_it) = 0
```

This is equivalent to Mundlak approach with RE.

## References

- Baltagi, B. (2021). *Econometric Analysis of Panel Data*, Ch. 2-4.
- Mundlak, Y. (1978). "On the Pooling of Time Series and Cross Section Data." *Econometrica*.
- Hausman, J. (1978). "Specification Tests in Econometrics." *Econometrica*.
- Wooldridge, J. (2010). *Econometric Analysis of Cross Section and Panel Data*, Ch. 10.
