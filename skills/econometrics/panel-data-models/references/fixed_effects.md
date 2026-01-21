# Fixed Effects Estimation

## Overview

Fixed Effects (FE) estimation controls for time-invariant unobserved heterogeneity by including entity-specific intercepts or using within-transformation.

## Within Transformation

### Derivation

Model with entity effects:
```
Y_it = X_it * beta + alpha_i + epsilon_it
```

Entity mean:
```
Y_i_bar = X_i_bar * beta + alpha_i + epsilon_i_bar
```

Within transformation (demeaning):
```
(Y_it - Y_i_bar) = (X_it - X_i_bar) * beta + (epsilon_it - epsilon_i_bar)
```

The entity effect `alpha_i` is eliminated.

### Properties

| Property | Within Estimator |
|----------|------------------|
| Consistency | Under strict exogeneity |
| Efficiency | Less efficient than RE if RE valid |
| Time-invariant X | Cannot be estimated (absorbed) |
| Degrees of freedom | N*T - N - K (lose N for entity effects) |

## First-Difference Estimator

Alternative to within-transformation:
```
Delta_Y_it = Delta_X_it * beta + Delta_epsilon_it
```

**When to prefer first-difference:**
- Random walk errors: epsilon_it = epsilon_{i,t-1} + v_it
- Very short T: FD loses fewer observations
- Dynamic panel: base for GMM

**When to prefer within:**
- Serially uncorrelated errors
- More efficient use of variation

## Time Fixed Effects

Controls for common time shocks:
```
Y_it = X_it * beta + alpha_i + gamma_t + epsilon_it
```

Implementation in linearmodels:
```python
from linearmodels.panel import PanelOLS

# Entity effects only
model_fe = PanelOLS.from_formula('y ~ x + EntityEffects', data=df)

# Both entity and time effects
model_twfe = PanelOLS.from_formula(
    'y ~ x + EntityEffects + TimeEffects',
    data=df,
    drop_absorbed=True  # Required when including both
)
```

## Two-Way Fixed Effects

### Model
```
Y_it = X_it * beta + alpha_i + gamma_t + epsilon_it
```

### Assumptions
1. **Strict exogeneity**: E[epsilon_it | X_i, alpha_i, gamma_t] = 0 for all t
2. **No perfect collinearity**: X varies within entity AND time

### Warning for Treatment Effects

With staggered treatment adoption, TWFE can give biased estimates due to:
- Negative weights on some group-time ATTs (Goodman-Bacon 2021)
- Heterogeneous treatment effects over time

**Recommendation**: Use specialized DID estimators (Callaway-Sant'Anna, Sun-Abraham) for treatment effects.

## F-Test for Fixed Effects

Test whether entity fixed effects are jointly significant:

H0: alpha_1 = alpha_2 = ... = alpha_N

```python
# F-statistic from linearmodels
result = model.fit()
print(f"F-statistic: {result.f_statistic.stat}")
print(f"P-value: {result.f_statistic.pval}")
```

Critical values (approximate):
| Entities (N) | Observations (N*T) | F(0.05) |
|--------------|-------------------|---------|
| 50 | 500 | 1.36 |
| 100 | 1000 | 1.24 |
| 500 | 5000 | 1.11 |

## References

- Wooldridge, J. (2010). *Econometric Analysis of Cross Section and Panel Data*, Ch. 10.
- Baltagi, B. (2021). *Econometric Analysis of Panel Data*, Ch. 2.
- Goodman-Bacon, A. (2021). "Difference-in-Differences with Variation in Treatment Timing." *Journal of Econometrics*.
