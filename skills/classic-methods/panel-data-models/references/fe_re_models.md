# Fixed Effects vs Random Effects Models

## Overview

Panel data models exploit the longitudinal structure of data to control for unobserved heterogeneity. The choice between fixed effects (FE) and random effects (RE) depends on the relationship between unobserved effects and regressors.

## The Panel Data Model

Consider the standard panel model:

$$y_{it} = X_{it}\beta + \alpha_i + \epsilon_{it}$$

Where:
- $y_{it}$: outcome for entity $i$ at time $t$
- $X_{it}$: observed covariates (including treatment)
- $\alpha_i$: unobserved entity-specific effect (time-invariant)
- $\epsilon_{it}$: idiosyncratic error

## Fixed Effects (FE) Model

### Assumption
$$\text{Cov}(\alpha_i, X_{it}) \neq 0$$

The unobserved effect may be correlated with regressors (endogeneity).

### Within Transformation

Subtract entity means to eliminate $\alpha_i$:

$$y_{it} - \bar{y}_i = (X_{it} - \bar{X}_i)\beta + (\epsilon_{it} - \bar{\epsilon}_i)$$

Or equivalently:
$$\ddot{y}_{it} = \ddot{X}_{it}\beta + \ddot{\epsilon}_{it}$$

### Key Properties

1. **Consistency**: FE is consistent even if $\alpha_i$ correlated with $X_{it}$
2. **Cannot estimate time-invariant effects**: Variables that don't vary within entity are eliminated
3. **Uses within-entity variation only**: Ignores between-entity information
4. **Less efficient** than RE when RE assumptions hold

### Implementation

```python
from panel_estimator import PanelEstimator

estimator = PanelEstimator(data, 'entity', 'time', 'y', ['x1', 'x2'])
fe_result = estimator.fit_fixed_effects(entity_effects=True)
```

### LSDV Equivalence

Fixed effects via within transformation is equivalent to Least Squares Dummy Variables:

$$y_{it} = X_{it}\beta + \sum_{i=1}^{N} \alpha_i D_i + \epsilon_{it}$$

where $D_i$ are entity dummies.

## Random Effects (RE) Model

### Assumption
$$\text{Cov}(\alpha_i, X_{it}) = 0$$

The unobserved effect is uncorrelated with all regressors (strict exogeneity).

### Composite Error

$$y_{it} = X_{it}\beta + u_{it}$$

where $u_{it} = \alpha_i + \epsilon_{it}$ is the composite error with:
- $\text{Var}(\alpha_i) = \sigma^2_\alpha$
- $\text{Var}(\epsilon_{it}) = \sigma^2_\epsilon$
- $\text{Corr}(u_{it}, u_{is}) = \rho = \frac{\sigma^2_\alpha}{\sigma^2_\alpha + \sigma^2_\epsilon}$ for $t \neq s$

### GLS Estimation (Quasi-Demeaning)

Transform using $\theta = 1 - \sqrt{\frac{\sigma^2_\epsilon}{\sigma^2_\epsilon + T\sigma^2_\alpha}}$:

$$y_{it} - \theta\bar{y}_i = (1-\theta)\alpha + (X_{it} - \theta\bar{X}_i)\beta + (u_{it} - \theta\bar{u}_i)$$

### Key Properties

1. **Efficiency**: RE is more efficient than FE when assumptions hold
2. **Can estimate time-invariant effects**: Unlike FE
3. **Uses both within and between variation**
4. **Inconsistent if $\text{Cov}(\alpha_i, X_{it}) \neq 0$**

### Implementation

```python
re_result = estimator.fit_random_effects()
print(f"Variance decomposition: {re_result.variance_decomposition}")
```

## Hausman Specification Test

### Hypotheses
- $H_0$: RE is consistent and efficient (use RE)
- $H_1$: RE is inconsistent (use FE)

### Test Statistic

$$H = (\hat{\beta}_{FE} - \hat{\beta}_{RE})'[\text{Var}(\hat{\beta}_{FE}) - \text{Var}(\hat{\beta}_{RE})]^{-1}(\hat{\beta}_{FE} - \hat{\beta}_{RE})$$

Under $H_0$: $H \sim \chi^2_k$

### Interpretation

| p-value | Conclusion |
|---------|------------|
| < 0.05 | Reject RE, use FE |
| >= 0.05 | Cannot reject RE, RE is more efficient |

### Implementation

```python
hausman = estimator.hausman_test()
print(hausman.conclusion)
```

### Caveats

1. **Large samples**: Test may reject RE even with negligible practical difference
2. **Negative test statistic**: Can occur with finite samples; use robust version
3. **Model misspecification**: Both FE and RE may be inconsistent

## Mundlak (Correlated Random Effects) Approach

Alternative test: include entity means in RE model.

$$y_{it} = X_{it}\beta + \bar{X}_i\gamma + \alpha_i + \epsilon_{it}$$

- Test $H_0: \gamma = 0$
- If rejected: correlation exists, FE preferred
- Advantage: Provides coefficient estimates for both within and between effects

```python
mundlak = estimator.within_between_test()
```

## First-Differencing

Alternative to within transformation:

$$\Delta y_{it} = \Delta X_{it}\beta + \Delta\epsilon_{it}$$

### When to Prefer First-Differencing

1. **Serial correlation**: FD better if $\epsilon_{it}$ follows random walk
2. **Measurement error**: FD may exacerbate or reduce bias depending on structure
3. **Unbalanced panels**: FD uses all adjacent observations

## R-Squared Measures

### Within R-squared
Variation explained in demeaned data:
$$R^2_{within} = 1 - \frac{\sum_{it}(\ddot{y}_{it} - \ddot{X}_{it}\hat{\beta})^2}{\sum_{it}\ddot{y}_{it}^2}$$

### Between R-squared
Variation explained in entity means:
$$R^2_{between} = 1 - \frac{\sum_i(\bar{y}_i - \bar{X}_i\hat{\beta})^2}{\sum_i(\bar{y}_i - \bar{\bar{y}})^2}$$

### Overall R-squared
Total variation explained (approximate for FE):
$$R^2_{overall} \approx \text{Corr}(y_{it}, X_{it}\hat{\beta})^2$$

## Decision Flowchart

```
                    Panel Data
                         |
                         v
          +-----------------------------+
          |   Hausman Test / Mundlak    |
          +-----------------------------+
                    /         \
                   /           \
           p < 0.05          p >= 0.05
              |                   |
              v                   v
    +-----------------+   +-----------------+
    |  Fixed Effects  |   | Random Effects  |
    | (within only)   |   | (more efficient)|
    +-----------------+   +-----------------+
              |                   |
              v                   v
    Need time-invariant    Can estimate
    effects? Use            time-invariant
    Correlated RE           effects directly
```

## Practical Recommendations

### Use Fixed Effects When:
- Treatment/policy varies within entity over time
- Concerned about omitted variable bias from time-invariant confounders
- Entity characteristics likely correlated with regressors
- Focus on causal inference

### Use Random Effects When:
- Interest in time-invariant characteristics
- Hausman test doesn't reject
- Entities are random sample from population
- Need efficiency with small T

### Always Consider:
- Clustered standard errors (entity level typically)
- Time effects in addition to entity effects
- Potential heterogeneous effects
- Serial correlation in residuals

## References

- Wooldridge, J.M. (2010). Econometric Analysis of Cross Section and Panel Data, Ch. 10-11
- Baltagi, B.H. (2013). Econometric Analysis of Panel Data, Ch. 2-4
- Mundlak, Y. (1978). On the Pooling of Time Series and Cross Section Data. Econometrica
