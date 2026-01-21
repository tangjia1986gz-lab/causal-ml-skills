# Clustered Standard Errors in Panel Data

## Overview

Panel data typically exhibits serial correlation within entities and/or cross-sectional correlation across entities. Clustered standard errors address these violations of the iid assumption.

## Types of Clustering

### 1. Entity Clustering (Most Common)

Accounts for arbitrary correlation within entity over time.

```python
result = model.fit(cov_type='clustered', cluster_entity=True)
```

**Use when:**
- Serial correlation in errors (AR(1), etc.)
- Heteroskedasticity across entities
- Default choice for panel data

### 2. Time Clustering

Accounts for cross-sectional correlation at each time point.

```python
result = model.fit(cov_type='clustered', cluster_time=True)
```

**Use when:**
- Common shocks affect all entities simultaneously
- Macro variables included (affect all entities)
- Spatial correlation across entities

### 3. Two-Way Clustering (Cameron-Gelbach-Miller)

Accounts for both entity and time clustering simultaneously.

```python
result = model.fit(
    cov_type='clustered',
    cluster_entity=True,
    cluster_time=True
)
```

**Formula:**
```
V_two_way = V_entity + V_time - V_intersection
```

**Use when:**
- Both serial and cross-sectional correlation
- Macro-financial data
- Multi-country panels with global shocks

## Minimum Cluster Requirements

| Number of Clusters | Reliability | Recommendation |
|-------------------|-------------|----------------|
| < 20 | Poor | Use bootstrap or wild bootstrap |
| 20-50 | Moderate | Use with caution, report both |
| > 50 | Good | Standard clustering reliable |

## Finite Sample Corrections

### Degrees of Freedom Adjustment

```
V_adjusted = V_raw * (G / (G-1)) * ((N-1) / (N-K))

where:
- G = number of clusters
- N = number of observations
- K = number of parameters
```

linearmodels applies this automatically with `debiased=True` (default).

## Comparison of SE Types

| SE Type | Assumption | When Invalid |
|---------|------------|--------------|
| OLS (iid) | Independent, homoskedastic | Almost always in panels |
| Robust (HC1) | Heteroskedasticity ok | Serial correlation present |
| Clustered (entity) | Within-cluster correlation | Cross-sectional correlation |
| Clustered (time) | Time-cluster correlation | Serial correlation |
| Two-way | Both | - |

## Code Example

```python
from linearmodels.panel import PanelOLS

model = PanelOLS.from_formula('y ~ x1 + x2 + EntityEffects', data=df)

# Compare different SE types
se_types = {
    'Homoskedastic': model.fit(cov_type='unadjusted'),
    'Robust': model.fit(cov_type='robust'),
    'Clustered (entity)': model.fit(cov_type='clustered', cluster_entity=True),
    'Clustered (time)': model.fit(cov_type='clustered', cluster_time=True),
    'Two-way': model.fit(cov_type='clustered', cluster_entity=True, cluster_time=True)
}

# Print comparison
for name, result in se_types.items():
    print(f"{name}: SE(x1) = {result.std_errors['x1']:.4f}")
```

## Wild Bootstrap for Few Clusters

When clusters < 30, use wild cluster bootstrap:

```python
from linearmodels.panel import PanelOLS
import numpy as np

def wild_bootstrap_se(model, data, n_boot=999, seed=42):
    """Wild cluster bootstrap for few clusters."""
    np.random.seed(seed)

    result = model.fit()
    beta_hat = result.params
    residuals = result.resids

    entities = data.index.get_level_values(0).unique()
    n_entities = len(entities)

    boot_betas = []
    for _ in range(n_boot):
        # Rademacher weights by cluster
        weights = np.random.choice([-1, 1], size=n_entities)

        # Create bootstrapped residuals
        boot_resid = residuals.copy()
        for i, entity in enumerate(entities):
            mask = data.index.get_level_values(0) == entity
            boot_resid[mask] *= weights[i]

        # Bootstrap Y
        data_boot = data.copy()
        data_boot['y_boot'] = result.fitted_values + boot_resid

        # Re-estimate
        model_boot = PanelOLS.from_formula(
            'y_boot ~ x1 + x2 + EntityEffects',
            data=data_boot
        )
        boot_result = model_boot.fit()
        boot_betas.append(boot_result.params)

    # Bootstrap SE
    boot_se = pd.DataFrame(boot_betas).std()
    return boot_se
```

## Practical Recommendations

1. **Default**: Always cluster by entity for panel data
2. **Time effects**: If including time FE, still cluster by entity
3. **Few clusters**: Use wild bootstrap and report both
4. **Macro data**: Consider two-way clustering
5. **Sensitivity**: Report multiple SE types as robustness

## Common Pitfall

**Do NOT use robust SE without clustering** for panel data:
- Robust SE assumes independent observations
- Panel data has correlated observations within entity
- Will severely underestimate SE

## References

- Cameron, A.C., Gelbach, J.B., & Miller, D.L. (2011). "Robust Inference with Multiway Clustering." *Journal of Business & Economic Statistics*.
- Petersen, M.A. (2009). "Estimating Standard Errors in Finance Panel Data Sets." *Review of Financial Studies*.
- Bertrand, M., Duflo, E., & Mullainathan, S. (2004). "How Much Should We Trust Differences-in-Differences Estimates?" *QJE*.
