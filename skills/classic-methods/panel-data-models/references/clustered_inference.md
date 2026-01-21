# Clustered Standard Errors and Robust Inference

## Why Cluster?

Standard errors assume independent observations. Panel data violates this:
- Observations within entity are correlated over time
- Errors may be correlated within clusters (firms, states, schools)
- Ignoring clustering leads to **understated standard errors** and **over-rejection**

## The Clustering Problem

### Within-Cluster Correlation

For entity $i$ with $T$ observations, the error correlation structure:

$$\text{Corr}(\epsilon_{it}, \epsilon_{is}) = \rho \neq 0 \text{ for } t \neq s$$

### Consequences of Ignoring Clustering

If true $\rho = 0.5$ with 50 clusters of 20 observations each:
- Effective sample size is much smaller than 1000
- Naive SE underestimated by factor of ~3
- Type I error rate inflated (5% test has ~25% rejection rate)

## Cluster-Robust Standard Errors

### The "Sandwich" Estimator

$$\hat{V}_{CR} = (X'X)^{-1}\left(\sum_{g=1}^{G} X_g'\hat{u}_g\hat{u}_g'X_g\right)(X'X)^{-1}$$

Where:
- $G$ = number of clusters
- $X_g$ = design matrix for cluster $g$
- $\hat{u}_g$ = residual vector for cluster $g$

### Finite Sample Correction

Standard adjustment:
$$\hat{V}_{CR,adj} = \frac{G}{G-1} \cdot \frac{N-1}{N-K} \cdot \hat{V}_{CR}$$

### Implementation

```python
from panel_estimator import PanelEstimator

estimator = PanelEstimator(data, 'entity', 'time', 'y', ['x1', 'x2'])
fe = estimator.fit_fixed_effects()

# Entity-clustered SE
fe_clustered = estimator.cluster_robust_inference(
    fe,
    cluster_col='entity',
    method='stata'
)
```

## Clustering Levels

### Rule of Thumb
**Cluster at the level of treatment variation** or higher.

### Common Scenarios

| Treatment Varies By | Recommended Clustering |
|---------------------|----------------------|
| Individual over time | Individual |
| Firm (all workers same treatment) | Firm |
| State-year | State |
| Industry-year | Industry |

### Two-Way Clustering

When errors correlated both within entity AND within time:

$$\hat{V}_{2way} = \hat{V}_{entity} + \hat{V}_{time} - \hat{V}_{entity \times time}$$

```python
# Two-way clustering (entity and time)
# Requires custom implementation or use linearmodels package
import linearmodels.panel as plm

mod = plm.PanelOLS(y, X, entity_effects=True)
result = mod.fit(cov_type='clustered', cluster_entity=True, cluster_time=True)
```

## Small Number of Clusters Problem

### The Problem

Clustered SE rely on asymptotic approximation with $G \to \infty$.

| Number of Clusters | Reliability |
|-------------------|-------------|
| G < 10 | Very unreliable |
| 10 < G < 30 | Somewhat unreliable |
| 30 < G < 50 | Generally OK |
| G > 50 | Usually reliable |

### Symptoms
- Over-rejection (true 5% test rejects more than 5%)
- Confidence intervals too narrow
- P-values too small

## Solutions for Few Clusters

### 1. Degrees of Freedom Correction

Use $t_{G-1}$ distribution instead of normal:

```python
# Use t-distribution with G-1 df
from scipy import stats
p_value = 2 * (1 - stats.t.cdf(abs(t_stat), G - 1))
```

### 2. Wild Cluster Bootstrap

More reliable with few clusters (G < 30).

**Algorithm**:
1. Estimate model, get residuals $\hat{u}_{ig}$
2. For $b = 1, ..., B$:
   - Draw Rademacher weights $w_g \in \{-1, 1\}$ for each cluster
   - Create bootstrap residuals: $u^*_{ig} = w_g \cdot \hat{u}_{ig}$
   - Create bootstrap $y^*_{ig} = X_{ig}\hat{\beta} + u^*_{ig}$
   - Re-estimate, get $\hat{\beta}^*_b$
3. Compute SE from distribution of $\hat{\beta}^*$

```python
fe_wild = estimator.cluster_robust_inference(
    fe,
    cluster_col='entity',
    method='wild',
    n_bootstrap=9999
)
```

### 3. Cluster Bootstrap

Resample entire clusters:

```python
fe_boot = estimator.cluster_robust_inference(
    fe,
    cluster_col='entity',
    method='bootstrap',
    n_bootstrap=1000
)
```

### 4. Aggregation

Collapse data to cluster level:
- Average observations within cluster
- Run analysis on cluster means
- Simple but loses within-cluster information

## Heteroskedasticity and Serial Correlation

### Heteroskedasticity-Robust (HC)

For independent but heteroskedastic errors:

$$\hat{V}_{HC} = (X'X)^{-1}\left(\sum_i \hat{u}_i^2 x_i x_i'\right)(X'X)^{-1}$$

Variants: HC0, HC1, HC2, HC3 (differ in finite-sample corrections)

### Serial Correlation

Within-entity serial correlation handled by entity clustering.

**Driscoll-Kraay SE**: For cross-sectional dependence
- Assumes correlation decays with distance
- Robust to heteroskedasticity, autocorrelation, AND cross-sectional correlation

## Diagnostic Tests

### Test for Clustering Need

Compare clustered vs. non-clustered SE:
- If substantially different, clustering matters
- Ratio > 2 suggests significant within-cluster correlation

### Wooldridge Test for Serial Correlation

Test for first-order autocorrelation in FE residuals:

$$\hat{\epsilon}_{it} = \rho \hat{\epsilon}_{i,t-1} + v_{it}$$

- $H_0: \rho = 0$ (no serial correlation)
- F-test on $\rho$

## Practical Guidelines

### Always Cluster By:
1. The level at which treatment varies
2. The level at which sampling was done
3. The highest level of aggregation you're comfortable with

### Conservative Approach
When in doubt, cluster at higher level:
- Overly fine clustering: incorrect SE
- Overly coarse clustering: conservative (wider CI), but valid

### Reporting
Always report:
- Number of clusters
- Clustering level
- Type of SE (robust, clustered, bootstrap)

```
Standard errors clustered at firm level (G = 127)
```

## Code Example: Comprehensive Inference

```python
from panel_estimator import PanelEstimator
import numpy as np

estimator = PanelEstimator(data, 'firm', 'year', 'revenue', ['treatment', 'size'])

# Fit with entity FE
fe = estimator.fit_fixed_effects(entity_effects=True)

# Different SE estimators
se_methods = {
    'default': fe,
    'entity_cluster': estimator.cluster_robust_inference(fe, 'firm', 'stata'),
    'bootstrap': estimator.cluster_robust_inference(fe, 'firm', 'bootstrap'),
    'wild_bootstrap': estimator.cluster_robust_inference(fe, 'firm', 'wild'),
}

# Compare
print("Standard Error Comparison:")
print("-" * 50)
for method, result in se_methods.items():
    se_treat = result.std_errors['treatment']
    print(f"{method:20s}: SE = {se_treat:.4f}")
```

## References

- Cameron, A.C. & Miller, D.L. (2015). A Practitioner's Guide to Cluster-Robust Inference. Journal of Human Resources.
- Cameron, A.C., Gelbach, J.B., & Miller, D.L. (2008). Bootstrap-Based Improvements for Inference with Clustered Errors. Review of Economics and Statistics.
- Abadie, A., Athey, S., Imbens, G.W., & Wooldridge, J.M. (2023). When Should You Adjust Standard Errors for Clustering?
- MacKinnon, J.G. & Webb, M.D. (2018). The Wild Bootstrap for Few (Treated) Clusters. Econometrics Journal.
