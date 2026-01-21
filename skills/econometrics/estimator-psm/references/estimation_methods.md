# PSM Estimation Methods

## 1. Propensity Score Estimation

### Logistic Regression (Standard)

```python
from sklearn.linear_model import LogisticRegression

ps_model = LogisticRegression(max_iter=1000)
ps = ps_model.fit(X, T).predict_proba(X)[:, 1]
```

**Pros:** Simple, interpretable, fast
**Cons:** Assumes linear relationship in log-odds

### Gradient Boosted Models

```python
from sklearn.ensemble import GradientBoostingClassifier

ps_model = GradientBoostingClassifier(n_estimators=100, max_depth=3)
ps = ps_model.fit(X, T).predict_proba(X)[:, 1]
```

**Pros:** Captures non-linearities, interactions
**Cons:** Can overfit, less interpretable

### Generalized Boosted Models (GBM)

Iteratively optimizes balance directly:

```python
# Use cross-validation to prevent overfitting
from sklearn.model_selection import cross_val_predict

ps = cross_val_predict(ps_model, X, T, cv=5, method='predict_proba')[:, 1]
```

## 2. Matching Methods

### Nearest Neighbor Matching

Match each treated unit to closest control on propensity score.

```python
from sklearn.neighbors import NearestNeighbors

nn = NearestNeighbors(n_neighbors=1, metric='euclidean')
nn.fit(control[['pscore']])
distances, indices = nn.kneighbors(treated[['pscore']])
```

**Options:**
- With/without replacement
- 1:1 or 1:k matching
- With caliper

### Caliper Matching

Only match within maximum distance:

```python
caliper = 0.2 * df['pscore'].std()  # Common choice
within_caliper = distances[:, 0] <= caliper
```

### Mahalanobis Distance Matching

Match on covariate distance, not propensity score:

```python
from scipy.spatial.distance import mahalanobis

# Compute Mahalanobis distance between units
cov_inv = np.linalg.inv(np.cov(X.T))
```

**Use when:** Propensity score alone doesn't achieve balance

### Coarsened Exact Matching (CEM)

1. Coarsen continuous covariates into bins
2. Exact match on coarsened values
3. Prune unmatched strata

**Use when:** Have discrete/categorical covariates

## 3. Weighting Methods

### Inverse Probability Weighting (IPW)

**For ATT:**
```python
# Treated weight = 1
# Control weight = ps / (1 - ps)
w = np.where(T == 1, 1, ps / (1 - ps))
```

**For ATE (Horvitz-Thompson):**
```python
# Treated weight = 1 / ps
# Control weight = 1 / (1 - ps)
w = np.where(T == 1, 1/ps, 1/(1-ps))
```

### Stabilized Weights

Reduce variance from extreme weights:

```python
p_treat = T.mean()
# For ATE
sw = np.where(T == 1, p_treat/ps, (1-p_treat)/(1-ps))
```

### Overlap Weights

Target population with clinical equipoise:

```python
# Weight by probability of opposite treatment
w = np.where(T == 1, 1-ps, ps)
```

**Advantage:** Automatically handles overlap violations

## 4. Doubly Robust Methods

### Augmented IPW (AIPW)

Combines propensity score and outcome model:

```
AIPW = (1/n) * Σ[T*Y/ps - (T-ps)/ps * μ₁(X)]
     - (1/n) * Σ[(1-T)*Y/(1-ps) + (T-ps)/(1-ps) * μ₀(X)]
```

**Property:** Consistent if EITHER ps model OR outcome model is correct.

```python
# Outcome models
mu1 = outcome_model_treated.predict(X)
mu0 = outcome_model_control.predict(X)

# AIPW components
aipw_y1 = (T * Y / ps + (1 - T/ps) * mu1).mean()
aipw_y0 = ((1-T) * Y / (1-ps) + (1 - (1-T)/(1-ps)) * mu0).mean()
ate = aipw_y1 - aipw_y0
```

### Targeted Maximum Likelihood (TMLE)

More sophisticated doubly robust estimator with better finite-sample properties.

## 5. Entropy Balancing

Directly optimize weights to achieve exact covariate balance:

```
min Σ wᵢ log(wᵢ)
s.t. Σ wᵢ X̄ᵢ = X̄_treated (moment conditions)
     Σ wᵢ = 1
```

**Implementation:** `ebalance` package in R; can implement via scipy in Python

## Comparison Summary

| Method | Pros | Cons | Best For |
|--------|------|------|----------|
| **NN Matching** | Simple, transparent | May discard data | Moderate samples |
| **Caliper** | Avoids bad matches | More exclusions | Sparse overlap |
| **IPW** | Uses all data | Extreme weights | Good overlap |
| **AIPW** | Doubly robust | More complex | Unknown confounders |
| **Entropy** | Exact balance | Computationally harder | Many covariates |

## Standard Errors

### For Matching

Bootstrap the entire matching procedure:

```python
boot_estimates = []
for _ in range(1000):
    boot_df = df.sample(frac=1, replace=True)
    # Re-estimate ps, re-match, re-estimate effect
    boot_estimates.append(effect)
se = np.std(boot_estimates)
```

### For IPW

Use sandwich estimator or bootstrap:

```python
# Bootstrap
boot_estimates = []
for _ in range(1000):
    boot_df = df.sample(frac=1, replace=True)
    # Re-compute weights and effect
    boot_estimates.append(effect)
se = np.std(boot_estimates)
```

### For AIPW

Influence function-based variance:

```
Var(τ) ≈ (1/n) * Var(ψ)
```

Where ψ is the influence function.

## Key References

1. Abadie & Imbens (2006): "Large Sample Properties of Matching Estimators"
2. Hirano, Imbens & Ridder (2003): "Efficient Estimation of Average Treatment Effects"
3. Bang & Robins (2005): "Doubly Robust Estimation"
4. Hainmueller (2012): "Entropy Balancing"
5. Imai & Ratkovic (2014): "Covariate Balancing Propensity Score"
