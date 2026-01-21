# Estimation Methods for Causal Forests

## Overview

This document covers the core estimation methods in the causal forest family: standard causal forests, generalized random forests (GRF), and honest trees. We focus on the Athey-Wager methodology implemented in the R `grf` package.

---

## 1. Standard Causal Forests

### Definition

A causal forest is an ensemble of causal trees that estimate the Conditional Average Treatment Effect (CATE):

$$\tau(x) = \mathbb{E}[Y(1) - Y(0) | X = x]$$

### Algorithm

**Training Phase**:
```
For b = 1 to B (number of trees):
    1. Draw subsample of size s from n observations
    2. Split subsample into splitting sample and estimation sample (honest splitting)
    3. Grow causal tree using splitting sample:
       - At each node, find split maximizing heterogeneity in treatment effects
       - Stop when minimum node size reached
    4. Populate leaves with estimation sample
    5. In each leaf, estimate treatment effect as difference in means
```

**Prediction Phase**:
```
For new observation x:
    1. Drop x down each tree to find its leaf
    2. Compute forest weights: alpha_i(x) = fraction of trees where i and x share a leaf
    3. Return weighted average: tau_hat(x) = sum_i alpha_i(x) * tau_i
```

### Splitting Criterion

**Objective**: Find split that maximizes heterogeneity in treatment effects between child nodes.

**Criterion (Athey-Imbens)**:
$$\max_{j,s} \left[ n_L \cdot \hat{\tau}_L^2 + n_R \cdot \hat{\tau}_R^2 \right]$$

Where:
- $j$ = splitting variable, $s$ = split point
- $n_L, n_R$ = observations in left/right children
- $\hat{\tau}_L, \hat{\tau}_R$ = estimated treatment effects in children

### Python Implementation (econml)

```python
from econml.grf import CausalForest
from econml.dml import CausalForestDML

# Basic CausalForest
cf = CausalForest(
    n_estimators=2000,
    min_samples_leaf=5,
    max_depth=None,  # Grow until min_samples_leaf
    honest=True,
    inference=True
)

cf.fit(Y, T, X=X)
cate = cf.predict(X_test)
cate_interval = cf.predict_interval(X_test, alpha=0.05)

# CausalForestDML (with nuisance model flexibility)
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier

cf_dml = CausalForestDML(
    model_y=GradientBoostingRegressor(n_estimators=100, max_depth=4),
    model_t=GradientBoostingClassifier(n_estimators=100, max_depth=4),
    n_estimators=2000,
    min_samples_leaf=5,
    criterion='het',  # Heterogeneity criterion
    honest=True
)

cf_dml.fit(Y, T, X=X, W=W)  # W = additional controls
tau = cf_dml.effect(X_test)
```

### R Implementation (grf)

```r
library(grf)

# Fit causal forest
cf <- causal_forest(
  X = X,           # Covariates (effect modifiers)
  Y = Y,           # Outcome
  W = W,           # Treatment indicator
  num.trees = 2000,
  honesty = TRUE,
  honesty.fraction = 0.5,
  min.node.size = 5,
  mtry = ceiling(sqrt(ncol(X))),
  sample.fraction = 0.5,
  clusters = NULL,  # Optional cluster IDs for clustered SEs
  tune.parameters = "all"  # Auto-tune parameters
)

# Predictions
pred <- predict(cf, X.test, estimate.variance = TRUE)
tau_hat <- pred$predictions
tau_se <- sqrt(pred$variance.estimates)

# Confidence intervals
ci_lower <- tau_hat - 1.96 * tau_se
ci_upper <- tau_hat + 1.96 * tau_se
```

---

## 2. Generalized Random Forests (GRF)

### Framework

GRF generalizes random forests to solve moment conditions of the form:

$$\mathbb{E}[\psi_{\theta(x)}(O_i) | X_i = x] = 0$$

Where:
- $\theta(x)$ = parameter of interest (e.g., CATE, quantile, regression)
- $\psi$ = moment function (estimating equation)
- $O_i$ = observed data for unit $i$

### Causal Forest as GRF

For causal forests, the moment condition is:
$$\psi_{\tau}(Y, W, X) = (Y - \mu(X) - \tau(X) \cdot (W - e(X))) \cdot (W - e(X))$$

Where:
- $\mu(X) = \mathbb{E}[Y | X]$ (outcome regression)
- $e(X) = \mathbb{E}[W | X]$ (propensity score)
- $\tau(X)$ = CATE

### Local Centering

GRF implements local centering to reduce confounding bias:

```python
# Local centering pseudocode
def grf_causal_forest_with_centering(X, Y, W):
    # Step 1: Estimate nuisance functions locally
    mu_forest = regression_forest(X, Y)  # E[Y|X]
    e_forest = regression_forest(X, W)   # E[W|X]

    mu_hat = mu_forest.predict(X)
    e_hat = e_forest.predict(X)

    # Step 2: Center outcome and treatment
    Y_centered = Y - mu_hat
    W_centered = W - e_hat

    # Step 3: Estimate CATE using centered variables
    # tau(x) solves: E[(Y_c - tau * W_c) * W_c | X=x] = 0

    return causal_forest_on_centered(X, Y_centered, W_centered)
```

### R GRF Implementation

```r
# Standard causal forest (with automatic local centering)
cf <- causal_forest(X, Y, W)

# Custom nuisance models
Y.hat <- predict(regression_forest(X, Y))$predictions
W.hat <- predict(regression_forest(X, W))$predictions

# Using pre-computed nuisance estimates
cf_custom <- causal_forest(
  X, Y, W,
  Y.hat = Y.hat,
  W.hat = W.hat
)

# Multi-arm causal forest (continuous treatment)
cf_continuous <- causal_forest(X, Y, W)  # W can be continuous

# Instrumental forest (IV setting)
if_model <- instrumental_forest(X, Y, W, Z)  # Z = instrument
```

---

## 3. Honest Trees

### Definition

An honest tree uses separate data for:
1. **Splitting sample**: Determines tree structure (where to split)
2. **Estimation sample**: Estimates leaf predictions (treatment effects)

### Why Honesty?

**Problem with standard trees**:
- Same data used for splitting and estimation
- Splits chosen to maximize apparent heterogeneity
- Overfits to noise, creating spurious heterogeneity
- Invalid confidence intervals (too narrow)

**Honesty solution**:
- Independent data for structure vs. estimation
- Unbiased leaf estimates
- Valid asymptotic inference
- Correct confidence interval coverage

### Implementation

```python
class HonestCausalTree:
    def __init__(self, min_samples_leaf=5, honesty_fraction=0.5):
        self.min_samples_leaf = min_samples_leaf
        self.honesty_fraction = honesty_fraction

    def fit(self, X, y, treatment):
        n = len(y)

        # Split data
        split_size = int(n * (1 - self.honesty_fraction))
        indices = np.random.permutation(n)

        split_idx = indices[:split_size]  # For tree structure
        est_idx = indices[split_size:]     # For estimation

        # Build tree structure using splitting sample
        self.tree_structure = self._build_tree(
            X[split_idx], y[split_idx], treatment[split_idx]
        )

        # Populate leaves using estimation sample
        self._populate_leaves(
            X[est_idx], y[est_idx], treatment[est_idx]
        )

    def _build_tree(self, X, y, treatment):
        # Find best split maximizing treatment effect heterogeneity
        # ... standard recursive tree building
        pass

    def _populate_leaves(self, X, y, treatment):
        # For each leaf, assign estimation sample observations
        # Estimate treatment effect as difference in means
        for leaf in self.leaves:
            mask = self._get_leaf_mask(X, leaf)
            y_leaf = y[mask]
            t_leaf = treatment[mask]

            # Difference in means
            leaf.tau = (
                np.mean(y_leaf[t_leaf == 1]) -
                np.mean(y_leaf[t_leaf == 0])
            )

            # Variance estimation
            n1 = np.sum(t_leaf == 1)
            n0 = np.sum(t_leaf == 0)
            var1 = np.var(y_leaf[t_leaf == 1]) / n1 if n1 > 0 else 0
            var0 = np.var(y_leaf[t_leaf == 0]) / n0 if n0 > 0 else 0
            leaf.variance = var1 + var0
```

### R Configuration

```r
# Honest causal forest (default)
cf_honest <- causal_forest(
  X, Y, W,
  honesty = TRUE,           # Use honest estimation
  honesty.fraction = 0.5,   # Fraction for estimation
  honesty.prune.leaves = TRUE  # Remove empty leaves
)

# Non-honest (for prediction only, NOT inference)
cf_nonhonest <- causal_forest(
  X, Y, W,
  honesty = FALSE  # NOT recommended for confidence intervals
)
```

---

## 4. Forest Weights Interpretation

### Weight Calculation

Each causal forest prediction is a weighted average:

$$\hat{\tau}(x) = \sum_{i=1}^{n} \alpha_i(x) \cdot \hat{\tau}_i$$

Where weights $\alpha_i(x)$ represent how similar observation $i$ is to query point $x$:

$$\alpha_i(x) = \frac{1}{B} \sum_{b=1}^{B} \mathbf{1}\{i \in L_b(x)\} \cdot \frac{1}{|L_b(x)|}$$

- $L_b(x)$ = leaf containing $x$ in tree $b$
- $|L_b(x)|$ = number of observations in that leaf

### Extracting Weights

```r
# R grf
weights <- get_forest_weights(cf, X.test)

# For a single test point
alpha <- weights[1, ]  # Weights on all training points for first test point

# Check weight properties
print(sum(alpha))  # Should be ~1
print(sum(alpha > 0))  # Number of non-zero weights (effective neighbors)
```

### Interpretation

- High weight $\alpha_i(x)$: Observation $i$ is highly relevant for predicting $\tau(x)$
- Weights create adaptive neighborhoods based on treatment effect similarity
- Unlike k-NN: neighborhoods are asymmetric and effect-specific

---

## 5. Variance Estimation

### Forest-Based Variance

Primary method using infinitesimal jackknife:

$$\hat{V}(x) = \frac{n-1}{n} \sum_{i=1}^{n} \left( \hat{\tau}^{(-i)}(x) - \hat{\tau}(x) \right)^2$$

Where $\hat{\tau}^{(-i)}(x)$ is the prediction without observation $i$.

### Cluster-Robust Variance

For clustered data (e.g., households, schools):

```r
# R grf with clusters
cf_clustered <- causal_forest(
  X, Y, W,
  clusters = cluster_id,  # Cluster identifier
  equalize.cluster.weights = TRUE
)

# Standard errors account for within-cluster correlation
pred <- predict(cf_clustered, estimate.variance = TRUE)
```

### Bootstrap Variance (Alternative)

```python
def bootstrap_variance(cf_model, X_test, n_bootstrap=500):
    """Bootstrap confidence intervals for CATE."""
    predictions = []
    n_train = len(cf_model._X_train)

    for _ in range(n_bootstrap):
        # Resample training data
        idx = np.random.choice(n_train, size=n_train, replace=True)

        # Refit model
        cf_boot = fit_causal_forest(
            X_train[idx], y_train[idx], treatment_train[idx]
        )

        # Predict
        tau_boot, _ = cf_boot.predict(X_test, return_std=False)
        predictions.append(tau_boot)

    predictions = np.array(predictions)

    # Bootstrap confidence interval
    ci_lower = np.percentile(predictions, 2.5, axis=0)
    ci_upper = np.percentile(predictions, 97.5, axis=0)

    return ci_lower, ci_upper
```

---

## 6. Tuning Parameters

### Key Parameters

| Parameter | Default | Effect | Tuning Guidance |
|-----------|---------|--------|-----------------|
| `num.trees` | 2000 | More trees = lower variance | 2000-4000 typical |
| `min.node.size` | 5 | Larger = smoother estimates | 5-20 depending on n |
| `honesty.fraction` | 0.5 | Balance splitting/estimation | 0.5 standard |
| `mtry` | sqrt(p) | Variables per split | Can tune via CV |
| `sample.fraction` | 0.5 | Subsampling rate | 0.5 standard |
| `alpha` | 0.05 | Imbalance penalty | Default usually fine |

### Automatic Tuning

```r
# R grf auto-tuning
cf_tuned <- causal_forest(
  X, Y, W,
  tune.parameters = "all",  # Tune all parameters
  tune.num.trees = 200,     # Trees for tuning
  tune.num.reps = 50,       # Tuning repetitions
  tune.num.draws = 1000     # Random draws for search
)

# Check tuned parameters
print(cf_tuned$tuning.output)
```

### Manual Tuning

```python
from sklearn.model_selection import cross_val_score
import numpy as np

def tune_causal_forest(X, y, treatment, param_grid):
    """Simple grid search for causal forest parameters."""
    best_score = -np.inf
    best_params = None

    for min_leaf in param_grid['min_node_size']:
        for mtry in param_grid['mtry']:
            # Cross-validation
            scores = []
            for fold in range(5):
                # ... implement CV
                pass

            mean_score = np.mean(scores)
            if mean_score > best_score:
                best_score = mean_score
                best_params = {'min_node_size': min_leaf, 'mtry': mtry}

    return best_params
```

---

## 7. Comparison: econml vs grf

| Aspect | econml (Python) | grf (R) |
|--------|-----------------|---------|
| **Implementation** | Native Python | C++ backend via R |
| **Nuisance models** | Flexible sklearn models | Internal regression forests |
| **Confidence intervals** | Available | Available with better theory |
| **Calibration tests** | Manual implementation | Built-in `test_calibration()` |
| **Policy learning** | Via separate module | Via `policytree` package |
| **Speed** | Good | Excellent (C++ core) |
| **Documentation** | Good | Excellent |

### Using grf from Python (rpy2)

```python
import rpy2.robjects as ro
from rpy2.robjects import pandas2ri, numpy2ri
from rpy2.robjects.packages import importr

pandas2ri.activate()
numpy2ri.activate()

grf = importr('grf')

# Convert data
X_r = ro.r.matrix(X, nrow=X.shape[0], ncol=X.shape[1])
Y_r = ro.FloatVector(y)
W_r = ro.FloatVector(treatment)

# Fit causal forest
cf = grf.causal_forest(
    X_r, Y_r, W_r,
    num_trees=2000,
    honesty=True,
    honesty_fraction=0.5,
    min_node_size=5
)

# Predictions
pred = grf.predict_causal_forest(cf, X_r, estimate_variance=True)
tau_hat = np.array(pred.rx2('predictions'))
tau_var = np.array(pred.rx2('variance.estimates'))

# Variable importance
var_imp = grf.variable_importance(cf)

# Calibration test
cal_test = grf.test_calibration(cf)

# Best linear projection
blp = grf.best_linear_projection(cf, A_r)
```

---

## Summary: Method Selection

| Scenario | Recommended Method |
|----------|-------------------|
| Python workflow, sklearn integration | `econml.dml.CausalForestDML` |
| Best inference, calibration tests | R `grf::causal_forest` via rpy2 |
| Large datasets, speed critical | R `grf` (C++ backend) |
| Custom nuisance models | `econml.dml.CausalForestDML` |
| Clustered data | R `grf` with `clusters` argument |
| Policy learning | R `grf` + `policytree` package |

---

## References

1. Athey, S., Tibshirani, J., & Wager, S. (2019). Generalized Random Forests. *Annals of Statistics*, 47(2), 1148-1178.

2. Wager, S., & Athey, S. (2018). Estimation and Inference of Heterogeneous Treatment Effects using Random Forests. *JASA*, 113(523), 1228-1242.

3. Athey, S., & Imbens, G. (2016). Recursive Partitioning for Heterogeneous Causal Effects. *PNAS*, 113(27), 7353-7360.

4. grf package: https://grf-labs.github.io/grf/
5. econml package: https://econml.azurewebsites.net/
