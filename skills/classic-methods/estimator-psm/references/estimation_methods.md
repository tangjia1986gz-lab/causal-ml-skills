# PSM Estimation Methods

> **Reference Document** | Propensity Score Matching
> Based on Caliendo & Kopeinig (2008), Abadie & Imbens (2006)

## Overview

This document details the various matching and weighting methods used in propensity score analysis, including their implementations, strengths, and limitations.

---

## 1. Nearest Neighbor Matching

### 1.1 Basic Nearest Neighbor

**Algorithm**:
1. For each treated unit, find control with closest propensity score
2. Match treated to this control
3. Estimate effect using matched pairs

**Estimator**:
$$
\hat{\tau}_{ATT} = \frac{1}{N_T} \sum_{i: D_i=1} \left[ Y_i - Y_{j(i)} \right]
$$

Where $j(i)$ is the matched control for treated unit $i$.

**Variants**:

| Variant | Description |
|---------|-------------|
| 1:1 Matching | Each treated matched to one control |
| 1:K Matching | Each treated matched to K nearest controls |
| With Replacement | Controls can be reused |
| Without Replacement | Controls used only once |

### 1.2 Nearest Neighbor with Caliper

**Purpose**: Ensure minimum match quality by setting maximum PS distance.

**Algorithm**:
```
For each treated unit i:
    Find nearest control j where |PS_i - PS_j| < caliper
    If no such control exists:
        Leave i unmatched
    Else:
        Match i to j
```

**Caliper Selection**:

| Method | Formula | Typical Value |
|--------|---------|---------------|
| Rosenbaum & Rubin | 0.25 * SD(PS) | ~0.05 |
| Austin | 0.2 * SD(logit(PS)) | ~0.05 |
| Fixed | User-specified | 0.01-0.1 |

**Trade-offs**:
- Smaller caliper -> Better balance, more unmatched
- Larger caliper -> More matches, worse balance

### 1.3 Implementation

```python
def nearest_neighbor_match(
    ps_treated: np.ndarray,
    ps_control: np.ndarray,
    n_neighbors: int = 1,
    caliper: float = None,
    replacement: bool = True
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Perform nearest neighbor matching.

    Parameters
    ----------
    ps_treated : np.ndarray
        Propensity scores for treated units
    ps_control : np.ndarray
        Propensity scores for control units
    n_neighbors : int
        Number of matches per treated unit
    caliper : float, optional
        Maximum PS distance for valid match
    replacement : bool
        Allow matching with replacement

    Returns
    -------
    tuple
        (matched_treated_idx, matched_control_idx, weights)
    """
    from sklearn.neighbors import NearestNeighbors

    # Fit NN model on controls
    nn = NearestNeighbors(n_neighbors=n_neighbors)
    nn.fit(ps_control.reshape(-1, 1))

    # Find neighbors for treated
    distances, indices = nn.kneighbors(ps_treated.reshape(-1, 1))

    matched_t = []
    matched_c = []
    weights = []
    used_controls = set()

    for i, (dist, idx) in enumerate(zip(distances, indices)):
        for d, j in zip(dist, idx):
            # Check caliper
            if caliper and d > caliper:
                continue
            # Check replacement
            if not replacement and j in used_controls:
                continue

            matched_t.append(i)
            matched_c.append(j)
            weights.append(1.0)

            if not replacement:
                used_controls.add(j)
                break  # Only one match without replacement

    return np.array(matched_t), np.array(matched_c), np.array(weights)
```

---

## 2. Radius / Caliper Matching

### 2.1 Concept

Match each treated unit to ALL controls within a specified radius.

**Algorithm**:
```
For each treated unit i:
    Find all controls j where |PS_i - PS_j| < radius
    Match i to all qualifying controls
    Weight each match by 1/(number of matches)
```

### 2.2 Estimator

$$
\hat{\tau}_{ATT} = \frac{1}{N_T} \sum_{i: D_i=1} \left[ Y_i - \frac{\sum_{j \in N(i)} Y_j}{|N(i)|} \right]
$$

Where $N(i) = \{j: D_j=0, |PS_i - PS_j| < r\}$.

### 2.3 Advantages and Disadvantages

| Pros | Cons |
|------|------|
| Uses more information | Variable number of matches |
| Better for thick overlap | May have no matches |
| More efficient | Radius selection arbitrary |

---

## 3. Kernel Matching

### 3.1 Concept

Weight all controls by their PS distance, with closer controls getting higher weight.

### 3.2 Kernel Functions

| Kernel | Formula $K(u)$ | Support |
|--------|---------------|---------|
| Epanechnikov | $\frac{3}{4}(1-u^2)$ if $|u| \leq 1$ | [-1, 1] |
| Gaussian | $\frac{1}{\sqrt{2\pi}}e^{-u^2/2}$ | $(-\infty, \infty)$ |
| Uniform | $\frac{1}{2}$ if $|u| \leq 1$ | [-1, 1] |
| Triangular | $(1-|u|)$ if $|u| \leq 1$ | [-1, 1] |

Where $u = (PS_i - PS_j) / h$ and $h$ is bandwidth.

### 3.3 Estimator

$$
\hat{\tau}_{ATT} = \frac{1}{N_T} \sum_{i: D_i=1} \left[ Y_i - \frac{\sum_{j: D_j=0} K\left(\frac{PS_i - PS_j}{h}\right) Y_j}{\sum_{j: D_j=0} K\left(\frac{PS_i - PS_j}{h}\right)} \right]
$$

### 3.4 Bandwidth Selection

| Method | Description |
|--------|-------------|
| Silverman's Rule | $h = 1.06 \cdot \sigma \cdot n^{-1/5}$ |
| Cross-Validation | Minimize prediction error |
| Plug-in | Based on kernel moments |
| Fixed | User-specified |

### 3.5 Implementation

```python
def kernel_match(
    ps_treated: np.ndarray,
    ps_control: np.ndarray,
    y_control: np.ndarray,
    kernel: str = 'epanechnikov',
    bandwidth: str = 'silverman'
) -> np.ndarray:
    """
    Kernel matching to estimate counterfactual outcomes.

    Parameters
    ----------
    ps_treated : np.ndarray
        PS for treated units
    ps_control : np.ndarray
        PS for control units
    y_control : np.ndarray
        Outcomes for control units
    kernel : str
        Kernel function name
    bandwidth : str or float
        Bandwidth specification

    Returns
    -------
    np.ndarray
        Estimated counterfactual outcomes for treated
    """
    import numpy as np

    # Kernel functions
    kernels = {
        'epanechnikov': lambda u: np.where(np.abs(u) <= 1, 0.75*(1-u**2), 0),
        'gaussian': lambda u: np.exp(-0.5*u**2) / np.sqrt(2*np.pi),
        'uniform': lambda u: np.where(np.abs(u) <= 1, 0.5, 0)
    }
    K = kernels.get(kernel, kernels['epanechnikov'])

    # Bandwidth
    if bandwidth == 'silverman':
        h = 1.06 * np.std(np.concatenate([ps_treated, ps_control])) * \
            len(ps_control)**(-1/5)
    else:
        h = float(bandwidth)

    # Estimate counterfactuals
    y0_hat = np.zeros(len(ps_treated))

    for i, ps_t in enumerate(ps_treated):
        u = (ps_t - ps_control) / h
        weights = K(u)
        weight_sum = weights.sum()

        if weight_sum > 0:
            y0_hat[i] = np.sum(weights * y_control) / weight_sum
        else:
            y0_hat[i] = np.nan

    return y0_hat
```

---

## 4. Mahalanobis Distance Matching

### 4.1 Concept

Match on multivariate distance in covariate space, accounting for covariance.

### 4.2 Distance Metric

$$
d_M(X_i, X_j) = \sqrt{(X_i - X_j)' \Sigma^{-1} (X_i - X_j)}
$$

Where $\Sigma$ is the covariance matrix of covariates.

### 4.3 Comparison with PS Matching

| Aspect | Mahalanobis | PS Matching |
|--------|-------------|-------------|
| Dimension | Multi-dimensional | Scalar |
| Curse of Dimensionality | Severe with many X | Avoided |
| Exact Balance | On all X | Only on PS |
| Model Required | None | PS model |

### 4.4 Combined Approach

Match on Mahalanobis distance within PS caliper:
1. First, identify controls within PS caliper
2. Among those, select closest Mahalanobis match

---

## 5. Inverse Probability Weighting (IPW)

### 5.1 Concept

Instead of matching, weight observations by inverse of treatment probability.

### 5.2 Weights

**For ATT**:
$$
w_i = \begin{cases}
1 & \text{if } D_i = 1 \\
\frac{e(X_i)}{1 - e(X_i)} & \text{if } D_i = 0
\end{cases}
$$

**For ATE**:
$$
w_i = \begin{cases}
\frac{1}{e(X_i)} & \text{if } D_i = 1 \\
\frac{1}{1 - e(X_i)} & \text{if } D_i = 0
\end{cases}
$$

### 5.3 Estimators

**Horvitz-Thompson**:
$$
\hat{\tau}_{ATE}^{HT} = \frac{1}{n} \sum_{i=1}^{n} \left[ \frac{D_i Y_i}{e(X_i)} - \frac{(1-D_i) Y_i}{1-e(X_i)} \right]
$$

**Normalized/Hajek**:
$$
\hat{\tau}_{ATE}^{Hajek} = \frac{\sum_i D_i Y_i / e(X_i)}{\sum_i D_i / e(X_i)} - \frac{\sum_i (1-D_i) Y_i / (1-e(X_i))}{\sum_i (1-D_i) / (1-e(X_i))}
$$

### 5.4 Trimming

Extreme PS values create extreme weights. Trim by:
- Hard cutoff: Drop if PS < 0.1 or PS > 0.9
- Soft cutoff: Truncate weights at percentile

### 5.5 Implementation

```python
def ipw_estimator(
    y: np.ndarray,
    treatment: np.ndarray,
    propensity: np.ndarray,
    estimand: str = 'ATT',
    trim: float = 0.01,
    normalize: bool = True
) -> Tuple[float, float]:
    """
    IPW estimator for treatment effects.

    Parameters
    ----------
    y : np.ndarray
        Outcomes
    treatment : np.ndarray
        Treatment indicators
    propensity : np.ndarray
        Propensity scores
    estimand : str
        'ATT' or 'ATE'
    trim : float
        Trim PS to [trim, 1-trim]
    normalize : bool
        Use Hajek normalization

    Returns
    -------
    tuple
        (effect estimate, standard error)
    """
    # Trim propensity scores
    ps = np.clip(propensity, trim, 1 - trim)

    # Calculate weights
    if estimand == 'ATT':
        weights = np.where(treatment == 1, 1, ps / (1 - ps))
    else:  # ATE
        weights = np.where(treatment == 1, 1/ps, 1/(1-ps))

    # Estimate
    if normalize:
        mean_t = np.sum(treatment * weights * y) / np.sum(treatment * weights)
        mean_c = np.sum((1-treatment) * weights * y) / np.sum((1-treatment) * weights)
    else:
        n = len(y)
        mean_t = np.mean(treatment * y / ps)
        mean_c = np.mean((1-treatment) * y / (1-ps))

    effect = mean_t - mean_c

    # Standard error via influence function
    if estimand == 'ATT':
        psi = treatment * (y - effect) / ps - (1-treatment) * (y - effect) / (1-ps)
    else:
        psi = treatment * (y - mean_t) / ps - (1-treatment) * (y - mean_c) / (1-ps)

    se = np.std(psi) / np.sqrt(len(y))

    return effect, se
```

---

## 6. Doubly Robust Estimation

### 6.1 Concept

Combine PS weighting with outcome modeling. Consistent if EITHER model is correct.

### 6.2 AIPW Estimator

$$
\hat{\tau}_{DR} = \frac{1}{n} \sum_{i=1}^{n} \left[ \frac{D_i(Y_i - \hat{\mu}_1(X_i))}{e(X_i)} + \hat{\mu}_1(X_i) - \frac{(1-D_i)(Y_i - \hat{\mu}_0(X_i))}{1-e(X_i)} - \hat{\mu}_0(X_i) \right]
$$

Where:
- $\hat{\mu}_1(X)$: Predicted outcome under treatment
- $\hat{\mu}_0(X)$: Predicted outcome under control

### 6.3 Advantages

| Property | Description |
|----------|-------------|
| Double Robustness | Consistent if PS OR outcome model correct |
| Efficiency | Achieves semiparametric efficiency bound |
| Variance Reduction | Combines benefits of both approaches |

### 6.4 Implementation

```python
def doubly_robust_estimator(
    y: np.ndarray,
    treatment: np.ndarray,
    X: np.ndarray,
    propensity: np.ndarray = None
) -> Tuple[float, float]:
    """
    Doubly robust AIPW estimator.

    Parameters
    ----------
    y : np.ndarray
        Outcomes
    treatment : np.ndarray
        Treatment indicators
    X : np.ndarray
        Covariates
    propensity : np.ndarray, optional
        Pre-estimated propensity scores

    Returns
    -------
    tuple
        (effect estimate, standard error)
    """
    from sklearn.linear_model import LogisticRegression, LinearRegression

    # Estimate propensity if not provided
    if propensity is None:
        ps_model = LogisticRegression()
        ps_model.fit(X, treatment)
        propensity = ps_model.predict_proba(X)[:, 1]

    ps = np.clip(propensity, 0.01, 0.99)

    # Outcome models
    # mu1: E[Y|X, D=1]
    model1 = LinearRegression()
    model1.fit(X[treatment == 1], y[treatment == 1])
    mu1 = model1.predict(X)

    # mu0: E[Y|X, D=0]
    model0 = LinearRegression()
    model0.fit(X[treatment == 0], y[treatment == 0])
    mu0 = model0.predict(X)

    # AIPW estimator
    aipw = (
        treatment * (y - mu1) / ps + mu1 -
        (1 - treatment) * (y - mu0) / (1 - ps) - mu0
    )

    effect = np.mean(aipw)
    se = np.std(aipw) / np.sqrt(len(y))

    return effect, se
```

---

## 7. Method Selection Guide

```
+------------------------------------------------------------------+
|                    MATCHING METHOD SELECTION                      |
+------------------------------------------------------------------+
|                                                                  |
|  START: What is your sample size?                                |
|         |                                                        |
|         v                                                        |
|  +------+------+                                                 |
|  | N < 100?    |                                                 |
|  +------+------+                                                 |
|      |      |                                                    |
|     Yes     No                                                   |
|      |      |                                                    |
|      v      v                                                    |
| Exact     How many covariates?                                   |
| Matching  |                                                      |
|           +------+------+                                        |
|           | K > 10?     |                                        |
|           +------+------+                                        |
|               |      |                                           |
|              Yes     No                                          |
|               |      |                                           |
|               v      v                                           |
|           PS-based  Consider both                                |
|           Matching  PS and Mahalanobis                           |
|               |                                                  |
|               v                                                  |
|  +---------------------------+                                   |
|  | Overlap concerns?         |                                   |
|  +---------------------------+                                   |
|         |        |                                               |
|        Yes       No                                              |
|         |        |                                               |
|         v        v                                               |
|  NN with     Consider                                            |
|  caliper     Kernel or IPW                                       |
|                                                                  |
+------------------------------------------------------------------+
```

---

## 8. Standard Error Estimation

### 8.1 Abadie-Imbens Standard Errors

For matching estimators:
$$
\widehat{Var}(\hat{\tau}) = \frac{1}{N_T^2} \sum_{i: D_i=1} \left[ \hat{\sigma}_1^2(X_i) + \hat{\sigma}_0^2(X_i) \cdot (1 + K_M(i)) \right]
$$

Where $K_M(i)$ is the number of times control $i$ is used as a match.

### 8.2 Bootstrap

```python
def bootstrap_se(data, estimator_func, n_boot=1000, seed=42):
    """Bootstrap standard error for matching estimator."""
    np.random.seed(seed)
    estimates = []

    for _ in range(n_boot):
        boot_idx = np.random.choice(len(data), len(data), replace=True)
        boot_data = data.iloc[boot_idx]
        estimates.append(estimator_func(boot_data))

    return np.std(estimates)
```

---

## References

- Caliendo, M., & Kopeinig, S. (2008). Some Practical Guidance for the Implementation of Propensity Score Matching. *Journal of Economic Surveys*, 22(1), 31-72.
- Abadie, A., & Imbens, G. W. (2006). Large Sample Properties of Matching Estimators. *Econometrica*, 74(1), 235-267.
- Robins, J. M., Rotnitzky, A., & Zhao, L. P. (1994). Estimation of Regression Coefficients When Some Regressors Are Not Always Observed. *JASA*, 89, 846-866.
- Hirano, K., Imbens, G. W., & Ridder, G. (2003). Efficient Estimation of Average Treatment Effects Using the Estimated Propensity Score. *Econometrica*, 71(4), 1161-1189.
