# Identification Assumptions for Causal Forests

## Overview

Causal forests estimate Conditional Average Treatment Effects (CATE):
$$\tau(x) = \mathbb{E}[Y(1) - Y(0) | X = x]$$

Valid CATE estimation requires specific identification assumptions that allow us to recover causal effects from observational or experimental data.

---

## Core Identification Assumptions

### 1. Unconfoundedness (Selection on Observables)

**Definition**: Treatment assignment is independent of potential outcomes conditional on observed covariates.

$$Y(0), Y(1) \perp\!\!\!\perp W | X$$

**Interpretation**:
- Given covariates X, treatment assignment W is as good as random
- All confounders (variables affecting both treatment and outcome) are observed and controlled
- Also known as: Conditional Independence Assumption (CIA), Ignorability

**Violations**:
- Unobserved confounders exist
- Selection on unobservables
- Unmeasured risk factors affecting both treatment selection and outcomes

**Testing**:
- Cannot be directly tested (untestable assumption)
- Indirect evidence: covariate balance after adjustment
- Sensitivity analysis for unmeasured confounding

```python
# Covariate balance check (indirect support for unconfoundedness)
from causal_forest import check_covariate_balance

balance_stats = check_covariate_balance(
    X=data[covariates],
    treatment=data['treatment'],
    method='standardized_diff'
)

# Look for standardized differences < 0.1
print(balance_stats.summary())
```

### 2. Positivity (Overlap / Common Support)

**Definition**: Every unit has positive probability of receiving each treatment level.

$$0 < P(W = 1 | X = x) < 1 \quad \forall x \in \mathcal{X}$$

**Interpretation**:
- For any covariate profile, both treatment and control are possible
- No deterministic treatment assignment regions
- Propensity score bounded away from 0 and 1

**Violations**:
- Certain subgroups never receive treatment
- Extreme selection into treatment
- Structural zeros in treatment assignment

**Testing**:
```python
# Check propensity score overlap
from sklearn.ensemble import GradientBoostingClassifier
import numpy as np

# Estimate propensity scores
ps_model = GradientBoostingClassifier()
ps_model.fit(X, treatment)
propensity = ps_model.predict_proba(X)[:, 1]

# Check overlap
print(f"Propensity range: [{propensity.min():.4f}, {propensity.max():.4f}]")
print(f"Observations with extreme propensity (<0.05 or >0.95): {((propensity < 0.05) | (propensity > 0.95)).sum()}")

# Visualize overlap
import matplotlib.pyplot as plt
fig, ax = plt.subplots()
ax.hist(propensity[treatment==0], alpha=0.5, label='Control', bins=50)
ax.hist(propensity[treatment==1], alpha=0.5, label='Treated', bins=50)
ax.legend()
ax.set_xlabel('Propensity Score')
plt.show()
```

### 3. Stable Unit Treatment Value Assumption (SUTVA)

**Definition**:
1. **No interference**: One unit's outcome depends only on their own treatment, not others' treatments
2. **No hidden variations**: Treatment is well-defined with a single version

$$Y_i = Y_i(W_i) \quad \text{(no dependence on } W_{-i} \text{)}$$

**Violations**:
- **Interference**: Peer effects, spillovers, network effects
  - Social programs where treated individuals affect untreated neighbors
  - Vaccination (herd immunity)
  - Pricing experiments with competition
- **Multiple treatment versions**:
  - Variable dosage without accounting for it
  - Different treatment delivery mechanisms

**Addressing SUTVA violations**:
```python
# Cluster-robust inference for potential interference
# If interference is within clusters (e.g., households, schools)

from causal_forest import fit_causal_forest_clustered

cf_clustered = fit_causal_forest_clustered(
    X=data[effect_modifiers],
    y=data[outcome],
    treatment=data[treatment],
    cluster_id=data['cluster_id'],  # e.g., household_id, school_id
    config=config
)
```

---

## Honest Estimation Theory

### What is Honesty?

**Definition**: A tree is honest if the data used to determine splits (tree structure) is independent from the data used to estimate leaf predictions (treatment effects).

**Split**:
```
Training Data
├── Splitting Sample (50%): Determines where to split
└── Estimation Sample (50%): Estimates effects in each leaf
```

### Why Honesty Matters

**Problem with standard trees**: Using the same data for splitting and estimation leads to:
- Overfitting to noise in treatment effects
- Biased estimates (optimistically selected splits)
- Invalid confidence intervals (too narrow)

**Honesty provides**:
1. **Unbiased estimation**: Leaf predictions are unbiased for true CATE
2. **Valid asymptotic inference**: Confidence intervals have correct coverage
3. **Consistency**: Estimates converge to true values as n grows

### Mathematical Formulation

For honest estimation, the CATE estimator in leaf L is:
$$\hat{\tau}(x) = \frac{1}{|S_1(L)|}\sum_{i \in S_1(L)} Y_i - \frac{1}{|S_0(L)|}\sum_{i \in S_0(L)} Y_i$$

Where:
- $S_1(L)$ = treated units in estimation sample falling in leaf L
- $S_0(L)$ = control units in estimation sample falling in leaf L

The key insight: splits were determined using different observations, so this is an unbiased estimator.

### Honesty Configuration

```python
from causal_forest import CausalForestConfig

# Standard honest estimation
config = CausalForestConfig(
    honesty=True,
    honesty_fraction=0.5,  # 50% for estimation, 50% for splitting
)

# More aggressive splitting (better tree structure, noisier estimates)
config_aggressive = CausalForestConfig(
    honesty=True,
    honesty_fraction=0.3,  # 30% for estimation, 70% for splitting
)

# More stable estimates (better estimation, simpler trees)
config_stable = CausalForestConfig(
    honesty=True,
    honesty_fraction=0.7,  # 70% for estimation, 30% for splitting
)
```

---

## Asymptotic Theory

### Consistency

Under regularity conditions, causal forest CATE estimates are consistent:
$$\hat{\tau}(x) \xrightarrow{p} \tau(x) \quad \text{as } n \to \infty$$

**Required conditions**:
1. Unconfoundedness and overlap hold
2. Trees grow at appropriate rate
3. Subsampling fraction $s/n \to 0$ as $n \to \infty$
4. Sufficient observations in each leaf

### Asymptotic Normality

Under additional regularity conditions:
$$\frac{\hat{\tau}(x) - \tau(x)}{\hat{\sigma}(x)} \xrightarrow{d} N(0, 1)$$

Where $\hat{\sigma}(x)$ is a consistent variance estimator.

**Implications**:
- Valid confidence intervals: $[\hat{\tau}(x) \pm z_{\alpha/2} \hat{\sigma}(x)]$
- Valid hypothesis tests
- Proper uncertainty quantification

### Variance Estimation

Causal forests provide variance estimates via:

**1. Forest-based variance** (preferred):
$$\hat{V}(x) = \frac{1}{B(B-1)} \sum_{b=1}^{B} \left(\hat{\tau}_b(x) - \bar{\tau}(x)\right)^2$$

Where $\hat{\tau}_b(x)$ is the prediction from tree b.

**2. Bootstrap variance** (alternative):
```python
# Bootstrap confidence intervals
from causal_forest import estimate_cate

cate_results = estimate_cate(
    cf_model,
    X_test,
    variance_method='bootstrap',  # 'forest', 'bootstrap', or 'jackknife'
    n_bootstrap=1000
)
```

### Rate of Convergence

CATE estimates converge at rate:
$$|\hat{\tau}(x) - \tau(x)| = O_p(n^{-\beta})$$

Where $\beta$ depends on:
- Smoothness of true CATE function
- Dimension of effect modifiers
- Tree depth and leaf size

Typical rates: $\beta \approx 0.25$ to $0.5$ depending on conditions.

---

## Local Centering (Orthogonalization)

### Purpose

Local centering reduces bias from confounding and improves efficiency by orthogonalizing the treatment and outcome with respect to covariates.

### Procedure

**Step 1**: Estimate conditional expectations
- $\hat{m}(x) = \hat{\mathbb{E}}[Y|X=x]$ (outcome regression)
- $\hat{e}(x) = \hat{\mathbb{E}}[W|X=x]$ (propensity score)

**Step 2**: Create centered variables
- $\tilde{Y}_i = Y_i - \hat{m}(X_i)$
- $\tilde{W}_i = W_i - \hat{e}(X_i)$

**Step 3**: Estimate CATE using centered variables

### Benefits

1. **Bias reduction**: Removes first-order confounding bias
2. **Efficiency**: Reduces variance of CATE estimates
3. **Robustness**: Double robustness property (consistent if either nuisance model correct)

### Implementation

```python
# grf automatically performs local centering
# In econml, use CausalForestDML

from econml.dml import CausalForestDML
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier

cf_dml = CausalForestDML(
    model_y=GradientBoostingRegressor(n_estimators=100),  # For E[Y|X]
    model_t=GradientBoostingClassifier(n_estimators=100),  # For E[W|X]
    n_estimators=2000
)

cf_dml.fit(Y, T, X=X, W=W)  # W = additional controls for centering
```

---

## Practical Implications

### Sample Size Requirements

| Condition | Minimum Recommended N |
|-----------|----------------------|
| Basic CATE estimation | 1,000 |
| Valid confidence intervals | 2,000 |
| Subgroup analysis | 500 per subgroup |
| Policy learning | 5,000+ |

### Diagnostic Checklist

Before trusting causal forest results, verify:

- [ ] **Overlap**: Propensity scores bounded away from 0 and 1
- [ ] **Balance**: Covariates balanced after adjustment
- [ ] **Honesty**: Using honest estimation for inference
- [ ] **Sample size**: Sufficient observations overall and in subgroups
- [ ] **SUTVA**: No obvious interference or spillovers
- [ ] **Sensitivity**: Results robust to unmeasured confounding assumptions

### Common Pitfalls

1. **Weak overlap**: Extreme propensity scores lead to high variance
   - Solution: Trim or truncate extreme propensities

2. **Small leaves**: Insufficient observations for reliable estimation
   - Solution: Increase `min_node_size`

3. **Confusing prediction with inference**: Good prediction does not equal valid inference
   - Solution: Always use honesty; report confidence intervals

4. **Ignoring SUTVA violations**: Network/spillover effects bias estimates
   - Solution: Cluster analysis or different methods

---

## References

1. Wager, S., & Athey, S. (2018). Estimation and Inference of Heterogeneous Treatment Effects using Random Forests. *Journal of the American Statistical Association*, 113(523), 1228-1242.

2. Athey, S., Tibshirani, J., & Wager, S. (2019). Generalized Random Forests. *Annals of Statistics*, 47(2), 1148-1178.

3. Imbens, G. W., & Rubin, D. B. (2015). *Causal Inference for Statistics, Social, and Biomedical Sciences*. Cambridge University Press.

4. Chernozhukov, V., et al. (2018). Double/Debiased Machine Learning for Treatment and Structural Parameters. *The Econometrics Journal*, 21(1), C1-C68.
