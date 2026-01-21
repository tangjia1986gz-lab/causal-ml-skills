# DDML Identification Assumptions

> Reference document for Double/Debiased Machine Learning identification conditions

## Overview

Double/Debiased Machine Learning (DDML) relies on several key assumptions to produce valid causal estimates. Understanding these assumptions is critical for proper application and interpretation.

---

## 1. Neyman Orthogonality

### Definition

A score function $\psi(W; \theta, \eta)$ is **Neyman-orthogonal** at $(\theta_0, \eta_0)$ if:

$$
\partial_\eta E[\psi(W; \theta_0, \eta)]|_{\eta=\eta_0} = 0
$$

### Interpretation

- The moment condition is **locally insensitive** to perturbations in nuisance parameters $\eta$
- First-order errors in nuisance estimation do not affect inference on $\theta_0$
- Enables use of regularized/ML estimators without invalidating standard errors

### PLR Orthogonal Score

For Partially Linear Regression:

```
Y = D * theta + g(X) + epsilon
D = m(X) + V
```

The Neyman-orthogonal score is:

$$
\psi^{PLR}(W; \theta, \ell, m) = (Y - \ell(X) - \theta(D - m(X)))(D - m(X))
$$

**Verification of Orthogonality**:

Taking the pathwise derivative with respect to $\ell$ and $m$ at the truth:

1. $\partial_\ell E[\psi] = -E[(D - m_0(X))] = 0$ (since $E[V|X] = 0$)
2. $\partial_m E[\psi] = -E[(Y - \ell_0(X) - \theta_0(D - m_0(X)))] \cdot (-1) = 0$

### IRM Orthogonal Score (AIPW)

For Interactive Regression Model (binary treatment):

$$
\psi^{AIPW}(W; \theta, g, m) = g(1,X) - g(0,X) + \frac{D(Y-g(1,X))}{m(X)} - \frac{(1-D)(Y-g(0,X))}{1-m(X)} - \theta
$$

This is the efficient influence function for ATE under unconfoundedness.

---

## 2. Cross-Fitting

### Why Cross-Fitting is Required

**Problem without cross-fitting**:
- ML estimators are not $\sqrt{n}$-consistent
- Using same data to estimate nuisance and compute score creates bias
- Regularization bias can propagate to treatment effect

**Solution**:
K-fold cross-fitting eliminates this bias:

```
For k = 1, ..., K:
    Train nuisance functions on data EXCLUDING fold k
    Compute scores on fold k using trained functions
Aggregate scores across all folds
```

### Cross-Fitting Algorithm

```python
def cross_fitting(data, n_folds=5):
    """
    K-fold cross-fitting procedure.

    1. Split data randomly into K folds: I_1, ..., I_K
    2. For each k:
       - Train hat{eta}_k on data \ I_k (all data except fold k)
       - Compute hat{psi}_i for i in I_k using hat{eta}_k
    3. Aggregate: hat{theta} = solve sum_i hat{psi}_i = 0
    """
    pass
```

### Choosing K (Number of Folds)

| Sample Size | Recommended K | Rationale |
|-------------|---------------|-----------|
| n < 500     | 2-3           | Need sufficient training data per fold |
| 500-2000    | 5             | Standard choice, good balance |
| n > 2000    | 5-10          | Can afford more folds |

**Rule of thumb**: n/K > 100 for stable nuisance estimation

### Multiple Repetitions

Running cross-fitting multiple times with different random splits and averaging reduces variance:

```python
# Multiple repetitions for stability
n_rep = 10
theta_estimates = []
for rep in range(n_rep):
    theta_k = run_cross_fitting(data, random_seed=rep)
    theta_estimates.append(theta_k)

theta_final = np.mean(theta_estimates)
```

---

## 3. Rate Conditions

### Product Rate Condition

For valid inference, nuisance estimators must satisfy:

$$
\|\hat{\ell} - \ell_0\|_2 \cdot \|\hat{m} - m_0\|_2 = o_P(n^{-1/2})
$$

### Interpretation

- **Product** of errors must decay faster than $n^{-1/2}$
- Each nuisance function can converge at rate slower than $n^{-1/4}$
- Example: If both converge at $n^{-1/4}$, product is $n^{-1/2}$

### When Rate Conditions Hold

| Nuisance Structure | Achievable Rate | Sufficient for DDML? |
|--------------------|-----------------|----------------------|
| Parametric | $n^{-1/2}$ | Yes |
| Sparse (s << n) | $\sqrt{s \log p / n}$ | Yes, if s small |
| Smooth (Holder) | $n^{-\beta/(2\beta+d)}$ | Depends on d, beta |
| Neural nets | $n^{-1/4}$ to $n^{-1/2}$ | Often yes |
| Tree ensembles | $n^{-1/4}$ typical | Usually yes |

### Practical Implications

1. **Use sufficiently complex learners**: Simple linear models may not achieve required rates for complex nuisance
2. **High-dimensional sparsity helps**: Lasso achieves fast rates under sparsity
3. **Cross-validation for model selection**: Choose learners that minimize prediction error

---

## 4. Unconfoundedness (Selection on Observables)

### Assumption

$$
(Y(0), Y(1)) \perp D | X
$$

**Interpretation**: Conditional on observed covariates X, treatment assignment is as good as random.

### This is NOT testable

- Fundamentally untestable from observational data
- Must be justified by domain knowledge and study design
- DDML does not solve omitted variable bias

### Strengthening the Case for Unconfoundedness

| Approach | Description |
|----------|-------------|
| **Rich covariates** | Include all confounders, proxies for unobservables |
| **DAG/Causal diagram** | Explicitly model relationships |
| **Placebo tests** | Test effects on pseudo-outcomes |
| **Sensitivity analysis** | Assess robustness to violations |
| **Panel data** | Use fixed effects for time-invariant confounders |

### When Unconfoundedness Likely Fails

Consider alternative methods when:
- Treatment is highly correlated with unobserved factors (ability, motivation)
- Selection is based on future outcomes
- Important variables are clearly missing
- Domain experts believe assignment is non-random conditional on X

---

## 5. Overlap / Positivity

### Assumption

$$
0 < P(D=1|X) < 1 \quad \text{almost surely}
$$

For continuous treatment: $\text{Var}(D|X) > 0$

### Why Overlap Matters

- Extreme propensities lead to unstable weights in IPW-type estimators
- Near-zero propensities imply lack of comparable observations
- Violates the "counterfactual" nature of causal inference

### Diagnostics

```python
def check_overlap(propensity_scores, threshold=0.01):
    """
    Check propensity score overlap.
    """
    n_extreme_low = np.sum(propensity_scores < threshold)
    n_extreme_high = np.sum(propensity_scores > 1 - threshold)

    violations = n_extreme_low + n_extreme_high

    return {
        'n_violations': violations,
        'pct_violations': violations / len(propensity_scores) * 100,
        'min_propensity': propensity_scores.min(),
        'max_propensity': propensity_scores.max(),
        'recommendation': 'Trim or adjust' if violations > 0 else 'OK'
    }
```

### Remedies for Overlap Violations

| Strategy | Implementation | Trade-off |
|----------|----------------|-----------|
| **Trimming** | Drop observations with extreme propensities | Changes estimand to trimmed population |
| **Clipping** | Cap propensities at [eps, 1-eps] | Introduces small bias |
| **Overlap weights** | Weight by $m(X)(1-m(X))$ | Focuses on area of good overlap |
| **CATE estimation** | Estimate effects only where overlap exists | More limited conclusions |

---

## 6. Correct Model Class

### Assumption

True nuisance functions $\ell_0(X)$, $m_0(X)$ are in the model class or well-approximated by it.

### Practical Considerations

1. **Flexibility vs. variance**: More flexible models (RF, NN) can approximate better but have higher variance
2. **Ensemble approaches**: Combine multiple learners to hedge against misspecification
3. **Cross-validation**: Use CV to select best learner for the data

### Model Class Diagnostic

```python
def assess_model_fit(X, y, learner, cv_folds=5):
    """
    Assess how well learner fits the data via cross-validation.
    """
    from sklearn.model_selection import cross_val_score

    scores = cross_val_score(learner, X, y, cv=cv_folds,
                            scoring='neg_mean_squared_error')
    r2_scores = cross_val_score(learner, X, y, cv=cv_folds,
                               scoring='r2')

    return {
        'mse': -scores.mean(),
        'mse_std': scores.std(),
        'r2': r2_scores.mean(),
        'r2_std': r2_scores.std()
    }
```

---

## Summary: Assumption Checklist

| Assumption | Testable? | Validation Approach |
|------------|-----------|---------------------|
| Neyman orthogonality | Yes (by construction) | Use correct score function |
| Cross-fitting | Yes (by construction) | Implement K-fold properly |
| Rate conditions | Partially | CV performance, learner choice |
| Unconfoundedness | No | Domain knowledge, sensitivity |
| Overlap | Yes | Propensity distribution |
| Model class | Partially | CV model comparison |

---

## References

1. Chernozhukov, V., et al. (2018). Double/Debiased Machine Learning for Treatment and Structural Parameters. *The Econometrics Journal*, 21(1), C1-C68.

2. Chernozhukov, V., et al. (2022). Locally Robust Semiparametric Estimation. *Econometrica*, 90(4), 1501-1535.

3. Kennedy, E. H. (2022). Semiparametric Doubly Robust Targeted Double Machine Learning: A Review. arXiv:2203.06469.

4. Robins, J. M., & Rotnitzky, A. (1995). Semiparametric Efficiency in Multivariate Regression Models with Missing Data. *JASA*, 90(429), 122-129.
