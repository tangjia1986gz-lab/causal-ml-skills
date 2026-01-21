# Prior Selection Guide

## Overview

Choosing appropriate priors is one of the most important decisions in Bayesian analysis. This guide covers strategies for prior selection in econometric applications, from uninformative to highly informative priors.

## Prior Philosophy

### Types of Priors

| Prior Type | Description | When to Use |
|------------|-------------|-------------|
| **Flat/Uninformative** | Uniform over parameter space | Rarely recommended; can be improper |
| **Weakly Informative** | Rules out implausible values but otherwise vague | Default choice for most applications |
| **Informative** | Encodes specific prior knowledge | When strong theory or prior studies exist |
| **Reference** | Objective priors maximizing information from data | Formal Bayesian inference without subjectivity |

### The Principle of Weakly Informative Priors

Gelman et al. recommend priors that:
1. Allow reasonable parameter values
2. Rule out extreme/implausible values
3. Don't dominate the likelihood with moderate data

```python
import pymc as pm
import numpy as np

# BAD: Flat prior - can cause computational issues
# beta = pm.Flat("beta")

# GOOD: Weakly informative - allows wide range but rules out absurd
beta = pm.Normal("beta", mu=0, sigma=2.5)

# For standardized data (mean 0, SD 1), this allows effects from -7.5 to 7.5
# with 99% prior probability - nearly always sufficient
```

## Prior Selection by Parameter Type

### Regression Coefficients

#### Default Recommendation: Normal(0, 2.5)

For standardized predictors (mean 0, SD 1) and outcome:

```python
with pm.Model() as model:
    # Standardize data first
    X_std = (X - X.mean(axis=0)) / X.std(axis=0)
    y_std = (y - y.mean()) / y.std()

    # Weakly informative prior
    beta = pm.Normal("beta", mu=0, sigma=2.5, shape=X.shape[1])
```

#### Student-t for Robustness

When outliers may be present or want more uncertainty:

```python
# Heavier tails than Normal
beta = pm.StudentT("beta", nu=4, mu=0, sigma=2.5, shape=k)
```

#### Horseshoe Prior for Sparsity

When many coefficients expected to be zero:

```python
# Global shrinkage
tau = pm.HalfCauchy("tau", beta=1)

# Local shrinkage
lambda_ = pm.HalfCauchy("lambda", beta=1, shape=k)

# Coefficients
beta = pm.Normal("beta", mu=0, sigma=tau * lambda_, shape=k)
```

### Variance/Standard Deviation Parameters

#### Half-Normal (Default)

```python
# For error standard deviation
sigma = pm.HalfNormal("sigma", sigma=1)

# Reasoning: Assumes most residual SDs are between 0 and 2
# with standardized data
```

#### Half-Cauchy (More Flexibility)

When larger variances are plausible:

```python
sigma = pm.HalfCauchy("sigma", beta=2)
```

#### Inverse-Gamma (Conjugate)

For computational convenience in certain models:

```python
# Shape=2 gives variance, scale controls prior mean
sigma2 = pm.InverseGamma("sigma2", alpha=2, beta=1)
sigma = pm.Deterministic("sigma", pm.math.sqrt(sigma2))
```

### Intercepts

```python
# Wide prior centered at data mean
alpha = pm.Normal("alpha", mu=y.mean(), sigma=10*y.std())
```

### Binary/Logistic Regression Coefficients

```python
# Log-odds scale; effect of 2.5 moves probability from 0.5 to 0.92
beta = pm.Normal("beta", mu=0, sigma=2.5)

# More conservative
beta = pm.Normal("beta", mu=0, sigma=1)
```

## Prior Predictive Checks

**Always** simulate from the prior to verify it generates sensible data.

### Implementation

```python
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt

with pm.Model() as model:
    # Priors
    alpha = pm.Normal("alpha", mu=0, sigma=10)
    beta = pm.Normal("beta", mu=0, sigma=2.5, shape=k)
    sigma = pm.HalfNormal("sigma", sigma=1)

    # Linear predictor
    mu = alpha + pm.math.dot(X, beta)

    # Likelihood
    y_pred = pm.Normal("y_pred", mu=mu, sigma=sigma, shape=len(y))

    # Sample from prior
    prior_samples = pm.sample_prior_predictive(samples=1000)

# Convert to InferenceData
idata = az.from_pymc(prior_predictive=prior_samples)

# Visualize prior predictive distribution
fig, ax = plt.subplots(figsize=(10, 4))
az.plot_ppc(idata, group="prior", ax=ax)
ax.set_title("Prior Predictive Check")
plt.show()

# Check reasonable ranges
y_prior = prior_samples.prior_predictive["y_pred"].values.flatten()
print(f"Prior predictive y range: [{y_prior.min():.1f}, {y_prior.max():.1f}]")
print(f"Prior predictive y mean: {y_prior.mean():.1f}")
print(f"Prior predictive y std: {y_prior.std():.1f}")
```

### What to Look For

1. **Reasonable Range**: Does simulated y cover plausible values?
2. **Not Too Wide**: Is prior putting mass on impossible values?
3. **Not Too Narrow**: Is prior excluding plausible values?

### Example: Wage Regression

```python
# Wages are positive, typically $10-100/hour
# Log wages range roughly from 2 to 5

with pm.Model() as wage_model:
    alpha = pm.Normal("alpha", mu=3, sigma=1)  # Centered around log(20) ≈ 3
    beta_educ = pm.Normal("beta_educ", mu=0.05, sigma=0.05)  # ~5% return per year
    sigma = pm.HalfNormal("sigma", sigma=0.5)

    log_wage = alpha + beta_educ * education
    log_wage_obs = pm.Normal("log_wage", mu=log_wage, sigma=sigma,
                              observed=np.log(wages))

    prior_check = pm.sample_prior_predictive(samples=500)

# Verify: exp(log_wage) should give reasonable dollar amounts
```

## Informative Priors from Prior Studies

### Meta-Analysis Approach

Use posterior from previous study as prior:

```python
# Previous study found beta = 0.08 with SE = 0.02
# Use as informative prior

beta = pm.Normal("beta", mu=0.08, sigma=0.02)
```

### Elicited Priors

Convert expert knowledge to prior:

```python
# Expert believes effect is between 0.05 and 0.15 with 90% confidence
# This suggests Normal(0.10, 0.03)

from scipy import stats

# Find parameters matching expert beliefs
# P(0.05 < beta < 0.15) = 0.90
# Solving: mu = 0.10, sigma ≈ 0.03

beta = pm.Normal("beta", mu=0.10, sigma=0.03)
```

## Prior Sensitivity Analysis

### Implementation

```python
def fit_with_prior(y, X, prior_sigma, draws=2000):
    """Fit model with different prior scales."""
    with pm.Model():
        alpha = pm.Normal("alpha", mu=0, sigma=10)
        beta = pm.Normal("beta", mu=0, sigma=prior_sigma, shape=X.shape[1])
        sigma = pm.HalfNormal("sigma", sigma=1)

        mu = alpha + pm.math.dot(X, beta)
        y_obs = pm.Normal("y", mu=mu, sigma=sigma, observed=y)

        trace = pm.sample(draws, tune=1000, chains=4,
                         progressbar=False, random_seed=42)

    return az.summary(trace, var_names=["beta"])


# Test different prior scales
prior_sigmas = [0.5, 1.0, 2.5, 5.0, 10.0]
results = {}

for sigma in prior_sigmas:
    results[sigma] = fit_with_prior(y, X, sigma)
    print(f"\nPrior sigma = {sigma}")
    print(results[sigma])
```

### Interpreting Sensitivity

- **Robust**: Results similar across prior choices → data is informative
- **Sensitive**: Results change substantially → prior matters, report multiple analyses

## Common Pitfalls

### 1. Using Improper Flat Priors

```python
# AVOID: Can cause computational issues
beta = pm.Flat("beta")

# PREFER: Weakly informative
beta = pm.Normal("beta", mu=0, sigma=10)
```

### 2. Forgetting to Standardize

```python
# If X has large scale (e.g., income in dollars)
# Prior of Normal(0, 2.5) is too tight

# SOLUTION: Standardize or adjust prior
X_std = (X - X.mean(axis=0)) / X.std(axis=0)
```

### 3. Overconfident Priors

```python
# TOO NARROW: Prior will dominate
beta = pm.Normal("beta", mu=0, sigma=0.01)

# BETTER: Allow data to speak
beta = pm.Normal("beta", mu=0, sigma=2.5)
```

### 4. Ignoring Parameter Constraints

```python
# WRONG: Normal allows negative variance
sigma = pm.Normal("sigma", mu=1, sigma=0.5)

# CORRECT: Constrained to positive
sigma = pm.HalfNormal("sigma", sigma=1)
```

## Prior Recommendations Summary

| Parameter | Default Prior | Notes |
|-----------|--------------|-------|
| Regression coefficient (standardized) | `Normal(0, 2.5)` | Adjust scale if not standardized |
| Intercept | `Normal(y_mean, 10*y_std)` | Center at data mean |
| Standard deviation | `HalfNormal(sigma=1)` | For standardized data |
| Variance | `InverseGamma(2, 1)` | Conjugate option |
| Probability | `Beta(1, 1)` or `Beta(2, 2)` | Uniform or slight regularization |
| Count rate | `Gamma(2, 0.5)` | Positive, allows wide range |
| Correlation | `LKJCorr(eta=2)` | For correlation matrices |

## References

- Gelman, A. (2006). Prior distributions for variance parameters in hierarchical models.
- Stan Development Team. Prior Choice Recommendations.
- Lemoine, N. P. (2019). Moving beyond noninformative priors: why and how to choose weakly informative priors in Bayesian analyses.
