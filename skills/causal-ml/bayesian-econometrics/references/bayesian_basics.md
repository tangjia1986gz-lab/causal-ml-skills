# Bayesian Basics for Economists

## Overview

Bayesian inference provides a coherent framework for updating beliefs about parameters as data becomes available. For economists, this approach offers natural uncertainty quantification and the ability to incorporate prior knowledge.

## Bayes' Theorem

The foundation of Bayesian inference:

$$P(\theta | y) = \frac{P(y | \theta) P(\theta)}{P(y)}$$

Where:
- $P(\theta | y)$: **Posterior** - probability of parameters given data
- $P(y | \theta)$: **Likelihood** - probability of data given parameters
- $P(\theta)$: **Prior** - probability of parameters before seeing data
- $P(y)$: **Marginal likelihood** - normalizing constant

In practice, we work with the unnormalized form:
$$P(\theta | y) \propto P(y | \theta) \cdot P(\theta)$$

## Key Concepts

### Prior Distribution

The prior encodes beliefs about parameters before observing data:

```python
import pymc as pm

with pm.Model() as model:
    # Uninformative (flat) prior
    beta_flat = pm.Flat("beta_flat")

    # Weakly informative prior
    beta_weak = pm.Normal("beta_weak", mu=0, sigma=10)

    # Informative prior (based on theory or past studies)
    beta_inform = pm.Normal("beta_inform", mu=0.5, sigma=0.1)
```

### Likelihood Function

The likelihood specifies the data generating process:

```python
with pm.Model() as model:
    # Linear regression likelihood
    beta = pm.Normal("beta", mu=0, sigma=10)
    sigma = pm.HalfNormal("sigma", sigma=1)

    mu = pm.math.dot(X, beta)
    y_obs = pm.Normal("y_obs", mu=mu, sigma=sigma, observed=y)
```

### Posterior Distribution

The posterior combines prior and likelihood:
- Represents updated beliefs after seeing data
- Used for all inference (point estimates, intervals, predictions)

## Bayesian vs Frequentist Interpretation

| Aspect | Frequentist | Bayesian |
|--------|-------------|----------|
| Parameters | Fixed, unknown constants | Random variables with distributions |
| Probability | Long-run frequency | Degree of belief |
| Intervals | 95% of such intervals contain true value | 95% probability parameter is in interval |
| Point estimate | MLE (mode of likelihood) | Posterior mean, median, or mode |

### Credible vs Confidence Intervals

**Frequentist 95% CI**: "If we repeated this experiment many times, 95% of the computed intervals would contain the true parameter."

**Bayesian 95% Credible Interval**: "Given the data and prior, there is a 95% probability the parameter lies in this interval."

The Bayesian interpretation is often more intuitive for applied work.

## Common Distributions for Econometrics

### For Regression Coefficients

```python
# Standard normal - centered, weakly informative
beta = pm.Normal("beta", mu=0, sigma=2.5)

# Student-t - heavier tails, robust to outliers
beta = pm.StudentT("beta", nu=4, mu=0, sigma=2.5)

# Laplace - sparsity-inducing (Bayesian LASSO)
beta = pm.Laplace("beta", mu=0, b=1)
```

### For Variance Parameters

```python
# Half-normal - for standard deviations
sigma = pm.HalfNormal("sigma", sigma=1)

# Inverse-gamma - conjugate for variance
sigma2 = pm.InverseGamma("sigma2", alpha=2, beta=1)

# Half-Cauchy - weakly informative, allows large values
sigma = pm.HalfCauchy("sigma", beta=5)
```

### For Count Data

```python
# Poisson for counts
lambda_ = pm.Gamma("lambda", alpha=2, beta=1)
y = pm.Poisson("y", mu=lambda_, observed=counts)

# Negative binomial for overdispersed counts
mu = pm.Gamma("mu", alpha=2, beta=1)
alpha = pm.Gamma("alpha", alpha=1, beta=1)
y = pm.NegativeBinomial("y", mu=mu, alpha=alpha, observed=counts)
```

## The Bayesian Workflow

### 1. Model Specification

Define the complete probabilistic model:

```python
with pm.Model() as linear_model:
    # Priors
    alpha = pm.Normal("alpha", mu=0, sigma=10)
    beta = pm.Normal("beta", mu=0, sigma=2.5, shape=k)
    sigma = pm.HalfNormal("sigma", sigma=1)

    # Linear predictor
    mu = alpha + pm.math.dot(X, beta)

    # Likelihood
    y_obs = pm.Normal("y", mu=mu, sigma=sigma, observed=y)
```

### 2. Prior Predictive Check

Simulate data from the prior to verify sensibility:

```python
with linear_model:
    prior_pred = pm.sample_prior_predictive(samples=500)

# Check if simulated y values are in reasonable range
import arviz as az
az.plot_ppc(az.from_pymc(prior_predictive=prior_pred),
            group="prior", kind="cumulative")
```

### 3. Sampling

Draw from the posterior using MCMC:

```python
with linear_model:
    trace = pm.sample(
        draws=2000,      # posterior samples
        tune=1000,       # burn-in/warmup
        chains=4,        # independent chains
        cores=4,         # parallel computation
        target_accept=0.9  # NUTS acceptance rate
    )
```

### 4. Convergence Diagnostics

Verify the sampler has converged:

```python
import arviz as az

# Summary with diagnostics
summary = az.summary(trace, var_names=["alpha", "beta", "sigma"])
print(summary)

# Check Rhat (should be < 1.01)
# Check ESS (should be > 400)
```

### 5. Posterior Predictive Check

Compare model predictions to observed data:

```python
with linear_model:
    ppc = pm.sample_posterior_predictive(trace)

az.plot_ppc(az.from_pymc(trace, posterior_predictive=ppc))
```

### 6. Inference

Summarize posteriors and draw conclusions:

```python
# Posterior summary
az.summary(trace, hdi_prob=0.95)

# Probability that effect is positive
(trace.posterior["beta"].values > 0).mean()

# Posterior density plots
az.plot_posterior(trace, var_names=["beta"])
```

## Example: Bayesian Linear Regression

Complete example for wage regression:

```python
import numpy as np
import pandas as pd
import pymc as pm
import arviz as az

# Simulate data
np.random.seed(42)
n = 200
education = np.random.uniform(8, 20, n)
experience = np.random.uniform(0, 40, n)
wage = 2 + 0.1 * education + 0.05 * experience + np.random.normal(0, 0.5, n)

# Prepare data
X = np.column_stack([education, experience])

with pm.Model() as wage_model:
    # Priors
    alpha = pm.Normal("alpha", mu=0, sigma=5)
    beta = pm.Normal("beta", mu=0, sigma=1, shape=2)
    sigma = pm.HalfNormal("sigma", sigma=1)

    # Expected wage
    mu = alpha + pm.math.dot(X, beta)

    # Likelihood
    wage_obs = pm.Normal("wage", mu=mu, sigma=sigma, observed=wage)

    # Sample
    trace = pm.sample(2000, tune=1000, chains=4, random_seed=42)

# Results
print(az.summary(trace, var_names=["alpha", "beta", "sigma"]))

# Interpretation
beta_samples = trace.posterior["beta"].values.reshape(-1, 2)
print(f"\nReturns to education: {beta_samples[:, 0].mean():.3f}")
print(f"95% HDI: [{np.percentile(beta_samples[:, 0], 2.5):.3f}, "
      f"{np.percentile(beta_samples[:, 0], 97.5):.3f}]")
```

## Advantages for Econometrics

1. **Natural Uncertainty Quantification**: Posteriors directly give probability distributions over parameters

2. **Regularization**: Priors prevent overfitting, especially with small samples

3. **Hierarchical Modeling**: Natural framework for panel data and clustered observations

4. **Decision Theory**: Posterior integrates directly with loss functions for optimal decisions

5. **Model Comparison**: Bayes factors and posterior model probabilities for formal comparison

6. **Sequential Learning**: Posterior becomes prior when new data arrives

## Common Pitfalls

1. **Improper Priors**: Can lead to improper posteriors (infinite mass)
2. **Prior Sensitivity**: Results may depend heavily on prior choice - always do sensitivity analysis
3. **Ignoring Diagnostics**: Chains that haven't converged give meaningless results
4. **Overconfident Priors**: Too narrow priors dominate the data
5. **Computational Issues**: Some models are hard to sample - use reparameterization

## References

- Gelman, A., et al. (2013). *Bayesian Data Analysis*, 3rd Edition
- McElreath, R. (2020). *Statistical Rethinking*, 2nd Edition
- Koop, G. (2003). *Bayesian Econometrics*
- Lancaster, T. (2004). *An Introduction to Modern Bayesian Econometrics*
