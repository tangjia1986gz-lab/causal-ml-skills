# Hierarchical Models in Bayesian Econometrics

## Overview

Hierarchical (multilevel) models are a natural framework for data with group structure. In econometrics, this includes panel data, clustered observations, and repeated measures. Bayesian estimation provides exact finite-sample inference and natural handling of varying group sizes.

## The Pooling Spectrum

### Complete Pooling
All groups share the same parameters - ignores group heterogeneity.

```python
# All individuals have same effect
with pm.Model():
    beta = pm.Normal("beta", mu=0, sigma=2.5)  # Single coefficient
    y_hat = alpha + beta * X
```

### No Pooling
Each group has independent parameters - ignores commonalities.

```python
# Each individual has own effect, estimated independently
with pm.Model():
    beta = pm.Normal("beta", mu=0, sigma=2.5, shape=n_groups)  # One per group
    y_hat = alpha + beta[group_idx] * X
```

### Partial Pooling (Hierarchical)
Groups share information through common hyperpriors - the Bayesian solution.

```python
# Individual effects drawn from common distribution
with pm.Model():
    # Hyperpriors
    mu_beta = pm.Normal("mu_beta", mu=0, sigma=2.5)
    sigma_beta = pm.HalfNormal("sigma_beta", sigma=1)

    # Group effects
    beta = pm.Normal("beta", mu=mu_beta, sigma=sigma_beta, shape=n_groups)
    y_hat = alpha + beta[group_idx] * X
```

## Why Partial Pooling Works

### Shrinkage
- Groups with little data are pulled toward the grand mean
- Groups with lots of data retain their individual estimates
- Automatically adapts to data quality

### Borrowing Strength
- Information flows between groups via hyperpriors
- Small groups benefit from large groups
- More efficient than no-pooling when groups are related

## Hierarchical Linear Model

### Model Structure

For individual $i$ in group $j$:

$$y_{ij} = \alpha_j + \beta_j x_{ij} + \epsilon_{ij}$$

With hierarchical priors:

$$\alpha_j \sim N(\mu_\alpha, \sigma_\alpha)$$
$$\beta_j \sim N(\mu_\beta, \sigma_\beta)$$

### PyMC Implementation

```python
import pymc as pm
import numpy as np
import pandas as pd

def fit_hierarchical_model(y, X, groups, draws=2000, tune=1000):
    """
    Fit hierarchical linear model with varying intercepts and slopes.

    Parameters
    ----------
    y : array-like
        Outcome variable
    X : array-like
        Predictor variable (single)
    groups : array-like
        Group indicators (integers 0 to n_groups-1)
    draws : int
        Number of posterior samples
    tune : int
        Number of tuning samples

    Returns
    -------
    trace : arviz.InferenceData
        Posterior samples
    """
    n_groups = len(np.unique(groups))

    with pm.Model() as hierarchical_model:
        # Hyperpriors for intercepts
        mu_alpha = pm.Normal("mu_alpha", mu=0, sigma=10)
        sigma_alpha = pm.HalfNormal("sigma_alpha", sigma=2)

        # Hyperpriors for slopes
        mu_beta = pm.Normal("mu_beta", mu=0, sigma=2.5)
        sigma_beta = pm.HalfNormal("sigma_beta", sigma=1)

        # Group-level parameters (non-centered)
        alpha_offset = pm.Normal("alpha_offset", mu=0, sigma=1, shape=n_groups)
        alpha = pm.Deterministic("alpha", mu_alpha + sigma_alpha * alpha_offset)

        beta_offset = pm.Normal("beta_offset", mu=0, sigma=1, shape=n_groups)
        beta = pm.Deterministic("beta", mu_beta + sigma_beta * beta_offset)

        # Observation-level
        sigma = pm.HalfNormal("sigma", sigma=1)

        # Expected value
        mu = alpha[groups] + beta[groups] * X

        # Likelihood
        y_obs = pm.Normal("y_obs", mu=mu, sigma=sigma, observed=y)

        # Sample
        trace = pm.sample(draws, tune=tune, chains=4, random_seed=42)

    return trace
```

## Panel Data Applications

### Fixed Effects vs Random Effects

In Bayesian framework, the fixed/random distinction becomes a matter of prior specification:

**"Fixed Effects" (Uninformative)**
```python
# Flat prior on group effects - no shrinkage
alpha = pm.Normal("alpha", mu=0, sigma=100, shape=n_groups)
```

**"Random Effects" (Hierarchical)**
```python
# Hierarchical prior - shrinkage toward mean
mu_alpha = pm.Normal("mu_alpha", mu=0, sigma=10)
sigma_alpha = pm.HalfNormal("sigma_alpha", sigma=2)
alpha = pm.Normal("alpha", mu=mu_alpha, sigma=sigma_alpha, shape=n_groups)
```

### Correlated Random Effects

Allow intercepts and slopes to be correlated:

```python
import pymc as pm
import numpy as np

with pm.Model() as correlated_model:
    # Hyperpriors
    mu = pm.Normal("mu", mu=0, sigma=2.5, shape=2)  # [mu_alpha, mu_beta]

    # Cholesky decomposition for correlation
    chol, corr, stds = pm.LKJCholeskyCov(
        "chol_cov",
        n=2,
        eta=2.0,  # Prior on correlation (2 = weakly informative)
        sd_dist=pm.HalfNormal.dist(sigma=2),
        compute_corr=True
    )
    cov = pm.Deterministic("cov", chol @ chol.T)

    # Group effects (multivariate)
    effects = pm.MvNormal("effects", mu=mu, chol=chol, shape=(n_groups, 2))
    alpha = effects[:, 0]
    beta = effects[:, 1]

    # Observation model
    sigma = pm.HalfNormal("sigma", sigma=1)
    mu_y = alpha[groups] + beta[groups] * X
    y_obs = pm.Normal("y_obs", mu=mu_y, sigma=sigma, observed=y)
```

## Varying Intercepts Model (Random Intercepts)

Common in panel data - group-specific intercepts, common slopes:

```python
def fit_varying_intercepts(y, X, groups, draws=2000):
    """Hierarchical model with varying intercepts only."""
    n_groups = len(np.unique(groups))
    n_predictors = X.shape[1] if X.ndim > 1 else 1

    with pm.Model():
        # Hyperpriors for intercept
        mu_alpha = pm.Normal("mu_alpha", mu=0, sigma=10)
        sigma_alpha = pm.HalfNormal("sigma_alpha", sigma=2)

        # Group intercepts (non-centered)
        alpha_offset = pm.Normal("alpha_offset", mu=0, sigma=1, shape=n_groups)
        alpha = pm.Deterministic("alpha", mu_alpha + sigma_alpha * alpha_offset)

        # Common slopes
        beta = pm.Normal("beta", mu=0, sigma=2.5, shape=n_predictors)

        # Residual
        sigma = pm.HalfNormal("sigma", sigma=1)

        # Linear predictor
        if X.ndim == 1:
            mu = alpha[groups] + beta * X
        else:
            mu = alpha[groups] + pm.math.dot(X, beta)

        # Likelihood
        y_obs = pm.Normal("y_obs", mu=mu, sigma=sigma, observed=y)

        trace = pm.sample(draws, tune=1000, chains=4)

    return trace
```

## Cross-Classified Models

When observations belong to multiple grouping factors:

```python
# Students (i) nested in schools (j) and neighborhoods (k)
with pm.Model():
    # School effects
    mu_school = pm.Normal("mu_school", mu=0, sigma=2)
    sigma_school = pm.HalfNormal("sigma_school", sigma=1)
    school_effect = pm.Normal("school_effect", mu=mu_school,
                              sigma=sigma_school, shape=n_schools)

    # Neighborhood effects
    mu_neighborhood = pm.Normal("mu_neighborhood", mu=0, sigma=2)
    sigma_neighborhood = pm.HalfNormal("sigma_neighborhood", sigma=1)
    neighborhood_effect = pm.Normal("neighborhood_effect",
                                     mu=mu_neighborhood,
                                     sigma=sigma_neighborhood,
                                     shape=n_neighborhoods)

    # Combined effect
    mu = (alpha + school_effect[school_idx] +
          neighborhood_effect[neighborhood_idx] + pm.math.dot(X, beta))
```

## Three-Level Models

Observations within groups within higher-level groups:

```python
# Students (i) within classrooms (j) within schools (k)
with pm.Model():
    # School level (Level 3)
    mu_school = pm.Normal("mu_school", mu=0, sigma=5)
    sigma_school = pm.HalfNormal("sigma_school", sigma=2)
    school_effect = pm.Normal("school_effect", mu=mu_school,
                              sigma=sigma_school, shape=n_schools)

    # Classroom level (Level 2)
    sigma_classroom = pm.HalfNormal("sigma_classroom", sigma=1)
    classroom_effect = pm.Normal("classroom_effect",
                                  mu=school_effect[school_of_classroom],
                                  sigma=sigma_classroom,
                                  shape=n_classrooms)

    # Student level (Level 1)
    sigma = pm.HalfNormal("sigma", sigma=1)
    mu = classroom_effect[classroom_idx] + pm.math.dot(X, beta)
    y_obs = pm.Normal("y_obs", mu=mu, sigma=sigma, observed=y)
```

## Example: Wage Panel Data

```python
import pymc as pm
import arviz as az
import numpy as np
import pandas as pd

# Simulated panel: workers over time
np.random.seed(42)
n_workers = 50
n_years = 10
n_obs = n_workers * n_years

# True parameters
true_mu_alpha = 2.5  # Average log wage
true_sigma_alpha = 0.3  # Worker heterogeneity
true_beta_exp = 0.02  # Return to experience

# Generate data
worker_id = np.repeat(np.arange(n_workers), n_years)
year = np.tile(np.arange(n_years), n_workers)
experience = year + np.random.poisson(5, n_workers)[worker_id]

# Worker fixed effects
worker_effects = np.random.normal(true_mu_alpha, true_sigma_alpha, n_workers)

# Log wages
log_wage = (worker_effects[worker_id] +
            true_beta_exp * experience +
            np.random.normal(0, 0.2, n_obs))

# Fit model
with pm.Model() as wage_panel:
    # Hyperpriors
    mu_worker = pm.Normal("mu_worker", mu=0, sigma=5)
    sigma_worker = pm.HalfNormal("sigma_worker", sigma=1)

    # Worker effects (non-centered)
    worker_offset = pm.Normal("worker_offset", mu=0, sigma=1, shape=n_workers)
    worker_effect = pm.Deterministic("worker_effect",
                                      mu_worker + sigma_worker * worker_offset)

    # Experience effect
    beta_exp = pm.Normal("beta_exp", mu=0, sigma=0.1)

    # Residual
    sigma = pm.HalfNormal("sigma", sigma=0.5)

    # Expected log wage
    mu = worker_effect[worker_id] + beta_exp * experience

    # Likelihood
    log_wage_obs = pm.Normal("log_wage", mu=mu, sigma=sigma, observed=log_wage)

    # Sample
    trace = pm.sample(2000, tune=1000, chains=4, random_seed=42)

# Results
print(az.summary(trace, var_names=["mu_worker", "sigma_worker", "beta_exp", "sigma"]))

# Compare to true values
print(f"\nTrue mu_worker: {true_mu_alpha}, Estimated: {trace.posterior['mu_worker'].mean():.3f}")
print(f"True sigma_worker: {true_sigma_alpha}, Estimated: {trace.posterior['sigma_worker'].mean():.3f}")
print(f"True beta_exp: {true_beta_exp}, Estimated: {trace.posterior['beta_exp'].mean():.4f}")
```

## Shrinkage Visualization

```python
def plot_shrinkage(trace, y_bar_group, n_group, group_names=None):
    """
    Visualize shrinkage toward population mean.

    Parameters
    ----------
    trace : arviz.InferenceData
        Posterior samples
    y_bar_group : array
        Sample means by group
    n_group : array
        Sample sizes by group
    group_names : list, optional
        Names for groups
    """
    import matplotlib.pyplot as plt

    n_groups = len(y_bar_group)
    if group_names is None:
        group_names = [f"Group {i}" for i in range(n_groups)]

    # Get posterior means
    alpha_post = trace.posterior["alpha"].mean(dim=["chain", "draw"]).values
    mu_post = trace.posterior["mu_alpha"].mean().values

    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot raw means vs posterior means
    for i in range(n_groups):
        ax.annotate("",
                    xy=(alpha_post[i], i),
                    xytext=(y_bar_group[i], i),
                    arrowprops=dict(arrowstyle="->", color="steelblue", lw=1))

    ax.scatter(y_bar_group, range(n_groups), s=n_group*5, c="orange",
               label="Sample Mean", zorder=3)
    ax.scatter(alpha_post, range(n_groups), s=50, c="steelblue",
               label="Posterior Mean", marker="s", zorder=3)
    ax.axvline(mu_post, color="red", linestyle="--", label="Population Mean")

    ax.set_yticks(range(n_groups))
    ax.set_yticklabels(group_names)
    ax.set_xlabel("Effect Estimate")
    ax.set_title("Shrinkage: Sample Means → Posterior Means")
    ax.legend()

    plt.tight_layout()
    return fig
```

## Model Comparison

Compare pooling strategies:

```python
import pymc as pm
import arviz as az

# Fit all three models
models = {}

# Complete pooling
with pm.Model() as complete_pooling:
    alpha = pm.Normal("alpha", mu=0, sigma=10)
    beta = pm.Normal("beta", mu=0, sigma=2.5)
    sigma = pm.HalfNormal("sigma", sigma=1)
    mu = alpha + beta * X
    y_obs = pm.Normal("y", mu=mu, sigma=sigma, observed=y)
    models["complete"] = pm.sample(2000, tune=1000)

# No pooling
with pm.Model() as no_pooling:
    alpha = pm.Normal("alpha", mu=0, sigma=10, shape=n_groups)
    beta = pm.Normal("beta", mu=0, sigma=2.5)
    sigma = pm.HalfNormal("sigma", sigma=1)
    mu = alpha[groups] + beta * X
    y_obs = pm.Normal("y", mu=mu, sigma=sigma, observed=y)
    models["no_pool"] = pm.sample(2000, tune=1000)

# Partial pooling
with pm.Model() as partial_pooling:
    mu_alpha = pm.Normal("mu_alpha", mu=0, sigma=10)
    sigma_alpha = pm.HalfNormal("sigma_alpha", sigma=2)
    alpha = pm.Normal("alpha", mu=mu_alpha, sigma=sigma_alpha, shape=n_groups)
    beta = pm.Normal("beta", mu=0, sigma=2.5)
    sigma = pm.HalfNormal("sigma", sigma=1)
    mu = alpha[groups] + beta * X
    y_obs = pm.Normal("y", mu=mu, sigma=sigma, observed=y)
    models["partial"] = pm.sample(2000, tune=1000)

# Compare with LOO-CV
comparison = az.compare(models, ic="loo")
print(comparison)
```

## References

- Gelman, A., & Hill, J. (2006). Data Analysis Using Regression and Multilevel/Hierarchical Models.
- McElreath, R. (2020). Statistical Rethinking, 2nd Edition.
- Raudenbush, S. W., & Bryk, A. S. (2002). Hierarchical Linear Models.
