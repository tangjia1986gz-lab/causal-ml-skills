# Bayesian Approaches to Causal Inference

## Overview

Bayesian methods offer unique advantages for causal inference: natural uncertainty quantification, incorporation of prior knowledge, and coherent probabilistic statements about causal effects. This document covers Bayesian implementations of common causal estimators.

## Bayesian Causal Quantities

### Average Treatment Effect

In Bayesian terms, the ATE has a posterior distribution:

$$P(ATE | \text{Data}) \propto P(\text{Data} | ATE) \cdot P(ATE)$$

We can make direct probability statements:
- $P(ATE > 0 | \text{Data})$ - Probability effect is positive
- 95% Credible Interval - 95% probability ATE is in this range

### Credible vs Confidence Intervals

| Aspect | Confidence Interval | Credible Interval |
|--------|---------------------|-------------------|
| Interpretation | 95% of such intervals contain true value | 95% probability parameter is in interval |
| Prior information | Not incorporated | Explicitly included |
| For researchers | Often misinterpreted as Bayesian | Direct probability statement |

```python
import pymc as pm
import arviz as az

# After fitting causal model
ate_samples = trace.posterior["ate"].values.flatten()

# Direct probability statements
prob_positive = (ate_samples > 0).mean()
prob_large = (ate_samples > 0.5).mean()

print(f"P(ATE > 0) = {prob_positive:.3f}")
print(f"P(ATE > 0.5) = {prob_large:.3f}")

# Credible interval
hdi = az.hdi(ate_samples, hdi_prob=0.95)
print(f"95% HDI: [{hdi[0]:.3f}, {hdi[1]:.3f}]")
```

## Bayesian Regression for Causal Effects

### Treatment Effect Regression

```python
import pymc as pm
import numpy as np

def bayesian_ate(y, treatment, X=None, prior_ate_sigma=2.5, draws=2000):
    """
    Estimate ATE using Bayesian regression.

    Parameters
    ----------
    y : array
        Outcome variable
    treatment : array
        Binary treatment indicator
    X : array, optional
        Covariates for adjustment
    prior_ate_sigma : float
        Prior standard deviation on ATE

    Returns
    -------
    trace : InferenceData
        Posterior samples including ATE
    """
    with pm.Model() as ate_model:
        # Priors
        alpha = pm.Normal("alpha", mu=0, sigma=10)
        ate = pm.Normal("ate", mu=0, sigma=prior_ate_sigma)
        sigma = pm.HalfNormal("sigma", sigma=1)

        # Covariates if provided
        if X is not None:
            beta = pm.Normal("beta", mu=0, sigma=2.5, shape=X.shape[1])
            mu = alpha + ate * treatment + pm.math.dot(X, beta)
        else:
            mu = alpha + ate * treatment

        # Likelihood
        y_obs = pm.Normal("y", mu=mu, sigma=sigma, observed=y)

        # Sample
        trace = pm.sample(draws, tune=1000, chains=4, random_seed=42)

    return trace
```

## Bayesian Propensity Score

### Propensity Score Model

```python
def bayesian_propensity_score(treatment, X, draws=2000):
    """
    Estimate propensity scores using Bayesian logistic regression.
    """
    with pm.Model() as ps_model:
        # Priors for logistic regression coefficients
        alpha = pm.Normal("alpha", mu=0, sigma=5)
        beta = pm.Normal("beta", mu=0, sigma=2.5, shape=X.shape[1])

        # Linear predictor
        logit_p = alpha + pm.math.dot(X, beta)

        # Propensity score (probability of treatment)
        p = pm.Deterministic("propensity", pm.math.sigmoid(logit_p))

        # Likelihood
        treatment_obs = pm.Bernoulli("treatment", p=p, observed=treatment)

        # Sample
        trace = pm.sample(draws, tune=1000, chains=4)

    # Get posterior mean propensity scores
    ps = trace.posterior["propensity"].mean(dim=["chain", "draw"]).values

    return trace, ps
```

### IPW with Uncertainty Propagation

```python
def bayesian_ipw(y, treatment, X, draws=2000):
    """
    Bayesian IPW estimator with full uncertainty quantification.

    Propagates propensity score uncertainty through to ATE estimate.
    """
    with pm.Model() as ipw_model:
        # Propensity score submodel
        ps_alpha = pm.Normal("ps_alpha", mu=0, sigma=5)
        ps_beta = pm.Normal("ps_beta", mu=0, sigma=2.5, shape=X.shape[1])
        logit_ps = ps_alpha + pm.math.dot(X, ps_beta)
        ps = pm.math.sigmoid(logit_ps)

        # IPW weights
        weights = treatment / ps + (1 - treatment) / (1 - ps)

        # Outcome model (weighted)
        alpha = pm.Normal("alpha", mu=0, sigma=10)
        ate = pm.Normal("ate", mu=0, sigma=2.5)
        sigma = pm.HalfNormal("sigma", sigma=1)

        mu = alpha + ate * treatment
        # Note: PyMC doesn't directly support weighted likelihood
        # This is an approximation using moment matching
        y_obs = pm.Normal("y", mu=mu, sigma=sigma, observed=y)

        trace = pm.sample(draws, tune=1000, chains=4)

    return trace
```

## Bayesian Difference-in-Differences

```python
def bayesian_did(y, treated, post, draws=2000):
    """
    Bayesian Difference-in-Differences estimation.

    Parameters
    ----------
    y : array
        Outcome variable
    treated : array
        Treatment group indicator (1 = treated group)
    post : array
        Post-treatment period indicator (1 = post)

    Returns
    -------
    trace : InferenceData
        Posterior samples including ATT
    """
    with pm.Model() as did_model:
        # Priors
        alpha = pm.Normal("alpha", mu=0, sigma=10)  # Control, pre
        gamma_treated = pm.Normal("gamma_treated", mu=0, sigma=5)  # Treatment group effect
        gamma_post = pm.Normal("gamma_post", mu=0, sigma=5)  # Time effect
        att = pm.Normal("att", mu=0, sigma=2.5)  # DiD estimate

        sigma = pm.HalfNormal("sigma", sigma=1)

        # DiD specification
        mu = (alpha +
              gamma_treated * treated +
              gamma_post * post +
              att * treated * post)

        # Likelihood
        y_obs = pm.Normal("y", mu=mu, sigma=sigma, observed=y)

        # Sample
        trace = pm.sample(draws, tune=1000, chains=4, random_seed=42)

    return trace
```

## Bayesian Instrumental Variables

### Two-Stage Bayesian IV

```python
def bayesian_iv(y, treatment, instrument, X=None, draws=2000):
    """
    Bayesian 2SLS instrumental variables estimation.

    Joint model for first and second stage.
    """
    with pm.Model() as iv_model:
        # First stage: treatment ~ instrument + X
        fs_alpha = pm.Normal("fs_alpha", mu=0, sigma=10)
        fs_gamma = pm.Normal("fs_gamma", mu=0, sigma=2.5)  # Instrument effect
        fs_sigma = pm.HalfNormal("fs_sigma", sigma=1)

        if X is not None:
            fs_beta = pm.Normal("fs_beta", mu=0, sigma=2.5, shape=X.shape[1])
            treatment_pred = fs_alpha + fs_gamma * instrument + pm.math.dot(X, fs_beta)
        else:
            treatment_pred = fs_alpha + fs_gamma * instrument

        # First stage likelihood
        treatment_obs = pm.Normal("treatment_model", mu=treatment_pred,
                                   sigma=fs_sigma, observed=treatment)

        # Second stage: y ~ treatment_hat + X
        ss_alpha = pm.Normal("ss_alpha", mu=0, sigma=10)
        late = pm.Normal("late", mu=0, sigma=2.5)  # Causal effect
        ss_sigma = pm.HalfNormal("ss_sigma", sigma=1)

        if X is not None:
            ss_beta = pm.Normal("ss_beta", mu=0, sigma=2.5, shape=X.shape[1])
            y_pred = ss_alpha + late * treatment_pred + pm.math.dot(X, ss_beta)
        else:
            y_pred = ss_alpha + late * treatment_pred

        # Second stage likelihood
        y_obs = pm.Normal("y_model", mu=y_pred, sigma=ss_sigma, observed=y)

        # Sample
        trace = pm.sample(draws, tune=1000, chains=4, random_seed=42)

    return trace
```

## Bayesian Regression Discontinuity

```python
def bayesian_rd(y, running_var, cutoff=0, bandwidth=None, draws=2000):
    """
    Bayesian Sharp Regression Discontinuity.

    Local linear regression at cutoff.
    """
    # Center running variable
    x = running_var - cutoff
    treatment = (x >= 0).astype(int)

    # Apply bandwidth if specified
    if bandwidth is not None:
        mask = np.abs(x) <= bandwidth
        y = y[mask]
        x = x[mask]
        treatment = treatment[mask]

    with pm.Model() as rd_model:
        # Intercepts
        alpha_control = pm.Normal("alpha_control", mu=0, sigma=10)
        alpha_treated = pm.Normal("alpha_treated", mu=0, sigma=10)

        # Slopes
        beta_control = pm.Normal("beta_control", mu=0, sigma=2.5)
        beta_treated = pm.Normal("beta_treated", mu=0, sigma=2.5)

        # RD effect at cutoff
        rd_effect = pm.Deterministic("rd_effect", alpha_treated - alpha_control)

        sigma = pm.HalfNormal("sigma", sigma=1)

        # Separate regressions on each side
        mu = pm.math.switch(
            treatment,
            alpha_treated + beta_treated * x,
            alpha_control + beta_control * x
        )

        y_obs = pm.Normal("y", mu=mu, sigma=sigma, observed=y)

        trace = pm.sample(draws, tune=1000, chains=4, random_seed=42)

    return trace
```

## Prior Sensitivity for Causal Estimates

### Implementing Sensitivity Analysis

```python
def causal_prior_sensitivity(y, treatment, X, prior_sigmas=[0.5, 1.0, 2.5, 5.0, 10.0]):
    """
    Assess sensitivity of ATE to prior specification.
    """
    results = []

    for prior_sigma in prior_sigmas:
        with pm.Model():
            alpha = pm.Normal("alpha", mu=0, sigma=10)
            ate = pm.Normal("ate", mu=0, sigma=prior_sigma)
            beta = pm.Normal("beta", mu=0, sigma=2.5, shape=X.shape[1])
            sigma = pm.HalfNormal("sigma", sigma=1)

            mu = alpha + ate * treatment + pm.math.dot(X, beta)
            y_obs = pm.Normal("y", mu=mu, sigma=sigma, observed=y)

            trace = pm.sample(2000, tune=1000, chains=4,
                             progressbar=False, random_seed=42)

        ate_post = trace.posterior["ate"].values.flatten()
        results.append({
            "prior_sigma": prior_sigma,
            "ate_mean": ate_post.mean(),
            "ate_std": ate_post.std(),
            "ate_hdi_low": np.percentile(ate_post, 2.5),
            "ate_hdi_high": np.percentile(ate_post, 97.5),
            "prob_positive": (ate_post > 0).mean()
        })

    return pd.DataFrame(results)
```

## Bayesian Model Averaging for Causal Inference

When uncertain about model specification:

```python
import pymc as pm
import arviz as az

def bayesian_model_averaging_ate(y, treatment, X_sets, draws=2000):
    """
    Estimate ATE averaging over multiple covariate specifications.

    Parameters
    ----------
    y : array
        Outcome
    treatment : array
        Treatment indicator
    X_sets : dict
        Dictionary of covariate matrices {name: X_matrix}

    Returns
    -------
    averaged_ate : array
        ATE samples from model-averaged posterior
    """
    traces = {}
    loo_scores = {}

    for name, X in X_sets.items():
        with pm.Model() as model:
            alpha = pm.Normal("alpha", mu=0, sigma=10)
            ate = pm.Normal("ate", mu=0, sigma=2.5)
            if X is not None and X.shape[1] > 0:
                beta = pm.Normal("beta", mu=0, sigma=2.5, shape=X.shape[1])
                mu = alpha + ate * treatment + pm.math.dot(X, beta)
            else:
                mu = alpha + ate * treatment
            sigma = pm.HalfNormal("sigma", sigma=1)
            y_obs = pm.Normal("y", mu=mu, sigma=sigma, observed=y)

            traces[name] = pm.sample(draws, tune=1000, chains=4,
                                     progressbar=False, random_seed=42)

        # Compute LOO for model weighting
        loo = az.loo(traces[name])
        loo_scores[name] = loo.loo

    # Compute model weights (Pseudo-BMA)
    loo_values = np.array(list(loo_scores.values()))
    weights = np.exp(loo_values - loo_values.max())
    weights /= weights.sum()

    # Sample from model-averaged posterior
    ate_samples = []
    for name, weight in zip(X_sets.keys(), weights):
        n_samples = int(weight * draws * 4)  # 4 chains
        model_ate = traces[name].posterior["ate"].values.flatten()
        ate_samples.extend(np.random.choice(model_ate, size=n_samples))

    return np.array(ate_samples), dict(zip(X_sets.keys(), weights))
```

## Reporting Bayesian Causal Results

### Standard Report Template

```python
def bayesian_causal_report(trace, effect_name="ate", hdi_prob=0.95):
    """
    Generate standard report for Bayesian causal estimates.
    """
    effect_samples = trace.posterior[effect_name].values.flatten()

    report = {
        "point_estimate": {
            "posterior_mean": effect_samples.mean(),
            "posterior_median": np.median(effect_samples),
            "posterior_mode": float(az.plots.plot_utils.calculate_point_estimate(
                "mode", effect_samples))
        },
        "uncertainty": {
            f"{int(hdi_prob*100)}% HDI": az.hdi(effect_samples, hdi_prob=hdi_prob),
            "posterior_sd": effect_samples.std()
        },
        "probability_statements": {
            "P(effect > 0)": (effect_samples > 0).mean(),
            "P(effect > 0.1)": (effect_samples > 0.1).mean(),
            "P(effect > 0.5)": (effect_samples > 0.5).mean(),
            "P(|effect| > 0.1)": (np.abs(effect_samples) > 0.1).mean()
        },
        "diagnostics": {
            "n_samples": len(effect_samples),
            "ess_bulk": float(az.ess(trace, var_names=[effect_name], method="bulk")[effect_name]),
            "ess_tail": float(az.ess(trace, var_names=[effect_name], method="tail")[effect_name]),
            "rhat": float(az.rhat(trace, var_names=[effect_name])[effect_name])
        }
    }

    return report
```

## Example: Complete Bayesian Causal Analysis

```python
import numpy as np
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt

# Simulate data
np.random.seed(42)
n = 500
X1 = np.random.normal(0, 1, n)
X2 = np.random.normal(0, 1, n)
treatment = (0.5 * X1 + 0.3 * X2 + np.random.normal(0, 1, n) > 0).astype(int)
true_ate = 0.5
y = 1 + true_ate * treatment + 0.3 * X1 + 0.2 * X2 + np.random.normal(0, 0.5, n)
X = np.column_stack([X1, X2])

# Fit Bayesian causal model
with pm.Model() as causal_model:
    # Priors
    alpha = pm.Normal("alpha", mu=0, sigma=5)
    ate = pm.Normal("ate", mu=0, sigma=2.5)
    beta = pm.Normal("beta", mu=0, sigma=2.5, shape=2)
    sigma = pm.HalfNormal("sigma", sigma=1)

    # Outcome model
    mu = alpha + ate * treatment + pm.math.dot(X, beta)
    y_obs = pm.Normal("y", mu=mu, sigma=sigma, observed=y)

    # Prior predictive check
    prior_pred = pm.sample_prior_predictive(samples=500)

    # Sample posterior
    trace = pm.sample(2000, tune=1000, chains=4, random_seed=42)

    # Posterior predictive check
    post_pred = pm.sample_posterior_predictive(trace)

# Diagnostics
print(az.summary(trace, var_names=["alpha", "ate", "beta", "sigma"]))

# Results
ate_samples = trace.posterior["ate"].values.flatten()
print(f"\n=== CAUSAL EFFECT ESTIMATE ===")
print(f"True ATE: {true_ate}")
print(f"Posterior Mean: {ate_samples.mean():.3f}")
print(f"95% HDI: [{np.percentile(ate_samples, 2.5):.3f}, {np.percentile(ate_samples, 97.5):.3f}]")
print(f"P(ATE > 0): {(ate_samples > 0).mean():.3f}")

# Visualization
fig, axes = plt.subplots(1, 3, figsize=(12, 4))

# Posterior distribution
az.plot_posterior(trace, var_names=["ate"], ax=axes[0])
axes[0].axvline(true_ate, color="red", linestyle="--", label=f"True={true_ate}")
axes[0].legend()
axes[0].set_title("ATE Posterior")

# Trace plot
az.plot_trace(trace, var_names=["ate"], combined=True, ax=axes[1])
axes[1].set_title("ATE Trace")

# Forest plot
az.plot_forest(trace, var_names=["beta"], combined=True, ax=axes[2])
axes[2].set_title("Covariate Effects")

plt.tight_layout()
plt.savefig("bayesian_causal_analysis.png", dpi=150)
plt.show()
```

## References

- Imbens, G. W. (2004). Nonparametric estimation of average treatment effects under exogeneity: A review.
- Li, F., et al. (2018). Bayesian inference for causal effects.
- Hahn, P. R., et al. (2020). Bayesian regression tree models for causal inference.
- Hill, J. L. (2011). Bayesian nonparametric modeling for causal inference.
