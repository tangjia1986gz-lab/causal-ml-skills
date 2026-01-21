---
name: bayesian-econometrics
description: Use for Bayesian inference in econometric models. Triggers on Bayesian, prior, posterior, MCMC, Stan, PyMC, credible interval, hierarchical model.
---

# Bayesian Econometrics

## Purpose

This skill provides guidance and tools for applying Bayesian methods to econometric analysis. It covers the complete Bayesian workflow from prior specification through posterior inference, with emphasis on proper diagnostics and interpretation.

## When to Use

Activate this skill when:
- Estimating models with uncertainty quantification via posteriors
- Working with small samples where priors provide regularization
- Building hierarchical/multilevel models with partial pooling
- Needing credible intervals rather than confidence intervals
- Combining prior knowledge with data evidence
- Performing sensitivity analysis on assumptions

## Key Concepts

### Bayesian Inference Framework
- **Prior**: Beliefs about parameters before seeing data
- **Likelihood**: Data generating process given parameters
- **Posterior**: Updated beliefs after observing data
- **Bayes' Theorem**: Posterior ∝ Likelihood × Prior

### MCMC Sampling
- Markov Chain Monte Carlo draws samples from posterior
- Requires convergence diagnostics (Rhat, ESS)
- Chain mixing and stationarity checks essential

### Hierarchical Models
- Parameters themselves have distributions (hyperpriors)
- Partial pooling between complete pooling and no pooling
- Natural framework for panel data and grouped observations

## Workflow

1. **Specify Prior**: Choose priors based on domain knowledge
2. **Prior Predictive Check**: Simulate data from prior to verify sensibility
3. **Fit Model**: Run MCMC sampler
4. **Diagnose**: Check convergence, trace plots, ESS
5. **Posterior Predictive Check**: Compare generated vs observed data
6. **Summarize**: Report posterior means, medians, credible intervals
7. **Sensitivity Analysis**: Vary priors to check robustness

## Tools Provided

### Main Implementation
- `bayesian_estimator.py` - Core estimation functions

### Scripts
- `run_bayesian_model.py` - CLI for model fitting
- `diagnose_mcmc.py` - Diagnostic visualizations
- `prior_sensitivity.py` - Prior sensitivity analysis
- `visualize_posteriors.py` - Posterior plots

### References
- `bayesian_basics.md` - Foundational concepts
- `prior_selection.md` - Prior choice guidance
- `mcmc_diagnostics.md` - Convergence assessment
- `hierarchical_models.md` - Multilevel modeling
- `bayesian_causal.md` - Causal inference applications

## Dependencies

```python
pymc>=5.0
arviz>=0.15
numpy
pandas
matplotlib
```

## Quick Start

```python
from bayesian_estimator import BayesianEstimator

# Initialize estimator
estimator = BayesianEstimator()

# Fit Bayesian regression
result = estimator.fit_bayesian_regression(
    y=outcome,
    X=covariates,
    prior_scale=2.5,  # weakly informative
    draws=2000,
    tune=1000
)

# Check diagnostics
diagnostics = estimator.mcmc_diagnostics(result)

# Get posterior summary
summary = estimator.posterior_summary(result, hdi_prob=0.95)
```

## Integration with CausalML Framework

Returns `CausalOutput` objects compatible with the broader framework:
- Effect estimates as posterior means
- Uncertainty via highest density intervals
- Full posterior samples for downstream analysis
