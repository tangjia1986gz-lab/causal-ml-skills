# MCMC Diagnostics

## Overview

Markov Chain Monte Carlo (MCMC) methods draw samples from the posterior distribution. Because these are approximate methods, we must verify the sampler has converged to the target distribution before trusting results.

## Key Diagnostic Concepts

### Convergence vs Mixing

- **Convergence**: Chains have found the high-probability region of the posterior
- **Mixing**: Chains are efficiently exploring the posterior space
- **Stationarity**: Chain distribution no longer changes with more iterations

### When to Worry

If diagnostics fail:
- Parameter estimates may be wrong
- Uncertainty intervals may be misleading
- Model may need reparameterization

## Essential Diagnostics

### 1. Rhat (Potential Scale Reduction Factor)

Compares between-chain and within-chain variance.

**Interpretation:**
- Rhat = 1.0: Chains have converged
- Rhat < 1.01: Acceptable
- Rhat > 1.01: Chains have NOT converged - do not trust results

```python
import arviz as az

# Get Rhat for all parameters
rhat = az.rhat(trace)
print(rhat)

# Check if any Rhat > 1.01
problematic = {k: v for k, v in rhat.items() if v > 1.01}
if problematic:
    print(f"WARNING: These parameters have Rhat > 1.01: {problematic}")
```

### 2. Effective Sample Size (ESS)

Number of independent samples after accounting for autocorrelation.

**Types:**
- **ESS bulk**: For central tendency estimates (mean)
- **ESS tail**: For tail estimates (quantiles)

**Interpretation:**
- ESS > 400: Adequate for most purposes
- ESS > 100: Minimum for rough estimates
- ESS < 100: Increase samples or fix sampling issues

```python
# Bulk and tail ESS
ess_bulk = az.ess(trace, method="bulk")
ess_tail = az.ess(trace, method="tail")

print("Bulk ESS:", ess_bulk)
print("Tail ESS:", ess_tail)

# Check minimum ESS
min_ess = min(min(ess_bulk.values()), min(ess_tail.values()))
if min_ess < 400:
    print(f"WARNING: Minimum ESS is {min_ess:.0f}, consider more samples")
```

### 3. Trace Plots

Visual inspection of chain behavior.

```python
import matplotlib.pyplot as plt

# Trace plots for key parameters
az.plot_trace(trace, var_names=["alpha", "beta", "sigma"])
plt.tight_layout()
plt.savefig("trace_plots.png", dpi=150)
plt.show()
```

**What to Look For:**

Good trace plot:
- Chains overlap (different colors mixed together)
- "Fuzzy caterpillar" appearance
- No trends or drifts
- Similar distributions across chains (right panel)

Bad trace plot:
- Chains separated (stuck in different regions)
- Trends (still converging)
- Long periods stuck at one value
- Different chain distributions

### 4. Rank Plots

More robust alternative to trace plots.

```python
az.plot_rank(trace, var_names=["beta"])
plt.show()
```

**Interpretation:**
- Uniform distribution of ranks across chains = good mixing
- Distinct patterns per chain = poor mixing

## Additional Diagnostics

### 5. Autocorrelation

High autocorrelation means inefficient sampling.

```python
az.plot_autocorr(trace, var_names=["beta"], combined=True)
plt.show()
```

**Interpretation:**
- Rapid decay to zero = good
- Slow decay = high autocorrelation, low ESS

### 6. Divergences (NUTS Sampler)

Divergent transitions indicate the sampler encountered difficult geometry.

```python
# Check for divergences
divergences = trace.sample_stats.diverging.sum().values
print(f"Number of divergences: {divergences}")

if divergences > 0:
    print("WARNING: Divergences detected - consider:")
    print("  1. Increase target_accept (e.g., 0.95 or 0.99)")
    print("  2. Reparameterize the model")
    print("  3. Use more informative priors")

# Plot divergences
az.plot_parallel(trace, var_names=["alpha", "beta", "sigma"])
plt.show()
```

### 7. Energy Plot (NUTS)

Compares marginal and energy transition distributions.

```python
az.plot_energy(trace)
plt.show()
```

**Interpretation:**
- Overlapping distributions = good
- Large gap = poor exploration of posterior

## Posterior Predictive Checks

Verify the model fits the data.

```python
import pymc as pm

with model:
    ppc = pm.sample_posterior_predictive(trace)

# Add to inference data
idata = az.from_pymc(trace, posterior_predictive=ppc)

# Visual check
az.plot_ppc(idata, kind="cumulative", num_pp_samples=100)
plt.show()

# Overlay check
az.plot_ppc(idata, kind="kde", num_pp_samples=50)
plt.show()
```

**Interpretation:**
- Observed data (dark line) should fall within posterior predictive distribution
- Systematic deviations suggest model misspecification

## Diagnostic Summary Report

```python
def mcmc_diagnostic_report(trace, var_names=None):
    """Generate comprehensive MCMC diagnostic report."""
    import warnings

    print("=" * 60)
    print("MCMC DIAGNOSTIC REPORT")
    print("=" * 60)

    # Summary statistics
    summary = az.summary(trace, var_names=var_names)
    print("\nParameter Summary:")
    print(summary)

    # Convergence checks
    print("\n" + "-" * 40)
    print("CONVERGENCE CHECKS")
    print("-" * 40)

    # Rhat
    rhat = az.rhat(trace)
    max_rhat = max([v.max() if hasattr(v, 'max') else v
                    for v in rhat.values()])
    print(f"\nMax Rhat: {max_rhat:.3f}")
    if max_rhat > 1.01:
        print("  WARNING: Rhat > 1.01 detected - chains have not converged!")
    else:
        print("  OK: All Rhat < 1.01")

    # ESS
    ess_bulk = az.ess(trace, method="bulk")
    ess_tail = az.ess(trace, method="tail")

    min_bulk = min([v.min() if hasattr(v, 'min') else v
                    for v in ess_bulk.values()])
    min_tail = min([v.min() if hasattr(v, 'min') else v
                    for v in ess_tail.values()])

    print(f"\nMin Bulk ESS: {min_bulk:.0f}")
    print(f"Min Tail ESS: {min_tail:.0f}")

    if min_bulk < 400 or min_tail < 400:
        print("  WARNING: ESS < 400 - consider more samples")
    else:
        print("  OK: ESS > 400")

    # Divergences
    if hasattr(trace, 'sample_stats') and 'diverging' in trace.sample_stats:
        div = trace.sample_stats.diverging.sum().values
        print(f"\nDivergences: {div}")
        if div > 0:
            print("  WARNING: Divergent transitions detected!")
        else:
            print("  OK: No divergences")

    # Overall assessment
    print("\n" + "=" * 60)
    issues = []
    if max_rhat > 1.01:
        issues.append("Rhat > 1.01")
    if min_bulk < 400:
        issues.append("Low bulk ESS")
    if min_tail < 400:
        issues.append("Low tail ESS")
    if hasattr(trace, 'sample_stats') and 'diverging' in trace.sample_stats:
        if trace.sample_stats.diverging.sum().values > 0:
            issues.append("Divergences")

    if issues:
        print(f"ISSUES DETECTED: {', '.join(issues)}")
        print("Recommendations:")
        if "Rhat > 1.01" in issues:
            print("  - Run longer chains (more tune and/or draws)")
            print("  - Check for multimodality")
        if "Low bulk ESS" in issues or "Low tail ESS" in issues:
            print("  - Increase number of draws")
            print("  - Consider reparameterization")
        if "Divergences" in issues:
            print("  - Increase target_accept (e.g., 0.95)")
            print("  - Use non-centered parameterization")
            print("  - Add informative priors")
    else:
        print("ALL DIAGNOSTICS PASSED")
    print("=" * 60)

    return summary


# Usage
report = mcmc_diagnostic_report(trace, var_names=["alpha", "beta", "sigma"])
```

## Fixing Common Issues

### Issue: High Rhat

**Solutions:**
1. Run longer chains
2. Increase tune (warmup) samples
3. Check for multimodality in posterior

```python
# Increase samples
trace = pm.sample(draws=4000, tune=2000, chains=4)
```

### Issue: Low ESS

**Solutions:**
1. Increase number of draws
2. Reparameterize (non-centered)
3. Thin chains (last resort)

```python
# More samples
trace = pm.sample(draws=5000, tune=2000, chains=4)
```

### Issue: Divergences

**Solutions:**
1. Increase target_accept
2. Non-centered parameterization
3. More informative priors

```python
# Higher acceptance rate
trace = pm.sample(draws=2000, tune=1000, target_accept=0.95)
```

### Non-Centered Parameterization

Transform hierarchical models to improve sampling:

```python
# Centered (can be problematic)
with pm.Model():
    mu = pm.Normal("mu", 0, 1)
    sigma = pm.HalfNormal("sigma", 1)
    theta = pm.Normal("theta", mu, sigma, shape=n_groups)  # Problematic

# Non-centered (better sampling)
with pm.Model():
    mu = pm.Normal("mu", 0, 1)
    sigma = pm.HalfNormal("sigma", 1)
    theta_raw = pm.Normal("theta_raw", 0, 1, shape=n_groups)
    theta = pm.Deterministic("theta", mu + sigma * theta_raw)  # Reparameterized
```

## Diagnostic Visualization Suite

```python
def plot_diagnostic_suite(trace, var_names, figsize=(15, 10)):
    """Create comprehensive diagnostic visualization."""
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=figsize)

    # Trace plots
    ax1 = fig.add_subplot(2, 2, 1)
    az.plot_trace(trace, var_names=var_names, combined=True, ax=ax1)

    # Posterior distributions
    ax2 = fig.add_subplot(2, 2, 2)
    az.plot_posterior(trace, var_names=var_names, ax=ax2)

    # Autocorrelation
    ax3 = fig.add_subplot(2, 2, 3)
    az.plot_autocorr(trace, var_names=var_names, combined=True, ax=ax3)

    # Rank plot
    ax4 = fig.add_subplot(2, 2, 4)
    az.plot_rank(trace, var_names=var_names, ax=ax4)

    plt.tight_layout()
    return fig


# Usage
fig = plot_diagnostic_suite(trace, var_names=["beta"])
plt.savefig("diagnostics.png", dpi=150)
```

## References

- Vehtari, A., et al. (2021). Rank-normalization, folding, and localization: An improved Rhat for assessing convergence of MCMC.
- Betancourt, M. (2017). A Conceptual Introduction to Hamiltonian Monte Carlo.
- Gabry, J., et al. (2019). Visualization in Bayesian workflow.
