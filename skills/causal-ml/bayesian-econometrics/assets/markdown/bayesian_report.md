# Bayesian Analysis Report

## Executive Summary

| Metric | Value |
|--------|-------|
| **Model Type** | {{ model_type }} |
| **N Observations** | {{ n_obs }} |
| **N Parameters** | {{ n_params }} |
| **Estimation Method** | MCMC (NUTS) |
| **Convergence Status** | {{ convergence_status }} |

### Key Finding

{{ key_finding }}

---

## 1. Model Specification

### 1.1 Outcome Variable
- **Name**: {{ outcome_name }}
- **Type**: {{ outcome_type }}
- **Distribution**: {{ outcome_dist }}

### 1.2 Predictors
| Variable | Type | Prior |
|----------|------|-------|
{{ predictor_table }}

### 1.3 Prior Specification

**Rationale for prior choices:**

{{ prior_rationale }}

**Prior predictive check:**

{{ prior_predictive_summary }}

---

## 2. Posterior Results

### 2.1 Parameter Summary

| Parameter | Mean | SD | 95% HDI Lower | 95% HDI Upper | ESS | Rhat |
|-----------|------|-----|---------------|---------------|-----|------|
{{ parameter_table }}

### 2.2 Key Effect Estimates

**Primary Effect: {{ primary_effect_name }}**

- **Posterior Mean**: {{ effect_mean }}
- **Posterior SD**: {{ effect_sd }}
- **95% Highest Density Interval**: [{{ hdi_lower }}, {{ hdi_upper }}]
- **Probability effect > 0**: {{ prob_positive }}
- **Probability effect < 0**: {{ prob_negative }}

**Interpretation**: {{ effect_interpretation }}

### 2.3 Posterior Distributions

![Posterior Distributions]({{ posterior_plot_path }})

*Figure 1: Posterior density plots with 95% HDI (shaded) and posterior mean (vertical line).*

---

## 3. MCMC Diagnostics

### 3.1 Convergence Summary

| Diagnostic | Value | Status |
|------------|-------|--------|
| Max Rhat | {{ max_rhat }} | {{ rhat_status }} |
| Min Bulk ESS | {{ min_bulk_ess }} | {{ ess_bulk_status }} |
| Min Tail ESS | {{ min_tail_ess }} | {{ ess_tail_status }} |
| Divergences | {{ n_divergences }} | {{ divergence_status }} |

**Overall**: {{ overall_diagnostic_status }}

### 3.2 Trace Plots

![Trace Plots]({{ trace_plot_path }})

*Figure 2: Trace plots showing chain mixing. Good mixing appears as overlapping "fuzzy caterpillars."*

### 3.3 Rank Plots

![Rank Plots]({{ rank_plot_path }})

*Figure 3: Rank plots for assessing chain convergence. Uniform distributions indicate good mixing.*

---

## 4. Model Checking

### 4.1 Posterior Predictive Check

![Posterior Predictive Check]({{ ppc_plot_path }})

*Figure 4: Posterior predictive distribution (light lines) compared to observed data (dark line).*

**Assessment**: {{ ppc_assessment }}

### 4.2 Residual Analysis

{{ residual_analysis }}

---

## 5. Sensitivity Analysis

### 5.1 Prior Sensitivity

| Prior Scale | Posterior Mean | 95% HDI | P(effect > 0) |
|-------------|----------------|---------|---------------|
{{ sensitivity_table }}

**Robustness Assessment**: {{ robustness_assessment }}

### 5.2 Influence Analysis

{{ influence_analysis }}

---

## 6. Causal Interpretation

### 6.1 Identification Strategy

{{ identification_strategy }}

### 6.2 Assumptions

| Assumption | Assessment | Evidence |
|------------|------------|----------|
{{ assumptions_table }}

### 6.3 Causal Effect Statement

{{ causal_statement }}

---

## 7. Conclusions

### 7.1 Main Findings

1. {{ finding_1 }}
2. {{ finding_2 }}
3. {{ finding_3 }}

### 7.2 Limitations

{{ limitations }}

### 7.3 Recommendations

{{ recommendations }}

---

## Technical Appendix

### A.1 Sampling Details

| Setting | Value |
|---------|-------|
| Sampler | NUTS |
| Draws | {{ draws }} |
| Tune | {{ tune }} |
| Chains | {{ chains }} |
| Target Accept | {{ target_accept }} |
| Random Seed | {{ seed }} |

### A.2 Software Environment

```
Python: {{ python_version }}
PyMC: {{ pymc_version }}
ArviZ: {{ arviz_version }}
NumPy: {{ numpy_version }}
```

### A.3 Computation Time

- Total sampling time: {{ sampling_time }}
- Average time per draw: {{ time_per_draw }}

### A.4 Full Model Code

```python
{{ model_code }}
```

---

## References

{{ references }}

---

*Report generated: {{ timestamp }}*
*Analysis by: {{ analyst }}*
