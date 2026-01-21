# Power Analysis Reference

> **Document Type**: Reference | **Last Updated**: 2025-01
> **Related**: [hypothesis_testing.md](hypothesis_testing.md), [effect_sizes.md](effect_sizes.md), [common_errors.md](common_errors.md)

## Overview

Statistical power is the probability of correctly rejecting a false null hypothesis. Power analysis helps researchers plan sample sizes and evaluate the sensitivity of their studies. Underpowered studies waste resources and often produce misleading results.

---

## 1. Fundamental Concepts

### Definition of Power

**Power = 1 - Beta (Type II Error Rate)**

$$
\text{Power} = P(\text{Reject } H_0 \mid H_1 \text{ is true})
$$

### The Four Parameters

Power analysis involves four interdependent quantities:

| Parameter | Symbol | Description |
|-----------|--------|-------------|
| Effect Size | d, f, r, etc. | Magnitude of the effect |
| Sample Size | n | Number of observations |
| Significance Level | alpha | Type I error rate (typically 0.05) |
| Power | 1 - beta | Probability of detecting true effect |

**Given any three, you can calculate the fourth.**

### Types of Power Analysis

| Type | Known | Unknown | Purpose |
|------|-------|---------|---------|
| A Priori | alpha, power, effect | n | Sample size planning |
| Post-Hoc | alpha, n, effect | power | Evaluate completed study |
| Sensitivity | alpha, power, n | effect | Find detectable effect |
| Criterion | power, n, effect | alpha | Find needed alpha |

---

## 2. Power for Common Tests

### Two-Sample t-Test

**Non-centrality Parameter**:
$$
\lambda = d \sqrt{\frac{n_1 n_2}{n_1 + n_2}}
$$

**Power Function**:
$$
\text{Power} = 1 - F_{t,df}(t_{crit}; \lambda) + F_{t,df}(-t_{crit}; \lambda)
$$

Where $F_{t,df}$ is the non-central t distribution CDF.

**Sample Size Formula** (equal groups, two-sided):
$$
n = \frac{2(z_{1-\alpha/2} + z_{1-\beta})^2}{d^2}
$$

```python
from statistical_analysis import power_ttest

# Calculate power
power = power_ttest(
    effect_size=0.5,  # Cohen's d
    n1=50,
    n2=50,
    alpha=0.05,
    alternative='two-sided'
)
print(f"Power = {power:.3f}")

# Calculate required sample size
n = power_ttest(
    effect_size=0.5,
    power=0.80,
    alpha=0.05,
    alternative='two-sided',
    mode='sample_size'
)
print(f"Required n per group = {n}")

# Calculate minimum detectable effect
mde = power_ttest(
    n1=50,
    n2=50,
    power=0.80,
    alpha=0.05,
    mode='effect_size'
)
print(f"MDE (Cohen's d) = {mde:.3f}")
```

### One-Sample t-Test

$$
n = \frac{(z_{1-\alpha/2} + z_{1-\beta})^2}{d^2}
$$

```python
from statistical_analysis import power_ttest_onesample

n = power_ttest_onesample(
    effect_size=0.3,
    power=0.80,
    alpha=0.05,
    mode='sample_size'
)
```

### Paired t-Test

$$
n_{pairs} = \frac{(z_{1-\alpha/2} + z_{1-\beta})^2}{d^2}
$$

Where d is based on the correlation between pairs:
$$
d = \frac{\mu_d}{\sigma_d} = \frac{\mu_1 - \mu_2}{\sigma \sqrt{2(1-\rho)}}
$$

```python
from statistical_analysis import power_ttest_paired

n = power_ttest_paired(
    effect_size=0.5,
    power=0.80,
    alpha=0.05,
    correlation=0.5,  # Pre-post correlation
    mode='sample_size'
)
```

### One-Way ANOVA

**Non-centrality Parameter**:
$$
\lambda = n \cdot f^2 \cdot k
$$

Where f is Cohen's f for ANOVA.

**Sample Size** (per group):
$$
n = \frac{\lambda}{f^2 \cdot k}
$$

```python
from statistical_analysis import power_anova

# For 3 groups
n_per_group = power_anova(
    effect_size=0.25,  # Cohen's f
    k=3,  # Number of groups
    power=0.80,
    alpha=0.05,
    mode='sample_size'
)
print(f"Required n per group = {n_per_group}")
```

### Chi-Squared Test

**Non-centrality Parameter**:
$$
\lambda = n \cdot w^2
$$

Where w is Cohen's w (effect size for chi-squared).

```python
from statistical_analysis import power_chisq

n = power_chisq(
    effect_size=0.3,  # Cohen's w
    df=2,  # Degrees of freedom
    power=0.80,
    alpha=0.05,
    mode='sample_size'
)
```

### Correlation Test

$$
n = \frac{(z_{1-\alpha/2} + z_{1-\beta})^2}{(\tanh^{-1}(r))^2} + 3
$$

```python
from statistical_analysis import power_correlation

n = power_correlation(
    r=0.3,  # Expected correlation
    power=0.80,
    alpha=0.05,
    mode='sample_size'
)
```

### Regression (R-squared)

**For testing if R-squared > 0**:
$$
f^2 = \frac{R^2}{1 - R^2}
$$

$$
\lambda = f^2 \cdot n
$$

```python
from statistical_analysis import power_regression

n = power_regression(
    f_squared=0.15,  # Cohen's f-squared
    n_predictors=3,
    power=0.80,
    alpha=0.05,
    mode='sample_size'
)
```

### Difference-in-Differences

For DID, power depends on:
- Effect size
- Number of clusters
- Number of time periods
- Within-cluster correlation (ICC)
- Variance decomposition

```python
from statistical_analysis import power_did

n_clusters = power_did(
    effect_size=0.3,
    n_periods=10,
    treatment_periods=5,
    icc=0.1,
    cluster_size=50,
    power=0.80,
    alpha=0.05,
    mode='sample_size'
)
```

---

## 3. Minimum Detectable Effect (MDE)

### Definition

The smallest effect size that can be detected with a given power, sample size, and alpha level.

$$
MDE = (z_{1-\alpha/2} + z_{1-\beta}) \times SE
$$

### Practical Importance

MDE helps answer: "Is my study powered to detect effects that matter?"

Steps:
1. Define what effect size would be policy-relevant
2. Calculate MDE for your design
3. Compare: Is MDE <= meaningful effect?

```python
from statistical_analysis import minimum_detectable_effect

# For a two-sample comparison
mde = minimum_detectable_effect(
    n1=100,
    n2=100,
    alpha=0.05,
    power=0.80,
    test_type='two_sample_ttest'
)

print(f"MDE (Cohen's d) = {mde:.3f}")
print(f"Can detect effects >= {mde:.3f} SD units")
```

### MDE Tables

**Two-Sample t-Test** (alpha=0.05, power=0.80):

| n per group | MDE (d) |
|-------------|---------|
| 20 | 0.91 |
| 50 | 0.57 |
| 100 | 0.40 |
| 200 | 0.28 |
| 500 | 0.18 |
| 1000 | 0.13 |

**Correlation** (alpha=0.05, power=0.80):

| n | MDE (r) |
|---|---------|
| 20 | 0.58 |
| 50 | 0.38 |
| 100 | 0.27 |
| 200 | 0.19 |
| 500 | 0.12 |

---

## 4. Sample Size Determination

### General Framework

```python
from statistical_analysis import required_sample_size

n = required_sample_size(
    effect_size=0.3,
    alpha=0.05,
    power=0.80,
    test_type='two_sample_ttest'
)
```

### Sample Size Tables

**Two-Sample t-Test** (alpha=0.05, two-sided):

| Effect Size (d) | Power=0.80 | Power=0.90 | Power=0.95 |
|-----------------|------------|------------|------------|
| 0.2 (small) | 394 | 526 | 651 |
| 0.5 (medium) | 64 | 86 | 105 |
| 0.8 (large) | 26 | 34 | 42 |

**One-Way ANOVA** (3 groups, alpha=0.05, power=0.80):

| Effect Size (f) | n per group |
|-----------------|-------------|
| 0.10 | 322 |
| 0.25 | 52 |
| 0.40 | 21 |

### Adjustments

**For Unequal Sample Sizes**:
Optimal allocation ratio depends on relative costs and variances.

$$
\frac{n_1}{n_2} = \sqrt{\frac{\sigma_1^2 / c_1}{\sigma_2^2 / c_2}}
$$

**For Clustering** (ICC adjustment):
$$
n_{eff} = \frac{n}{1 + (m-1) \times ICC}
$$

Where m is cluster size.

```python
from statistical_analysis import adjusted_sample_size

# Adjust for clustering
n_adjusted = adjusted_sample_size(
    base_n=100,
    cluster_size=20,
    icc=0.1
)
print(f"Need {n_adjusted} total (vs {100} if independent)")
```

---

## 5. Post-Hoc Power Analysis

### The Controversy

Post-hoc power analysis (using observed effect size) is **not recommended** because:

1. **Tautological**: Observed power is a function of p-value
2. **Misleading**: Low power after non-significant result is uninformative
3. **Misused**: Often used to excuse underpowered studies

### What to Do Instead

For completed studies, report:

1. **Confidence Intervals**: These show range of plausible effects
2. **Sensitivity Analysis**: What effects could have been detected?
3. **Meta-analysis**: Combine with other studies

```python
# AVOID: Post-hoc power with observed effect
# BAD: power_observed = power_ttest(effect_size=observed_d, n1=n1, n2=n2)

# INSTEAD: Report confidence interval
from statistical_analysis import ttest_ci

ci = ttest_ci(
    mean_diff=2.5,
    se=1.2,
    df=98,
    alpha=0.05
)
print(f"Effect: 2.5, 95% CI [{ci[0]:.2f}, {ci[1]:.2f}]")

# INSTEAD: Sensitivity analysis
mde = minimum_detectable_effect(n1=50, n2=50, alpha=0.05, power=0.80)
print(f"This study could detect effects >= {mde:.2f}")
```

---

## 6. Power for Complex Designs

### Multi-Level / Clustered Designs

**Design Effect**:
$$
DE = 1 + (m - 1) \times ICC
$$

**Required Clusters**:
$$
J = \frac{2(z_{1-\alpha/2} + z_{1-\beta})^2 \times DE \times \sigma^2}{m \times \delta^2}
$$

```python
from statistical_analysis import power_clustered

# Cluster RCT
n_clusters = power_clustered(
    effect_size=0.3,
    icc=0.05,
    cluster_size=30,
    power=0.80,
    alpha=0.05,
    mode='clusters'
)
print(f"Need {n_clusters} clusters per arm")
```

### Factorial Designs

Power for main effects and interactions differs. Interactions typically require larger samples.

```python
from statistical_analysis import power_factorial

# 2x2 factorial
n = power_factorial(
    main_effect_size=0.5,
    interaction_effect_size=0.25,
    n_factors=2,
    levels_per_factor=2,
    power=0.80,
    alpha=0.05
)
```

### Longitudinal Designs

Power depends on:
- Number of time points
- Autocorrelation structure
- Attrition
- Time-varying effects

```python
from statistical_analysis import power_longitudinal

n = power_longitudinal(
    effect_size=0.3,
    n_timepoints=5,
    autocorrelation=0.5,
    attrition_rate=0.1,
    power=0.80,
    alpha=0.05
)
```

### Mediation Analysis

Power for mediation depends on:
- Path a (X -> M) effect
- Path b (M -> Y) effect
- Sample size
- Whether paths are assumed or tested

```python
from statistical_analysis import power_mediation

n = power_mediation(
    path_a=0.3,  # X -> M standardized
    path_b=0.4,  # M -> Y standardized
    power=0.80,
    alpha=0.05,
    method='sobel'  # or 'bootstrap'
)
```

---

## 7. Practical Considerations

### Inflation Factors

Account for:
- **Attrition**: Multiply n by 1/(1-attrition_rate)
- **Non-compliance**: Inflate by design effect
- **Clustering**: Apply ICC adjustment
- **Covariates**: Can reduce required n if highly predictive

```python
from statistical_analysis import inflated_sample_size

n_final = inflated_sample_size(
    base_n=200,
    expected_attrition=0.15,
    clustering_effect=1.2,
    covariate_r_squared=0.3  # Reduces variance
)
print(f"Plan for n = {n_final}")
```

### Sequential Analysis and Interim Looks

Multiple looks at data inflate Type I error. Use:
- Alpha spending functions (e.g., O'Brien-Fleming)
- Group sequential designs

```python
from statistical_analysis import sequential_sample_size

n = sequential_sample_size(
    effect_size=0.5,
    n_looks=3,
    alpha=0.05,
    power=0.80,
    spending_function='obrien_fleming'
)
```

### Bayesian Sample Size

Bayesian approaches focus on:
- Precision of posterior
- Probability of detecting meaningful effect
- Expected posterior evidence

```python
from statistical_analysis import bayesian_sample_size

n = bayesian_sample_size(
    prior_mean=0,
    prior_sd=0.5,
    expected_effect=0.3,
    desired_ci_width=0.2,
    credible_level=0.95
)
```

---

## 8. Power Analysis Workflow

### A Priori (Planning)

```
1. Define research question and hypothesis
           |
           v
2. Choose appropriate statistical test
           |
           v
3. Specify minimum effect size of interest
   - Based on theory
   - Based on practical significance
   - Based on literature
           |
           v
4. Set alpha (usually 0.05) and power (usually 0.80)
           |
           v
5. Calculate required sample size
           |
           v
6. Assess feasibility
   - Budget constraints?
   - Time constraints?
   - Access to participants?
           |
           v
7. If infeasible, adjust:
   - Accept lower power
   - Detect only larger effects
   - Use more efficient design
```

### Sensitivity (Completed Study)

```
1. Study completed with fixed n
           |
           v
2. Calculate MDE for your sample
           |
           v
3. Compare MDE to meaningful effects
           |
           v
4. Report what effects could/could not be detected
           |
           v
5. Contextualize results:
   - Significant: Effect >= MDE
   - Non-significant: Cannot distinguish from effects < MDE
```

---

## 9. Common Mistakes

### 1. Using Wrong Effect Size Metric

**Mistake**: Using Cohen's d benchmarks for Cohen's f tests.

**Solution**: Use appropriate effect size for each test.

### 2. Ignoring Clustering

**Mistake**: Calculating n as if observations are independent.

**Solution**: Apply design effect adjustment.

### 3. Overlooking Multiple Comparisons

**Mistake**: Powering for single test when making multiple comparisons.

**Solution**: Adjust alpha for multiple testing in power calculation.

### 4. Unrealistic Effect Sizes

**Mistake**: Using "medium" effects without justification.

**Solution**: Base effect sizes on pilot data, literature, or minimum meaningful effect.

### 5. Post-Hoc Power Abuse

**Mistake**: Computing power with observed effect size after non-significant result.

**Solution**: Report confidence intervals and sensitivity analysis instead.

---

## 10. Software and Tools

### Python
```python
from statistical_analysis import power_analysis
# Or use statsmodels.stats.power
```

### R
```r
library(pwr)
pwr.t.test(d = 0.5, power = 0.80, sig.level = 0.05)
```

### Online Calculators
- G*Power (free desktop app)
- PowerUp! (for education research)
- PASS (commercial)

---

## References

### Foundational
- Cohen, J. (1988). *Statistical Power Analysis for the Behavioral Sciences* (2nd ed.). Lawrence Erlbaum.
- Cohen, J. (1992). A power primer. *Psychological Bulletin*, 112(1), 155-159.

### Methodological
- Hoenig, J. M., & Heisey, D. M. (2001). The abuse of power: The pervasive fallacy of power calculations for data analysis. *The American Statistician*, 55(1), 19-24.
- Maxwell, S. E., Kelley, K., & Rausch, J. R. (2008). Sample size planning for statistical power and accuracy in parameter estimation. *Annual Review of Psychology*, 59, 537-563.

### For Clustered Designs
- Hemming, K., Girling, A. J., Sitch, A. J., Marsh, J., & Lilford, R. J. (2011). Sample size calculations for cluster randomised controlled trials with a fixed number of clusters. *BMC Medical Research Methodology*, 11, 102.

### Economics Applications
- Duflo, E., Glennerster, R., & Kremer, M. (2007). Using randomization in development economics research: A toolkit. *Handbook of Development Economics*, 4, 3895-3962.
