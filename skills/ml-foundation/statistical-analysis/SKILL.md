---
name: statistical-analysis
description: Use for statistical hypothesis testing, effect sizes, and power analysis in econometric research. Triggers on t-test, F-test, chi-squared, effect size, Cohen's d, power analysis.
---

# Statistical Analysis Skill

> **Version**: 1.0.0 | **Type**: Foundation
> **Aliases**: Statistical Inference, Hypothesis Testing, Effect Size Analysis

## Overview

Statistical inference forms the foundation of empirical research in economics and social sciences. This skill provides comprehensive tools for hypothesis testing, effect size calculation, and power analysis, with special attention to the nuances of causal inference applications.

**Key Focus Areas**:
- Classical hypothesis tests (t-test, F-test, chi-squared)
- Effect size measures (Cohen's d, odds ratios, standardized coefficients)
- Power analysis and sample size determination
- Multiple testing corrections
- Practical significance vs. statistical significance

## When to Use

### Ideal Scenarios
- Comparing group means before/after treatment
- Testing regression coefficient significance
- Calculating effect sizes for meta-analysis or policy relevance
- Planning sample sizes for experiments or quasi-experiments
- Adjusting for multiple hypothesis tests
- Interpreting regression results in practical terms

### Data Requirements
- [ ] Continuous or categorical outcome variable
- [ ] Clear comparison groups or regression model
- [ ] Sufficient sample size for chosen test (see power analysis)
- [ ] Understanding of variable distributions

### When NOT to Use
- For causal effect estimation -> Use `estimator-did`, `estimator-psm`, etc.
- For machine learning prediction -> Use `ml-model-*` skills
- For exploratory analysis -> Use `econometric-eda`

---

## Workflow

```
+-------------------------------------------------------------+
|              STATISTICAL ANALYSIS WORKFLOW                   |
+-------------------------------------------------------------+
|  1. SETUP          -> Define hypothesis, choose test         |
|  2. ASSUMPTIONS    -> Check distributional requirements      |
|  3. TESTING        -> Run appropriate statistical test       |
|  4. EFFECT SIZE    -> Calculate practical significance       |
|  5. POWER          -> Assess or plan for adequate power      |
|  6. REPORTING      -> Generate publication-ready output      |
+-------------------------------------------------------------+
```

### Phase 1: Setup

**Define Research Question**:
```python
from statistical_analysis import StatisticalAnalysis

# Initialize analysis
analysis = StatisticalAnalysis(df)

# Define comparison
analysis.define_comparison(
    outcome='income',
    group_var='treatment',
    controls=['age', 'education']
)
```

**Hypothesis Specification**:
- H0 (Null): No difference between groups / coefficient = 0
- H1 (Alternative): One-sided or two-sided

### Phase 2: Assumption Checking

```python
from statistical_analysis import check_assumptions

# Check normality
normality = check_assumptions.normality_test(df['outcome'])

# Check homogeneity of variance
homogeneity = check_assumptions.levene_test(df, 'outcome', 'group')

# Check independence (for time series)
independence = check_assumptions.durbin_watson(residuals)
```

### Phase 3: Hypothesis Testing

**Two-Sample Tests**:
```python
from statistical_analysis import (
    run_ttest,
    run_welch_test,
    run_mann_whitney,
    run_ftest,
    run_chi_squared
)

# Independent samples t-test
result = run_ttest(
    data=df,
    outcome='income',
    group='treatment',
    alternative='two-sided',
    equal_var=True
)

# Welch's t-test (unequal variances)
result = run_welch_test(
    data=df,
    outcome='income',
    group='treatment'
)

# Mann-Whitney U (non-parametric)
result = run_mann_whitney(
    data=df,
    outcome='income',
    group='treatment'
)
```

**ANOVA and Extensions**:
```python
from statistical_analysis import run_anova, run_kruskal

# One-way ANOVA
result = run_anova(
    data=df,
    outcome='income',
    groups='education_level'
)

# Kruskal-Wallis (non-parametric)
result = run_kruskal(
    data=df,
    outcome='income',
    groups='education_level'
)
```

**Chi-Squared Tests**:
```python
# Test of independence
result = run_chi_squared(
    data=df,
    var1='treatment',
    var2='success',
    test_type='independence'
)

# Goodness of fit
result = run_chi_squared(
    observed=observed_counts,
    expected=expected_counts,
    test_type='goodness_of_fit'
)
```

### Phase 4: Effect Size Calculation

```python
from statistical_analysis import (
    calculate_cohens_d,
    calculate_hedges_g,
    calculate_glass_delta,
    calculate_odds_ratio,
    calculate_relative_risk,
    calculate_eta_squared,
    calculate_cohens_f
)

# Cohen's d for two groups
d = calculate_cohens_d(
    data=df,
    outcome='income',
    group='treatment'
)

# Hedges' g (bias-corrected)
g = calculate_hedges_g(
    data=df,
    outcome='income',
    group='treatment'
)

# Odds ratio for binary outcomes
or_result = calculate_odds_ratio(
    data=df,
    outcome='success',
    exposure='treatment'
)

# Eta-squared for ANOVA
eta_sq = calculate_eta_squared(anova_result)

# Standardized regression coefficient
from statistical_analysis import standardize_coefficient
beta = standardize_coefficient(
    coef=2.5,
    se_x=10.0,
    se_y=25.0
)
```

### Phase 5: Power Analysis

```python
from statistical_analysis import (
    power_analysis,
    minimum_detectable_effect,
    required_sample_size
)

# Calculate power for existing study
power = power_analysis(
    effect_size=0.3,
    n1=100,
    n2=100,
    alpha=0.05,
    test_type='two_sample_ttest'
)

# Find minimum detectable effect
mde = minimum_detectable_effect(
    n1=100,
    n2=100,
    alpha=0.05,
    power=0.80
)

# Calculate required sample size
n = required_sample_size(
    effect_size=0.3,
    alpha=0.05,
    power=0.80,
    test_type='two_sample_ttest'
)
```

### Phase 6: Multiple Testing Corrections

```python
from statistical_analysis import (
    bonferroni_correction,
    holm_correction,
    benjamini_hochberg,
    adjust_pvalues
)

# Adjust p-values
p_values = [0.01, 0.04, 0.03, 0.08, 0.002]

# Bonferroni (most conservative)
adjusted = bonferroni_correction(p_values)

# Holm-Bonferroni (step-down)
adjusted = holm_correction(p_values)

# Benjamini-Hochberg (FDR control)
adjusted = benjamini_hochberg(p_values, alpha=0.05)
```

---

## Complete Analysis Workflow

```python
from statistical_analysis import run_full_statistical_analysis

# Run complete analysis
result = run_full_statistical_analysis(
    data=df,
    outcome='income',
    group='treatment',
    controls=['age', 'education'],
    alpha=0.05,
    power_target=0.80,
    effect_size_type='cohens_d'
)

print(result.summary_table)
print(result.diagnostics)
print(result.interpretation)
```

**Returns**:
```python
StatisticalOutput(
    test_statistic=3.45,
    p_value=0.0012,
    effect_size=0.48,
    effect_size_ci=(0.22, 0.74),
    power=0.92,
    mde=0.28,
    interpretation="...",
    diagnostics={
        'normality': NormalityResult(...),
        'homogeneity': HomogeneityResult(...),
        'n_per_group': {'treatment': 150, 'control': 200}
    },
    summary_table="..."
)
```

---

## Effect Size Interpretation

### Cohen's Conventions (Use with Caution)

| Effect Size | Cohen's d | r | Eta-squared | Interpretation |
|-------------|-----------|---|-------------|----------------|
| Small | 0.2 | 0.1 | 0.01 | Subtle effect, may need large N |
| Medium | 0.5 | 0.3 | 0.06 | Moderate practical significance |
| Large | 0.8 | 0.5 | 0.14 | Substantial practical impact |

**Caution**: These benchmarks are arbitrary. Always interpret effect sizes in the **context of your research domain**.

### Context-Specific Guidelines

For **economics/policy research**:
- Compare to other interventions in the literature
- Express in meaningful units (dollars, percentage points)
- Consider policy-relevant thresholds

For **clinical research**:
- Use Number Needed to Treat (NNT) for binary outcomes
- Reference clinically meaningful differences (MCID)

---

## Common Mistakes

### 1. Confusing Statistical and Practical Significance

**Mistake**: Reporting only p-values without effect sizes.

**Why it's wrong**: With large samples, even trivial effects can be statistically significant. With small samples, important effects may be missed.

**Correct approach**:
```python
# WRONG: Only report p-value
print(f"p = {p_value:.4f}")

# CORRECT: Report effect size with confidence interval
print(f"Cohen's d = {d:.2f}, 95% CI [{d_ci[0]:.2f}, {d_ci[1]:.2f}]")
print(f"This represents a {interpret_cohens_d(d)} effect")
print(f"p = {p_value:.4f}")
```

### 2. Ignoring Multiple Comparisons

**Mistake**: Testing many hypotheses and reporting significant results without adjustment.

**Why it's wrong**: With 20 independent tests at alpha=0.05, expected false positives = 1.

**Correct approach**:
```python
# WRONG: Report unadjusted p-values
for var in outcome_vars:
    result = run_ttest(data, var, 'treatment')
    if result.p_value < 0.05:
        print(f"{var}: p = {result.p_value:.4f}")  # Inflated false positive rate

# CORRECT: Adjust for multiple testing
p_values = [run_ttest(data, var, 'treatment').p_value for var in outcome_vars]
adjusted = benjamini_hochberg(p_values)

for var, p_adj in zip(outcome_vars, adjusted):
    if p_adj < 0.05:
        print(f"{var}: p_adj = {p_adj:.4f}")  # Controls FDR
```

### 3. Underpowered Studies

**Mistake**: Running a study with insufficient power to detect meaningful effects.

**Why it's wrong**: Low power means (1) high probability of missing true effects, (2) any significant findings may be inflated (winner's curse).

**Correct approach**:
```python
# BEFORE collecting data, calculate required sample size
n_required = required_sample_size(
    effect_size=0.3,  # Minimum effect of interest
    alpha=0.05,
    power=0.80
)

print(f"Need n = {n_required} per group")

# If you have fixed sample size, report what you CAN detect
mde = minimum_detectable_effect(n1=50, n2=50, alpha=0.05, power=0.80)
print(f"With current sample, can detect effects >= {mde:.2f}")
```

### 4. Misusing One-Sided Tests

**Mistake**: Using one-sided tests to achieve significance when two-sided test fails.

**Why it's wrong**: One-sided tests should be pre-specified based on theory, not chosen post-hoc.

**Correct approach**:
```python
# WRONG: Switching to one-sided after seeing direction
result_two_sided = run_ttest(data, 'y', 'treatment', alternative='two-sided')
# p = 0.08, not significant
result_one_sided = run_ttest(data, 'y', 'treatment', alternative='greater')
# p = 0.04, "significant"

# CORRECT: Pre-register hypothesis direction based on theory
# If theory strongly predicts direction, specify BEFORE analysis
result = run_ttest(
    data, 'y', 'treatment',
    alternative='greater'  # Pre-specified based on theory
)
```

### 5. Treating p = 0.049 and p = 0.051 Differently

**Mistake**: Making binary significant/not-significant decisions at arbitrary thresholds.

**Why it's wrong**: These p-values represent nearly identical evidence.

**Correct approach**:
```python
# WRONG: Binary interpretation
if p_value < 0.05:
    print("Significant!")
else:
    print("Not significant")

# CORRECT: Report exact p-value, confidence interval, and effect size
print(f"Effect: {effect:.3f}")
print(f"95% CI: [{ci_low:.3f}, {ci_high:.3f}]")
print(f"p = {p_value:.3f}")
print(f"Effect size (d): {cohens_d:.2f}")
```

---

## CLI Usage

```bash
# Run t-test
python scripts/run_statistical_tests.py \
    --data data.csv \
    --outcome income \
    --group treatment \
    --test ttest \
    --alternative two-sided

# Calculate effect sizes
python scripts/calculate_effect_sizes.py \
    --data data.csv \
    --outcome income \
    --group treatment \
    --type cohens_d,hedges_g

# Power analysis
python scripts/power_analysis.py \
    --effect-size 0.3 \
    --alpha 0.05 \
    --power 0.80 \
    --mode sample_size

# Visualize distributions
python scripts/visualize_distributions.py \
    --data data.csv \
    --vars income,age \
    --output figures/
```

---

## Output Formats

### Standard Output Table

```
+----------------------------------------------------------+
|           Table X: Statistical Comparison                 |
+----------------------------------------------------------+
|                         Treatment  Control   Difference   |
+----------------------------------------------------------+
| Mean                       45.2      38.7        6.5      |
| Std. Dev.                  12.3      11.8        -        |
| N                          150       200         -        |
|                                                           |
| Test Statistic (t)                              3.45      |
| P-value (two-sided)                            0.001      |
|                                                           |
| Effect Size (Cohen's d)                         0.48      |
| 95% CI                                    [0.22, 0.74]    |
|                                                           |
| Power (achieved)                               0.92       |
+----------------------------------------------------------+
| Notes: Welch's t-test used due to unequal variances.      |
| Effect size indicates medium practical significance.      |
+----------------------------------------------------------+
```

---

## References

### Seminal Papers
- Cohen, J. (1988). *Statistical Power Analysis for the Behavioral Sciences* (2nd ed.). Lawrence Erlbaum.
- Wasserstein, R. L., & Lazar, N. A. (2016). The ASA Statement on p-Values. *The American Statistician*, 70(2), 129-133.
- Benjamini, Y., & Hochberg, Y. (1995). Controlling the False Discovery Rate. *JRSS-B*, 57(1), 289-300.

### Practical Guides
- Lakens, D. (2013). Calculating and reporting effect sizes. *Frontiers in Psychology*, 4, 863.
- Cumming, G. (2014). The New Statistics: Why and How. *Psychological Science*, 25(1), 7-29.
- Simmons, J. P., Nelson, L. D., & Simonsohn, U. (2011). False-Positive Psychology. *Psychological Science*, 22(11), 1359-1366.

### Econometrics References
- Angrist, J. D., & Pischke, J. S. (2009). *Mostly Harmless Econometrics*. Princeton University Press.
- Imbens, G. W., & Rubin, D. B. (2015). *Causal Inference for Statistics, Social, and Biomedical Sciences*. Cambridge University Press.

---

## Related Skills

| Skill | When to Use Instead |
|-------|---------------------|
| `econometric-eda` | For exploratory data analysis before hypothesis testing |
| `ml-preprocessing` | For data preparation and cleaning |
| `estimator-*` | For causal effect estimation (DID, IV, RD, PSM) |
| `causal-ddml` | For machine learning-based causal inference |

---

## Appendix: Mathematical Details

### T-Test Statistic

For two independent samples:
$$
t = \frac{\bar{X}_1 - \bar{X}_2}{\sqrt{\frac{s_1^2}{n_1} + \frac{s_2^2}{n_2}}}
$$

Degrees of freedom (Welch-Satterthwaite):
$$
df = \frac{\left(\frac{s_1^2}{n_1} + \frac{s_2^2}{n_2}\right)^2}{\frac{s_1^4}{n_1^2(n_1-1)} + \frac{s_2^4}{n_2^2(n_2-1)}}
$$

### Cohen's d

$$
d = \frac{\bar{X}_1 - \bar{X}_2}{s_p}
$$

Where pooled standard deviation:
$$
s_p = \sqrt{\frac{(n_1-1)s_1^2 + (n_2-1)s_2^2}{n_1 + n_2 - 2}}
$$

### Power Function

For two-sample t-test:
$$
\text{Power} = 1 - \beta = P\left(T > t_{\alpha/2} - \frac{\delta}{\sigma/\sqrt{n}} \mid H_1\right)
$$

Where $\delta$ is the true effect and $\sigma$ is the population standard deviation.

### Benjamini-Hochberg Procedure

1. Order p-values: $p_{(1)} \leq p_{(2)} \leq ... \leq p_{(m)}$
2. Find largest $k$ such that $p_{(k)} \leq \frac{k}{m} \alpha$
3. Reject all $H_{(i)}$ for $i \leq k$

This controls the False Discovery Rate (FDR) at level $\alpha$.
