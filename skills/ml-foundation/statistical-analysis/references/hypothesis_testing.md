# Hypothesis Testing Reference

> **Document Type**: Reference | **Last Updated**: 2025-01
> **Related**: [effect_sizes.md](effect_sizes.md), [power_analysis.md](power_analysis.md), [common_errors.md](common_errors.md)

## Overview

Hypothesis testing is a framework for making statistical inferences about population parameters based on sample data. This reference covers the most commonly used tests in economics and social science research.

---

## 1. Student's t-Test

### One-Sample t-Test

**Purpose**: Test whether a population mean equals a hypothesized value.

**Hypotheses**:
- H0: $\mu = \mu_0$
- H1: $\mu \neq \mu_0$ (two-sided) or $\mu > \mu_0$ / $\mu < \mu_0$ (one-sided)

**Test Statistic**:
$$
t = \frac{\bar{X} - \mu_0}{s / \sqrt{n}}
$$

**Degrees of Freedom**: $df = n - 1$

**Assumptions**:
1. Random sampling
2. Normal distribution (or large n for CLT)
3. Independence of observations

**Python Implementation**:
```python
from statistical_analysis import run_ttest_one_sample

result = run_ttest_one_sample(
    data=df['income'],
    hypothesized_mean=50000,
    alternative='two-sided'
)

print(f"t = {result.statistic:.3f}")
print(f"p = {result.p_value:.4f}")
print(f"95% CI: [{result.ci_lower:.2f}, {result.ci_upper:.2f}]")
```

### Independent Samples t-Test

**Purpose**: Compare means of two independent groups.

**Test Statistic (Equal Variances)**:
$$
t = \frac{\bar{X}_1 - \bar{X}_2}{s_p \sqrt{\frac{1}{n_1} + \frac{1}{n_2}}}
$$

Where pooled standard deviation:
$$
s_p = \sqrt{\frac{(n_1-1)s_1^2 + (n_2-1)s_2^2}{n_1 + n_2 - 2}}
$$

**Degrees of Freedom**: $df = n_1 + n_2 - 2$

**Test Statistic (Welch's, Unequal Variances)**:
$$
t = \frac{\bar{X}_1 - \bar{X}_2}{\sqrt{\frac{s_1^2}{n_1} + \frac{s_2^2}{n_2}}}
$$

**Welch-Satterthwaite Degrees of Freedom**:
$$
df = \frac{\left(\frac{s_1^2}{n_1} + \frac{s_2^2}{n_2}\right)^2}{\frac{s_1^4}{n_1^2(n_1-1)} + \frac{s_2^4}{n_2^2(n_2-1)}}
$$

**When to Use Which**:
| Scenario | Test |
|----------|------|
| Equal sample sizes, similar variances | Pooled t-test |
| Unequal sample sizes or variances | Welch's t-test |
| Unknown variance equality | Welch's t-test (default) |

**Python Implementation**:
```python
from statistical_analysis import run_ttest, run_welch_test

# Equal variances assumed
result = run_ttest(
    data=df,
    outcome='income',
    group='treatment',
    equal_var=True
)

# Welch's test (recommended as default)
result = run_welch_test(
    data=df,
    outcome='income',
    group='treatment'
)
```

### Paired Samples t-Test

**Purpose**: Compare means of paired or matched observations.

**Test Statistic**:
$$
t = \frac{\bar{D}}{s_D / \sqrt{n}}
$$

Where $D_i = X_{1i} - X_{2i}$ are the paired differences.

**Use Cases**:
- Before/after measurements on same subjects
- Matched pairs design
- Repeated measures

**Python Implementation**:
```python
from statistical_analysis import run_ttest_paired

result = run_ttest_paired(
    data=df,
    var1='income_before',
    var2='income_after'
)
```

---

## 2. F-Test and ANOVA

### One-Way ANOVA

**Purpose**: Compare means across more than two groups.

**Hypotheses**:
- H0: $\mu_1 = \mu_2 = ... = \mu_k$
- H1: At least one mean differs

**Test Statistic**:
$$
F = \frac{MS_{between}}{MS_{within}} = \frac{SS_{between}/(k-1)}{SS_{within}/(N-k)}
$$

**Components**:
$$
SS_{total} = \sum_{i,j} (X_{ij} - \bar{X})^2
$$
$$
SS_{between} = \sum_j n_j (\bar{X}_j - \bar{X})^2
$$
$$
SS_{within} = \sum_{i,j} (X_{ij} - \bar{X}_j)^2
$$

**Degrees of Freedom**: $df_1 = k-1$, $df_2 = N-k$

**Assumptions**:
1. Independence of observations
2. Normality within groups
3. Homogeneity of variances (homoscedasticity)

**Python Implementation**:
```python
from statistical_analysis import run_anova

result = run_anova(
    data=df,
    outcome='income',
    groups='education_level'
)

print(f"F = {result.statistic:.3f}")
print(f"p = {result.p_value:.4f}")
print(f"Eta-squared = {result.effect_size:.3f}")
```

### Post-Hoc Tests

When ANOVA is significant, use post-hoc tests to identify which groups differ:

| Test | Use Case | Controls |
|------|----------|----------|
| Tukey HSD | All pairwise comparisons | FWER |
| Bonferroni | Few planned comparisons | FWER |
| Scheffe | Any contrasts | FWER |
| Games-Howell | Unequal variances | FWER |
| Benjamini-Hochberg | Many comparisons | FDR |

```python
from statistical_analysis import run_posthoc_tukey, run_posthoc_bonferroni

# Tukey HSD
posthoc = run_posthoc_tukey(
    data=df,
    outcome='income',
    groups='education_level'
)

print(posthoc.pairwise_results)
```

### Two-Way ANOVA

**Purpose**: Test effects of two factors and their interaction.

**Model**:
$$
Y_{ijk} = \mu + \alpha_i + \beta_j + (\alpha\beta)_{ij} + \epsilon_{ijk}
$$

**Python Implementation**:
```python
from statistical_analysis import run_anova_twoway

result = run_anova_twoway(
    data=df,
    outcome='income',
    factor1='gender',
    factor2='education'
)

print(result.main_effects)
print(result.interaction_effect)
```

### Welch's ANOVA

**Purpose**: Compare means when variances are unequal.

```python
from statistical_analysis import run_welch_anova

result = run_welch_anova(
    data=df,
    outcome='income',
    groups='education_level'
)
```

### Kruskal-Wallis Test

**Purpose**: Non-parametric alternative to one-way ANOVA.

**Use When**:
- Data violates normality
- Ordinal outcome variable
- Small sample sizes with non-normal data

```python
from statistical_analysis import run_kruskal

result = run_kruskal(
    data=df,
    outcome='income',
    groups='education_level'
)
```

---

## 3. Chi-Squared Tests

### Test of Independence

**Purpose**: Test association between two categorical variables.

**Test Statistic**:
$$
\chi^2 = \sum_{i,j} \frac{(O_{ij} - E_{ij})^2}{E_{ij}}
$$

Where expected frequency:
$$
E_{ij} = \frac{(\text{row total}_i) \times (\text{column total}_j)}{N}
$$

**Degrees of Freedom**: $df = (r-1)(c-1)$

**Assumptions**:
1. Random sampling
2. Independence of observations
3. Expected frequencies $\geq 5$ in at least 80% of cells

**Python Implementation**:
```python
from statistical_analysis import run_chi_squared

result = run_chi_squared(
    data=df,
    var1='treatment',
    var2='success',
    test_type='independence'
)

print(f"Chi-squared = {result.statistic:.3f}")
print(f"p = {result.p_value:.4f}")
print(f"Cramer's V = {result.effect_size:.3f}")
```

### Goodness of Fit Test

**Purpose**: Test whether observed frequencies match expected distribution.

**Test Statistic**:
$$
\chi^2 = \sum_i \frac{(O_i - E_i)^2}{E_i}
$$

**Degrees of Freedom**: $df = k - 1$ (or $k - 1 - p$ if p parameters estimated)

```python
from statistical_analysis import run_chi_squared_gof

result = run_chi_squared_gof(
    observed=[50, 30, 20],
    expected=[40, 35, 25]  # or probabilities [0.4, 0.35, 0.25]
)
```

### Fisher's Exact Test

**Purpose**: Alternative to chi-squared for small samples (2x2 tables).

**Use When**:
- Any expected frequency < 5
- Total sample size < 20

```python
from statistical_analysis import run_fisher_exact

result = run_fisher_exact(
    data=df,
    var1='treatment',
    var2='success'
)
```

### McNemar's Test

**Purpose**: Compare paired proportions (before/after binary outcomes).

```python
from statistical_analysis import run_mcnemar

result = run_mcnemar(
    data=df,
    var1='success_before',
    var2='success_after'
)
```

---

## 4. Non-Parametric Tests

### Mann-Whitney U Test

**Purpose**: Non-parametric alternative to independent samples t-test.

**Also Known As**: Wilcoxon rank-sum test

**Test Statistic**:
$$
U = n_1 n_2 + \frac{n_1(n_1 + 1)}{2} - R_1
$$

Where $R_1$ is the sum of ranks for group 1.

**Use When**:
- Non-normal distributions
- Ordinal data
- Presence of outliers

```python
from statistical_analysis import run_mann_whitney

result = run_mann_whitney(
    data=df,
    outcome='satisfaction_score',  # Ordinal
    group='treatment'
)
```

### Wilcoxon Signed-Rank Test

**Purpose**: Non-parametric alternative to paired t-test.

```python
from statistical_analysis import run_wilcoxon

result = run_wilcoxon(
    data=df,
    var1='score_before',
    var2='score_after'
)
```

### Spearman Correlation

**Purpose**: Test monotonic relationship between two variables.

```python
from statistical_analysis import run_spearman_correlation

result = run_spearman_correlation(
    data=df,
    var1='education_years',
    var2='income'
)
```

---

## 5. Multiple Testing Corrections

### The Multiple Comparisons Problem

When performing $m$ independent tests at significance level $\alpha$:
- Probability of at least one false positive: $1 - (1-\alpha)^m$
- With 20 tests at $\alpha = 0.05$: $1 - 0.95^{20} = 0.64$

### Family-Wise Error Rate (FWER) Control

**Bonferroni Correction**:
$$
\alpha_{adjusted} = \frac{\alpha}{m}
$$

**Holm-Bonferroni (Step-Down)**:
1. Order p-values: $p_{(1)} \leq p_{(2)} \leq ... \leq p_{(m)}$
2. Compare $p_{(k)}$ to $\frac{\alpha}{m - k + 1}$
3. Reject until first non-rejection

**Hochberg (Step-Up)**:
1. Order p-values in reverse
2. Compare $p_{(k)}$ to $\frac{\alpha \cdot k}{m}$
3. Reject if any comparison passes

```python
from statistical_analysis import bonferroni_correction, holm_correction

p_values = [0.01, 0.04, 0.03, 0.08, 0.002]

# Bonferroni (most conservative)
adjusted_bonf = bonferroni_correction(p_values)

# Holm-Bonferroni (less conservative)
adjusted_holm = holm_correction(p_values)
```

### False Discovery Rate (FDR) Control

**Benjamini-Hochberg Procedure**:
1. Order p-values: $p_{(1)} \leq p_{(2)} \leq ... \leq p_{(m)}$
2. Find largest $k$ such that $p_{(k)} \leq \frac{k}{m} \alpha$
3. Reject all $H_{(i)}$ for $i \leq k$

**Interpretation**: Expected proportion of false discoveries among rejections $\leq \alpha$

```python
from statistical_analysis import benjamini_hochberg

p_values = [0.01, 0.04, 0.03, 0.08, 0.002]

# Control FDR at 5%
adjusted = benjamini_hochberg(p_values, alpha=0.05)
significant = [i for i, p in enumerate(adjusted) if p < 0.05]
```

### When to Use Which

| Method | Controls | Best For |
|--------|----------|----------|
| Bonferroni | FWER | Few tests, confirmatory research |
| Holm-Bonferroni | FWER | Moderate tests, want more power than Bonf |
| Benjamini-Hochberg | FDR | Many tests, exploratory research |
| No correction | Nothing | Single pre-registered hypothesis |

---

## 6. Assumption Testing

### Normality Tests

| Test | Use Case | Null Hypothesis |
|------|----------|-----------------|
| Shapiro-Wilk | Small to moderate n (<50) | Data is normal |
| Kolmogorov-Smirnov | Large n, known parameters | Data follows specified distribution |
| Anderson-Darling | Better for tails | Data is normal |
| D'Agostino-Pearson | Large n | Data is normal (skew + kurtosis) |

```python
from statistical_analysis import check_normality

result = check_normality(data=df['income'], method='shapiro')
print(f"W = {result.statistic:.4f}, p = {result.p_value:.4f}")
print(f"Normal: {result.passed}")
```

**Important**: Normality tests become overly sensitive with large samples. Use visual inspection (Q-Q plots) alongside formal tests.

### Variance Homogeneity Tests

| Test | Assumptions | Use Case |
|------|-------------|----------|
| Levene's | None | Most robust to non-normality |
| Bartlett's | Normality | More powerful when normal |
| Brown-Forsythe | None | Median-based variant of Levene's |

```python
from statistical_analysis import check_homogeneity

result = check_homogeneity(
    data=df,
    outcome='income',
    group='treatment',
    method='levene'
)
```

### Independence Tests

For time series or clustered data:

```python
from statistical_analysis import durbin_watson_test

# Check autocorrelation in residuals
dw = durbin_watson_test(residuals)
print(f"Durbin-Watson = {dw.statistic:.3f}")
# Values near 2 suggest no autocorrelation
# < 2 suggests positive autocorrelation
# > 2 suggests negative autocorrelation
```

---

## Summary Decision Tree

```
Start: Compare groups/test hypotheses
       |
       v
How many groups?
       |
       +-- 1 group vs. value --> One-sample t-test
       |
       +-- 2 groups
       |       |
       |       +-- Paired? --> Paired t-test (or Wilcoxon)
       |       |
       |       +-- Independent --> Two-sample t-test (or Mann-Whitney)
       |
       +-- 3+ groups --> ANOVA (or Kruskal-Wallis)
                |
                +-- Significant? --> Post-hoc tests

For categorical variables:
       |
       +-- 2x2 table, small n --> Fisher's exact
       |
       +-- Larger tables --> Chi-squared test
       |
       +-- Paired binary --> McNemar's test
```

---

## References

### Textbooks
- Rice, J. A. (2007). *Mathematical Statistics and Data Analysis* (3rd ed.). Cengage Learning.
- Wasserman, L. (2004). *All of Statistics*. Springer.
- Agresti, A. (2013). *Categorical Data Analysis* (3rd ed.). Wiley.

### Key Papers
- Welch, B. L. (1947). The generalization of Student's problem when several different population variances are involved. *Biometrika*, 34, 28-35.
- Benjamini, Y., & Hochberg, Y. (1995). Controlling the false discovery rate. *JRSS-B*, 57(1), 289-300.
- Student. (1908). The probable error of a mean. *Biometrika*, 6(1), 1-25.
