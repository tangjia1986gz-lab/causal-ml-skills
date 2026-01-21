# Effect Sizes Reference

> **Document Type**: Reference | **Last Updated**: 2025-01
> **Related**: [hypothesis_testing.md](hypothesis_testing.md), [power_analysis.md](power_analysis.md), [common_errors.md](common_errors.md)

## Overview

Effect sizes quantify the magnitude of a phenomenon independent of sample size. While p-values indicate whether an effect exists, effect sizes tell us how large that effect is in practical terms. This is crucial for policy relevance in economics and social sciences.

---

## 1. Standardized Mean Difference (d-family)

### Cohen's d

**Definition**: Difference between means in pooled standard deviation units.

$$
d = \frac{\bar{X}_1 - \bar{X}_2}{s_p}
$$

Where pooled standard deviation:
$$
s_p = \sqrt{\frac{(n_1-1)s_1^2 + (n_2-1)s_2^2}{n_1 + n_2 - 2}}
$$

**Confidence Interval**:
$$
SE_d = \sqrt{\frac{n_1 + n_2}{n_1 n_2} + \frac{d^2}{2(n_1 + n_2)}}
$$

$$
CI_{95\%} = d \pm 1.96 \times SE_d
$$

**Python Implementation**:
```python
from statistical_analysis import calculate_cohens_d

result = calculate_cohens_d(
    data=df,
    outcome='income',
    group='treatment'
)

print(f"Cohen's d = {result.effect_size:.3f}")
print(f"95% CI: [{result.ci_lower:.3f}, {result.ci_upper:.3f}]")
```

### Hedges' g (Bias-Corrected)

**Purpose**: Corrects upward bias in Cohen's d for small samples.

$$
g = d \times J
$$

Where the correction factor:
$$
J = 1 - \frac{3}{4(n_1 + n_2) - 9}
$$

**When to Use**:
- Sample sizes < 20 per group
- Meta-analysis (standard practice)
- Small sample research

```python
from statistical_analysis import calculate_hedges_g

result = calculate_hedges_g(
    data=df,
    outcome='income',
    group='treatment'
)
```

### Glass's Delta

**Definition**: Uses only the control group's standard deviation.

$$
\Delta = \frac{\bar{X}_1 - \bar{X}_2}{s_{control}}
$$

**When to Use**:
- Treatment changes variability
- One group is a "natural" baseline
- Comparing treatment to untreated norm

```python
from statistical_analysis import calculate_glass_delta

result = calculate_glass_delta(
    data=df,
    outcome='income',
    group='treatment',
    control_value=0  # Value indicating control group
)
```

### Interpreting d-family Effect Sizes

**Cohen's Benchmarks** (use cautiously):

| Size | Cohen's d | Interpretation |
|------|-----------|----------------|
| Small | 0.2 | Effect barely perceptible |
| Medium | 0.5 | Effect noticeable without statistics |
| Large | 0.8 | Substantial practical difference |

**Context-Specific Interpretation**:

| Field | "Small" | "Medium" | "Large" |
|-------|---------|----------|---------|
| Psychology | 0.2 | 0.5 | 0.8 |
| Education | 0.2 | 0.4 | 0.6 |
| Medicine | 0.1 | 0.3 | 0.5 |
| Economics | Varies by context | | |

**Important**: Always interpret effect sizes in your research context. A d = 0.3 might be:
- Small for a psychological intervention
- Large for a policy affecting millions
- Meaningful or trivial depending on costs

---

## 2. Correlation-Based Effect Sizes (r-family)

### Pearson's r

**Definition**: Standardized covariance between two continuous variables.

$$
r = \frac{\sum(X_i - \bar{X})(Y_i - \bar{Y})}{\sqrt{\sum(X_i - \bar{X})^2 \sum(Y_i - \bar{Y})^2}}
$$

**Range**: -1 to +1

**Confidence Interval** (Fisher's z transformation):
$$
z_r = \frac{1}{2} \ln\left(\frac{1+r}{1-r}\right)
$$
$$
SE_{z_r} = \frac{1}{\sqrt{n-3}}
$$

```python
from statistical_analysis import calculate_correlation

result = calculate_correlation(
    data=df,
    var1='education',
    var2='income',
    method='pearson'
)
```

### Point-Biserial Correlation

**Definition**: Correlation between continuous and binary variable.

**Equivalent to**: Pearson r when one variable is dichotomous

**Relationship to d**:
$$
r_{pb} = \frac{d}{\sqrt{d^2 + \frac{n_1 + n_2}{n_1 n_2}(n_1 + n_2 - 2)}}
$$

```python
from statistical_analysis import calculate_point_biserial

result = calculate_point_biserial(
    data=df,
    continuous='income',
    binary='treatment'
)
```

### Interpreting r

| Size | r | r-squared | Variance Explained |
|------|---|-----------|-------------------|
| Small | 0.1 | 0.01 | 1% |
| Medium | 0.3 | 0.09 | 9% |
| Large | 0.5 | 0.25 | 25% |

---

## 3. Variance-Explained Effect Sizes

### Eta-Squared (ANOVA)

**Definition**: Proportion of total variance explained by group membership.

$$
\eta^2 = \frac{SS_{between}}{SS_{total}}
$$

**Interpretation**: % of outcome variance attributable to group differences

**Limitation**: Positively biased; tends to overestimate population effect

```python
from statistical_analysis import calculate_eta_squared

result = calculate_eta_squared(anova_result)
print(f"Eta-squared = {result.effect_size:.3f}")
```

### Partial Eta-Squared

**Definition**: Effect size for one factor controlling for others.

$$
\eta^2_p = \frac{SS_{effect}}{SS_{effect} + SS_{error}}
$$

**Use in**: Factorial ANOVA, controlling for covariates

```python
from statistical_analysis import calculate_partial_eta_squared

result = calculate_partial_eta_squared(
    ss_effect=150,
    ss_error=450
)
```

### Omega-Squared (Less Biased)

**Definition**: Population estimate, less biased than eta-squared.

$$
\omega^2 = \frac{SS_{between} - (k-1)MS_{within}}{SS_{total} + MS_{within}}
$$

**Recommended over eta-squared for accurate population estimates**

```python
from statistical_analysis import calculate_omega_squared

result = calculate_omega_squared(
    ss_between=200,
    ss_within=800,
    ss_total=1000,
    k=3,  # Number of groups
    n=150  # Total sample size
)
```

### Cohen's f (ANOVA Effect Size)

**Definition**: Standardized effect for F-tests.

$$
f = \sqrt{\frac{\eta^2}{1 - \eta^2}}
$$

**Benchmarks**:
| Size | Cohen's f | Eta-squared |
|------|-----------|-------------|
| Small | 0.10 | 0.01 |
| Medium | 0.25 | 0.06 |
| Large | 0.40 | 0.14 |

```python
from statistical_analysis import calculate_cohens_f

result = calculate_cohens_f(eta_squared=0.06)
```

---

## 4. Effect Sizes for Categorical Data

### Odds Ratio (OR)

**Definition**: Ratio of odds of outcome in exposed vs. unexposed group.

$$
OR = \frac{a/c}{b/d} = \frac{ad}{bc}
$$

For 2x2 table:
|  | Outcome+ | Outcome- |
|--|----------|----------|
| Treatment | a | b |
| Control | c | d |

**Confidence Interval**:
$$
SE_{\ln(OR)} = \sqrt{\frac{1}{a} + \frac{1}{b} + \frac{1}{c} + \frac{1}{d}}
$$

$$
CI_{95\%} = \exp\left(\ln(OR) \pm 1.96 \times SE_{\ln(OR)}\right)
$$

**Interpretation**:
- OR = 1: No association
- OR > 1: Higher odds in treatment group
- OR < 1: Lower odds in treatment group

```python
from statistical_analysis import calculate_odds_ratio

result = calculate_odds_ratio(
    data=df,
    outcome='success',
    exposure='treatment'
)

print(f"OR = {result.effect_size:.2f}")
print(f"95% CI: [{result.ci_lower:.2f}, {result.ci_upper:.2f}]")
```

### Relative Risk (Risk Ratio, RR)

**Definition**: Ratio of probabilities of outcome.

$$
RR = \frac{a/(a+b)}{c/(c+d)}
$$

**Interpretation**: How many times more (or less) likely the outcome is in treatment vs. control.

**When to Use OR vs. RR**:
| Scenario | Preferred |
|----------|-----------|
| Rare outcomes (<10%) | OR and RR similar |
| Common outcomes | RR more intuitive |
| Case-control studies | OR only valid |
| Cohort studies | Either (RR preferred) |
| Logistic regression | OR (model output) |

```python
from statistical_analysis import calculate_relative_risk

result = calculate_relative_risk(
    data=df,
    outcome='success',
    exposure='treatment'
)
```

### Risk Difference (Attributable Risk)

**Definition**: Absolute difference in probabilities.

$$
RD = \frac{a}{a+b} - \frac{c}{c+d}
$$

**Interpretation**: Percentage point change in probability

**Number Needed to Treat (NNT)**:
$$
NNT = \frac{1}{|RD|}
$$

```python
from statistical_analysis import calculate_risk_difference, calculate_nnt

rd = calculate_risk_difference(
    data=df,
    outcome='success',
    exposure='treatment'
)

nnt = calculate_nnt(risk_difference=rd.effect_size)
print(f"NNT = {nnt:.1f}")
```

### Phi Coefficient (2x2 Tables)

**Definition**: Correlation coefficient for two binary variables.

$$
\phi = \frac{ad - bc}{\sqrt{(a+b)(c+d)(a+c)(b+d)}}
$$

**Range**: -1 to +1 (like Pearson r)

```python
from statistical_analysis import calculate_phi

result = calculate_phi(
    data=df,
    var1='treatment',
    var2='success'
)
```

### Cramer's V (Larger Tables)

**Definition**: Association measure for any contingency table.

$$
V = \sqrt{\frac{\chi^2}{n \times \min(r-1, c-1)}}
$$

**Range**: 0 to 1

**Interpretation**:
| Size | Cramer's V |
|------|------------|
| Small | 0.1 |
| Medium | 0.3 |
| Large | 0.5 |

```python
from statistical_analysis import calculate_cramers_v

result = calculate_cramers_v(
    data=df,
    var1='education_level',
    var2='income_bracket'
)
```

---

## 5. Regression Effect Sizes

### Standardized Regression Coefficient (Beta)

**Definition**: Regression coefficient when variables are standardized.

$$
\beta = b \times \frac{s_X}{s_Y}
$$

**Interpretation**: 1 SD increase in X associated with beta SD change in Y

```python
from statistical_analysis import standardize_coefficient

beta = standardize_coefficient(
    unstandardized_coef=2.5,
    sd_x=10.0,
    sd_y=25.0
)
print(f"Standardized coefficient: {beta:.3f}")
```

### R-Squared

**Definition**: Proportion of variance explained by the model.

$$
R^2 = 1 - \frac{SS_{residual}}{SS_{total}}
$$

**Adjusted R-Squared** (penalizes for predictors):
$$
R^2_{adj} = 1 - \frac{(1-R^2)(n-1)}{n-k-1}
$$

### Cohen's f-squared (Regression)

**Definition**: Effect size for additional variance explained.

$$
f^2 = \frac{R^2_{AB} - R^2_A}{1 - R^2_{AB}}
$$

Where:
- $R^2_{AB}$: R-squared with all predictors
- $R^2_A$: R-squared without predictor(s) of interest

**Benchmarks**:
| Size | f-squared |
|------|-----------|
| Small | 0.02 |
| Medium | 0.15 |
| Large | 0.35 |

```python
from statistical_analysis import calculate_cohens_f_squared

result = calculate_cohens_f_squared(
    r_squared_full=0.45,
    r_squared_reduced=0.38
)
```

---

## 6. Converting Between Effect Sizes

### Common Conversions

**d to r**:
$$
r = \frac{d}{\sqrt{d^2 + 4}}
$$

**r to d**:
$$
d = \frac{2r}{\sqrt{1-r^2}}
$$

**d to OR** (approximately):
$$
OR \approx \exp\left(\frac{\pi d}{\sqrt{3}}\right) \approx \exp(1.81 \times d)
$$

**Cohen's d to Cohen's f**:
$$
f = \frac{d}{2}
$$

**Eta-squared to Cohen's f**:
$$
f = \sqrt{\frac{\eta^2}{1 - \eta^2}}
$$

```python
from statistical_analysis import convert_effect_size

# d to r
r = convert_effect_size(d=0.5, from_type='d', to_type='r')

# r to d
d = convert_effect_size(r=0.3, from_type='r', to_type='d')

# d to OR
or_value = convert_effect_size(d=0.5, from_type='d', to_type='or')
```

---

## 7. Practical Significance Guidelines

### Domain-Specific Benchmarks

**Economics/Policy**:
| Context | Small | Medium | Large |
|---------|-------|--------|-------|
| Earnings impact | 2-5% | 5-10% | >10% |
| Employment effect | 1-3 pp | 3-5 pp | >5 pp |
| Test scores (SD) | 0.1 | 0.2-0.3 | >0.5 |

**Education**:
| Intervention Type | Typical Effect (d) |
|-------------------|-------------------|
| Most interventions | 0.1-0.3 |
| Effective programs | 0.3-0.5 |
| Exceptional | >0.5 |

**Medicine**:
| Outcome | Meaningful Change |
|---------|-------------------|
| Blood pressure | 5-10 mmHg |
| Pain (0-10 scale) | 1.5-2 points |
| QOL instruments | 0.5 SD |

### Cost-Effectiveness Considerations

Effect size alone is insufficient; consider:

1. **Cost per unit effect**: Is the intervention cost-effective?
2. **Population reach**: Small effects x large population = big impact
3. **Alternative uses**: Opportunity cost of resources
4. **Heterogeneity**: Who benefits most?

```python
from statistical_analysis import cost_effectiveness_analysis

cea = cost_effectiveness_analysis(
    effect_size=0.3,
    per_person_cost=100,
    population_size=10000,
    outcome_value_per_sd=500  # Dollar value of 1 SD improvement
)

print(f"Cost per SD improvement: ${cea['cost_per_sd']:.2f}")
print(f"Total benefit: ${cea['total_benefit']:,.0f}")
print(f"Net benefit: ${cea['net_benefit']:,.0f}")
```

---

## 8. Reporting Effect Sizes

### APA Style Guidelines

**For d-family**:
> The treatment group (M = 45.2, SD = 12.3) scored significantly higher than the control group (M = 38.7, SD = 11.8), t(348) = 3.45, p = .001, d = 0.48, 95% CI [0.22, 0.74].

**For r**:
> Education and income were positively correlated, r(248) = .42, p < .001, 95% CI [.31, .52].

**For OR**:
> Treatment significantly increased odds of success, OR = 2.15, 95% CI [1.45, 3.19], p = .002.

### Complete Reporting Checklist

- [ ] Point estimate of effect size
- [ ] 95% confidence interval
- [ ] Sample sizes per group
- [ ] Type of effect size and why chosen
- [ ] Context for interpreting magnitude
- [ ] Any corrections applied (e.g., Hedges' g)

---

## References

### Foundational
- Cohen, J. (1988). *Statistical Power Analysis for the Behavioral Sciences* (2nd ed.). Lawrence Erlbaum.
- Rosenthal, R. (1991). *Meta-Analytic Procedures for Social Research*. Sage.

### Methodological
- Lakens, D. (2013). Calculating and reporting effect sizes. *Frontiers in Psychology*, 4, 863.
- Cumming, G. (2012). *Understanding the New Statistics*. Routledge.
- McGraw, K. O., & Wong, S. P. (1992). A common language effect size statistic. *Psychological Bulletin*, 111(2), 361-365.

### Economics-Specific
- Bloom, H. S., Hill, C. J., Black, A. R., & Lipsey, M. W. (2008). Performance trajectories and performance gaps as achievement effect-size benchmarks for educational interventions. *Journal of Research on Educational Effectiveness*, 1(4), 289-328.
