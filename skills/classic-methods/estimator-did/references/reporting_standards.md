# Reporting Standards for Difference-in-Differences

> **Document Type**: Reference | **Last Updated**: 2025-01
> **Related**: [did_table.tex](../assets/latex/did_table.tex), [did_report.md](../assets/markdown/did_report.md)

## Overview

This document provides standards for reporting DID results in academic papers, following best practices from top economics journals (AER, QJE, REStud, Econometrica).

---

## 1. Main Results Table

### Standard Format

```
Table X: Difference-in-Differences Estimates
===============================================================================
                              (1)          (2)          (3)          (4)
Dependent Variable:            Y            Y            Y            Y
-------------------------------------------------------------------------------
Treatment Effect            2.345***     2.287***     2.156***     2.089***
                           (0.456)      (0.423)      (0.512)      (0.498)

Controls                      No          Yes          Yes          Yes
Unit Fixed Effects           Yes          Yes          Yes          Yes
Time Fixed Effects           Yes          Yes          Yes          Yes
Unit-Specific Trends          No           No          Yes           No
Additional Controls           No           No           No          Yes

Observations               10,000       10,000        9,856       10,000
R-squared                   0.234        0.312        0.345        0.328
Number of Units               500          500          492          500
Pre-trend p-value           0.672        0.672        0.701        0.672
-------------------------------------------------------------------------------
Notes: Robust standard errors clustered at the unit level in parentheses.
*** p<0.01, ** p<0.05, * p<0.1. Treatment variable is an indicator equal to 1
for treated units in post-treatment periods. Sample period: 2010-2020.
Treatment began in 2015.
===============================================================================
```

### Required Elements

| Element | Description | Notes |
|---------|-------------|-------|
| Dependent variable | Clearly labeled | Use descriptive names |
| Treatment coefficient | Main parameter of interest | Bold or highlighted |
| Standard errors | In parentheses below coefficient | Specify clustering |
| Fixed effects | Which FE are included | Unit, time, unit-specific trends |
| Controls | What covariates are included | Reference appendix for full list |
| N observations | Sample size | After any sample restrictions |
| R-squared | Model fit | Within R-squared for FE models |
| Pre-trend test | P-value for parallel trends test | Essential diagnostic |

### Column Progression

Typical progression from simple to complex specification:

1. **Column 1**: No controls (baseline)
2. **Column 2**: With controls
3. **Column 3**: Additional fixed effects or trends
4. **Column 4**: Alternative specification or robustness

### Python Code for Table Generation

```python
from did_estimator import estimate_did_panel
from table_formatter import create_regression_table

# Run specifications
results = []

# Specification 1: Basic
r1 = estimate_did_panel(df, "y", "treated", "id", "year", cluster="id")
results.append({
    'treatment_effect': r1.effect,
    'treatment_se': r1.se,
    'treatment_pval': r1.p_value,
    'controls': False,
    'fixed_effects': 'Unit + Time',
    'n_obs': r1.diagnostics['n_obs'],
    'r_squared': r1.diagnostics['r_squared_within'],
    'pretrend_pval': trends_result.p_value
})

# Specification 2: With controls
r2 = estimate_did_panel(df, "y", "treated", "id", "year",
                        controls=["x1", "x2"], cluster="id")
results.append({...})

# Generate table
table = create_regression_table(
    results=results,
    column_names=["(1)", "(2)", "(3)", "(4)"],
    title="Table 1: Difference-in-Differences Estimates",
    notes="Robust standard errors clustered at unit level in parentheses."
)

print(table)
```

---

## 2. Event Study Figure

### Standard Format

An event study figure should include:

1. **Point estimates** for each relative time period
2. **Confidence intervals** (typically 95%)
3. **Reference period** clearly marked (usually t=-1)
4. **Vertical line** at treatment time (t=0)
5. **Horizontal line** at y=0
6. **Clear axis labels**

### Python Code

```python
import matplotlib.pyplot as plt
from did_estimator import event_study_plot

fig = event_study_plot(
    data=df,
    outcome="y",
    treatment_time_var="first_treated",
    unit_id="id",
    time_id="year",
    reference_period=-1,
    pre_periods=4,
    post_periods=4,
    figsize=(10, 6)
)

# Customize for publication
ax = fig.axes[0]
ax.set_xlabel('Years Relative to Treatment', fontsize=12)
ax.set_ylabel('Estimated Effect on Y', fontsize=12)
ax.set_title('Figure 1: Event Study - Dynamic Treatment Effects', fontsize=14)

# Add note about normalization
fig.text(0.5, 0.02,
         'Notes: Point estimates and 95% confidence intervals shown. '
         'Coefficients normalized to zero at t=-1. '
         'Standard errors clustered at unit level.',
         ha='center', fontsize=9, style='italic')

plt.tight_layout(rect=[0, 0.05, 1, 1])
fig.savefig('event_study.pdf', dpi=300, bbox_inches='tight')
```

### Interpretation Section Template

```markdown
Figure X presents the event study estimates. The pre-treatment coefficients
(periods -4 to -1) are close to zero and statistically insignificant,
providing support for the parallel trends assumption (joint F-test p = 0.67).

The treatment effect emerges immediately in period 0 with an estimated
effect of [X] (SE = [Y]). The effect [persists/grows/diminishes] over
the post-treatment periods, reaching [Z] by period [N].
```

---

## 3. Parallel Trends Visualization

### Standard Format

```python
from did_estimator import plot_parallel_trends

fig = plot_parallel_trends(
    data=df,
    outcome="y",
    treatment_group="treated_ever",
    time_id="year",
    treatment_time=2015,
    figsize=(10, 6),
    title="Figure 2: Pre-Treatment Trends"
)

# Add vertical line at treatment
ax = fig.axes[0]
ax.axvline(x=2015-0.5, color='red', linestyle='--', linewidth=2, label='Treatment')

# Add legend
ax.legend(loc='upper left')

# Add note
fig.text(0.5, 0.02,
         'Notes: Mean outcomes by treatment status with 95% confidence intervals.',
         ha='center', fontsize=9, style='italic')

fig.savefig('parallel_trends.pdf', dpi=300, bbox_inches='tight')
```

---

## 4. Robustness Table

### Standard Format

```
Table X: Robustness Checks
===============================================================================
                              (1)        (2)        (3)        (4)        (5)
                           Baseline  Alt. SE   Alt. Spec  Placebo   Staggered
-------------------------------------------------------------------------------
Panel A: Main Results
Treatment Effect            2.34***    2.34**    2.18***     0.12      2.21***
                           (0.45)     (0.89)     (0.52)    (0.38)     (0.48)

Panel B: Alternative Samples
Balanced Panel Only         2.41***
                           (0.48)
Excluding Outliers          2.29***
                           (0.44)
Different Time Window       2.15***
                           (0.51)

Panel C: Alternative Specifications
Log Outcome                 0.089***
                           (0.018)
Including Trends            2.08***
                           (0.55)
Propensity Score Weighted   2.52***
                           (0.51)
-------------------------------------------------------------------------------
```

### What to Include

| Robustness Check | Purpose |
|-----------------|---------|
| Alternative clustering | Test SE sensitivity |
| Balanced panel | Address composition changes |
| Different time windows | Test stability across periods |
| Placebo tests | Validate parallel trends |
| Alternative estimators | Test sensitivity to method |
| Different controls | Test sensitivity to specification |
| Weighted estimation | Address selection |

---

## 5. Balance Table

### Standard Format

```
Table X: Summary Statistics and Balance
===============================================================================
                                 Treatment    Control    Difference   p-value
-------------------------------------------------------------------------------
Panel A: Pre-Treatment Characteristics
Age                               42.3         41.8         0.5        0.234
                                 (8.2)        (8.5)
Income (1000s)                    52.4         51.1         1.3        0.156
                                (15.3)       (14.8)
Education (years)                 14.2         14.0         0.2        0.342
                                 (2.8)        (2.9)
Employment Rate                   0.85         0.84        0.01        0.521
                                (0.36)       (0.37)

Panel B: Pre-Treatment Outcomes
Outcome (2012)                    8.34         8.21        0.13        0.456
                                 (2.1)        (2.2)
Outcome (2013)                    8.56         8.41        0.15        0.412
                                 (2.2)        (2.3)
Outcome (2014)                    8.72         8.58        0.14        0.398
                                 (2.1)        (2.2)

Joint F-test p-value                                                   0.672
-------------------------------------------------------------------------------
Notes: Standard deviations in parentheses. P-values from t-tests of equality
of means. Joint F-test tests whether all covariates jointly predict treatment.
Sample restricted to pre-treatment period (2012-2014).
===============================================================================
```

### Python Code

```python
from scripts.robustness_checks import create_balance_table

balance_table = create_balance_table(
    data=df[df['year'] < 2015],  # Pre-treatment only
    treatment_group="treated_ever",
    covariates=["age", "income", "education", "employed"],
    pre_outcomes={"outcome": [2012, 2013, 2014]},
    output_format="latex"  # or "markdown"
)
```

---

## 6. Goodman-Bacon Decomposition Table

### Standard Format (for staggered DID)

```
Table X: Goodman-Bacon Decomposition
===============================================================================
Comparison Type                           Weight    Estimate    Contribution
-------------------------------------------------------------------------------
Earlier vs. Later Treated                  0.35       2.89         1.01
Later vs. Earlier Treated                  0.15       1.45         0.22
Treated vs. Never Treated                  0.50       2.21         1.11
-------------------------------------------------------------------------------
TWFE Estimate                                                      2.34
Robust Estimator (Callaway-Sant'Anna)                             2.21
===============================================================================
Notes: Decomposition following Goodman-Bacon (2021). Earlier vs. Later Treated
comparisons use already-treated units as controls, which can bias estimates
when treatment effects are heterogeneous.
```

---

## 7. Results Section Template

### Introduction

```markdown
## X. Results

### X.1 Main Results

Table X presents our main difference-in-differences estimates. Column (1) shows
the baseline specification with unit and time fixed effects. The estimated
treatment effect is [EFFECT] (SE = [SE]), implying that [TREATMENT] led to a
[DIRECTION] in [OUTCOME] of [MAGNITUDE].

This effect is statistically significant at the [1%/5%/10%] level and represents
a [X%] change relative to the pre-treatment mean of [MEAN] in the treatment group.
```

### Parallel Trends

```markdown
### X.2 Parallel Trends and Identification

Figure X presents evidence on the parallel trends assumption. Panel A shows the
raw outcome means for treatment and control groups over time. The pre-treatment
trends appear parallel, with both groups following [PATTERN] from [YEAR] to [YEAR].

Panel B shows the event study estimates. The pre-treatment coefficients are
individually and jointly insignificant (F-test p = [P-VALUE]), supporting the
parallel trends assumption. We observe no evidence of anticipation effects in
the periods immediately preceding treatment.

The treatment effect emerges [immediately/gradually] in period [T], consistent
with [MECHANISM]. The effect [persists/grows/diminishes] in subsequent periods.
```

### Robustness

```markdown
### X.3 Robustness

Table X presents robustness checks. Our results are robust to:

1. **Alternative standard errors**: Column (2) clusters at [LEVEL], yielding
   [SIMILAR/DIFFERENT] inference.

2. **Alternative specifications**: Column (3) [DESCRIPTION], with an estimated
   effect of [EFFECT] (SE = [SE]).

3. **Placebo test**: Column (4) uses a fake treatment [N] years before actual
   treatment. The placebo effect is small and insignificant ([EFFECT], p = [P]),
   supporting our identification.

4. **Alternative estimators**: Column (5) uses the Callaway-Sant'Anna estimator,
   which addresses potential biases from staggered treatment timing. The estimate
   of [EFFECT] is [SIMILAR TO/DIFFERENT FROM] our baseline TWFE estimate.
```

---

## 8. Appendix Materials

### Standard Appendix Sections

1. **Data Appendix**
   - Variable definitions
   - Sample construction
   - Data sources

2. **Additional Results**
   - Heterogeneity analysis
   - Subgroup estimates
   - Additional robustness checks

3. **Technical Details**
   - Full model specifications
   - Derivation of standard errors
   - Computational details

### Heterogeneity Table Format

```
Table A.X: Heterogeneous Treatment Effects
===============================================================================
                              (1)        (2)        (3)        (4)
                            Overall    Low X      High X    Diff (2)-(3)
-------------------------------------------------------------------------------
Treatment Effect            2.34***    1.89***    2.78***     0.89**
                           (0.45)     (0.52)     (0.61)      (0.42)

Observations               10,000      4,800      5,200
Pre-treatment mean          8.45       7.21       9.58
% Change                   27.7%      26.2%      29.0%
-------------------------------------------------------------------------------
```

---

## 9. Common Mistakes to Avoid

### Reporting Errors

| Mistake | Why It's Wrong | Correct Approach |
|---------|---------------|------------------|
| Not reporting clustering | SEs likely understated | Always specify clustering level |
| Showing only significant specs | Publication bias | Show range of specifications |
| Ignoring pre-trends | Threatens identification | Always test and report |
| Not showing event study | Can't assess dynamics | Include as main or appendix |
| Vague treatment timing | Unclear identification | Specify exact treatment periods |

### Interpretation Errors

| Mistake | Why It's Wrong | Correct Approach |
|---------|---------------|------------------|
| Claiming causality without discussing assumptions | Assumptions may not hold | Discuss and test assumptions |
| Extrapolating to different populations | External validity concerns | Discuss generalizability |
| Ignoring heterogeneity | Average effect may hide variation | Test for heterogeneity |
| Not discussing mechanisms | Reduced policy relevance | Discuss potential mechanisms |

---

## 10. Journal-Specific Requirements

### American Economic Review (AER)

- Data and code availability required
- Detailed replication package
- Pre-registration encouraged for new studies

### Quarterly Journal of Economics (QJE)

- Clear identification section
- Extensive robustness checks expected
- Emphasis on economic magnitude

### Review of Economic Studies (REStud)

- Technical precision valued
- Full econometric details
- Online appendix for supplementary material

### General Best Practices

1. **Transparency**: Report all specifications attempted
2. **Replicability**: Provide complete code and data
3. **Completeness**: Include all diagnostic tests
4. **Clarity**: Make identification strategy explicit

---

## References

1. Christensen, G., & Miguel, E. (2018). "Transparency, Reproducibility, and the Credibility of Economics Research." *Journal of Economic Literature*.

2. Gentzkow, M., & Shapiro, J. M. (2014). "Code and Data for the Social Sciences: A Practitioner's Guide." *University of Chicago*.

3. AER Data and Code Availability Policy: https://www.aeaweb.org/journals/data

---

## See Also

- [did_table.tex](../assets/latex/did_table.tex) - LaTeX table template
- [did_report.md](../assets/markdown/did_report.md) - Full report template
- [visualize_event_study.py](../scripts/visualize_event_study.py) - Figure generation code
