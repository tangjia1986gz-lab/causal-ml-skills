# Common Errors in Statistical Analysis

> **Document Type**: Reference | **Last Updated**: 2025-01
> **Related**: [hypothesis_testing.md](hypothesis_testing.md), [effect_sizes.md](effect_sizes.md), [power_analysis.md](power_analysis.md)

## Overview

Statistical analysis is prone to systematic errors that can invalidate conclusions. This reference catalogs common mistakes in hypothesis testing, effect size interpretation, and inference, with guidance on how to avoid them.

---

## 1. P-Hacking and Data Dredging

### Description

**P-hacking**: Manipulating data or analysis until p < 0.05 is achieved.

**Common Forms**:
- Running many tests and reporting only significant ones
- Adding/removing controls until significance appears
- Excluding "outliers" selectively
- Transforming variables until results are significant
- Optional stopping (collecting data until p < 0.05)

### Example

```python
# BAD: P-hacking through specification search
results = []
for outcome in ['y1', 'y2', 'y3', 'y4', 'y5']:
    for controls in [[], ['x1'], ['x1', 'x2'], ['x1', 'x2', 'x3']]:
        for transform in [None, 'log', 'sqrt']:
            result = run_analysis(outcome, controls, transform)
            if result.p_value < 0.05:
                results.append(result)
                print(f"Found significant: {outcome} with {controls}")  # Cherry picking!
```

### Why It's Wrong

With 20 independent tests at alpha=0.05, P(at least one false positive) = 0.64

This inflates false discovery rate far above the nominal alpha level.

### Solution

```python
# GOOD: Pre-registration and correction
# 1. Pre-register analysis plan before looking at data
# 2. If multiple tests are needed, apply corrections

from statistical_analysis import benjamini_hochberg

# Run all pre-specified analyses
p_values = [run_analysis(outcome).p_value for outcome in outcomes]

# Apply FDR correction
adjusted = benjamini_hochberg(p_values, alpha=0.05)

# Report ALL results with adjusted p-values
for outcome, p_adj in zip(outcomes, adjusted):
    print(f"{outcome}: p_adjusted = {p_adj:.4f}")
```

---

## 2. Misinterpretation of P-Values

### Common Misunderstandings

| Misconception | Truth |
|---------------|-------|
| "p = 0.03 means 3% chance H0 is true" | p-value is P(data at least this extreme | H0 true) |
| "p < 0.05 means effect is important" | Statistical significance != practical significance |
| "p = 0.06 is 'approaching significance'" | Either pre-register alpha or treat as non-significant |
| "p = 0.001 is more significant than p = 0.01" | Both are significant; compare effect sizes instead |
| "Large p means H0 is true" | Absence of evidence != evidence of absence |

### Example

```python
# BAD: Incorrect interpretation
if p_value < 0.05:
    print("There is a 95% probability that the effect is real")  # WRONG!

# GOOD: Correct interpretation
if p_value < 0.05:
    print(f"If there were no effect, seeing data this extreme would occur "
          f"{p_value*100:.1f}% of the time.")
    print(f"Effect size: d = {cohens_d:.2f}, 95% CI [{ci_low:.2f}, {ci_high:.2f}]")
```

### Solution

Always report:
1. Exact p-value (not just "p < 0.05")
2. Effect size with confidence interval
3. Context for practical significance

---

## 3. Multiple Comparisons Problem

### Description

Performing multiple statistical tests without adjusting for inflated false positive rate.

### Example

```python
# BAD: 20 comparisons without adjustment
outcomes = ['y' + str(i) for i in range(1, 21)]
for outcome in outcomes:
    result = run_ttest(df, outcome, 'treatment')
    if result.p_value < 0.05:
        print(f"{outcome} is significant!")  # Expected ~1 false positive by chance!

# GOOD: Adjust for multiple comparisons
p_values = [run_ttest(df, outcome, 'treatment').p_value for outcome in outcomes]
adjusted = benjamini_hochberg(p_values, alpha=0.05)

significant = [outcomes[i] for i, p in enumerate(adjusted) if p < 0.05]
print(f"Significant after FDR correction: {significant}")
```

### When Adjustments Are Needed

| Scenario | Adjustment Needed? |
|----------|-------------------|
| Single pre-registered hypothesis | No |
| Multiple pre-specified tests | Yes (FWER or FDR) |
| Exploratory analysis | Yes (FDR) |
| Different research questions | Maybe (consider context) |
| Robustness checks | Usually no |

---

## 4. Confusing Statistical and Practical Significance

### Description

Treating any p < 0.05 result as meaningful without considering effect size.

### Example

```python
# Study with n = 100,000
# BAD: Only report significance
result = run_ttest(df, 'income', 'treatment')
if result.p_value < 0.05:
    print(f"Treatment significantly increased income (p < 0.001)")  # Uninformative!

# GOOD: Report effect size and practical meaning
print(f"Treatment effect: ${result.mean_diff:.2f}")
print(f"Cohen's d: {result.cohens_d:.3f}")  # d = 0.02 (trivial)
print(f"This represents a {result.mean_diff/baseline_income*100:.2f}% increase")
```

### Guidelines

| d | r | Interpretation for Policy |
|---|---|---------------------------|
| 0.01 | 0.005 | Trivial - not worth implementing |
| 0.10 | 0.05 | Small - consider if cheap/scalable |
| 0.20 | 0.10 | Modest - worth considering |
| 0.50 | 0.24 | Medium - meaningful impact |
| 0.80+ | 0.37+ | Large - substantial effect |

---

## 5. Optional Stopping (Peeking)

### Description

Repeatedly testing and stopping when p < 0.05 is achieved, inflating false positive rate.

### Example

```python
# BAD: Optional stopping
n = 0
p_value = 1.0
while p_value >= 0.05 and n < 500:
    n += 10  # Collect 10 more
    result = run_ttest(df[:n], 'y', 'treatment')
    p_value = result.p_value
    if p_value < 0.05:
        print(f"Significant at n={n}!")  # Inflated false positive rate!

# GOOD: Pre-specify sample size
n_required = 128  # From power analysis
# Collect all data, then analyze once
result = run_ttest(df[:n_required], 'y', 'treatment')
```

### Solution: Sequential Analysis

If interim looks are necessary, use proper sequential methods:

```python
from statistical_analysis import sequential_test

# O'Brien-Fleming boundaries
boundaries = sequential_test(
    n_looks=3,
    alpha=0.05,
    spending_function='obrien_fleming'
)

# At each interim: compare observed z to boundary
for look in range(3):
    result = analyze_data_at_look(look)
    if abs(result.z_score) > boundaries[look]:
        print(f"Stop at look {look+1}: Significant")
        break
```

---

## 6. HARKing (Hypothesizing After Results Known)

### Description

Formulating hypotheses after seeing the data, then presenting as if they were a priori.

### Example

```python
# BAD: Explore -> Find pattern -> Pretend it was predicted
# Run exploratory analysis
for subgroup in ['male', 'female', 'young', 'old', 'urban', 'rural']:
    result = run_ttest(df[df.subgroup == subgroup], 'y', 'treatment')
    if result.p_value < 0.05:
        # WRONG: "We hypothesized that treatment would work for [subgroup]"
        pass

# GOOD: Clearly separate exploratory and confirmatory
# Exploratory (hypothesis generating):
exploratory_results = {}
for subgroup in subgroups:
    exploratory_results[subgroup] = run_ttest(df_explore[df_explore.subgroup == subgroup], ...)

# Confirmatory (hypothesis testing on held-out data):
# Pre-register specific hypothesis based on exploratory
result = run_ttest(df_confirm[df_confirm.subgroup == 'male'], ...)
```

---

## 7. Garden of Forking Paths

### Description

The many undisclosed researcher degrees of freedom that allow finding significance through analytic flexibility.

### Common Decision Points

| Decision | Options |
|----------|---------|
| Outlier handling | Keep all, winsorize, remove >2SD, remove >3SD |
| Missing data | Listwise deletion, imputation, multiple imputation |
| Variable transformation | None, log, sqrt, inverse |
| Control variables | None, some, all, kitchen sink |
| Subgroup analysis | Which subgroups to test |
| Model specification | Linear, polynomial, interactions |

### Example

With just 5 binary decisions, there are 2^5 = 32 different analyses possible.

### Solution: Multiverse Analysis

```python
from statistical_analysis import multiverse_analysis

# Specify all reasonable choices
specifications = {
    'outliers': ['keep', 'remove_2sd', 'remove_3sd'],
    'controls': [[], ['x1'], ['x1', 'x2']],
    'transform': [None, 'log']
}

# Run all combinations
results = multiverse_analysis(
    data=df,
    outcome='y',
    treatment='treatment',
    specifications=specifications
)

# Report distribution of results
print(f"Effect sizes range: {results.effects.min():.3f} to {results.effects.max():.3f}")
print(f"Significant in {results.pct_significant:.1f}% of specifications")
```

---

## 8. Regression to the Mean

### Description

Selecting subjects based on extreme values, then observing "improvement" that's actually statistical artifact.

### Example

```python
# BAD: Select based on pre-test, then claim improvement
# Select students with lowest pre-test scores
low_performers = df[df['pretest'] < df['pretest'].quantile(0.25)]

# "Improvement" is partly regression to the mean!
improvement = low_performers['posttest'].mean() - low_performers['pretest'].mean()
print(f"Improvement: {improvement:.2f}")  # MISLEADING!

# GOOD: Use proper control group
# Compare to control group selected the same way
treatment_low = df[(df['treatment']==1) & (df['pretest'] < threshold)]
control_low = df[(df['treatment']==0) & (df['pretest'] < threshold)]

# DID to remove regression to mean
treatment_change = treatment_low['posttest'].mean() - treatment_low['pretest'].mean()
control_change = control_low['posttest'].mean() - control_low['pretest'].mean()
true_effect = treatment_change - control_change
```

---

## 9. Base Rate Fallacy

### Description

Ignoring prevalence when interpreting test results or rare effects.

### Example

Testing for a rare condition:
- Condition prevalence: 1%
- Test sensitivity: 99%
- Test specificity: 95%

```python
# If test is positive, what's P(have condition)?
# BAD thinking: "Test is 99% accurate, so 99% chance I have it"

# GOOD: Apply Bayes' theorem
from statistical_analysis import positive_predictive_value

ppv = positive_predictive_value(
    sensitivity=0.99,
    specificity=0.95,
    prevalence=0.01
)
print(f"P(condition | positive test) = {ppv:.2%}")  # Only ~17%!
```

---

## 10. Simpson's Paradox

### Description

A trend that appears in aggregated data reverses when data is disaggregated by a confounding variable.

### Example

```python
# Aggregated: Treatment appears harmful
overall_result = run_ttest(df, 'outcome', 'treatment')
print(f"Overall effect: {overall_result.mean_diff:.2f}")  # Negative!

# But stratified by severity...
for severity in ['mild', 'severe']:
    subset = df[df['severity'] == severity]
    result = run_ttest(subset, 'outcome', 'treatment')
    print(f"{severity}: effect = {result.mean_diff:.2f}")  # Both positive!

# Explanation: More severe patients got treatment, and they have worse outcomes
# regardless of treatment -> confounding!
```

### Solution

Check for confounding before aggregating:

```python
from statistical_analysis import check_simpsons_paradox

result = check_simpsons_paradox(
    data=df,
    outcome='outcome',
    treatment='treatment',
    potential_confounders=['severity', 'age_group', 'hospital']
)

if result.paradox_detected:
    print(f"Warning: Simpson's paradox detected for {result.confounders}")
```

---

## 11. Dichotomizing Continuous Variables

### Description

Converting continuous predictors into categories (e.g., median split), losing information and creating artifacts.

### Problems

1. Loses statistical power (equivalent to discarding 1/3 of data)
2. Creates artificial discontinuity at cutpoint
3. Cutpoint choice is arbitrary
4. Assumes step function relationship

### Example

```python
# BAD: Median split
df['income_high'] = (df['income'] > df['income'].median()).astype(int)
result = run_ttest(df, 'outcome', 'income_high')

# GOOD: Use continuous variable
from statsmodels.api import OLS
model = OLS(df['outcome'], df[['income', 'constant']]).fit()

# Or if non-linear: use splines
from statistical_analysis import spline_regression
result = spline_regression(df, 'outcome', 'income', knots=3)
```

---

## 12. Ecological Fallacy

### Description

Inferring individual-level relationships from aggregate data.

### Example

```python
# BAD: Aggregate data -> individual inference
# State-level data shows: states with higher education have higher crime
# WRONG conclusion: "More educated people commit more crimes"

# The fallacy: relationship at aggregate level may not hold at individual level
# Reality: within states, education is negatively correlated with crime

# GOOD: Use individual-level data when available
# Or explicitly note ecological inference limitations
```

---

## 13. Ignoring Effect Heterogeneity

### Description

Reporting only average effects when effects vary substantially across subgroups.

### Example

```python
# BAD: Report only overall effect
result = run_ttest(df, 'outcome', 'treatment')
print(f"Average effect: {result.mean_diff:.2f}")

# GOOD: Check for heterogeneity
from statistical_analysis import heterogeneity_analysis

het = heterogeneity_analysis(
    data=df,
    outcome='outcome',
    treatment='treatment',
    moderators=['gender', 'age', 'baseline_severity']
)

print("Subgroup effects:")
for subgroup, effect in het.subgroup_effects.items():
    print(f"  {subgroup}: {effect:.2f}")

if het.significant_heterogeneity:
    print("Warning: Significant treatment effect heterogeneity detected")
```

---

## 14. Survivorship Bias

### Description

Analyzing only "survivors" while ignoring those who dropped out, failed, or died.

### Example

```python
# BAD: Analyze only complete cases
df_complete = df.dropna()  # Lost 30% of sample!
result = run_analysis(df_complete)

# GOOD: Investigate attrition
from statistical_analysis import attrition_analysis

attrition = attrition_analysis(
    data=df,
    treatment='treatment',
    covariates=['age', 'income', 'severity']
)

if attrition.differential:
    print("WARNING: Differential attrition detected!")
    print(f"Treatment attrition: {attrition.treatment_rate:.1%}")
    print(f"Control attrition: {attrition.control_rate:.1%}")
    print("Results may be biased.")
```

---

## 15. Confirmation Bias in Analysis

### Description

Unconsciously favoring analyses that confirm prior beliefs.

### Signs

- Only running analyses expected to "work"
- Dismissing contradictory results as "noise"
- Preferring specifications that show desired effects
- Stopping analysis when desired result is found

### Solution

```python
# Pre-registration protocol
# 1. State hypotheses BEFORE analysis
# 2. Specify exact statistical tests
# 3. Specify data exclusion criteria
# 4. Specify any planned subgroup analyses
# 5. Commit to reporting ALL pre-registered analyses

# Adversarial collaboration
# - Have someone with opposite hypothesis involved
# - Agree on analysis plan together

# Robustness checks
# - Report results across multiple reasonable specifications
# - Highlight when results are sensitive to choices
```

---

## Summary: Pre-Analysis Checklist

### Before Analysis

- [ ] Hypotheses clearly stated
- [ ] Sample size justified by power analysis
- [ ] Statistical tests pre-specified
- [ ] Handling of missing data planned
- [ ] Outlier criteria defined
- [ ] Multiple comparison adjustment planned
- [ ] Subgroup analyses limited and justified

### During Analysis

- [ ] Follow pre-registered plan
- [ ] Document any deviations
- [ ] Label exploratory analyses clearly
- [ ] Check assumptions of statistical tests
- [ ] Calculate effect sizes with CIs

### Reporting

- [ ] Report ALL pre-registered analyses
- [ ] Report exact p-values (not just <0.05)
- [ ] Report effect sizes with confidence intervals
- [ ] Clearly separate confirmatory and exploratory
- [ ] Discuss practical significance
- [ ] Report sensitivity analyses
- [ ] Acknowledge limitations

---

## References

### Key Papers

- Simmons, J. P., Nelson, L. D., & Simonsohn, U. (2011). False-positive psychology. *Psychological Science*, 22(11), 1359-1366.
- Gelman, A., & Loken, E. (2014). The statistical crisis in science. *American Scientist*, 102(6), 460-465.
- Wasserstein, R. L., & Lazar, N. A. (2016). The ASA statement on p-values. *The American Statistician*, 70(2), 129-133.
- Benjamin, D. J., et al. (2018). Redefine statistical significance. *Nature Human Behaviour*, 2(1), 6-10.
- Ioannidis, J. P. (2005). Why most published research findings are false. *PLoS Medicine*, 2(8), e124.

### Practical Guides

- Lakens, D. (2022). *Improving Your Statistical Inferences*. Online course.
- Cumming, G. (2012). *Understanding the New Statistics*. Routledge.
- McShane, B. B., & Gal, D. (2016). Statistical significance and the dichotomization of evidence. *JASA*, 112(519), 885-895.
