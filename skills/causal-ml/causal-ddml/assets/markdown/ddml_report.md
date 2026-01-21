# Double/Debiased Machine Learning Analysis Report

> Template for DDML analysis reporting

**Study**: [STUDY TITLE]
**Date**: [DATE]
**Analyst**: [NAME]

---

## Executive Summary

[Brief summary of key findings - 2-3 sentences]

- **Treatment Effect**: [X.XXX] (SE = [X.XXX])
- **95% Confidence Interval**: [[X.XXX], [X.XXX]]
- **Statistical Significance**: [Yes/No at 5% level]
- **Robustness**: [Robust/Sensitive to specification choices]

---

## 1. Research Question

**Question**: [State the causal question]

**Treatment Variable**: [Name and description]
- Type: [Binary/Continuous]
- Variation: [Source of variation in treatment]

**Outcome Variable**: [Name and description]

**Identification Strategy**: [Brief description of why unconfoundedness is plausible]

---

## 2. Data Description

### 2.1 Sample

| Characteristic | Value |
|----------------|-------|
| Total observations | [N] |
| Treatment group | [N1] |
| Control group | [N0] |
| Time period | [PERIOD] |
| Geographic scope | [REGION] |

### 2.2 Variables

**Outcome**: [Y]
- Mean: [X.XX]
- Std Dev: [X.XX]
- Range: [[MIN], [MAX]]

**Treatment**: [D]
- Type: [Binary/Continuous]
- Proportion treated / Mean: [X.XX]

**Controls (p = [N])**:
- [Category 1]: [list key variables]
- [Category 2]: [list key variables]
- [Category 3]: [list key variables]

### 2.3 Missing Data

| Variable | N Missing | % Missing | Handling |
|----------|-----------|-----------|----------|
| [VAR1] | [N] | [%] | [METHOD] |
| [VAR2] | [N] | [%] | [METHOD] |

---

## 3. Methodology

### 3.1 Model Specification

We estimate treatment effects using **Double/Debiased Machine Learning** (Chernozhukov et al., 2018).

**Model**: [PLR / IRM]

For PLR (Partially Linear Regression):
```
Y = D * theta + g(X) + epsilon
D = m(X) + V
```

For IRM (Interactive Regression Model):
```
Y = g(D, X) + epsilon
D ~ Bernoulli(m(X))
```

### 3.2 Nuisance Functions

| Function | Learner | Tuning |
|----------|---------|--------|
| E[Y\|X] | [LEARNER] | [5-fold CV] |
| E[D\|X] | [LEARNER] | [5-fold CV] |

**Learner Selection**: [Automatic via CV / Specified based on domain knowledge]

### 3.3 Cross-Fitting

- **Number of folds (K)**: [5]
- **Number of repetitions**: [10]
- **Score function**: [Partialling out / AIPW]

### 3.4 Inference

- Standard errors: [Influence function based]
- Confidence intervals: [Normal approximation / Bootstrap]
- Multiple testing correction: [None / Bonferroni / FDR]

---

## 4. Results

### 4.1 Main Results

| Specification | Effect | SE | 95% CI | P-value |
|---------------|--------|-----|--------|---------|
| (1) Baseline | [X.XXX] | [X.XXX] | [[X.XXX], [X.XXX]] | [X.XXX] |
| (2) Alt Learner | [X.XXX] | [X.XXX] | [[X.XXX], [X.XXX]] | [X.XXX] |
| (3) Alt Controls | [X.XXX] | [X.XXX] | [[X.XXX], [X.XXX]] | [X.XXX] |

**Interpretation**: [Describe the treatment effect in substantive terms]

### 4.2 Learner Comparison

| Learner | Effect | SE | P-value |
|---------|--------|-----|---------|
| Lasso | [X.XXX] | [X.XXX] | [X.XXX] |
| Ridge | [X.XXX] | [X.XXX] | [X.XXX] |
| Random Forest | [X.XXX] | [X.XXX] | [X.XXX] |
| XGBoost | [X.XXX] | [X.XXX] | [X.XXX] |

**Sensitivity Summary**:
- Effect range: [[X.XXX], [X.XXX]]
- Coefficient of variation: [X.X]%
- All specifications significant: [Yes/No]
- All same sign: [Yes/No]

### 4.3 PLR vs IRM Comparison (Binary Treatment)

| Model | Effect | SE | Estimand |
|-------|--------|-----|----------|
| PLR | [X.XXX] | [X.XXX] | ATE (constant effect) |
| IRM | [X.XXX] | [X.XXX] | ATE (heterogeneous) |

**Difference**: [X.X]%
**Interpretation**: [Do results suggest effect heterogeneity?]

---

## 5. Diagnostics

### 5.1 Nuisance Model Performance

| Model | R-squared (CV) | MSE (CV) |
|-------|----------------|----------|
| E[Y\|X] | [X.XX] | [X.XX] |
| E[D\|X] | [X.XX] | [X.XX] |

**Assessment**: [Good/Moderate/Low fit - interpretation]

### 5.2 Propensity Score Overlap (IRM)

| Statistic | Value |
|-----------|-------|
| Min propensity | [X.XX] |
| Max propensity | [X.XX] |
| % extreme (< 0.05 or > 0.95) | [X.X]% |
| N trimmed | [N] |

**Assessment**: [Good overlap / Overlap concerns]

### 5.3 Cross-Fitting Stability

| Metric | Value |
|--------|-------|
| Estimate std across folds | [X.XXX] |
| Estimate std across repetitions | [X.XXX] |
| CV (fold variation) | [X.X]% |

**Assessment**: [Stable / Some instability]

---

## 6. Robustness Checks

### 6.1 Sensitivity to Specifications

| Variation | Effect | Change from Baseline |
|-----------|--------|---------------------|
| Baseline | [X.XXX] | -- |
| Different learner | [X.XXX] | [+/-X.X]% |
| Different K | [X.XXX] | [+/-X.X]% |
| Different trimming | [X.XXX] | [+/-X.X]% |
| Subset of controls | [X.XXX] | [+/-X.X]% |

### 6.2 Subgroup Analysis

| Subgroup | N | Effect | SE |
|----------|---|--------|-----|
| [GROUP1] | [N] | [X.XXX] | [X.XXX] |
| [GROUP2] | [N] | [X.XXX] | [X.XXX] |

### 6.3 Placebo Tests

| Test | Effect | P-value | Pass? |
|------|--------|---------|-------|
| [PLACEBO1] | [X.XXX] | [X.XXX] | [Yes/No] |
| [PLACEBO2] | [X.XXX] | [X.XXX] | [Yes/No] |

---

## 7. Discussion

### 7.1 Interpretation

[Interpret the treatment effect in context of the research question]

### 7.2 Assumptions

**Unconfoundedness**: [Discussion of whether assumption is plausible]
- Included controls: [List key confounders]
- Potential omitted variables: [Discuss concerns]

**Overlap**: [Discussion of propensity overlap]

**Correct Model Class**: [Discussion of nuisance model adequacy]

### 7.3 Limitations

1. [Limitation 1]
2. [Limitation 2]
3. [Limitation 3]

### 7.4 Comparison to Prior Work

[How do results compare to existing literature?]

---

## 8. Conclusion

[Summary of findings and implications]

**Key Finding**: The treatment effect is [X.XXX] (95% CI: [[X.XXX], [X.XXX]]), indicating that [interpretation].

**Robustness**: Results are [robust/sensitive] to specification choices.

**Policy Implications**: [If applicable]

---

## References

1. Chernozhukov, V., Chetverikov, D., Demirer, M., Duflo, E., Hansen, C., Newey, W., & Robins, J. (2018). Double/Debiased Machine Learning for Treatment and Structural Parameters. *The Econometrics Journal*, 21(1), C1-C68.

2. [Additional references]

---

## Appendix

### A. Variable Definitions

| Variable | Definition | Source |
|----------|------------|--------|
| [VAR1] | [DEFINITION] | [SOURCE] |
| [VAR2] | [DEFINITION] | [SOURCE] |

### B. Additional Results

[Include additional tables, figures, or sensitivity analyses]

### C. Code Availability

Analysis code available at: [REPOSITORY URL]

Software versions:
- Python: [VERSION]
- DoubleML: [VERSION]
- scikit-learn: [VERSION]

---

*Generated using DDML Analysis Framework*
*Reference: Chernozhukov et al. (2018)*
