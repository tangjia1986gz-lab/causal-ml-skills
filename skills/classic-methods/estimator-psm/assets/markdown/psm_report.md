# Propensity Score Matching Analysis Report

> **Analysis Date**: [PLACEHOLDER: YYYY-MM-DD]
> **Analyst**: [PLACEHOLDER]
> **Data Source**: [PLACEHOLDER]

---

## Executive Summary

This report presents the results of a propensity score matching (PSM) analysis examining the effect of [PLACEHOLDER: treatment] on [PLACEHOLDER: outcome]. The estimated Average Treatment Effect on the Treated (ATT) is **[PLACEHOLDER: effect]** (95% CI: [PLACEHOLDER: ci_lower, ci_upper], p = [PLACEHOLDER: p_value]).

**Key Findings**:
- [PLACEHOLDER: Finding 1]
- [PLACEHOLDER: Finding 2]
- [PLACEHOLDER: Finding 3]

---

## 1. Research Question and Identification Strategy

### 1.1 Research Question

[PLACEHOLDER: State the causal question being addressed]

### 1.2 Treatment Definition

- **Treatment**: [PLACEHOLDER: Define treatment precisely]
- **Control**: [PLACEHOLDER: Define control condition]
- **Timing**: [PLACEHOLDER: When treatment was administered]

### 1.3 Outcome Definition

- **Primary Outcome**: [PLACEHOLDER: Define primary outcome]
- **Measurement**: [PLACEHOLDER: How outcome was measured]
- **Timing**: [PLACEHOLDER: When outcome was measured relative to treatment]

### 1.4 Identification Assumptions

We rely on the following assumptions for causal identification:

1. **Conditional Independence (Unconfoundedness)**: Selection into treatment is based solely on observed covariates X. Formally: $(Y_0, Y_1) \perp D | X$

2. **Common Support (Overlap)**: For all covariate values, both treated and untreated observations exist. Formally: $0 < P(D=1|X) < 1$

3. **SUTVA**: No interference between units; treatment is well-defined.

**Justification for Unconfoundedness**:
[PLACEHOLDER: Explain why selection on observables is plausible. What confounders are included? Are there potential unobserved confounders?]

---

## 2. Data and Sample

### 2.1 Data Source

[PLACEHOLDER: Describe data source, collection methods, time period]

### 2.2 Sample Construction

| Criterion | N |
|-----------|---|
| Initial sample | [PLACEHOLDER] |
| After exclusion 1: [PLACEHOLDER] | [PLACEHOLDER] |
| After exclusion 2: [PLACEHOLDER] | [PLACEHOLDER] |
| Final analysis sample | [PLACEHOLDER] |

### 2.3 Sample Characteristics

| Variable | Treated (N=[PLACEHOLDER]) | Control (N=[PLACEHOLDER]) |
|----------|---------------------------|---------------------------|
| [PLACEHOLDER: var1] | [PLACEHOLDER] | [PLACEHOLDER] |
| [PLACEHOLDER: var2] | [PLACEHOLDER] | [PLACEHOLDER] |
| [PLACEHOLDER: var3] | [PLACEHOLDER] | [PLACEHOLDER] |

---

## 3. Propensity Score Estimation

### 3.1 Model Specification

**Estimation Method**: [PLACEHOLDER: Logistic regression / GBM / Random Forest / LASSO]

**Covariates Included**:
- Demographic: [PLACEHOLDER: e.g., age, sex, race]
- Socioeconomic: [PLACEHOLDER: e.g., income, education, employment]
- Health/Risk factors: [PLACEHOLDER]
- Prior outcomes: [PLACEHOLDER: e.g., lagged outcome]

**Functional Form**: [PLACEHOLDER: Main effects only / With interactions / Polynomial terms]

### 3.2 Model Diagnostics

| Metric | Value |
|--------|-------|
| AUC (ROC) | [PLACEHOLDER] |
| Pseudo R-squared | [PLACEHOLDER] |
| Hosmer-Lemeshow p-value | [PLACEHOLDER] |

**Feature Importance** (top 5):

| Variable | Importance/Coefficient |
|----------|------------------------|
| [PLACEHOLDER] | [PLACEHOLDER] |
| [PLACEHOLDER] | [PLACEHOLDER] |
| [PLACEHOLDER] | [PLACEHOLDER] |
| [PLACEHOLDER] | [PLACEHOLDER] |
| [PLACEHOLDER] | [PLACEHOLDER] |

---

## 4. Common Support Assessment

### 4.1 Propensity Score Distribution

| Statistic | Treated | Control |
|-----------|---------|---------|
| Minimum | [PLACEHOLDER] | [PLACEHOLDER] |
| 5th percentile | [PLACEHOLDER] | [PLACEHOLDER] |
| Median | [PLACEHOLDER] | [PLACEHOLDER] |
| Mean | [PLACEHOLDER] | [PLACEHOLDER] |
| 95th percentile | [PLACEHOLDER] | [PLACEHOLDER] |
| Maximum | [PLACEHOLDER] | [PLACEHOLDER] |

### 4.2 Common Support Region

| Metric | Value |
|--------|-------|
| Common support bounds | [[PLACEHOLDER], [PLACEHOLDER]] |
| Overlap ratio | [PLACEHOLDER] |
| % Treated in common support | [PLACEHOLDER]% |
| % Control in common support | [PLACEHOLDER]% |

**Assessment**: [PLACEHOLDER: Good overlap / Limited overlap / Poor overlap]

**Action Taken**: [PLACEHOLDER: No trimming / Trimmed to common support / Used caliper matching]

### 4.3 Overlap Visualization

[PLACEHOLDER: Include propensity score distribution figure]

---

## 5. Matching

### 5.1 Matching Method

| Parameter | Value |
|-----------|-------|
| Algorithm | [PLACEHOLDER: Nearest neighbor / Kernel / Mahalanobis] |
| Match ratio | [PLACEHOLDER: 1:1 / 1:K / variable] |
| With replacement | [PLACEHOLDER: Yes / No] |
| Caliper | [PLACEHOLDER: None / 0.1 SD / etc.] |
| Distance metric | [PLACEHOLDER: Propensity score / Mahalanobis] |

### 5.2 Matching Results

| Metric | Value |
|--------|-------|
| Treated units | [PLACEHOLDER] |
| Matched treated | [PLACEHOLDER] |
| Unmatched treated | [PLACEHOLDER] ([PLACEHOLDER]%) |
| Total matched controls | [PLACEHOLDER] |
| Unique matched controls | [PLACEHOLDER] |
| Mean PS distance | [PLACEHOLDER] |
| Max PS distance | [PLACEHOLDER] |

---

## 6. Covariate Balance

### 6.1 Balance Statistics

| Variable | Before Matching ||| After Matching |||
|----------|---------|---------|-----|---------|---------|-----|
|          | Treated | Control | SMD | Treated | Control | SMD |
| [PLACEHOLDER] | [PLACEHOLDER] | [PLACEHOLDER] | [PLACEHOLDER] | [PLACEHOLDER] | [PLACEHOLDER] | [PLACEHOLDER] |
| [PLACEHOLDER] | [PLACEHOLDER] | [PLACEHOLDER] | [PLACEHOLDER] | [PLACEHOLDER] | [PLACEHOLDER] | [PLACEHOLDER] |
| [PLACEHOLDER] | [PLACEHOLDER] | [PLACEHOLDER] | [PLACEHOLDER] | [PLACEHOLDER] | [PLACEHOLDER] | [PLACEHOLDER] |

### 6.2 Balance Summary

| Metric | Before | After |
|--------|--------|-------|
| Max |SMD| | [PLACEHOLDER] | [PLACEHOLDER] |
| Mean |SMD| | [PLACEHOLDER] | [PLACEHOLDER] |
| # covariates with |SMD| > 0.1 | [PLACEHOLDER] | [PLACEHOLDER] |
| # covariates with |SMD| > 0.25 | [PLACEHOLDER] | [PLACEHOLDER] |

**Balance Achieved**: [PLACEHOLDER: Yes / No]

### 6.3 Love Plot

[PLACEHOLDER: Include Love plot figure]

---

## 7. Treatment Effect Estimation

### 7.1 Primary Results

| Estimand | Estimate | SE | 95% CI | p-value |
|----------|----------|-----|--------|---------|
| **ATT** | [PLACEHOLDER] | [PLACEHOLDER] | [[PLACEHOLDER], [PLACEHOLDER]] | [PLACEHOLDER] |

**Interpretation**: [PLACEHOLDER: Interpret the treatment effect in context]

### 7.2 Robustness Checks

| Specification | Estimate | SE | 95% CI |
|---------------|----------|-----|--------|
| Primary | [PLACEHOLDER] | [PLACEHOLDER] | [PLACEHOLDER] |
| Alternative matching (NN without replacement) | [PLACEHOLDER] | [PLACEHOLDER] | [PLACEHOLDER] |
| Alternative matching (Kernel) | [PLACEHOLDER] | [PLACEHOLDER] | [PLACEHOLDER] |
| IPW estimator | [PLACEHOLDER] | [PLACEHOLDER] | [PLACEHOLDER] |
| Doubly robust | [PLACEHOLDER] | [PLACEHOLDER] | [PLACEHOLDER] |
| Alternative PS model | [PLACEHOLDER] | [PLACEHOLDER] | [PLACEHOLDER] |

---

## 8. Sensitivity Analysis

### 8.1 Rosenbaum Bounds

| Gamma | Effect Lower | Effect Upper | P-value (Upper) |
|-------|--------------|--------------|-----------------|
| 1.00 | [PLACEHOLDER] | [PLACEHOLDER] | [PLACEHOLDER] |
| 1.25 | [PLACEHOLDER] | [PLACEHOLDER] | [PLACEHOLDER] |
| 1.50 | [PLACEHOLDER] | [PLACEHOLDER] | [PLACEHOLDER] |
| 1.75 | [PLACEHOLDER] | [PLACEHOLDER] | [PLACEHOLDER] |
| 2.00 | [PLACEHOLDER] | [PLACEHOLDER] | [PLACEHOLDER] |
| 2.25 | [PLACEHOLDER] | [PLACEHOLDER] | [PLACEHOLDER] |
| 2.50 | [PLACEHOLDER] | [PLACEHOLDER] | [PLACEHOLDER] |

**Critical Gamma**: [PLACEHOLDER]

**Interpretation**: An unobserved confounder would need to change the odds of treatment by a factor of [PLACEHOLDER] to render the treatment effect statistically insignificant.

### 8.2 Sensitivity Assessment

[PLACEHOLDER: Discuss what magnitude of hidden bias is plausible given domain knowledge]

---

## 9. Limitations

### 9.1 Identification Limitations

1. **Unobserved Confounding**: [PLACEHOLDER: Discuss potential unobserved confounders]

2. **Selection on Observables**: [PLACEHOLDER: Discuss plausibility of this assumption]

3. **Measurement Error**: [PLACEHOLDER: Discuss any measurement issues]

### 9.2 Data Limitations

1. [PLACEHOLDER: Limitation 1]
2. [PLACEHOLDER: Limitation 2]

### 9.3 External Validity

[PLACEHOLDER: Discuss generalizability of results]

---

## 10. Conclusions

### 10.1 Main Findings

[PLACEHOLDER: Summarize main findings]

### 10.2 Policy Implications

[PLACEHOLDER: Discuss implications for policy or practice]

### 10.3 Future Research

[PLACEHOLDER: Suggest directions for future research]

---

## Technical Appendix

### A.1 Software and Packages

- Python version: [PLACEHOLDER]
- Key packages:
  - psm_estimator (custom)
  - scikit-learn
  - pandas
  - numpy
  - scipy

### A.2 Reproducibility

- Random seed: [PLACEHOLDER]
- Code repository: [PLACEHOLDER]
- Data availability: [PLACEHOLDER]

### A.3 Additional Tables and Figures

[PLACEHOLDER: Include any additional material]

---

## References

[PLACEHOLDER: Include relevant references]

- Rosenbaum, P. R., & Rubin, D. B. (1983). The Central Role of the Propensity Score in Observational Studies for Causal Effects. *Biometrika*, 70(1), 41-55.
- Stuart, E. A. (2010). Matching Methods for Causal Inference: A Review and a Look Forward. *Statistical Science*, 25(1), 1-21.
- Imbens, G. W., & Rubin, D. B. (2015). *Causal Inference for Statistics, Social, and Biomedical Sciences*. Cambridge University Press.

---

*Report generated by PSM Analysis Framework*
