# Instrumental Variables Analysis Report

> **Generated**: {{TIMESTAMP}}
> **Analyst**: {{ANALYST}}
> **Version**: 1.0

---

## Executive Summary

This report presents the results of an Instrumental Variables (IV) analysis estimating the causal effect of **{{TREATMENT}}** on **{{OUTCOME}}**. The analysis uses **{{INSTRUMENTS}}** as instrument(s).

**Key Finding**: The IV estimate indicates that {{TREATMENT}} has a **{{DIRECTION}}** effect on {{OUTCOME}} of **{{EFFECT}}** (SE = {{SE}}, 95% CI: [{{CI_LOWER}}, {{CI_UPPER}}]).

**Validity Assessment**: {{VALIDITY_SUMMARY}}

---

## 1. Research Question

**Primary Question**: What is the causal effect of {{TREATMENT}} on {{OUTCOME}}?

**Identification Challenge**: {{IDENTIFICATION_CHALLENGE}}

**Proposed Solution**: We use instrumental variables estimation with {{INSTRUMENTS}} as instruments for {{TREATMENT}}.

---

## 2. Data and Variables

### 2.1 Sample

| Statistic | Value |
|-----------|-------|
| Sample size | {{N_OBS}} |
| Time period | {{TIME_PERIOD}} |
| Unit of observation | {{UNIT}} |

### 2.2 Variable Definitions

| Variable | Definition | Mean (SD) |
|----------|------------|-----------|
| **Outcome** (Y): {{OUTCOME}} | {{OUTCOME_DEF}} | {{OUTCOME_MEAN}} ({{OUTCOME_SD}}) |
| **Treatment** (D): {{TREATMENT}} | {{TREATMENT_DEF}} | {{TREATMENT_MEAN}} ({{TREATMENT_SD}}) |
| **Instruments** (Z): {{INSTRUMENTS}} | {{INSTRUMENTS_DEF}} | {{INSTRUMENTS_MEAN}} |
| **Controls** (X): {{CONTROLS}} | {{CONTROLS_DEF}} | - |

---

## 3. Identification Strategy

### 3.1 The Endogeneity Problem

{{ENDOGENEITY_DISCUSSION}}

### 3.2 Instrument Description

**Instrument(s)**: {{INSTRUMENTS}}

{{INSTRUMENT_DESCRIPTION}}

### 3.3 Identification Assumptions

#### Relevance (Testable)
{{RELEVANCE_ARGUMENT}}

**Empirical Evidence**: First-stage F-statistic = {{F_STAT}} ({{F_ASSESSMENT}})

#### Independence/Exogeneity (Untestable)
{{INDEPENDENCE_ARGUMENT}}

**Supporting Evidence**:
- Balance tests: {{BALANCE_RESULTS}}
- Placebo tests: {{PLACEBO_RESULTS}}

#### Exclusion Restriction (Untestable)
{{EXCLUSION_ARGUMENT}}

**Potential Threats**:
1. {{THREAT_1}}
2. {{THREAT_2}}

#### Monotonicity (for LATE interpretation)
{{MONOTONICITY_ARGUMENT}}

---

## 4. Estimation Results

### 4.1 First-Stage Results

The first-stage regression examines the relationship between the instrument(s) and the endogenous treatment:

$$
{{TREATMENT}} = \pi_0 + \pi_1 \cdot {{INSTRUMENT}} + X'\gamma + v
$$

| Variable | Coefficient | Std. Error | t-statistic | p-value |
|----------|-------------|------------|-------------|---------|
| {{INSTRUMENT_1}} | {{PI_1}} | {{PI_SE_1}} | {{PI_T_1}} | {{PI_P_1}} |
| {{INSTRUMENT_2}} | {{PI_2}} | {{PI_SE_2}} | {{PI_T_2}} | {{PI_P_2}} |

**First-Stage Diagnostics**:
- F-statistic: {{F_STAT}}
- Partial R-squared: {{PARTIAL_R2}}
- Stock-Yogo 10% critical value: {{STOCK_YOGO_CV}}

**Assessment**: {{FIRST_STAGE_ASSESSMENT}}

### 4.2 Reduced Form Results

The reduced form shows the direct relationship between instruments and outcome:

$$
{{OUTCOME}} = \theta_0 + \theta_1 \cdot {{INSTRUMENT}} + X'\lambda + u
$$

| Variable | Coefficient | Std. Error | p-value |
|----------|-------------|------------|---------|
| {{INSTRUMENT_1}} | {{THETA_1}} | {{THETA_SE_1}} | {{THETA_P_1}} |

**Note**: The ratio of reduced form to first stage = {{THETA_1}} / {{PI_1}} = {{WALD_ESTIMATE}}

### 4.3 IV Estimates

| Method | Estimate | Std. Error | 95% CI | p-value |
|--------|----------|------------|--------|---------|
| OLS | {{OLS_EST}} | {{OLS_SE}} | [{{OLS_CI_L}}, {{OLS_CI_U}}] | {{OLS_P}} |
| 2SLS | {{2SLS_EST}} | {{2SLS_SE}} | [{{2SLS_CI_L}}, {{2SLS_CI_U}}] | {{2SLS_P}} |
| LIML | {{LIML_EST}} | {{LIML_SE}} | [{{LIML_CI_L}}, {{LIML_CI_U}}] | {{LIML_P}} |

**Difference (IV - OLS)**: {{IV_OLS_DIFF}}

---

## 5. Diagnostic Tests

### 5.1 Weak Instruments

| Test | Statistic | Critical Value | Result |
|------|-----------|----------------|--------|
| First-stage F | {{F_STAT}} | 10 (rule of thumb) | {{F_RESULT}} |
| Stock-Yogo (10% bias) | {{F_STAT}} | {{SY_CV}} | {{SY_RESULT}} |

**Interpretation**: {{WEAK_IV_INTERPRETATION}}

### 5.2 Overidentification Test (if applicable)

| Test | Statistic | p-value | Result |
|------|-----------|---------|--------|
| Sargan-Hansen J | {{J_STAT}} | {{J_P}} | {{J_RESULT}} |

**Interpretation**: {{OVERID_INTERPRETATION}}

### 5.3 Endogeneity Test

| Test | Statistic | p-value | Result |
|------|-----------|---------|--------|
| Wu-Hausman | {{WH_STAT}} | {{WH_P}} | {{WH_RESULT}} |

**Interpretation**: {{ENDOGENEITY_INTERPRETATION}}

### 5.4 Estimator Comparison

| Comparison | Assessment |
|------------|------------|
| 2SLS vs LIML difference | {{2SLS_LIML_DIFF}} |
| OLS vs IV difference | {{OLS_IV_DIFF}} |
| Sign consistency | {{SIGN_CONSISTENT}} |

---

## 6. Robustness Checks

### 6.1 Alternative Specifications

| Specification | Estimate | Std. Error | Notes |
|---------------|----------|------------|-------|
| Baseline | {{BASELINE_EST}} | {{BASELINE_SE}} | Main specification |
| {{ROBUSTNESS_1_NAME}} | {{ROBUSTNESS_1_EST}} | {{ROBUSTNESS_1_SE}} | {{ROBUSTNESS_1_NOTES}} |
| {{ROBUSTNESS_2_NAME}} | {{ROBUSTNESS_2_EST}} | {{ROBUSTNESS_2_SE}} | {{ROBUSTNESS_2_NOTES}} |

### 6.2 Instrument Subsets (if multiple instruments)

| Instruments Used | Estimate | Std. Error |
|------------------|----------|------------|
| All instruments | {{ALL_EST}} | {{ALL_SE}} |
| {{SUBSET_1}} only | {{SUBSET_1_EST}} | {{SUBSET_1_SE}} |
| {{SUBSET_2}} only | {{SUBSET_2_EST}} | {{SUBSET_2_SE}} |

### 6.3 Weak-IV Robust Inference

| Method | 95% Confidence Interval |
|--------|-------------------------|
| Standard (Wald) | [{{WALD_CI_L}}, {{WALD_CI_U}}] |
| Anderson-Rubin | [{{AR_CI_L}}, {{AR_CI_U}}] |

---

## 7. Interpretation

### 7.1 LATE Interpretation

The IV estimate represents the **Local Average Treatment Effect (LATE)** for compliers - units whose treatment status was affected by the instrument(s).

**Complier Population**: {{COMPLIER_DESCRIPTION}}

**Share of Compliers**: Approximately {{COMPLIER_SHARE}} of the sample.

### 7.2 Comparison to OLS

| Estimator | Estimate | Interpretation |
|-----------|----------|----------------|
| OLS | {{OLS_EST}} | {{OLS_INTERPRETATION}} |
| IV | {{IV_EST}} | {{IV_INTERPRETATION}} |

**The difference suggests**: {{OLS_IV_DIFFERENCE_INTERPRETATION}}

### 7.3 Economic/Policy Significance

{{ECONOMIC_SIGNIFICANCE}}

---

## 8. Limitations and Caveats

### 8.1 Internal Validity Concerns

1. **Exclusion Restriction**: {{EXCLUSION_CONCERN}}
2. **Weak Instruments**: {{WEAK_IV_CONCERN}}
3. **Specification**: {{SPECIFICATION_CONCERN}}

### 8.2 External Validity

- **LATE vs ATE**: {{LATE_VS_ATE}}
- **Sample Representativeness**: {{REPRESENTATIVENESS}}
- **Temporal/Geographic Scope**: {{SCOPE}}

---

## 9. Conclusion

{{CONCLUSION}}

---

## Appendix

### A. Variable Construction

{{VARIABLE_CONSTRUCTION_DETAILS}}

### B. Sample Selection

{{SAMPLE_SELECTION_DETAILS}}

### C. Additional Tables and Figures

{{ADDITIONAL_MATERIALS}}

---

## References

1. Angrist, J. D., & Pischke, J. S. (2009). *Mostly Harmless Econometrics*. Princeton University Press.
2. Stock, J. H., & Yogo, M. (2005). Testing for Weak Instruments in Linear IV Regression.
3. {{ADDITIONAL_REFERENCES}}

---

*Report generated using the IV Estimator Skill from Causal ML Skills.*
