# Difference-in-Differences Analysis Report

> **Project**: {{PROJECT_NAME}}
> **Analyst**: {{ANALYST_NAME}}
> **Date**: {{DATE}}
> **Version**: 1.0

---

## Executive Summary

This report presents the results of a difference-in-differences (DID) analysis examining the effect of {{TREATMENT_DESCRIPTION}} on {{OUTCOME_DESCRIPTION}}. The analysis uses {{DATA_DESCRIPTION}} covering {{N_UNITS}} units over {{N_PERIODS}} time periods ({{PERIOD_START}} to {{PERIOD_END}}).

**Key Findings**:
- The estimated treatment effect is **{{EFFECT}}** (SE = {{SE}}), representing a **{{PERCENT_CHANGE}}%** change from the pre-treatment mean
- This effect is {{SIGNIFICANCE_STATEMENT}}
- The parallel trends assumption is {{TRENDS_ASSESSMENT}}
- Results are {{ROBUSTNESS_ASSESSMENT}}

---

## 1. Introduction

### 1.1 Research Question

{{RESEARCH_QUESTION}}

### 1.2 Treatment and Timing

- **Treatment**: {{TREATMENT_DEFINITION}}
- **Treatment Timing**: {{TREATMENT_TIMING}}
- **Treated Units**: {{TREATED_UNITS_DESCRIPTION}}
- **Control Units**: {{CONTROL_UNITS_DESCRIPTION}}

### 1.3 Identification Strategy

We employ a difference-in-differences design that compares changes in {{OUTCOME}} between treated and control units before and after treatment implementation. This approach identifies the **Average Treatment Effect on the Treated (ATT)** under the assumption that, absent treatment, both groups would have followed parallel outcome trajectories.

---

## 2. Data and Sample

### 2.1 Data Sources

{{DATA_SOURCE_DESCRIPTION}}

### 2.2 Sample Construction

| Criterion | Observations Dropped | Remaining |
|-----------|---------------------|-----------|
| Initial sample | --- | {{N_INITIAL}} |
| {{RESTRICTION_1}} | {{DROP_1}} | {{REMAIN_1}} |
| {{RESTRICTION_2}} | {{DROP_2}} | {{REMAIN_2}} |
| Missing values | {{DROP_MISSING}} | {{REMAIN_FINAL}} |
| **Final sample** | --- | **{{N_FINAL}}** |

### 2.3 Variable Definitions

| Variable | Definition | Source |
|----------|------------|--------|
| {{OUTCOME_VAR}} | {{OUTCOME_DEFINITION}} | {{OUTCOME_SOURCE}} |
| {{TREATMENT_VAR}} | {{TREATMENT_VAR_DEFINITION}} | {{TREATMENT_SOURCE}} |
| {{CONTROL_1}} | {{CONTROL_1_DEFINITION}} | {{CONTROL_1_SOURCE}} |
| {{CONTROL_2}} | {{CONTROL_2_DEFINITION}} | {{CONTROL_2_SOURCE}} |

### 2.4 Summary Statistics

| Variable | N | Mean | Std. Dev. | Min | Max |
|----------|---|------|-----------|-----|-----|
| {{OUTCOME_VAR}} | {{N}} | {{MEAN_Y}} | {{SD_Y}} | {{MIN_Y}} | {{MAX_Y}} |
| {{CONTROL_1}} | {{N}} | {{MEAN_X1}} | {{SD_X1}} | {{MIN_X1}} | {{MAX_X1}} |
| {{CONTROL_2}} | {{N}} | {{MEAN_X2}} | {{SD_X2}} | {{MIN_X2}} | {{MAX_X2}} |

---

## 3. Empirical Strategy

### 3.1 Model Specification

We estimate the following two-way fixed effects model:

$$
Y_{it} = \alpha_i + \gamma_t + \delta \cdot D_{it} + X_{it}'\beta + \varepsilon_{it}
$$

Where:
- $Y_{it}$ = {{OUTCOME_DESCRIPTION}} for unit $i$ at time $t$
- $\alpha_i$ = unit fixed effects
- $\gamma_t$ = time fixed effects
- $D_{it}$ = treatment indicator (= 1 if unit $i$ is treated at time $t$)
- $X_{it}$ = control variables ({{CONTROLS_LIST}})
- $\delta$ = **parameter of interest** (ATT under parallel trends)

### 3.2 Identification Assumptions

1. **Parallel Trends**: In the absence of treatment, treated and control units would have experienced the same change in outcomes over time

2. **No Anticipation**: Units do not change behavior in anticipation of treatment

3. **SUTVA**: Treatment of one unit does not affect outcomes of other units

4. **No Composition Changes**: Sample composition is stable over time

### 3.3 Standard Errors

Standard errors are clustered at the {{CLUSTER_LEVEL}} level to account for serial correlation within {{CLUSTER_UNITS}}. There are {{N_CLUSTERS}} clusters in the sample.

---

## 4. Results

### 4.1 Parallel Trends Assessment

#### Visual Inspection

![Parallel Trends](figures/parallel_trends.png)

*Figure 1: Pre-treatment outcome trends for treatment and control groups. Shaded areas represent 95% confidence intervals.*

#### Statistical Tests

| Test | Statistic | P-value | Assessment |
|------|-----------|---------|------------|
| Linear trend divergence | {{TREND_STAT}} | {{TREND_PVAL}} | {{TREND_ASSESS}} |
| Joint F-test (pre-coefficients) | {{F_STAT}} | {{F_PVAL}} | {{F_ASSESS}} |
| Individual pre-period t-2 | {{T_M2}} | {{P_M2}} | {{ASSESS_M2}} |
| Individual pre-period t-3 | {{T_M3}} | {{P_M3}} | {{ASSESS_M3}} |

**Interpretation**: {{PARALLEL_TRENDS_INTERPRETATION}}

### 4.2 Main Results

#### Table 1: Difference-in-Differences Estimates

|  | (1) | (2) | (3) | (4) |
|--|-----|-----|-----|-----|
| **Treatment Effect** | **{{EFFECT_1}}{{STARS_1}}** | **{{EFFECT_2}}{{STARS_2}}** | **{{EFFECT_3}}{{STARS_3}}** | **{{EFFECT_4}}{{STARS_4}}** |
|  | ({{SE_1}}) | ({{SE_2}}) | ({{SE_3}}) | ({{SE_4}}) |
| Controls | No | Yes | Yes | Yes |
| Unit FE | Yes | Yes | Yes | Yes |
| Time FE | Yes | Yes | Yes | Yes |
| Trends | No | No | Yes | No |
| Estimator | TWFE | TWFE | TWFE | C-S |
| Observations | {{N_1}} | {{N_2}} | {{N_3}} | {{N_4}} |
| R-squared | {{R2_1}} | {{R2_2}} | {{R2_3}} | --- |

*Notes: Robust standard errors clustered at {{CLUSTER_LEVEL}} level in parentheses. \*\*\* p<0.01, \*\* p<0.05, \* p<0.1.*

**Interpretation**: {{MAIN_RESULTS_INTERPRETATION}}

### 4.3 Event Study

![Event Study](figures/event_study.png)

*Figure 2: Event study coefficients with 95% confidence intervals. Coefficients normalized to zero at t-1.*

**Pre-treatment Period (t < 0)**:
- All pre-treatment coefficients are {{PRE_COEF_ASSESSMENT}}
- Joint F-test: F = {{F_PRE}}, p = {{P_PRE}}
- {{PRE_INTERPRETATION}}

**Post-treatment Period (t >= 0)**:
- Treatment effect emerges {{TIMING_EMERGENCE}}
- Effect {{DYNAMIC_PATTERN}} over time
- By period t+{{FINAL_PERIOD}}, effect is {{FINAL_EFFECT}} (SE = {{FINAL_SE}})

### 4.4 Economic Significance

| Metric | Value |
|--------|-------|
| Point estimate | {{EFFECT}} |
| Pre-treatment mean (treated) | {{PRETX_MEAN}} |
| Percent change | {{PERCENT_CHANGE}}% |
| Elasticity (if applicable) | {{ELASTICITY}} |
| Cost per unit outcome | {{COST_EFFECTIVENESS}} |

**Interpretation**: {{ECONOMIC_SIGNIFICANCE_INTERPRETATION}}

---

## 5. Robustness Checks

### 5.1 Placebo Tests

#### Placebo Timing

| Placebo Year | Effect | SE | P-value | Assessment |
|--------------|--------|----|---------| -----------|
| {{PLACEBO_1}} | {{PLAC_EFF_1}} | {{PLAC_SE_1}} | {{PLAC_P_1}} | {{PLAC_ASSESS_1}} |
| {{PLACEBO_2}} | {{PLAC_EFF_2}} | {{PLAC_SE_2}} | {{PLAC_P_2}} | {{PLAC_ASSESS_2}} |

#### Placebo Outcomes

| Outcome | Effect | SE | P-value | Assessment |
|---------|--------|----|---------| -----------|
| {{PLAC_OUT_1}} | {{PLAC_OUT_EFF_1}} | {{PLAC_OUT_SE_1}} | {{PLAC_OUT_P_1}} | {{PLAC_OUT_ASSESS_1}} |
| {{PLAC_OUT_2}} | {{PLAC_OUT_EFF_2}} | {{PLAC_OUT_SE_2}} | {{PLAC_OUT_P_2}} | {{PLAC_OUT_ASSESS_2}} |

### 5.2 Alternative Specifications

| Specification | Effect | SE | 95% CI |
|---------------|--------|----| -------|
| Baseline | {{BASELINE_EFF}} | {{BASELINE_SE}} | [{{BASELINE_CI_L}}, {{BASELINE_CI_U}}] |
| No controls | {{NOCTL_EFF}} | {{NOCTL_SE}} | [{{NOCTL_CI_L}}, {{NOCTL_CI_U}}] |
| Extended controls | {{EXTCTL_EFF}} | {{EXTCTL_SE}} | [{{EXTCTL_CI_L}}, {{EXTCTL_CI_U}}] |
| Log outcome | {{LOG_EFF}} | {{LOG_SE}} | [{{LOG_CI_L}}, {{LOG_CI_U}}] |
| Balanced panel only | {{BAL_EFF}} | {{BAL_SE}} | [{{BAL_CI_L}}, {{BAL_CI_U}}] |

### 5.3 Inference Robustness

| Method | SE | 95% CI | P-value |
|--------|----| -------|---------|
| Clustered ({{CLUSTER_LEVEL}}) | {{CLUSTER_SE}} | [{{CLUSTER_CI_L}}, {{CLUSTER_CI_U}}] | {{CLUSTER_P}} |
| Clustered ({{ALT_CLUSTER}}) | {{ALT_CLUSTER_SE}} | [{{ALT_CLUSTER_CI_L}}, {{ALT_CLUSTER_CI_U}}] | {{ALT_CLUSTER_P}} |
| Wild bootstrap | {{BOOT_SE}} | [{{BOOT_CI_L}}, {{BOOT_CI_U}}] | {{BOOT_P}} |
| Randomization inference | --- | --- | {{RI_P}} |

### 5.4 Goodman-Bacon Decomposition

| Comparison | Weight | Estimate |
|------------|--------|----------|
| Earlier vs. Later Treated | {{W_EL}} | {{E_EL}} |
| Later vs. Earlier Treated | {{W_LE}} | {{E_LE}} |
| Treated vs. Never Treated | {{W_TN}} | {{E_TN}} |
| **TWFE Estimate** | --- | **{{TWFE_EST}}** |
| **Callaway-Sant'Anna** | --- | **{{CS_EST}}** |

**Assessment**: {{BACON_ASSESSMENT}}

### 5.5 Robustness Summary

| Check | Result | Assessment |
|-------|--------|------------|
| Parallel trends test | p = {{TREND_P}} | {{TREND_CHECK}} |
| Placebo timing | {{PLAC_RESULT}} | {{PLAC_CHECK}} |
| Placebo outcomes | {{PLAC_OUT_RESULT}} | {{PLAC_OUT_CHECK}} |
| Alternative SEs | {{ALT_SE_RESULT}} | {{ALT_SE_CHECK}} |
| Bacon decomposition | Weight = {{BAD_WEIGHT}} | {{BACON_CHECK}} |
| **Overall** | | **{{OVERALL_ROBUST}}** |

---

## 6. Heterogeneity Analysis

### 6.1 By Subgroup

| Subgroup | N | Effect | SE | P-value |
|----------|---|--------|----| --------|
| {{SUBGROUP_1}} | {{N_SUB_1}} | {{EFF_SUB_1}} | {{SE_SUB_1}} | {{P_SUB_1}} |
| {{SUBGROUP_2}} | {{N_SUB_2}} | {{EFF_SUB_2}} | {{SE_SUB_2}} | {{P_SUB_2}} |
| {{SUBGROUP_3}} | {{N_SUB_3}} | {{EFF_SUB_3}} | {{SE_SUB_3}} | {{P_SUB_3}} |
| Test for heterogeneity | | p = {{P_HET}} | | |

### 6.2 By Treatment Cohort

| Cohort | N | Effect | SE |
|--------|---|--------|----|
| {{COHORT_1}} | {{N_COH_1}} | {{EFF_COH_1}} | {{SE_COH_1}} |
| {{COHORT_2}} | {{N_COH_2}} | {{EFF_COH_2}} | {{SE_COH_2}} |
| {{COHORT_3}} | {{N_COH_3}} | {{EFF_COH_3}} | {{SE_COH_3}} |

---

## 7. Discussion

### 7.1 Summary of Findings

{{FINDINGS_SUMMARY}}

### 7.2 Interpretation

{{INTERPRETATION_DISCUSSION}}

### 7.3 Mechanisms

{{MECHANISMS_DISCUSSION}}

### 7.4 Limitations

1. **Parallel Trends**: {{LIMITATION_TRENDS}}
2. **External Validity**: {{LIMITATION_EXTERNAL}}
3. **SUTVA**: {{LIMITATION_SUTVA}}
4. **Data Limitations**: {{LIMITATION_DATA}}

### 7.5 Policy Implications

{{POLICY_IMPLICATIONS}}

---

## 8. Conclusion

{{CONCLUSION}}

---

## References

1. Angrist, J. D., & Pischke, J. S. (2009). *Mostly Harmless Econometrics*. Princeton University Press.

2. Callaway, B., & Sant'Anna, P. H. (2021). Difference-in-Differences with Multiple Time Periods. *Journal of Econometrics*, 225(2), 200-230.

3. Goodman-Bacon, A. (2021). Difference-in-Differences with Variation in Treatment Timing. *Journal of Econometrics*, 225(2), 254-277.

4. Roth, J. (2022). Pretest with Caution: Event-Study Estimates after Testing for Parallel Trends. *American Economic Review: Insights*, 4(3), 305-322.

{{ADDITIONAL_REFERENCES}}

---

## Appendix

### A.1 Additional Results Tables

{{APPENDIX_TABLES}}

### A.2 Additional Figures

{{APPENDIX_FIGURES}}

### A.3 Variable Construction Details

{{VARIABLE_DETAILS}}

### A.4 Data Cleaning Procedures

{{DATA_CLEANING}}

---

*Report generated using Causal ML Skills - DID Estimator*
*Last updated: {{LAST_UPDATED}}*
