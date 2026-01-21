# Online Appendix: Robustness Checks

## {{PAPER_TITLE}}

**Authors:** {{AUTHORS}}
**Date:** {{DATE}}

---

## Table of Contents

1. [Alternative Specifications](#1-alternative-specifications)
2. [Sensitivity Analysis](#2-sensitivity-analysis)
3. [Placebo and Falsification Tests](#3-placebo-and-falsification-tests)
4. [Sample Robustness](#4-sample-robustness)
5. [Measurement Robustness](#5-measurement-robustness)
6. [Inference Robustness](#6-inference-robustness)
7. [Method-Specific Diagnostics](#7-method-specific-diagnostics)
8. [Additional Results](#8-additional-results)

---

## 1. Alternative Specifications

### 1.1 Functional Form

**Table A1: Alternative Functional Forms**

| Specification | Estimate | SE | 95% CI | Notes |
|---------------|----------|-----|--------|-------|
| Baseline (linear) | {{EST_LINEAR}} | {{SE_LINEAR}} | [{{CI_LINEAR}}] | Main specification |
| Quadratic | {{EST_QUAD}} | {{SE_QUAD}} | [{{CI_QUAD}}] | $Y = \beta_1 D + \beta_2 D^2$ |
| Log outcome | {{EST_LOG}} | {{SE_LOG}} | [{{CI_LOG}}] | Semi-elasticity |
| Inverse hyperbolic sine | {{EST_IHS}} | {{SE_IHS}} | [{{CI_IHS}}] | IHS transformation |
| Poisson | {{EST_POIS}} | {{SE_POIS}} | [{{CI_POIS}}] | Count outcome |

### 1.2 Control Variables

**Table A2: Sensitivity to Control Variables**

| Controls Included | Estimate | SE | Change from Baseline |
|-------------------|----------|-----|----------------------|
| No controls | {{EST_NOCTL}} | {{SE_NOCTL}} | {{CHG_NOCTL}} |
| Demographics only | {{EST_DEMO}} | {{SE_DEMO}} | {{CHG_DEMO}} |
| + Economic | {{EST_ECON}} | {{SE_ECON}} | {{CHG_ECON}} |
| + Geographic | {{EST_GEO}} | {{SE_GEO}} | {{CHG_GEO}} |
| + Pre-treatment outcomes | {{EST_PRE}} | {{SE_PRE}} | {{CHG_PRE}} |
| Full controls (baseline) | {{EST_FULL}} | {{SE_FULL}} | --- |
| + High-dimensional controls | {{EST_HDIM}} | {{SE_HDIM}} | {{CHG_HDIM}} |

### 1.3 Fixed Effects

**Table A3: Alternative Fixed Effects Structures**

| Fixed Effects | Estimate | SE | Within R² | N |
|--------------|----------|-----|-----------|---|
| None | {{EST_NOFE}} | {{SE_NOFE}} | {{R2_NOFE}} | {{N_NOFE}} |
| Unit FE | {{EST_UFE}} | {{SE_UFE}} | {{R2_UFE}} | {{N_UFE}} |
| Time FE | {{EST_TFE}} | {{SE_TFE}} | {{R2_TFE}} | {{N_TFE}} |
| Unit + Time (baseline) | {{EST_TWFE}} | {{SE_TWFE}} | {{R2_TWFE}} | {{N_TWFE}} |
| Unit + Time + Unit-trends | {{EST_TREND}} | {{SE_TREND}} | {{R2_TREND}} | {{N_TREND}} |
| Unit $\times$ Time | {{EST_INT}} | {{SE_INT}} | {{R2_INT}} | {{N_INT}} |

---

## 2. Sensitivity Analysis

### 2.1 Omitted Variable Bias

**Oster (2019) Bounds**

| Parameter | Value | Interpretation |
|-----------|-------|----------------|
| $\delta$ | {{DELTA}} | Proportional selection |
| $R^2_{max}$ | {{R2_MAX}} | Assumed maximum R² |
| Identified set | [{{BOUND_LO}}, {{BOUND_HI}}] | Range of consistent estimates |
| $\delta$ for $\beta = 0$ | {{DELTA_ZERO}} | Selection needed to nullify |

**Interpretation:** {{OSTER_INTERPRETATION}}

### 2.2 Coefficient Stability (Altonji et al., 2005)

**Table A4: Coefficient Stability Across Specifications**

| Controls Added | Estimate | Change | % of Baseline |
|----------------|----------|--------|---------------|
| Step 0: No controls | {{EST_S0}} | --- | {{PCT_S0}}% |
| Step 1: + Demographics | {{EST_S1}} | {{CHG_S1}} | {{PCT_S1}}% |
| Step 2: + Economic | {{EST_S2}} | {{CHG_S2}} | {{PCT_S2}}% |
| Step 3: + Geographic | {{EST_S3}} | {{CHG_S3}} | {{PCT_S3}}% |
| Step 4: Full (baseline) | {{EST_S4}} | {{CHG_S4}} | 100% |

**Stability Ratio:** {{STABILITY_RATIO}}
(Ratio of change from Step 3→4 to change from Step 0→4)

### 2.3 Rosenbaum Bounds (for Matching)

**Table A5: Sensitivity to Hidden Bias (Rosenbaum Bounds)**

| $\Gamma$ | Upper p-value | Lower p-value | Significant? |
|----------|---------------|---------------|--------------|
| 1.0 | {{P_G1_U}} | {{P_G1_L}} | {{SIG_G1}} |
| 1.1 | {{P_G11_U}} | {{P_G11_L}} | {{SIG_G11}} |
| 1.2 | {{P_G12_U}} | {{P_G12_L}} | {{SIG_G12}} |
| 1.3 | {{P_G13_U}} | {{P_G13_L}} | {{SIG_G13}} |
| 1.4 | {{P_G14_U}} | {{P_G14_L}} | {{SIG_G14}} |
| 1.5 | {{P_G15_U}} | {{P_G15_L}} | {{SIG_G15}} |

**Critical $\Gamma$:** {{GAMMA_CRIT}}

**Interpretation:** Results remain significant until hidden bias would need to increase treatment odds by {{GAMMA_CRIT_PCT}}%.

---

## 3. Placebo and Falsification Tests

### 3.1 Temporal Placebos

**Table A6: Placebo Tests in Time**

| Placebo | Estimate | SE | p-value | Expected |
|---------|----------|-----|---------|----------|
| Treatment at $t-3$ | {{EST_PT3}} | {{SE_PT3}} | {{P_PT3}} | 0 |
| Treatment at $t-2$ | {{EST_PT2}} | {{SE_PT2}} | {{P_PT2}} | 0 |
| Treatment at $t-1$ | {{EST_PT1}} | {{SE_PT1}} | {{P_PT1}} | 0 |
| Actual treatment | {{EST_ACTUAL}} | {{SE_ACTUAL}} | {{P_ACTUAL}} | $\neq 0$ |

**Joint test (all placebos = 0):** F = {{F_PLACEBO}}, p = {{P_PLACEBO_JOINT}}

### 3.2 Outcome Placebos

**Table A7: Placebo Outcomes**

| Outcome | Should be Affected? | Estimate | SE | p-value |
|---------|---------------------|----------|-----|---------|
| {{PLACEBO_OUT_1}} | No | {{EST_PO1}} | {{SE_PO1}} | {{P_PO1}} |
| {{PLACEBO_OUT_2}} | No | {{EST_PO2}} | {{SE_PO2}} | {{P_PO2}} |
| {{PLACEBO_OUT_3}} | No | {{EST_PO3}} | {{SE_PO3}} | {{P_PO3}} |
| {{ACTUAL_OUT}} (baseline) | Yes | {{EST_ACTUAL_OUT}} | {{SE_ACTUAL_OUT}} | {{P_ACTUAL_OUT}} |

### 3.3 Placebo Treatment

**Table A8: Placebo Treatment Groups**

| Placebo Treatment | Estimate | SE | p-value | Notes |
|-------------------|----------|-----|---------|-------|
| Randomized placebo | {{EST_RAND}} | {{SE_RAND}} | {{P_RAND}} | Random assignment |
| Neighboring units | {{EST_NEIGHBOR}} | {{SE_NEIGHBOR}} | {{P_NEIGHBOR}} | Spillover test |
| Same industry/region | {{EST_IND}} | {{SE_IND}} | {{P_IND}} | Peer effects |

### 3.4 Pre-Treatment Effects (Event Study)

**Table A9: Pre-Treatment Coefficients**

| Period | Coefficient | SE | 95% CI | p-value |
|--------|-------------|-----|--------|---------|
| $t-5$ | {{COEF_M5}} | {{SE_M5}} | [{{CI_M5}}] | {{P_M5}} |
| $t-4$ | {{COEF_M4}} | {{SE_M4}} | [{{CI_M4}}] | {{P_M4}} |
| $t-3$ | {{COEF_M3}} | {{SE_M3}} | [{{CI_M3}}] | {{P_M3}} |
| $t-2$ | {{COEF_M2}} | {{SE_M2}} | [{{CI_M2}}] | {{P_M2}} |
| $t-1$ | --- | --- | --- | --- |

**Joint F-test (pre-trends = 0):** F({{DF_PRE}}) = {{F_PRE}}, p = {{P_PRE}}

---

## 4. Sample Robustness

### 4.1 Sample Restrictions

**Table A10: Sample Sensitivity**

| Sample | N | Estimate | SE | Change |
|--------|---|----------|-----|--------|
| Full sample (baseline) | {{N_FULL}} | {{EST_FULL}} | {{SE_FULL}} | --- |
| Drop outliers (1%) | {{N_OUT1}} | {{EST_OUT1}} | {{SE_OUT1}} | {{CHG_OUT1}} |
| Drop outliers (5%) | {{N_OUT5}} | {{EST_OUT5}} | {{SE_OUT5}} | {{CHG_OUT5}} |
| Balanced panel only | {{N_BAL}} | {{EST_BAL}} | {{SE_BAL}} | {{CHG_BAL}} |
| Exclude {{EXCLUDE_1}} | {{N_EX1}} | {{EST_EX1}} | {{SE_EX1}} | {{CHG_EX1}} |
| Exclude {{EXCLUDE_2}} | {{N_EX2}} | {{EST_EX2}} | {{SE_EX2}} | {{CHG_EX2}} |

### 4.2 Time Period Sensitivity

**Table A11: Sensitivity to Sample Period**

| Period | N | Estimate | SE | Notes |
|--------|---|----------|-----|-------|
| Full period (baseline) | {{N_FULL_T}} | {{EST_FULL_T}} | {{SE_FULL_T}} | {{PERIOD_FULL}} |
| Pre-crisis | {{N_PRE}} | {{EST_PRE_T}} | {{SE_PRE_T}} | {{PERIOD_PRE}} |
| Post-crisis | {{N_POST}} | {{EST_POST_T}} | {{SE_POST_T}} | {{PERIOD_POST}} |
| Drop first year | {{N_DF}} | {{EST_DF}} | {{SE_DF}} | Exclude {{FIRST_YEAR}} |
| Drop last year | {{N_DL}} | {{EST_DL}} | {{SE_DL}} | Exclude {{LAST_YEAR}} |

### 4.3 Leave-One-Out Analysis

**Figure A1: Leave-One-Out by Unit/Region**

{{FIGURE_LOO_DESCRIPTION}}

| Excluded | Estimate | 95% CI | Significant? |
|----------|----------|--------|--------------|
| {{LOO_1}} | {{EST_LOO1}} | [{{CI_LOO1}}] | {{SIG_LOO1}} |
| {{LOO_2}} | {{EST_LOO2}} | [{{CI_LOO2}}] | {{SIG_LOO2}} |
| {{LOO_3}} | {{EST_LOO3}} | [{{CI_LOO3}}] | {{SIG_LOO3}} |
| ... | ... | ... | ... |

**Range of LOO estimates:** [{{LOO_MIN}}, {{LOO_MAX}}]

---

## 5. Measurement Robustness

### 5.1 Alternative Outcome Measures

**Table A12: Alternative Outcome Definitions**

| Outcome Definition | Estimate | SE | Correlation with Baseline |
|--------------------|----------|-----|--------------------------|
| Baseline | {{EST_OUT_BASE}} | {{SE_OUT_BASE}} | 1.00 |
| {{ALT_OUT_1}} | {{EST_ALT1}} | {{SE_ALT1}} | {{CORR_ALT1}} |
| {{ALT_OUT_2}} | {{EST_ALT2}} | {{SE_ALT2}} | {{CORR_ALT2}} |
| Standardized | {{EST_STD}} | {{SE_STD}} | {{CORR_STD}} |
| Winsorized (1%) | {{EST_WINS}} | {{SE_WINS}} | {{CORR_WINS}} |

### 5.2 Alternative Treatment Measures

**Table A13: Alternative Treatment Definitions**

| Treatment Definition | Estimate | SE | Notes |
|----------------------|----------|-----|-------|
| Binary (baseline) | {{EST_TRT_BASE}} | {{SE_TRT_BASE}} | $D \in \{0,1\}$ |
| Continuous intensity | {{EST_TRT_CONT}} | {{SE_TRT_CONT}} | Dose-response |
| Terciles | {{EST_TRT_TERC}} | {{SE_TRT_TERC}} | Low/Med/High |
| Alternative threshold | {{EST_TRT_THRESH}} | {{SE_TRT_THRESH}} | Threshold = {{ALT_THRESH}} |

### 5.3 Index Construction

**Table A14: Alternative Index Construction**

| Index Method | Estimate | SE | Components |
|--------------|----------|-----|------------|
| Simple average | {{EST_IDX_AVG}} | {{SE_IDX_AVG}} | Equal weights |
| PCA first factor | {{EST_IDX_PCA}} | {{SE_IDX_PCA}} | {{PCA_VAR}}% variance |
| Anderson (2008) | {{EST_IDX_AND}} | {{SE_IDX_AND}} | GLS weighted |
| Individual components | --- | --- | See Panel B |

---

## 6. Inference Robustness

### 6.1 Standard Error Specifications

**Table A15: Alternative Standard Errors**

| SE Specification | SE | p-value | 95% CI |
|------------------|-----|---------|--------|
| Conventional | {{SE_CONV}} | {{P_CONV}} | [{{CI_CONV}}] |
| Heteroskedasticity-robust | {{SE_ROBUST}} | {{P_ROBUST}} | [{{CI_ROBUST}}] |
| Clustered (unit) | {{SE_CLUST_U}} | {{P_CLUST_U}} | [{{CI_CLUST_U}}] |
| Clustered (time) | {{SE_CLUST_T}} | {{P_CLUST_T}} | [{{CI_CLUST_T}}] |
| Two-way clustered | {{SE_CLUST_2}} | {{P_CLUST_2}} | [{{CI_CLUST_2}}] |
| Conley spatial HAC | {{SE_CONLEY}} | {{P_CONLEY}} | [{{CI_CONLEY}}] |
| Bootstrap (1000) | {{SE_BOOT}} | {{P_BOOT}} | [{{CI_BOOT}}] |
| Wild cluster bootstrap | {{SE_WILD}} | {{P_WILD}} | [{{CI_WILD}}] |

### 6.2 Finite Sample Corrections

**Table A16: Small Sample Adjustments**

| Adjustment | t-statistic | p-value | Critical Value |
|------------|-------------|---------|----------------|
| Normal approximation | {{T_NORM}} | {{P_NORM}} | 1.96 |
| t-distribution (N-k) | {{T_T}} | {{P_T}} | {{CV_T}} |
| Effective clusters | {{T_EFF}} | {{P_EFF}} | {{CV_EFF}} |

**Number of clusters:** {{N_CLUSTERS}}
**Effective degrees of freedom:** {{EFF_DF}}

### 6.3 Multiple Hypothesis Testing

**Table A17: Multiple Testing Corrections**

| Outcome | Unadjusted p | Bonferroni | Holm | BH (FDR) | Westfall-Young |
|---------|--------------|------------|------|----------|----------------|
| {{OUT_1}} | {{P_1}} | {{P_1_BONF}} | {{P_1_HOLM}} | {{P_1_BH}} | {{P_1_WY}} |
| {{OUT_2}} | {{P_2}} | {{P_2_BONF}} | {{P_2_HOLM}} | {{P_2_BH}} | {{P_2_WY}} |
| {{OUT_3}} | {{P_3}} | {{P_3_BONF}} | {{P_3_HOLM}} | {{P_3_BH}} | {{P_3_WY}} |

---

## 7. Method-Specific Diagnostics

### 7.1 Difference-in-Differences

**Parallel Trends Assessment**

| Test | Statistic | p-value | Interpretation |
|------|-----------|---------|----------------|
| Joint pre-trend F-test | {{F_PRETREND}} | {{P_PRETREND}} | {{INTERP_PRETREND}} |
| Rambachan-Roth sensitivity | {{RR_MBAR}} | --- | Robust to $\bar{M}$ = {{MBAR}} |

**Staggered DID Robustness**

| Estimator | ATT | SE | Notes |
|-----------|-----|-----|-------|
| TWFE (baseline) | {{ATT_TWFE}} | {{SE_TWFE}} | May be biased |
| Callaway-Sant'Anna | {{ATT_CS}} | {{SE_CS}} | Never-treated control |
| Sun-Abraham | {{ATT_SA}} | {{SE_SA}} | Interaction-weighted |
| Borusyak et al. | {{ATT_BJS}} | {{SE_BJS}} | Imputation |

### 7.2 Instrumental Variables

**First Stage Strength**

| Diagnostic | Statistic | Threshold | Pass? |
|------------|-----------|-----------|-------|
| First-stage F | {{F_FIRST}} | > 10 | {{PASS_F}} |
| Kleibergen-Paap F | {{KP_F}} | > 10 | {{PASS_KP}} |
| Anderson-Rubin | {{AR_STAT}} | p < 0.05 | {{PASS_AR}} |

**Overidentification**

| Test | Statistic | p-value | Interpretation |
|------|-----------|---------|----------------|
| Sargan | {{SARGAN}} | {{P_SARGAN}} | {{INTERP_SARGAN}} |
| Hansen J | {{HANSEN}} | {{P_HANSEN}} | {{INTERP_HANSEN}} |

**IV Robustness**

| Estimator | Estimate | SE | Notes |
|-----------|----------|-----|-------|
| 2SLS (baseline) | {{EST_2SLS}} | {{SE_2SLS}} | Main |
| LIML | {{EST_LIML}} | {{SE_LIML}} | Less biased with weak IV |
| Fuller(1) | {{EST_FULL1}} | {{SE_FULL1}} | Median-unbiased |
| JIVE | {{EST_JIVE}} | {{SE_JIVE}} | Jackknife |
| Reduced form | {{EST_RF}} | {{SE_RF}} | Intent-to-treat |

### 7.3 Regression Discontinuity

**Bandwidth Sensitivity**

| Bandwidth | Estimate | SE | N (left) | N (right) |
|-----------|----------|-----|----------|-----------|
| 0.5 $\times$ optimal | {{EST_BW05}} | {{SE_BW05}} | {{NL_05}} | {{NR_05}} |
| 0.75 $\times$ optimal | {{EST_BW075}} | {{SE_BW075}} | {{NL_075}} | {{NR_075}} |
| Optimal (baseline) | {{EST_BW1}} | {{SE_BW1}} | {{NL_1}} | {{NR_1}} |
| 1.5 $\times$ optimal | {{EST_BW15}} | {{SE_BW15}} | {{NL_15}} | {{NR_15}} |
| 2 $\times$ optimal | {{EST_BW2}} | {{SE_BW2}} | {{NL_2}} | {{NR_2}} |

**Polynomial Order**

| Order | Estimate | SE | AIC |
|-------|----------|-----|-----|
| Local linear (baseline) | {{EST_P1}} | {{SE_P1}} | {{AIC_P1}} |
| Local quadratic | {{EST_P2}} | {{SE_P2}} | {{AIC_P2}} |
| Local cubic | {{EST_P3}} | {{SE_P3}} | {{AIC_P3}} |

**Manipulation Tests**

| Test | Statistic | p-value | Interpretation |
|------|-----------|---------|----------------|
| McCrary density | {{MCCRARY}} | {{P_MCCRARY}} | {{INTERP_MCCRARY}} |
| Cattaneo et al. | {{CJM}} | {{P_CJM}} | {{INTERP_CJM}} |

### 7.4 Propensity Score Matching

**Matching Method Comparison**

| Method | ATT | SE | Matched Control N | Balance |
|--------|-----|-----|-------------------|---------|
| Nearest neighbor (1:1) | {{ATT_NN1}} | {{SE_NN1}} | {{N_NN1}} | {{BAL_NN1}} |
| Nearest neighbor (1:4) | {{ATT_NN4}} | {{SE_NN4}} | {{N_NN4}} | {{BAL_NN4}} |
| Caliper (0.1) | {{ATT_CAL1}} | {{SE_CAL1}} | {{N_CAL1}} | {{BAL_CAL1}} |
| Caliper (0.05) | {{ATT_CAL05}} | {{SE_CAL05}} | {{N_CAL05}} | {{BAL_CAL05}} |
| Kernel | {{ATT_KERN}} | {{SE_KERN}} | {{N_KERN}} | {{BAL_KERN}} |
| IPW | {{ATT_IPW}} | {{SE_IPW}} | {{N_IPW}} | {{BAL_IPW}} |
| Doubly robust | {{ATT_DR}} | {{SE_DR}} | {{N_DR}} | {{BAL_DR}} |

---

## 8. Additional Results

### 8.1 Mechanism Analysis

**Table A18: Mechanisms**

| Mechanism | Estimate | SE | % of Total |
|-----------|----------|-----|------------|
| Total effect | {{EST_TOTAL}} | {{SE_TOTAL}} | 100% |
| Through {{MECH_1}} | {{EST_M1}} | {{SE_M1}} | {{PCT_M1}}% |
| Through {{MECH_2}} | {{EST_M2}} | {{SE_M2}} | {{PCT_M2}}% |
| Direct effect | {{EST_DIRECT}} | {{SE_DIRECT}} | {{PCT_DIRECT}}% |

### 8.2 Heterogeneous Effects

**Table A19: Extended Heterogeneity Analysis**

| Subgroup | N | Estimate | SE | p (vs. baseline) |
|----------|---|----------|-----|------------------|
| Full sample | {{N_ALL}} | {{EST_ALL}} | {{SE_ALL}} | --- |
| {{SUBGROUP_1}} | {{N_SG1}} | {{EST_SG1}} | {{SE_SG1}} | {{P_SG1}} |
| {{SUBGROUP_2}} | {{N_SG2}} | {{EST_SG2}} | {{SE_SG2}} | {{P_SG2}} |
| {{SUBGROUP_3}} | {{N_SG3}} | {{EST_SG3}} | {{SE_SG3}} | {{P_SG3}} |
| {{SUBGROUP_4}} | {{N_SG4}} | {{EST_SG4}} | {{SE_SG4}} | {{P_SG4}} |

### 8.3 Spillover Analysis

**Table A20: Spillover Effects**

| Distance/Relationship | Estimate | SE | p-value |
|-----------------------|----------|-----|---------|
| Direct treatment | {{EST_DIRECT_SP}} | {{SE_DIRECT_SP}} | {{P_DIRECT_SP}} |
| {{SPILLOVER_1}} | {{EST_SP1}} | {{SE_SP1}} | {{P_SP1}} |
| {{SPILLOVER_2}} | {{EST_SP2}} | {{SE_SP2}} | {{P_SP2}} |
| {{SPILLOVER_3}} | {{EST_SP3}} | {{SE_SP3}} | {{P_SP3}} |

---

## Summary of Robustness

**All specifications produce estimates within [{{ROBUST_MIN}}, {{ROBUST_MAX}}] of the baseline.**

| Category | # Checks | # Consistent | Range |
|----------|----------|--------------|-------|
| Specifications | {{N_SPEC}} | {{N_CONSIST_SPEC}} | [{{RANGE_SPEC}}] |
| Sample | {{N_SAMP}} | {{N_CONSIST_SAMP}} | [{{RANGE_SAMP}}] |
| Inference | {{N_INF}} | {{N_CONSIST_INF}} | [{{RANGE_INF}}] |
| Placebo | {{N_PLAC}} | {{N_NULL_PLAC}} | Expected null |

---

*Appendix generated using the Causal ML Skills framework.*
*Template version: 1.0.0*

---

## Template Usage Instructions

### Required Sections

Not all sections will be relevant for every study. Include only:

- **DID**: Sections 3.4, 7.1
- **IV**: Sections 7.2
- **RD**: Sections 7.3
- **PSM**: Sections 7.4

### Placeholder Naming Convention

- `EST_*`: Point estimates
- `SE_*`: Standard errors
- `P_*`: p-values
- `CI_*`: Confidence intervals
- `N_*`: Sample sizes
- `CHG_*`: Changes from baseline

### Best Practices

1. **Report all specifications** tested, not just favorable ones
2. **Pre-register** which robustness checks will be performed
3. **Discuss** any discrepant results honestly
4. **Quantify** the range of estimates across specifications
5. **Interpret** economic significance, not just statistical
