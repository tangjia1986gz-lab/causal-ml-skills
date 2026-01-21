# Regression Discontinuity Analysis Report

> **Template Version**: 1.0.0
> **Generated**: [DATE]
> **Analyst**: [ANALYST_NAME]

---

## Executive Summary

[2-3 sentence summary of the research question, design, and key finding]

**Key Result**: [Treatment description] causes a [EFFECT_DIRECTION] of **[EFFECT] [UNITS]** (95% CI: [[CI_LOWER], [CI_UPPER]]) in [outcome description] for individuals at the margin of the [cutoff description].

---

## 1. Research Design

### 1.1 Research Question

[What causal question does this RD address?]

### 1.2 Setting and Treatment

| Component | Description |
|-----------|-------------|
| **Treatment** | [What is the treatment/intervention?] |
| **Running Variable** | [What determines treatment assignment?] |
| **Cutoff Value** | [VALUE] |
| **Design Type** | [Sharp / Fuzzy] RD |
| **Sample Period** | [Time period of data] |

### 1.3 Identification Strategy

Treatment is assigned based on whether the running variable [RUNNING_DESCRIPTION] crosses the threshold of [CUTOFF]. We estimate the causal effect of treatment by comparing outcomes for units just above versus just below this cutoff, where treatment assignment is effectively random.

**Key Assumptions**:
1. Continuity of potential outcomes at the cutoff
2. No precise manipulation of the running variable
3. No other discontinuities at the same threshold

---

## 2. Data and Sample

### 2.1 Data Source

[Describe data source, collection, and any preprocessing]

### 2.2 Sample Construction

| Statistic | Value |
|-----------|-------|
| Total Observations | [N_TOTAL] |
| Below Cutoff | [N_BELOW] |
| Above Cutoff | [N_ABOVE] |
| Effective N (within bandwidth) | [N_EFF] |
| Sample Period | [PERIOD] |

### 2.3 Variable Definitions

| Variable | Definition | Source |
|----------|------------|--------|
| Running Variable | [RUNNING_DEF] | [SOURCE] |
| Outcome | [OUTCOME_DEF] | [SOURCE] |
| Treatment (if Fuzzy) | [TREATMENT_DEF] | [SOURCE] |

---

## 3. Design Validity

### 3.1 Manipulation Test (McCrary)

The McCrary density test checks for discontinuities in the density of the running variable at the cutoff, which would suggest manipulation.

| Statistic | Value |
|-----------|-------|
| Log Density Difference | [MCCRARY_STAT] |
| P-value | [MCCRARY_PVAL] |
| **Result** | **[PASSED/FAILED]** |

**Interpretation**: [Interpretation of McCrary test result]

[INSERT HISTOGRAM OF RUNNING VARIABLE]

### 3.2 Covariate Balance

Pre-treatment covariates should be balanced at the cutoff. Significant discontinuities suggest sorting or confounding.

| Covariate | Mean Below | Mean Above | Difference | P-value | Status |
|-----------|------------|------------|------------|---------|--------|
| [COV_1] | [MEAN_B_1] | [MEAN_A_1] | [DIFF_1] | [PVAL_1] | [OK/FAIL] |
| [COV_2] | [MEAN_B_2] | [MEAN_A_2] | [DIFF_2] | [PVAL_2] | [OK/FAIL] |
| [COV_3] | [MEAN_B_3] | [MEAN_A_3] | [DIFF_3] | [PVAL_3] | [OK/FAIL] |
| [COV_4] | [MEAN_B_4] | [MEAN_A_4] | [DIFF_4] | [PVAL_4] | [OK/FAIL] |

**Joint Test P-value**: [JOINT_PVAL]

**Interpretation**: [Are covariates balanced? Any concerns?]

---

## 4. Main Results

### 4.1 RD Plot

[INSERT RD PLOT WITH BINNED SCATTER AND FITTED LINES]

*Notes: Circles represent bin means. Solid lines are local polynomial fits. Shaded areas show 95% confidence intervals. Dashed vertical line indicates the cutoff.*

### 4.2 Point Estimates

| Specification | Effect | SE | 95% CI | P-value | N_eff |
|---------------|--------|-----|--------|---------|-------|
| Baseline (Linear) | **[EFFECT]** | [SE] | [[CI_LOWER], [CI_UPPER]] | [PVAL] | [N_EFF] |
| Quadratic | [EFFECT_2] | [SE_2] | [[CI_2]] | [PVAL_2] | [N_EFF_2] |
| Half Bandwidth | [EFFECT_3] | [SE_3] | [[CI_3]] | [PVAL_3] | [N_EFF_3] |
| Double Bandwidth | [EFFECT_4] | [SE_4] | [[CI_4]] | [PVAL_4] | [N_EFF_4] |

**Baseline Specification**:
- Bandwidth: [BANDWIDTH] (MSE-optimal)
- Kernel: Triangular
- Polynomial Order: 1 (local linear)
- Inference: Robust bias-corrected (CCT)

### 4.3 Interpretation

[2-3 paragraphs interpreting the main result]

**Effect Size**: The estimated effect of [EFFECT] [UNITS] represents [interpretation in context - e.g., X% of the outcome mean, Y standard deviations].

**Statistical Significance**: The effect is statistically significant at the [X]% level (p = [PVAL]).

**Practical Significance**: [Discussion of whether the effect is economically/practically meaningful]

---

## 5. Robustness Analysis

### 5.1 Bandwidth Sensitivity

The table below shows how the effect estimate changes across different bandwidth choices.

| Bandwidth | Ratio | Effect | SE | 95% CI | P-value | N_eff |
|-----------|-------|--------|-----|--------|---------|-------|
| [BW_1] | 0.50x | [EFF_1] | [SE_1] | [[CI_1]] | [PVAL_1] | [N_1] |
| [BW_2] | 0.75x | [EFF_2] | [SE_2] | [[CI_2]] | [PVAL_2] | [N_2] |
| [BW_3]* | 1.00x | [EFF_3] | [SE_3] | [[CI_3]] | [PVAL_3] | [N_3] |
| [BW_4] | 1.25x | [EFF_4] | [SE_4] | [[CI_4]] | [PVAL_4] | [N_4] |
| [BW_5] | 1.50x | [EFF_5] | [SE_5] | [[CI_5]] | [PVAL_5] | [N_5] |
| [BW_6] | 2.00x | [EFF_6] | [SE_6] | [[CI_6]] | [PVAL_6] | [N_6] |

*Optimal bandwidth marked with asterisk*

[INSERT BANDWIDTH SENSITIVITY PLOT]

**Assessment**: [Are results robust to bandwidth choice?]

### 5.2 Placebo Cutoff Tests

We test for effects at placebo cutoffs where no effect should exist.

| Placebo Cutoff | Effect | P-value | Result |
|----------------|--------|---------|--------|
| [PC_1] | [EFF_1] | [PVAL_1] | [PASS/FAIL] |
| [PC_2] | [EFF_2] | [PVAL_2] | [PASS/FAIL] |
| [PC_3] | [EFF_3] | [PVAL_3] | [PASS/FAIL] |
| [PC_4] | [EFF_4] | [PVAL_4] | [PASS/FAIL] |

**Assessment**: [Do placebo tests pass?]

### 5.3 Donut Hole Analysis

[If applicable, include donut hole results]

| Donut Radius | Effect | SE | N Excluded |
|--------------|--------|-----|------------|
| 0 (none) | [EFF_0] | [SE_0] | 0 |
| [RAD_1] | [EFF_1] | [SE_1] | [N_1] |
| [RAD_2] | [EFF_2] | [SE_2] | [N_2] |

### 5.4 Polynomial Order Sensitivity

| Order | Effect | SE | 95% CI |
|-------|--------|-----|--------|
| Linear (p=1) | [EFF_1] | [SE_1] | [[CI_1]] |
| Quadratic (p=2) | [EFF_2] | [SE_2] | [[CI_2]] |

---

## 6. Additional Analyses

### 6.1 Heterogeneity

[If applicable, present heterogeneity analysis by subgroups]

### 6.2 Fuzzy RD Details

[If fuzzy RD, include first stage and reduced form]

| Component | Estimate | SE |
|-----------|----------|-----|
| First Stage (Treatment Jump) | [FS] | [FS_SE] |
| Reduced Form (Outcome Jump) | [RF] | [RF_SE] |
| **LATE (Fuzzy RD)** | **[LATE]** | [LATE_SE] |

First-stage F-statistic: [F_STAT]

---

## 7. Limitations and Caveats

### 7.1 External Validity

The RD estimate identifies the **Local Average Treatment Effect (LATE)** for units at the margin of the cutoff. This estimate may not generalize to:
- Units far from the cutoff
- The average treatment effect for all treated units
- Settings with different cutoff values

### 7.2 Design-Specific Concerns

[List any design-specific concerns:]
- [Concern 1 - e.g., McCrary test marginal]
- [Concern 2 - e.g., Some covariate imbalance]
- [Concern 3 - e.g., Discrete running variable]

### 7.3 Other Limitations

- [Data limitations]
- [Measurement issues]
- [Sample representativeness]

---

## 8. Conclusions

### 8.1 Summary of Findings

[1-2 paragraph summary]

### 8.2 Policy Implications

[If applicable, discuss policy implications]

### 8.3 Future Research

[Suggestions for future work]

---

## Technical Appendix

### A.1 Estimation Details

| Parameter | Value |
|-----------|-------|
| Bandwidth Selection | MSE-optimal (Calonico, Cattaneo, Titiunik 2014) |
| Kernel Function | Triangular |
| Polynomial Order | 1 (local linear) |
| Standard Errors | Robust bias-corrected |
| Software | [Python/R/Stata] using [rdrobust/rd_estimator] |

### A.2 Mathematical Specification

The sharp RD estimand is:
$$\tau_{SRD} = \lim_{x \downarrow c} E[Y|X=x] - \lim_{x \uparrow c} E[Y|X=x]$$

Estimated using local polynomial regression:
$$\hat{\tau} = \hat{\alpha}_+ - \hat{\alpha}_-$$

where $\hat{\alpha}_+$ and $\hat{\alpha}_-$ are the intercepts from weighted regressions above and below the cutoff.

### A.3 Replication

Code and data to replicate this analysis are available at: [REPOSITORY_URL]

---

## References

1. Cattaneo, M. D., Idrobo, N., & Titiunik, R. (2020). *A Practical Introduction to Regression Discontinuity Designs: Foundations*. Cambridge University Press.

2. Calonico, S., Cattaneo, M. D., & Titiunik, R. (2014). Robust nonparametric confidence intervals for regression-discontinuity designs. *Econometrica*, 82(6), 2295-2326.

3. Lee, D. S., & Lemieux, T. (2010). Regression discontinuity designs in economics. *Journal of Economic Literature*, 48(2), 281-355.

4. McCrary, J. (2008). Manipulation of the running variable in the regression discontinuity design: A density test. *Journal of Econometrics*, 142(2), 698-714.

---

*Report generated using Causal ML Skills - RD Estimator*
