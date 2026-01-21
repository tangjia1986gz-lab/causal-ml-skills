# {{TITLE}}: Causal Analysis Report

**Author(s):** {{AUTHORS}}
**Date:** {{DATE}}
**Version:** {{VERSION}}

---

## Executive Summary

{{EXECUTIVE_SUMMARY}}

**Key Finding:** {{KEY_FINDING}}

**Treatment Effect:** {{EFFECT_SIZE}} (95% CI: [{{CI_LOWER}}, {{CI_UPPER}}])

---

## 1. Introduction

### 1.1 Research Question

{{RESEARCH_QUESTION}}

### 1.2 Motivation

{{MOTIVATION}}

### 1.3 Preview of Results

{{PREVIEW}}

---

## 2. Data

### 2.1 Data Source

| Attribute | Description |
|-----------|-------------|
| **Source** | {{DATA_SOURCE}} |
| **Time Period** | {{TIME_PERIOD}} |
| **Unit of Observation** | {{UNIT}} |
| **Sample Size** | {{SAMPLE_SIZE}} |

### 2.2 Variable Definitions

| Variable | Type | Description | Source |
|----------|------|-------------|--------|
| {{OUTCOME}} | Outcome (Y) | {{OUTCOME_DESC}} | {{OUTCOME_SOURCE}} |
| {{TREATMENT}} | Treatment (D) | {{TREATMENT_DESC}} | {{TREATMENT_SOURCE}} |
| {{CONTROL_1}} | Control | {{CONTROL_1_DESC}} | {{CONTROL_1_SOURCE}} |
| {{CONTROL_2}} | Control | {{CONTROL_2_DESC}} | {{CONTROL_2_SOURCE}} |

### 2.3 Sample Selection

```
Initial sample:                           N = {{N_INITIAL}}
  - Drop: Missing outcome                 N = {{N_DROP_1}} ({{PCT_DROP_1}}%)
  - Drop: Missing treatment               N = {{N_DROP_2}} ({{PCT_DROP_2}}%)
  - Drop: Outside sample period           N = {{N_DROP_3}} ({{PCT_DROP_3}}%)
  - Drop: Outliers (>3 SD)                N = {{N_DROP_4}} ({{PCT_DROP_4}}%)
                                         ──────────────────
Final analysis sample:                    N = {{N_FINAL}}
```

### 2.4 Summary Statistics

| Variable | N | Mean | SD | Min | Max | P25 | P50 | P75 |
|----------|---|------|----|----|-----|-----|-----|-----|
| {{VAR_1}} | {{N_1}} | {{MEAN_1}} | {{SD_1}} | {{MIN_1}} | {{MAX_1}} | {{P25_1}} | {{P50_1}} | {{P75_1}} |
| {{VAR_2}} | {{N_2}} | {{MEAN_2}} | {{SD_2}} | {{MIN_2}} | {{MAX_2}} | {{P25_2}} | {{P50_2}} | {{P75_2}} |
| {{VAR_3}} | {{N_3}} | {{MEAN_3}} | {{SD_3}} | {{MIN_3}} | {{MAX_3}} | {{P25_3}} | {{P50_3}} | {{P75_3}} |

---

## 3. Empirical Strategy

### 3.1 Identification Strategy

**Method:** {{METHOD}}

{{IDENTIFICATION_NARRATIVE}}

### 3.2 Specification

**Main Equation:**

$$
{{MAIN_EQUATION}}
$$

where:
- $Y_{it}$ = {{Y_DEFINITION}}
- $D_{it}$ = {{D_DEFINITION}}
- $X_{it}$ = {{X_DEFINITION}}
- $\beta$ = {{BETA_INTERPRETATION}}

### 3.3 Identification Assumptions

| Assumption | Description | Testable? | Evidence |
|------------|-------------|-----------|----------|
| {{ASSUMPTION_1}} | {{ASSUMPTION_1_DESC}} | {{TESTABLE_1}} | {{EVIDENCE_1}} |
| {{ASSUMPTION_2}} | {{ASSUMPTION_2_DESC}} | {{TESTABLE_2}} | {{EVIDENCE_2}} |
| {{ASSUMPTION_3}} | {{ASSUMPTION_3_DESC}} | {{TESTABLE_3}} | {{EVIDENCE_3}} |

### 3.4 Threats to Identification

| Threat | Concern | Mitigation |
|--------|---------|------------|
| {{THREAT_1}} | {{CONCERN_1}} | {{MITIGATION_1}} |
| {{THREAT_2}} | {{CONCERN_2}} | {{MITIGATION_2}} |
| {{THREAT_3}} | {{CONCERN_3}} | {{MITIGATION_3}} |

---

## 4. Results

### 4.1 Main Results

**Table 1: Main Treatment Effect Estimates**

| Specification | (1) | (2) | (3) | (4) |
|---------------|-----|-----|-----|-----|
| **Treatment** | {{COEF_1}}{{STARS_1}} | {{COEF_2}}{{STARS_2}} | {{COEF_3}}{{STARS_3}} | {{COEF_4}}{{STARS_4}} |
| | ({{SE_1}}) | ({{SE_2}}) | ({{SE_3}}) | ({{SE_4}}) |
| Controls | No | Yes | Yes | Yes |
| Year FE | No | No | Yes | Yes |
| Region FE | No | No | No | Yes |
| N | {{N_1}} | {{N_2}} | {{N_3}} | {{N_4}} |
| R² | {{R2_1}} | {{R2_2}} | {{R2_3}} | {{R2_4}} |

*Notes: Standard errors clustered at {{CLUSTER_LEVEL}} level in parentheses. \* p<0.10, \*\* p<0.05, \*\*\* p<0.01*

### 4.2 Interpretation

{{MAIN_INTERPRETATION}}

**Economic Magnitude:** {{ECONOMIC_MAGNITUDE}}

**Statistical Significance:** {{STAT_SIGNIFICANCE}}

### 4.3 Diagnostic Tests

| Test | Statistic | p-value | Interpretation |
|------|-----------|---------|----------------|
| {{DIAG_1}} | {{STAT_1}} | {{PVAL_1}} | {{INTERP_1}} |
| {{DIAG_2}} | {{STAT_2}} | {{PVAL_2}} | {{INTERP_2}} |
| {{DIAG_3}} | {{STAT_3}} | {{PVAL_3}} | {{INTERP_3}} |

---

## 5. Robustness Checks

### 5.1 Sensitivity Analysis

| Robustness Check | Point Estimate | SE | 95% CI | Note |
|------------------|----------------|-----|--------|------|
| Baseline | {{BASELINE_EST}} | {{BASELINE_SE}} | [{{BASELINE_CI}}] | Main specification |
| {{ROBUST_1}} | {{ROBUST_1_EST}} | {{ROBUST_1_SE}} | [{{ROBUST_1_CI}}] | {{ROBUST_1_NOTE}} |
| {{ROBUST_2}} | {{ROBUST_2_EST}} | {{ROBUST_2_SE}} | [{{ROBUST_2_CI}}] | {{ROBUST_2_NOTE}} |
| {{ROBUST_3}} | {{ROBUST_3_EST}} | {{ROBUST_3_SE}} | [{{ROBUST_3_CI}}] | {{ROBUST_3_NOTE}} |
| {{ROBUST_4}} | {{ROBUST_4_EST}} | {{ROBUST_4_SE}} | [{{ROBUST_4_CI}}] | {{ROBUST_4_NOTE}} |

### 5.2 Placebo Tests

{{PLACEBO_DESCRIPTION}}

| Placebo | Estimate | SE | p-value | Interpretation |
|---------|----------|-----|---------|----------------|
| {{PLACEBO_1}} | {{PLACEBO_1_EST}} | {{PLACEBO_1_SE}} | {{PLACEBO_1_P}} | {{PLACEBO_1_INTERP}} |
| {{PLACEBO_2}} | {{PLACEBO_2_EST}} | {{PLACEBO_2_SE}} | {{PLACEBO_2_P}} | {{PLACEBO_2_INTERP}} |

### 5.3 Alternative Samples

| Sample | N | Estimate | SE | Note |
|--------|---|----------|-----|------|
| Full Sample | {{N_FULL}} | {{EST_FULL}} | {{SE_FULL}} | Baseline |
| {{ALT_SAMPLE_1}} | {{N_ALT_1}} | {{EST_ALT_1}} | {{SE_ALT_1}} | {{NOTE_ALT_1}} |
| {{ALT_SAMPLE_2}} | {{N_ALT_2}} | {{EST_ALT_2}} | {{SE_ALT_2}} | {{NOTE_ALT_2}} |

---

## 6. Heterogeneous Effects

### 6.1 Pre-Specified Subgroups

| Subgroup | N | CATE | SE | 95% CI |
|----------|---|------|-----|--------|
| {{SUBGROUP_1}} | {{N_SG1}} | {{CATE_SG1}} | {{SE_SG1}} | [{{CI_SG1}}] |
| {{SUBGROUP_2}} | {{N_SG2}} | {{CATE_SG2}} | {{SE_SG2}} | [{{CI_SG2}}] |
| {{SUBGROUP_3}} | {{N_SG3}} | {{CATE_SG3}} | {{SE_SG3}} | [{{CI_SG3}}] |

**Test for Heterogeneity:** {{HET_TEST}} (p = {{HET_P}})

### 6.2 Machine Learning-Based Heterogeneity

{{ML_HET_DESCRIPTION}}

| Metric | Value |
|--------|-------|
| CATE Variance | {{CATE_VAR}} |
| AUTOC | {{AUTOC}} |
| Top vs. Bottom Quartile | {{TOP_BOTTOM}} |

---

## 7. Discussion

### 7.1 Summary of Findings

{{SUMMARY_FINDINGS}}

### 7.2 Comparison with Existing Literature

| Study | Context | Method | Finding | Our Result |
|-------|---------|--------|---------|------------|
| {{STUDY_1}} | {{CONTEXT_1}} | {{METHOD_1}} | {{FINDING_1}} | {{COMPARE_1}} |
| {{STUDY_2}} | {{CONTEXT_2}} | {{METHOD_2}} | {{FINDING_2}} | {{COMPARE_2}} |

### 7.3 Limitations

{{LIMITATIONS}}

### 7.4 Policy Implications

{{POLICY_IMPLICATIONS}}

---

## 8. Conclusion

{{CONCLUSION}}

---

## References

{{REFERENCES}}

---

## Appendix

### A. Additional Tables and Figures

*See separate appendix document.*

### B. Data Construction Details

{{DATA_CONSTRUCTION}}

### C. Replication Information

- **Code Repository:** {{CODE_REPO}}
- **Data Availability:** {{DATA_AVAILABILITY}}
- **Computational Requirements:** {{COMPUTE_REQ}}
- **Contact:** {{CONTACT}}

---

*Report generated using the Causal ML Skills framework.*
*Template version: 1.0.0*

---

## Template Usage Instructions

### Placeholders

Replace all `{{PLACEHOLDER}}` markers with your actual content:

1. **Metadata**: TITLE, AUTHORS, DATE, VERSION
2. **Data Section**: All variable definitions, sample statistics
3. **Results**: Coefficients, standard errors, test statistics
4. **Interpretation**: Narrative sections explaining your findings

### Customization

- Add/remove sections as needed for your specific analysis
- Modify table structures to match your specification
- Include relevant figures inline or in appendix

### Best Practices

1. **Be specific** about identification assumptions and their validity
2. **Quantify uncertainty** with confidence intervals throughout
3. **Pre-register** heterogeneity analyses to avoid p-hacking
4. **Report all specifications** tested, not just significant ones
5. **Compare magnitudes** to existing literature

### Export Options

- Export to PDF via pandoc with LaTeX
- Convert to Word using pandoc
- Render in Jupyter/R Markdown for reproducibility
