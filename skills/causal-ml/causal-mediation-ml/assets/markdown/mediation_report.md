# Causal Mediation Analysis Report

**Study**: {{STUDY_NAME}}
**Generated**: {{DATE}}
**Analyst**: {{ANALYST}}

---

## Executive Summary

This report presents the results of a causal mediation analysis examining whether the effect of **{{TREATMENT}}** on **{{OUTCOME}}** operates through **{{MEDIATOR}}**.

**Key Findings**:
- Total Effect: {{TOTAL_EFFECT}} ({{TOTAL_SIG}})
- Direct Effect (ADE): {{ADE}} ({{ADE_SIG}})
- Indirect Effect (ACME): {{ACME}} ({{ACME_SIG}})
- Proportion Mediated: {{PROP_MEDIATED}}%

---

## 1. Research Design

### 1.1 Research Question

Does {{TREATMENT}} affect {{OUTCOME}} through {{MEDIATOR}}?

### 1.2 Causal Structure

```
{{TREATMENT}} ─────────────────────────────────────> {{OUTCOME}}
      │                Direct Effect (ADE)                ↑
      │                                                   │
      └────────> {{MEDIATOR}} ─────────────────────────>─┘
              Indirect Effect (ACME)
```

### 1.3 Variables

| Variable | Role | Description |
|----------|------|-------------|
| {{TREATMENT}} | Treatment (D) | {{TREATMENT_DESC}} |
| {{MEDIATOR}} | Mediator (M) | {{MEDIATOR_DESC}} |
| {{OUTCOME}} | Outcome (Y) | {{OUTCOME_DESC}} |
| {{CONTROLS}} | Controls (X) | {{CONTROLS_DESC}} |

### 1.4 Sample

- **Sample Size**: n = {{N}}
- **Treatment Distribution**: {{TREATMENT_DIST}}
- **Missing Values**: {{MISSING_INFO}}

---

## 2. Identification Assumptions

Causal mediation analysis requires **sequential ignorability**:

### 2.1 Part 1: Treatment Ignorability

> Conditional on controls X, treatment assignment is independent of potential outcomes and potential mediator values.

**Assessment**: {{PART1_ASSESSMENT}}

### 2.2 Part 2: Mediator Ignorability

> Conditional on treatment D and controls X, the mediator is independent of potential outcomes.

**Assessment**: {{PART2_ASSESSMENT}}

**Important**: This is a *strong* assumption that cannot be tested. Even in randomized experiments, the mediator is typically not randomized.

---

## 3. Results

### 3.1 Effect Decomposition

| Effect | Estimate | Std. Error | 95% CI | p-value |
|--------|----------|------------|--------|---------|
| Total Effect | {{TOTAL_EFFECT}} | {{TOTAL_SE}} | {{TOTAL_CI}} | {{TOTAL_P}} |
| Direct Effect (ADE) | {{ADE}} | {{ADE_SE}} | {{ADE_CI}} | {{ADE_P}} |
| Indirect Effect (ACME) | {{ACME}} | {{ACME_SE}} | {{ACME_CI}} | {{ACME_P}} |

**Proportion Mediated**: {{PROP_MEDIATED}}% (SE = {{PROP_SE}}%)

### 3.2 Path Coefficients

| Path | Coefficient | Std. Error | Interpretation |
|------|-------------|------------|----------------|
| a (D -> M) | {{ALPHA}} | {{ALPHA_SE}} | Effect of treatment on mediator |
| b (M -> Y\|D) | {{BETA_M}} | {{BETA_M_SE}} | Effect of mediator on outcome, controlling for treatment |
| c' (D -> Y\|M) | {{BETA_D}} | {{BETA_D_SE}} | Direct effect of treatment on outcome |

### 3.3 Pathway Diagram

```
{{TREATMENT}}              {{MEDIATOR}}              {{OUTCOME}}
    │                          │                         ↑
    │    a = {{ALPHA}}         │    b = {{BETA_M}}       │
    └─────────────────────────>┼────────────────────────>┘
                               │
    └────────────────────────────────────────────────────┘
                    c' = {{BETA_D}} (direct)
```

---

## 4. Sensitivity Analysis

### 4.1 Robustness to Unmeasured Confounding

The sensitivity parameter **rho** represents the correlation between unmeasured confounders of the mediator-outcome relationship and the outcome.

| Metric | Value | Interpretation |
|--------|-------|----------------|
| Breakpoint (rho*) | {{BREAKPOINT}} | ACME = 0 at this rho |
| R-squared equivalent | {{R2_BREAKPOINT}} | Variance explained by confounder |
| Robustness Level | {{ROBUSTNESS}} | {{ROBUSTNESS_INTERP}} |

### 4.2 Sensitivity Plot

{{SENSITIVITY_PLOT}}

### 4.3 Interpretation

{{SENSITIVITY_INTERPRETATION}}

---

## 5. Robustness Checks

### 5.1 Alternative Specifications

| Specification | ACME | SE | Notes |
|---------------|------|-----|-------|
| Baseline | {{ACME}} | {{ACME_SE}} | Primary specification |
| {{ALT_SPEC_1}} | {{ALT_ACME_1}} | {{ALT_SE_1}} | {{ALT_NOTES_1}} |
| {{ALT_SPEC_2}} | {{ALT_ACME_2}} | {{ALT_SE_2}} | {{ALT_NOTES_2}} |

### 5.2 Subgroup Analysis

| Subgroup | n | ACME | ADE | % Mediated |
|----------|---|------|-----|------------|
| Full Sample | {{N}} | {{ACME}} | {{ADE}} | {{PROP_MEDIATED}}% |
| {{SUBGROUP_1}} | {{N_1}} | {{ACME_1}} | {{ADE_1}} | {{PROP_1}}% |
| {{SUBGROUP_2}} | {{N_2}} | {{ACME_2}} | {{ADE_2}} | {{PROP_2}}% |

---

## 6. Interpretation

### 6.1 Main Finding

{{MAIN_INTERPRETATION}}

### 6.2 Direct Effect

{{DIRECT_INTERPRETATION}}

### 6.3 Indirect Effect

{{INDIRECT_INTERPRETATION}}

### 6.4 Proportion Mediated

{{PROPORTION_INTERPRETATION}}

---

## 7. Limitations

1. **Sequential Ignorability**: {{LIMITATION_SEQ_IGN}}

2. **Unmeasured Confounding**: {{LIMITATION_CONFOUNDING}}

3. **Functional Form**: {{LIMITATION_FUNCTIONAL}}

4. **Generalizability**: {{LIMITATION_EXTERNAL}}

---

## 8. Conclusions

{{CONCLUSIONS}}

---

## Appendix A: Technical Details

### A.1 Estimation Method

- **Method**: {{METHOD}}
- **Bootstrap Replications**: {{N_BOOTSTRAP}}
- **Confidence Level**: {{CONFIDENCE_LEVEL}}%
- **ML Learners**: {{ML_LEARNERS}}

### A.2 Model Specifications

**Mediator Model**:
```
M = {{MEDIATOR_MODEL}}
```

**Outcome Model**:
```
Y = {{OUTCOME_MODEL}}
```

### A.3 Software

- **Package**: causal-mediation-ml skill
- **Python Version**: {{PYTHON_VERSION}}
- **Key Dependencies**: {{DEPENDENCIES}}

---

## Appendix B: Data Summary

### B.1 Variable Statistics

| Variable | Mean | SD | Min | Max |
|----------|------|-----|-----|-----|
| {{OUTCOME}} | {{Y_MEAN}} | {{Y_SD}} | {{Y_MIN}} | {{Y_MAX}} |
| {{TREATMENT}} | {{D_MEAN}} | {{D_SD}} | {{D_MIN}} | {{D_MAX}} |
| {{MEDIATOR}} | {{M_MEAN}} | {{M_SD}} | {{M_MIN}} | {{M_MAX}} |

### B.2 Covariate Balance

| Control | Treated Mean | Control Mean | Std. Diff | p-value |
|---------|--------------|--------------|-----------|---------|
| {{CONTROL_1}} | {{TREAT_MEAN_1}} | {{CTRL_MEAN_1}} | {{STD_DIFF_1}} | {{BAL_P_1}} |
| {{CONTROL_2}} | {{TREAT_MEAN_2}} | {{CTRL_MEAN_2}} | {{STD_DIFF_2}} | {{BAL_P_2}} |

---

## References

1. Baron, R. M., & Kenny, D. A. (1986). The Moderator-Mediator Variable Distinction in Social Psychological Research. *Journal of Personality and Social Psychology*, 51(6), 1173-1182.

2. Imai, K., Keele, L., & Tingley, D. (2010). A General Approach to Causal Mediation Analysis. *Psychological Methods*, 15(4), 309-334.

3. Imai, K., Keele, L., & Yamamoto, T. (2010). Identification, Inference and Sensitivity Analysis for Causal Mediation Effects. *Statistical Science*, 25(1), 51-71.

4. VanderWeele, T. J. (2015). *Explanation in Causal Inference: Methods for Mediation and Interaction*. Oxford University Press.

---

*Report generated by causal-mediation-ml skill*
*Version: 1.0.0*
