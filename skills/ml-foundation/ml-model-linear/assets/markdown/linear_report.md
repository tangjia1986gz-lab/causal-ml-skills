# Linear Regression Analysis Report

**Project:** {{PROJECT_NAME}}
**Date:** {{DATE}}
**Analyst:** {{ANALYST_NAME}}

---

## Executive Summary

{{EXECUTIVE_SUMMARY}}

**Key Findings:**
- Treatment effect estimate: {{TREATMENT_EFFECT}} (95% CI: [{{CI_LOWER}}, {{CI_UPPER}}])
- Model selected {{N_SELECTED}} of {{N_TOTAL}} potential control variables
- Cross-validated RMSE: {{CV_RMSE}}

---

## 1. Data Description

### 1.1 Sample Overview

| Metric | Value |
|--------|-------|
| Observations | {{N_OBS}} |
| Features | {{N_FEATURES}} |
| Missing values | {{N_MISSING}} |
| Outcome variable | {{OUTCOME_NAME}} |
| Treatment variable | {{TREATMENT_NAME}} (if applicable) |

### 1.2 Variable Summary

| Variable | Mean | Std Dev | Min | Max | Missing |
|----------|------|---------|-----|-----|---------|
| {{VAR1_NAME}} | {{VAR1_MEAN}} | {{VAR1_STD}} | {{VAR1_MIN}} | {{VAR1_MAX}} | {{VAR1_MISS}} |
| {{VAR2_NAME}} | {{VAR2_MEAN}} | {{VAR2_STD}} | {{VAR2_MIN}} | {{VAR2_MAX}} | {{VAR2_MISS}} |
| ... | ... | ... | ... | ... | ... |

### 1.3 Correlation Structure

{{CORRELATION_NOTES}}

---

## 2. Model Specification

### 2.1 Research Question

{{RESEARCH_QUESTION}}

### 2.2 Estimation Strategy

**Primary specification:**
```
Y = beta_0 + beta_1 * X_1 + beta_2 * X_2 + ... + epsilon
```

**Regularization approach:** {{REGULARIZATION_APPROACH}}
- Ridge (L2): For multicollinearity, all variables believed relevant
- Lasso (L1): For variable selection, sparse models
- Elastic Net: For grouped selection with correlated features
- Post-Double-Selection: For causal inference with high-dimensional controls

### 2.3 Variable Selection Method

{{SELECTION_METHOD_DESCRIPTION}}

---

## 3. Model Results

### 3.1 OLS Baseline

| Variable | Coefficient | Std. Error | t-stat | p-value |
|----------|-------------|------------|--------|---------|
| (Intercept) | {{CONST_COEF}} | {{CONST_SE}} | {{CONST_T}} | {{CONST_P}} |
| {{VAR1_NAME}} | {{VAR1_COEF}} | {{VAR1_SE}} | {{VAR1_T}} | {{VAR1_P}} |
| {{VAR2_NAME}} | {{VAR2_COEF}} | {{VAR2_SE}} | {{VAR2_T}} | {{VAR2_P}} |
| ... | ... | ... | ... | ... |

**Model Statistics:**
- R-squared: {{R2}}
- Adjusted R-squared: {{ADJ_R2}}
- F-statistic: {{F_STAT}} (p = {{F_PVAL}})
- Residual Std. Error: {{RSE}}

### 3.2 Regularized Models

| Model | Alpha | L1 Ratio | CV RMSE | CV R2 | Non-zero |
|-------|-------|----------|---------|-------|----------|
| Ridge | {{ALPHA_RIDGE}} | 0.00 | {{RMSE_RIDGE}} | {{R2_RIDGE}} | {{NZ_RIDGE}} |
| Lasso | {{ALPHA_LASSO}} | 1.00 | {{RMSE_LASSO}} | {{R2_LASSO}} | {{NZ_LASSO}} |
| Elastic Net | {{ALPHA_ENET}} | {{L1_ENET}} | {{RMSE_ENET}} | {{R2_ENET}} | {{NZ_ENET}} |

### 3.3 Selected Variables (Lasso/Elastic Net)

| Rank | Variable | Coefficient | Selection Probability |
|------|----------|-------------|----------------------|
| 1 | {{SEL_VAR1}} | {{SEL_COEF1}} | {{SEL_PROB1}} |
| 2 | {{SEL_VAR2}} | {{SEL_COEF2}} | {{SEL_PROB2}} |
| 3 | {{SEL_VAR3}} | {{SEL_COEF3}} | {{SEL_PROB3}} |
| ... | ... | ... | ... |

---

## 4. Causal Inference (if applicable)

### 4.1 Post-Double-Selection Results

**Treatment:** {{TREATMENT_NAME}}

| Specification | Effect | Std. Error | 95% CI | p-value |
|---------------|--------|------------|--------|---------|
| Naive (all controls) | {{NAIVE_EFFECT}} | {{NAIVE_SE}} | [{{NAIVE_CI_L}}, {{NAIVE_CI_U}}] | {{NAIVE_P}} |
| Single Selection | {{SINGLE_EFFECT}} | {{SINGLE_SE}} | [{{SINGLE_CI_L}}, {{SINGLE_CI_U}}] | {{SINGLE_P}} |
| **Double Selection** | **{{DOUBLE_EFFECT}}** | **{{DOUBLE_SE}}** | **[{{DOUBLE_CI_L}}, {{DOUBLE_CI_U}}]** | **{{DOUBLE_P}}** |

### 4.2 Variable Selection Details

| Step | Variables Selected | Alpha |
|------|-------------------|-------|
| Outcome model (Y ~ X) | {{N_SEL_Y}} | {{ALPHA_Y}} |
| Treatment model (D ~ X) | {{N_SEL_D}} | {{ALPHA_D}} |
| Union (final) | {{N_SEL_UNION}} | --- |

**Selected by Y model only:** {{VARS_Y_ONLY}}
**Selected by D model only:** {{VARS_D_ONLY}}
**Selected by both:** {{VARS_BOTH}}

### 4.3 Interpretation

{{CAUSAL_INTERPRETATION}}

**Important caveats:**
1. Double selection controls for *observable* confounders only
2. Assumes linear relationship between controls and outcome/treatment
3. Does not address reverse causality or measurement error

---

## 5. Diagnostic Tests

### 5.1 Residual Diagnostics

| Test | Statistic | p-value | Conclusion |
|------|-----------|---------|------------|
| Shapiro-Wilk (normality) | {{SW_STAT}} | {{SW_P}} | {{SW_CONC}} |
| Jarque-Bera (normality) | {{JB_STAT}} | {{JB_P}} | {{JB_CONC}} |
| Breusch-Pagan (heteroskedasticity) | {{BP_STAT}} | {{BP_P}} | {{BP_CONC}} |
| White (heteroskedasticity) | {{WHITE_STAT}} | {{WHITE_P}} | {{WHITE_CONC}} |
| Durbin-Watson (autocorrelation) | {{DW_STAT}} | --- | {{DW_CONC}} |

### 5.2 Multicollinearity

| Variable | VIF | Concern Level |
|----------|-----|---------------|
| {{VAR1_NAME}} | {{VAR1_VIF}} | {{VAR1_VIF_CONC}} |
| {{VAR2_NAME}} | {{VAR2_VIF}} | {{VAR2_VIF_CONC}} |
| ... | ... | ... |

**Condition Number:** {{COND_NUM}} ({{COND_CONC}})

### 5.3 Influential Observations

| Diagnostic | Threshold | Count Exceeding | Max Value | Max Index |
|------------|-----------|-----------------|-----------|-----------|
| Cook's Distance | {{COOK_THRESH}} | {{COOK_N}} | {{COOK_MAX}} | {{COOK_IDX}} |
| Leverage | {{LEV_THRESH}} | {{LEV_N}} | {{LEV_MAX}} | {{LEV_IDX}} |
| DFFITS | {{DFFIT_THRESH}} | {{DFFIT_N}} | {{DFFIT_MAX}} | {{DFFIT_IDX}} |

---

## 6. Robustness Checks

### 6.1 Alternative Specifications

| Specification | Treatment Effect | Std. Error | Notes |
|---------------|------------------|------------|-------|
| Baseline (Double Selection) | {{BASE_EFFECT}} | {{BASE_SE}} | --- |
| With interaction terms | {{INT_EFFECT}} | {{INT_SE}} | {{INT_NOTES}} |
| Excluding outliers | {{EXCL_EFFECT}} | {{EXCL_SE}} | Removed {{N_EXCL}} obs |
| Different alpha rule | {{ALT_EFFECT}} | {{ALT_SE}} | 1-SE rule |

### 6.2 Sensitivity to Control Selection

{{SENSITIVITY_NOTES}}

---

## 7. Conclusions

### 7.1 Main Findings

{{MAIN_FINDINGS}}

### 7.2 Limitations

1. {{LIMITATION_1}}
2. {{LIMITATION_2}}
3. {{LIMITATION_3}}

### 7.3 Recommendations

{{RECOMMENDATIONS}}

---

## Appendix

### A. Full Coefficient Table

See attached file: `{{COEF_TABLE_FILE}}`

### B. Diagnostic Plots

- Residuals vs Fitted: `{{PLOT_RESID_FITTED}}`
- Q-Q Plot: `{{PLOT_QQ}}`
- Scale-Location: `{{PLOT_SCALE_LOC}}`
- Cook's Distance: `{{PLOT_COOKS}}`
- Regularization Path: `{{PLOT_REG_PATH}}`

### C. Code Reproducibility

```bash
# Fit linear model
python run_linear_model.py {{DATA_FILE}} \
    --outcome {{OUTCOME_NAME}} \
    --model double-selection \
    --treatment {{TREATMENT_NAME}} \
    --cv 5

# Run diagnostics
python diagnose_assumptions.py {{DATA_FILE}} \
    --outcome {{OUTCOME_NAME}} \
    --all-features \
    --plot \
    --plot-dir ./diagnostic_plots

# Visualize coefficients
python visualize_coefficients.py {{DATA_FILE}} \
    --outcome {{OUTCOME_NAME}} \
    --model lasso \
    --plot-type path \
    --output reg_path.png
```

### D. References

1. Belloni, A., Chernozhukov, V., & Hansen, C. (2014). Inference on treatment effects after selection among high-dimensional controls. *Review of Economic Studies*, 81(2), 608-650.

2. Tibshirani, R. (1996). Regression shrinkage and selection via the Lasso. *JRSS-B*, 58(1), 267-288.

3. Zou, H., & Hastie, T. (2005). Regularization and variable selection via the elastic net. *JRSS-B*, 67(2), 301-320.

---

*Report generated using ml-model-linear skill*
*{{GENERATION_DATE}}*
