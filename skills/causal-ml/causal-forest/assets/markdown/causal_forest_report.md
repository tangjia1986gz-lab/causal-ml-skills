# Causal Forest Analysis Report

**Generated**: {{TIMESTAMP}}
**Analysis ID**: {{ANALYSIS_ID}}

---

## Executive Summary

This report presents the results of a causal forest analysis estimating heterogeneous treatment effects (HTEs). The analysis identifies how treatment effects vary across individuals based on their characteristics, enabling personalized treatment recommendations.

### Key Findings

| Metric | Value |
|--------|-------|
| Average Treatment Effect (ATE) | {{ATE}} (SE: {{ATE_SE}}) |
| CATE Standard Deviation | {{CATE_STD}} |
| Significant Heterogeneity | {{HETEROGENEITY_SIGNIFICANT}} |
| Top Heterogeneity Driver | {{TOP_DRIVER}} |
| Optimal Treatment Rate | {{OPTIMAL_TREATMENT_RATE}} |
| Policy Improvement | {{POLICY_IMPROVEMENT}} |

---

## 1. Study Design

### 1.1 Data Description

| Characteristic | Value |
|----------------|-------|
| Sample Size | {{N_TOTAL}} |
| Treatment Variable | {{TREATMENT_NAME}} |
| Outcome Variable | {{OUTCOME_NAME}} |
| Effect Modifiers | {{EFFECT_MODIFIERS}} |
| Additional Confounders | {{CONFOUNDERS}} |

### 1.2 Treatment Summary

| Group | N | % |
|-------|---|---|
| Treated | {{N_TREATED}} | {{PCT_TREATED}} |
| Control | {{N_CONTROL}} | {{PCT_CONTROL}} |

### 1.3 Outcome Summary

| Statistic | Treated | Control | Difference |
|-----------|---------|---------|------------|
| Mean | {{Y_MEAN_TREATED}} | {{Y_MEAN_CONTROL}} | {{Y_DIFF}} |
| Std Dev | {{Y_STD_TREATED}} | {{Y_STD_CONTROL}} | - |

---

## 2. Identification Strategy

### 2.1 Assumptions

This analysis relies on the following identification assumptions:

1. **Unconfoundedness (Selection on Observables)**
   - Treatment assignment is independent of potential outcomes conditional on observed covariates
   - All confounders are observed and included in the model

2. **Positivity (Overlap)**
   - Every covariate profile has positive probability of receiving each treatment
   - No deterministic treatment assignment regions

3. **SUTVA (Stable Unit Treatment Value Assumption)**
   - No interference between units
   - Single version of treatment

### 2.2 Assumption Diagnostics

#### Overlap Assessment

| Metric | Value | Threshold | Status |
|--------|-------|-----------|--------|
| Propensity Score Min | {{PS_MIN}} | > 0.05 | {{PS_MIN_STATUS}} |
| Propensity Score Max | {{PS_MAX}} | < 0.95 | {{PS_MAX_STATUS}} |
| Common Support Coverage | {{COMMON_SUPPORT}} | > 90% | {{CS_STATUS}} |

#### Covariate Balance

| Variable | Std Diff | Status |
|----------|----------|--------|
{{BALANCE_TABLE}}

---

## 3. Model Specification

### 3.1 Causal Forest Configuration

| Parameter | Value |
|-----------|-------|
| Backend | {{BACKEND}} |
| Number of Trees | {{N_TREES}} |
| Honesty | {{HONESTY}} |
| Honesty Fraction | {{HONESTY_FRACTION}} |
| Minimum Node Size | {{MIN_NODE_SIZE}} |
| Sample Fraction | {{SAMPLE_FRACTION}} |
| Random Seed | {{RANDOM_SEED}} |

### 3.2 Nuisance Model Specification

- **Outcome Model**: {{OUTCOME_MODEL}}
- **Propensity Model**: {{PROPENSITY_MODEL}}

---

## 4. Treatment Effect Results

### 4.1 Average Treatment Effect

$$\hat{\tau}_{ATE} = {{ATE}} \quad (95\% \text{ CI: } [{{ATE_CI_LOWER}}, {{ATE_CI_UPPER}}])$$

**Interpretation**: On average, treatment {{ATE_DIRECTION}} the outcome by {{ATE_ABS}} units.

### 4.2 CATE Distribution

| Statistic | Value |
|-----------|-------|
| Mean | {{CATE_MEAN}} |
| Standard Deviation | {{CATE_STD}} |
| Minimum | {{CATE_MIN}} |
| Maximum | {{CATE_MAX}} |
| 25th Percentile | {{CATE_Q25}} |
| Median | {{CATE_MEDIAN}} |
| 75th Percentile | {{CATE_Q75}} |

### 4.3 Effect Direction Analysis

| Category | N | Percentage |
|----------|---|------------|
| Positive Effects | {{N_POSITIVE}} | {{PCT_POSITIVE}} |
| Negative Effects | {{N_NEGATIVE}} | {{PCT_NEGATIVE}} |
| Significantly Positive | {{N_SIG_POSITIVE}} | {{PCT_SIG_POSITIVE}} |
| Significantly Negative | {{N_SIG_NEGATIVE}} | {{PCT_SIG_NEGATIVE}} |

---

## 5. Heterogeneity Analysis

### 5.1 Omnibus Heterogeneity Test

| Metric | Value |
|--------|-------|
| Test Statistic | {{HET_STAT}} |
| p-value | {{HET_PVALUE}} |
| Conclusion | {{HET_CONCLUSION}} |

### 5.2 Variable Importance

The following variables drive treatment effect heterogeneity:

| Rank | Variable | Importance | Cumulative |
|------|----------|------------|------------|
{{IMPORTANCE_TABLE}}

### 5.3 Best Linear Projection

The CATE can be approximated by a linear function of covariates:

$$\hat{\tau}(x) \approx {{BLP_INTERCEPT}} {{BLP_FORMULA}}$$

| Variable | Coefficient | Std Error | t-stat | p-value |
|----------|-------------|-----------|--------|---------|
| Intercept | {{BLP_INTERCEPT}} | {{BLP_INTERCEPT_SE}} | {{BLP_INTERCEPT_T}} | {{BLP_INTERCEPT_P}} |
{{BLP_TABLE}}

**R-squared**: {{BLP_R2}}

**Interpretation**: {{BLP_INTERPRETATION}}

---

## 6. Subgroup Analysis

### 6.1 Group Average Treatment Effects (GATES)

| Quintile | N | Predicted CATE | Actual ATE | 95% CI |
|----------|---|----------------|------------|--------|
{{GATES_TABLE}}

### 6.2 High-Benefit Subgroup Profile

Individuals in the top quartile of treatment benefit have the following characteristics:

| Variable | Top Quartile Mean | Rest Mean | Std Diff |
|----------|-------------------|-----------|----------|
{{HIGH_BENEFIT_PROFILE}}

---

## 7. Policy Recommendations

### 7.1 Optimal Treatment Policy

| Metric | Value |
|--------|-------|
| Policy Method | {{POLICY_METHOD}} |
| Treatment Cost | {{TREATMENT_COST}} |
| Budget Constraint | {{BUDGET_CONSTRAINT}} |
| Optimal Treatment Rate | {{OPTIMAL_TREATMENT_RATE}} |
| Policy Value | {{POLICY_VALUE}} |
| Improvement over Treat-All | {{IMPROVEMENT_TREAT_ALL}} |
| Improvement over Treat-None | {{IMPROVEMENT_TREAT_NONE}} |

### 7.2 Policy Rules

{{POLICY_RULES}}

### 7.3 Target Population

Based on the analysis, treatment should be prioritized for individuals with:

{{TARGET_POPULATION_DESCRIPTION}}

---

## 8. Robustness Checks

### 8.1 Calibration Test

| Component | Estimate | Std Error | Interpretation |
|-----------|----------|-----------|----------------|
| Mean Forest Prediction | {{CAL_MEAN}} | {{CAL_MEAN_SE}} | {{CAL_MEAN_INTERP}} |
| Differential Prediction | {{CAL_DIFF}} | {{CAL_DIFF_SE}} | {{CAL_DIFF_INTERP}} |

### 8.2 Sensitivity Analysis

{{SENSITIVITY_RESULTS}}

### 8.3 Alternative Specifications

{{ALTERNATIVE_SPECS}}

---

## 9. Limitations

1. **Unverifiable Assumptions**: The unconfoundedness assumption cannot be directly tested. Results are valid only if all confounders are observed and controlled.

2. **External Validity**: Results apply to the study population. Generalization to other populations requires careful consideration of distributional differences.

3. **CATE Uncertainty**: Individual CATE estimates have higher uncertainty than the ATE. Recommendations should focus on subgroups rather than individuals.

4. **Model Dependence**: Results may vary with different hyperparameter choices. Robustness checks are recommended.

{{ADDITIONAL_LIMITATIONS}}

---

## 10. Conclusions

{{CONCLUSIONS}}

---

## Appendix

### A. Technical Details

#### A.1 Data Preprocessing

{{DATA_PREPROCESSING}}

#### A.2 Missing Data Handling

{{MISSING_DATA}}

#### A.3 Variable Transformations

{{TRANSFORMATIONS}}

### B. Supplementary Tables

#### B.1 Full CATE Summary by Covariate

{{FULL_CATE_TABLE}}

#### B.2 Policy Sensitivity Analysis

{{POLICY_SENSITIVITY}}

### C. Supplementary Figures

- `cate_distribution.png` - CATE distribution histogram and violin plot
- `cate_sorted.png` - Sorted CATEs with confidence intervals
- `variable_importance.png` - Variable importance bar chart
- `partial_dependence.png` - Partial dependence plots for top variables
- `gates.png` - Group Average Treatment Effects plot
- `policy_analysis.png` - Policy analysis summary

---

## References

1. Wager, S., & Athey, S. (2018). Estimation and Inference of Heterogeneous Treatment Effects using Random Forests. *Journal of the American Statistical Association*, 113(523), 1228-1242.

2. Athey, S., Tibshirani, J., & Wager, S. (2019). Generalized Random Forests. *Annals of Statistics*, 47(2), 1148-1178.

3. Athey, S., & Wager, S. (2021). Policy Learning with Observational Data. *Econometrica*, 89(1), 133-161.

4. Chernozhukov, V., et al. (2018). Generic Machine Learning Inference on Heterogeneous Treatment Effects in Randomized Experiments.

---

*Report generated using Causal Forest Analysis Toolkit*
*For questions or issues, contact: {{CONTACT}}*
