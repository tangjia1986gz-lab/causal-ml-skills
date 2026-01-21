# Econometric EDA Report

**Generated**: {{generated_at}}
**Dataset**: {{dataset_name}}

---

## 1. Dataset Overview

| Metric | Value |
|--------|-------|
| Observations | {{n_observations}} |
| Variables | {{n_variables}} |
| Numeric Variables | {{n_numeric}} |
| Categorical Variables | {{n_categorical}} |
| Complete Cases | {{n_complete_cases}} ({{pct_complete_cases}}%) |

### Variable Types

{{variable_types_table}}

---

## 2. Summary Statistics

### Numeric Variables

{{numeric_summary_table}}

### Categorical Variables

{{categorical_summary_table}}

---

## 3. Missing Data Analysis

### Summary

- **Variables with missing data**: {{n_vars_missing}}
- **Complete case rate**: {{pct_complete_cases}}%
- **MCAR test p-value**: {{mcar_pvalue}} ({{mcar_interpretation}})

### Missing by Variable

{{missing_summary_table}}

### Missing Data Patterns

The top 5 most common missing patterns:

{{missing_patterns_table}}

### Missing Correlations

Variables that tend to be missing together (correlation > 0.3):

{{missing_correlations}}

### Implications for Causal Inference

{{missing_implications}}

---

## 4. Outlier Analysis

### Univariate Outliers (IQR Method, k=1.5)

{{outlier_summary_table}}

### Multivariate Outliers (Mahalanobis Distance)

- **Outliers detected**: {{n_multivariate_outliers}} ({{pct_multivariate_outliers}}%)
- **Chi-square threshold (p=0.001)**: {{chi2_threshold}}

### Influential Observations (if regression specified)

{{influence_diagnostics_table}}

### Implications for Causal Inference

{{outlier_implications}}

---

## 5. Variable Relationships

### Correlation Matrix

{{correlation_matrix}}

**Significance**: * p<0.05, ** p<0.01, *** p<0.001

### High Correlations (|r| > 0.7)

{{high_correlations}}

### Multicollinearity Diagnostics

| Metric | Value | Concern |
|--------|-------|---------|
| Condition Number | {{condition_number}} | {{condition_concern}} |
| Max VIF | {{max_vif}} | {{vif_concern}} |
| Variables with VIF > 10 | {{n_high_vif}} | |

#### Variance Inflation Factors

{{vif_table}}

### Implications for Causal Inference

{{relationship_implications}}

---

## 6. Covariate Balance (Treatment vs. Control)

**Treatment variable**: {{treatment_var}}

### Balance Table

{{balance_table}}

### Standardized Differences

- **Variables with |d| > 0.1**: {{n_imbalanced}}
- **Max |standardized difference|**: {{max_std_diff}}

### Balance Interpretation

{{balance_interpretation}}

---

## 7. Panel Data Analysis

### Panel Structure

| Metric | Value |
|--------|-------|
| Entities | {{n_entities}} |
| Time Periods | {{n_periods}} |
| Panel Type | {{panel_type}} |
| Observations per Entity | {{obs_per_entity_range}} |

### Within/Between Variation Decomposition

{{variation_table}}

**Interpretation**: Variables with high between-variation (>80%) are mostly time-invariant and will be absorbed by fixed effects. Variables with high within-variation are suitable for within-estimator identification.

### Attrition Analysis

- **Initial sample (first period)**: {{initial_sample}}
- **Final sample (retained)**: {{final_sample}}
- **Overall attrition rate**: {{attrition_rate}}%

### Attrition Bias Test

Testing whether baseline characteristics differ between stayers and leavers:

{{attrition_bias_table}}

**Concern level**: {{attrition_concern}}

---

## 8. Data Quality Summary

### Issues Identified

{{data_issues_list}}

### Recommendations

{{recommendations_list}}

### Causal Inference Readiness Checklist

| Check | Status | Notes |
|-------|--------|-------|
| Missing data < 5% in key variables | {{missing_check}} | {{missing_notes}} |
| No severe multicollinearity (VIF < 10) | {{vif_check}} | {{vif_notes}} |
| Outliers addressed | {{outlier_check}} | {{outlier_notes}} |
| Covariate balance acceptable | {{balance_check}} | {{balance_notes}} |
| Panel attrition not systematic | {{attrition_check}} | {{attrition_notes}} |

---

## Appendix: Figures

### A1. Distribution Plots
![Distributions](figures/distributions.png)

### A2. Missing Data Patterns
![Missing Patterns](figures/missing_patterns.png)

### A3. Correlation Heatmap
![Correlations](figures/correlation_matrix.png)

### A4. Covariate Balance
![Balance](figures/balance_plot.png)

### A5. Panel Structure
![Panel](figures/panel_structure.png)

---

*Report generated using the Econometric EDA Skill*
*For causal inference research applications*
