# Data Quality Report Template

**Dataset**: [Dataset Name]
**Date**: [Report Date]
**Analyst**: [Name]
**Purpose**: [Brief description of analysis purpose]

---

## Executive Summary

| Metric | Value | Status |
|--------|-------|--------|
| Total Observations | [N] | - |
| Complete Cases | [N] ([%]%) | [Good/Warning/Critical] |
| Treatment Rate | [%]% | - |
| Missing Rate | [%]% | [Good/Warning/Critical] |
| Imbalanced Covariates | [N] | [Good/Warning/Critical] |

**Overall Assessment**: [Summary statement about data quality and readiness for causal inference]

---

## 1. Sample Characteristics

### 1.1 Sample Size

| Group | N | Percentage |
|-------|---|------------|
| Total | | |
| Treated | | |
| Control | | |

### 1.2 Variables

| Variable | Type | Role | Notes |
|----------|------|------|-------|
| [outcome_var] | Continuous/Binary | Outcome | |
| [treatment_var] | Binary | Treatment | |
| [control_1] | [Type] | Control | |
| [control_2] | [Type] | Control | |
| ... | | | |

---

## 2. Missing Data Analysis

### 2.1 Overall Missingness

| Metric | Value |
|--------|-------|
| Total cells with missing | [N] |
| Overall missing rate | [%]% |
| Variables with missing | [N] of [Total] |
| Complete cases | [N] ([%]%) |

### 2.2 Missing by Variable

| Variable | N Missing | % Missing | Recommendation |
|----------|-----------|-----------|----------------|
| [var1] | | | [Keep/Impute/Investigate] |
| [var2] | | | |
| ... | | | |

### 2.3 Missing by Treatment Group

| Variable | % Missing (Treated) | % Missing (Control) | Difference | Concern? |
|----------|---------------------|---------------------|------------|----------|
| [var1] | | | | [Yes/No] |
| [var2] | | | | |

**Assessment**: [Summary of whether differential missingness is a concern for causal identification]

### 2.4 Missingness Mechanism

- [ ] MCAR likely (Little's test p-value > 0.05)
- [ ] MAR suspected (missingness correlates with observed variables)
- [ ] MNAR possible (describe reasons)

**Recommended Handling**: [Listwise deletion / Mean imputation / Multiple imputation / Other]

---

## 3. Outlier Analysis

### 3.1 Detection Results

| Method | N Outliers | % of Sample | Key Variables Affected |
|--------|------------|-------------|----------------------|
| IQR (1.5x) | | | |
| Z-score (3 SD) | | | |
| Multivariate | | | |

### 3.2 Outlier Characterization

For each major outlier group:

**Group 1: [Description]**
- N observations: [N]
- Characteristics: [Key differences from main sample]
- Treatment distribution: [N treated, N control]
- Recommendation: [Keep/Remove/Investigate]

### 3.3 Sensitivity Assessment

| Specification | Treatment Effect | SE | 95% CI |
|---------------|-----------------|----|----|
| With all data | | | |
| Excluding IQR outliers | | | |
| Excluding multivariate outliers | | | |

**Assessment**: [Are results sensitive to outlier handling?]

---

## 4. Covariate Balance

### 4.1 Balance Summary

| Status | N Variables |
|--------|-------------|
| Well-balanced (|SMD| < 0.1) | |
| Moderately imbalanced (0.1-0.25) | |
| Severely imbalanced (> 0.25) | |

### 4.2 Detailed Balance Statistics

| Variable | Mean (Treated) | Mean (Control) | SMD | Status |
|----------|---------------|----------------|-----|--------|
| [var1] | | | | |
| [var2] | | | | |
| ... | | | | |

### 4.3 Balance Visualization

[Include Love plot or balance table visualization]

**Assessment**: [Summary of balance concerns and recommended adjustments]

---

## 5. Distribution Analysis

### 5.1 Outcome Variable

| Statistic | Overall | Treated | Control |
|-----------|---------|---------|---------|
| Mean | | | |
| SD | | | |
| Median | | | |
| Min | | | |
| Max | | | |
| Skewness | | | |

**Distribution Type**: [Normal / Skewed / Bimodal / Other]
**Transformation Needed**: [None / Log / Square root / Other]

### 5.2 Key Covariates

For variables requiring transformation:

| Variable | Original Skewness | After Log | After Sqrt | Recommendation |
|----------|-------------------|-----------|------------|----------------|
| [var1] | | | | |
| [var2] | | | | |

---

## 6. Overlap Assessment

### 6.1 Propensity Score Distribution

| Group | Min PS | Max PS | Mean PS |
|-------|--------|--------|---------|
| Treated | | | |
| Control | | | |

### 6.2 Overlap Region

- **Overlap range**: [min, max]
- **Units in overlap**: [N] ([%]%)
- **Units outside overlap**: [N] ([%]%)

### 6.3 Overlap Visualization

[Include propensity score histogram or density plot]

**Assessment**: [Good overlap / Moderate concern / Severe lack of overlap]

**Recommendation**:
- [ ] Proceed with full sample
- [ ] Trim to common support
- [ ] Use alternative estimand (ATT vs ATE)
- [ ] Consider different identification strategy

---

## 7. Data Quality Issues

### 7.1 Duplicates

| Type | N Found | Action Taken |
|------|---------|--------------|
| Exact duplicates | | |
| Key duplicates | | |

### 7.2 Data Entry Errors

| Issue | N Affected | Resolution |
|-------|------------|------------|
| [Describe issue] | | |
| | | |

### 7.3 Logical Inconsistencies

| Issue | N Affected | Resolution |
|-------|------------|------------|
| [e.g., negative ages] | | |
| | | |

---

## 8. Pre-Treatment Variable Verification

### 8.1 Variable Timing

| Variable | Timing | Include as Control? | Notes |
|----------|--------|---------------------|-------|
| [var1] | Pre-treatment | Yes | |
| [var2] | Post-treatment | No | Potential mediator |
| [var3] | Unknown | Investigate | |

### 8.2 Potential Bad Controls

| Variable | Concern | Decision |
|----------|---------|----------|
| [var1] | Possible mediator | Exclude |
| [var2] | Possible collider | Exclude |

---

## 9. Preprocessing Decisions

### 9.1 Final Preprocessing Pipeline

1. **Missing Values**: [Strategy and rationale]
2. **Outliers**: [Strategy and rationale]
3. **Transformations**: [Variables transformed and method]
4. **Encoding**: [Categorical encoding approach]
5. **Scaling**: [Standardization approach]

### 9.2 Variables Excluded

| Variable | Reason for Exclusion |
|----------|---------------------|
| [var1] | Post-treatment |
| [var2] | Too much missing data |
| [var3] | Not pre-treatment |

### 9.3 Final Analysis Sample

| Metric | Value |
|--------|-------|
| Starting observations | |
| After missing handling | |
| After outlier removal | |
| Final sample size | |
| % of original retained | |

---

## 10. Recommendations

### 10.1 For Causal Analysis

1. **Primary recommendation**: [Main approach]
2. **Robustness checks**: [List of sensitivity analyses]
3. **Limitations**: [Key data quality concerns]

### 10.2 Sensitivity Analyses to Report

- [ ] Analysis with/without outliers
- [ ] Analysis with different missing data handling
- [ ] Analysis trimmed to common support
- [ ] Analysis with alternative control sets

---

## Appendix

### A. Code for Reproducing This Report

```python
# Data loading
import pandas as pd
df = pd.read_csv('[data_path]')

# Run diagnostics
from scripts.diagnose_data_quality import main
# python diagnose_data_quality.py [input] --treatment D --outcome Y --report-path report.json

# Generate visualizations
# python visualize_distributions.py [input] --output-dir plots/ --treatment D --outcome Y
```

### B. Session Information

- Python version: [version]
- Key packages: pandas [version], numpy [version], scikit-learn [version]
- Date generated: [date]

### C. Additional Figures

[Include any additional diagnostic plots]

---

**Approval**

| Role | Name | Date | Signature |
|------|------|------|-----------|
| Analyst | | | |
| Reviewer | | | |
