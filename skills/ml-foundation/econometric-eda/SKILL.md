---
name: econometric-eda
description: Use for exploratory data analysis in econometric research. Triggers on EDA, data exploration, data quality, missing values, outliers, distributions, correlation analysis, multicollinearity, panel data variation, attrition analysis.
---

# Econometric EDA Skill

Exploratory data analysis tailored for econometric research, focusing on data quality issues that could affect causal inference validity.

## When to Use

- Starting a new econometric analysis project
- Assessing data quality before estimation
- Investigating missing data patterns (MCAR/MAR/MNAR)
- Detecting outliers and influential observations
- Checking for multicollinearity among regressors
- Analyzing panel data structure (within/between variation)
- Generating EDA reports for research documentation

## Key Capabilities

### Data Quality Assessment
- Missing value analysis with missingness mechanism diagnosis
- Duplicate detection and data inconsistency checks
- Data type validation and range checks

### Outlier Detection
- Univariate outliers (IQR, Z-score, MAD methods)
- Multivariate outliers (Mahalanobis distance, isolation forest)
- Influential observations for regression (Cook's distance, DFBETAS)

### Variable Relationships
- Correlation analysis with significance testing
- Multicollinearity diagnostics (VIF, condition number)
- Scatterplot matrices with regression lines

### Panel Data EDA
- Within and between variation decomposition
- Attrition analysis and pattern detection
- Time series properties by entity

## Quick Start

```python
from econometric_eda import EconometricEDA

# Initialize with data
eda = EconometricEDA(df)

# Run complete EDA pipeline
report = eda.full_report(
    outcome_var='y',
    treatment_var='treatment',
    covariates=['x1', 'x2', 'x3'],
    panel_id='entity_id',
    time_var='year'
)

# Generate HTML report
eda.export_report(report, format='html', path='eda_report.html')
```

## CLI Usage

```bash
# Complete EDA pipeline
python scripts/run_eda.py --data data.csv --outcome y --treatment D --covariates x1,x2,x3

# Missing value analysis
python scripts/diagnose_missing.py --data data.csv --vars y,x1,x2,x3

# Outlier detection
python scripts/detect_outliers.py --data data.csv --vars y,x1,x2 --method mahalanobis

# Visualization suite
python scripts/visualize_eda.py --data data.csv --output figures/
```

## References

- `references/data_quality.md` - Missing values, duplicates, inconsistencies
- `references/outlier_detection.md` - Univariate and multivariate outlier methods
- `references/variable_relationships.md` - Correlations and multicollinearity
- `references/panel_eda.md` - Panel-specific EDA techniques

## Causal Inference Considerations

EDA for causal inference requires special attention to:

1. **Selection bias indicators**: Covariate balance before matching/weighting
2. **Missing data mechanisms**: MAR vs MNAR affects identification
3. **Outlier influence**: Extreme observations can drive treatment effects
4. **Multicollinearity**: Affects standard errors but not point estimates
5. **Panel attrition**: Non-random attrition threatens internal validity

## Output

The skill generates:
- Summary statistics tables
- Missing data heatmaps and pattern analysis
- Outlier diagnostic plots
- Correlation matrices with significance stars
- Panel variation decomposition tables
- Comprehensive Markdown/HTML reports
