# Data Dictionary

## {{DATASET_NAME}}

**Version:** {{VERSION}}
**Last Updated:** {{LAST_UPDATED}}
**Maintainer:** {{MAINTAINER}}

---

## Overview

| Attribute | Value |
|-----------|-------|
| **Description** | {{DESCRIPTION}} |
| **Source** | {{SOURCE}} |
| **Time Period** | {{TIME_PERIOD}} |
| **Unit of Observation** | {{UNIT_OF_OBS}} |
| **Sample Size** | {{N_OBS}} observations |
| **File Format** | {{FILE_FORMAT}} |
| **File Size** | {{FILE_SIZE}} |
| **Encoding** | {{ENCODING}} |

---

## Quick Reference

### Key Variables

| Variable | Type | Role | Description |
|----------|------|------|-------------|
| `{{OUTCOME_VAR}}` | {{OUTCOME_TYPE}} | Outcome (Y) | {{OUTCOME_DESC}} |
| `{{TREATMENT_VAR}}` | {{TREATMENT_TYPE}} | Treatment (D) | {{TREATMENT_DESC}} |
| `{{ID_VAR}}` | {{ID_TYPE}} | Identifier | {{ID_DESC}} |
| `{{TIME_VAR}}` | {{TIME_TYPE}} | Time | {{TIME_DESC}} |

### Data Structure

```
Panel Structure:
  - Units ({{ID_VAR}}):     {{N_UNITS}}
  - Time periods ({{TIME_VAR}}): {{N_PERIODS}}
  - Balance:                {{BALANCE_STATUS}}
```

---

## Variable Definitions

### Identifiers

| Variable | Type | Description | Example | Missing |
|----------|------|-------------|---------|---------|
| `{{ID_VAR}}` | {{ID_TYPE}} | {{ID_FULL_DESC}} | {{ID_EXAMPLE}} | {{ID_MISSING}}% |
| `{{TIME_VAR}}` | {{TIME_TYPE}} | {{TIME_FULL_DESC}} | {{TIME_EXAMPLE}} | {{TIME_MISSING}}% |

---

### Outcome Variables

#### `{{OUTCOME_1}}`

| Attribute | Value |
|-----------|-------|
| **Type** | {{OUT1_TYPE}} |
| **Description** | {{OUT1_DESC}} |
| **Unit** | {{OUT1_UNIT}} |
| **Range** | [{{OUT1_MIN}}, {{OUT1_MAX}}] |
| **Mean (SD)** | {{OUT1_MEAN}} ({{OUT1_SD}}) |
| **Missing** | {{OUT1_MISSING}}% ({{OUT1_MISSING_N}} obs) |
| **Source** | {{OUT1_SOURCE}} |
| **Notes** | {{OUT1_NOTES}} |

**Distribution:**
```
Min:     {{OUT1_MIN}}
P25:     {{OUT1_P25}}
Median:  {{OUT1_P50}}
P75:     {{OUT1_P75}}
Max:     {{OUT1_MAX}}
```

#### `{{OUTCOME_2}}`

| Attribute | Value |
|-----------|-------|
| **Type** | {{OUT2_TYPE}} |
| **Description** | {{OUT2_DESC}} |
| **Unit** | {{OUT2_UNIT}} |
| **Range** | [{{OUT2_MIN}}, {{OUT2_MAX}}] |
| **Mean (SD)** | {{OUT2_MEAN}} ({{OUT2_SD}}) |
| **Missing** | {{OUT2_MISSING}}% |
| **Source** | {{OUT2_SOURCE}} |

---

### Treatment Variables

#### `{{TREATMENT_1}}`

| Attribute | Value |
|-----------|-------|
| **Type** | {{TRT1_TYPE}} |
| **Description** | {{TRT1_DESC}} |
| **Coding** | {{TRT1_CODING}} |
| **Treatment Group (D=1)** | {{TRT1_N_TREAT}} ({{TRT1_PCT_TREAT}}%) |
| **Control Group (D=0)** | {{TRT1_N_CTRL}} ({{TRT1_PCT_CTRL}}%) |
| **Missing** | {{TRT1_MISSING}}% |
| **Source** | {{TRT1_SOURCE}} |

**Value Labels:**
| Value | Label | N | % |
|-------|-------|---|---|
| 0 | {{TRT1_LABEL_0}} | {{TRT1_N_0}} | {{TRT1_PCT_0}} |
| 1 | {{TRT1_LABEL_1}} | {{TRT1_N_1}} | {{TRT1_PCT_1}} |

#### `{{TREATMENT_2}}` (Continuous/Intensity)

| Attribute | Value |
|-----------|-------|
| **Type** | {{TRT2_TYPE}} |
| **Description** | {{TRT2_DESC}} |
| **Unit** | {{TRT2_UNIT}} |
| **Range** | [{{TRT2_MIN}}, {{TRT2_MAX}}] |
| **Mean (SD)** | {{TRT2_MEAN}} ({{TRT2_SD}}) |

---

### Control Variables

#### Demographics

| Variable | Type | Description | Values/Range | Mean (SD) | Missing |
|----------|------|-------------|--------------|-----------|---------|
| `age` | Continuous | Age in years | [{{AGE_MIN}}, {{AGE_MAX}}] | {{AGE_MEAN}} ({{AGE_SD}}) | {{AGE_MISS}}% |
| `female` | Binary | 1 = Female | {0, 1} | {{FEM_MEAN}} | {{FEM_MISS}}% |
| `education` | Continuous | Years of schooling | [{{EDU_MIN}}, {{EDU_MAX}}] | {{EDU_MEAN}} ({{EDU_SD}}) | {{EDU_MISS}}% |
| `married` | Binary | 1 = Married | {0, 1} | {{MAR_MEAN}} | {{MAR_MISS}}% |
| `race` | Categorical | Race/ethnicity | See labels | --- | {{RACE_MISS}}% |

**`race` Value Labels:**
| Value | Label | N | % |
|-------|-------|---|---|
| 1 | {{RACE_1}} | {{RACE_N_1}} | {{RACE_PCT_1}} |
| 2 | {{RACE_2}} | {{RACE_N_2}} | {{RACE_PCT_2}} |
| 3 | {{RACE_3}} | {{RACE_N_3}} | {{RACE_PCT_3}} |
| 4 | {{RACE_4}} | {{RACE_N_4}} | {{RACE_PCT_4}} |

#### Economic

| Variable | Type | Description | Values/Range | Mean (SD) | Missing |
|----------|------|-------------|--------------|-----------|---------|
| `income` | Continuous | Annual income ($) | [{{INC_MIN}}, {{INC_MAX}}] | {{INC_MEAN}} ({{INC_SD}}) | {{INC_MISS}}% |
| `ln_income` | Continuous | Log income | [{{LINC_MIN}}, {{LINC_MAX}}] | {{LINC_MEAN}} ({{LINC_SD}}) | {{LINC_MISS}}% |
| `employed` | Binary | 1 = Employed | {0, 1} | {{EMP_MEAN}} | {{EMP_MISS}}% |
| `wealth_index` | Continuous | Wealth index (standardized) | [{{WLTH_MIN}}, {{WLTH_MAX}}] | 0.00 (1.00) | {{WLTH_MISS}}% |

#### Geographic

| Variable | Type | Description | Values | N Categories | Missing |
|----------|------|-------------|--------|--------------|---------|
| `region` | Categorical | Geographic region | See labels | {{REG_NCAT}} | {{REG_MISS}}% |
| `urban` | Binary | 1 = Urban residence | {0, 1} | 2 | {{URB_MISS}}% |
| `state` | Categorical | State FIPS code | 01-56 | {{STATE_NCAT}} | {{STATE_MISS}}% |

---

### Instrumental Variables (if applicable)

#### `{{IV_1}}`

| Attribute | Value |
|-----------|-------|
| **Type** | {{IV1_TYPE}} |
| **Description** | {{IV1_DESC}} |
| **Relevance** | {{IV1_RELEVANCE}} |
| **Exclusion Argument** | {{IV1_EXCLUSION}} |
| **Source** | {{IV1_SOURCE}} |
| **Range** | [{{IV1_MIN}}, {{IV1_MAX}}] |
| **Mean (SD)** | {{IV1_MEAN}} ({{IV1_SD}}) |
| **First-Stage F** | {{IV1_F}} |

---

### Panel Structure Variables

| Variable | Type | Description | Values |
|----------|------|-------------|--------|
| `{{POST_VAR}}` | Binary | 1 = Post-treatment period | {0, 1} |
| `{{TREAT_GROUP_VAR}}` | Binary | 1 = Treatment group | {0, 1} |
| `{{DID_VAR}}` | Binary | Treatment $\times$ Post | {0, 1} |
| `{{COHORT_VAR}}` | Categorical | Treatment cohort (year) | {{COHORT_VALUES}} |
| `{{REL_TIME_VAR}}` | Integer | Periods relative to treatment | [{{REL_MIN}}, {{REL_MAX}}] |

---

### Derived Variables

| Variable | Formula | Description |
|----------|---------|-------------|
| `{{DERIVED_1}}` | {{FORMULA_1}} | {{DERIVED_1_DESC}} |
| `{{DERIVED_2}}` | {{FORMULA_2}} | {{DERIVED_2_DESC}} |
| `{{INDEX_VAR}}` | {{INDEX_FORMULA}} | {{INDEX_DESC}} |

**Index Construction Details:**

```
{{INDEX_VAR}} = standardize({{INDEX_COMPONENTS}})

Method: {{INDEX_METHOD}}
Components:
  - {{COMP_1}} (weight: {{WEIGHT_1}})
  - {{COMP_2}} (weight: {{WEIGHT_2}})
  - {{COMP_3}} (weight: {{WEIGHT_3}})

Cronbach's alpha: {{CRONBACH}}
```

---

## Data Quality

### Missing Data

| Variable | N Missing | % Missing | Pattern |
|----------|-----------|-----------|---------|
| {{VAR_1}} | {{MISS_N_1}} | {{MISS_PCT_1}} | {{MISS_PAT_1}} |
| {{VAR_2}} | {{MISS_N_2}} | {{MISS_PCT_2}} | {{MISS_PAT_2}} |
| {{VAR_3}} | {{MISS_N_3}} | {{MISS_PCT_3}} | {{MISS_PAT_3}} |

**Missing Data Handling:**
- {{MISSING_HANDLING}}

### Outliers

| Variable | Outlier Definition | N Outliers | % | Treatment |
|----------|-------------------|------------|---|-----------|
| `{{VAR_1}}` | {{OUT_DEF_1}} | {{OUT_N_1}} | {{OUT_PCT_1}} | {{OUT_TRT_1}} |
| `{{VAR_2}}` | {{OUT_DEF_2}} | {{OUT_N_2}} | {{OUT_PCT_2}} | {{OUT_TRT_2}} |

### Data Validation

| Check | Result | Notes |
|-------|--------|-------|
| Duplicate IDs | {{DUP_RESULT}} | {{DUP_NOTES}} |
| Range validation | {{RANGE_RESULT}} | {{RANGE_NOTES}} |
| Cross-variable consistency | {{CONSIST_RESULT}} | {{CONSIST_NOTES}} |
| Panel balance | {{BALANCE_RESULT}} | {{BALANCE_NOTES}} |

---

## Sample Construction

### Initial Sample

```
Source data:                          N = {{N_SOURCE}}
```

### Sample Restrictions

| Step | Restriction | N Dropped | N Remaining | % of Initial |
|------|-------------|-----------|-------------|--------------|
| 1 | {{RESTRICT_1}} | {{DROP_1}} | {{REMAIN_1}} | {{PCT_1}}% |
| 2 | {{RESTRICT_2}} | {{DROP_2}} | {{REMAIN_2}} | {{PCT_2}}% |
| 3 | {{RESTRICT_3}} | {{DROP_3}} | {{REMAIN_3}} | {{PCT_3}}% |
| 4 | {{RESTRICT_4}} | {{DROP_4}} | {{REMAIN_4}} | {{PCT_4}}% |
| **Final** | --- | --- | **{{N_FINAL}}** | **{{PCT_FINAL}}%** |

---

## Data Sources

### Primary Sources

| Source | Variables | Access | Citation |
|--------|-----------|--------|----------|
| {{SOURCE_1}} | {{VARS_1}} | {{ACCESS_1}} | {{CITE_1}} |
| {{SOURCE_2}} | {{VARS_2}} | {{ACCESS_2}} | {{CITE_2}} |

### Merge Details

| Merge | Key Variables | Match Rate | Notes |
|-------|---------------|------------|-------|
| {{MERGE_1}} | {{KEY_1}} | {{RATE_1}} | {{MERGE_NOTES_1}} |
| {{MERGE_2}} | {{KEY_2}} | {{RATE_2}} | {{MERGE_NOTES_2}} |

---

## Code Reference

### Variable Creation

```python
# Example: Creating treatment indicator
df['treated'] = (df['{{TREAT_CONDITION}}']) * 1

# Example: Creating outcome variable
df['{{OUTCOME_VAR}}'] = {{OUTCOME_FORMULA}}

# Example: Creating index
from sklearn.preprocessing import StandardScaler
df['{{INDEX_VAR}}'] = StandardScaler().fit_transform(
    df[{{INDEX_COMPONENTS}}].mean(axis=1).values.reshape(-1, 1)
)
```

### Data Loading

```python
import pandas as pd

# Load main analysis file
df = pd.read_parquet('{{DATA_PATH}}')

# Verify dimensions
assert df.shape == ({{N_OBS}}, {{N_VARS}}), "Data dimensions mismatch"

# Check key variables
required_vars = ['{{ID_VAR}}', '{{TIME_VAR}}', '{{OUTCOME_VAR}}', '{{TREATMENT_VAR}}']
assert all(v in df.columns for v in required_vars), "Missing required variables"
```

---

## Version History

| Version | Date | Changes | Author |
|---------|------|---------|--------|
| {{V1}} | {{DATE_1}} | {{CHANGES_1}} | {{AUTHOR_1}} |
| {{V2}} | {{DATE_2}} | {{CHANGES_2}} | {{AUTHOR_2}} |

---

## Contact

For questions about this data dictionary:
- **Email:** {{CONTACT_EMAIL}}
- **Repository:** {{REPO_URL}}

---

*Data dictionary generated using the Causal ML Skills framework.*
*Template version: 1.0.0*

---

## Template Usage Instructions

### Required Sections

At minimum, include:
1. Overview
2. Key Variables (Outcome, Treatment, ID)
3. Variable Definitions for all analysis variables
4. Sample Construction

### Variable Documentation Standards

For each variable, document:
- **Name**: Exact column name in dataset
- **Type**: Continuous, Binary, Categorical, Date, String
- **Description**: Plain language explanation
- **Values/Range**: Possible values or numeric range
- **Missing**: % missing and handling approach
- **Source**: Original data source

### Best Practices

1. **Be explicit** about units (dollars, years, etc.)
2. **Document transformations** (log, standardization)
3. **Include value labels** for all categorical variables
4. **Note any top/bottom coding** or winsorization
5. **Specify missing value codes** (., -9, NA, etc.)
6. **Document merge keys** and match rates
7. **Include code snippets** for complex derivations

### Export Options

- Convert to PDF using pandoc
- Generate codebook in Stata format
- Create interactive HTML documentation
