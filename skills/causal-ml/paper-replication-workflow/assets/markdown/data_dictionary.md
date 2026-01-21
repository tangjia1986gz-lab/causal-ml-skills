# Data Dictionary

## Dataset: {{DATASET_NAME}}

**File:** `{{FILE_PATH}}`

**Description:** {{DATASET_DESCRIPTION}}

**Observations:** {{N_OBSERVATIONS}}

**Variables:** {{N_VARIABLES}}

**Last Updated:** {{DATE}}

**Source:** {{DATA_SOURCE}}

---

## Variable Summary

### Quick Reference

| Variable | Type | Description | Values/Range |
|----------|------|-------------|--------------|
{{VARIABLE_TABLE}}

---

## Detailed Variable Descriptions

### Outcome Variables

#### {{OUTCOME_VAR_1}}

| Property | Value |
|----------|-------|
| **Type** | {{TYPE}} (continuous/binary/categorical) |
| **Description** | {{DESCRIPTION}} |
| **Source** | {{SOURCE}} (e.g., Survey Q15, Admin Record) |
| **Unit** | {{UNIT}} |
| **Range** | {{MIN}} to {{MAX}} |
| **Mean** | {{MEAN}} |
| **Std. Dev.** | {{STD}} |
| **Missing** | {{N_MISSING}} ({{PCT_MISSING}}%) |

**Construction:**
```python
{{CONSTRUCTION_CODE}}
```

**Notes:**
{{NOTES}}

---

### Treatment Variables

#### {{TREATMENT_VAR}}

| Property | Value |
|----------|-------|
| **Type** | Binary |
| **Description** | {{DESCRIPTION}} |
| **Source** | {{SOURCE}} |
| **Values** | 0 = Control, 1 = Treatment |
| **N (Treatment)** | {{N_TREATED}} ({{PCT_TREATED}}%) |
| **N (Control)** | {{N_CONTROL}} ({{PCT_CONTROL}}%) |
| **Missing** | {{N_MISSING}} |

**Assignment Mechanism:**
{{ASSIGNMENT_DESCRIPTION}}

---

### Control Variables

#### {{CONTROL_VAR_1}}

| Property | Value |
|----------|-------|
| **Type** | {{TYPE}} |
| **Description** | {{DESCRIPTION}} |
| **Source** | {{SOURCE}} |
| **Range/Values** | {{VALUES}} |
| **Mean/Mode** | {{MEAN_OR_MODE}} |
| **Missing** | {{N_MISSING}} ({{PCT_MISSING}}%) |

**Notes:**
{{NOTES}}

---

#### {{CONTROL_VAR_2}}

*(Repeat structure for each control variable)*

---

### Identifier Variables

#### {{ID_VAR}}

| Property | Value |
|----------|-------|
| **Type** | {{TYPE}} (integer/string) |
| **Description** | Unique identifier for {{UNIT}} |
| **Unique Values** | {{N_UNIQUE}} |
| **Duplicates** | {{HAS_DUPLICATES}} |

**Notes:**
- {{ANONYMIZATION_NOTE}}
- {{LINKAGE_NOTE}}

---

### Time Variables

#### {{TIME_VAR}}

| Property | Value |
|----------|-------|
| **Type** | {{TYPE}} (date/integer) |
| **Description** | {{DESCRIPTION}} |
| **Range** | {{MIN_DATE}} to {{MAX_DATE}} |
| **Frequency** | {{FREQUENCY}} (daily/monthly/yearly) |
| **Missing** | {{N_MISSING}} |

---

## Value Labels

### {{CATEGORICAL_VAR}}

| Code | Label | N | Percent |
|------|-------|---|---------|
| 1 | {{LABEL_1}} | {{N_1}} | {{PCT_1}}% |
| 2 | {{LABEL_2}} | {{N_2}} | {{PCT_2}}% |
| 3 | {{LABEL_3}} | {{N_3}} | {{PCT_3}}% |
| ... | ... | ... | ... |

---

## Missing Value Coding

| Code | Meaning | Variables Using |
|------|---------|-----------------|
| `.` / `NaN` | Standard missing | All variables |
| `-9` | Refused to answer | Survey variables |
| `-8` | Don't know | Survey variables |
| `-7` | Not applicable | Conditional questions |
| `999` | Out of range | {{SPECIFIC_VARS}} |

---

## Variable Construction Notes

### Constructed Variables

#### {{CONSTRUCTED_VAR}}

**Definition:** {{DEFINITION}}

**Formula:**
```python
df['{{CONSTRUCTED_VAR}}'] = {{FORMULA}}
```

**Source Variables:** {{SOURCE_VARS}}

**Notes:** {{NOTES}}

---

### Sample Restrictions

The analysis sample was constructed with the following restrictions:

| Step | Restriction | N Before | N After | N Dropped |
|------|-------------|----------|---------|-----------|
| 1 | {{RESTRICTION_1}} | {{N_1}} | {{N_2}} | {{DROP_1}} |
| 2 | {{RESTRICTION_2}} | {{N_2}} | {{N_3}} | {{DROP_2}} |
| 3 | {{RESTRICTION_3}} | {{N_3}} | {{N_4}} | {{DROP_3}} |
| **Final** | | | **{{N_FINAL}}** | |

---

## Data Quality Notes

### Known Issues

1. **{{ISSUE_1}}:** {{DESCRIPTION_1}}
   - Affected variables: {{VARS_1}}
   - Resolution: {{RESOLUTION_1}}

2. **{{ISSUE_2}}:** {{DESCRIPTION_2}}
   - Affected variables: {{VARS_2}}
   - Resolution: {{RESOLUTION_2}}

### Validation Checks

- [ ] No duplicate identifiers
- [ ] All required variables present
- [ ] Value ranges within expected bounds
- [ ] Missing value patterns as expected
- [ ] Sample size matches documentation

---

## Summary Statistics

### Continuous Variables

| Variable | N | Mean | Std. Dev. | Min | P25 | Median | P75 | Max |
|----------|---|------|-----------|-----|-----|--------|-----|-----|
{{CONTINUOUS_SUMMARY_TABLE}}

### Categorical Variables

| Variable | N | Categories | Mode | Mode Freq |
|----------|---|------------|------|-----------|
{{CATEGORICAL_SUMMARY_TABLE}}

### Binary Variables

| Variable | N | N (=1) | Percent (=1) |
|----------|---|--------|--------------|
{{BINARY_SUMMARY_TABLE}}

---

## Cross-References

### Related Files

| File | Relationship | Key Variable |
|------|--------------|--------------|
| `{{RELATED_FILE_1}}` | {{RELATIONSHIP_1}} | {{KEY_1}} |
| `{{RELATED_FILE_2}}` | {{RELATIONSHIP_2}} | {{KEY_2}} |

### Documentation

- Original codebook: `docs/{{CODEBOOK_FILE}}`
- Survey instrument: `docs/{{SURVEY_FILE}}`
- Administrative documentation: `docs/{{ADMIN_FILE}}`

---

## Change Log

| Date | Version | Changes | Author |
|------|---------|---------|--------|
| {{DATE_1}} | 1.0 | Initial creation | {{AUTHOR_1}} |
| {{DATE_2}} | 1.1 | {{CHANGE_1}} | {{AUTHOR_2}} |

---

## References

{{REFERENCES}}

---

*This data dictionary follows best practices from the [ICPSR Guide to Social Science Data Preparation and Archiving](https://www.icpsr.umich.edu/web/pages/deposit/guide/).*
