# Documentation Checklist for Replication Packages

> **Document Type**: Reference | **Last Updated**: 2025-01
> **Related**: [replication_standards.md](replication_standards.md), [data_management.md](data_management.md), [code_organization.md](code_organization.md)

## Overview

This checklist ensures replication packages meet the documentation standards required by major economics journals and best practices for computational reproducibility. Use this before submitting to a journal or sharing research code.

---

## Master Checklist

### Required Documentation

- [ ] **README.md** - Main documentation file
- [ ] **LICENSE** - Code and data license
- [ ] **Data Availability Statement** - How to access data
- [ ] **Computational Requirements** - Software, hardware, runtime

### Recommended Documentation

- [ ] **CITATION.cff** - Machine-readable citation
- [ ] **Variable Dictionary** - All variables documented
- [ ] **Codebook** - For survey/administrative data
- [ ] **Analysis Notes** - Methodological decisions

---

## README Requirements

### Essential Sections

#### 1. Title and Authors

```markdown
# Replication Package: [Paper Title]

**Authors:** [Author 1], [Author 2], [Author 3]

**Journal:** [Journal Name]

**DOI:** [Paper DOI if available]

**Last Updated:** [Date]
```

#### 2. Overview

```markdown
## Overview

This replication package contains the data and code required to reproduce
all tables and figures in "[Paper Title]."

### Contents

| Folder | Description |
|--------|-------------|
| `code/` | All analysis scripts |
| `data/` | Data files (see Data Availability) |
| `output/` | Generated tables and figures |
| `docs/` | Additional documentation |
```

#### 3. Data Availability Statement

```markdown
## Data Availability

### Included Data

The following data are included in this replication package:

| File | Description | Source | License |
|------|-------------|--------|---------|
| `data/raw/public_data.csv` | [Description] | [URL] | Public Domain |

### Restricted/External Data

The following data must be obtained separately:

| Dataset | Source | Access | Instructions |
|---------|--------|--------|--------------|
| [Name] | [Organization] | [Application URL] | See `docs/data_access.md` |

### Simulated Data

For code verification, simulated data with similar properties is provided:
- `data/simulated/sim_data.csv`

This simulated data allows running all code but will not reproduce exact results.
```

#### 4. Computational Requirements

```markdown
## Computational Requirements

### Software

- **Python**: 3.10 or higher
- **R**: 4.2 or higher (optional, for specific robustness checks)
- **Stata**: 17 or higher (optional, for comparison)

### Python Packages

Install required packages:
```bash
pip install -r requirements.txt
```

Key dependencies (see `requirements.txt` for full list):
- pandas >= 2.0.0
- numpy >= 1.24.0
- statsmodels >= 0.14.0
- econml >= 0.14.0

### Hardware

- **Memory**: Minimum 8GB RAM (16GB recommended)
- **Storage**: 5GB free disk space
- **Processor**: Standard laptop/desktop sufficient

### Runtime

| Step | Approximate Time |
|------|------------------|
| Data cleaning | 5 minutes |
| Main analysis | 30 minutes |
| Robustness checks | 45 minutes |
| Bootstrap inference | 60 minutes |
| **Total** | **~2.5 hours** |

*Tested on: [Machine specs, e.g., MacBook Pro M1, 16GB RAM]*
```

#### 5. Instructions

```markdown
## Instructions

### Quick Start

```bash
# Clone repository
git clone https://github.com/[username]/[repo].git
cd [repo]

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run replication (all steps)
python code/00_master.py
```

### Step-by-Step

1. **Set up environment** (see Computational Requirements)

2. **Obtain data** (see Data Availability)
   - Download public data using `python code/01_download_data.py`
   - Place restricted data in `data/raw/`

3. **Run analysis**
   ```bash
   python code/00_master.py
   ```

4. **Check output**
   - Tables: `output/tables/`
   - Figures: `output/figures/`
   - Logs: `output/logs/`

### Running Individual Steps

To run specific analysis steps:

```bash
# Data cleaning only
python code/02_clean_data.py

# Main analysis only
python code/04_analysis_main.py

# Tables only
python code/06_tables.py --skip-analysis
```
```

#### 6. Output Mapping

```markdown
## Output Mapping

### Tables

| Table | Script | Output File |
|-------|--------|-------------|
| Table 1 | `06_tables.py` | `output/tables/table_1.tex` |
| Table 2 | `06_tables.py` | `output/tables/table_2.tex` |
| Table 3 | `06_tables.py` | `output/tables/table_3.tex` |
| Table A1 | `06_tables.py` | `output/tables/table_a1.tex` |

### Figures

| Figure | Script | Output File |
|--------|--------|-------------|
| Figure 1 | `07_figures.py` | `output/figures/figure_1.pdf` |
| Figure 2 | `07_figures.py` | `output/figures/figure_2.pdf` |
| Figure A1 | `07_figures.py` | `output/figures/figure_a1.pdf` |

### In-Text Statistics

Key statistics referenced in the paper are computed in `04_analysis_main.py`
and logged to `output/logs/statistics.txt`.
```

#### 7. Contact

```markdown
## Contact

For questions about the code or data:

- **Corresponding Author:** [Name] ([email])
- **GitHub Issues:** [Repository Issues URL]

For data access questions:
- **[Data Provider]:** [Contact Information]
```

---

## Variable Dictionary Template

### Structure

```markdown
# Variable Dictionary

## Dataset: [Dataset Name]

**File:** `data/processed/analysis_sample.csv`
**Observations:** [N]
**Last Updated:** [Date]

## Variables

### Outcome Variables

| Variable | Type | Description | Source | Notes |
|----------|------|-------------|--------|-------|
| `earnings` | continuous | Annual earnings in USD | Survey Q15 | Top-coded at 99th percentile |
| `employed` | binary | Employed (=1) or not (=0) | Survey Q12 | Based on work in past week |

### Treatment Variables

| Variable | Type | Description | Source | Notes |
|----------|------|-------------|--------|-------|
| `treat` | binary | Treatment assignment | Randomization | Intent-to-treat |
| `takeup` | binary | Actually received treatment | Admin records | For IV analysis |

### Control Variables

| Variable | Type | Description | Source | Notes |
|----------|------|-------------|--------|-------|
| `age` | continuous | Age in years | Survey Q2 | At baseline |
| `female` | binary | Female (=1) | Survey Q3 | |
| `education` | categorical | Education level (1-5) | Survey Q8 | See codebook for levels |

### Identifiers

| Variable | Type | Description | Notes |
|----------|------|-------------|-------|
| `id` | integer | Unique individual identifier | Anonymized |
| `wave` | integer | Survey wave (1-3) | |
| `region` | categorical | Geographic region (1-10) | |
```

### Detailed Variable Documentation

```markdown
## Variable: earnings

### Definition
Annual labor earnings in 2020 US dollars.

### Construction
```python
# From survey question Q15: "What was your total income from
# wages and salaries in the past 12 months?"

# Adjustments:
# 1. Converted to 2020 dollars using CPI
# 2. Top-coded at 99th percentile to reduce influence of outliers
# 3. Set to 0 for non-employed respondents

earnings = (
    survey["q15_raw"]
    * cpi_adjustment[survey["year"]]
)
earnings = earnings.clip(upper=earnings.quantile(0.99))
earnings = earnings.fillna(0) where not employed
```

### Values
- **Range:** $0 - $[max value]
- **Mean:** $[mean]
- **Median:** $[median]
- **Missing:** [N] observations ([%])

### Notes
- Includes wages, salaries, and self-employment income
- Excludes capital income, transfers, and other non-labor income
- See also: `earnings_hourly`, `earnings_monthly`
```

---

## Log Files

### Execution Log Template

```
================================================================================
REPLICATION LOG
================================================================================
Date: 2024-06-15 14:30:22
User: jsmith
Machine: MacBook Pro M1, 16GB RAM, macOS 14.1
Python: 3.10.12
Working directory: /Users/jsmith/replication_package

================================================================================
STEP 1: Data Cleaning (02_clean_data.py)
================================================================================
Start time: 14:30:22
Input file: data/raw/survey_raw.csv
  - Rows: 10,500
  - Columns: 150

Cleaning operations:
  1. Remove duplicates: 10,500 -> 10,498 (2 removed)
  2. Drop missing outcome: 10,498 -> 10,312 (186 removed)
  3. Restrict to ages 18-65: 10,312 -> 10,105 (207 removed)
  4. Handle outliers: 10,105 -> 10,003 (102 removed)

Output file: data/processed/analysis_sample.csv
  - Rows: 10,003
  - Columns: 50

End time: 14:31:45
Duration: 83 seconds

================================================================================
STEP 2: Main Analysis (04_analysis_main.py)
================================================================================
Start time: 14:31:46
Input file: data/processed/analysis_sample.csv

Analysis 1: OLS Regression
  - N: 10,003
  - R-squared: 0.342
  - Treatment effect: 1,523 (SE: 412)

Analysis 2: IV Estimation
  - N: 10,003
  - First-stage F: 42.3
  - Treatment effect: 2,145 (SE: 687)

...

================================================================================
SUMMARY
================================================================================
Total runtime: 2 hours 15 minutes
All steps completed successfully.
Output files generated in: output/
```

### Statistics Log

```markdown
# Key Statistics from Analysis

## Paper Reference: "Title" (Authors, Year)

### Sample Characteristics (Table 1)

| Statistic | Value | Script | Line |
|-----------|-------|--------|------|
| Total observations | 10,003 | 04_analysis_main.py | 45 |
| Treatment group | 4,856 | 04_analysis_main.py | 46 |
| Control group | 5,147 | 04_analysis_main.py | 47 |
| Mean age | 38.2 | 04_analysis_main.py | 52 |
| Percent female | 51.3% | 04_analysis_main.py | 53 |

### Main Results (Table 2)

| Estimate | Value | SE | p-value | Script | Line |
|----------|-------|----|---------|--------|------|
| OLS | 1,523 | 412 | 0.000 | 04_analysis_main.py | 89 |
| IV | 2,145 | 687 | 0.002 | 04_analysis_main.py | 112 |
| PSM | 1,678 | 534 | 0.002 | 04_analysis_main.py | 135 |

### In-Text Statistics

| Location | Text | Value | Script | Line |
|----------|------|-------|--------|------|
| p.12 | "The treatment increased..." | $1,523 | 04_analysis_main.py | 89 |
| p.14 | "First-stage F-statistic..." | 42.3 | 04_analysis_main.py | 108 |
```

---

## Pre-Submission Checklist

### Documentation Complete?

- [ ] README includes all required sections
- [ ] Data availability statement is accurate
- [ ] Computational requirements are complete
- [ ] Instructions can be followed by someone unfamiliar with project
- [ ] Output mapping covers all tables and figures
- [ ] Variable dictionary documents all variables

### Code Verification

- [ ] Master script runs without errors on clean machine
- [ ] All output files are generated
- [ ] Output matches manuscript tables/figures
- [ ] No hardcoded paths to specific machines
- [ ] Random seeds are set for reproducibility

### Data Verification

- [ ] All included data is properly licensed
- [ ] No personally identifiable information (PII)
- [ ] Restricted data access instructions are clear
- [ ] Data citations are complete

### Repository Ready?

- [ ] Repository is organized per standard structure
- [ ] LICENSE file is included
- [ ] .gitignore excludes appropriate files
- [ ] No large files in git (use LFS or DVC)
- [ ] Repository is public or properly shared

---

## Common Issues and Solutions

### Issue: Paths Don't Work on Different Machines

**Problem:** Code uses absolute paths like `C:/Users/jsmith/project/data/file.csv`

**Solution:**
```python
# BAD
path = "C:/Users/jsmith/project/data/file.csv"

# GOOD
from pathlib import Path
PROJECT_ROOT = Path(__file__).parent.parent
path = PROJECT_ROOT / "data" / "file.csv"
```

### Issue: Missing Package Versions

**Problem:** Code fails because package versions changed

**Solution:** Pin exact versions in `requirements.txt`:
```
pandas==2.0.3
numpy==1.24.3
statsmodels==0.14.0
```

### Issue: Results Don't Match

**Problem:** Replicated results differ from paper

**Solution:** Create verification script comparing outputs:
```python
def verify_results(generated, expected, tolerance=0.01):
    """Compare generated results to expected values."""
    for key, expected_val in expected.items():
        actual_val = generated[key]
        if abs(actual_val - expected_val) / abs(expected_val) > tolerance:
            print(f"MISMATCH: {key} - expected {expected_val}, got {actual_val}")
```

### Issue: Data Too Large for Repository

**Problem:** Data files are too large for git

**Solution:**
1. Use Git LFS for files < 2GB
2. Use DVC for larger files
3. Host on data repository (Zenodo, Dataverse)
4. Provide download script

---

## References

### AEA Requirements

- [AEA Data and Code Availability Policy](https://www.aeaweb.org/journals/data/data-code-policy)
- [AEA Data Editor Guidance](https://aeadataeditor.github.io/)
- [Template README](https://social-science-data-editors.github.io/template_README/)

### Best Practices

- Gentzkow, M., & Shapiro, J. M. (2014). "Code and Data for the Social Sciences."
- Christensen, G., & Miguel, E. (2018). "Transparency, Reproducibility, and the Credibility of Economics Research." *JEL*.

---

## See Also

- [replication_standards.md](replication_standards.md) - Journal-specific requirements
- [data_management.md](data_management.md) - Data handling best practices
- [code_organization.md](code_organization.md) - Code structure guidelines
