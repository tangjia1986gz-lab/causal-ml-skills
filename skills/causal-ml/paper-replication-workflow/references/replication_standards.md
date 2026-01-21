# Replication Standards and Journal Requirements

> **Document Type**: Reference | **Last Updated**: 2025-01
> **Related**: [data_management.md](data_management.md), [code_organization.md](code_organization.md), [documentation_checklist.md](documentation_checklist.md)

## Overview

This document outlines standards for replication packages based on requirements from major economics journals, particularly the American Economic Association (AEA) Data Editor requirements. Following these standards ensures computational reproducibility and increases the credibility and impact of research.

---

## AEA Data and Code Availability Policy

### Core Requirements

The AEA requires authors to provide:

1. **Replication package** containing all data and code
2. **README file** documenting data sources and instructions
3. **Data Availability Statement** describing how data can be accessed
4. **License** for code and data reuse

### Submission Checklist

Before submitting a replication package:

- [ ] All code runs from start to finish without errors
- [ ] README provides step-by-step instructions
- [ ] Data sources are documented with access information
- [ ] Computational requirements are specified
- [ ] Output matches figures and tables in the manuscript
- [ ] License file is included

### README Requirements

The README must contain:

| Section | Description | Required |
|---------|-------------|----------|
| **Data Availability** | Statement on how data can be obtained | Yes |
| **Computational Requirements** | Software, packages, hardware | Yes |
| **Instructions** | Step-by-step reproduction guide | Yes |
| **List of Tables/Figures** | Mapping code to outputs | Yes |
| **Data Citations** | Proper citations for datasets | Yes |
| **Time Requirements** | Expected runtime | Recommended |
| **Storage Requirements** | Disk space needed | Recommended |

---

## Data Availability Statement Templates

### Publicly Available Data

```markdown
## Data Availability Statement

The data used in this study are publicly available from [Source Name].
- Dataset: [Dataset Name]
- URL: [Access URL]
- Access Date: [Date data were downloaded]
- Documentation: [URL to codebook/documentation]

Instructions:
1. Navigate to [URL]
2. Download [specific files]
3. Place files in `data/raw/` directory
```

### Restricted-Access Data

```markdown
## Data Availability Statement

The data used in this study are confidential/restricted-access data from [Source].

**Access Conditions:**
- Application required: [Yes/No]
- Approval process: [Description]
- Typical timeline: [Duration]
- Contact: [Email/URL]

**Data Access Requirements:**
- Institutional affiliation: [Requirements]
- IRB approval: [Required/Not required]
- Data use agreement: [Description]

**Alternative:**
[Description of any publicly available subset or simulated data]
```

### Proprietary Data

```markdown
## Data Availability Statement

This study uses proprietary data from [Organization/Company].

**Access:**
- These data cannot be shared due to contractual restrictions
- Researchers may contact [Organization] at [contact information]
  to inquire about data access

**Replication Notes:**
- The replication package includes all code
- Synthetic data with similar properties is provided for code verification
- See `data/synthetic/` for simulated data
```

### Mixed Data Sources

```markdown
## Data Availability Statement

This study combines data from multiple sources:

| Dataset | Source | Access | Instructions |
|---------|--------|--------|--------------|
| [Name 1] | [Source] | Public | Download from [URL] |
| [Name 2] | [Source] | Restricted | Apply via [URL] |
| [Name 3] | Constructed | Included | See `data/constructed/` |

Public data download script: `code/1_download_data.py`
```

---

## Software and Environment Documentation

### Required Information

```markdown
## Computational Requirements

### Software Requirements
- Python: 3.10+
- R: 4.2+
- Stata: 17+

### Python Packages
See `requirements.txt` for complete list.

Key dependencies:
- pandas >= 2.0.0
- numpy >= 1.24.0
- statsmodels >= 0.14.0
- linearmodels >= 5.0
- causalml >= 0.15.0

### Hardware Requirements
- Processor: Standard laptop/desktop sufficient
- Memory: Minimum 8GB RAM (16GB recommended)
- Storage: 2GB free disk space

### Runtime
- Total runtime: approximately 2 hours on standard hardware
- Most time-consuming step: `03_bootstrap_inference.py` (~90 minutes)
```

### Environment Specification Files

**Python (requirements.txt)**:
```
# Core dependencies
pandas>=2.0.0,<3.0.0
numpy>=1.24.0,<2.0.0
scipy>=1.10.0

# Statistical/Econometric
statsmodels>=0.14.0
linearmodels>=5.0

# Causal Inference
econml>=0.14.0
doubleml>=0.7.0
causalml>=0.15.0

# Visualization
matplotlib>=3.7.0
seaborn>=0.12.0
```

**Conda (environment.yml)**:
```yaml
name: replication_env
channels:
  - conda-forge
  - defaults
dependencies:
  - python=3.10
  - pandas>=2.0
  - numpy>=1.24
  - statsmodels>=0.14
  - pip:
    - doubleml>=0.7.0
    - causalml>=0.15.0
```

**R (renv.lock)** or **sessionInfo()**:
```r
# Include output of:
sessionInfo()
# Or use renv::snapshot() for lockfile
```

---

## Code Organization Standards

### Recommended Directory Structure

```
replication_package/
├── README.md              # Main documentation
├── LICENSE                # License file
├── CITATION.cff           # Citation file
├── data/
│   ├── raw/              # Original, immutable data
│   ├── processed/        # Cleaned data (can be regenerated)
│   └── README.md         # Data documentation
├── code/
│   ├── 00_master.py      # Master script
│   ├── 01_data_prep.py   # Data cleaning
│   ├── 02_analysis.py    # Main analysis
│   ├── 03_robustness.py  # Robustness checks
│   └── lib/              # Helper functions
├── output/
│   ├── tables/           # Generated tables
│   ├── figures/          # Generated figures
│   └── logs/             # Log files
├── manuscript/           # Paper and supplementary materials
└── docs/                 # Additional documentation
```

### Naming Conventions

**Scripts**:
- Use numbered prefixes: `01_`, `02_`, etc.
- Use descriptive names: `01_clean_data.py`, `02_main_analysis.py`
- Separate by function: data prep, analysis, visualization

**Data Files**:
- Include version/date: `survey_data_v2.csv`, `census_2020.dta`
- Indicate processing stage: `raw_`, `clean_`, `final_`
- Use lowercase with underscores

**Output Files**:
- Match paper references: `table_1.tex`, `figure_2.pdf`
- Include descriptive component: `table_1_summary_stats.tex`

---

## Master Script Requirements

### Purpose

A master script ensures one-click reproducibility:

```python
#!/usr/bin/env python
"""
Master Script for Replication Package
=====================================

Paper: [Paper Title]
Authors: [Author Names]
Journal: [Journal Name]

This script reproduces all results in the paper.

Usage:
    python 00_master.py [--skip-data] [--tables-only]

Runtime: Approximately 2 hours on standard hardware

Requirements:
    See requirements.txt for dependencies
"""

import subprocess
import sys
import time
from pathlib import Path

# Configuration
SCRIPTS = [
    ("01_download_data.py", "Downloading data"),
    ("02_clean_data.py", "Cleaning data"),
    ("03_main_analysis.py", "Running main analysis"),
    ("04_robustness.py", "Running robustness checks"),
    ("05_create_tables.py", "Generating tables"),
    ("06_create_figures.py", "Generating figures"),
]

def run_script(script_name, description):
    """Run a single script with logging."""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Script: {script_name}")
    print(f"{'='*60}")

    start_time = time.time()

    result = subprocess.run(
        [sys.executable, script_name],
        capture_output=True,
        text=True
    )

    elapsed = time.time() - start_time

    if result.returncode != 0:
        print(f"ERROR in {script_name}:")
        print(result.stderr)
        return False

    print(f"Completed in {elapsed:.1f} seconds")
    return True

def main():
    """Run all scripts in sequence."""
    print("="*60)
    print("REPLICATION PACKAGE - MASTER SCRIPT")
    print("="*60)

    total_start = time.time()
    failed = []

    for script, description in SCRIPTS:
        if not run_script(script, description):
            failed.append(script)
            print(f"WARNING: {script} failed, continuing...")

    total_elapsed = time.time() - total_start

    print("\n" + "="*60)
    print("REPLICATION COMPLETE")
    print(f"Total runtime: {total_elapsed/60:.1f} minutes")

    if failed:
        print(f"Failed scripts: {failed}")
        sys.exit(1)
    else:
        print("All scripts completed successfully!")

if __name__ == "__main__":
    main()
```

---

## Version Control and Reproducibility

### Git Best Practices

1. **Initialize repository early**
2. **Use meaningful commits**
3. **Tag releases** (submission, revision, accepted)
4. **Include `.gitignore`** for large files, outputs

**.gitignore template**:
```
# Data (too large for git)
data/raw/*.csv
data/raw/*.dta

# Output (can be regenerated)
output/tables/*.tex
output/figures/*.pdf

# Python
__pycache__/
*.pyc
.venv/

# R
.Rhistory
.RData

# Stata
*.log

# OS
.DS_Store
Thumbs.db
```

### Data Version Control

For large data files:
- Use **Git LFS** for files under 2GB
- Use **DVC** (Data Version Control) for larger files
- Document exact data versions and access dates

---

## Journal-Specific Requirements

### American Economic Review (AER)

- Follows AEA Data Editor policy
- Requires openICPSR deposit
- Embargo period available if needed

### Quarterly Journal of Economics (QJE)

- Similar to AEA requirements
- Harvard Dataverse deposit
- README template provided

### Review of Economic Studies (REStud)

- Zenodo deposit preferred
- Strict pre-acceptance code verification
- Docker containers encouraged

### Journal of Political Economy (JPE)

- University of Chicago repository
- Code verification before acceptance
- Detailed computational appendix required

### Econometrica

- Similar to REStud
- Requires pre-acceptance verification
- Supplementary materials hosted on journal website

---

## Pre-Submission Verification

### Self-Verification Checklist

Before submission, verify in a clean environment:

```bash
# Create fresh environment
conda create -n verify_replication python=3.10
conda activate verify_replication

# Install requirements
pip install -r requirements.txt

# Run master script
python code/00_master.py

# Compare outputs
diff output/tables/ expected_output/tables/
```

### Verification Script

```python
"""
Verification Script
==================
Compares generated output to expected output.
"""

import hashlib
from pathlib import Path

def file_hash(filepath):
    """Compute MD5 hash of file."""
    with open(filepath, 'rb') as f:
        return hashlib.md5(f.read()).hexdigest()

def verify_outputs(generated_dir, expected_dir):
    """Compare generated outputs to expected."""
    generated = Path(generated_dir)
    expected = Path(expected_dir)

    results = []

    for expected_file in expected.glob("**/*"):
        if expected_file.is_file():
            relative_path = expected_file.relative_to(expected)
            generated_file = generated / relative_path

            if not generated_file.exists():
                results.append((relative_path, "MISSING"))
            elif file_hash(generated_file) != file_hash(expected_file):
                results.append((relative_path, "DIFFERENT"))
            else:
                results.append((relative_path, "MATCH"))

    return results

if __name__ == "__main__":
    results = verify_outputs("output/", "expected_output/")

    for path, status in results:
        print(f"{status}: {path}")
```

---

## References

### AEA Resources

- [AEA Data and Code Availability Policy](https://www.aeaweb.org/journals/data/data-code-policy)
- [AEA Data Editor Repository](https://github.com/AEADataEditor)
- [AEA README Template](https://social-science-data-editors.github.io/template_README/)

### Best Practices Guides

- Gentzkow, M., & Shapiro, J. M. (2014). "Code and Data for the Social Sciences: A Practitioner's Guide."
- Vilhuber, L. (2020). "Reproducibility and Replicability in Economics." *Harvard Data Science Review*.
- Christensen, G., & Miguel, E. (2018). "Transparency, Reproducibility, and the Credibility of Economics Research." *Journal of Economic Literature*.

### Tools and Repositories

- [Social Science Data Editors](https://social-science-data-editors.github.io/)
- [openICPSR](https://www.openicpsr.org/)
- [Harvard Dataverse](https://dataverse.harvard.edu/)
- [Zenodo](https://zenodo.org/)

---

## See Also

- [data_management.md](data_management.md) - Data acquisition and versioning
- [code_organization.md](code_organization.md) - Code structure best practices
- [documentation_checklist.md](documentation_checklist.md) - Complete documentation requirements
