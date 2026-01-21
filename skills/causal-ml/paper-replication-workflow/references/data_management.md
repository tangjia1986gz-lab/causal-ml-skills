# Data Management for Replication

> **Document Type**: Reference | **Last Updated**: 2025-01
> **Related**: [replication_standards.md](replication_standards.md), [code_organization.md](code_organization.md)

## Overview

Proper data management is fundamental to computational reproducibility. This document covers data acquisition, cleaning documentation, version control, and the creation of data provenance records for replication packages.

---

## Data Acquisition Best Practices

### Documentation Requirements

For every dataset used in your research, document:

| Field | Description | Example |
|-------|-------------|---------|
| **Dataset Name** | Official name | Current Population Survey (CPS) |
| **Source** | Provider organization | U.S. Census Bureau |
| **Access URL** | Where to obtain | https://www.census.gov/cps |
| **Access Date** | When downloaded | 2024-06-15 |
| **File Names** | Original filenames | cps_march_2020.dat |
| **Version** | Data version if applicable | IPUMS-CPS 2023 |
| **License** | Usage terms | Public domain |
| **Citation** | How to cite | Flood et al. (2023) |

### Download Script Template

```python
"""
Data Download Script
====================

This script downloads raw data from original sources.
Manual downloads are documented with instructions.

Usage:
    python 01_download_data.py

Output:
    data/raw/ - Downloaded data files
    data/raw/download_log.json - Download metadata
"""

import os
import json
import hashlib
import requests
from datetime import datetime
from pathlib import Path

# Configuration
RAW_DATA_DIR = Path("data/raw")
DOWNLOAD_LOG = RAW_DATA_DIR / "download_log.json"

def download_file(url, filename, description):
    """Download a file and log metadata."""
    filepath = RAW_DATA_DIR / filename

    print(f"Downloading: {description}")
    print(f"  URL: {url}")
    print(f"  Destination: {filepath}")

    response = requests.get(url, stream=True)
    response.raise_for_status()

    with open(filepath, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)

    # Compute hash for verification
    file_hash = compute_hash(filepath)

    return {
        "filename": filename,
        "description": description,
        "url": url,
        "download_date": datetime.now().isoformat(),
        "file_size_bytes": filepath.stat().st_size,
        "md5_hash": file_hash
    }

def compute_hash(filepath):
    """Compute MD5 hash of file."""
    hash_md5 = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

def main():
    """Download all data files."""
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

    download_log = []

    # Example: Download LaLonde data from package
    datasets = [
        {
            "url": "https://example.com/data/lalonde.csv",
            "filename": "lalonde.csv",
            "description": "LaLonde NSW experimental data"
        },
        # Add more datasets here
    ]

    for dataset in datasets:
        try:
            metadata = download_file(**dataset)
            download_log.append(metadata)
            print(f"  Success: {metadata['md5_hash']}")
        except Exception as e:
            print(f"  ERROR: {e}")
            download_log.append({
                **dataset,
                "error": str(e),
                "download_date": datetime.now().isoformat()
            })

    # Save download log
    with open(DOWNLOAD_LOG, 'w') as f:
        json.dump(download_log, f, indent=2)

    print(f"\nDownload log saved to: {DOWNLOAD_LOG}")

if __name__ == "__main__":
    main()
```

### Manual Download Documentation

For data requiring manual download (registration, terms acceptance):

```markdown
## Manual Download Instructions

### Dataset: [Name]

**Source**: [URL]

**Steps**:
1. Navigate to [URL]
2. Create account / log in
3. Accept data use agreement
4. Navigate to [specific page]
5. Select the following options:
   - Variable X: [value]
   - Time period: [value]
   - Format: [value]
6. Download file(s)
7. Rename to: `[standardized_name]`
8. Place in: `data/raw/`

**Expected file properties**:
- Size: approximately [X] MB
- Rows: approximately [N]
- MD5 hash: [hash value]

**Access restrictions**:
- Registration required: [Yes/No]
- Institutional affiliation: [Required/Not required]
- Data use agreement: [Link to agreement]
```

---

## Data Cleaning Documentation

### Principles

1. **Preserve raw data**: Never modify original files
2. **Document all transformations**: Every cleaning step recorded
3. **Create reproducible scripts**: Code generates cleaned data from raw
4. **Version intermediate files**: Track changes over project lifecycle

### Cleaning Log Template

```python
"""
Data Cleaning Script
====================

Input: data/raw/survey_raw.csv
Output: data/processed/survey_clean.csv

This script documents all data cleaning steps with justification.
"""

import pandas as pd
from dataclasses import dataclass
from typing import List

@dataclass
class CleaningStep:
    """Record of a single cleaning operation."""
    step_number: int
    operation: str
    justification: str
    rows_before: int
    rows_after: int
    rows_affected: int

cleaning_log: List[CleaningStep] = []

def log_step(df_before, df_after, operation, justification, step_num):
    """Log a cleaning step."""
    step = CleaningStep(
        step_number=step_num,
        operation=operation,
        justification=justification,
        rows_before=len(df_before),
        rows_after=len(df_after),
        rows_affected=len(df_before) - len(df_after)
    )
    cleaning_log.append(step)
    print(f"Step {step_num}: {operation}")
    print(f"  Rows: {step.rows_before} -> {step.rows_after} ({step.rows_affected} removed)")

def clean_data():
    """Main cleaning function."""
    # Load raw data
    df = pd.read_csv("data/raw/survey_raw.csv")
    step = 0

    # Step 1: Drop duplicate observations
    step += 1
    df_before = df.copy()
    df = df.drop_duplicates()
    log_step(df_before, df,
             "Remove duplicate rows",
             "Duplicates likely from data collection error")

    # Step 2: Handle missing values
    step += 1
    df_before = df.copy()
    df = df.dropna(subset=["outcome", "treatment"])
    log_step(df_before, df,
             "Drop observations missing outcome or treatment",
             "Cannot estimate effects without these variables")

    # Step 3: Restrict age range
    step += 1
    df_before = df.copy()
    df = df[(df["age"] >= 18) & (df["age"] <= 65)]
    log_step(df_before, df,
             "Restrict to working-age population (18-65)",
             "Following original paper's sample definition")

    # Step 4: Handle outliers
    step += 1
    df_before = df.copy()
    p99 = df["income"].quantile(0.99)
    df = df[df["income"] <= p99]
    log_step(df_before, df,
             f"Top-code income at 99th percentile (${p99:,.0f})",
             "Extreme values likely measurement error")

    # Save cleaned data
    df.to_csv("data/processed/survey_clean.csv", index=False)

    # Save cleaning log
    save_cleaning_log()

    return df

def save_cleaning_log():
    """Save cleaning log to markdown file."""
    with open("data/processed/cleaning_log.md", "w") as f:
        f.write("# Data Cleaning Log\n\n")
        f.write("| Step | Operation | Justification | Rows Before | Rows After | Removed |\n")
        f.write("|------|-----------|---------------|-------------|------------|----------|\n")

        for step in cleaning_log:
            f.write(f"| {step.step_number} | {step.operation} | {step.justification} | "
                   f"{step.rows_before:,} | {step.rows_after:,} | {step.rows_affected:,} |\n")

if __name__ == "__main__":
    clean_data()
```

---

## Variable Construction

### Documentation Requirements

For every constructed variable:

```markdown
## Variable: [variable_name]

**Type**: [continuous/binary/categorical]

**Definition**: [Clear description]

**Construction**:
```python
# Code that creates the variable
df['variable_name'] = (df['var1'] + df['var2']) / df['var3']
```

**Source variables**: [var1, var2, var3]

**Missing values**: [How handled]

**Validation**: [How verified]

**Notes**: [Any caveats or assumptions]
```

### Variable Dictionary Template

```python
"""
Variable Dictionary Generator
=============================

Generates comprehensive variable documentation.
"""

import pandas as pd

def generate_variable_dictionary(df, output_path="docs/variable_dictionary.md"):
    """Generate variable dictionary from dataframe."""

    lines = [
        "# Variable Dictionary\n",
        "## Dataset Overview\n",
        f"- **Observations**: {len(df):,}\n",
        f"- **Variables**: {len(df.columns)}\n",
        f"- **Memory Usage**: {df.memory_usage(deep=True).sum() / 1e6:.1f} MB\n\n",
        "## Variable Descriptions\n\n",
        "| Variable | Type | Non-Missing | Unique | Mean/Mode | Std/Freq |\n",
        "|----------|------|-------------|--------|-----------|----------|\n"
    ]

    for col in df.columns:
        dtype = str(df[col].dtype)
        non_missing = df[col].notna().sum()
        n_unique = df[col].nunique()

        if pd.api.types.is_numeric_dtype(df[col]):
            mean_val = f"{df[col].mean():.2f}"
            std_val = f"{df[col].std():.2f}"
        else:
            mean_val = str(df[col].mode().iloc[0]) if len(df[col].mode()) > 0 else "N/A"
            std_val = f"{df[col].value_counts().iloc[0]:,}" if n_unique > 0 else "N/A"

        lines.append(
            f"| {col} | {dtype} | {non_missing:,} | {n_unique:,} | {mean_val} | {std_val} |\n"
        )

    with open(output_path, 'w') as f:
        f.writelines(lines)

    print(f"Variable dictionary saved to: {output_path}")
```

---

## Data Version Control

### Using Git LFS

For data files under 2GB:

```bash
# Install Git LFS
git lfs install

# Track data files
git lfs track "data/raw/*.csv"
git lfs track "data/raw/*.dta"
git lfs track "data/raw/*.rds"

# Commit tracking configuration
git add .gitattributes
git commit -m "Configure Git LFS for data files"

# Add and commit data
git add data/raw/
git commit -m "Add raw data files"
```

**.gitattributes**:
```
data/raw/*.csv filter=lfs diff=lfs merge=lfs -text
data/raw/*.dta filter=lfs diff=lfs merge=lfs -text
data/raw/*.rds filter=lfs diff=lfs merge=lfs -text
data/raw/*.parquet filter=lfs diff=lfs merge=lfs -text
```

### Using DVC (Data Version Control)

For larger files or more complex workflows:

```bash
# Initialize DVC
dvc init

# Add data to DVC
dvc add data/raw/large_dataset.csv

# Configure remote storage (e.g., S3)
dvc remote add -d myremote s3://mybucket/dvc-storage

# Push data to remote
dvc push

# Commit DVC files to git
git add data/raw/large_dataset.csv.dvc data/raw/.gitignore
git commit -m "Add large dataset with DVC"
```

### Data Version Manifest

```json
{
    "data_version": "1.0.0",
    "last_updated": "2024-06-15",
    "files": [
        {
            "path": "data/raw/survey_2020.csv",
            "description": "Main survey data, 2020 wave",
            "source": "ICPSR Study 38417",
            "download_date": "2024-06-15",
            "md5_hash": "abc123...",
            "rows": 50000,
            "columns": 150
        },
        {
            "path": "data/raw/admin_records.csv",
            "description": "Administrative records from partner agency",
            "source": "Data sharing agreement #2024-001",
            "download_date": "2024-06-10",
            "md5_hash": "def456...",
            "rows": 75000,
            "columns": 25
        }
    ],
    "derived_files": [
        {
            "path": "data/processed/analysis_sample.csv",
            "description": "Final analysis sample",
            "created_by": "code/02_clean_data.py",
            "created_from": ["data/raw/survey_2020.csv", "data/raw/admin_records.csv"],
            "rows": 45000,
            "columns": 50
        }
    ]
}
```

---

## Data Validation

### Automated Validation Checks

```python
"""
Data Validation Script
======================

Validates data integrity before analysis.
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import List, Callable

@dataclass
class ValidationCheck:
    name: str
    check_function: Callable
    expected: any
    critical: bool = True

def validate_dataset(df, checks: List[ValidationCheck]):
    """Run all validation checks on dataset."""
    results = []

    for check in checks:
        try:
            actual = check.check_function(df)
            passed = actual == check.expected
        except Exception as e:
            actual = f"Error: {e}"
            passed = False

        results.append({
            "check": check.name,
            "expected": check.expected,
            "actual": actual,
            "passed": passed,
            "critical": check.critical
        })

        status = "PASS" if passed else "FAIL"
        print(f"[{status}] {check.name}: expected {check.expected}, got {actual}")

    return results

# Define validation checks
validation_checks = [
    ValidationCheck(
        name="Sample size",
        check_function=lambda df: len(df),
        expected=722,
        critical=True
    ),
    ValidationCheck(
        name="Treatment assignment count",
        check_function=lambda df: df['treat'].sum(),
        expected=185,
        critical=True
    ),
    ValidationCheck(
        name="No missing treatment",
        check_function=lambda df: df['treat'].isna().sum(),
        expected=0,
        critical=True
    ),
    ValidationCheck(
        name="No missing outcome",
        check_function=lambda df: df['re78'].isna().sum(),
        expected=0,
        critical=True
    ),
    ValidationCheck(
        name="Age range valid",
        check_function=lambda df: (df['age'] >= 16).all() and (df['age'] <= 65).all(),
        expected=True,
        critical=False
    ),
    ValidationCheck(
        name="No negative earnings",
        check_function=lambda df: (df['re78'] >= 0).all(),
        expected=True,
        critical=True
    ),
]

if __name__ == "__main__":
    df = pd.read_csv("data/processed/analysis_sample.csv")
    results = validate_dataset(df, validation_checks)

    critical_failures = [r for r in results if not r["passed"] and r["critical"]]
    if critical_failures:
        print(f"\nCRITICAL: {len(critical_failures)} validation checks failed!")
        exit(1)
    else:
        print("\nAll critical validation checks passed.")
```

### Expected vs. Actual Comparisons

```python
def compare_summary_stats(df, expected_stats):
    """Compare computed statistics to expected values."""

    comparisons = []

    for var, stats in expected_stats.items():
        computed = {
            "mean": df[var].mean(),
            "std": df[var].std(),
            "min": df[var].min(),
            "max": df[var].max(),
            "n": df[var].notna().sum()
        }

        for stat_name, expected_val in stats.items():
            actual_val = computed.get(stat_name)
            if actual_val is not None:
                diff = abs(actual_val - expected_val)
                pct_diff = diff / abs(expected_val) * 100 if expected_val != 0 else float('inf')

                comparisons.append({
                    "variable": var,
                    "statistic": stat_name,
                    "expected": expected_val,
                    "actual": actual_val,
                    "difference": diff,
                    "pct_difference": pct_diff,
                    "match": pct_diff < 1.0  # Within 1%
                })

    return pd.DataFrame(comparisons)

# Expected statistics from paper
expected_stats = {
    "re78": {"mean": 6349.14, "std": 7867.40, "n": 722},
    "age": {"mean": 25.82, "std": 7.16, "n": 722},
    "education": {"mean": 10.35, "std": 2.01, "n": 722},
}
```

---

## Data Security and Privacy

### Sensitive Data Handling

```markdown
## Data Security Protocol

### Data Classification

| Level | Description | Examples | Handling |
|-------|-------------|----------|----------|
| **Public** | No restrictions | Published statistics | Standard practices |
| **Internal** | Not public but not sensitive | Survey without PII | Restricted sharing |
| **Confidential** | Contains PII or sensitive info | Admin records | Secure environment |
| **Restricted** | Highly sensitive | Health records | IRB approval, secure room |

### Security Measures

For confidential/restricted data:

1. **Access Control**
   - Data stored on encrypted drives
   - Access limited to approved researchers
   - Access logged and audited

2. **De-identification**
   - Remove direct identifiers (names, SSN)
   - Aggregate or bin indirect identifiers
   - Use synthetic data for code sharing

3. **Secure Computing**
   - Analysis on secure servers
   - No data downloaded to personal machines
   - Results reviewed before export
```

### Synthetic Data Generation

```python
"""
Generate Synthetic Data
=======================

Creates synthetic data that preserves statistical properties
while protecting privacy.
"""

import pandas as pd
import numpy as np
from scipy import stats

def generate_synthetic_data(original_df, n_synthetic=None, seed=42):
    """
    Generate synthetic data preserving marginal distributions
    and approximate correlations.
    """
    np.random.seed(seed)

    if n_synthetic is None:
        n_synthetic = len(original_df)

    synthetic = {}

    for col in original_df.columns:
        if pd.api.types.is_numeric_dtype(original_df[col]):
            # Fit normal distribution
            mean = original_df[col].mean()
            std = original_df[col].std()
            synthetic[col] = np.random.normal(mean, std, n_synthetic)
        else:
            # Sample from empirical distribution
            values = original_df[col].dropna().values
            probs = original_df[col].value_counts(normalize=True)
            synthetic[col] = np.random.choice(
                probs.index,
                size=n_synthetic,
                p=probs.values
            )

    return pd.DataFrame(synthetic)
```

---

## References

- Vilhuber, L. (2020). "Reproducibility and Replicability in Economics." *Harvard Data Science Review*.
- Christensen, G., Freese, J., & Miguel, E. (2019). *Transparent and Reproducible Social Science Research*. University of California Press.
- ICPSR. "Guide to Social Science Data Preparation and Archiving."
- UK Data Service. "Research Data Management."

---

## See Also

- [replication_standards.md](replication_standards.md) - Journal requirements
- [code_organization.md](code_organization.md) - Code structure
- [documentation_checklist.md](documentation_checklist.md) - Documentation requirements
