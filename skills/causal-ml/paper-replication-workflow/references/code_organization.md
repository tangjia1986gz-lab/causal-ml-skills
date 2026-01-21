# Code Organization for Replication Packages

> **Document Type**: Reference | **Last Updated**: 2025-01
> **Related**: [replication_standards.md](replication_standards.md), [data_management.md](data_management.md)

## Overview

Well-organized code is essential for computational reproducibility. This document provides best practices for directory structure, master scripts, coding standards, and ensuring long-term reproducibility of research code.

---

## Directory Structure

### Standard Layout

```
replication_package/
├── README.md                 # Main documentation (required)
├── LICENSE                   # Code/data license (required)
├── CITATION.cff             # Citation metadata (recommended)
├── requirements.txt         # Python dependencies
├── environment.yml          # Conda environment
├── renv.lock               # R dependencies (if applicable)
│
├── code/
│   ├── 00_master.py         # Master script (runs everything)
│   ├── 01_download_data.py  # Data acquisition
│   ├── 02_clean_data.py     # Data cleaning
│   ├── 03_construct_vars.py # Variable construction
│   ├── 04_analysis_main.py  # Main analysis
│   ├── 05_analysis_robust.py# Robustness checks
│   ├── 06_tables.py         # Generate tables
│   ├── 07_figures.py        # Generate figures
│   └── lib/                 # Helper modules
│       ├── __init__.py
│       ├── data_utils.py
│       ├── estimation.py
│       └── visualization.py
│
├── data/
│   ├── raw/                 # Original data (read-only)
│   │   └── README.md        # Data source documentation
│   ├── processed/           # Cleaned data
│   └── temp/                # Temporary files (gitignored)
│
├── output/
│   ├── tables/              # LaTeX/CSV tables
│   ├── figures/             # PDF/PNG figures
│   └── logs/                # Execution logs
│
├── docs/
│   ├── variable_dictionary.md
│   ├── analysis_notes.md
│   └── codebook.pdf
│
└── manuscript/              # Paper files (optional)
    ├── main.tex
    └── figures/             # Symlink to output/figures
```

### Directory Purposes

| Directory | Purpose | Git Tracked |
|-----------|---------|-------------|
| `code/` | All analysis code | Yes |
| `data/raw/` | Original, unmodified data | Yes (LFS) or No |
| `data/processed/` | Cleaned datasets | No (regenerate) |
| `data/temp/` | Temporary files | No |
| `output/tables/` | Generated tables | Optional |
| `output/figures/` | Generated figures | Optional |
| `output/logs/` | Execution logs | No |
| `docs/` | Documentation | Yes |

---

## Master Script Design

### Purpose

The master script provides **one-click reproducibility**: a single command that regenerates all results from raw data to final output.

### Template: Python Master Script

```python
#!/usr/bin/env python
"""
Master Script for Replication Package
=====================================

Paper: [Full Paper Title]
Authors: [Author Names]
Journal: [Journal Name], [Year]

This script reproduces all tables and figures in the paper.

Usage:
    python 00_master.py [options]

Options:
    --skip-download    Skip data download (use existing data)
    --skip-clean       Skip data cleaning (use existing processed data)
    --tables-only      Only regenerate tables
    --figures-only     Only regenerate figures
    --parallel         Run independent steps in parallel

Requirements:
    Python 3.10+, see requirements.txt for packages

Runtime:
    Full replication: ~2 hours
    Tables/figures only: ~10 minutes

Output:
    output/tables/     LaTeX and CSV tables
    output/figures/    PDF and PNG figures
    output/logs/       Execution logs
"""

import argparse
import subprocess
import sys
import time
import logging
from pathlib import Path
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor

# Configuration
PROJECT_ROOT = Path(__file__).parent.parent
CODE_DIR = PROJECT_ROOT / "code"
OUTPUT_DIR = PROJECT_ROOT / "output"
LOG_DIR = OUTPUT_DIR / "logs"

# Ordered list of scripts to run
PIPELINE = [
    # (script_name, description, dependencies)
    ("01_download_data.py", "Download raw data", []),
    ("02_clean_data.py", "Clean and prepare data", ["01_download_data.py"]),
    ("03_construct_vars.py", "Construct analysis variables", ["02_clean_data.py"]),
    ("04_analysis_main.py", "Run main analysis", ["03_construct_vars.py"]),
    ("05_analysis_robust.py", "Run robustness checks", ["04_analysis_main.py"]),
    ("06_tables.py", "Generate tables", ["04_analysis_main.py", "05_analysis_robust.py"]),
    ("07_figures.py", "Generate figures", ["04_analysis_main.py"]),
]

def setup_logging():
    """Configure logging to file and console."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = LOG_DIR / f"replication_{timestamp}.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )

    return log_file

def run_script(script_name: str, description: str) -> tuple:
    """Run a single script and capture output."""
    script_path = CODE_DIR / script_name
    log = logging.getLogger(__name__)

    log.info(f"Starting: {description}")
    log.info(f"Script: {script_name}")

    start_time = time.time()

    try:
        result = subprocess.run(
            [sys.executable, str(script_path)],
            capture_output=True,
            text=True,
            cwd=str(CODE_DIR)
        )

        elapsed = time.time() - start_time

        if result.returncode != 0:
            log.error(f"FAILED: {script_name}")
            log.error(f"Stderr: {result.stderr}")
            return (script_name, False, elapsed, result.stderr)

        log.info(f"Completed: {script_name} ({elapsed:.1f}s)")
        return (script_name, True, elapsed, result.stdout)

    except Exception as e:
        elapsed = time.time() - start_time
        log.error(f"Exception in {script_name}: {e}")
        return (script_name, False, elapsed, str(e))

def filter_pipeline(args) -> list:
    """Filter pipeline based on command-line arguments."""
    pipeline = PIPELINE.copy()

    if args.skip_download:
        pipeline = [(s, d, deps) for s, d, deps in pipeline
                   if s != "01_download_data.py"]

    if args.skip_clean:
        pipeline = [(s, d, deps) for s, d, deps in pipeline
                   if s not in ["02_clean_data.py", "03_construct_vars.py"]]

    if args.tables_only:
        pipeline = [(s, d, deps) for s, d, deps in pipeline
                   if "table" in s.lower()]

    if args.figures_only:
        pipeline = [(s, d, deps) for s, d, deps in pipeline
                   if "figure" in s.lower()]

    return pipeline

def main():
    """Run the complete replication pipeline."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--skip-clean", action="store_true")
    parser.add_argument("--tables-only", action="store_true")
    parser.add_argument("--figures-only", action="store_true")
    parser.add_argument("--parallel", action="store_true")
    args = parser.parse_args()

    log_file = setup_logging()
    log = logging.getLogger(__name__)

    log.info("=" * 60)
    log.info("REPLICATION PACKAGE - MASTER SCRIPT")
    log.info("=" * 60)
    log.info(f"Log file: {log_file}")
    log.info(f"Python version: {sys.version}")
    log.info(f"Start time: {datetime.now().isoformat()}")

    pipeline = filter_pipeline(args)
    log.info(f"Running {len(pipeline)} scripts")

    total_start = time.time()
    results = []
    failed = []

    for script_name, description, dependencies in pipeline:
        result = run_script(script_name, description)
        results.append(result)

        if not result[1]:  # Failed
            failed.append(script_name)
            log.warning(f"Continuing despite failure in {script_name}")

    total_elapsed = time.time() - total_start

    # Summary
    log.info("=" * 60)
    log.info("REPLICATION SUMMARY")
    log.info("=" * 60)
    log.info(f"Total runtime: {total_elapsed/60:.1f} minutes")
    log.info(f"Scripts run: {len(results)}")
    log.info(f"Successful: {len(results) - len(failed)}")
    log.info(f"Failed: {len(failed)}")

    if failed:
        log.error(f"Failed scripts: {failed}")
        log.info("Review the log file for details")
        sys.exit(1)
    else:
        log.info("All scripts completed successfully!")
        log.info("Check output/ directory for results")

if __name__ == "__main__":
    main()
```

### Template: Stata Master Script

```stata
/*******************************************************************************
Master Script for Replication Package

Paper: [Paper Title]
Authors: [Authors]
Journal: [Journal], [Year]

This script reproduces all results in the paper.

Usage:
    stata -b do 00_master.do

Runtime: Approximately 2 hours
*******************************************************************************/

clear all
set more off
set maxvar 10000

* Set project root (modify as needed)
global root "C:/replication_package"

* Define paths
global code    "${root}/code"
global data    "${root}/data"
global rawdata "${data}/raw"
global procdata "${data}/processed"
global output  "${root}/output"
global tables  "${output}/tables"
global figures "${output}/figures"
global logs    "${output}/logs"

* Create output directories
cap mkdir "${output}"
cap mkdir "${tables}"
cap mkdir "${figures}"
cap mkdir "${logs}"

* Start log
log using "${logs}/replication_`c(current_date)'.log", replace

* Display system information
di "Stata version: `c(stata_version)'"
di "Date: `c(current_date)' `c(current_time)'"
di "User: `c(username)'"

timer clear
timer on 1

/*******************************************************************************
Step 1: Data Preparation
*******************************************************************************/

di _n(2) "=" * 60
di "STEP 1: Data Preparation"
di "=" * 60

do "${code}/01_clean_data.do"

/*******************************************************************************
Step 2: Main Analysis
*******************************************************************************/

di _n(2) "=" * 60
di "STEP 2: Main Analysis"
di "=" * 60

do "${code}/02_analysis_main.do"

/*******************************************************************************
Step 3: Robustness Checks
*******************************************************************************/

di _n(2) "=" * 60
di "STEP 3: Robustness Checks"
di "=" * 60

do "${code}/03_robustness.do"

/*******************************************************************************
Step 4: Generate Tables
*******************************************************************************/

di _n(2) "=" * 60
di "STEP 4: Generate Tables"
di "=" * 60

do "${code}/04_tables.do"

/*******************************************************************************
Step 5: Generate Figures
*******************************************************************************/

di _n(2) "=" * 60
di "STEP 5: Generate Figures"
di "=" * 60

do "${code}/05_figures.do"

/*******************************************************************************
Summary
*******************************************************************************/

timer off 1
timer list 1

di _n(2) "=" * 60
di "REPLICATION COMPLETE"
di "=" * 60
di "Total runtime: `r(t1)' seconds"
di "Check ${output} for results"

log close
```

---

## Coding Standards

### General Principles

1. **Readability over cleverness**: Code should be easy to understand
2. **One purpose per script**: Each script does one thing well
3. **Explicit over implicit**: Avoid hidden dependencies
4. **Fail loudly**: Errors should be visible, not hidden

### Python Style Guide

```python
"""
Script Template: [Description]
==============================

Purpose: [What this script does]

Input:
    - data/processed/input_file.csv

Output:
    - output/tables/table_1.tex
    - output/figures/figure_1.pdf

Dependencies:
    - 02_clean_data.py (must run first)

Author: [Name]
Date: [Date]
"""

# Standard library imports
import os
import sys
from pathlib import Path

# Third-party imports
import numpy as np
import pandas as pd
import statsmodels.api as sm

# Local imports
from lib.estimation import run_regression
from lib.visualization import create_figure

# Configuration
# Use absolute paths from project root
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_DIR = PROJECT_ROOT / "output"

# Constants (use UPPER_CASE)
OUTCOME_VAR = "earnings"
TREATMENT_VAR = "treat"
COVARIATE_VARS = ["age", "education", "experience"]

def load_data(filepath: Path) -> pd.DataFrame:
    """
    Load and validate analysis data.

    Parameters
    ----------
    filepath : Path
        Path to data file

    Returns
    -------
    pd.DataFrame
        Loaded dataframe

    Raises
    ------
    FileNotFoundError
        If data file does not exist
    ValueError
        If required columns are missing
    """
    if not filepath.exists():
        raise FileNotFoundError(f"Data file not found: {filepath}")

    df = pd.read_csv(filepath)

    # Validate required columns
    required_cols = [OUTCOME_VAR, TREATMENT_VAR] + COVARIATE_VARS
    missing_cols = set(required_cols) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    return df

def main():
    """Main function."""
    print("=" * 60)
    print("Running: [Script Name]")
    print("=" * 60)

    # Load data
    print("\nLoading data...")
    df = load_data(DATA_DIR / "analysis_sample.csv")
    print(f"  Observations: {len(df):,}")

    # Run analysis
    print("\nRunning analysis...")
    results = run_analysis(df)

    # Save results
    print("\nSaving results...")
    save_results(results)

    print("\nComplete!")

if __name__ == "__main__":
    main()
```

### R Style Guide

```r
#' =============================================================================
#' Script: [Description]
#' =============================================================================
#'
#' Purpose: [What this script does]
#'
#' Input:
#'   - data/processed/input_file.csv
#'
#' Output:
#'   - output/tables/table_1.tex
#'   - output/figures/figure_1.pdf
#'
#' Dependencies:
#'   - 02_clean_data.R (must run first)
#'
#' Author: [Name]
#' Date: [Date]
#' =============================================================================

# Clear workspace
rm(list = ls())

# Load packages
library(tidyverse)
library(fixest)
library(modelsummary)

# Configuration
root_dir <- here::here()
data_dir <- file.path(root_dir, "data", "processed")
output_dir <- file.path(root_dir, "output")

# Constants
OUTCOME_VAR <- "earnings"
TREATMENT_VAR <- "treat"
COVARIATE_VARS <- c("age", "education", "experience")

# =============================================================================
# Functions
# =============================================================================

#' Load and validate analysis data
#'
#' @param filepath Path to data file
#' @return Data frame
load_data <- function(filepath) {
  if (!file.exists(filepath)) {
    stop(paste("Data file not found:", filepath))
  }

  df <- read_csv(filepath, show_col_types = FALSE)

  # Validate required columns
  required_cols <- c(OUTCOME_VAR, TREATMENT_VAR, COVARIATE_VARS)
  missing_cols <- setdiff(required_cols, names(df))
  if (length(missing_cols) > 0) {
    stop(paste("Missing required columns:", paste(missing_cols, collapse = ", ")))
  }

  return(df)
}

# =============================================================================
# Main
# =============================================================================

main <- function() {
  cat("=" %>% rep(60) %>% paste(collapse = ""), "\n")
  cat("Running: [Script Name]\n")
  cat("=" %>% rep(60) %>% paste(collapse = ""), "\n")

  # Load data
  cat("\nLoading data...\n")
  df <- load_data(file.path(data_dir, "analysis_sample.csv"))
  cat(sprintf("  Observations: %s\n", format(nrow(df), big.mark = ",")))

  # Run analysis
  cat("\nRunning analysis...\n")
  results <- run_analysis(df)

  # Save results
  cat("\nSaving results...\n")
  save_results(results)

  cat("\nComplete!\n")
}

# Run
main()
```

---

## Relative Path Handling

### Problem

Hardcoded absolute paths break reproducibility across different machines.

### Solution: Use Project-Relative Paths

```python
# BAD: Hardcoded absolute path
data_path = "C:/Users/jsmith/project/data/file.csv"

# GOOD: Relative to script location
from pathlib import Path
PROJECT_ROOT = Path(__file__).parent.parent  # Go up from code/ to root
data_path = PROJECT_ROOT / "data" / "file.csv"

# GOOD: Environment variable
import os
PROJECT_ROOT = Path(os.environ.get("PROJECT_ROOT", Path(__file__).parent.parent))
```

### Cross-Platform Compatibility

```python
from pathlib import Path

# BAD: OS-specific path separators
path = "data\\raw\\file.csv"  # Windows only
path = "data/raw/file.csv"    # Usually works but not guaranteed

# GOOD: Use pathlib
path = Path("data") / "raw" / "file.csv"  # Works on all platforms
```

---

## Error Handling

### Fail Early, Fail Loudly

```python
def load_required_data(filepath: Path) -> pd.DataFrame:
    """Load data with validation."""

    # Check file exists
    if not filepath.exists():
        raise FileNotFoundError(
            f"Required data file not found: {filepath}\n"
            f"Run 02_clean_data.py first to generate this file."
        )

    # Load data
    df = pd.read_csv(filepath)

    # Validate
    if len(df) == 0:
        raise ValueError(f"Data file is empty: {filepath}")

    if df.duplicated().any():
        n_dups = df.duplicated().sum()
        raise ValueError(f"Data contains {n_dups} duplicate rows")

    return df
```

### Graceful Degradation

```python
def load_optional_data(filepath: Path) -> pd.DataFrame | None:
    """Load optional data, returning None if unavailable."""

    if not filepath.exists():
        logging.warning(f"Optional data file not found: {filepath}")
        return None

    try:
        return pd.read_csv(filepath)
    except Exception as e:
        logging.warning(f"Failed to load {filepath}: {e}")
        return None
```

---

## Logging and Progress

### Structured Logging

```python
import logging
from datetime import datetime

def setup_logging(log_file: Path):
    """Configure logging."""

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

# Usage
log = logging.getLogger(__name__)

log.info("Starting analysis")
log.debug("Detailed debugging info")
log.warning("Something unexpected happened")
log.error("An error occurred")
```

### Progress Indicators

```python
from tqdm import tqdm

# For loops
for i in tqdm(range(1000), desc="Processing"):
    process_item(i)

# For pandas operations
from tqdm.auto import tqdm
tqdm.pandas()
df['result'] = df['input'].progress_apply(expensive_function)
```

---

## Reproducibility Checklist

### Before Sharing Code

- [ ] All paths are relative to project root
- [ ] No hardcoded machine-specific paths
- [ ] Dependencies are pinned to specific versions
- [ ] Random seeds are set for stochastic operations
- [ ] Master script runs without errors on clean environment
- [ ] Output files match expected results

### Environment Verification

```python
def verify_environment():
    """Verify required packages and versions."""

    import importlib
    import sys

    required = {
        "pandas": "2.0.0",
        "numpy": "1.24.0",
        "statsmodels": "0.14.0",
    }

    print(f"Python version: {sys.version}")
    print("\nPackage versions:")

    all_ok = True
    for package, min_version in required.items():
        try:
            mod = importlib.import_module(package)
            version = getattr(mod, "__version__", "unknown")
            status = "OK" if version >= min_version else "WARNING: older than required"
            print(f"  {package}: {version} ({status})")
        except ImportError:
            print(f"  {package}: NOT INSTALLED")
            all_ok = False

    return all_ok

if __name__ == "__main__":
    if not verify_environment():
        print("\nERROR: Missing required packages. Install with:")
        print("  pip install -r requirements.txt")
        sys.exit(1)
```

---

## References

- Gentzkow, M., & Shapiro, J. M. (2014). "Code and Data for the Social Sciences: A Practitioner's Guide."
- Wilson, G., et al. (2017). "Good Enough Practices in Scientific Computing." *PLOS Computational Biology*.
- AEA Data Editor. "Unofficial Guidance on Code and Data."

---

## See Also

- [replication_standards.md](replication_standards.md) - Journal requirements
- [data_management.md](data_management.md) - Data handling
- [documentation_checklist.md](documentation_checklist.md) - Documentation requirements
