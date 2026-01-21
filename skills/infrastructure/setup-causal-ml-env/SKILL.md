---
name: setup-causal-ml-env
description: Set up and validate the causal inference ML environment with Python, R, and optional Stata integration
triggers:
  - setup causal ml environment
  - check causal dependencies
  - install causal inference packages
  - configure rpy2 bridge
  - causal ml setup
  - environment setup for causal inference
---

# Setup Causal ML Environment

This skill helps you set up, validate, and troubleshoot your causal inference machine learning environment. It covers Python packages, R integration, and optional Stata connectivity.

## Overview

Causal inference work requires a comprehensive stack of specialized packages across multiple languages:

- **Python**: Core ML and causal inference libraries
- **R**: Statistical packages for advanced causal methods (via rpy2 bridge)
- **Stata**: Optional, for econometric workflows

## Quick Start

### Check Current Environment

Run the diagnostic script to see what's installed and what's missing:

```bash
python env_check.py
```

### Install All Python Dependencies

```bash
pip install -r requirements.txt
```

## Python Environment Requirements

### Minimum Version
- **Python 3.10+** (required for modern type hints and package compatibility)

### Core Causal Inference Packages

| Package | Version | Purpose |
|---------|---------|---------|
| econml | >=0.15.0 | Microsoft's causal ML library (DML, IV, policy learning) |
| doubleml | >=0.7.0 | Double/Debiased ML for causal effects |
| causalml | >=0.15.0 | Uber's uplift modeling and causal inference |
| dowhy | >=0.11.0 | Microsoft's causal inference framework |

### Statistical & Econometric Packages

| Package | Version | Purpose |
|---------|---------|---------|
| statsmodels | >=0.14.0 | Statistical models, tests, and data exploration |
| linearmodels | >=6.0 | Panel data, IV, and system estimation |

### Machine Learning Packages

| Package | Version | Purpose |
|---------|---------|---------|
| scikit-learn | >=1.3.0 | Core ML algorithms |
| xgboost | >=2.0.0 | Gradient boosting for heterogeneous effects |
| lightgbm | >=4.0.0 | Fast gradient boosting |

### Data & Visualization

| Package | Version | Purpose |
|---------|---------|---------|
| pandas | >=2.0.0 | Data manipulation |
| numpy | >=1.24.0 | Numerical computing |
| matplotlib | >=3.7.0 | Plotting |
| seaborn | >=0.13.0 | Statistical visualization |

## R Environment Requirements

### Minimum Version
- **R 4.0+** (required for modern tidyverse compatibility)

### Required R Packages

| Package | Purpose |
|---------|---------|
| grf | Generalized Random Forests for causal effects |
| mediation | Causal mediation analysis |
| rdrobust | Robust RDD estimation |
| rddensity | RDD density tests (McCrary) |

### Installing R Packages

Open R or RStudio and run:

```r
install.packages(c("grf", "mediation", "rdrobust", "rddensity"))
```

## rpy2 Bridge Configuration

The rpy2 package allows calling R from Python. This is essential for using R's causal inference packages within Python workflows.

### Installation

```bash
pip install rpy2>=3.5.0
```

### Configuration Requirements

1. **R must be installed and on PATH**
2. **R_HOME environment variable must be set** (if not auto-detected)

#### Windows Setup

```powershell
# Check R installation
where R

# Set R_HOME if needed (example path)
$env:R_HOME = "C:\Program Files\R\R-4.3.0"
```

#### Linux/macOS Setup

```bash
# Check R installation
which R

# Set R_HOME if needed (example path)
export R_HOME=/usr/lib/R
```

### Testing rpy2 Bridge

```python
import rpy2.robjects as ro

# Test basic R execution
result = ro.r('1 + 1')
print(f"R returned: {result[0]}")

# Test loading grf
ro.r('library(grf)')
print("grf loaded successfully")
```

## Stata Integration (Optional)

Stata integration is optional but useful for replicating traditional econometric workflows.

### Detection

The environment check will look for:
- `stata` or `stata-mp` on PATH
- Common installation directories

### Python-Stata Bridge Options

1. **pystata** (Stata 17+): Official Python integration
2. **stata_setup**: Configuration helper for pystata

```python
# Example: Using pystata
import stata_setup
stata_setup.config("C:/Program Files/Stata17", "mp")

from pystata import stata
stata.run("sysuse auto")
```

## Installation Commands

### Full Installation (Recommended)

```bash
# Create virtual environment (recommended)
python -m venv causal-env
# Windows
causal-env\Scripts\activate
# Linux/macOS
source causal-env/bin/activate

# Install Python packages
pip install -r requirements.txt

# Install R packages (run in R)
# install.packages(c("grf", "mediation", "rdrobust", "rddensity"))
```

### Minimal Installation (Python Only)

```bash
pip install econml doubleml statsmodels scikit-learn pandas numpy matplotlib
```

### With GPU Support (CUDA)

```bash
# XGBoost with GPU
pip install xgboost --upgrade

# LightGBM with GPU (requires build from source on some platforms)
pip install lightgbm --config-settings=cmake.define.USE_GPU=ON
```

## Troubleshooting

### Common Issues

#### 1. econml Installation Fails

econml has complex dependencies. Try:

```bash
pip install cython numpy scipy
pip install econml --no-build-isolation
```

#### 2. rpy2 Cannot Find R

Ensure R_HOME is set correctly:

```python
import os
os.environ['R_HOME'] = '/path/to/R'  # Set before importing rpy2
import rpy2.robjects as ro
```

#### 3. causalml SHAP Conflicts

causalml may have version conflicts with SHAP. Install in order:

```bash
pip install shap==0.42.1
pip install causalml
```

#### 4. LightGBM on macOS (Apple Silicon)

```bash
brew install libomp
pip install lightgbm
```

### Getting Help

If the diagnostic script reports issues:

1. Check the specific error messages
2. Ensure you're using Python 3.10+
3. Try installing in a fresh virtual environment
4. Check package GitHub issues for known problems

## Environment Verification

After setup, verify everything works:

```python
# Quick verification script
from econml.dml import DML
from doubleml import DoubleMLData
from causalml.inference.meta import BaseSRegressor
from statsmodels.regression.linear_model import OLS
from linearmodels.iv import IV2SLS
import sklearn
import xgboost
import lightgbm
import pandas as pd
import numpy as np

print("All critical packages imported successfully!")
```

## Next Steps

Once your environment is set up:

1. Run `python env_check.py` to verify installation
2. Check out the causal inference tutorials in this project
3. Start with simple examples before complex analyses
