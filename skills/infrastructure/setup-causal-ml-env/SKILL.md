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

## Quick Reference

| Resource | Description |
|----------|-------------|
| [Python Packages](references/python_packages.md) | Detailed package documentation |
| [R Packages](references/r_packages.md) | R integration via rpy2 |
| [Stata Integration](references/stata_integration.md) | Optional Stata setup |
| [Troubleshooting](references/troubleshooting.md) | Common issues and solutions |

## Quick Start

### 1. Check Current Environment

Run the comprehensive diagnostic script:

```bash
python scripts/check_environment.py
```

Options:
- `--verbose` - Show detailed information
- `--json` - Output in JSON format
- `--fix` - Show fix commands for issues

### 2. Install Dependencies

**Automated Installation (Recommended):**
```bash
python scripts/install_dependencies.py
```

Options:
- `--minimal` - Core packages only
- `--full` - All packages (default)
- `--with-r` - Include R packages
- `--dry-run` - Preview without installing

**Manual Installation:**
```bash
pip install -r requirements.txt
```

### 3. Verify Installation

```python
# Quick verification
from econml.dml import DML
from doubleml import DoubleMLData
from causalml.inference.meta import BaseSRegressor
from statsmodels.regression.linear_model import OLS
print("All critical packages imported successfully!")
```

## Environment Overview

### Required Components

| Component | Version | Status |
|-----------|---------|--------|
| Python | 3.10+ | Required |
| Core Causal Packages | See below | Required |
| R + rpy2 | 4.0+ | Optional |
| Stata | 17+ | Optional |

### Core Python Packages

| Package | Purpose | Min Version |
|---------|---------|-------------|
| econml | DML, Causal Forests, IV | 0.15.0 |
| doubleml | Double/Debiased ML | 0.7.0 |
| causalml | Uplift modeling, Meta-learners | 0.15.0 |
| dowhy | Causal graphs, Refutation | 0.11.0 |
| statsmodels | Statistical models | 0.14.0 |
| linearmodels | Panel data, IV | 6.0 |

See [references/python_packages.md](references/python_packages.md) for complete details.

### R Packages (via rpy2)

| Package | Purpose |
|---------|---------|
| grf | Generalized Random Forests |
| mediation | Causal mediation analysis |
| rdrobust | Robust RDD estimation |
| rddensity | RDD density tests |

See [references/r_packages.md](references/r_packages.md) for setup and usage.

## Installation Paths

### Full Installation (Python + R)

```bash
# 1. Create virtual environment
python -m venv causal-env
# Windows: causal-env\Scripts\activate
# Linux/macOS: source causal-env/bin/activate

# 2. Install Python packages
python scripts/install_dependencies.py --with-r

# 3. Install R packages (in R console)
install.packages(c("grf", "mediation", "rdrobust", "rddensity"))

# 4. Verify
python scripts/check_environment.py
```

### Minimal Installation (Python Only)

```bash
python scripts/install_dependencies.py --minimal
```

### Conda Installation

```bash
conda create -n causal-ml python=3.11
conda activate causal-ml
pip install -r requirements.txt
```

## Platform-Specific Notes

### Windows

1. **Visual C++ Build Tools** required for some packages
   - Download from: https://visualstudio.microsoft.com/visual-cpp-build-tools/

2. **R Setup:**
   ```powershell
   $env:R_HOME = "C:\Program Files\R\R-4.3.0"
   $env:PATH = "$env:R_HOME\bin\x64;$env:PATH"
   ```

3. **Long paths:** Enable in registry for deep directory structures

### macOS

1. **Apple Silicon (M1/M2):**
   ```bash
   brew install libomp  # Required for LightGBM
   ```

2. **R Setup:**
   ```bash
   export R_HOME=/Library/Frameworks/R.framework/Resources
   ```

### Linux

1. **System dependencies:**
   ```bash
   # Ubuntu/Debian
   sudo apt-get install r-base-dev libcurl4-openssl-dev libssl-dev

   # Fedora/RHEL
   sudo dnf install R-devel libcurl-devel openssl-devel
   ```

2. **R Setup:**
   ```bash
   export R_HOME=/usr/lib/R
   ```

## Troubleshooting

### Common Issues

| Issue | Solution |
|-------|----------|
| econml build fails | Pre-install: `pip install cython numpy scipy` |
| causalml SHAP conflict | Install in order: `pip install shap==0.42.1` then `causalml` |
| rpy2 R not found | Set `R_HOME` environment variable |
| LightGBM macOS error | Run: `brew install libomp` |
| NumPy 2.0 conflicts | Pin: `pip install "numpy<2.0"` |

See [references/troubleshooting.md](references/troubleshooting.md) for detailed solutions.

### Getting Diagnostic Info

```bash
# Full environment report
python scripts/check_environment.py --verbose

# JSON output for debugging
python scripts/check_environment.py --json > env_report.json
```

## Directory Structure

```
setup-causal-ml-env/
├── SKILL.md                 # This file
├── requirements.txt         # Python dependencies
├── env_check.py            # Legacy check script
├── references/
│   ├── python_packages.md  # Python package details
│   ├── r_packages.md       # R package details
│   ├── stata_integration.md # Stata setup
│   └── troubleshooting.md  # Common issues
└── scripts/
    ├── check_environment.py    # Comprehensive diagnostics
    └── install_dependencies.py # Automated installer
```

## Next Steps

After environment setup:

1. **Verify installation:** `python scripts/check_environment.py`
2. **Check out estimator skills** in `skills/estimators/`
3. **Start with simple examples** before complex analyses

## Version Compatibility

| Python | Status | Notes |
|--------|--------|-------|
| 3.10 | Fully Supported | Recommended |
| 3.11 | Fully Supported | Recommended |
| 3.12 | Partial | Some packages may not support |
| 3.9 | Legacy | May work but not tested |

## Related Skills

- `causal-ddml` - Double Machine Learning workflows
- `estimator-iv` - Instrumental Variables estimation
- `estimator-rd` - Regression Discontinuity designs
- `causal-mediation-ml` - Mediation analysis
