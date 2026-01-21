# R Packages for Causal Inference

This document covers R packages used via the rpy2 bridge for advanced causal inference methods.

## Overview

R provides specialized packages for causal inference methods that are either:
- Not available in Python
- More mature/validated in R
- Required for replicating published research

The rpy2 bridge enables calling these packages from Python workflows.

---

## Core Causal Inference Packages

### grf (Generalized Random Forests)

The gold standard for causal forests and heterogeneous treatment effects.

| Attribute | Value |
|-----------|-------|
| Package | `grf` |
| CRAN | https://cran.r-project.org/package=grf |
| Repository | https://github.com/grf-labs/grf |
| Key Authors | Athey, Wager, Tibshirani |
| License | GPL-3 |

**Key Functions:**
- `causal_forest()` - Estimate heterogeneous treatment effects
- `regression_forest()` - Nonparametric regression
- `instrumental_forest()` - IV with heterogeneous effects
- `probability_forest()` - Classification
- `quantile_forest()` - Quantile regression

**Installation:**
```r
install.packages("grf")
```

**Usage via rpy2:**
```python
import rpy2.robjects as ro
from rpy2.robjects import pandas2ri
from rpy2.robjects.packages import importr

pandas2ri.activate()
grf = importr('grf')

# Prepare data
X = ro.r['as.matrix'](ro.conversion.py2rpy(X_df))
W = ro.FloatVector(treatment_vector)
Y = ro.FloatVector(outcome_vector)

# Fit causal forest
cf = grf.causal_forest(X, Y, W)

# Get treatment effects
tau_hat = grf.predict(cf)
```

---

### mediation (Causal Mediation Analysis)

Implements Imai, Keele, and Tingley's mediation analysis framework.

| Attribute | Value |
|-----------|-------|
| Package | `mediation` |
| CRAN | https://cran.r-project.org/package=mediation |
| Key Authors | Imai, Keele, Tingley, Yamamoto |
| License | GPL-2 |

**Key Functions:**
- `mediate()` - Main mediation analysis function
- `summary()` - Results summary with confidence intervals
- `plot()` - Visualization of effects

**Estimated Quantities:**
- ACME (Average Causal Mediation Effect)
- ADE (Average Direct Effect)
- Total Effect
- Proportion Mediated

**Installation:**
```r
install.packages("mediation")
```

**Usage via rpy2:**
```python
import rpy2.robjects as ro
from rpy2.robjects.packages import importr

mediation = importr('mediation')

# Fit mediator model
med_model = ro.r('lm(mediator ~ treatment + X1 + X2, data = df)')

# Fit outcome model
out_model = ro.r('lm(outcome ~ treatment + mediator + X1 + X2, data = df)')

# Run mediation analysis
med_result = mediation.mediate(
    med_model, out_model,
    treat='treatment', mediator='mediator',
    sims=1000
)
```

---

### rdrobust (Robust RDD Estimation)

Industry-standard package for regression discontinuity designs.

| Attribute | Value |
|-----------|-------|
| Package | `rdrobust` |
| CRAN | https://cran.r-project.org/package=rdrobust |
| Key Authors | Cattaneo, Idrobo, Titiunik |
| License | GPL-2 |

**Key Functions:**
- `rdrobust()` - Main RDD estimation
- `rdplot()` - RDD visualization
- `rdbwselect()` - Bandwidth selection

**Features:**
- Optimal bandwidth selection (MSE, CER)
- Robust bias-corrected inference
- Local polynomial regression
- Triangular, uniform, Epanechnikov kernels

**Installation:**
```r
install.packages("rdrobust")
```

**Usage via rpy2:**
```python
import rpy2.robjects as ro
from rpy2.robjects.packages import importr

rdrobust = importr('rdrobust')

# Convert data
Y = ro.FloatVector(outcome_vector)
X = ro.FloatVector(running_variable)

# Run RDD estimation
result = rdrobust.rdrobust(Y, X, c=0)  # c is the cutoff

# Extract results
ro.r('summary')(result)
```

---

### rddensity (RDD Density Tests)

Implements McCrary and Cattaneo-Jansson-Ma density tests for RDD validity.

| Attribute | Value |
|-----------|-------|
| Package | `rddensity` |
| CRAN | https://cran.r-project.org/package=rddensity |
| Key Authors | Cattaneo, Jansson, Ma |
| License | GPL-2 |

**Key Functions:**
- `rddensity()` - Density discontinuity test
- `rdplotdensity()` - Density plot at cutoff

**Purpose:**
Tests the identifying assumption that units cannot precisely manipulate the running variable to cross the treatment threshold.

**Installation:**
```r
install.packages("rddensity")
```

**Usage via rpy2:**
```python
import rpy2.robjects as ro
from rpy2.robjects.packages import importr

rddensity = importr('rddensity')

X = ro.FloatVector(running_variable)
result = rddensity.rddensity(X, c=0)
```

---

## Additional Recommended Packages

### did (Difference-in-Differences)

Implements Callaway and Sant'Anna's DiD estimator for staggered treatment.

| Attribute | Value |
|-----------|-------|
| Package | `did` |
| CRAN | https://cran.r-project.org/package=did |
| Key Authors | Callaway, Sant'Anna |

**Key Features:**
- Staggered treatment timing
- Group-time average treatment effects
- Aggregation methods
- Doubly robust estimation

**Installation:**
```r
install.packages("did")
```

---

### synthdid (Synthetic Difference-in-Differences)

Arkhangelsky et al.'s synthetic DiD method.

| Attribute | Value |
|-----------|-------|
| Package | `synthdid` |
| CRAN | https://cran.r-project.org/package=synthdid |

**Installation:**
```r
install.packages("synthdid")
```

---

### Matching

Propensity score and covariate matching.

| Attribute | Value |
|-----------|-------|
| Package | `Matching` |
| CRAN | https://cran.r-project.org/package=Matching |

**Installation:**
```r
install.packages("Matching")
```

---

### MatchIt

Flexible matching methods.

| Attribute | Value |
|-----------|-------|
| Package | `MatchIt` |
| CRAN | https://cran.r-project.org/package=MatchIt |

**Installation:**
```r
install.packages("MatchIt")
```

---

### WeightIt

Propensity score and balancing weights.

| Attribute | Value |
|-----------|-------|
| Package | `WeightIt` |
| CRAN | https://cran.r-project.org/package=WeightIt |

**Installation:**
```r
install.packages("WeightIt")
```

---

## Installation Commands

### Install All Core Packages
```r
# Core causal inference packages
install.packages(c(
  "grf",
  "mediation",
  "rdrobust",
  "rddensity"
))

# Additional recommended packages
install.packages(c(
  "did",
  "synthdid",
  "Matching",
  "MatchIt",
  "WeightIt"
))
```

### Check Installation
```r
# Verify packages are installed
packages <- c("grf", "mediation", "rdrobust", "rddensity")
for (pkg in packages) {
  if (require(pkg, character.only = TRUE)) {
    cat(sprintf("%s: version %s\n", pkg, packageVersion(pkg)))
  } else {
    cat(sprintf("%s: NOT INSTALLED\n", pkg))
  }
}
```

---

## rpy2 Bridge Setup

### Prerequisites
1. R must be installed and on PATH
2. R_HOME environment variable set (if not auto-detected)
3. rpy2 Python package installed

### Environment Variables

**Windows:**
```powershell
$env:R_HOME = "C:\Program Files\R\R-4.3.0"
$env:PATH = "$env:R_HOME\bin\x64;$env:PATH"
```

**Linux/macOS:**
```bash
export R_HOME=/usr/lib/R  # or /usr/local/Cellar/r/4.3.0 on Homebrew
export PATH=$R_HOME/bin:$PATH
```

### Verifying the Bridge

```python
import os

# Set R_HOME if needed (before importing rpy2)
# os.environ['R_HOME'] = '/path/to/R'

import rpy2.robjects as ro
from rpy2.robjects.packages import importr

# Test basic R execution
result = ro.r('1 + 1')
print(f"R returned: {result[0]}")

# Test package loading
try:
    grf = importr('grf')
    print("grf loaded successfully")
except Exception as e:
    print(f"Failed to load grf: {e}")
```

---

## Common Patterns for Using R in Python

### Converting Data Between Python and R

```python
import pandas as pd
import rpy2.robjects as ro
from rpy2.robjects import pandas2ri
from rpy2.robjects.conversion import localconverter

# Create sample data
df = pd.DataFrame({
    'Y': [1.0, 2.0, 3.0],
    'X': [0.1, 0.2, 0.3],
    'W': [0, 1, 0]
})

# Convert pandas DataFrame to R dataframe
with localconverter(ro.default_converter + pandas2ri.converter):
    r_df = ro.conversion.py2rpy(df)

# Now r_df can be used in R functions
ro.globalenv['df'] = r_df
result = ro.r('summary(df)')
print(result)
```

### Calling R Functions

```python
from rpy2.robjects.packages import importr
from rpy2.robjects import FloatVector

# Import R package
stats = importr('stats')

# Call R function
result = stats.lm('Y ~ X + W', data=r_df)

# Get summary
summary = ro.r['summary'](result)
print(summary)
```

### Extracting Results

```python
# Get coefficients from lm object
coeffs = ro.r['coef'](result)

# Convert to Python
import numpy as np
coeffs_py = np.array(coeffs)

# Get specific named elements
intercept = coeffs.rx2('(Intercept)')[0]
```

---

## Version Compatibility

| R Package | R 4.0 | R 4.1 | R 4.2 | R 4.3 |
|-----------|-------|-------|-------|-------|
| grf | Yes | Yes | Yes | Yes |
| mediation | Yes | Yes | Yes | Yes |
| rdrobust | Yes | Yes | Yes | Yes |
| rddensity | Yes | Yes | Yes | Yes |
| did | Yes | Yes | Yes | Yes |

**Recommendation:** Use R 4.2+ for best compatibility with recent package versions.
