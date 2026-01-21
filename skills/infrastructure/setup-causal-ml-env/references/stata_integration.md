# Stata Integration for Causal Inference

This document covers integrating Stata with Python for econometric workflows.

## Overview

Stata integration is **optional** but useful for:
- Replicating published econometric research
- Using Stata-specific estimators
- Collaborating with researchers using Stata
- Accessing proprietary Stata commands

## Detection and Requirements

### Stata Versions
- **Stata 17+**: Native Python integration via `pystata`
- **Stata 15-16**: Requires third-party bridges
- **Older versions**: Limited integration options

### License Requirements
Stata is commercial software requiring a valid license:
- Stata/MP (multiprocessor)
- Stata/SE (standard edition)
- Stata/BE (basic edition)

---

## Stata 17+ Integration (pystata)

Stata 17 introduced official Python integration through the `pystata` module.

### Configuration

```python
import stata_setup

# Configure Stata - adjust path to your installation
# Windows
stata_setup.config("C:/Program Files/Stata17", "mp")

# macOS
stata_setup.config("/Applications/Stata", "mp")

# Linux
stata_setup.config("/usr/local/stata17", "mp")
```

### Basic Usage

```python
from pystata import stata

# Run Stata commands
stata.run("sysuse auto")
stata.run("summarize price mpg")
stata.run("regress price mpg weight")

# Get output
stata.run("display _b[mpg]")
```

### Data Transfer

```python
import pandas as pd
from pystata import stata

# Load data from Python to Stata
df = pd.DataFrame({
    'x': [1, 2, 3, 4, 5],
    'y': [2, 4, 5, 4, 5]
})
stata.pdataframe_to_data(df)

# Run analysis in Stata
stata.run("regress y x")

# Get results back to Python
results = stata.get_return()
```

### Matrices and Scalars

```python
from pystata import stata
import numpy as np

# Get Stata matrix
stata.run("matrix A = (1, 2 \\ 3, 4)")
A = np.array(stata.get_return()['A'])

# Get scalars
stata.run("scalar r2 = e(r2)")
r2 = stata.get_return()['r2']
```

---

## Alternative: stata_kernel (Jupyter)

For Jupyter notebook workflows, `stata_kernel` provides Stata cells.

### Installation

```bash
pip install stata_kernel
python -m stata_kernel.install
```

### Configuration

Create `~/.stata_kernel.conf`:
```ini
[stata_kernel]
stata_path = C:\Program Files\Stata17\StataMP-64.exe
cache_directory = ~/.stata_kernel_cache
```

### Usage in Jupyter

```stata
%%stata
sysuse auto
regress price mpg weight
```

---

## Alternative: ipystata

Another option for Stata-Jupyter integration.

### Installation

```bash
pip install ipystata
```

### Configuration

```python
import ipystata
from ipystata.config import config_stata
config_stata("C:/Program Files/Stata17/StataMP-64.exe")
```

---

## Common Causal Inference Commands in Stata

### Regression Discontinuity

```stata
* Using rdrobust (install: ssc install rdrobust)
ssc install rdrobust
rdrobust Y X, c(0)
rdplot Y X, c(0)
```

### Difference-in-Differences

```stata
* Using did_multiplegt (Chaisemartin & D'Haultfoeuille)
ssc install did_multiplegt
did_multiplegt Y G T D, robust_dynamic

* Using csdid (Callaway & Sant'Anna)
ssc install csdid
csdid Y X, time(year) gvar(first_treat) method(dripw)
```

### Instrumental Variables

```stata
* 2SLS
ivregress 2sls Y (X = Z), robust

* Weak instrument tests
estat firststage
estat overid
```

### Matching

```stata
* Propensity score matching
ssc install psmatch2
psmatch2 treat X1 X2 X3, outcome(Y) neighbor(1)

* Nearest neighbor matching
ssc install nnmatch
nnmatch Y treat X1 X2, m(4) tc(att)
```

### Synthetic Control

```stata
* Using synth
ssc install synth
synth depvar predictors, trunit(1) trperiod(2000) figure
```

---

## Stata Package Installation

Many causal inference commands require additional packages.

### Install from SSC

```stata
* Core causal packages
ssc install rdrobust
ssc install rddensity
ssc install rdlocrand
ssc install psmatch2
ssc install nnmatch
ssc install synth
ssc install did_multiplegt
ssc install csdid
ssc install did_imputation
ssc install eventstudyinteract
ssc install bacondecomp
```

### Install from GitHub

```stata
* Using net install
net install csdid, from("https://raw.githubusercontent.com/friosavila/csdid_drdid/main/code/")
```

---

## Cross-Platform Installation Paths

### Windows
```
C:\Program Files\Stata17\
C:\Program Files\Stata18\
C:\Program Files (x86)\Stata\
```

### macOS
```
/Applications/Stata/
/Applications/Stata 17/
/Applications/Stata 18/
```

### Linux
```
/usr/local/stata17/
/usr/local/stata18/
/opt/stata17/
```

---

## Troubleshooting

### pystata Not Found

```python
# Add Stata utilities to path
import sys
sys.path.append("C:/Program Files/Stata17/utilities")
import stata_setup
```

### License Errors

Ensure your Stata license is valid:
```stata
query
about
```

### Path Issues on Windows

Use raw strings or forward slashes:
```python
# Either works
stata_setup.config(r"C:\Program Files\Stata17", "mp")
stata_setup.config("C:/Program Files/Stata17", "mp")
```

### Memory Issues

For large datasets:
```python
stata.run("set maxvar 10000")
stata.run("set matsize 10000")
```

---

## Comparison: When to Use Stata vs Python

| Use Case | Stata | Python |
|----------|-------|--------|
| Replicating published research | Preferred | Secondary |
| New causal ML methods | Secondary | Preferred |
| Traditional econometrics | Strong | Good |
| Machine learning integration | Limited | Excellent |
| Visualization | Good | Excellent |
| Big data | Limited | Preferred |
| Team collaboration (academia) | Common | Growing |
| Reproducibility | Good | Excellent |

---

## Example Workflow: Python + Stata

```python
import pandas as pd
import numpy as np
from pystata import stata
import stata_setup

# Configure Stata
stata_setup.config("C:/Program Files/Stata17", "mp")

# Create data in Python
np.random.seed(42)
n = 1000
df = pd.DataFrame({
    'X': np.random.randn(n),
    'Z': np.random.randn(n),
    'treat': np.random.binomial(1, 0.5, n)
})
df['Y'] = 1 + 2*df['treat'] + 0.5*df['X'] + np.random.randn(n)

# Send to Stata
stata.pdataframe_to_data(df)

# Run analysis in Stata
stata.run('''
    * Summary statistics
    summarize Y X treat

    * OLS
    regress Y treat X

    * Store results
    scalar ate = _b[treat]
    scalar se = _se[treat]
''')

# Get results back
results = stata.get_return()
print(f"ATE: {results['ate']:.3f} (SE: {results['se']:.3f})")
```

---

## Resources

- [Stata Python Integration](https://www.stata.com/python/)
- [pystata Documentation](https://www.stata.com/python/pystata/)
- [stata_kernel GitHub](https://github.com/kylebarron/stata_kernel)
- [Stata Journal](https://www.stata-journal.com/)
