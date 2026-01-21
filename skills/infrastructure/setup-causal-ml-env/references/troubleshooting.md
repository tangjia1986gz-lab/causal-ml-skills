# Troubleshooting Guide

This document covers common installation issues and solutions for the causal inference ML environment.

## Quick Diagnostics

Before troubleshooting, run the environment check:
```bash
python scripts/check_environment.py
```

---

## Python Package Issues

### econml Installation Fails

**Symptoms:**
- Build errors during `pip install econml`
- Missing compiler errors
- Cython/NumPy import errors

**Solutions:**

1. **Pre-install build dependencies:**
```bash
pip install cython numpy scipy
pip install econml --no-build-isolation
```

2. **Windows: Install Visual C++ Build Tools:**
   - Download from: https://visualstudio.microsoft.com/visual-cpp-build-tools/
   - Install "Desktop development with C++"

3. **Use conda-forge:**
```bash
conda install -c conda-forge econml
```

---

### causalml SHAP Conflicts

**Symptoms:**
- Version conflicts between causalml and shap
- Import errors after installation
- SHAP visualization failures

**Solutions:**

1. **Install in correct order:**
```bash
pip uninstall shap causalml -y
pip install shap==0.42.1
pip install causalml
```

2. **Use specific versions:**
```bash
pip install causalml==0.15.0 shap==0.42.1
```

---

### LightGBM on macOS (Apple Silicon)

**Symptoms:**
- `Library not loaded: libomp.dylib`
- Segmentation faults on import
- Build failures on M1/M2 Macs

**Solutions:**

1. **Install OpenMP:**
```bash
brew install libomp
```

2. **Set library path:**
```bash
export LDFLAGS="-L/opt/homebrew/opt/libomp/lib"
export CPPFLAGS="-I/opt/homebrew/opt/libomp/include"
pip install lightgbm
```

3. **Use conda:**
```bash
conda install -c conda-forge lightgbm
```

---

### NumPy 2.0 Compatibility

**Symptoms:**
- Import errors mentioning NumPy ABI
- "Module was compiled against NumPy 1.x but running with 2.x"

**Solutions:**

1. **Pin NumPy to 1.x:**
```bash
pip install "numpy<2.0"
pip install --force-reinstall econml doubleml causalml
```

2. **Rebuild all packages:**
```bash
pip install --force-reinstall --no-cache-dir econml
```

---

### scikit-learn Version Conflicts

**Symptoms:**
- Incompatible scikit-learn version warnings
- Missing estimator methods

**Solutions:**

1. **Use compatible versions:**
```bash
pip install "scikit-learn>=1.3.0,<1.4.0"
```

2. **Reinstall dependent packages:**
```bash
pip install --force-reinstall econml doubleml
```

---

## R Integration Issues

### rpy2 Cannot Find R

**Symptoms:**
- `R_HOME` not set error
- "R shared library not found"
- rpy2 import failures

**Solutions:**

1. **Windows - Set R_HOME:**
```powershell
# Find R installation
where R

# Set environment variable
$env:R_HOME = "C:\Program Files\R\R-4.3.0"
$env:PATH = "$env:R_HOME\bin\x64;$env:PATH"

# Make permanent (run as admin)
[Environment]::SetEnvironmentVariable("R_HOME", "C:\Program Files\R\R-4.3.0", "Machine")
```

2. **Linux - Set R_HOME:**
```bash
# Find R
which R
R RHOME

# Set in ~/.bashrc or ~/.zshrc
export R_HOME=/usr/lib/R
export LD_LIBRARY_PATH=$R_HOME/lib:$LD_LIBRARY_PATH
```

3. **macOS - Set R_HOME:**
```bash
# Homebrew R
export R_HOME=/opt/homebrew/Cellar/r/4.3.0/lib/R

# CRAN R
export R_HOME=/Library/Frameworks/R.framework/Resources
```

---

### rpy2 Import Error on Windows

**Symptoms:**
- DLL load failed
- "R shared library" errors
- Unable to load R.dll

**Solutions:**

1. **Add R bin to PATH:**
```powershell
$env:PATH = "C:\Program Files\R\R-4.3.0\bin\x64;$env:PATH"
```

2. **Use 64-bit consistently:**
   - Ensure Python is 64-bit
   - Ensure R is 64-bit
   - Use `R-4.x.x\bin\x64` not `bin\i386`

3. **Reinstall rpy2:**
```bash
pip uninstall rpy2
pip install rpy2 --no-cache-dir
```

---

### R Package Installation Fails

**Symptoms:**
- Compilation errors for R packages
- Missing system libraries
- Timeout during installation

**Solutions:**

1. **Windows - Install Rtools:**
   - Download Rtools from: https://cran.r-project.org/bin/windows/Rtools/
   - Add to PATH

2. **Linux - Install system dependencies:**
```bash
# Ubuntu/Debian
sudo apt-get install r-base-dev libcurl4-openssl-dev libssl-dev libxml2-dev

# Fedora/RHEL
sudo dnf install R-devel libcurl-devel openssl-devel libxml2-devel
```

3. **macOS - Install Xcode tools:**
```bash
xcode-select --install
```

---

### grf Installation Issues

**Symptoms:**
- Compilation fails
- "cannot find -lgfortran"
- RcppEigen errors

**Solutions:**

1. **Install dependencies first:**
```r
install.packages(c("Rcpp", "RcppEigen", "Matrix"))
install.packages("grf")
```

2. **Linux - Install gfortran:**
```bash
sudo apt-get install gfortran
```

---

## Stata Integration Issues

### pystata Import Error

**Symptoms:**
- ModuleNotFoundError: pystata
- Stata utilities not found

**Solutions:**

1. **Add Stata utilities to Python path:**
```python
import sys
sys.path.insert(0, "C:/Program Files/Stata17/utilities")
import stata_setup
stata_setup.config("C:/Program Files/Stata17", "mp")
```

2. **Verify Stata version:**
   - pystata requires Stata 17+

---

### Stata License Errors

**Symptoms:**
- "Cannot find license" errors
- Stata initialization fails

**Solutions:**

1. **Verify license:**
```stata
query
about
```

2. **Check license file location:**
   - Windows: `C:\Program Files\Stata17\stata.lic`
   - macOS: `/Applications/Stata/stata.lic`
   - Linux: Check `$HOME/stata.lic` or installation directory

---

## Environment Issues

### Virtual Environment Conflicts

**Symptoms:**
- Packages work in one environment but not another
- Version mismatches between environments

**Solutions:**

1. **Create fresh environment:**
```bash
# Using venv
python -m venv causal-env-new
source causal-env-new/bin/activate  # Linux/macOS
causal-env-new\Scripts\activate     # Windows

pip install -r requirements.txt
```

2. **Using conda:**
```bash
conda create -n causal-ml python=3.11
conda activate causal-ml
pip install -r requirements.txt
```

---

### Conda vs Pip Conflicts

**Symptoms:**
- Packages installed but not importable
- Multiple versions detected
- ImportErrors after conda install

**Solutions:**

1. **Prefer pip in conda environments:**
```bash
conda create -n causal-ml python=3.11
conda activate causal-ml
pip install -r requirements.txt  # Use pip, not conda
```

2. **Or use conda-forge consistently:**
```bash
conda install -c conda-forge econml doubleml statsmodels
```

---

### Windows Long Path Issues

**Symptoms:**
- "The filename or extension is too long"
- Installation fails with path errors

**Solutions:**

1. **Enable long paths:**
```powershell
# Run as Administrator
New-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" -Name "LongPathsEnabled" -Value 1 -PropertyType DWORD -Force
```

2. **Use shorter install paths:**
   - Install Python to `C:\Python311` instead of deep paths
   - Create virtual environments in short paths

---

## Platform-Specific Notes

### Windows

- Use Python from python.org (not Windows Store version)
- Install Visual C++ Build Tools for compiling packages
- Use PowerShell or Windows Terminal for better color support
- Path separators: Use `/` or raw strings `r"C:\path"`

### macOS

- Homebrew Python recommended: `brew install python@3.11`
- May need Xcode Command Line Tools: `xcode-select --install`
- Apple Silicon (M1/M2): Some packages need Rosetta or native builds

### Linux

- Install python3-dev/python3-devel for header files
- Install build-essential/Development Tools for compilers
- Check system Python vs user Python in PATH

---

## Getting Help

If issues persist:

1. **Check package GitHub issues:**
   - https://github.com/py-why/EconML/issues
   - https://github.com/DoubleML/doubleml-for-py/issues
   - https://github.com/uber/causalml/issues

2. **Stack Overflow:**
   - Tag: `[econml]`, `[causalml]`, `[rpy2]`

3. **Provide diagnostic information:**
```bash
python --version
pip list
python -c "import sys; print(sys.platform)"
python scripts/check_environment.py
```

---

## Clean Reinstall Procedure

When all else fails:

```bash
# 1. Create new virtual environment
python -m venv causal-ml-clean
source causal-ml-clean/bin/activate  # Linux/macOS
# causal-ml-clean\Scripts\activate   # Windows

# 2. Upgrade pip
pip install --upgrade pip setuptools wheel

# 3. Install in stages
pip install numpy scipy cython

pip install scikit-learn pandas matplotlib seaborn

pip install statsmodels linearmodels

pip install econml doubleml

pip install shap==0.42.1
pip install causalml

pip install dowhy networkx

pip install rpy2  # Only if R is configured

# 4. Verify
python scripts/check_environment.py
```
