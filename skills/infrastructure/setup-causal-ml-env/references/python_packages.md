# Python Packages for Causal Inference

This document provides detailed information about Python packages used in causal inference workflows.

## Core Causal Inference Packages

### EconML (Microsoft)

Microsoft's library for machine learning-based causal inference.

| Attribute | Value |
|-----------|-------|
| Package | `econml` |
| Minimum Version | 0.15.0 |
| Python Support | 3.8-3.11 |
| License | MIT |
| Repository | https://github.com/py-why/EconML |

**Key Features:**
- Double Machine Learning (DML)
- Causal Forests (Orthogonal Random Forest)
- Instrumental Variable estimation
- Policy learning and optimization
- Heterogeneous treatment effect estimation

**Installation:**
```bash
pip install econml>=0.15.0
```

**Common Issues:**
- Requires Cython and NumPy pre-installed for some build configurations
- May conflict with older scikit-learn versions

---

### DoubleML

Double/Debiased Machine Learning for causal effects.

| Attribute | Value |
|-----------|-------|
| Package | `doubleml` |
| Minimum Version | 0.7.0 |
| Python Support | 3.8-3.11 |
| License | MIT |
| Repository | https://github.com/DoubleML/doubleml-for-py |

**Key Features:**
- Partially Linear Regression (PLR)
- Interactive Regression Model (IRM)
- Instrumental Variables Regression (IIVM)
- Panel data support
- Cross-fitting infrastructure

**Installation:**
```bash
pip install doubleml>=0.7.0
```

**Key Classes:**
- `DoubleMLData` - Data container for DML
- `DoubleMLPLR` - Partially Linear Regression
- `DoubleMLIRM` - Interactive Regression Model
- `DoubleMLIIVM` - Interactive IV Model

---

### CausalML (Uber)

Uber's package for uplift modeling and causal inference.

| Attribute | Value |
|-----------|-------|
| Package | `causalml` |
| Minimum Version | 0.15.0 |
| Python Support | 3.8-3.11 |
| License | Apache 2.0 |
| Repository | https://github.com/uber/causalml |

**Key Features:**
- Meta-learners (S, T, X, R, DR)
- Tree-based methods (Uplift Tree, Causal Tree)
- Propensity score methods
- SHAP-based interpretability
- Sensitivity analysis

**Installation:**
```bash
pip install causalml>=0.15.0
```

**Important:** CausalML has specific SHAP version requirements. Install SHAP first if you encounter conflicts:
```bash
pip install shap==0.42.1
pip install causalml
```

---

### DoWhy (Microsoft)

End-to-end causal inference framework.

| Attribute | Value |
|-----------|-------|
| Package | `dowhy` |
| Minimum Version | 0.11.0 |
| Python Support | 3.8-3.11 |
| License | MIT |
| Repository | https://github.com/py-why/dowhy |

**Key Features:**
- Causal graph specification (DAGs)
- Multiple identification strategies
- Estimation with various methods
- Refutation tests
- Sensitivity analysis

**Installation:**
```bash
pip install dowhy>=0.11.0
```

---

## Statistical & Econometric Packages

### Statsmodels

Comprehensive statistical modeling library.

| Attribute | Value |
|-----------|-------|
| Package | `statsmodels` |
| Minimum Version | 0.14.0 |
| Python Support | 3.8+ |
| License | BSD-3-Clause |
| Repository | https://github.com/statsmodels/statsmodels |

**Key Features:**
- OLS, WLS, GLS regression
- Time series models (ARIMA, VAR)
- Generalized Linear Models
- Statistical tests
- Robust standard errors

**Key Modules for Causal Inference:**
- `statsmodels.regression.linear_model.OLS`
- `statsmodels.stats.diagnostic` - Specification tests
- `statsmodels.stats.outliers_influence` - Influence diagnostics

---

### Linearmodels

Panel data and instrumental variables estimation.

| Attribute | Value |
|-----------|-------|
| Package | `linearmodels` |
| Minimum Version | 6.0 |
| Python Support | 3.9+ |
| License | BSD-3-Clause |
| Repository | https://github.com/bashtage/linearmodels |

**Key Features:**
- Panel data models (Fixed Effects, Random Effects)
- Instrumental Variables (2SLS, LIML, GMM)
- System estimation (SUR, 3SLS)
- Clustered standard errors

**Key Classes:**
- `linearmodels.iv.IV2SLS` - Two-stage least squares
- `linearmodels.panel.PanelOLS` - Panel OLS with fixed effects
- `linearmodels.panel.RandomEffects` - Random effects model

---

## Machine Learning Packages

### Scikit-learn

Foundation for machine learning in Python.

| Attribute | Value |
|-----------|-------|
| Package | `scikit-learn` |
| Minimum Version | 1.3.0 |
| Python Support | 3.9+ |
| License | BSD-3-Clause |

**Causal Inference Uses:**
- Cross-validation infrastructure
- Propensity score estimation (LogisticRegression)
- Outcome modeling (RandomForest, GradientBoosting)
- Feature preprocessing (StandardScaler, OneHotEncoder)

---

### XGBoost

Gradient boosting for heterogeneous effects.

| Attribute | Value |
|-----------|-------|
| Package | `xgboost` |
| Minimum Version | 2.0.0 |
| Python Support | 3.8+ |
| License | Apache 2.0 |

**Causal Inference Uses:**
- First-stage learners in DML
- Propensity score models
- Outcome regression in meta-learners
- Handling high-dimensional confounders

**GPU Support:**
```bash
pip install xgboost --upgrade  # GPU support included by default
```

---

### LightGBM

Fast gradient boosting framework.

| Attribute | Value |
|-----------|-------|
| Package | `lightgbm` |
| Minimum Version | 4.0.0 |
| Python Support | 3.8+ |
| License | MIT |

**Causal Inference Uses:**
- Alternative to XGBoost for nuisance estimation
- Faster training for large datasets
- Categorical feature handling

**macOS (Apple Silicon) Installation:**
```bash
brew install libomp
pip install lightgbm
```

---

## Data & Visualization Packages

### Pandas

Data manipulation and analysis.

| Attribute | Value |
|-----------|-------|
| Package | `pandas` |
| Minimum Version | 2.0.0 |
| Python Support | 3.9+ |
| License | BSD-3-Clause |

---

### NumPy

Numerical computing foundation.

| Attribute | Value |
|-----------|-------|
| Package | `numpy` |
| Minimum Version | 1.24.0 |
| Python Support | 3.9+ |
| License | BSD-3-Clause |

**Note:** NumPy 2.0+ may have compatibility issues with older packages. Use NumPy 1.x if you encounter problems.

---

### Matplotlib

Plotting library.

| Attribute | Value |
|-----------|-------|
| Package | `matplotlib` |
| Minimum Version | 3.7.0 |
| Python Support | 3.8+ |
| License | PSF |

---

### Seaborn

Statistical visualization.

| Attribute | Value |
|-----------|-------|
| Package | `seaborn` |
| Minimum Version | 0.13.0 |
| Python Support | 3.8+ |
| License | BSD-3-Clause |

---

## Additional Dependencies

### SHAP

Shapley values for model interpretability.

| Attribute | Value |
|-----------|-------|
| Package | `shap` |
| Minimum Version | 0.42.0 |
| Critical For | CausalML interpretability |

**Version Conflicts:**
CausalML may require specific SHAP versions. Install SHAP before CausalML:
```bash
pip install shap==0.42.1
```

---

### NetworkX

Graph algorithms for causal DAGs.

| Attribute | Value |
|-----------|-------|
| Package | `networkx` |
| Minimum Version | 3.0 |
| Used By | DoWhy causal graphs |

---

### SciPy

Scientific computing.

| Attribute | Value |
|-----------|-------|
| Package | `scipy` |
| Minimum Version | 1.10.0 |
| Used By | Most causal packages |

---

## Version Compatibility Matrix

| Package | Python 3.10 | Python 3.11 | Python 3.12 |
|---------|-------------|-------------|-------------|
| econml | Yes | Yes | Partial |
| doubleml | Yes | Yes | Yes |
| causalml | Yes | Yes | Partial |
| dowhy | Yes | Yes | Yes |
| statsmodels | Yes | Yes | Yes |
| linearmodels | Yes | Yes | Yes |

**Recommendation:** Use Python 3.10 or 3.11 for maximum compatibility.

---

## Quick Installation Reference

### Minimal Installation
```bash
pip install econml doubleml statsmodels scikit-learn pandas numpy matplotlib
```

### Full Installation
```bash
pip install -r requirements.txt
```

### Conda Installation
```bash
conda install -c conda-forge econml doubleml statsmodels scikit-learn pandas numpy matplotlib seaborn
pip install causalml dowhy linearmodels  # Not all available on conda-forge
```
