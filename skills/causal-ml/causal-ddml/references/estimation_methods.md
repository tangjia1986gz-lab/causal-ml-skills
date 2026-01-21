# DDML Estimation Methods

> Comprehensive reference for Double/Debiased Machine Learning estimation approaches

## Overview

This document covers the main DDML estimation methods:
1. **PLR** - Partially Linear Regression
2. **IRM** - Interactive Regression Model
3. **IIVM** - Interactive IV Model
4. **PLIV** - Partially Linear IV Model

All methods use Neyman-orthogonal scores and cross-fitting.

---

## 1. Partially Linear Regression (PLR)

### Model Specification

$$
Y = D \cdot \theta_0 + g_0(X) + \epsilon, \quad E[\epsilon|D,X] = 0
$$
$$
D = m_0(X) + V, \quad E[V|X] = 0
$$

### Key Features

| Aspect | Description |
|--------|-------------|
| Treatment | Continuous or binary |
| Effect | Constant (homogeneous) ATE |
| Nuisance functions | $\ell_0(X) = E[Y|X]$, $m_0(X) = E[D|X]$ |
| Estimand | $\theta_0 = E[\partial Y / \partial D]$ |

### Orthogonal Score (Partialling Out)

$$
\psi^{PLR}(W; \theta, \ell, m) = (Y - \ell(X) - \theta(D - m(X)))(D - m(X))
$$

Setting $E[\psi] = 0$ and solving:

$$
\hat{\theta} = \frac{\sum_i (Y_i - \hat{\ell}(X_i))(D_i - \hat{m}(X_i))}{\sum_i (D_i - \hat{m}(X_i))^2}
$$

### Alternative: IV-Type Score

$$
\psi^{IV}(W; \theta, m) = (Y - \theta D)(D - m(X))
$$

This uses $D - m(X)$ as an instrument for $D$.

### Implementation

```python
from doubleml import DoubleMLPLR
from doubleml import DoubleMLData
from sklearn.linear_model import LassoCV
from sklearn.ensemble import RandomForestRegressor

# Prepare data
dml_data = DoubleMLData(df, y_col='outcome', d_cols='treatment',
                        x_cols=control_vars)

# Specify learners
ml_l = LassoCV()  # For E[Y|X]
ml_m = LassoCV()  # For E[D|X]

# Estimate
dml_plr = DoubleMLPLR(dml_data, ml_l=ml_l, ml_m=ml_m, n_folds=5)
dml_plr.fit()

print(dml_plr.summary)
```

### When to Use PLR

- Continuous or binary treatment
- Assume constant treatment effect across X
- Interest in ATE (average treatment effect)
- High-dimensional controls with potential nonlinear confounding

---

## 2. Interactive Regression Model (IRM)

### Model Specification

$$
Y = g_0(D, X) + \epsilon, \quad E[\epsilon|D,X] = 0
$$
$$
D = m_0(X) + V, \quad E[V|X] = 0
$$

where $D \in \{0, 1\}$ is binary.

### Key Features

| Aspect | Description |
|--------|-------------|
| Treatment | Binary only |
| Effect | Allows heterogeneous effects |
| Nuisance functions | $g_0(d,X) = E[Y|D=d,X]$, $m_0(X) = P(D=1|X)$ |
| Estimand | ATE = $E[g_0(1,X) - g_0(0,X)]$ |

### Estimands

**Average Treatment Effect (ATE)**:
$$
\theta_0^{ATE} = E[Y(1) - Y(0)] = E[g_0(1,X) - g_0(0,X)]
$$

**Average Treatment Effect on Treated (ATTE)**:
$$
\theta_0^{ATTE} = E[Y(1) - Y(0)|D=1] = E[g_0(1,X) - g_0(0,X)|D=1]
$$

### Orthogonal Score (AIPW/Doubly Robust)

For ATE:
$$
\psi^{ATE}(W; \theta, g, m) = g(1,X) - g(0,X) + \frac{D(Y-g(1,X))}{m(X)} - \frac{(1-D)(Y-g(0,X))}{1-m(X)} - \theta
$$

This is the **efficient influence function** and has the doubly robust property:
- Consistent if either $g$ or $m$ is correctly specified (but not both)
- Achieves semiparametric efficiency bound if both are correctly specified

### Implementation

```python
from doubleml import DoubleMLIRM
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

# Prepare data (treatment must be binary)
dml_data = DoubleMLData(df, y_col='outcome', d_cols='treatment',
                        x_cols=control_vars)

# Specify learners
ml_g = RandomForestRegressor(n_estimators=200)  # For E[Y|D,X]
ml_m = RandomForestClassifier(n_estimators=200)  # For P(D=1|X)

# Estimate ATE
dml_irm = DoubleMLIRM(dml_data, ml_g=ml_g, ml_m=ml_m,
                       n_folds=5, score='ATE')
dml_irm.fit()

print(dml_irm.summary)

# For ATTE
dml_irm_atte = DoubleMLIRM(dml_data, ml_g=ml_g, ml_m=ml_m,
                            n_folds=5, score='ATTE')
dml_irm_atte.fit()
```

### When to Use IRM

- Binary treatment
- Potential heterogeneous treatment effects
- Want doubly robust estimation
- Interest in ATE or ATTE

---

## 3. Interactive IV Model (IIVM)

### Model Specification

For settings with endogenous treatment and binary instrument:

$$
Y = g_0(D, X) + \epsilon, \quad E[\epsilon|Z,X] = 0
$$
$$
D = r_0(Z, X) + V, \quad E[V|Z,X] = 0
$$
$$
Z = m_0(X) + U, \quad E[U|X] = 0
$$

where $Z \in \{0, 1\}$ is a binary instrument.

### Key Features

| Aspect | Description |
|--------|-------------|
| Treatment | Can be endogenous |
| Instrument | Binary |
| Effect | Local ATE (LATE) for compliers |
| Nuisance functions | $g_0$, $r_0$, $m_0$ |

### Estimand: LATE

$$
\theta_0^{LATE} = \frac{E[Y(Z=1) - Y(Z=0)]}{E[D(Z=1) - D(Z=0)]}
$$

This is the effect for **compliers** - units whose treatment status is affected by the instrument.

### Orthogonal Score

$$
\psi^{IIVM} = \frac{g(1,X) - g(0,X)}{E[D|Z=1,X] - E[D|Z=0,X]} + \text{correction terms}
$$

### Implementation

```python
from doubleml import DoubleMLIIVM

# Data with instrument
dml_data = DoubleMLData(df, y_col='outcome', d_cols='treatment',
                        x_cols=control_vars, z_cols='instrument')

# Specify learners
ml_g = RandomForestRegressor()  # For E[Y|Z,X]
ml_m = RandomForestClassifier()  # For P(Z=1|X)
ml_r = RandomForestClassifier()  # For P(D=1|Z,X)

dml_iivm = DoubleMLIIVM(dml_data, ml_g=ml_g, ml_m=ml_m, ml_r=ml_r,
                         n_folds=5)
dml_iivm.fit()

print(dml_iivm.summary)
```

### When to Use IIVM

- Unconfoundedness fails (endogenous treatment)
- Valid binary instrument available
- Interest in LATE for compliers
- High-dimensional controls

---

## 4. Partially Linear IV Model (PLIV)

### Model Specification

$$
Y = D \cdot \theta_0 + g_0(X) + \epsilon, \quad E[\epsilon|Z,X] = 0
$$
$$
Z = m_0(X) + V, \quad E[V|X] = 0
$$

### Key Features

| Aspect | Description |
|--------|-------------|
| Treatment | Can be endogenous, continuous or binary |
| Instrument | Continuous or binary |
| Effect | Constant ATE (like PLR but with IV) |
| Nuisance functions | $\ell_0(X)$, $m_0(X)$, $r_0(X) = E[D|X]$ |

### Implementation

```python
from doubleml import DoubleMLPLIV

dml_data = DoubleMLData(df, y_col='outcome', d_cols='treatment',
                        x_cols=control_vars, z_cols='instrument')

ml_l = LassoCV()  # For E[Y|X]
ml_m = LassoCV()  # For E[Z|X]
ml_r = LassoCV()  # For E[D|X]

dml_pliv = DoubleMLPLIV(dml_data, ml_l=ml_l, ml_m=ml_m, ml_r=ml_r,
                         n_folds=5)
dml_pliv.fit()
```

---

## 5. Method Selection Guide

### Decision Tree

```
Is treatment endogenous?
├── No (Selection on observables)
│   ├── Binary treatment?
│   │   ├── Yes: Use IRM (allows heterogeneity) or PLR
│   │   └── No: Use PLR
│   └── Heterogeneous effects important?
│       ├── Yes: Use IRM (binary) or consider CATE methods
│       └── No: Use PLR
└── Yes (Need instrument)
    ├── Binary instrument?
    │   ├── Yes: Use IIVM (LATE)
    │   └── No: Use PLIV
    └── Constant effect assumption OK?
        ├── Yes: Use PLIV
        └── No: Use IIVM (for binary Z)
```

### Summary Table

| Method | Treatment | Instrument | Effect | Main Assumption |
|--------|-----------|------------|--------|-----------------|
| PLR | Any | None | Constant ATE | Unconfoundedness |
| IRM | Binary | None | Heterogeneous ATE | Unconfoundedness |
| PLIV | Any | Any | Constant ATE | Valid IV |
| IIVM | Any | Binary | LATE | Valid IV |

---

## 6. DoubleML Package Integration

### Installation

```bash
pip install doubleml
# or
conda install -c conda-forge doubleml
```

### Basic Workflow

```python
import doubleml as dml
from doubleml import DoubleMLData, DoubleMLPLR, DoubleMLIRM
from sklearn.linear_model import LassoCV
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

# 1. Prepare data
dml_data = DoubleMLData(
    df,
    y_col='outcome',
    d_cols='treatment',
    x_cols=control_vars
)

# 2. Choose learners
ml_l = LassoCV()
ml_m = LassoCV()  # or RandomForestClassifier for binary D

# 3. Initialize model
dml_model = DoubleMLPLR(
    dml_data,
    ml_l=ml_l,
    ml_m=ml_m,
    n_folds=5,
    n_rep=1,
    score='partialling out'
)

# 4. Fit
dml_model.fit()

# 5. Results
print(dml_model.summary)
print(f"Effect: {dml_model.coef[0]:.4f}")
print(f"SE: {dml_model.se[0]:.4f}")
print(f"95% CI: {dml_model.confint()}")
```

### Advanced Features

```python
# Multiple treatments
dml_data = DoubleMLData(df, y_col='y', d_cols=['d1', 'd2'], x_cols=x_cols)

# Clustered standard errors
dml_model = DoubleMLPLR(dml_data, ml_l=ml_l, ml_m=ml_m)
dml_model.fit()
# Bootstrap for clustered SEs
dml_model.bootstrap(method='wild', n_rep=500)

# Sensitivity analysis (with sensemakr)
dml_model.sensitivity_analysis()
```

---

## 7. Comparison: PLR vs IRM

### Monte Carlo Evidence

When true model is **partially linear** (constant effect):
- PLR: Unbiased, efficient
- IRM: Unbiased but slightly less efficient

When true model has **heterogeneous effects**:
- PLR: Estimates weighted average (may be biased for policy)
- IRM: Correctly estimates ATE

### Practical Recommendation

1. **Start with PLR** for continuous treatment or when heterogeneity is not the focus
2. **Use IRM for binary treatment** when effects may vary with X
3. **Compare both** when uncertain - if estimates differ substantially, heterogeneity matters

---

## 8. Asymptotic Theory Summary

### Key Result (Chernozhukov et al. 2018)

Under regularity conditions:

$$
\sqrt{n}(\hat{\theta} - \theta_0) \xrightarrow{d} N(0, \sigma^2)
$$

where:

$$
\sigma^2 = E[\psi^2] / (E[\partial_\theta \psi])^2
$$

### Rate Requirements

For valid inference, nuisance estimators must satisfy:

$$
\|\hat{\eta} - \eta_0\| = o_P(n^{-1/4})
$$

and the product condition:

$$
\|\hat{\ell} - \ell_0\| \cdot \|\hat{m} - m_0\| = o_P(n^{-1/2})
$$

### What This Means Practically

1. ML learners can converge slower than $\sqrt{n}$
2. But their **product** must be fast enough
3. Cross-fitting eliminates overfitting bias
4. Standard errors are valid for inference

---

## References

1. Chernozhukov, V., et al. (2018). Double/Debiased Machine Learning for Treatment and Structural Parameters. *The Econometrics Journal*, 21(1), C1-C68.

2. Bach, P., et al. (2022). DoubleML: An Object-Oriented Implementation of Double Machine Learning in Python. *Journal of Machine Learning Research*, 23(53), 1-6.

3. DoubleML Documentation: https://docs.doubleml.org/

4. Semenova, V., & Chernozhukov, V. (2021). Debiased Machine Learning of Conditional Average Treatment Effects. *The Econometrics Journal*, 24(2), 264-289.
