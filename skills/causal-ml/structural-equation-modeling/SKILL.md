# Structural Equation Modeling (SEM)

> **Version**: 1.0.0 | **Type**: Causal Method
> **Aliases**: SEM, Latent Variable Modeling, Covariance Structure Analysis, LISREL
> **Languages**: Python (semopy), R (lavaan)

## Overview

Structural Equation Modeling (SEM) is a multivariate statistical framework that combines factor analysis and path analysis to model complex relationships among observed and latent variables. SEM allows researchers to specify, estimate, and test theoretical models involving both measurement error and structural (causal) relationships.

**Key Features**:
- Models both measurement error and structural relationships
- Handles latent (unobserved) constructs
- Tests overall model fit against data
- Decomposes effects into direct, indirect, and total

**Primary Use Cases**:
1. Confirmatory Factor Analysis (CFA) for construct validation
2. Path analysis with latent variables
3. Mediation analysis with measurement error correction
4. Multi-group comparison and invariance testing

## When to Use

### Ideal Scenarios
- Testing theory-driven hypotheses about variable relationships
- Modeling constructs measured by multiple indicators
- Examining mediation with latent mediators
- Comparing structural relationships across groups
- Correcting for measurement error in regression models

### Data Requirements
- [ ] Sample size: N ≥ 200 (minimum), N ≥ 10-20 per free parameter (recommended)
- [ ] Continuous or ordinal indicators (5+ categories for ML estimation)
- [ ] Multivariate normality for ML estimation (or use robust methods)
- [ ] No severe multicollinearity among indicators
- [ ] Complete data or appropriate handling of missing values

### When NOT to Use
- Purely exploratory analysis → Consider `Exploratory Factor Analysis (EFA)`
- Causal identification from observational data alone → Consider `causal-ddml` or `estimator-iv`
- Small samples (N < 100) → Consider PLS-SEM or Bayesian SEM
- Non-recursive models without proper identification → Reconsider model specification

## Model Specification

### Components of a SEM Model

**1. Measurement Model (CFA)**:
$$
\mathbf{x} = \Lambda_x \boldsymbol{\xi} + \boldsymbol{\delta}
$$
$$
\mathbf{y} = \Lambda_y \boldsymbol{\eta} + \boldsymbol{\varepsilon}
$$

Where:
- $\mathbf{x}, \mathbf{y}$: Observed indicators
- $\boldsymbol{\xi}$: Exogenous latent variables
- $\boldsymbol{\eta}$: Endogenous latent variables
- $\Lambda_x, \Lambda_y$: Factor loading matrices
- $\boldsymbol{\delta}, \boldsymbol{\varepsilon}$: Measurement errors

**2. Structural Model**:
$$
\boldsymbol{\eta} = B\boldsymbol{\eta} + \Gamma\boldsymbol{\xi} + \boldsymbol{\zeta}
$$

Where:
- $B$: Matrix of coefficients among endogenous variables
- $\Gamma$: Matrix of coefficients from exogenous to endogenous
- $\boldsymbol{\zeta}$: Structural disturbances

### Implied Covariance Matrix

SEM estimates parameters by minimizing the discrepancy between:
- **Observed covariance matrix**: $\mathbf{S}$
- **Model-implied covariance matrix**: $\boldsymbol{\Sigma}(\boldsymbol{\theta})$

$$
\boldsymbol{\Sigma}(\boldsymbol{\theta}) = \begin{bmatrix}
\boldsymbol{\Sigma}_{yy} & \boldsymbol{\Sigma}_{yx} \\
\boldsymbol{\Sigma}_{xy} & \boldsymbol{\Sigma}_{xx}
\end{bmatrix}
$$

---

## Identification Assumptions

### Necessary Conditions for Identification

| Condition | Description | Check |
|-----------|-------------|-------|
| **Scaling** | Each latent variable has a defined scale | Fix one loading = 1 or fix variance = 1 |
| **t-Rule** | df = p(p+1)/2 - t ≥ 0 | Count free parameters |
| **Three-Indicator Rule** | Each factor has ≥ 3 indicators (for just-identification) | Or 2 with constraints |
| **Recursive Structure** | No feedback loops OR proper instruments | Check model diagram |

### Model Identification Formula

Degrees of freedom:
$$
df = \frac{p(p+1)}{2} - t
$$

Where:
- $p$: Number of observed variables
- $t$: Number of free parameters to estimate
- Requirement: $df \geq 0$ (necessary but not sufficient)

### Common Identification Problems

1. **Under-identification**: More unknowns than equations → Add constraints or indicators
2. **Empirical under-identification**: Model identified in theory but not estimable with current data
3. **Heywood cases**: Negative variance estimates → Check model specification

---

## Estimation Methods

### Maximum Likelihood (ML) - Default

$$
F_{ML} = \log|\boldsymbol{\Sigma}(\boldsymbol{\theta})| + \text{tr}(\mathbf{S}\boldsymbol{\Sigma}^{-1}(\boldsymbol{\theta})) - \log|\mathbf{S}| - p
$$

**Assumptions**:
- Multivariate normality
- Large sample (asymptotic properties)

**Advantages**:
- Efficient under normality
- Well-understood properties
- Most software defaults

### Robust ML (MLR/MLM)

For non-normal data:
- **MLM**: Satorra-Bentler scaled chi-square
- **MLR**: Robust standard errors with Yuan-Bentler correction

### Weighted Least Squares (WLS/DWLS/WLSMV)

For ordinal data:
$$
F_{WLS} = (\mathbf{s} - \boldsymbol{\sigma}(\boldsymbol{\theta}))' \mathbf{W}^{-1} (\mathbf{s} - \boldsymbol{\sigma}(\boldsymbol{\theta}))
$$

**WLSMV**: Diagonally weighted least squares with mean and variance adjusted test statistic (recommended for ordinal data)

---

## Model Fit Assessment

### Fit Indices Reference Table

| Index | Acceptable | Good | Formula/Interpretation |
|-------|------------|------|------------------------|
| **χ²/df** | < 3 | < 2 | Lower is better; sensitive to N |
| **CFI** | ≥ 0.90 | ≥ 0.95 | Compares to null model |
| **TLI** | ≥ 0.90 | ≥ 0.95 | Penalizes complexity |
| **RMSEA** | < 0.08 | < 0.06 | Parsimony-adjusted |
| **SRMR** | < 0.10 | < 0.08 | Standardized residual |

### Fit Index Formulas

**Comparative Fit Index (CFI)**:
$$
CFI = 1 - \frac{\max(\chi^2_t - df_t, 0)}{\max(\chi^2_t - df_t, \chi^2_n - df_n, 0)}
$$

**Root Mean Square Error of Approximation (RMSEA)**:
$$
RMSEA = \sqrt{\frac{\chi^2 - df}{df(N-1)}}
$$

**Standardized Root Mean Square Residual (SRMR)**:
$$
SRMR = \sqrt{\frac{\sum\sum(s_{ij} - \hat{\sigma}_{ij})^2}{p(p+1)/2}}
$$

### Model Comparison

| Method | Use Case |
|--------|----------|
| **χ² Difference Test** | Nested models (ML estimation) |
| **AIC/BIC** | Non-nested models, model selection |
| **Satorra-Bentler Scaled Difference** | Robust estimation |

---

## Workflow

```
+----------------------------------------------------------------+
|                    SEM WORKFLOW                                  |
+----------------------------------------------------------------+
|  1. SPECIFICATION  -> Define measurement & structural models     |
|  2. IDENTIFICATION -> Check model can be estimated              |
|  3. ESTIMATION     -> ML, WLSMV, or Bayesian                    |
|  4. EVALUATION     -> Assess fit, residuals, modification       |
|  5. RE-SPECIFICATION -> Modify based on theory & diagnostics    |
|  6. REPORTING      -> Path diagrams, standardized coefficients  |
+----------------------------------------------------------------+
```

---

## Implementation

### Python Implementation (semopy)

```python
# pip install semopy pandas numpy

import semopy
from semopy import Model
from semopy.stats import calc_stats
import pandas as pd
import numpy as np

# Example: Generate sample data with 3 factors
np.random.seed(42)
n = 300
F1 = np.random.randn(n)
F2 = np.random.randn(n)
F3 = 0.5 * F1 + 0.3 * F2 + 0.5 * np.random.randn(n)

df = pd.DataFrame({
    'x1': 0.7 * F1 + 0.3 * np.random.randn(n),
    'x2': 0.8 * F1 + 0.3 * np.random.randn(n),
    'x3': 0.7 * F1 + 0.3 * np.random.randn(n),
    'y1': 0.7 * F2 + 0.3 * np.random.randn(n),
    'y2': 0.8 * F2 + 0.3 * np.random.randn(n),
    'y3': 0.7 * F2 + 0.3 * np.random.randn(n),
    'z1': 0.7 * F3 + 0.3 * np.random.randn(n),
    'z2': 0.8 * F3 + 0.3 * np.random.randn(n),
    'z3': 0.7 * F3 + 0.3 * np.random.randn(n),
})

# Define model in lavaan-style syntax
model_syntax = """
    # Measurement model
    LatentA =~ x1 + x2 + x3
    LatentB =~ y1 + y2 + y3
    LatentC =~ z1 + z2 + z3

    # Structural model
    LatentC ~ LatentA + LatentB
"""

# Fit model
sem_model = Model(model_syntax)
sem_model.fit(df)

# View parameter estimates (with standardized)
params = sem_model.inspect(std_est=True)
print(params)

# Get fit indices
stats = calc_stats(sem_model)
print(f"\nFit Indices:")
print(f"  Chi-square: {stats['chi2'].values[0]:.2f} (df={int(stats['DoF'].values[0])})")
print(f"  CFI: {stats['CFI'].values[0]:.3f}")
print(f"  TLI: {stats['TLI'].values[0]:.3f}")
print(f"  RMSEA: {stats['RMSEA'].values[0]:.3f}")
print(f"  AIC: {stats['AIC'].values[0]:.1f}")
```

### R Implementation (lavaan)

```r
library(lavaan)

# Define model
model <- '
    # Measurement model
    LatentA =~ x1 + x2 + x3
    LatentB =~ y1 + y2 + y3
    LatentC =~ z1 + z2 + z3

    # Structural model
    LatentC ~ LatentA + LatentB
    LatentB ~ LatentA
'

# Fit model
fit <- sem(model, data = df, estimator = "ML")

# Summary with fit indices
summary(fit, fit.measures = TRUE, standardized = TRUE)

# Modification indices
modindices(fit, sort = TRUE, minimum.value = 10)
```

---

## Mediation Analysis in SEM

### Advantages over Baron-Kenny
1. Accounts for measurement error
2. Simultaneous estimation of all paths
3. Better handling of multiple mediators
4. Proper standard errors for indirect effects

### Indirect Effect Testing

```python
# Define mediation model with labeled paths
mediation_model = """
    # Measurement
    M =~ m1 + m2 + m3
    Y =~ y1 + y2 + y3

    # Structural (X is observed)
    M ~ a*X
    Y ~ b*M + c*X

    # Define indirect effect
    indirect := a*b
    total := c + a*b
"""

# Fit model
sem_model = Model(mediation_model)
sem_model.fit(df)

# Get parameters including defined indirect effect
params = sem_model.inspect()
print(params)
```

### Checking Mediation Effect Significance

```python
# The indirect effect significance can be assessed from the params output
# For bootstrap confidence intervals, use R lavaan or manual bootstrap:

def bootstrap_indirect(df, model, n_boot=1000):
    """Bootstrap indirect effect for CI estimation."""
    indirect_effects = []
    for i in range(n_boot):
        boot_df = df.sample(n=len(df), replace=True)
        try:
            m = Model(model)
            m.fit(boot_df)
            # Extract a and b coefficients
            params = m.inspect()
            a = params[params['rval'] == 'X']['Estimate'].values[0]
            b = params[(params['lval'] == 'Y') & (params['rval'] == 'M')]['Estimate'].values[0]
            indirect_effects.append(a * b)
        except:
            continue
    return np.percentile(indirect_effects, [2.5, 97.5])
```

---

## Multi-Group Analysis

### Measurement Invariance Testing

| Level | Constrained | Test |
|-------|-------------|------|
| **Configural** | None (same structure) | Baseline |
| **Metric** | Factor loadings | Δχ², ΔCFI |
| **Scalar** | + Intercepts | Δχ², ΔCFI |
| **Strict** | + Residual variances | Δχ², ΔCFI |

### Implementation

Note: Multi-group analysis in semopy is limited. For full measurement invariance testing, use R lavaan:

```r
# R code for multi-group analysis
library(lavaan)

# Configural invariance (baseline)
fit_config <- cfa(model, data = df, group = "gender")

# Metric invariance (constrain loadings)
fit_metric <- cfa(model, data = df, group = "gender",
                  group.equal = c("loadings"))

# Scalar invariance (constrain loadings + intercepts)
fit_scalar <- cfa(model, data = df, group = "gender",
                  group.equal = c("loadings", "intercepts"))

# Compare models
anova(fit_config, fit_metric, fit_scalar)

# Check ΔCFI (should be < 0.01 for invariance)
fitMeasures(fit_config, c("cfi", "rmsea"))
fitMeasures(fit_metric, c("cfi", "rmsea"))
```

For Python, split data by group and fit separately:

```python
# Simple group comparison in Python
groups = df.groupby('gender')
results = {}
for name, group_df in groups:
    sem_model = Model(model_syntax)
    sem_model.fit(group_df.drop('gender', axis=1))
    results[name] = calc_stats(sem_model)
    print(f"Group {name}: CFI={results[name]['CFI'].values[0]:.3f}")
```

---

## Common Mistakes

### 1. Ignoring Identification Issues

**Mistake**: Specifying models with too few indicators per factor.

**Correct**: Each factor needs ≥ 3 indicators for local identification.

```python
# WRONG: Only 2 indicators
bad_model = "F =~ x1 + x2"

# CORRECT: 3+ indicators or constraints
good_model = "F =~ x1 + x2 + x3"
```

### 2. Over-relying on Modification Indices

**Mistake**: Adding correlated errors based solely on modification indices.

**Correct**: Only add paths with theoretical justification.

```python
# Check modification indices
mod_idx = result.modification_indices

# Only consider if theoretically justified
# e.g., same-method variance, similar wording
```

### 3. Treating SEM as Proving Causation

**Mistake**: Claiming causal effects based on good model fit.

**Correct**: SEM tests consistency with causal theory, not causation itself.

### 4. Using Chi-Square Alone for Fit Assessment

**Mistake**: Rejecting models based only on significant χ².

**Correct**: Use multiple fit indices; χ² is sensitive to sample size.

```python
# Check multiple indices
stats = calc_stats(sem_model)
print(f"χ² = {stats['chi2'].values[0]:.2f}, df = {int(stats['DoF'].values[0])}")
print(f"p = {stats['chi2 p-value'].values[0]:.4f}")
print(f"CFI = {stats['CFI'].values[0]:.3f}")
print(f"TLI = {stats['TLI'].values[0]:.3f}")
print(f"RMSEA = {stats['RMSEA'].values[0]:.3f}")
```

---

## Diagnostics

### Residual Analysis

```python
# Get observed and model-implied covariance matrices
import numpy as np

# Observed covariance
S = df.cov().values

# Model-implied covariance (from semopy)
sigma = sem_model.calc_sigma()

# Residual matrix
residuals = S - sigma

# Standardized residuals (simplified)
std_residuals = residuals / np.sqrt(np.diag(S).reshape(-1, 1) @ np.diag(S).reshape(1, -1))

# Large residuals (|r| > 2.58) indicate misfit
n_large = np.sum(np.abs(std_residuals) > 2.58)
print(f"Number of large standardized residuals (|r| > 2.58): {n_large}")
```

### Factor Loading Standards

| Loading | Interpretation |
|---------|----------------|
| < 0.40 | Poor indicator |
| 0.40 - 0.70 | Acceptable |
| > 0.70 | Good indicator |

### Reliability Measures

**Composite Reliability (ω)**:
$$
\omega = \frac{(\sum \lambda_i)^2}{(\sum \lambda_i)^2 + \sum \theta_{ii}}
$$

**Average Variance Extracted (AVE)**:
$$
AVE = \frac{\sum \lambda_i^2}{p}
$$

---

## PLS-SEM (Partial Least Squares SEM)

PLS-SEM is a variance-based alternative to covariance-based SEM (CB-SEM).

### When to Use PLS-SEM

| Scenario | Recommendation |
|----------|----------------|
| Theory testing, well-established constructs | CB-SEM (lavaan) |
| Prediction-oriented research | PLS-SEM |
| Small sample sizes (N < 100) | PLS-SEM |
| Complex models with many constructs | PLS-SEM |
| Formative measurement models | PLS-SEM |

### PLS-SEM in R

```r
library(seminr)

# Define measurement model
measurements <- constructs(
    composite("Satisfaction", c("sat1", "sat2", "sat3")),
    composite("Loyalty", c("loy1", "loy2", "loy3"))
)

# Define structural model
structure <- relationships(
    paths(from = "Satisfaction", to = "Loyalty")
)

# Estimate model
model <- estimate_pls(data, measurements, structure)
summary(model)
```

### PLS-SEM Quality Criteria

| Criterion | Threshold |
|-----------|-----------|
| Composite Reliability | > 0.70 |
| AVE | > 0.50 |
| HTMT | < 0.85 (conservative) or < 0.90 |
| VIF (inner model) | < 5 |
| R² | 0.25 weak, 0.50 moderate, 0.75 substantial |
| Q² (predictive relevance) | > 0 |

---

## References

### Seminal Works
- Bollen, K. A. (1989). *Structural Equations with Latent Variables*. Wiley. [10,826 citations]
- Jöreskog, K. G. (1969). A general approach to confirmatory maximum likelihood factor analysis. *Psychometrika*, 34(2), 183-202.
- Kline, R. B. (2016). *Principles and Practice of Structural Equation Modeling* (4th ed.). Guilford Press. [50,064 citations]

### Model Fit
- Hu, L., & Bentler, P. M. (1999). Cutoff criteria for fit indexes in covariance structure analysis. *Structural Equation Modeling*, 6(1), 1-55. [101,667 citations]
- Marsh, H. W., Hau, K. T., & Wen, Z. (2004). In Search of Golden Rules: Comment on Hypothesis-Testing Approaches. *SEM*, 11(3), 320-341. [6,322 citations]
- Satorra, A., & Bentler, P. M. (2001). A scaled difference chi-square test statistic. *Psychometrika*, 66(4), 507-514.

### PLS-SEM
- Hair, J. F., et al. (2022). Review of Partial Least Squares Structural Equation Modeling (PLS-SEM). *Structural Equation Modeling*, 31(3), 470-516. [19,070 citations]
- Hair, J. F., et al. (2021). *Partial Least Squares Structural Equation Modeling (PLS-SEM) Using R*. Springer. [5,534 citations]

### Software
- Rosseel, Y. (2012). lavaan: An R package for structural equation modeling. *Journal of Statistical Software*, 48(2), 1-36. [23,344 citations]
- semopy (Python): https://semopy.com/
- seminr (R, PLS-SEM): https://cran.r-project.org/package=seminr
- SmartPLS: https://www.smartpls.com/

### Applications
- MacKinnon, D. P. (2008). *Introduction to Statistical Mediation Analysis*. Erlbaum.
- Little, T. D. (2013). *Longitudinal Structural Equation Modeling*. Guilford Press.

---

## Related Skills

| Skill | Relationship |
|-------|--------------|
| `causal-mediation-ml` | SEM approach to mediation vs. ML approach |
| `causal-concept-guide` | Causal identification theory |
| `bayesian-econometrics` | Bayesian SEM estimation |
| `ml-preprocessing` | Data preparation for SEM |
| `statistical-analysis` | Basic statistical concepts used in SEM |
