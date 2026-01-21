# IV Estimation Methods

> **Reference Document** | IV Estimator Skill
> **Last Updated**: 2024

## Overview

This document covers the main estimation methods for Instrumental Variables models: 2SLS, LIML, GMM, JIVE, and approaches for handling many instruments. Each method has different properties regarding bias, efficiency, and robustness to weak instruments.

---

## 1. Two-Stage Least Squares (2SLS)

### 1.1 Basic Formulation

**Structural Equation**:
$$
Y_i = X_i'\beta + D_i\gamma + \epsilon_i
$$

where $D_i$ is endogenous: $Cov(D_i, \epsilon_i) \neq 0$

**First Stage**:
$$
D_i = X_i'\pi + Z_i'\delta + v_i
$$

**Second Stage**:
$$
Y_i = X_i'\beta + \hat{D}_i\gamma + u_i
$$

where $\hat{D}_i = X_i'\hat{\pi} + Z_i'\hat{\delta}$

### 1.2 Matrix Formulation

Let $W = [X, Z]$ be all exogenous variables.

**2SLS Estimator**:
$$
\hat{\gamma}_{2SLS} = (D'P_W D)^{-1} D'P_W Y
$$

where $P_W = W(W'W)^{-1}W'$ is the projection matrix.

### 1.3 Properties

| Property | 2SLS |
|----------|------|
| **Consistency** | Yes (if instruments valid and strong) |
| **Finite sample bias** | Toward OLS as instruments weaken |
| **Efficiency** | Optimal under homoskedasticity |
| **Robustness to weak IV** | Poor |

### 1.4 Implementation

```python
from iv_estimator import estimate_2sls

result_2sls = estimate_2sls(
    data=df,
    outcome="y",
    treatment="d",
    instruments=["z1", "z2"],
    controls=["x1", "x2"]
)

print(f"2SLS Estimate: {result_2sls.effect:.4f}")
print(f"Standard Error: {result_2sls.se:.4f}")
print(f"95% CI: [{result_2sls.ci_lower:.4f}, {result_2sls.ci_upper:.4f}]")
```

### 1.5 Standard Error Computation

**Homoskedastic Standard Errors**:
$$
\widehat{Var}(\hat{\gamma}) = \hat{\sigma}^2 (D'P_W D)^{-1}
$$

where $\hat{\sigma}^2 = \frac{1}{n-k}(Y - D\hat{\gamma})'(Y - D\hat{\gamma})$

**Important**: Use residuals from actual D, not fitted $\hat{D}$!

**Heteroskedasticity-Robust (HC1)**:
$$
\widehat{Var}_{HC1}(\hat{\gamma}) = (D'P_W D)^{-1} D'P_W \hat{\Omega} P_W D (D'P_W D)^{-1}
$$

where $\hat{\Omega} = diag(\hat{u}_i^2)$

---

## 2. Limited Information Maximum Likelihood (LIML)

### 2.1 Motivation

LIML is more robust to weak instruments than 2SLS. While 2SLS minimizes the sum of squared residuals in the second stage, LIML maximizes the likelihood under normality assumptions.

### 2.2 Formulation

The LIML estimator solves:
$$
\min_{\gamma} \frac{(Y - D\gamma)'M_X(Y - D\gamma)}{(Y - D\gamma)'M_W(Y - D\gamma)}
$$

where:
- $M_X = I - X(X'X)^{-1}X'$ (residual maker for controls only)
- $M_W = I - W(W'W)^{-1}W'$ (residual maker for controls + instruments)

### 2.3 The $\kappa$ Parameter

LIML can be written as:
$$
\hat{\gamma}_{LIML} = (D'M_X D - \hat{\kappa} D'M_W D)^{-1}(D'M_X Y - \hat{\kappa} D'M_W Y)
$$

where $\hat{\kappa}$ is the smallest eigenvalue of:
$$
\begin{bmatrix} Y'M_W Y & Y'M_W D \\ D'M_W Y & D'M_W D \end{bmatrix}^{-1}
\begin{bmatrix} Y'M_X Y & Y'M_X D \\ D'M_X Y & D'M_X D \end{bmatrix}
$$

**Note**: When $\hat{\kappa} = 1$, LIML = 2SLS.

### 2.4 Properties

| Property | LIML |
|----------|------|
| **Consistency** | Yes (if instruments valid) |
| **Median bias** | Approximately median-unbiased |
| **Efficiency** | Slightly less efficient than 2SLS with strong IV |
| **Robustness to weak IV** | **Much better than 2SLS** |

### 2.5 Implementation

```python
from iv_estimator import estimate_liml

result_liml = estimate_liml(
    data=df,
    outcome="y",
    treatment="d",
    instruments=["z1", "z2"],
    controls=["x1", "x2"]
)

print(f"LIML Estimate: {result_liml.effect:.4f}")
print(f"Kappa: {result_liml.diagnostics.get('kappa', 'N/A')}")
```

### 2.6 Fuller's Modified LIML

Fuller (1977) proposed a bias-reduced version:

$$
\hat{\gamma}_{Fuller} = (D'M_X D - \hat{\kappa}_\alpha D'M_W D)^{-1}(D'M_X Y - \hat{\kappa}_\alpha D'M_W Y)
$$

where $\hat{\kappa}_\alpha = \hat{\kappa} - \frac{\alpha}{n-K}$ for small constant $\alpha$ (typically $\alpha = 1$ or 4).

**Properties**:
- Further reduces bias
- Has finite moments (unlike standard LIML with very weak instruments)
- $\alpha = 1$ approximately minimizes MSE
- $\alpha = 4$ provides better confidence interval coverage

---

## 3. Generalized Method of Moments (GMM)

### 3.1 Moment Conditions

IV estimation can be framed as GMM with moment conditions:
$$
E[Z_i(Y_i - X_i'\beta - D_i\gamma)] = 0
$$

### 3.2 GMM Estimator

$$
\hat{\gamma}_{GMM} = \arg\min_\gamma g(\gamma)' W g(\gamma)
$$

where:
- $g(\gamma) = \frac{1}{n}\sum_i Z_i(Y_i - X_i'\beta - D_i\gamma)$ (sample moments)
- $W$ is a positive definite weighting matrix

### 3.3 Two-Step Efficient GMM

1. **First step**: Use $W = (Z'Z/n)^{-1}$ (yields 2SLS)
2. Get residuals $\hat{u}_i = Y_i - X_i'\hat{\beta} - D_i\hat{\gamma}$
3. **Second step**: Use optimal weighting matrix:
   $$
   \hat{W}_{opt} = \left(\frac{1}{n}\sum_i Z_i Z_i' \hat{u}_i^2\right)^{-1}
   $$

### 3.4 Properties

| Property | GMM |
|----------|-----|
| **Consistency** | Yes |
| **Efficiency** | Efficient under heteroskedasticity (with optimal W) |
| **J-statistic** | Natural overidentification test |
| **Robustness to weak IV** | Similar to 2SLS |

### 3.5 Implementation

```python
from iv_estimator import estimate_gmm

result_gmm = estimate_gmm(
    data=df,
    outcome="y",
    treatment="d",
    instruments=["z1", "z2"],
    controls=["x1", "x2"]
)

print(f"GMM Estimate: {result_gmm.effect:.4f}")
print(f"J-statistic: {result_gmm.diagnostics['j_statistic']:.4f}")
print(f"J p-value: {result_gmm.diagnostics['j_pvalue']:.4f}")
```

### 3.6 Continuously Updating GMM (CUE)

The CUE estimator uses the optimal weighting matrix evaluated at the parameter estimate:
$$
\hat{\gamma}_{CUE} = \arg\min_\gamma g(\gamma)' W(\gamma) g(\gamma)
$$

**Properties**:
- More robust to weak instruments than two-step GMM
- Similar properties to LIML in linear IV case
- Computationally more intensive

---

## 4. Jackknife Instrumental Variables Estimator (JIVE)

### 4.1 Motivation

JIVE reduces the many-instruments bias by using leave-one-out predicted values.

### 4.2 Formulation

Instead of $\hat{D}_i = W_i'\hat{\pi}$ from the full sample, use:
$$
\hat{D}_{i,-i} = W_i'\hat{\pi}_{-i}
$$

where $\hat{\pi}_{-i}$ is estimated leaving out observation $i$.

### 4.3 JIVE1 (Angrist, Imbens, Krueger 1999)

$$
\hat{\gamma}_{JIVE1} = \frac{\sum_i (D_i - \bar{D}_{-i})Y_i}{\sum_i (D_i - \bar{D}_{-i})D_i}
$$

where $\bar{D}_{-i}$ is the leave-one-out mean.

### 4.4 JIVE2 (Ackerberg and Devereux 2009)

Uses a modified formula that further reduces bias:
$$
\hat{\gamma}_{JIVE2} = (D'\tilde{P}D)^{-1}D'\tilde{P}Y
$$

where $\tilde{P}_{ij} = P_{ij}$ for $i \neq j$ and $\tilde{P}_{ii} = 0$.

### 4.5 Properties

| Property | JIVE |
|----------|------|
| **Consistency** | Yes |
| **Bias** | Substantially less than 2SLS with many instruments |
| **Efficiency** | Lower than 2SLS |
| **When to use** | Many instruments, potential many-IV bias |

### 4.6 Implementation

```python
def estimate_jive(data, outcome, treatment, instruments, controls=None):
    """
    Estimate IV model using JIVE (Jackknife IV Estimator).
    """
    import statsmodels.api as sm

    df = data.copy()
    n = len(df)

    # Build instrument matrix
    Z_vars = instruments.copy()
    if controls:
        Z_vars.extend(controls)
    Z = sm.add_constant(df[Z_vars])
    D = df[treatment].values
    Y = df[outcome].values

    # Compute leave-one-out fitted values
    D_hat_jive = np.zeros(n)

    for i in range(n):
        # Leave out observation i
        mask = np.ones(n, dtype=bool)
        mask[i] = False

        Z_minus_i = Z.iloc[mask]
        D_minus_i = D[mask]

        # Fit first stage without i
        model = sm.OLS(D_minus_i, Z_minus_i).fit()

        # Predict for observation i
        D_hat_jive[i] = model.predict(Z.iloc[[i]])[0]

    # JIVE estimator
    numerator = np.sum(D_hat_jive * Y)
    denominator = np.sum(D_hat_jive * D)
    gamma_jive = numerator / denominator

    return gamma_jive
```

---

## 5. Many Instruments Problem

### 5.1 The Problem

With many instruments (K grows with n), 2SLS becomes biased toward OLS even with strong instruments.

**Finite-sample bias approximation**:
$$
E[\hat{\gamma}_{2SLS} - \gamma] \approx \frac{K}{n-K} \cdot (\hat{\gamma}_{OLS} - \gamma)
$$

As K/n increases, bias worsens.

### 5.2 Solutions

| Method | Approach | When to Use |
|--------|----------|-------------|
| **LIML** | Likelihood-based | Moderate many-IV |
| **JIVE** | Leave-one-out | Substantial many-IV |
| **Fuller** | Bias-corrected LIML | Moderate to substantial |
| **HFUL** | Heteroskedasticity-robust Fuller | Heteroskedasticity + many IV |
| **Regularization** | LASSO/Ridge selection | Very many potential instruments |

### 5.3 Regularized IV (LASSO-IV)

When you have many potential instruments, use LASSO to select relevant ones:

**Step 1**: First-stage LASSO
$$
\hat{\pi} = \arg\min_\pi \|D - Z\pi\|^2 + \lambda\|\pi\|_1
$$

**Step 2**: Use selected instruments for 2SLS

```python
from sklearn.linear_model import LassoCV

def lasso_iv(data, outcome, treatment, instruments, controls=None, cv=5):
    """
    IV estimation with LASSO instrument selection.
    """
    df = data.copy()

    # First stage: LASSO selection
    Z = df[instruments].values
    D = df[treatment].values

    lasso = LassoCV(cv=cv)
    lasso.fit(Z, D)

    # Selected instruments (non-zero coefficients)
    selected = [instruments[i] for i in range(len(instruments))
                if abs(lasso.coef_[i]) > 1e-6]

    if len(selected) == 0:
        raise ValueError("LASSO selected no instruments. Consider relaxing penalty.")

    print(f"LASSO selected {len(selected)} of {len(instruments)} instruments")

    # Second stage: Standard 2SLS with selected instruments
    from iv_estimator import estimate_2sls
    return estimate_2sls(data, outcome, treatment, selected, controls)
```

### 5.4 Practical Guidelines

1. **Check K/n ratio**: If K > 0.1n, many-instruments bias is a concern
2. **Compare estimators**: Large differences between 2SLS and LIML/JIVE indicate problems
3. **Use few strong instruments**: Prefer quality over quantity
4. **Regularization**: Only if you have conceptual justification for many instruments

---

## 6. Comparison of Estimators

### 6.1 Bias-Variance Tradeoff

| Estimator | Bias (weak IV) | Variance | Recommended When |
|-----------|----------------|----------|------------------|
| 2SLS | High | Lowest | Strong instruments |
| LIML | Low | Higher | Moderate weak IV |
| Fuller(1) | Very low | Moderate | Weak IV, need point estimate |
| Fuller(4) | Low | Moderate | Weak IV, need CIs |
| GMM | High | Efficient (het.) | Strong IV + heteroskedasticity |
| JIVE | Low | Higher | Many instruments |

### 6.2 Decision Flowchart

```
START
  |
  v
First-stage F > 10?
  |
  +-- YES --> Homoskedastic?
  |             |
  |             +-- YES --> Use 2SLS
  |             |
  |             +-- NO --> Use GMM
  |
  +-- NO --> F > 5?
              |
              +-- YES --> Use LIML or Fuller
              |
              +-- NO --> Consider:
                          - Find better instruments
                          - Anderson-Rubin CI
                          - Bounds analysis
```

### 6.3 Robustness Check: Compare All Estimators

```python
from iv_estimator import estimate_2sls, estimate_liml, estimate_gmm

def compare_iv_estimators(data, outcome, treatment, instruments, controls):
    """
    Compare multiple IV estimators for robustness.
    """
    results = {}

    # 2SLS
    r_2sls = estimate_2sls(data, outcome, treatment, instruments, controls)
    results['2SLS'] = {'estimate': r_2sls.effect, 'se': r_2sls.se}

    # LIML
    r_liml = estimate_liml(data, outcome, treatment, instruments, controls)
    results['LIML'] = {'estimate': r_liml.effect, 'se': r_liml.se}

    # GMM
    r_gmm = estimate_gmm(data, outcome, treatment, instruments, controls)
    results['GMM'] = {'estimate': r_gmm.effect, 'se': r_gmm.se}

    # Summary
    print("Estimator Comparison:")
    print("-" * 50)
    for method, vals in results.items():
        print(f"{method:10s}: {vals['estimate']:.4f} (SE: {vals['se']:.4f})")

    # Check for large discrepancies (potential weak IV)
    estimates = [v['estimate'] for v in results.values()]
    if max(estimates) - min(estimates) > 0.5 * np.mean([v['se'] for v in results.values()]):
        print("\nWARNING: Large discrepancy between estimators.")
        print("This may indicate weak instruments.")

    return results
```

---

## 7. Software Implementation Notes

### 7.1 Python (linearmodels)

```python
from linearmodels.iv import IV2SLS, IVLIML, IVGMM

# 2SLS
model = IV2SLS(dependent=df['y'],
               exog=df[['const'] + controls],
               endog=df[[treatment]],
               instruments=df[instruments])
result = model.fit(cov_type='robust')

# LIML
model = IVLIML(...)
result = model.fit(cov_type='robust')

# GMM
model = IVGMM(...)
result = model.fit(cov_type='robust')
```

### 7.2 R (ivreg)

```r
library(ivreg)

# 2SLS
result <- ivreg(y ~ d + x1 + x2 | z1 + z2 + x1 + x2, data = df)
summary(result, diagnostics = TRUE)

# With robust SEs
library(sandwich)
coeftest(result, vcov = vcovHC(result, type = "HC1"))
```

### 7.3 Stata

```stata
* 2SLS
ivregress 2sls y x1 x2 (d = z1 z2), robust

* LIML
ivregress liml y x1 x2 (d = z1 z2), robust

* GMM
ivregress gmm y x1 x2 (d = z1 z2), robust
```

---

## References

### Core Methods
- Angrist, J. D., & Pischke, J. S. (2009). *Mostly Harmless Econometrics*, Chapter 4.
- Wooldridge, J. M. (2010). *Econometric Analysis of Cross Section and Panel Data*, Chapter 5.

### LIML and Fuller
- Anderson, T. W., & Rubin, H. (1949). Estimation of the Parameters of a Single Equation in a Complete System of Stochastic Equations.
- Fuller, W. A. (1977). Some Properties of a Modification of the Limited Information Estimator.

### GMM
- Hansen, L. P. (1982). Large Sample Properties of Generalized Method of Moments Estimators.

### JIVE and Many Instruments
- Angrist, J. D., Imbens, G. W., & Krueger, A. B. (1999). Jackknife Instrumental Variables Estimation.
- Ackerberg, D. A., & Devereux, P. J. (2009). Improved JIVE Estimators.
- Bekker, P. A. (1994). Alternative Approximations to the Distributions of Instrumental Variable Estimators.
