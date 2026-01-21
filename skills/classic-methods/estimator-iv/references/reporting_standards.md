# IV Reporting Standards

> **Reference Document** | IV Estimator Skill
> **Last Updated**: 2024

## Overview

This document establishes reporting standards for Instrumental Variables analysis following best practices from top economics journals and the Angrist-Pischke methodology. Proper reporting is essential for reproducibility and allows readers to assess the validity of IV estimates.

---

## 1. Required Elements

Every IV analysis should report:

### 1.1 First-Stage Results

**Must include**:
- First-stage regression coefficients for all instruments
- Standard errors (preferably robust)
- First-stage F-statistic for excluded instruments
- Partial R-squared

**Table Format**:
```
Table X: First-Stage Results
Dependent Variable: Treatment (D)
-----------------------------------------------------------
                        (1)           (2)
                    First Stage   First Stage
                                  with Controls
-----------------------------------------------------------
Instrument Z1         0.452***      0.438***
                     (0.089)       (0.091)

Instrument Z2         0.287***      0.271***
                     (0.076)       (0.078)

Controls                No           Yes
Observations          1,000         1,000
R-squared            0.234         0.312
Partial R-squared    0.156         0.148
F-statistic          45.67         42.31
-----------------------------------------------------------
Notes: Robust standard errors in parentheses.
*** p<0.01, ** p<0.05, * p<0.1
F-statistic is for excluded instruments.
```

### 1.2 Reduced Form Results

The reduced form regresses the outcome directly on instruments (important for interpretation):

$$
Y_i = \theta_0 + \theta_1 Z_i + X_i'\lambda + u_i
$$

**Why Report**:
- Shows the "intent-to-treat" effect
- Numerator of the Wald estimator
- Useful when first stage is weak
- More robust to specification choices

```
Table X: Reduced Form Results
Dependent Variable: Outcome (Y)
-----------------------------------------------------------
                        (1)           (2)
-----------------------------------------------------------
Instrument Z1         0.475***      0.461***
                     (0.124)       (0.127)

Instrument Z2         0.301**       0.285**
                     (0.103)       (0.106)

Controls                No           Yes
-----------------------------------------------------------
```

### 1.3 Second-Stage (Structural) Results

**Must include**:
- IV coefficient estimate
- Standard errors (robust)
- 95% confidence interval
- OLS comparison (to show difference)

```
Table X: IV and OLS Estimates
Dependent Variable: Outcome (Y)
-----------------------------------------------------------
                        (1)           (2)           (3)
                       OLS          2SLS          LIML
-----------------------------------------------------------
Treatment (D)         0.453***      1.052***      1.048***
                     (0.078)       (0.231)       (0.235)

Controls               Yes           Yes           Yes
Observations          1,000         1,000         1,000
R-squared            0.345           -             -

First-stage F           -          45.67         45.67
Sargan p-value          -          0.456         0.456
Wu-Hausman p-value      -          0.003           -
-----------------------------------------------------------
Notes: Robust standard errors in parentheses.
*** p<0.01, ** p<0.05, * p<0.1
First-stage F is Kleibergen-Paap rk Wald F statistic.
```

---

## 2. Diagnostic Statistics

### 2.1 Always Report

| Statistic | Purpose | Threshold |
|-----------|---------|-----------|
| First-stage F | Weak IV assessment | F > 10 |
| Stock-Yogo critical value | Formal weak IV test | Compare to F |
| Wu-Hausman test | Endogeneity confirmation | p < 0.05 |

### 2.2 Report if Overidentified

| Statistic | Purpose | Interpretation |
|-----------|---------|----------------|
| Sargan/Hansen J | Instrument validity | p > 0.05 desirable |
| Degrees of freedom | Context for J-test | K - 1 |

### 2.3 Report for Robustness

| Statistic | Purpose |
|-----------|---------|
| 2SLS vs LIML difference | Weak IV sensitivity |
| Anderson-Rubin CI | Weak-IV robust inference |
| Subset IV estimates | Instrument heterogeneity |

---

## 3. Anderson-Rubin Confidence Intervals

When instruments may be weak, the Anderson-Rubin (AR) confidence set is valid regardless of instrument strength.

### 3.1 The AR Test

The AR statistic tests $H_0: \gamma = \gamma_0$:
$$
AR(\gamma_0) = \frac{(Y - D\gamma_0)'P_Z(Y - D\gamma_0) / K}{(Y - D\gamma_0)'M_Z(Y - D\gamma_0) / (n-K)} \sim F(K, n-K)
$$

### 3.2 AR Confidence Set

The $(1-\alpha)$ AR confidence set is:
$$
CS_{AR} = \{\gamma_0 : AR(\gamma_0) \leq F_{K, n-K, 1-\alpha}\}
$$

**Properties**:
- Valid with any instrument strength
- May be unbounded if instruments very weak
- More conservative than standard Wald CI

### 3.3 Implementation

```python
def anderson_rubin_ci(data, outcome, treatment, instruments, controls=None,
                      alpha=0.05, grid_points=1000):
    """
    Compute Anderson-Rubin confidence interval.

    Parameters
    ----------
    data : pd.DataFrame
        Dataset
    outcome : str
        Outcome variable
    treatment : str
        Treatment variable
    instruments : List[str]
        Instrument names
    controls : List[str], optional
        Control variable names
    alpha : float
        Significance level (default 0.05 for 95% CI)
    grid_points : int
        Number of grid points to evaluate

    Returns
    -------
    Dict
        AR confidence interval bounds
    """
    import statsmodels.api as sm
    from scipy import stats

    df = data.copy()
    y = df[outcome].values
    d = df[treatment].values

    # Build instrument matrix
    Z_vars = instruments.copy()
    if controls:
        Z_vars.extend(controls)
    Z = sm.add_constant(df[Z_vars]).values

    n, K = Z.shape

    # Projection matrices
    P_Z = Z @ np.linalg.inv(Z.T @ Z) @ Z.T
    M_Z = np.eye(n) - P_Z

    # Critical value
    f_crit = stats.f.ppf(1 - alpha, K, n - K)

    # Grid search for CI
    # Start with 2SLS estimate as center
    from iv_estimator import estimate_2sls
    result = estimate_2sls(data, outcome, treatment, instruments, controls)
    center = result.effect
    width = 10 * result.se

    gamma_grid = np.linspace(center - width, center + width, grid_points)
    in_ci = []

    for gamma_0 in gamma_grid:
        residual = y - d * gamma_0

        numerator = (residual @ P_Z @ residual) / K
        denominator = (residual @ M_Z @ residual) / (n - K)

        if denominator > 0:
            ar_stat = numerator / denominator
            if ar_stat <= f_crit:
                in_ci.append(gamma_0)

    if len(in_ci) == 0:
        return {
            'ci_lower': np.nan,
            'ci_upper': np.nan,
            'bounded': False,
            'note': 'Empty confidence set - instruments may be invalid'
        }

    # Check if CI extends to boundaries (unbounded)
    ci_lower = min(in_ci)
    ci_upper = max(in_ci)
    bounded = (ci_lower > gamma_grid[0] + 0.01 * width and
               ci_upper < gamma_grid[-1] - 0.01 * width)

    return {
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'bounded': bounded,
        'note': 'Bounded interval' if bounded else 'May be unbounded - expand grid'
    }
```

### 3.4 Reporting AR Intervals

```
Table X: Weak-IV Robust Inference
-----------------------------------------------------------
                     Point       Standard    Anderson-Rubin
                    Estimate       CI          95% CI
-----------------------------------------------------------
Treatment Effect     1.052     [0.60, 1.50]  [0.52, 1.72]
-----------------------------------------------------------
Notes: Standard CI based on asymptotic normal approximation.
Anderson-Rubin CI is robust to weak instruments.
First-stage F = 8.5 (below Stock-Yogo threshold).
```

---

## 4. Complete Reporting Template

### 4.1 LaTeX Table Template

See `assets/latex/iv_table.tex` for the complete template.

### 4.2 Markdown Report Template

See `assets/markdown/iv_report.md` for the complete template.

### 4.3 Minimum Reporting Checklist

**For the text**:
- [ ] State the research question and why IV is needed
- [ ] Describe instruments and argue for validity
- [ ] Report first-stage F and interpret strength
- [ ] Report main estimate with SE and CI
- [ ] Discuss LATE interpretation
- [ ] Address potential threats to validity

**For tables**:
- [ ] First-stage regression table
- [ ] Main results table with OLS comparison
- [ ] Diagnostic statistics (F, Sargan, Hausman)
- [ ] Robustness checks (LIML, AR CI if weak)

---

## 5. Common Mistakes to Avoid

### 5.1 Reporting Issues

| Mistake | Problem | Solution |
|---------|---------|----------|
| Not reporting first-stage F | Cannot assess weak IV | Always report |
| Wrong SE calculation | Using fitted D residuals | Use actual D residuals |
| Missing OLS comparison | Cannot see IV correction | Report side-by-side |
| No LATE discussion | Misleading interpretation | Discuss complier population |

### 5.2 Interpretation Issues

**Wrong**:
> "IV shows that X causes Y to increase by 1.05 for everyone."

**Correct**:
> "IV estimates a Local Average Treatment Effect (LATE) of 1.05 for compliers - those whose treatment status was affected by the instrument. This may differ from the average effect in the population if treatment effects are heterogeneous."

### 5.3 Validity Discussion

**Wrong**:
> "The Sargan test passes so the instruments are valid."

**Correct**:
> "The Sargan test cannot reject the overidentifying restrictions (p = 0.45), which is consistent with instrument validity. However, this test cannot validate the exclusion restriction if all instruments are invalid in the same way. We argue for the exclusion restriction based on [institutional argument]."

---

## 6. Journal-Specific Requirements

### 6.1 AER/QJE/Econometrica

- First-stage F required
- Robustness with LIML
- Extended validity discussion
- Falsification tests

### 6.2 NBER Working Papers

- Stock-Yogo critical values
- Anderson-Rubin CI if F < 10
- Multiple instrument comparison

### 6.3 Applied Journals

- Clear first-stage and reduced form
- OLS comparison
- Intuitive LATE discussion

---

## 7. Example Write-up

> We estimate the effect of education on wages using proximity to college as an instrument (Card, 1995). Table 2 presents the first-stage results, showing that college proximity significantly predicts educational attainment (F = 45.67, well above the Stock-Yogo 10% maximal bias critical value of 16.38). The 2SLS estimate indicates that an additional year of education increases log wages by 0.105 (SE = 0.023), substantially larger than the OLS estimate of 0.045.
>
> The difference between OLS and IV suggests either downward ability bias in OLS or that the LATE for compliers exceeds the ATE. The instrument affects those on the margin of college attendance, who may have higher returns to education than the average person.
>
> Table 3 reports diagnostic tests. The Wu-Hausman test rejects exogeneity (p = 0.003), confirming the need for IV. The Sargan test cannot reject instrument validity (p = 0.456). As a robustness check, LIML yields nearly identical estimates (0.103), suggesting weak instruments are not a concern.

---

## References

- Angrist, J. D., & Pischke, J. S. (2009). *Mostly Harmless Econometrics*. Princeton University Press.
- Stock, J. H., Wright, J. H., & Yogo, M. (2002). A Survey of Weak Instruments and Weak Identification in Generalized Method of Moments. *JBES*, 20(4), 518-529.
- Andrews, I., Stock, J. H., & Sun, L. (2019). Weak Instruments in IV Regression: Theory and Practice. *Annual Review of Economics*, 11, 727-753.
