---
name: estimator-iv
description: Instrumental Variables (IV) and Two-Stage Least Squares (2SLS) for causal inference with endogeneity. Use when OLS is biased due to omitted variables, simultaneity, or measurement error. Provides 2SLS, LIML, GMM estimators with weak instrument diagnostics via linearmodels and statsmodels.
license: MIT
metadata:
    skill-author: Causal-ML-Skills
---

# Instrumental Variables (IV): Causal Inference with Endogeneity

## Overview

Instrumental Variables (IV) is a powerful method for estimating causal effects when standard regression (OLS) fails due to endogeneity - correlation between the regressor and the error term. This skill provides comprehensive guidance for implementing rigorous IV analyses in Python, from basic Two-Stage Least Squares (2SLS) to advanced GMM estimators with proper weak instrument diagnostics.

Apply this skill when you suspect your key explanatory variable is endogenous (correlated with unobserved factors affecting the outcome), and you have access to a valid instrument - a variable that affects the outcome only through its effect on the endogenous variable.

## When to Use This Skill

This skill should be used when:

- Estimating causal effects with potential endogeneity (omitted variable bias)
- Dealing with simultaneity or reverse causality
- Correcting for measurement error in explanatory variables
- Implementing Two-Stage Least Squares (2SLS) regression
- Testing instrument validity (relevance and exogeneity)
- Diagnosing weak instruments (Stock-Yogo critical values)
- Running overidentification tests (Sargan-Hansen)
- Comparing OLS and IV estimates (Hausman test)
- Estimating Local Average Treatment Effects (LATE)

## Quick Start Guide

### Basic 2SLS with linearmodels

```python
import pandas as pd
import numpy as np
from linearmodels.iv import IV2SLS

# Example: Effect of education on wages, using distance to college as instrument
# Endogeneity: ability affects both education and wages
# Instrument: distance to nearest college (affects education, not wages directly)

# Load your data
df = pd.read_csv('data.csv')

# Add constant
df['const'] = 1

# 2SLS Estimation
# Model: wage = beta_0 + beta_1 * education + epsilon
# First stage: education = gamma_0 + gamma_1 * distance + nu

model = IV2SLS(
    dependent=df['wage'],           # Y: outcome
    exog=df[['const']],             # Exogenous controls (including constant)
    endog=df[['education']],        # X: endogenous variable
    instruments=df[['distance']]    # Z: instrument(s)
)

# Fit with robust standard errors
result = model.fit(cov_type='robust')

# View results
print(result.summary)

# Key diagnostics
print(f"\n{'='*50}")
print("KEY DIAGNOSTICS")
print(f"{'='*50}")
print(f"First-stage F-statistic: {result.first_stage.diagnostics['f.stat'].stat:.2f}")
print(f"First-stage F p-value: {result.first_stage.diagnostics['f.stat'].pval:.4f}")

# Weak instrument check (Stock-Yogo: F > 10 for 10% maximal IV size)
if result.first_stage.diagnostics['f.stat'].stat > 10:
    print("PASS: Instrument appears strong (F > 10)")
else:
    print("WARNING: Potential weak instrument (F < 10)")
```

### Complete IV Analysis with Diagnostics

```python
import pandas as pd
import numpy as np
from linearmodels.iv import IV2SLS, IVLIML, IVGMM
import statsmodels.api as sm
from scipy import stats

# Load data
df = pd.read_csv('data.csv')
df['const'] = 1

# Define variables
outcome = 'log_wage'
endogenous = 'years_education'
instruments = ['distance_college', 'tuition']  # Multiple instruments
controls = ['experience', 'experience_sq', 'female']

# ============================================
# 1. FIRST STAGE REGRESSION
# ============================================
print("="*60)
print("FIRST STAGE: Instrument Relevance")
print("="*60)

# First stage: education ~ instruments + controls
X_first = df[instruments + controls + ['const']]
y_first = df[endogenous]

first_stage = sm.OLS(y_first, X_first).fit(cov_type='HC1')
print(first_stage.summary())

# F-test for excluded instruments
r_matrix = np.zeros((len(instruments), len(X_first.columns)))
for i, inst in enumerate(instruments):
    r_matrix[i, X_first.columns.get_loc(inst)] = 1

f_test = first_stage.f_test(r_matrix)
print(f"\nF-statistic for excluded instruments: {f_test.fvalue[0][0]:.2f}")
print(f"P-value: {f_test.pvalue:.4f}")

# Stock-Yogo critical values (for 2 instruments, 1 endogenous)
# 10% maximal IV size: 19.93, 15% size: 11.59, 20% size: 8.75
print("\nStock-Yogo Critical Values (10% maximal IV size):")
print("  2 instruments: 19.93")
print("  3 instruments: 22.30")

# ============================================
# 2. TWO-STAGE LEAST SQUARES (2SLS)
# ============================================
print("\n" + "="*60)
print("SECOND STAGE: 2SLS Estimation")
print("="*60)

model_2sls = IV2SLS(
    dependent=df[outcome],
    exog=df[controls + ['const']],
    endog=df[[endogenous]],
    instruments=df[instruments]
)
result_2sls = model_2sls.fit(cov_type='robust')
print(result_2sls.summary)

# ============================================
# 3. OLS FOR COMPARISON
# ============================================
print("\n" + "="*60)
print("OLS COMPARISON")
print("="*60)

X_ols = df[[endogenous] + controls + ['const']]
model_ols = sm.OLS(df[outcome], X_ols).fit(cov_type='HC1')
print(f"OLS coefficient on {endogenous}: {model_ols.params[endogenous]:.4f}")
print(f"2SLS coefficient on {endogenous}: {result_2sls.params[endogenous]:.4f}")
print(f"Difference: {result_2sls.params[endogenous] - model_ols.params[endogenous]:.4f}")

# ============================================
# 4. HAUSMAN TEST (Endogeneity)
# ============================================
print("\n" + "="*60)
print("HAUSMAN TEST: Is IV needed?")
print("="*60)

if result_2sls.wu_hausman is not None:
    print(f"Wu-Hausman statistic: {result_2sls.wu_hausman.stat:.4f}")
    print(f"P-value: {result_2sls.wu_hausman.pval:.4f}")
    if result_2sls.wu_hausman.pval < 0.05:
        print("REJECT H0: Endogeneity present, IV is needed")
    else:
        print("FAIL TO REJECT H0: No evidence of endogeneity, OLS may be sufficient")

# ============================================
# 5. SARGAN-HANSEN TEST (Overidentification)
# ============================================
print("\n" + "="*60)
print("SARGAN-HANSEN TEST: Instrument Validity")
print("="*60)

if len(instruments) > 1:  # Only valid if overidentified
    if result_2sls.sargan is not None:
        print(f"Sargan-Hansen J-statistic: {result_2sls.sargan.stat:.4f}")
        print(f"P-value: {result_2sls.sargan.pval:.4f}")
        if result_2sls.sargan.pval < 0.05:
            print("REJECT H0: At least one instrument may be invalid")
        else:
            print("FAIL TO REJECT H0: Instruments appear valid")
else:
    print("Model is just-identified. Cannot test overidentification.")
```

### LIML Estimator (Weak Instrument Robust)

```python
from linearmodels.iv import IVLIML

# LIML is more robust to weak instruments than 2SLS
model_liml = IVLIML(
    dependent=df[outcome],
    exog=df[controls + ['const']],
    endog=df[[endogenous]],
    instruments=df[instruments]
)
result_liml = model_liml.fit(cov_type='robust')

print("LIML Estimation Results:")
print(result_liml.summary)

# Compare 2SLS and LIML
print(f"\n2SLS coefficient: {result_2sls.params[endogenous]:.4f}")
print(f"LIML coefficient: {result_liml.params[endogenous]:.4f}")
print("Note: Large difference suggests weak instrument bias in 2SLS")
```

### Anderson-Rubin Confidence Interval (Weak IV Robust)

```python
from linearmodels.iv import IV2SLS
import numpy as np
from scipy import stats

def anderson_rubin_ci(df, outcome, endogenous, instruments, controls, alpha=0.05):
    """
    Compute Anderson-Rubin confidence interval.
    Valid even with weak instruments.
    """
    # Grid search over beta values
    beta_grid = np.linspace(-2, 2, 1000)
    ar_stats = []

    for beta in beta_grid:
        # Construct Y - beta*X
        y_adjusted = df[outcome] - beta * df[endogenous]

        # Regress on instruments and controls
        X = df[instruments + controls + ['const']]
        model = sm.OLS(y_adjusted, X).fit()

        # F-test for instruments
        r_matrix = np.zeros((len(instruments), len(X.columns)))
        for i, inst in enumerate(instruments):
            r_matrix[i, X.columns.get_loc(inst)] = 1

        f_test = model.f_test(r_matrix)
        ar_stats.append(f_test.fvalue[0][0])

    # Critical value
    df1 = len(instruments)
    df2 = len(df) - len(instruments) - len(controls) - 1
    critical_value = stats.f.ppf(1 - alpha, df1, df2)

    # Find confidence set
    in_ci = np.array(ar_stats) < critical_value
    ci_lower = beta_grid[in_ci].min() if in_ci.any() else np.nan
    ci_upper = beta_grid[in_ci].max() if in_ci.any() else np.nan

    return ci_lower, ci_upper

# Usage
ar_ci = anderson_rubin_ci(df, outcome, endogenous, instruments, controls)
print(f"Anderson-Rubin 95% CI: [{ar_ci[0]:.4f}, {ar_ci[1]:.4f}]")
print("Note: This CI is valid even with weak instruments")
```

### Fuzzy Regression Discontinuity as IV

```python
from linearmodels.iv import IV2SLS

# Example: Effect of scholarship on graduation
# Running variable: test score
# Cutoff: score >= 70 gets scholarship
# But compliance is imperfect (fuzzy RD)

# Create instruments from RD design
df['above_cutoff'] = (df['test_score'] >= 70).astype(int)

# First stage: scholarship ~ above_cutoff (+ controls)
# Second stage: graduation ~ scholarship (instrumented by above_cutoff)

# Local sample around cutoff
bandwidth = 10
df_local = df[(df['test_score'] >= 70 - bandwidth) &
              (df['test_score'] <= 70 + bandwidth)]
df_local['const'] = 1

model_frd = IV2SLS(
    dependent=df_local['graduation'],
    exog=df_local[['const', 'test_score']],  # Control for running variable
    endog=df_local[['scholarship']],
    instruments=df_local[['above_cutoff']]
)
result_frd = model_frd.fit(cov_type='robust')

print("Fuzzy RD as IV:")
print(result_frd.summary)
print(f"\nLATE estimate: {result_frd.params['scholarship']:.4f}")
```

## Core Capabilities

### 1. IV Estimators

**Two-Stage Least Squares (2SLS)**:
- Most common IV estimator
- Consistent but can be biased with weak instruments
- Use `linearmodels.iv.IV2SLS`

**Limited Information Maximum Likelihood (LIML)**:
- More robust to weak instruments than 2SLS
- Approximately median-unbiased
- Use `linearmodels.iv.IVLIML`

**Generalized Method of Moments (GMM)**:
- Efficient with heteroskedasticity
- Optimal weighting of moment conditions
- Use `linearmodels.iv.IVGMM`

**When to use each:**

| Estimator | Best For | Weakness |
|-----------|----------|----------|
| **2SLS** | Strong instruments, homoskedasticity | Biased with weak IV |
| **LIML** | Potentially weak instruments | Less efficient than 2SLS |
| **GMM** | Heteroskedasticity, many instruments | Can be biased in small samples |

**Reference:** See `references/estimation_methods.md` for detailed comparison.

### 2. Instrument Diagnostics

**First-Stage F-statistic:**
- Tests instrument relevance
- Rule of thumb: F > 10 (Stock & Yogo, 2005)
- Stricter: F > 104.7 for <5% bias (Lee et al., 2022)

**Stock-Yogo Critical Values:**
- Account for number of instruments and endogenous variables
- Test for maximal IV size (bias relative to OLS)
- Available in `linearmodels` diagnostics

**Weak Instrument Robust Inference:**
- Anderson-Rubin test and confidence intervals
- Conditional likelihood ratio (CLR) test
- Valid regardless of instrument strength

**Reference:** See `references/diagnostic_tests.md` for critical value tables.

### 3. Validity Tests

**Sargan-Hansen J-test (Overidentification):**
- Tests if instruments are uncorrelated with error
- Only valid when overidentified (more instruments than endogenous)
- H0: All instruments are valid

**Wu-Hausman Test (Endogeneity):**
- Compares OLS and IV estimates
- H0: OLS is consistent (no endogeneity)
- Significant result → need IV

**Reference:** See `references/identification_assumptions.md` for interpretation.

### 4. Standard Errors

**Robust (Heteroskedasticity-consistent):**
```python
result = model.fit(cov_type='robust')
```

**Clustered:**
```python
result = model.fit(cov_type='clustered', clusters=df['cluster_var'])
```

**HAC (Heteroskedasticity and Autocorrelation Consistent):**
```python
result = model.fit(cov_type='kernel')
```

## Common Workflows

### Workflow 1: Standard IV Analysis

```
1. Theoretical Justification
   ├── Why is X endogenous? (omitted variables, simultaneity, measurement error)
   ├── Why is Z a valid instrument? (relevance + exclusion)
   └── What is the estimand? (LATE for binary instrument)

2. First Stage Analysis
   ├── Regress X on Z (and controls)
   ├── Check F-statistic (F > 10 or use Stock-Yogo)
   ├── Interpret coefficient sign and magnitude
   └── If weak, consider LIML or AR inference

3. Second Stage (2SLS)
   ├── Run IV2SLS
   ├── Report coefficient, SE, CI
   └── Compare with OLS

4. Diagnostic Tests
   ├── Wu-Hausman test (is IV needed?)
   ├── Sargan-Hansen test (if overidentified)
   └── Sensitivity analysis

5. Reporting
   ├── First-stage results with F-stat
   ├── Second-stage results
   ├── OLS comparison
   └── All diagnostic tests
```

### Workflow 2: Weak Instrument Robust Analysis

```
1. Standard First Stage
   └── If F < 10, proceed with caution

2. Multiple Estimators
   ├── 2SLS (baseline)
   ├── LIML (weak IV robust)
   └── Compare estimates

3. Weak IV Robust Inference
   ├── Anderson-Rubin confidence interval
   ├── CLR test
   └── Report both standard and robust CIs

4. Sensitivity Analysis
   ├── Vary instrument set
   ├── Add/remove controls
   └── Check stability
```

## Best Practices

### Instrument Selection

1. **Theoretical foundation**: Explain WHY instrument satisfies exclusion restriction
2. **Avoid forbidden regressions**: Don't regress Y on Z to "check" exclusion
3. **Multiple instruments**: Allows overidentification test, but beware many weak IVs
4. **Report reduced form**: Y on Z shows intent-to-treat effect

### Estimation

1. **Always check first stage**: F-statistic is minimum requirement
2. **Use robust SEs**: Heteroskedasticity is common
3. **Compare 2SLS and LIML**: Large difference indicates weak IV problem
4. **Report OLS alongside IV**: Shows direction and magnitude of bias

### Inference

1. **Weak IV robust CIs**: Anderson-Rubin or CLR when F < 10
2. **Don't ignore Sargan test failures**: Indicates invalid instruments
3. **Interpret LATE correctly**: IV identifies effect for compliers only
4. **Be cautious with many instruments**: Can cause GMM/2SLS bias

### Reporting

1. **First-stage regression table**: With F-statistic prominently displayed
2. **Stock-Yogo critical values**: Reference for instrument strength
3. **All diagnostic tests**: Hausman, Sargan-Hansen
4. **Reduced form**: Shows total effect of Z on Y

## Reference Documentation

This skill includes comprehensive reference files:

### references/identification_assumptions.md
- Exclusion restriction (formal definition)
- Relevance condition
- Monotonicity (for LATE interpretation)
- Independence assumption
- SUTVA
- When assumptions fail

### references/estimation_methods.md
- 2SLS derivation and properties
- LIML and k-class estimators
- GMM and efficient IV
- Jackknife IV (JIVE)
- Split-sample IV
- Comparison and selection guide

### references/diagnostic_tests.md
- First-stage F-statistic
- Stock-Yogo critical values table
- Anderson-Rubin test
- Conditional likelihood ratio (CLR)
- Sargan-Hansen J-test
- Wu-Hausman test
- Cragg-Donald statistic

### references/reporting_standards.md
- AER/QJE table formats
- Required diagnostics
- LaTeX templates
- First-stage reporting

### references/common_errors.md
- Using weak instruments
- Forbidden regression
- Manual 2SLS (wrong SEs)
- Ignoring LATE interpretation
- Too many instruments

## Common Pitfalls to Avoid

1. **Using weak instruments without adjustment**: If F < 10, use LIML or AR inference
2. **Manual two-step OLS**: SEs will be wrong; use `IV2SLS` or correct the SEs
3. **Ignoring first stage**: Always report F-statistic
4. **Forbidden regression**: Don't regress Y on Z to "test" exclusion
5. **Misinterpreting LATE**: IV estimates effect for compliers, not ATE
6. **Too many instruments**: Causes finite-sample bias (rule: instruments < sqrt(n))
7. **Ignoring Sargan test failure**: Indicates at least one instrument is invalid
8. **Not reporting OLS comparison**: Readers need to see the bias direction
9. **Claiming exclusion is "tested"**: Exclusion restriction is untestable
10. **Using irrelevant controls**: Only include controls that affect Y or instrument validity
11. **Clustering at wrong level**: Cluster at level of instrument variation
12. **Ignoring heterogeneous effects**: LATE may not equal ATE
13. **Using generated instruments**: Creates additional uncertainty (Murphy-Topel SEs needed)
14. **Weak IV with multiple endogenous**: Stock-Yogo doesn't apply directly
15. **Not checking reduced form**: Should show significant Z→Y relationship

## Troubleshooting

### First-Stage F-statistic is Low (< 10)

**Issue:** Weak instrument problem

**Solutions:**
```python
# 1. Use LIML instead of 2SLS
result_liml = IVLIML(...).fit()

# 2. Compute Anderson-Rubin CI
ar_ci = anderson_rubin_ci(...)

# 3. Find stronger instruments
# 4. Consider alternative identification strategies
```

### Sargan Test Rejects

**Issue:** At least one instrument may be invalid

**Solutions:**
- Re-examine exclusion restriction theoretically
- Try different instrument subsets
- Consider if heterogeneous effects cause rejection

### 2SLS and LIML Give Very Different Results

**Issue:** Indicates weak instrument bias

**Solution:** Trust LIML more; report both with caveat about weak instruments

### linearmodels Import Error

**Issue:** Package not installed

**Solution:**
```bash
pip install linearmodels
```

### Singular Matrix Error

**Issue:** Perfect collinearity

**Solution:**
```python
# Check for collinear variables
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Remove redundant instruments or controls
```

## Additional Resources

### Official Documentation
- linearmodels: https://bashtage.github.io/linearmodels/
- statsmodels IV: https://www.statsmodels.org/stable/generated/statsmodels.sandbox.regression.gmm.IV2SLS.html

### Key Papers
- Angrist, Imbens & Rubin (1996): "Identification of Causal Effects Using Instrumental Variables"
- Stock & Yogo (2005): "Testing for Weak Instruments in Linear IV Regression"
- Andrews, Stock & Sun (2019): "Weak Instruments in IV Regression"
- Lee et al. (2022): "Valid t-ratio Inference for IV"

### Textbooks
- Angrist & Pischke (2009): *Mostly Harmless Econometrics*, Ch. 4
- Wooldridge (2010): *Econometric Analysis of Cross Section and Panel Data*, Ch. 5
- Cameron & Trivedi (2005): *Microeconometrics*, Ch. 4

## Installation

```bash
# Core packages
pip install linearmodels statsmodels pandas numpy scipy

# For visualization
pip install matplotlib seaborn

# Full installation
pip install linearmodels statsmodels pandas numpy scipy matplotlib seaborn
```

## Related Skills

| Skill | When to Use Instead |
|-------|---------------------|
| `estimator-did` | Panel data with treatment timing |
| `estimator-rd` | Sharp cutoff in assignment (reduced form) |
| `estimator-psm` | Selection on observables only |
| `causal-ddml` | High-dimensional controls with IV |
| `panel-data-models` | Panel data without endogeneity |
