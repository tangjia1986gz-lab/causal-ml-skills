# IV Diagnostic Tests

> **Reference Document** | IV Estimator Skill
> **Last Updated**: 2024

## Overview

This document covers the essential diagnostic tests for Instrumental Variables estimation. These tests help assess instrument validity, detect weak instruments, and verify the need for IV over OLS.

---

## 1. Weak Instrument Tests

### 1.1 First-Stage F-Test

The most common weak instrument diagnostic is the F-statistic from the first-stage regression.

**Test Setup**:
$$
D_i = \pi_0 + \pi_1 Z_i + X_i'\gamma + v_i
$$

Test: $H_0: \pi_1 = 0$ (instruments have no predictive power)

**Implementation**:
```python
from iv_estimator import first_stage_test

first_stage = first_stage_test(
    data=df,
    treatment="d",
    instruments=["z1", "z2"],
    controls=["x1", "x2"]
)

print(f"First-stage F: {first_stage['f_statistic']:.2f}")
print(f"P-value: {first_stage['f_pvalue']:.4f}")
print(f"Partial R-squared: {first_stage['partial_r2']:.4f}")
```

**Interpretation**:
- F > 10: Strong instruments (Staiger & Stock rule of thumb)
- F between 5-10: Weak instruments, consider LIML
- F < 5: Very weak instruments, IV unreliable

### 1.2 Stock-Yogo Critical Values

More rigorous than the F > 10 rule, Stock-Yogo (2005) provides critical values based on acceptable bias or size distortion.

**Critical Values for Maximal IV Relative Bias**:

| # Instruments | 5% | 10% | 20% | 30% |
|---------------|-----|------|------|------|
| 1 | 16.38 | 16.38 | 6.66 | 5.53 |
| 2 | 19.93 | 19.93 | 8.75 | 7.25 |
| 3 | 22.30 | 22.30 | 9.54 | 7.80 |
| 5 | 26.87 | 26.87 | 10.27 | 8.84 |
| 10 | 43.01 | 43.01 | 14.20 | 11.52 |

**Critical Values for Maximal IV Size (10% Wald test)**:

| # Instruments | 10% | 15% | 20% | 25% |
|---------------|------|------|------|------|
| 1 | 16.38 | 8.96 | 6.66 | 5.53 |
| 2 | 19.93 | 11.59 | 8.75 | 7.25 |
| 3 | 22.30 | 12.83 | 9.54 | 7.80 |
| 5 | 26.87 | 15.09 | 10.27 | 8.84 |

**Implementation**:
```python
from iv_estimator import weak_iv_diagnostics

weak_iv = weak_iv_diagnostics(
    first_stage_f=first_stage['f_statistic'],
    n_instruments=len(instruments),
    max_bias=10  # 10% maximal bias
)

print(weak_iv.interpretation)
print(f"Critical value: {weak_iv.threshold:.2f}")
print(f"Passed: {weak_iv.passed}")
```

### 1.3 Cragg-Donald Statistic

For multiple endogenous variables, the Cragg-Donald (1993) statistic generalizes the first-stage F.

**Formula**:
$$
CD = \frac{N - K_2 - K_1}{L} \cdot \lambda_{min}
$$

Where:
- N = sample size
- K_1 = number of exogenous regressors
- K_2 = number of endogenous regressors
- L = number of excluded instruments
- $\lambda_{min}$ = minimum eigenvalue of the concentration matrix

**Use Case**: When you have multiple endogenous variables.

### 1.4 Kleibergen-Paap Statistic

Robust version of Cragg-Donald for heteroskedasticity and clustering.

**Implementation** (using linearmodels):
```python
from linearmodels.iv import IV2SLS

model = IV2SLS(y, exog, endog, instruments)
results = model.fit(cov_type='robust')

# The Kleibergen-Paap F is automatically computed with robust SEs
```

---

## 2. Overidentification Tests

When you have more instruments than endogenous variables, you can test whether all instruments are valid.

### 2.1 Sargan Test (Homoskedasticity)

**Null Hypothesis**: All instruments are valid (exogenous)

**Procedure**:
1. Estimate IV model, obtain residuals $\hat{u}$
2. Regress $\hat{u}$ on all exogenous variables (instruments + controls)
3. Test statistic: $J = nR^2 \sim \chi^2(m-k)$
   - m = number of instruments
   - k = number of endogenous variables

**Implementation**:
```python
from iv_estimator import overidentification_test

j_test = overidentification_test(
    model_result=iv_result,
    data=df,
    outcome="y",
    treatment="d",
    instruments=["z1", "z2"],
    controls=["x1", "x2"]
)

print(f"J-statistic: {j_test.statistic:.4f}")
print(f"P-value: {j_test.p_value:.4f}")
print(f"Degrees of freedom: {j_test.details['degrees_of_freedom']}")
```

### 2.2 Hansen J-Test (Robust)

The Hansen J-test is the heteroskedasticity-robust version used with GMM estimation.

**Implementation**:
```python
from iv_estimator import estimate_gmm

gmm_result = estimate_gmm(df, "y", "d", ["z1", "z2"], ["x1", "x2"])

j_stat = gmm_result.diagnostics['j_statistic']
j_pval = gmm_result.diagnostics['j_pvalue']

print(f"Hansen J: {j_stat:.4f}")
print(f"P-value: {j_pval:.4f}")
```

**Interpretation**:
- **Do not reject** (p > 0.05): Cannot reject that all instruments are valid
- **Reject** (p < 0.05): Evidence that at least one instrument is invalid

**Caution**:
- This test has **low power** - failure to reject does not prove validity
- Cannot detect if ALL instruments are invalid in the same way
- Subject to "weak instrument" distortions

### 2.3 Difference-in-Sargan Test

Test individual instruments by comparing Sargan statistics from nested models.

**Procedure**:
1. Estimate with full instrument set, obtain $J_{full}$
2. Estimate with subset (excluding suspect instrument), obtain $J_{reduced}$
3. Test statistic: $J_{full} - J_{reduced} \sim \chi^2(1)$

```python
def difference_in_sargan(data, outcome, treatment,
                         all_instruments, suspect_instrument, controls):
    """
    Test if a specific instrument is valid using difference-in-Sargan.
    """
    # Full model
    result_full = estimate_gmm(data, outcome, treatment,
                               all_instruments, controls)
    j_full = result_full.diagnostics['j_statistic']

    # Reduced model (without suspect instrument)
    reduced_instruments = [z for z in all_instruments if z != suspect_instrument]
    result_reduced = estimate_gmm(data, outcome, treatment,
                                  reduced_instruments, controls)
    j_reduced = result_reduced.diagnostics['j_statistic']

    # Test statistic
    diff_stat = j_full - j_reduced
    p_value = 1 - stats.chi2.cdf(diff_stat, df=1)

    return {
        'statistic': diff_stat,
        'p_value': p_value,
        'rejected': p_value < 0.05
    }
```

---

## 3. Endogeneity Tests

These tests assess whether IV is needed or if OLS is consistent.

### 3.1 Wu-Hausman Test

**Null Hypothesis**: Treatment is exogenous (OLS is consistent)

**Procedure** (Regression-based):
1. Run first-stage: $D = Z\pi + X\gamma + v$
2. Get first-stage residuals: $\hat{v}$
3. Run augmented regression: $Y = D\beta + X\theta + \hat{v}\rho + \epsilon$
4. Test: $H_0: \rho = 0$

**Implementation**:
```python
from iv_estimator import endogeneity_test

hausman = endogeneity_test(
    data=df,
    outcome="y",
    treatment="d",
    instruments=["z1", "z2"],
    controls=["x1", "x2"]
)

print(f"Wu-Hausman t-stat: {hausman.statistic:.4f}")
print(f"P-value: {hausman.p_value:.4f}")
print(hausman.interpretation)
```

**Interpretation**:
- **Reject** (p < 0.05): Treatment is endogenous, IV is appropriate
- **Do not reject** (p > 0.05): Cannot reject exogeneity, OLS may be consistent
  - But IV is still valid if instruments are truly exogenous

### 3.2 Durbin-Wu-Hausman Test

Alternative formulation comparing OLS and IV estimates directly.

**Test Statistic**:
$$
DWH = (\hat{\beta}_{IV} - \hat{\beta}_{OLS})'[\widehat{Var}(\hat{\beta}_{IV}) - \widehat{Var}(\hat{\beta}_{OLS})]^{-1}(\hat{\beta}_{IV} - \hat{\beta}_{OLS})
$$

Under $H_0$: $DWH \sim \chi^2(k)$ where k = number of endogenous variables

```python
def durbin_wu_hausman(data, outcome, treatment, instruments, controls):
    """
    Compare OLS and IV estimates to test for endogeneity.
    """
    from iv_estimator import estimate_2sls, estimate_ols
    from scipy import stats

    iv_result = estimate_2sls(data, outcome, treatment, instruments, controls)
    ols_result = estimate_ols(data, outcome, treatment, controls)

    # Difference in estimates
    diff = iv_result.effect - ols_result.effect

    # Variance of difference (using approximation)
    var_diff = iv_result.se**2 - ols_result.se**2

    if var_diff > 0:
        dwh_stat = diff**2 / var_diff
        p_value = 1 - stats.chi2.cdf(dwh_stat, df=1)
    else:
        dwh_stat = np.nan
        p_value = np.nan

    return {
        'statistic': dwh_stat,
        'p_value': p_value,
        'iv_estimate': iv_result.effect,
        'ols_estimate': ols_result.effect,
        'difference': diff
    }
```

---

## 4. Diagnostic Summary Table

| Test | Null Hypothesis | When to Use | Interpretation |
|------|-----------------|-------------|----------------|
| **First-stage F** | Instruments are irrelevant | Always | F > 10 = strong instruments |
| **Stock-Yogo** | Weak instruments | Always | Compare to critical values |
| **Sargan/Hansen J** | All instruments valid | Overidentified | p > 0.05 = cannot reject validity |
| **Wu-Hausman** | Treatment is exogenous | Always | p < 0.05 = endogeneity present |
| **Anderson-Rubin** | True effect = 0 | Weak instruments | Robust inference |

---

## 5. Comprehensive Diagnostic Workflow

```python
from iv_estimator import (
    first_stage_test,
    weak_iv_diagnostics,
    estimate_2sls,
    estimate_liml,
    overidentification_test,
    endogeneity_test
)

def run_iv_diagnostics(data, outcome, treatment, instruments, controls):
    """
    Run complete IV diagnostic battery.
    """
    results = {}

    # 1. First-stage diagnostics
    print("=" * 60)
    print("1. FIRST-STAGE DIAGNOSTICS")
    print("=" * 60)

    first_stage = first_stage_test(data, treatment, instruments, controls)
    results['first_stage_f'] = first_stage['f_statistic']
    results['partial_r2'] = first_stage['partial_r2']

    print(f"First-stage F-statistic: {first_stage['f_statistic']:.2f}")
    print(f"Partial R-squared: {first_stage['partial_r2']:.4f}")

    # Stock-Yogo assessment
    weak_iv = weak_iv_diagnostics(first_stage['f_statistic'], len(instruments))
    results['weak_iv_passed'] = weak_iv.passed
    print(f"\nWeak IV Assessment: {weak_iv.interpretation}")

    # 2. Estimation
    print("\n" + "=" * 60)
    print("2. ESTIMATION")
    print("=" * 60)

    result_2sls = estimate_2sls(data, outcome, treatment, instruments, controls)
    result_liml = estimate_liml(data, outcome, treatment, instruments, controls)

    results['2sls'] = result_2sls.effect
    results['liml'] = result_liml.effect

    print(f"2SLS estimate: {result_2sls.effect:.4f} (SE: {result_2sls.se:.4f})")
    print(f"LIML estimate: {result_liml.effect:.4f} (SE: {result_liml.se:.4f})")
    print(f"Difference: {abs(result_2sls.effect - result_liml.effect):.4f}")

    # 3. Overidentification test
    if len(instruments) > 1:
        print("\n" + "=" * 60)
        print("3. OVERIDENTIFICATION TEST")
        print("=" * 60)

        j_test = overidentification_test(
            result_2sls, data, outcome, treatment, instruments, controls
        )
        results['j_stat'] = j_test.statistic
        results['j_pvalue'] = j_test.p_value
        results['overid_passed'] = j_test.passed

        print(j_test.interpretation)

    # 4. Endogeneity test
    print("\n" + "=" * 60)
    print("4. ENDOGENEITY TEST")
    print("=" * 60)

    endog = endogeneity_test(data, outcome, treatment, instruments, controls)
    results['endogeneity_stat'] = endog.statistic
    results['endogeneity_pvalue'] = endog.p_value
    results['is_endogenous'] = endog.passed

    print(endog.interpretation)

    # 5. Summary
    print("\n" + "=" * 60)
    print("DIAGNOSTIC SUMMARY")
    print("=" * 60)

    issues = []
    if not results['weak_iv_passed']:
        issues.append("Weak instruments detected")
    if len(instruments) > 1 and not results.get('overid_passed', True):
        issues.append("Overidentification test rejected")
    if not results['is_endogenous']:
        issues.append("Endogeneity not detected (OLS may be consistent)")

    if issues:
        print("CONCERNS:")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print("All diagnostics passed. IV estimation appears valid.")

    return results
```

---

## 6. Best Practices

### Always Report:
1. First-stage F-statistic
2. First-stage coefficients and standard errors
3. Overidentification test (if applicable)
4. Comparison of 2SLS and LIML (for weak instrument robustness)

### Red Flags:
- First-stage F < 10
- Large difference between 2SLS and LIML
- Sargan/Hansen test rejection
- Opposite signs for different instruments

### When Tests Conflict:
- Trust economic intuition over statistical tests
- Tests have low power against certain alternatives
- Multiple instruments failing overid test may indicate one bad instrument
- Consider sensitivity analysis

---

## References

- Stock, J. H., & Yogo, M. (2005). Testing for Weak Instruments in Linear IV Regression.
- Sargan, J. D. (1958). The Estimation of Economic Relationships Using Instrumental Variables.
- Hansen, L. P. (1982). Large Sample Properties of Generalized Method of Moments Estimators.
- Cragg, J. G., & Donald, S. G. (1993). Testing Identifiability and Specification in Instrumental Variable Models.
- Kleibergen, F., & Paap, R. (2006). Generalized Reduced Rank Tests Using the Singular Value Decomposition.
