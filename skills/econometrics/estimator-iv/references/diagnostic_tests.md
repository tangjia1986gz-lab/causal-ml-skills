# IV Diagnostic Tests

## 1. Weak Instrument Test (First-Stage F)

### Stock-Yogo Critical Values

Test for maximal IV size (bias relative to OLS):

| # Instruments | 10% Size | 15% Size | 20% Size | 25% Size |
|:-------------:|:--------:|:--------:|:--------:|:--------:|
| 1 | 16.38 | 8.96 | 6.66 | 5.53 |
| 2 | 19.93 | 11.59 | 8.75 | 7.25 |
| 3 | 22.30 | 12.83 | 9.54 | 7.80 |
| 4 | 24.58 | 13.96 | 10.26 | 8.31 |
| 5 | 26.87 | 15.09 | 10.98 | 8.84 |

**Rule of Thumb:** F > 10 (Staiger & Stock, 1997)

**Stricter Threshold:** F > 104.7 for <5% bias (Lee et al., 2022)

### Python Implementation

```python
import statsmodels.api as sm
import numpy as np

def weak_iv_test(df, endogenous, instruments, controls=None):
    """
    Test instrument strength via first-stage F-statistic.
    """
    if controls is None:
        controls = []

    X = df[instruments + controls + ['const']]
    y = df[endogenous]

    result = sm.OLS(y, X).fit(cov_type='HC1')

    # F-test for excluded instruments
    n_inst = len(instruments)
    r_matrix = np.zeros((n_inst, len(X.columns)))
    for i, inst in enumerate(instruments):
        r_matrix[i, list(X.columns).index(inst)] = 1

    f_test = result.f_test(r_matrix)

    return {
        'f_stat': float(f_test.fvalue),
        'p_value': float(f_test.pvalue),
        'is_strong': float(f_test.fvalue) > 10
    }
```

## 2. Sargan-Hansen Test (Overidentification)

**Null Hypothesis:** All instruments are valid (uncorrelated with error)

**Statistic:**
```
J = n * R^2 from residual regression
```
Where residuals from 2SLS are regressed on all exogenous variables including instruments.

**Distribution:** Chi-squared with (L - K) degrees of freedom
- L = number of instruments
- K = number of endogenous variables

**Interpretation:**
- Reject H0 (p < 0.05): At least one instrument may be invalid
- Fail to reject: Instruments appear valid

**Caveat:** Only tests overidentifying restrictions. If all instruments are invalid in similar ways, test won't detect it.

### Python (linearmodels)

```python
from linearmodels.iv import IV2SLS

result = IV2SLS(...).fit()
sargan = result.sargan()  # Call as method
print(f"J-stat: {sargan.stat:.4f}, p-value: {sargan.pval:.4f}")
```

## 3. Wu-Hausman Test (Endogeneity)

**Null Hypothesis:** OLS is consistent (X is exogenous)

**Alternative:** X is endogenous, IV is needed

**Interpretation:**
- Reject H0 (p < 0.05): Endogeneity present, use IV
- Fail to reject: OLS may be consistent, but IV is still consistent

### Python (linearmodels)

```python
result = IV2SLS(...).fit()
hausman = result.wu_hausman()  # Call as method
print(f"Stat: {hausman.stat:.4f}, p-value: {hausman.pval:.4f}")
```

## 4. Anderson-Rubin Test (Weak IV Robust)

**Purpose:** Valid inference even with weak instruments

**Null Hypothesis:** beta = beta_0

**Test Statistic:**
```
AR(beta_0) = (SSR_restricted - SSR_unrestricted) / (k * sigma^2)
```

**Distribution:** F(k, n-k-1) under H0

### Python Implementation

```python
def anderson_rubin_test(df, outcome, endogenous, instruments, controls, beta_0):
    """
    Anderson-Rubin test for coefficient.
    """
    import statsmodels.api as sm
    from scipy import stats

    if controls is None:
        controls = []

    # Construct Y - beta_0 * X
    y_adj = df[outcome] - beta_0 * df[endogenous]

    X = df[instruments + controls + ['const']]
    result = sm.OLS(y_adj, X).fit()

    # F-test for instruments
    n_inst = len(instruments)
    r_matrix = np.zeros((n_inst, len(X.columns)))
    for i, inst in enumerate(instruments):
        r_matrix[i, list(X.columns).index(inst)] = 1

    f_test = result.f_test(r_matrix)

    return {
        'ar_stat': float(f_test.fvalue),
        'p_value': float(f_test.pvalue)
    }
```

## 5. Cragg-Donald Statistic

Generalization of first-stage F for multiple endogenous variables.

**Formula:**
```
CD = lambda_min / (1 + K/T)
```
Where lambda_min is the minimum eigenvalue of a matrix involving first-stage residuals.

## Test Decision Tree

```
1. First-Stage F-test
   │
   ├── F > 10: Strong instruments
   │   └── Proceed with standard 2SLS inference
   │
   └── F < 10: Weak instruments
       │
       ├── Use LIML instead of 2SLS
       │
       └── Use Anderson-Rubin CI for inference

2. Hausman Test (if F > 10)
   │
   ├── Reject (p < 0.05): Endogeneity present
   │   └── Report IV estimates as main results
   │
   └── Fail to reject: No clear endogeneity
       └── Report both OLS and IV

3. Sargan Test (if overidentified and F > 10)
   │
   ├── Reject: Instrument validity concern
   │   └── Re-examine instrument choice
   │
   └── Fail to reject: Instruments appear valid
       └── Proceed with IV estimates
```
