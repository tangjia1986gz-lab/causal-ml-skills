# Panel Data Diagnostic Tests

## 1. Hausman Test (FE vs RE)

### Purpose
Test whether Random Effects assumption (Cov(alpha_i, X_it) = 0) holds.

### Hypotheses
- H0: Random Effects is consistent (use RE for efficiency)
- H1: Fixed Effects needed (effects correlated with regressors)

### Test Statistic
```
H = (b_FE - b_RE)' * [Var(b_FE) - Var(b_RE)]^{-1} * (b_FE - b_RE)

H ~ Chi-squared(K)  under H0
```

### Implementation
```python
def hausman_test(fe_result, re_result):
    """Hausman specification test."""
    import numpy as np
    from scipy import stats

    # Get common parameters
    params = fe_result.params.index.intersection(re_result.params.index)
    params = params.drop('Intercept', errors='ignore')

    b_fe = fe_result.params[params].values
    b_re = re_result.params[params].values
    diff = b_fe - b_re

    cov_fe = fe_result.cov.loc[params, params].values
    cov_re = re_result.cov.loc[params, params].values
    var_diff = cov_fe - cov_re

    # Hausman statistic
    H = float(diff @ np.linalg.inv(var_diff) @ diff)
    df = len(params)
    p_value = 1 - stats.chi2.cdf(H, df)

    return {'statistic': H, 'df': df, 'p_value': p_value}
```

### Decision Rule

| P-value | Decision | Interpretation |
|---------|----------|----------------|
| < 0.01 | Reject H0 | Strong evidence for FE |
| 0.01-0.05 | Reject H0 | Moderate evidence for FE |
| 0.05-0.10 | Marginal | Consider context |
| > 0.10 | Fail to reject | RE may be appropriate |

## 2. F-Test for Fixed Effects

### Purpose
Test joint significance of entity fixed effects.

### Hypotheses
- H0: alpha_1 = alpha_2 = ... = alpha_N (no entity effects)
- H1: At least one alpha_i differs

### Test Statistic
```
F = [(RSS_pooled - RSS_FE) / (N - 1)] / [RSS_FE / (NT - N - K)]

F ~ F(N-1, NT-N-K)
```

### Implementation
```python
result = model.fit()

# F-statistic from linearmodels
f_stat = result.f_statistic.stat
p_value = result.f_statistic.pval

print(f"F({result.f_statistic.df1}, {result.f_statistic.df2}) = {f_stat:.4f}")
print(f"P-value: {p_value:.4f}")
```

## 3. Breusch-Pagan LM Test (RE vs Pooled)

### Purpose
Test for presence of random effects (variance of entity effect > 0).

### Hypotheses
- H0: sigma_u^2 = 0 (no random effects, Pooled OLS sufficient)
- H1: sigma_u^2 > 0 (RE appropriate)

### Test Statistic
```
LM = [NT / 2(T-1)] * [sum_i (sum_t e_it)^2 / sum_i sum_t e_it^2 - 1]^2

LM ~ Chi-squared(1)
```

### Implementation
```python
def breusch_pagan_test(pooled_result, n_entities, n_periods):
    """Breusch-Pagan LM test for random effects."""
    from scipy import stats

    resid = pooled_result.resids
    N, T = n_entities, n_periods

    # Reshape residuals by entity
    resid_matrix = resid.values.reshape(N, T)

    # Calculate test statistic
    num = (resid_matrix.sum(axis=1)**2).sum()
    den = (resid_matrix**2).sum()

    LM = (N * T / (2 * (T - 1))) * (num / den - 1)**2

    p_value = 1 - stats.chi2.cdf(LM, 1)

    return {'statistic': LM, 'df': 1, 'p_value': p_value}
```

## 4. Serial Correlation Tests

### Wooldridge Test

Tests for first-order serial correlation in FE residuals.

```python
def wooldridge_test(fe_result, df):
    """Wooldridge test for serial correlation in panel FE."""
    from statsmodels.regression.linear_model import OLS

    # Get residuals
    resid = fe_result.resids.reset_index()
    resid['lag_resid'] = resid.groupby('entity_id')['resid'].shift(1)

    # Regress residual on lagged residual
    clean = resid.dropna()
    model = OLS(clean['resid'], clean['lag_resid']).fit()

    # Null: rho = 0 (no serial correlation)
    return {
        'rho': model.params[0],
        't_stat': model.tvalues[0],
        'p_value': model.pvalues[0]
    }
```

## 5. Cross-Sectional Dependence Tests

### Pesaran CD Test

Tests for cross-sectional dependence in panel residuals.

```python
def pesaran_cd_test(residuals, entities, time):
    """Pesaran CD test for cross-sectional dependence."""
    import numpy as np
    from scipy import stats

    # Reshape residuals
    df_resid = pd.DataFrame({
        'entity': entities,
        'time': time,
        'resid': residuals
    })
    resid_wide = df_resid.pivot(index='time', columns='entity', values='resid')

    # Pairwise correlations
    N = resid_wide.shape[1]
    T = resid_wide.shape[0]

    # CD statistic
    corr_matrix = resid_wide.corr()
    rho_sum = corr_matrix.values[np.triu_indices(N, k=1)].sum()

    CD = np.sqrt(2 * T / (N * (N - 1))) * rho_sum

    p_value = 2 * (1 - stats.norm.cdf(abs(CD)))

    return {'statistic': CD, 'p_value': p_value}
```

## 6. Unit Root Tests for Panels

### Levin-Lin-Chu (LLC) Test

Tests for unit root in panel data.

- H0: All panels contain unit root
- H1: All panels are stationary

### Im-Pesaran-Shin (IPS) Test

Allows heterogeneous autoregressive parameters.

- H0: All panels contain unit root
- H1: Some panels are stationary

```python
from arch.unitroot import IPS

# Requires arch package
ips_test = IPS(y_panel, lags=1)
print(ips_test.summary())
```

## Diagnostic Decision Tree

```
Start
│
├─ Is panel balanced?
│  └─ No → Check for selection, consider unbalanced methods
│
├─ Run Hausman test
│  ├─ p < 0.05 → Use Fixed Effects
│  └─ p >= 0.05 → Consider Random Effects
│
├─ Check serial correlation (Wooldridge test)
│  └─ Significant → Cluster SE by entity
│
├─ Check cross-sectional dependence (Pesaran CD)
│  └─ Significant → Consider Driscoll-Kraay SE or two-way clustering
│
└─ Check for unit roots (if macro panel)
   └─ Non-stationary → First-difference or error-correction model
```

## Summary Table

| Test | H0 | Use Case |
|------|-----|----------|
| Hausman | RE consistent | FE vs RE selection |
| F-test (FE) | No entity effects | Pooled vs FE |
| Breusch-Pagan | No random effects | Pooled vs RE |
| Wooldridge | No serial correlation | SE clustering |
| Pesaran CD | No cross-sectional dependence | SE clustering |
| LLC/IPS | Unit root | Stationarity |

## References

- Wooldridge, J. (2010). *Econometric Analysis of Cross Section and Panel Data*, Ch. 10-11.
- Baltagi, B. (2021). *Econometric Analysis of Panel Data*, Ch. 5.
- Pesaran, M.H. (2004). "General Diagnostic Tests for Cross Section Dependence in Panels." *Cambridge Working Papers*.
