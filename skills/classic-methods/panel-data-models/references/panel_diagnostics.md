# Panel Data Diagnostics

## Overview

Before and after estimation, diagnostic tests help ensure model validity. Key concerns:
- Serial correlation in errors
- Heteroskedasticity
- Cross-sectional dependence
- Unit roots (non-stationarity)
- Model specification

## Serial Correlation Tests

### Wooldridge Test for FE Models

Tests for first-order autocorrelation in idiosyncratic errors.

**Procedure**:
1. Run FE regression, get residuals $\hat{u}_{it}$
2. Regress $\hat{u}_{it}$ on $\hat{u}_{i,t-1}$
3. F-test for $H_0: \rho = 0$

```python
import statsmodels.api as sm
from scipy import stats

def wooldridge_test(residuals, entity_col, time_col, data):
    """Wooldridge test for serial correlation in FE panels."""
    # Get lagged residuals
    df = data.copy()
    df['resid'] = residuals
    df['resid_lag'] = df.groupby(entity_col)['resid'].shift(1)

    # Drop missing
    valid = df['resid_lag'].notna()

    # Regress residuals on lagged residuals
    y = df.loc[valid, 'resid'].values
    X = sm.add_constant(df.loc[valid, 'resid_lag'].values)

    model = sm.OLS(y, X).fit()

    # F-test
    f_stat = model.tvalues[1] ** 2
    p_value = 1 - stats.f.cdf(f_stat, 1, model.df_resid)

    return {
        'rho': model.params[1],
        'f_statistic': f_stat,
        'p_value': p_value,
        'conclusion': 'Serial correlation detected' if p_value < 0.05
                      else 'No evidence of serial correlation'
    }
```

### Breusch-Godfrey Test

General test for AR(p) serial correlation:

$$\hat{u}_{it} = X_{it}\gamma + \sum_{s=1}^{p} \rho_s \hat{u}_{i,t-s} + v_{it}$$

Test $H_0: \rho_1 = \rho_2 = ... = \rho_p = 0$

### Durbin-Watson (Modified for Panels)

Panel-adjusted DW statistic:
- DW near 2: No serial correlation
- DW near 0: Positive serial correlation
- DW near 4: Negative serial correlation

**Caveat**: Less reliable with unbalanced panels or entity effects.

## Heteroskedasticity Tests

### Modified Wald Test

Tests for groupwise heteroskedasticity in FE residuals.

**Procedure**:
1. Compute variance of residuals for each entity: $\hat{\sigma}^2_i$
2. Test $H_0: \sigma^2_1 = \sigma^2_2 = ... = \sigma^2_N$

```python
def modified_wald_test(residuals, entity_col, data):
    """Modified Wald test for groupwise heteroskedasticity."""
    df = data.copy()
    df['resid'] = residuals

    # Entity-specific variances
    group_vars = df.groupby(entity_col)['resid'].var()

    # Overall variance
    sigma2 = residuals.var()

    # Test statistic (chi-squared)
    N = len(group_vars)
    T_i = df.groupby(entity_col).size()

    W = sum((T_i[i] - 1) * ((group_vars[i] - sigma2) / sigma2) ** 2
            for i in group_vars.index)

    p_value = 1 - stats.chi2.cdf(W, N - 1)

    return {
        'statistic': W,
        'df': N - 1,
        'p_value': p_value,
        'conclusion': 'Heteroskedasticity detected' if p_value < 0.05
                      else 'No evidence of heteroskedasticity'
    }
```

### Breusch-Pagan Test

Tests if error variance depends on regressors:

$$\hat{u}^2_{it} = X_{it}\delta + v_{it}$$

Test $H_0: \delta = 0$

### White Test

Non-parametric test including squares and cross-products of regressors.

## Cross-Sectional Dependence Tests

### Pesaran CD Test

Tests for cross-sectional dependence (spatial correlation):

$$CD = \sqrt{\frac{2T}{N(N-1)}} \sum_{i=1}^{N-1} \sum_{j=i+1}^{N} \hat{\rho}_{ij}$$

Where $\hat{\rho}_{ij}$ is the correlation between residuals of entities $i$ and $j$.

Under $H_0$ (independence): $CD \sim N(0,1)$

```python
def pesaran_cd_test(residuals, entity_col, time_col, data):
    """Pesaran CD test for cross-sectional dependence."""
    df = data.copy()
    df['resid'] = residuals

    # Pivot to entity x time matrix
    resid_wide = df.pivot(index=time_col, columns=entity_col, values='resid')

    N = resid_wide.shape[1]
    T = resid_wide.shape[0]

    # Pairwise correlations
    corr_sum = 0
    for i in range(N - 1):
        for j in range(i + 1, N):
            # Get common time periods
            valid = resid_wide.iloc[:, [i, j]].dropna()
            if len(valid) > 2:
                corr_sum += valid.iloc[:, 0].corr(valid.iloc[:, 1])

    CD = np.sqrt(2 * T / (N * (N - 1))) * corr_sum
    p_value = 2 * (1 - stats.norm.cdf(abs(CD)))

    return {
        'CD_statistic': CD,
        'p_value': p_value,
        'conclusion': 'Cross-sectional dependence detected' if p_value < 0.05
                      else 'No evidence of cross-sectional dependence'
    }
```

### Frees Test

More powerful alternative based on average squared correlation.

## Unit Root Tests

### Panel Unit Root Tests

For each entity, test:
$$\Delta y_{it} = \alpha_i + \rho y_{i,t-1} + \epsilon_{it}$$

$H_0: \rho = 0$ (unit root) vs $H_1: \rho < 0$ (stationary)

### Im-Pesaran-Shin (IPS) Test

Averages individual ADF t-statistics:
$$\bar{t} = \frac{1}{N} \sum_{i=1}^{N} t_i$$

Standardized: $W = \frac{\sqrt{N}(\bar{t} - E[\bar{t}])}{\sqrt{Var(\bar{t})}} \sim N(0,1)$

### Levin-Lin-Chu (LLC) Test

Pools all observations, assumes common $\rho$:
$$\Delta y_{it} = \alpha_i + \rho y_{i,t-1} + \sum_{k=1}^{p} \gamma_k \Delta y_{i,t-k} + \epsilon_{it}$$

### Hadri Test

Reverses hypotheses:
- $H_0$: All series are stationary
- $H_1$: Some series have unit roots

## Specification Tests

### RESET Test (Ramsey)

Tests functional form by adding powers of fitted values:

$$y_{it} = X_{it}\beta + \gamma_1 \hat{y}^2_{it} + \gamma_2 \hat{y}^3_{it} + \epsilon_{it}$$

$H_0: \gamma_1 = \gamma_2 = 0$

### Poolability Test

Tests whether coefficients are constant across entities:

$$y_{it} = X_{it}\beta_i + \epsilon_{it}$$

$H_0: \beta_1 = \beta_2 = ... = \beta_N = \beta$

F-test comparing restricted (pooled) vs unrestricted (entity-specific) model.

```python
def poolability_test(data, entity_col, time_col, y_col, x_cols):
    """Test if coefficients are constant across entities."""
    # Restricted model (pooled)
    y = data[y_col].values
    X = sm.add_constant(data[x_cols].values)
    model_r = sm.OLS(y, X).fit()
    RSS_r = model_r.ssr
    df_r = model_r.df_resid

    # Unrestricted model (entity-specific)
    RSS_ur = 0
    df_ur = 0
    for entity in data[entity_col].unique():
        mask = data[entity_col] == entity
        y_i = data.loc[mask, y_col].values
        X_i = sm.add_constant(data.loc[mask, x_cols].values)
        model_i = sm.OLS(y_i, X_i).fit()
        RSS_ur += model_i.ssr
        df_ur += model_i.df_resid

    # F-test
    k = len(x_cols) + 1  # Including constant
    N = data[entity_col].nunique()

    F = ((RSS_r - RSS_ur) / ((N - 1) * k)) / (RSS_ur / df_ur)
    p_value = 1 - stats.f.cdf(F, (N - 1) * k, df_ur)

    return {
        'F_statistic': F,
        'df1': (N - 1) * k,
        'df2': df_ur,
        'p_value': p_value,
        'conclusion': 'Reject poolability (coefficients differ)' if p_value < 0.05
                      else 'Cannot reject poolability'
    }
```

## Diagnostic Summary Table

| Issue | Test | H0 | If Rejected |
|-------|------|-----|-------------|
| Serial correlation | Wooldridge | No AR(1) | Use HAC SE or AR(1) correction |
| Heteroskedasticity | Modified Wald | Homoskedasticity | Use robust/clustered SE |
| Cross-sectional dep. | Pesaran CD | Independence | Use Driscoll-Kraay SE |
| Unit roots | IPS, LLC | Unit root | Difference or cointegration |
| FE vs RE | Hausman | RE consistent | Use FE |
| Poolability | F-test | Same coefficients | Random coefficients model |

## Diagnostic Workflow

```python
def run_panel_diagnostics(estimator, result):
    """Comprehensive panel diagnostics."""
    diagnostics = {}

    # 1. Serial correlation
    diagnostics['serial_correlation'] = wooldridge_test(
        result.residuals,
        estimator.entity_col,
        estimator.time_col,
        estimator.data
    )

    # 2. Heteroskedasticity
    diagnostics['heteroskedasticity'] = modified_wald_test(
        result.residuals,
        estimator.entity_col,
        estimator.data
    )

    # 3. Cross-sectional dependence
    diagnostics['cross_sectional_dep'] = pesaran_cd_test(
        result.residuals,
        estimator.entity_col,
        estimator.time_col,
        estimator.data
    )

    # 4. Model specification
    diagnostics['hausman'] = estimator.hausman_test()
    diagnostics['mundlak'] = estimator.within_between_test()

    # Summary
    print("\n" + "="*60)
    print("PANEL DIAGNOSTICS SUMMARY")
    print("="*60)

    for test, result in diagnostics.items():
        if isinstance(result, dict):
            print(f"\n{test.upper()}:")
            print(f"  p-value: {result.get('p_value', 'N/A'):.4f}")
            print(f"  Conclusion: {result.get('conclusion', 'N/A')}")

    return diagnostics
```

## Recommendations Based on Diagnostics

### If Serial Correlation Detected:
1. Use clustered standard errors (at minimum)
2. Consider Newey-West (HAC) standard errors
3. For dynamic models, check AR(2) in Arellano-Bond

### If Heteroskedasticity Detected:
1. Use heteroskedasticity-robust SE
2. Consider weighted least squares
3. Log transformation if variance increases with level

### If Cross-Sectional Dependence Detected:
1. Use Driscoll-Kraay standard errors
2. Consider spatial econometric models
3. Add time effects if not already included

### If Non-Stationarity:
1. First-difference the data
2. Consider panel cointegration if variables cointegrated
3. Use panel error correction model

## References

- Wooldridge, J.M. (2002). Econometric Analysis of Cross Section and Panel Data, Ch. 10
- Pesaran, M.H. (2021). General Diagnostic Tests for Cross-Sectional Dependence in Panels
- Drukker, D.M. (2003). Testing for Serial Correlation in Linear Panel-Data Models. Stata Journal
- Im, K.S., Pesaran, M.H., & Shin, Y. (2003). Testing for Unit Roots in Heterogeneous Panels
