# Estimation Methods for Causal Mediation Analysis

> **Reference Document** | Part of `causal-mediation-ml` skill
> **Version**: 1.0.0

## Overview

This document covers three main estimation approaches for causal mediation analysis:
1. **Product Method** (Baron-Kenny) - Classic parametric approach
2. **Difference Method** - Alternative identification strategy
3. **ML-Based Methods** - Modern approaches using machine learning

Each method has trade-offs in terms of assumptions, flexibility, and interpretability.

---

## Product Method (Baron-Kenny, 1986)

### Core Idea

Decompose the total effect into direct and indirect components using regression coefficients.

### Model Specification

**Step 1 - Mediator Model**:
$$
M_i = \alpha_0 + \alpha_1 D_i + \alpha_2' X_i + \epsilon_{Mi}
$$

**Step 2 - Outcome Model**:
$$
Y_i = \beta_0 + \beta_1 D_i + \beta_2 M_i + \beta_3' X_i + \epsilon_{Yi}
$$

### Effect Decomposition

| Effect | Formula | Interpretation |
|--------|---------|----------------|
| Indirect (ACME) | $\alpha_1 \times \beta_2$ | D -> M -> Y pathway |
| Direct (ADE) | $\beta_1$ | D -> Y holding M fixed |
| Total | $\beta_1 + \alpha_1 \beta_2$ | Overall D -> Y effect |

### Implementation

```python
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm

def baron_kenny_mediation(
    data: pd.DataFrame,
    outcome: str,
    treatment: str,
    mediator: str,
    controls: list = None,
    robust_se: bool = True
) -> dict:
    """
    Baron-Kenny (1986) mediation analysis with product method.

    Parameters
    ----------
    data : pd.DataFrame
        Input data
    outcome : str
        Name of outcome variable (Y)
    treatment : str
        Name of treatment variable (D)
    mediator : str
        Name of mediator variable (M)
    controls : list, optional
        Names of control variables (X)
    robust_se : bool
        Use heteroskedasticity-robust standard errors

    Returns
    -------
    dict with:
        - alpha: D->M coefficient
        - beta_d: D->Y coefficient (direct effect)
        - beta_m: M->Y coefficient
        - acme: indirect effect (alpha * beta_m)
        - ade: direct effect (beta_d)
        - total: total effect
        - se_*: standard errors
        - p_*: p-values
    """
    if controls is None:
        controls = []

    df = data[[outcome, treatment, mediator] + controls].dropna()

    # Mediator model: M ~ D + X
    X_m = sm.add_constant(df[[treatment] + controls])
    model_m = sm.OLS(df[mediator], X_m).fit(
        cov_type='HC3' if robust_se else 'nonrobust'
    )
    alpha = model_m.params[treatment]
    se_alpha = model_m.bse[treatment]

    # Outcome model: Y ~ D + M + X
    X_y = sm.add_constant(df[[treatment, mediator] + controls])
    model_y = sm.OLS(df[outcome], X_y).fit(
        cov_type='HC3' if robust_se else 'nonrobust'
    )
    beta_d = model_y.params[treatment]
    se_beta_d = model_y.bse[treatment]
    beta_m = model_y.params[mediator]
    se_beta_m = model_y.bse[mediator]

    # Calculate effects
    acme = alpha * beta_m
    ade = beta_d
    total = ade + acme

    # Sobel standard error for ACME (delta method)
    se_acme = np.sqrt(alpha**2 * se_beta_m**2 + beta_m**2 * se_alpha**2)

    # Inference
    z_acme = acme / se_acme
    p_acme = 2 * (1 - stats.norm.cdf(abs(z_acme)))

    z_ade = ade / se_beta_d
    p_ade = 2 * (1 - stats.norm.cdf(abs(z_ade)))

    # Proportion mediated
    prop_mediated = acme / total if abs(total) > 1e-10 else np.nan

    return {
        'alpha': alpha,
        'se_alpha': se_alpha,
        'beta_d': beta_d,
        'se_beta_d': se_beta_d,
        'beta_m': beta_m,
        'se_beta_m': se_beta_m,
        'acme': acme,
        'se_acme': se_acme,
        'p_acme': p_acme,
        'ci_acme': (acme - 1.96*se_acme, acme + 1.96*se_acme),
        'ade': ade,
        'se_ade': se_beta_d,
        'p_ade': p_ade,
        'ci_ade': (ade - 1.96*se_beta_d, ade + 1.96*se_beta_d),
        'total': total,
        'prop_mediated': prop_mediated,
        'n': len(df),
        'model_mediator': model_m,
        'model_outcome': model_y
    }
```

### Advantages

- Simple and intuitive
- Closed-form standard errors (Sobel test)
- Easy to interpret coefficients
- Works with small samples

### Limitations

- Assumes linear relationships
- Requires correct functional form
- Sensitive to model misspecification
- Does not handle high-dimensional confounders well

---

## Difference Method

### Core Idea

Estimate the indirect effect as the difference between total and direct effects.

### Model Specification

**Total Effect Model** (without mediator):
$$
Y_i = \gamma_0 + \gamma_1 D_i + \gamma_2' X_i + \nu_i
$$

**Direct Effect Model** (with mediator):
$$
Y_i = \beta_0 + \beta_1 D_i + \beta_2 M_i + \beta_3' X_i + \epsilon_{Yi}
$$

### Effect Decomposition

| Effect | Formula |
|--------|---------|
| Total | $\gamma_1$ |
| Direct (ADE) | $\beta_1$ |
| Indirect (ACME) | $\gamma_1 - \beta_1$ |

### Implementation

```python
def difference_method_mediation(
    data: pd.DataFrame,
    outcome: str,
    treatment: str,
    mediator: str,
    controls: list = None,
    robust_se: bool = True
) -> dict:
    """
    Difference method for mediation analysis.

    ACME = Total Effect - Direct Effect
    """
    if controls is None:
        controls = []

    df = data[[outcome, treatment, mediator] + controls].dropna()

    # Total effect model: Y ~ D + X
    X_total = sm.add_constant(df[[treatment] + controls])
    model_total = sm.OLS(df[outcome], X_total).fit(
        cov_type='HC3' if robust_se else 'nonrobust'
    )
    total_effect = model_total.params[treatment]
    se_total = model_total.bse[treatment]

    # Direct effect model: Y ~ D + M + X
    X_direct = sm.add_constant(df[[treatment, mediator] + controls])
    model_direct = sm.OLS(df[outcome], X_direct).fit(
        cov_type='HC3' if robust_se else 'nonrobust'
    )
    direct_effect = model_direct.params[treatment]
    se_direct = model_direct.bse[treatment]

    # Indirect effect (ACME)
    acme = total_effect - direct_effect

    # Standard error via bootstrap (recommended)
    # Approximation using delta method:
    # Note: This underestimates SE; use bootstrap for precise inference
    cov_total_direct = np.cov([
        model_total.resid * X_total[treatment],
        model_direct.resid * X_direct[treatment]
    ])[0, 1] / len(df)
    se_acme_approx = np.sqrt(se_total**2 + se_direct**2 - 2*cov_total_direct)

    return {
        'total_effect': total_effect,
        'se_total': se_total,
        'direct_effect': direct_effect,
        'se_direct': se_direct,
        'acme': acme,
        'se_acme': se_acme_approx,
        'ade': direct_effect,
        'prop_mediated': acme / total_effect if abs(total_effect) > 1e-10 else np.nan,
        'n': len(df)
    }
```

### Equivalence with Product Method

Under linearity and no treatment-mediator interaction:
$$
\gamma_1 - \beta_1 = \alpha_1 \times \beta_2
$$

The two methods give identical point estimates but may differ in standard errors.

---

## ML-Based Methods

### Overview

Modern approaches use machine learning to:
1. Flexibly estimate nuisance functions (E[M|D,X], E[Y|D,M,X])
2. Handle high-dimensional confounders
3. Maintain valid inference through cross-fitting

### Method 1: Simulation-Based (Imai et al., 2010)

Uses Monte Carlo simulation with fitted models.

```python
def simulation_based_mediation(
    data: pd.DataFrame,
    outcome: str,
    treatment: str,
    mediator: str,
    controls: list,
    n_simulations: int = 1000,
    ml_mediator: str = 'lasso',
    ml_outcome: str = 'lasso'
) -> dict:
    """
    Simulation-based mediation analysis (Imai et al., 2010).

    Simulates potential mediator values and outcomes to estimate
    causal mediation effects under general (nonlinear) models.
    """
    from sklearn.model_selection import cross_val_predict
    from sklearn.linear_model import LassoCV
    from sklearn.ensemble import RandomForestRegressor

    df = data[[outcome, treatment, mediator] + controls].dropna()
    n = len(df)

    # Prepare features
    X = df[controls].values
    D = df[treatment].values
    M = df[mediator].values
    Y = df[outcome].values

    # Get learners
    learner_m = LassoCV(cv=5) if ml_mediator == 'lasso' else RandomForestRegressor()
    learner_y = LassoCV(cv=5) if ml_outcome == 'lasso' else RandomForestRegressor()

    # Fit mediator model: M ~ D, X
    DX = np.column_stack([D, X])
    learner_m.fit(DX, M)

    # Fit outcome model: Y ~ D, M, X
    DMX = np.column_stack([D, M, X])
    learner_y.fit(DMX, Y)

    # Simulate potential outcomes
    acme_d1_sims = []
    acme_d0_sims = []
    ade_d1_sims = []
    ade_d0_sims = []

    for _ in range(n_simulations):
        # Sample from residual distribution
        m_pred = learner_m.predict(DX)
        m_resid = M - m_pred
        resid_sample = np.random.choice(m_resid, size=n, replace=True)

        # Potential mediators
        M_1 = learner_m.predict(np.column_stack([np.ones(n), X])) + resid_sample
        M_0 = learner_m.predict(np.column_stack([np.zeros(n), X])) + resid_sample

        # Potential outcomes for ACME(1): Y(1, M(1)) - Y(1, M(0))
        Y_1_M1 = learner_y.predict(np.column_stack([np.ones(n), M_1, X]))
        Y_1_M0 = learner_y.predict(np.column_stack([np.ones(n), M_0, X]))
        acme_d1_sims.append(np.mean(Y_1_M1 - Y_1_M0))

        # ACME(0): Y(0, M(1)) - Y(0, M(0))
        Y_0_M1 = learner_y.predict(np.column_stack([np.zeros(n), M_1, X]))
        Y_0_M0 = learner_y.predict(np.column_stack([np.zeros(n), M_0, X]))
        acme_d0_sims.append(np.mean(Y_0_M1 - Y_0_M0))

        # ADE(1): Y(1, M(1)) - Y(0, M(1))
        ade_d1_sims.append(np.mean(Y_1_M1 - Y_0_M1))

        # ADE(0): Y(1, M(0)) - Y(0, M(0))
        ade_d0_sims.append(np.mean(Y_1_M0 - Y_0_M0))

    # Aggregate
    acme = np.mean(acme_d1_sims)  # Average ACME
    ade = np.mean(ade_d1_sims)

    return {
        'acme': acme,
        'acme_se': np.std(acme_d1_sims),
        'acme_d1': np.mean(acme_d1_sims),
        'acme_d0': np.mean(acme_d0_sims),
        'ade': ade,
        'ade_se': np.std(ade_d1_sims),
        'ade_d1': np.mean(ade_d1_sims),
        'ade_d0': np.mean(ade_d0_sims),
        'total': acme + ade,
        'n_simulations': n_simulations
    }
```

### Method 2: DDML-Style Cross-Fitting

Combines double machine learning principles with mediation.

```python
def ddml_mediation(
    data: pd.DataFrame,
    outcome: str,
    treatment: str,
    mediator: str,
    controls: list,
    ml_m: str = 'lasso',
    ml_y: str = 'lasso',
    n_folds: int = 5
) -> dict:
    """
    DDML-style mediation analysis with cross-fitting.

    Uses out-of-sample predictions to avoid overfitting bias.
    """
    from sklearn.model_selection import KFold
    from sklearn.linear_model import LassoCV
    from sklearn.ensemble import RandomForestRegressor

    df = data[[outcome, treatment, mediator] + controls].dropna()
    n = len(df)

    X = df[controls].values
    D = df[treatment].values
    M = df[mediator].values
    Y = df[outcome].values

    # Cross-fitting storage
    m_hat_d1 = np.zeros(n)
    m_hat_d0 = np.zeros(n)
    y_hat_d1_m = np.zeros(n)
    y_hat_d0_m = np.zeros(n)

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)

    for train_idx, test_idx in kf.split(X):
        # Split data
        X_train, X_test = X[train_idx], X[test_idx]
        D_train, D_test = D[train_idx], D[test_idx]
        M_train, M_test = M[train_idx], M[test_idx]
        Y_train = Y[train_idx]

        # Mediator model
        DX_train = np.column_stack([D_train, X_train])
        learner_m = LassoCV(cv=3) if ml_m == 'lasso' else RandomForestRegressor()
        learner_m.fit(DX_train, M_train)

        # Predict M under D=1 and D=0
        m_hat_d1[test_idx] = learner_m.predict(np.column_stack([np.ones(len(test_idx)), X_test]))
        m_hat_d0[test_idx] = learner_m.predict(np.column_stack([np.zeros(len(test_idx)), X_test]))

        # Outcome model
        DMX_train = np.column_stack([D_train, M_train, X_train])
        learner_y = LassoCV(cv=3) if ml_y == 'lasso' else RandomForestRegressor()
        learner_y.fit(DMX_train, Y_train)

        # Predict Y under D=1 and D=0 with observed M
        y_hat_d1_m[test_idx] = learner_y.predict(np.column_stack([np.ones(len(test_idx)), M_test, X_test]))
        y_hat_d0_m[test_idx] = learner_y.predict(np.column_stack([np.zeros(len(test_idx)), M_test, X_test]))

    # Calculate effects
    # ADE using observed mediator values
    ade = np.mean(y_hat_d1_m - y_hat_d0_m)

    # Treatment effect on mediator
    alpha = np.mean(m_hat_d1 - m_hat_d0)

    # Total effect (separate estimation)
    from sklearn.linear_model import LinearRegression
    DX = np.column_stack([D, X])
    total_model = LinearRegression().fit(DX, Y)
    total_effect = total_model.coef_[0]

    # ACME = Total - ADE
    acme = total_effect - ade

    # Bootstrap for standard errors
    n_boot = 500
    acme_boot = []
    ade_boot = []

    for _ in range(n_boot):
        idx = np.random.choice(n, n, replace=True)
        ade_b = np.mean((y_hat_d1_m - y_hat_d0_m)[idx])
        total_b = np.mean(Y[idx][D[idx]==1]) - np.mean(Y[idx][D[idx]==0]) if D[idx].std() > 0 else total_effect
        acme_boot.append(total_b - ade_b)
        ade_boot.append(ade_b)

    return {
        'acme': acme,
        'acme_se': np.std(acme_boot),
        'ade': ade,
        'ade_se': np.std(ade_boot),
        'total': total_effect,
        'alpha': alpha,
        'n_folds': n_folds,
        'n': n
    }
```

### Method 3: Causal Forests for Heterogeneous Mediation

Using Generalized Random Forests for conditional effects.

```python
def grf_mediation(
    data: pd.DataFrame,
    outcome: str,
    treatment: str,
    mediator: str,
    controls: list
) -> dict:
    """
    Causal forest approach for heterogeneous mediation effects.

    Uses R's grf package via rpy2.

    Returns conditional ACME and ADE as functions of X.
    """
    try:
        import rpy2.robjects as ro
        from rpy2.robjects import pandas2ri
        from rpy2.robjects.packages import importr

        pandas2ri.activate()
        grf = importr('grf')

        df = data[[outcome, treatment, mediator] + controls].dropna()

        # Convert to R
        r_df = pandas2ri.py2rpy(df)

        # This would require custom GRF mediation functions
        # Placeholder for actual implementation

        return {
            'method': 'grf_mediation',
            'status': 'requires_r_implementation'
        }

    except ImportError:
        return {
            'error': 'rpy2 or grf not available',
            'recommendation': 'Use Python-based methods or install R dependencies'
        }
```

---

## Method Comparison

### Summary Table

| Method | Flexibility | High-Dim X | Inference | Sample Size |
|--------|-------------|------------|-----------|-------------|
| Baron-Kenny | Low (linear) | Poor | Analytical | Small OK |
| Difference | Low (linear) | Poor | Bootstrap | Small OK |
| Simulation-Based | Medium | Medium | Simulation | Medium |
| DDML Cross-Fit | High | Excellent | Bootstrap | Large (>500) |
| Causal Forest | Very High | Excellent | Built-in | Large (>1000) |

### Decision Guide

```
START
  |
  v
Is n > 500?
  |
  +--No--> Use Baron-Kenny or Difference Method
  |
  +--Yes--> Are there many controls (p > 20)?
              |
              +--No--> Baron-Kenny is fine, ML optional
              |
              +--Yes--> Is heterogeneity important?
                          |
                          +--No--> DDML Cross-Fitting
                          |
                          +--Yes--> Causal Forest (if R available)
                                    or DDML with subgroup analysis
```

---

## Inference Methods

### Sobel Test (Product Method)

Standard error using delta method:
$$
SE_{ACME} = \sqrt{\alpha^2 \cdot SE_{\beta_M}^2 + \beta_M^2 \cdot SE_{\alpha}^2}
$$

### Bootstrap

Recommended for all methods, especially ML-based:

```python
def bootstrap_mediation_inference(
    data: pd.DataFrame,
    estimation_func,
    n_bootstrap: int = 1000,
    confidence_level: float = 0.95,
    **kwargs
) -> dict:
    """
    Bootstrap confidence intervals for mediation effects.
    """
    n = len(data)
    acme_boot = []
    ade_boot = []

    for _ in range(n_bootstrap):
        boot_idx = np.random.choice(n, n, replace=True)
        boot_data = data.iloc[boot_idx].reset_index(drop=True)

        try:
            result = estimation_func(boot_data, **kwargs)
            acme_boot.append(result['acme'])
            ade_boot.append(result['ade'])
        except:
            continue

    alpha = 1 - confidence_level
    lower_q = alpha / 2 * 100
    upper_q = (1 - alpha / 2) * 100

    return {
        'acme_se': np.std(acme_boot),
        'acme_ci': (np.percentile(acme_boot, lower_q),
                    np.percentile(acme_boot, upper_q)),
        'ade_se': np.std(ade_boot),
        'ade_ci': (np.percentile(ade_boot, lower_q),
                   np.percentile(ade_boot, upper_q)),
        'n_successful': len(acme_boot)
    }
```

---

## References

### Classic Methods
- Baron, R. M., & Kenny, D. A. (1986). The Moderator-Mediator Variable Distinction. *JPSP*.
- MacKinnon, D. P. (2008). *Introduction to Statistical Mediation Analysis*. Erlbaum.

### Modern Approaches
- Imai, K., Keele, L., & Tingley, D. (2010). A General Approach to Causal Mediation Analysis. *Psychological Methods*.
- Farbmacher, H., et al. (2022). Causal Mediation Analysis with Double Machine Learning. *Econometrics Journal*.

### Software
- `mediation` R package: https://cran.r-project.org/package=mediation
- `econml` Python: https://econml.azurewebsites.net/
