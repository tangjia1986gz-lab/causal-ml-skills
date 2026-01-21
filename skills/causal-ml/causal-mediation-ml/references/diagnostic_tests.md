# Diagnostic Tests for Causal Mediation Analysis

> **Reference Document** | Part of `causal-mediation-ml` skill
> **Version**: 1.0.0

## Overview

Since sequential ignorability cannot be directly tested, diagnostic tests for mediation analysis focus on:
1. **Sensitivity analysis** - How robust are results to unmeasured confounding?
2. **Confounding diagnostics** - Evidence for/against assumption violations
3. **Model specification** - Are functional forms appropriate?
4. **Balance and overlap** - Data quality checks

---

## Sensitivity Analysis (Imai et al., 2010)

### The Sensitivity Parameter (rho)

The sensitivity parameter $\rho$ captures the correlation between the error terms in the mediator and outcome models that would arise from unmeasured confounding.

**Setup**:
- Mediator model: $M_i = \alpha_0 + \alpha_1 D_i + \alpha_2' X_i + \epsilon_{Mi}$
- Outcome model: $Y_i = \beta_0 + \beta_1 D_i + \beta_2 M_i + \beta_3' X_i + \epsilon_{Yi}$
- Sensitivity parameter: $\rho = Corr(\epsilon_M, \epsilon_Y)$

**Under sequential ignorability**: $\rho = 0$ (errors are independent conditional on D, X)

### Sensitivity Formula

The ACME under sensitivity parameter $\rho$:

$$
ACME(\rho) = ACME(0) - \rho \cdot \sigma_{\epsilon_M} \cdot \sigma_{\epsilon_Y | D, M, X}
$$

Where:
- $ACME(0)$ is the estimated ACME assuming no confounding
- $\sigma_{\epsilon_M}$ is the standard deviation of mediator model residuals
- $\sigma_{\epsilon_Y | D, M, X}$ is the residual SD from outcome model

### Finding the Breakpoint

The **breakpoint** $\rho^*$ is the value where ACME = 0:

$$
\rho^* = \frac{ACME(0)}{\sigma_{\epsilon_M} \cdot \sigma_{\epsilon_Y | D, M, X}}
$$

### Implementation

```python
import numpy as np
from scipy import stats

def sensitivity_analysis_imai(
    acme: float,
    acme_se: float,
    sigma_m: float,
    sigma_y_resid: float,
    rho_range: np.ndarray = None,
    alpha: float = 0.05
) -> dict:
    """
    Perform Imai et al. (2010) sensitivity analysis for ACME.

    Parameters
    ----------
    acme : float
        Estimated ACME assuming no confounding
    acme_se : float
        Standard error of ACME
    sigma_m : float
        Standard deviation of mediator model residuals
    sigma_y_resid : float
        Standard deviation of outcome model residuals (conditional on D, M, X)
    rho_range : np.ndarray, optional
        Range of sensitivity parameter values. Default: [-0.9, 0.9]
    alpha : float
        Significance level for confidence intervals

    Returns
    -------
    dict with:
        - rho_values: Sensitivity parameter values
        - acme_values: ACME under each rho
        - ci_lower, ci_upper: Confidence intervals
        - breakpoint: rho where ACME = 0
        - robustness_value: R^2 interpretation
        - interpretation: Text description
    """
    if rho_range is None:
        rho_range = np.linspace(-0.9, 0.9, 37)

    # Calculate ACME under each rho
    sigma_product = sigma_m * sigma_y_resid
    acme_values = acme - rho_range * sigma_product

    # Calculate confidence intervals
    z_crit = stats.norm.ppf(1 - alpha / 2)
    ci_lower = acme_values - z_crit * acme_se
    ci_upper = acme_values + z_crit * acme_se

    # Find breakpoint
    if sigma_product > 0:
        breakpoint = acme / sigma_product
    else:
        breakpoint = np.nan

    # R^2 interpretation (squared correlation)
    r2_breakpoint = breakpoint ** 2 if not np.isnan(breakpoint) else np.nan

    # Generate interpretation
    if np.isnan(breakpoint):
        interpretation = "Cannot compute breakpoint (zero variance)"
    elif abs(breakpoint) > 0.9:
        interpretation = (
            f"Very robust: breakpoint |rho| = {abs(breakpoint):.2f} > 0.9. "
            "Near-perfect unmeasured confounding required to nullify effect."
        )
    elif abs(breakpoint) > 0.5:
        interpretation = (
            f"Robust: breakpoint |rho| = {abs(breakpoint):.2f}. "
            "Strong unmeasured confounding required to nullify effect."
        )
    elif abs(breakpoint) > 0.3:
        interpretation = (
            f"Moderately robust: breakpoint |rho| = {abs(breakpoint):.2f}. "
            "Moderate confounding could nullify effect."
        )
    elif abs(breakpoint) > 0.1:
        interpretation = (
            f"Sensitive: breakpoint |rho| = {abs(breakpoint):.2f}. "
            "Modest confounding could nullify effect."
        )
    else:
        interpretation = (
            f"Very sensitive: breakpoint |rho| = {abs(breakpoint):.2f}. "
            "Even weak unmeasured confounding could nullify effect."
        )

    return {
        'rho_values': rho_range,
        'acme_values': acme_values,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'breakpoint': breakpoint,
        'r2_breakpoint': r2_breakpoint,
        'sigma_m': sigma_m,
        'sigma_y_resid': sigma_y_resid,
        'interpretation': interpretation
    }
```

### Interpreting the Breakpoint

| Breakpoint |rho|| Interpretation | Confidence Level |
|------------|----------------|------------------|
| > 0.9 | Near-perfect confounding needed | Very High |
| 0.5 - 0.9 | Strong confounding needed | High |
| 0.3 - 0.5 | Moderate confounding could eliminate | Medium |
| 0.1 - 0.3 | Modest confounding is problematic | Low |
| < 0.1 | Very sensitive to any confounding | Very Low |

### R-squared Interpretation

Convert $\rho$ to $R^2$ for intuition:

$$
R^2 = \rho^2
$$

**Example**: If $\rho^* = 0.5$, then $R^2 = 0.25$ means an unmeasured confounder explaining 25% of residual variance in both M and Y could eliminate the effect.

---

## Confounding Function Approach

### VanderWeele's Bias Formula

For binary treatment and mediator:

$$
Bias_{ACME} = \delta \cdot (\gamma_{M|D=1} - \gamma_{M|D=0})
$$

Where:
- $\delta$ = effect of unmeasured confounder U on Y
- $\gamma_{M|D=d}$ = difference in U between M=1 and M=0 within treatment group d

### Implementation

```python
def bias_formula_acme(
    delta_u_on_y: float,
    gamma_u_m_d1: float,
    gamma_u_m_d0: float
) -> float:
    """
    Calculate bias in ACME from unmeasured confounding.

    Parameters
    ----------
    delta_u_on_y : float
        Effect of unmeasured confounder U on outcome Y
    gamma_u_m_d1 : float
        Difference in U between M=1 and M=0 when D=1
    gamma_u_m_d0 : float
        Difference in U between M=1 and M=0 when D=0

    Returns
    -------
    float
        Bias in ACME estimate
    """
    return delta_u_on_y * (gamma_u_m_d1 - gamma_u_m_d0)
```

---

## Treatment Effect on Mediator Tests

### First-Stage Test

Check if treatment actually affects the mediator:

```python
def test_first_stage(data, treatment, mediator, controls):
    """
    Test if treatment significantly affects mediator.

    A weak first stage (small alpha) means:
    - Small indirect effect by construction
    - Imprecise ACME estimates
    """
    import statsmodels.api as sm

    X = data[[treatment] + controls]
    X = sm.add_constant(X)
    y = data[mediator]

    model = sm.OLS(y, X).fit(cov_type='HC3')

    alpha = model.params[treatment]
    alpha_se = model.bse[treatment]
    alpha_pval = model.pvalues[treatment]
    f_stat = (alpha / alpha_se) ** 2

    result = {
        'alpha': alpha,
        'alpha_se': alpha_se,
        'alpha_pval': alpha_pval,
        'f_statistic': f_stat,
        'first_stage_strong': f_stat > 10,
        'interpretation': ''
    }

    if f_stat < 4:
        result['interpretation'] = (
            "Very weak first stage (F < 4). Treatment has minimal effect on mediator. "
            "ACME will be near zero and poorly estimated."
        )
    elif f_stat < 10:
        result['interpretation'] = (
            f"Weak first stage (F = {f_stat:.1f}). Consider whether mediation "
            "is the appropriate framework."
        )
    else:
        result['interpretation'] = (
            f"Strong first stage (F = {f_stat:.1f}). Treatment has meaningful "
            f"effect on mediator (alpha = {alpha:.4f})."
        )

    return result
```

---

## Placebo Tests

### Pre-Treatment Outcome Test

If treatment affects pre-treatment outcomes through the "mediator," assumptions are violated:

```python
def placebo_pretreatment(data, pre_outcome, treatment, mediator, controls):
    """
    Test mediation on a pre-treatment outcome.

    ACME should be ~0 since treatment cannot affect pre-treatment outcomes.
    Significant ACME suggests confounding.
    """
    from mediation_estimator import estimate_baron_kenny

    result = estimate_baron_kenny(
        data=data,
        outcome=pre_outcome,
        treatment=treatment,
        mediator=mediator,
        controls=controls
    )

    placebo_acme = result['acme']
    placebo_se = result['acme_se']
    placebo_pval = result['acme_pvalue']

    return {
        'placebo_acme': placebo_acme,
        'placebo_se': placebo_se,
        'placebo_pval': placebo_pval,
        'passes': placebo_pval > 0.05,
        'interpretation': (
            "PASSED: No significant mediation effect on pre-treatment outcome."
            if placebo_pval > 0.05 else
            f"FAILED: Significant placebo ACME = {placebo_acme:.4f} (p = {placebo_pval:.4f}). "
            "Suggests confounding between mediator and outcomes."
        )
    }
```

### Placebo Mediator Test

Use a variable that SHOULD NOT mediate the effect:

```python
def placebo_mediator(data, outcome, treatment, placebo_mediator, controls):
    """
    Test mediation with a theoretically unrelated mediator.

    ACME should be ~0 if the placebo variable doesn't mediate.
    """
    from mediation_estimator import estimate_baron_kenny

    result = estimate_baron_kenny(
        data=data,
        outcome=outcome,
        treatment=treatment,
        mediator=placebo_mediator,
        controls=controls
    )

    return {
        'placebo_acme': result['acme'],
        'placebo_se': result['acme_se'],
        'placebo_pval': result['acme_pvalue'],
        'passes': result['acme_pvalue'] > 0.05
    }
```

---

## Model Specification Tests

### Linearity Test

Check if mediator-outcome relationship is linear:

```python
def test_linearity(data, outcome, treatment, mediator, controls):
    """
    Test linearity of mediator-outcome relationship.

    Adds polynomial terms and tests joint significance.
    """
    import statsmodels.api as sm

    # Linear model
    X_lin = data[[treatment, mediator] + controls]
    X_lin = sm.add_constant(X_lin)
    model_lin = sm.OLS(data[outcome], X_lin).fit()

    # Add squared mediator
    data_test = data.copy()
    data_test['m_sq'] = data[mediator] ** 2
    X_quad = data_test[[treatment, mediator, 'm_sq'] + controls]
    X_quad = sm.add_constant(X_quad)
    model_quad = sm.OLS(data[outcome], X_quad).fit()

    # LR test
    lr_stat = 2 * (model_quad.llf - model_lin.llf)
    lr_pval = 1 - stats.chi2.cdf(lr_stat, 1)

    return {
        'linear_r2': model_lin.rsquared,
        'quadratic_r2': model_quad.rsquared,
        'lr_statistic': lr_stat,
        'lr_pval': lr_pval,
        'nonlinearity_detected': lr_pval < 0.05,
        'recommendation': (
            "Consider flexible (ML-based) outcome model"
            if lr_pval < 0.05 else
            "Linear specification appears adequate"
        )
    }
```

### Interaction Test

Test for treatment-mediator interaction:

```python
def test_interaction(data, outcome, treatment, mediator, controls):
    """
    Test for treatment x mediator interaction.

    Significant interaction means ADE/ACME depend on treatment level.
    """
    import statsmodels.api as sm

    # Without interaction
    X_no_int = data[[treatment, mediator] + controls]
    X_no_int = sm.add_constant(X_no_int)
    model_no_int = sm.OLS(data[outcome], X_no_int).fit()

    # With interaction
    data_test = data.copy()
    data_test['d_x_m'] = data[treatment] * data[mediator]
    X_int = data_test[[treatment, mediator, 'd_x_m'] + controls]
    X_int = sm.add_constant(X_int)
    model_int = sm.OLS(data[outcome], X_int).fit()

    int_coef = model_int.params['d_x_m']
    int_pval = model_int.pvalues['d_x_m']

    return {
        'interaction_coef': int_coef,
        'interaction_pval': int_pval,
        'interaction_present': int_pval < 0.05,
        'interpretation': (
            f"Significant interaction (coef = {int_coef:.4f}, p = {int_pval:.4f}). "
            "ADE and ACME differ between treatment groups. "
            "Report ADE(0), ADE(1), ACME(0), ACME(1) separately."
            if int_pval < 0.05 else
            "No significant interaction. Pooled ADE/ACME appropriate."
        )
    }
```

---

## Balance and Overlap Diagnostics

### Covariate Balance by Treatment

```python
def balance_table(data, treatment, controls):
    """
    Generate covariate balance table.
    """
    from scipy import stats

    rows = []
    for var in controls:
        treated = data.loc[data[treatment] == 1, var]
        control = data.loc[data[treatment] == 0, var]

        stat, pval = stats.ttest_ind(treated, control)
        std_diff = (treated.mean() - control.mean()) / data[var].std()

        rows.append({
            'Variable': var,
            'Control Mean': control.mean(),
            'Treated Mean': treated.mean(),
            'Std. Diff': std_diff,
            'p-value': pval
        })

    return pd.DataFrame(rows)
```

### Mediator Overlap by Treatment

```python
def mediator_overlap_diagnostic(data, treatment, mediator):
    """
    Assess overlap in mediator distributions by treatment.
    """
    import matplotlib.pyplot as plt

    m_treated = data.loc[data[treatment] == 1, mediator]
    m_control = data.loc[data[treatment] == 0, mediator]

    overlap_min = max(m_treated.min(), m_control.min())
    overlap_max = min(m_treated.max(), m_control.max())

    # Proportion in overlap region
    prop_in_overlap = (
        ((data[mediator] >= overlap_min) & (data[mediator] <= overlap_max)).mean()
    )

    # KS test
    ks_stat, ks_pval = stats.ks_2samp(m_treated, m_control)

    return {
        'overlap_range': (overlap_min, overlap_max),
        'prop_in_overlap': prop_in_overlap,
        'ks_statistic': ks_stat,
        'ks_pval': ks_pval,
        'good_overlap': prop_in_overlap > 0.8 and ks_stat < 0.3,
        'interpretation': (
            "Good overlap in mediator distributions."
            if prop_in_overlap > 0.8 else
            f"Poor overlap: only {prop_in_overlap:.1%} of observations in overlap region."
        )
    }
```

---

## Comprehensive Diagnostic Report

```python
def full_mediation_diagnostics(
    data, outcome, treatment, mediator, controls,
    acme, acme_se, sigma_m, sigma_y_resid
):
    """
    Run all diagnostic tests and generate report.
    """
    diagnostics = {
        'sensitivity': sensitivity_analysis_imai(
            acme, acme_se, sigma_m, sigma_y_resid
        ),
        'first_stage': test_first_stage(data, treatment, mediator, controls),
        'linearity': test_linearity(data, outcome, treatment, mediator, controls),
        'interaction': test_interaction(data, outcome, treatment, mediator, controls),
        'balance': balance_table(data, treatment, controls),
        'overlap': mediator_overlap_diagnostic(data, treatment, mediator)
    }

    # Summary
    issues = []
    if abs(diagnostics['sensitivity']['breakpoint']) < 0.3:
        issues.append("Sensitive to unmeasured confounding")
    if not diagnostics['first_stage']['first_stage_strong']:
        issues.append("Weak first stage")
    if diagnostics['linearity']['nonlinearity_detected']:
        issues.append("Nonlinear mediator-outcome relationship")
    if not diagnostics['overlap']['good_overlap']:
        issues.append("Poor mediator overlap")

    diagnostics['summary'] = {
        'n_issues': len(issues),
        'issues': issues,
        'overall_quality': (
            'High' if len(issues) == 0 else
            'Medium' if len(issues) == 1 else
            'Low'
        )
    }

    return diagnostics
```

---

## References

- Imai, K., Keele, L., & Tingley, D. (2010). A General Approach to Causal Mediation Analysis. *Psychological Methods*.
- Imai, K., Keele, L., & Yamamoto, T. (2010). Identification, Inference and Sensitivity Analysis for Causal Mediation Effects. *Statistical Science*.
- VanderWeele, T. J. (2010). Bias Formulas for Sensitivity Analysis for Direct and Indirect Effects. *Epidemiology*.
- VanderWeele, T. J. (2015). *Explanation in Causal Inference*. Oxford University Press.
