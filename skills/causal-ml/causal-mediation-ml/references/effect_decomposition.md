# Effect Decomposition in Causal Mediation Analysis

> **Reference Document** | Part of `causal-mediation-ml` skill
> **Version**: 1.0.0

## Overview

Causal mediation analysis decomposes the **Total Effect** of treatment on outcome into:
1. **Direct Effect (ADE)** - Effect NOT through the mediator
2. **Indirect Effect (ACME)** - Effect THROUGH the mediator

This document provides formal definitions, estimation formulas, and interpretation guidance.

---

## Potential Outcomes Framework

### Notation

| Symbol | Definition |
|--------|------------|
| $D_i$ | Treatment (observed) |
| $M_i$ | Mediator (observed) |
| $Y_i$ | Outcome (observed) |
| $M_i(d)$ | Potential mediator under treatment $d$ |
| $Y_i(d, m)$ | Potential outcome under treatment $d$ and mediator $m$ |
| $Y_i(d, M_i(d'))$ | Potential outcome under treatment $d$ with mediator at its value under $d'$ |

### Observed Values

Under consistency assumption:
$$
M_i = M_i(D_i)
$$
$$
Y_i = Y_i(D_i, M_i(D_i))
$$

---

## Effect Definitions

### Average Direct Effect (ADE)

$$
ADE(d) = E[Y_i(1, M_i(d)) - Y_i(0, M_i(d))]
$$

**Interpretation**: The expected change in outcome when treatment changes from 0 to 1, **holding the mediator fixed** at what it would have been under treatment level $d$.

**Two versions**:
- $ADE(0)$: Direct effect holding mediator at control level
- $ADE(1)$: Direct effect holding mediator at treated level

### Average Causal Mediation Effect (ACME)

$$
ACME(d) = E[Y_i(d, M_i(1)) - Y_i(d, M_i(0))]
$$

**Interpretation**: The expected change in outcome when the mediator changes from its control value to its treated value, **holding treatment fixed** at level $d$.

**Two versions**:
- $ACME(0)$: Indirect effect with treatment fixed at 0
- $ACME(1)$: Indirect effect with treatment fixed at 1

### Total Effect

$$
TE = E[Y_i(1, M_i(1)) - Y_i(0, M_i(0))]
$$

**Decomposition** (under no interaction):
$$
TE = ADE(d) + ACME(1-d) = ADE(0) + ACME(1) = ADE(1) + ACME(0)
$$

---

## Proportion Mediated

### Definition

$$
\pi_{med} = \frac{ACME}{TE}
$$

**Interpretation**: The fraction of the total effect that operates through the mediator.

### Implementation

```python
def proportion_mediated(
    acme: float,
    total_effect: float,
    acme_se: float = None,
    total_se: float = None
) -> dict:
    """
    Calculate proportion mediated with uncertainty.

    Parameters
    ----------
    acme : float
        Average Causal Mediation Effect (indirect effect)
    total_effect : float
        Total effect of treatment on outcome
    acme_se : float, optional
        Standard error of ACME
    total_se : float, optional
        Standard error of total effect

    Returns
    -------
    dict with:
        - proportion: Point estimate
        - se: Standard error (if provided)
        - ci: 95% confidence interval (if SE provided)
        - interpretation: Text description
    """
    import numpy as np
    from scipy import stats

    # Handle edge cases
    if abs(total_effect) < 1e-10:
        return {
            'proportion': np.nan,
            'interpretation': 'Undefined: Total effect is approximately zero'
        }

    proportion = acme / total_effect

    # Standard error via delta method
    if acme_se is not None and total_se is not None:
        # Var(ACME/TE) approx= (1/TE)^2 * Var(ACME) + (ACME/TE^2)^2 * Var(TE)
        var_prop = (1/total_effect)**2 * acme_se**2 + \
                   (acme/total_effect**2)**2 * total_se**2
        se_prop = np.sqrt(var_prop)
        ci = (proportion - 1.96 * se_prop, proportion + 1.96 * se_prop)
    else:
        se_prop = None
        ci = None

    # Interpretation
    if proportion < 0:
        interpretation = (
            f"Negative proportion ({proportion:.1%}): "
            "Suppression effect - mediator suppresses direct effect. "
            "Direct and indirect effects have opposite signs."
        )
    elif proportion > 1:
        interpretation = (
            f"Proportion > 100% ({proportion:.1%}): "
            "Inconsistent mediation - direct and indirect effects have opposite signs. "
            "The indirect effect is larger than total effect."
        )
    elif proportion > 0.8:
        interpretation = (
            f"Strong mediation ({proportion:.1%}): "
            "Most of the total effect operates through the mediator."
        )
    elif proportion > 0.5:
        interpretation = (
            f"Substantial mediation ({proportion:.1%}): "
            "More than half of the effect operates through the mediator."
        )
    elif proportion > 0.2:
        interpretation = (
            f"Moderate mediation ({proportion:.1%}): "
            "A meaningful portion of the effect operates through the mediator."
        )
    else:
        interpretation = (
            f"Weak mediation ({proportion:.1%}): "
            "Most of the effect is direct, not through the mediator."
        )

    return {
        'proportion': proportion,
        'proportion_pct': proportion * 100,
        'se': se_prop,
        'ci': ci,
        'interpretation': interpretation
    }
```

### Special Cases

| Proportion | Meaning | Implication |
|------------|---------|-------------|
| 0 - 20% | Weak mediation | Mediator not main pathway |
| 20 - 50% | Moderate mediation | Both pathways matter |
| 50 - 80% | Substantial mediation | Mediator is primary pathway |
| 80 - 100% | Strong mediation | Nearly full mediation |
| > 100% | Inconsistent | Opposite-sign effects |
| < 0% | Suppression | Mediator reduces effect |

---

## Linear Model Formulas

Under linear models:

**Mediator Model**:
$$
M_i = \alpha_0 + \alpha_1 D_i + \alpha_2' X_i + \epsilon_{Mi}
$$

**Outcome Model**:
$$
Y_i = \beta_0 + \beta_1 D_i + \beta_2 M_i + \beta_3' X_i + \epsilon_{Yi}
$$

### Closed-Form Effects

| Effect | Formula | Description |
|--------|---------|-------------|
| ACME | $\alpha_1 \cdot \beta_2$ | Product of D->M and M->Y coefficients |
| ADE | $\beta_1$ | Direct effect coefficient |
| Total | $\beta_1 + \alpha_1 \cdot \beta_2$ | Sum of direct and indirect |
| Proportion | $\frac{\alpha_1 \cdot \beta_2}{\beta_1 + \alpha_1 \cdot \beta_2}$ | Indirect / Total |

### Implementation

```python
def decompose_effects_linear(
    alpha_1: float,
    beta_1: float,
    beta_2: float,
    se_alpha: float = None,
    se_beta_1: float = None,
    se_beta_2: float = None
) -> dict:
    """
    Decompose effects under linear model assumptions.

    Parameters
    ----------
    alpha_1 : float
        Coefficient of D in mediator model (D -> M)
    beta_1 : float
        Coefficient of D in outcome model (direct effect)
    beta_2 : float
        Coefficient of M in outcome model (M -> Y | D)
    se_* : float, optional
        Standard errors for inference

    Returns
    -------
    dict with all effects and standard errors
    """
    import numpy as np

    # Point estimates
    acme = alpha_1 * beta_2
    ade = beta_1
    total = ade + acme

    # Standard errors (Sobel/delta method)
    if all(se is not None for se in [se_alpha, se_beta_1, se_beta_2]):
        # Var(alpha*beta) = alpha^2*Var(beta) + beta^2*Var(alpha)
        var_acme = alpha_1**2 * se_beta_2**2 + beta_2**2 * se_alpha**2
        se_acme = np.sqrt(var_acme)

        # Var(total) = Var(ade) + Var(acme) + 2*Cov(ade, acme)
        # Assuming independence: Cov = 0
        se_total = np.sqrt(se_beta_1**2 + var_acme)
    else:
        se_acme = None
        se_total = None

    # Proportion
    if abs(total) > 1e-10:
        prop = acme / total
    else:
        prop = np.nan

    return {
        'acme': acme,
        'se_acme': se_acme,
        'ade': ade,
        'se_ade': se_beta_1,
        'total': total,
        'se_total': se_total,
        'proportion_mediated': prop,
        'coefficients': {
            'alpha_1': alpha_1,
            'beta_1': beta_1,
            'beta_2': beta_2
        }
    }
```

---

## Nonlinear / Heterogeneous Effects

### Treatment-Mediator Interaction

When effects differ by treatment status:

**Outcome Model with Interaction**:
$$
Y_i = \beta_0 + \beta_1 D_i + \beta_2 M_i + \beta_3 (D_i \times M_i) + \beta_4' X_i + \epsilon_{Yi}
$$

**Effect Modifications**:
- $ADE(0) \neq ADE(1)$
- $ACME(0) \neq ACME(1)$

```python
def decompose_with_interaction(
    alpha_1: float,
    beta_1: float,
    beta_2: float,
    beta_3: float,  # Interaction coefficient
    mean_m: float   # Mean of mediator
) -> dict:
    """
    Effect decomposition with treatment-mediator interaction.
    """
    # ACME varies by treatment
    # ACME(1) = alpha_1 * (beta_2 + beta_3)
    # ACME(0) = alpha_1 * beta_2
    acme_d1 = alpha_1 * (beta_2 + beta_3)
    acme_d0 = alpha_1 * beta_2

    # ADE varies by mediator level
    # Evaluated at mean of M:
    # ADE = beta_1 + beta_3 * M
    ade_at_mean = beta_1 + beta_3 * mean_m

    # Total effect
    # TE = ADE(0) + ACME(1) = ADE(1) + ACME(0)
    total = beta_1 + alpha_1 * beta_2 + alpha_1 * beta_3

    return {
        'acme_d0': acme_d0,
        'acme_d1': acme_d1,
        'ade_at_mean_m': ade_at_mean,
        'total': total,
        'interaction_coef': beta_3,
        'has_interaction': abs(beta_3) > 0.01
    }
```

### Heterogeneous Effects by Subgroup

```python
def decompose_by_subgroup(
    data: pd.DataFrame,
    outcome: str,
    treatment: str,
    mediator: str,
    controls: list,
    subgroup_var: str
) -> pd.DataFrame:
    """
    Estimate mediation effects separately by subgroup.
    """
    from mediation_estimator import estimate_baron_kenny

    results = []
    for group_val in data[subgroup_var].unique():
        subset = data[data[subgroup_var] == group_val]

        if len(subset) < 50:
            continue

        result = estimate_baron_kenny(
            subset, outcome, treatment, mediator, controls
        )

        results.append({
            'subgroup': group_val,
            'n': len(subset),
            'acme': result['acme'],
            'acme_se': result['acme_se'],
            'ade': result['ade'],
            'ade_se': result['ade_se'],
            'total': result['total_effect'],
            'prop_mediated': result['prop_mediated']
        })

    return pd.DataFrame(results)
```

---

## Decomposition Summary Table

### Standard Format

```python
def create_decomposition_table(results: dict) -> str:
    """
    Create publication-ready decomposition table.
    """
    from scipy import stats

    def stars(pval):
        if pval < 0.01:
            return '***'
        elif pval < 0.05:
            return '**'
        elif pval < 0.1:
            return '*'
        return ''

    def format_ci(lower, upper):
        return f"[{lower:.4f}, {upper:.4f}]"

    lines = [
        "=" * 70,
        "EFFECT DECOMPOSITION".center(70),
        "=" * 70,
        "",
        f"{'Effect':<25} {'Estimate':>12} {'Std.Err':>10} {'95% CI':>22}",
        "-" * 70,
    ]

    # Total Effect
    total = results.get('total_effect', results.get('total'))
    total_se = results.get('total_se', results.get('se_total', np.nan))
    total_ci = format_ci(total - 1.96*total_se, total + 1.96*total_se)
    z_total = total / total_se if total_se else 0
    p_total = 2 * (1 - stats.norm.cdf(abs(z_total)))
    lines.append(
        f"{'Total Effect':<25} {total:>12.4f}{stars(p_total):<4} ({total_se:>7.4f}) {total_ci}"
    )

    # Direct Effect
    ade = results.get('ade')
    ade_se = results.get('ade_se', results.get('se_ade'))
    ade_ci = format_ci(
        results.get('ade_ci_lower', ade - 1.96*ade_se),
        results.get('ade_ci_upper', ade + 1.96*ade_se)
    )
    z_ade = ade / ade_se if ade_se else 0
    p_ade = 2 * (1 - stats.norm.cdf(abs(z_ade)))
    lines.append(
        f"{'Direct Effect (ADE)':<25} {ade:>12.4f}{stars(p_ade):<4} ({ade_se:>7.4f}) {ade_ci}"
    )

    # Indirect Effect
    acme = results.get('acme')
    acme_se = results.get('acme_se', results.get('se_acme'))
    acme_ci = format_ci(
        results.get('acme_ci_lower', acme - 1.96*acme_se),
        results.get('acme_ci_upper', acme + 1.96*acme_se)
    )
    z_acme = acme / acme_se if acme_se else 0
    p_acme = 2 * (1 - stats.norm.cdf(abs(z_acme)))
    lines.append(
        f"{'Indirect Effect (ACME)':<25} {acme:>12.4f}{stars(p_acme):<4} ({acme_se:>7.4f}) {acme_ci}"
    )

    # Proportion
    prop = results.get('prop_mediated', results.get('proportion_mediated'))
    lines.extend([
        "-" * 70,
        f"Proportion Mediated: {prop*100:.1f}%",
        "",
        "=" * 70,
        "Notes: *** p<0.01, ** p<0.05, * p<0.1",
    ])

    return "\n".join(lines)
```

### Example Output

```
======================================================================
                         EFFECT DECOMPOSITION
======================================================================

Effect                       Estimate     Std.Err                95% CI
----------------------------------------------------------------------
Total Effect                   0.1500***  ( 0.0350) [0.0814, 0.2186]
Direct Effect (ADE)            0.0900***  ( 0.0280) [0.0351, 0.1449]
Indirect Effect (ACME)         0.0600***  ( 0.0180) [0.0247, 0.0953]
----------------------------------------------------------------------
Proportion Mediated: 40.0%

======================================================================
Notes: *** p<0.01, ** p<0.05, * p<0.1
```

---

## Interpretation Guidelines

### Reading the Decomposition

1. **Total Effect**: Overall impact of treatment
   - If insignificant, mediation analysis may not be meaningful
   - Proceed with caution

2. **Direct Effect (ADE)**: What remains after accounting for mediator
   - Captures all non-mediated pathways
   - May include other unmeasured mediators

3. **Indirect Effect (ACME)**: The mediation effect
   - Primary quantity of interest
   - Requires strongest assumptions

4. **Proportion Mediated**: Summary statistic
   - Useful for communication
   - Can be unstable; report with caution

### Reporting Recommendations

**Do Report**:
- Point estimates with standard errors/CIs
- Significance levels
- Proportion mediated (with caveats)
- Sensitivity analysis results

**Do NOT Report**:
- Proportion mediated without uncertainty
- Effects without discussing assumptions
- "Full mediation" claims without testing

---

## References

- Imai, K., Keele, L., & Tingley, D. (2010). A General Approach to Causal Mediation Analysis. *Psychological Methods*.
- Pearl, J. (2001). Direct and Indirect Effects. *UAI*.
- VanderWeele, T. J. (2015). *Explanation in Causal Inference*. Oxford University Press.
- MacKinnon, D. P. (2008). *Introduction to Statistical Mediation Analysis*. Erlbaum.
