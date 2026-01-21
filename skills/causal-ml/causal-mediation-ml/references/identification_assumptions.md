# Identification Assumptions for Causal Mediation Analysis

> **Reference Document** | Part of `causal-mediation-ml` skill
> **Version**: 1.0.0

## Overview

Causal mediation analysis requires strong assumptions to identify the Average Causal Mediation Effect (ACME) and Average Direct Effect (ADE). Unlike standard treatment effect estimation, mediation involves identifying **two causal relationships** simultaneously: Treatment -> Mediator and Mediator -> Outcome.

---

## Sequential Ignorability (Imai et al., 2010)

Sequential ignorability is the **fundamental identification assumption** for causal mediation analysis. It consists of two parts.

### Part 1: Treatment Ignorability

$$
\{Y_i(d', m), M_i(d)\} \perp\!\!\!\perp D_i | X_i = x
$$

**Interpretation**: Conditional on observed covariates X, treatment assignment D is independent of:
- All potential outcomes Y(d', m) under any treatment d' and mediator m
- All potential mediator values M(d) under any treatment d

**Plain Language**: After controlling for X, treatment is "as good as random" - there are no unmeasured confounders of the treatment-outcome or treatment-mediator relationships.

**Satisfied By**:
- Randomized experiments (for treatment)
- Natural experiments with exogenous treatment variation
- Observational studies with rich controls (under stronger assumptions)

### Part 2: Mediator Ignorability

$$
Y_i(d', m) \perp\!\!\!\perp M_i(d) | D_i = d, X_i = x
$$

**Interpretation**: Conditional on treatment D and observed covariates X, the mediator is independent of potential outcomes.

**Plain Language**: After controlling for treatment and X, the mediator value is "as good as random" - there are no unmeasured confounders of the mediator-outcome relationship.

**This is the STRONGER assumption** because:
1. Even in randomized experiments, the mediator is typically NOT randomized
2. Post-treatment confounders may affect both M and Y
3. Unobserved heterogeneity in response to treatment can violate this

---

## Cross-World Independence

### The Cross-World Counterfactual Problem

Mediation effects involve **cross-world counterfactuals** - outcomes that could never be observed in the same world:

$$
ACME(d) = E[Y_i(d, M_i(1)) - Y_i(d, M_i(0))]
$$

This compares:
- $Y_i(d, M_i(1))$: Outcome under treatment d with mediator at its value under treatment 1
- $Y_i(d, M_i(0))$: Outcome under treatment d with mediator at its value under treatment 0

**Problem**: We can never observe both $M_i(1)$ and $M_i(0)$ for the same unit.

### Cross-World Independence Assumption

The assumption implicitly embedded in sequential ignorability:

$$
Y_i(d, m) \perp\!\!\!\perp M_i(d') | X_i \quad \text{for } d \neq d'
$$

**Interpretation**: Potential outcomes under one treatment condition are independent of potential mediator values under a different treatment condition.

**Violation Example**:
- Individual's unobserved motivation affects both:
  - Their potential skill gain from training (M(1))
  - Their potential earnings even without skill gain (Y(1, m))
- This creates dependence between M(1) and Y(1, m) that violates the assumption

---

## No Interaction Assumption

### Treatment-Mediator Interaction on Unmeasured Confounders

$$
\text{No unmeasured confounder of } M \rightarrow Y \text{ is affected by treatment } D
$$

### Formal Statement

Let U be unmeasured confounders of the M->Y relationship:

$$
U_i(d) = U_i(d') \quad \forall d, d'
$$

**Interpretation**: Treatment does not affect the unmeasured confounders of the mediator-outcome relationship.

### Why This Matters

If treatment affects unmeasured confounders that also affect the mediator-outcome relationship, we cannot separate:
- True mediation effect (D -> M -> Y)
- Spurious correlation induced by post-treatment confounding (D -> U -> M and U -> Y)

### DAG Representation

**Valid Structure** (No Interaction):
```
X ──────────────────────┐
│                       │
↓                       ↓
D ───────→ M ──────→ Y
│                   ↑
└───────────────────┘
    (direct effect)
```

**Invalid Structure** (Interaction via U):
```
X ──────────────────────┐
│                       │
↓                       ↓
D ───────→ M ──────→ Y
│          ↑          ↑
│          │          │
└────→ U ─────────────┘
    (post-treatment confounder)
```

---

## Positivity Assumption

### Statement

$$
0 < P(D = d | X = x) < 1 \quad \text{and} \quad 0 < P(M = m | D = d, X = x) < 1
$$

for all values of d, m, and x in the support.

### Interpretation

1. **Treatment Positivity**: Every unit with covariates X has positive probability of receiving any treatment value
2. **Mediator Positivity**: Every unit with covariates X and treatment D has positive probability of any mediator value

### Violations

**Structural Violations**:
- Perfect predictors of treatment (deterministic assignment)
- Mediator values impossible for certain treatment groups

**Practical Violations**:
- Very low probability (near-positivity violations)
- Sparse covariate strata with no variation

### Checking Positivity

```python
def check_positivity(data, treatment, mediator, controls):
    """
    Check positivity assumption for mediation analysis.

    Returns
    -------
    Dict with:
    - treatment_overlap: Overlap statistics
    - mediator_overlap: Mediator overlap by treatment
    - warnings: List of potential violations
    """
    warnings = []

    # Treatment positivity
    prop_treated = data[treatment].mean()
    if prop_treated < 0.05 or prop_treated > 0.95:
        warnings.append(
            f"Extreme treatment prevalence: {prop_treated:.1%}"
        )

    # Mediator overlap by treatment
    m_treated = data.loc[data[treatment] == 1, mediator]
    m_control = data.loc[data[treatment] == 0, mediator]

    # Check overlap
    overlap_min = max(m_treated.min(), m_control.min())
    overlap_max = min(m_treated.max(), m_control.max())

    if overlap_max < overlap_min:
        warnings.append(
            "No overlap in mediator distributions between treatment groups!"
        )

    return {
        'treatment_prop': prop_treated,
        'mediator_overlap_range': (overlap_min, overlap_max),
        'warnings': warnings
    }
```

---

## SUTVA (Stable Unit Treatment Value Assumption)

### No Interference

$$
Y_i(d_1, ..., d_N, m_1, ..., m_N) = Y_i(d_i, m_i)
$$

**Interpretation**: Unit i's outcome depends only on their own treatment and mediator, not others'.

### No Hidden Versions

$$
D_i = d \implies M_i(d) = M_i(d')  \text{ for all versions } d, d' \text{ of treatment } d
$$

**Interpretation**: There is only one "version" of each treatment level.

### Violations in Mediation Context

1. **Social Spillovers**: Others' treatment affects my mediator
2. **General Equilibrium**: Market-level effects change mediator-outcome relationship
3. **Multiple Treatment Components**: Training program has multiple "versions"

---

## Testing Assumptions

### What Can Be Tested

| Assumption | Testable? | Available Test |
|------------|-----------|----------------|
| Treatment Ignorability (Part 1) | Partially | Balance tests, placebo outcomes |
| Mediator Ignorability (Part 2) | No | Sensitivity analysis only |
| Cross-World Independence | No | Sensitivity analysis only |
| No Interaction | No | Sensitivity analysis only |
| Positivity | Yes | Overlap diagnostics |
| SUTVA | Partially | Design-based arguments |

### Balance Tests for Part 1

```python
def balance_test_mediation(data, treatment, controls):
    """
    Test covariate balance between treatment groups.

    Significant imbalance suggests treatment ignorability may fail.
    """
    from scipy import stats

    results = []
    for var in controls:
        treated = data.loc[data[treatment] == 1, var]
        control = data.loc[data[treatment] == 0, var]

        stat, pval = stats.ttest_ind(treated, control)
        std_diff = (treated.mean() - control.mean()) / data[var].std()

        results.append({
            'variable': var,
            'treated_mean': treated.mean(),
            'control_mean': control.mean(),
            'std_diff': std_diff,
            'p_value': pval
        })

    return pd.DataFrame(results)
```

### Pre-Treatment Placebo Test

```python
def placebo_mediation_test(data, pre_outcome, treatment, mediator, controls):
    """
    Test if treatment affects a pre-treatment outcome.

    If significant, suggests unmeasured confounding.
    """
    # Run mediation analysis on pre-treatment outcome
    # ACME should be ~0 if assumptions hold
    pass
```

---

## Sensitivity to Violations

### When Assumptions Fail

| Assumption Violated | Consequence |
|--------------------|-------------|
| Part 1 (Treatment) | Biased estimates of both ADE and ACME |
| Part 2 (Mediator) | Biased ACME; ADE may still be valid |
| Cross-World | ACME not identified; only descriptive |
| No Interaction | ACME confounded with post-treatment confounding |
| Positivity | Extrapolation bias, unstable estimates |

### Robustness Strategy

1. **Sensitivity Analysis**: Use Imai et al. sensitivity parameter (see `diagnostic_tests.md`)
2. **Bounds Analysis**: Compute bounds under partial identification
3. **Alternative Designs**: Experimental manipulation of mediator
4. **Multiple Mediators**: Model alternative pathways

---

## References

### Core Papers
- Imai, K., Keele, L., & Tingley, D. (2010). A General Approach to Causal Mediation Analysis. *Psychological Methods*, 15(4), 309-334.
- Imai, K., Keele, L., & Yamamoto, T. (2010). Identification, Inference and Sensitivity Analysis for Causal Mediation Effects. *Statistical Science*, 25(1), 51-71.

### Identification Theory
- Pearl, J. (2001). Direct and Indirect Effects. *Proceedings of the Seventeenth Conference on Uncertainty in Artificial Intelligence*.
- Robins, J. M., & Greenland, S. (1992). Identifiability and Exchangeability for Direct and Indirect Effects. *Epidemiology*, 3(2), 143-155.

### Sensitivity Analysis
- VanderWeele, T. J. (2010). Bias Formulas for Sensitivity Analysis for Direct and Indirect Effects. *Epidemiology*, 21(4), 540-551.

### Textbooks
- VanderWeele, T. J. (2015). *Explanation in Causal Inference: Methods for Mediation and Interaction*. Oxford University Press.
