# Extensions to Difference-in-Differences

> **Document Type**: Reference | **Last Updated**: 2025-01
> **Related**: [estimation_methods.md](estimation_methods.md), [identification_assumptions.md](identification_assumptions.md)

## Overview

This document covers advanced DID methods and extensions for handling complex research designs, relaxing assumptions, and addressing specific empirical challenges.

---

## 1. Heterogeneous Treatment Effects

### 1.1 Group-Level Heterogeneity

**Motivation**: Treatment effects may vary systematically across observable groups.

**Approach**: Estimate separate effects by subgroup or interact treatment with group indicators.

```python
def estimate_heterogeneous_effects(
    data: pd.DataFrame,
    outcome: str,
    treatment: str,
    unit_id: str,
    time_id: str,
    heterogeneity_var: str,
    cluster: str = None
) -> Dict[str, CausalOutput]:
    """
    Estimate treatment effects separately by subgroup.
    """
    results = {}
    groups = data[heterogeneity_var].unique()

    for group in groups:
        group_data = data[data[heterogeneity_var] == group]
        result = estimate_did_panel(
            group_data, outcome, treatment, unit_id, time_id,
            cluster=cluster
        )
        results[group] = result

    # Test for heterogeneity
    effects = [r.effect for r in results.values()]
    ses = [r.se for r in results.values()]
    heterogeneity_stat = np.var(effects) / np.mean([s**2 for s in ses])

    return results, heterogeneity_stat

# Usage
results, het_stat = estimate_heterogeneous_effects(
    data=df,
    outcome="y",
    treatment="treated",
    unit_id="id",
    time_id="year",
    heterogeneity_var="firm_size_category"
)

for group, result in results.items():
    print(f"{group}: {result.effect:.4f} (SE = {result.se:.4f})")
```

### 1.2 Continuous Heterogeneity (CATT)

**Motivation**: Treatment effects may vary continuously with some characteristic.

```python
from scripts.robustness_checks import conditional_att

# Estimate CATT(x) for different values of x
catt_results = conditional_att(
    data=df,
    outcome="y",
    treatment="treated",
    unit_id="id",
    time_id="year",
    moderator="firm_size",
    method="kernel",  # or "binned", "linear_interaction"
    n_points=20
)

# Plot CATT function
import matplotlib.pyplot as plt
fig, ax = plt.subplots()
ax.plot(catt_results['x_values'], catt_results['catt_values'])
ax.fill_between(
    catt_results['x_values'],
    catt_results['ci_lower'],
    catt_results['ci_upper'],
    alpha=0.3
)
ax.set_xlabel('Firm Size')
ax.set_ylabel('Conditional ATT')
```

### 1.3 Machine Learning for Heterogeneity

**Approach**: Use causal forests or other ML methods to discover heterogeneity.

```python
from econml.dml import CausalForestDML
import numpy as np

# Prepare data for causal forest
# (assuming treatment is binary and occurs at known time)

# Create did-type features
def prepare_cf_data(df, outcome, treatment, time_id, treatment_time, covariates):
    """Prepare data for causal forest with DID structure."""
    pre = df[df[time_id] < treatment_time].groupby('id')[outcome].mean()
    post = df[df[time_id] >= treatment_time].groupby('id')[outcome].mean()
    treatment_status = df.groupby('id')[treatment].max()
    X = df.groupby('id')[covariates].first()

    return {
        'Y_change': post - pre,
        'treatment': treatment_status,
        'X': X
    }

cf_data = prepare_cf_data(df, 'y', 'treated_ever', 'year', 2015, ['x1', 'x2', 'x3'])

# Fit causal forest
cf = CausalForestDML(
    model_y='auto',
    model_t='auto',
    discrete_treatment=True,
    cv=5
)
cf.fit(
    cf_data['Y_change'],
    cf_data['treatment'],
    X=cf_data['X']
)

# Get heterogeneous effects
individual_effects = cf.effect(cf_data['X'])
print(f"Mean effect: {individual_effects.mean():.4f}")
print(f"Std of effects: {individual_effects.std():.4f}")
```

---

## 2. Triple Difference (DDD)

### 2.1 Concept

Triple difference adds a third comparison dimension to address differential trends.

**Example**: Comparing (treated state, affected industry) to:
- (treated state, unaffected industry)
- (control state, affected industry)
- (control state, unaffected industry)

### 2.2 Model Specification

$$
Y_{ijt} = \alpha_i + \lambda_j + \gamma_t + \beta_1(Treat_i \times Post_t) + \beta_2(Affected_j \times Post_t) + \beta_3(Treat_i \times Affected_j) + \delta(Treat_i \times Affected_j \times Post_t) + \varepsilon_{ijt}
$$

### 2.3 Implementation

```python
def estimate_triple_diff(
    data: pd.DataFrame,
    outcome: str,
    treat_group: str,
    affected_group: str,
    post: str,
    unit_id: str,
    time_id: str,
    cluster: str = None
) -> CausalOutput:
    """
    Estimate triple difference model.

    Parameters
    ----------
    treat_group : str
        First difference dimension (e.g., treated state)
    affected_group : str
        Second difference dimension (e.g., affected industry)
    post : str
        Third difference dimension (post-treatment indicator)
    """
    from linearmodels.panel import PanelOLS
    import statsmodels.api as sm

    df = data.copy()

    # Create interaction terms
    df['treat_post'] = df[treat_group] * df[post]
    df['affected_post'] = df[affected_group] * df[post]
    df['treat_affected'] = df[treat_group] * df[affected_group]
    df['triple'] = df[treat_group] * df[affected_group] * df[post]

    # Set up panel
    df_panel = df.set_index([unit_id, time_id])

    y = df_panel[outcome]
    X = df_panel[['treat_post', 'affected_post', 'treat_affected', 'triple']]

    model = PanelOLS(y, X, entity_effects=True, time_effects=True)

    if cluster:
        result = model.fit(cov_type='clustered', cluster_entity=True)
    else:
        result = model.fit(cov_type='robust')

    ddd_coef = result.params['triple']
    ddd_se = result.std_errors['triple']
    ddd_pval = result.pvalues['triple']

    return CausalOutput(
        effect=ddd_coef,
        se=ddd_se,
        ci_lower=ddd_coef - 1.96 * ddd_se,
        ci_upper=ddd_coef + 1.96 * ddd_se,
        p_value=ddd_pval,
        diagnostics={'method': 'Triple Difference'}
    )

# Usage
result = estimate_triple_diff(
    data=df,
    outcome="employment",
    treat_group="treated_state",
    affected_group="affected_industry",
    post="post_policy",
    unit_id="firm_id",
    time_id="year",
    cluster="state"
)
```

### 2.4 When to Use DDD

| Situation | DDD Helpful? |
|-----------|-------------|
| Differential trends by state | Yes - controls for state-time effects |
| Spillovers to other industries | Yes - distinguishes direct from spillover effects |
| Concern about state-level confounders | Yes - adds within-state comparison |
| Simple treatment/control | Usually not needed |

---

## 3. Synthetic Control Comparison

### 3.1 When DID Fails, SC May Help

If parallel trends is clearly violated, synthetic control creates a better counterfactual by weighting control units to match pre-treatment trends.

### 3.2 Comparison

| Feature | DID | Synthetic Control |
|---------|-----|-------------------|
| # Treated units | Many | Few (often 1) |
| Parallel trends | Required | Not required |
| Control weighting | Equal | Optimized |
| Pre-treatment fit | May differ | By construction matches |
| Inference | Standard | Permutation-based |

### 3.3 Combining DID and SC

**Synthetic DID (Arkhangelsky et al., 2021)**:

```python
from synthdid import synthetic_diff_in_diff

result = synthetic_diff_in_diff(
    data=df,
    outcome="y",
    unit="id",
    time="year",
    treatment="treated"
)

print(f"Synthetic DID estimate: {result['att']:.4f}")
print(f"Standard error: {result['se']:.4f}")
```

**Key Idea**: SDID combines:
1. Unit weights (like SC) to improve pre-treatment parallel trends
2. Time weights to focus on relevant comparison periods
3. Fixed effects (like DID) for efficiency

---

## 4. Changes-in-Changes (CIC)

### 4.1 Concept

CIC relaxes parallel trends to a weaker assumption: the distribution of potential outcomes is stable within groups.

**Assumption**: For any quantile $q$:
$$
F_{Y(0),T=1,t=1}^{-1}(q) - F_{Y(0),T=1,t=0}^{-1}(q) = F_{Y(0),T=0,t=1}^{-1}(q) - F_{Y(0),T=0,t=0}^{-1}(q)
$$

### 4.2 Implementation

```python
def changes_in_changes(
    data: pd.DataFrame,
    outcome: str,
    treatment_group: str,
    post: str,
    n_quantiles: int = 100
) -> Dict:
    """
    Estimate Changes-in-Changes (Athey & Imbens, 2006).
    """
    # Get outcome distributions for each group-period
    y_t0_pre = data[(data[treatment_group] == 1) & (data[post] == 0)][outcome]
    y_t0_post = data[(data[treatment_group] == 1) & (data[post] == 1)][outcome]
    y_c_pre = data[(data[treatment_group] == 0) & (data[post] == 0)][outcome]
    y_c_post = data[(data[treatment_group] == 0) & (data[post] == 1)][outcome]

    # Create counterfactual distribution for treated in post period
    # F_{Y(0)|T=1,Post=1}^{-1}(q) = F_{Y|T=1,Pre=1}^{-1}(F_{Y|T=0,Pre=1}(F_{Y|T=0,Post=1}^{-1}(q)))

    quantiles = np.linspace(0.01, 0.99, n_quantiles)

    # Step 1: Get quantiles of control post distribution
    y_c_post_quantiles = np.quantile(y_c_post, quantiles)

    # Step 2: Map through control pre distribution (get ranks)
    from scipy.interpolate import interp1d
    F_c_pre = interp1d(
        np.sort(y_c_pre),
        np.linspace(0, 1, len(y_c_pre)),
        bounds_error=False,
        fill_value=(0, 1)
    )
    ranks = F_c_pre(y_c_post_quantiles)

    # Step 3: Map through treated pre distribution (get counterfactual)
    F_t_pre_inv = interp1d(
        np.linspace(0, 1, len(y_t0_pre)),
        np.sort(y_t0_pre),
        bounds_error=False,
        fill_value=(y_t0_pre.min(), y_t0_pre.max())
    )
    y_counterfactual = F_t_pre_inv(ranks)

    # Step 4: Compare actual to counterfactual
    y_actual = np.quantile(y_t0_post, quantiles)
    qte = y_actual - y_counterfactual  # Quantile treatment effects

    # Average treatment effect
    att = qte.mean()

    return {
        'att': att,
        'qte': qte,
        'quantiles': quantiles,
        'counterfactual': y_counterfactual,
        'actual': y_actual
    }

# Usage
cic_result = changes_in_changes(df, 'y', 'treated_ever', 'post')
print(f"CIC ATT: {cic_result['att']:.4f}")
```

---

## 5. Regression Discontinuity + DID (RD-DID)

### 5.1 Concept

Combine RD identification (for treatment at threshold) with DID (for before/after comparison).

**Example**: School funding formula that provides extra money to schools below enrollment threshold, evaluated using before/after policy change.

### 5.2 Implementation

```python
def rdid_estimator(
    data: pd.DataFrame,
    outcome: str,
    running_var: str,
    threshold: float,
    post: str,
    bandwidth: float = None,
    kernel: str = 'triangular'
) -> Dict:
    """
    Estimate RD-DID model.
    """
    from rdrobust import rdrobust

    # Normalize running variable
    data = data.copy()
    data['x_norm'] = data[running_var] - threshold

    # Pre-period RD
    pre_data = data[data[post] == 0]
    rd_pre = rdrobust(pre_data[outcome], pre_data['x_norm'], kernel=kernel, h=bandwidth)

    # Post-period RD
    post_data = data[data[post] == 1]
    rd_post = rdrobust(post_data[outcome], post_data['x_norm'], kernel=kernel, h=bandwidth)

    # RD-DID: Difference in RD effects
    rdid_effect = rd_post.Estimate['tau.bc'] - rd_pre.Estimate['tau.bc']

    # Standard error (simplified; should use bootstrap for full inference)
    rdid_se = np.sqrt(rd_post.se['se.rb']**2 + rd_pre.se['se.rb']**2)

    return {
        'rdid_effect': rdid_effect,
        'rdid_se': rdid_se,
        'rd_pre': rd_pre.Estimate['tau.bc'],
        'rd_post': rd_post.Estimate['tau.bc']
    }
```

---

## 6. Dealing with Anticipation

### 6.1 Modeling Anticipation

When anticipation is expected, explicitly model it:

```python
def estimate_with_anticipation(
    data: pd.DataFrame,
    outcome: str,
    treatment_time: str,
    unit_id: str,
    time_id: str,
    anticipation_periods: int = 1
) -> CausalOutput:
    """
    Estimate DID allowing for anticipation effects.
    """
    df = data.copy()

    # Redefine treatment timing to account for anticipation
    df['effective_treatment_time'] = df[treatment_time] - anticipation_periods

    # Create treatment indicator based on effective timing
    df['treated_with_anticipation'] = (
        df[time_id] >= df['effective_treatment_time']
    ).astype(int)

    result = estimate_did_staggered(
        data=df,
        outcome=outcome,
        treatment_time='effective_treatment_time',
        unit_id=unit_id,
        time_id=time_id,
        anticipation=anticipation_periods
    )

    return result
```

### 6.2 Bounding Approach

If anticipation is uncertain, bound the effect:

```python
def bound_with_anticipation(
    data: pd.DataFrame,
    outcome: str,
    treatment_time: str,
    max_anticipation: int = 2
) -> Dict:
    """
    Estimate bounds under different anticipation assumptions.
    """
    bounds = {}

    for ant in range(max_anticipation + 1):
        result = estimate_with_anticipation(
            data, outcome, treatment_time, 'id', 'year',
            anticipation_periods=ant
        )
        bounds[f'anticipation_{ant}'] = {
            'effect': result.effect,
            'se': result.se
        }

    # Report range
    effects = [b['effect'] for b in bounds.values()]
    print(f"Effect range: [{min(effects):.4f}, {max(effects):.4f}]")

    return bounds
```

---

## 7. Incomplete Adoption (Fuzzy DID)

### 7.1 Concept

When treatment assignment differs from treatment receipt (like fuzzy RD):
- Some assigned units don't take treatment
- Some unassigned units do take treatment

### 7.2 Implementation

```python
def fuzzy_did(
    data: pd.DataFrame,
    outcome: str,
    assignment: str,  # Intent to treat
    treatment: str,   # Actual treatment
    unit_id: str,
    time_id: str
) -> Dict:
    """
    Estimate DID with incomplete compliance (Fuzzy DID).

    Uses assignment as instrument for treatment.
    """
    from linearmodels.iv import IV2SLS

    df = data.copy()
    df = df.set_index([unit_id, time_id])

    # First stage: treatment ~ assignment
    # Second stage: outcome ~ treatment (instrumented by assignment)

    # Create interaction with post dummy
    post_periods = df[df[treatment] == 1].index.get_level_values(1).unique()
    df['post'] = df.index.get_level_values(1).isin(post_periods).astype(int)

    df['assignment_post'] = df[assignment] * df['post']
    df['treatment_post'] = df[treatment] * df['post']

    # 2SLS estimation
    model = IV2SLS.from_formula(
        f'{outcome} ~ 1 + [treatment_post ~ assignment_post] + EntityEffects + TimeEffects',
        data=df.reset_index()
    )
    result = model.fit(cov_type='clustered', clusters=df.index.get_level_values(0))

    # LATE interpretation
    late = result.params['treatment_post']
    late_se = result.std_errors['treatment_post']

    # First stage F-statistic
    first_stage_f = result.first_stage.diagnostics['f.stat']

    return {
        'late': late,
        'late_se': late_se,
        'first_stage_f': first_stage_f,
        'itt': None  # Can also compute ITT
    }
```

---

## 8. Continuous Treatment

### 8.1 Dose-Response DID

When treatment is continuous (not binary):

```python
def dose_response_did(
    data: pd.DataFrame,
    outcome: str,
    treatment_intensity: str,
    unit_id: str,
    time_id: str,
    method: str = 'linear'  # or 'flexible'
) -> Dict:
    """
    Estimate dose-response DID with continuous treatment.
    """
    from linearmodels.panel import PanelOLS

    df = data.set_index([unit_id, time_id])
    y = df[outcome]

    if method == 'linear':
        # Linear dose-response
        X = df[[treatment_intensity]]
        model = PanelOLS(y, X, entity_effects=True, time_effects=True)
        result = model.fit(cov_type='clustered', cluster_entity=True)

        return {
            'marginal_effect': result.params[treatment_intensity],
            'se': result.std_errors[treatment_intensity]
        }

    elif method == 'flexible':
        # Flexible (binned) dose-response
        df['dose_bin'] = pd.qcut(
            df[treatment_intensity],
            q=5,
            labels=['Q1', 'Q2', 'Q3', 'Q4', 'Q5']
        )
        # ... continue with categorical analysis

    return result
```

---

## 9. Summary: Choosing Extensions

```
Standard DID not working?
├── Parallel trends violated?
│   ├── Use synthetic control or SDID
│   └── Consider CIC if distributional assumption plausible
│
├── Need within-group comparison?
│   └── Consider Triple Difference (DDD)
│
├── Treatment varies continuously?
│   └── Use dose-response DID
│
├── Assignment ≠ Treatment?
│   └── Use Fuzzy DID (IV approach)
│
├── Want to estimate heterogeneity?
│   ├── Observable: interaction terms or subgroup analysis
│   └── Unobservable: causal forests
│
└── Threshold-based assignment?
    └── Consider RD-DID
```

---

## References

1. Arkhangelsky, D., Athey, S., Hirshberg, D. A., Imbens, G. W., & Wager, S. (2021). "Synthetic Difference-in-Differences." *American Economic Review*.

2. Athey, S., & Imbens, G. W. (2006). "Identification and Inference in Nonlinear Difference-in-Differences Models." *Econometrica*.

3. de Chaisemartin, C., & D'Haultfoeuille, X. (2018). "Fuzzy Differences-in-Differences." *Review of Economic Studies*.

4. Callaway, B., Goodman-Bacon, A., & Sant'Anna, P. H. (2024). "Difference-in-Differences with a Continuous Treatment." *NBER Working Paper*.

---

## See Also

- [estimation_methods.md](estimation_methods.md) - Core DID methods
- [identification_assumptions.md](identification_assumptions.md) - When extensions are needed
- [common_errors.md](common_errors.md) - Mistakes to avoid with extensions
