# Identification Assumptions for Difference-in-Differences

> **Document Type**: Reference | **Last Updated**: 2026-01
> **Related**: [diagnostic_tests.md](diagnostic_tests.md), [common_errors.md](common_errors.md)
> **Key Reference**: Callaway & Sant'Anna (2021), Journal of Econometrics

## Overview

The validity of Difference-in-Differences (DID) estimation rests on several key assumptions. Violation of any core assumption invalidates the causal interpretation of the estimated effect. This document provides detailed explanations of each assumption, their formal definitions, and practical guidance for assessing their plausibility.

**Note**: This document covers both classical DID assumptions and the more general framework from Callaway & Sant'Anna (2021) for staggered adoption designs.

---

## 1. Parallel Trends Assumption

### Formal Definition

The parallel trends assumption states that, in the absence of treatment, the average outcomes for treated and control groups would have followed parallel paths over time:

$$
E[Y_{it}(0) | G_i = 1, t] - E[Y_{it}(0) | G_i = 1, t-1] = E[Y_{it}(0) | G_i = 0, t] - E[Y_{it}(0) | G_i = 0, t-1]
$$

Where:
- $Y_{it}(0)$ is the potential outcome under no treatment
- $G_i = 1$ indicates units in the treatment group
- $t$ indexes time periods

### Intuitive Explanation

If treatment had never occurred, the treatment and control groups would have experienced the same **change** in outcomes over time. This does NOT require that the groups have the same **level** of outcomes - only that their trends would be parallel.

### What Parallel Trends Allows

1. **Time-invariant unobserved confounders**: Any characteristic that differs between groups but is constant over time is differenced out
2. **Common time shocks**: Any factor that affects both groups equally in a given period is controlled for
3. **Group-level fixed effects**: Permanent differences between treatment and control groups

### What Parallel Trends Does NOT Allow

1. **Group-specific time trends**: If one group was on an upward trajectory while the other was flat
2. **Differential responses to common shocks**: If both groups experience a recession but respond differently
3. **Selection based on anticipated treatment effects**: If units selected into treatment because they anticipated larger benefits

### Testing Parallel Trends

The parallel trends assumption cannot be directly tested because it concerns **counterfactual** outcomes. However, we can examine **pre-treatment trends**:

```python
# Visual inspection (essential first step)
from did_estimator import plot_parallel_trends

fig = plot_parallel_trends(
    data=df,
    outcome="y",
    treatment_group="treated_ever",
    time_id="year",
    treatment_time=2015
)

# Statistical test
from did_estimator import test_parallel_trends

result = test_parallel_trends(
    data=df,
    outcome="y",
    treatment_group="treated_ever",
    time_id="year",
    unit_id="id",
    treatment_time=2015,
    n_pre_periods=4
)
```

**Warning**: Parallel pre-trends do NOT guarantee parallel counterfactual post-trends. The pre-trends test is necessary but not sufficient.

### Conditional Parallel Trends (Callaway & Sant'Anna)

For staggered adoption designs, Callaway & Sant'Anna (2021) introduce a more flexible **conditional** parallel trends assumption:

**Formal Definition (Assumption 4 in C-S)**:
$$
E[Y_t(0) - Y_{t-1}(0) | X, G_g = 1] = E[Y_t(0) - Y_{t-1}(0) | X, C = 1]
$$

Where:
- $G_g = 1$ indicates units first treated at time $g$ (cohort $g$)
- $C = 1$ indicates never-treated units (comparison group)
- $X$ are pre-treatment covariates
- The equality holds for all $g \in \mathcal{G}$ and $t \geq 2$

**Intuitive Explanation**:
After conditioning on observed covariates $X$, the average change in potential untreated outcomes would be the same for units in cohort $g$ as for never-treated units. This is **weaker** than unconditional parallel trends because it allows for:
- Covariate-driven differences in trends between groups
- Different baseline characteristics between cohorts

**When to Use Conditional vs Unconditional**:

| Scenario | Use Unconditional | Use Conditional |
|----------|------------------|-----------------|
| Groups have similar observable characteristics | ✓ | Either |
| Observable differences explain selection | | ✓ |
| Limited covariates available | ✓ | |
| Strong theoretical reason for covariate adjustment | | ✓ |

**Python Implementation**:
```python
from did_estimator import estimate_did_staggered

# Conditional parallel trends (with covariates)
result = estimate_did_staggered(
    data=df,
    outcome="y",
    treatment_time="first_treated",
    unit_id="id",
    time_id="year",
    covariates=["x1", "x2", "x3"],  # Conditioning covariates
    control_group="nevertreated"
)

# Unconditional parallel trends (no covariates)
result_unconditional = estimate_did_staggered(
    data=df,
    outcome="y",
    treatment_time="first_treated",
    unit_id="id",
    time_id="year",
    covariates=None,  # No conditioning
    control_group="nevertreated"
)
```

### Common Violations

| Violation Type | Example | Solution |
|---------------|---------|----------|
| Diverging pre-trends | Treatment group already improving faster | Synthetic control, matching |
| Anticipation effects | Units change behavior before treatment | Adjust treatment timing, model anticipation |
| Composition changes | Different units observed over time | Balanced panel, explicit modeling |
| Differential shocks | Economic shock affects groups differently | Triple-difference, within-group variation |

---

## 2. SUTVA (Stable Unit Treatment Value Assumption)

### Formal Definition

SUTVA consists of two components:

**2.1. No Interference (Spillovers)**:
$$
Y_{it}(D_1, D_2, ..., D_N) = Y_{it}(D_i)
$$

The potential outcome for unit $i$ depends only on unit $i$'s treatment status, not on other units' treatment status.

**2.2. Treatment Homogeneity**:
There is only one version of treatment. The potential outcome under treatment, $Y_i(1)$, is well-defined.

### Why SUTVA Matters

If SUTVA is violated, the estimated treatment effect is a mixture of direct effects and spillover effects, making interpretation difficult.

### Common Violations

#### Geographic Spillovers
- Minimum wage increase in state A affects neighboring state B
- Policy in one school district affects students who can move between districts

#### Market-Level Spillovers
- Subsidy to treated firms affects competitors (general equilibrium effects)
- Training program for some workers affects wages of untrained workers

#### Social Network Spillovers
- Treatment of friends/family affects individual outcomes
- Information spillovers through networks

### Testing for SUTVA Violations

SUTVA is generally **not directly testable** from the data. Assessment relies on:

1. **Domain knowledge**: Consider plausible spillover channels
2. **Design features**:
   - Geographic distance between treated and control units
   - Market structure and competition
   - Social network structure

3. **Indirect tests**:
```python
# Test if control outcomes change when nearby units are treated
# (suggests spillovers if significant)
def test_spillovers(data, outcome, treatment, distance_var, threshold):
    """
    Test for spatial spillovers by examining control units near treated units.
    """
    data['near_treated'] = data[distance_var] < threshold
    # Compare control units near treated vs. far from treated
    ...
```

### Addressing SUTVA Violations

| Strategy | When to Use |
|----------|-------------|
| **Expand treatment definition** | Include spillover effects as part of treatment |
| **Geographic distance controls** | When spatial spillovers are plausible |
| **Exclude affected controls** | Remove control units likely affected by spillovers |
| **Clustering at higher level** | Treatment and inference at market/region level |

---

## 3. No Anticipation / Limited Anticipation Assumption

### Formal Definition (Strict No Anticipation)

Units do not change their behavior in anticipation of future treatment:

$$
E[Y_{it}(0) | G_i = 1, t < t^*] = E[Y_{it}(D_i) | G_i = 1, t < t^*]
$$

Where $t^*$ is the treatment timing. For periods before $t^*$, outcomes should be the same regardless of future treatment status.

### Limited Treatment Anticipation (Callaway & Sant'Anna)

Callaway & Sant'Anna (2021) introduce a more flexible **limited anticipation** assumption:

**Formal Definition (Assumption 5 in C-S)**:
$$
E[Y_t(g) - Y_t(0) | X, G_g = 1] = 0 \quad \text{for all } t < g - \delta
$$

Where:
- $g$ is the time period when treatment begins
- $\delta \geq 0$ is the number of periods during which anticipation effects are allowed
- For $\delta = 0$, this reduces to the standard no anticipation assumption

**Intuitive Explanation**:
Treatment effects can begin up to $\delta$ periods before the official treatment start date. This accommodates:
- Policy announcements before implementation
- Gradual rollout where units know treatment is coming
- Behavioral adjustments in anticipation of known future treatment

**Setting the Anticipation Parameter**:

| Scenario | Recommended δ |
|----------|---------------|
| Treatment surprise (no prior knowledge) | 0 |
| Policy announced 1 period before implementation | 1 |
| Gradual rollout with advance notice | 1-2 |
| Long-term planning horizon | 2+ |

**Python Implementation**:
```python
from did_estimator import estimate_did_staggered

# Allow for 1 period of anticipation
result = estimate_did_staggered(
    data=df,
    outcome="y",
    treatment_time="first_treated",
    unit_id="id",
    time_id="year",
    anticipation=1,  # Allow 1 period anticipation
    control_group="nevertreated"
)

# Sensitivity analysis: compare different anticipation assumptions
for delta in [0, 1, 2]:
    result = estimate_did_staggered(
        data=df, outcome="y", treatment_time="first_treated",
        unit_id="id", time_id="year", anticipation=delta
    )
    print(f"δ={delta}: ATT = {result.effect:.4f} (SE={result.se:.4f})")
```

### Why No Anticipation Matters

If units anticipate treatment:
- Pre-treatment periods may already reflect (partial) treatment effects
- The "parallel trends" in pre-treatment data may be affected by anticipation
- The post-treatment estimate may understate the full effect

### Common Anticipation Scenarios

1. **Policy announcements**: Policy announced 6 months before implementation
2. **Gradual rollout**: Units know they will receive treatment soon
3. **Behavioral response to expectations**: Investment changes based on expected future policy
4. **Strategic timing**: Units delay/accelerate actions based on treatment timing

### Testing for Anticipation

Examine event study coefficients for periods just before treatment:

```python
from did_estimator import event_study_plot

fig = event_study_plot(
    data=df,
    outcome="y",
    treatment_time_var="first_treated",
    unit_id="id",
    time_id="year",
    reference_period=-1,
    pre_periods=6,  # Look back further
    post_periods=4
)
```

**Signs of anticipation**:
- Significant coefficients in periods t-1, t-2, etc.
- Trending coefficients approaching treatment time
- "Dip" or "spike" pattern before treatment

### Addressing Anticipation

| Strategy | Implementation |
|----------|---------------|
| **Redefine treatment timing** | Set $t^* = $ announcement date, not implementation |
| **Model anticipation explicitly** | Allow for anticipation effects in event study |
| **Exclude anticipation periods** | Drop periods between announcement and implementation |
| **Use announcement as instrument** | Instrument for actual treatment with announcement |

---

## 4. No Composition Changes

### Formal Definition

The composition of treatment and control groups should remain stable over time:

$$
Pr(i \in Sample_t | G_i = g) = Pr(i \in Sample_{t'} | G_i = g) \quad \forall t, t'
$$

### Why Composition Matters

If different units enter/exit the sample around treatment time:
- Changes in group means may reflect composition changes, not treatment effects
- Selection into/out of the sample may be correlated with treatment

### Common Violations

1. **Mortality/attrition**: Units leaving the sample (firms going bankrupt, people dying)
2. **Entry**: New units entering (new firms, births)
3. **Migration**: Units switching between treatment and control (moving between states)
4. **Sample selection changes**: Changes in survey response rates

### Testing for Composition Changes

```python
# Check for balanced panel
def check_balance(data, unit_id, time_id):
    """Check if panel is balanced."""
    obs_per_unit = data.groupby(unit_id)[time_id].nunique()
    max_periods = data[time_id].nunique()

    n_balanced = (obs_per_unit == max_periods).sum()
    n_total = len(obs_per_unit)

    print(f"Balanced units: {n_balanced}/{n_total} ({100*n_balanced/n_total:.1f}%)")

    if obs_per_unit.nunique() > 1:
        print(f"Observations per unit range: {obs_per_unit.min()} - {obs_per_unit.max()}")

    return n_balanced == n_total

# Check composition changes around treatment
def check_composition_stability(data, unit_id, time_id, treatment_time, treatment_group):
    """Check if sample composition changes around treatment."""
    pre_units = set(data[data[time_id] < treatment_time][unit_id])
    post_units = set(data[data[time_id] >= treatment_time][unit_id])

    entered = post_units - pre_units
    exited = pre_units - post_units

    print(f"Units entering after treatment: {len(entered)}")
    print(f"Units exiting after treatment: {len(exited)}")

    # Check if entry/exit differs by treatment group
    ...
```

### Addressing Composition Changes

| Strategy | When to Use |
|----------|-------------|
| **Balanced panel** | Restrict to units observed in all periods |
| **Inverse probability weighting** | Weight by probability of remaining in sample |
| **Lee bounds** | Bound treatment effect under selection |
| **Model selection explicitly** | Joint model of selection and outcomes |

---

## 5. Common Support

### Formal Definition

There should be sufficient overlap in the distribution of covariates between treatment and control groups:

$$
0 < Pr(G_i = 1 | X_i) < 1 \quad \forall X_i
$$

### Why Common Support Matters

If treatment and control groups have non-overlapping characteristics:
- We rely heavily on model extrapolation
- Parallel trends may be implausible across very different units
- Treatment effects may not be identified in regions without overlap

### Testing Common Support

```python
from did_estimator import balance_test_did

balance = balance_test_did(
    data=df,
    treatment_group="treated_ever",
    covariates=["age", "income", "education", "firm_size"],
    time_id="year",
    pre_period=2014
)

# Check overlap in propensity scores
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

# Fit propensity score model
X = df_pre[covariates]
y = df_pre[treatment_group]
model = LogisticRegression()
model.fit(X, y)
pscore = model.predict_proba(X)[:, 1]

# Plot distributions
fig, ax = plt.subplots()
ax.hist(pscore[y == 0], bins=50, alpha=0.5, label='Control', density=True)
ax.hist(pscore[y == 1], bins=50, alpha=0.5, label='Treatment', density=True)
ax.legend()
ax.set_xlabel('Propensity Score')
ax.set_title('Common Support Check')
```

### Improving Common Support

| Strategy | Implementation |
|----------|---------------|
| **Trimming** | Exclude units with extreme propensity scores |
| **Matching** | Match treated units to similar controls |
| **Reweighting** | IPW to equalize covariate distributions |
| **Restrict sample** | Limit to subpopulation with overlap |

---

## 6. Irreversibility of Treatment (Staggered DID)

### Formal Definition

**Callaway & Sant'Anna Assumption 6**:
$$
D_{i,t} = 1 \Rightarrow D_{i,t+s} = 1 \quad \text{for all } s \geq 0
$$

Once a unit receives treatment, it remains treated in all subsequent periods. Treatment is **absorbing** - units can enter treatment but not exit.

### Intuitive Explanation

In staggered adoption settings, the irreversibility assumption means:
- Once a firm adopts a new technology, it doesn't revert to the old one
- Once a state implements a policy, the policy remains in effect
- Once an individual receives training, they don't "unlearn"

This assumption is **implicit** in the definition of treatment cohorts $G_g$, which partition units by when they were **first** treated.

### Why Irreversibility Matters

Without irreversibility:
1. **Group definition becomes ambiguous**: Units could belong to multiple "cohorts" if they switch treatment status
2. **Counterfactual reasoning breaks down**: $Y_t(g)$ assumes treatment started at $g$ and continued
3. **Aggregation schemes invalid**: The selective timing and dynamic effect aggregations assume treatment persists

### When Irreversibility is Violated

| Scenario | Example | Implication |
|----------|---------|-------------|
| Policy reversal | Tax implemented then repealed | Units can be "untreated" |
| Temporary treatment | Short-term program participation | Treatment status fluctuates |
| Switching behavior | Firms adopt/drop practices | Multiple treatment episodes |

### Addressing Irreversibility Violations

**Option 1: de Chaisemartin & D'Haultfoeuille Estimator**

When treatment can turn on AND off:

```python
from scripts.robustness_checks import dechaisemartin_dhaultfoeuille

# DID_M estimator handles treatment reversals
result = dechaisemartin_dhaultfoeuille(
    data=df,
    outcome="y",
    treatment="treatment_status",  # Can be 0 or 1 in any period
    unit_id="id",
    time_id="year"
)
```

**Option 2: Redefine Treatment**

```python
# Instead of current treatment status, use "ever treated"
df['ever_treated'] = df.groupby('id')['treated'].transform('max')

# Or define treatment as "duration of exposure"
df['treatment_duration'] = df.groupby('id')['treated'].cumsum()
```

**Option 3: Restrict Sample**

```python
# Keep only units that never reverse treatment
def check_irreversibility(group):
    """Check if treatment is monotonically non-decreasing."""
    return (group['treated'].diff().dropna() >= 0).all()

valid_units = df.groupby('id').apply(check_irreversibility)
df_irreversible = df[df['id'].isin(valid_units[valid_units].index)]
```

### Testing for Irreversibility

```python
def test_irreversibility(data, unit_id, time_id, treatment):
    """
    Test if treatment is irreversible (absorbing).

    Returns the number and percentage of units that violate irreversibility.
    """
    df = data.sort_values([unit_id, time_id])

    violations = []
    for unit, group in df.groupby(unit_id):
        treated_periods = group[group[treatment] == 1][time_id].values
        untreated_periods = group[group[treatment] == 0][time_id].values

        if len(treated_periods) > 0 and len(untreated_periods) > 0:
            # Check if any untreated period comes after a treated period
            if untreated_periods.max() > treated_periods.min():
                violations.append(unit)

    n_violations = len(violations)
    n_total = data[unit_id].nunique()

    print(f"Irreversibility violations: {n_violations}/{n_total} units ({100*n_violations/n_total:.1f}%)")

    if n_violations > 0:
        print("WARNING: Treatment is reversible for some units.")
        print("Consider: de Chaisemartin-D'Haultfoeuille estimator or sample restriction.")

    return {'violations': violations, 'n_violations': n_violations, 'pct': n_violations/n_total}

# Usage
irrev_test = test_irreversibility(df, 'id', 'year', 'treated')
```

---

## 7. Overlap / Generalized Propensity Score

### Formal Definition (Callaway & Sant'Anna)

**Generalized Propensity Score (Assumption 2)**:
$$
p_g(X) = P(G_g = 1 | X, G_g + C = 1)
$$

Where:
- $G_g = 1$ indicates the unit belongs to cohort $g$ (first treated at time $g$)
- $C = 1$ indicates the unit is never treated
- $G_g + C = 1$ restricts to units that are either in cohort $g$ or never treated

**Overlap Condition (Assumption 3)**:
$$
p_g(X) < 1 \quad \text{for all } X \text{ in the support and all } g \in \mathcal{G}
$$

### Intuitive Explanation

The generalized propensity score $p_g(X)$ is the probability that a unit belongs to treatment cohort $g$, conditional on:
1. Observable characteristics $X$
2. Being either in cohort $g$ or never treated

This differs from the standard propensity score because:
- It compares cohort $g$ to never-treated units specifically
- Not-yet-treated units from other cohorts are excluded from this comparison
- A separate propensity score is estimated for each cohort

**Overlap** ensures that for any covariate profile $X$, there exist both cohort $g$ units and never-treated units. Without overlap, we cannot estimate $ATT(g,t)$ for those covariate values.

### Python Implementation

```python
from sklearn.linear_model import LogisticRegression
import numpy as np

def estimate_generalized_propensity_score(
    data, cohort_indicator, never_treated_indicator, covariates
):
    """
    Estimate the generalized propensity score p_g(X).

    Parameters
    ----------
    data : DataFrame
        Panel data
    cohort_indicator : str
        Binary indicator for cohort g
    never_treated_indicator : str
        Binary indicator for never-treated
    covariates : list
        Covariate names
    """
    # Restrict to cohort g or never-treated
    mask = (data[cohort_indicator] == 1) | (data[never_treated_indicator] == 1)
    df_subset = data[mask].copy()

    # Get unique units (one observation per unit for propensity score)
    df_units = df_subset.groupby('id').first().reset_index()

    X = df_units[covariates]
    y = df_units[cohort_indicator]

    # Fit logistic regression
    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)

    # Predict propensity scores
    p_g = model.predict_proba(X)[:, 1]

    # Check overlap
    print(f"Propensity score range: [{p_g.min():.4f}, {p_g.max():.4f}]")

    if p_g.max() > 0.99:
        print("WARNING: Near-violation of overlap (p_g close to 1)")

    return p_g, model

# Estimate for cohort 2015
df['cohort_2015'] = (df['first_treated'] == 2015).astype(int)
df['never_treated'] = (df['first_treated'].isna() | (df['first_treated'] == np.inf)).astype(int)

p_scores, model = estimate_generalized_propensity_score(
    df, 'cohort_2015', 'never_treated', ['x1', 'x2', 'x3']
)
```

### Checking Overlap

```python
import matplotlib.pyplot as plt

def plot_propensity_overlap(data, p_scores, cohort_indicator, never_treated_indicator):
    """Visualize overlap in generalized propensity scores."""
    mask_g = data[cohort_indicator] == 1
    mask_c = data[never_treated_indicator] == 1

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(p_scores[mask_c], bins=50, alpha=0.5, label='Never-treated', density=True)
    ax.hist(p_scores[mask_g], bins=50, alpha=0.5, label=f'Cohort g', density=True)
    ax.set_xlabel('Generalized Propensity Score')
    ax.set_ylabel('Density')
    ax.legend()
    ax.set_title('Overlap Check: Generalized Propensity Score')

    return fig

# Plot overlap
fig = plot_propensity_overlap(df_units, p_scores, 'cohort_2015', 'never_treated')
```

---

## Summary Table: Assumption Checklist

### Core Assumptions (All DID Designs)

| Assumption | Testable? | Pre-Analysis Check | Main Risk |
|------------|-----------|-------------------|-----------|
| Parallel Trends | Partially | Visual + statistical pre-trends | Bias from differential trends |
| SUTVA | No | Domain knowledge review | Spillover contamination |
| No Anticipation | Partially | Event study pre-coefficients | Underestimated effects |
| No Composition Changes | Yes | Panel balance check | Selection bias |
| Common Support | Yes | Balance tests, overlap plots | Extrapolation bias |

### Additional Assumptions (Staggered DID / Callaway-Sant'Anna)

| Assumption | Testable? | Pre-Analysis Check | Main Risk |
|------------|-----------|-------------------|-----------|
| Conditional Parallel Trends | Partially | Pre-trends conditional on X | Model misspecification |
| Limited Anticipation (δ periods) | Partially | Event study coefficients at g-δ | Incorrect anticipation parameter |
| Irreversibility of Treatment | Yes | Check for treatment reversals | Invalid group definitions |
| Overlap (Generalized PS) | Yes | Propensity score distribution | Unstable IPW estimates |

---

## References

### Key Methodology Papers

1. Abadie, A. (2005). "Semiparametric Difference-in-Differences Estimators." *Review of Economic Studies*, 72(1), 1-19.

2. Roth, J. (2022). "Pretest with Caution: Event-Study Estimates after Testing for Parallel Trends." *American Economic Review: Insights*, 4(3), 305-322.

3. Rambachan, A., & Roth, J. (2023). "A More Credible Approach to Parallel Trends." *Review of Economic Studies*, 90(5), 2555-2591.

4. Callaway, B., & Sant'Anna, P. H. (2021). "Difference-in-Differences with Multiple Time Periods." *Journal of Econometrics*, 225(2), 200-230.

### Textbook References

- Angrist, J. D., & Pischke, J. S. (2009). *Mostly Harmless Econometrics*. Chapter 5.
- Cunningham, S. (2021). *Causal Inference: The Mixtape*. Chapter 9.
- Huntington-Klein, N. (2022). *The Effect*. Chapter 18.

---

## See Also

- [diagnostic_tests.md](diagnostic_tests.md) - How to test these assumptions
- [common_errors.md](common_errors.md) - Common mistakes in assumption assessment
- [extensions.md](extensions.md) - Methods when assumptions are violated
