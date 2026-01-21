# Identification Assumptions for Difference-in-Differences

> **Document Type**: Reference | **Last Updated**: 2025-01
> **Related**: [diagnostic_tests.md](diagnostic_tests.md), [common_errors.md](common_errors.md)

## Overview

The validity of Difference-in-Differences (DID) estimation rests on several key assumptions. Violation of any core assumption invalidates the causal interpretation of the estimated effect. This document provides detailed explanations of each assumption, their formal definitions, and practical guidance for assessing their plausibility.

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

## 3. No Anticipation Assumption

### Formal Definition

Units do not change their behavior in anticipation of future treatment:

$$
E[Y_{it}(0) | G_i = 1, t < t^*] = E[Y_{it}(D_i) | G_i = 1, t < t^*]
$$

Where $t^*$ is the treatment timing. For periods before $t^*$, outcomes should be the same regardless of future treatment status.

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

## Summary Table: Assumption Checklist

| Assumption | Testable? | Pre-Analysis Check | Main Risk |
|------------|-----------|-------------------|-----------|
| Parallel Trends | Partially | Visual + statistical pre-trends | Bias from differential trends |
| SUTVA | No | Domain knowledge review | Spillover contamination |
| No Anticipation | Partially | Event study pre-coefficients | Underestimated effects |
| No Composition Changes | Yes | Panel balance check | Selection bias |
| Common Support | Yes | Balance tests, overlap plots | Extrapolation bias |

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
