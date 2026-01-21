# Estimation Methods for Difference-in-Differences

> **Document Type**: Reference | **Last Updated**: 2025-01
> **Related**: [SKILL.md](../SKILL.md), [did_estimator.py](../did_estimator.py)

## Overview

This document provides detailed guidance on different DID estimation methods, their assumptions, advantages, and when to use each approach.

---

## 1. Classic 2x2 Difference-in-Differences

### Model Specification

$$
Y_{it} = \alpha + \beta_1 \cdot \text{Treat}_i + \beta_2 \cdot \text{Post}_t + \delta \cdot (\text{Treat}_i \times \text{Post}_t) + \varepsilon_{it}
$$

Where:
- $\text{Treat}_i$ = 1 if unit $i$ is in treatment group
- $\text{Post}_t$ = 1 if time $t$ is post-treatment
- $\delta$ = DID estimate (ATT)

### Equivalent Cell-Mean Formula

$$
\hat{\delta}_{DID} = (\bar{Y}_{T,Post} - \bar{Y}_{T,Pre}) - (\bar{Y}_{C,Post} - \bar{Y}_{C,Pre})
$$

### When to Use

- Two time periods only (pre and post)
- Single treatment timing
- Simple comparison groups

### Implementation

```python
from did_estimator import estimate_did_2x2

result = estimate_did_2x2(
    data=df,
    outcome="y",
    treatment_group="treated_ever",
    post="post",
    controls=None,  # or ["x1", "x2"]
    robust_se=True
)
```

### Advantages and Limitations

| Advantages | Limitations |
|------------|-------------|
| Simple and transparent | Cannot test parallel trends (only 2 periods) |
| Easy to interpret | No dynamic effects |
| Clear identification | Inefficient with panel data |

---

## 2. Two-Way Fixed Effects (TWFE)

### Model Specification

$$
Y_{it} = \alpha_i + \lambda_t + \delta \cdot D_{it} + X_{it}'\beta + \varepsilon_{it}
$$

Where:
- $\alpha_i$ = unit fixed effects
- $\lambda_t$ = time fixed effects
- $D_{it}$ = treatment indicator (1 if treated at time $t$)

### When to Use

- Multiple time periods
- Single treatment timing (all treated units treated at same time)
- Homogeneous treatment effects

### Implementation

```python
from did_estimator import estimate_did_panel

result = estimate_did_panel(
    data=df,
    outcome="y",
    treatment="treated",
    unit_id="id",
    time_id="year",
    controls=["x1", "x2"],
    cluster="id",
    entity_effects=True,
    time_effects=True
)
```

### Understanding TWFE

The TWFE estimator identifies $\delta$ by comparing:
1. **Within-unit variation**: How does outcome change for a unit when it becomes treated?
2. **Controlling for time**: Net of common time effects

**Fixed Effects Remove**:
- Unit-level time-invariant confounders
- Period-specific common shocks

### Standard Errors

```python
# Different SE options
from linearmodels.panel import PanelOLS

# Robust (heteroskedasticity-consistent)
result_robust = model.fit(cov_type='robust')

# Clustered at entity level (standard for DID)
result_cluster = model.fit(cov_type='clustered', cluster_entity=True)

# Clustered at time level
result_time = model.fit(cov_type='clustered', cluster_time=True)

# Double-clustered (entity and time)
result_double = model.fit(cov_type='clustered', cluster_entity=True, cluster_time=True)
```

### TWFE Pitfalls

**Problem**: With staggered treatment timing, TWFE can be biased.

**Why**: Already-treated units serve as controls for later-treated units. If treatment effects are heterogeneous or dynamic, this creates bias.

```
Time:        t=1    t=2    t=3    t=4
Group A:      C      C      T      T     (treated at t=3)
Group B:      C      C      C      T     (treated at t=4)
Group C:      C      C      C      C     (never treated)

At t=4, TWFE uses:
- Group C (never treated) as control for B   [OK]
- Group A (already treated) as control for B [PROBLEM!]
```

---

## 3. Staggered DID: Callaway-Sant'Anna (2021)

### Concept

Estimate group-time average treatment effects $ATT(g,t)$ separately for each treatment cohort $g$ and time period $t$, then aggregate.

### Notation

| Symbol | Definition |
|--------|------------|
| $G_g$ | Indicator for cohort $g$ (units first treated at time $g$) |
| $C$ | Indicator for never-treated units |
| $\mathcal{G}$ | Set of all treatment cohorts |
| $\mathcal{T}$ | Last time period in the panel |
| $Y_t(g)$ | Potential outcome at time $t$ if first treated at $g$ |
| $Y_t(0)$ | Potential outcome at time $t$ if never treated |

### Target Parameter: ATT(g,t)

**Definition**:
$$
ATT(g,t) = E[Y_t(g) - Y_t(0) | G_g = 1]
$$

This is the average treatment effect at time $t$ for units that were first treated at time $g$.

### Identification (Theorem 1 in C-S)

Under Assumptions 1-6 (random sampling, overlap, conditional parallel trends, limited anticipation, irreversibility), the $ATT(g,t)$ is identified as:

**Inverse Probability Weighting (IPW) Formula**:
$$
ATT(g,t) = E\left[\left(\frac{G_g}{E[G_g]} - \frac{\frac{p_g(X) \cdot C}{1-p_g(X)}}{E\left[\frac{p_g(X) \cdot C}{1-p_g(X)}\right]}\right)(Y_t - Y_{g-\delta-1})\right]
$$

Where:
- $p_g(X) = P(G_g = 1 | X, G_g + C = 1)$ is the generalized propensity score
- $\delta$ is the anticipation parameter (periods before treatment with potential effects)
- $g - \delta - 1$ is the last "clean" pre-treatment period

**Outcome Regression (OR) Formula**:
$$
ATT(g,t) = E\left[\frac{G_g}{E[G_g]}\left((Y_t - Y_{g-\delta-1}) - (m_{g,t}(X) - m_{g,g-\delta-1}(X))\right)\right]
$$

Where $m_{g,t}(X) = E[Y_t | X, C = 1]$ is the conditional mean for never-treated units.

**Doubly Robust (DR) Formula**:
$$
ATT(g,t) = E\left[\left(\frac{G_g}{E[G_g]} - \frac{\frac{p_g(X) \cdot C}{1-p_g(X)}}{E\left[\frac{p_g(X) \cdot C}{1-p_g(X)}\right]}\right)\left((Y_t - Y_{g-\delta-1}) - (m_{g,t}(X) - m_{g,g-\delta-1}(X))\right)\right]
$$

The doubly robust estimator is consistent if **either** the propensity score model **or** the outcome regression model is correctly specified.

### Simplified Model (Unconditional)

For each $(g,t)$ combination with no covariates:
$$
ATT(g,t) = E[Y_t - Y_{g-1} | G = g] - E[Y_t - Y_{g-1} | C = 1]
$$

Where:
- $g$ = treatment cohort (time when group was first treated)
- Control = never-treated units (C = 1)

### Aggregation Schemes

The $ATT(g,t)$ estimates can be aggregated in different ways depending on the research question.

#### 3.1 Overall Simple Aggregation

$$
ATT = \sum_{g \in \mathcal{G}} \sum_{t=g}^{\mathcal{T}} w_{g,t} \cdot ATT(g,t)
$$

#### 3.2 Selective Treatment Timing (θ^S)

**Question**: What is the average effect for a unit that was treated at a randomly drawn treatment time?

**Group-level average**:
$$
\theta^S(g) = \frac{1}{\mathcal{T} - g + 1} \sum_{t=g}^{\mathcal{T}} ATT(g,t)
$$

**Overall selective timing effect**:
$$
\theta^S = \sum_{g \in \mathcal{G}} \theta^S(g) \cdot P(G = g | G \in \mathcal{G})
$$

Where $P(G = g | G \in \mathcal{G})$ is the share of units in cohort $g$ among all eventually-treated units.

**Interpretation**: The average effect across all treated cohorts, where each cohort is weighted by its size.

#### 3.3 Dynamic Treatment Effects (θ^D)

**Question**: How does the treatment effect evolve with exposure length?

**Definition** (Effect at exposure $e$):
$$
\theta^D(e) = \sum_{g \in \mathcal{G}} \mathbf{1}\{g + e \leq \mathcal{T}\} \cdot ATT(g, g+e) \cdot P(G = g | G + e \leq \mathcal{T})
$$

Where:
- $e$ = length of exposure (periods since treatment started)
- $e = 0$ is the first period of treatment
- $e < 0$ represents pre-treatment periods (for pre-trends analysis)

**Interpretation**: The average treatment effect among units that have been exposed to treatment for exactly $e$ periods.

**Python Implementation**:
```python
def aggregate_dynamic_effects(att_gt, data, time_id, treatment_time, max_e=5):
    """
    Compute dynamic treatment effects θ^D(e).

    Parameters
    ----------
    att_gt : dict
        Dictionary of ATT(g,t) estimates: {(g, t): estimate}
    data : DataFrame
        Panel data
    time_id : str
        Time variable name
    treatment_time : str
        Variable indicating when unit was first treated
    max_e : int
        Maximum exposure length to compute

    Returns
    -------
    dict
        {e: theta_D(e)} for e in range(-max_e, max_e+1)
    """
    import numpy as np

    T = data[time_id].max()
    cohorts = data[treatment_time].dropna().unique()

    theta_d = {}

    for e in range(-max_e, max_e + 1):
        # Find cohorts that can contribute to this exposure
        valid_cohorts = [g for g in cohorts if g + e <= T and g + e >= 1]

        if not valid_cohorts:
            continue

        # Weight by cohort size
        weights = {}
        total_weight = 0
        for g in valid_cohorts:
            n_g = (data[treatment_time] == g).sum() / len(data[data[treatment_time] == g][time_id].unique())
            weights[g] = n_g
            total_weight += n_g

        # Normalize weights
        weights = {g: w / total_weight for g, w in weights.items()}

        # Compute weighted average
        theta = 0
        for g in valid_cohorts:
            t = g + e
            if (g, t) in att_gt:
                theta += weights[g] * att_gt[(g, t)]

        theta_d[e] = theta

    return theta_d
```

#### 3.4 Calendar Time Effects (θ^C)

**Question**: What is the average effect at a specific calendar time?

**Definition**:
$$
\theta^C(t) = \sum_{g \in \mathcal{G}} \mathbf{1}\{g \leq t\} \cdot ATT(g,t) \cdot P(G = g | G \leq t)
$$

**Interpretation**: The average treatment effect at calendar time $t$ across all cohorts that have been treated by time $t$.

#### 3.5 Event-Time Aggregation (Simplified)

$$
ATT(e) = \sum_g w_g \cdot ATT(g, g+e)
$$

Where $w_g$ is the share of cohort $g$ among cohorts that can be observed at exposure $e$.

### Implementation

```python
from did_estimator import estimate_did_staggered

result = estimate_did_staggered(
    data=df,
    outcome="y",
    treatment_time="first_treated",  # When unit was first treated
    unit_id="id",
    time_id="year",
    control_group="nevertreated",  # or "notyettreated"
    anticipation=0,  # Periods to allow for anticipation
    covariates=["x1", "x2"]
)

# Access group-time effects
att_gt = result.diagnostics['att_gt']
for (g, t), att in att_gt.items():
    print(f"ATT({g},{t}) = {att:.4f}")
```

### Control Group Options

| Option | Description | Pros | Cons |
|--------|-------------|------|------|
| `nevertreated` | Only never-treated units | Avoids contamination | May have limited overlap |
| `notyettreated` | Never-treated + not-yet-treated | More comparison units | May violate no-anticipation |

### Two-Step Semiparametric Estimation

Callaway & Sant'Anna (2021) propose a two-step estimation procedure:

**Step 1: Estimate Generalized Propensity Score**

For each cohort $g$, estimate:
$$
\hat{p}_g(X) = P(G_g = 1 | X, G_g + C = 1)
$$

Using logistic regression or other flexible methods on the subsample of cohort $g$ and never-treated units.

**Step 2: Compute ATT(g,t) Estimates**

Using the estimated propensity scores, compute the sample analog of the identification formula:

$$
\widehat{ATT}(g,t) = \frac{1}{n} \sum_{i=1}^{n} \left(\frac{G_{g,i}}{\hat{p}_g} - \frac{\frac{\hat{p}_g(X_i) \cdot C_i}{1-\hat{p}_g(X_i)}}{\frac{1}{n}\sum_j \frac{\hat{p}_g(X_j) \cdot C_j}{1-\hat{p}_g(X_j)}}\right)(Y_{i,t} - Y_{i,g-\delta-1})
$$

Where $\hat{p}_g = \frac{1}{n}\sum_i G_{g,i}$ is the sample share of cohort $g$.

### Inference: Multiplier Bootstrap

Standard errors and confidence intervals are computed using the **multiplier bootstrap**, which is computationally efficient and valid for simultaneous inference across multiple ATT(g,t) parameters.

**Procedure**:
1. Compute the influence function $\psi_i(g,t)$ for each observation
2. Generate i.i.d. weights $\xi_i \sim N(0,1)$ (or Rademacher)
3. Compute bootstrap statistic: $T_b = \frac{1}{\sqrt{n}} \sum_i \xi_i \cdot \psi_i(g,t)$
4. Repeat $B$ times to obtain bootstrap distribution
5. Use bootstrap quantiles for confidence bands

**Pointwise vs. Simultaneous Confidence Bands**:

| Type | Use Case | Coverage |
|------|----------|----------|
| Pointwise | Individual ATT(g,t) | 95% for each parameter separately |
| Simultaneous | Multiple ATT(g,t) jointly | 95% that ALL parameters are covered |

**Python Implementation**:
```python
def multiplier_bootstrap_inference(
    att_gt, influence_functions, n_bootstrap=1000, alpha=0.05
):
    """
    Compute simultaneous confidence bands via multiplier bootstrap.

    Parameters
    ----------
    att_gt : dict
        Point estimates {(g,t): estimate}
    influence_functions : dict
        {(g,t): array of influence function values}
    n_bootstrap : int
        Number of bootstrap replications
    alpha : float
        Significance level

    Returns
    -------
    dict
        Confidence bands for each (g,t)
    """
    import numpy as np

    n = len(list(influence_functions.values())[0])
    gt_pairs = list(att_gt.keys())

    # Store bootstrap statistics
    boot_stats = {gt: [] for gt in gt_pairs}

    for b in range(n_bootstrap):
        # Generate multiplier weights
        xi = np.random.normal(0, 1, n)

        for gt in gt_pairs:
            psi = influence_functions[gt]
            t_b = np.sum(xi * psi) / np.sqrt(n)
            boot_stats[gt].append(t_b)

    # Compute critical value for simultaneous inference
    # Use supremum of t-statistics across all parameters
    sup_stats = []
    for b in range(n_bootstrap):
        max_t = max(abs(boot_stats[gt][b]) for gt in gt_pairs)
        sup_stats.append(max_t)

    c_alpha = np.quantile(sup_stats, 1 - alpha)

    # Construct confidence bands
    results = {}
    for gt in gt_pairs:
        se = np.std(boot_stats[gt])
        est = att_gt[gt]
        results[gt] = {
            'estimate': est,
            'se': se,
            'ci_pointwise': (est - 1.96 * se, est + 1.96 * se),
            'ci_simultaneous': (est - c_alpha * se, est + c_alpha * se)
        }

    return results
```

### When to Use

- Staggered treatment adoption
- Heterogeneous treatment effects suspected
- Dynamic treatment effects
- Want to avoid TWFE bias

---

## 4. Sun and Abraham (2021) Interaction-Weighted Estimator

### Concept

Saturate the TWFE model with cohort-specific relative time indicators to avoid bad comparisons.

### Model

$$
Y_{it} = \alpha_i + \lambda_t + \sum_g \sum_{e \neq -1} \delta_{ge} \cdot \mathbf{1}[G_i = g] \cdot \mathbf{1}[K_{it} = e] + \varepsilon_{it}
$$

Where:
- $G_i$ = cohort of unit $i$
- $K_{it}$ = event time (time relative to treatment)
- $e = -1$ is the reference period

### Aggregation

$$
\hat{\delta}_e = \sum_g \hat{w}_g \hat{\delta}_{ge}
$$

### Implementation

```python
from scripts.robustness_checks import sun_abraham_estimator

result = sun_abraham_estimator(
    data=df,
    outcome="y",
    treatment_time="first_treated",
    unit_id="id",
    time_id="year",
    pre_periods=4,
    post_periods=4,
    reference_period=-1
)
```

### When to Use

- Want to visualize dynamic effects
- Need cohort-specific treatment effects
- Event study with staggered adoption

---

## 5. de Chaisemartin and D'Haultfoeuille (2020)

### Concept

Identify "switching" units (those whose treatment status changes) and use appropriate comparisons.

### DID_M Estimator

Uses only clean comparisons:
- Newly treated vs. not-yet-treated
- Newly untreated vs. still-treated

### Implementation

```python
# Requires did_multiplegt package
# See: https://github.com/chaisemartinPackages/did_multiplegt

from scripts.robustness_checks import dechaisemartin_dhaultfoeuille

result = dechaisemartin_dhaultfoeuille(
    data=df,
    outcome="y",
    treatment="treated",
    unit_id="id",
    time_id="year",
    controls=["x1", "x2"]
)
```

### When to Use

- Treatment turns on AND off (not just adoption)
- Want minimal assumptions
- Concerned about treatment effect heterogeneity

---

## 6. Borusyak, Jaravel, and Spiess (2024) Imputation Estimator

### Concept

Impute counterfactual outcomes for treated units using untreated observations, then compare actual to imputed.

### Steps

1. Estimate fixed effects model on untreated observations only
2. Impute $\hat{Y}_{it}(0)$ for treated unit-periods
3. Calculate $\hat{\tau}_{it} = Y_{it} - \hat{Y}_{it}(0)$
4. Aggregate to desired level

### Advantages

- Efficient use of data
- Flexible heterogeneity
- Natural connection to synthetic control

### Implementation

```python
from scripts.robustness_checks import imputation_estimator

result = imputation_estimator(
    data=df,
    outcome="y",
    treatment="treated",
    unit_id="id",
    time_id="year",
    controls=["x1", "x2"],
    horizon=4  # Post-treatment periods to estimate
)
```

---

## 7. Event Study Estimation

### Model Specification

$$
Y_{it} = \alpha_i + \lambda_t + \sum_{k=-K_{pre}}^{K_{post}} \beta_k \cdot D_{it}^k + X_{it}'\gamma + \varepsilon_{it}
$$

Where:
- $D_{it}^k$ = indicator for being $k$ periods from treatment
- $\beta_{-1} = 0$ (normalization)

### Implementation

```python
from did_estimator import event_study_plot

fig = event_study_plot(
    data=df,
    outcome="y",
    treatment_time_var="first_treated",
    unit_id="id",
    time_id="year",
    reference_period=-1,
    pre_periods=4,
    post_periods=4,
    cluster="id",
    title="Event Study: Treatment Effects Over Time"
)
```

### Interpreting Event Study

| Pattern | Interpretation |
|---------|----------------|
| Flat pre-trend at 0 | Supports parallel trends |
| Trending pre-coefficients | Parallel trends concern |
| Jump at t=0 | Immediate treatment effect |
| Gradual increase post | Treatment effect builds over time |
| Decay after initial effect | Treatment effect fades |

### Binning Endpoints

For distant event-times with few observations:

```python
# Bin endpoints to avoid noise
def create_event_study_vars(df, treatment_time, time_id, pre_trim=-3, post_trim=3):
    """Create binned event study indicators."""
    df = df.copy()
    df['event_time'] = df[time_id] - df[treatment_time]

    # Bin early and late periods
    df['event_time_binned'] = df['event_time'].clip(lower=pre_trim, upper=post_trim)

    return df
```

---

## 8. Comparison of Estimators

### Decision Guide

```
                              Is treatment timing staggered?
                                    /           \
                                  YES            NO
                                   |              |
                    Are effects likely heterogeneous?     Use TWFE
                           /              \
                         YES              NO
                          |                |
              Use Callaway-Sant'Anna    TWFE may be OK
              or Sun-Abraham            but check Bacon
                                        decomposition
```

### Summary Table

| Estimator | Staggered OK? | Heterogeneity OK? | Dynamic Effects? | Complexity |
|-----------|--------------|-------------------|-----------------|------------|
| 2x2 DID | No | N/A | No | Low |
| TWFE | Caution | No | Limited | Low |
| Callaway-Sant'Anna | Yes | Yes | Yes | Medium |
| Sun-Abraham | Yes | Yes | Yes | Medium |
| de Chaisemartin-D'Haultfoeuille | Yes | Yes | Yes | Medium |
| Imputation | Yes | Yes | Yes | Medium |

---

## 9. Implementation Recommendations

### Standard Workflow

1. **Start Simple**: Run basic TWFE to understand the data
2. **Check for Issues**: Run Bacon decomposition if staggered
3. **Run Robust Estimator**: Callaway-Sant'Anna or similar
4. **Compare Results**: Large differences indicate heterogeneity/dynamics
5. **Report Both**: Show TWFE and robust estimator for transparency

### Code Template

```python
from did_estimator import (
    estimate_did_panel,
    estimate_did_staggered,
    event_study_plot
)
from scripts.robustness_checks import bacon_decomposition

# Step 1: Basic TWFE
result_twfe = estimate_did_panel(
    data=df, outcome="y", treatment="treated",
    unit_id="id", time_id="year", cluster="id"
)

# Step 2: Check Bacon decomposition (if staggered)
bacon = bacon_decomposition(df, "y", "treated", "id", "year")
print(f"Problematic weight share: {bacon['bad_comparison_weight']:.3f}")

# Step 3: Robust estimator
result_cs = estimate_did_staggered(
    data=df, outcome="y", treatment_time="first_treated",
    unit_id="id", time_id="year", control_group="nevertreated"
)

# Step 4: Compare
print(f"TWFE estimate: {result_twfe.effect:.4f} (SE={result_twfe.se:.4f})")
print(f"C-S estimate: {result_cs.effect:.4f} (SE={result_cs.se:.4f})")
print(f"Difference: {result_twfe.effect - result_cs.effect:.4f}")

# Step 5: Event study for dynamics
fig = event_study_plot(
    data=df, outcome="y", treatment_time_var="first_treated",
    unit_id="id", time_id="year"
)
fig.savefig("event_study.png")
```

---

## References

### Core Methodology

1. Callaway, B., & Sant'Anna, P. H. (2021). "Difference-in-Differences with Multiple Time Periods." *Journal of Econometrics*.

2. Sun, L., & Abraham, S. (2021). "Estimating Dynamic Treatment Effects in Event Studies with Heterogeneous Treatment Effects." *Journal of Econometrics*.

3. de Chaisemartin, C., & D'Haultfoeuille, X. (2020). "Two-Way Fixed Effects Estimators with Heterogeneous Treatment Effects." *American Economic Review*.

4. Goodman-Bacon, A. (2021). "Difference-in-Differences with Variation in Treatment Timing." *Journal of Econometrics*.

5. Borusyak, K., Jaravel, X., & Spiess, J. (2024). "Revisiting Event Study Designs: Robust and Efficient Estimation." *Review of Economic Studies*.

### Software

- Python: `linearmodels`, `pyfixest`
- R: `did`, `fixest`, `did2s`
- Stata: `csdid`, `did_multiplegt`, `eventstudyinteract`

---

## See Also

- [SKILL.md](../SKILL.md) - Main skill documentation
- [did_estimator.py](../did_estimator.py) - Python implementation
- [diagnostic_tests.md](diagnostic_tests.md) - How to test which method is appropriate
