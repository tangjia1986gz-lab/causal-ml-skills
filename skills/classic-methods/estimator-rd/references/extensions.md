# RD Extensions

> **Document Type**: Advanced Methods Reference
> **Last Updated**: 2024-01
> **Purpose**: Extensions and variations of the standard RD design

## Overview

This document covers extensions to the basic sharp and fuzzy RD designs, including specialized approaches for particular data structures and research questions.

---

## 1. Fuzzy RD in Detail

### When Sharp Becomes Fuzzy

Standard sharp RD assumes deterministic treatment:
$$D_i = \mathbf{1}(X_i \geq c)$$

Fuzzy RD relaxes this to probabilistic treatment:
$$P(D_i = 1 | X_i = x) \text{ is discontinuous at } c$$

### Sources of Fuzziness

| Source | Example | Implication |
|--------|---------|-------------|
| **Non-compliance** | Students above cutoff don't enroll | Need to measure actual treatment |
| **Discretion** | Eligibility determined with discretion | May include manipulation |
| **Measurement lag** | Score measured before treatment decision | Running variable may change |
| **Multiple thresholds** | Program has additional criteria | Crossing running variable cutoff isn't sufficient |

### Estimation

The fuzzy RD estimator is a local Wald (IV) estimator:

$$
\hat{\tau}_{FRD} = \frac{\hat{\mu}_{Y+} - \hat{\mu}_{Y-}}{\hat{\mu}_{D+} - \hat{\mu}_{D-}}
$$

Where:
- Numerator: Reduced form (jump in outcome)
- Denominator: First stage (jump in treatment probability)

### Implementation

```python
from rd_estimator import estimate_fuzzy_rd

# Estimate fuzzy RD
result = estimate_fuzzy_rd(
    data=df,
    running="score",
    outcome="earnings",
    treatment="enrolled",  # Actual treatment received
    cutoff=0.0,
    bandwidth=None  # Auto-select
)

print(f"LATE at cutoff: {result.effect:.4f}")
print(f"First stage: {result.diagnostics['first_stage']:.4f}")
print(f"Reduced form: {result.diagnostics['reduced_form']:.4f}")
```

### Interpreting Fuzzy RD

**Compliers**: Units who would be treated above the cutoff and untreated below.

The fuzzy RD estimate is:
$$\tau_{FRD} = E[Y(1) - Y(0) | \text{Complier at } c]$$

**Key Points**:
- Effect is for compliers only, not all units
- First stage must be sufficiently strong (> 0.05)
- Monotonicity assumption required (no defiers)

---

## 2. Kink RD (Regression Kink Design)

### Concept

Instead of a discontinuity in the LEVEL of treatment, there's a discontinuity in the SLOPE (derivative) of treatment.

**Example**: Tax rate changes at income threshold
- Below threshold: marginal tax rate = 20%
- Above threshold: marginal tax rate = 30%

The treatment (tax liability) is continuous, but its derivative with respect to income is discontinuous.

### Identification

The kink RD estimand:
$$
\tau_{RK} = \frac{\lim_{x \downarrow c} \frac{\partial E[Y|X=x]}{\partial x} - \lim_{x \uparrow c} \frac{\partial E[Y|X=x]}{\partial x}}{\lim_{x \downarrow c} \frac{\partial E[D|X=x]}{\partial x} - \lim_{x \uparrow c} \frac{\partial E[D|X=x]}{\partial x}}
$$

### Implementation

```python
def estimate_kink_rd(
    data: pd.DataFrame,
    running: str,
    outcome: str,
    treatment: str,
    cutoff: float,
    bandwidth: float = None
) -> dict:
    """
    Estimate Regression Kink Design.

    Parameters
    ----------
    data : pd.DataFrame
        Dataset
    running : str
        Running variable
    outcome : str
        Outcome variable
    treatment : str
        Treatment variable (continuous)
    cutoff : float
        Kink point

    Returns
    -------
    dict
        Kink RD estimates
    """
    from rd_estimator import select_bandwidth

    x = data[running].values
    y = data[outcome].values
    d = data[treatment].values

    if bandwidth is None:
        bandwidth = select_bandwidth(x, y, cutoff)

    # Fit local quadratic on each side (need derivative estimates)
    below = (x >= cutoff - bandwidth) & (x < cutoff)
    above = (x >= cutoff) & (x < cutoff + bandwidth)

    # Below cutoff
    x_b = x[below] - cutoff
    y_b = y[below]
    d_b = d[below]

    # Above cutoff
    x_a = x[above] - cutoff
    y_a = y[above]
    d_a = d[above]

    # Fit quadratic: y = a + bx + cx^2
    # Derivative at cutoff = b
    from numpy.polynomial import polynomial as P

    # Outcome derivatives
    coef_y_below = np.polyfit(x_b, y_b, 2)
    coef_y_above = np.polyfit(x_a, y_a, 2)
    slope_y_below = coef_y_below[1]  # Linear coefficient
    slope_y_above = coef_y_above[1]

    # Treatment derivatives
    coef_d_below = np.polyfit(x_b, d_b, 2)
    coef_d_above = np.polyfit(x_a, d_a, 2)
    slope_d_below = coef_d_below[1]
    slope_d_above = coef_d_above[1]

    # Kink RD estimate
    numerator = slope_y_above - slope_y_below
    denominator = slope_d_above - slope_d_below

    if abs(denominator) > 1e-10:
        kink_estimate = numerator / denominator
    else:
        kink_estimate = np.nan

    return {
        'estimate': kink_estimate,
        'slope_y_change': numerator,
        'slope_d_change': denominator,
        'bandwidth': bandwidth
    }
```

### Applications

- Tax policy (Nielsen, Sorensen & Taber, 2010)
- Unemployment insurance (Card et al., 2015)
- Social security (Gelber et al., 2017)

---

## 3. Geographic RD

### Concept

Treatment is determined by geographic location relative to a boundary (border, district line, etc.).

**Examples**:
- School district boundaries
- State/country borders
- Electoral district lines
- Zoning boundaries

### Key Challenges

| Challenge | Description | Solution |
|-----------|-------------|----------|
| **Running variable** | Distance to boundary, not single dimension | Use 2D distance or boundary segments |
| **Manipulation** | People can move (self-selection) | Test for sorting, use pre-treatment location |
| **Spillovers** | Effects may cross boundary | Use donut approach, test for spatial correlation |
| **Multiple boundaries** | Same area has many borders | Clearly define which boundary is relevant |

### Distance Measures

```python
import numpy as np

def distance_to_boundary(lat, lon, boundary_coords):
    """
    Calculate minimum distance to a boundary.

    Parameters
    ----------
    lat, lon : float
        Point coordinates
    boundary_coords : list of tuples
        [(lat1, lon1), (lat2, lon2), ...] boundary points

    Returns
    -------
    float
        Signed distance (negative = one side, positive = other)
    """
    from shapely.geometry import Point, LineString

    point = Point(lon, lat)
    boundary = LineString(boundary_coords)

    distance = point.distance(boundary)

    # Determine sign based on which side of boundary
    # (Implementation depends on boundary orientation)

    return distance
```

### Implementation Approaches

**Approach 1: Distance-based RD**
```python
# Calculate distance to boundary for each observation
df['distance_to_border'] = df.apply(
    lambda row: distance_to_boundary(row['lat'], row['lon'], boundary),
    axis=1
)

# Standard RD with distance as running variable
result = estimate_sharp_rd(
    data=df,
    running='distance_to_border',
    outcome='y',
    cutoff=0.0  # Boundary
)
```

**Approach 2: Boundary Segment Fixed Effects**
```python
# Assign each observation to nearest boundary segment
df['segment'] = assign_to_segment(df['lat'], df['lon'], boundary_segments)

# RD with segment fixed effects
result = estimate_sharp_rd(
    data=df,
    running='distance_to_border',
    outcome='y',
    cutoff=0.0,
    covariates=['segment_fe_1', 'segment_fe_2', ...]
)
```

### Geographic RD Validity Checks

```python
# 1. Test for sorting at boundary
# Compare population density on each side
density_below = len(df[df['distance_to_border'] < 0]) / area_below
density_above = len(df[df['distance_to_border'] >= 0]) / area_above

# 2. Test pre-treatment characteristics
geographic_balance = covariate_balance_rd(
    data=df,
    running='distance_to_border',
    cutoff=0.0,
    covariates=['pre_income', 'pre_population', 'land_value']
)

# 3. Test for spatial autocorrelation
from scipy.spatial.distance import cdist
# Use Moran's I or similar spatial correlation test
```

---

## 4. Multi-Cutoff RD

### Concept

Treatment is determined by multiple cutoffs of the same running variable, or different cutoffs apply to different populations.

**Examples**:
- Test score thresholds vary by school
- Income cutoffs differ by family size
- Age thresholds vary by birth cohort

### Estimation Strategies

**Strategy 1: Normalize and Pool**
```python
# Normalize running variable relative to each cutoff
df['normalized_score'] = df.apply(
    lambda row: row['score'] - get_cutoff(row['group']),
    axis=1
)

# Pool and estimate with group fixed effects
result = estimate_sharp_rd(
    data=df,
    running='normalized_score',
    outcome='y',
    cutoff=0.0,  # Normalized cutoff
    covariates=['group_fe_1', 'group_fe_2']  # Group fixed effects
)
```

**Strategy 2: Estimate Separately and Combine**
```python
from scipy import stats

results_by_cutoff = {}
effects = []
ses = []

for cutoff in unique_cutoffs:
    subset = df[df['applicable_cutoff'] == cutoff]

    result = estimate_sharp_rd(
        data=subset,
        running='score',
        outcome='y',
        cutoff=cutoff
    )

    results_by_cutoff[cutoff] = result
    effects.append(result.effect)
    ses.append(result.se)

# Inverse variance weighted average
weights = [1/se**2 for se in ses]
pooled_effect = np.average(effects, weights=weights)
pooled_se = np.sqrt(1 / sum(weights))

print(f"Pooled effect: {pooled_effect:.4f} (SE: {pooled_se:.4f})")
```

### Testing Heterogeneity

```python
# Test if effects differ across cutoffs
from scipy.stats import chi2

effects = np.array(effects)
ses = np.array(ses)
weights = 1 / ses**2

# Cochran's Q test for heterogeneity
pooled = np.average(effects, weights=weights)
Q = np.sum(weights * (effects - pooled)**2)
df_test = len(effects) - 1
p_value = 1 - chi2.cdf(Q, df_test)

print(f"Heterogeneity test: Q = {Q:.2f}, p = {p_value:.4f}")
```

---

## 5. Multi-Dimensional RD

### Concept

Treatment determined by multiple running variables, each with its own cutoff.

**Example**: College admission based on both GPA >= 3.0 AND SAT >= 1200.

### Types of Multi-Dimensional RD

| Type | Rule | Challenge |
|------|------|-----------|
| **Intersection** | $X_1 \geq c_1$ AND $X_2 \geq c_2$ | Complex boundary |
| **Union** | $X_1 \geq c_1$ OR $X_2 \geq c_2$ | Multiple margins |
| **Weighted** | $w_1 X_1 + w_2 X_2 \geq c$ | Single combined score |

### Estimation for Intersection Rule

```python
def multidimensional_rd(
    data: pd.DataFrame,
    running1: str,
    running2: str,
    cutoff1: float,
    cutoff2: float,
    outcome: str
) -> dict:
    """
    Estimate multi-dimensional RD with intersection rule.

    Uses distance to boundary as the running variable.
    """
    df = data.copy()

    # Calculate distance to each cutoff
    df['dist1'] = df[running1] - cutoff1
    df['dist2'] = df[running2] - cutoff2

    # Treatment: above BOTH cutoffs
    df['treated'] = (df['dist1'] >= 0) & (df['dist2'] >= 0)

    # Running variable: minimum distance to the boundary
    # (Distance to the binding constraint)
    df['distance_to_boundary'] = np.where(
        df['dist1'] < df['dist2'],
        df['dist1'],
        df['dist2']
    )

    # Identify which dimension is binding for each unit
    df['binding_dim'] = np.where(df['dist1'] < df['dist2'], 1, 2)

    # Estimate separately for each binding dimension
    results = {}

    for dim in [1, 2]:
        subset = df[df['binding_dim'] == dim]
        dist_col = f'dist{dim}'

        if len(subset) > 50:  # Minimum sample
            result = estimate_sharp_rd(
                data=subset,
                running=dist_col,
                outcome=outcome,
                cutoff=0.0
            )
            results[f'dimension_{dim}'] = result

    return results
```

---

## 6. RD with Covariates

### When to Use Covariates

**Potential Benefits**:
1. Precision improvement (reduce residual variance)
2. Address local imbalances (though RD shouldn't need this)

**Cautions**:
1. Not necessary for identification
2. Can introduce overfitting in small samples
3. Only use pre-treatment covariates

### Implementation

```python
# Covariate-adjusted RD (if the estimator supports it)
result_adjusted = estimate_sharp_rd(
    data=df,
    running="score",
    outcome="y",
    cutoff=0.0,
    covariates=["pre_treatment_x1", "pre_treatment_x2"]
)

# Compare with unadjusted
result_unadjusted = estimate_sharp_rd(
    data=df,
    running="score",
    outcome="y",
    cutoff=0.0
)

print(f"Unadjusted: {result_unadjusted.effect:.4f} (SE: {result_unadjusted.se:.4f})")
print(f"Adjusted: {result_adjusted.effect:.4f} (SE: {result_adjusted.se:.4f})")
```

### Covariate Selection

```python
# Only include covariates that:
# 1. Are determined before treatment
# 2. Pass balance tests (not significantly discontinuous at cutoff)

covariates = ['age', 'gender', 'prior_score']

# Test each
valid_covariates = []
for cov in covariates:
    balance = estimate_sharp_rd(df, 'score', cov, 0.0)
    if balance.p_value > 0.10:  # Not significantly imbalanced
        valid_covariates.append(cov)

print(f"Valid covariates for adjustment: {valid_covariates}")
```

---

## 7. Local Randomization Inference

### Concept

Alternative to local polynomial methods. Assumes that within a narrow window around the cutoff, treatment is as-if randomly assigned.

### When to Use

- Discrete running variable with few values
- Very narrow window of interest
- Want finite-sample exact inference
- As sensitivity analysis

### Implementation

```python
def local_randomization_rd(
    data: pd.DataFrame,
    running: str,
    outcome: str,
    cutoff: float,
    window: float
) -> dict:
    """
    Local randomization RD inference.

    Within the window, treatment is assumed random.
    Uses permutation inference.
    """
    # Subset to window
    in_window = (data[running] >= cutoff - window) & (data[running] < cutoff + window)
    df_window = data[in_window].copy()

    # Treatment indicator
    df_window['treated'] = (df_window[running] >= cutoff).astype(int)

    # Observed difference in means
    treated = df_window[df_window['treated'] == 1][outcome]
    control = df_window[df_window['treated'] == 0][outcome]
    obs_diff = treated.mean() - control.mean()

    # Permutation test
    n_permutations = 10000
    y = df_window[outcome].values
    n_treated = df_window['treated'].sum()
    n_total = len(df_window)

    perm_diffs = []
    for _ in range(n_permutations):
        perm = np.random.permutation(n_total)
        perm_treated = y[perm[:n_treated]]
        perm_control = y[perm[n_treated:]]
        perm_diffs.append(perm_treated.mean() - perm_control.mean())

    perm_diffs = np.array(perm_diffs)

    # Two-sided p-value
    p_value = np.mean(np.abs(perm_diffs) >= np.abs(obs_diff))

    return {
        'effect': obs_diff,
        'p_value': p_value,
        'n_window': n_total,
        'n_treated': n_treated,
        'n_control': n_total - n_treated,
        'window': window
    }

# Example usage
result = local_randomization_rd(
    data=df,
    running='score',
    outcome='y',
    cutoff=0.0,
    window=0.1  # Narrow window for local randomization
)
```

### Window Selection

The window should be narrow enough that local randomization is plausible:
- Covariates should be balanced within window
- Running variable should be approximately uniform within window

```python
# Test different windows
windows = [0.05, 0.10, 0.15, 0.20]

for w in windows:
    # Check balance within window
    in_window = (df[running] >= cutoff - w) & (df[running] < cutoff + w)
    df_w = df[in_window]

    # Covariate balance
    treated = df_w[df_w[running] >= cutoff]
    control = df_w[df_w[running] < cutoff]

    for cov in covariates:
        stat, p = stats.ttest_ind(treated[cov], control[cov])
        print(f"Window {w}, {cov}: p = {p:.3f}")
```

---

## 8. RD with Panel Data

### Concept

Combining RD with multiple time periods allows for:
- Dynamic treatment effects
- Additional robustness checks
- Difference-in-RD designs

### Dynamic RD Effects

```python
def dynamic_rd_effects(
    data: pd.DataFrame,
    running: str,
    outcome_prefix: str,
    cutoff: float,
    periods: list
) -> dict:
    """
    Estimate RD effects at multiple time periods.
    """
    results = {}

    for t in periods:
        outcome = f"{outcome_prefix}_{t}"

        if outcome in data.columns:
            result = estimate_sharp_rd(
                data=data,
                running=running,
                outcome=outcome,
                cutoff=cutoff
            )
            results[t] = {
                'effect': result.effect,
                'se': result.se,
                'p_value': result.p_value
            }

    return results

# Example: Effect on outcomes in years 0, 1, 2, 3 after treatment
dynamic = dynamic_rd_effects(
    data=df,
    running='score_at_assignment',
    outcome_prefix='outcome_year',
    cutoff=0.0,
    periods=[0, 1, 2, 3]
)

for t, res in dynamic.items():
    print(f"Year {t}: Effect = {res['effect']:.4f} (SE = {res['se']:.4f})")
```

### Difference-in-RD

Combines RD with a before-after comparison:

```python
def difference_in_rd(
    data: pd.DataFrame,
    running: str,
    outcome_before: str,
    outcome_after: str,
    cutoff: float
) -> dict:
    """
    Difference-in-RD: Difference out pre-treatment RD (should be zero).
    """
    # RD on before outcome (should be null)
    rd_before = estimate_sharp_rd(data, running, outcome_before, cutoff)

    # RD on after outcome
    rd_after = estimate_sharp_rd(data, running, outcome_after, cutoff)

    # Difference-in-RD
    diff_rd = rd_after.effect - rd_before.effect
    diff_se = np.sqrt(rd_after.se**2 + rd_before.se**2)  # Assumes independence

    return {
        'rd_before': rd_before.effect,
        'rd_after': rd_after.effect,
        'difference': diff_rd,
        'se': diff_se,
        'p_value': 2 * (1 - stats.norm.cdf(abs(diff_rd / diff_se)))
    }
```

---

## Summary Table of Extensions

| Extension | Use Case | Key Additional Assumption | Implementation Complexity |
|-----------|----------|--------------------------|---------------------------|
| Fuzzy RD | Imperfect compliance | Monotonicity | Low |
| Kink RD | Treatment intensity changes at cutoff | Smoothness of higher derivatives | Medium |
| Geographic RD | Spatial treatment boundaries | No sorting across boundary | Medium-High |
| Multi-cutoff RD | Different thresholds for different groups | Homogeneous effects (if pooling) | Medium |
| Multi-dimensional RD | Multiple eligibility criteria | Correct boundary specification | High |
| Local Randomization | Discrete running variable | Local random assignment | Low |
| Panel RD | Multiple time periods | No anticipation | Medium |

---

## References

- Card, D., Lee, D. S., Pei, Z., & Weber, A. (2015). Inference on causal effects in a generalized regression kink design. *Econometrica*, 83(6), 2453-2483.
- Dell, M. (2010). The persistent effects of Peru's mining mita. *Econometrica*, 78(6), 1863-1903.
- Cattaneo, M. D., Keele, L., Titiunik, R., & Vazquez-Bare, G. (2016). Interpreting regression discontinuity designs with multiple cutoffs. *Journal of Politics*, 78(4), 1229-1248.
- Cattaneo, M. D., Frandsen, B. R., & Titiunik, R. (2015). Randomization inference in the regression discontinuity design. *Journal of Causal Inference*, 3(1), 1-24.
