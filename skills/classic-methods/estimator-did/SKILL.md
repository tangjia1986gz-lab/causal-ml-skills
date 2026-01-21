---
name: estimator-did
description: Use when estimating causal effects with Difference-in-Differences. Triggers on DID, diff-in-diff, difference-in-differences, parallel trends, panel data treatment effects.
---

# Estimator: Difference-in-Differences (DID)

> **Version**: 1.0.0 | **Type**: Estimator
> **Aliases**: DID, Diff-in-Diff, DD, Double Difference

## Overview

Difference-in-Differences (DID) estimates causal effects by comparing changes in outcomes over time between a treatment group and a control group. The method removes time-invariant unobserved confounders by differencing, while the control group accounts for common time trends.

**Key Identification Assumption**: In the absence of treatment, the treatment and control groups would have followed parallel trends in the outcome variable.

## When to Use

### Ideal Scenarios
- Policy evaluation where treatment timing varies across units
- Natural experiments with clear pre/post treatment periods
- Program effects when randomization is not feasible
- Settings where selection into treatment is based on time-invariant characteristics

### Data Requirements
- [ ] Panel data with at least one pre-treatment and one post-treatment period
- [ ] Clear treatment and control groups identified
- [ ] Binary or staggered treatment adoption
- [ ] Outcome variable measured consistently across time
- [ ] Sufficient pre-treatment periods for parallel trends testing (recommended: 3+)

### When NOT to Use
- Parallel trends clearly violated in pre-treatment data -> Consider `estimator-synthetic-control`
- Treatment timing varies continuously -> Consider `estimator-event-study`
- Strong anticipation effects present -> Consider `estimator-rdid` (RD + DID)
- Treatment and control groups fundamentally different -> Consider `estimator-psm` (with DID)

## Identification Assumptions

| Assumption | Description | Testable? |
|------------|-------------|-----------|
| **Parallel Trends** | Treatment and control groups would have followed the same trend absent treatment | Partially (pre-trends) |
| **No Anticipation** | Treatment effect does not occur before actual treatment | Yes (pre-period coefficients) |
| **SUTVA** | Treatment of one unit does not affect outcomes of others | No (domain knowledge) |
| **No Composition Changes** | Group composition remains stable over time | Yes (balance tests) |

---

## Workflow

```
+-------------------------------------------------------------+
|                    DID ESTIMATOR WORKFLOW                     |
+-------------------------------------------------------------+
|  1. SETUP          -> Define treatment, time, outcome vars   |
|  2. PRE-ESTIMATION -> Test parallel trends, visualize        |
|  3. ESTIMATION     -> 2x2 DID, Panel DID, Staggered DID      |
|  4. DIAGNOSTICS    -> Placebo tests, event study, robustness |
|  5. REPORTING      -> Generate tables & interpretation       |
+-------------------------------------------------------------+
```

### Phase 1: Setup

**Objective**: Prepare panel data and define model specification

**Inputs Required**:
```python
# Standard CausalInput structure for DID
outcome = "y"                    # Outcome variable name
treatment = "treated"            # Treatment indicator (0/1 or treatment timing)
unit_id = "id"                   # Panel: unit identifier
time_id = "year"                 # Panel: time identifier
treatment_time = 2015            # When treatment began (or column with unit-specific timing)
controls = ["x1", "x2"]          # Optional control variables
```

**Data Validation Checklist**:
- [ ] Panel is balanced or explicitly handle unbalanced panels
- [ ] No missing values in key variables (or document handling strategy)
- [ ] Treatment indicator is correctly coded (0 pre-treatment, 1 post-treatment for treated units)
- [ ] Sufficient observations in both treatment and control groups
- [ ] Multiple pre-treatment periods available for trend testing

**Data Structure Verification**:
```python
from did_estimator import validate_did_data

# Check data structure
validation = validate_did_data(
    data=df,
    outcome="y",
    treatment="treated",
    unit_id="id",
    time_id="year"
)
print(validation.summary())
```

### Phase 2: Pre-Estimation Checks

**Parallel Trends Test: Visual and Statistical**

```python
from did_estimator import test_parallel_trends, plot_parallel_trends

# Visual inspection (ALWAYS do this first)
fig = plot_parallel_trends(
    data=df,
    outcome="y",
    treatment_group="treatment_group",  # 0/1 indicator for ever-treated
    time_id="year",
    treatment_time=2015
)

# Statistical test
trends_result = test_parallel_trends(
    data=df,
    outcome="y",
    treatment_group="treatment_group",
    time_id="year",
    unit_id="id",
    treatment_time=2015,
    n_pre_periods=4
)

print(trends_result)
```

**Interpretation**:
- PASS if: Pre-treatment coefficients are jointly insignificant (F-test p > 0.05) AND visual inspection shows parallel paths
- WARNING if: Some individual pre-period coefficients marginally significant but joint test passes
- FAIL if: Significant divergence in pre-treatment trends - DO NOT proceed without addressing

**Balance Test: Covariate Distribution**

```python
from did_estimator import balance_test_did

balance = balance_test_did(
    data=df,
    treatment_group="treatment_group",
    covariates=["x1", "x2", "x3"],
    time_id="year",
    pre_period=2014  # Last pre-treatment period
)
print(balance.summary_table)
```

---

### Phase 3: Main Estimation

**Model Specification**:

**Classic 2x2 DID**:
$$
Y_{it} = \alpha + \beta_1 \cdot \text{Treated}_i + \beta_2 \cdot \text{Post}_t + \delta \cdot (\text{Treated}_i \times \text{Post}_t) + \epsilon_{it}
$$

Where $\delta$ is the **Average Treatment Effect on the Treated (ATT)**.

**Panel DID with Two-Way Fixed Effects (TWFE)**:
$$
Y_{it} = \alpha_i + \gamma_t + \delta \cdot D_{it} + X_{it}'\beta + \epsilon_{it}
$$

Where:
- $\alpha_i$: Unit fixed effects
- $\gamma_t$: Time fixed effects
- $D_{it}$: Treatment indicator (1 if unit $i$ is treated at time $t$)
- $\delta$: **ATT** (under parallel trends)

**Python Implementation**:

```python
from did_estimator import (
    estimate_did_2x2,
    estimate_did_panel,
    estimate_did_staggered,
    run_full_did_analysis
)

# Option 1: Classic 2x2 DID
result_2x2 = estimate_did_2x2(
    data=df,
    outcome="y",
    treatment_group="treatment_group",
    post="post",
    controls=["x1", "x2"]
)

# Option 2: Panel DID with TWFE
result_panel = estimate_did_panel(
    data=df,
    outcome="y",
    treatment="treated",  # Post-treatment indicator for treated units
    unit_id="id",
    time_id="year",
    controls=["x1", "x2"],
    cluster="id"  # Cluster standard errors at unit level
)

# Option 3: Staggered DID (Callaway-Sant'Anna)
# Use when treatment timing varies across units
result_staggered = estimate_did_staggered(
    data=df,
    outcome="y",
    treatment_time="first_treated",  # Year unit first received treatment
    unit_id="id",
    time_id="year",
    control_group="nevertreated"  # or "notyettreated"
)

# Option 4: Full workflow (recommended)
result = run_full_did_analysis(
    data=df,
    outcome="y",
    treatment="treated",
    unit_id="id",
    time_id="year",
    treatment_time=2015,
    controls=["x1", "x2"],
    cluster="id"
)

print(result.summary_table)
print(result.diagnostics)
```

**Returns**:
```python
CausalOutput(
    effect=2.34,           # Point estimate of ATT
    se=0.45,               # Clustered standard error
    ci_lower=1.46,         # 95% CI lower bound
    ci_upper=3.22,         # 95% CI upper bound
    p_value=0.0001,        # Two-sided p-value
    diagnostics={
        'parallel_trends': DiagnosticResult(...),
        'n_treated': 150,
        'n_control': 350,
        'n_pre_periods': 4,
        'n_post_periods': 3
    },
    summary_table="...",
    interpretation="..."
)
```

### Phase 4: Robustness Checks

| Check | Purpose | Implementation |
|-------|---------|----------------|
| Placebo Test | Validate no pre-treatment effects | `placebo_test()` |
| Event Study | Examine dynamic effects | `event_study_plot()` |
| Alternative Controls | Test specification sensitivity | Re-run with different X |
| Different Clustering | Check SE robustness | Change `cluster` parameter |
| Trimmed Sample | Remove extreme observations | Filter data |

**Placebo Test (Fake Treatment Timing)**:

```python
from did_estimator import placebo_test

# Run placebo with fake treatment 2 years before actual treatment
placebo_result = placebo_test(
    data=df,
    outcome="y",
    treatment_group="treatment_group",
    unit_id="id",
    time_id="year",
    actual_treatment_time=2015,
    placebo_treatment_time=2013  # Should find NO effect here
)

print(placebo_result)
# Expected: Insignificant placebo effect (p > 0.1)
```

**Event Study Plot (Dynamic Effects)**:

```python
from did_estimator import event_study_plot

# Visualize treatment effects over time
fig = event_study_plot(
    data=df,
    outcome="y",
    treatment_time_var="first_treated",
    unit_id="id",
    time_id="year",
    reference_period=-1,  # Normalize to period before treatment
    pre_periods=4,
    post_periods=3
)

fig.savefig("event_study.png")
```

**Interpretation of Event Study**:
- Pre-treatment coefficients should be close to zero and statistically insignificant
- Post-treatment coefficients show dynamic treatment effects
- Increasing post-treatment effects may indicate treatment building over time

### Phase 5: Reporting

**Standard Output Table Format**:

```
+----------------------------------------------------------+
|             Table X: Difference-in-Differences            |
+----------------------------------------------------------+
|                         (1)        (2)        (3)         |
|                        2x2      Panel FE   Staggered      |
+----------------------------------------------------------+
| Treatment Effect      2.34***    2.28***    2.15***       |
|                      (0.45)     (0.42)     (0.48)         |
|                                                           |
| Controls               No         Yes        Yes          |
| Unit FE                No         Yes        Yes          |
| Time FE                No         Yes        Yes          |
| Clustering             No         Unit       Unit         |
|                                                           |
| Observations          2,000      2,000      2,000         |
| R-squared             0.234      0.456      0.445         |
| Pre-trend p-value     0.672      0.672      0.672         |
+----------------------------------------------------------+
| Notes: Robust standard errors in parentheses, clustered   |
| at unit level in columns (2)-(3).                         |
| *** p<0.01, ** p<0.05, * p<0.1                           |
+----------------------------------------------------------+
```

**Interpretation Template**:

```markdown
## Results Interpretation

Using a difference-in-differences design, we estimate that [treatment description]
leads to a [increase/decrease] of **[effect]** (SE = [se]) in [outcome description].

### Identification
The parallel trends assumption is [supported/not rejected] based on:
1. Visual inspection of pre-treatment trends (Figure X)
2. Joint F-test of pre-treatment coefficients (p = [p-value])
3. Individual pre-period coefficients are statistically insignificant

### Robustness
- Placebo test with fake treatment timing: [result]
- Alternative control variables: [result]
- Event study shows [dynamic pattern description]

### Economic Significance
The estimated effect of [magnitude] represents a [X%] change relative to
the pre-treatment mean of [Y_mean] in the treatment group.

### Caveats
- [Any violations or concerns about assumptions]
- [Sample limitations]
- [External validity considerations]
```

---

## Common Mistakes

### 1. Using TWFE with Staggered Treatment Adoption

**Mistake**: Applying standard two-way fixed effects when units are treated at different times.

**Why it's wrong**: TWFE can produce biased estimates when treatment effects are heterogeneous across time or groups, because already-treated units serve as controls for later-treated units.

**Correct approach**:
```python
# WRONG: Standard TWFE with staggered adoption
model = PanelOLS(y ~ treated + EntityEffects + TimeEffects)

# CORRECT: Use Callaway-Sant'Anna or other staggered DID methods
from did_estimator import estimate_did_staggered

result = estimate_did_staggered(
    data=df,
    outcome="y",
    treatment_time="first_treated",
    unit_id="id",
    time_id="year"
)
```

### 2. Ignoring Pre-Trend Violations

**Mistake**: Proceeding with DID despite visual or statistical evidence of non-parallel pre-trends.

**Why it's wrong**: If trends differ pre-treatment, the difference in post-treatment changes is not a valid estimate of the treatment effect.

**Correct approach**:
```python
# ALWAYS check pre-trends first
trends = test_parallel_trends(data, ...)

if not trends.passed:
    print("WARNING: Parallel trends assumption may be violated!")
    print("Consider:")
    print("1. Synthetic control methods")
    print("2. Including group-specific linear trends")
    print("3. Matching on pre-treatment outcomes")
```

### 3. Clustering Standard Errors Incorrectly

**Mistake**: Not clustering standard errors at the level of treatment assignment.

**Why it's wrong**: Observations within the same unit (or treatment cluster) are correlated over time. Ignoring this leads to understated standard errors and over-rejection.

**Correct approach**:
```python
# WRONG: Robust but unclustered
result = estimate_did_panel(data, ..., cluster=None)

# CORRECT: Cluster at unit level (typical)
result = estimate_did_panel(data, ..., cluster="id")

# CORRECT: Cluster at state level (if treatment varies by state)
result = estimate_did_panel(data, ..., cluster="state")
```

### 4. Interpreting Composition Changes as Treatment Effects

**Mistake**: Not accounting for units entering or exiting the sample around treatment.

**Why it's wrong**: If the composition of treatment/control groups changes, differences may reflect sample changes rather than treatment effects.

**Correct approach**:
```python
# Check for balanced panel
n_obs_per_unit = df.groupby('id')['year'].count()
if n_obs_per_unit.nunique() > 1:
    print("WARNING: Unbalanced panel detected")

# Option 1: Restrict to balanced subsample
balanced_units = n_obs_per_unit[n_obs_per_unit == n_obs_per_unit.max()].index
df_balanced = df[df['id'].isin(balanced_units)]

# Option 2: Explicitly model entry/exit
```

---

## Examples

### Example 1: Minimum Wage and Employment (Card & Krueger Style)

**Research Question**: What is the effect of minimum wage increase on employment?

**Data**: Fast food restaurants in NJ (treatment) and PA (control) before/after NJ minimum wage increase.

```python
import pandas as pd
from did_estimator import run_full_did_analysis, plot_parallel_trends

# Load data
data = pd.read_csv("card_krueger.csv")

# Data structure:
# - store_id: restaurant identifier
# - state: NJ (treatment) or PA (control)
# - period: 0 (before) or 1 (after)
# - fte: full-time equivalent employment
# - wage: starting wage
# - chain: restaurant chain (BK, KFC, etc.)

# Create indicators
data['treated'] = ((data['state'] == 'NJ') & (data['period'] == 1)).astype(int)
data['treatment_group'] = (data['state'] == 'NJ').astype(int)
data['post'] = data['period']

# Visualize pre-trends (if multiple periods available)
# In classic Card-Krueger, only 2 periods

# Run DID analysis
result = run_full_did_analysis(
    data=data,
    outcome="fte",
    treatment="treated",
    unit_id="store_id",
    time_id="period",
    treatment_time=1,
    controls=["chain"],
    cluster="store_id"
)

# View results
print(result.summary_table)
print(f"\nATT: {result.effect:.3f} ({result.se:.3f})")
print(f"95% CI: [{result.ci_lower:.3f}, {result.ci_upper:.3f}]")
```

**Output**:
```
### Difference-in-Differences Results

| Variable | (1) Basic | (2) With Controls |
|:---------|:---------:|:-----------------:|
| Treatment Effect | 2.76** | 2.54** |
| | (1.04) | (0.98) |
| Controls | No | Yes |
| Unit FE | Yes | Yes |
| Time FE | Yes | Yes |
| Observations | 784 | 784 |
| R-squared | 0.089 | 0.112 |

*Notes: Robust standard errors clustered at store level.*

ATT: 2.54 (0.98)
95% CI: [0.62, 4.46]
```

**Interpretation**:
The minimum wage increase in New Jersey led to an increase of approximately 2.54 full-time equivalent employees per restaurant (SE = 0.98), contrary to the standard competitive labor market prediction. This effect is statistically significant at the 5% level.

### Example 2: Staggered DID with Synthetic Data

```python
import pandas as pd
import numpy as np
from did_estimator import (
    estimate_did_staggered,
    event_study_plot,
    test_parallel_trends
)

# Import synthetic data generator
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / 'lib' / 'python'))
from data_loader import generate_synthetic_did_data

# Generate synthetic panel data
data, true_params = generate_synthetic_did_data(
    n_units=100,
    n_periods=10,
    treatment_period=5,
    treatment_effect=2.0,
    treatment_share=0.5,
    noise_std=1.0,
    random_state=42
)

print(f"True ATT: {true_params['true_ate']}")

# Test parallel trends
trends = test_parallel_trends(
    data=data,
    outcome="y",
    treatment="treatment_group",
    time_id="time",
    unit_id="unit_id",
    treatment_time=5,
    n_pre_periods=4
)
print(f"\nParallel trends test: {'PASSED' if trends.passed else 'FAILED'}")
print(f"  Slope of trend difference: {trends.statistic:.4f}")
print(f"  P-value: {trends.p_value:.4f}")

# Run estimation
from did_estimator import estimate_did_panel

result = estimate_did_panel(
    data=data,
    outcome="y",
    treatment="treated",
    unit_id="unit_id",
    time_id="time",
    controls=["x1", "x2"],
    cluster="unit_id"
)

print(f"\nEstimated ATT: {result.effect:.4f}")
print(f"Standard Error: {result.se:.4f}")
print(f"95% CI: [{result.ci_lower:.4f}, {result.ci_upper:.4f}]")
print(f"Bias: {(result.effect - true_params['true_ate']) / true_params['true_ate'] * 100:.2f}%")
```

**Output**:
```
True ATT: 2.0

Parallel trends test: PASSED
  Slope of trend difference: 0.0234
  P-value: 0.7823

Estimated ATT: 2.0156
Standard Error: 0.1423
95% CI: [1.7367, 2.2945]
Bias: 0.78%
```

---

## References

### Seminal Papers
- Card, D., & Krueger, A. B. (1994). Minimum Wages and Employment: A Case Study of the Fast-Food Industry in New Jersey and Pennsylvania. *American Economic Review*, 84(4), 772-793.
- Angrist, J. D., & Pischke, J. S. (2009). *Mostly Harmless Econometrics*. Princeton University Press. Chapter 5.
- Callaway, B., & Sant'Anna, P. H. (2021). Difference-in-Differences with Multiple Time Periods. *Journal of Econometrics*, 225(2), 200-230.

### Methodological Extensions
- Goodman-Bacon, A. (2021). Difference-in-Differences with Variation in Treatment Timing. *Journal of Econometrics*, 225(2), 254-277.
- Sun, L., & Abraham, S. (2021). Estimating Dynamic Treatment Effects in Event Studies with Heterogeneous Treatment Effects. *Journal of Econometrics*, 225(2), 175-199.
- de Chaisemartin, C., & D'Haultfoeuille, X. (2020). Two-Way Fixed Effects Estimators with Heterogeneous Treatment Effects. *American Economic Review*, 110(9), 2964-2996.

### Textbook Treatments
- Angrist, J. D., & Pischke, J. S. (2015). *Mastering 'Metrics*. Princeton University Press. Chapter 5.
- Cunningham, S. (2021). *Causal Inference: The Mixtape*. Yale University Press. Chapter 9.
- Huntington-Klein, N. (2022). *The Effect*. CRC Press. Chapter 18.

### Software Documentation
- `linearmodels`: https://bashtage.github.io/linearmodels/
- `did` (R): https://bcallaway11.github.io/did/
- `csdid` (Stata): https://friosavila.github.io/playgrounds/csdid/csdid.html

---

## Related Estimators

| Estimator | When to Use Instead |
|-----------|---------------------|
| `estimator-synthetic-control` | Parallel trends violated; small number of treated units |
| `estimator-rdid` | Treatment assigned by threshold with time variation |
| `estimator-event-study` | Focus on dynamic treatment effects over time |
| `estimator-psm-did` | Need to improve covariate balance before DID |
| `estimator-cic` | Want to relax parallel trends to changes-in-changes |

---

## Appendix: Mathematical Details

### Derivation of 2x2 DID Estimator

The DID estimator can be derived as:

$$
\hat{\delta}_{DID} = (\bar{Y}_{T,1} - \bar{Y}_{T,0}) - (\bar{Y}_{C,1} - \bar{Y}_{C,0})
$$

Where:
- $\bar{Y}_{T,1}$: Mean outcome for treated group in post-period
- $\bar{Y}_{T,0}$: Mean outcome for treated group in pre-period
- $\bar{Y}_{C,1}$: Mean outcome for control group in post-period
- $\bar{Y}_{C,0}$: Mean outcome for control group in pre-period

This is equivalent to the OLS coefficient on $\text{Treated} \times \text{Post}$ in the regression:

$$
Y_{it} = \alpha + \beta_1 \text{Treated}_i + \beta_2 \text{Post}_t + \delta (\text{Treated}_i \times \text{Post}_t) + \epsilon_{it}
$$

### Asymptotic Properties

Under standard regularity conditions and the parallel trends assumption:

$$
\sqrt{N}(\hat{\delta} - \delta_0) \xrightarrow{d} N(0, V)
$$

Where $V$ depends on the variance of the error term and the design. With clustered errors at the unit level:

$$
\hat{V} = (X'X)^{-1} \left( \sum_{i=1}^{N} X_i' \hat{u}_i \hat{u}_i' X_i \right) (X'X)^{-1}
$$

### Goodman-Bacon Decomposition

For staggered DID with TWFE, the overall estimate is a weighted average:

$$
\hat{\delta}^{TWFE} = \sum_{k} \sum_{l \neq k} s_{kl} \hat{\delta}_{kl}
$$

Where $\hat{\delta}_{kl}$ is a 2x2 DID comparing timing group $k$ to timing group $l$, and $s_{kl}$ are weights that depend on group sizes and variance.

**Problem**: Some $\hat{\delta}_{kl}$ compare later-treated to earlier-treated units, using already-treated units as controls. If treatment effects are heterogeneous, this can lead to bias.

**Solution**: Use Callaway-Sant'Anna or other heterogeneity-robust estimators.
