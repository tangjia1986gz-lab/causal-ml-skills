---
name: estimator-psm
description: Use when estimating causal effects with Propensity Score Matching. Triggers on PSM, propensity score, matching, ATT, ATE, selection on observables, nearest neighbor, caliper, kernel matching, Mahalanobis distance.
---

# Estimator: Propensity Score Matching (PSM)

> **Version**: 1.0.0 | **Type**: Estimator
> **Aliases**: PSM, Propensity Score Matching, Matching Estimator, Selection on Observables

## Overview

Propensity Score Matching (PSM) estimates causal effects by matching treated units to control units with similar propensity scores (predicted probability of receiving treatment given observed covariates). By comparing outcomes between matched treated and control units, PSM removes confounding bias due to observed covariates.

**Key Identification Assumption**: Conditional Independence Assumption (CIA) / Unconfoundedness - conditional on observed covariates X, treatment assignment D is independent of potential outcomes: $(Y_0, Y_1) \perp D | X$

## When to Use

### Ideal Scenarios
- Selection into treatment is based on observed characteristics (selection on observables)
- Rich set of pre-treatment covariates available
- Observational study where randomization is not feasible
- Want to create a matched comparison group that resembles the treatment group
- Need to check covariate balance explicitly

### Data Requirements
- [ ] Cross-sectional or panel data with clearly defined treatment and control groups
- [ ] Binary treatment indicator
- [ ] Pre-treatment covariates that predict treatment assignment
- [ ] Outcome variable measured post-treatment
- [ ] Sufficient overlap in covariate distributions between treatment and control

### When NOT to Use
- Unobserved confounding is likely present -> Consider `estimator-iv` (Instrumental Variables)
- Poor overlap / limited common support -> Consider `estimator-synthetic-control` or trimming
- Treatment effect varies by propensity score in complex ways -> Consider `estimator-causal-forest`
- Panel data with parallel trends -> Consider `estimator-did` (often stronger design)
- Regression discontinuity design available -> Consider `estimator-rd`

## Identification Assumptions

| Assumption | Description | Testable? |
|------------|-------------|-----------|
| **Unconfoundedness (CIA)** | No unmeasured confounders; treatment assignment is ignorable given X | No (fundamentally untestable) |
| **Common Support (Overlap)** | For all X, probability of treatment is strictly between 0 and 1: $0 < P(D=1|X) < 1$ | Yes (propensity score distribution) |
| **SUTVA** | No interference between units; stable unit treatment values | No (domain knowledge) |
| **Correct PS Model** | Propensity score model is correctly specified | Partially (balance checks) |

---

## Workflow

```
+-------------------------------------------------------------+
|                    PSM ESTIMATOR WORKFLOW                     |
+-------------------------------------------------------------+
|  1. SETUP          -> Define outcome, treatment, covariates   |
|  2. PS ESTIMATION  -> Estimate propensity scores (logit/ML)   |
|  3. OVERLAP CHECK  -> Verify common support assumption        |
|  4. MATCHING       -> Match treated to controls               |
|  5. BALANCE CHECK  -> Verify covariate balance improved       |
|  6. ESTIMATION     -> Estimate ATT/ATE on matched sample      |
|  7. SENSITIVITY    -> Rosenbaum bounds for hidden bias        |
+-------------------------------------------------------------+
```

### Phase 1: Setup

**Objective**: Prepare data and define model specification

**Inputs Required**:
```python
# Standard CausalInput structure for PSM
outcome = "y"                    # Outcome variable name
treatment = "treated"            # Treatment indicator (0/1)
covariates = ["x1", "x2", "x3"]  # Pre-treatment covariates
```

**Data Validation Checklist**:
- [ ] Treatment is binary (0/1)
- [ ] Covariates are measured PRE-treatment (no post-treatment variables!)
- [ ] No missing values in key variables (or explicit handling strategy)
- [ ] Sufficient treated and control observations
- [ ] Covariates should predict treatment assignment

**Data Structure Verification**:
```python
from psm_estimator import validate_psm_data

# Check data structure
validation = validate_psm_data(
    data=df,
    outcome="y",
    treatment="treated",
    covariates=["x1", "x2", "x3"]
)
print(validation.summary())
```

### Phase 2: Propensity Score Estimation

**Propensity Score Definition**:
$$
e(X) = P(D = 1 | X) = E[D | X]
$$

The propensity score $e(X)$ is the conditional probability of receiving treatment given covariates.

**Estimation Methods**:

| Method | Pros | Cons |
|--------|------|------|
| Logistic Regression | Simple, interpretable | May miss nonlinearities |
| Probit | Similar to logit | Slightly different functional form |
| GBM/Random Forest | Captures complex patterns | Less interpretable, may overfit |
| LASSO Logistic | Variable selection built-in | Requires tuning |

```python
from psm_estimator import estimate_propensity_score

# Option 1: Logistic regression (default)
ps_result = estimate_propensity_score(
    data=df,
    treatment="treated",
    covariates=["x1", "x2", "x3"],
    method="logit"
)

# Option 2: Machine learning methods
ps_result = estimate_propensity_score(
    data=df,
    treatment="treated",
    covariates=["x1", "x2", "x3"],
    method="gbm",  # or "random_forest", "lasso"
    cv_folds=5
)

print(f"Propensity scores range: [{ps_result.ps.min():.3f}, {ps_result.ps.max():.3f}]")
```

### Phase 3: Common Support / Overlap Check

**Common Support Requirement**:
For valid causal inference, both treatment and control groups must have positive probability at all covariate values. Check overlap in propensity score distributions.

```python
from psm_estimator import check_common_support, plot_propensity_overlap

# Visual inspection
fig = plot_propensity_overlap(
    ps_treated=df.loc[df['treated']==1, 'propensity_score'],
    ps_control=df.loc[df['treated']==0, 'propensity_score']
)

# Statistical check
overlap_result = check_common_support(
    ps_treated=df.loc[df['treated']==1, 'propensity_score'],
    ps_control=df.loc[df['treated']==0, 'propensity_score'],
    method="minmax"  # or "trim_5pct", "optimal"
)

print(overlap_result)
```

**Interpretation**:
- PASS if: Substantial overlap in PS distributions
- WARNING if: Limited overlap, many observations outside common support
- FAIL if: No overlap region - matching not valid

**Handling Poor Overlap**:
1. **Trimming**: Drop observations with extreme propensity scores
2. **Caliper matching**: Only match within specified PS distance
3. **Reassess covariates**: Remove covariates that perfectly predict treatment

---

### Phase 4: Matching

**Matching Algorithms Comparison**:

| Method | Description | Pros | Cons |
|--------|-------------|------|------|
| **Nearest Neighbor** | Match to closest PS | Simple, intuitive | May create poor matches if no close control |
| **Nearest Neighbor with Caliper** | NN within max PS distance | Controls match quality | May discard unmatched treated |
| **Radius/Caliper** | Match to all controls within caliper | Uses more information | May have variable # matches |
| **Kernel** | Weighted average of all controls | Uses all data, smooth | Bandwidth selection crucial |
| **Mahalanobis Distance** | Match on multivariate distance | Exact covariate matching | Curse of dimensionality |
| **Coarsened Exact Matching** | Exact match on coarsened bins | Guaranteed balance | May discard many observations |

**Python Implementation**:

```python
from psm_estimator import (
    match_nearest_neighbor,
    match_kernel,
    match_mahalanobis,
    match_with_caliper
)

# Option 1: Nearest Neighbor (1:1, with replacement)
matched = match_nearest_neighbor(
    data=df,
    propensity_score="ps",
    treatment="treated",
    n_neighbors=1,
    replacement=True
)

# Option 2: Nearest Neighbor with Caliper
matched = match_with_caliper(
    data=df,
    propensity_score="ps",
    treatment="treated",
    caliper=0.1,  # Max PS difference (0.1 = 10% of PS)
    caliper_scale="ps_std"  # or "ps_raw"
)

# Option 3: Kernel Matching
matched = match_kernel(
    data=df,
    propensity_score="ps",
    treatment="treated",
    kernel="epanechnikov",  # or "gaussian"
    bandwidth="optimal"  # or float
)

# Option 4: Mahalanobis Distance (on covariates directly)
matched = match_mahalanobis(
    data=df,
    treatment="treated",
    covariates=["x1", "x2", "x3"],
    n_neighbors=1
)

print(f"Matched sample: {matched.n_matched} treated, {matched.n_controls} controls")
print(f"Unmatched treated: {matched.n_unmatched}")
```

### Phase 5: Balance Checking

**Critical Step**: After matching, verify that covariate balance has improved. Matching is only valid if it achieves good balance.

**Balance Metrics**:
- **Standardized Mean Difference (SMD)**: $\frac{\bar{X}_T - \bar{X}_C}{\sqrt{(S_T^2 + S_C^2)/2}}$
  - Target: |SMD| < 0.1 (some use 0.25)
- **Variance Ratio**: $S_T^2 / S_C^2$
  - Target: Between 0.5 and 2.0

```python
from psm_estimator import check_balance, create_balance_table, plot_balance

# Calculate balance statistics
balance = check_balance(
    data=df,
    treatment="treated",
    covariates=["x1", "x2", "x3"],
    weights="match_weights"  # or None for unmatched
)

# Create balance table
balance_table = create_balance_table(
    data=df,
    treatment="treated",
    covariates=["x1", "x2", "x3"],
    weights="match_weights",
    show_before_after=True
)
print(balance_table)

# Visual balance plot (Love plot)
fig = plot_balance(
    balance_before=balance_unmatched,
    balance_after=balance_matched,
    threshold=0.1
)
```

**Balance Table Output**:
```
+------------------------------------------------------------------+
|                    Covariate Balance Table                        |
+------------------------------------------------------------------+
|              |    Before Matching    |    After Matching     |
| Variable     | Treated  Control  SMD | Treated  Control  SMD |
+------------------------------------------------------------------+
| x1           |   0.52    0.31  0.42  |   0.48    0.46  0.04  |
| x2           |   0.68    0.45  0.47  |   0.62    0.60  0.04  |
| x3           |   1.23    0.98  0.31  |   1.15    1.12  0.04  |
+------------------------------------------------------------------+
| All SMD < 0.1: No -> Yes                                          |
+------------------------------------------------------------------+
```

**Interpretation**:
- PASS if: All covariates have |SMD| < 0.1 after matching
- WARNING if: Some covariates have 0.1 < |SMD| < 0.25
- FAIL if: Any covariate has |SMD| > 0.25 - re-specify PS model or matching method

---

### Phase 6: Effect Estimation

**Estimands**:

| Estimand | Definition | When to Use |
|----------|------------|-------------|
| **ATT** (Average Treatment Effect on Treated) | $E[Y_1 - Y_0 | D=1]$ | Focus on treated population |
| **ATE** (Average Treatment Effect) | $E[Y_1 - Y_0]$ | Effect on entire population |
| **ATC** (Average Treatment Effect on Controls) | $E[Y_1 - Y_0 | D=0]$ | Less common |

**ATT Estimation** (Most Common for PSM):

$$
\hat{\tau}_{ATT} = \frac{1}{N_T} \sum_{i: D_i=1} \left[ Y_i - \sum_{j: D_j=0} w_{ij} Y_j \right]
$$

Where $w_{ij}$ are matching weights.

**Python Implementation**:

```python
from psm_estimator import estimate_att, estimate_ate

# ATT on matched sample
att_result = estimate_att(
    data=df,
    outcome="y",
    treatment="treated",
    weights="match_weights",
    se_method="abadie_imbens"  # or "bootstrap", "analytical"
)

print(f"ATT: {att_result.effect:.4f} (SE: {att_result.se:.4f})")
print(f"95% CI: [{att_result.ci_lower:.4f}, {att_result.ci_upper:.4f}]")

# ATE using IPW (Inverse Propensity Weighting)
ate_result = estimate_ate(
    data=df,
    outcome="y",
    treatment="treated",
    propensity_score="ps",
    estimator="ipw"  # or "doubly_robust"
)
```

**Returns**:
```python
CausalOutput(
    effect=1.52,           # Point estimate of ATT
    se=0.34,               # Standard error
    ci_lower=0.86,         # 95% CI lower bound
    ci_upper=2.18,         # 95% CI upper bound
    p_value=0.0001,        # Two-sided p-value
    diagnostics={
        'n_treated': 200,
        'n_control_matched': 180,
        'n_unmatched': 20,
        'balance_achieved': True,
        'mean_smd': 0.05
    }
)
```

### Phase 7: Sensitivity Analysis

**Rosenbaum Bounds**: Since unconfoundedness cannot be tested, sensitivity analysis examines how much hidden bias (unobserved confounding) would be needed to invalidate the results.

**Gamma ($\Gamma$)**: Odds ratio of differential treatment assignment due to unobserved confounder
- $\Gamma = 1$: No hidden bias (unconfoundedness holds)
- $\Gamma = 2$: Unobserved confounder could double odds of treatment

```python
from psm_estimator import rosenbaum_sensitivity

# Sensitivity analysis
sensitivity = rosenbaum_sensitivity(
    data=df_matched,
    outcome="y",
    treatment="treated",
    gamma_range=[1.0, 1.5, 2.0, 2.5, 3.0]
)

print(sensitivity.summary())
```

**Output**:
```
Rosenbaum Sensitivity Analysis
------------------------------
Gamma   Lower Bound   Upper Bound   P-value (upper)
1.0     1.52          1.52          0.0001
1.5     0.98          2.06          0.0023
2.0     0.52          2.52          0.0156
2.5     0.08          2.96          0.0512
3.0    -0.34          3.38          0.1203

Interpretation: Results are sensitive to unobserved confounding
at Gamma = 2.5 (unobserved confounder with 2.5x odds ratio
could make the effect insignificant).
```

---

## PSM + DID Combination

When both selection on observables AND time trends matter, combine PSM with DID:

```python
from psm_estimator import psm_did

# Step 1: Match on pre-treatment characteristics
# Step 2: Run DID on matched sample

result = psm_did(
    data=df,
    outcome="y",
    treatment="treated",
    time_id="year",
    unit_id="id",
    treatment_time=2015,
    covariates=["x1", "x2"],  # For matching
    match_method="nearest_neighbor"
)
```

This approach:
1. Improves covariate balance between treatment and control
2. Uses DID to account for time trends
3. Requires BOTH unconfoundedness AND parallel trends

---

## Common Mistakes

### 1. Including Post-Treatment Variables in Propensity Score

**Mistake**: Including variables affected by treatment in the PS model.

**Why it's wrong**: Post-treatment variables are "bad controls" - they can induce bias by blocking causal pathways or opening collider paths.

**Correct approach**:
```python
# WRONG: Including post-treatment variable
ps = estimate_propensity_score(
    covariates=["x1", "x2", "post_treatment_mediator"]  # BAD!
)

# CORRECT: Only pre-treatment covariates
ps = estimate_propensity_score(
    covariates=["x1_baseline", "x2_baseline"]  # Good
)
```

### 2. Ignoring Common Support Violations

**Mistake**: Proceeding with matching despite poor overlap in propensity score distributions.

**Why it's wrong**: Extrapolation beyond common support leads to model-dependent, unreliable estimates.

**Correct approach**:
```python
# ALWAYS check overlap first
overlap = check_common_support(ps_treated, ps_control)

if not overlap.sufficient:
    print("WARNING: Poor overlap detected!")
    print("Consider:")
    print("1. Trimming observations outside common support")
    print("2. Using caliper matching")
    print("3. Re-specifying the propensity score model")
    print("4. Acknowledging limited external validity")
```

### 3. Not Checking Balance After Matching

**Mistake**: Assuming matching automatically produces balance without verification.

**Why it's wrong**: Matching may fail to achieve balance, especially with poor PS model or limited overlap.

**Correct approach**:
```python
# ALWAYS check balance after matching
balance = check_balance(data, treatment, covariates, weights)

if any(abs(smd) > 0.1 for smd in balance.smd.values()):
    print("WARNING: Balance not achieved!")
    print("Consider:")
    print("1. Adding interaction terms to PS model")
    print("2. Using different matching method")
    print("3. Exact matching on key covariates")
```

### 4. Claiming Unconfoundedness is Satisfied

**Mistake**: Stating that unconfoundedness "holds" because balance is achieved.

**Why it's wrong**: Balance on observed covariates does not guarantee no unobserved confounding.

**Correct approach**:
```python
# In interpretation, be clear about assumptions
interpretation = """
Results rely on the unconfoundedness assumption, which is
fundamentally untestable. Sensitivity analysis suggests
results are robust to hidden bias up to Gamma = 2.0.
"""
```

### 5. Using Matching Weights Incorrectly

**Mistake**: Forgetting to use matching weights in subsequent analysis.

**Why it's wrong**: Unweighted analysis on matched sample may be biased.

**Correct approach**:
```python
# WRONG: Unweighted regression on matched sample
model = OLS(y ~ treatment, data=matched_data).fit()

# CORRECT: Use matching weights
model = WLS(y ~ treatment, data=matched_data, weights=match_weights).fit()
```

---

## Examples

### Example 1: LaLonde (1986) Job Training Program

**Research Question**: What is the effect of job training on earnings?

**Data**: National Supported Work (NSW) experimental data matched with observational controls.

```python
import pandas as pd
from psm_estimator import run_full_psm_analysis, plot_balance

# Load LaLonde data (example)
# Treatment: participation in job training program
# Outcome: earnings in 1978
# Covariates: age, education, race, married, earnings in 1974-75

data = pd.read_csv("lalonde.csv")

# Define variables
outcome = "re78"  # Real earnings 1978
treatment = "treat"  # Treatment indicator
covariates = [
    "age", "education", "black", "hispanic",
    "married", "nodegree", "re74", "re75"
]

# Run full PSM analysis
result = run_full_psm_analysis(
    data=data,
    outcome=outcome,
    treatment=treatment,
    covariates=covariates,
    match_method="nearest_neighbor",
    caliper=0.1,
    estimand="ATT"
)

# View results
print(result.summary_table)
print(result.balance_table)

# Sensitivity analysis
print(result.sensitivity_analysis)
```

**Output**:
```
============================================================
         PROPENSITY SCORE MATCHING ANALYSIS RESULTS
============================================================

Treatment Effect (ATT): $1,794.34
Standard Error: $632.85
95% CI: [$553.96, $3,034.72]
P-value: 0.0046

------------------------------------------------------------
COVARIATE BALANCE
------------------------------------------------------------
               Before Matching          After Matching
Variable     Treated  Control   SMD   Treated  Control   SMD
------------------------------------------------------------
age            25.82    34.85  -0.94    25.82    26.15  -0.04
education      10.35    12.03  -0.64    10.35    10.42  -0.03
black           0.84     0.25   1.34     0.84     0.82   0.05
hispanic        0.06     0.03   0.12     0.06     0.05   0.04
married         0.19     0.51  -0.72     0.19     0.21  -0.05
nodegree        0.71     0.31   0.87     0.71     0.69   0.04
re74         2095.57  19428.7  -0.82   2095.57  2234.12  -0.03
re75         1532.06  21553.9  -0.91   1532.06  1789.43  -0.05
------------------------------------------------------------
Balance achieved: YES (all SMD < 0.1)

------------------------------------------------------------
SENSITIVITY ANALYSIS (Rosenbaum Bounds)
------------------------------------------------------------
Gamma    Effect Bounds      P-value
1.0      [1794, 1794]       0.005
1.5      [1124, 2464]       0.032
2.0      [521, 3067]        0.098
2.5      [-38, 3627]        0.184

Conclusion: Results sensitive at Gamma = 2.5
============================================================
```

**Interpretation**:
Job training participation increased earnings by approximately $1,794 (SE = $633), which is statistically significant at the 5% level. After matching, all covariates are well-balanced (SMD < 0.1). Sensitivity analysis suggests that an unobserved confounder would need to increase the odds of treatment by 2.5 times to render the effect statistically insignificant.

### Example 2: Synthetic Data Validation

```python
import numpy as np
import pandas as pd
from psm_estimator import run_full_psm_analysis, validate_estimator

# Generate synthetic data with known effect
np.random.seed(42)
n = 2000

# Covariates
x1 = np.random.normal(0, 1, n)
x2 = np.random.binomial(1, 0.5, n)
x3 = np.random.normal(0, 1, n)

# Treatment probability depends on covariates
ps_true = 1 / (1 + np.exp(-(0.5 * x1 + 0.8 * x2 + 0.3 * x3)))
treatment = np.random.binomial(1, ps_true)

# True treatment effect
true_effect = 2.0

# Outcome with confounding
y = 1 + 0.5 * x1 + 0.3 * x2 + 0.2 * x3 + true_effect * treatment + np.random.normal(0, 1, n)

data = pd.DataFrame({
    'y': y, 'treatment': treatment,
    'x1': x1, 'x2': x2, 'x3': x3
})

# Run PSM
result = run_full_psm_analysis(
    data=data,
    outcome="y",
    treatment="treatment",
    covariates=["x1", "x2", "x3"]
)

print(f"True effect: {true_effect}")
print(f"Estimated ATT: {result.effect:.4f}")
print(f"Bias: {(result.effect - true_effect) / true_effect * 100:.2f}%")
```

---

## References

### Seminal Papers
- Rosenbaum, P. R., & Rubin, D. B. (1983). The Central Role of the Propensity Score in Observational Studies for Causal Effects. *Biometrika*, 70(1), 41-55.
- LaLonde, R. J. (1986). Evaluating the Econometric Evaluations of Training Programs with Experimental Data. *American Economic Review*, 76(4), 604-620.
- Dehejia, R. H., & Wahba, S. (1999). Causal Effects in Nonexperimental Studies: Reevaluating the Evaluation of Training Programs. *Journal of the American Statistical Association*, 94(448), 1053-1062.

### Methodological Extensions
- Abadie, A., & Imbens, G. W. (2006). Large Sample Properties of Matching Estimators for Average Treatment Effects. *Econometrica*, 74(1), 235-267.
- Abadie, A., & Imbens, G. W. (2011). Bias-Corrected Matching Estimators for Average Treatment Effects. *Journal of Business & Economic Statistics*, 29(1), 1-11.
- Rosenbaum, P. R. (2002). *Observational Studies* (2nd ed.). Springer.

### Practical Guides
- Caliendo, M., & Kopeinig, S. (2008). Some Practical Guidance for the Implementation of Propensity Score Matching. *Journal of Economic Surveys*, 22(1), 31-72.
- Stuart, E. A. (2010). Matching Methods for Causal Inference: A Review and a Look Forward. *Statistical Science*, 25(1), 1-21.

### Textbook Treatments
- Angrist, J. D., & Pischke, J. S. (2009). *Mostly Harmless Econometrics*. Princeton University Press. Chapter 3.
- Cunningham, S. (2021). *Causal Inference: The Mixtape*. Yale University Press. Chapter 5.
- Huntington-Klein, N. (2022). *The Effect*. CRC Press. Chapter 14.

### Software Documentation
- `MatchIt` (R): https://kosukeimai.github.io/MatchIt/
- `causalml` (Python): https://causalml.readthedocs.io/
- `pymatch` (Python): https://github.com/benmiroglio/pymatch

---

## Related Estimators

| Estimator | When to Use Instead |
|-----------|---------------------|
| `estimator-did` | Panel data with parallel trends assumption |
| `estimator-iv` | Unobserved confounding with valid instrument |
| `estimator-rd` | Treatment assigned by threshold rule |
| `estimator-synthetic-control` | Few treated units, aggregate data |
| `estimator-causal-forest` | Heterogeneous treatment effects |
| `estimator-doubly-robust` | Want robustness to PS or outcome model misspecification |

---

## Appendix: Mathematical Details

### Propensity Score Theorem (Rosenbaum & Rubin, 1983)

If unconfoundedness holds given X:
$$
(Y_0, Y_1) \perp D | X
$$

Then unconfoundedness also holds given the propensity score:
$$
(Y_0, Y_1) \perp D | e(X)
$$

**Implication**: Instead of matching on high-dimensional X, we can match on the scalar propensity score $e(X)$.

### ATT Identification

Under unconfoundedness and overlap:
$$
\tau_{ATT} = E[Y_1 - Y_0 | D = 1]
$$
$$
= E[Y_1 | D = 1] - E[Y_0 | D = 1]
$$
$$
= E[Y | D = 1] - E[E[Y | D = 0, e(X)] | D = 1]
$$

The second term is identified by matching treated units to controls with similar propensity scores.

### Abadie-Imbens Standard Errors

For matching estimators, Abadie & Imbens (2006) derive:
$$
Var(\hat{\tau}_{ATT}) = \frac{1}{N_T} \left[ Var(Y_1 | D=1) + E[Var(Y_0 | e(X)) | D=1] \cdot (1 + K_M) \right]
$$

Where $K_M$ accounts for the number of times each control is used as a match.

### Rosenbaum Bounds

Under hidden bias, the odds of treatment may differ for two units with the same X:
$$
\frac{1}{\Gamma} \leq \frac{P(D_i = 1 | X)}{P(D_j = 1 | X)} \cdot \frac{1 - P(D_j = 1 | X)}{1 - P(D_i = 1 | X)} \leq \Gamma
$$

For each $\Gamma$, we compute bounds on the treatment effect and test statistic. The critical $\Gamma$ where significance is lost indicates sensitivity to hidden bias.
