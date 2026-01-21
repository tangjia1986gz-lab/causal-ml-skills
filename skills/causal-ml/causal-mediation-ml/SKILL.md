---
name: causal-mediation-ml
description: Use when decomposing treatment effects into direct and indirect (mediated) pathways. Triggers on mediation, mediator, ADE, ACME, direct effect, indirect effect, mechanism, pathway, causal mechanism, mediation analysis.
---

# Estimator: ML-Enhanced Causal Mediation Analysis

> **Version**: 1.0.0 | **Type**: Estimator
> **Aliases**: Mediation Analysis, ACME, ADE, Direct/Indirect Effects, Mechanism Analysis

## Overview

Causal Mediation Analysis decomposes the **Total Effect** of a treatment on an outcome into two components:

1. **Average Direct Effect (ADE)**: The effect of treatment on outcome that does NOT operate through the mediator
2. **Average Causal Mediation Effect (ACME)**: The effect of treatment on outcome that operates THROUGH the mediator (indirect effect)

$$
\text{Total Effect} = \text{Direct Effect (ADE)} + \text{Indirect Effect (ACME)}
$$

**Key Innovation**: This skill combines traditional mediation analysis (Baron & Kenny, 1986) with modern Double/Debiased Machine Learning (DDML) to handle high-dimensional confounders while maintaining valid statistical inference.

```
Treatment (D) ─────────────────────────────────→ Outcome (Y)
      │                  Direct Effect                 ↑
      │                                                │
      └───────→ Mediator (M) ───────────────────────→─┘
           Indirect Effect (via M)
```

## When to Use

### Ideal Scenarios
- **Mechanism investigation**: Understanding HOW or WHY a treatment works
- **Policy design**: Identifying which channels are most effective
- **Program evaluation**: Decomposing intervention effects by pathway
- **Theory testing**: Testing theoretical causal mechanisms
- **Resource allocation**: Deciding whether to target direct vs. indirect pathways

### Data Requirements
- [ ] Treatment variable (binary or continuous)
- [ ] Mediator variable (measured post-treatment, pre-outcome)
- [ ] Outcome variable
- [ ] Control variables (observed confounders of all relationships)
- [ ] Sufficient sample size (n > 500 recommended for ML-enhanced version)
- [ ] Temporal ordering: Treatment → Mediator → Outcome

### When NOT to Use
- **Mediator is a collider**: If M is affected by both unmeasured causes of D and Y → Biased estimates
- **Post-treatment confounding**: If there's unmeasured confounding between M and Y that is affected by D → Consider `causal-forest` for heterogeneous effects instead
- **Multiple simultaneous mediators**: Complex mediation structures may require specialized software
- **No theoretical basis**: Mediation without theory is exploratory, not confirmatory
- **Time ordering unclear**: When treatment, mediator, and outcome timing is ambiguous → Collect better data

## Identification Assumptions

| Assumption | Description | Testable? |
|------------|-------------|-----------|
| **Sequential Ignorability (Part 1)** | Treatment D is independent of potential outcomes and mediators given controls X | No (fundamentally untestable) |
| **Sequential Ignorability (Part 2)** | Mediator M is independent of potential outcomes given D and X | No (fundamentally untestable) |
| **No Interaction Between D and M on Unmeasured Confounders** | No unmeasured confounder of M→Y relationship is affected by treatment D | No |
| **Positivity** | All units have positive probability of any treatment and mediator value | Yes (check distributions) |
| **Correct Functional Form** | Models for E[M|D,X] and E[Y|D,M,X] are correctly specified | Partially (via ML flexibility) |

**Critical**: Sequential ignorability is a STRONG assumption. The treatment must be "as-if randomized" conditional on X, AND the mediator must be "as-if randomized" conditional on D and X. This is rarely satisfied without experimental manipulation of both D and M.

---

## Workflow

```
+-------------------------------------------------------------+
|                MEDIATION ANALYSIS WORKFLOW                   |
+-------------------------------------------------------------+
|  1. SETUP          -> Define Y, D, M, X with temporal order  |
|  2. MODEL SPEC     -> Choose outcome & mediator models       |
|  3. ESTIMATION     -> Traditional Baron-Kenny OR ML-enhanced |
|  4. DECOMPOSITION  -> Calculate ADE, ACME, proportion med.   |
|  5. SENSITIVITY    -> How robust to unmeasured confounding   |
|  6. REPORTING      -> Decomposition table, pathway diagram   |
+-------------------------------------------------------------+
```

### Phase 1: Setup

**Objective**: Define the causal structure and verify temporal ordering

**Inputs Required**:
```python
# Standard CausalInput structure for Mediation
outcome = "y"                    # Outcome variable (Y)
treatment = "d"                  # Treatment variable (D)
mediator = "m"                   # Mediator variable (M)
controls = ["x1", "x2", ...]     # Confounders of D→Y, D→M, M→Y
```

**Causal Structure (DAG)**:
```
       X (Controls)
      /   \     \
     ↓     ↓     ↓
    D ──→  M ──→ Y
     \___________↗
        (direct)
```

**Data Validation Checklist**:
- [ ] Treatment occurs BEFORE mediator (check data collection timing)
- [ ] Mediator occurs BEFORE outcome
- [ ] No missing values in key variables
- [ ] Mediator has sufficient variation in both treatment groups
- [ ] Controls include all observed confounders of all relationships

**Data Structure Verification**:
```python
from mediation_estimator import validate_mediation_setup

# Verify data structure
validation = validate_mediation_setup(
    data=df,
    outcome="y",
    treatment="d",
    mediator="m",
    controls=control_vars
)
print(validation)
```

### Phase 2: Model Specification

**Key Decision**: Choose models for two equations:

| Model | Equation | Purpose |
|-------|----------|---------|
| **Mediator Model** | $M = f(D, X) + \epsilon_M$ | Predict mediator from treatment and controls |
| **Outcome Model** | $Y = g(D, M, X) + \epsilon_Y$ | Predict outcome from treatment, mediator, and controls |

**Traditional (Parametric) Approach**:
```python
# Baron-Kenny linear models
M = alpha_0 + alpha_1 * D + alpha_2 * X + epsilon_M
Y = beta_0 + beta_1 * D + beta_2 * M + beta_3 * X + epsilon_Y

# Mediation effects (linear case)
ACME = alpha_1 * beta_2  # Indirect effect
ADE = beta_1             # Direct effect
```

**ML-Enhanced Approach**:
Use flexible ML models for both equations to handle:
- High-dimensional controls
- Nonlinear relationships
- Complex interactions

```python
from mediation_estimator import estimate_ml_mediation

# Auto-select best learners
result = estimate_ml_mediation(
    data=df,
    outcome="y",
    treatment="d",
    mediator="m",
    controls=control_vars,
    ml_m='lasso',      # Learner for M|D,X
    ml_y='random_forest'  # Learner for Y|D,M,X
)
```

### Phase 3: Estimation Methods

**Method A: Traditional Baron-Kenny (1986)**

Classic 4-step approach:
1. Show D → Y relationship exists (total effect)
2. Show D → M relationship exists
3. Show M → Y relationship (controlling for D)
4. Direct effect should decrease when M is added

```python
from mediation_estimator import estimate_baron_kenny

result = estimate_baron_kenny(
    data=df,
    outcome="y",
    treatment="d",
    mediator="m",
    controls=control_vars
)

print(f"Total Effect: {result['total_effect']:.4f}")
print(f"Direct Effect (ADE): {result['ade']:.4f}")
print(f"Indirect Effect (ACME): {result['acme']:.4f}")
print(f"Proportion Mediated: {result['prop_mediated']:.1%}")
```

**Method B: ML-Enhanced Mediation**

For high-dimensional settings, use DDML approach:

```python
from mediation_estimator import estimate_ml_mediation

result = estimate_ml_mediation(
    data=df,
    outcome="y",
    treatment="d",
    mediator="m",
    controls=control_vars,
    ml_m='lasso',
    ml_y='random_forest',
    n_folds=5
)
```

**Method C: Simulation-Based (Imai et al., 2010)**

Most general approach using potential outcomes framework:

```python
from mediation_estimator import run_full_mediation_analysis

result = run_full_mediation_analysis(
    data=df,
    outcome="y",
    treatment="d",
    mediator="m",
    controls=control_vars,
    n_simulations=1000
)
```

### Phase 4: Effect Decomposition

**Formal Definitions**:

Let $Y_i(d, m)$ denote the potential outcome for unit $i$ under treatment $d$ and mediator $m$.
Let $M_i(d)$ denote the potential mediator value under treatment $d$.

**Average Direct Effect (ADE)**:
$$
ADE(d) = E[Y_i(1, M_i(d)) - Y_i(0, M_i(d))]
$$
Effect of changing D while holding M at its natural value under treatment $d$.

**Average Causal Mediation Effect (ACME)**:
$$
ACME(d) = E[Y_i(d, M_i(1)) - Y_i(d, M_i(0))]
$$
Effect of changing M (from what it would be under control to what it would be under treatment) while holding D fixed at $d$.

**Total Effect**:
$$
TE = ADE(0) + ACME(1) = ADE(1) + ACME(0)
$$

**Proportion Mediated**:
$$
\pi_{med} = \frac{ACME}{Total Effect}
$$

**Interpretation Table**:

| Effect | Formula | Interpretation |
|--------|---------|----------------|
| Total Effect | $E[Y(1) - Y(0)]$ | Overall causal effect |
| ADE | $E[Y(1, M(0)) - Y(0, M(0))]$ | Direct pathway contribution |
| ACME | $E[Y(1, M(1)) - Y(1, M(0))]$ | Indirect (mediated) contribution |
| Proportion Mediated | ACME / Total | Share of effect through mediator |

### Phase 5: Sensitivity Analysis

**Why Sensitivity Analysis?**
Sequential ignorability is untestable. We must assess how sensitive results are to violations.

**Sensitivity Parameter ($\rho$)**:
Correlation between error terms in mediator and outcome models that would arise from unmeasured confounding.

```python
from mediation_estimator import sensitivity_analysis_mediation

# Run sensitivity analysis
sensitivity = sensitivity_analysis_mediation(
    acme=result['acme'],
    acme_se=result['acme_se'],
    rho_range=np.arange(-0.5, 0.51, 0.1)
)

# Find breakpoint: what rho makes ACME = 0?
print(f"ACME crosses zero at rho = {sensitivity['breakpoint']:.3f}")
```

**Interpretation**:
- If $|\rho_{breakpoint}|$ is small (e.g., < 0.1), results are SENSITIVE to confounding
- If $|\rho_{breakpoint}|$ is large (e.g., > 0.3), results are ROBUST

### Phase 6: Reporting

**Standard Output**:

```python
from mediation_estimator import run_full_mediation_analysis

result = run_full_mediation_analysis(
    data=df,
    outcome="earnings",
    treatment="training",
    mediator="skills",
    controls=control_vars
)

print(result.summary_table)
```

**Output Table Format**:
```
==================================================================
               CAUSAL MEDIATION ANALYSIS RESULTS
==================================================================

Causal Pathway: training -> skills -> earnings

------------------------------------------------------------------
Effect Decomposition
------------------------------------------------------------------
                          Estimate    Std.Err    95% CI
------------------------------------------------------------------
Total Effect              0.150***    (0.035)    [0.081, 0.219]
Direct Effect (ADE)       0.090***    (0.028)    [0.035, 0.145]
Indirect Effect (ACME)    0.060***    (0.018)    [0.025, 0.095]
------------------------------------------------------------------
Proportion Mediated       40.0%

------------------------------------------------------------------
Sensitivity Analysis
------------------------------------------------------------------
ACME = 0 at rho = 0.25 (moderately robust)

==================================================================
```

---

## Traditional vs. ML Approach Comparison

| Aspect | Baron-Kenny | ML-Enhanced |
|--------|-------------|-------------|
| Functional Form | Linear (parametric) | Flexible (nonparametric) |
| High-Dimensional X | May fail | Handles well |
| Interpretation | Coefficients are intuitive | Black-box nuisance |
| Sample Size | Works with smaller n | Needs larger n (>500) |
| Inference | Standard SEs | Cross-fitting + bootstrap |
| Nonlinear Effects | Limited | Captures naturally |
| When to Use | Simple settings, clear theory | Complex confounding, many X |

---

## Common Mistakes

### 1. Ignoring Post-Treatment Confounding

**Mistake**: Not accounting for variables that are affected by D and affect both M and Y.

**Why it's wrong**: Creates bias in mediation estimates even if sequential ignorability holds for observables.

**Correct approach**:
```python
# WRONG: Ignoring post-treatment confounder Z
result = estimate_baron_kenny(data, "y", "d", "m", controls=["x1", "x2"])

# CORRECT: Either include Z (if it blocks the backdoor)
# or acknowledge limitation
# Better: Use design-based approaches (experimental manipulation of M)

# For observational data, conduct sensitivity analysis
sensitivity = sensitivity_analysis_mediation(result['acme'], result['acme_se'])
```

### 2. Confusing Correlation for Mediation

**Mistake**: Claiming mediation just because D→M and M→Y are correlated.

**Why it's wrong**: Correlation doesn't imply the causal chain D→M→Y.

**Correct approach**:
```python
# Always verify temporal ordering
# 1. D must precede M in time
# 2. M must precede Y in time

# WRONG: Using cross-sectional data where M and Y measured simultaneously
# CORRECT: Use panel data or experimental design
```

### 3. Treating Proportion Mediated as Stable

**Mistake**: Reporting proportion mediated as a fixed quantity.

**Why it's wrong**: Proportion depends on scale of total effect; can be > 100% or negative.

**Correct approach**:
```python
# Report with caveats
prop_med = result['acme'] / result['total_effect']

if prop_med > 1.0:
    print("Warning: Proportion > 100% suggests inconsistent mediation "
          "(direct and indirect effects have opposite signs)")
elif prop_med < 0:
    print("Warning: Negative proportion suggests suppression effect")
else:
    print(f"Proportion mediated: {prop_med:.1%}")
    print("Note: This is effect-scale dependent and may vary across samples")
```

### 4. Not Conducting Sensitivity Analysis

**Mistake**: Reporting mediation results without assessing robustness to unmeasured confounding.

**Why it's wrong**: Sequential ignorability is untestable; results may be fragile.

**Correct approach**:
```python
# ALWAYS include sensitivity analysis
from mediation_estimator import sensitivity_analysis_mediation

sensitivity = sensitivity_analysis_mediation(
    acme=result['acme'],
    acme_se=result['acme_se'],
    rho_range=np.arange(-0.5, 0.51, 0.05)
)

# Report breakpoint
print(f"Results robust up to rho = {sensitivity['breakpoint']:.2f}")
print(f"Interpretation: {sensitivity['interpretation']}")
```

### 5. Using Mediation with Colliders

**Mistake**: Including a variable M that is caused by both D and Y (or their common causes).

**Why it's wrong**: Conditioning on a collider opens backdoor paths and biases all estimates.

**Correct approach**:
```python
# Draw DAG first!
# If M is a collider (common effect), DO NOT use mediation analysis

# Example collider structure (WRONG to use M as mediator):
#    D ──────────→ Y
#     \          /
#      ↘        ↙
#         M (collider)

# Correct: Ensure M is on causal pathway D → M → Y
# Verify temporal ordering: D precedes M precedes Y
```

---

## Examples

### Example 1: Job Training, Skills, and Earnings

**Research Question**: Does a job training program increase earnings through skill acquisition?

**Causal Structure**:
```
Training (D) ──→ Skills (M) ──→ Earnings (Y)
      \____________________________↗
              (direct effect)
```

**Analysis**:
```python
import pandas as pd
from mediation_estimator import run_full_mediation_analysis

# Load data
data = pd.read_csv("job_training_study.csv")

# Define variables
outcome = "log_earnings"
treatment = "training"  # Binary: 0/1
mediator = "skill_score"  # Post-training skill assessment
controls = [
    "age", "education", "prior_earnings",
    "married", "black", "hispanic"
]

# Run full mediation analysis
result = run_full_mediation_analysis(
    data=data,
    outcome=outcome,
    treatment=treatment,
    mediator=mediator,
    controls=controls,
    method='ml_enhanced'  # Use DDML for high-dimensional controls
)

# View results
print(result.summary_table)

# Sensitivity analysis
print("\nSensitivity to Unmeasured Confounding:")
print(f"ACME becomes zero at rho = {result.diagnostics['sensitivity']['breakpoint']:.3f}")
```

**Expected Output**:
```
==================================================================
               CAUSAL MEDIATION ANALYSIS RESULTS
==================================================================

Research Question: Effect of training on log_earnings via skill_score

------------------------------------------------------------------
Effect Decomposition
------------------------------------------------------------------
                          Estimate    Std.Err    95% CI
------------------------------------------------------------------
Total Effect              0.082***    (0.015)    [0.053, 0.111]
Direct Effect (ADE)       0.049***    (0.012)    [0.026, 0.072]
Indirect Effect (ACME)    0.033***    (0.008)    [0.017, 0.049]
------------------------------------------------------------------
Proportion Mediated       40.2%       (7.5%)     [25.5%, 54.9%]

------------------------------------------------------------------
Interpretation
------------------------------------------------------------------
The job training program increases log earnings by 0.082 (8.2%).
Of this total effect:
  - 40.2% operates through improved skills (ACME = 0.033)
  - 59.8% operates through other channels (ADE = 0.049)

------------------------------------------------------------------
Sensitivity Analysis
------------------------------------------------------------------
The indirect effect (ACME) crosses zero at rho = 0.28
Interpretation: Results are MODERATELY ROBUST to unmeasured confounding.
An unmeasured confounder would need to induce a correlation of 0.28
between M and Y residuals to eliminate the mediation effect.

==================================================================
```

### Example 2: ML-Enhanced Mediation with High-Dimensional Controls

```python
from mediation_estimator import estimate_ml_mediation, compare_mediation_methods

# High-dimensional setting
data = pd.read_csv("education_study.csv")

# Many potential confounders
controls = [f"x{i}" for i in range(1, 101)]  # 100 controls

# Compare traditional vs ML approach
comparison = compare_mediation_methods(
    data=data,
    outcome="test_scores",
    treatment="tutoring",
    mediator="study_hours",
    controls=controls
)

print(comparison.summary_table)
```

**Output**:
```
### Mediation Analysis: Method Comparison

| Method | ACME | SE | 95% CI | Prop. Med. |
|:-------|:----:|:--:|:------:|:----------:|
| Baron-Kenny (OLS) | 0.045*** | (0.018) | [0.010, 0.080] | 35.2% |
| ML-Enhanced (Lasso) | 0.052*** | (0.015) | [0.023, 0.081] | 38.8% |
| ML-Enhanced (RF) | 0.048*** | (0.016) | [0.017, 0.079] | 36.4% |

*Notes: ML methods use 5-fold cross-fitting. *** p<0.01*
```

---

## References

### Seminal Papers
- Baron, R. M., & Kenny, D. A. (1986). The Moderator-Mediator Variable Distinction in Social Psychological Research. *Journal of Personality and Social Psychology*, 51(6), 1173-1182.
- Imai, K., Keele, L., & Tingley, D. (2010). A General Approach to Causal Mediation Analysis. *Psychological Methods*, 15(4), 309-334.
- Imai, K., Keele, L., & Yamamoto, T. (2010). Identification, Inference and Sensitivity Analysis for Causal Mediation Effects. *Statistical Science*, 25(1), 51-71.

### ML Extensions
- Chernozhukov, V., et al. (2018). Double/Debiased Machine Learning for Treatment and Structural Parameters. *The Econometrics Journal*, 21(1), C1-C68.
- Farbmacher, H., Huber, M., Laffers, L., Langen, H., & Spindler, M. (2022). Causal Mediation Analysis with Double Machine Learning. *The Econometrics Journal*, 25(2), 277-300.
- Semenova, V., Goldman, M., Chernozhukov, V., & Taddy, M. (2024). Estimation and Inference on Causal Effects in High-Dimensional Mediation. arXiv:2101.00282.

### Textbook Treatments
- VanderWeele, T. J. (2015). *Explanation in Causal Inference: Methods for Mediation and Interaction*. Oxford University Press.
- Pearl, J. (2012). The Mediation Formula: A Guide to the Assessment of Causal Pathways in Nonlinear Models. In *Causality: Statistical Perspectives and Applications* (pp. 151-179).

### Software Documentation
- `mediation` (R): https://cran.r-project.org/web/packages/mediation/
- `lavaan` (R): https://lavaan.ugent.be/
- `econml` (Python): https://econml.azurewebsites.net/

---

## Related Estimators

| Estimator | When to Use Instead |
|-----------|---------------------|
| `causal-ddml` | No mediator; just want ATE with high-dimensional controls |
| `causal-forest` | Treatment effect heterogeneity (not mechanism) |
| `estimator-iv` | Mediator is endogenous, have instrument for M |
| `estimator-did` | Panel data, staggered treatment timing |

---

## Appendix: Mathematical Details

### Potential Outcomes Framework for Mediation

Define:
- $Y_i(d, m)$: Potential outcome under treatment $d$ and mediator $m$
- $M_i(d)$: Potential mediator under treatment $d$

**Observed Outcome**:
$$
Y_i = Y_i(D_i, M_i(D_i))
$$

**Causal Mediation Effect for Individual $i$**:
$$
\delta_i(d) = Y_i(d, M_i(1)) - Y_i(d, M_i(0))
$$

**Direct Effect for Individual $i$**:
$$
\zeta_i(d) = Y_i(1, M_i(d)) - Y_i(0, M_i(d))
$$

### Sequential Ignorability Assumption

**Part 1** (Treatment Assignment):
$$
\{Y_i(d', m), M_i(d)\} \perp\!\!\!\perp D_i | X_i = x
$$

**Part 2** (Mediator Assignment):
$$
Y_i(d', m) \perp\!\!\!\perp M_i(d) | D_i = d, X_i = x
$$

### Identification Under Sequential Ignorability

**ACME**:
$$
ACME(d) = \int \left[ E[Y|D=d, M=m_1, X=x] - E[Y|D=d, M=m_0, X=x] \right] dF_{M|D=1,X}(m_1|x) dF_{M|D=0,X}(m_0|x) dF_X(x)
$$

**ADE**:
$$
ADE(d) = \int \left[ E[Y|D=1, M=m, X=x] - E[Y|D=0, M=m, X=x] \right] dF_{M|D=d,X}(m|x) dF_X(x)
$$

### Linear Model Special Case

Under linear models:
$$
M_i = \alpha_0 + \alpha_1 D_i + \alpha_2' X_i + \epsilon_{Mi}
$$
$$
Y_i = \beta_0 + \beta_1 D_i + \beta_2 M_i + \beta_3' X_i + \epsilon_{Yi}
$$

The effects simplify to:
$$
ACME = \alpha_1 \cdot \beta_2
$$
$$
ADE = \beta_1
$$
$$
TE = \beta_1 + \alpha_1 \cdot \beta_2
$$

### Sensitivity Analysis Formula (Imai et al.)

Under sensitivity parameter $\rho$ (correlation between $\epsilon_M$ and $\epsilon_Y$):
$$
ACME(\rho) = ACME(0) - \rho \cdot \sigma_M \cdot \sigma_{Y|D,M,X}
$$

The breakpoint $\rho^*$ where $ACME(\rho^*) = 0$:
$$
\rho^* = \frac{ACME(0)}{\sigma_M \cdot \sigma_{Y|D,M,X}}
$$
