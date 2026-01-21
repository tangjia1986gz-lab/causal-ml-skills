# IV Identification Assumptions

> **Reference Document** | IV Estimator Skill
> **Last Updated**: 2024

## Overview

Instrumental Variables (IV) estimation requires strong assumptions to identify causal effects. Unlike OLS, which requires exogeneity of all regressors, IV isolates exogenous variation in the endogenous treatment using external instruments. This document details the identification assumptions following Angrist-Pischke methodology.

---

## The Four Core Assumptions

### 1. Relevance (First-Stage)

**Formal Statement**:
$$
Cov(Z_i, D_i) \neq 0
$$

Or in regression form:
$$
D_i = \pi_0 + \pi_1 Z_i + X_i'\gamma + v_i \quad \text{where } \pi_1 \neq 0
$$

**Interpretation**: The instrument must be correlated with the endogenous treatment. Without this correlation, there is no "lever" to move the treatment variable.

**Testability**: **YES** - This is the only IV assumption that can be directly tested.

**Testing Procedure**:
1. Regress treatment D on instruments Z and controls X
2. Compute F-statistic for joint significance of instruments
3. Compare to Stock-Yogo critical values

**Rule of Thumb** (Staiger & Stock, 1997):
- F > 10: Instruments considered strong
- F between 5-10: Weak instruments concern
- F < 5: Severe weak instruments problem

**Python Implementation**:
```python
from iv_estimator import first_stage_test, weak_iv_diagnostics

first_stage = first_stage_test(
    data=df,
    treatment="d",
    instruments=["z1", "z2"],
    controls=["x1", "x2"]
)

print(f"First-stage F: {first_stage['f_statistic']:.2f}")
print(f"Partial R-squared: {first_stage['partial_r2']:.4f}")

# Stock-Yogo assessment
weak_iv = weak_iv_diagnostics(
    first_stage_f=first_stage['f_statistic'],
    n_instruments=len(instruments)
)
print(weak_iv.interpretation)
```

---

### 2. Independence (Exogeneity)

**Formal Statement**:
$$
Cov(Z_i, \epsilon_i) = 0
$$

Or in potential outcomes framework:
$$
Z_i \perp\!\!\!\perp (Y_i(0), Y_i(1), D_i(0), D_i(1))
$$

**Interpretation**: The instrument must be uncorrelated with unobserved factors that affect the outcome. The instrument should be "as-good-as-randomly assigned."

**Testability**: **NO** - This assumption cannot be directly tested because it involves unobservables.

**Assessment Strategies**:

1. **Balance Tests**: Check if Z is correlated with observable characteristics
   ```python
   # Regress each covariate on instruments
   for covariate in covariates:
       model = OLS(df[covariate], sm.add_constant(df[instruments])).fit()
       print(f"{covariate}: F = {model.fvalue:.2f}, p = {model.f_pvalue:.4f}")
   ```

2. **Institutional Knowledge**: Understand the mechanism that generates variation in Z

3. **Falsification Tests**: Test if Z predicts outcomes it should not affect
   ```python
   # Test on pre-treatment outcomes (if available)
   placebo_result = estimate_reduced_form(
       data=df,
       outcome="pre_treatment_outcome",
       instruments=instruments,
       controls=controls
   )
   # Should be close to zero
   ```

4. **Sensitivity Analysis**: Examine how violations would affect estimates (Conley et al., 2012)

---

### 3. Exclusion Restriction

**Formal Statement**:
$$
Y_i = g(D_i, X_i, \epsilon_i) \quad \text{(Z enters only through D)}
$$

Or equivalently:
$$
\frac{\partial Y}{\partial Z} \bigg|_{D, X, \epsilon} = 0
$$

**Interpretation**: The instrument affects the outcome ONLY through its effect on the treatment. There is no direct path from Z to Y that bypasses D.

**Testability**: **NO** - This is a structural assumption about the causal mechanism.

**Common Violations**:

| Instrument Type | Potential Violation |
|-----------------|---------------------|
| Geographic distance | Direct effects on outcomes (local labor markets) |
| Birth timing | Cohort effects, parental characteristics |
| Lottery numbers | Psychological effects beyond treatment |
| Policy changes | Bundled with other policy changes |

**Assessment Approaches**:

1. **Causal Diagram Analysis**: Draw the DAG and identify all paths from Z to Y
   ```
   Valid:    Z --> D --> Y

   Invalid:  Z --> D --> Y
             |          ^
             +----------+
   ```

2. **Overidentification Test**: With multiple instruments, test if they give same estimate
   ```python
   from iv_estimator import overidentification_test

   j_test = overidentification_test(model_result, data, ...)
   # Rejection suggests at least one instrument is invalid
   ```

3. **Subset Analysis**: Compare estimates using different instrument subsets
   ```python
   # Estimate with each instrument separately
   for z in instruments:
       result_z = estimate_2sls(data, outcome, treatment, [z], controls)
       print(f"Instrument {z}: {result_z.effect:.4f}")
   # Estimates should be similar if all instruments are valid
   ```

---

### 4. Monotonicity (for LATE)

**Formal Statement**:
$$
D_i(Z=1) \geq D_i(Z=0) \quad \forall i
$$

Or:
$$
D_i(Z=1) \leq D_i(Z=0) \quad \forall i
$$

**Interpretation**: The instrument affects everyone's treatment in the same direction. There are no "defiers" - units who do the opposite of what the instrument encourages.

**Why It Matters**: Without monotonicity, IV identifies a weighted average of treatment effects where some weights can be negative, making interpretation problematic.

**Testability**: **NO** - This involves counterfactual treatment status.

**When Monotonicity is Plausible**:
- One-sided noncompliance (lottery to receive treatment, cannot refuse)
- Strong institutional constraints
- Clear economic incentives aligned with instrument

**When Monotonicity May Fail**:
- Two-sided noncompliance
- Heterogeneous responses to instrument
- Strategic behavior (defiance)

**Assessment**:
1. Examine institutional details
2. Look for negative weights in specification tests
3. Consider bounds analysis if monotonicity uncertain

---

## The LATE Framework

When treatment effects are heterogeneous, IV with binary instrument identifies the **Local Average Treatment Effect (LATE)** for compliers:

$$
\beta_{IV} = \frac{E[Y|Z=1] - E[Y|Z=0]}{E[D|Z=1] - E[D|Z=0]} = E[Y_i(1) - Y_i(0) | \text{Complier}]
$$

### Population Decomposition

| Type | Definition | $D(0)$ | $D(1)$ | Identified? |
|------|------------|--------|--------|-------------|
| **Always-takers** | Always treated | 1 | 1 | No |
| **Never-takers** | Never treated | 0 | 0 | No |
| **Compliers** | Follow instrument | 0 | 1 | Yes |
| **Defiers** | Oppose instrument | 1 | 0 | Ruled out |

### Share of Compliers

The first-stage coefficient estimates the share of compliers (in a binary setting):

$$
\pi_1 = E[D|Z=1] - E[D|Z=0] = P(\text{Complier})
$$

### External Validity Concerns

LATE may differ from ATE if:
- Compliers are not representative of population
- Treatment effects are heterogeneous
- Different instruments identify different complier groups

**Best Practice**: Be explicit about which population your estimate represents and discuss external validity carefully.

---

## Practical Checklist

Before running IV estimation, answer these questions:

### Relevance
- [ ] Is the first-stage F-statistic > 10?
- [ ] What is the economic/behavioral mechanism linking Z to D?
- [ ] Is the relationship robust to different specifications?

### Independence
- [ ] Is Z plausibly "as-good-as-randomly assigned"?
- [ ] Are observables balanced across Z values?
- [ ] Can you rule out confounders of the Z-Y relationship?

### Exclusion Restriction
- [ ] Can you articulate why Z affects Y only through D?
- [ ] Have you considered all possible direct effects of Z on Y?
- [ ] Does overidentification test pass (if applicable)?

### Monotonicity
- [ ] Is it plausible that Z affects D in the same direction for all units?
- [ ] Are there groups who might "defy" the instrument?
- [ ] Do institutional details support monotonicity?

---

## References

### Core Methodology
- Imbens, G. W., & Angrist, J. D. (1994). Identification and Estimation of Local Average Treatment Effects. *Econometrica*, 62(2), 467-475.
- Angrist, J. D., Imbens, G. W., & Rubin, D. B. (1996). Identification of Causal Effects Using Instrumental Variables. *JASA*, 91(434), 444-455.

### Weak Instruments
- Stock, J. H., & Yogo, M. (2005). Testing for Weak Instruments in Linear IV Regression. In *Identification and Inference for Econometric Models*.
- Staiger, D., & Stock, J. H. (1997). Instrumental Variables Regression with Weak Instruments. *Econometrica*, 65(3), 557-586.

### Exclusion Restriction & Sensitivity
- Conley, T. G., Hansen, C. B., & Rossi, P. E. (2012). Plausibly Exogenous. *Review of Economics and Statistics*, 94(1), 260-272.

### Textbooks
- Angrist, J. D., & Pischke, J. S. (2009). *Mostly Harmless Econometrics*. Princeton University Press. Chapter 4.
- Angrist, J. D., & Pischke, J. S. (2015). *Mastering 'Metrics*. Princeton University Press. Chapter 3.
