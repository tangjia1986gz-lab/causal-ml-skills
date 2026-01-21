# Potential Outcomes Framework (Rubin Causal Model)

## Overview

The potential outcomes framework, also known as the Rubin Causal Model (RCM), provides the mathematical foundation for defining and estimating causal effects. Developed by Donald Rubin building on earlier work by Jerzy Neyman, this framework is now the dominant paradigm in applied causal inference.

---

## Core Notation

### Unit-Level Potential Outcomes

For each unit $i$ in a population:

| Symbol | Definition |
|--------|------------|
| $Y_i(0)$ | Potential outcome if unit $i$ does NOT receive treatment |
| $Y_i(1)$ | Potential outcome if unit $i$ DOES receive treatment |
| $D_i$ | Treatment indicator: $D_i = 1$ if treated, $D_i = 0$ if control |
| $Y_i$ | Observed outcome |

### Fundamental Relationship

The observed outcome is determined by treatment status:

$$Y_i = D_i \cdot Y_i(1) + (1 - D_i) \cdot Y_i(0)$$

This is called the **switching equation** - we "switch" between potential outcomes based on treatment.

---

## The Fundamental Problem of Causal Inference

**Key Insight**: For any single unit, we can only ever observe ONE potential outcome.

- If $D_i = 1$: We observe $Y_i(1)$, but $Y_i(0)$ is the **counterfactual**
- If $D_i = 0$: We observe $Y_i(0)$, but $Y_i(1)$ is the **counterfactual**

### Example: Job Training Program

| Worker | Treatment (Training) | $Y(0)$ (Earnings without training) | $Y(1)$ (Earnings with training) | Observed |
|--------|---------------------|-----------------------------------|--------------------------------|----------|
| Alice | 1 (Trained) | ? | $50,000 | $50,000 |
| Bob | 0 (Not trained) | $35,000 | ? | $35,000 |
| Carol | 1 (Trained) | ? | $45,000 | $45,000 |
| Dave | 0 (Not trained) | $40,000 | ? | $40,000 |

The "?" entries are counterfactuals - fundamentally unobservable.

**Individual Treatment Effect**: $\tau_i = Y_i(1) - Y_i(0)$

This is **never directly observable** for any individual.

---

## Counterfactuals

### Definition

A counterfactual is "what would have happened" under a different treatment assignment.

### Types of Counterfactual Questions

1. **Effects of causes** (forward-looking):
   - "What would happen if we implement policy X?"
   - This is the standard causal inference question

2. **Causes of effects** (backward-looking):
   - "What caused outcome Y to occur?"
   - Much harder to answer; requires additional structure

### Counterfactual vs. Prediction

| Aspect | Counterfactual | Prediction |
|--------|---------------|------------|
| Question | What would have happened? | What will happen? |
| Time | Refers to same time point | Refers to future |
| Intervention | Different treatment | No intervention |
| Observability | Never observable | Eventually observable |

### The Counterfactual Mean

For identification, we need to estimate:
- $E[Y(0) \mid D=1]$ - What would treated units have experienced WITHOUT treatment?
- $E[Y(1) \mid D=0]$ - What would control units have experienced WITH treatment?

These counterfactual means are never directly observed.

---

## Treatment Effect Estimands

### Average Treatment Effect (ATE)

$$\text{ATE} = E[Y(1) - Y(0)] = E[Y(1)] - E[Y(0)]$$

**Interpretation**: The expected effect of treatment for a randomly chosen unit from the entire population.

**When to use**:
- Policy will be applied to entire population
- Random assignment from population
- General scientific interest in average effect

### Average Treatment Effect on the Treated (ATT)

$$\text{ATT} = E[Y(1) - Y(0) \mid D=1]$$

**Interpretation**: The expected effect of treatment for those who actually received treatment.

**When to use**:
- Evaluating an existing program
- Treatment is voluntary/selective
- Want to know if treatment helped those who chose it

### Average Treatment Effect on the Untreated (ATU)

$$\text{ATU} = E[Y(1) - Y(0) \mid D=0]$$

**Interpretation**: The expected effect if we were to treat those currently untreated.

**When to use**:
- Considering program expansion
- Want to predict effects for new participants

### Local Average Treatment Effect (LATE)

$$\text{LATE} = E[Y(1) - Y(0) \mid \text{Compliers}]$$

**Interpretation**: The effect for "compliers" - those whose treatment status is affected by an instrument.

**When to use**:
- Instrumental variables estimation
- Treatment assignment is imperfect (non-compliance)

### Conditional Average Treatment Effect (CATE)

$$\text{CATE}(x) = E[Y(1) - Y(0) \mid X=x]$$

**Interpretation**: The treatment effect for units with specific characteristics $X=x$.

**When to use**:
- Heterogeneous treatment effects
- Targeting/personalization
- Understanding effect modifiers

---

## Relationship Between Estimands

### When ATE = ATT = ATU

Under **random assignment**:
- $E[Y(0) \mid D=1] = E[Y(0) \mid D=0] = E[Y(0)]$
- $E[Y(1) \mid D=1] = E[Y(1) \mid D=0] = E[Y(1)]$
- Therefore: ATE = ATT = ATU

### When ATE ≠ ATT

**Selection on gains**: If units who benefit most self-select into treatment:
- $E[Y(1) - Y(0) \mid D=1] > E[Y(1) - Y(0)]$
- ATT > ATE

**Example**: Voluntary job training
- Those who enroll may be more motivated
- Training may benefit motivated workers more
- ATT (effect on enrollees) exceeds ATE (effect on random person)

### Decomposing Selection Bias

$$\underbrace{E[Y \mid D=1] - E[Y \mid D=0]}_{\text{Naive Comparison}} = \underbrace{E[Y(1) - Y(0) \mid D=1]}_{\text{ATT}} + \underbrace{E[Y(0) \mid D=1] - E[Y(0) \mid D=0]}_{\text{Selection Bias}}$$

The naive comparison of means equals the ATT plus selection bias.

---

## LATE and the Complier Framework

### Who are Compliers?

With an instrument $Z$:

| Type | Definition | $D$ when $Z=0$ | $D$ when $Z=1$ |
|------|------------|----------------|----------------|
| **Always-takers** | Always treated regardless of $Z$ | 1 | 1 |
| **Never-takers** | Never treated regardless of $Z$ | 0 | 0 |
| **Compliers** | Treatment follows instrument | 0 | 1 |
| **Defiers** | Treatment opposes instrument | 1 | 0 |

### LATE Interpretation

IV estimates the effect for **compliers only**:
$$\text{LATE} = \frac{E[Y \mid Z=1] - E[Y \mid Z=0]}{E[D \mid Z=1] - E[D \mid Z=0]}$$

### External Validity Concerns

LATE may not generalize to:
- Always-takers (they would take treatment anyway)
- Never-takers (they would never take treatment)
- Different complier populations with different instruments

---

## Assumptions for Identification

### SUTVA (Stable Unit Treatment Value Assumption)

**Two components**:

1. **No interference**: One unit's treatment does not affect another unit's outcome
   - $Y_i$ depends only on $D_i$, not on $D_j$ for $j \neq i$

2. **No hidden variations of treatment**: Treatment is well-defined
   - Only one version of "treatment" and one version of "control"

**Violations**:
- Social programs with spillovers
- Vaccines with herd immunity
- Treatments with different dosages lumped together

### Ignorability (Unconfoundedness)

$$\{Y(0), Y(1)\} \perp\!\!\!\perp D \mid X$$

Given covariates $X$, treatment assignment is independent of potential outcomes.

**Also called**:
- Conditional independence
- Selection on observables
- No unmeasured confounding

### Overlap (Positivity)

$$0 < P(D=1 \mid X=x) < 1 \quad \text{for all } x$$

Every unit has a positive probability of receiving either treatment or control.

**Violations**:
- Deterministic treatment assignment
- Structural zeros (certain groups never treated)

---

## Identification Under Random Assignment

Under randomization:
- Ignorability holds unconditionally: $(Y(0), Y(1)) \perp\!\!\!\perp D$
- No confounding by construction
- Simple difference in means identifies ATE:

$$\hat{\tau} = \bar{Y}_{\text{treated}} - \bar{Y}_{\text{control}}$$

This is why RCTs are the "gold standard" - they provide clean identification.

---

## Common Misunderstandings

### Misconception 1: "Potential outcomes exist in reality"

**Clarification**: Potential outcomes are a mathematical device. Only one outcome ever manifests; the other is a hypothetical construct useful for defining causal effects.

### Misconception 2: "We estimate individual treatment effects"

**Clarification**: We estimate **average** treatment effects. Individual effects ($\tau_i$) are fundamentally unidentifiable without additional assumptions.

### Misconception 3: "Matching finds the counterfactual"

**Clarification**: Matching finds **similar units** whose outcomes serve as an **estimate** of the counterfactual. This is valid only under selection-on-observables assumptions.

### Misconception 4: "ATE and ATT are interchangeable"

**Clarification**: They differ whenever treatment selection correlates with treatment effect magnitude. Policy implications can differ dramatically.

---

## Related Skills

- `estimator-psm` - Propensity Score Matching (requires conditional independence)
- `estimator-iv` - Instrumental Variables (estimates LATE)
- `estimator-did` - Difference-in-Differences (parallel trends assumption)
- `estimator-rd` - Regression Discontinuity (local identification)
- `causal-ddml` - Double ML (high-dimensional conditional independence)

---

## Key References

1. **Rubin, D. B. (1974)**. Estimating causal effects of treatments in randomized and nonrandomized studies. *Journal of Educational Psychology*, 66(5), 688-701.

2. **Holland, P. W. (1986)**. Statistics and causal inference. *Journal of the American Statistical Association*, 81(396), 945-960.

3. **Imbens, G. W., & Rubin, D. B. (2015)**. *Causal Inference for Statistics, Social, and Biomedical Sciences: An Introduction*. Cambridge University Press.

4. **Angrist, J. D., Imbens, G. W., & Rubin, D. B. (1996)**. Identification of causal effects using instrumental variables. *Journal of the American Statistical Association*, 91(434), 444-455.
