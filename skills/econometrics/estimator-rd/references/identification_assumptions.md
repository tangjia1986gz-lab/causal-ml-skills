# RD Identification Assumptions

## Core Assumption: Continuity

The fundamental identification assumption for RD designs:

### Formal Statement

**Assumption (Continuity):** The conditional expectation functions of potential outcomes are continuous at the cutoff:

$$\lim_{x \downarrow c} E[Y(0) | X = x] = \lim_{x \uparrow c} E[Y(0) | X = x]$$

$$\lim_{x \downarrow c} E[Y(1) | X = x] = \lim_{x \uparrow c} E[Y(1) | X = x]$$

### Intuition

- Units just below and just above the cutoff are comparable
- Only the treatment assignment changes discontinuously at the cutoff
- All other factors vary smoothly through the cutoff

### What This Allows

Under continuity, the causal effect at the cutoff is identified:

$$\tau_{RD} = \lim_{x \downarrow c} E[Y | X = x] - \lim_{x \uparrow c} E[Y | X = x]$$

For **Sharp RD**, this equals: $E[Y(1) - Y(0) | X = c]$

For **Fuzzy RD**, this equals the Local Average Treatment Effect (LATE):
$$\tau_{LATE} = \frac{\lim_{x \downarrow c} E[Y | X = x] - \lim_{x \uparrow c} E[Y | X = x]}{\lim_{x \downarrow c} E[D | X = x] - \lim_{x \uparrow c} E[D | X = x]}$$

---

## Sharp vs Fuzzy RD

### Sharp RD

**Definition:** Treatment is deterministically assigned based on running variable:
$$D_i = \mathbf{1}(X_i \geq c)$$

**Identification:** Effect at cutoff for all units

**Interpretation:** Average treatment effect at the cutoff

### Fuzzy RD

**Definition:** Treatment probability jumps at cutoff but not from 0 to 1:
$$\lim_{x \downarrow c} P(D = 1 | X = x) \neq \lim_{x \uparrow c} P(D = 1 | X = x)$$

**Additional Assumption (Monotonicity):**
- Crossing the cutoff weakly increases treatment probability
- No "defiers": no one refuses treatment when eligible but takes it when ineligible

**Identification:** LATE for compliers at the cutoff

**Interpretation:** Effect for units who comply with the treatment rule at the margin

---

## Testable Implications

The continuity assumption has several testable implications:

### 1. Density Continuity (No Manipulation)

If units cannot manipulate their running variable, the density should be continuous:
$$\lim_{x \downarrow c} f_X(x) = \lim_{x \uparrow c} f_X(x)$$

**Test:** McCrary (2008) or Cattaneo-Jansson-Ma (2020) density test

**Violation implies:** Units may be manipulating to get above/below cutoff

### 2. Covariate Continuity (Balance)

Pre-determined covariates should be continuous at the cutoff:
$$\lim_{x \downarrow c} E[W | X = x] = \lim_{x \uparrow c} E[W | X = x]$$

for any covariate $W$ determined before treatment assignment.

**Test:** RD estimation using covariates as outcomes

**Violation implies:** Confounding or manipulation

### 3. Placebo Cutoffs

No effect should be found at fake cutoffs where treatment doesn't change:
$$E[Y | X = c'] - \text{ is continuous for } c' \neq c$$

**Test:** Estimate RD at different cutoff values

---

## Common Violations

### Precise Manipulation

**Problem:** Units can precisely control their running variable

**Example:** Students retaking tests to score just above a scholarship threshold

**Detection:** McCrary test, bunching at cutoff

**Solutions:**
- Donut hole RD (exclude observations very close to cutoff)
- Find alternative design
- Report as limitation

### Compound Treatment

**Problem:** Multiple treatments change at the same cutoff

**Example:** Age 65 brings Medicare eligibility AND Social Security benefits

**Detection:** Theory-driven, examine all policies at cutoff

**Solutions:**
- Isolate individual treatment effects
- Interpret as compound effect
- Find setting with single treatment

### Discontinuous Covariates

**Problem:** Other factors besides treatment change at cutoff

**Detection:** Covariate balance tests

**Solutions:**
- Control for covariates (controversial)
- Question RD validity
- Find better running variable

---

## LATE Interpretation in Fuzzy RD

### Complier Types

| Type | Behavior | Identified? |
|------|----------|-------------|
| Compliers | $D_i = \mathbf{1}(X_i \geq c)$ | Yes (effect identified) |
| Always-takers | $D_i = 1$ always | No |
| Never-takers | $D_i = 0$ always | No |
| Defiers | $D_i = \mathbf{1}(X_i < c)$ | Assumed away |

### What LATE Tells Us

- Effect for marginal compliers at the cutoff
- NOT the average effect for all treated units
- NOT the effect for always-takers or never-takers

### External Validity

Fuzzy RD LATE may not generalize to:
- Units far from the cutoff
- Different populations with different compliance rates
- Policy changes that affect always/never-takers

---

## Key References

- Hahn, Todd, van der Klaauw (2001): "Identification and Estimation of Treatment Effects with a RDD" - foundational identification
- Lee & Lemieux (2010): "Regression Discontinuity Designs in Economics" - comprehensive JEL survey
- Imbens & Lemieux (2008): "Regression Discontinuity Designs: A Guide to Practice" - practical guidance
- Cattaneo, Idrobo, Titiunik (2020): "A Practical Introduction to RD Designs" - modern treatment
