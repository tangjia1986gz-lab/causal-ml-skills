# RD Identification Assumptions

> **Document Type**: Methodological Reference
> **Last Updated**: 2024-01
> **Key References**: Imbens & Lemieux (2008), Cattaneo, Idrobo & Titiunik (2020)

## Overview

Regression Discontinuity (RD) designs exploit discontinuous changes in treatment assignment at a known cutoff of a running variable. The causal interpretation relies on specific assumptions about the behavior of potential outcomes and agents near the threshold.

---

## Core Assumptions

### 1. Continuity of Potential Outcomes

**Statement**: The conditional expectation functions $E[Y(0)|X=x]$ and $E[Y(1)|X=x]$ are continuous in $x$ at the cutoff $c$.

**Mathematical Formulation**:
$$
\lim_{x \downarrow c} E[Y(0)|X=x] = E[Y(0)|X=c] \quad \text{and} \quad \lim_{x \uparrow c} E[Y(0)|X=x] = E[Y(0)|X=c]
$$

**Implications**:
- In the absence of treatment, outcomes would be smooth at the cutoff
- Any discontinuity in the observed outcome at $c$ is attributable to treatment
- This is the **fundamental identifying assumption** - it cannot be directly tested

**What Violates This**:
- Simultaneous policy changes at the same cutoff
- Other programs with the same eligibility threshold
- Measurement error in the running variable that varies at cutoff
- Anticipation effects where agents change behavior before crossing threshold

**Strengthening the Assumption**:
1. Use institutional knowledge to argue for no other discontinuities
2. Check for policy stacking at the cutoff
3. Verify measurement of running variable is consistent around cutoff

---

### 2. No Manipulation (No Precise Sorting)

**Statement**: Agents cannot precisely manipulate their value of the running variable to sort around the cutoff.

**Mathematical Formulation**:
The density of $X$ is continuous at $c$:
$$
\lim_{x \downarrow c} f_X(x) = \lim_{x \uparrow c} f_X(x)
$$

**Types of Manipulation**:

| Type | Description | Example | Testable |
|------|-------------|---------|----------|
| **Precise** | Agents can choose exact value of $X$ | Self-reported income | Yes (McCrary) |
| **Imprecise** | Agents influence $X$ but not precisely | Effort on test | Often not problematic |
| **Third-party** | Others manipulate on agents' behalf | Teachers inflating scores | Yes (McCrary) |

**Key Insight**: The assumption allows for **imprecise** influence on the running variable. What matters is that agents cannot **precisely** control whether they end up just above or just below the cutoff.

**Testing**:
```python
from rd_estimator import mccrary_test

# Conduct McCrary density test
result = mccrary_test(
    running=df["score"],
    cutoff=0.0,
    bandwidth=None  # Auto-select
)

if not result.passed:
    print("WARNING: Evidence of manipulation detected")
    print("Consider: donut hole RD, alternative identification")
```

---

### 3. Local Randomization Interpretation

**Statement**: Sufficiently close to the cutoff, treatment assignment is "as-if random."

**Formal Framework** (Cattaneo, Frandsen, Titiunik 2015):
Within a window $[c - w, c + w]$ around the cutoff:
1. Treatment $D$ is independent of potential outcomes: $D \perp\!\!\!\perp (Y(0), Y(1))$
2. Covariates $X$ are balanced across treatment status

**Practical Implications**:
- Near the cutoff, treated and control units are comparable
- Covariates should not show discontinuities at the cutoff
- Supports using finite-sample inference methods

**Testing Covariate Balance**:
```python
from rd_estimator import covariate_balance_rd

balance = covariate_balance_rd(
    data=df,
    running="score",
    cutoff=0.0,
    covariates=["age", "gender", "income", "education"],
    bandwidth=optimal_bw
)

for cov, result in balance.items():
    status = "Balanced" if result.passed else "IMBALANCED"
    print(f"{cov}: {status} (p={result.p_value:.3f})")
```

**Interpretation**:
- Balanced covariates support (but don't prove) local randomization
- Imbalanced covariates suggest sorting or confounding
- Even with balance, the RD estimate is a LATE, not an ATE

---

### 4. SUTVA (Stable Unit Treatment Value Assumption)

**Statement**:
1. No interference between units
2. Well-defined treatment levels (no hidden treatment variation)

**In RD Context**:
- Treatment effect for unit $i$ depends only on their own treatment status
- Crossing the cutoff has the same meaning for all units

**Potential Violations**:
- Peer effects when classmates/neighbors are also treated
- General equilibrium effects in large-scale programs
- Heterogeneous treatment intensity based on how far above cutoff

---

### 5. Correct Functional Form (For Parametric Approaches)

**Statement**: The relationship between $X$ and $E[Y|X]$ is correctly specified on each side of the cutoff.

**Why This Matters**:
- Global polynomial approaches require correct specification
- Local polynomial with appropriate bandwidth is more robust
- Higher-order polynomials can introduce bias near boundaries

**Best Practice**:
1. Use **local linear regression** (polynomial order 1) as default
2. With optimal bandwidth selection (MSE or CER)
3. Test robustness to polynomial order (1 vs 2)

```python
# Prefer local linear with optimal bandwidth
result = estimate_sharp_rd(
    data=df,
    running="score",
    outcome="y",
    cutoff=0.0,
    bandwidth=None,  # Auto MSE-optimal
    order=1          # Local linear (default)
)
```

---

## Assumption Hierarchy

```
                    +---------------------------+
                    |     CONTINUITY (Core)     |
                    |   Cannot be tested        |
                    +-------------+-------------+
                                  |
            +---------------------+---------------------+
            |                                           |
+-----------v-----------+                   +-----------v-----------+
|   NO MANIPULATION     |                   |  LOCAL RANDOMIZATION  |
| Testable via McCrary  |                   |  Testable via balance |
+-----------+-----------+                   +-----------+-----------+
            |                                           |
            +---------------------+---------------------+
                                  |
                    +-------------v-------------+
                    |         SUTVA             |
                    | Context-dependent         |
                    +---------------------------+
```

---

## Checking Assumptions: Practical Checklist

### Pre-Estimation Checks

- [ ] **Institutional Knowledge**: Is there only ONE discontinuity at the cutoff?
- [ ] **Running Variable**: Is it measured consistently around the cutoff?
- [ ] **Treatment Definition**: Is crossing the cutoff well-defined?
- [ ] **Sample Size**: Sufficient observations on both sides of cutoff?

### Statistical Tests

- [ ] **McCrary Test**: Run density test for manipulation
- [ ] **Covariate Balance**: Test all pre-treatment covariates at cutoff
- [ ] **Placebo Cutoffs**: Test for effects at fake cutoffs
- [ ] **Histogram Inspection**: Visually check for bunching

### Post-Estimation Sensitivity

- [ ] **Bandwidth Sensitivity**: Stable across reasonable bandwidth range?
- [ ] **Polynomial Order**: Similar results with linear vs quadratic?
- [ ] **Donut Hole**: Excluding observations near cutoff affects results?
- [ ] **Kernel Choice**: Robust to triangular vs uniform kernel?

---

## Common Threats to Validity

### Threat 1: Multiple Cutoffs at Same Threshold

**Problem**: Another program has the same eligibility threshold.

**Example**: Both a scholarship program AND tutoring eligibility are determined by the same test score cutoff.

**Solutions**:
1. Document all policies at the cutoff
2. Consider this as a bundled treatment
3. If possible, find variation where cutoffs differ

### Threat 2: Manipulation Through Third Parties

**Problem**: Someone other than the agent manipulates scores.

**Example**: Teachers round grades to help students pass; administrators adjust eligibility scores.

**Detection**:
- McCrary test may detect aggregate manipulation
- Look for discrete heaping at round numbers
- Check for implausible score patterns

### Threat 3: Anticipation Effects

**Problem**: Agents change behavior in anticipation of treatment before actually crossing the cutoff.

**Example**: Students expecting to receive a scholarship start studying harder before results are announced.

**Implications**:
- Continuity assumption may be violated
- Effect estimate captures more than the direct treatment effect

### Threat 4: Measurement Error in Running Variable

**Problem**: Running variable is measured with error, and the error structure differs around the cutoff.

**Detection**:
- Check for heaping at round numbers
- Verify data collection was consistent
- Consider bounds or sensitivity analysis

---

## Sharp vs Fuzzy RD: Implications for Identification

### Sharp RD

**Setup**: $D_i = \mathbf{1}(X_i \geq c)$

Treatment is **deterministic** function of running variable.

**Identification**: Under continuity, the RD estimand is:
$$
\tau_{SRD} = E[Y(1) - Y(0) | X = c]
$$

This is the **Average Treatment Effect at the Cutoff**.

### Fuzzy RD

**Setup**: $P(D_i = 1 | X_i = x)$ is discontinuous at $c$, but $0 < \lim_{x \uparrow c} P(D|X=x) < \lim_{x \downarrow c} P(D|X=x) < 1$

**Identification**: Under continuity + monotonicity:
$$
\tau_{FRD} = \frac{\lim_{x \downarrow c} E[Y|X=x] - \lim_{x \uparrow c} E[Y|X=x]}{\lim_{x \downarrow c} P(D=1|X=x) - \lim_{x \uparrow c} P(D=1|X=x)}
$$

This is the **Local Average Treatment Effect for Compliers at the Cutoff**.

**Additional Assumption**: Monotonicity - no defiers (units that would be treated below the cutoff but not above).

---

## References

- Hahn, J., Todd, P., & Van der Klaauw, W. (2001). Identification and estimation of treatment effects with a regression-discontinuity design. *Econometrica*, 69(1), 201-209.
- Imbens, G. W., & Lemieux, T. (2008). Regression discontinuity designs: A guide to practice. *Journal of Econometrics*, 142(2), 615-635.
- Lee, D. S., & Lemieux, T. (2010). Regression discontinuity designs in economics. *Journal of Economic Literature*, 48(2), 281-355.
- Cattaneo, M. D., Idrobo, N., & Titiunik, R. (2020). *A Practical Introduction to Regression Discontinuity Designs: Foundations*. Cambridge University Press.
- Cattaneo, M. D., Frandsen, B. R., & Titiunik, R. (2015). Randomization inference in the regression discontinuity design. *Journal of Causal Inference*, 3(1), 1-24.
