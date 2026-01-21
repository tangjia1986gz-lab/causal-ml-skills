# PSM Identification Assumptions

> **Reference Document** | Propensity Score Matching
> Based on Imbens & Rubin (2015), Rosenbaum & Rubin (1983)

## Overview

Propensity Score Matching (PSM) relies on three fundamental assumptions for valid causal identification. Understanding these assumptions is critical for proper application and interpretation.

---

## 1. Conditional Independence Assumption (CIA) / Unconfoundedness

### Formal Definition

$$
(Y_0, Y_1) \perp D | X
$$

Where:
- $Y_0$: Potential outcome under control
- $Y_1$: Potential outcome under treatment
- $D$: Treatment indicator
- $X$: Observed covariates

### Alternative Names
- **Ignorability** (Rosenbaum & Rubin, 1983)
- **Selection on Observables** (Heckman & Robb, 1985)
- **Exogeneity** (Econometrics)
- **No Unmeasured Confounders**

### Interpretation

Conditional on observed covariates $X$, treatment assignment is independent of potential outcomes. In other words:
- All confounders are observed and included in $X$
- After conditioning on $X$, treatment assignment is "as good as random"
- No unobserved factors jointly affect treatment and outcome

### Testability

**UNTESTABLE** - This is fundamentally untestable because we never observe both potential outcomes for any unit.

### What Can Be Checked

1. **Placebo Tests**: Test for "effects" on pre-treatment outcomes
   - If CIA holds, treatment should not "affect" lagged outcomes

2. **Sensitivity Analysis**: Rosenbaum bounds to assess robustness
   - How much hidden bias would invalidate results?

3. **Auxiliary Outcome Tests**: Check treatment effect on outcomes known to be unaffected
   - Finding an effect suggests unobserved confounding

### Violations

CIA is violated when:
- Important confounders are unobserved
- Selection is based on unobservables (ability, motivation, preferences)
- Reverse causality exists (outcome affects treatment)

### Example Violation

**Job Training Study**:
- Treatment: Participation in job training program
- Outcome: Post-program earnings
- **Problem**: Participants may be more motivated (unobserved) -> Higher earnings anyway
- CIA violated: Motivation affects both treatment and outcome

---

## 2. Common Support / Overlap Assumption

### Formal Definition

$$
0 < P(D = 1 | X = x) < 1 \quad \forall x \in \mathcal{X}
$$

Or equivalently:
$$
0 < e(X) < 1
$$

Where $e(X) = P(D = 1 | X)$ is the propensity score.

### Alternative Names
- **Positivity**
- **Overlap**
- **Common Support**

### Interpretation

For every combination of covariate values that occurs in the population:
- Some units receive treatment
- Some units do not receive treatment
- No covariate region is "treatment-only" or "control-only"

### Testability

**TESTABLE** - Check propensity score distributions

### Diagnostic Methods

1. **Visual Inspection**
   ```
   +----------------------------------------+
   |  PS Distribution by Treatment Status   |
   +----------------------------------------+
   |                                        |
   |  Control: |******..............|        |
   |  Treated: |.....*************|          |
   |                                        |
   |           0   0.25  0.5  0.75  1       |
   |              Propensity Score          |
   +----------------------------------------+
   ```

2. **Min-Max Method**: Common support = [max(min_T, min_C), min(max_T, max_C)]

3. **Trimming**: Drop observations with PS < 0.1 or PS > 0.9

### Violations

Overlap is violated when:
- Some covariate profiles predict treatment perfectly
- Treatment and control groups do not overlap in covariate space
- Extreme propensity scores (near 0 or 1)

### Consequences of Violation

- **Extrapolation**: Must impute counterfactuals outside observed data
- **Model Dependence**: Results depend heavily on modeling assumptions
- **Limited External Validity**: Can only estimate effects for overlap region

### Remedies

| Approach | Description | Trade-off |
|----------|-------------|-----------|
| **Trimming** | Drop extreme PS values | Lose observations, changes estimand |
| **Caliper Matching** | Only match within PS distance | Unmatched treated units |
| **Bounds** | Report range of possible effects | Wider uncertainty |
| **Alternative Method** | Use RD, IV, or DID | Different assumptions |

---

## 3. SUTVA (Stable Unit Treatment Value Assumption)

### Formal Definition

For all units $i$ and $j$:
$$
Y_i(d_1, ..., d_N) = Y_i(d_i)
$$

SUTVA has two components:

### Component 1: No Interference

One unit's treatment does not affect another unit's outcome.

$$
Y_i(D_i, D_j) = Y_i(D_i) \quad \forall i \neq j
$$

**Violations occur with**:
- Social networks (peer effects)
- Geographic spillovers
- Market equilibrium effects
- General equilibrium effects

### Component 2: No Hidden Versions of Treatment

Treatment is well-defined with a single version.

$$
D_i = D_j = 1 \Rightarrow Y_i^{(1)} \text{ and } Y_j^{(1)} \text{ represent same treatment}
$$

**Violations occur with**:
- Heterogeneous treatment intensity
- Different treatment providers
- Treatment quality variation
- Different timing of treatment

### Testability

**PARTIALLY TESTABLE** through design features

### Diagnostic Approaches

1. **Cluster-Level Analysis**: Group units and check for spillovers
2. **Spatial Analysis**: Test for geographic spillovers
3. **Network Analysis**: Test for peer effects
4. **Timing Analysis**: Check for anticipation effects

### Example Violations

| Setting | Violation Type | Example |
|---------|---------------|---------|
| Vaccination | Interference | Herd immunity affects unvaccinated |
| Training | Interference | Trained workers compete with untrained |
| Advertising | Interference | Ad to one household affects neighbors |
| Education | Hidden treatment | Different teachers, class sizes |
| Surgery | Hidden treatment | Different surgeons, techniques |

---

## Assumption Hierarchy

```
+---------------------------------------------------------------+
|                    ASSUMPTION IMPORTANCE                       |
+---------------------------------------------------------------+
|                                                               |
|  1. CIA (Most Critical)                                       |
|     - Strongest assumption                                    |
|     - Completely untestable                                   |
|     - Must rely on theory and domain knowledge               |
|                                                               |
|  2. Overlap (Critical, Testable)                              |
|     - Can be diagnosed empirically                           |
|     - Violations detectable via PS distribution              |
|     - Remedies available (trimming, caliper)                 |
|                                                               |
|  3. SUTVA (Critical, Design-Dependent)                        |
|     - Depends on research setting                            |
|     - Partially testable through design                      |
|     - May require different estimators if violated           |
|                                                               |
+---------------------------------------------------------------+
```

---

## Strengthening Assumptions

### For CIA

1. **Rich Covariate Set**: Include all potential confounders
2. **Pre-Treatment Variables**: Only use pre-treatment X
3. **Lagged Outcomes**: Include pre-treatment outcome as covariate
4. **Proxy Variables**: Include proxies for unobservables
5. **Administrative Data**: Use comprehensive administrative records

### For Overlap

1. **Careful PS Model**: Avoid overfitting that creates extreme PS
2. **Trimming**: Remove observations outside common support
3. **Caliper Matching**: Enforce match quality
4. **Reassess Covariates**: Remove variables with extreme predictive power

### For SUTVA

1. **Unit Definition**: Define units to minimize interaction
2. **Time Windows**: Use time windows that avoid spillovers
3. **Cluster Randomization**: Account for clustering in design
4. **Alternative Estimators**: Use methods that allow for interference

---

## Reporting Requirements

When using PSM, authors should:

1. **Justify CIA**:
   - List all included confounders
   - Discuss potential unobserved confounders
   - Explain why selection is on observables

2. **Demonstrate Overlap**:
   - Show propensity score distributions
   - Report common support region
   - Describe any trimming applied

3. **Address SUTVA**:
   - Discuss potential for interference
   - Justify treatment definition
   - Consider alternative estimators if needed

4. **Report Sensitivity**:
   - Rosenbaum bounds or similar analysis
   - Discussion of robustness to hidden bias

---

## References

- Imbens, G. W., & Rubin, D. B. (2015). *Causal Inference for Statistics, Social, and Biomedical Sciences*. Cambridge University Press.
- Rosenbaum, P. R., & Rubin, D. B. (1983). The Central Role of the Propensity Score. *Biometrika*, 70(1), 41-55.
- Heckman, J. J., & Robb, R. (1985). Alternative Methods for Evaluating the Impact of Interventions. *Journal of Econometrics*, 30, 239-267.
- Rubin, D. B. (1980). Bias Reduction Using Mahalanobis-Metric Matching. *Biometrics*, 36, 293-298.
