---
name: causal-concept-guide
description: Conceptual guidance for causal inference - helps users understand causal concepts, choose appropriate methods, and avoid common pitfalls. This is a knowledge skill that provides guidance, not code execution.
triggers:
  - causal inference concepts
  - which causal method
  - treatment effect
  - confounding
  - identification strategy
  - research design
  - potential outcomes
  - counterfactual
  - ATE ATT LATE
  - causal question
---

# Causal Inference Conceptual Guide

## Overview

This skill provides conceptual guidance for causal inference analysis. It helps users:
- Understand fundamental causal concepts
- Choose the appropriate estimation method for their research question
- Avoid common pitfalls in causal claims
- Frame research questions in the potential outcomes framework

**This is a knowledge skill** - it does NOT execute code but provides the conceptual foundation needed before implementing any causal method.

---

## Skill Contents

### Reference Documents

| Document | Description |
|----------|-------------|
| `references/potential_outcomes.md` | Rubin causal model, counterfactuals, ATE/ATT/LATE definitions |
| `references/dags_and_graphs.md` | DAGs, d-separation, backdoor/frontdoor criteria |
| `references/selection_guide.md` | Decision trees for choosing estimators |
| `references/common_pitfalls.md` | Bad controls, collider bias, selection bias |
| `references/econometrics_vs_ml.md` | Traditional vs. ML approaches |
| `references/glossary.md` | Key terms and definitions |

### Assets

| Asset | Description |
|-------|-------------|
| `assets/markdown/method_comparison.md` | Side-by-side method comparison tables |

---

## Core Concepts

### 1. The Fundamental Problem of Causal Inference

We can never observe both potential outcomes for the same unit at the same time. If unit $i$ receives treatment ($D_i = 1$), we observe $Y_i(1)$ but not $Y_i(0)$. This is why causal inference requires assumptions.

**See**: `references/potential_outcomes.md` for detailed treatment.

### 2. Confounding

**Definition**: A confounder is a variable that affects both the treatment and the outcome, creating a spurious association.

**Example 1 - Classic**:
- Question: Does coffee cause heart disease?
- Confounder: Smoking behavior (smokers drink more coffee AND have higher heart disease risk)
- Without controlling for smoking, coffee appears harmful

**Example 2 - Economics**:
- Question: Does education increase earnings?
- Confounder: Ability (higher ability leads to more education AND higher earnings)
- OLS regression overestimates the return to education

**Example 3 - Policy**:
- Question: Does police presence reduce crime?
- Confounder: Crime rate itself (more police are deployed to high-crime areas)
- Naive correlation shows police associated with MORE crime

**Graphical Representation (DAG)**:
```
    Confounder (C)
       /    \
      v      v
Treatment -> Outcome
   (D)        (Y)
```

**See**: `references/dags_and_graphs.md` for graphical models.

### 3. Reverse Causality

**Definition**: When the outcome affects the treatment, rather than (or in addition to) the treatment affecting the outcome.

**Example 1**:
- Observed: Hospitals have higher mortality than homes
- Reverse causality: Sick people go to hospitals (sickness causes hospital visits, not the reverse)

**Example 2**:
- Observed: Countries with more UN peacekeepers have more conflict
- Reverse causality: Peacekeepers are deployed TO conflict zones

**Example 3**:
- Observed: Companies with more R&D spending have lower profits
- Reverse causality: Profitable companies can afford R&D, OR struggling companies increase R&D

### 4. Selection Bias

**Definition**: Systematic differences between treated and control groups that are not due to the treatment.

**Types of Selection Bias**:

| Type | Description | Example |
|------|-------------|---------|
| Self-selection | Units choose treatment based on expected benefit | Only motivated students enroll in tutoring |
| Administrative selection | Authority assigns treatment non-randomly | Judges assign stricter sentences to repeat offenders |
| Survivorship bias | Only "survivors" observed in sample | Successful companies studied; failed ones ignored |
| Attrition bias | Differential dropout between groups | Sicker patients drop out of clinical trials |

**See**: `references/common_pitfalls.md` for detailed examples.

### 5. Potential Outcomes Framework (Rubin Causal Model)

**Notation**:
- $Y_i(0)$: Potential outcome for unit $i$ if NOT treated (control)
- $Y_i(1)$: Potential outcome for unit $i$ if treated
- $D_i$: Treatment indicator (1 = treated, 0 = control)
- $Y_i$: Observed outcome = $D_i \cdot Y_i(1) + (1-D_i) \cdot Y_i(0)$

**Individual Treatment Effect**:
$$\tau_i = Y_i(1) - Y_i(0)$$

This is fundamentally unobservable for any single unit.

**See**: `references/potential_outcomes.md` for complete framework.

### 6. Treatment Effect Estimands

| Estimand | Definition | Formula | Interpretation |
|----------|------------|---------|----------------|
| **ATE** | Average Treatment Effect | $E[Y(1) - Y(0)]$ | Effect for a randomly chosen unit from the population |
| **ATT** | Average Treatment Effect on the Treated | $E[Y(1) - Y(0) \mid D=1]$ | Effect for those who actually received treatment |
| **ATU** | Average Treatment Effect on the Untreated | $E[Y(1) - Y(0) \mid D=0]$ | Effect if control group had been treated |
| **LATE** | Local Average Treatment Effect | $E[Y(1) - Y(0) \mid \text{Compliers}]$ | Effect for units whose treatment is affected by the instrument |
| **CATE** | Conditional Average Treatment Effect | $E[Y(1) - Y(0) \mid X=x]$ | Effect for units with characteristics $X=x$ |

**When They Differ**:
- ATE = ATT when treatment is randomly assigned
- ATT > ATE when those who benefit most self-select into treatment
- LATE may not equal ATE if compliers are different from always-takers/never-takers

**See**: `references/glossary.md` for all estimand definitions.

---

## Identification Strategies Overview

**Identification** means having a credible strategy to isolate the causal effect from confounding, selection bias, and reverse causality.

### Key Identification Strategies

| Strategy | Source of Variation | Key Assumption | Threat |
|----------|---------------------|----------------|--------|
| **RCT** | Random assignment by researcher | Random assignment was executed properly | Non-compliance, attrition |
| **DID** | Policy change over time | Parallel trends (absent treatment) | Differential trends, anticipation |
| **RD** | Threshold-based assignment | No manipulation at cutoff | Sorting, discontinuous confounders |
| **IV** | External instrument | Exclusion restriction (instrument only affects Y through D) | Direct effect of instrument |
| **Matching/PSM** | Observable characteristics | Selection on observables only | Unobserved confounders |
| **Synthetic Control** | Weighted combination of controls | Pre-treatment fit implies good counterfactual | Poor pre-treatment fit |

**See**: `references/selection_guide.md` for detailed decision trees.
**See**: `assets/markdown/method_comparison.md` for side-by-side comparisons.

---

## Master Method Selection Flowchart

```
START: What is your research question?
│
├─► What causal effect do you want to estimate?
│   │
│   ├─► Effect for entire population (ATE)
│   │   └─► Methods: RCT, DDML (with overlap), DID (homogeneous effects)
│   │
│   ├─► Effect for those treated (ATT)
│   │   └─► Methods: PSM, Matching, DID, Synthetic Control
│   │
│   ├─► Effect for compliers (LATE)
│   │   └─► Methods: IV, Fuzzy RD
│   │
│   └─► Heterogeneous effects (CATE)
│       └─► Methods: Causal Forest, Subgroup analysis
│
├─► What variation in treatment do you have?
│   │
│   ├─► RANDOM ASSIGNMENT by design?
│   │   │
│   │   └─► YES → RCT Analysis
│   │       ├─► Full compliance? → Simple difference in means
│   │       ├─► Non-compliance? → IV/LATE with randomization as instrument
│   │       ├─► Clustering? → Cluster-robust standard errors
│   │       └─► Check: Balance tests, attrition analysis
│   │
│   ├─► POLICY/INTERVENTION at specific time?
│   │   │
│   │   └─► YES → Consider Difference-in-Differences (DID)
│   │       ├─► Two groups, one treatment time? → Classic 2x2 DID
│   │       ├─► Staggered adoption? → Callaway-Sant'Anna, Sun-Abraham
│   │       ├─► Single treated unit? → Synthetic Control
│   │       ├─► Continuous treatment? → Generalized DID
│   │       ├─► Check: Pre-trends parallel?
│   │       ├─► Check: No anticipation effects?
│   │       └─► Check: No composition changes?
│   │
│   ├─► THRESHOLD/CUTOFF-based assignment?
│   │   │
│   │   └─► YES → Consider Regression Discontinuity (RD)
│   │       ├─► Deterministic at threshold? → Sharp RD
│   │       ├─► Probabilistic at threshold? → Fuzzy RD
│   │       ├─► Multiple cutoffs? → Multi-cutoff RD
│   │       ├─► Geographic boundary? → Geographic RD
│   │       ├─► Check: McCrary density test (no bunching)
│   │       ├─► Check: Covariate balance at cutoff
│   │       └─► Check: Sufficient observations near cutoff
│   │
│   ├─► EXTERNAL INSTRUMENT available?
│   │   │
│   │   └─► YES → Consider Instrumental Variables (IV)
│   │       ├─► Single instrument? → 2SLS
│   │       ├─► Multiple instruments? → GMM, overidentification tests
│   │       ├─► Weak first stage (F < 10)? → LIML, Anderson-Rubin
│   │       ├─► Check: Relevance (first-stage F-stat)
│   │       ├─► Check: Exclusion restriction (argue from theory)
│   │       └─► Check: Monotonicity for LATE interpretation
│   │
│   ├─► SELECTION based on observable characteristics?
│   │   │
│   │   └─► YES → Consider Matching / Propensity Score Methods
│   │       ├─► Low-dimensional covariates? → Nearest-neighbor, PSM
│   │       ├─► High-dimensional covariates? → DDML, Causal Forest
│   │       ├─► Need doubly robust? → AIPW
│   │       ├─► Check: Conditional independence plausible?
│   │       ├─► Check: Overlap in propensity scores?
│   │       └─► WARNING: Cannot address unobserved confounders
│   │
│   └─► HETEROGENEOUS EFFECTS needed?
│       │
│       └─► YES → Consider Causal ML Methods
│           ├─► Exploratory heterogeneity? → Causal Forest / GRF
│           ├─► Pre-specified subgroups? → Interaction models
│           ├─► Policy targeting? → Optimal treatment regimes
│           ├─► Check: Unconfoundedness must hold
│           └─► Check: Sufficient sample size (1000+)
│
└─► UNKNOWN or COMPLEX selection?
    │
    └─► Consider:
        ├─► Can you find an instrument? → IV
        ├─► Can you find a discontinuity? → RD
        ├─► Can you exploit timing? → DID
        ├─► Can you collect better data? → Richer covariates for PSM
        ├─► Sensitivity analysis → How much confounding to change results?
        └─► Bounds analysis → What can be learned with weaker assumptions?
```

---

## Research Design Checklist

### Questions to Ask Before Choosing a Method

1. **What is the precise causal question?**
   - What is the treatment? The outcome? The population?
   - Can you state it in potential outcomes notation?

2. **What is the source of variation in treatment?**
   - How did some units end up treated and others not?
   - Is this variation plausibly exogenous?

3. **What are the threats to identification?**
   - What confounders might affect both treatment and outcome?
   - Could reverse causality be at play?
   - What selection process determined treatment?

4. **What assumptions are you willing to make?**
   - Selection on observables only? (Matching)
   - Parallel trends? (DID)
   - No manipulation? (RD)
   - Exclusion restriction? (IV)

5. **What estimand do you want?**
   - ATE for the whole population?
   - ATT for the treated?
   - LATE for compliers?

### Negative List: Designs That Rarely Work

**Do NOT do these**:

| Bad Practice | Why It Fails | What To Do Instead |
|--------------|--------------|---------------------|
| Regress Y on D without controls | Omitted variable bias | Add confounders or find better identification |
| Control for post-treatment variables | Collider bias, removes causal pathway | Only control for pre-treatment confounders |
| Control for mediators (unless intended) | Removes indirect effect | Decide if you want total or direct effect |
| Difference-in-differences without testing pre-trends | Violates parallel trends | Plot and test pre-trends |
| IV with weak instrument | Bias toward OLS | Check first-stage F-stat > 10 |
| Matching on outcome-related variables | Conditions on outcome | Match only on pre-treatment confounders |
| Claim causality from correlation | Confounding, reverse causality | Use proper identification strategy |
| Over-control (kitchen sink regression) | May include colliders or mediators | Draw DAG, think about each variable |
| Report only significant specifications | P-hacking | Pre-register, report all specifications |

**See**: `references/common_pitfalls.md` for comprehensive pitfall guide.

### Common Pitfalls in Causal Claims

1. **"We controlled for everything"** - You can never control for unobservables
2. **"The coefficient is significant"** - Statistical significance is not causal identification
3. **"The effect is robust to specifications"** - Robustness does not prove causality
4. **"We used machine learning"** - ML is for prediction, not causal identification
5. **"The R-squared is high"** - Explanatory power is not causation
6. **"We have a large sample"** - Sample size does not fix confounding

---

## Counterfactual Framework Template

Use this template to articulate your causal question precisely:

### Template

```
RESEARCH QUESTION TEMPLATE
==========================

1. TREATMENT (D):
   - What is the intervention/treatment/policy?
   - Binary, multi-valued, or continuous?
   - Example: D = 1 if received job training, 0 otherwise

2. OUTCOME (Y):
   - What outcome do you want to affect?
   - How is it measured? When is it measured?
   - Example: Y = earnings 12 months after training

3. POPULATION:
   - Who is the population of interest?
   - Example: Unemployed workers ages 25-55

4. POTENTIAL OUTCOMES:
   - Y(0) = outcome if NOT treated
   - Y(1) = outcome if treated
   - Example: Y(0) = earnings without training; Y(1) = earnings with training

5. ESTIMAND:
   - What treatment effect do you want to estimate?
   - ATE = E[Y(1) - Y(0)] for population
   - ATT = E[Y(1) - Y(0) | D=1] for treated
   - Example: ATT = average earnings gain for those who received training

6. IDENTIFICATION STRATEGY:
   - What variation identifies the causal effect?
   - What assumptions are required?
   - Example: Random assignment to training conditional on pre-training characteristics

7. MAIN THREATS:
   - What could violate your identifying assumptions?
   - Example: Selection into training based on motivation (unobserved)

8. COUNTERFACTUAL:
   - What would have happened to treated units without treatment?
   - How do you construct this counterfactual?
   - Example: Control group earnings serve as counterfactual for treated
```

### Example: Filled Template

```
RESEARCH QUESTION: Effect of Minimum Wage on Employment
=======================================================

1. TREATMENT: State increases minimum wage (D=1) vs. no change (D=0)

2. OUTCOME: Employment in fast food sector, measured monthly

3. POPULATION: Fast food restaurants in affected states vs. neighboring states

4. POTENTIAL OUTCOMES:
   - Y(0) = Employment if minimum wage did NOT increase
   - Y(1) = Employment if minimum wage DID increase

5. ESTIMAND: ATT - effect on states that raised minimum wage

6. IDENTIFICATION: Difference-in-differences using neighboring states as control
   - Assumption: Employment trends would have been parallel absent treatment

7. THREATS:
   - Economic shocks affecting treatment states differently
   - Anticipation effects (hiring before increase)
   - Spillovers to control states

8. COUNTERFACTUAL: Neighboring states' employment trends extrapolated to treatment states
```

---

## When to Use Each Estimator

| Method | Best For | Key Assumption | Data Requirements | Example Application |
|--------|----------|----------------|-------------------|---------------------|
| **RCT** | Clean causal identification | Random assignment | Randomized treatment | Clinical trials, A/B tests |
| **DID** | Policy evaluation | Parallel trends | Pre/post data for treated and control groups | Minimum wage effects |
| **RD** | Threshold-based programs | No manipulation at cutoff | Running variable, sufficient observations near cutoff | Class size effects (Maimonides rule) |
| **IV** | Endogenous treatment with external instrument | Exclusion restriction + relevance | Valid instrument | Returns to education (quarter of birth) |
| **Matching/PSM** | Observable selection | Conditional independence (selection on observables) | Rich covariates, overlap in propensity | Job training programs |
| **DDML** | High-dimensional confounding | Conditional independence | Many potential controls | Policy effects with many covariates |
| **Causal Forest** | Heterogeneous effects | Unconfoundedness | Moderate to large sample | Personalized treatment effects |
| **Synthetic Control** | Single treated unit | Pre-treatment fit | Long pre-treatment period | Policy effects on single country/state |
| **Bounds** | When point identification fails | Minimal assumptions | Varies | Partial identification under selection |

**See**: `assets/markdown/method_comparison.md` for detailed comparison tables.

---

## Key References

### Essential Textbooks

1. **Angrist, J. D., & Pischke, J.-S. (2009)**. *Mostly Harmless Econometrics: An Empiricist's Companion*. Princeton University Press.
   - Best introduction to modern applied econometrics
   - Focuses on IV, DID, RD

2. **Angrist, J. D., & Pischke, J.-S. (2014)**. *Mastering 'Metrics: The Path from Cause to Effect*. Princeton University Press.
   - More accessible version of MHE
   - Excellent for beginners

3. **Cunningham, S. (2021)**. *Causal Inference: The Mixtape*. Yale University Press.
   - Free online: https://mixtape.scunning.com/
   - Modern treatment with code examples

4. **Huntington-Klein, N. (2022)**. *The Effect: An Introduction to Research Design and Causality*. Chapman & Hall/CRC.
   - Free online: https://theeffectbook.net/
   - Excellent visualizations and intuition

5. **Imbens, G. W., & Rubin, D. B. (2015)**. *Causal Inference for Statistics, Social, and Biomedical Sciences*. Cambridge University Press.
   - Definitive treatment of potential outcomes framework
   - More technical

### Seminal Papers by Method

**Potential Outcomes & Matching**:
- Rosenbaum, P. R., & Rubin, D. B. (1983). The central role of the propensity score in observational studies for causal effects. *Biometrika*, 70(1), 41-55.

**Difference-in-Differences**:
- Card, D., & Krueger, A. B. (1994). Minimum wages and employment: A case study of the fast-food industry in New Jersey and Pennsylvania. *American Economic Review*, 84(4), 772-793.

**Regression Discontinuity**:
- Angrist, J. D., & Lavy, V. (1999). Using Maimonides' rule to estimate the effect of class size on scholastic achievement. *Quarterly Journal of Economics*, 114(2), 533-575.

**Instrumental Variables**:
- Angrist, J. D., & Krueger, A. B. (1991). Does compulsory school attendance affect schooling and earnings? *Quarterly Journal of Economics*, 106(4), 979-1014.

**Synthetic Control**:
- Abadie, A., Diamond, A., & Hainmueller, J. (2010). Synthetic control methods for comparative case studies. *Journal of the American Statistical Association*, 105(490), 493-505.

**Causal Forests & Machine Learning**:
- Athey, S., & Imbens, G. W. (2016). Recursive partitioning for heterogeneous causal effects. *Proceedings of the National Academy of Sciences*, 113(27), 7353-7360.
- Chernozhukov, V., Chetverikov, D., Demirer, M., Duflo, E., Hansen, C., Newey, W., & Robins, J. (2018). Double/debiased machine learning for treatment and structural parameters. *Econometrics Journal*, 21(1), C1-C68.

---

## Related Skills

### Implementation Skills

For code implementation, see the method-specific skills:

| Skill | Method | Description |
|-------|--------|-------------|
| `estimator-did` | Difference-in-Differences | DID implementation in R/Python/Stata |
| `estimator-rd` | Regression Discontinuity | RD implementation with rdrobust |
| `estimator-iv` | Instrumental Variables | 2SLS and weak-IV robust methods |
| `estimator-psm` | Propensity Score Matching | PSM and matching estimators |
| `causal-ddml` | Double/Debiased ML | High-dimensional causal inference |
| `causal-mediation-ml` | Mediation Analysis | Direct/indirect effect decomposition |

### Complementary Skills

| Skill | Purpose |
|-------|---------|
| `ml-preprocessing` | Data preparation for causal analysis |
| `ml-model-tree` | Tree-based methods (including Causal Forest) |
| `setup-causal-ml-env` | Environment setup for causal ML |

---

## Usage Notes

This skill is designed to be consulted BEFORE implementing causal methods. It provides:

1. **Conceptual clarity** - Understanding what causal inference means
2. **Method selection guidance** - Choosing the right tool for your problem
3. **Assumption checking** - Knowing what must hold for identification
4. **Pitfall avoidance** - Recognizing common mistakes

**Workflow recommendation**:
1. Start here to understand concepts and choose method
2. Consult `references/` documents for deeper understanding
3. Use `assets/markdown/method_comparison.md` for method selection
4. Move to implementation skill (e.g., `estimator-did`) for code
