# Estimator Selection Guide

## Overview

Choosing the right causal inference method depends on your research question, data structure, and the source of variation in treatment assignment. This guide provides decision trees and comparison tables to help you select the appropriate estimator.

---

## Quick Selection Matrix

### By Data Structure

| Data Structure | Recommended Methods |
|---------------|---------------------|
| Cross-sectional, rich covariates | PSM, Matching, DDML |
| Panel data, policy change | DID, Synthetic Control |
| Threshold-based assignment | RD (Sharp or Fuzzy) |
| External instrument available | IV, 2SLS |
| Single treated unit, many controls | Synthetic Control |
| Need heterogeneous effects | Causal Forest, GRF |
| High-dimensional controls | DDML, Causal Forest |

### By Research Design

| Design Feature | Best Methods |
|---------------|--------------|
| Random assignment | Simple comparison, RCT analysis |
| Natural experiment | DID, IV, RD |
| Selection on observables | PSM, Matching, DDML |
| Geographic variation | DID, Synthetic Control |
| Policy discontinuity | RD |
| Instrumental variation | IV, Fuzzy RD |

---

## Primary Decision Tree

```
START: How was treatment assigned?
│
├─► RANDOMIZED by researcher?
│   │
│   └─► YES → RCT Analysis
│       ├─► Full compliance? → Simple ITT analysis
│       └─► Non-compliance? → IV/LATE with randomization as instrument
│
├─► Based on observable THRESHOLD?
│   │
│   └─► YES → Regression Discontinuity (RD)
│       ├─► Deterministic at threshold? → Sharp RD
│       ├─► Probabilistic at threshold? → Fuzzy RD
│       └─► Multiple cutoffs? → Multi-cutoff RD / RD + DID
│
├─► POLICY CHANGE over time?
│   │
│   └─► YES → Difference-in-Differences variants
│       ├─► Single treatment time, two groups? → Classic 2x2 DID
│       ├─► Staggered adoption? → Staggered DID (Callaway-Sant'Anna, Sun-Abraham)
│       ├─► Single treated unit? → Synthetic Control
│       └─► Continuous treatment? → Continuous DID / Event study
│
├─► INSTRUMENTAL VARIABLE available?
│   │
│   └─► YES → IV / 2SLS
│       ├─► Strong first stage (F > 10)? → Standard 2SLS
│       ├─► Weak instrument? → LIML, Anderson-Rubin, or find better instrument
│       └─► Multiple instruments? → Overidentification tests, GMM
│
├─► Selection based on OBSERVABLES only?
│   │
│   └─► YES → Matching / Propensity Score Methods
│       ├─► Low-dimensional? → PSM, Nearest-neighbor matching
│       ├─► High-dimensional? → DDML, Causal Forest
│       ├─► Need ATT? → PSM, Matching
│       └─► Need ATE? → IPW, AIPW, DDML
│
└─► UNKNOWN or complex selection?
    │
    └─► Consider:
        ├─► Can you find an instrument? → IV
        ├─► Can you find a discontinuity? → RD
        ├─► Can you exploit timing? → DID
        └─► Bounds analysis if identification fails
```

---

## Secondary Decision Tree: Heterogeneous Effects

```
Do you need HETEROGENEOUS treatment effects?
│
├─► NO → Use average effect methods (see above)
│
└─► YES → What type of heterogeneity?
    │
    ├─► Pre-specified subgroups?
    │   └─► Subgroup analysis with any method
    │       └─► Multiple testing correction!
    │
    ├─► Exploratory, data-driven?
    │   └─► Causal Forest / GRF
    │       ├─► Requires unconfoundedness
    │       └─► Moderate to large sample size
    │
    ├─► By propensity score strata?
    │   └─► Stratified PSM / Blocking
    │
    └─► Continuous moderator?
        └─► Interaction terms or Causal Forest
```

---

## Estimand Selection Guide

### Which Treatment Effect Do You Want?

| Estimand | When to Use | Methods That Provide It |
|----------|-------------|------------------------|
| **ATE** | Effect for random person from population | RCT, DDML (with overlap), DID (with homogeneity) |
| **ATT** | Effect for those who received treatment | PSM, Matching, DID, Synthetic Control |
| **ATU** | Effect if we expand treatment to untreated | IPW (with extrapolation), Matching (reversed) |
| **LATE** | Effect for compliers (instrument-induced takers) | IV, Fuzzy RD |
| **CATE** | Effect varying by characteristics | Causal Forest, Interaction models |
| **Quantile TE** | Effect on distribution, not mean | Quantile regression, distributional methods |

### Estimand Flow Chart

```
What is your policy question?
│
├─► "Should we implement this policy for everyone?"
│   └─► Want: ATE
│
├─► "Did this policy help those who received it?"
│   └─► Want: ATT
│
├─► "What would happen if we expanded this program?"
│   └─► Want: ATU (extrapolation warning!)
│
├─► "Who benefits most from treatment?"
│   └─► Want: CATE / Heterogeneous effects
│
└─► "For whom does the instrument change behavior?"
    └─► Want: LATE (be clear about complier population)
```

---

## Method Comparison by Assumption

### Identifying Assumptions Required

| Method | Primary Assumption | Testable? | Failure Mode |
|--------|-------------------|-----------|--------------|
| **RCT** | Randomization was proper | Partially (balance tests) | Implementation failure |
| **DID** | Parallel trends | Partially (pre-trends) | Differential trends |
| **RD** | No manipulation at cutoff | Partially (density test) | Sorting, bunching |
| **IV** | Exclusion restriction | No | Direct effect of instrument |
| **PSM** | Selection on observables | No | Unobserved confounders |
| **DDML** | Conditional independence | No | Unobserved confounders |
| **Synthetic Control** | Pre-treatment fit → good counterfactual | Partially (fit quality) | Poor fit, interference |
| **Causal Forest** | Unconfoundedness | No | Unobserved confounders |

### Assumption Strength Ranking

From **strongest** (hardest to satisfy) to **weakest** (easiest to satisfy):

1. **Selection on observables** (PSM, DDML, Causal Forest)
   - Must observe ALL confounders
   - Rarely fully believable

2. **Exclusion restriction** (IV)
   - Instrument cannot directly affect outcome
   - Often debated

3. **Parallel trends** (DID)
   - Trends would have been same without treatment
   - Can partially test with pre-trends

4. **No manipulation** (RD)
   - Units cannot sort around cutoff
   - Often testable

5. **Random assignment** (RCT)
   - Treatment is randomized
   - Gold standard, but implementation matters

---

## Method Selection by Data Requirements

### Sample Size Requirements

| Method | Minimum Viable | Recommended | Notes |
|--------|---------------|-------------|-------|
| RCT | ~100 per arm | ~500+ per arm | Power calculations needed |
| DID | ~50 per group-time | ~200+ | More periods help |
| RD | ~100 near cutoff | ~500+ near cutoff | Bandwidth matters |
| IV | ~500 total | ~1000+ | Weak IV bias with small N |
| PSM | ~200 treated | ~500+ | Need overlap |
| DDML | ~500 total | ~1000+ | Cross-fitting needs data |
| Causal Forest | ~1000 total | ~5000+ | Data-hungry |
| Synthetic Control | ~20 time periods | ~50+ | One treated unit fine |

### Covariate Requirements

| Method | Covariate Needs |
|--------|----------------|
| RCT | Optional (for precision) |
| DID | Optional (for robustness) |
| RD | Running variable required |
| IV | Instrument required |
| PSM | Rich pre-treatment covariates |
| DDML | Many potential controls |
| Causal Forest | Moderate covariates for CATE |
| Synthetic Control | Pre-treatment outcomes |

---

## Decision Flowchart: By Research Question

### Policy Evaluation

```
Evaluating a policy intervention?
│
├─► Policy implemented at single time?
│   ├─► Multiple treated and control units? → DID
│   └─► Single treated unit? → Synthetic Control
│
├─► Policy phased in over time?
│   └─► Staggered DID (modern methods)
│
├─► Policy based on eligibility threshold?
│   └─► RD
│
└─► Policy assigned through lottery?
    └─► RCT analysis
```

### Program Evaluation

```
Evaluating a voluntary program?
│
├─► Random encouragement/lottery for access?
│   └─► IV with lottery as instrument
│
├─► Selection likely based on observables?
│   ├─► Low-dimensional? → PSM
│   └─► High-dimensional? → DDML
│
├─► Eligibility based on threshold?
│   └─► RD
│
└─► Strong selection on unobservables?
    └─► Seek instrument or accept limitations
```

### Treatment Effect Heterogeneity

```
Interested in who benefits most?
│
├─► Have pre-specified hypotheses?
│   └─► Subgroup analysis (with multiple testing correction)
│
├─► Want exploratory analysis?
│   └─► Causal Forest / GRF
│
└─► Want policy-relevant targeting?
    └─► Optimal treatment regimes (beyond this guide)
```

---

## Red Flags: When NOT to Use Each Method

### RCT
- Non-compliance is severe and non-random
- Attrition differs by treatment arm
- Randomization compromised

### DID
- Pre-trends are divergent
- Composition of groups changes over time
- Anticipation effects present

### RD
- Manipulation at cutoff (bunching in density)
- Too few observations near cutoff
- Cutoff is not locally random

### IV
- First-stage F-statistic < 10 (weak instrument)
- Exclusion restriction implausible
- Instrument affects outcome directly

### PSM / Matching
- Strong selection on unobservables expected
- Poor overlap in propensity scores
- Key confounders are unobserved

### DDML
- Conditional independence implausible
- Small sample (cross-fitting needs data)
- No meaningful covariates

### Synthetic Control
- Poor pre-treatment fit
- Interference between units
- Too few pre-treatment periods

### Causal Forest
- Small sample size
- Unconfoundedness implausible
- Treatment is determined by outcome

---

## Quick Reference: Method Strengths

| Method | Strongest When... |
|--------|-------------------|
| **RCT** | You can randomize treatment |
| **DID** | Clean policy timing with parallel pre-trends |
| **RD** | Sharp cutoff with no manipulation |
| **IV** | Strong, excludable instrument exists |
| **PSM** | Rich data captures all selection factors |
| **DDML** | High-dimensional confounders, large sample |
| **Causal Forest** | Heterogeneity is key, large sample |
| **Synthetic Control** | Single treated unit, long pre-period |

---

## Related Skills

For implementation details, see:

- `estimator-did` - Difference-in-Differences
- `estimator-rd` - Regression Discontinuity
- `estimator-iv` - Instrumental Variables
- `estimator-psm` - Propensity Score Matching
- `causal-ddml` - Double/Debiased Machine Learning
- `causal-mediation-ml` - Mediation Analysis

---

## Key References

1. **Angrist, J. D., & Pischke, J.-S. (2009)**. *Mostly Harmless Econometrics*. Princeton University Press.

2. **Cunningham, S. (2021)**. *Causal Inference: The Mixtape*. Yale University Press.

3. **Huntington-Klein, N. (2022)**. *The Effect*. Chapman & Hall/CRC.

4. **Imbens, G. W., & Wooldridge, J. M. (2009)**. Recent developments in the econometrics of program evaluation. *Journal of Economic Literature*, 47(1), 5-86.
