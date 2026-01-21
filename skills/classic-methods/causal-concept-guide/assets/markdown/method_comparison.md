# Causal Inference Method Comparison

## Overview

This document provides side-by-side comparisons of major causal inference methods to aid in estimator selection.

---

## Quick Comparison Matrix

### By Identification Strategy

| Method | Source of Variation | Primary Assumption | Testable? | Estimand |
|--------|---------------------|-------------------|-----------|----------|
| **RCT** | Random assignment | Proper randomization | Partially (balance) | ATE |
| **DID** | Policy timing | Parallel trends | Partially (pre-trends) | ATT |
| **RD** | Threshold rule | No manipulation | Partially (density) | Local ATE |
| **IV / 2SLS** | Instrumental variation | Exclusion restriction | No | LATE |
| **PSM / Matching** | Observable selection | Selection on observables | No | ATT (typically) |
| **DDML** | Observable selection | Conditional independence | No | ATE or ATT |
| **Causal Forest** | Observable selection | Unconfoundedness | No | CATE |
| **Synthetic Control** | Weighted donors | Good pre-treatment fit | Partially (fit) | ATT |

---

## Detailed Method Comparison

### Data Requirements

| Method | Sample Size | Time Periods | Covariates | Special Requirements |
|--------|-------------|--------------|------------|---------------------|
| **RCT** | 100+ per arm | 1+ | Optional | Randomization mechanism |
| **DID** | 50+ per group | 2+ (pre/post) | Optional | Treated & control groups |
| **RD** | 100+ near cutoff | 1+ | Optional | Running variable, cutoff |
| **IV** | 500+ | 1+ | Optional | Valid instrument |
| **PSM** | 200+ treated | 1+ | Essential | Rich pre-treatment data |
| **DDML** | 500+ | 1+ | Many | High-dimensional covariates |
| **Causal Forest** | 1000+ | 1+ | Moderate | Sufficient for splits |
| **Synthetic Control** | 1 treated unit | 20+ pre-treatment | Optional | Donor pool |

### Assumption Strength

| Method | Assumption Credibility | Comment |
|--------|----------------------|---------|
| **RCT** | Strongest | Gold standard, but implementation matters |
| **RD** | Very strong | Local randomization near cutoff |
| **IV** | Strong if exclusion holds | Exclusion restriction often debated |
| **DID** | Moderate | Pre-trends testable but not sufficient |
| **Synthetic Control** | Moderate | Fit quality observable |
| **PSM / DDML** | Weakest | Requires ALL confounders observed |
| **Causal Forest** | Weakest | Same as PSM but for heterogeneity |

### What Each Method Identifies

| Method | Population | Effect Type | External Validity |
|--------|------------|-------------|-------------------|
| **RCT** | Study population | ATE | Depends on sample |
| **DID** | Treated units | ATT | Depends on comparability |
| **RD** | Units near cutoff | Local ATE | Limited to cutoff region |
| **IV** | Compliers only | LATE | Limited to compliers |
| **PSM** | Overlap region | ATT (usually) | Limited to overlap |
| **DDML** | Overlap region | ATE or ATT | Limited to overlap |
| **Causal Forest** | Covariate space | CATE | Within covariate support |
| **Synthetic Control** | Single treated unit | ATT | That unit only |

---

## Use Case Comparison

### When to Prefer Each Method

| Research Setting | Recommended Method | Rationale |
|-----------------|-------------------|-----------|
| Can randomize treatment | RCT | Cleanest identification |
| Policy with clear timing | DID | Exploits temporal variation |
| Program with eligibility cutoff | RD | Threshold creates local experiment |
| Have external instrument | IV | Overcomes unobserved confounding |
| Rich covariates, selection on observables | PSM or DDML | Leverages observable data |
| Need heterogeneous effects | Causal Forest | Discovers effect variation |
| Single treated unit (state, country) | Synthetic Control | Creates comparable control |
| Very large observational data | DDML | Handles high dimensionality |

### Method Limitations

| Method | Cannot Handle | Workaround |
|--------|--------------|------------|
| **RCT** | Unethical treatments, non-compliance | ITT analysis, IV with randomization |
| **DID** | Differential trends | Triple-diff, event studies |
| **RD** | Manipulation at cutoff | Find manipulation-proof setting |
| **IV** | Weak/invalid instruments | Find better instrument, bounds |
| **PSM** | Unobserved confounders | Sensitivity analysis |
| **DDML** | Unobserved confounders | Sensitivity analysis |
| **Causal Forest** | Unobserved confounders | Use with RCT/quasi-experiment |
| **Synthetic Control** | Poor pre-treatment fit | Cannot use method |

---

## Statistical Properties

### Bias and Variance Trade-offs

| Method | Bias Risk | Variance | Notes |
|--------|-----------|----------|-------|
| **RCT** | Low (if proper) | Moderate | Randomization removes bias |
| **DID** | Moderate | Low | Depends on parallel trends |
| **RD** | Low near cutoff | High (local estimation) | Bandwidth affects trade-off |
| **IV** | Low if valid | High (especially weak IV) | Weak IV increases both |
| **PSM** | Potentially high | Moderate | Unobserved confounders bias |
| **DDML** | Low if correct | Moderate | Double robustness helps |
| **Causal Forest** | Potentially high | Moderate to low | Needs unconfoundedness |
| **Synthetic Control** | Depends on fit | High (single unit) | Pre-treatment fit crucial |

### Inference Quality

| Method | Standard Errors | Confidence Intervals | Notes |
|--------|----------------|---------------------|-------|
| **RCT** | Straightforward | Valid | Randomization inference available |
| **DID** | Cluster at unit level | Valid with clustering | Serial correlation concerns |
| **RD** | Bias-corrected robust | Valid with CCT corrections | Bandwidth selection matters |
| **IV** | Robust to heteroskedasticity | Valid if not weak | Anderson-Rubin for weak IV |
| **PSM** | Bootstrap or analytical | Valid with correct SE | Must account for estimation |
| **DDML** | Cross-fitting based | Valid | DML variance formula |
| **Causal Forest** | Built-in honest inference | Valid | Honesty is key |
| **Synthetic Control** | Permutation inference | Valid | Placebo-based |

---

## Software Comparison

### R Packages

| Method | Primary Package | Alternative |
|--------|----------------|-------------|
| **RCT** | `estimatr` | `sandwich` |
| **DID** | `did`, `fixest` | `bacondecomp` |
| **RD** | `rdrobust` | `rdd`, `rdmulti` |
| **IV** | `ivreg`, `fixest` | `AER` |
| **PSM** | `MatchIt`, `Matching` | `WeightIt` |
| **DDML** | `DoubleML` | `hdm` |
| **Causal Forest** | `grf` | `causalForest` |
| **Synthetic Control** | `Synth`, `gsynth` | `microsynth` |

### Python Packages

| Method | Primary Package | Alternative |
|--------|----------------|-------------|
| **RCT** | `statsmodels` | `scipy.stats` |
| **DID** | `linearmodels` | `differences` |
| **RD** | `rdrobust` | Custom |
| **IV** | `linearmodels` | `statsmodels` |
| **PSM** | `causalinference` | `pymatch` |
| **DDML** | `doubleml` | `econml` |
| **Causal Forest** | `econml` | `causalml` |
| **Synthetic Control** | `SyntheticControlMethods` | Custom |

### Stata Commands

| Method | Primary Command |
|--------|----------------|
| **RCT** | `regress`, `ttest` |
| **DID** | `did_imputation`, `csdid` |
| **RD** | `rdrobust` |
| **IV** | `ivreg2`, `ivregress` |
| **PSM** | `psmatch2`, `teffects` |
| **DDML** | `ddml` |
| **Causal Forest** | External via R/Python |
| **Synthetic Control** | `synth`, `synth_runner` |

---

## Decision Matrix: Key Questions

### Question 1: Is there a natural experiment?

```
Natural experiment available?
│
├─► YES
│   ├─► Based on timing? → DID
│   ├─► Based on threshold? → RD
│   └─► External instrument? → IV
│
└─► NO
    ├─► Can you randomize? → RCT
    └─► Selection on observables? → PSM/DDML
```

### Question 2: What data do you have?

```
What's your data structure?
│
├─► Cross-sectional only
│   ├─► Rich covariates? → PSM/DDML
│   └─► Threshold rule? → RD
│
├─► Panel data
│   ├─► Policy change? → DID
│   ├─► Single treated unit? → Synthetic Control
│   └─► Multiple time-varying treatments? → Staggered DID
│
└─► Experimental data
    └─► RCT analysis (with IV for non-compliance)
```

### Question 3: What effect do you want?

```
Target estimand?
│
├─► Average effect (ATE) → RCT, DDML, DID (with homogeneity)
│
├─► Effect on treated (ATT) → PSM, DID, Synthetic Control
│
├─► Effect for compliers (LATE) → IV
│
├─► Local effect at cutoff → RD
│
└─► Heterogeneous effects (CATE) → Causal Forest, subgroup analysis
```

---

## Summary Table: Method Selection Guide

| If You Have... | Consider... | Watch Out For... |
|---------------|-------------|------------------|
| Random assignment | RCT analysis | Non-compliance, attrition |
| Policy with timing variation | DID | Parallel trends violations |
| Threshold-based assignment | RD | Manipulation, sparse data near cutoff |
| Plausibly excludable instrument | IV | Weak instruments, exclusion violations |
| Rich pre-treatment covariates | PSM / DDML | Unobserved confounders |
| Interest in heterogeneity | Causal Forest | Sample size, unconfoundedness |
| Single treated unit, many controls | Synthetic Control | Pre-treatment fit |
| High-dimensional controls | DDML | Still need conditional independence |

---

## Related Documents

- `selection_guide.md` - Detailed decision trees
- `common_pitfalls.md` - Method-specific mistakes
- `econometrics_vs_ml.md` - When to use ML approaches
- `glossary.md` - Terminology definitions
