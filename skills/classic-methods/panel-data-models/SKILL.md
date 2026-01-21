---
name: panel-data-models
description: Use for panel data analysis with fixed/random effects. Triggers on panel data, fixed effects, random effects, FE, RE, Hausman test, clustered standard errors, dynamic panel, GMM, Arellano-Bond, two-way fixed effects, TWFE, within estimator, entity effects, time effects.
---

# Panel Data Models Skill

## Overview

This skill provides comprehensive tools for panel (longitudinal) data econometrics, including fixed effects, random effects, dynamic panel models, and proper inference with clustered standard errors.

## When to Use

Activate this skill when:
- User has panel/longitudinal data (repeated observations over entities/time)
- Questions about fixed effects vs random effects choice
- Need for Hausman specification test
- Clustered standard error computation required
- Dynamic panel models (lagged dependent variables)
- Two-way fixed effects for causal inference
- Concerns about TWFE with staggered treatment adoption

## Key Capabilities

### 1. Static Panel Models
- **Fixed Effects (FE)**: Controls for time-invariant unobserved heterogeneity
- **Random Effects (RE)**: Assumes unobserved effects uncorrelated with regressors
- **Within transformation**: Demeans data to eliminate entity effects
- **First-differencing**: Alternative to within transformation

### 2. Model Selection
- **Hausman test**: Tests FE vs RE specification
- **Within-between decomposition**: Mundlak approach
- **Poolability tests**: Whether pooled OLS is appropriate

### 3. Dynamic Panels
- **Arellano-Bond (difference GMM)**: For models with lagged DV
- **Blundell-Bond (system GMM)**: More efficient with persistent series
- **Instrument validity tests**: Sargan/Hansen overidentification

### 4. Robust Inference
- **Clustered standard errors**: Entity-level, time-level, two-way
- **Heteroskedasticity-robust**: Within-cluster correlation
- **Wild cluster bootstrap**: Small number of clusters
- **Driscoll-Kraay SE**: Cross-sectional dependence

### 5. Causal Inference with TWFE
- **Two-way fixed effects**: Entity + time effects
- **TWFE problems**: Heterogeneous treatment effects
- **Goodman-Bacon decomposition**: Understanding TWFE weights
- **Modern alternatives**: Callaway-Sant'Anna, Sun-Abraham, Borusyak et al.

## Workflow

```python
from panel_estimator import PanelEstimator

# Initialize with panel data
estimator = PanelEstimator(
    data=df,
    entity_col='firm_id',
    time_col='year',
    y_col='outcome',
    x_cols=['treatment', 'control1', 'control2']
)

# Fit models
fe_result = estimator.fit_fixed_effects(entity_effects=True, time_effects=True)
re_result = estimator.fit_random_effects()

# Specification test
hausman = estimator.hausman_test(fe_result, re_result)

# Robust inference
fe_clustered = estimator.cluster_robust_inference(
    fe_result,
    cluster_col='firm_id',
    method='stata'  # or 'robust', 'bootstrap', 'wild'
)
```

## Critical Considerations

### TWFE Causal Inference Warning

Two-way fixed effects can produce biased estimates under staggered treatment adoption with heterogeneous treatment effects. The TWFE estimator is a weighted average of many 2x2 DiD comparisons, potentially with **negative weights**.

**When to worry:**
- Staggered treatment timing across units
- Treatment effects vary over time or across units
- Already-treated units used as controls

**Diagnostic:**
```python
# Check for problematic negative weights
decomposition = estimator.goodman_bacon_decomposition()
print(decomposition.negative_weight_share)
```

**Alternatives:**
- Callaway-Sant'Anna (2021): Group-time ATT
- Sun-Abraham (2021): Interaction-weighted estimator
- Borusyak et al. (2024): Imputation approach

### Clustered Standard Errors

**Rule of thumb**: Cluster at the level of treatment variation (typically entity).

| Scenario | Recommended Clustering |
|----------|----------------------|
| Treatment varies by entity | Cluster by entity |
| Treatment varies by time | Cluster by time |
| Both entity and time variation | Two-way clustering |
| Few clusters (< 50) | Wild cluster bootstrap |

## File Structure

```
panel-data-models/
├── SKILL.md                    # This file
├── panel_estimator.py          # Main implementation
├── references/
│   ├── fe_re_models.md        # FE vs RE theory
│   ├── clustered_inference.md # Cluster-robust inference
│   ├── dynamic_panels.md      # GMM estimation
│   ├── panel_diagnostics.md   # Specification tests
│   └── causal_interpretation.md # TWFE for causal inference
├── scripts/
│   ├── run_panel_model.py     # CLI for panel estimation
│   ├── test_fe_vs_re.py       # Hausman test script
│   ├── cluster_robust_se.py   # SE computation
│   └── visualize_panel.py     # Panel visualization
└── assets/
    └── latex/
        └── panel_table.tex    # LaTeX table template
```

## Integration

This skill integrates with:
- `estimator-did`: For difference-in-differences
- `ml-preprocessing`: Data preparation
- `causal-concept-guide`: Method selection guidance

## References

- Wooldridge, J.M. (2010). Econometric Analysis of Cross Section and Panel Data
- Arellano, M. & Bond, S. (1991). Some tests of specification for panel data
- Cameron, A.C. & Miller, D.L. (2015). A Practitioner's Guide to Cluster-Robust Inference
- Goodman-Bacon, A. (2021). Difference-in-differences with variation in treatment timing
- Callaway, B. & Sant'Anna, P.H.C. (2021). Difference-in-Differences with multiple time periods
