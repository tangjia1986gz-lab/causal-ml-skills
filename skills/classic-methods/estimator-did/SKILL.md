---
name: estimator-did
description: Difference-in-Differences estimator with Top Journal standards (AER/MS). Includes rigorous diagnostics, placebo tests, and publication-quality reporting.
version: 3.0.0
type: estimator
aliases:
  - DID
  - Diff-in-Diff
  - TWFE
  - Staggered DID
triggers:
  - difference-in-differences
  - diff-in-diff
  - parallel trends
  - event study
  - policy evaluation
  - beck levine levkov
references:
  - references/identification_assumptions.md
  - references/diagnostic_tests.md
  - references/estimation_methods.md
  - references/reporting_standards.md
scripts:
  - scripts/run_did_analysis.py
  - scripts/test_parallel_trends.py
  - scripts/robustness_checks.py
assets:
  - assets/latex/did_table.tex
---

# Estimator: Difference-in-Differences (Top Journal Edition)

> **Version**: 3.0.0 | **Standard**: AER/Management Science | **Last Updated**: 2026-01

## 🏆 Gold Standard Benchmark

This skill is engineered to replicate the rigor of **Beck, Levine & Levkov (2010), "Big Bad Banks? The Winners and Losers from Bank Deregulation in the United States", Journal of Finance**.

- **Benchmark Data**: `benchmark/beck_levine_levkov_2010.csv`
- **Replication Target**: Table III (Baseline), Figure 1 (Dynamic Effect).

## 📊 Core Standards

### 1. Rigor (严谨性)
- **Standard Errors**: Must be clustered at the treatment assignment level (e.g., State).
- **Fixed Effects**: High-dimensional Fixed Effects (Unit + Time) are mandatory.
- **Dynamic Effects**: Event Study plots with confidence intervals are required for all analyses.

### 2. Robustness (稳健性)
- **Placebo Test**: Randomly assign treatment time/group 1000 times (permutation test).
- **Pre-trend Test**: Joint F-test of pre-treatment coefficients.
- **Goodman-Bacon Decomposition**: Mandatory check for staggered timing designs.

### 3. Reporting (规范性)
- **Output Format**: Stargazer-like tables (LaTeX/Markdown).
- **Significance Levels**: * p<0.1, ** p<0.05, *** p<0.01.
- **Summary Stats**: Obs, R-squared, Mean of Dep. Var.

## 🛠️ Implementation Workflow

### Step 1: Pre-Estimation Diagnostics
```python
from estimator_did.implementation.diagnostics import check_parallel_trends

# Visual & Statistical Test
check_parallel_trends(
    data=df, unit='state', time='year', 
    outcome='log_income', treatment='deregulation'
)
```

### Step 2: Main Estimation (TWFE / Callaway-Sant'Anna)
```python
from estimator_did.implementation.core import estimate_did

model = estimate_did(
    data=df,
    y='log_income',
    d='deregulation',
    unit='state',
    time='year',
    controls=['unemp_rate', 'gdp_growth'],
    cluster='state',
    estimator='twfe' # or 'cs' for Callaway-Sant'Anna
)

print(model.summary())
```

### Step 3: Robustness & Reporting
```python
from estimator_did.implementation.reporting import generate_publication_table

# Generate AER-style table
generate_publication_table(
    models=[model_baseline, model_controls, model_staggered],
    output_format='latex', # or 'markdown'
    filename='table_3_replication.tex'
)
```

## 📂 Directory Structure

```text
estimator-did/
├── SKILL.md                     # This file
├── implementation/              # Core Logic
│   ├── core.py                  # Estimator classes (TWFE, CS)
│   ├── diagnostics.py           # Parallel trends, Bacon decomp
│   └── reporting.py             # Stargazer-like typesetter
├── benchmark/                   # Gold Standard Data
│   ├── beck_levine_levkov_2010.csv
│   └── replication_targets.json
├── tests/                       # Verification
│   └── test_replication.py      # Assert result == Benchmark
└── examples/
    └── paper_replication.ipynb  # Teaching notebook
```

## ⚠️ Common Pitfalls (Reviewer Rejection Reasons)

1.  **"Testing" Parallel Trends via Statistical Significance Only**: 
    - *Correction*: Must provide Event Study plot; lack of significance is not proof of parallel trends.
2.  **Incorrect Clustering**:
    - *Correction*: Cluster at the level of treatment variation, not the unit of analysis (if different).
3.  **TWFE Bias in Staggered Designs**:
    - *Correction*: Use `estimator='cs'` (Callaway-Sant'Anna) if treatment timing varies.
