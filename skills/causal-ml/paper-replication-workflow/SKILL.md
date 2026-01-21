---
name: paper-replication-workflow
triggers:
  - replication
  - replicate paper
  - reproduce results
  - LaLonde
  - Card
  - empirical paper
  - replication study
  - reproduce findings
  - original paper
  - published results
---

# Paper Replication Workflow

## Overview

This skill provides a systematic workflow for replicating empirical economics papers. Replication is fundamental to scientific credibility and serves multiple purposes:

1. **Verification**: Confirm published results are reproducible
2. **Learning**: Deep understanding of methods through implementation
3. **Extension**: Build foundation for new research
4. **Teaching**: Demonstrate methods with real applications

This workflow guides you from paper analysis through result comparison, integrating with all estimator skills in the causal-ml toolkit.

## Workflow Stages

### Stage 1: Paper Analysis

Before touching any data or code, thoroughly analyze the paper.

#### 1.1 Research Question Identification

Extract the core causal question:
- What is the treatment?
- What is the outcome?
- What is the population of interest?
- What is the estimand (ATT, ATE, LATE)?

**Example - LaLonde (1986)**:
- Treatment: Job training program (NSW)
- Outcome: Post-training earnings (1978)
- Population: Disadvantaged workers
- Estimand: ATT (effect on the treated)

#### 1.2 Data Source Documentation

Document all data sources used:

```
Data Sources:
├── Primary Data
│   ├── Name: [dataset name]
│   ├── Source: [where to obtain]
│   ├── Time Period: [years covered]
│   └── Sample Size: [N observations]
├── Secondary Data (if any)
│   └── [repeat structure]
└── Variable Definitions
    ├── Treatment: [definition]
    ├── Outcome: [definition]
    └── Covariates: [list with definitions]
```

#### 1.3 Estimation Method Identification

Map the paper's method to our estimator skills:

| Paper Method | Skill | Key Identifying Features |
|-------------|-------|-------------------------|
| Difference-in-Differences | `estimator-did` | Treatment/control, pre/post periods |
| Regression Discontinuity | `estimator-rd` | Running variable, cutoff |
| Instrumental Variables | `estimator-iv` | Instrument, exclusion restriction argument |
| Propensity Score Matching | `estimator-psm` | Selection on observables, matched samples |
| Double/Debiased ML | `causal-ddml` | High-dimensional controls, ML methods |
| Synthetic Control | (future) | Aggregate units, donor pool |

#### 1.4 Specification Documentation

Extract all specifications run in the paper:

```python
paper_specifications = {
    "main": {
        "outcome": "re78",
        "treatment": "treat",
        "covariates": ["age", "education", "black", "hispanic",
                       "married", "nodegree", "re74", "re75"],
        "method": "psm",
        "method_details": {
            "matching_method": "nearest_neighbor",
            "caliper": None,
            "replacement": True
        }
    },
    "robustness": [
        {"name": "no_lagged_earnings", "drop_covariates": ["re74", "re75"]},
        {"name": "kernel_matching", "method_details": {"matching_method": "kernel"}},
        {"name": "radius_matching", "method_details": {"caliper": 0.1}}
    ]
}
```

### Stage 2: Data Acquisition

#### 2.1 Locate Original Data

Priority order for data sources:

1. **Author's replication package** - Check journal website, author's page, dataverse
2. **Standard repositories** - ICPSR, Harvard Dataverse, OSF
3. **Package-bundled data** - Many classic datasets in R/Python packages
4. **Reconstruct from description** - Last resort, document limitations

**Common Replication Datasets**:

| Dataset | Source | Papers Using It |
|---------|--------|-----------------|
| LaLonde NSW | `causalml`, `lalonde` R package | LaLonde (1986), Dehejia & Wahba (1999) |
| Card College | NLSY79 | Card (1995) |
| Lee Elections | Author's website | Lee (2008) |
| Card & Krueger MW | Author's files | Card & Krueger (1994) |
| Oregon Health | NBER | Finkelstein et al. (2012) |

#### 2.2 Variable Definition Verification

Create variable mapping table:

```python
variable_mapping = {
    "paper_name": "code_name",
    "Annual earnings 1978": "re78",
    "Treatment indicator": "treat",
    "Age in years": "age",
    "Years of education": "education",
    # ... etc
}
```

#### 2.3 Sample Construction Verification

Document sample restrictions:

```python
sample_construction = {
    "initial_n": 722,  # Full NSW data
    "restrictions": [
        {"condition": "age >= 18 and age <= 55", "remaining": 720},
        {"condition": "not missing(re78)", "remaining": 722},
    ],
    "final_n": 722,
    "paper_reported_n": 722,
    "match": True
}
```

### Stage 3: Method Selection

#### 3.1 Route to Correct Estimator Skill

```python
METHOD_SKILL_MAP = {
    # Difference-in-Differences variants
    "did": "estimator-did",
    "difference-in-differences": "estimator-did",
    "diff-in-diff": "estimator-did",
    "twfe": "estimator-did",
    "two-way fixed effects": "estimator-did",
    "event study": "estimator-did",

    # Regression Discontinuity variants
    "rd": "estimator-rd",
    "rdd": "estimator-rd",
    "regression discontinuity": "estimator-rd",
    "sharp rd": "estimator-rd",
    "fuzzy rd": "estimator-rd",

    # Instrumental Variables variants
    "iv": "estimator-iv",
    "2sls": "estimator-iv",
    "tsls": "estimator-iv",
    "instrumental variables": "estimator-iv",
    "liml": "estimator-iv",

    # Propensity Score Methods
    "psm": "estimator-psm",
    "propensity score": "estimator-psm",
    "matching": "estimator-psm",
    "ipw": "estimator-psm",
    "inverse probability weighting": "estimator-psm",

    # Machine Learning Methods
    "ddml": "causal-ddml",
    "double ml": "causal-ddml",
    "debiased ml": "causal-ddml",
    "dml": "causal-ddml",
    "causal forest": "causal-ddml",
}
```

#### 3.2 Extract Skill Parameters from Paper

Map paper specifications to skill function arguments:

**Example - PSM for LaLonde**:

```python
# Paper specification
paper_spec = {
    "method": "propensity score matching",
    "propensity_model": "logit",
    "matching": "nearest neighbor without replacement",
    "covariates": ["age", "educ", "black", "hispan", "married",
                   "nodegree", "re74", "re75"]
}

# Mapped to skill parameters
skill_params = {
    "treatment_col": "treat",
    "outcome_col": "re78",
    "covariate_cols": ["age", "educ", "black", "hispan", "married",
                       "nodegree", "re74", "re75"],
    "method": "nearest",
    "n_neighbors": 1,
    "replacement": False,
    "caliper": None
}
```

### Stage 4: Replication Execution

#### 4.1 Run Main Specification

```python
from replication_workflow import run_main_specification

# Load data
data = load_replication_data("lalonde")

# Define paper specification
paper_spec = {
    "paper_name": "LaLonde (1986)",
    "method": "psm",
    "treatment": "treat",
    "outcome": "re78",
    "covariates": ["age", "education", "black", "hispanic",
                   "married", "nodegree", "re74", "re75"],
    "original_estimate": 1794,
    "original_se": 633
}

# Run replication
results = run_main_specification(data, paper_spec)
```

#### 4.2 Run Robustness Checks

```python
robustness_specs = [
    {"name": "CPS comparison", "control_data": "cps"},
    {"name": "PSID comparison", "control_data": "psid"},
    {"name": "No lagged earnings", "drop_covariates": ["re74", "re75"]},
]

robustness_results = run_robustness_checks(data, paper_spec, robustness_specs)
```

#### 4.3 Document Execution

Keep detailed logs:

```
Replication Log - LaLonde (1986)
================================
Date: 2024-01-15
Software: Python 3.10, pandas 2.0, statsmodels 0.14

Step 1: Data Loading
- Loaded lalonde dataset: 722 obs (NSW + PSID comparison)
- Treatment: 185 treated, 537 control
- Outcome mean: $6,349 (treated), $4,555 (control)

Step 2: Main Specification
- Method: Propensity Score Matching (nearest neighbor)
- Covariates: 8 variables
- Matched sample: 185 treated, 185 control
- ATT estimate: $1,672 (SE: $712)

Step 3: Original Paper Values
- Reported ATT: $1,794 (SE: $633)
- Difference: -$122 (6.8% lower)
```

### Stage 5: Discrepancy Diagnosis

#### 5.1 Tolerance Levels

Define success criteria:

| Tolerance Level | Point Estimate | Standard Error | Interpretation |
|-----------------|----------------|----------------|----------------|
| **Exact** | < 1% difference | < 5% difference | Identical results |
| **Close** | < 5% difference | < 10% difference | Minor numerical differences |
| **Approximate** | < 10% difference | < 20% difference | Methodological variations |
| **Qualitative** | Same sign & significance | - | Conclusions match |
| **Failed** | Different conclusions | - | Cannot replicate |

#### 5.2 Common Sources of Differences

When results don't match, investigate systematically:

**Data Issues**:
- [ ] Sample size matches paper?
- [ ] Variable definitions identical?
- [ ] Missing value treatment same?
- [ ] Data vintage/corrections applied?

**Method Issues**:
- [ ] Estimation method identical?
- [ ] Standard error calculation same?
- [ ] Clustering/weighting applied correctly?
- [ ] Optimization algorithm converged?

**Software Issues**:
- [ ] Package version differences?
- [ ] Default settings differ?
- [ ] Numerical precision issues?
- [ ] Random seed for stochastic methods?

#### 5.3 Debugging Strategy

```python
def diagnose_discrepancy(original, replicated, data, spec):
    """Systematic diagnosis of replication discrepancies."""

    diagnosis = {
        "point_estimate_diff": replicated["estimate"] - original["estimate"],
        "percent_diff": (replicated["estimate"] - original["estimate"]) / original["estimate"] * 100,
        "se_diff": replicated["se"] - original["se"],
        "conclusion_match": (
            (replicated["pvalue"] < 0.05) == (original["pvalue"] < 0.05) and
            np.sign(replicated["estimate"]) == np.sign(original["estimate"])
        )
    }

    # Check potential sources
    checks = []

    # Sample size
    if data.shape[0] != original["n"]:
        checks.append(f"Sample size mismatch: {data.shape[0]} vs {original['n']}")

    # Variable means
    for var in spec["covariates"]:
        if var in original.get("means", {}):
            diff = abs(data[var].mean() - original["means"][var])
            if diff > 0.01:
                checks.append(f"Mean of {var} differs by {diff:.3f}")

    diagnosis["potential_issues"] = checks
    return diagnosis
```

### Stage 6: Report Generation

#### 6.1 Side-by-Side Comparison Table

```
═══════════════════════════════════════════════════════════════════
                    REPLICATION COMPARISON: LaLonde (1986)
═══════════════════════════════════════════════════════════════════

Specification          Original        Replicated       Difference
─────────────────────────────────────────────────────────────────
Main Estimate
  Point Estimate       $1,794          $1,672           -$122 (6.8%)
  Standard Error       $633            $712             +$79 (12.5%)
  95% CI              [$553, $3,035]  [$277, $3,067]
  p-value              0.005           0.019

Sample
  N (treated)          185             185              Match
  N (control)          185             185              Match

Robustness Checks
  CPS comparison       -$635           -$642            Match (< 5%)
  PSID comparison      $1,069          $1,045           Match (< 5%)
  No lagged earn       $886            $912             Match (< 5%)

─────────────────────────────────────────────────────────────────
REPLICATION STATUS: APPROXIMATE SUCCESS
Main conclusions replicated; minor numerical differences likely
due to software/algorithm variations.
═══════════════════════════════════════════════════════════════════
```

#### 6.2 Full Replication Report Structure

```markdown
# Replication Report: [Paper Citation]

## 1. Paper Summary
- Research question
- Main findings
- Methods used

## 2. Data
- Sources and access
- Sample construction
- Variable definitions

## 3. Methods
- Estimation strategy
- Skill used
- Parameter choices

## 4. Results Comparison
### 4.1 Main Specification
[Comparison table]

### 4.2 Robustness Checks
[Additional tables]

## 5. Discrepancy Analysis
- Sources of differences
- Resolution attempts
- Remaining issues

## 6. Conclusions
- Replication success level
- Confidence in original findings
- Recommendations

## Appendix
- Code
- Data dictionary
- Detailed output
```

## Classic Replication Examples

### Example 1: LaLonde (1986) - Job Training

**Paper**: LaLonde, R. (1986). "Evaluating the Econometric Evaluations of Training Programs with Experimental Data." *American Economic Review*.

**Research Question**: Can non-experimental methods recover experimental treatment effects?

**Methods**:
- Experimental benchmark (NSW)
- Propensity Score Matching
- Selection models

**Data**:
- NSW experimental data (185 treated, 260 control)
- CPS comparison group
- PSID comparison group

**Key Specifications**:
```python
lalonde_specs = {
    "experimental_benchmark": {
        "method": "simple_difference",
        "estimate": 1794,
        "se": 633
    },
    "psm_psid": {
        "method": "psm",
        "control_group": "psid",
        "estimate": 1069,
        "se": 886
    },
    "psm_cps": {
        "method": "psm",
        "control_group": "cps",
        "estimate": -635,
        "se": 1073
    }
}
```

**Replication Command**:
```python
from replication_workflow import replicate_lalonde
results = replicate_lalonde()
```

### Example 2: Card (1995) - College Proximity IV

**Paper**: Card, D. (1995). "Using Geographic Variation in College Proximity to Estimate the Return to Schooling."

**Research Question**: What is the causal return to education?

**Methods**:
- OLS (biased by ability)
- IV using college proximity as instrument

**Data**: NLSY79

**Key Specifications**:
```python
card_specs = {
    "ols": {
        "method": "ols",
        "outcome": "log_wage",
        "treatment": "education",
        "estimate": 0.073,
        "se": 0.004
    },
    "iv": {
        "method": "iv",
        "instrument": "nearc4",  # Near 4-year college
        "estimate": 0.132,
        "se": 0.055
    }
}
```

**Replication Command**:
```python
from replication_workflow import replicate_card_proximity
results = replicate_card_proximity()
```

### Example 3: Lee (2008) - Elections RD

**Paper**: Lee, D. (2008). "Randomized Experiments from Non-random Selection in U.S. House Elections."

**Research Question**: What is the incumbency advantage in elections?

**Methods**: Sharp Regression Discontinuity

**Data**: U.S. House elections

**Key Specifications**:
```python
lee_specs = {
    "main_rd": {
        "method": "rd",
        "running_variable": "vote_margin",  # Democrat margin at t
        "cutoff": 0.5,
        "outcome": "win_next",  # Democrat wins at t+1
        "bandwidth": "optimal",
        "estimate": 0.38,
        "se": 0.06
    }
}
```

**Replication Command**:
```python
from replication_workflow import replicate_lee_elections
results = replicate_lee_elections()
```

### Example 4: Card & Krueger (1994) - Minimum Wage DID

**Paper**: Card, D. and Krueger, A. (1994). "Minimum Wages and Employment: A Case Study of the Fast-Food Industry in New Jersey and Pennsylvania."

**Research Question**: Do minimum wage increases reduce employment?

**Methods**: Difference-in-Differences

**Data**: Survey of fast food restaurants

**Key Specifications**:
```python
card_krueger_specs = {
    "main_did": {
        "method": "did",
        "treatment_group": "nj",  # New Jersey
        "control_group": "pa",   # Pennsylvania
        "pre_period": "feb1992",
        "post_period": "nov1992",
        "outcome": "fte_employment",  # Full-time equivalent
        "estimate": 2.76,
        "se": 1.36
    }
}
```

## Result Comparison Framework

### Comparison Metrics

```python
def compute_comparison_metrics(original, replicated):
    """Compute comprehensive comparison metrics."""

    metrics = {
        # Point estimate comparison
        "estimate_diff": replicated["estimate"] - original["estimate"],
        "estimate_pct_diff": (replicated["estimate"] - original["estimate"]) / abs(original["estimate"]) * 100,

        # Standard error comparison
        "se_diff": replicated["se"] - original["se"],
        "se_pct_diff": (replicated["se"] - original["se"]) / original["se"] * 100,

        # Confidence interval overlap
        "ci_overlap": compute_ci_overlap(
            original["estimate"], original["se"],
            replicated["estimate"], replicated["se"]
        ),

        # Statistical conclusion
        "same_significance": (
            (replicated["pvalue"] < 0.05) == (original["pvalue"] < 0.05)
        ),
        "same_sign": np.sign(replicated["estimate"]) == np.sign(original["estimate"]),

        # Normalized difference
        "normalized_diff": (replicated["estimate"] - original["estimate"]) /
                           np.sqrt(original["se"]**2 + replicated["se"]**2)
    }

    return metrics
```

### Success Classification

```python
def classify_replication_success(metrics):
    """Classify replication success level."""

    if abs(metrics["estimate_pct_diff"]) < 1 and abs(metrics["se_pct_diff"]) < 5:
        return "EXACT", "Results match within numerical precision"

    elif abs(metrics["estimate_pct_diff"]) < 5 and abs(metrics["se_pct_diff"]) < 10:
        return "CLOSE", "Minor numerical differences, likely software variation"

    elif abs(metrics["estimate_pct_diff"]) < 10 and abs(metrics["se_pct_diff"]) < 20:
        return "APPROXIMATE", "Moderate differences, possibly different specifications"

    elif metrics["same_sign"] and metrics["same_significance"]:
        return "QUALITATIVE", "Conclusions match despite numerical differences"

    else:
        return "FAILED", "Cannot replicate main conclusions"
```

## Publishing Replication Results

### When Replication Succeeds

- Confirm findings with additional robustness checks
- Extend analysis with additional methods (e.g., DDML)
- Document methodology for teaching purposes

### When Replication Fails

1. **Document thoroughly** - Every step and discrepancy
2. **Contact authors** - Seek clarification before publishing
3. **Check for errata** - Paper may have known corrections
4. **Independent verification** - Have others attempt replication
5. **Publish constructively** - Focus on improving science

### Replication Study Venues

- **Journal of Applied Econometrics** - Replication section
- **AEA Data and Code Repository** - Verification studies
- **I4R** - Institute for Replication
- **Working papers** - Document and share

## Usage Example

```python
from replication_workflow import (
    parse_paper_specification,
    load_replication_data,
    run_main_specification,
    run_robustness_checks,
    compare_results,
    generate_replication_report
)

# Step 1: Define paper specification
paper_spec = {
    "paper_name": "LaLonde (1986)",
    "citation": "LaLonde, R. (1986). AER 76(4): 604-620",
    "method": "psm",
    "treatment": "treat",
    "outcome": "re78",
    "covariates": ["age", "education", "black", "hispanic",
                   "married", "nodegree", "re74", "re75"],
    "original_results": {
        "estimate": 1794,
        "se": 633,
        "n_treated": 185,
        "n_control": 260
    }
}

# Step 2: Load data
data = load_replication_data("lalonde_nsw")

# Step 3: Run main specification
results = run_main_specification(data, paper_spec)

# Step 4: Compare to original
comparison = compare_results(
    original=paper_spec["original_results"],
    replicated=results,
    tolerance="approximate"
)

# Step 5: Generate report
report = generate_replication_report(
    paper_spec=paper_spec,
    results=results,
    comparison=comparison
)

print(report)
```

## Integration with Other Skills

This skill integrates with all estimator skills:

```
paper-replication-workflow
    ├── estimator-did      → DID papers
    ├── estimator-rd       → RD papers
    ├── estimator-iv       → IV papers
    ├── estimator-psm      → Matching papers
    └── causal-ddml        → ML extension of any method
```

For each replication, the workflow:
1. Parses paper specification
2. Routes to appropriate estimator skill
3. Executes analysis using skill's validated implementation
4. Compares results systematically
5. Generates professional report
