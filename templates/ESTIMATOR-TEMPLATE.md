---
name: estimator-method-name
description: Use when estimating [causal effect type] with [method name]. Triggers on [DID/RD/IV/PSM/DDML/etc.].
---

# Estimator: [Method Full Name]

> **Version**: 0.1.0 | **Type**: Estimator
> **Aliases**: [Alternative names, e.g., DID, Diff-in-Diff]

## Overview

[Method name] estimates causal effects by [core mechanism in 1-2 sentences].

**Key Identification Assumption**: [Core assumption that enables causal interpretation]

## When to Use

### Ideal Scenarios
- [Research design scenario 1]
- [Research design scenario 2]

### Data Requirements
- [ ] [Data structure requirement 1]
- [ ] [Data structure requirement 2]
- [ ] [Variable requirements]

### When NOT to Use
- [Violation scenario 1] → Consider `[alternative-estimator]`
- [Violation scenario 2] → Consider `[alternative-estimator]`

## Identification Assumptions

| Assumption | Description | Testable? |
|------------|-------------|-----------|
| [Assumption 1] | [What it means] | Yes/No |
| [Assumption 2] | [What it means] | Yes/No |
| [Assumption 3] | [What it means] | Yes/No |

---

## Workflow

```
┌─────────────────────────────────────────────────────────────┐
│                    ESTIMATOR WORKFLOW                        │
├─────────────────────────────────────────────────────────────┤
│  1. SETUP          → Define variables, check data structure │
│  2. PRE-ESTIMATION → Validate identification assumptions    │
│  3. ESTIMATION     → Run main model                         │
│  4. DIAGNOSTICS    → Robustness & sensitivity checks        │
│  5. REPORTING      → Generate tables & interpretation       │
└─────────────────────────────────────────────────────────────┘
```

### Phase 1: Setup

**Objective**: Prepare data and define model specification

**Inputs Required**:
```python
# Standard CausalInput structure
outcome = "y_variable"       # Outcome variable name
treatment = "d_variable"     # Treatment variable name
controls = ["x1", "x2"]      # Control variables
unit_id = "id"               # Panel: unit identifier
time_id = "year"             # Panel: time identifier
```

**Data Validation Checklist**:
- [ ] No missing values in key variables (or explicit handling strategy)
- [ ] Treatment is binary/continuous as expected
- [ ] Panel structure is balanced (if applicable)
- [ ] Sufficient observations in treatment/control groups

### Phase 2: Pre-Estimation Checks

**[Assumption 1] Test: [Test Name]**

```python
# Code template for test
def test_assumption_1(data, treatment, outcome, **kwargs):
    """
    [Test description]

    Returns
    -------
    passed : bool
    statistics : dict
    visualization : Figure
    """
    pass
```

**Interpretation**:
- ✅ Pass if: [condition]
- ⚠️ Warning if: [condition]
- ❌ Fail if: [condition]

---

**[Assumption 2] Test: [Test Name]**

[Same structure as above]

---

### Phase 3: Main Estimation

**Model Specification**:

$$
Y_{it} = \alpha + \beta \cdot D_{it} + \gamma \cdot X_{it} + \epsilon_{it}
$$

Where:
- $Y_{it}$: Outcome for unit $i$ at time $t$
- $D_{it}$: Treatment indicator
- $X_{it}$: Control variables
- $\beta$: **Causal effect of interest**

**Python Implementation**:

```python
def estimate(data, outcome, treatment, controls=None, **method_params):
    """
    Main estimation function for [Method Name].

    Parameters
    ----------
    data : pd.DataFrame
        Panel or cross-sectional data
    outcome : str
        Name of outcome variable (Y)
    treatment : str
        Name of treatment variable (D)
    controls : list, optional
        Names of control variables (X)
    **method_params : dict
        Method-specific parameters:
        - param1 : type, description (default: value)
        - param2 : type, description (default: value)

    Returns
    -------
    CausalOutput
        effect : float - Point estimate of causal effect
        se : float - Standard error
        ci_lower, ci_upper : float - 95% confidence interval
        p_value : float
        diagnostics : dict - All diagnostic test results
        summary_table : str - Publication-ready table
    """
    import [required_packages]

    # Step 1: Data preparation
    # ...

    # Step 2: Model fitting
    # ...

    # Step 3: Extract results
    # ...

    return CausalOutput(...)
```

**R Implementation** (if applicable):

```r
# R code via rpy2 bridge
library(package_name)

estimate_r <- function(data, outcome, treatment, controls) {
    # Implementation
}
```

### Phase 4: Robustness Checks

| Check | Purpose | Implementation |
|-------|---------|----------------|
| Placebo Test | Validate no pre-treatment effects | `placebo_test()` |
| Sensitivity Analysis | Assess robustness to unmeasured confounding | `sensitivity_analysis()` |
| Alternative Specification | Test model specification | `alt_specification()` |
| Subsample Analysis | Check heterogeneity | `subsample_analysis()` |

**Placebo Test Template**:

```python
def placebo_test(data, outcome, treatment, placebo_treatment, **kwargs):
    """
    Run placebo test with fake treatment.

    The placebo effect should be statistically insignificant.
    If significant, identification assumption may be violated.
    """
    pass
```

### Phase 5: Reporting

**Standard Output Table Format**:

```
┌──────────────────────────────────────────────────────┐
│                   Table X: [Title]                   │
├──────────────────────────────────────────────────────┤
│                         (1)        (2)        (3)    │
│                        Base    + Controls   Full     │
├──────────────────────────────────────────────────────┤
│ Treatment Effect      0.XXX***   0.XXX***   0.XXX*** │
│                      (0.XXX)    (0.XXX)    (0.XXX)   │
│                                                      │
│ Controls               No         Yes        Yes     │
│ Fixed Effects          No         No         Yes     │
│                                                      │
│ Observations          X,XXX      X,XXX      X,XXX    │
│ R-squared             0.XXX      0.XXX      0.XXX    │
├──────────────────────────────────────────────────────┤
│ Notes: Standard errors in parentheses.               │
│ *** p<0.01, ** p<0.05, * p<0.1                       │
└──────────────────────────────────────────────────────┘
```

**Interpretation Template**:

```markdown
## Results Interpretation

The estimated treatment effect is **[β]** (SE = [se], p = [p-value]),
suggesting that [treatment description] leads to a [increase/decrease]
of [effect magnitude] in [outcome description].

This effect is [statistically significant at the X% level / not statistically significant].

**Economic Significance**: [Interpretation of magnitude in practical terms]

**Caveats**: [Any limitations or required assumptions]
```

---

## Common Mistakes

### 1. [Common Mistake Category]

**Mistake**: [What people do wrong]

**Why it's wrong**: [Explanation of the problem]

**Correct approach**:
```python
# Correct code
```

### 2. [Common Mistake Category]

[Same structure]

---

## Examples

### Example 1: [Classic Application]

**Research Question**: [Question]

**Data**: [Description]

```python
import pandas as pd
from causal_ml_skills import estimator_method

# Load data
data = pd.read_csv("example_data.csv")

# Define variables
outcome = "outcome_var"
treatment = "treatment_var"
controls = ["control1", "control2"]

# Run estimation
result = estimator_method.estimate(
    data=data,
    outcome=outcome,
    treatment=treatment,
    controls=controls
)

# View results
print(result.summary_table)

# Check diagnostics
print(result.diagnostics)
```

**Output**:
```
[Expected output]
```

**Interpretation**:
[How to interpret these results]

---

## References

### Seminal Papers
- [Author (Year). Title. Journal.]

### Textbook Treatments
- [Author. Book Title. Chapter X.]

### Software Documentation
- [Package documentation links]

---

## Related Estimators

| Estimator | When to Use Instead |
|-----------|---------------------|
| `[estimator-1]` | [Scenario] |
| `[estimator-2]` | [Scenario] |

---

## Appendix: Mathematical Details

### Derivation of Estimator

[Optional: Include for advanced users]

### Asymptotic Properties

[Optional: Include for advanced users]
