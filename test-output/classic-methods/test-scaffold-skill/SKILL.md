      ---
      name: test-scaffold-skill
      description: Use when estimating causal effects with Test Scaffold Skill. Triggers on method-specific-keywords, estimation, treatment-effect.
      version: 0.1.0
      type: estimator
      triggers:
        - method-specific-keywords
- estimation
- treatment-effect
      ---

# Estimator: Test Scaffold Skill

> **Version**: 0.1.0 | **Type**: Estimator
> **Aliases**: [Add alternative names]

## Overview

[Test Scaffold Skill] estimates causal effects by [describe core mechanism in 1-2 sentences].

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
- [Violation scenario 1] -> Consider `[alternative-estimator]`
- [Violation scenario 2] -> Consider `[alternative-estimator]`

## Identification Assumptions

| Assumption | Description | Testable? |
|------------|-------------|-----------|
| [Assumption 1] | [What it means] | Yes/No |
| [Assumption 2] | [What it means] | Yes/No |
| [Assumption 3] | [What it means] | Yes/No |

---

## Workflow

```
+-------------------------------------------------------------+
|                    ESTIMATOR WORKFLOW                        |
+-------------------------------------------------------------+
|  1. SETUP          -> Define variables, check data structure |
|  2. PRE-ESTIMATION -> Validate identification assumptions    |
|  3. ESTIMATION     -> Run main model                         |
|  4. DIAGNOSTICS    -> Robustness & sensitivity checks        |
|  5. REPORTING      -> Generate tables & interpretation       |
+-------------------------------------------------------------+
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

[TODO: Add assumption tests specific to this estimator]

### Phase 3: Main Estimation

**Model Specification**:

[TODO: Add mathematical model specification]

**Python Implementation**:

```python
from test_scaffold_skill_estimator import estimate

result = estimate(
    data=df,
    outcome="y",
    treatment="d",
    controls=["x1", "x2"]
)
```

### Phase 4: Robustness Checks

| Check | Purpose | Implementation |
|-------|---------|----------------|
| Placebo Test | Validate no pre-treatment effects | `placebo_test()` |
| Sensitivity Analysis | Assess robustness to unmeasured confounding | `sensitivity_analysis()` |
| Alternative Specification | Test model specification | `alt_specification()` |

### Phase 5: Reporting

[TODO: Add reporting templates and interpretation guide]

---

## Common Mistakes

### 1. [Common Mistake Category]

**Mistake**: [What people do wrong]

**Why it's wrong**: [Explanation of the problem]

**Correct approach**:
```python
# Correct code
```

---

## Examples

### Example 1: [Classic Application]

**Research Question**: [Question]

**Data**: [Description]

```python
import pandas as pd
from test_scaffold_skill_estimator import estimate

# Load data
data = pd.read_csv("example_data.csv")

# Run estimation
result = estimate(
    data=data,
    outcome="outcome_var",
    treatment="treatment_var",
    controls=["control1", "control2"]
)

print(result.summary_table)
```

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
