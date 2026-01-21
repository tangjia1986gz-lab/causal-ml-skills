---
name: my-workflow
description: Use when end-to-end process orchestration. Triggers on end-to-end, pipeline, full-analysis.
version: 0.1.0
type: workflow
triggers:
  - end-to-end
  - pipeline
  - full-analysis
---

# My Workflow

> **Version**: 0.1.0 | **Type**: Workflow

## Overview

[1-2 sentences defining what this skill does and its core value proposition]

## When to Use

Use this skill when:
- [Specific scenario 1]
- [Specific scenario 2]
- [Specific scenario 3]

**When NOT to use:**
- [Situation where this skill is inappropriate]
- [Alternative skill to use instead]

## Prerequisites

- [ ] Python environment with `[required_packages]`
- [ ] Data in DataFrame format with columns: `[required_columns]`
- [ ] [Other prerequisites]

## Quick Reference

| Parameter | Description | Default |
|-----------|-------------|---------|
| `param1` | Description | `value` |
| `param2` | Description | `value` |

## Core Workflow

### Phase 1: [Phase Name]

**Objective**: [What this phase accomplishes]

**Steps**:
1. [Step 1]
2. [Step 2]

**Verification**:
- [ ] [Checkpoint to verify before proceeding]

### Phase 2: [Phase Name]

[Continue pattern...]

## Implementation

### Python Code Template

```python
# Example implementation
def skill_function(data, outcome, treatment, controls=None):
    """
    [Function description]

    Parameters
    ----------
    data : pd.DataFrame
        Input data
    outcome : str
        Name of outcome variable
    treatment : str
        Name of treatment variable
    controls : list, optional
        List of control variable names

    Returns
    -------
    result : CausalOutput
        Estimation results
    """
    pass
```

## Diagnostics & Validation

### Required Checks
- [ ] [Check 1]: [What it validates]
- [ ] [Check 2]: [What it validates]

### Interpretation Guide
- **If [condition]**: [interpretation]
- **If [condition]**: [interpretation]

## Common Mistakes

1. **Mistake**: [Description]
   - **Symptom**: [How it manifests]
   - **Fix**: [How to correct]

## Examples

### Example 1: [Basic Usage]

```python
# Load data
import pandas as pd
data = pd.read_csv("example.csv")

# Run analysis
result = skill_function(
    data=data,
    outcome="y",
    treatment="d",
    controls=["x1", "x2"]
)

print(result.summary_table)
```

## References

- [Paper/Book 1]
- [Paper/Book 2]
- [Online Resource]

---

## Appendix: Related Skills

| Skill | Relationship |
|-------|-------------|
| `[related-skill-1]` | [How they relate] |
| `[related-skill-2]` | [How they relate] |
