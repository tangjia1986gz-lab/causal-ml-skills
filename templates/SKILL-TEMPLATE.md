---
name: skill-name-here
description: Use when [specific triggering conditions]. Triggers on keywords like [keyword1], [keyword2].
---

# Skill Name

> **Version**: 0.1.0 | **Type**: Knowledge/Tool/Estimator/Workflow

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

2. **Mistake**: [Description]
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

**Expected Output**:
```
[Sample output]
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

---

## K-Dense Directory Structure

This skill follows the K-Dense structure standard:

```
skill-name/
├── SKILL.md                 # This file (main documentation)
│
├── references/              # Reference documents (required)
│   ├── [topic_1].md         # Detailed topic documentation
│   ├── [topic_2].md         # Additional reference material
│   └── common_errors.md     # Common mistakes and fixes
│
├── scripts/                 # Support scripts (optional)
│   └── skill_name.py        # Python implementation
│
└── assets/                  # Resource files (optional)
    ├── latex/               # LaTeX templates
    └── markdown/            # Markdown templates
```

### Required Files for K-Dense Compliance

| File | Required | Purpose |
|------|----------|---------|
| `SKILL.md` | Yes | Main skill documentation |
| `references/` directory | Yes | Contains detailed reference docs |
| `references/common_errors.md` | Recommended | Common mistakes and solutions |
| `scripts/` directory | Optional | Python/R implementation files |
| `assets/` directory | Optional | Templates and resources |

### Creating This Skill

Use the scaffold generator for K-Dense compliant structure:

```bash
python scripts/generate_skill_scaffold.py \
    --name skill-name \
    --category [category] \
    --type [knowledge|tool|estimator|workflow]
```

### Validating This Skill

```bash
python scripts/validate_skill.py skills/[category]/skill-name
```
