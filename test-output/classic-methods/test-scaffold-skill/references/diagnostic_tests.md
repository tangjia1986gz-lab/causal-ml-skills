# Diagnostic Tests: Test Scaffold Skill

## Pre-Estimation Diagnostics

### Test 1: [Name]

**Purpose**: [What it tests]

**Implementation**:
```python
from test_scaffold_skill_estimator import test_name

result = test_name(data, ...)
```

**Interpretation**:
- Pass: [When to proceed]
- Warning: [When to be cautious]
- Fail: [When to stop or reconsider]

## Post-Estimation Diagnostics

[Similar structure for post-estimation tests]

## Diagnostic Decision Tree

```
Start
  |
  v
[Test 1] --Pass--> [Test 2] --Pass--> Proceed
  |                   |
  Fail                Fail
  |                   |
  v                   v
[Action]           [Action]
```
