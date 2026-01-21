# Causal Considerations in Data Preprocessing

## Overview

Preprocessing decisions can make or break causal inference validity. This document addresses the critical question: **What variables should be included or excluded to identify causal effects?** We cover pre-treatment variable selection, avoiding bad controls, and preserving causal identification through preprocessing.

## The Fundamental Principle: Pre-Treatment Only

### Why Pre-Treatment Matters

In causal inference, we want to compare potential outcomes Y(1) and Y(0). Valid controls must satisfy:

1. **Measured before treatment assignment** - Cannot be affected by treatment
2. **Block backdoor paths** - Account for common causes of treatment and outcome
3. **Not open collider paths** - Avoid conditioning on consequences of treatment and outcome

```
Timeline for valid control selection:

    Confounders (Z)     Treatment (D)     Mediators (M)     Outcome (Y)
         |                    |                |                |
    <----+------ PRE-TREATMENT ------>  <---- POST-TREATMENT ---->

    INCLUDE as controls      ---        DO NOT INCLUDE    ---
```

### Identifying Pre-Treatment Variables

```python
def classify_variables(df, treatment, outcome, variable_timing):
    """
    Classify variables by their relationship to treatment.

    Parameters
    ----------
    variable_timing : dict
        Maps variable names to 'pre', 'post', or 'unknown'

    Returns
    -------
    dict
        Classification of variables
    """
    classification = {
        'pre_treatment': [],
        'post_treatment': [],
        'unknown': [],
        'treatment': treatment,
        'outcome': outcome
    }

    for var, timing in variable_timing.items():
        if var not in [treatment, outcome]:
            classification[f'{timing}_treatment' if timing != 'unknown' else 'unknown'].append(var)

    return classification

# Example usage
variable_timing = {
    'age': 'pre',
    'education': 'pre',
    'baseline_income': 'pre',
    'current_income': 'post',  # Could be affected by treatment
    'job_satisfaction': 'unknown',  # Need domain knowledge
}

classification = classify_variables(df, 'training_program', 'productivity', variable_timing)
print(f"Valid controls: {classification['pre_treatment']}")
print(f"Exclude: {classification['post_treatment']}")
```

---

## Bad Controls: What to Avoid

### Types of Bad Controls

#### 1. Mediators (Post-Treatment Variables on Causal Path)

```
D --> M --> Y
      ^
      |
    BAD CONTROL
```

Controlling for mediators removes part of the causal effect you're trying to estimate.

**Example**: Training program (D) affects skills (M) which affects wages (Y)
- Controlling for skills blocks the causal pathway
- You would only estimate the "direct effect" not mediated by skills

```python
# WRONG: Including mediator
bad_controls = ['skills', 'productivity']  # Affected by training
effect = estimate_with_controls(df, 'training', 'wages', controls=bad_controls)  # BIASED

# CORRECT: Pre-treatment variables only
good_controls = ['baseline_skills', 'education', 'experience']
effect = estimate_with_controls(df, 'training', 'wages', controls=good_controls)
```

#### 2. Colliders (Common Effects of Treatment and Outcome)

```
D --> C <-- Y
      ^
      |
    BAD CONTROL
```

Controlling for colliders opens a spurious path between D and Y.

**Example**: Success (C) depends on both training (D) and innate ability (U)
```
Training (D) --> Success (C) <-- Ability (U)
                                     |
                                     v
                                 Wages (Y)
```

Controlling for "success" creates spurious association between training and wages through ability.

```python
# WRONG: Conditioning on collider
# Restricting to "successful" employees opens backdoor through ability
df_successful = df[df['is_successful'] == 1]
effect = estimate(df_successful, 'training', 'wages')  # BIASED

# CORRECT: Include full population
effect = estimate(df, 'training', 'wages')  # Unbiased
```

#### 3. M-Bias (Controlling for Pre-Treatment Collider)

```
U1 --> Z <-- U2
|            |
v            v
D            Y
```

Z is pre-treatment but still a collider of unmeasured confounders.

**Example**:
- Parental investment (U1) affects both child's test prep (Z) and college attendance (D)
- Innate ability (U2) affects both test prep (Z) and earnings (Y)
- Controlling for test prep (Z) opens path between college and earnings through abilities

```python
# Pre-treatment doesn't always mean safe to control
# Requires domain knowledge about causal structure

# Potentially problematic
effect_with_z = estimate(df, 'college', 'earnings', controls=['test_scores'])

# May need to consider unmeasured confounders
```

#### 4. Instruments (Causes of Treatment Only)

```
Z --> D --> Y
```

Including instruments in regression reduces precision without reducing bias.

```python
# Z is an instrument (e.g., lottery for program eligibility)
# In standard regression, including Z is inefficient but not biasing

# For IV estimation, Z should NOT be in the outcome equation
# Use Z to predict D, then use predicted D

from sklearn.linear_model import LinearRegression

# First stage: D on Z
first_stage = LinearRegression().fit(df[['Z'] + controls], df['D'])
D_hat = first_stage.predict(df[['Z'] + controls])

# Second stage: Y on D_hat
second_stage = LinearRegression().fit(
    np.column_stack([D_hat, df[controls].values]),
    df['Y']
)
iv_effect = second_stage.coef_[0]
```

---

## Variable Selection Framework

### Decision Framework

For each potential control variable X, ask:

```
1. Is X measured BEFORE treatment?
   - No  --> EXCLUDE (potential mediator/collider)
   - Yes --> Continue to 2

2. Does X affect BOTH treatment AND outcome?
   - Yes --> INCLUDE (confounder)
   - No  --> Continue to 3

3. Does X affect only outcome (not treatment)?
   - Yes --> MAY INCLUDE (precision variable, but not required for identification)
   - No  --> Continue to 4

4. Does X affect only treatment (not outcome)?
   - Yes --> EXCLUDE (instrument) or use for IV
   - No  --> EXCLUDE (irrelevant, adds noise)
```

### Implementation

```python
def select_causal_controls(
    df,
    treatment,
    outcome,
    candidate_controls,
    variable_metadata
):
    """
    Select valid controls for causal inference.

    Parameters
    ----------
    variable_metadata : dict
        For each variable:
        - 'timing': 'pre' or 'post' treatment
        - 'affects_treatment': bool
        - 'affects_outcome': bool

    Returns
    -------
    dict
        Selected controls and exclusion reasons
    """
    selected = []
    excluded = {}

    for var in candidate_controls:
        meta = variable_metadata.get(var, {})

        # Check timing
        if meta.get('timing') == 'post':
            excluded[var] = 'Post-treatment variable (potential mediator)'
            continue

        # Check if confounder
        affects_d = meta.get('affects_treatment', False)
        affects_y = meta.get('affects_outcome', False)

        if affects_d and affects_y:
            selected.append(var)
        elif affects_y and not affects_d:
            selected.append(var)  # Precision variable
        elif affects_d and not affects_y:
            excluded[var] = 'Instrument (affects treatment only)'
        else:
            excluded[var] = 'Irrelevant (affects neither)'

    return {
        'selected': selected,
        'excluded': excluded
    }

# Example
metadata = {
    'age': {'timing': 'pre', 'affects_treatment': True, 'affects_outcome': True},
    'education': {'timing': 'pre', 'affects_treatment': True, 'affects_outcome': True},
    'lottery_number': {'timing': 'pre', 'affects_treatment': True, 'affects_outcome': False},
    'post_skills': {'timing': 'post', 'affects_treatment': False, 'affects_outcome': True},
}

result = select_causal_controls(df, 'training', 'wages', list(metadata.keys()), metadata)
print(f"Use as controls: {result['selected']}")
print(f"Excluded: {result['excluded']}")
```

---

## Preserving Identification Through Preprocessing

### Overlap/Positivity Requirement

Both treatment groups need representation at all covariate values:

```
P(D=1|X) > 0 and P(D=0|X) > 0 for all X
```

**Preprocessing can violate overlap by**:
- Removing observations that provide overlap
- Creating features that perfectly predict treatment

```python
def check_overlap(df, treatment, controls, bins=10):
    """
    Check overlap in propensity score distribution.

    Returns
    -------
    dict
        Overlap diagnostics
    """
    from sklearn.linear_model import LogisticRegression

    # Estimate propensity scores
    ps_model = LogisticRegression(max_iter=1000)
    ps_model.fit(df[controls], df[treatment])
    ps = ps_model.predict_proba(df[controls])[:, 1]

    # Check overlap
    ps_treated = ps[df[treatment] == 1]
    ps_control = ps[df[treatment] == 0]

    overlap_region = (
        max(ps_treated.min(), ps_control.min()),
        min(ps_treated.max(), ps_control.max())
    )

    # Units outside overlap
    outside_overlap = (ps < overlap_region[0]) | (ps > overlap_region[1])

    return {
        'overlap_region': overlap_region,
        'n_outside_overlap': outside_overlap.sum(),
        'pct_outside_overlap': outside_overlap.mean() * 100,
        'ps_treated_range': (ps_treated.min(), ps_treated.max()),
        'ps_control_range': (ps_control.min(), ps_control.max())
    }

# Usage
overlap = check_overlap(df, 'treatment', controls)
print(f"Overlap region: {overlap['overlap_region']}")
print(f"Units outside overlap: {overlap['n_outside_overlap']} ({overlap['pct_outside_overlap']:.1f}%)")
```

### Maintaining Balance Across Groups

Preprocessing should maintain or improve covariate balance:

```python
def assess_balance(df, treatment, covariates):
    """
    Assess covariate balance between treatment groups.

    Returns
    -------
    pd.DataFrame
        Balance statistics for each covariate
    """
    treated = df[df[treatment] == 1]
    control = df[df[treatment] == 0]

    balance_stats = []

    for cov in covariates:
        mean_t = treated[cov].mean()
        mean_c = control[cov].mean()

        # Pooled standard deviation
        var_t = treated[cov].var()
        var_c = control[cov].var()
        pooled_std = np.sqrt((var_t + var_c) / 2)

        # Standardized mean difference
        smd = (mean_t - mean_c) / pooled_std if pooled_std > 0 else 0

        balance_stats.append({
            'variable': cov,
            'mean_treated': mean_t,
            'mean_control': mean_c,
            'std_diff': smd,
            'abs_std_diff': abs(smd)
        })

    return pd.DataFrame(balance_stats).sort_values('abs_std_diff', ascending=False)

# Check balance before and after preprocessing
balance_raw = assess_balance(df_raw, 'treatment', controls)
balance_processed = assess_balance(df_processed, 'treatment', controls)

# Flag if preprocessing worsened balance
for var in controls:
    raw_smd = balance_raw[balance_raw['variable'] == var]['abs_std_diff'].values[0]
    proc_smd = balance_processed[balance_processed['variable'] == var]['abs_std_diff'].values[0]

    if proc_smd > raw_smd + 0.05:
        print(f"Warning: Balance worsened for {var} after preprocessing")
```

---

## Domain-Specific Guidance

### Labor Economics / Program Evaluation

| Variable | Include? | Reasoning |
|----------|----------|-----------|
| Age, gender, race | Yes | Pre-treatment demographics |
| Baseline education | Yes | Pre-treatment human capital |
| Prior earnings (before treatment) | Yes | Key confounder |
| Current earnings (during treatment) | No | Potential mediator |
| Post-program employment | No | Mediator or outcome |

### Healthcare / Clinical Trials

| Variable | Include? | Reasoning |
|----------|----------|-----------|
| Baseline health status | Yes | Key confounder |
| Pre-treatment lab values | Yes | Important for selection |
| Post-treatment adherence | No | Affected by treatment |
| Side effects | No | Caused by treatment |
| Baseline comorbidities | Yes | Confounders |

### Marketing / Business

| Variable | Include? | Reasoning |
|----------|----------|-----------|
| Pre-campaign purchase history | Yes | Key confounder |
| Customer demographics | Yes | Pre-treatment |
| Post-campaign clicks | No | Mediator to conversion |
| Post-campaign purchases | Outcome | What we're measuring |
| Pre-campaign engagement | Yes | Selection confounder |

---

## Preprocessing Checklist for Causal Validity

### Before Preprocessing

- [ ] Identify treatment and outcome variables
- [ ] List all candidate control variables
- [ ] Determine timing (pre/post treatment) for each variable
- [ ] Draw causal DAG if possible
- [ ] Identify potential confounders, mediators, colliders, instruments

### During Preprocessing

- [ ] Only include pre-treatment variables as controls
- [ ] Apply same transformations to treatment and control groups
- [ ] Check overlap after each preprocessing step
- [ ] Preserve all observations if possible (document any exclusions)
- [ ] Fit scalers/encoders on full data (or training data), not by treatment group

### After Preprocessing

- [ ] Verify overlap is maintained
- [ ] Check balance did not worsen
- [ ] Document all preprocessing decisions
- [ ] Save preprocessing pipeline for reproducibility
- [ ] Plan sensitivity analyses for preprocessing choices

---

## Common Mistakes and Solutions

### Mistake 1: Including Post-Treatment Variables

```python
# WRONG
controls = ['age', 'education', 'current_job_performance']  # 'current' is post-treatment
effect = estimate(df, treatment, outcome, controls)  # BIASED

# CORRECT
controls = ['age', 'education', 'baseline_job_performance']
effect = estimate(df, treatment, outcome, controls)
```

### Mistake 2: Different Preprocessing by Treatment Group

```python
# WRONG
treated_scaled = StandardScaler().fit_transform(df[df['D']==1][controls])
control_scaled = StandardScaler().fit_transform(df[df['D']==0][controls])

# CORRECT
scaler = StandardScaler()
df[controls] = scaler.fit_transform(df[controls])  # Same transformation for all
```

### Mistake 3: Removing Observations Selectively

```python
# WRONG (if dropping relates to treatment/outcome)
df_clean = df[df['compliance'] > 0.5]  # Compliance is post-treatment
effect = estimate(df_clean, treatment, outcome, controls)  # BIASED

# CORRECT
# Either analyze full sample or use methods for non-compliance
# (IV with random assignment, bounds analysis, etc.)
```

### Mistake 4: Feature Selection Using Outcome

```python
# WRONG
from sklearn.feature_selection import SelectKBest, f_regression

# Selecting features based on outcome correlation
selector = SelectKBest(f_regression, k=10)
X_selected = selector.fit_transform(df[controls], df[outcome])  # Uses outcome!

# CORRECT
# Select based on theory, not data-driven outcome association
# Or use cross-fitting to avoid overfitting
```

---

## Summary

Causal inference requires careful variable selection:

| Principle | Rationale |
|-----------|-----------|
| Pre-treatment only | Post-treatment variables can be mediators or colliders |
| Block backdoor paths | Control for common causes of D and Y |
| Don't block causal paths | Avoid mediators |
| Don't open collider paths | Avoid colliders |
| Maintain overlap | Both groups need representation at all X values |
| Apply uniformly | Same preprocessing for treatment and control |

**When in doubt**: Draw a causal diagram, consult domain experts, and run sensitivity analyses.
