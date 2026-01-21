# Common PSM Errors and Pitfalls

> **Reference Document** | Propensity Score Matching
> Based on King & Nielsen (2019), Imbens (2015)

## Overview

This document catalogs common errors in PSM implementation, their consequences, and recommended corrections.

---

## 1. Including Post-Treatment Variables

### 1.1 The Error

Including variables measured AFTER treatment in the propensity score model.

### 1.2 Why It's Wrong

Post-treatment variables can be:
- **Mediators**: On the causal path from treatment to outcome
- **Colliders**: Affected by both treatment and outcome
- **Consequences of treatment**: Partially determined by treatment status

Including them violates the structure of causal inference and can introduce or amplify bias.

### 1.3 Causal Graph

```
CORRECT (pre-treatment confounder):

    X (pre-treatment)
   /  \
  v    v
 D --> Y

INCORRECT (post-treatment mediator):

 D --> M --> Y
       ^
       |
    Conditioning on M blocks the causal path!

INCORRECT (collider):

 D --> C <-- U --> Y
       ^
       |
    Conditioning on C opens a backdoor path!
```

### 1.4 Example

**Job Training Study**:
- Treatment: Job training program (year 1)
- Outcome: Earnings (year 3)
- **BAD**: Including "employed" in year 2 (mediator)
- **GOOD**: Including "employed" before year 1

### 1.5 How to Detect

1. **Timeline Analysis**: Map when each variable was measured
2. **Causal Graph**: Draw DAG and check variable timing
3. **Domain Knowledge**: Could treatment affect this variable?

### 1.6 Correction

**Rule**: Only include variables that:
- Were measured BEFORE treatment assignment
- Cannot be affected by treatment
- Predict treatment (selection) or outcome (confounding)

```python
# WRONG
covariates = ['age', 'education', 'post_treatment_job_status']

# CORRECT
covariates = ['age_at_baseline', 'education_at_baseline',
              'pre_treatment_employment']
```

---

## 2. Bad Propensity Score Model Specification

### 2.1 The Error

Incorrectly specifying the propensity score model through:
- Missing important confounders
- Overfitting with too many variables
- Wrong functional form
- Not including interactions

### 2.2 Types of Misspecification

| Type | Description | Consequence |
|------|-------------|-------------|
| **Omission** | Missing key confounders | Bias from uncontrolled confounding |
| **Overfitting** | Too many/collinear variables | Extreme PS values, poor overlap |
| **Functional Form** | Linear when nonlinear needed | Poor balance |
| **No Interactions** | Missing X1*X2 terms | Inadequate covariate control |

### 2.3 Signs of Misspecification

1. **Poor balance after matching** (SMD > 0.1)
2. **PS near 0 or 1** for many observations
3. **PS model AUC > 0.9** (possible overfitting or deterministic selection)
4. **Large imbalance on interactions** even with marginal balance

### 2.4 Corrections

**For Missing Variables**:
```python
# Think through potential confounders systematically
confounders = []

# 1. Demographic factors
confounders += ['age', 'sex', 'race', 'education']

# 2. Socioeconomic factors
confounders += ['income', 'employment', 'wealth']

# 3. Health factors (if relevant)
confounders += ['baseline_health', 'chronic_conditions']

# 4. Geographic factors
confounders += ['region', 'urban_rural']

# 5. Prior outcomes
confounders += ['lagged_outcome']
```

**For Functional Form**:
```python
# Option 1: Add polynomial terms
from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X)

# Option 2: Use flexible ML methods
from sklearn.ensemble import GradientBoostingClassifier
ps_model = GradientBoostingClassifier(n_estimators=100, max_depth=3)

# Option 3: Add key interactions manually
X['age_x_income'] = X['age'] * X['income']
X['education_x_employed'] = X['education'] * X['employed']
```

---

## 3. Ignoring Common Support Violations

### 3.1 The Error

Proceeding with analysis when treated and control groups don't overlap in propensity score distribution.

### 3.2 Why It's Wrong

Without common support:
- Matching requires **extrapolation**
- Results become **model-dependent**
- Estimates apply to **different populations** than intended

### 3.3 Visual Example

```
POOR OVERLAP:

Control:  ******.................
Treated:  ...............*******
          |----|----|----|----|
          0   0.25  0.5  0.75  1

          [No common support region!]

GOOD OVERLAP:

Control:  ***********..........
Treated:  .....*************
          |----|----|----|----|
          0   0.25  0.5  0.75  1

          [Common support: 0.25 to 0.5]
```

### 3.4 Detection

```python
def check_overlap(ps_treated, ps_control, threshold=0.1):
    """
    Check common support assumption.

    Returns warning level:
    - 'good': Strong overlap
    - 'moderate': Some concerns
    - 'poor': Serious violations
    """
    min_t, max_t = ps_treated.min(), ps_treated.max()
    min_c, max_c = ps_control.min(), ps_control.max()

    # Common support bounds
    lower = max(min_t, min_c)
    upper = min(max_t, max_c)

    # Check overlap
    if upper <= lower:
        return 'poor'

    # Percentage in common support
    pct_t = np.mean((ps_treated >= lower) & (ps_treated <= upper))
    pct_c = np.mean((ps_control >= lower) & (ps_control <= upper))

    if pct_t > 0.9 and pct_c > 0.9:
        return 'good'
    elif pct_t > 0.7 and pct_c > 0.7:
        return 'moderate'
    else:
        return 'poor'
```

### 3.5 Corrections

| Approach | Implementation | Trade-off |
|----------|---------------|-----------|
| **Trimming** | Drop PS < 0.1 or > 0.9 | Changes estimand |
| **Caliper** | Only match within radius | May lose treated |
| **Bounds** | Report range of estimates | Wider uncertainty |
| **Alternative Estimand** | Estimate ATT for common support only | Limited generalizability |

```python
# Trimming approach
def trim_to_common_support(data, ps_col, treatment_col, method='minmax'):
    """Trim data to common support region."""
    ps_t = data.loc[data[treatment_col] == 1, ps_col]
    ps_c = data.loc[data[treatment_col] == 0, ps_col]

    if method == 'minmax':
        lower = max(ps_t.min(), ps_c.min())
        upper = min(ps_t.max(), ps_c.max())
    elif method == 'percentile':
        lower = max(np.percentile(ps_t, 5), np.percentile(ps_c, 5))
        upper = min(np.percentile(ps_t, 95), np.percentile(ps_c, 95))

    mask = (data[ps_col] >= lower) & (data[ps_col] <= upper)

    n_dropped = len(data) - mask.sum()
    print(f"Trimmed {n_dropped} observations ({n_dropped/len(data)*100:.1f}%)")

    return data[mask].copy()
```

---

## 4. Not Checking Balance After Matching

### 4.1 The Error

Assuming matching automatically produces balance without verification.

### 4.2 Why It's Wrong

- Matching on propensity scores doesn't guarantee balance on individual covariates
- Poor PS model can fail to achieve balance
- Different matching methods achieve different balance
- Some covariates may remain imbalanced

### 4.3 Consequences

If balance is not achieved:
- Residual confounding remains
- Treatment effect estimates are biased
- Results are not credible

### 4.4 Correction

**Always check and report balance**:

```python
def verify_balance(data_matched, treatment, covariates, weights=None, threshold=0.1):
    """
    Verify balance after matching and raise warnings.

    Parameters
    ----------
    data_matched : pd.DataFrame
        Matched dataset
    treatment : str
        Treatment column
    covariates : list
        Covariates to check
    weights : str, optional
        Weights column
    threshold : float
        SMD threshold for balance

    Returns
    -------
    dict
        Balance statistics and warnings
    """
    results = {'balanced': True, 'warnings': [], 'smd': {}}

    for cov in covariates:
        smd = calculate_smd(
            data_matched[data_matched[treatment] == 1][cov],
            data_matched[data_matched[treatment] == 0][cov]
        )
        results['smd'][cov] = smd

        if abs(smd) > threshold:
            results['balanced'] = False
            results['warnings'].append(
                f"WARNING: {cov} has SMD = {smd:.3f} > {threshold}"
            )

    if not results['balanced']:
        print("BALANCE NOT ACHIEVED!")
        print("Consider:")
        print("1. Re-specifying propensity score model")
        print("2. Adding interaction terms")
        print("3. Using different matching method")
        print("4. Exact matching on problem covariates")
        for w in results['warnings']:
            print(f"  {w}")

    return results
```

---

## 5. Inappropriate Trimming

### 5.1 The Error

Trimming propensity scores in ways that:
- Are too aggressive (lose too much data)
- Are too conservative (don't address overlap issues)
- Change the estimand without acknowledgment
- Are applied asymmetrically

### 5.2 Types of Trimming Errors

| Error | Description | Consequence |
|-------|-------------|-------------|
| **Over-trimming** | PS in [0.2, 0.8] | Loses many observations, selection bias |
| **Under-trimming** | PS in [0.01, 0.99] | Doesn't address extreme weights |
| **Asymmetric** | Different rules for T/C | Changes estimand incorrectly |
| **Post-hoc** | Trimming based on results | Data dredging, p-hacking |

### 5.3 Best Practices

1. **Pre-specify trimming rules** before seeing results
2. **Use data-driven but principled approaches**:
   - Crump et al. (2009): Trim where PS < 0.1 or > 0.9
   - Optimal trimming: Minimize variance subject to overlap

3. **Report sensitivity to trimming choice**

```python
def sensitivity_to_trimming(data, outcome, treatment, ps_col,
                           trim_values=[0.01, 0.05, 0.10, 0.15]):
    """
    Check sensitivity of results to trimming choice.

    Returns effects under different trimming rules.
    """
    results = []

    for trim in trim_values:
        # Trim data
        mask = (data[ps_col] >= trim) & (data[ps_col] <= 1 - trim)
        data_trim = data[mask].copy()

        # Estimate effect
        effect, se = estimate_effect(data_trim, outcome, treatment, ps_col)

        results.append({
            'trim': trim,
            'n': len(data_trim),
            'pct_kept': len(data_trim) / len(data) * 100,
            'effect': effect,
            'se': se
        })

    return pd.DataFrame(results)
```

---

## 6. Misinterpreting Propensity Score

### 6.1 The Error

Believing that:
- High PS model AUC = good PSM study
- Propensity score is the probability of being treated
- Matching on PS removes all confounding
- PS balances unobserved confounders too

### 6.2 Clarifications

| Misconception | Reality |
|---------------|---------|
| High AUC is good | High AUC may indicate overlap problems |
| PS = true treatment probability | PS is an estimate, may be misspecified |
| PSM removes all confounding | Only balances on included covariates |
| PS handles unobservables | PS cannot address unobserved confounding |

### 6.3 Correct Understanding

- **PS is a dimension-reduction tool**, not a silver bullet
- **Balance on PS** implies balance on covariates in PS model
- **Unconfoundedness** is an assumption, not a result of matching
- **High AUC** may indicate perfect selection (bad for causal inference)

---

## 7. Ignoring Standard Error Issues

### 7.1 The Error

Using naive standard errors that ignore:
- Matching uncertainty
- Propensity score estimation
- Clustering in data
- Repeated use of controls (with replacement)

### 7.2 Why It's Wrong

Naive SEs typically understate uncertainty because:
- PS is estimated, not known
- Matching introduces correlation
- Controls used multiple times (with replacement)

### 7.3 Correct Approaches

| Method | When to Use |
|--------|-------------|
| **Bootstrap** | General purpose, accounts for PS estimation |
| **Abadie-Imbens SE** | Matching estimators |
| **Cluster-robust SE** | Clustered data |
| **Doubly robust SE** | AIPW estimators |

```python
def bootstrap_se_matching(data, outcome, treatment, covariates,
                         n_boot=1000, seed=42):
    """
    Bootstrap SE that accounts for PS estimation and matching.
    """
    np.random.seed(seed)
    estimates = []

    for b in range(n_boot):
        # Resample data
        boot_idx = np.random.choice(len(data), len(data), replace=True)
        boot_data = data.iloc[boot_idx].reset_index(drop=True)

        # Re-estimate PS on bootstrap sample
        ps = estimate_propensity_score(boot_data, treatment, covariates)
        boot_data['_ps'] = ps

        # Re-match on bootstrap sample
        matched = nearest_neighbor_match(boot_data, '_ps', treatment)

        # Estimate effect on bootstrap matched sample
        effect = estimate_att(matched, outcome, treatment)
        estimates.append(effect)

    return np.std(estimates)
```

---

## 8. Wrong Estimand

### 8.1 The Error

Estimating the wrong treatment effect (ATE vs ATT vs ATU) or not being clear about which estimand is targeted.

### 8.2 Estimand Differences

| Estimand | Definition | What PSM Estimates |
|----------|------------|-------------------|
| **ATT** | E[Y1-Y0\|D=1] | Effect on treated |
| **ATE** | E[Y1-Y0] | Effect on everyone |
| **ATU** | E[Y1-Y0\|D=0] | Effect on untreated |

### 8.3 Why It Matters

- Standard PSM estimates **ATT**, not ATE
- ATT and ATE differ when effects are heterogeneous
- Policies may require different estimands

### 8.4 Correction

**Know your estimand and match accordingly**:

```python
# For ATT (most common in PSM)
# Match controls to treated
weights_att = calculate_att_weights(ps, treatment)

# For ATE
# Need reweighting or bidirectional matching
weights_ate = calculate_ate_weights(ps, treatment)

# For ATU
# Match treated to controls (reverse matching)
weights_atu = calculate_atu_weights(ps, treatment)
```

---

## 9. Quick Reference: Error Prevention Checklist

```
PSM ERROR PREVENTION CHECKLIST
==============================

DATA PREPARATION
[ ] All covariates measured PRE-treatment
[ ] No post-treatment variables in PS model
[ ] Timeline documented for all variables

PS MODEL
[ ] Include all known confounders
[ ] Check functional form (interactions, nonlinearities)
[ ] Don't overfit (monitor AUC, balance)

OVERLAP
[ ] Plot PS distributions before matching
[ ] Check common support region
[ ] Plan trimming rule in advance

MATCHING
[ ] Choose method appropriate for data
[ ] Specify caliper if using NN
[ ] Document matching parameters

BALANCE
[ ] Check SMD for ALL covariates
[ ] Check variance ratios
[ ] Create Love plot
[ ] If balance fails, re-specify PS model

INFERENCE
[ ] Use appropriate SE method
[ ] Report correct confidence intervals
[ ] Conduct sensitivity analysis

REPORTING
[ ] State estimand clearly (ATT/ATE)
[ ] Report all trimming/exclusions
[ ] Acknowledge limitations
[ ] Discuss unobserved confounding
```

---

## References

- King, G., & Nielsen, R. (2019). Why Propensity Scores Should Not Be Used for Matching. *Political Analysis*, 27(4), 435-454.
- Imbens, G. W. (2015). Matching Methods in Practice: Three Examples. *Journal of Human Resources*, 50(2), 373-419.
- Crump, R. K., Hotz, V. J., Imbens, G. W., & Mitnik, O. A. (2009). Dealing with Limited Overlap in Estimation of Average Treatment Effects. *Biometrika*, 96(1), 187-199.
- Austin, P. C. (2011). Optimal Caliper Widths for Propensity-Score Matching. *American Journal of Epidemiology*, 173(9), 1168-1175.
