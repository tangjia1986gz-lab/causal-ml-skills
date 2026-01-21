# Common PSM Errors

## Error 1: Including Post-Treatment Variables

**Problem:**
Conditioning on variables affected by treatment introduces bias.

**Wrong:**
```python
# If 'job_satisfaction' occurs AFTER treatment
covariates = ['age', 'education', 'job_satisfaction']  # BAD
ps = LogisticRegression().fit(X[covariates], T)
```

**Correct:**
```python
# Only use PRE-treatment variables
covariates = ['age', 'education', 'prior_job']  # GOOD
```

**Rule:** Only include variables measured BEFORE treatment assignment.

## Error 2: Ignoring Overlap Violations

**Problem:**
Lack of common support leads to extrapolation and extreme weights.

**Wrong:**
```python
# Proceeding with extreme propensity scores
df['ps'] = model.predict_proba(X)[:, 1]
# Some ps near 0 or 1
ate = ipw_estimate(df)  # Unstable estimate
```

**Correct:**
```python
# Trim extreme values
df['ps'] = np.clip(df['ps'], 0.01, 0.99)

# Or check and report
if (df['ps'] < 0.05).any() or (df['ps'] > 0.95).any():
    print("WARNING: Limited overlap detected")
    # Consider trimming or changing estimand to ATT
```

## Error 3: Not Checking Balance

**Problem:**
Assuming propensity score adjustment worked without verification.

**Wrong:**
```python
# Match and estimate without checking
treated, control, _ = nearest_neighbor_matching(df, 'T', 'ps')
att = treated['Y'].mean() - control['Y'].mean()
print(f"ATT: {att}")  # No balance check!
```

**Correct:**
```python
# Always check balance
balance = balance_table(df, 'T', covariates, matched_df=matched_df)

# Verify all SMD < 0.1
if (balance['SMD_adj'].abs() > 0.1).any():
    print("WARNING: Imbalance remains!")
    print(balance[balance['SMD_adj'].abs() > 0.1])
    # Re-specify PS model
```

## Error 4: Wrong Estimand

**Problem:**
Using ATT weights when ATE is desired (or vice versa).

**Wrong:**
```python
# Claiming ATE but using ATT weights
w = np.where(T == 1, 1, ps / (1 - ps))  # These are ATT weights
ate = weighted_difference(Y, T, w)
print(f"ATE: {ate}")  # Actually ATT!
```

**Correct:**
```python
# Match estimand to weights
# For ATE:
w_ate = np.where(T == 1, 1/ps, 1/(1-ps))

# For ATT:
w_att = np.where(T == 1, 1, ps/(1-ps))

# Report correctly
print(f"ATT: {ipw_estimate(w_att)}")
print(f"ATE: {ipw_estimate(w_ate)}")
```

## Error 5: Overfitting Propensity Score

**Problem:**
Complex PS model achieves perfect prediction but poor generalization.

**Wrong:**
```python
# Overly complex model
from sklearn.ensemble import RandomForestClassifier
ps_model = RandomForestClassifier(n_estimators=1000, max_depth=None)
ps = ps_model.fit(X, T).predict_proba(X)[:, 1]
# AUC = 0.99 on training data!
```

**Correct:**
```python
# Use cross-validation
from sklearn.model_selection import cross_val_predict

ps = cross_val_predict(ps_model, X, T, cv=5, method='predict_proba')[:, 1]

# Or use regularization
ps_model = LogisticRegression(C=1.0, max_iter=1000)  # L2 regularization
```

## Error 6: King-Nielsen Critique

**Problem:**
PSM can INCREASE imbalance and model dependence (King & Nielsen, 2019).

**Wrong:**
```python
# Blindly trusting PSM
matched_treated, matched_control = psm_match(df, ps)
# Not checking if balance improved
```

**Correct:**
```python
# Check balance BEFORE and AFTER
balance_before = smd(treated_X, control_X)
balance_after = smd(matched_treated_X, matched_control_X)

if balance_after > balance_before:
    print("WARNING: PSM increased imbalance!")
    # Consider CEM or Mahalanobis matching instead
```

**Better alternatives:**
- Coarsened Exact Matching (CEM)
- Mahalanobis distance matching
- Entropy balancing

## Error 7: Using PS as Regression Covariate

**Problem:**
Including propensity score in outcome regression is redundant and can bias.

**Wrong:**
```python
# Adding PS to outcome model
model = sm.OLS(Y, sm.add_constant(df[['T', 'pscore'] + covariates]))
result = model.fit()
```

**Correct:**
```python
# Either stratify/match on PS...
# ...OR control for covariates directly
# But not both PS and covariates

# Correct: Stratify on PS
for stratum in ps_strata:
    stratum_effect = estimate_effect_in_stratum(stratum)

# Or: Just use covariates
model = sm.OLS(Y, sm.add_constant(df[['T'] + covariates]))
```

## Error 8: Not Conducting Sensitivity Analysis

**Problem:**
Claiming causal effect without testing robustness to unmeasured confounding.

**Wrong:**
```python
att = estimate_att_psm(...)
print(f"Job training CAUSES {att:.2f} increase in earnings")  # Too strong
```

**Correct:**
```python
att = estimate_att_psm(...)
bounds = rosenbaum_bounds(treated_Y, control_Y)

print(f"ATT estimate: {att:.4f}")
print(f"Effect robust to Gamma = {max_robust_gamma}")
print("Caveat: Assumes selection on observables")
```

## Error 9: Ignoring Clustering

**Problem:**
Not accounting for clustered data (e.g., patients within hospitals).

**Wrong:**
```python
# Standard errors assume iid
se = bootstrap_se(df)  # Wrong if clustered
```

**Correct:**
```python
# Cluster bootstrap
def cluster_bootstrap(df, cluster_var, n_boot=1000):
    clusters = df[cluster_var].unique()
    boot_estimates = []
    for _ in range(n_boot):
        sampled_clusters = np.random.choice(clusters, size=len(clusters), replace=True)
        boot_df = df[df[cluster_var].isin(sampled_clusters)]
        boot_estimates.append(estimate_effect(boot_df))
    return np.std(boot_estimates)
```

## Error 10: Poor Reporting

**Problem:**
Not providing enough information for readers to assess validity.

**Wrong:**
```
"We used propensity score matching. The ATT was 2.5 (p < 0.05)."
```

**Correct:**
```
"We estimated propensity scores using logistic regression with the following
covariates: age, education, income, and marital status. After 1:1 nearest-neighbor
matching with a 0.2 SD caliper, we retained 450 matched pairs (85% of treated).
Standardized mean differences were below 0.1 for all covariates (see Table 1).
The ATT was 2.5 (95% CI: 1.2-3.8). Sensitivity analysis indicates the finding
is robust to unmeasured confounding up to Gamma = 1.8."
```

## Quick Checklist

Before finalizing PSM analysis:

- [ ] Only pre-treatment covariates used
- [ ] Overlap checked and addressed
- [ ] Balance verified (SMD < 0.1)
- [ ] Correct estimand identified
- [ ] Cross-validation or regularization used for PS
- [ ] Balance improved (not worsened) by matching
- [ ] PS not included in outcome model
- [ ] Sensitivity analysis conducted
- [ ] Clustering accounted for (if applicable)
- [ ] Full details reported

## Key References

1. King & Nielsen (2019): "Why Propensity Scores Should Not Be Used for Matching"
2. Stuart (2010): "Matching methods for causal inference: A review"
3. Austin (2011): "An Introduction to Propensity Score Methods for Reducing Confounding"
4. Rubin (2001): "Using Propensity Scores to Help Design Observational Studies"
