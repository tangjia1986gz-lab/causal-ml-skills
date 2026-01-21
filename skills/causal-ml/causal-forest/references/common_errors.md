# Common Errors in Causal Forest Analysis

> **Reference Document** | Causal Forest / GRF
> Based on Wager & Athey (2018), Athey et al. (2019), grf documentation

## Overview

This document catalogs common errors in causal forest implementation, their consequences, and recommended corrections.

---

## 1. Confusing Causal Forests with Predictive Random Forests

### 1.1 The Error

Using standard random forests or treating causal forests as prediction models.

### 1.2 Why It's Wrong

| Random Forest | Causal Forest |
|---------------|---------------|
| Predicts Y | Predicts tau(x) = E[Y(1) - Y(0) \| X] |
| Minimizes prediction error | Maximizes heterogeneity in treatment effects |
| Splitting: variance reduction | Splitting: treatment effect heterogeneity |
| No causal interpretation | Estimates individual causal effects |

### 1.3 Consequences

- **Invalid causal claims** from predictive models
- **Biased treatment effect estimates**
- **No valid confidence intervals** for causal effects

### 1.4 Correction

```python
# WRONG: Using RandomForestRegressor for causal effects
from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor()
rf.fit(X, Y)  # This predicts Y, NOT treatment effects!

# CORRECT: Using CausalForest for treatment effect estimation
from econml.grf import CausalForest
cf = CausalForest(n_estimators=2000, honest=True)
cf.fit(Y, T, X=X)
cate = cf.predict(X_test)  # This estimates E[Y(1)-Y(0)|X]
```

**Reference**: Wager, S., & Athey, S. (2018). Estimation and Inference of Heterogeneous Treatment Effects using Random Forests. *JASA*, 113(523), 1228-1242.

---

## 2. Using Dishonest Estimation for Inference

### 2.1 The Error

Setting `honesty=False` to get "better" predictions, then reporting confidence intervals.

### 2.2 Why It's Wrong

Without honesty:
- Same data used for tree structure AND effect estimation
- **Confidence intervals are invalid** (severe undercoverage)
- **Overfitting to noise** in treatment effects
- Cannot trust p-values or statistical tests

### 2.3 The Honesty Principle

```
Honest Estimation:
├── Splitting Sample: Determines tree structure (where to split)
└── Estimation Sample: Estimates effects in leaves (what values)

By separating these, we avoid overfitting and get valid inference.
```

### 2.4 Correction

```python
# WRONG: Dishonest forest with inference claims
cf = CausalForest(honest=False)  # Invalid for inference!
cf.fit(Y, T, X=X)
cate, ci = cf.predict_and_var(X_test)  # CIs are WRONG

# CORRECT: Honest forest for valid inference
cf = CausalForest(
    honest=True,              # REQUIRED for valid CIs
    honesty_fraction=0.5      # Half for splitting, half for estimation
)
cf.fit(Y, T, X=X)
cate, ci = cf.predict_and_var(X_test)  # CIs are valid
```

**Reference**: Athey, S., Tibshirani, J., & Wager, S. (2019). Generalized Random Forests. *Annals of Statistics*, 47(2), 1148-1178.

---

## 3. Ignoring Confidence Intervals

### 3.1 The Error

Reporting point estimates of CATE without uncertainty quantification.

### 3.2 Why It's Wrong

Individual CATE estimates can be:
- **Highly variable** even with large samples
- **Misleading** without context on precision
- **Not actionable** if CIs include zero

### 3.3 Example of the Problem

```
BAD REPORTING:
"Customer A has CATE = $50, Customer B has CATE = $45"
→ Makes it seem we know effects precisely

GOOD REPORTING:
"Customer A: CATE = $50 (95% CI: $-10 to $110)
 Customer B: CATE = $45 (95% CI: $20 to $70)"
→ Shows Customer B's effect is more certain
```

### 3.4 Correction

```python
# ALWAYS report uncertainty
cate_pred = cf.predict(X_test)
cate_var = cf.predict_var(X_test)
cate_se = np.sqrt(cate_var)

# 95% confidence intervals
ci_lower = cate_pred - 1.96 * cate_se
ci_upper = cate_pred + 1.96 * cate_se

# Report with uncertainty
results = pd.DataFrame({
    'cate': cate_pred,
    'se': cate_se,
    'ci_lower': ci_lower,
    'ci_upper': ci_upper,
    'significant': (ci_lower > 0) | (ci_upper < 0)
})
```

---

## 4. Small Sample Heterogeneity Claims

### 4.1 The Error

Claiming significant treatment effect heterogeneity with small samples (N < 1000).

### 4.2 Why It's Wrong

Causal forests require substantial data because:
- Honest splitting effectively halves the sample
- Heterogeneity detection needs many observations per subgroup
- Individual CATEs have high variance with small N

### 4.3 Sample Size Guidelines

| Sample Size | Capability |
|-------------|------------|
| N < 500 | Use simpler methods; heterogeneity claims unreliable |
| 500-2000 | Can estimate ATE well; heterogeneity detection limited |
| 2000-10000 | Good heterogeneity detection; reliable subgroup analysis |
| N > 10000 | Fine-grained CATE estimation; individual-level targeting |

### 4.4 Correction

```python
# Check before running causal forest
n_obs = len(data)
n_treated = data[treatment].sum()
n_control = n_obs - n_treated

print(f"Sample size: {n_obs}")
print(f"Treated: {n_treated}, Control: {n_control}")

if n_obs < 1000:
    print("WARNING: Sample may be too small for reliable heterogeneity")
    print("Consider: IPW, regression adjustment, or DID for ATE")
elif min(n_treated, n_control) < 200:
    print("WARNING: Imbalanced treatment; consider IPW instead")
```

---

## 5. Over-interpreting Variable Importance

### 5.1 The Error

Treating variable importance as a definitive ranking of effect modifiers.

### 5.2 Why It's Wrong

Variable importance in causal forests:
- **Correlated variables share importance** (similar to random forests)
- **Does not indicate direction** of effect modification
- **May reflect splitting convenience**, not true importance
- **Can be unstable** across runs

### 5.3 Example

```
Suppose:
- Age and tenure are highly correlated (r = 0.8)
- Both modify treatment effect similarly

Variable importance might show:
  Age: 0.30
  Tenure: 0.15

But this doesn't mean age is "twice as important" - they're sharing.
```

### 5.4 Correction

```python
# Use BLP for quantitative interpretation
from grf import best_linear_projection

blp = best_linear_projection(cf, A=data[['age', 'tenure', 'income']])
print(blp.summary())

# Interpret BLP coefficients:
# "Each 10-year increase in age is associated with
#  a $X change in treatment effect (SE = Y, p = Z)"

# Also use partial dependence plots
from causal_forest import partial_dependence_plot
partial_dependence_plot(cf, X, feature='age')
```

---

## 6. Not Testing for Heterogeneity

### 6.1 The Error

Assuming heterogeneity exists without formal testing.

### 6.2 Why It's Wrong

Even when CATEs vary across the sample:
- Variation might be **noise, not signal**
- No heterogeneity means **uniform treatment is optimal**
- Resources wasted on targeting that doesn't help

### 6.3 Correction

```python
# R grf: Built-in calibration test
# calibration <- test_calibration(cf)

# Python: Heterogeneity test via BLP
from econml.grf import CausalForest

cf = CausalForest(n_estimators=2000)
cf.fit(Y, T, X=X)

# Test: Is the coefficient on predicted CATE significant?
from econml.tests import test_heterogeneity
het_test = test_heterogeneity(cf, X)
print(f"Heterogeneity test p-value: {het_test.pvalue:.4f}")

if het_test.pvalue > 0.05:
    print("No significant heterogeneity detected")
    print("Consider simpler methods for ATE estimation")
```

---

## 7. Treating CATE as Ground Truth for Individuals

### 7.1 The Error

Making individual treatment decisions based solely on CATE point estimates.

### 7.2 Why It's Wrong

CATE(x) estimates E[Y(1) - Y(0) | X = x], the **expected** effect for people with characteristics X, not the guaranteed effect for a specific individual.

### 7.3 What CATE Actually Represents

```
Individual i with X_i has CATE = $100

This means:
- Among people LIKE individual i (similar X values)
- The AVERAGE treatment effect is $100
- Individual i's actual effect could be much higher or lower
- Due to unmeasured heterogeneity
```

### 7.4 Correction

```python
# Use CATE for population-level decisions, not individual predictions
# Focus on: "Targeting this GROUP increases average outcomes"

# For policy:
# 1. Rank individuals by predicted CATE
# 2. Treat top X% where CATE is highest
# 3. Evaluate policy value, not individual outcomes

policy_value = cf.estimate_policy_value(
    X_test,
    treatment_budget=0.3  # Treat top 30%
)
print(f"Expected policy improvement: ${policy_value:.2f}")
```

---

## 8. Forgetting Overlap/Positivity

### 8.1 The Error

Running causal forests without checking treatment overlap across covariate space.

### 8.2 Why It's Wrong

If some covariate regions have:
- Only treated (no control comparisons)
- Only control (no treated examples)

Then CATE estimates extrapolate beyond the data.

### 8.3 Detection

```python
from sklearn.ensemble import GradientBoostingClassifier

# Estimate propensity score
ps_model = GradientBoostingClassifier()
ps_model.fit(X, T)
ps = ps_model.predict_proba(X)[:, 1]

# Check overlap
print(f"Propensity score range: [{ps.min():.3f}, {ps.max():.3f}]")
print(f"Treated with PS < 0.1: {(ps[T==1] < 0.1).mean():.1%}")
print(f"Control with PS > 0.9: {(ps[T==0] > 0.9).mean():.1%}")

# Trim if needed
overlap_mask = (ps > 0.1) & (ps < 0.9)
X_trimmed = X[overlap_mask]
```

### 8.4 Correction

Causal forests with grf automatically perform local centering, but extreme overlap violations still cause problems. Consider:
- Trimming extreme propensity scores
- Using inverse propensity weighting
- Reporting CATEs only in regions with overlap

---

## 9. Quick Reference: Error Prevention Checklist

```
CAUSAL FOREST ERROR PREVENTION CHECKLIST
========================================

BEFORE ANALYSIS
[ ] Sample size adequate (N > 1000 for heterogeneity)
[ ] Treatment and outcome clearly defined
[ ] Effect modifiers identified based on theory
[ ] Check overlap/positivity in covariate space

MODEL SPECIFICATION
[ ] Use honest=True for inference
[ ] Adequate number of trees (2000+)
[ ] Reasonable min_node_size for sample size
[ ] Include confounders in adjustment

VALIDATION
[ ] Formal heterogeneity test
[ ] Calibration assessment
[ ] Compare ATE to simpler methods
[ ] Check CATE distribution makes sense

INTERPRETATION
[ ] Report confidence intervals
[ ] Don't over-interpret variable importance
[ ] Use BLP for quantitative effects
[ ] Focus on group patterns, not individuals

POLICY
[ ] Evaluate policy value, not just CATEs
[ ] Consider treatment costs
[ ] Test on held-out data
[ ] Sensitivity analysis
```

---

## References

- Wager, S., & Athey, S. (2018). Estimation and Inference of Heterogeneous Treatment Effects using Random Forests. *Journal of the American Statistical Association*, 113(523), 1228-1242.
- Athey, S., Tibshirani, J., & Wager, S. (2019). Generalized Random Forests. *Annals of Statistics*, 47(2), 1148-1178.
- Athey, S., & Wager, S. (2021). Policy Learning with Observational Data. *Econometrica*, 89(1), 133-161.
- grf R package documentation: https://grf-labs.github.io/grf/
- econml Python package documentation: https://econml.azurewebsites.net/
