---
name: causal-forest
triggers:
  - causal forest
  - GRF
  - generalized random forest
  - CATE
  - heterogeneous treatment effects
  - conditional average treatment effect
  - policy learning
  - treatment effect heterogeneity
  - personalized treatment effects
  - individualized treatment effects
---

# Causal Forest / Generalized Random Forest

## Overview

Causal forests estimate **heterogeneous treatment effects** - how treatment effects vary across individuals based on their characteristics. Unlike methods that estimate a single average treatment effect (ATE), causal forests provide **Conditional Average Treatment Effects (CATE)**: the expected treatment effect for individuals with specific covariate values.

**CATE Definition**:
```
tau(x) = E[Y(1) - Y(0) | X = x]
```

Where:
- `Y(1)` is the potential outcome under treatment
- `Y(0)` is the potential outcome under control
- `X` are the covariates (effect modifiers)
- `tau(x)` is the treatment effect for individuals with characteristics `X = x`

## When to Use Causal Forests

**Ideal scenarios**:
- Detecting **treatment effect heterogeneity** across subgroups
- Developing **personalized treatment strategies** (who benefits most?)
- **Policy targeting**: identifying optimal treatment allocation rules
- Exploratory analysis of effect modification when many potential modifiers exist
- When you have moderate to large sample sizes (N > 1000 typically)
- RCT data or observational data with valid adjustment strategy

**Example applications**:
- Which customers respond best to a marketing intervention?
- Who benefits most from a medical treatment?
- Where should we target a policy intervention for maximum impact?
- Personalized pricing: optimal discount for each customer segment

## When NOT to Use Causal Forests

**Avoid when**:
- Sample size is small (N < 500) - insufficient for honest splitting
- You only need the **Average Treatment Effect** (use simpler methods)
- Treatment effect is known to be constant across individuals
- You need parametric confidence intervals (consider DML instead)
- Interpretability is paramount (consider policy trees or linear CATE models)
- High-dimensional treatments (forests handle binary/continuous treatment best)

## Key Concepts

### 1. Honest Estimation

Causal forests use **honesty** to provide valid inference:

```
Training Data Split:
├── Splitting Sample: Used to determine tree structure (where to split)
└── Estimation Sample: Used to estimate effects in leaves (what values)
```

**Why honesty matters**:
- Prevents overfitting to noise in treatment effects
- Enables valid confidence intervals
- Trades some efficiency for unbiased estimation

### 2. Local Centering (Orthogonalization)

Causal forests apply **local centering** to remove confounding:

```
Centered Outcome: Y - E[Y|X]
Centered Treatment: W - E[W|X]
```

This orthogonalization:
- Reduces bias from confounding
- Improves efficiency
- Is done via local linear forests or separate ML models

### 3. Weighted Nearest Neighbors Interpretation

Each causal forest prediction is a weighted average:

```
tau_hat(x) = sum_i alpha_i(x) * (Y_i^treated - Y_i^control)
```

Where weights `alpha_i(x)` are determined by how often observation `i` falls in the same leaf as `x` across all trees.

## Implementation Workflow

### Step 1: Setup - Define Components

```python
from causal_forest import fit_causal_forest, CausalForestConfig

# Identify your variables
outcome = 'revenue'           # Y: what we're trying to affect
treatment = 'discount'        # W: binary or continuous treatment
effect_modifiers = [          # X: variables that may modify treatment effect
    'customer_age',
    'tenure_months',
    'past_purchases',
    'segment'
]
confounders = [               # Additional adjustment variables
    'region',
    'signup_channel'
]

# Configuration
config = CausalForestConfig(
    n_estimators=2000,        # Number of trees
    honesty=True,             # Use honest splitting
    honesty_fraction=0.5,     # Fraction for estimation (vs splitting)
    min_node_size=5,          # Minimum observations per leaf
    mtry=None,                # Variables to try at each split (default: sqrt(p))
    sample_fraction=0.5       # Subsampling fraction per tree
)
```

### Step 2: Training - Fit Causal Forest

```python
# Fit the causal forest
cf_model = fit_causal_forest(
    X=data[effect_modifiers],
    y=data[outcome],
    treatment=data[treatment],
    X_adjust=data[confounders],  # Additional confounders
    config=config
)

print(f"Training complete. OOB R-squared: {cf_model.oob_score_:.3f}")
```

### Step 3: CATE Estimation - Individual Treatment Effects

```python
from causal_forest import estimate_cate

# Estimate treatment effects for each individual
cate_results = estimate_cate(cf_model, X_test=data[effect_modifiers])

# Results include:
# - cate_results.estimates: Point estimates of tau(x)
# - cate_results.std_errors: Standard errors
# - cate_results.ci_lower: Lower confidence bound
# - cate_results.ci_upper: Upper confidence bound

# Summary statistics
print(f"Average CATE: {cate_results.estimates.mean():.3f}")
print(f"CATE Range: [{cate_results.estimates.min():.3f}, {cate_results.estimates.max():.3f}]")
print(f"Proportion with positive effect: {(cate_results.estimates > 0).mean():.1%}")
```

### Step 4: Variable Importance - Heterogeneity Drivers

```python
from causal_forest import variable_importance, plot_variable_importance

# Extract variable importance
importance = variable_importance(cf_model)

# Plot importance scores
plot_variable_importance(
    importance_scores=importance.scores,
    feature_names=effect_modifiers,
    title="Drivers of Treatment Effect Heterogeneity"
)

# Top drivers
for name, score in importance.ranked[:5]:
    print(f"{name}: {score:.3f}")
```

### Step 5: Best Linear Projection (BLP)

The BLP summarizes heterogeneity as a linear function of covariates:

```python
from causal_forest import best_linear_projection

# Project CATE onto linear function of selected variables
blp_result = best_linear_projection(
    cf_model,
    A=data[['customer_age', 'tenure_months']]  # Variables to project onto
)

# Results
print("Best Linear Projection of CATE:")
print(blp_result.summary())

# Interpretation: CATE ≈ beta_0 + beta_1 * age + beta_2 * tenure
# Coefficients show how CATE varies with each variable
```

### Step 6: Heterogeneity Testing

```python
from causal_forest import heterogeneity_test

# Test whether heterogeneity is statistically significant
het_test = heterogeneity_test(cf_model)

print(f"Heterogeneity Test:")
print(f"  Chi-squared statistic: {het_test.statistic:.2f}")
print(f"  p-value: {het_test.pvalue:.4f}")
print(f"  Significant heterogeneity: {het_test.pvalue < 0.05}")
```

### Step 7: Policy Learning - Optimal Treatment Rules

```python
from causal_forest import policy_learning, PolicyConfig

# Learn optimal treatment policy
policy_config = PolicyConfig(
    treatment_cost=10,           # Cost of treating one person
    budget_fraction=0.3,         # Can only treat 30% of population
    method='policy_tree'         # 'threshold', 'policy_tree', or 'optimal'
)

policy = policy_learning(
    cf_model,
    X=data[effect_modifiers],
    config=policy_config
)

# Evaluate policy
print(f"Policy Value: {policy.value:.2f}")
print(f"Proportion Treated: {policy.treatment_rate:.1%}")
print(f"Improvement over treat-all: {policy.improvement:.1%}")

# Get treatment recommendations
recommendations = policy.recommend(new_data[effect_modifiers])
```

## R grf Package vs Python econml

### R grf (Gold Standard)

```r
library(grf)

# Fit causal forest
cf <- causal_forest(
  X = X,
  Y = Y,
  W = W,
  num.trees = 2000,
  honesty = TRUE,
  honesty.fraction = 0.5,
  min.node.size = 5
)

# Predict CATE
cate <- predict(cf, X.test, estimate.variance = TRUE)

# Variable importance
var_imp <- variable_importance(cf)

# Best linear projection
blp <- best_linear_projection(cf, A)

# Calibration test
cal_test <- test_calibration(cf)
```

**R grf advantages**:
- More mature implementation
- Better calibrated confidence intervals
- Calibration tests built-in
- Policy tree package integration

### Python econml

```python
from econml.grf import CausalForest
from econml.dml import CausalForestDML

# Basic CausalForest
cf = CausalForest(
    n_estimators=2000,
    min_samples_leaf=5,
    honest=True
)
cf.fit(Y, T, X=X)
cate = cf.predict(X_test)

# CausalForestDML (with flexible nuisance models)
from sklearn.ensemble import GradientBoostingRegressor

cf_dml = CausalForestDML(
    model_y=GradientBoostingRegressor(),
    model_t=GradientBoostingRegressor(),
    n_estimators=2000
)
cf_dml.fit(Y, T, X=X, W=W)  # W = additional controls
```

**Python econml advantages**:
- Integrates with scikit-learn ecosystem
- CausalForestDML allows flexible nuisance estimation
- Easier deployment in Python pipelines

## Tuning Guidance

### Critical Parameters

| Parameter | Default | Guidance |
|-----------|---------|----------|
| `n_estimators` | 2000 | More is generally better; 2000-4000 typical |
| `honesty` | True | **Always use True** for valid inference |
| `honesty_fraction` | 0.5 | Balance between splitting and estimation |
| `min_node_size` | 5 | Increase for smoother estimates; decrease for more heterogeneity |
| `mtry` | sqrt(p) | Can tune; higher values more exhaustive but slower |
| `sample_fraction` | 0.5 | Subsampling per tree |

### Tuning Strategy

```python
from causal_forest import tune_causal_forest

# Automated tuning
tuned_params = tune_causal_forest(
    X, y, treatment,
    param_grid={
        'min_node_size': [5, 10, 20],
        'mtry': [None, 3, 5],
        'honesty_fraction': [0.4, 0.5, 0.6]
    },
    cv_folds=5
)

# Fit with tuned parameters
cf_tuned = fit_causal_forest(X, y, treatment, config=tuned_params)
```

### Sample Size Considerations

| Sample Size | Recommendation |
|-------------|----------------|
| N < 500 | Consider simpler methods (IPW, regression adjustment) |
| 500-2000 | Use causal forest with larger min_node_size (20+) |
| 2000-10000 | Standard settings work well |
| N > 10000 | Can use smaller min_node_size for more heterogeneity |

## Visualization

### CATE Distribution

```python
from causal_forest import plot_cate_distribution

# Visualize distribution of treatment effects
plot_cate_distribution(
    cate_results.estimates,
    ci_lower=cate_results.ci_lower,
    ci_upper=cate_results.ci_upper,
    title="Distribution of Individual Treatment Effects"
)
```

### Group-wise CATEs

```python
from causal_forest import plot_cate_by_group

# Compare CATEs across groups
plot_cate_by_group(
    cate_estimates=cate_results.estimates,
    group_variable=data['customer_segment'],
    title="Treatment Effects by Customer Segment"
)
```

### CATE vs Covariates

```python
from causal_forest import plot_cate_vs_covariate

# Partial dependence of CATE on a covariate
plot_cate_vs_covariate(
    cf_model,
    X=data[effect_modifiers],
    covariate='customer_age',
    title="How Treatment Effect Varies with Age"
)
```

## Policy Trees for Interpretable Rules

For interpretable treatment rules, use policy trees:

```python
from causal_forest import fit_policy_tree

# Fit interpretable policy tree
policy_tree = fit_policy_tree(
    cate_estimates=cate_results.estimates,
    X=data[effect_modifiers],
    max_depth=3,  # Keep shallow for interpretability
    min_samples_leaf=100
)

# Visualize the tree
policy_tree.plot(feature_names=effect_modifiers)

# Extract rules
rules = policy_tree.get_rules()
for rule in rules:
    print(f"If {rule.condition}: Treat = {rule.treatment}")
```

## Common Mistakes to Avoid

### 1. Confusing with Random Forest

**Wrong**: "I'll use my existing random forest to estimate treatment effects"

**Right**: Causal forests have fundamentally different objectives:
- Random forests: predict Y
- Causal forests: predict tau(x) = E[Y(1) - Y(0) | X]

Causal forests use special splitting criteria that target heterogeneity in treatment effects, not outcome prediction.

### 2. Ignoring Confidence Intervals

**Wrong**: "The CATE for this individual is 5.2"

**Right**: "The CATE for this individual is 5.2 (95% CI: 1.1 to 9.3)"

Always report uncertainty - individual CATE estimates can be noisy.

### 3. Over-interpreting Variable Importance

**Wrong**: "Age is the most important driver of treatment effect heterogeneity"

**Right**: Variable importance shows which variables the forest uses for splitting, but:
- Correlated variables share importance
- Doesn't indicate direction of effect
- Use BLP for quantitative relationships

### 4. Using Dishonest Forests for Inference

**Wrong**: `honesty=False` for "better predictions"

**Right**: Always use `honesty=True` when you need valid confidence intervals or statistical tests.

### 5. Small Sample Heterogeneity Claims

**Wrong**: "Treatment effect varies significantly" (with N=200)

**Right**: With small samples, focus on ATE estimation. Heterogeneity detection requires larger samples.

## Complete Example: Personalized Marketing

```python
from causal_forest import (
    fit_causal_forest, estimate_cate, variable_importance,
    best_linear_projection, policy_learning, heterogeneity_test,
    run_full_cf_analysis, CausalOutput
)
import pandas as pd

# Load data from A/B test
data = pd.read_csv('marketing_ab_test.csv')

# Define analysis components
outcome = 'purchase_amount'
treatment = 'email_campaign'  # 1 = received email, 0 = no email
effect_modifiers = [
    'customer_age', 'tenure_days', 'past_purchases',
    'email_opens_30d', 'website_visits_30d', 'segment'
]

# Run complete analysis
results: CausalOutput = run_full_cf_analysis(
    data=data,
    outcome=outcome,
    treatment=treatment,
    effect_modifiers=effect_modifiers,
    output_dir='./causal_forest_results'
)

# Summary
print(results.summary())

# Key findings
print(f"\nAverage Treatment Effect: ${results.ate:.2f}")
print(f"CATE Range: ${results.cate_min:.2f} to ${results.cate_max:.2f}")
print(f"Significant Heterogeneity: {results.heterogeneity_significant}")

# Top heterogeneity drivers
print("\nTop Drivers of Heterogeneity:")
for var, imp in results.variable_importance[:3]:
    print(f"  {var}: {imp:.3f}")

# Policy recommendation
print(f"\nOptimal Policy:")
print(f"  Target {results.policy.treatment_rate:.1%} of customers")
print(f"  Expected value: ${results.policy.value:.2f} per customer")
print(f"  Improvement over treat-all: {results.policy.improvement:.1%}")

# Who to target
high_value_customers = data[results.policy.recommendations == 1]
print(f"\nHigh-value customer profile:")
print(high_value_customers[effect_modifiers].describe())
```

## Output Interpretation

### CATE Summary Statistics

| Metric | Interpretation |
|--------|----------------|
| Mean CATE | Average treatment effect (should match ATE from simpler methods) |
| CATE Std Dev | Amount of heterogeneity in treatment effects |
| % Positive | Proportion who benefit from treatment |
| % Significant | Proportion with statistically significant effects |

### Heterogeneity Test Interpretation

| Result | Interpretation | Action |
|--------|----------------|--------|
| p < 0.05 | Significant heterogeneity | Explore personalization |
| p >= 0.05 | No significant heterogeneity | Focus on ATE; uniform policy may suffice |

### Policy Value Metrics

| Metric | Definition |
|--------|------------|
| Policy Value | E[Y | follow policy] - E[Y | no treatment] |
| Improvement | (Policy value - ATE) / ATE |
| Treatment Rate | Proportion of population treated under optimal policy |

## References

1. Wager, S., & Athey, S. (2018). Estimation and Inference of Heterogeneous Treatment Effects using Random Forests. *JASA*.
2. Athey, S., Tibshirani, J., & Wager, S. (2019). Generalized Random Forests. *Annals of Statistics*.
3. Athey, S., & Wager, S. (2021). Policy Learning with Observational Data. *Econometrica*.
4. grf R package: https://grf-labs.github.io/grf/
5. econml Python package: https://econml.azurewebsites.net/

## See Also

- `double-ml` - For ATE estimation with cross-fitting
- `meta-learners` - Alternative CATE estimators (T-learner, X-learner)
- `sensitivity-analysis` - Testing robustness of causal claims
