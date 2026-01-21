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

---

## Quick Reference

### Core Concepts
- [Identification Assumptions](references/identification_assumptions.md) - Unconfoundedness, overlap, SUTVA, honest estimation theory
- [Estimation Methods](references/estimation_methods.md) - Causal forests, GRF, honest trees, R grf vs Python econml
- [Diagnostic Tests](references/diagnostic_tests.md) - Calibration, overlap assessment, heterogeneity tests
- [Heterogeneity Analysis](references/heterogeneity_analysis.md) - CATE estimation, subgroup discovery, BLP
- [Reporting Standards](references/reporting_standards.md) - Variable importance, CATE plots, policy trees

### CLI Tools
- [causal_forest_pipeline.py](scripts/causal_forest_pipeline.py) - Complete causal forest analysis pipeline with CATE estimation, heterogeneity testing, policy learning, and visualization

### Templates
- [Analysis Report Template](assets/markdown/causal_forest_report.md) - Comprehensive report template

---

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

---

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

See [Identification Assumptions](references/identification_assumptions.md) for detailed theory.

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

---

## Implementation Workflow

### Step 1: Setup - Define Components

```python
# pip install econml scikit-learn pandas numpy matplotlib

from econml.dml import CausalForestDML
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
import pandas as pd
import numpy as np

# Identify your variables
outcome = 'revenue'           # Y: what we're trying to affect
treatment = 'discount'        # W: binary or continuous treatment
effect_modifiers = [          # X: variables that may modify treatment effect
    'customer_age',
    'tenure_months',
    'past_purchases',
    'segment'
]
confounders = [               # W: Additional adjustment variables (controls)
    'region',
    'signup_channel'
]

# Prepare data matrices
Y = data[outcome].values
T = data[treatment].values
X = data[effect_modifiers].values
W = data[effect_modifiers + confounders].values  # All controls
```

### Step 2: Training - Fit Causal Forest

```python
# Initialize Causal Forest with flexible first-stage models
cf_model = CausalForestDML(
    model_y=GradientBoostingRegressor(n_estimators=100, max_depth=3),
    model_t=GradientBoostingClassifier(n_estimators=100, max_depth=3),
    n_estimators=2000,        # Number of trees
    min_samples_leaf=5,       # Minimum observations per leaf
    max_depth=None,           # Full tree depth
    honest=True,              # Use honest splitting
    cv=5,                     # Cross-fitting folds
    random_state=42
)

# Fit the model
cf_model.fit(Y=Y, T=T, X=X, W=W)
print("Training complete.")
```

### Step 3: CATE Estimation - Individual Treatment Effects

```python
# Estimate treatment effects for each individual
cate = cf_model.effect(X)                   # Point estimates
cate_interval = cf_model.effect_interval(X) # 95% CI
cate_lower, cate_upper = cate_interval

# Summary statistics
print(f"Average CATE: {cate.mean():.3f}")
print(f"CATE Range: [{cate.min():.3f}, {cate.max():.3f}]")
print(f"Proportion with positive effect: {(cate > 0).mean():.1%}")
print(f"Significant effects (CI excludes 0): {((cate_lower > 0) | (cate_upper < 0)).mean():.1%}")
```

### Step 4: Variable Importance - Heterogeneity Drivers

```python
# Extract feature importances from the causal forest
importance = cf_model.feature_importances_

# Create importance DataFrame
importance_df = pd.DataFrame({
    'feature': effect_modifiers,
    'importance': importance
}).sort_values('importance', ascending=False)

print("Variable Importance (Heterogeneity Drivers):")
print(importance_df.to_string(index=False))

# Plot
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 6))
plt.barh(importance_df['feature'], importance_df['importance'])
plt.xlabel('Importance')
plt.title('Drivers of Treatment Effect Heterogeneity')
plt.tight_layout()
plt.savefig('variable_importance.png', dpi=150)
```

### Step 5: Best Linear Projection (BLP)

The BLP summarizes heterogeneity as a linear function of covariates using OLS of CATE on X:

```python
import statsmodels.api as sm

# OLS of estimated CATE on covariates
X_with_const = sm.add_constant(pd.DataFrame(X, columns=effect_modifiers))
blp_model = sm.OLS(cate, X_with_const).fit(cov_type='HC1')

print("Best Linear Projection of CATE:")
print(blp_model.summary())

# Which covariates significantly predict heterogeneity?
significant_vars = blp_result[blp_result['P>|z|'] < 0.05].index.tolist()
print(f"Significant heterogeneity drivers: {significant_vars}")

# Interpretation: CATE ≈ beta_0 + beta_1 * age + beta_2 * tenure + ...
# Positive coefficient = higher CATE for higher covariate values
```

See [Heterogeneity Analysis](references/heterogeneity_analysis.md) for detailed guidance.

### Step 6: Heterogeneity Testing

Test whether treatment effect heterogeneity is statistically significant:

```python
from scipy import stats

# Simple heterogeneity test: variance of CATE estimates
cate_var = np.var(cate)
cate_mean_var = np.mean(((cate_upper - cate_lower) / 3.92)**2)  # Avg squared SE

# Test statistic: ratio of CATE variance to mean variance
het_ratio = cate_var / cate_mean_var
print(f"Heterogeneity Ratio (var(CATE)/mean(var)): {het_ratio:.2f}")
print(f"Ratio > 1 suggests significant heterogeneity")

# More rigorous: Calibration test (from econml)
# Are high predicted CATEs associated with high actual effects?
from econml.score import RScorer
scorer = RScorer(model_y=GradientBoostingRegressor(), model_t=GradientBoostingClassifier())
scorer.fit(Y, T, X=X, W=W)
rscore = scorer.score(cf_model)
print(f"R-score (calibration): {rscore:.3f}")
```

See [Diagnostic Tests](references/diagnostic_tests.md) for comprehensive diagnostics.

### Step 7: Policy Learning - Optimal Treatment Rules

```python
from econml.policy import PolicyTree

# Learn optimal treatment policy using policy tree
policy_tree = PolicyTree(
    max_depth=3,
    min_samples_leaf=50
)

# Fit policy tree to maximize treatment effect
policy_tree.fit(X, cate)

# Get treatment recommendations
recommendations = policy_tree.predict(X)  # 1=treat, 0=don't treat

# Evaluate policy value
treated_value = cate[recommendations == 1].mean()
untreated_value = cate[recommendations == 0].mean()
treatment_rate = recommendations.mean()

print(f"Policy Treatment Rate: {treatment_rate:.1%}")
print(f"Avg CATE for treated: {treated_value:.3f}")
print(f"Avg CATE for untreated: {untreated_value:.3f}")
print(f"Policy targets individuals with higher treatment effects")
```

---

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

### Using R grf from Python (rpy2)

See [Estimation Methods](references/estimation_methods.md) for detailed rpy2 integration guide.

---

## CLI Tools

### Complete Analysis Pipeline

```bash
# Run demo with simulated data
python scripts/causal_forest_pipeline.py --demo

# Run demo with strong heterogeneity
python scripts/causal_forest_pipeline.py --demo --heterogeneity strong

# Run with real data
python scripts/causal_forest_pipeline.py \
    --data experiment.csv \
    --outcome revenue \
    --treatment discount \
    --controls "age,income,tenure,segment,region,channel" \
    --output results/

# Output includes:
# - CATE estimates with confidence intervals
# - Variable importance ranking
# - Heterogeneity test results
# - Best linear projection coefficients
# - Visualization plots (cate_distribution.png, variable_importance.png)
```

### Available Functions in causal_forest_pipeline.py

The pipeline script provides these importable functions:

```python
from scripts.causal_forest_pipeline import (
    simulate_causal_forest_data,   # Generate simulated data with known CATE
    fit_causal_forest_dml,         # Fit CausalForestDML model
    variable_importance,           # Extract feature importances
    heterogeneity_test,            # Test for significant heterogeneity
    best_linear_projection,        # BLP of CATE on covariates
    simple_policy_tree,            # Policy learning
    plot_cate_distribution,        # CATE distribution plot
    plot_cate_by_covariate,        # CATE vs covariate scatter plots
    print_results,                 # Print formatted results
    generate_latex_table,          # Publication-ready LaTeX table
    run_full_analysis,             # Complete pipeline
)
```

---

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

### Sample Size Considerations

| Sample Size | Recommendation |
|-------------|----------------|
| N < 500 | Consider simpler methods (IPW, regression adjustment) |
| 500-2000 | Use causal forest with larger min_node_size (20+) |
| 2000-10000 | Standard settings work well |
| N > 10000 | Can use smaller min_node_size for more heterogeneity |

---

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

---

## Complete Example: Personalized Marketing

```python
from econml.dml import CausalForestDML
from econml.policy import PolicyTree
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
import statsmodels.api as sm
import pandas as pd
import numpy as np

# Load data from A/B test
data = pd.read_csv('marketing_ab_test.csv')

# Define analysis components
outcome = 'purchase_amount'
treatment = 'email_campaign'  # 1 = received email, 0 = no email
effect_modifiers = [
    'customer_age', 'tenure_days', 'past_purchases',
    'email_opens_30d', 'website_visits_30d', 'segment'
]

# Prepare matrices
Y = data[outcome].values
T = data[treatment].values
X = data[effect_modifiers].values

# Step 1: Fit Causal Forest
cf_model = CausalForestDML(
    model_y=GradientBoostingRegressor(n_estimators=100, max_depth=3),
    model_t=GradientBoostingClassifier(n_estimators=100, max_depth=3),
    n_estimators=2000,
    min_samples_leaf=10,
    honest=True,
    cv=5,
    random_state=42
)
cf_model.fit(Y=Y, T=T, X=X)

# Step 2: Estimate CATE
cate = cf_model.effect(X)
cate_lower, cate_upper = cf_model.effect_interval(X)

# Step 3: Summary statistics
print(f"Average Treatment Effect: ${cate.mean():.2f}")
print(f"CATE Range: ${cate.min():.2f} to ${cate.max():.2f}")
print(f"Significant effects: {((cate_lower > 0) | (cate_upper < 0)).mean():.1%}")

# Step 4: Variable importance
importance = cf_model.feature_importances_
importance_df = pd.DataFrame({
    'feature': effect_modifiers,
    'importance': importance
}).sort_values('importance', ascending=False)

print("\nTop Drivers of Heterogeneity:")
for _, row in importance_df.head(3).iterrows():
    print(f"  {row['feature']}: {row['importance']:.3f}")

# Step 5: Policy learning
policy_tree = PolicyTree(max_depth=3, min_samples_leaf=50)
policy_tree.fit(X, cate)
recommendations = policy_tree.predict(X)

treatment_rate = recommendations.mean()
value_treated = cate[recommendations == 1].mean()
value_all = cate.mean()
improvement = (value_treated - value_all) / value_all * 100

print(f"\nOptimal Policy:")
print(f"  Target {treatment_rate:.1%} of customers")
print(f"  Expected CATE for targeted: ${value_treated:.2f}")
print(f"  Improvement over treat-all: {improvement:.1f}%")

# Who to target (high-value segment profile)
high_value = data[recommendations == 1]
print(f"\nHigh-value customer profile (n={len(high_value)}):")
print(high_value[effect_modifiers].describe().round(2))
```

---

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

---

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
