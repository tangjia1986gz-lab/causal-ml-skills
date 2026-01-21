---
name: ml-model-tree
description: Tree-based machine learning models for prediction and causal inference
triggers:
  - random forest
  - xgboost
  - lightgbm
  - gradient boosting
  - decision tree
  - SHAP
  - feature importance
  - tree model
  - ensemble methods
  - boosting
---

# Tree-Based ML Models Skill

## Overview

Tree-based models are powerful nonparametric methods that excel at capturing complex, nonlinear relationships in data. This skill covers decision trees, random forests, gradient boosting (XGBoost, LightGBM), and their applications in both prediction and causal inference. These models are particularly valuable in causal ML as first-stage learners in Double/Debiased Machine Learning (DDML) and for propensity score estimation.

## Decision Trees (CART)

### Fundamentals

Classification and Regression Trees (CART) recursively partition the feature space to minimize prediction error:

```python
from tree_models import fit_decision_tree

# Regression tree
tree_reg = fit_decision_tree(X, y, task='regression', max_depth=5, min_samples_leaf=10)

# Classification tree
tree_clf = fit_decision_tree(X, y, task='classification', max_depth=5, min_samples_leaf=10)
```

### Key Parameters

| Parameter | Description | Guidance |
|-----------|-------------|----------|
| `max_depth` | Maximum tree depth | Start with 3-10; deeper = more complex |
| `min_samples_leaf` | Minimum samples per leaf | Higher = more regularization |
| `min_samples_split` | Minimum samples to split | Prevents splits on small groups |
| `max_features` | Features to consider per split | Controls randomness |

### Splitting Criteria

- **Regression**: Mean Squared Error (MSE), Mean Absolute Error (MAE)
- **Classification**: Gini impurity, Entropy (information gain)

### Pruning

Pruning prevents overfitting by removing branches that provide little predictive power:

```python
# Cost-complexity pruning (post-pruning)
tree = fit_decision_tree(X, y, task='regression', ccp_alpha=0.01)
```

### Interpretation

Decision trees are highly interpretable:
- **Feature importance**: Based on total impurity reduction
- **Decision rules**: Extract if-then rules from tree structure
- **Visualization**: Plot tree structure directly

## Random Forest

### Bagging Ensemble

Random Forest combines many decision trees using bagging (bootstrap aggregating):

```python
from tree_models import fit_random_forest

# Random Forest for regression
rf_reg = fit_random_forest(
    X, y,
    task='regression',
    n_estimators=100,
    max_depth=None,
    min_samples_leaf=5,
    max_features='sqrt'
)

# Random Forest for classification
rf_clf = fit_random_forest(
    X, y,
    task='classification',
    n_estimators=100,
    max_depth=None,
    class_weight='balanced'
)
```

### Key Parameters

| Parameter | Description | Guidance |
|-----------|-------------|----------|
| `n_estimators` | Number of trees | More is better (100-500 typical) |
| `max_depth` | Maximum tree depth | None for full trees, or limit |
| `max_features` | Features per split | 'sqrt' for classification, 'auto'/n/3 for regression |
| `min_samples_leaf` | Minimum samples per leaf | Higher = smoother predictions |
| `bootstrap` | Use bootstrap samples | True (default) for variance reduction |

### Out-of-Bag (OOB) Error

Each tree is trained on ~63% of data, allowing validation on the remaining 37%:

```python
rf = fit_random_forest(X, y, task='regression', n_estimators=100, oob_score=True)
print(f"OOB R-squared: {rf['oob_score']:.3f}")
```

### Feature Importance

```python
from tree_models import get_feature_importance

# Impurity-based importance (default)
importance = get_feature_importance(rf['model'], feature_names, method='impurity')

# Permutation importance (more reliable)
importance = get_feature_importance(rf['model'], feature_names, method='permutation', X=X, y=y)
```

## XGBoost

### Gradient Boosting Framework

XGBoost builds trees sequentially, each correcting errors of the ensemble:

```python
from tree_models import fit_xgboost

# XGBoost regression
xgb_model = fit_xgboost(
    X, y,
    task='regression',
    params={
        'max_depth': 6,
        'learning_rate': 0.1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'reg_alpha': 0.1,
        'reg_lambda': 1.0
    },
    n_rounds=100,
    early_stopping_rounds=10
)

# XGBoost classification
xgb_clf = fit_xgboost(
    X, y,
    task='classification',
    params={
        'max_depth': 4,
        'learning_rate': 0.1,
        'scale_pos_weight': 2.0  # For imbalanced classes
    },
    n_rounds=200,
    early_stopping_rounds=20
)
```

### Key Parameters

| Parameter | Description | Guidance |
|-----------|-------------|----------|
| `max_depth` | Tree depth | 3-10 (lower than RF) |
| `learning_rate` (eta) | Shrinkage factor | 0.01-0.3 (lower = more trees needed) |
| `n_estimators` | Number of boosting rounds | 100-1000 |
| `subsample` | Row sampling ratio | 0.5-1.0 |
| `colsample_bytree` | Column sampling ratio | 0.5-1.0 |
| `reg_alpha` (L1) | L1 regularization | 0-10 |
| `reg_lambda` (L2) | L2 regularization | 1-10 |

### Regularization in XGBoost

XGBoost's objective function includes regularization:

$$\text{Obj} = \sum_{i=1}^n L(y_i, \hat{y}_i) + \sum_{k=1}^K \Omega(f_k)$$

where $\Omega(f) = \gamma T + \frac{1}{2}\lambda \|w\|^2$ penalizes tree complexity.

### Early Stopping

Prevents overfitting by monitoring validation performance:

```python
xgb_model = fit_xgboost(
    X_train, y_train,
    task='regression',
    params={'max_depth': 6, 'learning_rate': 0.1},
    n_rounds=1000,
    early_stopping_rounds=50,
    eval_set=[(X_val, y_val)]
)
print(f"Best iteration: {xgb_model['best_iteration']}")
```

## LightGBM

### Leaf-Wise Growth

Unlike XGBoost's level-wise growth, LightGBM grows leaf-wise for efficiency:

```python
from tree_models import fit_lightgbm

# LightGBM regression
lgb_model = fit_lightgbm(
    X, y,
    task='regression',
    params={
        'num_leaves': 31,
        'max_depth': -1,
        'learning_rate': 0.1,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'lambda_l1': 0.1,
        'lambda_l2': 1.0
    },
    n_rounds=100,
    early_stopping_rounds=10
)
```

### Categorical Feature Handling

LightGBM natively handles categorical features:

```python
lgb_model = fit_lightgbm(
    X, y,
    task='classification',
    params={'num_leaves': 31},
    categorical_features=['region', 'education_level']
)
```

### Key Parameters

| Parameter | Description | Guidance |
|-----------|-------------|----------|
| `num_leaves` | Max leaves per tree | Primary complexity control (31 default) |
| `max_depth` | Max tree depth | -1 for no limit; use num_leaves instead |
| `learning_rate` | Shrinkage | 0.01-0.3 |
| `feature_fraction` | Column sampling | 0.5-1.0 |
| `bagging_fraction` | Row sampling | 0.5-1.0 |
| `min_data_in_leaf` | Min samples per leaf | Higher = more regularization |

### Speed Advantages

LightGBM is typically 2-10x faster than XGBoost due to:
- Histogram-based algorithm
- Leaf-wise growth
- Efficient categorical handling
- Gradient-based one-side sampling (GOSS)

## Hyperparameter Tuning

### Cross-Validation Tuning

```python
from tree_models import tune_hyperparameters

# Tune XGBoost hyperparameters
param_grid = {
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.3],
    'subsample': [0.7, 0.8, 0.9],
    'colsample_bytree': [0.7, 0.8, 0.9]
}

best_params = tune_hyperparameters(
    X, y,
    model_type='xgboost',
    param_grid=param_grid,
    cv=5,
    scoring='neg_mean_squared_error'
)
print(f"Best parameters: {best_params}")
```

### Tuning Strategy

1. **Start with defaults**, establish baseline
2. **Tune tree structure**: `max_depth`, `num_leaves`, `min_samples_leaf`
3. **Tune sampling**: `subsample`, `colsample_bytree`, `bagging_fraction`
4. **Tune regularization**: `reg_alpha`, `reg_lambda`
5. **Tune learning rate**: Lower rate + more iterations
6. **Final tuning**: Fine-tune best parameters

## Interpretability

### Feature Importance Methods

```python
from tree_models import get_feature_importance

# Impurity-based (fast but biased toward high-cardinality features)
imp_impurity = get_feature_importance(model, feature_names, method='impurity')

# Permutation importance (unbiased, but slower)
imp_perm = get_feature_importance(model, feature_names, method='permutation', X=X, y=y)
```

**Impurity-based** importance sums the decrease in impurity from splits on each feature. It can be biased toward features with many unique values.

**Permutation importance** measures performance decrease when a feature is randomly shuffled. It is model-agnostic and unbiased.

### Partial Dependence Plots (PDP)

PDPs show the marginal effect of features on predictions:

```python
from tree_models import partial_dependence_plot

# Single feature PDP
partial_dependence_plot(model, X, features=['income'])

# Two-feature interaction PDP
partial_dependence_plot(model, X, features=['income', 'age'])
```

PDPs answer: "What is the average predicted outcome as this feature varies, holding other features at their observed values?"

### SHAP Values

SHAP (SHapley Additive exPlanations) provides theoretically grounded feature attributions:

```python
from tree_models import compute_shap_values

# Compute SHAP values
shap_result = compute_shap_values(model, X)

# Access SHAP values and visualizations
shap_values = shap_result['shap_values']
shap_result['summary_plot']  # Displays summary plot
shap_result['bar_plot']      # Feature importance bar plot
```

**SHAP advantages**:
- Consistent and locally accurate
- Handles feature interactions
- Provides both global and local interpretations
- TreeSHAP is fast for tree models

**Understanding SHAP output**:
- Each sample gets one SHAP value per feature
- SHAP values sum to (prediction - expected value)
- Positive SHAP = feature pushes prediction higher

## Causal Inference Applications

### First-Stage Learners in DDML

Tree-based models are excellent first-stage learners for nuisance functions:

```python
from tree_models import fit_random_forest, fit_xgboost

# Propensity score model
propensity_model = fit_random_forest(X, D, task='classification', n_estimators=200)

# Outcome model
outcome_model = fit_xgboost(X, Y, task='regression', params={'max_depth': 5}, n_rounds=200)

# Use predictions in DDML
# See ddml skill for full implementation
```

**Why trees work well for DDML**:
- Capture nonlinear relationships automatically
- Handle high-dimensional controls
- Robust to outliers
- No need to specify functional form

### Propensity Score Estimation

```python
# Estimate propensity scores using gradient boosting
ps_model = fit_xgboost(
    X, treatment,
    task='classification',
    params={
        'max_depth': 4,
        'learning_rate': 0.1,
        'scale_pos_weight': n_control / n_treated
    },
    n_rounds=200
)

propensity_scores = ps_model['model'].predict_proba(X)[:, 1]

# Check overlap
print(f"Min PS: {propensity_scores.min():.3f}")
print(f"Max PS: {propensity_scores.max():.3f}")
```

**Best practices**:
- Check for extreme propensity scores
- Use regularization to avoid perfect prediction
- Consider trimming samples with extreme PS

### Connection to Causal Forest

For heterogeneous treatment effect estimation, see the **causal-forest** skill. Causal Forests modify the random forest algorithm to target treatment effect heterogeneity rather than prediction:

```python
# Standard prediction forest
rf_pred = fit_random_forest(X, Y, task='regression')

# Causal forest (see causal-forest skill)
# Estimates conditional average treatment effects: E[Y(1) - Y(0) | X]
from causal_forest import fit_causal_forest
cate_forest = fit_causal_forest(X, Y, D)
```

Key differences:
- **Prediction forest**: Minimizes MSE for Y
- **Causal forest**: Targets treatment effect heterogeneity
- **Splitting criterion**: Causal forest splits to maximize treatment effect variation

## Model Comparison and Selection

### Compare All Tree Models

```python
from tree_models import compare_tree_models

# Compare decision tree, random forest, XGBoost, and LightGBM
comparison = compare_tree_models(X, y, task='regression')

print(comparison['summary'])
# Shows cross-validated performance for each model

comparison['plot']  # Visual comparison
```

### Selection Guidance

| Use Case | Recommended Model | Reason |
|----------|------------------|--------|
| Small data, interpretability needed | Decision Tree | Fully transparent |
| General prediction | Random Forest | Robust, few hyperparameters |
| Maximum accuracy | XGBoost/LightGBM | Best performance |
| Very large data | LightGBM | Speed advantage |
| Propensity scores | Random Forest or XGBoost | Good probability calibration |
| DDML first stage | Any boosting method | Captures complex relationships |

### Ensemble Stacking

For maximum predictive performance, combine models:

```python
from sklearn.ensemble import StackingRegressor

estimators = [
    ('rf', RandomForestRegressor(n_estimators=100)),
    ('xgb', xgb.XGBRegressor(n_estimators=100)),
    ('lgb', lgb.LGBMRegressor(n_estimators=100))
]

stack = StackingRegressor(
    estimators=estimators,
    final_estimator=RidgeCV()
)
```

## Common Mistakes and Best Practices

### Mistakes to Avoid

1. **Overfitting boosting models**: Always use early stopping and validation set
2. **Ignoring feature importance bias**: Use permutation importance for reliable rankings
3. **Default hyperparameters**: Tune key parameters for your specific data
4. **Ignoring class imbalance**: Use `scale_pos_weight` or `class_weight`
5. **Feature leakage**: Be careful with time-series or post-treatment variables

### Best Practices

1. **Start simple**: Begin with random forest, move to boosting if needed
2. **Use cross-validation**: For both tuning and evaluation
3. **Monitor training curves**: Check for overfitting
4. **Understand your features**: Use SHAP for model debugging
5. **Validate on holdout**: Final evaluation on truly unseen data

### Computational Tips

- **Large data**: Use LightGBM with histogram binning
- **GPU acceleration**: XGBoost and LightGBM support GPU training
- **Parallel training**: Set `n_jobs=-1` for random forest
- **Memory constraints**: Reduce `max_depth` or use sampling

## Summary

Tree-based models provide a powerful toolkit for modern ML and causal inference:

1. **Decision trees** offer interpretability but limited accuracy
2. **Random forests** provide robust performance with minimal tuning
3. **XGBoost/LightGBM** achieve state-of-the-art performance with proper tuning
4. **Interpretability tools** (SHAP, PDP, feature importance) reveal model behavior
5. **Causal applications** include first-stage estimation, propensity scores, and treatment effect discovery

Key takeaways:
- Tree models automatically capture nonlinear relationships
- Regularization is crucial for boosting methods
- Use appropriate interpretability tools to understand predictions
- In causal ML, tree models excel as flexible nuisance function estimators
