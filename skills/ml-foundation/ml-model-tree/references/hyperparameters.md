# Hyperparameter Reference Guide

## Overview

This reference provides comprehensive guidance on key hyperparameters for tree-based models, with special emphasis on settings appropriate for DDML nuisance function estimation.

---

## Universal Tree Parameters

### Tree Depth Control

| Parameter | Models | Description | Typical Range |
|-----------|--------|-------------|---------------|
| `max_depth` | All | Maximum depth of tree | 3-15 (boosting), None (RF) |
| `num_leaves` | LightGBM | Maximum leaves per tree | 15-127 |
| `max_leaf_nodes` | sklearn | Alternative to max_depth | 8-256 |

**Guidelines**:
- **Boosting models**: Start with `max_depth=5-7`, increase if underfitting
- **Random Forest**: Often use `max_depth=None` (full trees)
- **LightGBM**: Use `num_leaves` instead; `num_leaves < 2^max_depth`

### Minimum Samples

| Parameter | Models | Description | Typical Range |
|-----------|--------|-------------|---------------|
| `min_samples_leaf` | sklearn, RF | Min samples in leaf | 1-50 |
| `min_samples_split` | sklearn, RF | Min samples to split | 2-100 |
| `min_child_weight` | XGBoost | Min sum of instance weight | 1-10 |
| `min_data_in_leaf` | LightGBM | Min samples in leaf | 10-100 |

**Guidelines**:
- Higher values = more regularization = simpler trees
- For DDML: Use moderate values (10-50) to prevent overfitting on folds

---

## Ensemble Size

### Number of Trees

| Parameter | Models | Description | Typical Range |
|-----------|--------|-------------|---------------|
| `n_estimators` | sklearn, all | Number of trees/rounds | 100-1000 |
| `num_iterations` | CatBoost | Number of trees | 100-1000 |

**Guidelines**:
- **Random Forest**: More trees = better (diminishing returns after 500)
- **Boosting**: Use early stopping to determine optimal number
- **DDML**: 100-300 trees usually sufficient; avoid excessive complexity

```python
# Determine optimal n_estimators with early stopping
model = fit_xgboost(
    X_train, y_train,
    params={'max_depth': 5, 'learning_rate': 0.1},
    n_rounds=1000,  # Upper bound
    early_stopping_rounds=50,  # Stop if no improvement
    eval_set=[(X_val, y_val)]
)
print(f"Optimal iterations: {model['best_iteration']}")
```

---

## Learning Rate and Shrinkage

### Boosting Learning Rate

| Parameter | Models | Description | Typical Range |
|-----------|--------|-------------|---------------|
| `learning_rate` / `eta` | XGBoost | Step size shrinkage | 0.01-0.3 |
| `learning_rate` | LightGBM | Step size shrinkage | 0.01-0.3 |
| `learning_rate` | CatBoost | Step size shrinkage | 0.01-0.3 |

**Trade-off**:
- Lower learning rate + more trees = better generalization
- Higher learning rate + fewer trees = faster training

**DDML Recommendation**:
```python
# Conservative: Better generalization for nuisance estimation
params_conservative = {
    'learning_rate': 0.05,
    'n_rounds': 500
}

# Aggressive: Faster training, may need more regularization
params_aggressive = {
    'learning_rate': 0.2,
    'n_rounds': 100
}
```

---

## Regularization Parameters

### L1 and L2 Regularization

| Parameter | Models | Description | Typical Range |
|-----------|--------|-------------|---------------|
| `reg_alpha` / `alpha` | XGBoost | L1 regularization | 0-10 |
| `reg_lambda` / `lambda` | XGBoost | L2 regularization | 0-10 |
| `lambda_l1` | LightGBM | L1 regularization | 0-10 |
| `lambda_l2` | LightGBM | L2 regularization | 0-10 |
| `l2_leaf_reg` | CatBoost | L2 regularization | 1-10 |

**Guidelines**:
- Start with default (L2=1, L1=0)
- Increase if overfitting observed
- L1 promotes sparsity (feature selection)
- L2 smooths predictions

### Tree Structure Regularization

| Parameter | Models | Description | Typical Range |
|-----------|--------|-------------|---------------|
| `gamma` / `min_split_loss` | XGBoost | Min loss reduction for split | 0-5 |
| `min_gain_to_split` | LightGBM | Min gain for split | 0-1 |
| `ccp_alpha` | sklearn | Cost-complexity pruning | 0-0.1 |

---

## Sampling Parameters

### Row Sampling (Bagging)

| Parameter | Models | Description | Typical Range |
|-----------|--------|-------------|---------------|
| `subsample` | XGBoost | Row sampling per tree | 0.5-1.0 |
| `bagging_fraction` | LightGBM | Row sampling per tree | 0.5-1.0 |
| `subsample` | CatBoost | Row sampling per tree | 0.5-1.0 |
| `bootstrap` | sklearn RF | Use bootstrap samples | True/False |

### Column Sampling

| Parameter | Models | Description | Typical Range |
|-----------|--------|-------------|---------------|
| `colsample_bytree` | XGBoost | Features per tree | 0.5-1.0 |
| `colsample_bylevel` | XGBoost | Features per level | 0.5-1.0 |
| `colsample_bynode` | XGBoost | Features per split | 0.5-1.0 |
| `feature_fraction` | LightGBM | Features per tree | 0.5-1.0 |
| `max_features` | sklearn RF | Features per split | 'sqrt', 'log2', float |

**DDML Recommendation**:
```python
# Moderate sampling for stable nuisance estimates
params_ddml = {
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    # OR for LightGBM:
    'bagging_fraction': 0.8,
    'feature_fraction': 0.8,
    'bagging_freq': 5
}
```

---

## Model-Specific Parameters

### XGBoost-Specific

| Parameter | Description | Default | Guidance |
|-----------|-------------|---------|----------|
| `booster` | Boosting type | 'gbtree' | Usually keep default |
| `tree_method` | Tree construction | 'auto' | 'hist' for speed, 'gpu_hist' for GPU |
| `scale_pos_weight` | Class imbalance | 1 | Set to n_neg/n_pos |
| `max_bin` | Histogram bins | 256 | Higher = more precision |

### LightGBM-Specific

| Parameter | Description | Default | Guidance |
|-----------|-------------|---------|----------|
| `boosting_type` | Boosting algorithm | 'gbdt' | 'dart' for regularization |
| `num_leaves` | Max leaves | 31 | Primary complexity control |
| `max_bin` | Histogram bins | 255 | Higher = more precision |
| `is_unbalance` | Handle imbalance | False | True for imbalanced classification |
| `categorical_feature` | Cat feature indices | None | Use for native categorical handling |

### CatBoost-Specific

| Parameter | Description | Default | Guidance |
|-----------|-------------|---------|----------|
| `cat_features` | Categorical indices | None | Essential for categorical data |
| `grow_policy` | Tree growth | 'SymmetricTree' | 'Lossguide' for asymmetric |
| `border_count` | Histogram bins | 254 | Higher for continuous features |
| `bootstrap_type` | Sampling method | 'MVS' | 'Bayesian' for small data |

---

## Tuning Strategies

### 1. Quick Start (Default + Early Stopping)

```python
# Start with sensible defaults, rely on early stopping
xgb_quick = fit_xgboost(
    X, y, task='regression',
    params={
        'max_depth': 6,
        'learning_rate': 0.1,
        'subsample': 0.8,
        'colsample_bytree': 0.8
    },
    n_rounds=500,
    early_stopping_rounds=50
)
```

### 2. Systematic Tuning (Grid Search)

```python
from tree_models import tune_hyperparameters

# Stage 1: Tree structure
param_grid_structure = {
    'max_depth': [3, 5, 7, 9],
    'min_child_weight': [1, 3, 5]
}

# Stage 2: Sampling
param_grid_sampling = {
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0]
}

# Stage 3: Regularization
param_grid_reg = {
    'reg_alpha': [0, 0.1, 1],
    'reg_lambda': [0, 1, 5]
}
```

### 3. Bayesian Optimization (Optuna)

```python
import optuna

def objective(trial):
    params = {
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True)
    }

    # Cross-validation score
    model = fit_xgboost(X, y, params=params, n_rounds=200)
    return cv_score  # Return validation metric

study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=100)
```

---

## DDML-Specific Tuning Guidelines

### For Nuisance Function Estimation

**Goal**: Achieve good predictive performance without overfitting, ensuring stable cross-fitting.

**Recommended Settings**:

```python
# Conservative settings for DDML nuisance estimation
ddml_params_xgboost = {
    'max_depth': 5,              # Moderate complexity
    'learning_rate': 0.05,       # Conservative learning
    'subsample': 0.8,            # Some randomness
    'colsample_bytree': 0.8,     # Feature subsampling
    'reg_alpha': 0.1,            # Light L1
    'reg_lambda': 1.0,           # Standard L2
    'min_child_weight': 3        # Prevent tiny leaves
}

ddml_params_lightgbm = {
    'num_leaves': 31,            # Standard complexity
    'learning_rate': 0.05,       # Conservative
    'feature_fraction': 0.8,     # Feature subsampling
    'bagging_fraction': 0.8,     # Row subsampling
    'bagging_freq': 5,           # Bagging frequency
    'lambda_l1': 0.1,            # Light L1
    'lambda_l2': 1.0,            # Standard L2
    'min_data_in_leaf': 20       # Prevent overfitting
}
```

### Cross-Fitting Considerations

1. **Consistency across folds**: Use fixed random seeds
2. **Avoid extreme predictions**: Regularization helps
3. **Sample splitting**: Models should generalize to holdout folds
4. **Propensity scores**: Clip extreme values (0.01-0.99)

```python
# Ensure reproducibility in cross-fitting
import numpy as np

def train_nuisance_models(X, y, n_folds=5, random_state=42):
    """Train nuisance models with cross-fitting."""
    from sklearn.model_selection import KFold

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    predictions = np.zeros(len(y))

    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        model = fit_xgboost(
            X[train_idx], y[train_idx],
            params=ddml_params_xgboost,
            n_rounds=200,
            early_stopping_rounds=30,
            random_state=random_state + fold  # Vary seed per fold
        )
        predictions[val_idx] = model['model'].predict(X[val_idx])

    return predictions
```

---

## Quick Reference Tables

### XGBoost Cheat Sheet

| Goal | Parameters to Adjust |
|------|---------------------|
| Reduce overfitting | Decrease `max_depth`, increase `min_child_weight`, decrease `learning_rate` |
| Speed up training | Increase `learning_rate`, use `tree_method='hist'` |
| Handle imbalance | Set `scale_pos_weight`, use `eval_metric='auc'` |
| Feature selection | Increase `reg_alpha`, decrease `colsample_bytree` |

### LightGBM Cheat Sheet

| Goal | Parameters to Adjust |
|------|---------------------|
| Reduce overfitting | Decrease `num_leaves`, increase `min_data_in_leaf` |
| Speed up training | Decrease `num_leaves`, use `feature_fraction` |
| Handle imbalance | Set `is_unbalance=True` or `scale_pos_weight` |
| Categorical data | Set `categorical_feature` for native handling |

---

## References

- XGBoost Parameters: https://xgboost.readthedocs.io/en/latest/parameter.html
- LightGBM Parameters: https://lightgbm.readthedocs.io/en/latest/Parameters.html
- CatBoost Parameters: https://catboost.ai/en/docs/references/training-parameters/
- Probst, P., Wright, M. N., & Boulesteix, A. L. (2019). Hyperparameters and tuning strategies for random forest.
