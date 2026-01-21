# Model Selection Guide

## Overview

This reference provides guidance for selecting the appropriate tree-based model for prediction tasks and causal inference applications, particularly for DDML nuisance function estimation.

## Decision Tree (CART)

### When to Use
- **Interpretability is paramount**: Need to explain individual predictions
- **Small datasets**: < 1,000 samples where ensembles may overfit
- **Baseline model**: Establishing simple benchmarks
- **Feature selection**: Identifying most predictive splits

### Characteristics
| Aspect | Value |
|--------|-------|
| Complexity | Low |
| Interpretability | High |
| Accuracy | Low-Medium |
| Training Speed | Fast |
| Risk of Overfitting | High |

### DDML Suitability
**Not recommended** for nuisance function estimation due to high variance. Single trees are too unstable for the cross-fitting procedure.

---

## Random Forest

### When to Use
- **General-purpose prediction**: Default choice for tabular data
- **Robustness required**: Data quality is uncertain
- **Minimal tuning**: Limited time for hyperparameter optimization
- **Parallel computing available**: Scales well across cores
- **Propensity score estimation**: Good probability calibration

### Characteristics
| Aspect | Value |
|--------|-------|
| Complexity | Medium |
| Interpretability | Medium (via feature importance) |
| Accuracy | High |
| Training Speed | Medium |
| Risk of Overfitting | Low |

### Key Advantages
1. **Variance reduction**: Bagging averages out individual tree variance
2. **Out-of-bag estimation**: Built-in validation without holdout set
3. **Feature importance**: Reliable with permutation method
4. **Few hyperparameters**: `n_estimators`, `max_features`, `min_samples_leaf`

### DDML Suitability
**Recommended** for:
- Propensity score models (treatment probability)
- Outcome models when relationships are moderately complex
- Settings with limited sample size (< 10,000)

```python
from tree_models import fit_random_forest

# For propensity score in DDML
ps_model = fit_random_forest(
    X, D,
    task='classification',
    n_estimators=200,
    max_depth=None,  # Full trees for expressiveness
    min_samples_leaf=5,  # Regularization
    oob_score=True  # Built-in validation
)
```

---

## XGBoost

### When to Use
- **Maximum predictive accuracy**: Kaggle-style competitions
- **Structured/tabular data**: Excels on non-image/text data
- **Gradient-based optimization needed**: Custom loss functions
- **GPU acceleration available**: Large datasets with GPU
- **Fine-grained regularization control**: Complex tuning required

### Characteristics
| Aspect | Value |
|--------|-------|
| Complexity | High |
| Interpretability | Low-Medium |
| Accuracy | Very High |
| Training Speed | Medium-Fast |
| Risk of Overfitting | Medium (with proper regularization) |

### Key Advantages
1. **Regularized objective**: Built-in L1/L2 regularization
2. **Handling missing values**: Native support
3. **Early stopping**: Prevents overfitting
4. **Second-order gradients**: More efficient optimization

### DDML Suitability
**Highly recommended** for:
- Outcome models with complex nonlinear relationships
- Large datasets (> 10,000 samples)
- Settings requiring maximum predictive performance

```python
from tree_models import fit_xgboost

# For outcome model in DDML
outcome_model = fit_xgboost(
    X, Y,
    task='regression',
    params={
        'max_depth': 5,
        'learning_rate': 0.1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'reg_alpha': 0.1,
        'reg_lambda': 1.0
    },
    n_rounds=200,
    early_stopping_rounds=20
)
```

---

## LightGBM

### When to Use
- **Large datasets**: > 100,000 samples
- **Speed is critical**: Time-constrained training
- **Categorical features**: Native categorical handling
- **Memory constraints**: More efficient than XGBoost
- **High-dimensional data**: Feature fraction helps

### Characteristics
| Aspect | Value |
|--------|-------|
| Complexity | High |
| Interpretability | Low-Medium |
| Accuracy | Very High |
| Training Speed | Very Fast |
| Risk of Overfitting | Medium |

### Key Advantages
1. **Histogram-based splitting**: 2-10x faster than XGBoost
2. **Leaf-wise growth**: More efficient than level-wise
3. **Categorical feature handling**: No need for one-hot encoding
4. **GOSS and EFB**: Gradient-based one-side sampling, exclusive feature bundling

### DDML Suitability
**Highly recommended** for:
- Large-scale DDML applications
- High-dimensional control sets
- Production pipelines with time constraints

```python
from tree_models import fit_lightgbm

# For large-scale DDML
nuisance_model = fit_lightgbm(
    X, Y,
    task='regression',
    params={
        'num_leaves': 31,
        'learning_rate': 0.1,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5
    },
    n_rounds=200,
    early_stopping_rounds=20,
    categorical_features=['region', 'education']
)
```

---

## CatBoost

### When to Use
- **Many categorical features**: Superior categorical handling
- **Minimal preprocessing desired**: Works with raw features
- **Ordered target encoding**: Prevents target leakage
- **Symmetric trees needed**: More regularization

### Characteristics
| Aspect | Value |
|--------|-------|
| Complexity | High |
| Interpretability | Low-Medium |
| Accuracy | Very High |
| Training Speed | Fast |
| Risk of Overfitting | Low |

### Key Advantages
1. **Ordered boosting**: Prevents prediction shift
2. **Native categorical handling**: No preprocessing needed
3. **Symmetric trees**: More regularized by design
4. **Built-in overfitting detection**: Automatic early stopping

### Installation
```bash
pip install catboost
```

### Usage
```python
from catboost import CatBoostRegressor, CatBoostClassifier

# For DDML with categorical features
model = CatBoostRegressor(
    iterations=200,
    depth=6,
    learning_rate=0.1,
    cat_features=['region', 'education', 'occupation'],
    verbose=False
)
model.fit(X, y)
```

---

## Model Selection Decision Tree

```
                    ┌─────────────────────┐
                    │ What's the priority? │
                    └─────────────────────┘
                              │
          ┌───────────────────┼───────────────────┐
          ▼                   ▼                   ▼
    Interpretability      Accuracy          Speed/Scale
          │                   │                   │
          ▼                   ▼                   ▼
    Decision Tree        XGBoost             LightGBM
    (or Random Forest    (or CatBoost
     with few trees)      if many cats)
```

## DDML-Specific Selection

### For Propensity Score (Treatment Model)
1. **Random Forest** - Good calibration, stable probabilities
2. **XGBoost** with careful tuning - Avoid extreme predictions

### For Outcome Model
1. **XGBoost/LightGBM** - Maximum flexibility for E[Y|X]
2. **Random Forest** - When sample size is limited

### For Cross-Fitting Stability
- Use models with low variance (ensembles preferred)
- Ensure consistent predictions across folds
- Consider model averaging across fold-specific models

### Sample Size Guidelines

| Sample Size | Recommended Model | Notes |
|-------------|-------------------|-------|
| < 1,000 | Random Forest | Avoid boosting, overfitting risk |
| 1,000-10,000 | Random Forest or XGBoost | XGBoost with regularization |
| 10,000-100,000 | XGBoost | Full hyperparameter tuning |
| > 100,000 | LightGBM | Speed advantage crucial |

---

## Performance Comparison Template

```python
from tree_models import compare_tree_models

# Compare all tree models on your data
comparison = compare_tree_models(X, y, task='regression', cv=5)
print(comparison['summary'])

# Use best model for DDML
best_model_name = comparison['summary'].iloc[0]['Model']
print(f"Recommended for DDML: {best_model_name}")
```

## References

- Friedman, J. H. (2001). Greedy function approximation: A gradient boosting machine.
- Breiman, L. (2001). Random forests. Machine learning, 45(1), 5-32.
- Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system.
- Ke, G., et al. (2017). LightGBM: A highly efficient gradient boosting decision tree.
- Prokhorenkova, L., et al. (2018). CatBoost: unbiased boosting with categorical features.
