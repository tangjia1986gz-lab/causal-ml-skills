# Ensemble Methods: Stacking, Blending, and Model Averaging

## Overview

Ensemble methods combine predictions from multiple base models to improve overall performance. Beyond bagging and boosting, advanced ensemble techniques include stacking, blending, and model averaging.

## Ensemble Strategy Comparison

| Method | Description | Pros | Cons |
|--------|-------------|------|------|
| **Averaging** | Simple mean of predictions | Easy, no overfitting | Limited improvement |
| **Weighted Averaging** | Weighted mean by performance | Better than simple | Weights can overfit |
| **Stacking** | Meta-model on base predictions | Most powerful | Complex, slow |
| **Blending** | Stacking on holdout set | Simpler than stacking | Wastes data |

## Model Averaging

### Simple Averaging

```python
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

# Fit base models
rf = RandomForestClassifier(n_estimators=100, random_state=42)
gb = GradientBoostingClassifier(n_estimators=100, random_state=42)
lr = LogisticRegression(random_state=42)

rf.fit(X_train, y_train)
gb.fit(X_train, y_train)
lr.fit(X_train, y_train)

# Average probability predictions
proba_rf = rf.predict_proba(X_test)[:, 1]
proba_gb = gb.predict_proba(X_test)[:, 1]
proba_lr = lr.predict_proba(X_test)[:, 1]

avg_proba = (proba_rf + proba_gb + proba_lr) / 3
predictions = (avg_proba >= 0.5).astype(int)
```

### Weighted Averaging

```python
from sklearn.model_selection import cross_val_score

# Get CV scores for weighting
scores = []
for model, name in [(rf, 'RF'), (gb, 'GB'), (lr, 'LR')]:
    cv_score = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy').mean()
    scores.append(cv_score)
    print(f"{name}: {cv_score:.4f}")

# Convert scores to weights
weights = np.array(scores)
weights = weights / weights.sum()  # Normalize

# Weighted average
weighted_proba = (
    weights[0] * proba_rf +
    weights[1] * proba_gb +
    weights[2] * proba_lr
)
predictions = (weighted_proba >= 0.5).astype(int)
```

### Using VotingClassifier

```python
from sklearn.ensemble import VotingClassifier

# Soft voting (probability averaging)
voting_clf = VotingClassifier(
    estimators=[
        ('rf', RandomForestClassifier(n_estimators=100)),
        ('gb', GradientBoostingClassifier(n_estimators=100)),
        ('lr', LogisticRegression())
    ],
    voting='soft',  # 'hard' for majority voting
    weights=[2, 2, 1]  # Optional weights
)

voting_clf.fit(X_train, y_train)
predictions = voting_clf.predict(X_test)
```

## Stacking

Stacking trains a meta-model (level-1) to combine predictions from base models (level-0).

### Concept

```
Level 0 (Base Models):
+------------------+    +------------------+    +------------------+
|  Random Forest   | -> |  Gradient Boost  | -> |  Neural Network  |
|    P1(x)         |    |    P2(x)         |    |    P3(x)         |
+------------------+    +------------------+    +------------------+
        |                       |                       |
        v                       v                       v
+---------------------------------------------------------------+
|              Meta-features: [P1, P2, P3]                      |
+---------------------------------------------------------------+
                                |
                                v
+---------------------------------------------------------------+
|           Level 1 Meta-Model (e.g., Logistic Regression)      |
|                      Final Prediction                          |
+---------------------------------------------------------------+
```

### Basic Stacking with sklearn

```python
from sklearn.ensemble import StackingClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

# Define base estimators
base_estimators = [
    ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
    ('gb', GradientBoostingClassifier(n_estimators=100, random_state=42)),
    ('svc', SVC(probability=True, random_state=42))
]

# Meta-learner
meta_learner = LogisticRegression()

# Stacking classifier
stacking_clf = StackingClassifier(
    estimators=base_estimators,
    final_estimator=meta_learner,
    cv=5,  # Cross-validation for generating meta-features
    stack_method='predict_proba',  # Use probabilities as meta-features
    passthrough=False  # Don't include original features
)

stacking_clf.fit(X_train, y_train)
predictions = stacking_clf.predict(X_test)
```

### Stacking with Original Features (Passthrough)

```python
# Include original features alongside meta-features
stacking_clf = StackingClassifier(
    estimators=base_estimators,
    final_estimator=LogisticRegression(),
    cv=5,
    passthrough=True  # Include X in meta-features
)
```

### Manual Stacking Implementation

For more control or understanding:

```python
from sklearn.model_selection import KFold
import numpy as np

def manual_stacking(X, y, base_models, meta_model, n_folds=5):
    """
    Manual stacking implementation with cross-validation.
    """
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)

    # Create meta-features matrix
    n_samples = len(X)
    n_base_models = len(base_models)
    meta_features = np.zeros((n_samples, n_base_models))

    # Generate out-of-fold predictions for each base model
    for i, model in enumerate(base_models):
        for train_idx, val_idx in kf.split(X):
            X_train_fold, X_val_fold = X[train_idx], X[val_idx]
            y_train_fold = y[train_idx]

            # Clone and fit model
            model_clone = clone(model)
            model_clone.fit(X_train_fold, y_train_fold)

            # Predict on validation fold
            if hasattr(model_clone, 'predict_proba'):
                meta_features[val_idx, i] = model_clone.predict_proba(X_val_fold)[:, 1]
            else:
                meta_features[val_idx, i] = model_clone.predict(X_val_fold)

    # Fit meta-model on meta-features
    meta_model.fit(meta_features, y)

    # Fit base models on full data for predictions
    fitted_base_models = []
    for model in base_models:
        model_clone = clone(model)
        model_clone.fit(X, y)
        fitted_base_models.append(model_clone)

    return fitted_base_models, meta_model

def predict_stacking(X, fitted_base_models, meta_model):
    """
    Make predictions with stacked model.
    """
    meta_features = np.zeros((len(X), len(fitted_base_models)))

    for i, model in enumerate(fitted_base_models):
        if hasattr(model, 'predict_proba'):
            meta_features[:, i] = model.predict_proba(X)[:, 1]
        else:
            meta_features[:, i] = model.predict(X)

    return meta_model.predict(meta_features)
```

### Stacking for Regression

```python
from sklearn.ensemble import StackingRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge

stacking_reg = StackingRegressor(
    estimators=[
        ('rf', RandomForestRegressor(n_estimators=100)),
        ('gb', GradientBoostingRegressor(n_estimators=100)),
    ],
    final_estimator=Ridge(),
    cv=5
)

stacking_reg.fit(X_train, y_train)
predictions = stacking_reg.predict(X_test)
```

## Blending

Blending is similar to stacking but uses a fixed holdout set instead of cross-validation.

### Blending vs Stacking

| Aspect | Stacking | Blending |
|--------|----------|----------|
| Data efficiency | Higher | Lower (holdout) |
| Leakage risk | Lower (CV) | Slightly higher |
| Implementation | More complex | Simpler |
| Speed | Slower (CV) | Faster |

### Blending Implementation

```python
from sklearn.model_selection import train_test_split

def blending(X, y, base_models, meta_model, blend_size=0.2):
    """
    Blending implementation with holdout set.
    """
    # Split data: training for base models, blending for meta-model
    X_base, X_blend, y_base, y_blend = train_test_split(
        X, y, test_size=blend_size, random_state=42
    )

    # Fit base models on base training data
    fitted_models = []
    blend_features = np.zeros((len(X_blend), len(base_models)))

    for i, model in enumerate(base_models):
        model.fit(X_base, y_base)
        fitted_models.append(model)

        # Generate blend features
        if hasattr(model, 'predict_proba'):
            blend_features[:, i] = model.predict_proba(X_blend)[:, 1]
        else:
            blend_features[:, i] = model.predict(X_blend)

    # Fit meta-model on blend features
    meta_model.fit(blend_features, y_blend)

    # Refit base models on full training data for final predictions
    for model in fitted_models:
        model.fit(X, y)

    return fitted_models, meta_model

# Usage
from sklearn.base import clone

base_models = [
    RandomForestClassifier(n_estimators=100, random_state=42),
    GradientBoostingClassifier(n_estimators=100, random_state=42),
    LogisticRegression(random_state=42)
]
meta_model = LogisticRegression()

fitted_base, fitted_meta = blending(X_train, y_train, base_models, meta_model)
```

## Choosing Base Models for Ensembles

### Diversity Principle

The key to effective ensembles is **diversity** - base models should make different errors.

**Good diversity sources:**
1. Different algorithm families (trees, linear, SVM, neural networks)
2. Different hyperparameters (depth, regularization)
3. Different feature subsets
4. Different training data (bootstrapping)

### Recommended Base Model Combinations

**For Classification:**
```python
base_models = [
    ('rf', RandomForestClassifier(n_estimators=200, max_depth=10)),
    ('et', ExtraTreesClassifier(n_estimators=200, max_depth=10)),
    ('gb', GradientBoostingClassifier(n_estimators=200, max_depth=5)),
    ('xgb', XGBClassifier(n_estimators=200, max_depth=5)),
    ('lgb', LGBMClassifier(n_estimators=200, max_depth=5)),
    ('lr', LogisticRegression(C=1.0)),
    ('svc', SVC(probability=True, kernel='rbf'))
]
```

**For Regression:**
```python
base_models = [
    ('rf', RandomForestRegressor(n_estimators=200)),
    ('gb', GradientBoostingRegressor(n_estimators=200)),
    ('xgb', XGBRegressor(n_estimators=200)),
    ('ridge', Ridge(alpha=1.0)),
    ('svr', SVR(kernel='rbf'))
]
```

### Avoiding Correlated Predictions

```python
from scipy.stats import spearmanr

# Calculate prediction correlation between base models
def check_diversity(models, X, y):
    """Check prediction diversity between models."""
    predictions = []
    for model in models:
        model.fit(X, y)
        pred = model.predict(X)
        predictions.append(pred)

    # Calculate pairwise correlations
    n_models = len(models)
    corr_matrix = np.zeros((n_models, n_models))

    for i in range(n_models):
        for j in range(n_models):
            corr, _ = spearmanr(predictions[i], predictions[j])
            corr_matrix[i, j] = corr

    return corr_matrix

# Aim for correlation < 0.7 between models
```

## Meta-Model Selection

### For Classification

| Meta-Model | When to Use |
|------------|-------------|
| LogisticRegression | Default choice, prevents overfitting |
| GradientBoosting | When base models are simple |
| Neural Network | When you have lots of base models |

### For Regression

| Meta-Model | When to Use |
|------------|-------------|
| Ridge | Default choice |
| ElasticNet | When sparsity might help |
| Linear SVR | Alternative to Ridge |

### Meta-Model Complexity

```python
# CORRECT: Simple meta-model
meta_model = LogisticRegression(C=0.1)  # Regularized

# RISKY: Complex meta-model may overfit
meta_model = GradientBoostingClassifier(n_estimators=200, max_depth=5)

# General rule: Keep meta-model simpler than base models
```

## Practical Considerations

### Cross-Validation for Final Evaluation

```python
from sklearn.model_selection import cross_val_score

# Evaluate stacking with CV
stacking_clf = StackingClassifier(
    estimators=base_estimators,
    final_estimator=LogisticRegression(),
    cv=5
)

# Outer CV for unbiased performance estimate
cv_scores = cross_val_score(stacking_clf, X, y, cv=5, scoring='accuracy')
print(f"Stacking CV accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})")
```

### Handling Different Feature Scales

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Create pipelines for models that need scaling
base_estimators = [
    ('rf', RandomForestClassifier(n_estimators=100)),  # No scaling needed
    ('svc', Pipeline([
        ('scaler', StandardScaler()),
        ('svc', SVC(probability=True))
    ])),
    ('mlp', Pipeline([
        ('scaler', StandardScaler()),
        ('mlp', MLPClassifier(hidden_layer_sizes=(100,)))
    ]))
]
```

### Computational Cost

```python
# Estimate training time
# Stacking with 5-fold CV: ~5x training time of base models
# Plus meta-model fitting

# Speed up with parallel training
from joblib import Parallel, delayed

def fit_base_model(model, X, y):
    return model.fit(X, y)

# Fit base models in parallel
fitted_models = Parallel(n_jobs=-1)(
    delayed(fit_base_model)(clone(m), X_train, y_train)
    for m in base_models
)
```

## When Ensembles Help Most

1. **Competition settings**: Marginal improvements matter
2. **High-stakes predictions**: Reduce variance
3. **Diverse data**: Different models capture different patterns
4. **Uncertainty estimation**: Ensemble variance as uncertainty

## When to Avoid Complex Ensembles

1. **Need interpretability**: Stick to single models
2. **Production latency constraints**: Simpler models faster
3. **Small datasets**: Overfitting risk
4. **Simple problems**: Diminishing returns

## Best Practices Summary

1. **Start with model averaging** - Simplest, often effective
2. **Ensure base model diversity** - Different algorithms, hyperparameters
3. **Use simple meta-models** - Regularized linear models
4. **Cross-validate properly** - Nested CV for unbiased estimates
5. **Check for correlation** - Low correlation = better ensemble
6. **Mind computational cost** - Stacking is expensive
7. **Consider blending** - When CV is too slow
8. **Don't over-ensemble** - Diminishing returns after 3-5 models
