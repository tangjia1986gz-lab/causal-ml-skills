# Causal Inference Applications of Tree Models

## Overview

Tree-based models play a crucial role in modern causal inference, particularly in Double/Debiased Machine Learning (DDML). This reference covers their use as nuisance function estimators, the connection to causal forests, and best practices for causal applications.

---

## Role in Double/Debiased Machine Learning (DDML)

### DDML Framework Recap

DDML estimates causal effects by:
1. Using ML models to estimate nuisance functions (propensity scores, outcome models)
2. Using sample splitting (cross-fitting) to avoid overfitting bias
3. Constructing orthogonal moment conditions that are robust to first-stage errors

The partially linear model:
$$Y = D\theta + g(X) + \epsilon$$
$$D = m(X) + \eta$$

where:
- $g(X) = E[Y|X]$ (outcome nuisance)
- $m(X) = E[D|X]$ (propensity nuisance for continuous) or $P(D=1|X)$ (for binary)
- $\theta$ is the causal parameter of interest

### Why Tree Models Excel for DDML

1. **Automatic nonlinearity**: Capture complex relationships without functional form specification
2. **Robustness**: Handle outliers, missing values, and irregular patterns
3. **Scalability**: Handle high-dimensional confounders
4. **No curse of dimensionality**: Trees adapt to local structure
5. **Built-in regularization**: Via depth limits, minimum samples, ensemble averaging

---

## Nuisance Function Estimation

### Outcome Model: E[Y|X]

```python
from tree_models import fit_xgboost, fit_lightgbm

# Outcome model with conservative regularization
def train_outcome_model(X, Y, model_type='xgboost'):
    """Train outcome regression model for DDML."""

    if model_type == 'xgboost':
        return fit_xgboost(
            X, Y,
            task='regression',
            params={
                'max_depth': 5,
                'learning_rate': 0.05,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'reg_alpha': 0.1,
                'reg_lambda': 1.0,
                'min_child_weight': 3
            },
            n_rounds=300,
            early_stopping_rounds=30
        )
    elif model_type == 'lightgbm':
        return fit_lightgbm(
            X, Y,
            task='regression',
            params={
                'num_leaves': 31,
                'learning_rate': 0.05,
                'feature_fraction': 0.8,
                'bagging_fraction': 0.8,
                'bagging_freq': 5,
                'lambda_l1': 0.1,
                'lambda_l2': 1.0,
                'min_data_in_leaf': 20
            },
            n_rounds=300,
            early_stopping_rounds=30
        )
```

### Propensity Model: P(D=1|X) or E[D|X]

```python
from tree_models import fit_random_forest, fit_xgboost

def train_propensity_model(X, D, model_type='random_forest'):
    """Train propensity score model for DDML."""

    # Check if binary or continuous treatment
    is_binary = len(np.unique(D)) == 2
    task = 'classification' if is_binary else 'regression'

    if model_type == 'random_forest':
        # RF often gives better calibrated probabilities
        return fit_random_forest(
            X, D,
            task=task,
            n_estimators=200,
            max_depth=None,
            min_samples_leaf=10,
            max_features='sqrt',
            oob_score=True
        )
    elif model_type == 'xgboost':
        # Balance classes for propensity estimation
        if is_binary:
            n_pos = D.sum()
            n_neg = len(D) - n_pos
            scale_pos_weight = n_neg / n_pos
        else:
            scale_pos_weight = 1

        return fit_xgboost(
            X, D,
            task=task,
            params={
                'max_depth': 4,  # Shallower for propensity
                'learning_rate': 0.05,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'scale_pos_weight': scale_pos_weight,
                'reg_lambda': 2.0  # More regularization
            },
            n_rounds=200,
            early_stopping_rounds=30
        )
```

### Propensity Score Clipping

Extreme propensity scores can destabilize DDML estimates. Always clip:

```python
def clip_propensity_scores(ps, lower=0.01, upper=0.99):
    """Clip propensity scores to avoid extreme weights."""
    return np.clip(ps, lower, upper)

# Example usage
ps_raw = propensity_model['model'].predict_proba(X)[:, 1]
ps_clipped = clip_propensity_scores(ps_raw, lower=0.05, upper=0.95)

# Diagnostics
print(f"Raw PS range: [{ps_raw.min():.4f}, {ps_raw.max():.4f}]")
print(f"Clipped PS range: [{ps_clipped.min():.4f}, {ps_clipped.max():.4f}]")
print(f"Samples clipped: {(ps_raw != ps_clipped).sum()}")
```

---

## Cross-Fitting Implementation

### K-Fold Cross-Fitting

```python
import numpy as np
from sklearn.model_selection import KFold

def cross_fit_nuisance(X, Y, D, n_folds=5, random_state=42):
    """
    Implement cross-fitting for DDML nuisance estimation.

    Returns out-of-fold predictions for all observations.
    """
    n = len(Y)
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)

    # Storage for out-of-fold predictions
    Y_hat = np.zeros(n)  # E[Y|X] predictions
    D_hat = np.zeros(n)  # E[D|X] or P(D=1|X) predictions

    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        X_train, X_val = X[train_idx], X[val_idx]
        Y_train, Y_val = Y[train_idx], Y[val_idx]
        D_train, D_val = D[train_idx], D[val_idx]

        # Train outcome model on training fold
        outcome_model = train_outcome_model(X_train, Y_train)
        Y_hat[val_idx] = outcome_model['model'].predict(X_val)

        # Train propensity model on training fold
        propensity_model = train_propensity_model(X_train, D_train)
        if len(np.unique(D)) == 2:
            D_hat[val_idx] = propensity_model['model'].predict_proba(X_val)[:, 1]
        else:
            D_hat[val_idx] = propensity_model['model'].predict(X_val)

    return Y_hat, D_hat


def ddml_ate(Y, D, Y_hat, D_hat, ps_clip=(0.05, 0.95)):
    """
    Compute ATE using DDML with pre-computed nuisance predictions.

    Uses the doubly-robust estimator (AIPW).
    """
    # Clip propensity scores
    D_hat_clipped = np.clip(D_hat, ps_clip[0], ps_clip[1])

    # Compute residuals
    Y_res = Y - Y_hat
    D_res = D - D_hat_clipped

    # DDML estimator: regress Y residuals on D residuals
    # theta = Cov(Y_res, D_res) / Var(D_res)
    theta = np.mean(Y_res * D_res) / np.mean(D_res ** 2)

    # Standard error (Neyman-style)
    n = len(Y)
    psi = (Y_res - theta * D_res) * D_res / np.mean(D_res ** 2)
    se = np.sqrt(np.var(psi) / n)

    return {
        'ate': theta,
        'se': se,
        'ci_lower': theta - 1.96 * se,
        'ci_upper': theta + 1.96 * se,
        'n': n
    }


# Full DDML pipeline
def run_ddml(X, Y, D, n_folds=5):
    """Complete DDML pipeline for ATE estimation."""

    # Step 1: Cross-fitted nuisance predictions
    Y_hat, D_hat = cross_fit_nuisance(X, Y, D, n_folds=n_folds)

    # Step 2: Compute ATE
    result = ddml_ate(Y, D, Y_hat, D_hat)

    print(f"DDML ATE: {result['ate']:.4f}")
    print(f"Standard Error: {result['se']:.4f}")
    print(f"95% CI: [{result['ci_lower']:.4f}, {result['ci_upper']:.4f}]")

    return result
```

---

## Honest Trees and Causal Forests

### What Makes Trees "Honest"?

Standard trees use the same data to:
1. Determine split points (structure)
2. Estimate leaf predictions (estimation)

**Honest trees** use sample splitting:
1. **Structure sample**: Determines splits
2. **Estimation sample**: Computes leaf predictions

This provides valid inference with:
- Asymptotically normal estimates
- Valid confidence intervals
- Correct coverage properties

### Connection to Causal Forests

Causal forests (Athey & Wager) modify random forests for treatment effect estimation:

| Aspect | Prediction Forest | Causal Forest |
|--------|------------------|---------------|
| Target | E[Y|X] | tau(X) = E[Y(1)-Y(0)|X] |
| Splitting criterion | MSE reduction | Treatment effect heterogeneity |
| Honesty | Optional | Required |
| Estimation | Mean of Y | Difference in treated/control means |

```python
# Standard prediction forest
from tree_models import fit_random_forest
outcome_model = fit_random_forest(X, Y, task='regression')

# Causal forest (requires econml or grf)
# See causal-forest skill for full implementation
from econml.dml import CausalForestDML

causal_forest = CausalForestDML(
    model_y=fit_random_forest(X, Y)['model'],
    model_t=fit_random_forest(X, D)['model'],
    n_estimators=100,
    min_samples_leaf=10,
    honest=True  # Key difference!
)
causal_forest.fit(Y, D, X=X, W=None)

# Get heterogeneous treatment effects
cate = causal_forest.effect(X)  # tau(x) for each observation
```

### Using Tree Models to Inform Causal Forests

Tree-based nuisance models can be combined with causal forests:

```python
from econml.dml import CausalForestDML
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

# Custom nuisance models
model_y = RandomForestRegressor(n_estimators=200, min_samples_leaf=5)
model_t = RandomForestClassifier(n_estimators=200, min_samples_leaf=5)

cf = CausalForestDML(
    model_y=model_y,
    model_t=model_t,
    discrete_treatment=True,
    n_estimators=200,
    honest=True,
    inference=True  # Enable confidence intervals
)

cf.fit(Y, D, X=X)

# Point estimates
tau = cf.effect(X)

# Confidence intervals
tau_lower, tau_upper = cf.effect_interval(X, alpha=0.05)
```

---

## Best Practices for Causal Applications

### 1. Model Selection for Nuisance Functions

| Scenario | Outcome Model | Propensity Model |
|----------|---------------|------------------|
| Small n (< 1,000) | Random Forest | Random Forest |
| Medium n (1,000-10,000) | XGBoost | Random Forest or XGBoost |
| Large n (> 10,000) | LightGBM | XGBoost or LightGBM |
| Many categoricals | CatBoost | CatBoost |

### 2. Regularization for Causal Stability

```python
# Conservative defaults for causal applications
causal_params = {
    # More regularization than prediction tasks
    'max_depth': 5,  # Not too deep
    'learning_rate': 0.05,  # Conservative
    'min_child_weight': 10,  # Larger leaves
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'reg_alpha': 0.5,  # Some L1
    'reg_lambda': 2.0  # More L2
}
```

### 3. Diagnostics for Nuisance Models

```python
def diagnose_nuisance_models(X, Y, D, Y_hat, D_hat):
    """Diagnostic checks for DDML nuisance models."""

    from sklearn.metrics import r2_score, roc_auc_score

    print("=== Nuisance Model Diagnostics ===\n")

    # Outcome model performance
    r2_outcome = r2_score(Y, Y_hat)
    print(f"Outcome Model R2: {r2_outcome:.4f}")

    # Propensity model performance
    if len(np.unique(D)) == 2:
        auc_propensity = roc_auc_score(D, D_hat)
        print(f"Propensity Model AUC: {auc_propensity:.4f}")
    else:
        r2_propensity = r2_score(D, D_hat)
        print(f"Treatment Model R2: {r2_propensity:.4f}")

    # Propensity score distribution
    print(f"\nPropensity Score Distribution:")
    print(f"  Min: {D_hat.min():.4f}")
    print(f"  25%: {np.percentile(D_hat, 25):.4f}")
    print(f"  50%: {np.percentile(D_hat, 50):.4f}")
    print(f"  75%: {np.percentile(D_hat, 75):.4f}")
    print(f"  Max: {D_hat.max():.4f}")

    # Check for extreme weights
    extreme_low = (D_hat < 0.05).sum()
    extreme_high = (D_hat > 0.95).sum()
    print(f"\nExtreme propensity scores:")
    print(f"  < 0.05: {extreme_low} ({100*extreme_low/len(D_hat):.1f}%)")
    print(f"  > 0.95: {extreme_high} ({100*extreme_high/len(D_hat):.1f}%)")

    # Residual diagnostics
    Y_res = Y - Y_hat
    D_res = D - D_hat

    print(f"\nResidual Statistics:")
    print(f"  Y residuals: mean={Y_res.mean():.4f}, std={Y_res.std():.4f}")
    print(f"  D residuals: mean={D_res.mean():.4f}, std={D_res.std():.4f}")

    # Check residual correlation (should be close to zero after partialing out)
    corr = np.corrcoef(Y_res, D_res)[0, 1]
    print(f"  Correlation(Y_res, D_res): {corr:.4f}")
```

### 4. Sensitivity Analysis

```python
def ddml_sensitivity_analysis(X, Y, D, n_folds_options=[3, 5, 10],
                               model_options=['random_forest', 'xgboost', 'lightgbm']):
    """
    Check sensitivity of DDML results to modeling choices.
    """
    results = []

    for n_folds in n_folds_options:
        for model_type in model_options:
            # Run DDML with these settings
            Y_hat, D_hat = cross_fit_nuisance(X, Y, D, n_folds=n_folds)
            result = ddml_ate(Y, D, Y_hat, D_hat)

            results.append({
                'n_folds': n_folds,
                'model': model_type,
                'ate': result['ate'],
                'se': result['se']
            })

    results_df = pd.DataFrame(results)
    print("DDML Sensitivity Analysis:")
    print(results_df.to_string(index=False))

    # Check consistency
    ate_range = results_df['ate'].max() - results_df['ate'].min()
    print(f"\nATE range across specifications: {ate_range:.4f}")

    return results_df
```

---

## Common Pitfalls and Solutions

### Pitfall 1: Overfitting Nuisance Models

**Problem**: Nuisance models memorize training data, leading to biased cross-fitted predictions.

**Solution**:
- Use regularization (depth limits, L1/L2, sampling)
- Monitor out-of-fold performance
- Use early stopping

### Pitfall 2: Extreme Propensity Scores

**Problem**: Near-zero or near-one propensity scores create extreme IPW weights.

**Solution**:
- Clip propensity scores (e.g., to [0.05, 0.95])
- Use doubly-robust estimators (less sensitive to PS extremes)
- Check overlap: ensure treatment and control groups overlap in X-space

### Pitfall 3: Positivity Violations

**Problem**: Some covariate regions have no treated or control units.

**Solution**:
- Examine propensity score distributions by treatment group
- Trim sample to common support region
- Use overlap weights

```python
def check_positivity(ps, D, threshold=0.1):
    """Check positivity assumption."""

    # PS distribution by treatment group
    ps_treated = ps[D == 1]
    ps_control = ps[D == 0]

    print("Propensity Score Overlap:")
    print(f"  Treated: [{ps_treated.min():.3f}, {ps_treated.max():.3f}]")
    print(f"  Control: [{ps_control.min():.3f}, {ps_control.max():.3f}]")

    # Proportion in overlap region
    overlap_region = (ps > threshold) & (ps < 1 - threshold)
    print(f"  In overlap region ({threshold}, {1-threshold}): {overlap_region.mean()*100:.1f}%")

    return overlap_region
```

### Pitfall 4: Model Misspecification

**Problem**: Nuisance models fail to capture true relationships.

**Solution**:
- Use flexible models (trees, neural networks)
- Compare multiple model types
- Check residual patterns

---

## References

- Chernozhukov, V., et al. (2018). Double/debiased machine learning for treatment and structural parameters.
- Athey, S., & Imbens, G. W. (2019). Machine learning methods that economists should know about.
- Athey, S., & Wager, S. (2019). Estimating treatment effects with causal forests: An application.
- Kennedy, E. H. (2020). Towards optimal doubly robust estimation of heterogeneous causal effects.
- Chernozhukov, V., et al. (2022). Automatic debiased machine learning of causal and structural effects.
