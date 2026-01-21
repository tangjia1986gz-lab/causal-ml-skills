# Feature Importance Methods

## Overview

Understanding which features drive predictions is crucial for model interpretation and debugging. This reference covers three main approaches: impurity-based importance, permutation importance, and SHAP values.

---

## Impurity-Based Importance

### How It Works

Impurity-based importance (also called Gini importance or Mean Decrease Impurity) measures the total reduction in impurity (variance for regression, Gini/entropy for classification) achieved by splits on each feature across all trees.

$$\text{Importance}(X_j) = \sum_{t \in T: v(t) = j} \frac{n_t}{n} \cdot \Delta I_t$$

where:
- $T$ is the set of all nodes in all trees
- $v(t)$ is the feature used for splitting at node $t$
- $n_t / n$ is the fraction of samples reaching node $t$
- $\Delta I_t$ is the impurity decrease at node $t$

### Usage

```python
from tree_models import get_feature_importance

# Impurity-based importance (fast)
importance = get_feature_importance(
    model=rf_model,
    feature_names=feature_names,
    method='impurity'
)

print(importance.head(10))
# Output:
#        feature  importance
# 0       income      0.2534
# 1          age      0.1823
# 2    education      0.1245
# ...
```

### Advantages
- Very fast to compute (no additional predictions needed)
- Available for all tree-based models
- Built-in to sklearn, XGBoost, LightGBM

### Limitations
- **Biased toward high-cardinality features**: Features with many unique values get more split opportunities
- **Biased toward continuous features**: Numerics preferred over categoricals
- **Not based on generalization**: Reflects training data, not test performance
- **Unreliable with correlated features**: Importance gets split among correlated features

### When to Use
- Quick exploration during model development
- When computational resources are limited
- As a first-pass feature screening tool
- **Not recommended** for final feature importance conclusions

---

## Permutation Importance

### How It Works

Permutation importance measures the decrease in model performance when a feature's values are randomly shuffled, breaking its relationship with the target.

$$\text{PI}(X_j) = s - \frac{1}{K}\sum_{k=1}^{K} s_{\pi_j}^{(k)}$$

where:
- $s$ is the original model score
- $s_{\pi_j}^{(k)}$ is the score after shuffling feature $j$ in repetition $k$

### Usage

```python
from tree_models import get_feature_importance

# Permutation importance (more reliable)
importance = get_feature_importance(
    model=rf_model,
    feature_names=feature_names,
    method='permutation',
    X=X_test,  # Use test data for unbiased estimates
    y=y_test,
    n_repeats=10
)

print(importance.head(10))
# Output:
#        feature  importance       std
# 0       income      0.1523    0.0089
# 1          age      0.0892    0.0056
# 2    education      0.0654    0.0043
# ...
```

### Advantages
- **Model-agnostic**: Works with any model
- **Based on test performance**: Reflects generalization
- **Unbiased**: No preference for high-cardinality features
- **Provides uncertainty estimates**: Via standard deviation across repeats

### Limitations
- **Slower**: Requires multiple model predictions
- **Correlated features**: May underestimate importance of correlated features
- **Extrapolation**: Shuffling creates unrealistic feature combinations

### When to Use
- Final feature importance rankings
- Comparing feature importance across different models
- When accuracy matters more than speed
- **Recommended** for DDML diagnostics

### Best Practices

```python
# Use test set for unbiased permutation importance
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
model.fit(X_train, y_train)

# Compute on test set
importance = get_feature_importance(
    model, feature_names,
    method='permutation',
    X=X_test, y=y_test,  # Test data!
    n_repeats=30  # More repeats for stability
)
```

---

## SHAP Values

### How It Works

SHAP (SHapley Additive exPlanations) uses game-theoretic Shapley values to attribute each prediction to individual features. For each prediction:

$$f(x) = \phi_0 + \sum_{j=1}^{M} \phi_j$$

where:
- $f(x)$ is the model prediction
- $\phi_0$ is the base value (expected prediction)
- $\phi_j$ is the SHAP value for feature $j$

The Shapley value for feature $j$:

$$\phi_j = \sum_{S \subseteq N \setminus \{j\}} \frac{|S|!(M-|S|-1)!}{M!} [f(S \cup \{j\}) - f(S)]$$

### Usage

```python
from tree_models import compute_shap_values

# Compute SHAP values
shap_result = compute_shap_values(
    model=xgb_model,
    X=X_test,
    feature_names=feature_names,
    plot_summary=True,
    plot_bar=True
)

# Global feature importance from SHAP
print(shap_result['feature_importance'].head(10))

# Access raw SHAP values for custom analysis
shap_values = shap_result['shap_values']  # Shape: (n_samples, n_features)
```

### TreeSHAP

For tree-based models, TreeSHAP provides exact, fast computation:

```python
import shap

# TreeSHAP is used automatically for tree models
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)
```

### SHAP Visualizations

#### Summary Plot
Shows feature importance and effect direction for all samples:

```python
import shap
shap.summary_plot(shap_values, X, feature_names=feature_names)
```

#### Bar Plot
Global feature importance (mean |SHAP|):

```python
shap.summary_plot(shap_values, X, plot_type='bar')
```

#### Dependence Plot
Shows how a feature affects predictions:

```python
shap.dependence_plot('income', shap_values, X, feature_names=feature_names)
```

#### Force Plot (Single Prediction)
Explains individual predictions:

```python
shap.force_plot(explainer.expected_value, shap_values[0], X.iloc[0])
```

### Advantages
- **Theoretically grounded**: Based on cooperative game theory
- **Local and global**: Explains both individual predictions and overall model
- **Consistent**: Satisfies desirable axioms
- **Handles interactions**: Properly accounts for feature interactions
- **Fast for trees**: TreeSHAP is O(TL^2) where T=trees, L=leaves

### Limitations
- **Computational cost**: O(2^M) for exact Shapley values (but TreeSHAP is fast)
- **Interpretation complexity**: Understanding joint contributions
- **Background data**: May need representative background set

### When to Use
- Detailed model explanation for stakeholders
- Debugging model behavior
- Understanding individual predictions
- Regulatory/compliance requirements
- **Highly recommended** for DDML model diagnostics

---

## Comparison of Methods

| Aspect | Impurity-Based | Permutation | SHAP |
|--------|---------------|-------------|------|
| Speed | Fast | Medium | Medium (TreeSHAP) |
| Bias toward high-cardinality | Yes | No | No |
| Works with correlations | Poorly | Poorly | Better |
| Local explanations | No | No | Yes |
| Theoretical foundation | Weak | Moderate | Strong |
| Model-agnostic | No | Yes | Yes |
| DDML recommendation | Avoid | Use | Use |

---

## Feature Importance for DDML

### Why Feature Importance Matters in DDML

1. **Confounder verification**: Confirm important controls are included
2. **Model diagnostics**: Ensure nuisance models capture relevant relationships
3. **Interpretation**: Understand which factors drive treatment/outcome
4. **Dimensionality reduction**: Identify irrelevant features to remove

### DDML-Specific Analysis

```python
from tree_models import fit_xgboost, get_feature_importance, compute_shap_values

# Train outcome model
outcome_model = fit_xgboost(X, Y, task='regression', params=ddml_params)

# Train propensity model
propensity_model = fit_xgboost(X, D, task='classification', params=ddml_params)

# Compare feature importance between models
outcome_importance = get_feature_importance(
    outcome_model['model'], feature_names, method='permutation', X=X, y=Y
)
propensity_importance = get_feature_importance(
    propensity_model['model'], feature_names, method='permutation', X=X, y=D
)

# Features important for both = potential confounders
import pandas as pd

comparison = pd.merge(
    outcome_importance.rename(columns={'importance': 'outcome_imp'}),
    propensity_importance.rename(columns={'importance': 'propensity_imp'}),
    on='feature'
)
comparison['total_imp'] = comparison['outcome_imp'] + comparison['propensity_imp']
comparison = comparison.sort_values('total_imp', ascending=False)

print("Key confounders (important for both Y and D):")
print(comparison.head(10))
```

### Verifying Confounder Control

```python
# SHAP analysis for confounding patterns
shap_outcome = compute_shap_values(outcome_model['model'], X, feature_names)
shap_propensity = compute_shap_values(propensity_model['model'], X, feature_names)

# Check if treatment assignment depends on important outcome predictors
# High overlap = important to control for
outcome_top = set(shap_outcome['feature_importance'].head(10)['feature'])
propensity_top = set(shap_propensity['feature_importance'].head(10)['feature'])

confounders = outcome_top & propensity_top
print(f"Potential confounders: {confounders}")
```

---

## Handling Correlated Features

All importance methods struggle with highly correlated features. Here are strategies:

### 1. Pre-screening with Correlation Analysis

```python
import numpy as np
from scipy import stats

# Compute correlation matrix
corr_matrix = np.corrcoef(X.T)

# Identify highly correlated pairs
high_corr_pairs = []
for i in range(len(feature_names)):
    for j in range(i+1, len(feature_names)):
        if abs(corr_matrix[i, j]) > 0.7:
            high_corr_pairs.append((feature_names[i], feature_names[j], corr_matrix[i, j]))

print("Highly correlated feature pairs:")
for f1, f2, corr in high_corr_pairs:
    print(f"  {f1} <-> {f2}: {corr:.3f}")
```

### 2. Grouped Permutation Importance

```python
# Permute correlated features together
def grouped_permutation_importance(model, X, y, feature_groups, n_repeats=10):
    """Compute permutation importance for feature groups."""
    from sklearn.metrics import r2_score

    base_score = r2_score(y, model.predict(X))
    group_importance = {}

    for group_name, features in feature_groups.items():
        feature_idx = [list(X.columns).index(f) for f in features]
        scores = []

        for _ in range(n_repeats):
            X_permuted = X.copy()
            # Permute all features in group together
            perm_idx = np.random.permutation(len(X))
            for idx in feature_idx:
                X_permuted.iloc[:, idx] = X_permuted.iloc[perm_idx, idx]

            scores.append(r2_score(y, model.predict(X_permuted)))

        group_importance[group_name] = base_score - np.mean(scores)

    return group_importance
```

### 3. SHAP Interaction Values

```python
# Compute SHAP interaction values for understanding correlated features
explainer = shap.TreeExplainer(model)
shap_interaction = explainer.shap_interaction_values(X)

# shap_interaction shape: (n_samples, n_features, n_features)
# Diagonal: main effects; Off-diagonal: interactions
```

---

## Practical Workflow

### Step 1: Quick Exploration (Impurity)

```python
# Fast initial screening
importance = get_feature_importance(model, feature_names, method='impurity')
print("Quick feature ranking:")
print(importance.head(20))
```

### Step 2: Reliable Ranking (Permutation)

```python
# Unbiased importance on test data
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
model.fit(X_train, y_train)

importance = get_feature_importance(
    model, feature_names, method='permutation',
    X=X_test, y=y_test, n_repeats=30
)
print("Reliable feature ranking:")
print(importance.head(20))
```

### Step 3: Deep Dive (SHAP)

```python
# Detailed analysis for final reporting
shap_result = compute_shap_values(model, X_test, feature_names)

# Summary visualization
import shap
shap.summary_plot(shap_result['shap_values'], X_test, feature_names=feature_names)

# Individual prediction explanations
idx = 0  # Explain first prediction
shap.force_plot(
    shap_result['expected_value'],
    shap_result['shap_values'][idx],
    X_test.iloc[idx]
)
```

---

## References

- Breiman, L. (2001). Random forests. Machine learning, 45(1), 5-32. (Impurity importance)
- Altmann, A., et al. (2010). Permutation importance: a corrected feature importance measure.
- Lundberg, S. M., & Lee, S. I. (2017). A unified approach to interpreting model predictions. (SHAP)
- Lundberg, S. M., et al. (2020). From local explanations to global understanding with explainable AI for trees.
- Molnar, C. (2022). Interpretable Machine Learning. https://christophm.github.io/interpretable-ml-book/
