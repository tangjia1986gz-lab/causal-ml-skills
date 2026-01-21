# Neural Networks for Tabular Data

## Overview

Multi-Layer Perceptrons (MLPs) are universal function approximators that can learn complex non-linear relationships. However, for tabular data, they often underperform compared to gradient boosting methods unless carefully tuned.

## When to Use Neural Networks for Tabular Data

### Good Use Cases

1. **Very large datasets** (>100,000 samples): Neural networks benefit from scale
2. **High-dimensional feature spaces**: When features exceed samples
3. **Complex interactions**: Non-linear patterns that tree-based models miss
4. **Transfer learning**: Pre-trained embeddings for categorical features
5. **Multi-task learning**: Shared representations across related tasks

### When to Avoid

1. **Small datasets** (<1,000 samples): Prefer simpler models
2. **Need interpretability**: Use linear models or trees
3. **Limited compute**: Tree-based models train faster
4. **Sparse data**: Gradient boosting often works better

## Architecture Design Principles

### Network Depth vs Width

```
Shallow & Wide:          Deep & Narrow:
(256,)                   (64, 64, 64, 64)
- Faster training        - Better feature hierarchies
- Lower capacity         - Higher capacity
- Good for linear        - Good for complex patterns
  relationships
```

**Rule of Thumb for Tabular Data:**
- Start with 2-3 hidden layers
- Width: 64-256 neurons per layer
- Gradually decrease width: (128, 64, 32)

### Architecture by Dataset Size

| Dataset Size | Recommended Architecture |
|--------------|--------------------------|
| < 1,000 | (32,) or (64,) |
| 1,000 - 10,000 | (64, 32) or (100, 50) |
| 10,000 - 100,000 | (128, 64, 32) |
| > 100,000 | (256, 128, 64) or deeper |

### Activation Functions

```python
# ReLU - Default choice for hidden layers
# Fast, sparse activations, but can have "dead neurons"
activation='relu'

# Leaky ReLU alternative (requires custom implementation)
# Addresses dead neuron problem

# tanh - Output in [-1, 1]
# Better gradient flow, but slower
activation='tanh'

# For output layer:
# - Classification: softmax (multi-class) or sigmoid (binary)
# - Regression: linear (identity)
```

## Regularization Techniques

### L2 Regularization (Weight Decay)

```python
from sklearn.neural_network import MLPClassifier

# alpha controls L2 penalty strength
mlp = MLPClassifier(
    hidden_layer_sizes=(100, 50),
    alpha=0.01  # Increase for stronger regularization
)
```

**Tuning alpha:**
- Start with 0.0001 (sklearn default)
- If overfitting: increase to 0.001, 0.01, 0.1
- If underfitting: decrease to 0.00001

### Early Stopping

```python
mlp = MLPClassifier(
    hidden_layer_sizes=(100, 50),
    early_stopping=True,
    validation_fraction=0.1,  # Use 10% for validation
    n_iter_no_change=10       # Stop after 10 epochs without improvement
)
```

### Dropout (PyTorch/TensorFlow)

sklearn's MLPClassifier does not support dropout. For dropout regularization:

```python
import torch.nn as nn

class TabularMLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim, dropout=0.3):
        super().__init__()
        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)
```

### Batch Normalization

Normalizes activations between layers. Helps with:
- Faster training
- Higher learning rates
- Less sensitive to initialization

```python
# PyTorch implementation
class TabularMLPWithBN(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim):
        super().__init__()
        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
            ])
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)
```

## Hyperparameter Tuning Strategy

### Priority Order

1. **Learning rate** - Most important
2. **Architecture** (depth and width)
3. **Regularization** (alpha, dropout)
4. **Batch size**
5. **Optimizer settings**

### Learning Rate Schedule

```python
# sklearn: adaptive learning rate
mlp = MLPClassifier(
    solver='sgd',
    learning_rate='adaptive',  # Divides by 5 when loss stagnates
    learning_rate_init=0.01
)

# For adam solver, start with 0.001
mlp = MLPClassifier(
    solver='adam',
    learning_rate_init=0.001
)
```

### Grid Search Example

```python
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

param_grid = {
    'mlp__hidden_layer_sizes': [(64,), (100, 50), (128, 64, 32)],
    'mlp__alpha': [0.0001, 0.001, 0.01],
    'mlp__learning_rate_init': [0.0001, 0.001, 0.01],
    'mlp__activation': ['relu', 'tanh']
}

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('mlp', MLPClassifier(max_iter=500, early_stopping=True, random_state=42))
])

search = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
search.fit(X, y)
```

## Feature Engineering for Neural Networks

### Numerical Features

```python
from sklearn.preprocessing import StandardScaler, QuantileTransformer

# Standard scaling - required for MLPs
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Quantile transformation - makes features more Gaussian
# Can help with skewed distributions
quantile = QuantileTransformer(output_distribution='normal')
X_transformed = quantile.fit_transform(X)
```

### Categorical Features

```python
from sklearn.preprocessing import OneHotEncoder

# One-hot encoding for low cardinality
encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
X_encoded = encoder.fit_transform(X_categorical)

# For high cardinality: use embeddings (PyTorch/TensorFlow)
# Or target encoding as preprocessing step
```

### Missing Values

Neural networks do not handle missing values natively:

```python
from sklearn.impute import SimpleImputer, KNNImputer

# Simple imputation
imputer = SimpleImputer(strategy='median')
X_imputed = imputer.fit_transform(X)

# KNN imputation for better estimates
knn_imputer = KNNImputer(n_neighbors=5)
X_imputed = knn_imputer.fit_transform(X)
```

## Common Pitfalls

### 1. No Feature Scaling

```python
# WRONG
mlp = MLPClassifier()
mlp.fit(X, y)  # Features not scaled!

# CORRECT
from sklearn.pipeline import Pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('mlp', MLPClassifier())
])
pipeline.fit(X, y)
```

### 2. Ignoring Convergence Warnings

```python
mlp = MLPClassifier(max_iter=500)
mlp.fit(X, y)

# Check convergence
if mlp.n_iter_ == mlp.max_iter:
    print("WARNING: Model did not converge!")
    print("Consider: increasing max_iter, adjusting learning_rate, or reducing alpha")
```

### 3. Overfitting on Small Data

```python
# WRONG for small dataset
mlp_overfit = MLPClassifier(
    hidden_layer_sizes=(512, 256, 128, 64),  # Too complex
    alpha=0.0001  # Weak regularization
)

# CORRECT for small dataset
mlp_simple = MLPClassifier(
    hidden_layer_sizes=(32,),  # Simple architecture
    alpha=0.01,  # Strong regularization
    early_stopping=True
)
```

### 4. Not Using Early Stopping

```python
# RISKY - may overfit
mlp = MLPClassifier(max_iter=1000)

# SAFER - stops when validation loss increases
mlp = MLPClassifier(
    max_iter=1000,
    early_stopping=True,
    n_iter_no_change=10
)
```

## Comparison: Neural Networks vs Gradient Boosting

| Aspect | Neural Networks | Gradient Boosting |
|--------|----------------|-------------------|
| Feature scaling | Required | Not required |
| Missing values | Must impute | Some handle natively |
| Categorical features | Encoding required | Native support (some) |
| Small data (<1k) | Often underperforms | Usually wins |
| Large data (>100k) | Competitive | Still strong |
| Training time | Longer | Faster |
| Interpretability | Very low | Medium (SHAP) |
| Hyperparameter tuning | Critical | Less sensitive |

## Best Practices Summary

1. **Always use pipelines** with StandardScaler
2. **Start simple**: (64, 32) before trying deeper networks
3. **Enable early stopping** to prevent overfitting
4. **Monitor convergence**: Check n_iter_ vs max_iter
5. **Tune learning rate first**, then architecture
6. **Consider gradient boosting** as strong baseline
7. **Use cross-validation** for hyperparameter selection
8. **Handle missing values** before training
