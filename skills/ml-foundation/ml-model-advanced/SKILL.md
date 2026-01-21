---
name: ml-model-advanced
triggers:
  - SVM
  - support vector machine
  - neural network
  - MLP
  - deep learning
  - kernel methods
  - multilayer perceptron
  - backpropagation
  - kernel trick
---

# Advanced Machine Learning Models

Advanced ML models for complex prediction tasks including Support Vector Machines (SVM) and Neural Networks (MLP).

## Overview

This skill covers two powerful model families that extend beyond tree-based methods:

1. **Support Vector Machines (SVM)**: Margin-based classifiers using the kernel trick
2. **Neural Networks (MLP)**: Multi-layer perceptrons for learning complex patterns

These models are particularly useful when:
- Data has complex, non-linear relationships
- High-dimensional feature spaces
- Need flexible function approximation
- Tree-based models underperform

## Support Vector Machines (SVM)

### Conceptual Foundation

SVMs find the optimal hyperplane that maximizes the margin between classes. Key concepts:

- **Support Vectors**: Data points closest to the decision boundary
- **Margin**: Distance between the hyperplane and nearest support vectors
- **Kernel Trick**: Implicitly map data to higher dimensions without computing coordinates

### SVM for Classification (SVC)

```python
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# IMPORTANT: SVM requires standardized features
svm_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('svc', SVC(kernel='rbf', C=1.0, gamma='scale'))
])

svm_pipeline.fit(X_train, y_train)
predictions = svm_pipeline.predict(X_test)

# For probability estimates (needed for propensity scores)
svm_proba = Pipeline([
    ('scaler', StandardScaler()),
    ('svc', SVC(kernel='rbf', probability=True))
])
```

### SVM for Regression (SVR)

```python
from sklearn.svm import SVR

svr_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('svr', SVR(kernel='rbf', C=1.0, epsilon=0.1))
])

svr_pipeline.fit(X_train, y_train)
predictions = svr_pipeline.predict(X_test)
```

### Kernel Functions

| Kernel | Formula | Best For |
|--------|---------|----------|
| **Linear** | `K(x,y) = x·y` | Linearly separable data, high dimensions |
| **RBF (Gaussian)** | `K(x,y) = exp(-γ||x-y||²)` | Most common, general purpose |
| **Polynomial** | `K(x,y) = (γx·y + r)^d` | When feature interactions matter |
| **Sigmoid** | `K(x,y) = tanh(γx·y + r)` | Similar to neural networks |

### Hyperparameter Tuning

**Key parameters:**

- **C (Regularization)**: Trade-off between margin width and misclassification
  - Small C → Wider margin, more misclassification allowed
  - Large C → Narrower margin, fewer misclassifications

- **gamma (RBF kernel)**: Defines influence of single training example
  - Small gamma → Far reach, smoother decision boundary
  - Large gamma → Close reach, more complex boundary

- **epsilon (SVR only)**: Width of the epsilon-tube with no penalty

```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'svc__C': [0.1, 1, 10, 100],
    'svc__gamma': ['scale', 'auto', 0.1, 1],
    'svc__kernel': ['rbf', 'linear', 'poly']
}

grid_search = GridSearchCV(svm_pipeline, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)
print(f"Best params: {grid_search.best_params_}")
```

### SVM Pros and Cons

**Advantages:**
- Effective in high-dimensional spaces
- Memory efficient (only stores support vectors)
- Kernel trick allows flexible decision boundaries
- Works well with clear margin of separation
- Robust to overfitting in high dimensions

**Disadvantages:**
- Doesn't scale well to very large datasets (O(n²) to O(n³))
- Sensitive to feature scaling (must standardize!)
- No direct probability estimates (calibration needed)
- Choice of kernel and parameters is crucial
- Hard to interpret (black box)

## Neural Networks (MLP)

### Multi-Layer Perceptron Basics

An MLP consists of:
1. **Input Layer**: Receives features
2. **Hidden Layers**: Learn representations
3. **Output Layer**: Produces predictions

```
Input → [Hidden Layer 1] → [Hidden Layer 2] → Output
   x₁ →    h₁ = σ(W₁x + b₁)  →  h₂ = σ(W₂h₁ + b₂)  →  ŷ
```

### Architecture Components

**Layers and Neurons:**
```python
from sklearn.neural_network import MLPClassifier, MLPRegressor

# hidden_layer_sizes defines architecture
# (100,) = 1 hidden layer with 100 neurons
# (100, 50) = 2 hidden layers with 100 and 50 neurons
# (64, 32, 16) = 3 hidden layers

mlp = MLPClassifier(hidden_layer_sizes=(100, 50))
```

**Activation Functions:**

| Activation | Formula | Properties |
|------------|---------|------------|
| **ReLU** | `max(0, x)` | Default, fast, can have dead neurons |
| **tanh** | `(eˣ - e⁻ˣ)/(eˣ + e⁻ˣ)` | Output in [-1, 1], centered |
| **logistic** | `1/(1 + e⁻ˣ)` | Output in [0, 1], sigmoid |
| **identity** | `x` | Linear, for output layer in regression |

```python
mlp = MLPClassifier(
    hidden_layer_sizes=(100, 50),
    activation='relu'  # 'tanh', 'logistic', 'identity'
)
```

### Training Neural Networks

**Backpropagation and Optimization:**

```python
mlp = MLPClassifier(
    hidden_layer_sizes=(100, 50),
    solver='adam',           # 'sgd', 'adam', 'lbfgs'
    learning_rate_init=0.001,
    max_iter=200,
    batch_size='auto',       # or integer
    random_state=42
)
```

**Solvers:**
- **adam**: Adaptive learning rate, best for large datasets
- **sgd**: Stochastic gradient descent, more control
- **lbfgs**: Quasi-Newton, good for small datasets

**Learning Rate Schedules (for SGD):**
```python
mlp = MLPClassifier(
    solver='sgd',
    learning_rate='adaptive',  # 'constant', 'invscaling', 'adaptive'
    learning_rate_init=0.01
)
```

### Regularization Techniques

**L2 Regularization (alpha):**
```python
mlp = MLPClassifier(
    hidden_layer_sizes=(100, 50),
    alpha=0.0001  # L2 penalty, increase to reduce overfitting
)
```

**Early Stopping:**
```python
mlp = MLPClassifier(
    hidden_layer_sizes=(100, 50),
    early_stopping=True,
    validation_fraction=0.1,
    n_iter_no_change=10
)
```

**Note on Dropout:** sklearn's MLPClassifier doesn't support dropout. For dropout regularization, use deep learning frameworks (PyTorch, TensorFlow).

### Complete MLP Pipeline

```python
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# IMPORTANT: Neural networks require standardized features
mlp_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('mlp', MLPClassifier(
        hidden_layer_sizes=(128, 64),
        activation='relu',
        solver='adam',
        alpha=0.0001,
        learning_rate_init=0.001,
        max_iter=500,
        early_stopping=True,
        validation_fraction=0.1,
        random_state=42
    ))
])

mlp_pipeline.fit(X_train, y_train)
```

### MLP Hyperparameter Tuning

```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'mlp__hidden_layer_sizes': [(50,), (100,), (100, 50), (128, 64, 32)],
    'mlp__activation': ['relu', 'tanh'],
    'mlp__alpha': [0.0001, 0.001, 0.01],
    'mlp__learning_rate_init': [0.001, 0.01]
}

grid_search = GridSearchCV(mlp_pipeline, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)
```

### MLP Pros and Cons

**Advantages:**
- Universal function approximators
- Can learn complex non-linear patterns
- Flexible architecture design
- Can handle large feature spaces
- Gradient-based optimization

**Disadvantages:**
- Requires careful hyperparameter tuning
- Sensitive to feature scaling (must standardize!)
- Can get stuck in local minima
- Risk of overfitting without regularization
- Black box, hard to interpret
- Computationally expensive for large networks

## Model Selection Framework

### When to Use Advanced Models vs. Simpler Methods

| Scenario | Recommended Model |
|----------|-------------------|
| Small dataset (< 1,000) | Logistic Regression, SVM |
| Medium dataset with interactions | Random Forest, Gradient Boosting |
| Large dataset with complex patterns | Neural Networks, XGBoost |
| High-dimensional sparse data | SVM with linear kernel |
| Need interpretability | Logistic Regression, Decision Trees |
| Propensity score estimation | Gradient Boosting > Neural Networks > SVM |

### Comparison with Tree-Based Models

| Aspect | SVM/MLP | Tree-Based |
|--------|---------|------------|
| **Feature Scaling** | Required | Not required |
| **Categorical Features** | Encoding required | Native support (some) |
| **Missing Values** | Must impute | Some handle natively |
| **Interpretability** | Low | Medium (feature importance) |
| **Training Speed** | Medium/Slow | Fast |
| **Prediction Speed** | Fast | Fast |
| **Non-linearity** | Kernel/Layers | Splits |

### Interpretability Trade-offs

```
High Interpretability ←————————————————→ Low Interpretability
                                         High Flexibility

Linear Regression → Decision Tree → Random Forest → SVM → Neural Network
       ↑                   ↑              ↑           ↑         ↑
  Coefficients         Rules      Feature Imp.    Hard     Very Hard
```

For causal inference, interpretability often matters more than raw predictive power.

## Causal Inference Applications

### Propensity Score Estimation with Neural Networks

Neural networks can estimate propensity scores, but use with caution:

```python
from sklearn.neural_network import MLPClassifier
from sklearn.calibration import CalibratedClassifierCV

# Neural net for propensity scores
mlp_ps = Pipeline([
    ('scaler', StandardScaler()),
    ('mlp', MLPClassifier(
        hidden_layer_sizes=(64, 32),
        activation='relu',
        early_stopping=True,
        random_state=42
    ))
])

# Calibrate for better probability estimates
calibrated_mlp = CalibratedClassifierCV(mlp_ps, cv=5)
calibrated_mlp.fit(X, treatment)
propensity_scores = calibrated_mlp.predict_proba(X)[:, 1]
```

**Considerations:**
- Neural nets may overfit propensity scores
- Calibration improves probability estimates
- Gradient boosting often more stable for propensity scores

### First-Stage Learners in DDML (with Caution)

Neural networks can serve as first-stage learners in Double/Debiased ML:

```python
from econml.dml import LinearDML

# Use MLP as first-stage learner
mlp_outcome = MLPRegressor(hidden_layer_sizes=(64, 32), early_stopping=True)
mlp_treatment = MLPClassifier(hidden_layer_sizes=(64, 32), early_stopping=True)

dml = LinearDML(
    model_y=Pipeline([('scaler', StandardScaler()), ('mlp', mlp_outcome)]),
    model_t=Pipeline([('scaler', StandardScaler()), ('mlp', mlp_treatment)]),
    cv=5
)
dml.fit(Y, T, X=X, W=W)
```

**Cautions:**
- First-stage models should not overfit (use regularization)
- Cross-fitting helps but doesn't eliminate all issues
- Tree-based models often more robust for DDML
- Sample splitting is critical with neural networks

### Representation Learning for Confounders

Neural networks can learn representations of confounders:

```python
# Learn compressed confounder representation
autoencoder_like = MLPRegressor(
    hidden_layer_sizes=(128, 32, 128),  # Bottleneck architecture
    activation='relu'
)
# Train on W to reconstruct W
# Extract hidden layer activations as representation
```

**Use cases:**
- High-dimensional confounders
- Confounder balancing representations
- Pre-training for causal estimation

## Common Mistakes

### 1. No Feature Standardization

```python
# WRONG - SVM/MLP without scaling
svm = SVC()
svm.fit(X, y)  # Will perform poorly!

# CORRECT - Always use pipeline with scaler
svm_scaled = Pipeline([
    ('scaler', StandardScaler()),
    ('svc', SVC())
])
svm_scaled.fit(X, y)
```

### 2. Overfitting Complex Models

```python
# WRONG - Too complex for small dataset
mlp_overfit = MLPClassifier(
    hidden_layer_sizes=(512, 256, 128, 64),
    max_iter=1000
)

# CORRECT - Appropriate complexity with regularization
mlp_proper = MLPClassifier(
    hidden_layer_sizes=(64, 32),
    alpha=0.01,  # Regularization
    early_stopping=True
)
```

### 3. Using Advanced Models When Simpler Suffice

```python
# If logistic regression achieves 85% accuracy
# and MLP achieves 86% accuracy...
# Prefer logistic regression for interpretability!

# Check if simpler model works first
from sklearn.linear_model import LogisticRegression
simple = LogisticRegression()
simple.fit(X_train, y_train)
print(f"Simple model accuracy: {simple.score(X_test, y_test):.3f}")

# Only use complex model if significant improvement
```

### 4. Ignoring Convergence Warnings

```python
# Monitor convergence
mlp = MLPClassifier(max_iter=500, verbose=True)
mlp.fit(X, y)

# Check if converged
if mlp.n_iter_ == mlp.max_iter:
    print("Warning: Did not converge! Increase max_iter or adjust learning rate")
```

### 5. Wrong Kernel Choice

```python
# Linear kernel for high-dimensional sparse data
svm_text = Pipeline([
    ('scaler', StandardScaler()),
    ('svc', SVC(kernel='linear'))  # Good for text/sparse
])

# RBF for general non-linear patterns
svm_general = Pipeline([
    ('scaler', StandardScaler()),
    ('svc', SVC(kernel='rbf'))
])
```

## Quick Reference

### SVM Quick Setup

```python
# Classification
from sklearn.svm import SVC
svc = Pipeline([
    ('scaler', StandardScaler()),
    ('svc', SVC(kernel='rbf', C=1.0, gamma='scale', probability=True))
])

# Regression
from sklearn.svm import SVR
svr = Pipeline([
    ('scaler', StandardScaler()),
    ('svr', SVR(kernel='rbf', C=1.0, epsilon=0.1))
])
```

### MLP Quick Setup

```python
# Classification
from sklearn.neural_network import MLPClassifier
mlp_clf = Pipeline([
    ('scaler', StandardScaler()),
    ('mlp', MLPClassifier(
        hidden_layer_sizes=(100, 50),
        activation='relu',
        solver='adam',
        alpha=0.001,
        early_stopping=True,
        random_state=42
    ))
])

# Regression
from sklearn.neural_network import MLPRegressor
mlp_reg = Pipeline([
    ('scaler', StandardScaler()),
    ('mlp', MLPRegressor(
        hidden_layer_sizes=(100, 50),
        activation='relu',
        solver='adam',
        alpha=0.001,
        early_stopping=True,
        random_state=42
    ))
])
```

## References

### In-Depth Guides

- [Neural Networks for Tabular Data](references/neural_networks.md) - MLP architecture, regularization, hyperparameter tuning
- [SVM and Kernel Methods](references/svm_kernel.md) - Kernel selection, SVM tuning, computational considerations
- [Ensemble Methods](references/ensemble_methods.md) - Stacking, blending, model averaging
- [Neural Networks for Causal Inference](references/causal_applications.md) - DragonNet, CEVAE, DDML with neural networks

### Scripts

- [run_advanced_model.py](scripts/run_advanced_model.py) - CLI for training SVM/MLP models
- [model_comparison.py](scripts/model_comparison.py) - Cross-validated model comparison
- [visualize_learning.py](scripts/visualize_learning.py) - Learning and validation curves

### Templates

- [Advanced Model Report](assets/markdown/advanced_report.md) - Comprehensive model evaluation report template

## Related Skills

- **ml-model-tree**: Tree-based models (prerequisites)
- **ml-model-linear**: Linear models (simpler alternatives)
- **ml-evaluation**: Model evaluation and validation
- **ddml-basics**: Using advanced models in causal estimation
