# Support Vector Machines and Kernel Methods

## Overview

Support Vector Machines (SVMs) find the optimal hyperplane that maximizes the margin between classes. The kernel trick allows SVMs to learn non-linear decision boundaries by implicitly mapping data to higher-dimensional spaces.

## When to Use SVMs

### Good Use Cases

1. **Small to medium datasets** (100 - 10,000 samples)
2. **High-dimensional data**: Text classification, genomics
3. **Clear margin of separation**: Well-separated classes
4. **Binary classification**: SVM excels at two-class problems
5. **Sparse data**: Document classification, feature selection

### When to Avoid

1. **Large datasets** (>50,000): Training is O(n^2) to O(n^3)
2. **Need probability estimates**: Calibration is expensive
3. **Multi-class problems**: One-vs-rest is less efficient
4. **Highly noisy data**: Outliers affect the margin

## The Kernel Trick

The kernel trick computes the dot product in a higher-dimensional space without explicitly computing the coordinates:

```
K(x, y) = phi(x) . phi(y)
```

This allows SVMs to find non-linear decision boundaries while maintaining computational efficiency.

### Linear Kernel

```python
from sklearn.svm import SVC

# K(x, y) = x . y
svc_linear = SVC(kernel='linear')
```

**Use when:**
- Data is linearly separable
- High-dimensional sparse data (text)
- Features >> samples
- Interpretability needed (coefficient access)

**Pros:**
- Fast training for sparse data
- Coefficient interpretation possible
- No gamma parameter to tune

**Cons:**
- Cannot learn non-linear patterns

### RBF (Radial Basis Function) Kernel

```python
# K(x, y) = exp(-gamma * ||x - y||^2)
svc_rbf = SVC(kernel='rbf', gamma='scale')
```

**Use when:**
- Non-linear patterns expected
- Don't know the data distribution
- General-purpose classification

**Gamma parameter:**
- `gamma='scale'`: 1 / (n_features * X.var()) - Default, recommended
- `gamma='auto'`: 1 / n_features
- Small gamma: Smooth, wider influence
- Large gamma: Complex, localized influence

```
Small gamma (0.001)     Large gamma (10)
+------------------+    +------------------+
|    .....         |    |  .   .           |
|  .........       |    | ...  ..          |
| ............     |    | .... ...         |
|  .........       |    |  ... ..          |
|    .....         |    |   .  .           |
+------------------+    +------------------+
 Smooth boundary         Complex boundary
```

### Polynomial Kernel

```python
# K(x, y) = (gamma * x . y + coef0)^degree
svc_poly = SVC(kernel='poly', degree=3, gamma='scale', coef0=1)
```

**Use when:**
- Feature interactions are important
- Known polynomial relationship
- Image classification (historically)

**Parameters:**
- `degree`: Polynomial degree (2, 3, 4)
- `coef0`: Independent term (0 or 1 typically)
- `gamma`: Coefficient for input vectors

### Sigmoid Kernel

```python
# K(x, y) = tanh(gamma * x . y + coef0)
svc_sigmoid = SVC(kernel='sigmoid', gamma='scale', coef0=0)
```

**Use when:**
- Neural network-like behavior desired
- Rarely used in practice; RBF usually better

## Hyperparameter Tuning

### C Parameter (Regularization)

C controls the trade-off between smooth decision boundary and classifying training points correctly:

```
Small C (0.1)           Large C (100)
+------------------+    +------------------+
|     /            |    |     |            |
|    /  o  o       |    |     | o  o       |
|   /    o         |    |     |   o        |
|  / x             |    | x   |            |
| /   x  x         |    |  x  | x          |
+------------------+    +------------------+
 Wide margin             Narrow margin
 More misclassification  Fewer misclassifications
```

**Tuning strategy:**
1. Start with C=1.0 (default)
2. If underfitting: increase C (1, 10, 100, 1000)
3. If overfitting: decrease C (0.1, 0.01, 0.001)

### Grid Search Strategy

```python
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# IMPORTANT: Always include scaler in pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('svc', SVC())
])

# Comprehensive grid
param_grid = {
    'svc__C': [0.01, 0.1, 1, 10, 100],
    'svc__gamma': ['scale', 0.001, 0.01, 0.1, 1],
    'svc__kernel': ['rbf', 'linear', 'poly']
}

# Reduced grid for poly kernel
param_grid_poly = {
    'svc__C': [0.1, 1, 10],
    'svc__kernel': ['poly'],
    'svc__degree': [2, 3, 4],
    'svc__coef0': [0, 1]
}

search = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
search.fit(X, y)
```

### Randomized Search for Large Grids

```python
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import loguniform

param_dist = {
    'svc__C': loguniform(0.01, 1000),
    'svc__gamma': loguniform(0.0001, 10),
    'svc__kernel': ['rbf', 'linear', 'poly']
}

search = RandomizedSearchCV(
    pipeline, param_dist,
    n_iter=100, cv=5,
    scoring='accuracy',
    random_state=42
)
search.fit(X, y)
```

## SVM for Regression (SVR)

### Epsilon-Insensitive Loss

SVR uses an epsilon-tube around predictions. No penalty for errors within the tube:

```
     |
  y  |    ........epsilon tube........
     |    .                          .
     |    .   x       x              .
pred |----+-------------------x------+----
     |    .        x                 .
     |    .                          .
     |    ............................
     |
     +------------------------------------> x
```

### SVR Parameters

```python
from sklearn.svm import SVR

svr = SVR(
    kernel='rbf',
    C=1.0,           # Regularization
    gamma='scale',   # Kernel coefficient
    epsilon=0.1      # Width of epsilon-tube
)
```

**Epsilon tuning:**
- Large epsilon: Simpler model, ignores small errors
- Small epsilon: More precise, may overfit

### Choosing SVR vs. Other Regressors

| Criterion | SVR | Ridge | Random Forest |
|-----------|-----|-------|---------------|
| Non-linear | Yes (RBF) | No | Yes |
| Scalability | Poor | Excellent | Good |
| Sparse data | Good | Good | Poor |
| Interpretability | Poor | Good | Medium |
| Outlier robust | Medium | Poor | Good |

## Probability Calibration

SVM does not naturally output probabilities. Enable probability estimates with calibration:

```python
# Enable probability estimates (slower)
svc = SVC(kernel='rbf', probability=True)
svc.fit(X_train, y_train)

# Get probabilities
proba = svc.predict_proba(X_test)

# Note: probability=True uses Platt scaling internally
# For better calibration, use explicit calibration:
from sklearn.calibration import CalibratedClassifierCV

svc_base = SVC(kernel='rbf')
calibrated_svc = CalibratedClassifierCV(svc_base, cv=5, method='isotonic')
calibrated_svc.fit(X_train, y_train)
proba = calibrated_svc.predict_proba(X_test)
```

### Calibration Methods

| Method | Description | Best For |
|--------|-------------|----------|
| Platt (sigmoid) | Fits sigmoid to SVM scores | SVM default |
| Isotonic | Non-parametric monotonic function | More flexible, needs more data |

## Computational Considerations

### Training Complexity

| Kernel | Training Time | Memory |
|--------|--------------|--------|
| Linear | O(n * d) | O(n * d) |
| RBF/Poly | O(n^2) to O(n^3) | O(n^2) |

### Strategies for Large Data

1. **Use LinearSVC for linear kernel:**
```python
from sklearn.svm import LinearSVC

# Much faster than SVC(kernel='linear')
linear_svc = LinearSVC(C=1.0, max_iter=10000)
```

2. **Subsample training data:**
```python
from sklearn.model_selection import train_test_split

# Train on subset
X_sub, _, y_sub, _ = train_test_split(X, y, train_size=10000)
svc.fit(X_sub, y_sub)
```

3. **Use SGD with hinge loss:**
```python
from sklearn.linear_model import SGDClassifier

# Approximate linear SVM
sgd_svm = SGDClassifier(loss='hinge', penalty='l2', max_iter=1000)
```

4. **Use approximate kernel methods:**
```python
from sklearn.kernel_approximation import RBFSampler, Nystroem

# RBF approximation with random Fourier features
rbf_approx = RBFSampler(gamma=1.0, n_components=100)
X_features = rbf_approx.fit_transform(X)
linear_svc.fit(X_features, y)
```

## Common Pitfalls

### 1. Forgetting to Scale Features

```python
# WRONG - SVM is very sensitive to feature scales
svc = SVC()
svc.fit(X, y)  # Unscaled features!

# CORRECT
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('svc', SVC())
])
pipeline.fit(X, y)
```

### 2. Using Default Gamma for Old sklearn

```python
# Old default (pre-sklearn 0.22) was gamma='auto' which often overfits
# Current default is gamma='scale' which is better

# Explicitly set to avoid version issues
svc = SVC(kernel='rbf', gamma='scale')
```

### 3. Expecting Fast Training on Large Data

```python
# SVM training is O(n^2) to O(n^3)
# For 100,000 samples, consider alternatives:

# Option 1: Use linear kernel with LinearSVC
from sklearn.svm import LinearSVC
linear_svc = LinearSVC(max_iter=10000)

# Option 2: Use gradient boosting
from sklearn.ensemble import GradientBoostingClassifier
gbc = GradientBoostingClassifier()

# Option 3: Use kernel approximation
from sklearn.kernel_approximation import Nystroem
nystroem = Nystroem(kernel='rbf', n_components=300)
X_approx = nystroem.fit_transform(X)
LinearSVC().fit(X_approx, y)
```

### 4. Ignoring Class Imbalance

```python
# Handle imbalanced classes
svc = SVC(
    kernel='rbf',
    class_weight='balanced'  # Adjust weights inversely proportional to frequencies
)

# Or set explicit weights
svc = SVC(class_weight={0: 1, 1: 10})  # Weight class 1 more heavily
```

## Best Practices Summary

1. **Always standardize features** - SVM is scale-sensitive
2. **Start with RBF kernel** - Most general-purpose
3. **Tune C and gamma together** - They interact
4. **Use grid search or random search** - Essential for good performance
5. **Consider LinearSVC for linear problems** - Much faster
6. **Monitor training time** - SVM doesn't scale to large datasets
7. **Use class weights for imbalanced data**
8. **Calibrate probabilities** if needed for downstream tasks
9. **Compare with gradient boosting** - Often more practical for tabular data
