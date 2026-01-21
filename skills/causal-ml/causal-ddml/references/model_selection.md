# DDML Model Selection

> Guidance for selecting nuisance learners and model specifications in DDML

## Overview

Model selection in DDML involves:
1. Choosing ML learners for nuisance functions
2. Tuning hyperparameters
3. Deciding between PLR vs IRM
4. Selecting number of folds and repetitions

---

## 1. Nuisance Learner Selection

### Available Learners

| Learner | Best For | Pros | Cons |
|---------|----------|------|------|
| **Lasso** | Sparse, linear | Fast, interpretable, variable selection | Assumes sparsity |
| **Ridge** | Dense effects | Stable, good with multicollinearity | No variable selection |
| **Elastic Net** | Mixed | Combines Lasso + Ridge | Two parameters to tune |
| **Random Forest** | Nonlinear, interactions | Handles nonlinearity well | Can overfit, slower |
| **XGBoost** | Complex patterns | State-of-art performance | Many hyperparameters |
| **LightGBM** | Large data | Fast, memory efficient | Similar to XGBoost |
| **Neural Networks** | Very complex | Ultimate flexibility | Requires much data |

### Selection Strategy

```python
def select_nuisance_learners(X, y, d, cv_folds=5):
    """
    Automated learner selection based on cross-validation.
    """
    from sklearn.model_selection import cross_val_score

    candidate_learners = {
        'lasso': LassoCV(cv=5),
        'ridge': RidgeCV(cv=5),
        'elastic_net': ElasticNetCV(cv=5),
        'random_forest': RandomForestRegressor(n_estimators=100, max_depth=10),
        'xgboost': XGBRegressor(n_estimators=100, max_depth=5)
    }

    # Evaluate for outcome model E[Y|X]
    scores_y = {}
    for name, learner in candidate_learners.items():
        try:
            cv_score = cross_val_score(learner, X, y, cv=cv_folds,
                                       scoring='neg_mean_squared_error')
            scores_y[name] = -cv_score.mean()
        except:
            scores_y[name] = np.inf

    best_y = min(scores_y, key=scores_y.get)

    # Evaluate for treatment model E[D|X]
    is_binary = len(np.unique(d)) == 2
    scores_d = {}

    for name, learner in candidate_learners.items():
        try:
            if is_binary:
                # Classification for binary treatment
                clf = get_classifier_version(learner)
                cv_score = cross_val_score(clf, X, d, cv=cv_folds,
                                          scoring='neg_brier_score')
            else:
                cv_score = cross_val_score(learner, X, d, cv=cv_folds,
                                          scoring='neg_mean_squared_error')
            scores_d[name] = -cv_score.mean()
        except:
            scores_d[name] = np.inf

    best_d = min(scores_d, key=scores_d.get)

    return {
        'best_outcome_learner': best_y,
        'best_treatment_learner': best_d,
        'outcome_cv_scores': scores_y,
        'treatment_cv_scores': scores_d
    }
```

### Decision Tree for Learner Selection

```
Data characteristics:
├── High-dimensional (p >> n)
│   └── Use Lasso or Elastic Net (sparsity assumption)
├── Moderate dimensions, expect nonlinearity
│   └── Use Random Forest or XGBoost
├── Very large n (n > 100k)
│   └── Use LightGBM (speed) or Lasso (simplicity)
├── Unknown structure
│   └── Compare multiple learners, report sensitivity
└── Small sample (n < 500)
    └── Use Lasso/Ridge (less variance than tree methods)
```

---

## 2. Hyperparameter Tuning

### Lasso/Ridge/Elastic Net

```python
from sklearn.linear_model import LassoCV, RidgeCV, ElasticNetCV

# Lasso with automatic lambda selection
lasso = LassoCV(
    cv=5,                    # 5-fold CV for lambda
    n_alphas=100,            # Grid of 100 lambda values
    max_iter=10000,          # Increase for convergence
    selection='random'       # Faster for large p
)

# Ridge
ridge = RidgeCV(
    cv=5,
    alphas=np.logspace(-4, 4, 100)
)

# Elastic Net
enet = ElasticNetCV(
    cv=5,
    l1_ratio=[0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 1.0],  # Mix of L1/L2
    n_alphas=50
)
```

### Random Forest

```python
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV

rf = RandomForestRegressor(random_state=42)

param_dist = {
    'n_estimators': [100, 200, 500],
    'max_depth': [5, 10, 15, None],
    'min_samples_leaf': [1, 2, 5, 10],
    'max_features': ['sqrt', 'log2', 0.3, 0.5]
}

rf_tuned = RandomizedSearchCV(
    rf, param_dist,
    n_iter=50,
    cv=5,
    scoring='neg_mean_squared_error',
    random_state=42
)
```

### XGBoost/LightGBM

```python
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV

xgb = XGBRegressor(random_state=42, verbosity=0)

param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.2],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0],
    'reg_alpha': [0, 0.1, 1],    # L1 regularization
    'reg_lambda': [1, 10]        # L2 regularization
}

xgb_tuned = GridSearchCV(
    xgb, param_grid,
    cv=5,
    scoring='neg_mean_squared_error'
)
```

### Automated Tuning with DoubleML

```python
from doubleml import DoubleMLPLR
from sklearn.model_selection import GridSearchCV

# Wrap tuned learner
def get_tuned_learner(X, y):
    """Return tuned learner via nested CV."""
    from sklearn.linear_model import LassoCV
    from sklearn.ensemble import RandomForestRegressor

    # Quick comparison
    lasso = LassoCV(cv=3).fit(X, y)
    rf = RandomForestRegressor(n_estimators=100).fit(X, y)

    # Use the one with better CV score
    lasso_score = lasso.score(X, y)
    rf_score = rf.score(X, y)

    return lasso if lasso_score > rf_score else rf
```

---

## 3. Ensemble Approaches

### Stacking Multiple Learners

```python
from sklearn.ensemble import StackingRegressor

def create_stacked_learner():
    """
    Ensemble multiple learners for robust nuisance estimation.
    """
    estimators = [
        ('lasso', LassoCV(cv=5)),
        ('rf', RandomForestRegressor(n_estimators=100, max_depth=10)),
        ('xgb', XGBRegressor(n_estimators=100, max_depth=5))
    ]

    stacker = StackingRegressor(
        estimators=estimators,
        final_estimator=RidgeCV(cv=5),
        cv=5
    )

    return stacker
```

### Super Learner (TMLE-style)

```python
def super_learner(X, y, learners, cv_folds=5):
    """
    Super Learner: optimal weighted combination of learners.

    1. Get cross-validated predictions from each learner
    2. Find optimal weights via constrained regression
    """
    from sklearn.model_selection import cross_val_predict

    # Get CV predictions from each learner
    cv_preds = {}
    for name, learner in learners.items():
        cv_preds[name] = cross_val_predict(learner, X, y, cv=cv_folds)

    # Stack predictions
    Z = np.column_stack([cv_preds[name] for name in learners])

    # Find optimal non-negative weights that sum to 1
    from scipy.optimize import minimize

    def objective(weights):
        pred = Z @ weights
        return np.mean((y - pred)**2)

    n_learners = len(learners)
    constraints = [
        {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},  # Sum to 1
    ]
    bounds = [(0, 1) for _ in range(n_learners)]  # Non-negative

    result = minimize(
        objective,
        x0=np.ones(n_learners) / n_learners,
        bounds=bounds,
        constraints=constraints
    )

    optimal_weights = result.x

    return {
        'weights': dict(zip(learners.keys(), optimal_weights)),
        'cv_mse': result.fun
    }
```

### Model Averaging for DDML

```python
def ddml_model_averaging(data, outcome, treatment, controls,
                         learner_list=['lasso', 'rf', 'xgboost']):
    """
    Run DDML with multiple learners and combine estimates.
    """
    estimates = []
    variances = []

    for learner in learner_list:
        result = estimate_plr(
            data=data, outcome=outcome, treatment=treatment,
            controls=controls, ml_l=learner, ml_m=learner
        )
        estimates.append(result.effect)
        variances.append(result.se**2)

    # Inverse variance weighting
    weights = [1/v for v in variances]
    total_weight = sum(weights)
    weights = [w/total_weight for w in weights]

    combined_estimate = sum(e*w for e, w in zip(estimates, weights))
    combined_variance = 1 / sum(1/v for v in variances)
    combined_se = np.sqrt(combined_variance)

    return {
        'combined_effect': combined_estimate,
        'combined_se': combined_se,
        'individual_estimates': dict(zip(learner_list, estimates)),
        'weights': dict(zip(learner_list, weights))
    }
```

---

## 4. PLR vs IRM Selection

### When to Use Each

| Criterion | PLR | IRM |
|-----------|-----|-----|
| Treatment type | Any (continuous/binary) | Binary only |
| Effect homogeneity | Constant effect | Allows heterogeneity |
| Efficiency | More efficient if constant | More robust |
| Interpretation | Clear coefficient | ATE (population average) |

### Decision Framework

```python
def recommend_model(data, treatment, outcome, controls):
    """
    Recommend PLR vs IRM based on data characteristics.
    """
    d = data[treatment]

    # Check if treatment is binary
    is_binary = set(d.unique()).issubset({0, 1})

    if not is_binary:
        return {
            'recommendation': 'PLR',
            'reason': 'Treatment is continuous; IRM requires binary treatment'
        }

    # If binary, consider both
    # Test for effect heterogeneity
    result_plr = estimate_plr(data, outcome, treatment, controls)
    result_irm = estimate_irm(data, outcome, treatment, controls)

    diff = abs(result_plr.effect - result_irm.effect)
    avg = (abs(result_plr.effect) + abs(result_irm.effect)) / 2
    relative_diff = diff / avg if avg > 0 else 0

    if relative_diff < 0.1:
        return {
            'recommendation': 'PLR (or either)',
            'reason': f'PLR and IRM give similar estimates (diff={relative_diff:.1%})',
            'plr_effect': result_plr.effect,
            'irm_effect': result_irm.effect
        }
    else:
        return {
            'recommendation': 'Report both',
            'reason': f'Substantial difference ({relative_diff:.1%}) suggests heterogeneity',
            'plr_effect': result_plr.effect,
            'irm_effect': result_irm.effect,
            'note': 'IRM may be preferred if heterogeneity is expected'
        }
```

---

## 5. Cross-Fitting Parameters

### Number of Folds (K)

| Sample Size | Recommended K | Rationale |
|-------------|---------------|-----------|
| n < 500 | 2-3 | Ensure sufficient training data |
| 500 - 2000 | 5 | Standard, good balance |
| 2000 - 10000 | 5-10 | Can afford more folds |
| n > 10000 | 5-10 | Diminishing returns beyond 10 |

### Number of Repetitions

| Goal | n_rep | Use Case |
|------|-------|----------|
| Quick analysis | 1 | Exploratory work |
| Standard | 1-5 | Most applications |
| Publication | 5-10 | Reduce sampling variability |
| High stakes | 10-50 | When precision critical |

### Implementation

```python
def select_cross_fitting_params(n, n_controls, precision_level='standard'):
    """
    Select cross-fitting parameters based on sample size and goals.
    """
    # Number of folds
    if n < 500:
        n_folds = min(3, max(2, n // 100))
    elif n < 2000:
        n_folds = 5
    else:
        n_folds = min(10, max(5, n // 500))

    # Number of repetitions
    n_rep_map = {
        'quick': 1,
        'standard': 3,
        'robust': 10,
        'publication': 50
    }
    n_rep = n_rep_map.get(precision_level, 3)

    return {
        'n_folds': n_folds,
        'n_rep': n_rep,
        'n_per_fold': n // n_folds,
        'note': f"With n={n}, using {n_folds} folds gives ~{n // n_folds} obs/fold"
    }
```

---

## 6. Model Selection Checklist

### Pre-Estimation

- [ ] **Data exploration**: Understand distributions, check for nonlinearity
- [ ] **Dimensionality**: Count p vs n, assess high-dimensional regime
- [ ] **Treatment type**: Binary or continuous?
- [ ] **Domain knowledge**: Expected effect pattern (constant vs heterogeneous)?

### Learner Selection

- [ ] **Cross-validate candidates**: Compare Lasso, RF, XGBoost on your data
- [ ] **Consider ensemble**: Stack learners for robustness
- [ ] **Match complexity to n**: Avoid overly complex models with small n

### Specification Testing

- [ ] **Compare PLR vs IRM** (for binary treatment)
- [ ] **Vary number of folds**: Check K=3, 5, 10
- [ ] **Multiple learners**: Report sensitivity
- [ ] **Multiple repetitions**: Reduce sampling noise

### Post-Estimation

- [ ] **Residual diagnostics**: Check nuisance model fit
- [ ] **Overlap check**: Verify propensity overlap (for IRM)
- [ ] **Sensitivity analysis**: Report range across specifications

---

## 7. Common Pitfalls

### 1. Over-tuning Nuisance Functions

**Problem**: Spending too much effort on nuisance tuning
**Solution**: DDML is robust to nuisance quality (orthogonality); simple CV suffices

### 2. Ignoring Learner Sensitivity

**Problem**: Reporting only one learner specification
**Solution**: Always compare 2-3 learners and report if results differ

### 3. Wrong K for Sample Size

**Problem**: Too many folds for small n (unstable nuisance)
**Solution**: Use K=2-3 for n < 500

### 4. Using IRM for Continuous Treatment

**Problem**: IRM is designed for binary treatment
**Solution**: Use PLR for continuous treatment

---

## References

1. Chernozhukov, V., et al. (2018). Double/Debiased Machine Learning. *Econometrics Journal*.

2. van der Laan, M. J., Polley, E. C., & Hubbard, A. E. (2007). Super Learner. *Statistical Applications in Genetics and Molecular Biology*.

3. DoubleML Documentation: https://docs.doubleml.org/stable/guide/learners.html

4. Athey, S., & Imbens, G. W. (2019). Machine Learning Methods Economists Should Know About. *Annual Review of Economics*.
