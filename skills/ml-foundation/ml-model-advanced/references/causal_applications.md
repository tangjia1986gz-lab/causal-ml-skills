# Neural Networks for Causal Inference

## Overview

Neural networks have been adapted for causal effect estimation, offering flexibility in modeling complex relationships. However, they require careful implementation to maintain valid causal interpretation.

## When Advanced Models Help in Causal Inference

### Good Use Cases

1. **Complex confounding**: Non-linear relationships between confounders and outcomes
2. **High-dimensional confounders**: Many covariates to adjust for
3. **Heterogeneous treatment effects**: Effect varies complexly with covariates
4. **Large datasets**: Enough data to estimate neural network parameters reliably

### When to Prefer Simpler Methods

1. **Small samples** (<1,000): Regularized linear models more stable
2. **Need interpretability**: Linear/tree models easier to explain
3. **Limited compute**: Gradient boosting often sufficient
4. **Simple treatment mechanism**: Complex models not necessary

## Propensity Score Estimation with Neural Networks

### Basic Approach

```python
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV

def estimate_propensity_mlp(X, treatment, calibrate=True):
    """
    Estimate propensity scores using MLP with calibration.

    Parameters
    ----------
    X : array-like
        Confounders matrix.
    treatment : array-like
        Binary treatment indicator.
    calibrate : bool
        Whether to calibrate probabilities.

    Returns
    -------
    propensity_scores : array
        Estimated P(T=1|X).
    """
    # Base MLP pipeline
    mlp_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('mlp', MLPClassifier(
            hidden_layer_sizes=(64, 32),
            activation='relu',
            alpha=0.01,  # Strong regularization
            early_stopping=True,
            validation_fraction=0.1,
            random_state=42
        ))
    ])

    if calibrate:
        # Calibration improves probability estimates
        model = CalibratedClassifierCV(mlp_pipeline, cv=5, method='isotonic')
    else:
        model = mlp_pipeline

    model.fit(X, treatment)
    propensity_scores = model.predict_proba(X)[:, 1]

    return propensity_scores
```

### Propensity Score Diagnostics

```python
def check_propensity_scores(propensity_scores, treatment):
    """
    Diagnostic checks for propensity scores.
    """
    treated = propensity_scores[treatment == 1]
    control = propensity_scores[treatment == 0]

    diagnostics = {
        'overlap': {
            'treated_range': (treated.min(), treated.max()),
            'control_range': (control.min(), control.max()),
            'overlap_warning': treated.min() > control.max() or control.min() > treated.max()
        },
        'extremes': {
            'near_zero': (propensity_scores < 0.01).sum(),
            'near_one': (propensity_scores > 0.99).sum(),
            'trim_recommended': (propensity_scores < 0.01).sum() + (propensity_scores > 0.99).sum() > 0
        },
        'balance': {
            'mean_treated': treated.mean(),
            'mean_control': control.mean(),
        }
    }

    return diagnostics
```

### Trimming Extreme Propensity Scores

```python
def trim_propensity_scores(propensity_scores, lower=0.01, upper=0.99):
    """
    Trim observations with extreme propensity scores.

    Extreme scores indicate positivity violations.
    """
    mask = (propensity_scores >= lower) & (propensity_scores <= upper)
    return mask
```

## Double/Debiased Machine Learning with Neural Networks

### DDML Overview

Double ML uses machine learning for nuisance parameter estimation (propensity scores, outcome regression) while maintaining valid inference through:
1. **Orthogonalization**: Removes regularization bias
2. **Cross-fitting**: Prevents overfitting bias

### Using MLPs as First-Stage Learners

```python
from econml.dml import LinearDML, NonParamDML
from sklearn.neural_network import MLPClassifier, MLPRegressor

# MLP for outcome model
mlp_outcome = Pipeline([
    ('scaler', StandardScaler()),
    ('mlp', MLPRegressor(
        hidden_layer_sizes=(64, 32),
        alpha=0.01,
        early_stopping=True,
        random_state=42
    ))
])

# MLP for treatment model (propensity)
mlp_treatment = Pipeline([
    ('scaler', StandardScaler()),
    ('mlp', MLPClassifier(
        hidden_layer_sizes=(64, 32),
        alpha=0.01,
        early_stopping=True,
        random_state=42
    ))
])

# Linear DML with MLP nuisance models
linear_dml = LinearDML(
    model_y=mlp_outcome,
    model_t=mlp_treatment,
    cv=5,  # Cross-fitting folds
    random_state=42
)

linear_dml.fit(Y, T, X=X, W=W)
ate = linear_dml.ate()
```

### Considerations for DDML with Neural Networks

**Advantages:**
- Can capture complex confounding
- Flexible function approximation
- Good for high-dimensional W

**Risks:**
- Overfitting despite cross-fitting
- Sensitivity to hyperparameters
- Slower than tree-based alternatives

**Recommendations:**
1. Use strong regularization (high alpha)
2. Enable early stopping
3. Compare with gradient boosting baseline
4. Check first-stage R-squared

```python
# Check first-stage model fit
from sklearn.model_selection import cross_val_score

# Outcome model quality
y_scores = cross_val_score(mlp_outcome, W, Y, cv=5, scoring='r2')
print(f"Outcome model R2: {y_scores.mean():.3f}")

# Treatment model quality (propensity)
t_scores = cross_val_score(mlp_treatment, W, T, cv=5, scoring='roc_auc')
print(f"Treatment model AUC: {t_scores.mean():.3f}")
```

## DragonNet: Targeted Regularization

### Concept

DragonNet (Shi et al., 2019) is a neural network architecture specifically designed for treatment effect estimation:

1. **Shared representation**: Common hidden layers learn balanced representations
2. **Three heads**: Outcome under control, outcome under treatment, propensity score
3. **Targeted regularization**: Propensity head regularizes the representation

```
                Input X
                   |
            [Shared Layers]
                   |
         +---------+---------+
         |         |         |
      [Q0 Head] [Q1 Head] [g Head]
         |         |         |
       Y|T=0     Y|T=1      P(T)
```

### PyTorch Implementation Sketch

```python
import torch
import torch.nn as nn

class DragonNet(nn.Module):
    """
    DragonNet for treatment effect estimation.

    Architecture:
    - Shared representation layers
    - Three output heads: Q0, Q1, propensity
    """

    def __init__(self, input_dim, hidden_dims=[200, 100, 100], dropout=0.3):
        super().__init__()

        # Shared representation
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ELU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim

        self.representation = nn.Sequential(*layers)

        # Outcome head for T=0
        self.q0_head = nn.Sequential(
            nn.Linear(hidden_dims[-1], 100),
            nn.ELU(),
            nn.Linear(100, 1)
        )

        # Outcome head for T=1
        self.q1_head = nn.Sequential(
            nn.Linear(hidden_dims[-1], 100),
            nn.ELU(),
            nn.Linear(100, 1)
        )

        # Propensity head
        self.g_head = nn.Sequential(
            nn.Linear(hidden_dims[-1], 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        rep = self.representation(x)
        q0 = self.q0_head(rep)
        q1 = self.q1_head(rep)
        g = self.g_head(rep)
        return q0, q1, g

    def predict_cate(self, x):
        """Predict conditional average treatment effect."""
        q0, q1, _ = self.forward(x)
        return q1 - q0
```

### Training DragonNet

```python
def dragonnet_loss(y_true, t, q0, q1, g, alpha=1.0):
    """
    DragonNet loss function.

    Components:
    - Regression loss for outcomes
    - Cross-entropy for propensity
    - Optional: targeted regularization
    """
    # Outcome loss
    y_pred = t * q1 + (1 - t) * q0
    regression_loss = nn.MSELoss()(y_pred, y_true)

    # Propensity loss
    propensity_loss = nn.BCELoss()(g, t)

    # Combined loss with propensity regularization
    total_loss = regression_loss + alpha * propensity_loss

    return total_loss
```

## CEVAE: Causal Effect Variational Autoencoder

### Concept

CEVAE (Louizos et al., 2017) learns latent confounders using variational inference:

1. **Latent variable model**: Assumes hidden confounders Z
2. **VAE framework**: Encoder q(Z|X), decoder p(X|Z)
3. **Causal structure**: Z affects both T and Y

### Architecture Overview

```
Encoder:        Latent:         Decoder:
   X    ------>   Z   ------->    X_hat
                  |
                  |------>  T (propensity)
                  |
                  |------>  Y (outcome)
```

### When to Use CEVAE

**Good for:**
- Proxy variables for hidden confounders
- High-dimensional treatments with latent structure
- When confounders are partially observed

**Limitations:**
- Strong assumptions about latent structure
- Difficult to train
- Requires large samples

### Simplified Implementation Sketch

```python
class CEVAE(nn.Module):
    """
    Simplified CEVAE structure.

    Note: Full implementation requires proper
    variational inference with reparameterization.
    """

    def __init__(self, x_dim, z_dim=20, hidden_dim=100):
        super().__init__()

        # Encoder: q(z|x)
        self.encoder = nn.Sequential(
            nn.Linear(x_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.z_mean = nn.Linear(hidden_dim, z_dim)
        self.z_logvar = nn.Linear(hidden_dim, z_dim)

        # Decoder: p(x|z)
        self.decoder_x = nn.Sequential(
            nn.Linear(z_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, x_dim)
        )

        # Treatment model: p(t|z)
        self.decoder_t = nn.Sequential(
            nn.Linear(z_dim, 1),
            nn.Sigmoid()
        )

        # Outcome model: p(y|z, t)
        self.decoder_y = nn.Sequential(
            nn.Linear(z_dim + 1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def encode(self, x):
        h = self.encoder(x)
        return self.z_mean(h), self.z_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, t):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)

        x_recon = self.decoder_x(z)
        t_pred = self.decoder_t(z)

        zt = torch.cat([z, t], dim=1)
        y_pred = self.decoder_y(zt)

        return x_recon, t_pred, y_pred, mu, logvar
```

## Practical Recommendations

### Model Selection for Causal Tasks

| Task | Recommended Approach |
|------|---------------------|
| ATE estimation | DDML with gradient boosting |
| CATE estimation | Causal Forest, DragonNet |
| High-dimensional confounders | DDML with neural networks |
| Latent confounders | CEVAE (with caution) |
| Propensity scores | Gradient boosting > calibrated MLP |

### Hyperparameter Guidelines

```python
# For propensity score models
propensity_config = {
    'hidden_layer_sizes': (64, 32),  # Not too deep
    'alpha': 0.01,  # Strong regularization
    'early_stopping': True,
    'validation_fraction': 0.1,
    'max_iter': 500
}

# For outcome models in DDML
outcome_config = {
    'hidden_layer_sizes': (128, 64),  # Can be slightly larger
    'alpha': 0.001,  # Moderate regularization
    'early_stopping': True,
    'max_iter': 500
}
```

### Validation Strategies

```python
def validate_causal_model(model, X, y, t, n_bootstrap=100):
    """
    Bootstrap validation for causal effect estimates.
    """
    import numpy as np

    effects = []
    n = len(X)

    for _ in range(n_bootstrap):
        # Bootstrap sample
        idx = np.random.choice(n, n, replace=True)
        X_boot, y_boot, t_boot = X[idx], y[idx], t[idx]

        # Fit and estimate
        model.fit(y_boot, t_boot, X=X_boot)
        effect = model.ate()
        effects.append(effect)

    effects = np.array(effects)

    return {
        'mean': effects.mean(),
        'std': effects.std(),
        'ci_lower': np.percentile(effects, 2.5),
        'ci_upper': np.percentile(effects, 97.5)
    }
```

## Common Pitfalls

### 1. Overfitting Propensity Scores

```python
# WRONG: Complex model overfits propensity
mlp_overfit = MLPClassifier(
    hidden_layer_sizes=(256, 128, 64),
    alpha=0.0001  # Weak regularization
)

# CORRECT: Simpler, regularized model
mlp_proper = MLPClassifier(
    hidden_layer_sizes=(64, 32),
    alpha=0.01  # Stronger regularization
)
```

### 2. Ignoring Positivity Violations

```python
# Always check overlap
ps = estimate_propensity_mlp(X, T)

# Trim extreme values
mask = (ps > 0.01) & (ps < 0.99)
X_trimmed, y_trimmed, t_trimmed = X[mask], y[mask], T[mask]
```

### 3. Not Comparing with Baselines

```python
# Always compare neural network approach with simpler methods
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier

models = {
    'Logistic': LogisticRegression(),
    'GradientBoosting': GradientBoostingClassifier(),
    'MLP': mlp_pipeline
}

for name, model in models.items():
    cv_auc = cross_val_score(model, X, T, cv=5, scoring='roc_auc').mean()
    print(f"{name}: AUC = {cv_auc:.3f}")
```

## Resources

### Papers
- Shi et al. (2019): "Adapting Neural Networks for the Estimation of Treatment Effects" (DragonNet)
- Louizos et al. (2017): "Causal Effect Inference with Deep Latent-Variable Models" (CEVAE)
- Chernozhukov et al. (2018): "Double/Debiased Machine Learning"

### Libraries
- **econml**: Microsoft's library for DDML and CATE estimation
- **causalml**: Uber's library with meta-learners
- **catenets**: Neural network architectures for CATE
