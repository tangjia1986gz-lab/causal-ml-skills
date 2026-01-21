# Binary Choice Models

## Overview

Binary choice models analyze outcomes that take only two values (0 or 1). Common applications include labor force participation, loan default, treatment uptake, and voting decisions.

## Model Comparison

### Linear Probability Model (LPM)

**Specification**:
$$P(Y=1|X) = X\beta$$

**Advantages**:
- Easy to interpret: coefficients are marginal effects
- Robust to distributional misspecification
- Computational simplicity
- Consistent under mild regularity conditions

**Disadvantages**:
- Predictions can fall outside [0,1]
- Heteroskedasticity by construction: $Var(\epsilon|X) = P(1-P)$
- May be inefficient

**When to use**:
- Quick robustness check
- When focus is on average treatment effects
- When predictions near boundaries are rare

```python
import statsmodels.api as sm

# LPM with robust standard errors
X_const = sm.add_constant(X)
lpm = sm.OLS(y, X_const).fit(cov_type='HC1')
```

### Logit Model

**Specification**:
$$P(Y=1|X) = \Lambda(X\beta) = \frac{e^{X\beta}}{1 + e^{X\beta}}$$

**Link Function**: Logit (log-odds)
$$\log\left(\frac{P}{1-P}\right) = X\beta$$

**Advantages**:
- Predictions bounded in [0,1]
- Odds ratio interpretation: $e^{\beta_j}$
- Closed-form for odds ratios

**Marginal Effects**:
$$\frac{\partial P}{\partial x_j} = \beta_j \cdot \Lambda(X\beta) \cdot [1 - \Lambda(X\beta)]$$

```python
from statsmodels.discrete.discrete_model import Logit

logit = Logit(y, X_const).fit()
# Marginal effects at means
mfx_mean = logit.get_margeff(at='mean')
# Average marginal effects
mfx_avg = logit.get_margeff(at='overall')
```

### Probit Model

**Specification**:
$$P(Y=1|X) = \Phi(X\beta)$$

where $\Phi$ is the standard normal CDF.

**Advantages**:
- Natural for latent variable interpretation
- Better for multivariate probit (correlated errors)
- Slightly more efficient if normality holds

**Marginal Effects**:
$$\frac{\partial P}{\partial x_j} = \beta_j \cdot \phi(X\beta)$$

where $\phi$ is the standard normal PDF.

```python
from statsmodels.discrete.discrete_model import Probit

probit = Probit(y, X_const).fit()
mfx = probit.get_margeff()
```

## Coefficient Comparison

Logit and probit coefficients are **not directly comparable** due to different scales:
- Probit coefficients $\approx$ Logit coefficients / 1.6
- Both relate to the latent variable $Y^* = X\beta + \epsilon$

**Rule of thumb**:
$$\beta_{probit} \approx \frac{\beta_{logit}}{1.6} \approx 0.625 \cdot \beta_{logit}$$

## Marginal Effects

### Types

1. **Marginal Effect at Mean (MEM)**:
   $$\frac{\partial P}{\partial x_j}\bigg|_{X=\bar{X}}$$

2. **Average Marginal Effect (AME)**:
   $$\frac{1}{N}\sum_{i=1}^{N}\frac{\partial P}{\partial x_j}\bigg|_{X=X_i}$$

3. **Marginal Effect at Representative Values (MER)**:
   Evaluated at specific covariate values of interest.

### AME vs MEM

| Aspect | AME | MEM |
|--------|-----|-----|
| Interpretation | Average effect in population | Effect for "average" person |
| Causal relevance | Better for ATE | May not represent anyone |
| Robustness | More robust to functional form | Sensitive to specification |
| Standard errors | Requires delta method or bootstrap | Simpler computation |

**Recommendation**: Use AME for causal interpretation (Angrist & Pischke, 2009).

### Computing Marginal Effects

```python
import numpy as np
from scipy import stats

def compute_ame_logit(X, beta):
    """Compute average marginal effects for logit."""
    linear_pred = X @ beta
    prob = 1 / (1 + np.exp(-linear_pred))
    # Derivative of logit CDF
    d_prob = prob * (1 - prob)
    # AME for each coefficient
    ame = np.mean(d_prob[:, np.newaxis] * beta, axis=0)
    return ame

def compute_ame_probit(X, beta):
    """Compute average marginal effects for probit."""
    linear_pred = X @ beta
    # Derivative of probit CDF (standard normal PDF)
    d_prob = stats.norm.pdf(linear_pred)
    # AME for each coefficient
    ame = np.mean(d_prob[:, np.newaxis] * beta, axis=0)
    return ame
```

## Discrete Change Effects

For binary regressors, the marginal effect is the discrete change:
$$\Delta P = P(Y=1|X, D=1) - P(Y=1|X, D=0)$$

```python
def discrete_effect(model, X, treatment_col):
    """Compute discrete change effect for binary treatment."""
    X1 = X.copy()
    X0 = X.copy()
    X1[:, treatment_col] = 1
    X0[:, treatment_col] = 0

    p1 = model.predict(X1)
    p0 = model.predict(X0)

    # Average treatment effect
    ate = np.mean(p1 - p0)
    return ate
```

## Interaction Effects

**Warning**: Interaction effects in nonlinear models are NOT simply the coefficient on the interaction term.

The interaction effect is:
$$\frac{\partial^2 P}{\partial x_1 \partial x_2} = \beta_3 \cdot f'(X\beta) + (\beta_1 + \beta_3 x_2)(\beta_2 + \beta_3 x_1) \cdot f''(X\beta)$$

where $f$ is the link function.

```python
def interaction_effect_logit(X, beta, var1_idx, var2_idx, interaction_idx):
    """
    Compute interaction effect in logit model.
    Following Ai & Norton (2003).
    """
    xb = X @ beta
    F = 1 / (1 + np.exp(-xb))  # CDF
    f = F * (1 - F)  # PDF (first derivative)
    f_prime = f * (1 - 2*F)  # Second derivative

    b1 = beta[var1_idx]
    b2 = beta[var2_idx]
    b12 = beta[interaction_idx]

    # Full interaction effect
    ie = b12 * f + (b1 + b12 * X[:, var2_idx]) * (b2 + b12 * X[:, var1_idx]) * f_prime / f

    return np.mean(ie)
```

## Model Selection

### Likelihood Ratio Test
```python
# Nested model comparison
lr_stat = 2 * (logit_full.llf - logit_restricted.llf)
p_value = stats.chi2.sf(lr_stat, df=num_restrictions)
```

### Information Criteria
```python
# AIC and BIC
aic = logit.aic
bic = logit.bic
```

### Goodness of Fit

1. **McFadden's Pseudo R-squared**:
   $$R^2_{McF} = 1 - \frac{\log L}{\log L_0}$$

2. **Percent Correctly Predicted**:
   ```python
   predictions = (logit.predict() > 0.5).astype(int)
   pcp = np.mean(predictions == y)
   ```

3. **ROC Curve and AUC**:
   ```python
   from sklearn.metrics import roc_auc_score, roc_curve
   auc = roc_auc_score(y, logit.predict())
   ```

## Standard Errors

### Robust (Heteroskedasticity-Consistent)
```python
logit_robust = Logit(y, X).fit(cov_type='HC1')
```

### Clustered
```python
logit_clustered = Logit(y, X).fit(cov_type='cluster', cov_kwds={'groups': cluster_var})
```

### Bootstrap
```python
from sklearn.utils import resample

def bootstrap_ame(y, X, n_bootstrap=1000):
    ames = []
    for _ in range(n_bootstrap):
        idx = resample(range(len(y)))
        model = Logit(y[idx], X[idx]).fit(disp=0)
        ame = model.get_margeff(at='overall').margeff
        ames.append(ame)
    return np.std(ames, axis=0)
```

## Practical Considerations

### Perfect/Quasi-Perfect Separation
When a predictor perfectly separates outcomes, MLE doesn't exist.

**Solutions**:
- Firth's penalized likelihood
- Exact logistic regression
- Drop problematic variables

```python
# Detect separation
from statsmodels.discrete.discrete_model import Logit
try:
    logit = Logit(y, X).fit(maxiter=100)
    if not logit.mle_retvals['converged']:
        print("Warning: Possible separation issue")
except:
    print("Separation detected")
```

### Sample Size Requirements
- Rule of thumb: 10-20 events per predictor
- Small sample corrections available (Firth)

## Causal Interpretation

### Under Conditional Independence
If treatment $D$ is independent of potential outcomes given $X$:
$$E[Y(1) - Y(0)] \approx AME_D$$

### With Selection on Observables
The AME on treatment approximates the ATE when:
1. No unmeasured confounders
2. Common support (overlap)
3. Correct functional form

### Limitations
- Coefficients in logit/probit are NOT treatment effects
- Must compute marginal effects
- Functional form assumptions matter more than in LPM
