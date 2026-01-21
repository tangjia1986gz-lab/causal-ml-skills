# Causal Interpretation with Discrete Outcomes

## Overview

Causal inference with discrete (binary, ordered, multinomial) and count outcomes requires special attention to:
1. **Functional form assumptions**: Nonlinear models impose structure
2. **Marginal effects vs coefficients**: Only marginal effects have causal interpretation
3. **Heterogeneity**: Effects vary across individuals
4. **Identification**: Same requirements as continuous outcomes

## The Fundamental Problem

### Coefficients Are NOT Treatment Effects

In models like:
$$P(Y=1|X, D) = \Lambda(X\beta + \gamma D)$$

The coefficient $\gamma$ is **not** the average treatment effect because:
1. The relationship is nonlinear
2. $\gamma$ captures effect on latent scale
3. Effect on probability depends on baseline probability

### What We Want

$$ATE = E[Y(1) - Y(0)] = E[P(Y=1|X, D=1)] - E[P(Y=1|X, D=0)]$$

This is the Average Marginal Effect (AME) on treatment $D$.

## Marginal Effects as Causal Parameters

### Binary Outcomes

For binary treatment $D$ in logit/probit:

**Average Treatment Effect (via AME)**:
$$\widehat{ATE} = \frac{1}{N}\sum_{i=1}^{N}\left[\hat{P}(Y_i=1|X_i, D_i=1) - \hat{P}(Y_i=1|X_i, D_i=0)\right]$$

```python
def ate_binary_model(model, X, treatment_col):
    """
    Compute ATE from binary choice model.

    Parameters
    ----------
    model : fitted Logit/Probit model
    X : array, covariates including treatment
    treatment_col : int, column index of treatment variable

    Returns
    -------
    ate : float, average treatment effect
    se : float, standard error (via bootstrap)
    """
    import numpy as np

    X1 = X.copy()
    X0 = X.copy()
    X1[:, treatment_col] = 1
    X0[:, treatment_col] = 0

    p1 = model.predict(X1)
    p0 = model.predict(X0)

    ate = np.mean(p1 - p0)

    return ate


def ate_with_bootstrap_se(model_class, y, X, treatment_col, n_bootstrap=500):
    """Compute ATE with bootstrap standard errors."""
    from sklearn.utils import resample

    ates = []
    n = len(y)

    for _ in range(n_bootstrap):
        idx = resample(range(n), n_samples=n)
        model_b = model_class(y[idx], X[idx]).fit(disp=0)
        ate_b = ate_binary_model(model_b, X, treatment_col)
        ates.append(ate_b)

    return np.mean(ates), np.std(ates)
```

### Continuous Covariates

For continuous covariate $X_j$ (not treatment):
$$AME_j = \frac{1}{N}\sum_{i=1}^{N}\frac{\partial P(Y_i=1|X_i)}{\partial X_{ij}}$$

This can be interpreted causally only if:
1. $X_j$ is exogenous (no confounding)
2. The effect is the same regardless of how $X_j$ changes
3. SUTVA holds

## LPM as Robustness Check

### Why LPM for Causal Inference?

Angrist and Pischke (2009) argue that LPM often suffices for causal questions:

1. **Direct interpretation**: Coefficients ARE marginal effects
2. **Robust to misspecification**: Consistent if mean correct
3. **Handles fixed effects**: Easy to include many dummies
4. **IV/2SLS works directly**: No special modifications needed

```python
import statsmodels.api as sm

def lpm_ate(y, X, treatment_col):
    """
    LPM estimate of ATE.
    Coefficient on treatment IS the ATE (under assumptions).
    """
    X_const = sm.add_constant(X)
    lpm = sm.OLS(y, X_const).fit(cov_type='HC1')

    # Treatment coefficient is ATE
    ate = lpm.params[treatment_col + 1]  # +1 for constant
    se = lpm.bse[treatment_col + 1]

    return ate, se, lpm
```

### When LPM May Fail

1. Predicted probabilities far outside [0,1]
2. Effects near boundary probabilities
3. Strong nonlinearity in true DGP

## Identification Requirements

### Conditional Independence

$$Y(d) \perp D | X$$

- Same as for continuous outcomes
- Selection on observables only
- All confounders included in $X$

### Common Support (Overlap)

$$0 < P(D=1|X) < 1$$

More critical for nonlinear models:
- Effects near $P=0$ or $P=1$ are poorly identified
- Extrapolation is dangerous

```python
def check_overlap(propensity_scores, threshold=0.05):
    """
    Check common support condition.
    """
    min_ps = propensity_scores.min()
    max_ps = propensity_scores.max()

    violations = ((propensity_scores < threshold) |
                  (propensity_scores > 1 - threshold)).sum()

    print(f"Propensity score range: [{min_ps:.3f}, {max_ps:.3f}]")
    print(f"Observations near boundary: {violations} ({100*violations/len(propensity_scores):.1f}%)")

    return violations == 0
```

### SUTVA

- No interference between units
- Single version of treatment
- Same requirements as continuous outcomes

## Heterogeneous Treatment Effects

### CATEs with Discrete Outcomes

$$CATE(x) = E[Y(1) - Y(0) | X = x] = P(Y=1|X=x, D=1) - P(Y=1|X=x, D=0)$$

```python
def compute_cate(model, X_grid, treatment_col):
    """
    Compute CATE across covariate values.
    """
    cates = []

    for x in X_grid:
        x1 = x.copy()
        x0 = x.copy()
        x1[treatment_col] = 1
        x0[treatment_col] = 0

        p1 = model.predict(x1.reshape(1, -1))[0]
        p0 = model.predict(x0.reshape(1, -1))[0]

        cates.append(p1 - p0)

    return np.array(cates)
```

### Natural Heterogeneity in Nonlinear Models

Even with constant $\gamma$, treatment effects vary:
$$\frac{\partial P}{\partial D} = \gamma \cdot f(X\beta + \gamma D)$$

This depends on $X\beta$ (baseline characteristics).

### Interaction Effects

To test for heterogeneity, include interactions:
$$Y^* = X\beta + \gamma D + \delta(X \times D) + \epsilon$$

But interpretation requires careful computation (see Ai & Norton, 2003).

```python
def interaction_marginal_effect(model, X, treatment_col, moderator_col, moderator_values):
    """
    Compute how treatment effect varies with moderator.
    """
    effects = []

    for m in moderator_values:
        X_mod = X.copy()
        X_mod[:, moderator_col] = m

        ate_m = ate_binary_model(model, X_mod, treatment_col)
        effects.append({'moderator': m, 'effect': ate_m})

    return pd.DataFrame(effects)
```

## Ordered Outcomes

### Treatment Effects on Category Probabilities

$$ATE_j = E[P(Y=j|D=1, X)] - E[P(Y=j|D=0, X)]$$

Effects sum to zero across categories.

```python
def ate_ordered(model, X, treatment_col):
    """
    Compute treatment effects on each ordered category.
    """
    X1 = X.copy()
    X0 = X.copy()
    X1[:, treatment_col] = 1
    X0[:, treatment_col] = 0

    p1 = model.predict(X1)  # (n, J) array
    p0 = model.predict(X0)

    # ATE for each category
    ate_by_category = np.mean(p1 - p0, axis=0)

    # Check: should sum to zero
    assert np.abs(ate_by_category.sum()) < 1e-10

    return ate_by_category
```

### Summary Measures

1. **Effect on expected value** (treating categories as numeric):
   $$E[Y|D=1] - E[Y|D=0] = \sum_j j \cdot ATE_j$$

2. **Effect on probability of exceeding threshold**:
   $$P(Y > k | D=1) - P(Y > k | D=0)$$

## Multinomial Outcomes

### Treatment Effects on Choice Probabilities

$$ATE_j = E[P(Y=j|D=1, X)] - E[P(Y=j|D=0, X)]$$

Effects sum to zero across alternatives.

### Substitution Patterns

Treatment may:
- Increase probability of some alternatives
- Decrease others
- Pattern depends on IIA and model specification

```python
def ate_multinomial(model, X, treatment_col):
    """
    Compute treatment effects on multinomial choice probabilities.
    """
    X1 = X.copy()
    X0 = X.copy()
    X1[:, treatment_col] = 1
    X0[:, treatment_col] = 0

    p1 = model.predict(X1)
    p0 = model.predict(X0)

    ate_by_alt = np.mean(p1 - p0, axis=0)

    return ate_by_alt
```

## Count Outcomes

### Average Treatment Effect

$$ATE = E[Y(1)] - E[Y(0)] = E[e^{X\beta + \gamma}] - E[e^{X\beta}]$$

Note: This is NOT simply $e^\gamma - 1$ times baseline.

```python
def ate_count(model, X, treatment_col):
    """
    Compute ATE for count outcome.
    """
    X1 = X.copy()
    X0 = X.copy()
    X1[:, treatment_col] = 1
    X0[:, treatment_col] = 0

    mu1 = model.predict(X1)
    mu0 = model.predict(X0)

    ate = np.mean(mu1 - mu0)

    # Also compute percent change
    pct_change = np.mean((mu1 - mu0) / mu0) * 100

    return ate, pct_change
```

### Incidence Rate Ratio

$$IRR = \frac{E[Y|D=1]}{E[Y|D=0]}$$

Under log-linear model: $IRR \approx e^\gamma$ (approximate, exact only at mean).

## Instrumental Variables with Discrete Outcomes

### Forbidden Regression Problem

Cannot simply plug fitted values from first stage into nonlinear second stage.

### Solutions

1. **Control Function Approach**:
   - First stage: $D = X\pi + Z\theta + v$
   - Include $\hat{v}$ in second stage: $P(Y=1) = \Lambda(X\beta + \gamma D + \rho\hat{v})$

2. **Special Regressor Method** (Lewbel):
   - Requires special regressor with large support
   - More complex but consistent

3. **2SLS with LPM**:
   - Use LPM for both stages
   - Simple and often adequate

```python
def control_function_probit(y, X, D, Z):
    """
    Control function approach for endogenous treatment in probit.
    """
    from statsmodels.discrete.discrete_model import Probit

    # First stage: regress D on X and Z
    XZ = np.column_stack([X, Z])
    XZ_const = sm.add_constant(XZ)
    first_stage = sm.OLS(D, XZ_const).fit()
    v_hat = first_stage.resid

    # Second stage: include residuals
    X_D_v = np.column_stack([X, D, v_hat])
    X_D_v_const = sm.add_constant(X_D_v)
    second_stage = Probit(y, X_D_v_const).fit()

    # Test endogeneity: coefficient on v_hat
    endo_test_stat = second_stage.tvalues[-1]

    return second_stage, endo_test_stat
```

## Difference-in-Differences with Discrete Outcomes

### Nonlinear DiD

Standard DiD assumes additive effects. With nonlinear models:
$$P(Y=1|T, Post) = \Lambda(\alpha + \beta \cdot Post + \gamma \cdot T + \delta \cdot (T \times Post))$$

$\delta$ is NOT the DiD effect on probability.

### Computing the DiD Effect

$$DiD = [P(Y=1|T=1, Post=1) - P(Y=1|T=1, Post=0)] -$$
$$[P(Y=1|T=0, Post=1) - P(Y=1|T=0, Post=0)]$$

```python
def did_nonlinear(model, X_template, treat_col, post_col):
    """
    Compute DiD effect for nonlinear model.
    """
    # Four predictions
    X_11 = X_template.copy()
    X_11[treat_col] = 1
    X_11[post_col] = 1

    X_10 = X_template.copy()
    X_10[treat_col] = 1
    X_10[post_col] = 0

    X_01 = X_template.copy()
    X_01[treat_col] = 0
    X_01[post_col] = 1

    X_00 = X_template.copy()
    X_00[treat_col] = 0
    X_00[post_col] = 0

    p_11 = model.predict(X_11.reshape(1, -1))[0]
    p_10 = model.predict(X_10.reshape(1, -1))[0]
    p_01 = model.predict(X_01.reshape(1, -1))[0]
    p_00 = model.predict(X_00.reshape(1, -1))[0]

    did = (p_11 - p_10) - (p_01 - p_00)

    return did
```

### Recommendation

Use LPM for DiD with binary outcomes for:
- Simple interpretation
- Consistent under parallel trends on probability scale
- Robust to some forms of misspecification

## Best Practices Summary

1. **Always report marginal effects** (preferably AME), not just coefficients
2. **Use LPM as robustness check** for binary outcomes
3. **Check overlap/common support** especially for nonlinear models
4. **Bootstrap standard errors** for marginal effects when possible
5. **Compute treatment effects correctly** accounting for nonlinearity
6. **Consider heterogeneity** - effects vary even with constant coefficients
7. **Test sensitivity** to functional form assumptions
8. **For IV/DiD**, consider control function or LPM approaches

## References

- Angrist, J. D., & Pischke, J.-S. (2009). Mostly Harmless Econometrics. Princeton University Press.
- Ai, C., & Norton, E. C. (2003). Interaction terms in logit and probit models. Economics Letters, 80(1), 123-129.
- Wooldridge, J. M. (2010). Econometric Analysis of Cross Section and Panel Data. MIT Press.
- Cameron, A. C., & Trivedi, P. K. (2005). Microeconometrics: Methods and Applications. Cambridge University Press.
