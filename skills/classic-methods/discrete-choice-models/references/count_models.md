# Count Data Models

## Overview

Count data models analyze non-negative integer outcomes representing counts of events. Examples include:
- Number of doctor visits
- Patent applications
- Traffic accidents
- Crime incidents
- Website clicks
- Paper citations

## Key Characteristics of Count Data

1. **Non-negative integers**: $Y \in \{0, 1, 2, ...\}$
2. **Often right-skewed**: Many zeros and small values
3. **Variance structure**: Often variance $\neq$ mean
4. **Overdispersion common**: Variance > mean

## Poisson Regression

### Distribution

$$P(Y = y | \mu) = \frac{e^{-\mu} \mu^y}{y!}$$

**Key property**: $E[Y] = Var(Y) = \mu$ (equidispersion)

### Model Specification

$$\log(\mu_i) = X_i\beta$$
$$\mu_i = e^{X_i\beta}$$

### Interpretation

- $\beta_j$: A one-unit increase in $x_j$ changes $\log(\mu)$ by $\beta_j$
- $e^{\beta_j}$: Incidence Rate Ratio (IRR) - multiplicative effect on count
- $e^{\beta_j} - 1$: Percent change in expected count

```python
import statsmodels.api as sm
from statsmodels.discrete.discrete_model import Poisson

X_const = sm.add_constant(X)
poisson = Poisson(y, X_const).fit()
print(poisson.summary())

# Incidence rate ratios
irr = np.exp(poisson.params)
print("IRR:", irr)
```

### Marginal Effects

For log-linear model:
$$\frac{\partial E[Y]}{\partial x_j} = \beta_j \cdot \mu = \beta_j \cdot e^{X\beta}$$

```python
# Average marginal effects
mfx = poisson.get_margeff()
print(mfx.summary())
```

### Limitations

1. **Equidispersion rarely holds**: Real data usually has $Var(Y) > E[Y]$
2. **Excess zeros**: Poisson may underpredict zeros
3. **Sensitive to outliers**: Large counts have high leverage

## Testing for Overdispersion

### Definition

Overdispersion: $Var(Y|X) > E[Y|X]$

### Cameron-Trivedi Test

Test $H_0$: $Var(Y) = E[Y]$ vs $H_1$: $Var(Y) = E[Y] + \alpha g(E[Y])$

```python
def cameron_trivedi_test(y, fitted_values):
    """
    Test for overdispersion in Poisson model.

    Returns
    -------
    t_stat : float
    p_value : float
    """
    from scipy import stats

    # Auxiliary regression
    mu = fitted_values
    aux_dep = ((y - mu)**2 - y) / mu

    # Regress on mu (or mu^2)
    aux_reg = sm.OLS(aux_dep, mu).fit()

    t_stat = aux_reg.tvalues[0]
    p_value = aux_reg.pvalues[0]

    return t_stat, p_value, aux_reg.params[0]

# Usage
t, p, alpha = cameron_trivedi_test(y, poisson.fittedvalues)
print(f"Overdispersion test: t={t:.3f}, p={p:.4f}")
if p < 0.05 and alpha > 0:
    print("Evidence of overdispersion")
```

### Dispersion Statistic

$$\hat{\alpha} = \frac{1}{n-k}\sum_{i=1}^{n}\frac{(y_i - \hat{\mu}_i)^2}{\hat{\mu}_i}$$

If $\hat{\alpha} > 1$: Overdispersion
If $\hat{\alpha} < 1$: Underdispersion

```python
def dispersion_statistic(y, fitted_values, n_params):
    """Calculate dispersion statistic."""
    n = len(y)
    pearson_chi2 = np.sum((y - fitted_values)**2 / fitted_values)
    return pearson_chi2 / (n - n_params)
```

## Negative Binomial Regression

### Distribution

$$P(Y = y | \mu, \alpha) = \frac{\Gamma(y + \alpha^{-1})}{\Gamma(\alpha^{-1})\Gamma(y+1)} \left(\frac{\alpha^{-1}}{\alpha^{-1} + \mu}\right)^{\alpha^{-1}} \left(\frac{\mu}{\alpha^{-1} + \mu}\right)^y$$

**Key property**: $Var(Y) = \mu + \alpha\mu^2$ (NB2) or $Var(Y) = \mu + \alpha\mu$ (NB1)

- $\alpha$: Overdispersion parameter
- $\alpha \to 0$: Converges to Poisson

### NB1 vs NB2

| Model | Variance Function | When to Use |
|-------|-------------------|-------------|
| NB1 | $\mu(1 + \alpha)$ | Linear overdispersion |
| NB2 | $\mu + \alpha\mu^2$ | Quadratic overdispersion |

```python
from statsmodels.discrete.discrete_model import NegativeBinomial, NegativeBinomialP

# NB2 (default, variance = mu + alpha*mu^2)
nb2 = NegativeBinomial(y, X_const).fit()
print(nb2.summary())

# NB1 (variance = mu(1 + alpha))
nb1 = NegativeBinomialP(y, X_const, p=1).fit()
```

### Interpretation

Same as Poisson:
- Coefficients are log IRRs
- $e^{\beta}$ is the IRR
- Marginal effects scale with $\mu$

## Quasi-Poisson (Robust Poisson)

When only the mean structure is correctly specified:

$$E[Y|X] = e^{X\beta}$$

But variance may be:
$$Var(Y|X) = \phi \cdot \mu$$

### Robust Standard Errors

Use Poisson MLE with robust (sandwich) standard errors:

```python
# Quasi-Poisson via robust SEs
poisson_robust = Poisson(y, X_const).fit(cov_type='HC0')

# Or using GLM
from statsmodels.genmod.generalized_linear_model import GLM
from statsmodels.genmod.families import Poisson as PoissonFamily

glm_poisson = GLM(y, X_const, family=PoissonFamily()).fit(scale='X2')
```

### When to Use

- Point estimates same as Poisson
- Standard errors robust to overdispersion
- Preferred when only interested in mean effects, not full distribution

## Zero-Inflated Models

### Motivation

Some processes generate excess zeros from a separate mechanism:
- Doctor visits: Some people never go (structural zeros)
- Patent counts: Some firms don't innovate (non-innovators)

### Model Structure

Two processes:
1. **Zero-generating process**: $P(\text{structural zero}) = \pi$
2. **Count process**: $P(Y=y | \text{not structural zero})$

$$P(Y = 0) = \pi + (1-\pi) \cdot f(0)$$
$$P(Y = y) = (1-\pi) \cdot f(y), \quad y > 0$$

### Zero-Inflated Poisson (ZIP)

```python
from statsmodels.discrete.count_model import ZeroInflatedPoisson

# inflate formula models the zero-generating process
zip_model = ZeroInflatedPoisson(y, X_const, exog_infl=Z_const).fit()
print(zip_model.summary())
```

### Zero-Inflated Negative Binomial (ZINB)

```python
from statsmodels.discrete.count_model import ZeroInflatedNegativeBinomialP

zinb = ZeroInflatedNegativeBinomialP(y, X_const, exog_infl=Z_const).fit()
```

### Interpretation

Two sets of coefficients:
1. **Count model coefficients**: Effect on count (among those who can have positive counts)
2. **Inflation model coefficients**: Effect on probability of being a structural zero

### Vuong Test

Test ZIP/ZINB against standard Poisson/NB:

```python
def vuong_test(y, ll_zip, ll_poisson):
    """
    Vuong test comparing non-nested models.

    H0: models are equivalent
    H1: ZIP is better (positive statistic) or Poisson is better (negative)
    """
    m = ll_zip - ll_poisson  # Individual log-likelihood differences
    n = len(y)

    v_stat = np.sqrt(n) * np.mean(m) / np.std(m)
    p_value = 2 * stats.norm.sf(np.abs(v_stat))

    return v_stat, p_value
```

## Hurdle Models

### Difference from Zero-Inflated

- **Zero-inflated**: Zeros come from two sources
- **Hurdle**: All zeros from one process, all positives from another

### Structure

1. **Hurdle process**: Binary model for $P(Y > 0)$
2. **Truncated count process**: $P(Y = y | Y > 0)$

$$P(Y = 0) = 1 - \Phi(X\gamma)$$
$$P(Y = y | Y > 0) = \frac{f(y)}{1 - f(0)}, \quad y > 0$$

```python
def fit_hurdle_poisson(y, X):
    """Fit hurdle Poisson model."""
    from statsmodels.discrete.discrete_model import Logit, Poisson

    # Stage 1: Binary model for y > 0
    y_binary = (y > 0).astype(int)
    hurdle = Logit(y_binary, X).fit(disp=0)

    # Stage 2: Truncated Poisson for y > 0
    mask = y > 0
    # Note: Should use truncated Poisson, approximation with regular
    count_model = Poisson(y[mask], X[mask]).fit(disp=0)

    return hurdle, count_model
```

## Model Selection

### Comparing Count Models

```python
def compare_count_models(y, X):
    """Compare Poisson, NB, ZIP, ZINB."""
    results = {}

    # Poisson
    pois = Poisson(y, X).fit(disp=0)
    results['Poisson'] = {'aic': pois.aic, 'bic': pois.bic, 'llf': pois.llf}

    # Negative Binomial
    nb = NegativeBinomial(y, X).fit(disp=0)
    results['NegBin'] = {'aic': nb.aic, 'bic': nb.bic, 'llf': nb.llf}

    # ZIP
    zip_m = ZeroInflatedPoisson(y, X, exog_infl=X).fit(disp=0)
    results['ZIP'] = {'aic': zip_m.aic, 'bic': zip_m.bic, 'llf': zip_m.llf}

    # ZINB
    zinb = ZeroInflatedNegativeBinomialP(y, X, exog_infl=X).fit(disp=0)
    results['ZINB'] = {'aic': zinb.aic, 'bic': zinb.bic, 'llf': zinb.llf}

    return pd.DataFrame(results).T
```

### Decision Tree

```
Start with Poisson
    │
    ├── Test overdispersion
    │       │
    │       ├── No → Poisson OK
    │       │
    │       └── Yes → Use NegBin
    │               │
    │               └── Check excess zeros
    │                       │
    │                       ├── Proportion zeros > expected → ZIP/ZINB
    │                       │
    │                       └── Zeros as expected → NegBin OK
```

## Exposure and Offsets

### Rate Models

When counts occur over different exposure periods/populations:
$$\log(\mu_i) = \log(t_i) + X_i\beta$$

where $t_i$ is the exposure (time, population, etc.).

```python
# Include offset for exposure
poisson_rate = Poisson(y, X_const, offset=np.log(exposure)).fit()

# Coefficients now represent effects on rate, not count
```

### Incidence Rates

$$\text{Rate} = \frac{\text{Count}}{\text{Exposure}} = \frac{\mu}{t}$$

## Marginal Effects

### Semi-Elasticity

$$\frac{\partial \log E[Y]}{\partial x_j} = \beta_j$$

### Marginal Effect

$$\frac{\partial E[Y]}{\partial x_j} = \beta_j \cdot \mu_i$$

### Average Marginal Effect

$$AME_j = \frac{1}{n}\sum_{i=1}^{n} \beta_j \cdot \hat{\mu}_i = \beta_j \cdot \bar{\hat{\mu}}$$

```python
def marginal_effects_count(model, X, method='average'):
    """
    Compute marginal effects for count model.

    Parameters
    ----------
    model : fitted count model
    X : array
    method : 'average', 'atmean', or array of values

    Returns
    -------
    mfx : array of marginal effects
    """
    beta = model.params
    mu = model.predict(X)

    if method == 'average':
        return beta * np.mean(mu)
    elif method == 'atmean':
        mu_mean = model.predict(X.mean(axis=0).reshape(1, -1))
        return beta * mu_mean
    else:
        return beta * model.predict(method.reshape(1, -1))
```

## Causal Interpretation

### Treatment Effects

For binary treatment $D$:
$$ATE = E[Y(1)] - E[Y(0)] = E[e^{X\beta + \beta_D}] - E[e^{X\beta}]$$

```python
def ate_count_model(model, X, treatment_col):
    """
    Compute ATE from count model.
    """
    X1 = X.copy()
    X0 = X.copy()
    X1[:, treatment_col] = 1
    X0[:, treatment_col] = 0

    mu1 = model.predict(X1)
    mu0 = model.predict(X0)

    ate = np.mean(mu1 - mu0)
    ate_pct = np.mean((mu1 - mu0) / mu0) * 100  # Percent change

    return ate, ate_pct
```

### Ratio vs Difference

Two ways to express treatment effect:
1. **Difference**: $E[Y(1)] - E[Y(0)]$ (additive)
2. **Ratio**: $E[Y(1)] / E[Y(0)] = e^{\beta_D}$ (multiplicative, IRR)

IRR is often more interpretable for counts.

## Practical Considerations

### Small Counts and Zeros

- Many zeros: Consider ZIP/ZINB or hurdle
- All zeros in some groups: Perfect prediction issues

### Large Counts

- Poisson approximation to binomial
- Consider whether normal approximation sufficient

### Clustered Data

```python
# Clustered standard errors
poisson_clustered = Poisson(y, X).fit(
    cov_type='cluster',
    cov_kwds={'groups': cluster_id}
)
```

### Panel Count Data

- Fixed effects Poisson
- Random effects Poisson/NB
- See panel data methods

```python
# Fixed effects Poisson
from statsmodels.discrete.conditional_models import ConditionalPoisson

fe_poisson = ConditionalPoisson(y, X, groups=panel_id).fit()
```

## Best Practices

1. **Always test for overdispersion** before using Poisson
2. **Compare observed vs predicted zeros** to check for excess zeros
3. **Use robust standard errors** as default
4. **Report IRRs along with coefficients** for interpretability
5. **Consider exposure/offset** when appropriate
6. **Check for influential observations** (large counts)
