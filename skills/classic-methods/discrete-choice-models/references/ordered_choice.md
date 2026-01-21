# Ordered Choice Models

## Overview

Ordered choice models are used when the dependent variable is categorical with a natural ordering but the distances between categories are unknown. Examples include:
- Survey responses (strongly disagree to strongly agree)
- Credit ratings (AAA to D)
- Education levels
- Health status (poor, fair, good, excellent)

## Latent Variable Framework

### Model Structure

Assume a latent continuous variable $Y^*$:
$$Y^* = X\beta + \epsilon$$

The observed ordinal outcome $Y$ is determined by:
$$Y = j \text{ if } \mu_{j-1} < Y^* \leq \mu_j$$

where $\mu_0 = -\infty$, $\mu_J = +\infty$, and $\mu_1 < \mu_2 < ... < \mu_{J-1}$ are threshold (cut-point) parameters.

### Probabilities

For $J$ ordered categories:
$$P(Y = j | X) = F(\mu_j - X\beta) - F(\mu_{j-1} - X\beta)$$

where $F$ is the CDF of $\epsilon$.

## Ordered Logit (Proportional Odds Model)

### Specification

$$\epsilon \sim \text{Logistic}(0, 1)$$
$$F(z) = \frac{e^z}{1 + e^z}$$

### Cumulative Probabilities

$$P(Y \leq j | X) = \frac{e^{\mu_j - X\beta}}{1 + e^{\mu_j - X\beta}}$$

### Proportional Odds Interpretation

$$\log\left(\frac{P(Y \leq j)}{P(Y > j)}\right) = \mu_j - X\beta$$

The odds ratio for $Y \leq j$ vs $Y > j$ is the same for all $j$:
$$OR = e^{-\beta}$$

```python
from statsmodels.miscmodels.ordinal_model import OrderedModel

# Ordered logit
ordered_logit = OrderedModel(y, X, distr='logit')
result = ordered_logit.fit(method='bfgs')
print(result.summary())
```

## Ordered Probit

### Specification

$$\epsilon \sim N(0, 1)$$
$$F(z) = \Phi(z)$$

```python
# Ordered probit
ordered_probit = OrderedModel(y, X, distr='probit')
result = ordered_probit.fit(method='bfgs')
```

### Comparison with Ordered Logit

| Aspect | Ordered Logit | Ordered Probit |
|--------|---------------|----------------|
| Distribution | Logistic | Normal |
| Tails | Heavier | Lighter |
| Odds ratio | Clean interpretation | No simple interpretation |
| Multivariate extension | Harder | Natural (correlated errors) |

## Parallel Lines Assumption

### Definition

The parallel lines (proportional odds) assumption states that the relationship between $X$ and $Y$ is the same regardless of which cumulative probability we consider.

Formally: $\beta$ does not vary across categories $j$.

### Intuition

If we collapsed the ordinal variable into any binary split:
- Categories 1 vs 2,3,4,...
- Categories 1,2 vs 3,4,...
- etc.

The coefficient $\beta$ would be the same for all splits.

### Testing the Assumption

#### Brant Test
```python
def brant_test(y, X):
    """
    Brant test for parallel lines assumption.
    Compares ordered model to series of binary logits.
    """
    import pandas as pd
    from statsmodels.discrete.discrete_model import Logit

    categories = sorted(y.unique())
    J = len(categories)

    # Fit ordered logit
    ordered = OrderedModel(y, X, distr='logit').fit(disp=0)
    beta_ordered = ordered.params[:-J+1]  # Exclude thresholds

    # Fit J-1 binary logits
    binary_betas = []
    for j in range(1, J):
        y_binary = (y > categories[j-1]).astype(int)
        binary_model = Logit(y_binary, sm.add_constant(X)).fit(disp=0)
        binary_betas.append(binary_model.params[1:])  # Exclude constant

    binary_betas = np.array(binary_betas)

    # Test statistic (chi-squared)
    # Simplified version - full test requires variance estimation
    variation = np.var(binary_betas, axis=0)
    test_stat = np.sum(variation) * len(y)

    return test_stat, binary_betas
```

#### Score Test (Generalized)
```python
from scipy import stats

def score_test_parallel_lines(ordered_result, y, X):
    """Score test for parallel lines assumption."""
    # Compute score contributions under null
    # Compare to chi-squared distribution
    pass  # Implementation depends on specific parameterization
```

### When Parallel Lines Fails

Options when the assumption is violated:

1. **Generalized Ordered Logit**: Allow some $\beta_j$ to vary
   ```python
   # Partial proportional odds model
   # Some variables have category-specific effects
   ```

2. **Stereotype Logit Model**: Intermediate between ordered and multinomial

3. **Multinomial Logit**: Abandon ordering assumption

4. **Adjacent Categories Model**: Model adjacent category comparisons

## Marginal Effects in Ordered Models

### Complexity

Unlike binary models, marginal effects in ordered models:
- Sum to zero across categories
- Can have opposite signs for different categories
- Depend on which category is of interest

### Formulas

For ordered logit:
$$\frac{\partial P(Y=j)}{\partial x_k} = -\beta_k [f(\mu_j - X\beta) - f(\mu_{j-1} - X\beta)]$$

where $f$ is the PDF (derivative of CDF).

### Computing Marginal Effects

```python
def marginal_effects_ordered(result, X, category=None):
    """
    Compute marginal effects for ordered choice model.

    Parameters
    ----------
    result : OrderedModel result
    X : array, covariate values
    category : int or None, specific category or all

    Returns
    -------
    effects : array of marginal effects
    """
    from scipy import stats

    beta = result.params[:-len(result.model.endog.unique())+1]
    thresholds = result.params[-len(result.model.endog.unique())+1:]

    # Add boundaries
    mu = np.concatenate([[-np.inf], thresholds, [np.inf]])

    xb = X @ beta
    n_cat = len(mu) - 1

    if result.model.distr == 'logit':
        pdf = lambda z: np.exp(z) / (1 + np.exp(z))**2
    else:  # probit
        pdf = stats.norm.pdf

    effects = []
    for j in range(n_cat):
        # ME for category j
        if j == 0:
            me = -beta * pdf(mu[1] - xb)
        elif j == n_cat - 1:
            me = beta * pdf(mu[-2] - xb)
        else:
            me = -beta * (pdf(mu[j+1] - xb) - pdf(mu[j] - xb))
        effects.append(me)

    if category is not None:
        return effects[category]
    return effects
```

### Average Marginal Effects

```python
def ame_ordered(result, X):
    """Average marginal effects across all observations."""
    n_cat = len(result.model.endog.unique())
    n_vars = len(result.params) - n_cat + 1

    ame = np.zeros((n_cat, n_vars))
    for i in range(len(X)):
        me = marginal_effects_ordered(result, X[i:i+1])
        for j in range(n_cat):
            ame[j] += me[j].flatten()

    return ame / len(X)
```

## Predicted Probabilities

```python
def predict_probs_ordered(result, X):
    """Predict probabilities for each category."""
    return result.predict(X)

# Example
probs = ordered_logit.predict(X_new)
# probs is (n_obs, n_categories) array
```

## Model Comparison

### Likelihood Ratio Test (Nested Models)
```python
# Compare restricted vs unrestricted
lr_stat = 2 * (ll_unrestricted - ll_restricted)
p_value = stats.chi2.sf(lr_stat, df=difference_in_params)
```

### AIC/BIC
```python
aic = result.aic
bic = result.bic
```

## Handling Partial Ordering

When some categories may not be strictly ordered:

1. **Continuation Ratio Model**: Sequential binary choices
2. **Adjacent Categories Model**: Compare consecutive categories
3. **Hybrid approaches**: Combine ordered and unordered components

## Implementation Details

### Identification

The model is identified by:
- Setting $\sigma_\epsilon = 1$ (scale normalization)
- No constant term (absorbed into thresholds)

### Optimization

```python
# Recommended optimization settings
result = OrderedModel(y, X, distr='logit').fit(
    method='bfgs',
    maxiter=1000,
    full_output=True
)
```

### Standard Errors

```python
# Robust standard errors
result_robust = OrderedModel(y, X).fit(cov_type='HC1')

# Clustered standard errors
result_cluster = OrderedModel(y, X).fit(
    cov_type='cluster',
    cov_kwds={'groups': cluster_id}
)
```

## Causal Interpretation

### Treatment Effects

With ordered outcomes, treatment effects can be defined as:
1. **Effect on latent variable**: $\beta_D$ (scale-dependent)
2. **Effect on category probabilities**: $\Delta P(Y=j)$
3. **Effect on expected category**: $E[Y|D=1] - E[Y|D=0]$

### Recommended Approach

Report marginal effects on probabilities for each category:
$$AME_j = E[\hat{P}(Y=j|D=1, X)] - E[\hat{P}(Y=j|D=0, X)]$$

```python
def treatment_effect_ordered(result, X, treatment_col):
    """
    Compute treatment effect on each category probability.
    """
    X1 = X.copy()
    X0 = X.copy()
    X1[:, treatment_col] = 1
    X0[:, treatment_col] = 0

    p1 = result.predict(X1)
    p0 = result.predict(X0)

    # Average effect on each category
    ate_by_category = np.mean(p1 - p0, axis=0)

    return ate_by_category
```

## Best Practices

1. **Always test parallel lines assumption** before interpreting
2. **Report marginal effects**, not just coefficients
3. **Show predicted probabilities** across covariate ranges
4. **Consider category-specific effects** if parallel lines fails
5. **Use robust standard errors** in practice
6. **Check sensitivity** to combining/splitting categories
