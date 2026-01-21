# Multinomial Choice Models

## Overview

Multinomial choice models analyze decisions among multiple unordered alternatives. Applications include:
- Transportation mode choice (car, bus, train, bike)
- Brand selection
- Occupational choice
- Residential location choice
- Voting among multiple candidates

## Random Utility Framework

### Setup

Decision maker $i$ chooses among $J$ alternatives. The utility from alternative $j$ is:
$$U_{ij} = V_{ij} + \epsilon_{ij}$$

where:
- $V_{ij} = X_i\beta_j + Z_{ij}\gamma$ is the systematic (deterministic) component
- $X_i$ = individual characteristics
- $Z_{ij}$ = alternative-specific characteristics
- $\epsilon_{ij}$ = random component

### Choice Rule

Individual $i$ chooses alternative $j$ if:
$$U_{ij} > U_{ik} \quad \forall k \neq j$$

## Multinomial Logit (MNL)

### Assumptions

$\epsilon_{ij}$ are independently and identically distributed Type I Extreme Value (Gumbel):
$$F(\epsilon) = e^{-e^{-\epsilon}}$$

### Choice Probabilities

$$P(Y_i = j | X_i) = \frac{e^{V_{ij}}}{\sum_{k=1}^{J} e^{V_{ik}}}$$

### Types of Variables

1. **Case-specific variables** ($X_i$): Same for all alternatives
   - Individual characteristics (age, income)
   - Result in alternative-specific coefficients $\beta_j$

2. **Alternative-specific variables** ($Z_{ij}$): Vary across alternatives
   - Price, travel time, quality
   - Result in generic coefficients $\gamma$

```python
import statsmodels.api as sm
from statsmodels.discrete.discrete_model import MNLogit

# Case-specific variables only
mnl = MNLogit(y, X)
result = mnl.fit()
print(result.summary())
```

### Identification

- One category must be the reference (base) category
- Coefficients interpreted relative to reference
- Typically choose most common category or theoretically meaningful baseline

```python
# y should be encoded 0, 1, 2, ..., J-1
# Category 0 is the reference by default
```

## Interpreting Coefficients

### Relative Risk Ratios (RRR)

$$RRR_{jk} = \frac{P(Y=j)/P(Y=k)}{P(Y=j')/P(Y=k')} = e^{\beta_j - \beta_k}$$

For a unit change in $x$:
$$\frac{P(Y=j|X+1)/P(Y=base|X+1)}{P(Y=j|X)/P(Y=base|X)} = e^{\beta_j}$$

```python
# Relative risk ratios
rrr = np.exp(result.params)
```

### Marginal Effects

The marginal effect of $x_k$ on $P(Y=j)$:
$$\frac{\partial P_j}{\partial x_k} = P_j \left[\beta_{jk} - \sum_{m=1}^{J} P_m \beta_{mk}\right]$$

**Key properties**:
- Sum to zero across alternatives
- Can be positive or negative even if $\beta_{jk} > 0$

```python
# Average marginal effects
mfx = result.get_margeff()
print(mfx.summary())
```

## Independence of Irrelevant Alternatives (IIA)

### Definition

The IIA property states that the ratio of choice probabilities between any two alternatives is independent of other alternatives:
$$\frac{P_j}{P_k} = \frac{e^{V_j}}{e^{V_k}}$$

### Red Bus/Blue Bus Problem

Classic example showing IIA violation:
- Original: Car (50%) vs Red Bus (50%)
- Add Blue Bus (identical to Red Bus)
- MNL predicts: Car (33%), Red Bus (33%), Blue Bus (33%)
- Reality: Car (50%), Red Bus (25%), Blue Bus (25%)

### Testing IIA

#### Hausman-McFadden Test

Compare full model to model excluding one alternative:
$$H = (\hat{\beta}_S - \hat{\beta}_F)'[\hat{V}_S - \hat{V}_F]^{-1}(\hat{\beta}_S - \hat{\beta}_F) \sim \chi^2_k$$

```python
def hausman_mcfadden_test(y, X, excluded_alt):
    """
    Test IIA by excluding one alternative.

    Parameters
    ----------
    y : array, choice variable
    X : array, covariates
    excluded_alt : int, alternative to exclude

    Returns
    -------
    stat : float, test statistic
    p_value : float
    """
    from scipy import stats

    # Full model
    full = MNLogit(y, X).fit(disp=0)
    beta_f = full.params.flatten()
    V_f = full.cov_params()

    # Restricted model (exclude observations choosing excluded_alt)
    mask = y != excluded_alt
    y_r = y[mask]
    X_r = X[mask]

    # Recode alternatives
    y_r_recoded = y_r.copy()
    y_r_recoded[y_r_recoded > excluded_alt] -= 1

    restricted = MNLogit(y_r_recoded, X_r).fit(disp=0)
    beta_s = restricted.params.flatten()

    # Align parameters (complex in practice)
    # This is a simplified version

    # Test statistic
    diff = beta_s - beta_f[:len(beta_s)]  # Simplified
    V_diff = restricted.cov_params() - V_f[:len(beta_s), :len(beta_s)]

    try:
        stat = diff @ np.linalg.inv(V_diff) @ diff
        p_value = stats.chi2.sf(stat, len(diff))
    except:
        stat, p_value = np.nan, np.nan

    return stat, p_value
```

#### Small-Hsiao Test

Uses split sample approach to avoid negative variance issues.

```python
def small_hsiao_test(y, X, excluded_alt, n_splits=2):
    """Small-Hsiao test for IIA."""
    # Split sample randomly
    n = len(y)
    perm = np.random.permutation(n)
    split1 = perm[:n//2]
    split2 = perm[n//2:]

    # Estimate on split 1
    full_1 = MNLogit(y[split1], X[split1]).fit(disp=0)

    # Estimate restricted on split 2
    mask = y[split2] != excluded_alt
    y_r = y[split2][mask]
    X_r = X[split2][mask]
    y_r_recoded = y_r.copy()
    y_r_recoded[y_r_recoded > excluded_alt] -= 1
    rest_2 = MNLogit(y_r_recoded, X_r).fit(disp=0)

    # Test statistic
    stat = -2 * (rest_2.llf - full_1.llf)
    k = full_1.params.size - rest_2.params.size
    p_value = stats.chi2.sf(stat, k)

    return stat, p_value
```

### When IIA Fails

Options when IIA is violated:

1. **Nested Logit**: Group similar alternatives
2. **Mixed (Random Parameters) Logit**: Allow coefficient heterogeneity
3. **Probit**: Correlated errors
4. **Specify more alternative-specific variables**

## Nested Logit

### Structure

Alternatives are grouped into nests. IIA holds within nests but not across nests.

$$P(Y=j) = P(Y=j|nest_m) \times P(nest_m)$$

### Utility Specification

$$V_{ij} = W_{im} + Y_{ij}$$

where:
- $W_{im}$ = characteristics of nest $m$
- $Y_{ij}$ = characteristics within nest

### Inclusive Value (Log-Sum)

$$IV_m = \log\left(\sum_{j \in B_m} e^{Y_{ij}/\lambda_m}\right)$$

- $\lambda_m \in (0,1]$: Nesting parameter (dissimilarity)
- $\lambda_m = 1$: Reduces to MNL (no correlation within nest)
- $\lambda_m$ close to 0: High correlation within nest

```python
# Nested logit is not in base statsmodels
# Use specialized packages like pylogit

# pip install pylogit
import pylogit as pl

# Define nesting structure
nest_spec = {
    'public_transit': [1, 2],  # bus, train
    'private': [3, 4]  # car, bike
}

# Estimate nested logit
# (requires long-format data)
```

## Mixed Logit (Random Parameters Logit)

### Specification

Allow coefficients to vary across individuals:
$$\beta_i \sim f(\beta | \theta)$$

Common distributions:
- Normal: $\beta_i \sim N(\mu, \sigma^2)$
- Log-normal: $\log(\beta_i) \sim N(\mu, \sigma^2)$ (ensures positive)
- Triangular

### Choice Probability

$$P(Y_i = j) = \int \frac{e^{V_{ij}(\beta)}}{\sum_k e^{V_{ik}(\beta)}} f(\beta|\theta) d\beta$$

Estimated via simulation (simulated maximum likelihood).

```python
# Mixed logit requires simulation
# Use pylogit or biogeme

# Example with pylogit
# Requires panel/long format data
```

## Conditional Logit

When only alternative-specific variables matter:
$$P(Y_i = j) = \frac{e^{Z_{ij}\gamma}}{\sum_k e^{Z_{ik}\gamma}}$$

All individuals have the same $\gamma$ (not alternative-specific).

```python
# For conditional logit with alternative-specific vars
# Data must be in long format (one row per alternative per individual)

# Using statsmodels
from statsmodels.discrete.conditional_models import ConditionalLogit

# group = individual identifier
clogit = ConditionalLogit(y_long, X_long, groups=group_id)
result = clogit.fit()
```

## Marginal Effects Details

### For Case-Specific Variables

$$\frac{\partial P_j}{\partial x_k} = P_j \left(\beta_{jk} - \sum_{m} P_m \beta_{mk}\right)$$

### For Alternative-Specific Variables

$$\frac{\partial P_j}{\partial z_{jk}} = P_j(1 - P_j) \gamma_k$$
$$\frac{\partial P_j}{\partial z_{mk}} = -P_j P_m \gamma_k \quad (m \neq j)$$

### Cross-Elasticities

For alternative-specific variable:
$$\eta_{jm} = \frac{\partial P_j}{\partial z_m} \cdot \frac{z_m}{P_j}$$

In MNL:
$$\eta_{jm} = -P_m \gamma z_m \quad (j \neq m)$$

All cross-elasticities are equal (IIA consequence).

## Practical Implementation

### Data Preparation

```python
def prepare_long_format(df, choice_col, id_col, alt_specific_vars):
    """
    Convert wide-format choice data to long format.
    """
    # Each row becomes J rows (one per alternative)
    n_alts = df[choice_col].nunique()

    long_data = []
    for _, row in df.iterrows():
        for alt in range(n_alts):
            new_row = {
                'id': row[id_col],
                'alternative': alt,
                'choice': 1 if row[choice_col] == alt else 0
            }
            # Add case-specific vars
            for col in df.columns:
                if col not in [choice_col, id_col] + alt_specific_vars:
                    new_row[col] = row[col]
            long_data.append(new_row)

    return pd.DataFrame(long_data)
```

### Standard Errors

```python
# Robust
result_robust = mnl.fit(cov_type='HC0')

# Clustered
result_cluster = mnl.fit(cov_type='cluster',
                         cov_kwds={'groups': cluster_var})
```

## Causal Interpretation

### Treatment Effects with Multinomial Outcomes

For binary treatment $D$:
$$ATE_j = E[P(Y=j|D=1, X)] - E[P(Y=j|D=0, X)]$$

```python
def treatment_effect_multinomial(result, X, treatment_col):
    """
    Compute treatment effects on each alternative probability.
    """
    X1 = X.copy()
    X0 = X.copy()
    X1[:, treatment_col] = 1
    X0[:, treatment_col] = 0

    p1 = result.predict(X1)
    p0 = result.predict(X0)

    # ATE for each alternative
    ate = np.mean(p1 - p0, axis=0)

    return ate
```

### Challenges

1. **Multiple outcomes**: Treatment can increase probability of one alternative while decreasing others
2. **Substitution patterns**: MNL imposes proportional substitution (IIA)
3. **Heterogeneity**: Average effects may mask important heterogeneity

## Best Practices

1. **Always test IIA** before trusting MNL results
2. **Report marginal effects**, not just coefficients
3. **Consider nested or mixed logit** if IIA fails
4. **Use alternative-specific variables** when available
5. **Check robustness** to reference category choice
6. **Examine predicted market shares** for reasonableness
