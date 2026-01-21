# Time Series Causality vs. Causal Inference

## The Fundamental Distinction

**Time Series "Causality"** (Granger, VAR, etc.):
- Tests for *predictive* relationships
- Based on temporal precedence
- "X helps forecast Y" ≠ "X causes Y"

**Causal Inference** (Rubin, Pearl frameworks):
- Tests for *interventional* relationships
- Based on potential outcomes
- "Changing X would change Y"

## Granger Causality: What It Really Tests

### Definition

X "Granger-causes" Y if:
```
E[Y_{t+h} | Y_t, Y_{t-1}, ..., X_t, X_{t-1}, ...] ≠ E[Y_{t+h} | Y_t, Y_{t-1}, ...]
```

In words: Past X improves prediction of Y beyond Y's own past.

### What This DOES Tell Us

1. **Predictive relationship exists**: X contains information about Y
2. **Temporal ordering**: X's past relates to Y's present
3. **Incremental forecasting power**: X adds value beyond Y alone

### What This DOES NOT Tell Us

1. **Causation**: No guarantee X causes Y
2. **Mechanism**: No insight into how X affects Y
3. **Intervention effect**: What happens if we change X
4. **Direction**: Maybe both caused by a third variable

## Why Granger Causality Fails as True Causality

### Example 1: The Rooster Problem

```
Observation: Rooster crowing Granger-causes sunrise
- Crowing time = 4:55 AM
- Sunrise = 5:00 AM
- Crowing predicts sunrise very well!

Reality: Both caused by Earth's rotation
- Silencing roosters won't stop sunrise
```

### Example 2: Confounding

```
True causal structure:
    Z (Confunder)
   / \
  v   v
  X   Y

What we see in time series:
- X Granger-causes Y
- Y Granger-causes X

Both are "caused" by Z, but we didn't observe Z
```

### Example 3: Reverse Causation

```
True: Y_t → X_{t+1}
(People anticipate Y and adjust X)

Granger test shows: X Granger-causes Y
(Because X_{t-1} was adjusted based on anticipated Y_t)
```

### Example 4: Common Trends

```
Stock price and ice cream sales both:
- Increase over time
- Are non-stationary

Spurious Granger causality detected!
```

## Proper Causal Inference with Time Series Data

### 1. Difference-in-Differences (DID)

When you have:
- Treatment and control groups
- Before and after periods
- Parallel trends assumption

```python
def did_estimator(data, outcome, treatment, post, entity, time):
    """
    Difference-in-Differences with time series data.

    Model: Y_it = α + β·Treat_i + γ·Post_t + δ·(Treat_i × Post_t) + ε_it

    δ is the causal effect (ATT)
    """
    import statsmodels.formula.api as smf

    # Create interaction
    data['treat_post'] = data[treatment] * data[post]

    # Estimate with entity and time fixed effects
    formula = f"{outcome} ~ {treatment} + {post} + treat_post + C({entity}) + C({time})"
    model = smf.ols(formula, data=data).fit(cov_type='cluster',
                                             cov_kwds={'groups': data[entity]})

    return {
        'att': model.params['treat_post'],
        'se': model.bse['treat_post'],
        'p_value': model.pvalues['treat_post'],
        'ci': model.conf_int().loc['treat_post'].tolist()
    }
```

### 2. Synthetic Control Method

When you have:
- Single treated unit
- Multiple control units
- Pre-treatment outcome data

```python
def synthetic_control(treated_series, control_matrix, pre_period, post_period):
    """
    Synthetic control estimator.

    Creates weighted average of controls to match treated pre-treatment.
    Effect = Treated - Synthetic in post-treatment.
    """
    from scipy.optimize import minimize
    import numpy as np

    # Pre-treatment fit
    Y_treated_pre = treated_series[pre_period]
    Y_controls_pre = control_matrix[pre_period]

    # Find weights that minimize pre-treatment distance
    def objective(w):
        synthetic = Y_controls_pre @ w
        return np.sum((Y_treated_pre - synthetic) ** 2)

    n_controls = control_matrix.shape[1]
    constraints = [
        {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},  # weights sum to 1
    ]
    bounds = [(0, 1)] * n_controls  # non-negative weights

    result = minimize(objective, x0=np.ones(n_controls)/n_controls,
                     constraints=constraints, bounds=bounds)

    weights = result.x

    # Compute synthetic control
    synthetic = control_matrix @ weights

    # Effect in post-period
    effect = treated_series[post_period] - synthetic[post_period]

    return {
        'weights': weights,
        'synthetic': synthetic,
        'effect': effect,
        'pre_treatment_fit': np.sqrt(result.fun / len(pre_period))
    }
```

### 3. Interrupted Time Series (ITS)

When you have:
- Single unit or aggregate data
- Clear intervention point
- Many pre/post observations

```python
def interrupted_time_series(data, outcome, time, intervention_time):
    """
    Interrupted time series analysis.

    Model: Y_t = β₀ + β₁·t + β₂·D_t + β₃·(t - T₀)·D_t + ε_t

    Where:
    - β₂ = immediate level change
    - β₃ = change in slope
    """
    import numpy as np
    import statsmodels.api as sm

    T0 = intervention_time

    # Create variables
    data['time'] = data[time]
    data['post'] = (data[time] >= T0).astype(int)
    data['time_since'] = np.maximum(0, data[time] - T0)

    # Estimate
    X = sm.add_constant(data[['time', 'post', 'time_since']])
    model = sm.OLS(data[outcome], X).fit(cov_type='HAC',
                                          cov_kwds={'maxlags': 5})

    return {
        'level_change': model.params['post'],
        'slope_change': model.params['time_since'],
        'level_change_se': model.bse['post'],
        'slope_change_se': model.bse['time_since'],
        'model': model
    }
```

### 4. Regression Discontinuity in Time

When you have:
- Sharp cutoff date
- Running variable is time
- Continuity of potential outcomes

```python
def regression_discontinuity_time(data, outcome, time, cutoff, bandwidth=None):
    """
    Regression discontinuity in time.

    Caution: Requires strong assumptions about no confounding at cutoff.
    """
    import numpy as np
    import statsmodels.api as sm

    # Normalize time around cutoff
    data['running'] = data[time] - cutoff
    data['treated'] = (data['running'] >= 0).astype(int)

    # Optimal bandwidth (simple rule)
    if bandwidth is None:
        bandwidth = 0.5 * data['running'].std()

    # Subset to bandwidth
    subset = data[np.abs(data['running']) <= bandwidth].copy()

    # Local linear regression
    subset['running_x_treated'] = subset['running'] * subset['treated']
    X = sm.add_constant(subset[['running', 'treated', 'running_x_treated']])

    model = sm.OLS(subset[outcome], X).fit()

    return {
        'effect': model.params['treated'],
        'se': model.bse['treated'],
        'bandwidth': bandwidth,
        'n_obs': len(subset)
    }
```

## When Can Time Series Methods Support Causal Claims?

### Conditions for Stronger Causal Interpretation

1. **Theory-driven**: Economic theory predicts causal direction
2. **Timing impossible to reverse**: X physically precedes Y
3. **Controlled for confounders**: All relevant variables included
4. **No anticipation**: Y doesn't respond in advance to expected X
5. **Structural breaks**: Exogenous policy changes as natural experiments

### Structural VAR for Causal Analysis

Requires identifying assumptions:

```python
def structural_var_for_causality(data, identification='cholesky', order=None):
    """
    Structural VAR with causal interpretation.

    Identification strategies:
    1. Cholesky (recursive): Assumes temporal ordering
    2. Sign restrictions: Based on economic theory
    3. External instruments: Uses exogenous variation

    WARNING: Results depend heavily on identification assumptions!
    """
    from statsmodels.tsa.api import VAR

    if identification == 'cholesky':
        if order is None:
            raise ValueError("Cholesky requires variable ordering")

        # Reorder data
        data = data[order]

        model = VAR(data)
        result = model.fit()

        # Cholesky decomposition gives orthogonalized IRFs
        irf = result.irf(periods=20)

        return {
            'irf': irf,
            'note': 'Causal interpretation depends on ordering being correct!',
            'ordering': order
        }

    raise NotImplementedError(f"Identification method {identification} not implemented")
```

## Best Practices

### DO

1. **Report Granger tests as "predictive"**, not "causal"
2. **Use proper causal methods** when causality is the goal
3. **Consider confounders** and common causes
4. **Test robustness** to different specifications
5. **Be explicit about assumptions**

### DON'T

1. **Claim causality** from Granger tests alone
2. **Ignore reverse causation** possibilities
3. **Over-interpret VAR** impulse responses
4. **Forget about omitted variables**
5. **Assume temporal precedence = causation**

## Checklist for Causal Claims with Time Series

Before making causal claims, verify:

```
[ ] Is there a credible source of exogenous variation?
[ ] Can reverse causality be ruled out?
[ ] Are all confounders observed and controlled?
[ ] Is anticipation unlikely?
[ ] Does economic theory support the direction?
[ ] Are results robust to specification changes?
[ ] Have you tried alternative causal methods?
[ ] Are the identifying assumptions plausible?
```

## Summary Table

| Method | Tests For | Causal Interpretation |
|--------|-----------|----------------------|
| Granger causality | Prediction | WEAK - only predictive |
| VAR impulse response | Dynamic correlations | WEAK - requires structural ID |
| Cointegration | Long-run equilibrium | NONE - statistical relationship |
| Difference-in-Differences | Treatment effect | STRONG - if parallel trends hold |
| Synthetic Control | Treatment effect | STRONG - if pre-fit good |
| Interrupted Time Series | Intervention effect | MODERATE - if no confounders at cutoff |
| RDD in Time | Discontinuity effect | MODERATE - if continuity holds |

## References

- Angrist, J.D. & Pischke, J.S. (2009). Mostly Harmless Econometrics. Princeton.
- Cunningham, S. (2021). Causal Inference: The Mixtape. Yale University Press.
- Stock, J.H. & Watson, M.W. (2018). Identification and Estimation of Dynamic Causal Effects in Macroeconomics. Economic Journal.
- Rambachan, A. & Roth, J. (2023). A More Credible Approach to Parallel Trends. Review of Economic Studies.
