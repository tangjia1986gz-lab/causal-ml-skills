# Common Errors in Panel Data Analysis

## 1. Forgetting to Set Multi-Index

**Error:**
```python
from linearmodels.panel import PanelOLS

df = pd.read_csv('data.csv')
model = PanelOLS.from_formula('y ~ x', data=df)
# Error: data must have a MultiIndex
```

**Correct:**
```python
df = pd.read_csv('data.csv')
df = df.set_index(['entity_id', 'time_id'])  # Set multi-index first
model = PanelOLS.from_formula('y ~ x + EntityEffects', data=df)
```

## 2. Not Clustering Standard Errors

**Error:**
```python
result = model.fit()  # Default: homoskedastic SE
# SE severely underestimated due to serial correlation
```

**Correct:**
```python
result = model.fit(cov_type='clustered', cluster_entity=True)
# Accounts for within-entity correlation
```

**Impact:** SE can be 2-3x larger with proper clustering.

## 3. Including Time-Invariant Variables with FE

**Error:**
```python
# gender is time-invariant
model = PanelOLS.from_formula('y ~ x + gender + EntityEffects', data=df)
# Warning: Collinearity - gender absorbed by entity effects
```

**Solution options:**
1. Use Random Effects (can estimate time-invariant)
2. Use Mundlak approach (include entity means)
3. Use between-effects model
4. Accept that within-entity effects cannot be estimated

## 4. Ignoring Hausman Test Results

**Error:**
```python
# Hausman test p-value = 0.001
# Still uses RE because "more efficient"
result_re = RandomEffects.from_formula('y ~ x', data=df).fit()
```

**Correct:**
```python
# p = 0.001 strongly rejects RE consistency
# Must use FE despite efficiency loss
result_fe = PanelOLS.from_formula('y ~ x + EntityEffects', data=df).fit()
```

## 5. TWFE with Staggered Treatment

**Error:**
```python
# Staggered DID setting
model = PanelOLS.from_formula('y ~ treated + EntityEffects + TimeEffects', data=df)
# TWFE gives biased estimates with heterogeneous effects
```

**Correct:**
```python
# Use specialized estimators
from differences import ATTgt  # Callaway-Sant'Anna

# Or Sun-Abraham interaction-weighted estimator
# Or de Chaisemartin-D'Haultfoeuille
```

**Why:** TWFE uses already-treated as controls and can assign negative weights.

## 6. Lagged Dependent Variable with FE (Nickell Bias)

**Error:**
```python
# Short T (e.g., T=5)
df['y_lag'] = df.groupby(level=0)['y'].shift(1)
model = PanelOLS.from_formula('y ~ y_lag + x + EntityEffects', data=df)
# Nickell bias: coefficient on y_lag biased downward by O(1/T)
```

**Correct:**
```python
# Use GMM for short panels
# Arellano-Bond uses lagged levels as instruments for differences
# Or use bias-corrected LSDV if T > 20
```

**Bias magnitude:** ~-1/T (e.g., 20% bias for T=5)

## 7. Reporting Only Overall R-squared

**Error:**
```python
print(f"R-squared: {result.rsquared}")  # Overall R²
# Misleading for FE models
```

**Correct:**
```python
print(f"R² (within): {result.rsquared_within}")    # Variation explained within entities
print(f"R² (between): {result.rsquared_between}")  # Variation explained across entities
print(f"R² (overall): {result.rsquared_overall}")  # Total variation
```

**Why:** Within R² is most relevant for FE; it shows how much within-entity variation is explained.

## 8. Forgetting drop_absorbed with TWFE

**Error:**
```python
model = PanelOLS.from_formula('y ~ x + EntityEffects + TimeEffects', data=df)
result = model.fit()
# May fail due to perfect collinearity
```

**Correct:**
```python
model = PanelOLS.from_formula(
    'y ~ x + EntityEffects + TimeEffects',
    data=df,
    drop_absorbed=True  # Required for numerical stability
)
```

## 9. Using Robust SE Instead of Clustered

**Error:**
```python
result = model.fit(cov_type='robust')  # Only heteroskedasticity-robust
# Ignores serial correlation!
```

**Correct:**
```python
result = model.fit(cov_type='clustered', cluster_entity=True)
# Accounts for both heteroskedasticity AND serial correlation
```

**Typical result:** Clustered SE >> Robust SE for panel data

## 10. Unbalanced Panel Without Investigation

**Error:**
```python
# 30% of observations missing
# Just run the model without checking why
model.fit()
```

**Correct:**
```python
# Investigate missingness pattern
obs_per_entity = df.groupby(level=0).size()
print(obs_per_entity.describe())

# Check if attrition is random
# Consider Heckman selection model if not
# Document attrition in paper
```

## 11. Wrong Index Order

**Error:**
```python
df = df.set_index(['time_id', 'entity_id'])  # Time first, entity second
# linearmodels expects entity first!
```

**Correct:**
```python
df = df.set_index(['entity_id', 'time_id'])  # Entity first, time second
```

## 12. Comparing FE and RE Coefficients Without Hausman

**Error:**
```python
# FE coefficient: 0.5
# RE coefficient: 0.8
# "They're different, must be a problem"
```

**Correct:**
```python
# Run formal Hausman test
# Difference could be significant (use FE) or not (RE is fine)
# Without test, can't determine if difference is meaningful
```

## 13. Using First-Difference When Within is Better

**Error:**
```python
# Serially uncorrelated errors
df['dy'] = df.groupby(level=0)['y'].diff()
df['dx'] = df.groupby(level=0)['x'].diff()
model = OLS('dy ~ dx', data=df.dropna())
# Less efficient than within estimator
```

**When to use FD:**
- Errors follow random walk
- Very short T (2-3 periods)
- Building block for GMM

## Summary Checklist

Before running panel models:
- [ ] Multi-index set (entity, time)?
- [ ] Checked for balance?
- [ ] Time-invariant variables identified?
- [ ] Hausman test planned?

During estimation:
- [ ] Clustered SE by entity?
- [ ] drop_absorbed=True for TWFE?
- [ ] Staggered treatment? (use specialized estimator)
- [ ] Lagged Y? (check for Nickell bias)

Reporting:
- [ ] Within R² for FE?
- [ ] Multiple SE types?
- [ ] Hausman test result?
- [ ] N entities, T periods, balance?
