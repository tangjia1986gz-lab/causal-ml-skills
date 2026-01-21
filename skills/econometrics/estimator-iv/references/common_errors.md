# Common IV Errors and How to Avoid Them

## Error 1: Using Weak Instruments

**Problem:**
Using instruments with first-stage F < 10 leads to:
- Severe finite-sample bias toward OLS
- Invalid standard errors and confidence intervals
- Misleading hypothesis tests

**Wrong:**
```python
# Ignoring weak instruments
result = IV2SLS(...).fit()
print(result.summary)  # Reporting without checking F
```

**Correct:**
```python
# Always check first stage
result = IV2SLS(...).fit()

# Get first-stage F
f_stat = first_stage_analysis(...)['f_statistic']

if f_stat < 10:
    print("WARNING: Weak instrument detected")
    # Use LIML instead
    result_liml = IVLIML(...).fit()
    # Or use Anderson-Rubin CI
    ar_ci = anderson_rubin_ci(...)
```

## Error 2: Manual Two-Step OLS

**Problem:**
Computing X_hat manually and plugging into OLS gives WRONG standard errors.

**Wrong:**
```python
# NEVER DO THIS
from sklearn.linear_model import LinearRegression

# First stage
reg1 = LinearRegression().fit(Z, X)
X_hat = reg1.predict(Z)

# Second stage - WRONG SEs!
reg2 = LinearRegression().fit(X_hat, Y)
print(reg2.coef_)  # SE is wrong!
```

**Correct:**
```python
# Use proper IV package
from linearmodels.iv import IV2SLS

result = IV2SLS(
    dependent=Y,
    exog=controls,
    endog=X,
    instruments=Z
).fit(cov_type='robust')
# SEs are correct!
```

## Error 3: Forbidden Regression

**Problem:**
Regressing Y on Z to "test" the exclusion restriction.

**Wrong:**
```python
# FORBIDDEN - This proves nothing
import statsmodels.api as sm
result = sm.OLS(Y, sm.add_constant(Z)).fit()
if result.pvalues['Z'] < 0.05:
    print("Instrument affects Y - bad?")  # WRONG interpretation
```

**Why it's wrong:**
- Z SHOULD affect Y through X (that's the reduced form)
- Finding Z affects Y is expected and desired
- Exclusion restriction is about the DIRECT effect, which is untestable

**Correct:**
- Make theoretical arguments for exclusion
- Report reduced form (Z → Y) as supporting evidence
- Don't claim you've "tested" exclusion

## Error 4: Ignoring LATE Interpretation

**Problem:**
Interpreting IV as Average Treatment Effect (ATE) when it's actually LATE.

**Wrong:**
```python
# Claiming universal effect
print(f"Education increases wages by {iv_coef:.2f} for everyone")
```

**Correct:**
```python
# Acknowledge LATE
print(f"Education increases wages by {iv_coef:.2f} for compliers")
print("Compliers: Those whose education is affected by the instrument")
print("This may differ from the effect for always-takers or never-takers")
```

## Error 5: Too Many Instruments

**Problem:**
Using many weak instruments causes severe bias, even if combined they pass F-test.

**Rule of Thumb:**
Number of instruments < sqrt(n)

**Wrong:**
```python
# 50 instruments with n=100
instruments = [f'z_{i}' for i in range(50)]
result = IV2SLS(..., instruments=df[instruments]).fit()
# Heavily biased even if first-stage F > 10
```

**Correct:**
```python
# Use fewer, stronger instruments
# Or use LIML/JIVE with many instruments
result = IVLIML(...).fit()  # More robust to many instruments
```

## Error 6: Ignoring Sargan Test Rejection

**Problem:**
Proceeding with IV despite Sargan test rejecting validity.

**Wrong:**
```python
result = IV2SLS(...).fit()
sargan = result.sargan()
if sargan.pval < 0.05:
    print("Sargan rejects, but we'll proceed anyway")  # BAD
```

**Correct:**
```python
if sargan.pval < 0.05:
    print("WARNING: Sargan test suggests at least one instrument invalid")
    # Actions:
    # 1. Re-examine instrument validity theoretically
    # 2. Try dropping instruments one at a time
    # 3. Check for heterogeneous effects
    # 4. Consider alternative identification
```

## Error 7: Not Clustering Standard Errors

**Problem:**
Not accounting for clustering in the data.

**Wrong:**
```python
# Data is clustered by state, but using default SEs
result = IV2SLS(...).fit()  # Default SEs
```

**Correct:**
```python
# Cluster at appropriate level
result = IV2SLS(...).fit(cov_type='clustered', clusters=df['state'])
```

## Error 8: Misusing Generated Instruments

**Problem:**
Using predicted values from another regression as instruments without adjusting SEs.

**Wrong:**
```python
# Generate instrument from another model
predicted_z = some_model.predict(X_other)
df['z_hat'] = predicted_z
result = IV2SLS(..., instruments=df[['z_hat']]).fit()  # SEs wrong!
```

**Correct:**
- Use Murphy-Topel standard error correction
- Or bootstrap the entire procedure
- Better: use original variables if possible

## Error 9: Not Reporting OLS Comparison

**Problem:**
Failing to show how IV differs from OLS (and why).

**Correct Reporting:**
```python
print("Estimation Results Comparison:")
print(f"  OLS coefficient: {ols_coef:.4f}")
print(f"  IV coefficient:  {iv_coef:.4f}")
print(f"  Difference:      {iv_coef - ols_coef:.4f}")
print(f"  Hausman p-value: {hausman_pval:.4f}")

# Interpretation
if iv_coef > ols_coef:
    print("IV > OLS suggests downward bias in OLS (attenuation?)")
else:
    print("IV < OLS suggests upward bias in OLS (omitted variables?)")
```

## Error 10: Wrong Degrees of Freedom in Tests

**Problem:**
Using wrong distribution for test statistics with small samples.

**Wrong:**
```python
# Using chi-squared when F is appropriate
from scipy.stats import chi2
p_value = 1 - chi2.cdf(test_stat, df)  # Wrong for small n
```

**Correct:**
```python
# Use F distribution for finite samples
from scipy.stats import f
p_value = 1 - f.cdf(test_stat, df1, df2)
```

## Quick Reference Checklist

Before submitting IV results, verify:

- [ ] First-stage F > 10 (or use LIML/AR)
- [ ] Used proper IV package (not manual two-step)
- [ ] Exclusion restriction argued theoretically (not "tested")
- [ ] LATE interpretation acknowledged
- [ ] Number of instruments reasonable
- [ ] Sargan test addressed (if overidentified)
- [ ] Appropriate standard errors (clustered if needed)
- [ ] OLS comparison reported
- [ ] All diagnostics documented
