# Common Errors in IV Estimation

> **Reference Document** | IV Estimator Skill
> **Last Updated**: 2024

## Overview

This document catalogs common errors in Instrumental Variables analysis, explains why they are problematic, and provides guidance on avoiding them. These errors can lead to biased estimates, incorrect inference, or misleading conclusions.

---

## 1. Weak Instruments

### 1.1 The Problem

**Error**: Proceeding with 2SLS when first-stage F-statistic is below 10.

**Why It's Wrong**:
- 2SLS is biased toward OLS in finite samples
- Bias increases as instruments weaken
- Standard errors understate uncertainty
- Confidence intervals have wrong coverage

**Mathematical Intuition**:
$$
\text{Bias}_{2SLS} \approx \frac{K}{F + 1} \times \text{Bias}_{OLS}
$$

With K instruments and first-stage F near K, bias approaches OLS bias.

### 1.2 Symptoms

- First-stage F < 10 (or below Stock-Yogo critical values)
- Large difference between 2SLS and LIML estimates
- Confidence intervals shift dramatically with small specification changes
- Implausibly large estimates

### 1.3 Solution

```python
from iv_estimator import first_stage_test, estimate_liml

# Check first-stage strength
first_stage = first_stage_test(data, treatment, instruments, controls)

if first_stage['f_statistic'] < 10:
    print("WARNING: Weak instruments detected")

    # Use LIML instead of 2SLS
    result = estimate_liml(data, outcome, treatment, instruments, controls)

    # Compute Anderson-Rubin CI
    ar_ci = anderson_rubin_ci(data, outcome, treatment, instruments, controls)
    print(f"AR 95% CI: [{ar_ci['ci_lower']:.3f}, {ar_ci['ci_upper']:.3f}]")

    # Consider finding better instruments
    print("Consider strengthening identification strategy")
```

### 1.4 Best Practices

1. **Always report first-stage F**
2. **Compare 2SLS and LIML** - large differences indicate problems
3. **Use Anderson-Rubin CI** when F < 10
4. **Find stronger instruments** if possible
5. **Consider bounds analysis** if instruments cannot be strengthened

---

## 2. Forbidden Regressions

### 2.1 The Problem

**Error**: Including predicted treatment $\hat{D}$ directly in regression without proper 2SLS.

**Examples of Forbidden Regressions**:

```python
# WRONG: "Control function" done incorrectly
from sklearn.linear_model import LinearRegression

# First stage
first_stage = LinearRegression().fit(Z, D)
D_hat = first_stage.predict(Z)

# Forbidden regression #1: Using D_hat directly
model = LinearRegression().fit(np.column_stack([D_hat, X]), Y)  # WRONG!

# Forbidden regression #2: Controlling for D_hat and D
model = LinearRegression().fit(np.column_stack([D, D_hat, X]), Y)  # WRONG!
```

**Why It's Wrong**:
- Does not produce consistent estimates
- Standard errors are incorrect
- Mixing different estimands

### 2.2 The Correct Approaches

**Option 1: Proper 2SLS**
```python
from iv_estimator import estimate_2sls

# This handles both stages correctly
result = estimate_2sls(data, outcome, treatment, instruments, controls)
```

**Option 2: Control Function (done correctly)**
```python
# First stage
first_stage = first_stage_test(data, treatment, instruments, controls)
v_hat = first_stage['residuals']  # Use RESIDUALS, not fitted values

# Second stage: Include original D and first-stage RESIDUALS
data['v_hat'] = v_hat
import statsmodels.api as sm

X_second = sm.add_constant(data[[treatment] + controls + ['v_hat']])
model = sm.OLS(data[outcome], X_second).fit()

# Coefficient on treatment is IV estimate
# Coefficient on v_hat tests endogeneity (Wu-Hausman)
```

### 2.3 Key Distinction

| Approach | What to Include | Purpose |
|----------|-----------------|---------|
| 2SLS | $\hat{D}$ (fitted values) | Estimate causal effect |
| Control Function | D + $\hat{v}$ (residuals) | Estimate + endogeneity test |
| Forbidden | $\hat{D}$ in OLS | **Invalid** |

---

## 3. Over-Identification Errors

### 3.1 Using Too Many Instruments

**Error**: Using many instruments to "increase power" or "improve efficiency."

**Why It's Wrong**:
- Many weak instruments bias 2SLS toward OLS
- Finite-sample bias: $\approx K/n \times \text{OLS bias}$
- Overidentification tests lose power with many instruments

**Example**:
```python
# WRONG: Using all available correlates as instruments
instruments = ["z1", "z2", "z3", "z4", "z5", "z6", "z7", "z8",
               "z9", "z10", "z11", "z12"]  # Too many!
result = estimate_2sls(data, outcome, treatment, instruments, controls)
```

### 3.2 Solution

```python
# CORRECT: Use few, strong instruments based on theory
# Select instruments based on institutional knowledge, not data mining
instruments = ["z1", "z2"]  # Two conceptually justified instruments

# If you have many potential instruments:
# 1. Use LASSO for selection (with caution)
# 2. Use JIVE estimator
# 3. Compare estimates from instrument subsets

# Check for many-instruments bias
n = len(data)
k = len(instruments)
if k / n > 0.05:
    print(f"Warning: K/n = {k/n:.3f} > 0.05")
    print("Many-instruments bias may be a concern")
```

### 3.3 Misinterpreting Overidentification Test

**Error**: Claiming "instruments are valid" because Sargan test doesn't reject.

**Why It's Wrong**:
- Sargan/Hansen test has low power
- Cannot detect if ALL instruments are invalid in the same way
- Does not validate the exclusion restriction

**Correct Interpretation**:
```python
j_test = overidentification_test(result, data, ...)

if j_test.passed:
    print("Cannot reject that instruments are valid.")
    print("NOTE: This does not prove validity!")
    print("- Test has low power")
    print("- All instruments could be invalid similarly")
    print("- Exclusion restriction still requires theoretical argument")
else:
    print("Evidence that at least one instrument may be invalid.")
    print("- Check each instrument separately")
    print("- Consider difference-in-Sargan tests")
```

---

## 4. Standard Error Mistakes

### 4.1 Using Wrong Residuals

**Error**: Computing standard errors using residuals from predicted treatment $\hat{D}$.

**Why It's Wrong**:
The variance formula uses actual residuals $u = Y - D\hat{\gamma}$, not $\tilde{u} = Y - \hat{D}\hat{\gamma}$.

**Correct**:
```python
# The IV packages handle this automatically
# If computing manually:

# Get IV estimate
gamma_iv = ...  # IV coefficient

# Compute residuals using ACTUAL D (not D_hat)
u = y - X @ beta - D * gamma_iv  # Correct!
# NOT: u = y - X @ beta - D_hat * gamma_iv  # Wrong!

# Standard error computation
se = np.sqrt(sigma2 * np.linalg.inv(D.T @ P_Z @ D))
```

### 4.2 Not Using Robust Standard Errors

**Error**: Using homoskedastic standard errors when heteroskedasticity is present.

**Solution**:
```python
from iv_estimator import estimate_2sls

# Always use robust standard errors by default
result = estimate_2sls(data, outcome, treatment, instruments, controls)
# The implementation uses cov_type='robust' by default
```

### 4.3 Ignoring Clustering

**Error**: Not clustering standard errors when data has group structure.

**When to Cluster**:
- Panel data with repeated observations
- Data sampled by cluster (schools, firms, regions)
- Instrument varies at cluster level

```python
from linearmodels.iv import IV2SLS

# With clustering
model = IV2SLS(y, exog, endog, instruments)
result = model.fit(cov_type='clustered', clusters=data['cluster_id'])
```

---

## 5. LATE Misinterpretation

### 5.1 Generalizing to ATE

**Error**: Interpreting the IV estimate as the Average Treatment Effect (ATE).

**Why It's Wrong**:
- IV identifies LATE (Local Average Treatment Effect) for compliers
- Compliers may be systematically different from population
- Treatment effects may be heterogeneous

**Example of Wrong Interpretation**:
> "Our IV estimate shows that the policy increased earnings by $5,000 for everyone."

**Correct Interpretation**:
> "Our IV estimate of $5,000 represents the effect for compliers - those whose treatment status was affected by the instrument. This may differ from the effect for always-takers or never-takers if treatment effects are heterogeneous."

### 5.2 Different Instruments, Different LATEs

**Error**: Assuming different instruments identify the same effect.

**Reality**:
- Different instruments identify effects for different complier populations
- Quarter-of-birth compliers ≠ College-proximity compliers
- Both are valid LATEs, but for different groups

**Best Practice**:
```python
# Compare estimates from different instruments
for z in instruments:
    result_z = estimate_2sls(data, outcome, treatment, [z], controls)
    print(f"Instrument {z}: {result_z.effect:.4f} (SE: {result_z.se:.4f})")

# Discuss: Are these estimates similar?
# If different, what does this imply about treatment effect heterogeneity?
```

---

## 6. Exclusion Restriction Violations

### 6.1 Ignoring Direct Effects

**Error**: Not considering pathways from instrument to outcome that bypass treatment.

**Examples**:

| Instrument | Treatment | Potential Violation |
|------------|-----------|---------------------|
| College proximity | Education | Local labor market effects |
| Vietnam draft lottery | Military service | Psychological effects of lottery |
| Rainfall | Agricultural output | Direct effects on non-farm activities |
| Distance to provider | Healthcare use | Transportation cost effects |

### 6.2 Assessment Framework

```python
def assess_exclusion_restriction(instrument, treatment, outcome,
                                 potential_mechanisms):
    """
    Framework for assessing exclusion restriction.
    """
    print(f"Instrument: {instrument}")
    print(f"Treatment: {treatment}")
    print(f"Outcome: {outcome}")
    print()
    print("Pathways analysis:")
    print("-" * 50)

    for mechanism in potential_mechanisms:
        print(f"  {instrument} -> {mechanism['intermediate']} -> {outcome}")
        print(f"    Threat level: {mechanism['threat_level']}")
        print(f"    Mitigation: {mechanism['mitigation']}")
        print()

    print("Recommended checks:")
    print("1. Balance tests on covariates")
    print("2. Placebo tests (pre-treatment outcomes)")
    print("3. Subgroup analysis")
    print("4. Sensitivity analysis (Conley et al. 2012)")
```

### 6.3 Partial Violation Sensitivity

When exclusion restriction may be partially violated:

```python
def sensitivity_to_exclusion_violation(result, direct_effect_range):
    """
    Assess sensitivity to exclusion restriction violations.

    If Z has direct effect on Y of size delta:
    True effect = IV estimate - delta / first_stage
    """
    iv_estimate = result.effect
    first_stage = result.diagnostics['first_stage']['coefficients']
    pi = list(first_stage.values())[0]  # First-stage coefficient

    print("Sensitivity Analysis: Exclusion Restriction")
    print("-" * 50)
    print(f"IV Estimate: {iv_estimate:.4f}")
    print(f"First-stage coefficient: {pi:.4f}")
    print()
    print("If direct effect (Z -> Y) is:")

    for delta in direct_effect_range:
        adjusted = iv_estimate - delta / pi
        print(f"  delta = {delta:.3f}: True effect = {adjusted:.4f}")
```

---

## 7. Data and Specification Issues

### 7.1 Sample Selection on Endogenous Variable

**Error**: Restricting sample based on treatment or outcome.

**Example**:
```python
# WRONG: Restricting to treated observations
df_treated = df[df['treatment'] == 1]
result = estimate_2sls(df_treated, outcome, treatment, instruments)  # Invalid!

# WRONG: Restricting based on outcome
df_positive = df[df['outcome'] > 0]
result = estimate_2sls(df_positive, ...)  # Invalid!
```

**Why It's Wrong**:
- Introduces selection bias
- Violates random sampling assumption
- Changes the complier population

### 7.2 Including Bad Controls

**Error**: Controlling for variables affected by treatment (post-treatment variables).

```python
# WRONG: Including post-treatment variable
controls_bad = ["x1", "x2", "employment_status"]  # If treatment affects employment
result = estimate_2sls(data, "wages", "education",
                       instruments, controls_bad)  # Biased!

# CORRECT: Only control for pre-treatment variables
controls_good = ["x1", "x2", "parental_education", "birth_year"]
result = estimate_2sls(data, "wages", "education",
                       instruments, controls_good)
```

### 7.3 Multicollinearity Between Instruments

**Error**: Using highly correlated instruments.

**Problem**:
- First-stage may appear strong but instruments don't add information
- Overidentification test may be meaningless
- Efficiency gains are illusory

```python
# Check instrument correlation
import seaborn as sns
import matplotlib.pyplot as plt

corr_matrix = data[instruments].corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title('Instrument Correlation Matrix')

# If correlations > 0.8, consider:
# 1. Using only one instrument
# 2. Creating orthogonal combinations
# 3. Using factor analysis
```

---

## 8. Error Checklist

Before finalizing IV analysis, verify:

### Data
- [ ] No sample selection on treatment or outcome
- [ ] No post-treatment controls included
- [ ] Instruments have sufficient variation
- [ ] Missing data handled appropriately

### First Stage
- [ ] F-statistic > 10 (or Stock-Yogo threshold)
- [ ] Instruments have expected signs
- [ ] No multicollinearity among instruments

### Estimation
- [ ] Used proper 2SLS (not forbidden regression)
- [ ] Standard errors computed correctly (actual residuals)
- [ ] Robust/clustered SEs if appropriate

### Diagnostics
- [ ] Compared 2SLS and LIML (similar estimates)
- [ ] Reported overidentification test (if applicable)
- [ ] Reported endogeneity test
- [ ] Considered AR CI if weak instruments

### Interpretation
- [ ] Discussed LATE vs ATE
- [ ] Identified complier population
- [ ] Defended exclusion restriction
- [ ] Noted limitations and caveats

---

## References

- Bound, J., Jaeger, D. A., & Baker, R. M. (1995). Problems with Instrumental Variables Estimation When the Correlation Between the Instruments and the Endogenous Explanatory Variable is Weak. *JASA*, 90(430), 443-450.
- Angrist, J. D., & Pischke, J. S. (2009). *Mostly Harmless Econometrics*. Princeton University Press.
- Young, A. (2022). Consistency without Inference: Instrumental Variables in Practical Application. *European Economic Review*, 147, 104112.
- Andrews, I., Stock, J. H., & Sun, L. (2019). Weak Instruments in IV Regression: Theory and Practice. *Annual Review of Economics*, 11, 727-753.
