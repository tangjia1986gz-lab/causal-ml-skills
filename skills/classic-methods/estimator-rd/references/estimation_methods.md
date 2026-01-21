# RD Estimation Methods

> **Document Type**: Technical Reference
> **Last Updated**: 2024-01
> **Key References**: Calonico, Cattaneo & Titiunik (2014), Imbens & Kalyanaraman (2012)

## Overview

This document covers the main estimation approaches for Regression Discontinuity designs, with emphasis on the local polynomial methods that represent current best practice.

---

## 1. Local Polynomial Regression

### Fundamental Idea

Instead of fitting a global model, fit separate polynomial regressions on each side of the cutoff using only observations within a bandwidth $h$ of the cutoff.

### Mathematical Formulation

**Below the cutoff** ($X_i < c$):
$$
\min_{\alpha_-, \beta_-, ...} \sum_{i: X_i < c} K\left(\frac{X_i - c}{h}\right) \cdot \left(Y_i - \alpha_- - \beta_-(X_i - c) - \frac{\gamma_-}{2}(X_i - c)^2 - ...\right)^2
$$

**Above the cutoff** ($X_i \geq c$):
$$
\min_{\alpha_+, \beta_+, ...} \sum_{i: X_i \geq c} K\left(\frac{X_i - c}{h}\right) \cdot \left(Y_i - \alpha_+ - \beta_+(X_i - c) - \frac{\gamma_+}{2}(X_i - c)^2 - ...\right)^2
$$

**RD Estimate**:
$$
\hat{\tau}_{RD} = \hat{\alpha}_+ - \hat{\alpha}_-
$$

### Key Components

| Component | Description | Typical Choices |
|-----------|-------------|-----------------|
| Polynomial Order $p$ | Flexibility of local fit | 1 (linear), 2 (quadratic) |
| Bandwidth $h$ | Width of estimation window | Data-driven optimal |
| Kernel $K(\cdot)$ | Weighting function | Triangular (default) |

---

## 2. Polynomial Order Selection

### Linear (p = 1) - Recommended Default

**Advantages**:
- Most robust to boundary bias
- Lower variance than higher orders
- Recommended by methodological literature (Gelman & Imbens, 2019)

**When to Use**:
- Default for all RD applications
- When relationship is approximately linear near cutoff
- When sample size is limited

**Implementation**:
```python
result = estimate_sharp_rd(
    data=df,
    running="score",
    outcome="y",
    cutoff=0.0,
    order=1  # Linear (default)
)
```

### Quadratic (p = 2)

**Advantages**:
- More flexible, can capture curvature
- Useful as robustness check

**When to Use**:
- When visual inspection suggests curvature
- As sensitivity analysis
- When sample is very large

**Caution**:
- Higher variance than linear
- More sensitive to bandwidth choice
- Can be erratic at boundaries

### Higher Orders (p > 2)

**Generally Discouraged** (Gelman & Imbens, 2019):
- High sensitivity to influential observations
- Erratic behavior near boundaries
- Global polynomials are especially problematic

**Quote from Gelman & Imbens (2019)**:
> "We recommend against the use of high-order global polynomial approximations... these estimators have poor properties in regression discontinuity settings."

---

## 3. Bandwidth Selection

### The Bandwidth Trade-off

| Small Bandwidth | Large Bandwidth |
|-----------------|-----------------|
| Lower bias | Higher bias |
| Higher variance | Lower variance |
| Fewer observations | More observations |
| More local | More global |

### MSE-Optimal Bandwidth (CCT)

**Objective**: Minimize Mean Squared Error of the RD estimator.

**Formula** (simplified):
$$
h_{MSE} \propto \left(\frac{\sigma^2(c)}{[m''(c^+) - m''(c^-)]^2 \cdot n}\right)^{1/5}
$$

Where:
- $\sigma^2(c)$: Variance of outcome near cutoff
- $m''(c)$: Second derivative of conditional mean
- $n$: Sample size

**Implementation**:
```python
from rd_estimator import select_bandwidth

h_mse = select_bandwidth(
    running=df["score"],
    outcome=df["y"],
    cutoff=0.0,
    method="mserd"  # MSE-optimal for RD
)
print(f"MSE-optimal bandwidth: {h_mse:.4f}")
```

### CER-Optimal Bandwidth

**Objective**: Minimize Coverage Error Rate of confidence intervals.

**Relationship to MSE**:
$$
h_{CER} = h_{MSE} \cdot n^{-1/20}
$$

CER-optimal bandwidth is smaller than MSE-optimal, leading to:
- More conservative inference
- Better coverage properties
- Wider confidence intervals

**Implementation**:
```python
h_cer = select_bandwidth(
    running=df["score"],
    outcome=df["y"],
    cutoff=0.0,
    method="cerrd"  # CER-optimal
)
print(f"CER-optimal bandwidth: {h_cer:.4f}")
```

### Imbens-Kalyanaraman (IK) Bandwidth

**Earlier Method** (Imbens & Kalyanaraman, 2012):
- Cross-validation based
- Widely used in applied work
- Similar to CCT MSE-optimal

**When to Use**:
- For comparison with older studies
- When CCT methods are unavailable

---

## 4. Kernel Functions

### Triangular Kernel (Recommended)

$$
K(u) = (1 - |u|) \cdot \mathbf{1}(|u| \leq 1)
$$

**Properties**:
- Gives more weight to observations close to cutoff
- Boundary optimal (Fan & Gijbels, 1996)
- Default in rdrobust

### Epanechnikov Kernel

$$
K(u) = \frac{3}{4}(1 - u^2) \cdot \mathbf{1}(|u| \leq 1)
$$

**Properties**:
- Optimal for interior point estimation
- Slightly less weight to boundary
- Commonly used in kernel density estimation

### Uniform Kernel

$$
K(u) = \frac{1}{2} \cdot \mathbf{1}(|u| \leq 1)
$$

**Properties**:
- Equal weight to all observations within bandwidth
- Equivalent to simple OLS within bandwidth
- Useful for simple comparisons

### Comparison

```python
# Compare kernels
for kernel in ["triangular", "epanechnikov", "uniform"]:
    result = estimate_sharp_rd(
        data=df,
        running="score",
        outcome="y",
        cutoff=0.0,
        bandwidth=h_opt,
        kernel=kernel
    )
    print(f"{kernel}: effect = {result.effect:.4f}, se = {result.se:.4f}")
```

---

## 5. Robust Bias-Corrected Inference

### The Problem with Conventional Inference

Conventional confidence intervals:
$$
CI = \hat{\tau} \pm z_{1-\alpha/2} \cdot \hat{SE}
$$

**Issue**: The point estimator $\hat{\tau}$ has bias of order $h^2$. For optimal bandwidth, this bias is first-order.

### CCT Robust Inference

**Solution** (Calonico, Cattaneo & Titiunik, 2014):
1. Estimate bias explicitly
2. Construct bias-corrected estimator
3. Use variance that accounts for bias estimation

**Implementation**:
```python
# rdrobust-style robust inference
result = estimate_sharp_rd(
    data=df,
    running="score",
    outcome="y",
    cutoff=0.0,
    bandwidth=h_mse  # MSE-optimal
)

# result.ci_lower and result.ci_upper are robust bias-corrected
print(f"Robust 95% CI: [{result.ci_lower:.4f}, {result.ci_upper:.4f}]")
```

### Practical Implications

| Inference Type | Bandwidth | Coverage | Width |
|----------------|-----------|----------|-------|
| Conventional | MSE-optimal | Under-covers | Narrower |
| Robust BC | MSE-optimal | Correct | Wider |
| Conventional | CER-optimal | Approximately correct | Wider |

**Recommendation**: Use robust bias-corrected inference with MSE-optimal bandwidth.

---

## 6. The rdrobust Package

### Overview

`rdrobust` (R/Stata/Python) is the gold standard implementation by Cattaneo, Calonico, and Titiunik.

### Key Features

1. **Bandwidth Selection**: MSE and CER optimal
2. **Robust Inference**: Bias-corrected confidence intervals
3. **Flexible Specification**: Kernels, polynomial orders, covariates
4. **Companion Packages**: rddensity, rdlocrand, rdmulti

### Python Interface

```python
# If rdrobust-python is available
try:
    from rdrobust import rdrobust

    result = rdrobust(
        Y=df["y"],
        X=df["score"],
        c=0.0,          # Cutoff
        p=1,            # Polynomial order
        kernel="tri",   # Triangular kernel
        bwselect="mserd"  # MSE-optimal bandwidth
    )

    print(result.summary())

except ImportError:
    # Use local implementation
    from rd_estimator import estimate_sharp_rd
    result = estimate_sharp_rd(df, "score", "y", 0.0)
```

### Interpretation of rdrobust Output

```
=======================================================================
              RD Estimates
=======================================================================
     Cutoff c = 0          | Left of c | Right of c | Number of obs = 1000
-----------------------------------------------------------------------
    Number of obs          |    487    |    513     | BW type = mserd
    Eff. number of obs     |    187    |    201     | Kernel = Triangular
    Order est. (p)         |      1    |      1     | BW est. (h) = 0.52
    Order bias (q)         |      2    |      2     | BW bias (b) = 0.89
-----------------------------------------------------------------------
              Coef.   Std. Err.   z    P>|z|  [95% Conf. Interval]
-----------------------------------------------------------------------
Conventional   0.523    0.142    3.68  0.000   [0.245 , 0.801]
Robust           -        -      3.21  0.001   [0.198 , 0.848]
=======================================================================
```

**Key Elements**:
- **Conventional**: Standard point estimate and CI
- **Robust**: Bias-corrected CI (use this for inference)
- **BW est (h)**: Bandwidth for point estimation
- **BW bias (b)**: Bandwidth for bias estimation (typically larger)

---

## 7. Fuzzy RD Estimation

### Setup

In fuzzy RD, crossing the cutoff affects treatment probability but not deterministically:
$$
\lim_{x \downarrow c} P(D=1|X=x) \neq \lim_{x \uparrow c} P(D=1|X=x)
$$

### Local Wald Estimator

$$
\hat{\tau}_{FRD} = \frac{\text{Reduced Form}}{\text{First Stage}} = \frac{\hat{\tau}_{Y,sharp}}{\hat{\tau}_{D,sharp}}
$$

### Implementation

```python
from rd_estimator import estimate_fuzzy_rd

result = estimate_fuzzy_rd(
    data=df,
    running="score",
    outcome="y",
    treatment="treated",  # Actual treatment received
    cutoff=0.0,
    bandwidth=h_opt
)

print(f"Fuzzy RD (LATE): {result.effect:.4f}")
print(f"First stage: {result.diagnostics['first_stage']:.4f}")
print(f"Reduced form: {result.diagnostics['reduced_form']:.4f}")
```

### First Stage Strength

**Weak First Stage Warning**:
- If first stage < 0.05, fuzzy RD is unreliable
- Large standard errors due to near-zero denominator
- Consider reporting reduced form (intent-to-treat) instead

```python
if abs(result.diagnostics['first_stage']) < 0.05:
    print("WARNING: Weak first stage. Consider:")
    print("1. Report reduced form (ITT) effect")
    print("2. Use bias-robust inference")
    print("3. Reconsider the fuzzy RD approach")
```

---

## 8. Covariate Adjustment

### When to Include Covariates

**Potential Benefits**:
- Reduce residual variance (precision gain)
- Control for local imbalances

**Potential Risks**:
- Overfitting in small samples
- Not necessary for identification (unlike matching)

### Implementation

```python
# With covariates (if supported)
result = estimate_sharp_rd(
    data=df,
    running="score",
    outcome="y",
    cutoff=0.0,
    covariates=["age", "gender"]  # Pre-treatment covariates only
)
```

### Best Practices

1. **Only pre-treatment covariates**: Never include post-treatment variables
2. **Balance first**: Check covariate balance before adjusting
3. **Report both**: With and without covariate adjustment
4. **Modest gains**: Precision improvement typically small in well-designed RD

---

## 9. Estimation with Discrete Running Variables

### Challenge

When the running variable takes few discrete values, standard local polynomial methods may not work well.

### Solutions

**Option 1: Local Randomization Approach**
- Treat units very close to cutoff as randomly assigned
- Use permutation inference
- Requires stronger local randomization assumption

**Option 2: Mass Points Correction**
- Adjust for discrete nature of running variable
- Use clustering at running variable values

**Implementation**:
```python
# Check for discrete running variable
n_unique = df["score"].nunique()
n_obs = len(df)

if n_unique < 50:
    print(f"WARNING: Running variable has only {n_unique} unique values")
    print("Consider local randomization approach or parametric methods")
```

---

## 10. Estimation Checklist

### Before Estimation

1. [ ] Validate data structure (running variable, cutoff, outcome)
2. [ ] Check for sufficient observations near cutoff
3. [ ] Run McCrary density test
4. [ ] Check covariate balance

### During Estimation

1. [ ] Select bandwidth (MSE or CER optimal)
2. [ ] Choose polynomial order (default: linear)
3. [ ] Specify kernel (default: triangular)
4. [ ] For fuzzy RD: verify first stage strength

### After Estimation

1. [ ] Report robust bias-corrected CI
2. [ ] Conduct bandwidth sensitivity
3. [ ] Test alternative polynomial orders
4. [ ] Run placebo cutoff tests
5. [ ] Create RD plot

---

## Software Comparison

| Feature | rdrobust (R/Stata) | rd_estimator (Python) | causalml |
|---------|-------------------|----------------------|----------|
| Sharp RD | Yes | Yes | Limited |
| Fuzzy RD | Yes | Yes | Limited |
| Robust BC inference | Yes | Yes | No |
| Optimal bandwidth | Yes | Yes | No |
| McCrary test | rddensity | Yes | No |
| RD plots | rdplot | Yes | No |
| Multiple cutoffs | rdmulti | No | No |

---

## References

- Calonico, S., Cattaneo, M. D., & Titiunik, R. (2014). Robust nonparametric confidence intervals for regression-discontinuity designs. *Econometrica*, 82(6), 2295-2326.
- Imbens, G., & Kalyanaraman, K. (2012). Optimal bandwidth choice for the regression discontinuity estimator. *Review of Economic Studies*, 79(3), 933-959.
- Fan, J., & Gijbels, I. (1996). *Local Polynomial Modelling and Its Applications*. Chapman & Hall.
- Gelman, A., & Imbens, G. (2019). Why high-order polynomials should not be used in regression discontinuity designs. *Journal of Business & Economic Statistics*, 37(3), 447-456.
- Cattaneo, M. D., Idrobo, N., & Titiunik, R. (2020). *A Practical Introduction to Regression Discontinuity Designs: Foundations*. Cambridge University Press.
