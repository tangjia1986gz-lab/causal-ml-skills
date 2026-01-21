# RD Estimation Methods

## Local Polynomial Regression

### Core Idea

Estimate conditional expectations separately on each side of the cutoff using weighted regression with observations close to the cutoff receiving more weight.

### Estimator Formula

For polynomial order $p$ and kernel $K$:

$$\hat{\mu}_{+} = \hat{\beta}_0^{+} \text{ where } (\hat{\beta}_0^{+}, \hat{\beta}_1^{+}, ..., \hat{\beta}_p^{+}) = \arg\min \sum_{i: X_i \geq c} K\left(\frac{X_i - c}{h}\right) \left[Y_i - \sum_{j=0}^{p} \beta_j (X_i - c)^j\right]^2$$

$$\hat{\mu}_{-} = \hat{\beta}_0^{-} \text{ analogously for } X_i < c$$

**RD Estimate:**
$$\hat{\tau}_{RD} = \hat{\mu}_{+} - \hat{\mu}_{-}$$

### Polynomial Orders

| Order | Name | Bias | Variance | Recommendation |
|-------|------|------|----------|----------------|
| p = 0 | Local constant | High at boundary | Low | Rarely used |
| p = 1 | Local linear | Low (rate h²) | Medium | **Default choice** |
| p = 2 | Local quadratic | Lower | Higher | For smoother data |

**Why Local Linear is Preferred:**
- Corrects boundary bias automatically
- Good bias-variance tradeoff
- Gelman & Imbens (2018) recommend against global high-order polynomials

---

## Kernel Functions

Kernels determine how observations are weighted by distance from cutoff.

### Common Kernels

**Triangular (Recommended):**
$$K(u) = (1 - |u|) \cdot \mathbf{1}(|u| \leq 1)$$

**Epanechnikov:**
$$K(u) = \frac{3}{4}(1 - u^2) \cdot \mathbf{1}(|u| \leq 1)$$

**Uniform:**
$$K(u) = \frac{1}{2} \cdot \mathbf{1}(|u| \leq 1)$$

### Comparison

| Kernel | Efficiency | Boundary Behavior | Smoothness |
|--------|------------|-------------------|------------|
| Triangular | Optimal at boundary | Best | Medium |
| Epanechnikov | MSE-optimal interior | Good | High |
| Uniform | Simple | Sharp boundary | None |

**Recommendation:** Use triangular kernel - it is optimal for boundary estimation and gives more weight to observations closest to the cutoff.

---

## Bandwidth Selection

### Imbens-Kalyanaraman (IK) Bandwidth

MSE-optimal bandwidth for point estimation:

$$h_{IK} = C_K \cdot \left[\frac{\hat{\sigma}^2(c)}{\hat{f}(c) \cdot (\hat{m}^{(2)+}(c) - \hat{m}^{(2)-}(c))^2}\right]^{1/5} \cdot n^{-1/5}$$

where:
- $C_K$ is a kernel-specific constant
- $\hat{\sigma}^2(c)$ is the conditional variance at cutoff
- $\hat{f}(c)$ is the density of running variable at cutoff
- $\hat{m}^{(2)}$ is the second derivative of the regression function

### Calonico-Cattaneo-Titiunik (CCT) Bandwidth

Coverage error rate (CER) optimal bandwidth for inference:

$$h_{CCT} = h_{IK} \cdot n^{-1/10}$$

- Slightly smaller than IK bandwidth
- Optimized for confidence interval coverage
- **Recommended for inference**

### Practical Guidelines

| Purpose | Bandwidth Choice |
|---------|------------------|
| Point estimation | IK or MSE-optimal |
| Confidence intervals | CCT or CER-optimal |
| Robustness check | Multiple bandwidths (0.5h, h, 1.5h, 2h) |

---

## Bias Correction

### The Bias Problem

Local polynomial estimates have leading bias term of order $O(h^{p+1})$.

For confidence intervals to have correct coverage, this bias must be addressed.

### Robust Bias-Corrected Inference (CCT 2014)

**Step 1:** Estimate bias using higher-order polynomial:
$$\hat{B} = \text{estimate using polynomial of order } p + 1$$

**Step 2:** Subtract bias from estimate:
$$\hat{\tau}_{BC} = \hat{\tau}_{RD} - \hat{B}$$

**Step 3:** Use robust standard errors that account for bias estimation:
$$\hat{SE}_{robust} = \sqrt{\hat{V}_{conventional} + \hat{V}_{bias}}$$

### rdrobust Output Interpretation

| Column | Description | When to Use |
|--------|-------------|-------------|
| Conventional | Standard local polynomial | Point estimation only |
| Bias-Corrected | Bias-corrected estimate | Better point estimate |
| Robust | Robust bias-corrected CI | **Always use for inference** |

---

## Variance Estimation

### Heteroskedasticity-Robust Variance

$$\hat{V} = (X'WX)^{-1}X'W\Sigma WX(X'WX)^{-1}$$

where:
- $W$ is the diagonal kernel weight matrix
- $\Sigma$ is diagonal with $\hat{\sigma}_i^2 = (Y_i - \hat{Y}_i)^2$

### Cluster-Robust Variance

When observations are clustered (e.g., by school, region):

$$\hat{V}_{cluster} = (X'WX)^{-1}\left(\sum_{g=1}^{G} X_g'W_g\hat{u}_g\hat{u}_g'W_gX_g\right)(X'WX)^{-1}$$

### Nearest-Neighbor Variance (NN)

Alternative variance estimator using nearby observations:
- More robust to specification
- Often preferred in practice

---

## Fuzzy RD Estimation

### Two-Stage Approach

**First Stage:**
$$P(D = 1 | X = x) = \mu_D(x)$$

Estimate the jump in treatment probability:
$$\hat{\pi} = \hat{\mu}_D^{+} - \hat{\mu}_D^{-}$$

**Second Stage (Reduced Form):**
Estimate the jump in outcome:
$$\hat{\rho} = \hat{\mu}_Y^{+} - \hat{\mu}_Y^{-}$$

**Fuzzy RD Estimate:**
$$\hat{\tau}_{FRD} = \frac{\hat{\rho}}{\hat{\pi}} = \frac{\text{Reduced Form}}{\text{First Stage}}$$

### Interpretation

- Fuzzy RD estimate is LATE: effect for compliers at the cutoff
- Requires first-stage jump $\hat{\pi}$ to be non-zero (strong enough)
- Standard errors must account for two-stage estimation

---

## Implementation in rdrobust

```python
from rdrobust import rdrobust, rdbwselect

# Basic usage
result = rdrobust(Y, X, c=cutoff)

# With options
result = rdrobust(
    Y, X, c=cutoff,
    h=bandwidth,           # Manual bandwidth
    p=1,                   # Polynomial order
    kernel='triangular',   # Kernel type
    bwselect='cerrd',      # Bandwidth selector
    vce='hc1',             # Variance estimator
    cluster=cluster_var,   # Cluster variable
    fuzzy=D                # Treatment for fuzzy RD
)

# Key outputs
result.coef     # [conventional, bias-corrected, robust]
result.se       # Standard errors
result.ci       # Confidence intervals
result.pv       # P-values
result.bws      # Bandwidths used
result.N        # Sample sizes
```

---

## Key References

- Hahn, Todd, van der Klaauw (2001): Foundational identification paper
- Imbens & Kalyanaraman (2012): MSE-optimal bandwidth selection
- Calonico, Cattaneo, Titiunik (2014): Robust bias-corrected inference
- Gelman & Imbens (2018): Why high-order polynomials are problematic
- Cattaneo, Idrobo, Titiunik (2020): Modern practical guide
