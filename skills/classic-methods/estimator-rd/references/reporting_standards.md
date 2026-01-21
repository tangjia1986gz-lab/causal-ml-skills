# RD Reporting Standards

> **Document Type**: Reporting Guidelines
> **Last Updated**: 2024-01
> **Key References**: Lee & Lemieux (2010), Cattaneo et al. (2020)

## Overview

Transparent and complete reporting is essential for evaluating the credibility of RD designs. This document provides standards for presenting RD analyses in academic papers, reports, and policy documents.

---

## 1. The Essential RD Plot

### Purpose
The RD plot is the most important visualization. It should clearly show:
1. The discontinuity in outcomes at the cutoff
2. The underlying relationship between running variable and outcome
3. The data distribution around the cutoff

### Components

```
+------------------------------------------------------------------+
|                      RD Plot Structure                            |
+------------------------------------------------------------------+
|                                                                   |
|    Binned Scatter Points     Fitted Polynomial Lines              |
|         (circles)                (smooth curves)                  |
|                                                                   |
|           o                              ____                     |
|          o o                       _____/                         |
|         o   o               ______/                               |
|        o     o        _____/                                      |
|       o       o _____|     <- DISCONTINUITY                       |
|      o       __|                                                  |
|     o   ____/  |                                                  |
|    ____/       |                                                  |
|   /            |                                                  |
|                c (cutoff)                                         |
+------------------------------------------------------------------+
```

### Implementation

```python
from rd_estimator import rd_plot

fig = rd_plot(
    data=df,
    running="score",
    outcome="y",
    cutoff=0.0,
    bandwidth=optimal_bw,
    n_bins=20,            # Number of bins for scatter
    poly_order=1,         # Linear fit (or 2 for quadratic)
    ci=True,              # Show confidence intervals
    figsize=(10, 6),
    title="Effect of [Treatment] on [Outcome]"
)

fig.savefig("rd_plot.png", dpi=300, bbox_inches='tight')
```

### Best Practices for RD Plots

| Do | Don't |
|----|-------|
| Show raw binned data | Show only fitted lines |
| Use appropriate bin width | Use too few bins (hides variation) |
| Include confidence bands | Omit uncertainty |
| Mark the cutoff clearly | Hide the threshold |
| Label axes informatively | Use generic labels |

### Publication-Ready Code

```python
import matplotlib.pyplot as plt
import numpy as np

def publication_rd_plot(data, running, outcome, cutoff, bandwidth, n_bins=20):
    """Create publication-quality RD plot."""
    fig, ax = plt.subplots(figsize=(10, 6))

    x = data[running].values
    y = data[outcome].values

    # Create bins
    bin_edges = np.linspace(x.min(), x.max(), n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Calculate bin means and SEs
    bin_means = []
    bin_ses = []
    for i in range(n_bins):
        mask = (x >= bin_edges[i]) & (x < bin_edges[i + 1])
        if mask.sum() > 0:
            bin_means.append(y[mask].mean())
            bin_ses.append(y[mask].std() / np.sqrt(mask.sum()))
        else:
            bin_means.append(np.nan)
            bin_ses.append(np.nan)

    bin_means = np.array(bin_means)
    bin_ses = np.array(bin_ses)

    # Color by side of cutoff
    below = bin_centers < cutoff
    above = ~below

    # Plot binned scatter with error bars
    ax.errorbar(bin_centers[below], bin_means[below], yerr=1.96*bin_ses[below],
                fmt='o', color='steelblue', markersize=6, capsize=3,
                label='Below cutoff')
    ax.errorbar(bin_centers[above], bin_means[above], yerr=1.96*bin_ses[above],
                fmt='o', color='indianred', markersize=6, capsize=3,
                label='Above cutoff')

    # Cutoff line
    ax.axvline(x=cutoff, color='black', linestyle='--', linewidth=1.5,
               label=f'Cutoff = {cutoff}')

    # Bandwidth markers
    ax.axvline(x=cutoff - bandwidth, color='gray', linestyle=':', alpha=0.5)
    ax.axvline(x=cutoff + bandwidth, color='gray', linestyle=':', alpha=0.5)

    # Labels
    ax.set_xlabel('Running Variable', fontsize=12)
    ax.set_ylabel('Outcome', fontsize=12)
    ax.legend(loc='best', frameon=True)
    ax.grid(True, alpha=0.3)

    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    return fig
```

---

## 2. Bandwidth Sensitivity Tables

### Purpose
Show that results are robust to bandwidth choice, the most important specification decision.

### Standard Format

```
Table X: Bandwidth Sensitivity Analysis
============================================================
                    (1)        (2)        (3)        (4)
                  0.5×h_opt   h_opt    1.5×h_opt   2×h_opt
------------------------------------------------------------
RD Estimate        0.48***    0.52***    0.49***    0.45***
                  (0.18)     (0.15)     (0.13)     (0.12)

95% Robust CI    [0.13,0.83] [0.23,0.81] [0.24,0.74] [0.22,0.68]

Bandwidth          0.175      0.35       0.525      0.70
Effective N         250        512        789       1024
------------------------------------------------------------
Optimal bandwidth (MSE): 0.35
Polynomial order: 1 (linear)
Kernel: Triangular
============================================================
Notes: Robust bias-corrected confidence intervals.
Standard errors in parentheses.
*** p<0.01, ** p<0.05, * p<0.1
```

### Implementation

```python
from rd_estimator import bandwidth_sensitivity

# Generate sensitivity table
sens = bandwidth_sensitivity(
    data=df,
    running="score",
    outcome="y",
    cutoff=0.0,
    bandwidth_range=[0.5*h_opt, 0.75*h_opt, h_opt, 1.25*h_opt, 1.5*h_opt, 2*h_opt]
)

# Format as LaTeX table
print(create_bandwidth_table_latex(sens))
```

---

## 3. Main Results Table

### Standard Format

```
Table X: Regression Discontinuity Estimates
================================================================
                          (1)         (2)         (3)         (4)
                       Sharp RD    Sharp RD    Fuzzy RD    Donut RD
----------------------------------------------------------------
RD Effect               0.52***     0.48***     0.67***     0.55***
                       (0.15)      (0.18)      (0.21)      (0.22)

95% Robust CI         [0.23,0.81] [0.13,0.83]  [0.26,1.08] [0.12,0.98]

----------------------------------------------------------------
Polynomial Order          1           2           1           1
Bandwidth               0.35        0.35        0.35        0.35
Donut Radius             --          --          --         0.05
Kernel               Triangular  Triangular  Triangular  Triangular

N (left of cutoff)      245         245         245         235
N (right of cutoff)     267         267         267         255
Effective N             512         512         512         490
----------------------------------------------------------------
First Stage              --          --        0.78         --
F-statistic (1st)        --          --        45.3         --
================================================================
Notes: Robust bias-corrected standard errors in parentheses.
Column (1): Baseline sharp RD with linear fit.
Column (2): Quadratic polynomial specification.
Column (3): Fuzzy RD with treatment as endogenous variable.
Column (4): Donut hole RD excluding |X-c| < 0.05.
*** p<0.01, ** p<0.05, * p<0.1
```

---

## 4. Diagnostic Summary Table

### Format

```
Table X: RD Design Validity Tests
================================================================
Panel A: Manipulation Test
----------------------------------------------------------------
McCrary Density Test
  Log difference in density at cutoff        -0.042
  Standard error                              0.089
  P-value                                     0.67
  Conclusion                                  No evidence of manipulation

================================================================
Panel B: Covariate Balance at Cutoff
----------------------------------------------------------------
Covariate           Mean Below    Mean Above    Difference    P-value
----------------------------------------------------------------
Age                   24.3          24.1          -0.2         0.72
Female                0.52          0.49          -0.03        0.45
Income ($1000)        45.2          46.1           0.9         0.61
Prior GPA             3.12          3.08          -0.04        0.58
----------------------------------------------------------------
Joint test (Fisher)                                            0.82

================================================================
Panel C: Placebo Cutoff Tests
----------------------------------------------------------------
Placebo Cutoff        Effect        SE          P-value
----------------------------------------------------------------
-0.50                  0.05        0.12          0.68
-0.25                 -0.08        0.11          0.47
 0.25                  0.03        0.10          0.76
 0.50                 -0.02        0.13          0.88
----------------------------------------------------------------
Joint test (Bonferroni-corrected)                              Pass
================================================================
```

---

## 5. Donut Hole Sensitivity Table

### Purpose
Show whether results are robust to excluding observations near the cutoff.

### Format

```
Table X: Donut Hole Sensitivity Analysis
============================================================
Donut Radius    Effect      SE        N Excluded    P-value
------------------------------------------------------------
0.00 (none)      0.52      0.15          0          0.001
0.01             0.53      0.16         23          0.001
0.02             0.54      0.17         41          0.002
0.05             0.55      0.22         89          0.012
0.10             0.51      0.28        156          0.068
------------------------------------------------------------
Notes: All specifications use MSE-optimal bandwidth (0.35)
and linear polynomial. Standard errors are robust bias-corrected.
============================================================
```

---

## 6. RD Report Template (Markdown)

```markdown
# Regression Discontinuity Analysis Report

## Executive Summary

[Brief summary of research question, design, and main finding]

---

## 1. Design Overview

### Research Question
[What causal question does this RD address?]

### Treatment and Cutoff
- **Treatment**: [Description]
- **Running Variable**: [Name and description]
- **Cutoff Value**: [Value]
- **Design Type**: [Sharp/Fuzzy]

### Sample
- Total observations: N = [total]
- Below cutoff: N = [below]
- Above cutoff: N = [above]
- Effective sample (within bandwidth): N = [effective]

---

## 2. Design Validity

### Manipulation Test (McCrary)
[Insert McCrary test results and interpretation]

### Covariate Balance
[Insert balance table]

### Density Plot
[Insert histogram of running variable]

---

## 3. Main Results

### RD Plot
[Insert RD plot]

### Effect Estimate
| Specification | Effect | SE | 95% CI | p-value |
|---------------|--------|-----|--------|---------|
| Baseline (linear) | | | | |
| Quadratic | | | | |
| Half bandwidth | | | | |
| Double bandwidth | | | | |

### Interpretation
[Plain language interpretation of the effect]

---

## 4. Robustness Checks

### Bandwidth Sensitivity
[Insert bandwidth sensitivity table/plot]

### Placebo Cutoffs
[Insert placebo test results]

### Donut Hole Analysis
[If applicable, insert donut hole results]

---

## 5. Limitations

1. **External Validity**: [RD estimates LATE at cutoff]
2. **Manipulation Concerns**: [Any residual concerns?]
3. **Data Limitations**: [Sample size, measurement issues]

---

## 6. Conclusions

[Summary of findings and policy implications]

---

## Technical Appendix

### Estimation Details
- Bandwidth selection method: [MSE-optimal / CER-optimal]
- Kernel: [Triangular / Uniform]
- Polynomial order: [1 / 2]
- Software: [rdrobust / rd_estimator]

### Code Availability
[Link to replication code]
```

---

## 7. Minimum Reporting Requirements

### Must Report

1. **Design Setup**
   - Cutoff value
   - Running variable description
   - Sharp or Fuzzy RD
   - Sample sizes (total, by side, effective)

2. **Main Estimate**
   - Point estimate with standard error
   - Confidence interval (robust bias-corrected)
   - Bandwidth used and selection method
   - Polynomial order

3. **Validity Evidence**
   - McCrary test statistic and p-value
   - Covariate balance (at least some covariates)

4. **Visual**
   - RD plot with binned scatter and fitted lines

### Should Report

5. **Robustness**
   - Bandwidth sensitivity (at least 3 bandwidths)
   - Alternative polynomial order
   - Placebo cutoffs

6. **Additional Diagnostics**
   - Histogram of running variable
   - Donut hole (if manipulation concerns)

### Nice to Have

7. **Extended Analysis**
   - Heterogeneity by subgroups
   - Alternative kernels
   - Local randomization inference

---

## 8. Common Reporting Mistakes

### Mistake 1: Only Showing Fitted Lines

**Problem**: RD plot shows only polynomial fits without underlying data.

**Fix**: Always include binned scatter points with the fitted lines.

### Mistake 2: Not Reporting Bandwidth

**Problem**: Paper reports effect but not the bandwidth used.

**Fix**: Always report bandwidth, selection method, and effective sample size.

### Mistake 3: Using Conventional CI

**Problem**: Reporting confidence intervals that don't account for bias.

**Fix**: Use robust bias-corrected confidence intervals from rdrobust.

### Mistake 4: No Sensitivity Analysis

**Problem**: Only reporting one specification.

**Fix**: Show results for multiple bandwidths and polynomial orders.

### Mistake 5: Ignoring Manipulation Test

**Problem**: Not testing or reporting McCrary density test.

**Fix**: Always test and report, even if it passes.

---

## 9. Checklist Before Submission

- [ ] RD plot with binned scatter and fitted lines
- [ ] McCrary test reported
- [ ] At least one covariate balance check
- [ ] Main estimate with robust SE and CI
- [ ] Bandwidth reported with selection method
- [ ] Effective sample size reported
- [ ] Bandwidth sensitivity (3+ bandwidths)
- [ ] Polynomial sensitivity (linear vs quadratic)
- [ ] Placebo cutoff tests
- [ ] Clear interpretation of LATE

---

## References

- Lee, D. S., & Lemieux, T. (2010). Regression discontinuity designs in economics. *Journal of Economic Literature*, 48(2), 281-355.
- Cattaneo, M. D., Idrobo, N., & Titiunik, R. (2020). *A Practical Introduction to Regression Discontinuity Designs: Foundations*. Cambridge University Press.
- Imbens, G. W., & Lemieux, T. (2008). Regression discontinuity designs: A guide to practice. *Journal of Econometrics*, 142(2), 615-635.
