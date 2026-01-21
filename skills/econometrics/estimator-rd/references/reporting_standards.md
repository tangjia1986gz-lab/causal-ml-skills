# RD Reporting Standards

## Required Elements

Every RD paper should include:

### 1. Design Description

- Clear statement of running variable and cutoff
- Institutional background explaining treatment rule
- Whether design is sharp or fuzzy
- Treatment compliance information (for fuzzy RD)

### 2. RD Plot

**Essential visualization showing:**
- Outcome vs running variable
- Binned scatter points on each side
- Fitted local polynomial lines
- Vertical line at cutoff
- Clear discontinuity (or lack thereof) at cutoff

### 3. Main Results Table

| Element | Required | Notes |
|---------|----------|-------|
| Point estimate | Yes | Use robust bias-corrected |
| Standard error | Yes | In parentheses below estimate |
| Confidence interval | Yes | 95% CI with robust correction |
| P-value | Optional | If reported, use robust |
| Bandwidth | Yes | Report optimal bandwidth used |
| Polynomial order | Yes | Usually p = 1 |
| Kernel | Yes | Usually triangular |
| N (left/right) | Yes | Sample sizes on each side |
| N effective | Yes | Within bandwidth |

### 4. Diagnostic Section

Must include:
- Manipulation test (McCrary or rddensity)
- Covariate balance table
- At least one robustness check (bandwidth sensitivity most important)

---

## LaTeX Table Template

### Main Results Table

```latex
\begin{table}[htbp]
\centering
\caption{Regression Discontinuity Estimates}
\label{tab:rd_main}
\begin{tabular}{lccc}
\toprule
 & (1) & (2) & (3) \\
 & Baseline & Controls & Optimal BW \\
\midrule
RD Estimate & 2.543*** & 2.481*** & 2.512*** \\
 & (0.412) & (0.398) & (0.405) \\
\addlinespace
95\% CI & [1.735, 3.351] & [1.701, 3.261] & [1.718, 3.306] \\
\addlinespace
Bandwidth & 0.152 & 0.152 & 0.168 \\
Polynomial & 1 & 1 & 1 \\
Kernel & Triangular & Triangular & Triangular \\
\addlinespace
N (left) & 2,456 & 2,456 & 2,456 \\
N (right) & 2,544 & 2,544 & 2,544 \\
N effective & 847 & 847 & 923 \\
\bottomrule
\multicolumn{4}{l}{\footnotesize Standard errors in parentheses. Robust bias-corrected inference.} \\
\multicolumn{4}{l}{\footnotesize *** p<0.01, ** p<0.05, * p<0.1} \\
\end{tabular}
\end{table}
```

### Diagnostic Tests Table

```latex
\begin{table}[htbp]
\centering
\caption{RD Validity Tests}
\label{tab:rd_validity}
\begin{tabular}{lcc}
\toprule
Test & Statistic & p-value \\
\midrule
\multicolumn{3}{l}{\textit{Panel A: Manipulation Test}} \\
McCrary density test & 0.847 & 0.397 \\
\addlinespace
\multicolumn{3}{l}{\textit{Panel B: Covariate Balance}} \\
Age & 0.152 (0.423) & 0.719 \\
Education & -0.081 (0.178) & 0.649 \\
Income (pre-treatment) & 234 (892) & 0.793 \\
Female & 0.021 (0.031) & 0.498 \\
\addlinespace
\multicolumn{3}{l}{\textit{Panel C: Placebo Cutoffs}} \\
Below true cutoff (median) & 0.089 (0.521) & 0.864 \\
Above true cutoff (median) & -0.143 (0.498) & 0.774 \\
\bottomrule
\multicolumn{3}{l}{\footnotesize Standard errors in parentheses for covariate balance.} \\
\end{tabular}
\end{table}
```

### Bandwidth Sensitivity Table

```latex
\begin{table}[htbp]
\centering
\caption{Bandwidth Sensitivity Analysis}
\label{tab:rd_bandwidth}
\begin{tabular}{lccccc}
\toprule
Bandwidth & Multiplier & Estimate & SE & 95\% CI & N effective \\
\midrule
0.076 & 0.50 & 2.687*** & 0.612 & [1.488, 3.886] & 421 \\
0.114 & 0.75 & 2.581*** & 0.478 & [1.644, 3.518] & 634 \\
0.152 & 1.00 & 2.543*** & 0.412 & [1.735, 3.351] & 847 \\
0.190 & 1.25 & 2.498*** & 0.367 & [1.779, 3.217] & 1,058 \\
0.228 & 1.50 & 2.431*** & 0.334 & [1.776, 3.086] & 1,270 \\
0.304 & 2.00 & 2.312*** & 0.289 & [1.746, 2.878] & 1,693 \\
\bottomrule
\multicolumn{6}{l}{\footnotesize *** p<0.01, ** p<0.05, * p<0.1. Robust bias-corrected inference.} \\
\end{tabular}
\end{table}
```

---

## RD Plot Guidelines

### Figure Elements

```
       Y |
         |           o o
         |          o
         |         o    ← Fitted line (right)
         |        o
    Y(c) |.......•      ← Discontinuity
         |      x
         |     x
         |    x        ← Fitted line (left)
         |   x x
         |  x
         |______________|____________
                        c             X
                     Cutoff
```

### Best Practices

1. **Binned scatter:** Use 15-25 bins on each side
2. **Fit lines:** Show local polynomial fits (order 1 or 2)
3. **Cutoff:** Clear vertical line at threshold
4. **Axes:** Label clearly with variable names
5. **No extrapolation:** Don't extend fitted lines beyond data range

### Python Code for Publication-Quality Plot

```python
import matplotlib.pyplot as plt
import numpy as np

def create_rd_plot(df, outcome, running, cutoff, n_bins=20, save_path=None):
    fig, ax = plt.subplots(figsize=(10, 6))

    x = df[running].values
    y = df[outcome].values

    left = x < cutoff
    right = x >= cutoff

    # Binned scatter
    for mask, color, label in [(left, 'blue', 'Below'), (right, 'red', 'Above')]:
        x_sub, y_sub = x[mask], y[mask]
        bins = np.linspace(x_sub.min(), x_sub.max(), n_bins // 2 + 1)

        for i in range(len(bins) - 1):
            bin_mask = (x_sub >= bins[i]) & (x_sub < bins[i+1])
            if bin_mask.sum() > 0:
                ax.scatter(x_sub[bin_mask].mean(), y_sub[bin_mask].mean(),
                          color=color, s=60, alpha=0.7)

    # Fitted lines
    if left.sum() > 5:
        coef = np.polyfit(x[left], y[left], 1)
        x_fit = np.linspace(x[left].min(), cutoff, 100)
        ax.plot(x_fit, np.polyval(coef, x_fit), 'b-', lw=2)

    if right.sum() > 5:
        coef = np.polyfit(x[right], y[right], 1)
        x_fit = np.linspace(cutoff, x[right].max(), 100)
        ax.plot(x_fit, np.polyval(coef, x_fit), 'r-', lw=2)

    ax.axvline(cutoff, color='gray', linestyle='--', lw=2)
    ax.set_xlabel(running, fontsize=12)
    ax.set_ylabel(outcome, fontsize=12)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig
```

---

## Journal-Specific Guidelines

### American Economic Review (AER)

- Preferred: Robust bias-corrected inference
- Report manipulation test prominently
- Include covariate balance in main text or appendix
- RD plot required in main text

### Quarterly Journal of Economics (QJE)

- Similar to AER
- Often request additional bandwidth robustness
- Prefer clear institutional description

### Journal of Political Economy (JPE)

- Emphasis on identification strategy description
- Clear statement of assumptions
- Robustness across specifications

### Review of Economic Studies (ReStud)

- Technical details in appendix acceptable
- Main text should highlight key findings
- Methodological rigor expected

---

## Common Reporting Mistakes

### Avoid These Errors

1. **Not reporting bandwidth:** Always state the bandwidth used
2. **Missing sample sizes:** Report N on each side and within bandwidth
3. **No RD plot:** Visual evidence is essential
4. **Ignoring manipulation test:** Must be reported even if passed
5. **Single specification only:** Need bandwidth sensitivity at minimum
6. **Global polynomial:** Don't use or report high-order global polynomials
7. **Wrong standard errors:** Use robust bias-corrected, not conventional
8. **No covariate balance:** Pre-determined covariates must be tested
9. **Missing interpretation:** Explain economic magnitude, not just significance

---

## Key References for Standards

- Lee & Lemieux (2010): JEL survey with reporting guidance
- Imbens & Lemieux (2008): Practical guide with examples
- Cattaneo, Idrobo, Titiunik (2020): Modern standards
- rdrobust documentation: Current best practices
