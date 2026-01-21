# Tables and Figures for Economics Papers

## General Principles

1. **Tables for precision, figures for intuition**
2. **Each table/figure should tell one clear story**
3. **Standalone**: Reader should understand without reading text
4. **Consistent formatting** throughout paper
5. **AER style** is the gold standard

---

## Table Formatting (AER Style)

### The Three-Line Rule

Economics tables use only **three horizontal lines**:
1. Top rule (above header)
2. Header rule (below header)
3. Bottom rule (below content)

**No vertical lines. Ever.**

### Basic Table Structure

```latex
\begin{table}[htbp]
\centering
\caption{Effect of Education on Earnings}
\label{tab:main_results}
\begin{tabular}{lccc}
\toprule
                    & (1)      & (2)      & (3)      \\
                    & OLS      & IV       & RDD      \\
\midrule
Years of Education  & 0.082*** & 0.092*** & 0.089*** \\
                    & (0.012)  & (0.024)  & (0.018)  \\
                    &          &          &          \\
Female              & -0.156***& -0.154***& -0.152***\\
                    & (0.018)  & (0.019)  & (0.021)  \\
                    &          &          &          \\
Experience          & 0.043*** & 0.041*** & 0.042*** \\
                    & (0.003)  & (0.004)  & (0.004)  \\
\midrule
Controls            & No       & Yes      & Yes      \\
State FE            & No       & No       & Yes      \\
Observations        & 50,000   & 50,000   & 45,000   \\
R-squared           & 0.234    & 0.198    & 0.267    \\
\bottomrule
\end{tabular}
\begin{tablenotes}
\small
\item Notes: Standard errors in parentheses, clustered at state level.
*** p<0.01, ** p<0.05, * p<0.1. Dependent variable is log hourly wage.
Column (1) presents OLS estimates. Column (2) instruments for education
using compulsory schooling laws. Column (3) uses regression discontinuity
design based on school entry age cutoffs.
\end{tablenotes}
\end{table}
```

### Visual Example

```
Table 1: Effect of Education on Earnings
═══════════════════════════════════════════════════════════
                        (1)         (2)         (3)
                        OLS         IV          RDD
───────────────────────────────────────────────────────────
Years of Education    0.082***    0.092***    0.089***
                      (0.012)     (0.024)     (0.018)

Female               -0.156***   -0.154***   -0.152***
                      (0.018)     (0.019)     (0.021)

Experience            0.043***    0.041***    0.042***
                      (0.003)     (0.004)     (0.004)
───────────────────────────────────────────────────────────
Controls                No          Yes         Yes
State FE                No          No          Yes
Observations          50,000      50,000      45,000
R-squared             0.234       0.198       0.267
═══════════════════════════════════════════════════════════
Notes: Standard errors in parentheses, clustered at state level.
*** p<0.01, ** p<0.05, * p<0.1.
```

---

## Regression Table Best Practices

### Column Organization

Build complexity across columns:
- **Column 1**: Simplest specification (bivariate or basic controls)
- **Columns 2-4**: Add controls, fixed effects progressively
- **Final column**: Preferred specification

### Row Organization

```
Variable of Interest    ← FIRST (this is what readers care about)
────────────────────
Key Controls            ← Important covariates (if showing)
────────────────────
[Control rows may be suppressed with note: "Controls include..."]
────────────────────
Specification Details   ← Fixed effects, clustering, etc.
Sample Size
Fit Statistics          ← R², F-stat, etc.
```

### Standard Errors

Always report standard errors in parentheses below coefficients:
```
0.092***
(0.024)
```

**State clustering clearly in notes**:
- "Standard errors clustered at state level"
- "Robust standard errors in parentheses"
- "Wild cluster bootstrap p-values in brackets"

### Significance Stars

Standard convention:
- *** p<0.01
- ** p<0.05
- * p<0.1

Some journals (QJE) discourage stars; check guidelines.

### What to Include in Notes

1. What's in parentheses (SEs, t-stats, p-values)
2. Clustering level
3. Significance levels
4. Dependent variable definition
5. Sample restrictions
6. Column-specific information

---

## Table Types

### Summary Statistics Table

```
Table 1: Summary Statistics
═══════════════════════════════════════════════════════════════
                              Mean      SD        Min      Max
───────────────────────────────────────────────────────────────
Panel A: Outcome Variables
  Log Hourly Wage            2.89     (0.54)    1.23     4.56
  Employed                   0.82     (0.38)    0        1

Panel B: Education
  Years of Schooling        13.2      (2.8)     8        20
  College Graduate           0.34     (0.47)    0        1

Panel C: Demographics
  Age                       42.3      (11.2)    25       65
  Female                     0.48     (0.50)    0        1

Observations               50,000
═══════════════════════════════════════════════════════════════
Notes: Data from Current Population Survey, 2015-2020. Sample
restricted to ages 25-65, employed at least 20 hours per week.
```

### Balance Table (RCT/RDD)

```
Table 2: Covariate Balance
═══════════════════════════════════════════════════════════════
                          Control    Treated    Diff      p-value
───────────────────────────────────────────────────────────────
Age                        42.1       42.3      0.2       0.654
Female                      0.48       0.49     0.01      0.782
College                     0.33       0.34     0.01      0.845
Baseline Earnings        45,234     45,891     657       0.412

Joint F-test                                              0.723
Observations               5,000      5,000
═══════════════════════════════════════════════════════════════
Notes: Column (3) reports difference in means. Column (4) reports
p-value from t-test of equality. Joint F-test is from regression of
treatment on all covariates.
```

### Heterogeneity Table

```
Table 4: Heterogeneous Effects by Subgroup
═══════════════════════════════════════════════════════════════
                          (1)         (2)         (3)
                         Full       Female       Male
───────────────────────────────────────────────────────────────
Treatment Effect        0.092***    0.124***    0.067**
                       (0.024)     (0.031)     (0.029)

p-value (Female=Male)              0.034
───────────────────────────────────────────────────────────────
Observations            50,000      24,000      26,000
═══════════════════════════════════════════════════════════════
```

---

## Figure Guidelines

### When to Use Figures

- **RDD**: Discontinuity plot is essential
- **DID**: Event study plot
- **Distributions**: Histograms, densities
- **Mechanisms**: Flow diagrams
- **Geographic variation**: Maps
- **Time series**: Trends

### Figure Formatting

```latex
\begin{figure}[htbp]
\centering
\includegraphics[width=0.8\textwidth]{figures/rdd_plot.pdf}
\caption{Effect of Education on Earnings: RDD Evidence}
\label{fig:rdd}
\begin{figurenotes}
Notes: Figure shows average log hourly earnings by years of
education relative to the school entry cutoff. Bin width is
0.1 years. Solid lines show local linear regression fit with
bandwidth of 2 years. Dashed lines show 95% confidence intervals.
Sample restricted to individuals born within 5 years of the cutoff.
\end{figurenotes}
\end{figure}
```

### RDD Plot

Essential elements:
1. Running variable on x-axis
2. Outcome on y-axis
3. Binned scatter points
4. Fitted lines on each side of cutoff
5. Clear vertical line at cutoff
6. Confidence intervals

```python
# Conceptual structure
plt.figure(figsize=(10, 6))
plt.scatter(x_bins, y_means, s=50, alpha=0.6)  # Binned means
plt.axvline(x=0, color='red', linestyle='--')  # Cutoff
plt.plot(x_left, fit_left, 'b-')               # Left fit
plt.plot(x_right, fit_right, 'b-')             # Right fit
plt.fill_between(x, ci_low, ci_high, alpha=0.2) # CI
plt.xlabel('Running Variable (Centered at Cutoff)')
plt.ylabel('Outcome')
```

### Event Study Plot

Essential elements:
1. Time relative to treatment on x-axis
2. Coefficient estimates on y-axis
3. Reference period (usually t=-1) normalized to zero
4. Confidence intervals (bars or bands)
5. Vertical line at t=0

```python
# Conceptual structure
plt.figure(figsize=(10, 6))
plt.errorbar(time_periods, coefficients, yerr=ci_width,
             fmt='o', capsize=3)
plt.axhline(y=0, color='gray', linestyle='--')
plt.axvline(x=-0.5, color='red', linestyle='--')
plt.xlabel('Periods Relative to Treatment')
plt.ylabel('Effect on Outcome')
```

### Density/McCrary Plot

For RDD manipulation tests:
```python
# Show density of running variable
plt.figure(figsize=(10, 6))
plt.hist(running_var[running_var < 0], bins=50, alpha=0.7, label='Below')
plt.hist(running_var[running_var >= 0], bins=50, alpha=0.7, label='Above')
plt.axvline(x=0, color='red', linestyle='--')
plt.xlabel('Running Variable')
plt.ylabel('Density')
```

---

## Color and Style Guidelines

### Color Palette

Use colorblind-friendly palettes:
```python
# Recommended colors
colors = {
    'primary': '#1f77b4',    # Blue
    'secondary': '#ff7f0e',  # Orange
    'tertiary': '#2ca02c',   # Green
    'accent': '#d62728',     # Red
    'gray': '#7f7f7f'
}
```

### Line Styles

For black-and-white printing:
```python
line_styles = ['-', '--', '-.', ':']
markers = ['o', 's', '^', 'D']
```

### Font Sizes

```python
plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10
})
```

---

## Converting Stata/R Output

### From Stata

```stata
* Export to LaTeX
esttab using "table1.tex", ///
    se star(* 0.1 ** 0.05 *** 0.01) ///
    label booktabs ///
    title("Main Results") ///
    note("Standard errors in parentheses")
```

### From R

```r
# Using stargazer
library(stargazer)
stargazer(model1, model2, model3,
          type = "latex",
          se = list(se1, se2, se3),
          title = "Main Results",
          notes = "Standard errors in parentheses")

# Using modelsummary
library(modelsummary)
modelsummary(list(model1, model2, model3),
             stars = TRUE,
             output = "latex")
```

### From Python

```python
# Using stargazer (Python port)
from stargazer.stargazer import Stargazer
stargazer = Stargazer([model1, model2, model3])
print(stargazer.render_latex())
```

---

## Common Mistakes

### Tables

1. **Too many columns**: Limit to 6-8 columns
2. **Too many rows**: Move to appendix if >15 variables
3. **Missing notes**: Always explain what's in parentheses
4. **Inconsistent formatting**: Same number of decimals throughout
5. **Variable names not labels**: "ln_wage" → "Log Hourly Wage"
6. **Vertical lines**: Never use them

### Figures

1. **Axis labels missing**: Always label axes
2. **Legend unclear**: Use descriptive labels
3. **Too small**: Ensure readability at print size
4. **Overcrowded**: One message per figure
5. **Wrong aspect ratio**: Usually 4:3 or 16:9
6. **Missing confidence intervals**: Always show uncertainty

---

## Appendix Tables

For robustness checks and additional results:
- Use same formatting as main tables
- Reference clearly in text: "Table A1 in the Appendix"
- Can be more detailed/technical
- Include full coefficient vectors if space allows

---

## Checklist Before Submission

### Tables
- [ ] Three-line format (no vertical lines)
- [ ] Standard errors in parentheses with clustering noted
- [ ] Significance stars defined in notes
- [ ] Dependent variable clearly stated
- [ ] Sample size and fit statistics included
- [ ] Variable labels (not code names)
- [ ] Consistent decimal places
- [ ] Referenced in text

### Figures
- [ ] Axis labels with units
- [ ] Legend if multiple series
- [ ] Confidence intervals shown
- [ ] Source data noted
- [ ] Readable at print size
- [ ] Black-and-white friendly
- [ ] Referenced in text
