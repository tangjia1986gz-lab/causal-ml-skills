# IV Reporting Standards

## Required Elements

### 1. First-Stage Results

Every IV paper MUST report:

- [ ] First-stage coefficient(s) on instrument(s)
- [ ] First-stage F-statistic
- [ ] Reference to Stock-Yogo critical values
- [ ] Number of observations

### 2. Second-Stage Results

- [ ] IV coefficient estimate
- [ ] Standard error (specify: robust, clustered, etc.)
- [ ] Confidence interval
- [ ] Number of observations
- [ ] R-squared (if meaningful)

### 3. Diagnostic Tests

- [ ] Hausman test (OLS vs IV)
- [ ] Sargan/Hansen test (if overidentified)
- [ ] Weak instrument assessment

### 4. Comparison

- [ ] OLS estimate for comparison
- [ ] Discussion of bias direction

## Table Format (AER/QJE Style)

```latex
\begin{table}[htbp]
\centering
\caption{Effect of Education on Wages: IV Estimates}
\label{tab:iv_results}
\begin{tabular}{lcccc}
\toprule
& (1) & (2) & (3) & (4) \\
& OLS & 2SLS & LIML & Reduced Form \\
\midrule
\multicolumn{5}{l}{\textit{Panel A: Second Stage}} \\
Years of Education & 0.075*** & 0.132*** & 0.130*** & \\
& (0.005) & (0.021) & (0.022) & \\
[0.5em]
\multicolumn{5}{l}{\textit{Panel B: First Stage}} \\
Distance to College & & -0.320*** & -0.320*** & -0.042*** \\
& & (0.045) & (0.045) & (0.008) \\
[0.5em]
First-stage F & & 50.54 & 50.54 & \\
[0.5em]
Observations & 5,000 & 5,000 & 5,000 & 5,000 \\
\midrule
Controls & Yes & Yes & Yes & Yes \\
State FE & Yes & Yes & Yes & Yes \\
\bottomrule
\end{tabular}
\begin{tablenotes}
\small
\item Notes: Robust standard errors in parentheses. *** p<0.01, ** p<0.05, * p<0.1.
\item The instrument is distance to the nearest 4-year college.
\item Stock-Yogo 10\% critical value with 1 instrument: 16.38.
\end{tablenotes}
\end{table}
```

## Narrative Structure

### Introduction Section

1. State the endogeneity concern clearly
2. Introduce the instrument with theoretical justification
3. Preview key findings

### Methodology Section

1. **Model specification:**
   - Structural equation
   - First-stage equation
   - Define all variables

2. **Instrument validity discussion:**
   - Relevance argument (why Z affects X)
   - Exclusion restriction argument (why Z doesn't directly affect Y)

### Results Section

1. First-stage results (strength of instrument)
2. Main IV estimates
3. Comparison with OLS
4. Diagnostic tests interpretation
5. Robustness checks

### Required Discussion Points

- What does the IV estimate identify? (LATE vs ATE)
- Who are the compliers?
- Is the LATE policy-relevant?

## Common Formatting Conventions

### Significance Stars

| Symbol | Meaning |
|--------|---------|
| *** | p < 0.01 |
| ** | p < 0.05 |
| * | p < 0.10 |

### Standard Error Presentation

```
0.132***
(0.021)     <- SE in parentheses
[0.091, 0.173]  <- 95% CI in brackets (optional)
```

### F-statistic Reporting

Always include:
1. F-statistic value
2. Reference critical value
3. Interpretation (strong/weak)

Example:
> "The first-stage F-statistic is 50.54, well above the Stock-Yogo (2005) critical value of 16.38 for 10% maximal IV size, suggesting our instrument is strong."

## Robustness Checks to Report

1. **Alternative instruments** (if available)
2. **Subset analysis** (different samples)
3. **Specification tests** (adding/removing controls)
4. **LIML alongside 2SLS** (weak IV robustness)
5. **Anderson-Rubin CI** (if F is borderline)

## What NOT to Do

- Don't report 2SLS without first-stage F
- Don't claim exclusion is "tested" (it's not testable)
- Don't ignore Sargan test rejections
- Don't present weak-IV results without robust inference
- Don't omit OLS comparison

## Python Code for LaTeX Table

```python
def generate_iv_latex_table(ols_result, iv_result, first_stage):
    latex = r"""
\begin{table}[htbp]
\centering
\caption{IV Estimation Results}
\begin{tabular}{lcc}
\toprule
& OLS & 2SLS \\
\midrule
"""
    # Add coefficient rows
    ols_coef = ols_result['coefficient']
    ols_se = ols_result['std_error']
    iv_coef = iv_result['coefficient']
    iv_se = iv_result['std_error']

    latex += f"Effect & {ols_coef:.4f} & {iv_coef:.4f} \\\\\n"
    latex += f" & ({ols_se:.4f}) & ({iv_se:.4f}) \\\\\n"

    # First-stage F
    f_stat = first_stage['f_statistic']
    latex += f"First-stage F & -- & {f_stat:.2f} \\\\\n"

    latex += r"""
\bottomrule
\end{tabular}
\end{table}
"""
    return latex
```
