# PSM Reporting Standards

> **Reference Document** | Propensity Score Matching
> Based on Austin (2011), STROBE Guidelines

## Overview

This document provides standards for reporting PSM analyses, including required tables, figures, and interpretive text.

---

## 1. Essential Reporting Elements

### 1.1 Minimum Requirements

Every PSM analysis should report:

```
REQUIRED REPORTING CHECKLIST
============================

DATA & DESIGN
[ ] Sample sizes (treated, control, total)
[ ] Data source and time period
[ ] Treatment definition
[ ] Outcome definition
[ ] Covariate selection rationale

PROPENSITY SCORE MODEL
[ ] PS estimation method (logit, probit, ML)
[ ] All covariates included in PS model
[ ] Model diagnostics (AUC, calibration)

MATCHING
[ ] Matching algorithm used
[ ] Matching parameters (caliper, replacement)
[ ] Number of matched pairs/units
[ ] Number of unmatched units

BALANCE
[ ] Balance table (before and after)
[ ] SMD for all covariates
[ ] Variance ratios (optional but recommended)
[ ] PS distribution overlap

RESULTS
[ ] Point estimate with SE/CI
[ ] Estimand (ATT/ATE)
[ ] Statistical significance

SENSITIVITY
[ ] Rosenbaum bounds or equivalent
[ ] Discussion of unobserved confounding
```

---

## 2. Balance Table

### 2.1 Standard Format

```
+--------------------------------------------------------------------------------+
|                        TABLE 1: COVARIATE BALANCE                              |
+--------------------------------------------------------------------------------+
|                |        Before Matching          |       After Matching        |
|                |----------------------------------|----------------------------|
| Variable       | Treated | Control | SMD  | VR   | Treated | Control | SMD | VR |
+--------------------------------------------------------------------------------+
| Age (years)    |   45.2  |  52.1   | 0.42 | 1.12 |   45.2  |  45.8   | 0.04| 1.02|
| Male (%)       |   62.3  |  58.1   | 0.09 | 1.04 |   62.3  |  61.5   | 0.02| 1.01|
| Income ($K)    |   75.4  |  68.2   | 0.31 | 1.25 |   75.4  |  74.1   | 0.06| 1.08|
| Education (yr) |   14.2  |  13.1   | 0.38 | 0.95 |   14.2  |  14.0   | 0.07| 0.98|
| Employed (%)   |   78.5  |  71.2   | 0.17 | 1.06 |   78.5  |  77.9   | 0.01| 1.00|
| Married (%)    |   55.4  |  48.9   | 0.13 | 1.02 |   55.4  |  54.8   | 0.01| 1.01|
+--------------------------------------------------------------------------------+
| N              |   500   |  1500   |      |      |   485   |   485   |     |     |
+--------------------------------------------------------------------------------+

Notes: SMD = Standardized Mean Difference; VR = Variance Ratio
Target: |SMD| < 0.1, VR in [0.5, 2.0]
```

### 2.2 Balance Table Code

```python
def create_balance_table(
    data_before: pd.DataFrame,
    data_after: pd.DataFrame,
    treatment: str,
    covariates: List[str],
    weights_col: str = None,
    output_format: str = 'markdown'
) -> str:
    """
    Create publication-quality balance table.

    Parameters
    ----------
    data_before : pd.DataFrame
        Data before matching
    data_after : pd.DataFrame
        Data after matching
    treatment : str
        Treatment column name
    covariates : List[str]
        List of covariate names
    weights_col : str, optional
        Matching weights column
    output_format : str
        'markdown', 'latex', or 'html'

    Returns
    -------
    str
        Formatted balance table
    """

    def calc_stats(df, treat_col, cov, weights=None):
        """Calculate means and SMD for one covariate."""
        treated = df[df[treat_col] == 1][cov]
        control = df[df[treat_col] == 0][cov]

        if weights and weights in df.columns:
            wt = df[df[treat_col] == 1][weights]
            wc = df[df[treat_col] == 0][weights]
            mean_t = np.average(treated, weights=wt)
            mean_c = np.average(control, weights=wc)
            var_t = np.average((treated - mean_t)**2, weights=wt)
            var_c = np.average((control - mean_c)**2, weights=wc)
        else:
            mean_t = treated.mean()
            mean_c = control.mean()
            var_t = treated.var()
            var_c = control.var()

        pooled_sd = np.sqrt((var_t + var_c) / 2)
        smd = (mean_t - mean_c) / pooled_sd if pooled_sd > 0 else 0
        vr = var_t / var_c if var_c > 0 else np.inf

        return mean_t, mean_c, smd, vr

    # Build table
    rows = []
    for cov in covariates:
        mt_b, mc_b, smd_b, vr_b = calc_stats(data_before, treatment, cov)
        mt_a, mc_a, smd_a, vr_a = calc_stats(data_after, treatment, cov, weights_col)
        rows.append({
            'Variable': cov,
            'Treat_Before': mt_b, 'Ctrl_Before': mc_b,
            'SMD_Before': smd_b, 'VR_Before': vr_b,
            'Treat_After': mt_a, 'Ctrl_After': mc_a,
            'SMD_After': smd_a, 'VR_After': vr_a
        })

    # Format output
    if output_format == 'latex':
        return _format_latex(rows)
    elif output_format == 'html':
        return _format_html(rows)
    else:
        return _format_markdown(rows)
```

---

## 3. Propensity Score Distribution

### 3.1 Required Visualization

```
+--------------------------------------------------------------------+
|              FIGURE 1: PROPENSITY SCORE DISTRIBUTIONS               |
+--------------------------------------------------------------------+
|                                                                    |
|  Panel A: Before Matching                                          |
|  +--------------------------------------------------------------+ |
|  |                                                              | |
|  |  Treated   |             ************                        | |
|  |  __________|_____________________________________            | |
|  |  Control   |  **************                                 | |
|  |                                                              | |
|  |            0        0.25       0.5       0.75        1       | |
|  +--------------------------------------------------------------+ |
|                                                                    |
|  Panel B: After Matching                                           |
|  +--------------------------------------------------------------+ |
|  |                                                              | |
|  |  Treated   |             ************                        | |
|  |  __________|_____________________________________            | |
|  |  Control   |             ************                        | |
|  |                                                              | |
|  |            0        0.25       0.5       0.75        1       | |
|  +--------------------------------------------------------------+ |
|                                                                    |
+--------------------------------------------------------------------+
```

### 3.2 Overlap Statistics to Report

| Statistic | Before | After |
|-----------|--------|-------|
| PS Range (Treated) | [0.08, 0.95] | [0.12, 0.88] |
| PS Range (Control) | [0.02, 0.75] | [0.10, 0.85] |
| Common Support | [0.08, 0.75] | [0.12, 0.85] |
| % Treated in CS | 92% | 100% |
| % Control in CS | 95% | 100% |

---

## 4. Love Plot

### 4.1 Standard Format

```
+--------------------------------------------------------------------+
|                     FIGURE 2: LOVE PLOT                            |
|              Standardized Mean Differences                          |
+--------------------------------------------------------------------+
|                                                                    |
| Age            o----------------------->*                           |
| Male           o-->*                                                |
| Income         o--------------->*                                   |
| Education      o---------->*                                        |
| Employed       o---->*                                              |
| Married        o--->*                                               |
|                |         |         |         |         |           |
|               0.0       0.1       0.2       0.3       0.4          |
|                         |                                          |
|                         v                                          |
|                    Threshold                                       |
|                                                                    |
| Legend:  o = Before matching   * = After matching                  |
|          Dashed line at |SMD| = 0.1                                |
+--------------------------------------------------------------------+
```

### 4.2 Love Plot Code

```python
def create_love_plot(
    smd_before: Dict[str, float],
    smd_after: Dict[str, float],
    threshold: float = 0.1,
    title: str = "Covariate Balance (Love Plot)",
    figsize: Tuple[int, int] = (10, 8)
) -> plt.Figure:
    """
    Create publication-quality Love plot.

    Parameters
    ----------
    smd_before : dict
        {covariate: SMD} before matching
    smd_after : dict
        {covariate: SMD} after matching
    threshold : float
        Balance threshold (usually 0.1)
    title : str
        Plot title
    figsize : tuple
        Figure dimensions

    Returns
    -------
    matplotlib.Figure
    """
    import matplotlib.pyplot as plt

    covariates = list(smd_before.keys())
    n_cov = len(covariates)

    fig, ax = plt.subplots(figsize=figsize)

    y_pos = np.arange(n_cov)

    # Before matching (circles)
    ax.scatter(
        [abs(smd_before[c]) for c in covariates],
        y_pos,
        marker='o', s=100, c='red', label='Before matching',
        zorder=3
    )

    # After matching (squares)
    ax.scatter(
        [abs(smd_after[c]) for c in covariates],
        y_pos,
        marker='s', s=100, c='blue', label='After matching',
        zorder=3
    )

    # Connecting lines
    for i, cov in enumerate(covariates):
        ax.plot(
            [abs(smd_before[cov]), abs(smd_after[cov])],
            [i, i],
            'k-', alpha=0.3, zorder=2
        )

    # Threshold line
    ax.axvline(x=threshold, color='green', linestyle='--',
               linewidth=2, label=f'Threshold ({threshold})')

    # Formatting
    ax.set_yticks(y_pos)
    ax.set_yticklabels(covariates)
    ax.set_xlabel('Absolute Standardized Mean Difference')
    ax.set_title(title)
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3, axis='x')
    ax.set_xlim(0, None)

    plt.tight_layout()
    return fig
```

---

## 5. Results Table

### 5.1 Standard Format

```
+--------------------------------------------------------------------+
|                 TABLE 2: TREATMENT EFFECT ESTIMATES                 |
+--------------------------------------------------------------------+
|                                                                    |
| Estimand: Average Treatment Effect on the Treated (ATT)            |
| Matching Method: Nearest Neighbor (1:1 with replacement)           |
| Caliper: 0.1 SD of propensity score                                |
|                                                                    |
+--------------------------------------------------------------------+
| Specification      | Effect   | SE      | 95% CI         | p-value |
+--------------------------------------------------------------------+
| Primary            | 2.34     | 0.45    | [1.46, 3.22]   | <0.001  |
| Without caliper    | 2.41     | 0.48    | [1.47, 3.35]   | <0.001  |
| Kernel matching    | 2.28     | 0.42    | [1.46, 3.10]   | <0.001  |
| IPW                | 2.52     | 0.51    | [1.52, 3.52]   | <0.001  |
| Doubly robust      | 2.39     | 0.44    | [1.53, 3.25]   | <0.001  |
+--------------------------------------------------------------------+
|                                                                    |
| N (Treated)        | 485                                           |
| N (Control)        | 485 matched (from 1500 available)             |
| Unmatched Treated  | 15 (3%)                                       |
+--------------------------------------------------------------------+
```

---

## 6. Sensitivity Analysis Table

### 6.1 Rosenbaum Bounds Format

```
+--------------------------------------------------------------------+
|          TABLE 3: SENSITIVITY TO HIDDEN BIAS (ROSENBAUM BOUNDS)    |
+--------------------------------------------------------------------+
|                                                                    |
| Gamma  | Effect Lower | Effect Upper | P-value (Upper Bound)       |
+--------------------------------------------------------------------+
| 1.0    |    2.34      |    2.34      |    < 0.001                  |
| 1.25   |    1.89      |    2.79      |    < 0.001                  |
| 1.5    |    1.52      |    3.16      |      0.002                  |
| 1.75   |    1.18      |    3.50      |      0.012                  |
| 2.0    |    0.87      |    3.81      |      0.038                  |
| 2.25   |    0.58      |    4.10      |      0.082                  |
| 2.5    |    0.31      |    4.37      |      0.142                  |
+--------------------------------------------------------------------+
|                                                                    |
| Critical Gamma: 2.25                                               |
| Interpretation: Results robust to unobserved confounders that      |
|                 could change treatment odds by up to 2.25x         |
+--------------------------------------------------------------------+
```

---

## 7. Sample Write-Up Template

### Methods Section

```
METHODS

Propensity Score Estimation
---------------------------
We estimated propensity scores using logistic regression with the
following covariates: [list all covariates]. The propensity score
model achieved an AUC of [X.XX], indicating [good/moderate/poor]
discrimination between treated and control groups.

Matching Procedure
------------------
We employed [matching method] matching with [parameters]. Each
treated unit was matched to [N] control unit(s) [with/without]
replacement. We used a caliper of [X] standard deviations of the
propensity score to ensure match quality. Of the [N] treated units,
[N] (XX%) were successfully matched.

Balance Assessment
------------------
We assessed covariate balance using standardized mean differences
(SMD) and variance ratios. Balance was considered adequate when
|SMD| < 0.1 for all covariates.

Effect Estimation
-----------------
We estimated the Average Treatment Effect on the Treated (ATT) as
the mean difference in outcomes between treated and matched control
units. Standard errors were estimated using [bootstrap/analytical]
methods.

Sensitivity Analysis
--------------------
We conducted Rosenbaum bounds sensitivity analysis to assess
robustness to potential unobserved confounding.
```

### Results Section

```
RESULTS

Sample Characteristics
----------------------
The study included [N] treated and [N] control observations.
Before matching, treated and control groups differed substantially
on [key variables] (Table 1, Figure 1).

Propensity Score Distribution
-----------------------------
Propensity scores ranged from [X] to [X] for treated and [X] to [X]
for controls, with [X]% of observations falling within the common
support region.

Covariate Balance
-----------------
After matching, all covariates achieved balance with |SMD| < 0.1
(Table 1, Figure 2). The mean absolute SMD decreased from [X.XX]
before matching to [X.XX] after matching.

Treatment Effect
----------------
The estimated ATT was [X.XX] (95% CI: [X.XX, X.XX], p = [X.XXX]),
suggesting that [interpretation]. Results were robust across
alternative specifications (Table 2).

Sensitivity Analysis
--------------------
Rosenbaum bounds analysis indicated that results remained statistically
significant for hidden bias up to Gamma = [X.XX] (Table 3). An
unobserved confounder would need to change the odds of treatment by
[X.XX]-fold to fully explain the observed effect.
```

---

## 8. Figure Specifications

### 8.1 PS Distribution Figure

| Element | Specification |
|---------|---------------|
| Figure size | 8 x 6 inches |
| DPI | 300 |
| Font | Arial or Helvetica |
| Font size | 12pt axes labels |
| Colors | Blue (control), Red (treated) |
| Transparency | alpha = 0.6 for histograms |

### 8.2 Love Plot Figure

| Element | Specification |
|---------|---------------|
| Figure size | 10 x 8 inches |
| DPI | 300 |
| Markers | Circle (before), Square (after) |
| Marker size | 100 |
| Colors | Red (before), Blue (after) |
| Grid | Light gray, x-axis only |

---

## 9. Common Reporting Mistakes

### 9.1 What NOT to Do

1. **Reporting p-values from t-tests instead of SMD**
   - t-tests are sample-size dependent
   - Use SMD for balance assessment

2. **Omitting unmatched observations**
   - Always report how many treated units went unmatched
   - Discuss implications for generalizability

3. **Claiming causality without addressing assumptions**
   - Explicitly state CIA assumption
   - Discuss potential violations

4. **Not reporting sensitivity analysis**
   - Rosenbaum bounds or similar are essential
   - Discuss robustness to unobserved confounding

5. **Using matched sample size for inference**
   - Report both original and matched sample sizes
   - Use appropriate SE estimation methods

---

## References

- Austin, P. C. (2011). An Introduction to Propensity Score Methods for Reducing the Effects of Confounding in Observational Studies. *Multivariate Behavioral Research*, 46(3), 399-424.
- von Elm, E., et al. (2007). The Strengthening the Reporting of Observational Studies in Epidemiology (STROBE) Statement. *Lancet*, 370, 1453-57.
- Ali, M. S., et al. (2015). Reporting of Covariate Selection and Balance Assessment in Propensity Score Analysis. *Journal of Clinical Epidemiology*, 68(2), 122-131.
