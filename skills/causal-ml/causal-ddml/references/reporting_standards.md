# DDML Reporting Standards

> Guidelines for presenting Double/Debiased Machine Learning results

## Overview

Proper reporting of DDML results requires:
1. Clear presentation of main estimates
2. Documentation of methodological choices
3. Sensitivity analysis across specifications
4. Appropriate discussion of assumptions

---

## 1. Main Results Table

### Standard Format

```
Table 1: Double/Debiased Machine Learning Estimates

                           (1)           (2)           (3)           (4)
                         Lasso          RF          XGBoost       Ensemble
------------------------------------------------------------------------
Treatment Effect         0.082***      0.079***      0.081***      0.080***
                        (0.008)       (0.009)       (0.008)       (0.007)
                        [0.066,       [0.061,       [0.065,       [0.066,
                         0.098]        0.097]        0.097]        0.094]

------------------------------------------------------------------------
Model                    PLR           PLR           PLR           PLR
ML Learner (Y|X)        Lasso         RF           XGBoost        Stack
ML Learner (D|X)        Lasso         RF           XGBoost        Stack
Cross-fitting folds      5             5             5             5
Repetitions              10            10            10            10
------------------------------------------------------------------------
Observations           15,000        15,000        15,000        15,000
Controls                 50            50            50            50
------------------------------------------------------------------------

Notes: Standard errors in parentheses. 95% confidence intervals in brackets.
*** p<0.01, ** p<0.05, * p<0.1. All specifications use 5-fold cross-fitting
with 10 repetitions. Controls include demographic variables, regional fixed
effects, and year fixed effects.
```

### LaTeX Code

```latex
\begin{table}[htbp]
\centering
\caption{Double/Debiased Machine Learning Estimates}
\label{tab:ddml_results}
\begin{tabular}{l*{4}{c}}
\toprule
& (1) & (2) & (3) & (4) \\
& Lasso & RF & XGBoost & Ensemble \\
\midrule
Treatment Effect & 0.082*** & 0.079*** & 0.081*** & 0.080*** \\
& (0.008) & (0.009) & (0.008) & (0.007) \\
& [0.066, 0.098] & [0.061, 0.097] & [0.065, 0.097] & [0.066, 0.094] \\
\midrule
Model & PLR & PLR & PLR & PLR \\
ML Learner (Y$|$X) & Lasso & RF & XGBoost & Stack \\
ML Learner (D$|$X) & Lasso & RF & XGBoost & Stack \\
Cross-fitting folds & 5 & 5 & 5 & 5 \\
\midrule
Observations & 15,000 & 15,000 & 15,000 & 15,000 \\
Controls & 50 & 50 & 50 & 50 \\
\bottomrule
\end{tabular}
\begin{tablenotes}
\small
\item Notes: Standard errors in parentheses. 95\% CI in brackets.
*** p$<$0.01, ** p$<$0.05, * p$<$0.1.
\end{tablenotes}
\end{table}
```

---

## 2. Required Elements

### 2.1 Estimation Details

Always report:

| Element | Description | Example |
|---------|-------------|---------|
| **Model type** | PLR, IRM, PLIV, IIVM | "PLR (Partially Linear)" |
| **Nuisance learners** | ML methods used | "Lasso for E[Y\|X], Logistic Lasso for P(D=1\|X)" |
| **Cross-fitting** | K and repetitions | "5-fold cross-fitting, 10 repetitions" |
| **Sample size** | n observations | "n = 15,000" |
| **Number of controls** | Dimension of X | "p = 50 control variables" |

### 2.2 Inference

| Element | What to Report |
|---------|----------------|
| Point estimate | Treatment effect coefficient |
| Standard error | In parentheses below estimate |
| Confidence interval | 95% CI in brackets |
| P-value or stars | Statistical significance |

### 2.3 Nuisance Model Performance

Report in supplementary materials:

```
Table A1: Nuisance Model Performance

                    Outcome Model      Treatment Model
                    E[Y|X]             E[D|X]
---------------------------------------------------------
R-squared (CV)      0.42               0.31
MSE (CV)            1.23               0.18
Best learner        XGBoost            Random Forest
---------------------------------------------------------
```

---

## 3. Sensitivity Analysis Reporting

### 3.1 Across Learners

```
Figure 1: Sensitivity to ML Learner Choice

[Effect estimates with 95% CI for each learner]

Lasso:     ====|====  0.082
RF:        =====|=====  0.079
XGBoost:   ====|====  0.081
LightGBM:  ====|====  0.080
Ensemble:  ===|===  0.080
           |-----+-----|
           0.06  0.08  0.10

Notes: Effect estimates robust across learner specifications.
Coefficient of variation: 1.8%
```

### 3.2 Across Specifications

```
Table 2: Specification Sensitivity

                    (1)        (2)        (3)        (4)
                  Baseline   Add FE    Drop Age   Trim 5%
------------------------------------------------------------
Treatment Effect   0.082***   0.079***   0.084***   0.081***
                  (0.008)    (0.009)    (0.009)    (0.008)
------------------------------------------------------------
```

### 3.3 Reporting Summary Statistics

```python
def sensitivity_summary(results_dict):
    """
    Generate sensitivity summary for reporting.
    """
    effects = [r['effect'] for r in results_dict.values()]

    summary = {
        'median_effect': np.median(effects),
        'range': f"[{min(effects):.4f}, {max(effects):.4f}]",
        'cv': np.std(effects) / abs(np.mean(effects)) * 100,
        'all_significant': all(r['p_value'] < 0.05 for r in results_dict.values()),
        'all_same_sign': all(e > 0 for e in effects) or all(e < 0 for e in effects)
    }

    interpretation = []
    if summary['cv'] < 5:
        interpretation.append("highly robust")
    elif summary['cv'] < 10:
        interpretation.append("robust")
    elif summary['cv'] < 20:
        interpretation.append("moderately sensitive")
    else:
        interpretation.append("sensitive")

    summary['interpretation'] = " ".join(interpretation)

    return summary
```

---

## 4. Confidence Intervals

### 4.1 Standard CI

```python
def compute_ci(effect, se, alpha=0.05):
    """Standard normal-based CI."""
    z = stats.norm.ppf(1 - alpha/2)
    return (effect - z * se, effect + z * se)
```

### 4.2 Bootstrap CI

For more robust inference, especially with small samples:

```python
def bootstrap_ci(dml_model, n_bootstrap=500, method='normal'):
    """
    Bootstrap confidence interval for DDML.

    Methods:
    - 'normal': Normal approximation
    - 'percentile': Percentile method
    - 'basic': Basic bootstrap
    """
    dml_model.bootstrap(method='wild', n_rep=n_bootstrap)
    ci = dml_model.confint(joint=False, level=0.95)
    return ci
```

### 4.3 Reporting CIs

```
Treatment Effect: 0.082
Standard Error: 0.008
95% CI (Normal): [0.066, 0.098]
95% CI (Bootstrap): [0.065, 0.099]
```

---

## 5. Writing Guidelines

### 5.1 Methods Section Template

```
We estimate the treatment effect using Double/Debiased Machine Learning
(Chernozhukov et al., 2018), which combines flexible machine learning
methods for nuisance parameter estimation with valid statistical inference.

Specifically, we estimate a Partially Linear Regression (PLR) model:
    Y = D * theta + g(X) + epsilon
where Y is the outcome, D is the treatment, and X is a vector of p = [N]
control variables including [list key controls].

For the nuisance functions E[Y|X] and E[D|X], we use [Lasso/Random Forest/
XGBoost] selected via cross-validation. We employ 5-fold cross-fitting
with [N] repetitions to ensure valid inference.

To assess robustness, we compare estimates across multiple ML learner
specifications (Lasso, Random Forest, XGBoost) and report all results.
```

### 5.2 Results Section Template

```
Table [N] presents our main DDML estimates. The treatment effect is
[X.XX] (SE = [X.XX]), statistically significant at the [1%/5%] level.
The 95% confidence interval is [[X.XX], [X.XX]].

Results are robust across ML specifications: estimates range from
[X.XX] to [X.XX] with a coefficient of variation of [X.X]%.
[All/Most] specifications yield statistically significant effects
with the same sign.

The nuisance models achieve R-squared of [X.XX] for the outcome
model and [X.XX] for the treatment model, indicating [adequate/strong]
predictive power for the confounding adjustment.
```

### 5.3 Discussion of Assumptions

```
Our DDML estimates rely on the selection-on-observables assumption:
conditional on the observed controls X, treatment assignment is as
good as random. While this assumption is fundamentally untestable,
we argue it is plausible in our setting because:

1. [Rich controls]: We include comprehensive controls for [factors]
2. [Institutional knowledge]: [Explain why selection is likely on X]
3. [Robustness]: Results are stable across specifications with
   different control sets

We also verify the overlap assumption by examining the propensity
score distribution, finding [describe overlap findings].
```

---

## 6. Supplementary Materials

### 6.1 Detailed Diagnostics

```
Appendix A: DDML Diagnostics

A1. Nuisance Model Cross-Validation Performance
    [Table of CV scores for each learner]

A2. Propensity Score Distribution
    [Histogram by treatment group]
    [Summary statistics]
    [N observations trimmed]

A3. Cross-Fitting Stability
    [Variation across folds]
    [Variation across repetitions]

A4. Residual Diagnostics
    [Residual plots]
    [Test statistics]
```

### 6.2 Code and Replication

```
Appendix B: Replication Code

All analyses conducted using:
- Python [version]
- DoubleML [version]
- scikit-learn [version]

Code available at: [repository URL]
```

---

## 7. Visualization Standards

### 7.1 Effect Comparison Plot

```python
def plot_learner_comparison(results_dict, title="DDML Estimates by Learner"):
    """
    Create forest plot of estimates across learners.
    """
    import matplotlib.pyplot as plt

    learners = list(results_dict.keys())
    effects = [r['effect'] for r in results_dict.values()]
    lower = [r['ci_lower'] for r in results_dict.values()]
    upper = [r['ci_upper'] for r in results_dict.values()]

    fig, ax = plt.subplots(figsize=(8, 5))

    y_pos = range(len(learners))
    ax.errorbar(effects, y_pos,
                xerr=[np.array(effects)-np.array(lower),
                      np.array(upper)-np.array(effects)],
                fmt='o', capsize=5, capthick=2)

    ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(learners)
    ax.set_xlabel('Treatment Effect')
    ax.set_title(title)

    plt.tight_layout()
    return fig
```

### 7.2 Propensity Distribution Plot

```python
def plot_propensity_overlap(ps, d, title="Propensity Score Distribution"):
    """
    Plot propensity score distributions by treatment group.
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.hist(ps[d == 0], bins=50, alpha=0.5, label='Control', density=True)
    ax.hist(ps[d == 1], bins=50, alpha=0.5, label='Treated', density=True)

    ax.set_xlabel('Propensity Score')
    ax.set_ylabel('Density')
    ax.set_title(title)
    ax.legend()

    # Add overlap statistics
    overlap_pct = np.mean((ps > 0.1) & (ps < 0.9)) * 100
    ax.text(0.95, 0.95, f'Overlap region: {overlap_pct:.1f}%',
            transform=ax.transAxes, ha='right', va='top')

    plt.tight_layout()
    return fig
```

---

## 8. Checklist for Reporting

### Methods

- [ ] Specify DDML model type (PLR, IRM, etc.)
- [ ] List ML learners used for nuisance functions
- [ ] Report cross-fitting parameters (K folds, n repetitions)
- [ ] Describe control variables included
- [ ] Cite Chernozhukov et al. (2018)

### Results

- [ ] Report point estimate with standard error
- [ ] Include 95% confidence interval
- [ ] Show significance (p-value or stars)
- [ ] Compare multiple learner specifications
- [ ] Report sensitivity summary (range, CV)

### Diagnostics

- [ ] Nuisance model performance (R-squared, MSE)
- [ ] Propensity score overlap (for IRM)
- [ ] Cross-fitting stability

### Discussion

- [ ] Discuss unconfoundedness assumption
- [ ] Address potential violations
- [ ] Interpret effect size meaningfully

---

## References

1. Chernozhukov, V., et al. (2018). Double/Debiased Machine Learning for Treatment and Structural Parameters. *The Econometrics Journal*, 21(1), C1-C68.

2. American Economic Association (2024). Data and Code Availability Policy.

3. JASA Guidelines for Statistical Reporting.
