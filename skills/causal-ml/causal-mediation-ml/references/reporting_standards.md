# Reporting Standards for Causal Mediation Analysis

> **Reference Document** | Part of `causal-mediation-ml` skill
> **Version**: 1.0.0

## Overview

This document provides standards and templates for reporting causal mediation analysis results in academic publications and applied research reports.

---

## Effect Tables

### Standard Decomposition Table

**LaTeX Template**:

```latex
\begin{table}[htbp]
\centering
\caption{Causal Mediation Analysis: Effect Decomposition}
\label{tab:mediation}
\begin{tabular}{lcccc}
\toprule
Effect & Estimate & Std. Err. & 95\% CI & $p$-value \\
\midrule
Total Effect & 0.150*** & (0.035) & [0.081, 0.219] & $<$0.001 \\
Direct Effect (ADE) & 0.090*** & (0.028) & [0.035, 0.145] & 0.001 \\
Indirect Effect (ACME) & 0.060*** & (0.018) & [0.025, 0.095] & 0.001 \\
\midrule
Proportion Mediated & 40.0\% & (7.5\%) & [25.5\%, 54.5\%] & --- \\
\bottomrule
\multicolumn{5}{l}{\footnotesize Notes: *** $p<0.01$, ** $p<0.05$, * $p<0.1$} \\
\multicolumn{5}{l}{\footnotesize 95\% CIs from 1,000 bootstrap replications.} \\
\end{tabular}
\end{table}
```

**Python Generator**:

```python
def generate_latex_table(results: dict, caption: str = None) -> str:
    """
    Generate LaTeX table for mediation results.
    """
    from scipy import stats

    def stars(pval):
        if pval < 0.01: return '***'
        elif pval < 0.05: return '**'
        elif pval < 0.1: return '*'
        return ''

    def format_ci(lower, upper):
        return f"[{lower:.3f}, {upper:.3f}]"

    total = results['total_effect']
    total_se = results.get('total_se', 0)
    z_total = total / total_se if total_se > 0 else 0
    p_total = 2 * (1 - stats.norm.cdf(abs(z_total)))

    ade = results['ade']
    ade_se = results['ade_se']
    p_ade = results.get('ade_pvalue', 2 * (1 - stats.norm.cdf(abs(ade/ade_se))))

    acme = results['acme']
    acme_se = results['acme_se']
    p_acme = results.get('acme_pvalue', 2 * (1 - stats.norm.cdf(abs(acme/acme_se))))

    prop = results.get('prop_mediated', acme/total if total != 0 else float('nan'))
    prop_se = results.get('prop_mediated_se', 0)

    latex = r"""
\begin{table}[htbp]
\centering
\caption{%s}
\label{tab:mediation}
\begin{tabular}{lcccc}
\toprule
Effect & Estimate & Std. Err. & 95\%% CI & $p$-value \\
\midrule
Total Effect & %.3f%s & (%.3f) & %s & %.3f \\
Direct Effect (ADE) & %.3f%s & (%.3f) & %s & %.3f \\
Indirect Effect (ACME) & %.3f%s & (%.3f) & %s & %.3f \\
\midrule
Proportion Mediated & %.1f\%% & (%.1f\%%) & --- & --- \\
\bottomrule
\multicolumn{5}{l}{\footnotesize Notes: *** $p<0.01$, ** $p<0.05$, * $p<0.1$} \\
\end{tabular}
\end{table}
""" % (
        caption or "Causal Mediation Analysis",
        total, stars(p_total), total_se, format_ci(total-1.96*total_se, total+1.96*total_se), p_total,
        ade, stars(p_ade), ade_se, format_ci(results['ade_ci_lower'], results['ade_ci_upper']), p_ade,
        acme, stars(p_acme), acme_se, format_ci(results['acme_ci_lower'], results['acme_ci_upper']), p_acme,
        prop * 100, prop_se * 100
    )

    return latex
```

### Multi-Method Comparison Table

```latex
\begin{table}[htbp]
\centering
\caption{Mediation Analysis: Method Comparison}
\label{tab:method_comparison}
\begin{tabular}{lccc}
\toprule
 & (1) Baron-Kenny & (2) ML-Lasso & (3) ML-RF \\
\midrule
\textbf{ACME} & 0.060*** & 0.058*** & 0.062*** \\
 & (0.018) & (0.016) & (0.019) \\
 & [0.025, 0.095] & [0.027, 0.089] & [0.025, 0.099] \\
\\
\textbf{ADE} & 0.090*** & 0.092*** & 0.088*** \\
 & (0.028) & (0.025) & (0.027) \\
\\
Prop. Mediated & 40.0\% & 38.7\% & 41.3\% \\
\midrule
$N$ & 1,500 & 1,500 & 1,500 \\
Controls & Yes & Yes & Yes \\
ML Learner & OLS & Lasso & Random Forest \\
\bottomrule
\end{tabular}
\end{table}
```

---

## Mediation Diagrams

### Standard Pathway Diagram

```python
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

def create_mediation_diagram(
    treatment_name: str,
    mediator_name: str,
    outcome_name: str,
    alpha: float,
    beta_m: float,
    direct: float,
    alpha_se: float = None,
    beta_m_se: float = None,
    direct_se: float = None,
    acme: float = None,
    save_path: str = None,
    figsize: tuple = (10, 6)
) -> plt.Figure:
    """
    Create publication-quality mediation pathway diagram.
    """
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis('off')

    # Box style
    box_style = dict(
        boxstyle='round,pad=0.4',
        facecolor='white',
        edgecolor='black',
        linewidth=2
    )

    # Draw variable boxes
    # Treatment (left)
    ax.text(1.5, 3, treatment_name, ha='center', va='center',
            fontsize=11, fontweight='bold', bbox=box_style)

    # Mediator (top center)
    ax.text(5, 5, mediator_name, ha='center', va='center',
            fontsize=11, fontweight='bold', bbox=box_style)

    # Outcome (right)
    ax.text(8.5, 3, outcome_name, ha='center', va='center',
            fontsize=11, fontweight='bold', bbox=box_style)

    # Arrow style
    arrow_props = dict(
        arrowstyle='-|>',
        color='black',
        linewidth=1.5,
        connectionstyle='arc3,rad=0'
    )

    # Treatment -> Mediator (a path)
    ax.annotate('', xy=(4.2, 4.8), xytext=(2.2, 3.3),
                arrowprops=arrow_props)
    a_label = f"a = {alpha:.3f}"
    if alpha_se:
        a_label += f"\n({alpha_se:.3f})"
    ax.text(2.8, 4.3, a_label, fontsize=10, ha='center', va='bottom')

    # Mediator -> Outcome (b path)
    ax.annotate('', xy=(7.8, 3.3), xytext=(5.8, 4.8),
                arrowprops=arrow_props)
    b_label = f"b = {beta_m:.3f}"
    if beta_m_se:
        b_label += f"\n({beta_m_se:.3f})"
    ax.text(7.2, 4.3, b_label, fontsize=10, ha='center', va='bottom')

    # Direct effect (c' path)
    ax.annotate('', xy=(7.6, 3), xytext=(2.4, 3),
                arrowprops=dict(**arrow_props, color='gray'))
    c_label = f"c' = {direct:.3f}"
    if direct_se:
        c_label += f" ({direct_se:.3f})"
    ax.text(5, 2.5, c_label, fontsize=10, ha='center', color='gray')

    # Effect decomposition box
    if acme is not None:
        total = direct + acme
        prop = acme / total * 100 if total != 0 else float('nan')

        decomp_text = (
            f"Indirect (ACME): {acme:.4f}\n"
            f"Direct (ADE): {direct:.4f}\n"
            f"Total: {total:.4f}\n"
            f"% Mediated: {prop:.1f}%"
        )

        ax.text(5, 0.7, decomp_text, ha='center', va='center',
                fontsize=9, family='monospace',
                bbox=dict(boxstyle='round', facecolor='lightyellow',
                         edgecolor='orange', alpha=0.9))

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight',
                   facecolor='white', edgecolor='none')

    return fig
```

### TikZ Version (LaTeX)

```latex
\begin{tikzpicture}[
    node distance=3cm,
    box/.style={rectangle, draw, minimum width=2cm, minimum height=0.8cm},
    arrow/.style={->, >=latex, thick}
]

% Nodes
\node[box] (T) {Treatment};
\node[box, above right=1.5cm and 2cm of T] (M) {Mediator};
\node[box, right=4cm of T] (Y) {Outcome};

% Arrows
\draw[arrow] (T) -- node[above, sloped] {$a = 0.60$} (M);
\draw[arrow] (M) -- node[above, sloped] {$b = 0.10$} (Y);
\draw[arrow, gray] (T) -- node[below] {$c' = 0.09$} (Y);

% Legend
\node[below=1cm of T, text width=6cm, align=left] {
    Indirect Effect (ACME): $a \times b = 0.06$ \\
    Direct Effect (ADE): $c' = 0.09$ \\
    Total Effect: $0.15$
};

\end{tikzpicture}
```

---

## Sensitivity Analysis Plots

### ACME vs. Rho Plot

```python
def plot_sensitivity_analysis(
    sensitivity_results: dict,
    save_path: str = None,
    figsize: tuple = (10, 6)
) -> plt.Figure:
    """
    Create sensitivity analysis plot showing ACME vs. rho.
    """
    import numpy as np

    fig, ax = plt.subplots(figsize=figsize)

    rho = np.array(sensitivity_results['rho_values'])
    acme = np.array(sensitivity_results['acme_values'])
    ci_lower = np.array(sensitivity_results.get('ci_lower', acme - 0.1))
    ci_upper = np.array(sensitivity_results.get('ci_upper', acme + 0.1))

    # Main line
    ax.plot(rho, acme, 'b-', linewidth=2, label='ACME')

    # Confidence band
    ax.fill_between(rho, ci_lower, ci_upper, alpha=0.2, color='blue',
                    label='95% CI')

    # Reference lines
    ax.axhline(y=0, color='red', linestyle='--', linewidth=1.5,
               label='Zero effect')
    ax.axvline(x=0, color='gray', linestyle=':', linewidth=1)

    # Breakpoint
    breakpoint = sensitivity_results.get('breakpoint')
    if breakpoint is not None and -1 < breakpoint < 1:
        ax.axvline(x=breakpoint, color='orange', linestyle='--', linewidth=2)
        ax.scatter([breakpoint], [0], color='orange', s=100, zorder=5,
                  label=f'Breakpoint (rho = {breakpoint:.2f})')

    ax.set_xlabel('Sensitivity Parameter (rho)', fontsize=12)
    ax.set_ylabel('ACME', fontsize=12)
    ax.set_title('Sensitivity Analysis for Unmeasured Confounding', fontsize=14)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig
```

---

## Results Narrative Templates

### Main Results Paragraph

```python
def generate_results_narrative(
    results: dict,
    treatment_name: str,
    mediator_name: str,
    outcome_name: str
) -> str:
    """
    Generate narrative paragraph for results section.
    """
    total = results['total_effect']
    ade = results['ade']
    acme = results['acme']
    prop = results.get('prop_mediated', acme/total if total != 0 else float('nan'))
    acme_pval = results.get('acme_pvalue', 0.001)
    ade_pval = results.get('ade_pvalue', 0.001)

    acme_sig = "statistically significant" if acme_pval < 0.05 else "not statistically significant"
    ade_sig = "statistically significant" if ade_pval < 0.05 else "not statistically significant"

    narrative = f"""
Table X presents the results of our causal mediation analysis examining
whether the effect of {treatment_name} on {outcome_name} operates through
{mediator_name}.

The total effect of {treatment_name} on {outcome_name} is {total:.4f}
(SE = {results.get('total_se', 0):.4f}). We decompose this into direct and
indirect components. The Average Direct Effect (ADE), representing the
effect of {treatment_name} not operating through {mediator_name}, is
{ade:.4f} (SE = {results['ade_se']:.4f}, p = {ade_pval:.3f}), which is
{ade_sig} at the 5% level.

The Average Causal Mediation Effect (ACME), representing the indirect effect
through {mediator_name}, is {acme:.4f} (SE = {results['acme_se']:.4f},
p = {acme_pval:.3f}). This indirect effect is {acme_sig}, indicating that
{mediator_name} {'does' if acme_pval < 0.05 else 'does not'} serve as a
significant mediator of the treatment effect.

The proportion of the total effect mediated through {mediator_name} is
{prop*100:.1f}%, suggesting {'substantial' if prop > 0.3 else 'modest'
if prop > 0.1 else 'limited'} mediation.
    """

    return narrative.strip()
```

### Sensitivity Analysis Paragraph

```python
def generate_sensitivity_narrative(sensitivity_results: dict) -> str:
    """
    Generate narrative for sensitivity analysis.
    """
    breakpoint = sensitivity_results.get('breakpoint', float('nan'))
    robustness = sensitivity_results.get('robustness', 'unknown')
    interpretation = sensitivity_results.get('interpretation', '')

    r2_equiv = breakpoint ** 2 if not np.isnan(breakpoint) else float('nan')

    narrative = f"""
We assess the robustness of our mediation findings to unmeasured confounding
using the sensitivity analysis framework of Imai et al. (2010). The key
sensitivity parameter rho represents the correlation between the error terms
in the mediator and outcome models that would arise from unmeasured confounding.

Under the sequential ignorability assumption, rho = 0. Our sensitivity
analysis examines how the ACME estimate changes as we allow for non-zero
values of rho. The "breakpoint" is the value of rho at which the ACME
becomes statistically indistinguishable from zero.

The estimated breakpoint is rho = {breakpoint:.3f}, corresponding to an
R-squared equivalent of {r2_equiv:.1%}. This means that an unmeasured
confounder would need to explain approximately {r2_equiv:.1%} of the
residual variance in both the mediator and outcome (conditional on
observables) to nullify our mediation finding.

{interpretation}

Based on this analysis, we conclude that our mediation results are
{'robust' if abs(breakpoint) > 0.3 else 'moderately robust' if abs(breakpoint) > 0.15 else 'sensitive'}
to unmeasured confounding.
    """

    return narrative.strip()
```

---

## Reporting Checklist

### Essential Elements

- [ ] **Causal Structure**: DAG or pathway diagram
- [ ] **Identification**: Statement of sequential ignorability
- [ ] **Data**: Sample size, key variable descriptions
- [ ] **Method**: Estimation approach (Baron-Kenny, ML, etc.)
- [ ] **Results Table**: Total, ADE, ACME with SE/CI
- [ ] **Proportion Mediated**: With appropriate caveats
- [ ] **Sensitivity Analysis**: Breakpoint and interpretation
- [ ] **Limitations**: Discuss assumption plausibility

### Optional but Recommended

- [ ] Alternative specifications (robustness)
- [ ] Subgroup analyses (heterogeneity)
- [ ] Multiple mediators (if applicable)
- [ ] Bootstrap confidence intervals
- [ ] Pre-registration reference

### What NOT to Report

- [ ] Proportion mediated without uncertainty
- [ ] Claims of "full mediation"
- [ ] Effects without discussing assumptions
- [ ] Results without sensitivity analysis

---

## Publication Guidelines

### Journal-Specific Requirements

| Journal Type | Key Requirements |
|--------------|------------------|
| Psychology | Prefer simulation-based (Imai et al.) |
| Economics | Emphasize identification, robustness |
| Epidemiology | VanderWeele methods, bounds |
| Management | Clear practical implications |

### Recommended Software Citations

```bibtex
@article{imai2010mediation,
  title={A general approach to causal mediation analysis},
  author={Imai, Kosuke and Keele, Luke and Tingley, Dustin},
  journal={Psychological Methods},
  volume={15},
  number={4},
  pages={309--334},
  year={2010}
}

@article{tingley2014mediation,
  title={mediation: R package for causal mediation analysis},
  author={Tingley, Dustin and Yamamoto, Teppei and Hirose, Kentaro
          and Keele, Luke and Imai, Kosuke},
  journal={Journal of Statistical Software},
  volume={59},
  number={5},
  pages={1--38},
  year={2014}
}
```

---

## References

- American Psychological Association. (2020). *Publication Manual* (7th ed.).
- Imai, K., Keele, L., & Tingley, D. (2010). A General Approach to Causal Mediation Analysis. *Psychological Methods*.
- VanderWeele, T. J. (2015). *Explanation in Causal Inference*. Oxford University Press.
