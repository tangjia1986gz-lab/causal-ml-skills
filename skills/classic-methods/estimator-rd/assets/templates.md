# RD Skill - Template Reference

## Shared Templates

The following templates from `assets/` are relevant for Regression Discontinuity analysis:

### LaTeX Tables and Figures
| Template | Location | Purpose |
|----------|----------|---------|
| `regression_table.tex` | `assets/latex/` | RD estimates with different bandwidths |
| `coef_plot.tex` | `assets/latex/` | RD coefficient plots |
| `summary_stats.tex` | `assets/latex/` | Descriptive statistics around cutoff |

### Markdown Reports
| Template | Location | Purpose |
|----------|----------|---------|
| `analysis_report.md` | `assets/markdown/` | Full analysis report |
| `robustness_appendix.md` | `assets/markdown/` | Bandwidth sensitivity, manipulation tests |

## RD-Specific Considerations

### Bandwidth Sensitivity Table
Use `robustness_appendix.md` Section 7.3 for:
- Optimal bandwidth (MSE-optimal, CER-optimal)
- Bandwidth multipliers (0.5x, 0.75x, 1x, 1.5x, 2x)
- Polynomial order sensitivity

### Manipulation Tests
Report in robustness appendix:
- McCrary density test
- Cattaneo-Jansson-Ma density test

### Covariate Continuity
Use `summary_stats.tex` Template 2 (Balance Table) adapted for:
- Just below cutoff vs. just above cutoff comparison
- Bandwidth-restricted sample

## RD-Specific Placeholders (Custom)
```
{{BANDWIDTH}}         - Selected bandwidth
{{CUTOFF}}            - Discontinuity cutoff value
{{EST_LOCAL}}         - Local linear estimate
{{EST_QUAD}}          - Local quadratic estimate
{{MCCRARY_STAT}}      - McCrary test statistic
{{MCCRARY_P}}         - McCrary p-value
{{N_LEFT}}            - Observations below cutoff
{{N_RIGHT}}           - Observations above cutoff
```

## RD Figure Template

For RD plots, use a custom adaptation of `coef_plot.tex` or create RD-specific visualizations:

```latex
% RD Plot with local polynomial fit
\begin{tikzpicture}
\begin{axis}[
    xlabel={Running Variable},
    ylabel={Outcome},
    xmin={{XMIN}}, xmax={{XMAX}},
    extra x ticks={{{CUTOFF}}},
    extra x tick style={grid=major, grid style={dashed, red}},
]
% Scatter points
\addplot[only marks, mark=o, mark size=1pt, gray]
    table[x=running, y=outcome]{data_left.csv};
\addplot[only marks, mark=o, mark size=1pt, gray]
    table[x=running, y=outcome]{data_right.csv};
% Local polynomial fits
\addplot[coefblue, line width=1.5pt, domain={{XMIN}}:{{CUTOFF}}]
    {{{FIT_LEFT}}};
\addplot[coefblue, line width=1.5pt, domain={{CUTOFF}}:{{XMAX}}]
    {{{FIT_RIGHT}}};
\end{axis}
\end{tikzpicture}
```

## Usage

```python
# Access shared assets
from pathlib import Path

ASSETS_DIR = Path(__file__).parent.parent.parent.parent / 'assets'
LATEX_DIR = ASSETS_DIR / 'latex'
MD_DIR = ASSETS_DIR / 'markdown'

# Load regression table template
regression_table = (LATEX_DIR / 'regression_table.tex').read_text()
```
