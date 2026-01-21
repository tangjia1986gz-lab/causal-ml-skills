# Causal Forest Skill - Template Reference

## Shared Templates

The following templates from `assets/` are relevant for Causal Forest / CATE analysis:

### LaTeX Tables
| Template | Location | Purpose |
|----------|----------|---------|
| `heterogeneity_table.tex` | `assets/latex/` | CATE by subgroup, variable importance |
| `coef_plot.tex` | `assets/latex/` | CATE distribution, targeting plots |
| `regression_table.tex` | `assets/latex/` | ATE with ML nuisance estimation |

### Markdown Reports
| Template | Location | Purpose |
|----------|----------|---------|
| `analysis_report.md` | `assets/markdown/` | Full analysis report |
| `robustness_appendix.md` | `assets/markdown/` | Sensitivity to tuning parameters |

## Key Sections in Templates

### heterogeneity_table.tex
- **Template 1**: CATE by subgroups (demographics, income, etc.)
- **Template 2**: Variable importance and Best Linear Projection
- **Template 3**: CATE distribution (quantiles, share positive/negative)
- **Template 4**: Policy targeting (AUTOC, QINI, calibration)
- **Template 5**: Sorted Group Average Treatment Effects (GATES)

### Causal Forest-Specific Placeholders
```
{{CATE_*}}            - Conditional average treatment effect
{{SE_*}}              - Standard error (honest inference)
{{VIMP_*}}            - Variable importance score
{{BLP_*}}             - Best linear projection coefficient
{{AUTOC}}             - Area Under Targeting Operating Characteristic
{{QINI}}              - QINI coefficient
{{SHARE_POS}}         - Share with positive CATE
{{SHARE_NEG}}         - Share with negative CATE
{{VAR_CATE}}          - Variance of CATE
{{P_HET}}             - p-value for heterogeneity test
```

## GATES Analysis

For Sorted Group Average Treatment Effects (Chernozhukov et al., 2020):

```python
# Example GATES structure
gates_results = {
    'group': [1, 2, 3, 4, 5],  # Quintiles by predicted CATE
    'gate': [...],             # Group average treatment effects
    'se': [...],               # Standard errors
    'deviation': [...],        # Deviation from overall ATE
}
```

## Variable Importance Visualization

Adapt `coef_plot.tex` for horizontal bar chart of VIMP scores:

```latex
% Modify coef_plot.tex for variable importance
\begin{axis}[
    xbar,
    xlabel={Variable Importance},
    symbolic y coords={Var1, Var2, Var3, ...},
    ytick=data,
]
\addplot[fill=coefblue] coordinates {
    ({{VIMP_1}}, Var1)
    ({{VIMP_2}}, Var2)
    ...
};
\end{axis}
```

## Usage

```python
# Access shared assets
from pathlib import Path

ASSETS_DIR = Path(__file__).parent.parent.parent.parent / 'assets'
LATEX_DIR = ASSETS_DIR / 'latex'
MD_DIR = ASSETS_DIR / 'markdown'

# Load heterogeneity template
hetero_table = (LATEX_DIR / 'heterogeneity_table.tex').read_text()
```

## R Integration (grf package)

The causal forest analysis typically uses the `grf` package in R. Template outputs should match:

```r
library(grf)

cf <- causal_forest(X, Y, W, ...)

# For heterogeneity_table.tex Template 2
vimp <- variable_importance(cf)

# For heterogeneity_table.tex Template 5
gates <- average_treatment_effect(cf, target.sample = "overlap")
```
