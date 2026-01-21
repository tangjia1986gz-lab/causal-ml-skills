# PSM Skill - Template Reference

## Shared Templates

The following templates from `assets/` are relevant for PSM analysis:

### LaTeX Tables
| Template | Location | Purpose |
|----------|----------|---------|
| `balance_table.tex` | `assets/latex/` | Before/after matching balance tables |
| `summary_stats.tex` | `assets/latex/` | Descriptive statistics |
| `regression_table.tex` | `assets/latex/` | Treatment effect estimates |
| `coef_plot.tex` | `assets/latex/` | Love plots for balance visualization |

### Markdown Reports
| Template | Location | Purpose |
|----------|----------|---------|
| `analysis_report.md` | `assets/markdown/` | Full analysis report |
| `robustness_appendix.md` | `assets/markdown/` | Sensitivity to matching methods |

## Key Sections in Templates

### balance_table.tex
- **Template 1**: Comprehensive before/after matching table
- **Template 2**: Matching method comparison
- **Template 3**: Love plot (visual balance)
- **Template 4**: Propensity score distribution

### PSM-Specific Placeholders
```
{{STDDIFF_*_BEFORE}}  - Standardized difference before matching
{{STDDIFF_*_AFTER}}   - Standardized difference after matching
{{VARRATIO_*}}        - Variance ratio (treated/control)
{{RUBIN_B_*}}         - Rubin's B statistic
{{RUBIN_R_*}}         - Rubin's R statistic
{{N_TREATED_*}}       - Number of treated units
{{N_CONTROL_*}}       - Number of control units
```

## Usage

```python
# Access shared assets
from pathlib import Path

ASSETS_DIR = Path(__file__).parent.parent.parent.parent / 'assets'
LATEX_DIR = ASSETS_DIR / 'latex'
MD_DIR = ASSETS_DIR / 'markdown'

# Load balance table template
balance_template = (LATEX_DIR / 'balance_table.tex').read_text()
```
