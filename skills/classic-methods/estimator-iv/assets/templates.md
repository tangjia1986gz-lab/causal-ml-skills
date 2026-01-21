# IV Skill - Template Reference

## Shared Templates

The following templates from `assets/` are relevant for Instrumental Variables analysis:

### LaTeX Tables
| Template | Location | Purpose |
|----------|----------|---------|
| `first_stage_table.tex` | `assets/latex/` | First stage coefficients and diagnostics |
| `regression_table.tex` | `assets/latex/` | 2SLS/IV regression results |
| `coef_plot.tex` | `assets/latex/` | Coefficient comparisons (OLS vs 2SLS) |

### Markdown Reports
| Template | Location | Purpose |
|----------|----------|---------|
| `analysis_report.md` | `assets/markdown/` | Full analysis report |
| `robustness_appendix.md` | `assets/markdown/` | Weak IV tests, overidentification |

## Key Sections in Templates

### first_stage_table.tex
- **Template 1**: Comprehensive first stage with diagnostics
- **Template 2**: Single instrument with detailed diagnostics
- **Template 3**: Multiple endogenous variables
- **Template 4**: First stage + reduced form + 2SLS comparison
- **Template 5**: Overidentification and validity tests

### IV-Specific Placeholders
```
{{COEF_Z*_*}}         - Instrument coefficient (instrument, column)
{{SE_Z*_*}}           - Standard error
{{F_*}}               - First-stage F-statistic
{{KP_*}}              - Kleibergen-Paap F-statistic
{{PARTIALR2_*}}       - Partial R-squared of instruments
{{SHEA_*}}            - Shea partial R-squared
{{SARGAN}}            - Sargan overidentification statistic
{{HANSEN}}            - Hansen J statistic
{{AR_STAT}}           - Anderson-Rubin statistic
```

## Stock-Yogo Critical Values Reference

| Instruments | Endogenous | 10% Bias | 15% Bias | 20% Bias |
|-------------|------------|----------|----------|----------|
| 1 | 1 | 16.38 | 8.96 | 6.66 |
| 2 | 1 | 19.93 | 11.59 | 8.75 |
| 3 | 1 | 22.30 | 12.83 | 9.54 |
| 3 | 2 | 13.43 | 8.18 | 6.40 |

## Usage

```python
# Access shared assets
from pathlib import Path

ASSETS_DIR = Path(__file__).parent.parent.parent.parent / 'assets'
LATEX_DIR = ASSETS_DIR / 'latex'
MD_DIR = ASSETS_DIR / 'markdown'

# Load first stage template
first_stage = (LATEX_DIR / 'first_stage_table.tex').read_text()
```
