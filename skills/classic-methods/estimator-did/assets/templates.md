# DID Skill - Template Reference

## Shared Templates

The following templates from `assets/` are relevant for Difference-in-Differences analysis:

### LaTeX Tables and Figures
| Template | Location | Purpose |
|----------|----------|---------|
| `event_study.tex` | `assets/latex/` | Event study plots with confidence intervals |
| `event_study_table.tex` | `assets/latex/` | Event study coefficient tables |
| `regression_table.tex` | `assets/latex/` | Main DID regression results |
| `summary_stats.tex` | `assets/latex/` | Descriptive statistics by treatment group |

### Markdown Reports
| Template | Location | Purpose |
|----------|----------|---------|
| `analysis_report.md` | `assets/markdown/` | Full analysis report |
| `robustness_appendix.md` | `assets/markdown/` | Pre-trends, placebo tests |

## Key Sections in Templates

### event_study.tex
- **Template 1**: Basic event study with shaded confidence intervals
- **Template 2**: Error bar style
- **Template 3**: Multiple groups (heterogeneous effects)
- **Template 4**: Panel event study (multiple outcomes)

### event_study_table.tex
- **Template 1**: Basic event study coefficients
- **Template 2**: Multiple outcomes comparison
- **Template 3**: Staggered DID with robust estimators (TWFE, CS, SA, BJS)
- **Template 4**: Compact with binned endpoints

### DID-Specific Placeholders
```
{{COEF_T*}}           - Coefficient at time T relative to treatment
{{SE_T*}}             - Standard error at time T
{{CI_LO_T*}}          - Lower confidence interval bound
{{CI_HI_T*}}          - Upper confidence interval bound
{{F_PRETREND}}        - Pre-trends F-statistic
{{P_PRETREND}}        - Pre-trends p-value
{{ATT_*}}             - Average treatment effect on treated
{{N_UNITS}}           - Number of units
{{N_PERIODS}}         - Number of time periods
```

## Staggered DID Considerations

For staggered adoption designs, use the estimator comparison table (Template 3):
- TWFE (baseline, potentially biased)
- Callaway-Sant'Anna (2021)
- Sun-Abraham (2021)
- Borusyak et al. (2024)

## Usage

```python
# Access shared assets
from pathlib import Path

ASSETS_DIR = Path(__file__).parent.parent.parent.parent / 'assets'
LATEX_DIR = ASSETS_DIR / 'latex'
MD_DIR = ASSETS_DIR / 'markdown'

# Load event study templates
event_study_figure = (LATEX_DIR / 'event_study.tex').read_text()
event_study_table = (LATEX_DIR / 'event_study_table.tex').read_text()
```
