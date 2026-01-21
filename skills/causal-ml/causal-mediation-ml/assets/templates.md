# Causal Mediation ML Skill - Template Reference

## Shared Templates

The following templates from `assets/` are relevant for causal mediation analysis:

### LaTeX Tables
| Template | Location | Purpose |
|----------|----------|---------|
| `regression_table.tex` | `assets/latex/` | Total, direct, and indirect effects |
| `heterogeneity_table.tex` | `assets/latex/` | Heterogeneous mediation effects |
| `coef_plot.tex` | `assets/latex/` | Effect decomposition visualization |

### Markdown Reports
| Template | Location | Purpose |
|----------|----------|---------|
| `analysis_report.md` | `assets/markdown/` | Full analysis report |
| `robustness_appendix.md` | `assets/markdown/` | Sensitivity analysis |

## Mediation-Specific Table

Create a custom table for effect decomposition:

### Effect Decomposition Table

| Effect | Estimate | SE | 95% CI | % of Total |
|--------|----------|-----|--------|------------|
| Total Effect (ATE) | {{ATE}} | {{SE_ATE}} | [{{CI_ATE}}] | 100% |
| Average Direct Effect (ADE) | {{ADE}} | {{SE_ADE}} | [{{CI_ADE}}] | {{PCT_ADE}}% |
| Average Causal Mediation Effect (ACME) | {{ACME}} | {{SE_ACME}} | [{{CI_ACME}}] | {{PCT_ACME}}% |

### By Treatment Status

| Effect | Treated (d=1) | Control (d=0) | Difference |
|--------|---------------|---------------|------------|
| Direct Effect | {{ADE_1}} | {{ADE_0}} | {{ADE_DIFF}} |
| Indirect Effect | {{ACME_1}} | {{ACME_0}} | {{ACME_DIFF}} |

## Mediation-Specific Placeholders
```
{{ATE}}               - Total treatment effect
{{ADE}}               - Average direct effect
{{ADE_0}}             - ADE for control group
{{ADE_1}}             - ADE for treated group
{{ACME}}              - Average causal mediation effect
{{ACME_0}}            - ACME for control group
{{ACME_1}}            - ACME for treated group
{{PCT_MEDIATED}}      - Proportion mediated
{{RHOS}}              - Sensitivity parameter values
```

## Sensitivity Analysis (Imai et al., 2010)

For robustness appendix, include sensitivity to sequential ignorability:

| $\rho$ | ACME | 95% CI | Significance |
|--------|------|--------|--------------|
| -0.3 | {{ACME_R1}} | [{{CI_R1}}] | {{SIG_R1}} |
| -0.2 | {{ACME_R2}} | [{{CI_R2}}] | {{SIG_R2}} |
| -0.1 | {{ACME_R3}} | [{{CI_R3}}] | {{SIG_R3}} |
| 0.0 | {{ACME_R4}} | [{{CI_R4}}] | {{SIG_R4}} |
| 0.1 | {{ACME_R5}} | [{{CI_R5}}] | {{SIG_R5}} |
| 0.2 | {{ACME_R6}} | [{{CI_R6}}] | {{SIG_R6}} |
| 0.3 | {{ACME_R7}} | [{{CI_R7}}] | {{SIG_R7}} |

$\rho^*$ (ACME = 0): {{RHO_STAR}}

## Effect Decomposition Figure

Adapt `coef_plot.tex` for horizontal bar chart:

```latex
% Mediation effect decomposition
\begin{axis}[
    xbar,
    xlabel={Effect Size},
    symbolic y coords={Total, Direct, Indirect},
    ytick=data,
]
\addplot[fill=coefblue] coordinates {
    ({{ATE}}, Total)
    ({{ADE}}, Direct)
    ({{ACME}}, Indirect)
};
% Error bars
\addplot[only marks, mark=none, error bars/.cd, x dir=both, x explicit]
    coordinates {
        ({{ATE}}, Total) +- ({{SE_ATE}}, 0)
        ({{ADE}}, Direct) +- ({{SE_ADE}}, 0)
        ({{ACME}}, Indirect) +- ({{SE_ACME}}, 0)
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

# Load base templates
regression_table = (LATEX_DIR / 'regression_table.tex').read_text()
coef_plot = (LATEX_DIR / 'coef_plot.tex').read_text()
```

## R Integration (mediation package)

```r
library(mediation)

# Fit mediation model
med.fit <- lm(Mediator ~ Treatment + Controls, data)
out.fit <- lm(Outcome ~ Mediator + Treatment + Controls, data)
med.out <- mediate(med.fit, out.fit, treat="Treatment", mediator="Mediator")

# Extract for templates
summary(med.out)
# ACME, ADE, Total Effect, Prop. Mediated
```

## Python Integration

```python
# Using linearmodels or custom implementation
# ACME = E[Y(1,M(1)) - Y(1,M(0))]
# ADE = E[Y(1,M(d)) - Y(0,M(d))]
```
