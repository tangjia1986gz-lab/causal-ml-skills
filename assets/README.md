# Causal ML Skills - Shared Assets

This directory contains publication-ready templates for causal inference analysis.

## Directory Structure

```
assets/
├── latex/                    # LaTeX table and figure templates
│   ├── common_preamble.tex   # Shared preamble (colors, commands)
│   ├── regression_table.tex  # General regression results
│   ├── summary_stats.tex     # Descriptive statistics
│   ├── coef_plot.tex         # Coefficient plots
│   ├── event_study.tex       # Event study figures
│   ├── balance_table.tex     # PSM balance tables
│   ├── event_study_table.tex # DID event study coefficients
│   ├── first_stage_table.tex # IV first stage diagnostics
│   └── heterogeneity_table.tex # CATE heterogeneity results
│
├── markdown/                 # Report and documentation templates
│   ├── analysis_report.md    # Generic causal analysis report
│   ├── replication_readme.md # Replication package README
│   ├── robustness_appendix.md # Robustness checks appendix
│   └── data_dictionary.md    # Variable dictionary
│
└── README.md                 # This file
```

## Usage by Skill

### DID (estimator-did)
- `event_study.tex` - Event study plots
- `event_study_table.tex` - Coefficient tables
- `regression_table.tex` - Main results

### RD (estimator-rd)
- `coef_plot.tex` - RD coefficient plots
- `regression_table.tex` - Main results

### IV (estimator-iv)
- `first_stage_table.tex` - First stage diagnostics
- `regression_table.tex` - 2SLS results

### PSM (estimator-psm)
- `balance_table.tex` - Balance before/after matching
- `summary_stats.tex` - Descriptive statistics

### Causal Forest / DDML
- `heterogeneity_table.tex` - CATE by subgroup
- `coef_plot.tex` - Variable importance

### Paper Replication
- `analysis_report.md` - Full analysis report
- `replication_readme.md` - Replication package
- `data_dictionary.md` - Variable documentation
- `robustness_appendix.md` - Robustness checks

## Template Conventions

### Placeholders
All templates use `{{PLACEHOLDER}}` syntax for values to be replaced:
- `{{TITLE}}` - Table/figure title
- `{{LABEL}}` - LaTeX label
- `{{COEF_*}}` - Coefficients
- `{{SE_*}}` - Standard errors
- `{{N_*}}` - Sample sizes

### Style
- AER/NBER publication standards
- booktabs for tables
- siunitx for number alignment
- pgfplots for figures

## Programmatic Use

### Python

```python
def fill_template(template_path, values):
    """Replace placeholders in template with actual values."""
    with open(template_path, 'r') as f:
        content = f.read()

    for key, value in values.items():
        content = content.replace(f'{{{{{key}}}}}', str(value))

    return content

# Example
values = {
    'TITLE': 'Treatment Effects',
    'COEF_1': '0.234',
    'SE_1': '0.045',
    'N_1': '10000'
}
filled = fill_template('latex/regression_table.tex', values)
```

### R

```r
fill_template <- function(template_path, values) {
  content <- readLines(template_path, warn = FALSE)
  content <- paste(content, collapse = "\n")

  for (key in names(values)) {
    pattern <- paste0("\\{\\{", key, "\\}\\}")
    content <- gsub(pattern, values[[key]], content)
  }

  return(content)
}
```

## Adding New Templates

1. Create template in appropriate directory (latex/ or markdown/)
2. Use `{{PLACEHOLDER}}` syntax for all dynamic values
3. Include comprehensive usage notes at end of file
4. Add to this README
5. Update skill-specific asset references if applicable

## Version

Template Suite v1.0.0
Compatible with causal-ml-skills v1.0.0
