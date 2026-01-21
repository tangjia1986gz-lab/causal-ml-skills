# DDML Skill - Template Reference

## Shared Templates

The following templates from `assets/` are relevant for Double/Debiased Machine Learning:

### LaTeX Tables
| Template | Location | Purpose |
|----------|----------|---------|
| `heterogeneity_table.tex` | `assets/latex/` | CATE heterogeneity (for CATE learners) |
| `regression_table.tex` | `assets/latex/` | ATE/LATE estimates with DML |
| `coef_plot.tex` | `assets/latex/` | Coefficient comparison across learners |

### Markdown Reports
| Template | Location | Purpose |
|----------|----------|---------|
| `analysis_report.md` | `assets/markdown/` | Full analysis report |
| `robustness_appendix.md` | `assets/markdown/` | ML model sensitivity |

## DDML-Specific Considerations

### Model Selection Table

Create a custom table showing results across different ML methods:

| ML Method | ATE Estimate | SE | Cross-Fit Folds |
|-----------|--------------|-----|-----------------|
| Linear/Logistic | {{EST_LINEAR}} | {{SE_LINEAR}} | {{K_LINEAR}} |
| Random Forest | {{EST_RF}} | {{SE_RF}} | {{K_RF}} |
| XGBoost | {{EST_XGB}} | {{SE_XGB}} | {{K_XGB}} |
| Neural Network | {{EST_NN}} | {{SE_NN}} | {{K_NN}} |
| Ensemble | {{EST_ENS}} | {{SE_ENS}} | {{K_ENS}} |

### Nuisance Function Quality

Report first-stage prediction performance:

| Nuisance | ML Method | R² / AUC | Notes |
|----------|-----------|----------|-------|
| $E[Y|X]$ | {{ML_Y}} | {{R2_Y}} | Outcome model |
| $E[D|X]$ | {{ML_D}} | {{AUC_D}} | Propensity score |
| $E[Y|D,X]$ | {{ML_YDX}} | {{R2_YDX}} | Conditional outcome |

### DDML-Specific Placeholders
```
{{ATE_DML}}           - DML average treatment effect
{{SE_DML}}            - Standard error (Neyman orthogonal)
{{K_FOLDS}}           - Number of cross-fitting folds
{{ML_METHOD}}         - ML method for nuisance
{{R2_NUISANCE}}       - Nuisance prediction R²
{{CATE_*}}            - Conditional effects (for CATE learners)
```

## Model Types

### Partially Linear Model (PLR)
$$Y = D\theta_0 + g_0(X) + U$$

### Interactive Regression Model (IRM)
$$Y = g_0(D, X) + U$$

### IIVM (IV with ML)
$$Y = D\theta_0 + g_0(X) + U, \quad D = m_0(Z, X) + V$$

## Usage

```python
# Access shared assets
from pathlib import Path

ASSETS_DIR = Path(__file__).parent.parent.parent.parent / 'assets'
LATEX_DIR = ASSETS_DIR / 'latex'
MD_DIR = ASSETS_DIR / 'markdown'

# Load regression table for DML results
regression_table = (LATEX_DIR / 'regression_table.tex').read_text()

# For heterogeneity analysis
hetero_table = (LATEX_DIR / 'heterogeneity_table.tex').read_text()
```

## Python Integration (doubleml, econml)

```python
from doubleml import DoubleMLPLR
from sklearn.ensemble import RandomForestRegressor

# Fit DML model
ml_g = RandomForestRegressor(n_estimators=500)
ml_m = RandomForestRegressor(n_estimators=500)
dml_plr = DoubleMLPLR(obj_dml_data, ml_g, ml_m)
dml_plr.fit()

# Extract for templates
values = {
    'ATE_DML': f'{dml_plr.coef[0]:.4f}',
    'SE_DML': f'{dml_plr.se[0]:.4f}',
    'CI_LO': f'{dml_plr.confint().iloc[0, 0]:.4f}',
    'CI_HI': f'{dml_plr.confint().iloc[0, 1]:.4f}',
}
```
