# Replication Package

## {{PAPER_TITLE}}

**Authors:** {{AUTHORS}}
**Journal:** {{JOURNAL}}
**Year:** {{YEAR}}
**DOI:** {{DOI}}

---

## Overview

This replication package contains all code and data necessary to reproduce the results in "{{PAPER_TITLE}}."

**Corresponding Author:** {{CORRESPONDING_AUTHOR}}
**Email:** {{EMAIL}}
**Last Updated:** {{LAST_UPDATED}}

---

## Data Availability Statement

{{DATA_AVAILABILITY_STATEMENT}}

| Data Source | Access | License | Notes |
|-------------|--------|---------|-------|
| {{DATA_1}} | {{ACCESS_1}} | {{LICENSE_1}} | {{NOTES_1}} |
| {{DATA_2}} | {{ACCESS_2}} | {{LICENSE_2}} | {{NOTES_2}} |
| {{DATA_3}} | {{ACCESS_3}} | {{LICENSE_3}} | {{NOTES_3}} |

### Data Access Instructions

{{DATA_ACCESS_INSTRUCTIONS}}

---

## Computational Requirements

### Software Requirements

| Software | Version | Purpose | Required |
|----------|---------|---------|----------|
| Python | {{PYTHON_VERSION}} | Main analysis | Yes |
| R | {{R_VERSION}} | {{R_PURPOSE}} | {{R_REQUIRED}} |
| Stata | {{STATA_VERSION}} | {{STATA_PURPOSE}} | {{STATA_REQUIRED}} |

### Python Dependencies

```
# Core packages
numpy=={{NUMPY_VERSION}}
pandas=={{PANDAS_VERSION}}
scipy=={{SCIPY_VERSION}}

# Causal inference
econml=={{ECONML_VERSION}}
doubleml=={{DOUBLEML_VERSION}}
causalml=={{CAUSALML_VERSION}}

# Machine learning
scikit-learn=={{SKLEARN_VERSION}}
xgboost=={{XGBOOST_VERSION}}
lightgbm=={{LIGHTGBM_VERSION}}

# Visualization
matplotlib=={{MPL_VERSION}}
seaborn=={{SEABORN_VERSION}}

# Tables
stargazer=={{STARGAZER_VERSION}}
```

### R Dependencies

```r
# Install required packages
install.packages(c(
    "grf",           # Causal forests
    "fixest",        # Fast fixed effects
    "did",           # Difference-in-differences
    "rdrobust",      # Regression discontinuity
    "MatchIt",       # Propensity score matching
    "cobalt",        # Balance diagnostics
    "modelsummary",  # Regression tables
    "ggplot2"        # Visualization
))
```

### Hardware Requirements

| Resource | Minimum | Recommended | Used in Paper |
|----------|---------|-------------|---------------|
| RAM | {{MIN_RAM}} | {{REC_RAM}} | {{USED_RAM}} |
| CPU Cores | {{MIN_CORES}} | {{REC_CORES}} | {{USED_CORES}} |
| Disk Space | {{MIN_DISK}} | {{REC_DISK}} | {{USED_DISK}} |
| GPU | {{GPU_REQ}} | {{GPU_REC}} | {{GPU_USED}} |

**Estimated Runtime:** {{TOTAL_RUNTIME}} on {{RUNTIME_HARDWARE}}

---

## Directory Structure

```
replication_package/
│
├── README.md                    # This file
├── LICENSE                      # License file
├── CHANGELOG.md                 # Version history
│
├── data/
│   ├── raw/                     # Original data (if redistributable)
│   │   ├── {{RAW_DATA_1}}.csv
│   │   └── {{RAW_DATA_2}}.dta
│   ├── processed/               # Cleaned analysis data
│   │   └── analysis_sample.parquet
│   └── README.md                # Data documentation
│
├── code/
│   ├── 00_master.py             # Master script
│   ├── 01_data_cleaning.py      # Data preparation
│   ├── 02_descriptives.py       # Summary statistics
│   ├── 03_main_analysis.py      # Main results (Tables 1-3)
│   ├── 04_robustness.py         # Robustness checks (Table 4)
│   ├── 05_heterogeneity.py      # Heterogeneous effects (Table 5)
│   ├── 06_figures.py            # All figures
│   └── utils/                   # Helper functions
│       ├── __init__.py
│       ├── data_utils.py
│       ├── estimation.py
│       └── visualization.py
│
├── output/
│   ├── tables/                  # LaTeX/CSV tables
│   │   ├── table1_summary.tex
│   │   ├── table2_main.tex
│   │   └── ...
│   ├── figures/                 # PDF/PNG figures
│   │   ├── figure1_eventstudy.pdf
│   │   └── ...
│   └── logs/                    # Execution logs
│
├── docs/
│   ├── data_dictionary.md       # Variable definitions
│   ├── codebook.pdf             # Survey codebook
│   └── additional_results.md    # Supplementary analyses
│
└── environment/
    ├── requirements.txt         # Python dependencies
    ├── environment.yml          # Conda environment
    └── renv.lock                # R dependencies
```

---

## Instructions for Replication

### Step 1: Environment Setup

**Option A: Conda (Recommended)**

```bash
# Clone repository
git clone {{REPO_URL}}
cd {{REPO_NAME}}

# Create conda environment
conda env create -f environment/environment.yml
conda activate {{ENV_NAME}}
```

**Option B: pip**

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r environment/requirements.txt
```

**Option C: Docker**

```bash
# Pull and run container
docker pull {{DOCKER_IMAGE}}
docker run -it -v $(pwd):/app {{DOCKER_IMAGE}}
```

### Step 2: Data Setup

{{DATA_SETUP_INSTRUCTIONS}}

```bash
# If data is included
unzip data/raw/data_archive.zip -d data/raw/

# If data must be downloaded
python code/00_download_data.py

# If data must be requested
# See DATA_ACCESS.md for instructions
```

### Step 3: Run Analysis

**Full Replication (All Results)**

```bash
python code/00_master.py
```

**Individual Scripts**

| Script | Output | Runtime |
|--------|--------|---------|
| `01_data_cleaning.py` | `data/processed/` | {{TIME_01}} |
| `02_descriptives.py` | Table 1 | {{TIME_02}} |
| `03_main_analysis.py` | Tables 2-3, Figures 1-2 | {{TIME_03}} |
| `04_robustness.py` | Table 4, Figure 3 | {{TIME_04}} |
| `05_heterogeneity.py` | Table 5, Figure 4 | {{TIME_05}} |
| `06_figures.py` | Appendix Figures | {{TIME_06}} |

### Step 4: Verify Results

Compare generated outputs with published results:

```bash
python code/verify_replication.py
```

---

## Table and Figure Concordance

### Tables

| Table | Script | Output File | Description |
|-------|--------|-------------|-------------|
| Table 1 | `02_descriptives.py` | `output/tables/table1_summary.tex` | Summary statistics |
| Table 2 | `03_main_analysis.py` | `output/tables/table2_main.tex` | Main results |
| Table 3 | `03_main_analysis.py` | `output/tables/table3_mechanisms.tex` | Mechanisms |
| Table 4 | `04_robustness.py` | `output/tables/table4_robust.tex` | Robustness |
| Table 5 | `05_heterogeneity.py` | `output/tables/table5_hetero.tex` | Heterogeneity |
| Table A1 | `03_main_analysis.py` | `output/tables/tableA1_balance.tex` | Balance table |
| Table A2 | `04_robustness.py` | `output/tables/tableA2_placebo.tex` | Placebo tests |

### Figures

| Figure | Script | Output File | Description |
|--------|--------|-------------|-------------|
| Figure 1 | `03_main_analysis.py` | `output/figures/figure1_eventstudy.pdf` | Event study plot |
| Figure 2 | `03_main_analysis.py` | `output/figures/figure2_coef.pdf` | Coefficient plot |
| Figure 3 | `04_robustness.py` | `output/figures/figure3_sensitivity.pdf` | Sensitivity analysis |
| Figure 4 | `05_heterogeneity.py` | `output/figures/figure4_cate.pdf` | CATE distribution |
| Figure A1 | `06_figures.py` | `output/figures/figureA1_trends.pdf` | Pre-trends |
| Figure A2 | `06_figures.py` | `output/figures/figureA2_pscore.pdf` | Propensity score |

---

## Known Issues and Troubleshooting

### Common Issues

| Issue | Solution |
|-------|----------|
| Memory error in main analysis | Reduce batch size in `config.py` |
| Missing R packages | Run `Rscript code/install_packages.R` |
| Stata license error | Ensure Stata is properly licensed |
| Floating point differences | Minor (<1e-6) due to platform differences |

### Platform-Specific Notes

**Windows:**
```
{{WINDOWS_NOTES}}
```

**macOS:**
```
{{MACOS_NOTES}}
```

**Linux:**
```
{{LINUX_NOTES}}
```

---

## Replication Verification

### Expected Outputs

Results should match published values within the following tolerances:

| Result | Published | Tolerance | Reason |
|--------|-----------|-----------|--------|
| Main treatment effect | {{MAIN_EFFECT}} | {{TOLERANCE_1}} | {{REASON_1}} |
| Bootstrap standard errors | {{SE_VALUE}} | {{TOLERANCE_2}} | Random seed variation |
| ML heterogeneity | {{ML_VALUE}} | {{TOLERANCE_3}} | Stochastic algorithms |

### Certification

This replication package was tested on:

| Tester | Date | Platform | Result |
|--------|------|----------|--------|
| {{TESTER_1}} | {{DATE_1}} | {{PLATFORM_1}} | {{RESULT_1}} |
| {{TESTER_2}} | {{DATE_2}} | {{PLATFORM_2}} | {{RESULT_2}} |

---

## License

{{LICENSE_TEXT}}

---

## Citation

If you use this replication package, please cite:

```bibtex
@article{{{BIBTEX_KEY}},
    author = {{{BIBTEX_AUTHOR}}},
    title = {{{BIBTEX_TITLE}}},
    journal = {{{BIBTEX_JOURNAL}}},
    year = {{{BIBTEX_YEAR}}},
    volume = {{{BIBTEX_VOLUME}}},
    pages = {{{BIBTEX_PAGES}}},
    doi = {{{BIBTEX_DOI}}}
}
```

---

## Contact

For questions about the replication package:

- **Email:** {{CONTACT_EMAIL}}
- **GitHub Issues:** {{GITHUB_ISSUES}}

---

## Acknowledgments

{{ACKNOWLEDGMENTS}}

---

*This replication package follows the AEA Data and Code Availability Policy and the Social Science Data Editors' guidelines.*

*Template version: 1.0.0*
*Generated using the Causal ML Skills framework.*

---

## Template Usage Instructions

### Required Replacements

1. **Paper Information**: PAPER_TITLE, AUTHORS, JOURNAL, DOI
2. **Data Details**: All DATA_* placeholders
3. **Software Versions**: All *_VERSION placeholders
4. **File Paths**: All references to specific files/directories
5. **Runtime Estimates**: TIME_* placeholders

### Customization

- Add/remove software requirements as needed
- Adjust directory structure to match your project
- Include additional troubleshooting notes
- Expand platform-specific instructions

### Best Practices

1. **Test on a clean machine** before submission
2. **Include exact package versions** (use `pip freeze` or `conda list`)
3. **Set random seeds** for reproducibility
4. **Document any manual steps** required
5. **Provide expected runtimes** for each script
6. **Include checksums** for data files when possible

### Quality Checklist

- [ ] All scripts run without error
- [ ] Output matches published tables/figures
- [ ] All data sources documented
- [ ] Dependencies pinned to specific versions
- [ ] README tested by independent user
- [ ] License file included
- [ ] Contact information current
