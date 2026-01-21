# Replication Package: {{PAPER_TITLE}}

**Authors:** {{AUTHORS}}

**Original Paper:** {{PAPER_CITATION}}

**Journal:** {{JOURNAL_NAME}}

**DOI:** {{PAPER_DOI}}

**Last Updated:** {{DATE}}

---

## Overview

This replication package contains the data and code required to reproduce all tables and figures in "{{PAPER_TITLE}}."

### Contents

| Folder | Description |
|--------|-------------|
| `code/` | All analysis scripts |
| `data/` | Data files (see Data Availability) |
| `output/` | Generated tables and figures |
| `docs/` | Additional documentation |

---

## Data Availability

### Included Data

The following data are included in this replication package:

| File | Description | Source | License |
|------|-------------|--------|---------|
| `data/raw/{{DATASET_1}}.csv` | {{DESCRIPTION_1}} | {{SOURCE_1}} | {{LICENSE_1}} |
| `data/raw/{{DATASET_2}}.csv` | {{DESCRIPTION_2}} | {{SOURCE_2}} | {{LICENSE_2}} |

### Restricted/External Data

The following data must be obtained separately:

| Dataset | Source | Access | Instructions |
|---------|--------|--------|--------------|
| {{RESTRICTED_DATASET}} | {{PROVIDER}} | {{ACCESS_URL}} | See `docs/data_access.md` |

**Data Access Requirements:**
- Registration required: {{YES_NO}}
- Institutional affiliation: {{REQUIRED_OR_NOT}}
- Data use agreement: {{DUA_LINK}}
- Estimated processing time: {{TIME_ESTIMATE}}

### Simulated Data

For code verification, simulated data with similar properties is provided:
- `data/simulated/sim_{{DATASET}}.csv`

This simulated data allows running all code but will not reproduce exact results.

---

## Computational Requirements

### Software

- **Python**: 3.10 or higher
- **R**: {{R_VERSION}} (optional, for {{R_COMPONENTS}})
- **Stata**: {{STATA_VERSION}} (optional, for {{STATA_COMPONENTS}})

### Python Packages

Install required packages:
```bash
pip install -r requirements.txt
```

Key dependencies (see `requirements.txt` for complete list with version pins):
- pandas >= 2.0.0
- numpy >= 1.24.0
- statsmodels >= 0.14.0
- econml >= 0.14.0
- {{ADDITIONAL_PACKAGES}}

### Hardware

- **Memory**: Minimum {{MIN_RAM}}GB RAM ({{RECOMMENDED_RAM}}GB recommended)
- **Storage**: {{STORAGE_GB}}GB free disk space
- **Processor**: {{PROCESSOR_REQUIREMENTS}}

### Runtime

| Step | Script | Approximate Time |
|------|--------|------------------|
| Data download | `01_download_data.py` | {{TIME_1}} |
| Data cleaning | `02_clean_data.py` | {{TIME_2}} |
| Main analysis | `04_analysis_main.py` | {{TIME_3}} |
| Robustness checks | `05_analysis_robust.py` | {{TIME_4}} |
| Tables and figures | `06_tables.py`, `07_figures.py` | {{TIME_5}} |
| **Total** | `00_master.py` | **{{TOTAL_TIME}}** |

*Tested on: {{TEST_MACHINE_SPECS}}*

---

## Instructions

### Quick Start

```bash
# Clone repository
git clone {{REPO_URL}}
cd {{REPO_NAME}}

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run replication (all steps)
python code/00_master.py
```

### Step-by-Step Instructions

#### 1. Set Up Environment

```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Verify installation
python -c "import pandas; import statsmodels; print('Setup complete')"
```

#### 2. Obtain Data

**Public data:**
```bash
python code/01_download_data.py
```

**Restricted data:**
1. Follow instructions in `docs/data_access.md`
2. Place obtained files in `data/raw/`
3. Verify: `python code/verify_data.py`

#### 3. Run Analysis

**Full replication:**
```bash
python code/00_master.py
```

**Individual steps:**
```bash
# Data preparation
python code/02_clean_data.py
python code/03_construct_vars.py

# Analysis
python code/04_analysis_main.py
python code/05_analysis_robust.py

# Output
python code/06_tables.py
python code/07_figures.py
```

#### 4. Check Output

Results will be in:
- Tables: `output/tables/`
- Figures: `output/figures/`
- Logs: `output/logs/`

Compare to expected output: `python code/verify_output.py`

---

## Output Mapping

### Tables

| Paper Table | Description | Script | Output File |
|-------------|-------------|--------|-------------|
| Table 1 | Summary Statistics | `06_tables.py` | `output/tables/table_1.tex` |
| Table 2 | Main Results | `06_tables.py` | `output/tables/table_2.tex` |
| Table 3 | Robustness Checks | `06_tables.py` | `output/tables/table_3.tex` |
| Table A1 | Balance Test | `06_tables.py` | `output/tables/table_a1.tex` |

### Figures

| Paper Figure | Description | Script | Output File |
|--------------|-------------|--------|-------------|
| Figure 1 | {{FIGURE_1_DESC}} | `07_figures.py` | `output/figures/figure_1.pdf` |
| Figure 2 | {{FIGURE_2_DESC}} | `07_figures.py` | `output/figures/figure_2.pdf` |
| Figure A1 | {{FIGURE_A1_DESC}} | `07_figures.py` | `output/figures/figure_a1.pdf` |

### In-Text Statistics

Key statistics referenced in the paper are computed in `code/04_analysis_main.py` and logged to `output/logs/statistics.txt`.

| Location | Statistic | Value | Script Line |
|----------|-----------|-------|-------------|
| p.{{PAGE}} | {{STAT_DESC}} | {{VALUE}} | `04_analysis_main.py:{{LINE}}` |

---

## Replication Notes

### Known Differences from Original

{{DIFFERENCES_FROM_ORIGINAL}}

### Random Seeds

For reproducibility, random seeds are set in:
- `code/config.py`: `RANDOM_SEED = {{SEED}}`

### Version-Specific Notes

{{VERSION_NOTES}}

---

## Troubleshooting

### Common Issues

**Issue:** Package installation fails
**Solution:** Ensure Python 3.10+ is installed. Try: `pip install --upgrade pip`

**Issue:** Memory error during analysis
**Solution:** Close other applications. If persists, modify chunk size in `config.py`

**Issue:** Data file not found
**Solution:** Run `01_download_data.py` first, or check restricted data instructions

### Getting Help

1. Check the `docs/FAQ.md` file
2. Review `output/logs/` for error messages
3. Contact authors (see below)

---

## Citation

If you use this replication package, please cite:

```bibtex
@misc{{{CITATION_KEY}},
  author = {{{AUTHORS}}},
  title = {Replication Package for "{{PAPER_TITLE}}"},
  year = {{{YEAR}}},
  publisher = {{{REPOSITORY}}},
  doi = {{{DOI}}}
}
```

---

## Contact

For questions about the code or data:

- **Corresponding Author:** {{AUTHOR_NAME}} ({{AUTHOR_EMAIL}})
- **GitHub Issues:** {{REPO_URL}}/issues

For data access questions:
- **{{DATA_PROVIDER}}:** {{DATA_CONTACT}}

---

## License

Code in this replication package is licensed under {{CODE_LICENSE}}.

Data usage is subject to the terms of their respective providers (see Data Availability).

---

## Acknowledgments

{{ACKNOWLEDGMENTS}}

---

*This README follows the [AEA Data Editor template](https://social-science-data-editors.github.io/template_README/) for replication packages.*
