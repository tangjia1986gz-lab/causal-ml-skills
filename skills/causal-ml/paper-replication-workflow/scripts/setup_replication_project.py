#!/usr/bin/env python
"""
Setup Replication Project
=========================

Initialize a replication project structure following AEA Data Editor
requirements and best practices for computational reproducibility.

This script creates the directory structure, templates, and configuration
files needed for a professional replication package.

Usage:
    python setup_replication_project.py [project_name] [options]

Examples:
    # Interactive mode
    python setup_replication_project.py

    # Non-interactive
    python setup_replication_project.py "LaLonde_1986_Replication" --author "John Smith"

    # With paper info
    python setup_replication_project.py "Card_Krueger_1994" \
        --title "Minimum Wages and Employment" \
        --author "Jane Doe" \
        --original-paper "Card & Krueger (1994)"

Output:
    Creates project directory with full replication structure
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional


# Project structure definition
PROJECT_STRUCTURE = {
    "code": {
        "lib": {},
    },
    "data": {
        "raw": {},
        "processed": {},
        "temp": {},
        "simulated": {},
    },
    "output": {
        "tables": {},
        "figures": {},
        "logs": {},
    },
    "docs": {},
    "manuscript": {},
}


def create_directory_structure(root: Path, structure: dict) -> None:
    """Recursively create directory structure."""
    for name, substructure in structure.items():
        dir_path = root / name
        dir_path.mkdir(parents=True, exist_ok=True)

        if substructure:
            create_directory_structure(dir_path, substructure)


def create_readme(
    root: Path,
    project_name: str,
    author: str,
    original_paper: str,
    title: str
) -> None:
    """Create main README.md file."""

    readme_content = f'''# Replication Package: {title}

**Authors:** {author}

**Original Paper:** {original_paper}

**Last Updated:** {datetime.now().strftime("%Y-%m-%d")}

---

## Overview

This replication package contains the data and code required to reproduce
all tables and figures in "{title}."

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

| File | Description | Source | License |
|------|-------------|--------|---------|
| *Add data files here* | | | |

### Restricted/External Data

| Dataset | Source | Access | Instructions |
|---------|--------|--------|--------------|
| *Add restricted data here* | | | |

### Simulated Data

For code verification, simulated data with similar properties is provided in `data/simulated/`.

---

## Computational Requirements

### Software

- **Python**: 3.10 or higher

### Python Packages

Install required packages:
```bash
pip install -r requirements.txt
```

### Hardware

- **Memory**: Minimum 8GB RAM (16GB recommended)
- **Storage**: [X]GB free disk space
- **Processor**: Standard laptop/desktop sufficient

### Runtime

| Step | Approximate Time |
|------|------------------|
| Data cleaning | X minutes |
| Main analysis | X minutes |
| **Total** | **~X hours** |

---

## Instructions

### Quick Start

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\\Scripts\\activate

# Install dependencies
pip install -r requirements.txt

# Run replication (all steps)
python code/00_master.py
```

### Step-by-Step

1. **Set up environment** (see Computational Requirements)
2. **Obtain data** (see Data Availability)
3. **Run analysis**: `python code/00_master.py`
4. **Check output**: `output/tables/` and `output/figures/`

---

## Output Mapping

### Tables

| Table | Script | Output File |
|-------|--------|-------------|
| Table 1 | `code/06_tables.py` | `output/tables/table_1.tex` |

### Figures

| Figure | Script | Output File |
|--------|--------|-------------|
| Figure 1 | `code/07_figures.py` | `output/figures/figure_1.pdf` |

---

## Contact

For questions about the code or data:

- **Corresponding Author:** {author}

---

## License

See LICENSE file for terms.
'''

    (root / "README.md").write_text(readme_content)


def create_license(root: Path, author: str) -> None:
    """Create LICENSE file (MIT by default)."""

    year = datetime.now().year
    license_content = f'''MIT License

Copyright (c) {year} {author}

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
'''

    (root / "LICENSE").write_text(license_content)


def create_citation(
    root: Path,
    project_name: str,
    author: str,
    title: str
) -> None:
    """Create CITATION.cff file."""

    # Parse author name (assume "First Last" format)
    author_parts = author.split()
    given_name = author_parts[0] if author_parts else "Author"
    family_name = " ".join(author_parts[1:]) if len(author_parts) > 1 else ""

    citation_content = f'''cff-version: 1.2.0
message: "If you use this replication package, please cite it as below."
authors:
  - family-names: "{family_name}"
    given-names: "{given_name}"
title: "{title} - Replication Package"
version: 1.0.0
date-released: {datetime.now().strftime("%Y-%m-%d")}
repository-code: "https://github.com/[username]/{project_name}"
license: MIT
type: software
keywords:
  - replication
  - economics
  - causal inference
'''

    (root / "CITATION.cff").write_text(citation_content)


def create_requirements(root: Path) -> None:
    """Create requirements.txt file."""

    requirements_content = '''# Core dependencies
pandas>=2.0.0
numpy>=1.24.0
scipy>=1.10.0

# Statistical/Econometric
statsmodels>=0.14.0
linearmodels>=5.0

# Causal Inference
econml>=0.14.0
doubleml>=0.7.0
# causalml>=0.15.0  # Uncomment if needed

# Machine Learning
scikit-learn>=1.2.0

# Visualization
matplotlib>=3.7.0
seaborn>=0.12.0

# Tables
tabulate>=0.9.0
stargazer>=0.0.5

# Utilities
tqdm>=4.65.0
'''

    (root / "requirements.txt").write_text(requirements_content)


def create_gitignore(root: Path) -> None:
    """Create .gitignore file."""

    gitignore_content = '''# Data (too large for git or regenerated)
data/raw/*.csv
data/raw/*.dta
data/raw/*.xlsx
data/processed/
data/temp/

# Output (can be regenerated)
output/tables/*.tex
output/tables/*.csv
output/figures/*.pdf
output/figures/*.png
output/logs/

# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Virtual environments
.env
.venv
env/
venv/
ENV/

# Jupyter Notebook
.ipynb_checkpoints

# IDE
.idea/
.vscode/
*.swp
*.swo

# R
.Rhistory
.Rdata
.RData
*.Rproj.user

# Stata
*.log
*.smcl

# OS
.DS_Store
Thumbs.db

# Secrets (never commit)
.env
*.pem
credentials.json
'''

    (root / ".gitignore").write_text(gitignore_content)


def create_master_script(root: Path, project_name: str) -> None:
    """Create master script (00_master.py)."""

    master_content = f'''#!/usr/bin/env python
"""
Master Script for Replication Package
=====================================

Project: {project_name}

This script reproduces all tables and figures in the paper.

Usage:
    python 00_master.py [options]

Options:
    --skip-download    Skip data download
    --skip-clean       Skip data cleaning
    --tables-only      Only regenerate tables
    --figures-only     Only regenerate figures

Runtime: Approximately X hours
"""

import argparse
import subprocess
import sys
import time
import logging
from pathlib import Path
from datetime import datetime

# Configuration
PROJECT_ROOT = Path(__file__).parent.parent
CODE_DIR = PROJECT_ROOT / "code"
OUTPUT_DIR = PROJECT_ROOT / "output"
LOG_DIR = OUTPUT_DIR / "logs"

# Pipeline definition: (script_name, description)
PIPELINE = [
    ("01_download_data.py", "Download raw data"),
    ("02_clean_data.py", "Clean and prepare data"),
    ("03_construct_vars.py", "Construct analysis variables"),
    ("04_analysis_main.py", "Run main analysis"),
    ("05_analysis_robust.py", "Run robustness checks"),
    ("06_tables.py", "Generate tables"),
    ("07_figures.py", "Generate figures"),
]


def setup_logging():
    """Configure logging."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = LOG_DIR / f"replication_{{timestamp}}.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )

    return log_file


def run_script(script_name: str, description: str) -> tuple:
    """Run a single script."""
    script_path = CODE_DIR / script_name
    log = logging.getLogger(__name__)

    if not script_path.exists():
        log.warning(f"Script not found: {{script_name}} - Skipping")
        return (script_name, True, 0, "Skipped (not found)")

    log.info(f"Starting: {{description}}")
    log.info(f"Script: {{script_name}}")

    start_time = time.time()

    try:
        result = subprocess.run(
            [sys.executable, str(script_path)],
            capture_output=True,
            text=True,
            cwd=str(CODE_DIR)
        )

        elapsed = time.time() - start_time

        if result.returncode != 0:
            log.error(f"FAILED: {{script_name}}")
            log.error(f"Stderr: {{result.stderr}}")
            return (script_name, False, elapsed, result.stderr)

        log.info(f"Completed: {{script_name}} ({{elapsed:.1f}}s)")
        return (script_name, True, elapsed, result.stdout)

    except Exception as e:
        elapsed = time.time() - start_time
        log.error(f"Exception in {{script_name}}: {{e}}")
        return (script_name, False, elapsed, str(e))


def main():
    """Run the replication pipeline."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--skip-clean", action="store_true")
    parser.add_argument("--tables-only", action="store_true")
    parser.add_argument("--figures-only", action="store_true")
    args = parser.parse_args()

    log_file = setup_logging()
    log = logging.getLogger(__name__)

    log.info("=" * 60)
    log.info("REPLICATION PACKAGE - MASTER SCRIPT")
    log.info("=" * 60)
    log.info(f"Log file: {{log_file}}")
    log.info(f"Start time: {{datetime.now().isoformat()}}")

    # Filter pipeline based on arguments
    pipeline = PIPELINE.copy()

    if args.skip_download:
        pipeline = [(s, d) for s, d in pipeline if "download" not in s.lower()]

    if args.skip_clean:
        pipeline = [(s, d) for s, d in pipeline
                   if "clean" not in s.lower() and "construct" not in s.lower()]

    if args.tables_only:
        pipeline = [(s, d) for s, d in pipeline if "table" in s.lower()]

    if args.figures_only:
        pipeline = [(s, d) for s, d in pipeline if "figure" in s.lower()]

    log.info(f"Running {{len(pipeline)}} scripts")

    total_start = time.time()
    results = []
    failed = []

    for script_name, description in pipeline:
        result = run_script(script_name, description)
        results.append(result)

        if not result[1]:
            failed.append(script_name)

    total_elapsed = time.time() - total_start

    # Summary
    log.info("=" * 60)
    log.info("REPLICATION SUMMARY")
    log.info("=" * 60)
    log.info(f"Total runtime: {{total_elapsed/60:.1f}} minutes")
    log.info(f"Scripts run: {{len(results)}}")
    log.info(f"Successful: {{len(results) - len(failed)}}")
    log.info(f"Failed: {{len(failed)}}")

    if failed:
        log.error(f"Failed scripts: {{failed}}")
        sys.exit(1)
    else:
        log.info("All scripts completed successfully!")


if __name__ == "__main__":
    main()
'''

    code_dir = root / "code"
    (code_dir / "00_master.py").write_text(master_content)


def create_script_templates(root: Path) -> None:
    """Create template scripts."""

    code_dir = root / "code"

    # Data download template
    download_content = '''#!/usr/bin/env python
"""
Download Data
=============

Downloads raw data from original sources.

Input: None
Output: data/raw/
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"


def main():
    """Download data files."""
    print("=" * 60)
    print("Downloading data...")
    print("=" * 60)

    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

    # TODO: Add data download logic
    # Example:
    # import requests
    # response = requests.get(DATA_URL)
    # (RAW_DATA_DIR / "data.csv").write_bytes(response.content)

    print("Data download complete!")


if __name__ == "__main__":
    main()
'''

    # Data cleaning template
    clean_content = '''#!/usr/bin/env python
"""
Clean Data
==========

Cleans raw data and creates analysis sample.

Input: data/raw/
Output: data/processed/analysis_sample.csv
"""

import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROC_DATA_DIR = PROJECT_ROOT / "data" / "processed"


def main():
    """Clean data."""
    print("=" * 60)
    print("Cleaning data...")
    print("=" * 60)

    PROC_DATA_DIR.mkdir(parents=True, exist_ok=True)

    # TODO: Add data cleaning logic
    # df = pd.read_csv(RAW_DATA_DIR / "raw_data.csv")
    # ...cleaning steps...
    # df.to_csv(PROC_DATA_DIR / "analysis_sample.csv", index=False)

    print("Data cleaning complete!")


if __name__ == "__main__":
    main()
'''

    # Analysis template
    analysis_content = '''#!/usr/bin/env python
"""
Main Analysis
=============

Runs main analysis specifications.

Input: data/processed/analysis_sample.csv
Output: output/tables/, output/logs/
"""

import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_DIR = PROJECT_ROOT / "output"


def main():
    """Run main analysis."""
    print("=" * 60)
    print("Running main analysis...")
    print("=" * 60)

    # Load data
    # df = pd.read_csv(DATA_DIR / "analysis_sample.csv")

    # TODO: Add analysis logic

    print("Main analysis complete!")


if __name__ == "__main__":
    main()
'''

    # Write templates
    (code_dir / "01_download_data.py").write_text(download_content)
    (code_dir / "02_clean_data.py").write_text(clean_content)
    (code_dir / "04_analysis_main.py").write_text(analysis_content)

    # Create lib __init__.py
    (code_dir / "lib" / "__init__.py").write_text('"""Helper modules for analysis."""\n')


def create_data_readme(root: Path) -> None:
    """Create README for data directory."""

    readme_content = '''# Data Directory

## Structure

```
data/
├── raw/           # Original, unmodified data (read-only)
├── processed/     # Cleaned data (can be regenerated)
├── temp/          # Temporary files (gitignored)
└── simulated/     # Simulated data for code verification
```

## Data Sources

| File | Source | Access | Notes |
|------|--------|--------|-------|
| *Add files here* | | | |

## Instructions

### Obtaining Data

1. **Public data**: Run `python code/01_download_data.py`
2. **Restricted data**: Follow instructions in main README

### Data Processing

The cleaning script `code/02_clean_data.py` transforms raw data into
the analysis sample. Run this before analysis.

## Version Control

- Raw data is tracked with Git LFS (if included) or must be obtained separately
- Processed data is not tracked (regenerate using cleaning scripts)
- Temp files are gitignored
'''

    (root / "data" / "README.md").write_text(readme_content)


def create_project_config(root: Path, project_name: str, author: str) -> None:
    """Create project configuration file."""

    config = {
        "project_name": project_name,
        "author": author,
        "created": datetime.now().isoformat(),
        "python_version": f"{sys.version_info.major}.{sys.version_info.minor}",
        "structure_version": "1.0"
    }

    (root / ".replication_config.json").write_text(
        json.dumps(config, indent=2)
    )


def setup_project(
    project_name: str,
    author: str,
    original_paper: str,
    title: str,
    output_dir: Optional[Path] = None
) -> Path:
    """
    Set up a complete replication project.

    Parameters
    ----------
    project_name : str
        Name for the project directory (no spaces)
    author : str
        Author name(s)
    original_paper : str
        Citation for the paper being replicated
    title : str
        Title of the paper
    output_dir : Path, optional
        Parent directory for the project. Defaults to current directory.

    Returns
    -------
    Path
        Path to the created project directory
    """
    if output_dir is None:
        output_dir = Path.cwd()

    # Clean project name (replace spaces with underscores)
    project_name_clean = project_name.replace(" ", "_")
    project_root = output_dir / project_name_clean

    if project_root.exists():
        raise ValueError(f"Directory already exists: {project_root}")

    print(f"Creating replication project: {project_name}")
    print(f"Location: {project_root}")
    print()

    # Create structure
    print("Creating directory structure...")
    project_root.mkdir(parents=True)
    create_directory_structure(project_root, PROJECT_STRUCTURE)

    # Create files
    print("Creating documentation files...")
    create_readme(project_root, project_name, author, original_paper, title)
    create_license(project_root, author)
    create_citation(project_root, project_name, author, title)
    create_requirements(project_root)
    create_gitignore(project_root)

    print("Creating script templates...")
    create_master_script(project_root, project_name)
    create_script_templates(project_root)

    print("Creating data documentation...")
    create_data_readme(project_root)

    print("Creating project configuration...")
    create_project_config(project_root, project_name, author)

    print()
    print("=" * 60)
    print("Project created successfully!")
    print("=" * 60)
    print()
    print("Next steps:")
    print(f"  1. cd {project_root}")
    print("  2. Edit README.md with your specific information")
    print("  3. Add your data to data/raw/")
    print("  4. Implement the analysis scripts in code/")
    print("  5. Run: python code/00_master.py")
    print()

    return project_root


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Set up a replication project structure.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python setup_replication_project.py
  python setup_replication_project.py "LaLonde_1986" --author "John Smith"
  python setup_replication_project.py "Card_Krueger" --title "Minimum Wages"
        """
    )

    parser.add_argument(
        "project_name",
        nargs="?",
        help="Name for the project directory"
    )
    parser.add_argument(
        "--author",
        help="Author name(s)"
    )
    parser.add_argument(
        "--title",
        help="Paper title"
    )
    parser.add_argument(
        "--original-paper",
        help="Citation for the original paper"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Parent directory for the project"
    )

    args = parser.parse_args()

    # Interactive mode if no project name provided
    if args.project_name is None:
        print("=" * 60)
        print("Replication Project Setup")
        print("=" * 60)
        print()

        project_name = input("Project name (e.g., LaLonde_1986_Replication): ").strip()
        if not project_name:
            print("Error: Project name is required")
            sys.exit(1)

        author = input("Author name(s): ").strip() or "Author"
        title = input("Paper title: ").strip() or project_name
        original_paper = input("Original paper citation: ").strip() or "Original Paper"
    else:
        project_name = args.project_name
        author = args.author or "Author"
        title = args.title or project_name
        original_paper = args.original_paper or "Original Paper"

    try:
        setup_project(
            project_name=project_name,
            author=author,
            original_paper=original_paper,
            title=title,
            output_dir=args.output_dir
        )
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
