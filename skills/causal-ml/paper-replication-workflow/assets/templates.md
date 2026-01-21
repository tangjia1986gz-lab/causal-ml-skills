# Paper Replication Workflow - Template Reference

## Shared Templates

This skill uses all shared templates for comprehensive paper replication:

### LaTeX Templates
| Template | Location | Purpose |
|----------|----------|---------|
| `common_preamble.tex` | `assets/latex/` | Shared LaTeX setup |
| `regression_table.tex` | `assets/latex/` | Main results tables |
| `summary_stats.tex` | `assets/latex/` | Descriptive statistics |
| `coef_plot.tex` | `assets/latex/` | Coefficient visualizations |
| `event_study.tex` | `assets/latex/` | Event study figures (DID) |
| `event_study_table.tex` | `assets/latex/` | Event study tables (DID) |
| `balance_table.tex` | `assets/latex/` | Balance tables (PSM) |
| `first_stage_table.tex` | `assets/latex/` | First stage (IV) |
| `heterogeneity_table.tex` | `assets/latex/` | CATE heterogeneity |

### Markdown Templates
| Template | Location | Purpose |
|----------|----------|---------|
| `analysis_report.md` | `assets/markdown/` | Full analysis writeup |
| `replication_readme.md` | `assets/markdown/` | Replication package README |
| `robustness_appendix.md` | `assets/markdown/` | Online appendix |
| `data_dictionary.md` | `assets/markdown/` | Variable documentation |

## Workflow Integration

### Phase 1: Data Preparation
Use templates:
- `data_dictionary.md` - Document all variables
- `summary_stats.tex` - Generate descriptive tables

### Phase 2: Identification
Select method-specific templates:
- DID: `event_study.tex`, `event_study_table.tex`
- IV: `first_stage_table.tex`
- PSM: `balance_table.tex`
- RD: Custom adaptation of `coef_plot.tex`

### Phase 3: Main Results
Use templates:
- `regression_table.tex` - Main specifications
- `coef_plot.tex` - Visualize key estimates

### Phase 4: Robustness
Use templates:
- `robustness_appendix.md` - Comprehensive robustness checks
- Method-specific diagnostics

### Phase 5: Heterogeneity
Use templates:
- `heterogeneity_table.tex` - Subgroup analysis
- CATE analysis (if using ML methods)

### Phase 6: Documentation
Use templates:
- `analysis_report.md` - Full paper writeup
- `replication_readme.md` - Replication package
- `data_dictionary.md` - Final variable codebook

## Master Template Generator

```python
from pathlib import Path

class ReplicationTemplates:
    """Helper class for accessing all replication templates."""

    def __init__(self, project_root):
        self.assets = Path(project_root) / 'assets'
        self.latex = self.assets / 'latex'
        self.markdown = self.assets / 'markdown'

    def get_latex(self, name):
        """Load a LaTeX template."""
        return (self.latex / f'{name}.tex').read_text()

    def get_markdown(self, name):
        """Load a Markdown template."""
        return (self.markdown / f'{name}.md').read_text()

    def fill(self, template, values):
        """Fill template with values."""
        content = template
        for key, value in values.items():
            content = content.replace(f'{{{{{key}}}}}', str(value))
        return content

    def generate_replication_package(self, output_dir, metadata):
        """Generate complete replication package structure."""
        output = Path(output_dir)

        # Create README
        readme = self.get_markdown('replication_readme')
        readme = self.fill(readme, metadata)
        (output / 'README.md').write_text(readme)

        # Create data dictionary
        datadict = self.get_markdown('data_dictionary')
        (output / 'docs' / 'data_dictionary.md').write_text(datadict)

        # Copy LaTeX templates
        (output / 'output' / 'templates').mkdir(parents=True)
        for tex in self.latex.glob('*.tex'):
            (output / 'output' / 'templates' / tex.name).write_text(
                tex.read_text()
            )

# Usage
templates = ReplicationTemplates('/path/to/causal-ml-skills')
templates.generate_replication_package('./replication/', metadata)
```

## Replication Standards

This workflow follows:
- **AEA Data and Code Availability Policy**
- **TIER Protocol** for reproducible research
- **Social Science Data Editors' guidelines**

## Checklist

Before submission, verify:
- [ ] All tables generated from templates
- [ ] README.md complete with all placeholders filled
- [ ] Data dictionary covers all variables
- [ ] Robustness appendix includes all checks
- [ ] Code runs on clean environment
- [ ] Random seeds documented

## Usage

```python
# Access all templates
from pathlib import Path

ASSETS_DIR = Path(__file__).parent.parent.parent.parent / 'assets'
LATEX_DIR = ASSETS_DIR / 'latex'
MD_DIR = ASSETS_DIR / 'markdown'

# Load all templates
templates = {
    'regression': (LATEX_DIR / 'regression_table.tex').read_text(),
    'summary': (LATEX_DIR / 'summary_stats.tex').read_text(),
    'report': (MD_DIR / 'analysis_report.md').read_text(),
    'readme': (MD_DIR / 'replication_readme.md').read_text(),
    'datadict': (MD_DIR / 'data_dictionary.md').read_text(),
    'appendix': (MD_DIR / 'robustness_appendix.md').read_text(),
}
```
