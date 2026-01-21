"""
Publication-quality table formatting for causal inference results.
"""

from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass
import numpy as np


@dataclass
class RegressionResult:
    """Standard regression result for table formatting."""
    variable: str
    coefficient: float
    std_error: float
    p_value: float
    ci_lower: float = None
    ci_upper: float = None


def format_coefficient(
    coef: float,
    se: float,
    p_value: float,
    decimal_places: int = 3
) -> str:
    """Format coefficient with significance stars."""
    stars = ""
    if p_value < 0.01:
        stars = "***"
    elif p_value < 0.05:
        stars = "**"
    elif p_value < 0.1:
        stars = "*"

    return f"{coef:.{decimal_places}f}{stars}"


def format_std_error(se: float, decimal_places: int = 3) -> str:
    """Format standard error in parentheses."""
    return f"({se:.{decimal_places}f})"


def create_regression_table(
    results: List[Dict[str, Any]],
    column_names: List[str] = None,
    title: str = "Regression Results",
    notes: str = None,
    output_format: str = "markdown"
) -> str:
    """
    Create publication-quality regression table.

    Parameters
    ----------
    results : List[Dict[str, Any]]
        List of result dictionaries, one per column
        Each dict should have:
        - 'treatment_effect': float
        - 'treatment_se': float
        - 'treatment_pval': float
        - 'controls': bool
        - 'fixed_effects': bool (optional)
        - 'n_obs': int
        - 'r_squared': float
    column_names : List[str], optional
        Column headers
    title : str
        Table title
    notes : str, optional
        Table footnotes
    output_format : str
        'markdown', 'latex', or 'text'

    Returns
    -------
    str
        Formatted table string
    """
    n_cols = len(results)

    if column_names is None:
        column_names = [f"({i+1})" for i in range(n_cols)]

    if output_format == "latex":
        return _create_latex_table(results, column_names, title, notes)
    elif output_format == "markdown":
        return _create_markdown_table(results, column_names, title, notes)
    else:
        return _create_text_table(results, column_names, title, notes)


def _create_markdown_table(
    results: List[Dict[str, Any]],
    column_names: List[str],
    title: str,
    notes: str
) -> str:
    """Create markdown-formatted regression table."""
    lines = []
    lines.append(f"### {title}")
    lines.append("")

    # Header
    header = "| Variable | " + " | ".join(column_names) + " |"
    separator = "|:---------|" + "|".join([":------:" for _ in column_names]) + "|"
    lines.append(header)
    lines.append(separator)

    # Treatment effect row
    effect_row = "| Treatment Effect | "
    for r in results:
        coef_str = format_coefficient(
            r.get('treatment_effect', 0),
            r.get('treatment_se', 0),
            r.get('treatment_pval', 1)
        )
        effect_row += f"{coef_str} | "
    lines.append(effect_row)

    # Standard error row
    se_row = "| | "
    for r in results:
        se_str = format_std_error(r.get('treatment_se', 0))
        se_row += f"{se_str} | "
    lines.append(se_row)

    # Empty row
    lines.append("| | " + " | ".join(["" for _ in column_names]) + " |")

    # Controls row
    controls_row = "| Controls | "
    for r in results:
        controls_row += ("Yes | " if r.get('controls', False) else "No | ")
    lines.append(controls_row)

    # Fixed effects row (if any)
    if any(r.get('fixed_effects') is not None for r in results):
        fe_row = "| Fixed Effects | "
        for r in results:
            fe = r.get('fixed_effects')
            if fe is None:
                fe_row += "- | "
            elif isinstance(fe, str):
                fe_row += f"{fe} | "
            else:
                fe_row += ("Yes | " if fe else "No | ")
        lines.append(fe_row)

    # Empty row
    lines.append("| | " + " | ".join(["" for _ in column_names]) + " |")

    # Observations
    obs_row = "| Observations | "
    for r in results:
        n = r.get('n_obs', 0)
        obs_row += f"{n:,} | "
    lines.append(obs_row)

    # R-squared
    r2_row = "| R-squared | "
    for r in results:
        r2 = r.get('r_squared', 0)
        r2_row += f"{r2:.3f} | "
    lines.append(r2_row)

    lines.append("")

    # Notes
    if notes is None:
        notes = "Standard errors in parentheses. *** p<0.01, ** p<0.05, * p<0.1"
    lines.append(f"*Notes: {notes}*")

    return "\n".join(lines)


def _create_latex_table(
    results: List[Dict[str, Any]],
    column_names: List[str],
    title: str,
    notes: str
) -> str:
    """Create LaTeX-formatted regression table."""
    n_cols = len(results)
    col_spec = "l" + "c" * n_cols

    lines = []
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    lines.append(f"\\caption{{{title}}}")
    lines.append(r"\begin{tabular}{" + col_spec + "}")
    lines.append(r"\hline\hline")

    # Header
    header = " & " + " & ".join(column_names) + r" \\"
    lines.append(header)
    lines.append(r"\hline")

    # Treatment effect
    effect_row = "Treatment Effect & "
    for r in results:
        coef_str = format_coefficient(
            r.get('treatment_effect', 0),
            r.get('treatment_se', 0),
            r.get('treatment_pval', 1)
        )
        effect_row += f"{coef_str} & "
    effect_row = effect_row[:-3] + r" \\"
    lines.append(effect_row)

    # Standard errors
    se_row = " & "
    for r in results:
        se_str = format_std_error(r.get('treatment_se', 0))
        se_row += f"{se_str} & "
    se_row = se_row[:-3] + r" \\"
    lines.append(se_row)

    lines.append(r"\\")

    # Controls
    controls_row = "Controls & "
    for r in results:
        controls_row += ("Yes & " if r.get('controls', False) else "No & ")
    controls_row = controls_row[:-3] + r" \\"
    lines.append(controls_row)

    # Fixed effects
    if any(r.get('fixed_effects') is not None for r in results):
        fe_row = "Fixed Effects & "
        for r in results:
            fe = r.get('fixed_effects')
            if fe is None:
                fe_row += "- & "
            elif isinstance(fe, str):
                fe_row += f"{fe} & "
            else:
                fe_row += ("Yes & " if fe else "No & ")
        fe_row = fe_row[:-3] + r" \\"
        lines.append(fe_row)

    lines.append(r"\\")

    # Observations
    obs_row = "Observations & "
    for r in results:
        n = r.get('n_obs', 0)
        obs_row += f"{n:,} & "
    obs_row = obs_row[:-3] + r" \\"
    lines.append(obs_row)

    # R-squared
    r2_row = "R-squared & "
    for r in results:
        r2 = r.get('r_squared', 0)
        r2_row += f"{r2:.3f} & "
    r2_row = r2_row[:-3] + r" \\"
    lines.append(r2_row)

    lines.append(r"\hline\hline")
    lines.append(r"\end{tabular}")

    # Notes
    if notes is None:
        notes = "Standard errors in parentheses. *** p<0.01, ** p<0.05, * p<0.1"
    lines.append(r"\begin{tablenotes}")
    lines.append(r"\small")
    lines.append(f"\\item {notes}")
    lines.append(r"\end{tablenotes}")
    lines.append(r"\end{table}")

    return "\n".join(lines)


def _create_text_table(
    results: List[Dict[str, Any]],
    column_names: List[str],
    title: str,
    notes: str
) -> str:
    """Create plain text formatted regression table."""
    width = 60
    col_width = 12

    lines = []
    lines.append("=" * width)
    lines.append(title.center(width))
    lines.append("=" * width)

    # Header
    header = "Variable".ljust(20) + "".join(c.center(col_width) for c in column_names)
    lines.append(header)
    lines.append("-" * width)

    # Treatment effect
    effect_row = "Treatment Effect".ljust(20)
    for r in results:
        coef_str = format_coefficient(
            r.get('treatment_effect', 0),
            r.get('treatment_se', 0),
            r.get('treatment_pval', 1)
        )
        effect_row += coef_str.center(col_width)
    lines.append(effect_row)

    # Standard errors
    se_row = "".ljust(20)
    for r in results:
        se_str = format_std_error(r.get('treatment_se', 0))
        se_row += se_str.center(col_width)
    lines.append(se_row)

    lines.append("")

    # Controls
    controls_row = "Controls".ljust(20)
    for r in results:
        controls_row += ("Yes" if r.get('controls', False) else "No").center(col_width)
    lines.append(controls_row)

    lines.append("")

    # Observations
    obs_row = "Observations".ljust(20)
    for r in results:
        n = r.get('n_obs', 0)
        obs_row += f"{n:,}".center(col_width)
    lines.append(obs_row)

    # R-squared
    r2_row = "R-squared".ljust(20)
    for r in results:
        r2 = r.get('r_squared', 0)
        r2_row += f"{r2:.3f}".center(col_width)
    lines.append(r2_row)

    lines.append("=" * width)

    # Notes
    if notes is None:
        notes = "Standard errors in parentheses. *** p<0.01, ** p<0.05, * p<0.1"
    lines.append(f"Notes: {notes}")

    return "\n".join(lines)


def create_diagnostic_report(
    diagnostics: Dict[str, Any],
    title: str = "Diagnostic Tests"
) -> str:
    """
    Create formatted diagnostic report.

    Parameters
    ----------
    diagnostics : Dict[str, Any]
        Dictionary of diagnostic results
    title : str
        Report title

    Returns
    -------
    str
        Formatted diagnostic report
    """
    lines = []
    lines.append(f"## {title}")
    lines.append("")

    for test_name, result in diagnostics.items():
        if hasattr(result, 'passed'):
            # DiagnosticResult object
            status = "PASS" if result.passed else "FAIL"
            lines.append(f"### {result.test_name}")
            lines.append(f"- **Status**: {status}")
            lines.append(f"- **Statistic**: {result.statistic:.4f}")
            if not np.isnan(result.p_value):
                lines.append(f"- **P-value**: {result.p_value:.4f}")
            lines.append(f"- **Interpretation**: {result.interpretation}")
            lines.append("")
        elif isinstance(result, dict):
            # Dictionary result
            lines.append(f"### {test_name}")
            for k, v in result.items():
                if isinstance(v, float):
                    lines.append(f"- **{k}**: {v:.4f}")
                else:
                    lines.append(f"- **{k}**: {v}")
            lines.append("")
        else:
            lines.append(f"### {test_name}: {result}")
            lines.append("")

    return "\n".join(lines)
