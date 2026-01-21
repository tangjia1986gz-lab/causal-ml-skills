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


@dataclass
class BalanceTableRow:
    """Row structure for balance tables."""
    variable: str
    treated_mean: float
    control_mean: float
    difference: float
    std_error: Optional[float] = None
    p_value: Optional[float] = None
    smd: Optional[float] = None  # Standardized mean difference


def format_balance_table(
    balance_data: List[Union[BalanceTableRow, Dict[str, Any]]],
    title: str = "Covariate Balance Table",
    before_after: bool = False,
    output_format: str = "markdown",
    decimal_places: int = 3,
    show_smd: bool = True,
    show_pvalue: bool = True,
    notes: Optional[str] = None
) -> str:
    """
    Format a PSM covariate balance table for publication.

    Creates a table comparing covariate means between treatment and control
    groups, typically used to assess the quality of propensity score matching.

    Parameters
    ----------
    balance_data : List[Union[BalanceTableRow, Dict[str, Any]]]
        List of balance results, either BalanceTableRow objects or dicts with keys:
        - variable: str
        - treated_mean: float
        - control_mean: float
        - difference: float (optional, computed if missing)
        - std_error: float (optional)
        - p_value: float (optional)
        - smd: float (optional, standardized mean difference)
    title : str
        Table title
    before_after : bool
        If True, shows before/after matching comparison (expects pairs of rows)
    output_format : str
        Output format: 'markdown', 'latex', 'text', or 'html'
    decimal_places : int
        Number of decimal places for numbers
    show_smd : bool
        Whether to show standardized mean difference column
    show_pvalue : bool
        Whether to show p-value column
    notes : Optional[str]
        Custom footnotes

    Returns
    -------
    str
        Formatted balance table
    """
    # Convert dicts to BalanceTableRow if needed
    rows = []
    for item in balance_data:
        if isinstance(item, dict):
            diff = item.get('difference', item.get('treated_mean', 0) - item.get('control_mean', 0))
            rows.append(BalanceTableRow(
                variable=item.get('variable', ''),
                treated_mean=item.get('treated_mean', 0),
                control_mean=item.get('control_mean', 0),
                difference=diff,
                std_error=item.get('std_error'),
                p_value=item.get('p_value'),
                smd=item.get('smd')
            ))
        else:
            rows.append(item)

    if output_format == "latex":
        return _format_balance_table_latex(rows, title, show_smd, show_pvalue, decimal_places, notes)
    elif output_format == "html":
        return _format_balance_table_html(rows, title, show_smd, show_pvalue, decimal_places, notes)
    elif output_format == "text":
        return _format_balance_table_text(rows, title, show_smd, show_pvalue, decimal_places, notes)
    else:  # markdown
        return _format_balance_table_markdown(rows, title, show_smd, show_pvalue, decimal_places, notes)


def _format_balance_table_markdown(
    rows: List[BalanceTableRow],
    title: str,
    show_smd: bool,
    show_pvalue: bool,
    decimal_places: int,
    notes: Optional[str]
) -> str:
    """Create markdown balance table."""
    lines = [f"### {title}", ""]

    # Build header
    header_parts = ["| Variable", "Treated Mean", "Control Mean", "Difference"]
    if show_smd:
        header_parts.append("SMD")
    if show_pvalue:
        header_parts.append("P-value")
    header = " | ".join(header_parts) + " |"
    lines.append(header)

    # Separator
    sep_parts = ["|:------"] * len(header_parts)
    sep_parts[-1] = sep_parts[-1] + "|"
    separator = sep_parts[0] + "|" + "|".join(sep_parts[1:])
    lines.append(separator)

    # Data rows
    for row in rows:
        parts = [
            f"| {row.variable}",
            f"{row.treated_mean:.{decimal_places}f}",
            f"{row.control_mean:.{decimal_places}f}",
            f"{row.difference:.{decimal_places}f}"
        ]
        if show_smd:
            smd_str = f"{row.smd:.{decimal_places}f}" if row.smd is not None else "-"
            parts.append(smd_str)
        if show_pvalue:
            pval_str = f"{row.p_value:.{decimal_places}f}" if row.p_value is not None else "-"
            parts.append(pval_str)
        lines.append(" | ".join(parts) + " |")

    lines.append("")

    # Notes
    if notes is None:
        notes = "SMD = Standardized Mean Difference. |SMD| < 0.1 indicates good balance."
    lines.append(f"*Notes: {notes}*")

    return "\n".join(lines)


def _format_balance_table_latex(
    rows: List[BalanceTableRow],
    title: str,
    show_smd: bool,
    show_pvalue: bool,
    decimal_places: int,
    notes: Optional[str]
) -> str:
    """Create LaTeX balance table."""
    n_cols = 4 + (1 if show_smd else 0) + (1 if show_pvalue else 0)
    col_spec = "l" + "c" * (n_cols - 1)

    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        f"\\caption{{{title}}}",
        r"\begin{tabular}{" + col_spec + "}",
        r"\hline\hline"
    ]

    # Header
    header_parts = ["Variable", "Treated", "Control", "Diff."]
    if show_smd:
        header_parts.append("SMD")
    if show_pvalue:
        header_parts.append("P-value")
    lines.append(" & ".join(header_parts) + r" \\")
    lines.append(r"\hline")

    # Data
    for row in rows:
        parts = [
            row.variable,
            f"{row.treated_mean:.{decimal_places}f}",
            f"{row.control_mean:.{decimal_places}f}",
            f"{row.difference:.{decimal_places}f}"
        ]
        if show_smd:
            parts.append(f"{row.smd:.{decimal_places}f}" if row.smd is not None else "-")
        if show_pvalue:
            parts.append(f"{row.p_value:.{decimal_places}f}" if row.p_value is not None else "-")
        lines.append(" & ".join(parts) + r" \\")

    lines.extend([
        r"\hline\hline",
        r"\end{tabular}",
        r"\begin{tablenotes}",
        r"\small"
    ])

    if notes is None:
        notes = "SMD = Standardized Mean Difference. $|SMD| < 0.1$ indicates good balance."
    lines.append(f"\\item {notes}")
    lines.extend([
        r"\end{tablenotes}",
        r"\end{table}"
    ])

    return "\n".join(lines)


def _format_balance_table_html(
    rows: List[BalanceTableRow],
    title: str,
    show_smd: bool,
    show_pvalue: bool,
    decimal_places: int,
    notes: Optional[str]
) -> str:
    """Create HTML balance table."""
    lines = [
        '<table class="balance-table">',
        f'<caption>{title}</caption>',
        '<thead>',
        '<tr>'
    ]

    # Header
    headers = ["Variable", "Treated Mean", "Control Mean", "Difference"]
    if show_smd:
        headers.append("SMD")
    if show_pvalue:
        headers.append("P-value")
    for h in headers:
        lines.append(f'<th>{h}</th>')
    lines.extend(['</tr>', '</thead>', '<tbody>'])

    # Data rows
    for row in rows:
        lines.append('<tr>')
        lines.append(f'<td>{row.variable}</td>')
        lines.append(f'<td>{row.treated_mean:.{decimal_places}f}</td>')
        lines.append(f'<td>{row.control_mean:.{decimal_places}f}</td>')
        lines.append(f'<td>{row.difference:.{decimal_places}f}</td>')
        if show_smd:
            smd_str = f"{row.smd:.{decimal_places}f}" if row.smd is not None else "-"
            lines.append(f'<td>{smd_str}</td>')
        if show_pvalue:
            pval_str = f"{row.p_value:.{decimal_places}f}" if row.p_value is not None else "-"
            lines.append(f'<td>{pval_str}</td>')
        lines.append('</tr>')

    lines.extend(['</tbody>', '</table>'])

    if notes is None:
        notes = "SMD = Standardized Mean Difference. |SMD| < 0.1 indicates good balance."
    lines.append(f'<p class="table-notes"><em>Notes: {notes}</em></p>')

    return "\n".join(lines)


def _format_balance_table_text(
    rows: List[BalanceTableRow],
    title: str,
    show_smd: bool,
    show_pvalue: bool,
    decimal_places: int,
    notes: Optional[str]
) -> str:
    """Create plain text balance table."""
    width = 80
    col_width = 12

    lines = [
        "=" * width,
        title.center(width),
        "=" * width
    ]

    # Header
    header = "Variable".ljust(20) + "Treated".center(col_width) + "Control".center(col_width) + "Diff.".center(col_width)
    if show_smd:
        header += "SMD".center(col_width)
    if show_pvalue:
        header += "P-value".center(col_width)
    lines.append(header)
    lines.append("-" * width)

    # Data
    for row in rows:
        line = row.variable[:18].ljust(20)
        line += f"{row.treated_mean:.{decimal_places}f}".center(col_width)
        line += f"{row.control_mean:.{decimal_places}f}".center(col_width)
        line += f"{row.difference:.{decimal_places}f}".center(col_width)
        if show_smd:
            smd_str = f"{row.smd:.{decimal_places}f}" if row.smd is not None else "-"
            line += smd_str.center(col_width)
        if show_pvalue:
            pval_str = f"{row.p_value:.{decimal_places}f}" if row.p_value is not None else "-"
            line += pval_str.center(col_width)
        lines.append(line)

    lines.append("=" * width)

    if notes is None:
        notes = "SMD = Standardized Mean Difference. |SMD| < 0.1 indicates good balance."
    lines.append(f"Notes: {notes}")

    return "\n".join(lines)


@dataclass
class EventStudyCoefficient:
    """Coefficient for event study table."""
    period: int
    coefficient: float
    std_error: float
    ci_lower: Optional[float] = None
    ci_upper: Optional[float] = None
    p_value: Optional[float] = None


def format_event_study_table(
    coefficients: List[Union[EventStudyCoefficient, Dict[str, Any]]],
    title: str = "Event Study Estimates",
    reference_period: int = -1,
    output_format: str = "markdown",
    decimal_places: int = 3,
    show_ci: bool = True,
    notes: Optional[str] = None
) -> str:
    """
    Format an event study coefficients table for DID analysis.

    Creates a table showing period-by-period treatment effects relative
    to a reference period (typically -1, the period before treatment).

    Parameters
    ----------
    coefficients : List[Union[EventStudyCoefficient, Dict[str, Any]]]
        List of coefficient results, either EventStudyCoefficient objects or dicts with:
        - period: int (relative time period)
        - coefficient: float
        - std_error: float
        - ci_lower: float (optional)
        - ci_upper: float (optional)
        - p_value: float (optional)
    title : str
        Table title
    reference_period : int
        The omitted reference period (default -1)
    output_format : str
        Output format: 'markdown', 'latex', 'text', or 'html'
    decimal_places : int
        Number of decimal places
    show_ci : bool
        Whether to show confidence intervals
    notes : Optional[str]
        Custom footnotes

    Returns
    -------
    str
        Formatted event study table
    """
    # Convert dicts to EventStudyCoefficient
    rows = []
    for item in coefficients:
        if isinstance(item, dict):
            rows.append(EventStudyCoefficient(
                period=item.get('period', 0),
                coefficient=item.get('coefficient', 0),
                std_error=item.get('std_error', 0),
                ci_lower=item.get('ci_lower'),
                ci_upper=item.get('ci_upper'),
                p_value=item.get('p_value')
            ))
        else:
            rows.append(item)

    # Sort by period
    rows.sort(key=lambda x: x.period)

    if output_format == "latex":
        return _format_event_study_latex(rows, title, reference_period, decimal_places, show_ci, notes)
    elif output_format == "html":
        return _format_event_study_html(rows, title, reference_period, decimal_places, show_ci, notes)
    elif output_format == "text":
        return _format_event_study_text(rows, title, reference_period, decimal_places, show_ci, notes)
    else:  # markdown
        return _format_event_study_markdown(rows, title, reference_period, decimal_places, show_ci, notes)


def _format_event_study_markdown(
    rows: List[EventStudyCoefficient],
    title: str,
    reference_period: int,
    decimal_places: int,
    show_ci: bool,
    notes: Optional[str]
) -> str:
    """Create markdown event study table."""
    lines = [f"### {title}", ""]

    # Header
    header_parts = ["| Period", "Coefficient", "Std. Error"]
    if show_ci:
        header_parts.append("95% CI")
    header = " | ".join(header_parts) + " |"
    lines.append(header)

    # Separator
    sep = "|:------" + "|:------:" * (len(header_parts) - 1) + "|"
    lines.append(sep)

    # Data rows
    for row in rows:
        if row.period == reference_period:
            parts = [f"| {row.period} (ref)", "0.000", "-"]
            if show_ci:
                parts.append("-")
        else:
            coef_str = format_coefficient(row.coefficient, row.std_error,
                                          row.p_value if row.p_value else 1.0, decimal_places)
            parts = [
                f"| {row.period}",
                coef_str,
                f"({row.std_error:.{decimal_places}f})"
            ]
            if show_ci:
                if row.ci_lower is not None and row.ci_upper is not None:
                    ci_str = f"[{row.ci_lower:.{decimal_places}f}, {row.ci_upper:.{decimal_places}f}]"
                else:
                    ci_str = "-"
                parts.append(ci_str)
        lines.append(" | ".join(parts) + " |")

    lines.append("")

    if notes is None:
        notes = f"Reference period: {reference_period}. *** p<0.01, ** p<0.05, * p<0.1"
    lines.append(f"*Notes: {notes}*")

    return "\n".join(lines)


def _format_event_study_latex(
    rows: List[EventStudyCoefficient],
    title: str,
    reference_period: int,
    decimal_places: int,
    show_ci: bool,
    notes: Optional[str]
) -> str:
    """Create LaTeX event study table."""
    n_cols = 3 + (1 if show_ci else 0)
    col_spec = "r" + "c" * (n_cols - 1)

    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        f"\\caption{{{title}}}",
        r"\begin{tabular}{" + col_spec + "}",
        r"\hline\hline"
    ]

    # Header
    header_parts = ["Period", "Coefficient", "Std. Error"]
    if show_ci:
        header_parts.append("95\\% CI")
    lines.append(" & ".join(header_parts) + r" \\")
    lines.append(r"\hline")

    # Data
    for row in rows:
        if row.period == reference_period:
            parts = [f"{row.period} (ref)", "0.000", "-"]
            if show_ci:
                parts.append("-")
        else:
            coef_str = format_coefficient(row.coefficient, row.std_error,
                                          row.p_value if row.p_value else 1.0, decimal_places)
            parts = [
                str(row.period),
                coef_str,
                f"({row.std_error:.{decimal_places}f})"
            ]
            if show_ci:
                if row.ci_lower is not None and row.ci_upper is not None:
                    ci_str = f"[{row.ci_lower:.{decimal_places}f}, {row.ci_upper:.{decimal_places}f}]"
                else:
                    ci_str = "-"
                parts.append(ci_str)
        lines.append(" & ".join(parts) + r" \\")

    lines.extend([
        r"\hline\hline",
        r"\end{tabular}",
        r"\begin{tablenotes}",
        r"\small"
    ])

    if notes is None:
        notes = f"Reference period: {reference_period}. *** p<0.01, ** p<0.05, * p<0.1"
    lines.append(f"\\item {notes}")
    lines.extend([r"\end{tablenotes}", r"\end{table}"])

    return "\n".join(lines)


def _format_event_study_html(
    rows: List[EventStudyCoefficient],
    title: str,
    reference_period: int,
    decimal_places: int,
    show_ci: bool,
    notes: Optional[str]
) -> str:
    """Create HTML event study table."""
    lines = [
        '<table class="event-study-table">',
        f'<caption>{title}</caption>',
        '<thead><tr>'
    ]

    headers = ["Period", "Coefficient", "Std. Error"]
    if show_ci:
        headers.append("95% CI")
    for h in headers:
        lines.append(f'<th>{h}</th>')
    lines.extend(['</tr></thead>', '<tbody>'])

    for row in rows:
        lines.append('<tr>')
        if row.period == reference_period:
            lines.append(f'<td>{row.period} (ref)</td>')
            lines.append('<td>0.000</td>')
            lines.append('<td>-</td>')
            if show_ci:
                lines.append('<td>-</td>')
        else:
            coef_str = format_coefficient(row.coefficient, row.std_error,
                                          row.p_value if row.p_value else 1.0, decimal_places)
            lines.append(f'<td>{row.period}</td>')
            lines.append(f'<td>{coef_str}</td>')
            lines.append(f'<td>({row.std_error:.{decimal_places}f})</td>')
            if show_ci:
                if row.ci_lower is not None and row.ci_upper is not None:
                    ci_str = f"[{row.ci_lower:.{decimal_places}f}, {row.ci_upper:.{decimal_places}f}]"
                else:
                    ci_str = "-"
                lines.append(f'<td>{ci_str}</td>')
        lines.append('</tr>')

    lines.extend(['</tbody>', '</table>'])

    if notes is None:
        notes = f"Reference period: {reference_period}. *** p<0.01, ** p<0.05, * p<0.1"
    lines.append(f'<p class="table-notes"><em>Notes: {notes}</em></p>')

    return "\n".join(lines)


def _format_event_study_text(
    rows: List[EventStudyCoefficient],
    title: str,
    reference_period: int,
    decimal_places: int,
    show_ci: bool,
    notes: Optional[str]
) -> str:
    """Create plain text event study table."""
    width = 70
    lines = [
        "=" * width,
        title.center(width),
        "=" * width
    ]

    header = "Period".center(10) + "Coefficient".center(15) + "Std. Error".center(15)
    if show_ci:
        header += "95% CI".center(25)
    lines.extend([header, "-" * width])

    for row in rows:
        if row.period == reference_period:
            line = f"{row.period} (ref)".center(10) + "0.000".center(15) + "-".center(15)
            if show_ci:
                line += "-".center(25)
        else:
            coef_str = format_coefficient(row.coefficient, row.std_error,
                                          row.p_value if row.p_value else 1.0, decimal_places)
            line = str(row.period).center(10) + coef_str.center(15) + f"({row.std_error:.{decimal_places}f})".center(15)
            if show_ci:
                if row.ci_lower is not None and row.ci_upper is not None:
                    ci_str = f"[{row.ci_lower:.{decimal_places}f}, {row.ci_upper:.{decimal_places}f}]"
                else:
                    ci_str = "-"
                line += ci_str.center(25)
        lines.append(line)

    lines.append("=" * width)

    if notes is None:
        notes = f"Reference period: {reference_period}. *** p<0.01, ** p<0.05, * p<0.1"
    lines.append(f"Notes: {notes}")

    return "\n".join(lines)


@dataclass
class FirstStageResult:
    """First stage regression result for IV table."""
    instrument: str
    coefficient: float
    std_error: float
    t_statistic: Optional[float] = None
    p_value: Optional[float] = None


def format_first_stage_table(
    results: List[Union[FirstStageResult, Dict[str, Any]]],
    f_statistic: float,
    r_squared: float,
    n_obs: int,
    title: str = "First Stage Results",
    endogenous_var: str = "Endogenous Variable",
    output_format: str = "markdown",
    decimal_places: int = 3,
    notes: Optional[str] = None
) -> str:
    """
    Format an IV first stage regression table.

    Creates a table showing how instruments predict the endogenous variable,
    including the F-statistic for weak instrument diagnostics.

    Parameters
    ----------
    results : List[Union[FirstStageResult, Dict[str, Any]]]
        List of instrument coefficients, either FirstStageResult or dicts with:
        - instrument: str (instrument name)
        - coefficient: float
        - std_error: float
        - t_statistic: float (optional)
        - p_value: float (optional)
    f_statistic : float
        First-stage F-statistic
    r_squared : float
        First-stage R-squared
    n_obs : int
        Number of observations
    title : str
        Table title
    endogenous_var : str
        Name of the endogenous variable
    output_format : str
        Output format: 'markdown', 'latex', 'text', or 'html'
    decimal_places : int
        Number of decimal places
    notes : Optional[str]
        Custom footnotes

    Returns
    -------
    str
        Formatted first stage table
    """
    # Convert dicts to FirstStageResult
    rows = []
    for item in results:
        if isinstance(item, dict):
            rows.append(FirstStageResult(
                instrument=item.get('instrument', ''),
                coefficient=item.get('coefficient', 0),
                std_error=item.get('std_error', 0),
                t_statistic=item.get('t_statistic'),
                p_value=item.get('p_value')
            ))
        else:
            rows.append(item)

    if output_format == "latex":
        return _format_first_stage_latex(rows, f_statistic, r_squared, n_obs, title,
                                          endogenous_var, decimal_places, notes)
    elif output_format == "html":
        return _format_first_stage_html(rows, f_statistic, r_squared, n_obs, title,
                                         endogenous_var, decimal_places, notes)
    elif output_format == "text":
        return _format_first_stage_text(rows, f_statistic, r_squared, n_obs, title,
                                         endogenous_var, decimal_places, notes)
    else:  # markdown
        return _format_first_stage_markdown(rows, f_statistic, r_squared, n_obs, title,
                                             endogenous_var, decimal_places, notes)


def _format_first_stage_markdown(
    rows: List[FirstStageResult],
    f_statistic: float,
    r_squared: float,
    n_obs: int,
    title: str,
    endogenous_var: str,
    decimal_places: int,
    notes: Optional[str]
) -> str:
    """Create markdown first stage table."""
    lines = [f"### {title}", "", f"**Dependent Variable**: {endogenous_var}", ""]

    lines.append("| Instrument | Coefficient | Std. Error | t-stat |")
    lines.append("|:-----------|:----------:|:----------:|:------:|")

    for row in rows:
        pval = row.p_value if row.p_value is not None else 1.0
        coef_str = format_coefficient(row.coefficient, row.std_error, pval, decimal_places)
        t_str = f"{row.t_statistic:.2f}" if row.t_statistic is not None else "-"
        lines.append(f"| {row.instrument} | {coef_str} | ({row.std_error:.{decimal_places}f}) | {t_str} |")

    lines.extend(["", "| Statistic | Value |", "|:----------|------:|"])
    lines.append(f"| F-statistic | {f_statistic:.2f} |")
    lines.append(f"| R-squared | {r_squared:.{decimal_places}f} |")
    lines.append(f"| Observations | {n_obs:,} |")

    lines.append("")

    # Weak instrument warning
    weak_iv_warning = ""
    if f_statistic < 10:
        weak_iv_warning = " **WARNING: F < 10 suggests weak instruments.**"

    if notes is None:
        notes = f"*** p<0.01, ** p<0.05, * p<0.1. Stock-Yogo critical value (10% maximal IV size): 16.38.{weak_iv_warning}"
    lines.append(f"*Notes: {notes}*")

    return "\n".join(lines)


def _format_first_stage_latex(
    rows: List[FirstStageResult],
    f_statistic: float,
    r_squared: float,
    n_obs: int,
    title: str,
    endogenous_var: str,
    decimal_places: int,
    notes: Optional[str]
) -> str:
    """Create LaTeX first stage table."""
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        f"\\caption{{{title}}}",
        f"\\label{{tab:first_stage}}",
        r"\begin{tabular}{lccc}",
        r"\hline\hline",
        f"\\multicolumn{{4}}{{l}}{{Dependent Variable: {endogenous_var}}} \\\\",
        r"\hline",
        r"Instrument & Coefficient & Std. Error & t-stat \\"
    ]

    for row in rows:
        pval = row.p_value if row.p_value is not None else 1.0
        coef_str = format_coefficient(row.coefficient, row.std_error, pval, decimal_places)
        t_str = f"{row.t_statistic:.2f}" if row.t_statistic is not None else "-"
        lines.append(f"{row.instrument} & {coef_str} & ({row.std_error:.{decimal_places}f}) & {t_str} \\\\")

    lines.extend([
        r"\hline",
        f"F-statistic & \\multicolumn{{3}}{{c}}{{{f_statistic:.2f}}} \\\\",
        f"R-squared & \\multicolumn{{3}}{{c}}{{{r_squared:.{decimal_places}f}}} \\\\",
        f"Observations & \\multicolumn{{3}}{{c}}{{{n_obs:,}}} \\\\",
        r"\hline\hline",
        r"\end{tabular}",
        r"\begin{tablenotes}",
        r"\small"
    ])

    if notes is None:
        weak_warning = " F < 10 suggests weak instruments." if f_statistic < 10 else ""
        notes = f"*** p<0.01, ** p<0.05, * p<0.1.{weak_warning}"
    lines.append(f"\\item {notes}")
    lines.extend([r"\end{tablenotes}", r"\end{table}"])

    return "\n".join(lines)


def _format_first_stage_html(
    rows: List[FirstStageResult],
    f_statistic: float,
    r_squared: float,
    n_obs: int,
    title: str,
    endogenous_var: str,
    decimal_places: int,
    notes: Optional[str]
) -> str:
    """Create HTML first stage table."""
    lines = [
        '<table class="first-stage-table">',
        f'<caption>{title}</caption>',
        f'<tr><td colspan="4"><strong>Dependent Variable:</strong> {endogenous_var}</td></tr>',
        '<thead><tr>',
        '<th>Instrument</th><th>Coefficient</th><th>Std. Error</th><th>t-stat</th>',
        '</tr></thead>',
        '<tbody>'
    ]

    for row in rows:
        pval = row.p_value if row.p_value is not None else 1.0
        coef_str = format_coefficient(row.coefficient, row.std_error, pval, decimal_places)
        t_str = f"{row.t_statistic:.2f}" if row.t_statistic is not None else "-"
        lines.append(f'<tr><td>{row.instrument}</td><td>{coef_str}</td><td>({row.std_error:.{decimal_places}f})</td><td>{t_str}</td></tr>')

    lines.extend([
        '</tbody>',
        '<tfoot>',
        f'<tr><td>F-statistic</td><td colspan="3">{f_statistic:.2f}</td></tr>',
        f'<tr><td>R-squared</td><td colspan="3">{r_squared:.{decimal_places}f}</td></tr>',
        f'<tr><td>Observations</td><td colspan="3">{n_obs:,}</td></tr>',
        '</tfoot>',
        '</table>'
    ])

    if notes is None:
        weak_warning = " <strong>F < 10 suggests weak instruments.</strong>" if f_statistic < 10 else ""
        notes = f"*** p<0.01, ** p<0.05, * p<0.1.{weak_warning}"
    lines.append(f'<p class="table-notes"><em>Notes: {notes}</em></p>')

    return "\n".join(lines)


def _format_first_stage_text(
    rows: List[FirstStageResult],
    f_statistic: float,
    r_squared: float,
    n_obs: int,
    title: str,
    endogenous_var: str,
    decimal_places: int,
    notes: Optional[str]
) -> str:
    """Create plain text first stage table."""
    width = 65
    lines = [
        "=" * width,
        title.center(width),
        "=" * width,
        f"Dependent Variable: {endogenous_var}",
        "-" * width
    ]

    header = "Instrument".ljust(20) + "Coefficient".center(15) + "Std. Error".center(15) + "t-stat".center(10)
    lines.extend([header, "-" * width])

    for row in rows:
        pval = row.p_value if row.p_value is not None else 1.0
        coef_str = format_coefficient(row.coefficient, row.std_error, pval, decimal_places)
        t_str = f"{row.t_statistic:.2f}" if row.t_statistic is not None else "-"
        line = row.instrument[:18].ljust(20) + coef_str.center(15) + f"({row.std_error:.{decimal_places}f})".center(15) + t_str.center(10)
        lines.append(line)

    lines.extend([
        "-" * width,
        f"F-statistic: {f_statistic:.2f}",
        f"R-squared: {r_squared:.{decimal_places}f}",
        f"Observations: {n_obs:,}",
        "=" * width
    ])

    if notes is None:
        weak_warning = " WARNING: F < 10 suggests weak instruments." if f_statistic < 10 else ""
        notes = f"*** p<0.01, ** p<0.05, * p<0.1.{weak_warning}"
    lines.append(f"Notes: {notes}")

    return "\n".join(lines)


def format_panel_table(
    results: List[Dict[str, Any]],
    column_names: Optional[List[str]] = None,
    title: str = "Panel Regression Results",
    fe_indicators: Optional[Dict[str, List[bool]]] = None,
    cluster_indicator: Optional[List[str]] = None,
    output_format: str = "markdown",
    decimal_places: int = 3,
    notes: Optional[str] = None
) -> str:
    """
    Format a panel regression table with fixed effects indicators.

    Creates a publication-quality table for panel data regressions,
    showing which fixed effects are included in each specification.

    Parameters
    ----------
    results : List[Dict[str, Any]]
        List of regression results, one per column. Each dict should have:
        - treatment_effect: float
        - treatment_se: float
        - treatment_pval: float
        - n_obs: int
        - r_squared: float
        - r_squared_within: float (optional)
        - additional_vars: Dict[str, Tuple[float, float, float]] (optional)
          mapping var_name to (coef, se, pval)
    column_names : Optional[List[str]]
        Column headers (default: (1), (2), etc.)
    title : str
        Table title
    fe_indicators : Optional[Dict[str, List[bool]]]
        Fixed effects indicators, e.g., {'Unit FE': [False, True, True], 'Time FE': [False, False, True]}
    cluster_indicator : Optional[List[str]]
        Clustering level for each column, e.g., ['None', 'Unit', 'Unit']
    output_format : str
        Output format: 'markdown', 'latex', 'text', or 'html'
    decimal_places : int
        Number of decimal places
    notes : Optional[str]
        Custom footnotes

    Returns
    -------
    str
        Formatted panel regression table
    """
    n_cols = len(results)

    if column_names is None:
        column_names = [f"({i+1})" for i in range(n_cols)]

    if output_format == "latex":
        return _format_panel_table_latex(results, column_names, title, fe_indicators,
                                          cluster_indicator, decimal_places, notes)
    elif output_format == "html":
        return _format_panel_table_html(results, column_names, title, fe_indicators,
                                         cluster_indicator, decimal_places, notes)
    elif output_format == "text":
        return _format_panel_table_text(results, column_names, title, fe_indicators,
                                         cluster_indicator, decimal_places, notes)
    else:  # markdown
        return _format_panel_table_markdown(results, column_names, title, fe_indicators,
                                             cluster_indicator, decimal_places, notes)


def _format_panel_table_markdown(
    results: List[Dict[str, Any]],
    column_names: List[str],
    title: str,
    fe_indicators: Optional[Dict[str, List[bool]]],
    cluster_indicator: Optional[List[str]],
    decimal_places: int,
    notes: Optional[str]
) -> str:
    """Create markdown panel table."""
    lines = [f"### {title}", ""]

    # Header
    header = "| Variable | " + " | ".join(column_names) + " |"
    sep = "|:---------|" + "|".join([":------:" for _ in column_names]) + "|"
    lines.extend([header, sep])

    # Treatment effect
    effect_row = "| Treatment | "
    for r in results:
        coef_str = format_coefficient(
            r.get('treatment_effect', 0),
            r.get('treatment_se', 0),
            r.get('treatment_pval', 1),
            decimal_places
        )
        effect_row += f"{coef_str} | "
    lines.append(effect_row)

    # SE row
    se_row = "| | "
    for r in results:
        se_row += f"({r.get('treatment_se', 0):.{decimal_places}f}) | "
    lines.append(se_row)

    # Additional variables if present
    all_vars = set()
    for r in results:
        if 'additional_vars' in r:
            all_vars.update(r['additional_vars'].keys())

    for var in sorted(all_vars):
        var_row = f"| {var} | "
        se_row = "| | "
        for r in results:
            if 'additional_vars' in r and var in r['additional_vars']:
                coef, se, pval = r['additional_vars'][var]
                coef_str = format_coefficient(coef, se, pval, decimal_places)
                var_row += f"{coef_str} | "
                se_row += f"({se:.{decimal_places}f}) | "
            else:
                var_row += "- | "
                se_row += "- | "
        lines.extend([var_row, se_row])

    lines.append("| | " + " | ".join(["" for _ in column_names]) + " |")

    # Fixed effects indicators
    if fe_indicators:
        for fe_name, indicators in fe_indicators.items():
            fe_row = f"| {fe_name} | "
            for i, ind in enumerate(indicators):
                fe_row += ("Yes | " if ind else "No | ")
            lines.append(fe_row)

    # Clustering
    if cluster_indicator:
        cluster_row = "| Clustering | "
        for cl in cluster_indicator:
            cluster_row += f"{cl} | "
        lines.append(cluster_row)

    lines.append("| | " + " | ".join(["" for _ in column_names]) + " |")

    # Statistics
    obs_row = "| Observations | "
    for r in results:
        obs_row += f"{r.get('n_obs', 0):,} | "
    lines.append(obs_row)

    r2_row = "| R-squared | "
    for r in results:
        r2_row += f"{r.get('r_squared', 0):.{decimal_places}f} | "
    lines.append(r2_row)

    # Within R-squared if available
    if any('r_squared_within' in r for r in results):
        r2w_row = "| R-squared (within) | "
        for r in results:
            r2w = r.get('r_squared_within')
            r2w_row += (f"{r2w:.{decimal_places}f} | " if r2w is not None else "- | ")
        lines.append(r2w_row)

    lines.append("")

    if notes is None:
        notes = "Standard errors in parentheses. *** p<0.01, ** p<0.05, * p<0.1"
    lines.append(f"*Notes: {notes}*")

    return "\n".join(lines)


def _format_panel_table_latex(
    results: List[Dict[str, Any]],
    column_names: List[str],
    title: str,
    fe_indicators: Optional[Dict[str, List[bool]]],
    cluster_indicator: Optional[List[str]],
    decimal_places: int,
    notes: Optional[str]
) -> str:
    """Create LaTeX panel table."""
    n_cols = len(results)
    col_spec = "l" + "c" * n_cols

    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        f"\\caption{{{title}}}",
        r"\begin{tabular}{" + col_spec + "}",
        r"\hline\hline"
    ]

    # Header
    header = " & " + " & ".join(column_names) + r" \\"
    lines.extend([header, r"\hline"])

    # Treatment
    effect_row = "Treatment & "
    for r in results:
        coef_str = format_coefficient(
            r.get('treatment_effect', 0),
            r.get('treatment_se', 0),
            r.get('treatment_pval', 1),
            decimal_places
        )
        effect_row += f"{coef_str} & "
    effect_row = effect_row[:-3] + r" \\"
    lines.append(effect_row)

    se_row = " & "
    for r in results:
        se_row += f"({r.get('treatment_se', 0):.{decimal_places}f}) & "
    se_row = se_row[:-3] + r" \\"
    lines.append(se_row)

    lines.append(r"\\")

    # Fixed effects
    if fe_indicators:
        for fe_name, indicators in fe_indicators.items():
            fe_row = f"{fe_name} & "
            for ind in indicators:
                fe_row += ("Yes & " if ind else "No & ")
            fe_row = fe_row[:-3] + r" \\"
            lines.append(fe_row)

    # Clustering
    if cluster_indicator:
        cluster_row = "Clustering & "
        for cl in cluster_indicator:
            cluster_row += f"{cl} & "
        cluster_row = cluster_row[:-3] + r" \\"
        lines.append(cluster_row)

    lines.append(r"\\")

    # Statistics
    obs_row = "Observations & "
    for r in results:
        obs_row += f"{r.get('n_obs', 0):,} & "
    obs_row = obs_row[:-3] + r" \\"
    lines.append(obs_row)

    r2_row = "R-squared & "
    for r in results:
        r2_row += f"{r.get('r_squared', 0):.{decimal_places}f} & "
    r2_row = r2_row[:-3] + r" \\"
    lines.append(r2_row)

    lines.extend([
        r"\hline\hline",
        r"\end{tabular}",
        r"\begin{tablenotes}",
        r"\small"
    ])

    if notes is None:
        notes = "Standard errors in parentheses. *** p<0.01, ** p<0.05, * p<0.1"
    lines.append(f"\\item {notes}")
    lines.extend([r"\end{tablenotes}", r"\end{table}"])

    return "\n".join(lines)


def _format_panel_table_html(
    results: List[Dict[str, Any]],
    column_names: List[str],
    title: str,
    fe_indicators: Optional[Dict[str, List[bool]]],
    cluster_indicator: Optional[List[str]],
    decimal_places: int,
    notes: Optional[str]
) -> str:
    """Create HTML panel table."""
    lines = [
        '<table class="panel-table">',
        f'<caption>{title}</caption>',
        '<thead><tr>',
        '<th>Variable</th>'
    ]

    for name in column_names:
        lines.append(f'<th>{name}</th>')
    lines.extend(['</tr></thead>', '<tbody>'])

    # Treatment
    lines.append('<tr><td>Treatment</td>')
    for r in results:
        coef_str = format_coefficient(
            r.get('treatment_effect', 0),
            r.get('treatment_se', 0),
            r.get('treatment_pval', 1),
            decimal_places
        )
        lines.append(f'<td>{coef_str}</td>')
    lines.append('</tr>')

    lines.append('<tr><td></td>')
    for r in results:
        lines.append(f'<td>({r.get("treatment_se", 0):.{decimal_places}f})</td>')
    lines.append('</tr>')

    # Fixed effects
    if fe_indicators:
        for fe_name, indicators in fe_indicators.items():
            lines.append(f'<tr><td>{fe_name}</td>')
            for ind in indicators:
                lines.append(f'<td>{"Yes" if ind else "No"}</td>')
            lines.append('</tr>')

    # Clustering
    if cluster_indicator:
        lines.append('<tr><td>Clustering</td>')
        for cl in cluster_indicator:
            lines.append(f'<td>{cl}</td>')
        lines.append('</tr>')

    # Statistics
    lines.append('<tr><td>Observations</td>')
    for r in results:
        lines.append(f'<td>{r.get("n_obs", 0):,}</td>')
    lines.append('</tr>')

    lines.append('<tr><td>R-squared</td>')
    for r in results:
        lines.append(f'<td>{r.get("r_squared", 0):.{decimal_places}f}</td>')
    lines.append('</tr>')

    lines.extend(['</tbody>', '</table>'])

    if notes is None:
        notes = "Standard errors in parentheses. *** p<0.01, ** p<0.05, * p<0.1"
    lines.append(f'<p class="table-notes"><em>Notes: {notes}</em></p>')

    return "\n".join(lines)


def _format_panel_table_text(
    results: List[Dict[str, Any]],
    column_names: List[str],
    title: str,
    fe_indicators: Optional[Dict[str, List[bool]]],
    cluster_indicator: Optional[List[str]],
    decimal_places: int,
    notes: Optional[str]
) -> str:
    """Create plain text panel table."""
    col_width = 12
    width = 20 + col_width * len(column_names)

    lines = [
        "=" * width,
        title.center(width),
        "=" * width
    ]

    header = "Variable".ljust(20) + "".join(c.center(col_width) for c in column_names)
    lines.extend([header, "-" * width])

    # Treatment
    effect_row = "Treatment".ljust(20)
    for r in results:
        coef_str = format_coefficient(
            r.get('treatment_effect', 0),
            r.get('treatment_se', 0),
            r.get('treatment_pval', 1),
            decimal_places
        )
        effect_row += coef_str.center(col_width)
    lines.append(effect_row)

    se_row = "".ljust(20)
    for r in results:
        se_row += f"({r.get('treatment_se', 0):.{decimal_places}f})".center(col_width)
    lines.append(se_row)

    lines.append("")

    # Fixed effects
    if fe_indicators:
        for fe_name, indicators in fe_indicators.items():
            fe_row = fe_name.ljust(20)
            for ind in indicators:
                fe_row += ("Yes" if ind else "No").center(col_width)
            lines.append(fe_row)

    # Clustering
    if cluster_indicator:
        cluster_row = "Clustering".ljust(20)
        for cl in cluster_indicator:
            cluster_row += cl.center(col_width)
        lines.append(cluster_row)

    lines.append("")

    # Statistics
    obs_row = "Observations".ljust(20)
    for r in results:
        obs_row += f"{r.get('n_obs', 0):,}".center(col_width)
    lines.append(obs_row)

    r2_row = "R-squared".ljust(20)
    for r in results:
        r2_row += f"{r.get('r_squared', 0):.{decimal_places}f}".center(col_width)
    lines.append(r2_row)

    lines.append("=" * width)

    if notes is None:
        notes = "Standard errors in parentheses. *** p<0.01, ** p<0.05, * p<0.1"
    lines.append(f"Notes: {notes}")

    return "\n".join(lines)


def export_to_latex(
    table_content: str,
    output_format: str = "standalone",
    document_class: str = "article",
    packages: Optional[List[str]] = None
) -> str:
    """
    Export any table to a complete LaTeX document or fragment.

    Parameters
    ----------
    table_content : str
        Table content (can be from any format_* function with output_format='latex')
    output_format : str
        Output type:
        - 'standalone': Complete compilable document
        - 'fragment': Just the table (for inclusion in larger documents)
        - 'beamer': Slide-ready format
    document_class : str
        LaTeX document class for standalone output
    packages : Optional[List[str]]
        Additional LaTeX packages to include

    Returns
    -------
    str
        LaTeX output ready for compilation or inclusion
    """
    if output_format == "fragment":
        return table_content

    # Default packages
    default_packages = [
        "booktabs",
        "threeparttable",
        "adjustbox",
        "array"
    ]

    if packages:
        default_packages.extend(packages)

    if output_format == "beamer":
        lines = [
            r"\documentclass{beamer}",
        ]
    else:
        lines = [
            f"\\documentclass{{{document_class}}}",
        ]

    # Add packages
    for pkg in default_packages:
        lines.append(f"\\usepackage{{{pkg}}}")

    lines.extend([
        "",
        r"\begin{document}",
        ""
    ])

    if output_format == "beamer":
        lines.extend([
            r"\begin{frame}",
            r"\frametitle{Results}",
            r"\centering",
            r"\scalebox{0.8}{",
            table_content,
            r"}",
            r"\end{frame}"
        ])
    else:
        lines.append(table_content)

    lines.extend([
        "",
        r"\end{document}"
    ])

    return "\n".join(lines)


def export_to_html(
    table_content: str,
    output_format: str = "standalone",
    title: str = "Results",
    css_style: Optional[str] = None
) -> str:
    """
    Export any table to a complete HTML document or styled fragment.

    Parameters
    ----------
    table_content : str
        Table content (can be from any format_* function with output_format='html')
    output_format : str
        Output type:
        - 'standalone': Complete HTML document with styling
        - 'fragment': Just the table with inline styles
        - 'notebook': Jupyter notebook compatible
    title : str
        Page title for standalone output
    css_style : Optional[str]
        Custom CSS to include

    Returns
    -------
    str
        HTML output ready for display or inclusion
    """
    # Default CSS for publication-quality tables
    default_css = """
    <style>
    table {
        border-collapse: collapse;
        margin: 20px auto;
        font-family: 'Times New Roman', Times, serif;
        font-size: 12pt;
    }
    caption {
        font-weight: bold;
        margin-bottom: 10px;
        font-size: 14pt;
    }
    th, td {
        border: none;
        padding: 8px 12px;
        text-align: center;
    }
    thead tr {
        border-bottom: 2px solid black;
    }
    tbody tr:first-child td {
        padding-top: 12px;
    }
    tfoot tr:first-child td {
        border-top: 1px solid black;
        padding-top: 8px;
    }
    tfoot tr:last-child td {
        border-bottom: 2px solid black;
    }
    .table-notes {
        font-size: 10pt;
        text-align: left;
        max-width: 600px;
        margin: 10px auto;
    }
    /* Column alignment */
    td:first-child, th:first-child {
        text-align: left;
    }
    </style>
    """

    if css_style:
        default_css = f"<style>\n{css_style}\n</style>"

    if output_format == "fragment":
        return f"{default_css}\n{table_content}"

    if output_format == "notebook":
        # IPython display compatible
        return f"""
        <div>
        {default_css}
        {table_content}
        </div>
        """

    # Standalone HTML document
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    {default_css}
</head>
<body>
    {table_content}
</body>
</html>"""

    return html
