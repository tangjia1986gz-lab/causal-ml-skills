#!/usr/bin/env python3
"""
Format Tables for Economics Papers

Converts regression output from various sources (Stata, R, Python)
to publication-quality LaTeX tables in AER style.

Usage:
    python format_tables.py --input stata_output.txt --style aer
    python format_tables.py --input results.csv --style qje --output table1.tex
"""

import argparse
import re
import sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Dict, List
import csv


@dataclass
class RegressionColumn:
    """Represents one regression specification."""
    name: str = ""
    dependent_var: str = ""
    coefficients: Dict[str, float] = field(default_factory=dict)
    std_errors: Dict[str, float] = field(default_factory=dict)
    p_values: Dict[str, float] = field(default_factory=dict)
    n_obs: Optional[int] = None
    r_squared: Optional[float] = None
    adj_r_squared: Optional[float] = None
    controls: bool = False
    fixed_effects: List[str] = field(default_factory=list)


@dataclass
class RegressionTable:
    """Represents a complete regression table."""
    title: str = "Regression Results"
    columns: List[RegressionColumn] = field(default_factory=list)
    notes: str = ""
    label: str = "tab:results"


class TableFormatter:
    """Format regression tables for publication."""

    # Star thresholds
    STARS = {
        0.01: '***',
        0.05: '**',
        0.10: '*'
    }

    def __init__(self, style: str = 'aer'):
        self.style = style
        self.decimal_places = 3

    def format_coefficient(self, coef: float, se: float,
                          p_value: Optional[float] = None) -> str:
        """Format a coefficient with standard error."""
        # Format coefficient
        coef_str = f"{coef:.{self.decimal_places}f}"

        # Add stars based on p-value
        if p_value is not None:
            stars = ''
            for threshold, star in sorted(self.STARS.items()):
                if p_value < threshold:
                    stars = star
                    break
            coef_str += stars

        # Format standard error
        se_str = f"({se:.{self.decimal_places}f})"

        return coef_str, se_str

    def format_number(self, value: float, decimals: Optional[int] = None) -> str:
        """Format a number with appropriate decimals."""
        if decimals is None:
            decimals = self.decimal_places

        if value >= 1000:
            return f"{int(value):,}"
        else:
            return f"{value:.{decimals}f}"

    def to_latex(self, table: RegressionTable) -> str:
        """Convert table to LaTeX format."""
        lines = []

        n_cols = len(table.columns)
        col_spec = 'l' + 'c' * n_cols

        # Begin table
        lines.append(r'\begin{table}[htbp]')
        lines.append(r'\centering')
        lines.append(f'\\caption{{{table.title}}}')
        lines.append(f'\\label{{{table.label}}}')

        if self.style == 'aer':
            lines.append(r'\begin{tabular}{' + col_spec + '}')
            lines.append(r'\toprule')
        else:
            lines.append(r'\begin{tabular}{' + col_spec + '}')
            lines.append(r'\hline\hline')

        # Header row with column numbers
        header1 = ' & ' + ' & '.join(f'({i+1})' for i in range(n_cols)) + r' \\'
        lines.append(header1)

        # Header row with column names if any
        if any(col.name for col in table.columns):
            header2 = ' & ' + ' & '.join(col.name or '' for col in table.columns) + r' \\'
            lines.append(header2)

        if self.style == 'aer':
            lines.append(r'\midrule')
        else:
            lines.append(r'\hline')

        # Get all unique variables
        all_vars = []
        for col in table.columns:
            for var in col.coefficients.keys():
                if var not in all_vars:
                    all_vars.append(var)

        # Output coefficients
        for var in all_vars:
            # Coefficient row
            coef_row = [self._format_variable_name(var)]
            se_row = ['']

            for col in table.columns:
                if var in col.coefficients:
                    coef = col.coefficients[var]
                    se = col.std_errors.get(var, 0)
                    p = col.p_values.get(var)

                    coef_str, se_str = self.format_coefficient(coef, se, p)
                    coef_row.append(coef_str)
                    se_row.append(se_str)
                else:
                    coef_row.append('')
                    se_row.append('')

            lines.append(' & '.join(coef_row) + r' \\')
            lines.append(' & '.join(se_row) + r' \\')
            lines.append('')  # Blank line between variables

        # Separator before statistics
        if self.style == 'aer':
            lines.append(r'\midrule')
        else:
            lines.append(r'\hline')

        # Controls row
        has_controls = any(col.controls for col in table.columns)
        if has_controls:
            controls_row = ['Controls'] + [
                'Yes' if col.controls else 'No' for col in table.columns
            ]
            lines.append(' & '.join(controls_row) + r' \\')

        # Fixed effects rows
        all_fe = set()
        for col in table.columns:
            all_fe.update(col.fixed_effects)

        for fe in sorted(all_fe):
            fe_row = [f'{fe} FE'] + [
                'Yes' if fe in col.fixed_effects else 'No'
                for col in table.columns
            ]
            lines.append(' & '.join(fe_row) + r' \\')

        # Observations
        obs_row = ['Observations'] + [
            self.format_number(col.n_obs, 0) if col.n_obs else ''
            for col in table.columns
        ]
        lines.append(' & '.join(obs_row) + r' \\')

        # R-squared
        r2_row = ['R-squared'] + [
            self.format_number(col.r_squared) if col.r_squared else ''
            for col in table.columns
        ]
        lines.append(' & '.join(r2_row) + r' \\')

        # End table
        if self.style == 'aer':
            lines.append(r'\bottomrule')
        else:
            lines.append(r'\hline\hline')

        lines.append(r'\end{tabular}')

        # Notes
        if table.notes or True:  # Always add notes
            default_notes = (
                "Standard errors in parentheses. "
                "*** p<0.01, ** p<0.05, * p<0.1."
            )
            notes_text = table.notes if table.notes else default_notes

            if self.style == 'aer':
                lines.append(r'\begin{tablenotes}')
                lines.append(r'\small')
                lines.append(f'\\item Notes: {notes_text}')
                lines.append(r'\end{tablenotes}')
            else:
                lines.append(r'\begin{minipage}{\textwidth}')
                lines.append(r'\footnotesize')
                lines.append(f'Notes: {notes_text}')
                lines.append(r'\end{minipage}')

        lines.append(r'\end{table}')

        return '\n'.join(lines)

    def _format_variable_name(self, name: str) -> str:
        """Convert variable names to readable labels."""
        # Common transformations
        name = name.replace('_', ' ')
        name = name.replace('ln ', 'Log ')
        name = name.replace('log ', 'Log ')
        name = name.title()
        return name


class StataParser:
    """Parse Stata regression output."""

    def parse(self, text: str) -> RegressionTable:
        """Parse Stata output text."""
        table = RegressionTable()
        col = RegressionColumn()

        lines = text.strip().split('\n')

        in_results = False
        for line in lines:
            line = line.strip()

            # Look for coefficient lines
            # Typical format: varname | coef stderr t P>|t| [95% Conf. Interval]
            coef_match = re.match(
                r'(\w+)\s+\|\s+([-\d.]+)\s+([\d.]+)\s+([-\d.]+)\s+([\d.]+)',
                line
            )

            if coef_match:
                var_name = coef_match.group(1)
                coefficient = float(coef_match.group(2))
                std_error = float(coef_match.group(3))
                p_value = float(coef_match.group(5))

                col.coefficients[var_name] = coefficient
                col.std_errors[var_name] = std_error
                col.p_values[var_name] = p_value

            # Look for N
            n_match = re.search(r'Number of obs\s*=\s*([\d,]+)', line)
            if n_match:
                col.n_obs = int(n_match.group(1).replace(',', ''))

            # Look for R-squared
            r2_match = re.search(r'R-squared\s*=\s*([\d.]+)', line)
            if r2_match:
                col.r_squared = float(r2_match.group(1))

        table.columns.append(col)
        return table


class CSVParser:
    """Parse CSV regression output."""

    def parse(self, filepath: Path) -> RegressionTable:
        """Parse CSV file with regression results."""
        table = RegressionTable()

        with open(filepath, 'r') as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        if not rows:
            return table

        # Determine number of columns from headers
        # Expected format: variable, coef1, se1, coef2, se2, ...
        headers = list(rows[0].keys())

        # Find coefficient columns
        coef_cols = [h for h in headers if h.startswith('coef') or h.startswith('b_')]
        n_specs = len(coef_cols)

        for i in range(n_specs):
            col = RegressionColumn(name=f"({i+1})")
            table.columns.append(col)

        for row in rows:
            var_name = row.get('variable', row.get('var', ''))

            if var_name.lower() in ['n', 'observations', 'obs']:
                for i, col in enumerate(table.columns):
                    val = row.get(f'coef{i+1}', row.get(f'b_{i+1}', ''))
                    if val:
                        col.n_obs = int(float(val))
            elif var_name.lower() in ['r2', 'r_squared', 'rsquared']:
                for i, col in enumerate(table.columns):
                    val = row.get(f'coef{i+1}', row.get(f'b_{i+1}', ''))
                    if val:
                        col.r_squared = float(val)
            elif var_name:
                for i, col in enumerate(table.columns):
                    coef_val = row.get(f'coef{i+1}', row.get(f'b_{i+1}', ''))
                    se_val = row.get(f'se{i+1}', row.get(f'se_{i+1}', ''))
                    p_val = row.get(f'p{i+1}', row.get(f'p_{i+1}', ''))

                    if coef_val:
                        col.coefficients[var_name] = float(coef_val)
                        if se_val:
                            col.std_errors[var_name] = float(se_val)
                        if p_val:
                            col.p_values[var_name] = float(p_val)

        return table


def create_example_table() -> RegressionTable:
    """Create an example table for demonstration."""
    table = RegressionTable(
        title="Effect of Education on Earnings",
        label="tab:main_results"
    )

    # Column 1: OLS
    col1 = RegressionColumn(
        name="OLS",
        coefficients={
            "years_education": 0.082,
            "experience": 0.043,
            "female": -0.156
        },
        std_errors={
            "years_education": 0.012,
            "experience": 0.003,
            "female": 0.018
        },
        p_values={
            "years_education": 0.001,
            "experience": 0.001,
            "female": 0.001
        },
        n_obs=50000,
        r_squared=0.234,
        controls=False
    )

    # Column 2: With controls
    col2 = RegressionColumn(
        name="OLS",
        coefficients={
            "years_education": 0.087,
            "experience": 0.041,
            "female": -0.154
        },
        std_errors={
            "years_education": 0.014,
            "experience": 0.004,
            "female": 0.019
        },
        p_values={
            "years_education": 0.001,
            "experience": 0.001,
            "female": 0.001
        },
        n_obs=50000,
        r_squared=0.267,
        controls=True
    )

    # Column 3: IV
    col3 = RegressionColumn(
        name="IV",
        coefficients={
            "years_education": 0.092,
            "experience": 0.041,
            "female": -0.152
        },
        std_errors={
            "years_education": 0.024,
            "experience": 0.004,
            "female": 0.021
        },
        p_values={
            "years_education": 0.001,
            "experience": 0.001,
            "female": 0.001
        },
        n_obs=50000,
        r_squared=0.198,
        controls=True,
        fixed_effects=["State"]
    )

    table.columns = [col1, col2, col3]
    table.notes = (
        "Standard errors clustered at state level in parentheses. "
        "*** p<0.01, ** p<0.05, * p<0.1. "
        "Dependent variable is log hourly wage."
    )

    return table


def main():
    parser = argparse.ArgumentParser(
        description="Format regression tables for economics papers"
    )
    parser.add_argument(
        '--input', '-i',
        type=Path,
        help="Input file (Stata output or CSV)"
    )
    parser.add_argument(
        '--output', '-o',
        type=Path,
        help="Output LaTeX file (default: stdout)"
    )
    parser.add_argument(
        '--style', '-s',
        choices=['aer', 'qje', 'standard'],
        default='aer',
        help="Table style (default: aer)"
    )
    parser.add_argument(
        '--title', '-t',
        default="Regression Results",
        help="Table title"
    )
    parser.add_argument(
        '--example',
        action='store_true',
        help="Generate example table"
    )

    args = parser.parse_args()

    formatter = TableFormatter(style=args.style)

    if args.example:
        table = create_example_table()
    elif args.input:
        if not args.input.exists():
            print(f"Error: File not found: {args.input}")
            sys.exit(1)

        if args.input.suffix == '.csv':
            parser_obj = CSVParser()
            table = parser_obj.parse(args.input)
        else:
            # Assume Stata output
            parser_obj = StataParser()
            text = args.input.read_text()
            table = parser_obj.parse(text)

        table.title = args.title
    else:
        print("Error: Specify --input or --example")
        parser.print_help()
        sys.exit(1)

    latex = formatter.to_latex(table)

    if args.output:
        args.output.write_text(latex)
        print(f"Table written to {args.output}")
    else:
        print(latex)


if __name__ == '__main__':
    main()
