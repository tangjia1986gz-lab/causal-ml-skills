#!/usr/bin/env python
"""
Generate Comparison Report
==========================

Compare replicated results against original paper values and generate
a detailed comparison report for replication studies.

This script:
- Loads original paper estimates and replicated results
- Computes comparison metrics (differences, CI overlap, etc.)
- Classifies replication success level
- Generates formatted comparison tables and reports

Usage:
    python generate_comparison.py [options]

Examples:
    # From JSON specification
    python generate_comparison.py --spec paper_spec.json --results results.json

    # Interactive mode
    python generate_comparison.py --interactive

    # Generate LaTeX table
    python generate_comparison.py --spec paper_spec.json --results results.json --format latex

Output:
    - Comparison tables (text, LaTeX, or markdown)
    - Detailed discrepancy analysis
    - Success classification report
"""

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Optional, Union

import numpy as np


class ReplicationSuccess(Enum):
    """Classification of replication success level."""
    EXACT = "exact"
    CLOSE = "close"
    APPROXIMATE = "approximate"
    QUALITATIVE = "qualitative"
    FAILED = "failed"


@dataclass
class EstimateResult:
    """Single estimation result."""
    estimate: float
    se: float
    pvalue: Optional[float] = None
    ci_lower: Optional[float] = None
    ci_upper: Optional[float] = None
    n: Optional[int] = None
    method: str = ""

    def __post_init__(self):
        """Compute derived values."""
        if self.pvalue is None and self.se > 0:
            from scipy import stats
            z = abs(self.estimate / self.se)
            self.pvalue = 2 * (1 - stats.norm.cdf(z))

        if self.ci_lower is None:
            self.ci_lower = self.estimate - 1.96 * self.se

        if self.ci_upper is None:
            self.ci_upper = self.estimate + 1.96 * self.se


@dataclass
class ComparisonMetrics:
    """Metrics comparing original and replicated results."""
    estimate_diff: float
    estimate_pct_diff: float
    se_diff: float
    se_pct_diff: float
    ci_overlap: float
    same_sign: bool
    same_significance: bool
    normalized_diff: float
    success_level: ReplicationSuccess
    success_message: str


def compute_ci_overlap(est1: float, se1: float, est2: float, se2: float) -> float:
    """
    Compute overlap of 95% confidence intervals.

    Returns fraction of CI width that overlaps (0 to 1).
    """
    ci1_lower = est1 - 1.96 * se1
    ci1_upper = est1 + 1.96 * se1
    ci2_lower = est2 - 1.96 * se2
    ci2_upper = est2 + 1.96 * se2

    overlap_lower = max(ci1_lower, ci2_lower)
    overlap_upper = min(ci1_upper, ci2_upper)

    if overlap_lower >= overlap_upper:
        return 0.0

    overlap_width = overlap_upper - overlap_lower
    max_width = max(ci1_upper - ci1_lower, ci2_upper - ci2_lower)

    return overlap_width / max_width if max_width > 0 else 0.0


def classify_replication_success(
    estimate_pct_diff: float,
    se_pct_diff: float,
    same_sign: bool,
    same_significance: bool
) -> tuple:
    """
    Classify replication success level.

    Classification criteria:
    - EXACT: < 1% estimate diff, < 5% SE diff
    - CLOSE: < 5% estimate diff, < 10% SE diff
    - APPROXIMATE: < 10% estimate diff, < 20% SE diff
    - QUALITATIVE: Same sign and significance
    - FAILED: Different conclusions

    Returns
    -------
    tuple
        (ReplicationSuccess, explanation string)
    """
    est_diff = abs(estimate_pct_diff)
    se_diff = abs(se_pct_diff)

    if est_diff < 1 and se_diff < 5:
        return (
            ReplicationSuccess.EXACT,
            "Results match within numerical precision"
        )

    if est_diff < 5 and se_diff < 10:
        return (
            ReplicationSuccess.CLOSE,
            "Minor numerical differences, likely software/algorithm variation"
        )

    if est_diff < 10 and se_diff < 20:
        return (
            ReplicationSuccess.APPROXIMATE,
            "Moderate differences, possibly different specifications or data versions"
        )

    if same_sign and same_significance:
        return (
            ReplicationSuccess.QUALITATIVE,
            "Conclusions match despite numerical differences"
        )

    return (
        ReplicationSuccess.FAILED,
        "Cannot replicate main conclusions"
    )


def compare_results(
    original: Union[dict, EstimateResult],
    replicated: Union[dict, EstimateResult]
) -> ComparisonMetrics:
    """
    Compare original and replicated results.

    Parameters
    ----------
    original : dict or EstimateResult
        Original paper results with keys: estimate, se, (optionally) pvalue
    replicated : dict or EstimateResult
        Replicated results

    Returns
    -------
    ComparisonMetrics
        Comprehensive comparison metrics
    """
    # Convert to EstimateResult if dict
    if isinstance(original, dict):
        original = EstimateResult(**original)
    if isinstance(replicated, dict):
        replicated = EstimateResult(**replicated)

    # Compute differences
    estimate_diff = replicated.estimate - original.estimate
    estimate_pct_diff = (
        estimate_diff / abs(original.estimate) * 100
        if original.estimate != 0 else float("inf")
    )

    se_diff = replicated.se - original.se
    se_pct_diff = (
        se_diff / original.se * 100
        if original.se != 0 else float("inf")
    )

    # CI overlap
    ci_overlap = compute_ci_overlap(
        original.estimate, original.se,
        replicated.estimate, replicated.se
    )

    # Statistical conclusions
    orig_sig = original.pvalue < 0.05 if original.pvalue else False
    repl_sig = replicated.pvalue < 0.05 if replicated.pvalue else False
    same_significance = orig_sig == repl_sig

    same_sign = np.sign(original.estimate) == np.sign(replicated.estimate)

    # Normalized difference
    combined_se = np.sqrt(original.se**2 + replicated.se**2)
    normalized_diff = estimate_diff / combined_se if combined_se > 0 else float("inf")

    # Classify success
    success_level, success_message = classify_replication_success(
        estimate_pct_diff, se_pct_diff, same_sign, same_significance
    )

    return ComparisonMetrics(
        estimate_diff=estimate_diff,
        estimate_pct_diff=estimate_pct_diff,
        se_diff=se_diff,
        se_pct_diff=se_pct_diff,
        ci_overlap=ci_overlap,
        same_sign=same_sign,
        same_significance=same_significance,
        normalized_diff=normalized_diff,
        success_level=success_level,
        success_message=success_message
    )


def generate_comparison_table(
    original: Union[dict, EstimateResult],
    replicated: Union[dict, EstimateResult],
    paper_name: str = "Paper",
    specification: str = "Main"
) -> str:
    """
    Generate side-by-side comparison table in text format.

    Parameters
    ----------
    original : dict or EstimateResult
        Original paper results
    replicated : dict or EstimateResult
        Replicated results
    paper_name : str
        Name of paper for title
    specification : str
        Name of specification being compared

    Returns
    -------
    str
        Formatted comparison table
    """
    # Convert to EstimateResult if dict
    if isinstance(original, dict):
        original = EstimateResult(**original)
    if isinstance(replicated, dict):
        replicated = EstimateResult(**replicated)

    comparison = compare_results(original, replicated)

    # Format numbers
    def fmt_num(x, decimals=2, prefix=""):
        if x is None:
            return "N/A"
        if abs(x) >= 1000:
            return f"{prefix}{x:,.0f}"
        return f"{prefix}{x:.{decimals}f}"

    def fmt_pct(x):
        if x == float("inf") or x != x:  # Check for inf or nan
            return "N/A"
        sign = "+" if x > 0 else ""
        return f"{sign}{x:.1f}%"

    lines = [
        "=" * 75,
        f"{'REPLICATION COMPARISON: ' + paper_name:^75}",
        f"{'Specification: ' + specification:^75}",
        "=" * 75,
        "",
        f"{'Metric':<25} {'Original':>15} {'Replicated':>15} {'Difference':>15}",
        "-" * 75,
        "Point Estimates",
        f"  {'Estimate':<23} {fmt_num(original.estimate):>15} {fmt_num(replicated.estimate):>15} {fmt_pct(comparison.estimate_pct_diff):>15}",
        f"  {'Standard Error':<23} {fmt_num(original.se):>15} {fmt_num(replicated.se):>15} {fmt_pct(comparison.se_pct_diff):>15}",
        "",
        "Confidence Intervals (95%)",
        f"  {'Lower Bound':<23} {fmt_num(original.ci_lower):>15} {fmt_num(replicated.ci_lower):>15}",
        f"  {'Upper Bound':<23} {fmt_num(original.ci_upper):>15} {fmt_num(replicated.ci_upper):>15}",
        f"  {'CI Overlap':<23} {'-':>15} {'-':>15} {f'{comparison.ci_overlap:.1%}':>15}",
        "",
        "Statistical Inference",
        f"  {'p-value':<23} {fmt_num(original.pvalue, 4):>15} {fmt_num(replicated.pvalue, 4):>15}",
        f"  {'Significant (5%)':<23} {'Yes' if (original.pvalue and original.pvalue < 0.05) else 'No':>15} {'Yes' if (replicated.pvalue and replicated.pvalue < 0.05) else 'No':>15} {'Match' if comparison.same_significance else 'DIFFER':>15}",
        f"  {'Sign':<23} {'+' if original.estimate > 0 else '-':>15} {'+' if replicated.estimate > 0 else '-':>15} {'Match' if comparison.same_sign else 'DIFFER':>15}",
        "",
        "-" * 75,
        "Summary Metrics",
        f"  {'Normalized Difference':<23} {fmt_num(comparison.normalized_diff)} standard errors",
        "",
        "-" * 75,
        f"REPLICATION STATUS: {comparison.success_level.value.upper()}",
        comparison.success_message,
        "=" * 75,
    ]

    return "\n".join(lines)


def generate_latex_table(
    original: Union[dict, EstimateResult],
    replicated: Union[dict, EstimateResult],
    paper_name: str = "Paper",
    specification: str = "Main",
    label: str = "tab:replication"
) -> str:
    """
    Generate comparison table in LaTeX format.

    Parameters
    ----------
    original : dict or EstimateResult
        Original paper results
    replicated : dict or EstimateResult
        Replicated results
    paper_name : str
        Name of paper for title
    specification : str
        Name of specification
    label : str
        LaTeX label for the table

    Returns
    -------
    str
        LaTeX table code
    """
    # Convert to EstimateResult if dict
    if isinstance(original, dict):
        original = EstimateResult(**original)
    if isinstance(replicated, dict):
        replicated = EstimateResult(**replicated)

    comparison = compare_results(original, replicated)

    def fmt(x, decimals=3):
        if x is None:
            return "--"
        if abs(x) >= 1000:
            return f"{x:,.0f}"
        return f"{x:.{decimals}f}"

    latex = f"""\\begin{{table}}[htbp]
\\centering
\\caption{{Replication Comparison: {paper_name}}}
\\label{{{label}}}
\\begin{{tabular}}{{lccc}}
\\toprule
 & Original & Replicated & Difference (\\%) \\\\
\\midrule
\\multicolumn{{4}}{{l}}{{\\textit{{{specification}}}}} \\\\
\\addlinespace
Point Estimate & {fmt(original.estimate)} & {fmt(replicated.estimate)} & {comparison.estimate_pct_diff:+.1f}\\% \\\\
Standard Error & ({fmt(original.se)}) & ({fmt(replicated.se)}) & {comparison.se_pct_diff:+.1f}\\% \\\\
\\addlinespace
95\\% CI Lower & {fmt(original.ci_lower)} & {fmt(replicated.ci_lower)} & \\\\
95\\% CI Upper & {fmt(original.ci_upper)} & {fmt(replicated.ci_upper)} & \\\\
\\addlinespace
$p$-value & {fmt(original.pvalue, 4)} & {fmt(replicated.pvalue, 4)} & \\\\
\\midrule
CI Overlap & \\multicolumn{{3}}{{c}}{{{comparison.ci_overlap:.1%}}} \\\\
Same Sign & \\multicolumn{{3}}{{c}}{{{'Yes' if comparison.same_sign else 'No'}}} \\\\
Same Significance & \\multicolumn{{3}}{{c}}{{{'Yes' if comparison.same_significance else 'No'}}} \\\\
\\midrule
\\textbf{{Replication Status}} & \\multicolumn{{3}}{{c}}{{\\textbf{{{comparison.success_level.value.upper()}}}}} \\\\
\\bottomrule
\\end{{tabular}}
\\begin{{tablenotes}}
\\small
\\item \\textit{{Note:}} {comparison.success_message}
\\end{{tablenotes}}
\\end{{table}}
"""

    return latex


def generate_markdown_table(
    original: Union[dict, EstimateResult],
    replicated: Union[dict, EstimateResult],
    paper_name: str = "Paper",
    specification: str = "Main"
) -> str:
    """
    Generate comparison table in Markdown format.

    Parameters
    ----------
    original : dict or EstimateResult
        Original paper results
    replicated : dict or EstimateResult
        Replicated results
    paper_name : str
        Name of paper for title
    specification : str
        Name of specification

    Returns
    -------
    str
        Markdown table
    """
    # Convert to EstimateResult if dict
    if isinstance(original, dict):
        original = EstimateResult(**original)
    if isinstance(replicated, dict):
        replicated = EstimateResult(**replicated)

    comparison = compare_results(original, replicated)

    def fmt(x, decimals=3):
        if x is None:
            return "--"
        if abs(x) >= 1000:
            return f"{x:,.0f}"
        return f"{x:.{decimals}f}"

    md = f"""## Replication Comparison: {paper_name}

### {specification}

| Metric | Original | Replicated | Difference |
|--------|----------|------------|------------|
| **Point Estimate** | {fmt(original.estimate)} | {fmt(replicated.estimate)} | {comparison.estimate_pct_diff:+.1f}% |
| Standard Error | {fmt(original.se)} | {fmt(replicated.se)} | {comparison.se_pct_diff:+.1f}% |
| 95% CI | [{fmt(original.ci_lower)}, {fmt(original.ci_upper)}] | [{fmt(replicated.ci_lower)}, {fmt(replicated.ci_upper)}] | |
| p-value | {fmt(original.pvalue, 4)} | {fmt(replicated.pvalue, 4)} | |

### Summary

| Metric | Value |
|--------|-------|
| CI Overlap | {comparison.ci_overlap:.1%} |
| Same Sign | {'Yes' if comparison.same_sign else 'No'} |
| Same Significance | {'Yes' if comparison.same_significance else 'No'} |
| Normalized Difference | {comparison.normalized_diff:.2f} SE |

### Replication Status: **{comparison.success_level.value.upper()}**

{comparison.success_message}

---

*Generated: {datetime.now().strftime("%Y-%m-%d %H:%M")}*
"""

    return md


def generate_multiple_comparisons(
    specifications: list,
    paper_name: str = "Paper",
    format: str = "text"
) -> str:
    """
    Generate comparison for multiple specifications.

    Parameters
    ----------
    specifications : list
        List of dicts with keys: name, original, replicated
    paper_name : str
        Name of paper
    format : str
        Output format (text, latex, markdown)

    Returns
    -------
    str
        Formatted comparison report
    """
    outputs = []

    for spec in specifications:
        name = spec.get("name", "Specification")
        original = spec["original"]
        replicated = spec["replicated"]

        if format == "text":
            output = generate_comparison_table(original, replicated, paper_name, name)
        elif format == "latex":
            label = f"tab:replication_{name.lower().replace(' ', '_')}"
            output = generate_latex_table(original, replicated, paper_name, name, label)
        elif format == "markdown":
            output = generate_markdown_table(original, replicated, paper_name, name)
        else:
            raise ValueError(f"Unknown format: {format}")

        outputs.append(output)

    return "\n\n".join(outputs)


def load_specification(spec_path: Path) -> dict:
    """Load paper specification from JSON file."""
    with open(spec_path) as f:
        return json.load(f)


def load_results(results_path: Path) -> dict:
    """Load replication results from JSON file."""
    with open(results_path) as f:
        return json.load(f)


def interactive_mode():
    """Run in interactive mode, prompting for values."""
    print("=" * 60)
    print("Replication Comparison - Interactive Mode")
    print("=" * 60)
    print()

    # Get paper name
    paper_name = input("Paper name/citation: ").strip() or "Paper"
    spec_name = input("Specification name: ").strip() or "Main"

    print("\nOriginal paper values:")
    orig_est = float(input("  Estimate: "))
    orig_se = float(input("  Standard error: "))

    print("\nReplicated values:")
    repl_est = float(input("  Estimate: "))
    repl_se = float(input("  Standard error: "))

    original = EstimateResult(estimate=orig_est, se=orig_se)
    replicated = EstimateResult(estimate=repl_est, se=repl_se)

    print()
    print(generate_comparison_table(original, replicated, paper_name, spec_name))


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Generate replication comparison report.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python generate_comparison.py --spec paper.json --results results.json
  python generate_comparison.py --interactive
  python generate_comparison.py --spec paper.json --results results.json --format latex
        """
    )

    parser.add_argument(
        "--spec",
        type=Path,
        help="Path to paper specification JSON file"
    )
    parser.add_argument(
        "--results",
        type=Path,
        help="Path to replication results JSON file"
    )
    parser.add_argument(
        "--format",
        choices=["text", "latex", "markdown"],
        default="text",
        help="Output format (default: text)"
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output file (default: stdout)"
    )
    parser.add_argument(
        "--interactive", "-i",
        action="store_true",
        help="Run in interactive mode"
    )

    args = parser.parse_args()

    # Interactive mode
    if args.interactive:
        interactive_mode()
        return

    # File-based mode
    if not args.spec or not args.results:
        parser.error("Either --interactive or both --spec and --results are required")

    spec = load_specification(args.spec)
    results = load_results(args.results)

    # Check if multiple specifications
    if "specifications" in spec:
        # Multiple specifications
        specs = []
        for s in spec["specifications"]:
            name = s["name"]
            original = s["original_results"]
            replicated = results.get(name, results)
            specs.append({
                "name": name,
                "original": original,
                "replicated": replicated
            })

        output = generate_multiple_comparisons(
            specs,
            paper_name=spec.get("paper_name", "Paper"),
            format=args.format
        )
    else:
        # Single specification
        original = spec.get("original_results", spec)
        replicated = results

        if args.format == "text":
            output = generate_comparison_table(
                original, replicated,
                spec.get("paper_name", "Paper"),
                spec.get("specification", "Main")
            )
        elif args.format == "latex":
            output = generate_latex_table(
                original, replicated,
                spec.get("paper_name", "Paper"),
                spec.get("specification", "Main")
            )
        elif args.format == "markdown":
            output = generate_markdown_table(
                original, replicated,
                spec.get("paper_name", "Paper"),
                spec.get("specification", "Main")
            )

    # Output
    if args.output:
        args.output.write_text(output)
        print(f"Report saved to: {args.output}")
    else:
        print(output)


if __name__ == "__main__":
    main()
