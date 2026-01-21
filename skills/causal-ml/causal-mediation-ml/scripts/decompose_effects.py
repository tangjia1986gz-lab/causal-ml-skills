#!/usr/bin/env python3
"""
Effect Decomposition with Uncertainty Quantification

Decomposes the total treatment effect into direct (ADE) and indirect (ACME)
components with full uncertainty quantification via bootstrap.

Features:
- Multiple decomposition methods (linear, simulation-based)
- Bootstrap confidence intervals (percentile, BCa)
- Heterogeneous effects by subgroup
- Publication-ready output tables

Usage:
    python decompose_effects.py data.csv \
        --outcome earnings --treatment training --mediator skills \
        --controls age education --n-bootstrap 1000

Author: Causal ML Skills
Version: 1.0.0
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import numpy as np
from scipy import stats

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from mediation_estimator import (
    estimate_baron_kenny,
    create_mediation_data
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Decompose treatment effects into direct and indirect components.',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument('data_file', type=str, help='Path to CSV data file')

    # Core variables
    parser.add_argument('-y', '--outcome', type=str, required=True,
                        help='Outcome variable name')
    parser.add_argument('-d', '--treatment', type=str, required=True,
                        help='Treatment variable name')
    parser.add_argument('-m', '--mediator', type=str, required=True,
                        help='Mediator variable name')
    parser.add_argument('--controls', type=str, nargs='*', default=[],
                        help='Control variable names')

    # Bootstrap options
    parser.add_argument('--n-bootstrap', type=int, default=1000,
                        help='Number of bootstrap replications (default: 1000)')
    parser.add_argument('--ci-method', type=str,
                        choices=['percentile', 'bca', 'both'],
                        default='percentile',
                        help='Bootstrap CI method (default: percentile)')
    parser.add_argument('--confidence-level', type=float, default=0.95,
                        help='Confidence level (default: 0.95)')

    # Subgroup analysis
    parser.add_argument('--subgroup', type=str, default=None,
                        help='Variable for subgroup analysis')

    # Output
    parser.add_argument('-o', '--output', type=str, default=None,
                        help='Output file (CSV)')
    parser.add_argument('--latex', action='store_true',
                        help='Output LaTeX table')
    parser.add_argument('--verbose', action='store_true',
                        help='Verbose output')

    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')

    return parser.parse_args()


def bootstrap_decomposition(
    data: pd.DataFrame,
    outcome: str,
    treatment: str,
    mediator: str,
    controls: List[str],
    n_bootstrap: int = 1000,
    confidence_level: float = 0.95,
    ci_method: str = 'percentile',
    random_state: int = 42
) -> Dict:
    """
    Bootstrap effect decomposition with confidence intervals.

    Parameters
    ----------
    data : pd.DataFrame
        Input data
    outcome, treatment, mediator : str
        Variable names
    controls : List[str]
        Control variable names
    n_bootstrap : int
        Number of bootstrap replications
    confidence_level : float
        Confidence level for intervals
    ci_method : str
        'percentile', 'bca', or 'both'
    random_state : int
        Random seed

    Returns
    -------
    Dict with decomposition results and bootstrap CIs
    """
    np.random.seed(random_state)
    n = len(data)

    # Point estimates
    point_est = estimate_baron_kenny(data, outcome, treatment, mediator, controls)

    # Bootstrap storage
    boot_total = []
    boot_ade = []
    boot_acme = []
    boot_alpha = []
    boot_beta_m = []
    boot_prop = []

    # Bootstrap loop
    for b in range(n_bootstrap):
        idx = np.random.choice(n, n, replace=True)
        boot_data = data.iloc[idx].reset_index(drop=True)

        try:
            result = estimate_baron_kenny(
                boot_data, outcome, treatment, mediator, controls
            )
            boot_total.append(result['total_effect'])
            boot_ade.append(result['ade'])
            boot_acme.append(result['acme'])
            boot_alpha.append(result['alpha'])
            boot_beta_m.append(result['beta_m'])

            if abs(result['total_effect']) > 1e-10:
                boot_prop.append(result['acme'] / result['total_effect'])
        except:
            continue

    boot_total = np.array(boot_total)
    boot_ade = np.array(boot_ade)
    boot_acme = np.array(boot_acme)
    boot_alpha = np.array(boot_alpha)
    boot_beta_m = np.array(boot_beta_m)
    boot_prop = np.array(boot_prop)

    # Calculate CIs
    alpha = 1 - confidence_level
    lower_q = alpha / 2 * 100
    upper_q = (1 - alpha / 2) * 100

    def percentile_ci(boot_vals):
        return (np.percentile(boot_vals, lower_q),
                np.percentile(boot_vals, upper_q))

    def bca_ci(boot_vals, point_est, jackknife_vals=None):
        """Bias-corrected and accelerated CI."""
        # Bias correction
        prop_below = np.mean(boot_vals < point_est)
        z0 = stats.norm.ppf(prop_below) if 0 < prop_below < 1 else 0

        # Acceleration (from jackknife)
        if jackknife_vals is not None:
            mean_jack = np.mean(jackknife_vals)
            num = np.sum((mean_jack - jackknife_vals) ** 3)
            denom = 6 * (np.sum((mean_jack - jackknife_vals) ** 2) ** 1.5)
            a = num / denom if denom != 0 else 0
        else:
            a = 0

        # Adjusted percentiles
        z_alpha = stats.norm.ppf(alpha / 2)
        z_1_alpha = stats.norm.ppf(1 - alpha / 2)

        alpha1 = stats.norm.cdf(z0 + (z0 + z_alpha) / (1 - a * (z0 + z_alpha)))
        alpha2 = stats.norm.cdf(z0 + (z0 + z_1_alpha) / (1 - a * (z0 + z_1_alpha)))

        return (np.percentile(boot_vals, alpha1 * 100),
                np.percentile(boot_vals, alpha2 * 100))

    # Compute CIs
    results = {
        'point_estimates': {
            'total': point_est['total_effect'],
            'ade': point_est['ade'],
            'acme': point_est['acme'],
            'alpha': point_est['alpha'],
            'beta_m': point_est['beta_m'],
            'prop_mediated': point_est['prop_mediated']
        },
        'bootstrap': {
            'n_successful': len(boot_total),
            'n_requested': n_bootstrap
        }
    }

    # Standard errors
    results['se'] = {
        'total': np.std(boot_total),
        'ade': np.std(boot_ade),
        'acme': np.std(boot_acme),
        'alpha': np.std(boot_alpha),
        'beta_m': np.std(boot_beta_m),
        'prop_mediated': np.std(boot_prop) if len(boot_prop) > 0 else np.nan
    }

    # Percentile CIs
    if ci_method in ['percentile', 'both']:
        results['ci_percentile'] = {
            'total': percentile_ci(boot_total),
            'ade': percentile_ci(boot_ade),
            'acme': percentile_ci(boot_acme),
            'alpha': percentile_ci(boot_alpha),
            'beta_m': percentile_ci(boot_beta_m),
            'prop_mediated': percentile_ci(boot_prop) if len(boot_prop) > 0 else (np.nan, np.nan)
        }

    # BCa CIs
    if ci_method in ['bca', 'both']:
        results['ci_bca'] = {
            'total': bca_ci(boot_total, point_est['total_effect']),
            'ade': bca_ci(boot_ade, point_est['ade']),
            'acme': bca_ci(boot_acme, point_est['acme'])
        }

    # P-values from bootstrap
    results['p_values'] = {
        'acme': 2 * min(np.mean(boot_acme <= 0), np.mean(boot_acme >= 0)),
        'ade': 2 * min(np.mean(boot_ade <= 0), np.mean(boot_ade >= 0))
    }

    return results


def decompose_by_subgroup(
    data: pd.DataFrame,
    outcome: str,
    treatment: str,
    mediator: str,
    controls: List[str],
    subgroup_var: str,
    n_bootstrap: int = 500,
    confidence_level: float = 0.95
) -> pd.DataFrame:
    """
    Decompose effects separately by subgroup.
    """
    results = []

    for group_val in data[subgroup_var].unique():
        subset = data[data[subgroup_var] == group_val]

        if len(subset) < 50:
            print(f"Warning: Subgroup {group_val} has only {len(subset)} obs, skipping")
            continue

        decomp = bootstrap_decomposition(
            subset, outcome, treatment, mediator, controls,
            n_bootstrap=n_bootstrap,
            confidence_level=confidence_level
        )

        ci_acme = decomp.get('ci_percentile', {}).get('acme', (np.nan, np.nan))
        ci_ade = decomp.get('ci_percentile', {}).get('ade', (np.nan, np.nan))

        results.append({
            'subgroup': group_val,
            'n': len(subset),
            'total': decomp['point_estimates']['total'],
            'ade': decomp['point_estimates']['ade'],
            'ade_se': decomp['se']['ade'],
            'ade_ci_lower': ci_ade[0],
            'ade_ci_upper': ci_ade[1],
            'acme': decomp['point_estimates']['acme'],
            'acme_se': decomp['se']['acme'],
            'acme_ci_lower': ci_acme[0],
            'acme_ci_upper': ci_acme[1],
            'prop_mediated': decomp['point_estimates']['prop_mediated']
        })

    return pd.DataFrame(results)


def format_decomposition_table(results: Dict, confidence_level: float = 0.95) -> str:
    """Format results as text table."""
    ci_key = 'ci_percentile' if 'ci_percentile' in results else 'ci_bca'
    ci = results.get(ci_key, {})
    se = results['se']
    pe = results['point_estimates']

    def stars(p):
        if p < 0.01: return '***'
        elif p < 0.05: return '**'
        elif p < 0.1: return '*'
        return ''

    def format_ci(ci_tuple):
        if ci_tuple and not any(np.isnan(ci_tuple)):
            return f"[{ci_tuple[0]:.4f}, {ci_tuple[1]:.4f}]"
        return "N/A"

    lines = [
        "=" * 75,
        "EFFECT DECOMPOSITION WITH UNCERTAINTY".center(75),
        "=" * 75,
        "",
        f"{'Component':<20} {'Estimate':>12} {'Std.Err':>10} {f'{int(confidence_level*100)}% CI':>22}",
        "-" * 75,
    ]

    # Total effect
    lines.append(
        f"{'Total Effect':<20} {pe['total']:>12.4f} ({se['total']:>8.4f}) {format_ci(ci.get('total'))}"
    )

    # Path coefficients
    lines.append(
        f"{'  a (D -> M)':<20} {pe['alpha']:>12.4f} ({se['alpha']:>8.4f}) {format_ci(ci.get('alpha'))}"
    )
    lines.append(
        f"{'  b (M -> Y|D)':<20} {pe['beta_m']:>12.4f} ({se['beta_m']:>8.4f}) {format_ci(ci.get('beta_m'))}"
    )

    lines.append("-" * 75)

    # Direct effect
    p_ade = results['p_values'].get('ade', 0)
    lines.append(
        f"{'Direct (ADE)':<20} {pe['ade']:>12.4f}{stars(p_ade):<3} ({se['ade']:>8.4f}) {format_ci(ci.get('ade'))}"
    )

    # Indirect effect
    p_acme = results['p_values'].get('acme', 0)
    lines.append(
        f"{'Indirect (ACME)':<20} {pe['acme']:>12.4f}{stars(p_acme):<3} ({se['acme']:>8.4f}) {format_ci(ci.get('acme'))}"
    )

    lines.extend([
        "-" * 75,
        f"Proportion Mediated: {pe['prop_mediated']*100:.1f}% (SE = {se['prop_mediated']*100:.1f}%)",
        "",
        f"Bootstrap replications: {results['bootstrap']['n_successful']}/{results['bootstrap']['n_requested']}",
        "Notes: *** p<0.01, ** p<0.05, * p<0.1",
        "=" * 75,
    ])

    return "\n".join(lines)


def generate_latex_table(results: Dict, confidence_level: float = 0.95) -> str:
    """Generate LaTeX table."""
    ci = results.get('ci_percentile', results.get('ci_bca', {}))
    se = results['se']
    pe = results['point_estimates']

    def stars(p):
        if p < 0.01: return '***'
        elif p < 0.05: return '**'
        elif p < 0.1: return '*'
        return ''

    def fmt_ci(ci_tuple):
        if ci_tuple and not any(np.isnan(ci_tuple)):
            return f"[{ci_tuple[0]:.3f}, {ci_tuple[1]:.3f}]"
        return "---"

    p_ade = results['p_values'].get('ade', 0)
    p_acme = results['p_values'].get('acme', 0)

    latex = r"""
\begin{table}[htbp]
\centering
\caption{Effect Decomposition}
\label{tab:decomposition}
\begin{tabular}{lccc}
\toprule
Component & Estimate & Std. Err. & %d\%% CI \\
\midrule
Total Effect & %.4f & (%.4f) & %s \\
\quad $a$ (D $\to$ M) & %.4f & (%.4f) & %s \\
\quad $b$ (M $\to$ Y$|$D) & %.4f & (%.4f) & %s \\
\midrule
Direct Effect (ADE) & %.4f%s & (%.4f) & %s \\
Indirect Effect (ACME) & %.4f%s & (%.4f) & %s \\
\midrule
Proportion Mediated & %.1f\%% & (%.1f\%%) & --- \\
\bottomrule
\multicolumn{4}{l}{\footnotesize Notes: *** $p<0.01$, ** $p<0.05$, * $p<0.1$} \\
\multicolumn{4}{l}{\footnotesize %d bootstrap replications.} \\
\end{tabular}
\end{table}
""" % (
        int(confidence_level * 100),
        pe['total'], se['total'], fmt_ci(ci.get('total')),
        pe['alpha'], se['alpha'], fmt_ci(ci.get('alpha')),
        pe['beta_m'], se['beta_m'], fmt_ci(ci.get('beta_m')),
        pe['ade'], stars(p_ade), se['ade'], fmt_ci(ci.get('ade')),
        pe['acme'], stars(p_acme), se['acme'], fmt_ci(ci.get('acme')),
        pe['prop_mediated'] * 100, se['prop_mediated'] * 100,
        results['bootstrap']['n_successful']
    )

    return latex


def main():
    """Main entry point."""
    args = parse_args()
    np.random.seed(args.seed)

    # Load data
    data = pd.read_csv(args.data_file)
    if args.verbose:
        print(f"Loaded {len(data)} observations")

    # Validate variables
    required = [args.outcome, args.treatment, args.mediator] + args.controls
    missing = [v for v in required if v not in data.columns]
    if missing:
        print(f"Error: Variables not found: {missing}", file=sys.stderr)
        sys.exit(1)

    if args.subgroup:
        if args.subgroup not in data.columns:
            print(f"Error: Subgroup variable not found: {args.subgroup}", file=sys.stderr)
            sys.exit(1)

        if args.verbose:
            print(f"Running subgroup analysis by {args.subgroup}...")

        subgroup_results = decompose_by_subgroup(
            data, args.outcome, args.treatment, args.mediator, args.controls,
            args.subgroup, args.n_bootstrap, args.confidence_level
        )

        print("\nSubgroup Analysis Results:")
        print(subgroup_results.to_string(index=False))

        if args.output:
            subgroup_results.to_csv(args.output, index=False)
            print(f"\nResults saved to: {args.output}")

    else:
        if args.verbose:
            print(f"Running bootstrap decomposition with {args.n_bootstrap} replications...")

        results = bootstrap_decomposition(
            data, args.outcome, args.treatment, args.mediator, args.controls,
            n_bootstrap=args.n_bootstrap,
            confidence_level=args.confidence_level,
            ci_method=args.ci_method,
            random_state=args.seed
        )

        # Print text table
        print(format_decomposition_table(results, args.confidence_level))

        # Print LaTeX if requested
        if args.latex:
            print("\nLaTeX Table:")
            print(generate_latex_table(results, args.confidence_level))

        # Save to CSV if requested
        if args.output:
            df = pd.DataFrame([{
                'component': k,
                'estimate': v,
                'se': results['se'].get(k, np.nan),
                'ci_lower': results.get('ci_percentile', {}).get(k, (np.nan, np.nan))[0],
                'ci_upper': results.get('ci_percentile', {}).get(k, (np.nan, np.nan))[1]
            } for k, v in results['point_estimates'].items()])
            df.to_csv(args.output, index=False)
            print(f"\nResults saved to: {args.output}")


if __name__ == '__main__':
    main()
