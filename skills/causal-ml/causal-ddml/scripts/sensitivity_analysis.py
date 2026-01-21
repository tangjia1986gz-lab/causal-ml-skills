#!/usr/bin/env python3
"""
DDML Sensitivity Analysis

Comprehensive sensitivity analysis for DDML estimates:
- Sensitivity to ML learner choice
- Sensitivity to number of folds
- Sensitivity to trimming threshold (IRM)
- Sensitivity to control variable sets
- Robustness to specification choices

Usage:
    python sensitivity_analysis.py --data data.csv --outcome y --treatment d --controls "x1,x2"
    python sensitivity_analysis.py --data data.csv --outcome y --treatment d --all-controls --output results.json

Reference:
    Chernozhukov, V., et al. (2018). Double/Debiased Machine Learning.
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime
import warnings

import pandas as pd
import numpy as np
from scipy import stats

warnings.filterwarnings('ignore')

# Add parent for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from ddml_estimator import estimate_plr, estimate_irm, create_ddml_data


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run sensitivity analysis for DDML"
    )

    parser.add_argument('--data', '-d', required=True,
                        help='Path to CSV data file')
    parser.add_argument('--outcome', '-y', required=True,
                        help='Outcome variable name')
    parser.add_argument('--treatment', '-t', required=True,
                        help='Treatment variable name')
    parser.add_argument('--controls', '-x',
                        help='Comma-separated control variables')
    parser.add_argument('--all-controls', action='store_true',
                        help='Use all other columns as controls')
    parser.add_argument('--model', '-m', default='auto',
                        choices=['plr', 'irm', 'auto'],
                        help='Model type (default: auto)')
    parser.add_argument('--analyses', '-a',
                        default='learner,folds,trimming,controls',
                        help='Analyses to run (comma-separated)')
    parser.add_argument('--output', '-o',
                        help='Output file for results (JSON)')
    parser.add_argument('--plot', action='store_true',
                        help='Generate sensitivity plots')
    parser.add_argument('--verbose', '-v', action='store_true')

    return parser.parse_args()


def learner_sensitivity(data, outcome, treatment, controls, model='plr',
                        learners=None, n_folds=5, verbose=False):
    """
    Analyze sensitivity to ML learner choice.
    """
    if learners is None:
        learners = ['lasso', 'ridge', 'random_forest', 'xgboost']

    results = {}

    for learner in learners:
        if verbose:
            print(f"  Testing {learner}...")

        try:
            if model == 'plr':
                result = estimate_plr(
                    data=data, outcome=outcome, treatment=treatment,
                    controls=controls, ml_l=learner, ml_m=learner,
                    n_folds=n_folds
                )
            else:
                ml_m = 'logistic_lasso' if learner in ['lasso', 'ridge'] else learner
                result = estimate_irm(
                    data=data, outcome=outcome, treatment=treatment,
                    controls=controls, ml_g=learner, ml_m=ml_m,
                    n_folds=n_folds
                )

            results[learner] = {
                'effect': result.effect,
                'se': result.se,
                'ci_lower': result.ci_lower,
                'ci_upper': result.ci_upper,
                'p_value': result.p_value
            }
        except Exception as e:
            results[learner] = {'error': str(e)}

    # Compute summary statistics
    valid_effects = [r['effect'] for r in results.values() if 'effect' in r]

    if valid_effects:
        summary = {
            'n_learners': len(valid_effects),
            'mean': np.mean(valid_effects),
            'std': np.std(valid_effects),
            'cv': np.std(valid_effects) / abs(np.mean(valid_effects)) * 100,
            'range': (min(valid_effects), max(valid_effects)),
            'all_same_sign': all(e > 0 for e in valid_effects) or all(e < 0 for e in valid_effects),
            'all_significant': all(r.get('p_value', 1) < 0.05 for r in results.values() if 'p_value' in r)
        }
    else:
        summary = {'error': 'No valid results'}

    return {'by_learner': results, 'summary': summary}


def folds_sensitivity(data, outcome, treatment, controls, model='plr',
                      learner='lasso', fold_range=(2, 10), verbose=False):
    """
    Analyze sensitivity to number of cross-fitting folds.
    """
    results = {}

    for n_folds in range(fold_range[0], fold_range[1] + 1):
        if verbose:
            print(f"  Testing K={n_folds}...")

        try:
            if model == 'plr':
                result = estimate_plr(
                    data=data, outcome=outcome, treatment=treatment,
                    controls=controls, ml_l=learner, ml_m=learner,
                    n_folds=n_folds
                )
            else:
                result = estimate_irm(
                    data=data, outcome=outcome, treatment=treatment,
                    controls=controls, ml_g=learner,
                    n_folds=n_folds
                )

            results[n_folds] = {
                'effect': result.effect,
                'se': result.se,
                'ci_lower': result.ci_lower,
                'ci_upper': result.ci_upper
            }
        except Exception as e:
            results[n_folds] = {'error': str(e)}

    valid_effects = [r['effect'] for r in results.values() if 'effect' in r]

    summary = {
        'n_tested': len(valid_effects),
        'mean': np.mean(valid_effects) if valid_effects else None,
        'std': np.std(valid_effects) if valid_effects else None,
        'cv': np.std(valid_effects) / abs(np.mean(valid_effects)) * 100 if valid_effects and np.mean(valid_effects) != 0 else None,
        'stable': np.std(valid_effects) / abs(np.mean(valid_effects)) < 0.05 if valid_effects and np.mean(valid_effects) != 0 else None
    }

    return {'by_folds': results, 'summary': summary}


def trimming_sensitivity(data, outcome, treatment, controls,
                         thresholds=None, learner='random_forest', n_folds=5,
                         verbose=False):
    """
    Analyze sensitivity to propensity score trimming (IRM only).
    """
    if thresholds is None:
        thresholds = [0.0, 0.01, 0.02, 0.05, 0.10, 0.15]

    results = {}

    for thresh in thresholds:
        if verbose:
            print(f"  Testing threshold={thresh}...")

        try:
            result = estimate_irm(
                data=data, outcome=outcome, treatment=treatment,
                controls=controls, ml_g=learner,
                trimming_threshold=thresh, n_folds=n_folds
            )

            results[thresh] = {
                'effect': result.effect,
                'se': result.se,
                'n_trimmed': result.diagnostics.get('n_trimmed', 0),
                'ci_lower': result.ci_lower,
                'ci_upper': result.ci_upper
            }
        except Exception as e:
            results[thresh] = {'error': str(e)}

    valid_effects = [r['effect'] for r in results.values() if 'effect' in r]

    summary = {
        'n_tested': len(valid_effects),
        'mean': np.mean(valid_effects) if valid_effects else None,
        'std': np.std(valid_effects) if valid_effects else None,
        'sensitivity': 'LOW' if valid_effects and np.std(valid_effects) < 0.1 * abs(np.mean(valid_effects)) else 'MODERATE'
    }

    return {'by_threshold': results, 'summary': summary}


def controls_sensitivity(data, outcome, treatment, all_controls,
                         model='plr', learner='lasso', n_folds=5,
                         verbose=False):
    """
    Analyze sensitivity to control variable sets.
    """
    results = {}

    # Test different control sets
    control_sets = {
        'all': all_controls,
        'half': all_controls[:len(all_controls)//2],
        'quarter': all_controls[:len(all_controls)//4] if len(all_controls) >= 4 else all_controls[:1],
        'minimal': all_controls[:3] if len(all_controls) >= 3 else all_controls
    }

    for name, controls in control_sets.items():
        if not controls:
            continue

        if verbose:
            print(f"  Testing {name} ({len(controls)} controls)...")

        try:
            if model == 'plr':
                result = estimate_plr(
                    data=data, outcome=outcome, treatment=treatment,
                    controls=controls, ml_l=learner, ml_m=learner,
                    n_folds=n_folds
                )
            else:
                result = estimate_irm(
                    data=data, outcome=outcome, treatment=treatment,
                    controls=controls, ml_g=learner, n_folds=n_folds
                )

            results[name] = {
                'n_controls': len(controls),
                'effect': result.effect,
                'se': result.se,
                'ci_lower': result.ci_lower,
                'ci_upper': result.ci_upper
            }
        except Exception as e:
            results[name] = {'error': str(e)}

    valid_effects = [r['effect'] for r in results.values() if 'effect' in r]

    summary = {
        'n_tested': len(valid_effects),
        'mean': np.mean(valid_effects) if valid_effects else None,
        'std': np.std(valid_effects) if valid_effects else None,
        'sensitivity': 'LOW' if valid_effects and np.std(valid_effects) < 0.15 * abs(np.mean(valid_effects)) else 'MODERATE'
    }

    return {'by_control_set': results, 'summary': summary}


def run_comprehensive_sensitivity(data, outcome, treatment, controls, model,
                                  analyses, verbose=False):
    """
    Run comprehensive sensitivity analysis.
    """
    results = {
        'timestamp': datetime.now().isoformat(),
        'model': model,
        'n_obs': len(data),
        'n_controls': len(controls)
    }

    # Learner sensitivity
    if 'learner' in analyses:
        print("\nAnalyzing learner sensitivity...")
        results['learner_sensitivity'] = learner_sensitivity(
            data, outcome, treatment, controls, model, verbose=verbose
        )

    # Folds sensitivity
    if 'folds' in analyses:
        print("Analyzing folds sensitivity...")
        results['folds_sensitivity'] = folds_sensitivity(
            data, outcome, treatment, controls, model, verbose=verbose
        )

    # Trimming sensitivity (IRM only)
    if 'trimming' in analyses and model == 'irm':
        print("Analyzing trimming sensitivity...")
        results['trimming_sensitivity'] = trimming_sensitivity(
            data, outcome, treatment, controls, verbose=verbose
        )

    # Controls sensitivity
    if 'controls' in analyses:
        print("Analyzing controls sensitivity...")
        results['controls_sensitivity'] = controls_sensitivity(
            data, outcome, treatment, controls, model, verbose=verbose
        )

    return results


def generate_sensitivity_report(results, output_path: Path = None):
    """Generate markdown sensitivity report."""
    report = f"""# DDML Sensitivity Analysis Report

Generated: {results['timestamp']}

## Overview

- **Model**: {results['model'].upper()}
- **Observations**: {results['n_obs']:,}
- **Controls**: {results['n_controls']}

"""

    # Learner sensitivity
    if 'learner_sensitivity' in results:
        ls = results['learner_sensitivity']
        report += "## Learner Sensitivity\n\n"
        report += "| Learner | Effect | SE | 95% CI | P-value |\n"
        report += "|---------|--------|-----|--------|----------|\n"

        for learner, r in ls['by_learner'].items():
            if 'effect' in r:
                sig = "***" if r['p_value'] < 0.01 else "**" if r['p_value'] < 0.05 else "*" if r['p_value'] < 0.1 else ""
                report += f"| {learner} | {r['effect']:.4f}{sig} | {r['se']:.4f} | [{r['ci_lower']:.4f}, {r['ci_upper']:.4f}] | {r['p_value']:.4f} |\n"
            else:
                report += f"| {learner} | ERROR | - | - | - |\n"

        summary = ls['summary']
        if 'mean' in summary:
            report += f"\n**Summary**: CV = {summary['cv']:.2f}%, Range = [{summary['range'][0]:.4f}, {summary['range'][1]:.4f}]\n"
            report += f"All same sign: {summary['all_same_sign']}, All significant: {summary['all_significant']}\n\n"

    # Folds sensitivity
    if 'folds_sensitivity' in results:
        fs = results['folds_sensitivity']
        report += "## Folds Sensitivity\n\n"
        report += "| K | Effect | SE | 95% CI |\n"
        report += "|---|--------|-----|--------|\n"

        for k, r in fs['by_folds'].items():
            if 'effect' in r:
                report += f"| {k} | {r['effect']:.4f} | {r['se']:.4f} | [{r['ci_lower']:.4f}, {r['ci_upper']:.4f}] |\n"

        if fs['summary'].get('cv'):
            report += f"\n**Summary**: CV = {fs['summary']['cv']:.2f}%, Stable: {fs['summary']['stable']}\n\n"

    # Trimming sensitivity
    if 'trimming_sensitivity' in results:
        ts = results['trimming_sensitivity']
        report += "## Trimming Sensitivity (IRM)\n\n"
        report += "| Threshold | Effect | SE | N Trimmed |\n"
        report += "|-----------|--------|-----|------------|\n"

        for thresh, r in ts['by_threshold'].items():
            if 'effect' in r:
                report += f"| {thresh} | {r['effect']:.4f} | {r['se']:.4f} | {r['n_trimmed']} |\n"

        report += f"\n**Sensitivity**: {ts['summary']['sensitivity']}\n\n"

    # Controls sensitivity
    if 'controls_sensitivity' in results:
        cs = results['controls_sensitivity']
        report += "## Controls Sensitivity\n\n"
        report += "| Control Set | N | Effect | SE | 95% CI |\n"
        report += "|-------------|---|--------|-----|--------|\n"

        for name, r in cs['by_control_set'].items():
            if 'effect' in r:
                report += f"| {name} | {r['n_controls']} | {r['effect']:.4f} | {r['se']:.4f} | [{r['ci_lower']:.4f}, {r['ci_upper']:.4f}] |\n"

        report += f"\n**Sensitivity**: {cs['summary']['sensitivity']}\n\n"

    # Overall assessment
    report += "## Overall Assessment\n\n"

    assessments = []
    if 'learner_sensitivity' in results and results['learner_sensitivity']['summary'].get('cv'):
        cv = results['learner_sensitivity']['summary']['cv']
        if cv < 5:
            assessments.append("Learner choice: HIGHLY ROBUST")
        elif cv < 15:
            assessments.append("Learner choice: ROBUST")
        else:
            assessments.append("Learner choice: SENSITIVE - interpret with caution")

    if 'folds_sensitivity' in results and results['folds_sensitivity']['summary'].get('stable') is not None:
        if results['folds_sensitivity']['summary']['stable']:
            assessments.append("Cross-fitting: STABLE")
        else:
            assessments.append("Cross-fitting: Some instability")

    for a in assessments:
        report += f"- {a}\n"

    if output_path:
        with open(output_path, 'w') as f:
            f.write(report)

    return report


def main():
    args = parse_args()

    print("=" * 60)
    print("DDML SENSITIVITY ANALYSIS")
    print("=" * 60)

    # Load data
    data = pd.read_csv(args.data)

    # Get controls
    if args.controls:
        controls = [c.strip() for c in args.controls.split(',')]
    elif args.all_controls:
        controls = [c for c in data.columns if c not in [args.outcome, args.treatment]]
    else:
        controls = [c for c in data.columns if c not in [args.outcome, args.treatment]]

    # Drop NAs
    df = data[[args.outcome, args.treatment] + controls].dropna()

    # Determine model
    model = args.model
    if model == 'auto':
        d = df[args.treatment]
        is_binary = set(d.unique()).issubset({0, 1})
        model = 'irm' if is_binary else 'plr'
        print(f"Auto-selected model: {model.upper()}")

    print(f"\nData: {len(df)} observations, {len(controls)} controls")
    print(f"Model: {model.upper()}")

    # Parse analyses
    analyses = [a.strip() for a in args.analyses.split(',')]

    # Run sensitivity
    results = run_comprehensive_sensitivity(
        df, args.outcome, args.treatment, controls, model,
        analyses, verbose=args.verbose
    )

    # Generate report
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)

    report = generate_sensitivity_report(results)
    print(report)

    # Save results
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\nResults saved to: {args.output}")

        report_path = Path(args.output).with_suffix('.md')
        generate_sensitivity_report(results, report_path)
        print(f"Report saved to: {report_path}")


if __name__ == '__main__':
    main()
