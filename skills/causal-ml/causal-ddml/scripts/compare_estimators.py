#!/usr/bin/env python3
"""
Compare DDML Estimators

Systematically compares different DDML model specifications:
- PLR vs IRM
- Different score functions
- With and without cross-fitting repetitions
- Alternative nuisance model combinations

Usage:
    python compare_estimators.py --data data.csv --outcome y --treatment d
    python compare_estimators.py --data data.csv --outcome y --treatment d --output comparison.json

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
from ddml_estimator import (
    estimate_plr, estimate_irm, create_ddml_data,
    compare_learners, validate_ddml_setup
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare DDML estimator specifications"
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
    parser.add_argument('--learners', '-l',
                        default='lasso,random_forest,xgboost',
                        help='Learners to compare')
    parser.add_argument('--n-folds', '-k', type=int, default=5,
                        help='Cross-fitting folds')
    parser.add_argument('--n-rep', '-r', type=int, default=1,
                        help='Cross-fitting repetitions')
    parser.add_argument('--output', '-o',
                        help='Output file (JSON)')
    parser.add_argument('--verbose', '-v', action='store_true')

    return parser.parse_args()


def compare_plr_irm(data, outcome, treatment, controls, learner='lasso',
                    n_folds=5, n_rep=1, verbose=False):
    """
    Compare PLR and IRM estimators (binary treatment only).
    """
    results = {}

    # PLR
    if verbose:
        print("  Estimating PLR...")

    try:
        plr = estimate_plr(
            data=data, outcome=outcome, treatment=treatment,
            controls=controls, ml_l=learner, ml_m=learner,
            n_folds=n_folds, n_rep=n_rep
        )
        results['plr'] = {
            'effect': plr.effect,
            'se': plr.se,
            'ci_lower': plr.ci_lower,
            'ci_upper': plr.ci_upper,
            'p_value': plr.p_value,
            'model': 'Partially Linear Regression',
            'estimand': 'ATE (constant effect assumption)'
        }
    except Exception as e:
        results['plr'] = {'error': str(e)}

    # IRM
    if verbose:
        print("  Estimating IRM...")

    try:
        ml_m = 'logistic_lasso' if learner in ['lasso', 'ridge'] else learner
        irm = estimate_irm(
            data=data, outcome=outcome, treatment=treatment,
            controls=controls, ml_g=learner, ml_m=ml_m,
            n_folds=n_folds, n_rep=n_rep
        )
        results['irm'] = {
            'effect': irm.effect,
            'se': irm.se,
            'ci_lower': irm.ci_lower,
            'ci_upper': irm.ci_upper,
            'p_value': irm.p_value,
            'model': 'Interactive Regression Model',
            'estimand': 'ATE (allows heterogeneous effects)'
        }
    except Exception as e:
        results['irm'] = {'error': str(e)}

    # Compare
    if 'effect' in results.get('plr', {}) and 'effect' in results.get('irm', {}):
        plr_effect = results['plr']['effect']
        irm_effect = results['irm']['effect']
        diff = abs(plr_effect - irm_effect)
        avg = (abs(plr_effect) + abs(irm_effect)) / 2

        results['comparison'] = {
            'difference': diff,
            'relative_difference': diff / avg * 100 if avg > 0 else 0,
            'suggests_heterogeneity': diff / avg > 0.1 if avg > 0 else False,
            'recommendation': 'Models agree' if diff / avg < 0.1 else 'Consider heterogeneous effects'
        }

    return results


def compare_score_functions(data, outcome, treatment, controls, learner='lasso',
                            n_folds=5, verbose=False):
    """
    Compare different score functions for PLR.
    """
    results = {}

    # Partialling out score
    if verbose:
        print("  Partialling out score...")

    try:
        po = estimate_plr(
            data=data, outcome=outcome, treatment=treatment,
            controls=controls, ml_l=learner, ml_m=learner,
            n_folds=n_folds, score='partialling out'
        )
        results['partialling_out'] = {
            'effect': po.effect,
            'se': po.se,
            'ci_lower': po.ci_lower,
            'ci_upper': po.ci_upper,
            'description': 'Uses both Y|X and D|X nuisance functions'
        }
    except Exception as e:
        results['partialling_out'] = {'error': str(e)}

    # IV-type score
    if verbose:
        print("  IV-type score...")

    try:
        iv = estimate_plr(
            data=data, outcome=outcome, treatment=treatment,
            controls=controls, ml_l=learner, ml_m=learner,
            n_folds=n_folds, score='IV-type'
        )
        results['iv_type'] = {
            'effect': iv.effect,
            'se': iv.se,
            'ci_lower': iv.ci_lower,
            'ci_upper': iv.ci_upper,
            'description': 'Uses D-m(X) as instrument'
        }
    except Exception as e:
        results['iv_type'] = {'error': str(e)}

    return results


def compare_learner_combinations(data, outcome, treatment, controls,
                                 learners_outcome=None, learners_treatment=None,
                                 n_folds=5, verbose=False):
    """
    Compare different combinations of outcome and treatment learners.
    """
    if learners_outcome is None:
        learners_outcome = ['lasso', 'random_forest']
    if learners_treatment is None:
        learners_treatment = ['lasso', 'random_forest']

    results = {}

    for ml_l in learners_outcome:
        for ml_m in learners_treatment:
            key = f"{ml_l}_outcome_{ml_m}_treatment"
            if verbose:
                print(f"  Testing {key}...")

            try:
                result = estimate_plr(
                    data=data, outcome=outcome, treatment=treatment,
                    controls=controls, ml_l=ml_l, ml_m=ml_m,
                    n_folds=n_folds
                )
                results[key] = {
                    'ml_l': ml_l,
                    'ml_m': ml_m,
                    'effect': result.effect,
                    'se': result.se,
                    'ci_lower': result.ci_lower,
                    'ci_upper': result.ci_upper,
                    'p_value': result.p_value
                }
            except Exception as e:
                results[key] = {'error': str(e)}

    # Summary
    valid = [r for r in results.values() if 'effect' in r]
    if valid:
        effects = [r['effect'] for r in valid]
        results['summary'] = {
            'n_combinations': len(valid),
            'mean_effect': np.mean(effects),
            'std_effect': np.std(effects),
            'cv': np.std(effects) / abs(np.mean(effects)) * 100 if np.mean(effects) != 0 else 0,
            'range': (min(effects), max(effects))
        }

    return results


def compare_repetitions(data, outcome, treatment, controls, learner='lasso',
                        n_folds=5, rep_values=None, verbose=False):
    """
    Compare estimates with different numbers of cross-fitting repetitions.
    """
    if rep_values is None:
        rep_values = [1, 3, 5, 10]

    results = {}

    for n_rep in rep_values:
        if verbose:
            print(f"  Testing {n_rep} repetitions...")

        try:
            result = estimate_plr(
                data=data, outcome=outcome, treatment=treatment,
                controls=controls, ml_l=learner, ml_m=learner,
                n_folds=n_folds, n_rep=n_rep
            )
            results[n_rep] = {
                'effect': result.effect,
                'se': result.se,
                'ci_lower': result.ci_lower,
                'ci_upper': result.ci_upper
            }
        except Exception as e:
            results[n_rep] = {'error': str(e)}

    # Analyze stability
    valid = [r for r in results.values() if 'effect' in r]
    if valid:
        effects = [r['effect'] for r in valid]
        ses = [r['se'] for r in valid]
        results['stability'] = {
            'effect_std': np.std(effects),
            'se_trend': 'decreasing' if ses[-1] < ses[0] else 'stable',
            'converged': np.std(effects[-3:]) < 0.01 * abs(np.mean(effects)) if len(effects) >= 3 else False
        }

    return results


def run_full_comparison(data, outcome, treatment, controls, learners,
                        n_folds=5, n_rep=1, verbose=False):
    """
    Run comprehensive estimator comparison.
    """
    # Check if binary treatment
    ddml_data = create_ddml_data(data, outcome, treatment, controls)
    is_binary = ddml_data.treatment_type == 'binary'

    results = {
        'timestamp': datetime.now().isoformat(),
        'n_obs': len(data),
        'n_controls': len(controls),
        'treatment_type': ddml_data.treatment_type,
        'comparisons': {}
    }

    # Compare learners (common analysis)
    print("\n1. Comparing ML learners...")
    learner_list = [l.strip() for l in learners.split(',')]
    results['comparisons']['learners'] = {}

    for model in ['plr'] + (['irm'] if is_binary else []):
        if verbose:
            print(f"  Model: {model.upper()}")
        comparison = compare_learners(
            data=data, outcome=outcome, treatment=treatment,
            controls=controls, learner_list=learner_list,
            model=model, n_folds=n_folds
        )
        results['comparisons']['learners'][model] = {
            learner: {
                'effect': r.effect,
                'se': r.se,
                'p_value': r.p_value
            }
            for learner, r in comparison.results.items()
        }
        results['comparisons']['learners'][f'{model}_sensitivity'] = comparison.sensitivity

    # PLR vs IRM (binary treatment only)
    if is_binary:
        print("\n2. Comparing PLR vs IRM...")
        results['comparisons']['plr_vs_irm'] = compare_plr_irm(
            data, outcome, treatment, controls,
            learner=learner_list[0], n_folds=n_folds, n_rep=n_rep,
            verbose=verbose
        )

    # Compare score functions (PLR only)
    print("\n3. Comparing score functions...")
    results['comparisons']['score_functions'] = compare_score_functions(
        data, outcome, treatment, controls,
        learner=learner_list[0], n_folds=n_folds, verbose=verbose
    )

    # Compare learner combinations
    print("\n4. Comparing learner combinations...")
    results['comparisons']['learner_combinations'] = compare_learner_combinations(
        data, outcome, treatment, controls,
        learners_outcome=learner_list[:2],
        learners_treatment=learner_list[:2],
        n_folds=n_folds, verbose=verbose
    )

    # Compare repetitions
    print("\n5. Comparing cross-fitting repetitions...")
    results['comparisons']['repetitions'] = compare_repetitions(
        data, outcome, treatment, controls,
        learner=learner_list[0], n_folds=n_folds,
        verbose=verbose
    )

    return results


def generate_comparison_report(results):
    """Generate comparison report."""
    report = f"""# DDML Estimator Comparison Report

Generated: {results['timestamp']}

## Data Summary

- **Observations**: {results['n_obs']:,}
- **Controls**: {results['n_controls']}
- **Treatment Type**: {results['treatment_type']}

## 1. Learner Comparison

"""

    # Learner results
    for model, learner_results in results['comparisons']['learners'].items():
        if model.endswith('_sensitivity'):
            continue
        report += f"\n### {model.upper()}\n\n"
        report += "| Learner | Effect | SE | P-value |\n"
        report += "|---------|--------|-----|----------|\n"

        for learner, r in learner_results.items():
            if 'effect' in r:
                sig = "***" if r['p_value'] < 0.01 else "**" if r['p_value'] < 0.05 else "*" if r['p_value'] < 0.1 else ""
                report += f"| {learner} | {r['effect']:.4f}{sig} | {r['se']:.4f} | {r['p_value']:.4f} |\n"

    # PLR vs IRM
    if 'plr_vs_irm' in results['comparisons']:
        report += "\n## 2. PLR vs IRM Comparison\n\n"
        pvi = results['comparisons']['plr_vs_irm']

        report += "| Model | Effect | SE | 95% CI | Estimand |\n"
        report += "|-------|--------|-----|--------|----------|\n"

        for model in ['plr', 'irm']:
            if model in pvi and 'effect' in pvi[model]:
                r = pvi[model]
                report += f"| {model.upper()} | {r['effect']:.4f} | {r['se']:.4f} | [{r['ci_lower']:.4f}, {r['ci_upper']:.4f}] | {r['estimand']} |\n"

        if 'comparison' in pvi:
            report += f"\n**Difference**: {pvi['comparison']['relative_difference']:.1f}%\n"
            report += f"**Recommendation**: {pvi['comparison']['recommendation']}\n"

    # Score functions
    report += "\n## 3. Score Function Comparison (PLR)\n\n"
    sf = results['comparisons']['score_functions']
    report += "| Score | Effect | SE | Description |\n"
    report += "|-------|--------|-----|-------------|\n"

    for score, r in sf.items():
        if 'effect' in r:
            report += f"| {score} | {r['effect']:.4f} | {r['se']:.4f} | {r['description']} |\n"

    # Learner combinations
    report += "\n## 4. Learner Combinations\n\n"
    lc = results['comparisons']['learner_combinations']
    report += "| Outcome Model | Treatment Model | Effect | SE |\n"
    report += "|---------------|-----------------|--------|-----|\n"

    for key, r in lc.items():
        if key != 'summary' and 'effect' in r:
            report += f"| {r['ml_l']} | {r['ml_m']} | {r['effect']:.4f} | {r['se']:.4f} |\n"

    if 'summary' in lc:
        report += f"\n**CV**: {lc['summary']['cv']:.2f}%\n"

    # Repetitions
    report += "\n## 5. Cross-Fitting Repetitions\n\n"
    rep = results['comparisons']['repetitions']
    report += "| Repetitions | Effect | SE |\n"
    report += "|-------------|--------|-----|\n"

    for n_rep, r in rep.items():
        if n_rep != 'stability' and 'effect' in r:
            report += f"| {n_rep} | {r['effect']:.4f} | {r['se']:.4f} |\n"

    if 'stability' in rep:
        report += f"\n**SE Trend**: {rep['stability']['se_trend']}\n"

    # Overall recommendation
    report += "\n## Overall Recommendation\n\n"

    recommendations = []

    # Check learner robustness
    for model in ['plr', 'irm']:
        sens_key = f'{model}_sensitivity'
        if sens_key in results['comparisons']['learners']:
            sens = results['comparisons']['learners'][sens_key]
            if sens.get('all_significant', False) and sens.get('all_same_sign', False):
                recommendations.append(f"{model.upper()}: Results robust across learners")

    # Check PLR vs IRM
    if 'plr_vs_irm' in results['comparisons']:
        comp = results['comparisons']['plr_vs_irm'].get('comparison', {})
        if comp.get('suggests_heterogeneity', False):
            recommendations.append("Consider heterogeneous treatment effects (PLR and IRM differ)")
        else:
            recommendations.append("PLR and IRM agree - constant effect assumption appears reasonable")

    for rec in recommendations:
        report += f"- {rec}\n"

    return report


def main():
    args = parse_args()

    print("=" * 60)
    print("DDML ESTIMATOR COMPARISON")
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

    print(f"\nData: {len(df)} observations, {len(controls)} controls")
    print(f"Learners: {args.learners}")

    # Run comparison
    results = run_full_comparison(
        df, args.outcome, args.treatment, controls,
        learners=args.learners, n_folds=args.n_folds, n_rep=args.n_rep,
        verbose=args.verbose
    )

    # Generate report
    print("\n" + "=" * 60)
    print("COMPARISON REPORT")
    print("=" * 60)

    report = generate_comparison_report(results)
    print(report)

    # Save results
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\nResults saved to: {args.output}")

        report_path = Path(args.output).with_suffix('.md')
        with open(report_path, 'w') as f:
            f.write(report)
        print(f"Report saved to: {report_path}")


if __name__ == '__main__':
    main()
