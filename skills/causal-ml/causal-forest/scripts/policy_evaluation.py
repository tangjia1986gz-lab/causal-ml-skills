#!/usr/bin/env python3
"""
Policy Learning and Evaluation CLI

Learn optimal treatment policies from causal forest CATE estimates
and evaluate policy value.

Usage:
    python policy_evaluation.py --cate-file cate.csv --data data.csv \\
        --effect-modifiers X1 X2 X3 --treatment-cost 10 --output policy/

Author: Causal ML Skills
"""

import argparse
import sys
import os
import pickle
import json
from pathlib import Path
from typing import Optional, List, Dict, Any

import numpy as np
import pandas as pd

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Learn and evaluate treatment policies from CATE estimates',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic policy learning
    python policy_evaluation.py --cate-file cate.csv --treatment-cost 10 \\
        --output policy_results/

    # With budget constraint
    python policy_evaluation.py --cate-file cate.csv --treatment-cost 10 \\
        --budget 0.3 --output policy_results/

    # Policy tree for interpretable rules
    python policy_evaluation.py --cate-file cate.csv --data data.csv \\
        --effect-modifiers X1 X2 X3 --method policy_tree --max-depth 3 \\
        --output policy_results/

    # From fitted model
    python policy_evaluation.py --model model.pkl --data data.csv \\
        --effect-modifiers X1 X2 X3 --treatment-cost 10 --output policy_results/
        """
    )

    # Input options
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--cate-file', help='Path to CATE estimates CSV')
    input_group.add_argument('--model', help='Path to fitted model pickle')

    parser.add_argument('--data', help='Path to data file')
    parser.add_argument('--effect-modifiers', nargs='+',
                        help='Effect modifier variable names')

    # Policy specification
    parser.add_argument('--treatment-cost', type=float, default=0.0,
                        help='Cost of treatment per unit (default: 0)')
    parser.add_argument('--budget', type=float, default=None,
                        help='Budget constraint as fraction (e.g., 0.3 = treat max 30%)')
    parser.add_argument('--method', choices=['threshold', 'policy_tree', 'optimal'],
                        default='threshold',
                        help='Policy learning method (default: threshold)')

    # Policy tree options
    parser.add_argument('--max-depth', type=int, default=3,
                        help='Max depth for policy tree (default: 3)')
    parser.add_argument('--min-samples-leaf', type=int, default=50,
                        help='Min samples per leaf for policy tree (default: 50)')

    # Evaluation options
    parser.add_argument('--outcome', help='Outcome variable for policy value estimation')
    parser.add_argument('--treatment', help='Treatment variable for policy value estimation')
    parser.add_argument('--cross-validate', action='store_true',
                        help='Cross-validate policy value')
    parser.add_argument('--n-folds', type=int, default=5,
                        help='Number of CV folds (default: 5)')

    # Output options
    parser.add_argument('--output', '-o', default='./policy_results',
                        help='Output directory')
    parser.add_argument('--save-recommendations', action='store_true', default=True,
                        help='Save individual treatment recommendations')
    parser.add_argument('--visualize', action='store_true',
                        help='Generate policy visualizations')

    parser.add_argument('--verbose', '-v', action='store_true')

    return parser.parse_args()


def load_cate_data(cate_file: str) -> np.ndarray:
    """Load CATE estimates from file."""
    if not os.path.exists(cate_file):
        raise FileNotFoundError(f"CATE file not found: {cate_file}")

    df = pd.read_csv(cate_file)
    if 'cate' not in df.columns:
        raise ValueError("CATE file must contain 'cate' column")

    return df['cate'].values


def load_data(data_path: str) -> pd.DataFrame:
    """Load data file."""
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")

    if data_path.endswith('.csv'):
        return pd.read_csv(data_path)
    elif data_path.endswith('.parquet'):
        return pd.read_parquet(data_path)
    else:
        return pd.read_csv(data_path)


def threshold_policy(tau: np.ndarray, treatment_cost: float,
                     budget: Optional[float] = None) -> Dict[str, Any]:
    """
    Learn threshold-based policy.

    Treats if CATE > treatment_cost, subject to budget constraint.
    """
    n = len(tau)
    net_benefit = tau - treatment_cost

    # Initial recommendations
    recommendations = (net_benefit > 0).astype(int)

    threshold = 0.0

    # Apply budget constraint if specified
    if budget is not None:
        max_treated = int(n * budget)
        if recommendations.sum() > max_treated:
            # Find threshold that satisfies budget
            sorted_benefit = np.sort(net_benefit)[::-1]
            threshold = sorted_benefit[max_treated - 1]
            recommendations = (net_benefit >= threshold).astype(int)

    treatment_rate = recommendations.mean()
    policy_value = np.mean(tau * recommendations) - treatment_cost * treatment_rate

    return {
        'recommendations': recommendations,
        'threshold': threshold,
        'treatment_rate': treatment_rate,
        'policy_value': policy_value,
        'method': 'threshold'
    }


def policy_tree_learning(tau: np.ndarray, X: np.ndarray,
                         feature_names: List[str],
                         treatment_cost: float,
                         budget: Optional[float],
                         max_depth: int,
                         min_samples_leaf: int) -> Dict[str, Any]:
    """
    Learn interpretable policy tree.
    """
    from sklearn.tree import DecisionTreeRegressor, export_text

    n = len(tau)
    net_benefit = tau - treatment_cost

    # Fit tree on net benefit
    tree = DecisionTreeRegressor(
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf
    )
    tree.fit(X, net_benefit)

    # Get predictions
    pred_benefit = tree.predict(X)
    recommendations = (pred_benefit > 0).astype(int)

    # Apply budget constraint
    if budget is not None:
        max_treated = int(n * budget)
        if recommendations.sum() > max_treated:
            sorted_idx = np.argsort(pred_benefit)[::-1]
            recommendations = np.zeros(n, dtype=int)
            recommendations[sorted_idx[:max_treated]] = 1

    treatment_rate = recommendations.mean()
    policy_value = np.mean(tau * recommendations) - treatment_cost * treatment_rate

    # Extract rules
    rules_text = export_text(tree, feature_names=feature_names)

    # Get leaf statistics
    leaves = tree.apply(X)
    leaf_stats = []
    for leaf_id in np.unique(leaves):
        mask = leaves == leaf_id
        leaf_stats.append({
            'leaf_id': int(leaf_id),
            'n_samples': int(mask.sum()),
            'mean_cate': float(tau[mask].mean()),
            'mean_net_benefit': float(net_benefit[mask].mean()),
            'recommendation': 'TREAT' if net_benefit[mask].mean() > 0 else 'CONTROL'
        })

    return {
        'recommendations': recommendations,
        'treatment_rate': treatment_rate,
        'policy_value': policy_value,
        'method': 'policy_tree',
        'tree': tree,
        'rules_text': rules_text,
        'leaf_stats': leaf_stats
    }


def optimal_policy(tau: np.ndarray, treatment_cost: float,
                   budget: Optional[float] = None) -> Dict[str, Any]:
    """
    Learn optimal (greedy) policy.

    Assigns treatment to maximize total benefit subject to budget.
    """
    n = len(tau)
    net_benefit = tau - treatment_cost

    if budget is not None:
        max_treated = int(n * budget)
        # Treat top max_treated by net benefit
        top_idx = np.argsort(net_benefit)[::-1][:max_treated]
        recommendations = np.zeros(n, dtype=int)
        recommendations[top_idx] = 1
    else:
        # Treat all with positive net benefit
        recommendations = (net_benefit > 0).astype(int)

    treatment_rate = recommendations.mean()
    policy_value = np.mean(tau * recommendations) - treatment_cost * treatment_rate

    return {
        'recommendations': recommendations,
        'treatment_rate': treatment_rate,
        'policy_value': policy_value,
        'method': 'optimal'
    }


def evaluate_policy_value(recommendations: np.ndarray, y: np.ndarray,
                          treatment: np.ndarray, treatment_cost: float) -> Dict[str, float]:
    """
    Evaluate policy value using inverse propensity weighting.
    """
    n = len(y)

    # Simple evaluation assuming randomization
    # (More sophisticated methods would use IPW or AIPW)

    # Estimate propensity
    p = treatment.mean()

    # Policy value components
    treated_rec = (treatment == 1) & (recommendations == 1)
    control_rec = (treatment == 0) & (recommendations == 0)
    treated_not_rec = (treatment == 1) & (recommendations == 0)
    control_not_rec = (treatment == 0) & (recommendations == 1)

    # IPW estimates
    if treated_rec.sum() > 0 and control_rec.sum() > 0:
        value_follow_policy = (
            np.sum(y[treated_rec] / p) +
            np.sum(y[control_rec] / (1 - p))
        ) / n
    else:
        value_follow_policy = np.nan

    # Baseline values
    value_treat_all = np.mean(y[treatment == 1]) - treatment_cost
    value_treat_none = np.mean(y[treatment == 0])

    return {
        'policy_value_ipw': value_follow_policy,
        'treat_all_value': value_treat_all,
        'treat_none_value': value_treat_none,
        'improvement_over_treat_all': (value_follow_policy - value_treat_all) / abs(value_treat_all) if not np.isnan(value_follow_policy) else np.nan,
        'improvement_over_treat_none': (value_follow_policy - value_treat_none) / abs(value_treat_none) if not np.isnan(value_follow_policy) else np.nan
    }


def cross_validate_policy(tau: np.ndarray, X: Optional[np.ndarray],
                          y: np.ndarray, treatment: np.ndarray,
                          method: str, treatment_cost: float,
                          budget: Optional[float],
                          n_folds: int,
                          feature_names: Optional[List[str]] = None,
                          max_depth: int = 3,
                          min_samples_leaf: int = 50) -> Dict[str, Any]:
    """
    Cross-validate policy learning.
    """
    from sklearn.model_selection import KFold

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    fold_results = []

    for fold, (train_idx, test_idx) in enumerate(kf.split(tau)):
        # Train policy on training fold
        tau_train = tau[train_idx]
        X_train = X[train_idx] if X is not None else None

        if method == 'threshold':
            policy = threshold_policy(tau_train, treatment_cost, budget)
        elif method == 'policy_tree' and X is not None:
            policy = policy_tree_learning(
                tau_train, X_train, feature_names,
                treatment_cost, budget, max_depth, min_samples_leaf
            )
        else:
            policy = optimal_policy(tau_train, treatment_cost, budget)

        # Get recommendations for test fold
        if method == 'policy_tree' and 'tree' in policy:
            X_test = X[test_idx]
            pred_benefit = policy['tree'].predict(X_test)
            test_rec = (pred_benefit > 0).astype(int)
        else:
            tau_test = tau[test_idx]
            net_benefit = tau_test - treatment_cost
            test_rec = (net_benefit > policy.get('threshold', 0)).astype(int)

        # Evaluate on test fold
        y_test = y[test_idx]
        t_test = treatment[test_idx]

        eval_results = evaluate_policy_value(test_rec, y_test, t_test, treatment_cost)

        fold_results.append({
            'fold': fold,
            'treatment_rate': test_rec.mean(),
            **eval_results
        })

    # Aggregate results
    fold_df = pd.DataFrame(fold_results)

    return {
        'mean_policy_value': fold_df['policy_value_ipw'].mean(),
        'std_policy_value': fold_df['policy_value_ipw'].std(),
        'mean_treatment_rate': fold_df['treatment_rate'].mean(),
        'fold_results': fold_results
    }


def visualize_policy(tau: np.ndarray, recommendations: np.ndarray,
                     X: Optional[pd.DataFrame], policy_result: Dict,
                     output_dir: str, verbose: bool):
    """Generate policy visualizations."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        if verbose:
            print("matplotlib not available, skipping visualizations")
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # 1. CATE distribution by recommendation
    ax1 = axes[0, 0]
    ax1.hist(tau[recommendations == 1], bins=30, alpha=0.6,
             label='Treat', color='green', density=True)
    ax1.hist(tau[recommendations == 0], bins=30, alpha=0.6,
             label='Control', color='red', density=True)
    ax1.axvline(0, color='black', linestyle='--', linewidth=2)
    ax1.set_xlabel('CATE')
    ax1.set_ylabel('Density')
    ax1.set_title('CATE by Policy Recommendation')
    ax1.legend()

    # 2. Net benefit distribution
    ax2 = axes[0, 1]
    treatment_cost = policy_result.get('treatment_cost', 0)
    net_benefit = tau - treatment_cost
    ax2.hist(net_benefit, bins=50, alpha=0.7, color='steelblue')
    ax2.axvline(0, color='red', linestyle='--', linewidth=2,
                label='Treatment threshold')
    if 'threshold' in policy_result:
        ax2.axvline(policy_result['threshold'], color='green',
                   linestyle='-', linewidth=2, label='Budget threshold')
    ax2.set_xlabel('Net Benefit (CATE - Cost)')
    ax2.set_ylabel('Count')
    ax2.set_title('Net Benefit Distribution')
    ax2.legend()

    # 3. Policy summary
    ax3 = axes[1, 0]
    ax3.axis('off')
    summary = f"""
    Policy Summary
    ==============

    Method: {policy_result.get('method', 'N/A')}
    Treatment Rate: {policy_result.get('treatment_rate', 0):.1%}
    Policy Value: {policy_result.get('policy_value', 0):.4f}

    N Treated: {recommendations.sum():,}
    N Control: {(~recommendations.astype(bool)).sum():,}

    Mean CATE (Treated): {tau[recommendations == 1].mean():.4f}
    Mean CATE (Control): {tau[recommendations == 0].mean():.4f}

    Treatment Cost: {treatment_cost:.2f}
    Budget: {policy_result.get('budget', 'None')}
    """
    ax3.text(0.1, 0.5, summary, fontsize=11, family='monospace',
             verticalalignment='center')

    # 4. Cumulative benefit curve
    ax4 = axes[1, 1]
    sorted_idx = np.argsort(net_benefit)[::-1]
    cumulative_benefit = np.cumsum(net_benefit[sorted_idx])
    fraction_treated = np.arange(1, len(tau)+1) / len(tau)

    ax4.plot(fraction_treated, cumulative_benefit, 'b-', linewidth=2)
    ax4.axhline(0, color='gray', linestyle='--', alpha=0.5)

    # Mark optimal point
    if len(cumulative_benefit) > 0:
        optimal_idx = np.argmax(cumulative_benefit)
        ax4.scatter([fraction_treated[optimal_idx]], [cumulative_benefit[optimal_idx]],
                   color='red', s=100, zorder=5, label=f'Optimal: {fraction_treated[optimal_idx]:.1%}')

    ax4.set_xlabel('Fraction Treated')
    ax4.set_ylabel('Cumulative Net Benefit')
    ax4.set_title('Cumulative Benefit Curve')
    ax4.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'policy_analysis.png'),
               dpi=300, bbox_inches='tight')
    plt.close()

    if verbose:
        print(f"  Visualization saved to {output_dir}/policy_analysis.png")


def main():
    """Main entry point."""
    args = parse_args()

    try:
        os.makedirs(args.output, exist_ok=True)

        # Load CATE estimates
        if args.cate_file:
            tau = load_cate_data(args.cate_file)
        else:
            # Load from model
            with open(args.model, 'rb') as f:
                model = pickle.load(f)
            data = load_data(args.data)
            X = data[args.effect_modifiers].values
            tau, _ = model.predict(X, return_std=False)

        if args.verbose:
            print(f"Loaded {len(tau)} CATE estimates")
            print(f"CATE range: [{tau.min():.4f}, {tau.max():.4f}]")

        # Load data if needed
        X = None
        feature_names = None
        if args.data and args.effect_modifiers:
            data = load_data(args.data)
            X = data[args.effect_modifiers].values
            feature_names = args.effect_modifiers

        # Learn policy
        if args.verbose:
            print(f"\nLearning policy (method: {args.method})...")
            print(f"  Treatment cost: {args.treatment_cost}")
            print(f"  Budget: {args.budget if args.budget else 'None'}")

        if args.method == 'threshold':
            policy = threshold_policy(tau, args.treatment_cost, args.budget)
        elif args.method == 'policy_tree':
            if X is None:
                raise ValueError("--data and --effect-modifiers required for policy_tree")
            policy = policy_tree_learning(
                tau, X, feature_names, args.treatment_cost, args.budget,
                args.max_depth, args.min_samples_leaf
            )
        else:
            policy = optimal_policy(tau, args.treatment_cost, args.budget)

        policy['treatment_cost'] = args.treatment_cost
        policy['budget'] = args.budget

        if args.verbose:
            print(f"\nPolicy Results:")
            print(f"  Treatment rate: {policy['treatment_rate']:.1%}")
            print(f"  Policy value: {policy['policy_value']:.4f}")

        # Cross-validate if requested
        cv_results = None
        if args.cross_validate and args.outcome and args.treatment and args.data:
            if args.verbose:
                print(f"\nCross-validating policy ({args.n_folds} folds)...")

            data = load_data(args.data)
            y = data[args.outcome].values
            treatment_var = data[args.treatment].values

            cv_results = cross_validate_policy(
                tau, X, y, treatment_var,
                args.method, args.treatment_cost, args.budget,
                args.n_folds, feature_names,
                args.max_depth, args.min_samples_leaf
            )

            if args.verbose:
                print(f"  CV Policy Value: {cv_results['mean_policy_value']:.4f} "
                      f"(+/- {cv_results['std_policy_value']:.4f})")

        # Save results
        results = {
            'method': args.method,
            'treatment_cost': args.treatment_cost,
            'budget': args.budget,
            'treatment_rate': policy['treatment_rate'],
            'policy_value': policy['policy_value'],
            'n_treated': int(policy['recommendations'].sum()),
            'n_total': len(policy['recommendations'])
        }

        if 'threshold' in policy:
            results['threshold'] = policy['threshold']

        if 'rules_text' in policy:
            results['rules_text'] = policy['rules_text']
            results['leaf_stats'] = policy['leaf_stats']

        if cv_results:
            results['cv_mean_value'] = cv_results['mean_policy_value']
            results['cv_std_value'] = cv_results['std_policy_value']
            results['cv_fold_results'] = cv_results['fold_results']

        # Save JSON results
        with open(os.path.join(args.output, 'policy_results.json'), 'w') as f:
            json.dump(results, f, indent=2, default=str)

        # Save recommendations
        if args.save_recommendations:
            rec_df = pd.DataFrame({
                'recommendation': policy['recommendations'],
                'cate': tau,
                'net_benefit': tau - args.treatment_cost
            })
            rec_df.to_csv(os.path.join(args.output, 'recommendations.csv'), index=False)

        # Save policy tree rules
        if 'rules_text' in policy:
            with open(os.path.join(args.output, 'policy_rules.txt'), 'w') as f:
                f.write(policy['rules_text'])

            leaf_df = pd.DataFrame(policy['leaf_stats'])
            leaf_df.to_csv(os.path.join(args.output, 'leaf_statistics.csv'), index=False)

        # Visualize
        if args.visualize:
            visualize_policy(tau, policy['recommendations'], None, policy,
                           args.output, args.verbose)

        # Print summary
        print(f"\n{'='*50}")
        print("POLICY LEARNING SUMMARY")
        print('='*50)
        print(f"Method: {args.method}")
        print(f"Treatment rate: {policy['treatment_rate']:.1%}")
        print(f"Policy value: {policy['policy_value']:.4f}")
        print(f"N treated: {policy['recommendations'].sum():,} / {len(tau):,}")

        if cv_results:
            print(f"\nCross-Validated Value: {cv_results['mean_policy_value']:.4f} "
                  f"(SD: {cv_results['std_policy_value']:.4f})")

        print(f"\nResults saved to {args.output}/")
        print('='*50)

        return 0

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
