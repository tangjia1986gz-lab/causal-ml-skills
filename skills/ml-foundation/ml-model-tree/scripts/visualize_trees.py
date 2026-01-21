#!/usr/bin/env python3
"""
Tree Visualization Script

This script provides visualizations for tree-based models including:
- Tree structure visualization
- Partial dependence plots (PDP)
- Individual conditional expectation (ICE) plots
- SHAP visualizations

Usage:
    python visualize_trees.py model.joblib data.csv --target y --pdp income,age
    python visualize_trees.py model.joblib data.csv --target y --tree-plot
    python visualize_trees.py model.joblib data.csv --target y --shap-summary
"""

import argparse
import sys
import warnings
from pathlib import Path
from typing import Dict, Any, List, Optional

import numpy as np
import pandas as pd
import joblib

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


def load_model_and_data(model_path: str, data_path: str, target: str,
                        features: Optional[str] = None) -> Dict[str, Any]:
    """Load model and data for visualization."""
    print(f"Loading model from: {model_path}")
    model = joblib.load(model_path)

    print(f"Loading data from: {data_path}")
    df = pd.read_csv(data_path)

    if features:
        feature_cols = [f.strip() for f in features.split(',')]
    else:
        feature_cols = [c for c in df.columns if c != target]

    X = df[feature_cols]
    y = df[target].values

    # Determine task
    unique_values = len(np.unique(y))
    task = 'classification' if unique_values < 10 else 'regression'

    return {
        'model': model,
        'X': X,
        'y': y,
        'feature_names': feature_cols,
        'task': task,
        'df': df
    }


def plot_tree_structure(model: Any, feature_names: List[str], max_depth: int = 4,
                        output_path: Optional[str] = None) -> None:
    """Visualize tree structure for Decision Tree or first tree in ensemble."""
    import matplotlib.pyplot as plt
    from sklearn.tree import plot_tree

    print("\n=== Tree Structure Visualization ===")

    # Extract tree for different model types
    if hasattr(model, 'tree_'):
        # Single decision tree
        tree_to_plot = model
        title = "Decision Tree Structure"
    elif hasattr(model, 'estimators_'):
        # Random Forest or other ensemble
        tree_to_plot = model.estimators_[0]
        title = "First Tree in Ensemble (of {})".format(len(model.estimators_))
    elif hasattr(model, 'get_booster'):
        # XGBoost
        print("For XGBoost trees, use XGBoost's built-in plotting:")
        print("  import xgboost as xgb")
        print("  xgb.plot_tree(model, num_trees=0)")
        return
    elif hasattr(model, 'booster_'):
        # LightGBM
        print("For LightGBM trees, use LightGBM's built-in plotting:")
        print("  import lightgbm as lgb")
        print("  lgb.plot_tree(model, tree_index=0)")
        return
    else:
        print("Tree structure visualization not supported for this model type.")
        return

    fig, ax = plt.subplots(figsize=(20, 12))

    plot_tree(
        tree_to_plot,
        feature_names=feature_names,
        filled=True,
        rounded=True,
        ax=ax,
        max_depth=max_depth,
        fontsize=10
    )

    ax.set_title(title, fontsize=14)

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved to: {output_path}")

    plt.tight_layout()
    plt.show()


def plot_partial_dependence(data: Dict[str, Any], features: List[str],
                            grid_resolution: int = 50,
                            output_path: Optional[str] = None) -> None:
    """Generate partial dependence plots."""
    import matplotlib.pyplot as plt
    from sklearn.inspection import PartialDependenceDisplay

    print(f"\n=== Partial Dependence Plots ===")
    print(f"Features: {features}")

    model = data['model']
    X = data['X']
    feature_names = data['feature_names']

    # Convert feature names to indices
    feature_indices = []
    for f in features:
        if f in feature_names:
            feature_indices.append(feature_names.index(f))
        else:
            print(f"Warning: Feature '{f}' not found, skipping.")

    if not feature_indices:
        print("No valid features to plot.")
        return

    # Number of plots
    n_features = len(feature_indices)
    n_cols = min(3, n_features)
    n_rows = (n_features + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
    if n_features == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    PartialDependenceDisplay.from_estimator(
        model, X, feature_indices,
        feature_names=feature_names,
        grid_resolution=grid_resolution,
        ax=axes[:n_features]
    )

    # Hide unused axes
    for i in range(n_features, len(axes)):
        axes[i].set_visible(False)

    plt.suptitle('Partial Dependence Plots', fontsize=14, y=1.02)
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved to: {output_path}")

    plt.show()


def plot_ice(data: Dict[str, Any], feature: str, n_samples: int = 100,
             centered: bool = False, output_path: Optional[str] = None) -> None:
    """Generate Individual Conditional Expectation plots."""
    import matplotlib.pyplot as plt
    from sklearn.inspection import PartialDependenceDisplay

    print(f"\n=== ICE Plot for: {feature} ===")

    model = data['model']
    X = data['X']
    feature_names = data['feature_names']

    if feature not in feature_names:
        print(f"Feature '{feature}' not found.")
        return

    feature_idx = feature_names.index(feature)

    # Subsample if needed
    if len(X) > n_samples:
        X_sample = X.sample(n=n_samples, random_state=42)
    else:
        X_sample = X

    fig, ax = plt.subplots(figsize=(10, 6))

    kind = 'both'  # Show both ICE curves and PDP

    display = PartialDependenceDisplay.from_estimator(
        model, X_sample, [feature_idx],
        feature_names=feature_names,
        kind=kind,
        centered=centered,
        ax=ax,
        ice_lines_kw={'alpha': 0.3, 'linewidth': 0.5},
        pd_line_kw={'linewidth': 3, 'color': 'red'}
    )

    ax.set_title(f'ICE Plot: {feature}', fontsize=14)

    if centered:
        ax.set_ylabel('Centered Prediction')

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved to: {output_path}")

    plt.show()


def plot_2d_pdp(data: Dict[str, Any], features: List[str],
                output_path: Optional[str] = None) -> None:
    """Generate 2D partial dependence plot (interaction)."""
    import matplotlib.pyplot as plt
    from sklearn.inspection import PartialDependenceDisplay

    if len(features) != 2:
        print("2D PDP requires exactly 2 features.")
        return

    print(f"\n=== 2D Partial Dependence Plot ===")
    print(f"Features: {features[0]} x {features[1]}")

    model = data['model']
    X = data['X']
    feature_names = data['feature_names']

    # Get feature indices
    feature_indices = [(feature_names.index(features[0]), feature_names.index(features[1]))]

    fig, ax = plt.subplots(figsize=(10, 8))

    PartialDependenceDisplay.from_estimator(
        model, X, feature_indices,
        feature_names=feature_names,
        ax=ax
    )

    ax.set_title(f'2D PDP: {features[0]} x {features[1]}', fontsize=14)

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved to: {output_path}")

    plt.show()


def plot_shap_summary(data: Dict[str, Any], max_samples: int = 500,
                      output_path: Optional[str] = None) -> None:
    """Generate SHAP summary plot."""
    try:
        import shap
    except ImportError:
        print("SHAP not installed. Install with: pip install shap")
        return

    import matplotlib.pyplot as plt

    print("\n=== SHAP Summary Plot ===")

    model = data['model']
    X = data['X']
    feature_names = data['feature_names']

    # Subsample if needed
    if len(X) > max_samples:
        X_sample = X.sample(n=max_samples, random_state=42).values
    else:
        X_sample = X.values

    # Create explainer
    try:
        explainer = shap.TreeExplainer(model)
    except Exception:
        explainer = shap.Explainer(model, X_sample)

    shap_values = explainer.shap_values(X_sample)

    # Handle multi-class
    if isinstance(shap_values, list):
        if len(shap_values) == 2:
            shap_values = shap_values[1]
        else:
            shap_values = shap_values[0]

    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, X_sample, feature_names=feature_names, show=False)

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved to: {output_path}")

    plt.show()


def plot_shap_dependence(data: Dict[str, Any], feature: str, interaction_feature: Optional[str] = None,
                         max_samples: int = 500, output_path: Optional[str] = None) -> None:
    """Generate SHAP dependence plot."""
    try:
        import shap
    except ImportError:
        print("SHAP not installed. Install with: pip install shap")
        return

    import matplotlib.pyplot as plt

    print(f"\n=== SHAP Dependence Plot: {feature} ===")

    model = data['model']
    X = data['X']
    feature_names = data['feature_names']

    if feature not in feature_names:
        print(f"Feature '{feature}' not found.")
        return

    # Subsample if needed
    if len(X) > max_samples:
        X_sample = X.sample(n=max_samples, random_state=42).values
    else:
        X_sample = X.values

    # Create explainer
    try:
        explainer = shap.TreeExplainer(model)
    except Exception:
        explainer = shap.Explainer(model, X_sample)

    shap_values = explainer.shap_values(X_sample)

    # Handle multi-class
    if isinstance(shap_values, list):
        if len(shap_values) == 2:
            shap_values = shap_values[1]
        else:
            shap_values = shap_values[0]

    plt.figure(figsize=(10, 6))

    if interaction_feature and interaction_feature in feature_names:
        shap.dependence_plot(
            feature, shap_values, X_sample,
            feature_names=feature_names,
            interaction_index=interaction_feature,
            show=False
        )
    else:
        shap.dependence_plot(
            feature, shap_values, X_sample,
            feature_names=feature_names,
            show=False
        )

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved to: {output_path}")

    plt.show()


def plot_shap_force(data: Dict[str, Any], sample_index: int = 0,
                    output_path: Optional[str] = None) -> None:
    """Generate SHAP force plot for a single prediction."""
    try:
        import shap
    except ImportError:
        print("SHAP not installed. Install with: pip install shap")
        return

    print(f"\n=== SHAP Force Plot (Sample {sample_index}) ===")

    model = data['model']
    X = data['X'].values
    feature_names = data['feature_names']

    # Create explainer
    try:
        explainer = shap.TreeExplainer(model)
    except Exception:
        explainer = shap.Explainer(model, X)

    shap_values = explainer.shap_values(X[sample_index:sample_index+1])

    # Handle multi-class
    if isinstance(shap_values, list):
        if len(shap_values) == 2:
            shap_values = shap_values[1]
        else:
            shap_values = shap_values[0]

    expected_value = explainer.expected_value
    if isinstance(expected_value, list):
        expected_value = expected_value[1] if len(expected_value) == 2 else expected_value[0]

    # For notebook display
    shap.initjs()

    force_plot = shap.force_plot(
        expected_value,
        shap_values[0],
        X[sample_index],
        feature_names=feature_names,
        matplotlib=True
    )

    if output_path:
        import matplotlib.pyplot as plt
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved to: {output_path}")


def plot_feature_importance_comparison(data: Dict[str, Any],
                                       output_path: Optional[str] = None) -> None:
    """Plot feature importance with confidence intervals (for ensembles)."""
    import matplotlib.pyplot as plt

    print("\n=== Feature Importance (Ensemble Comparison) ===")

    model = data['model']
    feature_names = data['feature_names']

    if not hasattr(model, 'estimators_'):
        print("Ensemble comparison only available for Random Forest.")
        return

    # Get importance from each tree
    importances = np.array([tree.feature_importances_ for tree in model.estimators_])

    mean_importance = importances.mean(axis=0)
    std_importance = importances.std(axis=0)

    # Sort by mean importance
    sorted_idx = np.argsort(mean_importance)[::-1][:20]  # Top 20

    fig, ax = plt.subplots(figsize=(10, 8))

    y_pos = np.arange(len(sorted_idx))

    ax.barh(y_pos, mean_importance[sorted_idx], xerr=std_importance[sorted_idx],
            align='center', alpha=0.8, capsize=3)

    ax.set_yticks(y_pos)
    ax.set_yticklabels([feature_names[i] for i in sorted_idx])
    ax.invert_yaxis()

    ax.set_xlabel('Importance')
    ax.set_title('Feature Importance with Std Dev (across trees)')

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved to: {output_path}")

    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description='Tree model visualization tools',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Visualize tree structure
  python visualize_trees.py model.joblib data.csv --target y --tree-plot

  # Partial dependence plots
  python visualize_trees.py model.joblib data.csv --target y --pdp income,age,education

  # ICE plot for a single feature
  python visualize_trees.py model.joblib data.csv --target y --ice income

  # 2D interaction PDP
  python visualize_trees.py model.joblib data.csv --target y --pdp-2d income,age

  # SHAP summary plot
  python visualize_trees.py model.joblib data.csv --target y --shap-summary

  # SHAP dependence plot
  python visualize_trees.py model.joblib data.csv --target y --shap-dep income --shap-interact age
        """
    )

    parser.add_argument('model', type=str, help='Path to saved model (.joblib)')
    parser.add_argument('data', type=str, help='Path to CSV data file')
    parser.add_argument('--target', '-y', type=str, required=True,
                        help='Target variable column name')
    parser.add_argument('--features', '-X', type=str, default=None,
                        help='Comma-separated feature column names')

    # Visualization options
    parser.add_argument('--tree-plot', action='store_true',
                        help='Plot tree structure')
    parser.add_argument('--tree-depth', type=int, default=4,
                        help='Max depth for tree visualization (default: 4)')
    parser.add_argument('--pdp', type=str, default=None,
                        help='Comma-separated features for PDP')
    parser.add_argument('--ice', type=str, default=None,
                        help='Feature for ICE plot')
    parser.add_argument('--pdp-2d', type=str, default=None,
                        help='Two comma-separated features for 2D PDP')
    parser.add_argument('--shap-summary', action='store_true',
                        help='Generate SHAP summary plot')
    parser.add_argument('--shap-dep', type=str, default=None,
                        help='Feature for SHAP dependence plot')
    parser.add_argument('--shap-interact', type=str, default=None,
                        help='Interaction feature for SHAP dependence')
    parser.add_argument('--shap-force', type=int, default=None,
                        help='Sample index for SHAP force plot')
    parser.add_argument('--importance-plot', action='store_true',
                        help='Feature importance comparison plot')

    # Options
    parser.add_argument('--max-samples', type=int, default=500,
                        help='Max samples for SHAP (default: 500)')
    parser.add_argument('--output', '-o', type=str, default=None,
                        help='Output directory for plots')
    parser.add_argument('--quiet', '-q', action='store_true',
                        help='Suppress verbose output')

    args = parser.parse_args()

    if args.quiet:
        warnings.filterwarnings('ignore')

    # Ensure at least one visualization is requested
    viz_requested = any([
        args.tree_plot, args.pdp, args.ice, args.pdp_2d,
        args.shap_summary, args.shap_dep, args.shap_force is not None,
        args.importance_plot
    ])

    if not viz_requested:
        parser.print_help()
        print("\nError: No visualization requested. Use --help to see options.")
        return

    # Setup output directory
    output_dir = Path(args.output) if args.output else None
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)

    # Load model and data
    data = load_model_and_data(args.model, args.data, args.target, args.features)

    # Generate visualizations
    if args.tree_plot:
        output_path = str(output_dir / 'tree_structure.png') if output_dir else None
        plot_tree_structure(data['model'], data['feature_names'],
                           max_depth=args.tree_depth, output_path=output_path)

    if args.pdp:
        features = [f.strip() for f in args.pdp.split(',')]
        output_path = str(output_dir / 'pdp.png') if output_dir else None
        plot_partial_dependence(data, features, output_path=output_path)

    if args.ice:
        output_path = str(output_dir / f'ice_{args.ice}.png') if output_dir else None
        plot_ice(data, args.ice, output_path=output_path)

    if args.pdp_2d:
        features = [f.strip() for f in args.pdp_2d.split(',')]
        output_path = str(output_dir / 'pdp_2d.png') if output_dir else None
        plot_2d_pdp(data, features, output_path=output_path)

    if args.shap_summary:
        output_path = str(output_dir / 'shap_summary.png') if output_dir else None
        plot_shap_summary(data, max_samples=args.max_samples, output_path=output_path)

    if args.shap_dep:
        output_path = str(output_dir / f'shap_dep_{args.shap_dep}.png') if output_dir else None
        plot_shap_dependence(data, args.shap_dep, args.shap_interact,
                            max_samples=args.max_samples, output_path=output_path)

    if args.shap_force is not None:
        output_path = str(output_dir / f'shap_force_{args.shap_force}.png') if output_dir else None
        plot_shap_force(data, sample_index=args.shap_force, output_path=output_path)

    if args.importance_plot:
        output_path = str(output_dir / 'importance_comparison.png') if output_dir else None
        plot_feature_importance_comparison(data, output_path=output_path)

    print("\nVisualization complete!")


if __name__ == '__main__':
    main()
