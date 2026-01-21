#!/usr/bin/env python3
"""
First Stage and Reduced Form Visualization.

This script creates publication-ready visualizations for IV analysis:
- First-stage scatter plots
- Reduced form visualization
- Instrument strength diagnostics
- Binned scatter plots (a la Angrist-Pischke)

Usage:
    python visualize_first_stage.py --data data.csv --outcome y --treatment d \
        --instruments z1 --controls x1 x2 --output figures/

Author: Causal ML Skills
Version: 1.0.0
"""

import argparse
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from iv_estimator import first_stage_test, estimate_2sls, estimate_ols


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Visualize first stage and reduced form for IV analysis"
    )

    parser.add_argument("--data", "-d", type=str, required=True)
    parser.add_argument("--outcome", "-y", type=str, required=True)
    parser.add_argument("--treatment", "-t", type=str, required=True)
    parser.add_argument("--instruments", "-z", type=str, nargs="+", required=True)
    parser.add_argument("--controls", "-x", type=str, nargs="*", default=None)
    parser.add_argument("--output", "-o", type=str, default="figures")
    parser.add_argument("--n-bins", type=int, default=20,
                        help="Number of bins for binned scatter plots")
    parser.add_argument("--format", type=str, choices=["png", "pdf", "svg"],
                        default="png", help="Output format")
    parser.add_argument("--dpi", type=int, default=150)
    parser.add_argument("--style", type=str, default="seaborn-v0_8-whitegrid")

    return parser.parse_args()


def setup_plotting():
    """Import and configure plotting libraries."""
    try:
        import matplotlib.pyplot as plt
        import matplotlib as mpl

        # Try to use a nice style
        try:
            plt.style.use('seaborn-v0_8-whitegrid')
        except:
            try:
                plt.style.use('seaborn-whitegrid')
            except:
                pass

        # Set default figure parameters
        mpl.rcParams['figure.figsize'] = (8, 6)
        mpl.rcParams['font.size'] = 11
        mpl.rcParams['axes.titlesize'] = 12
        mpl.rcParams['axes.labelsize'] = 11
        mpl.rcParams['xtick.labelsize'] = 10
        mpl.rcParams['ytick.labelsize'] = 10
        mpl.rcParams['legend.fontsize'] = 10

        return plt
    except ImportError:
        raise ImportError("matplotlib required. Install with: pip install matplotlib")


def residualize(
    data: pd.DataFrame,
    var: str,
    controls: Optional[List[str]]
) -> np.ndarray:
    """
    Residualize a variable with respect to controls.

    Parameters
    ----------
    data : pd.DataFrame
        Dataset
    var : str
        Variable to residualize
    controls : List[str], optional
        Control variables

    Returns
    -------
    np.ndarray
        Residualized values
    """
    if not controls:
        return data[var].values

    import statsmodels.api as sm
    from statsmodels.regression.linear_model import OLS

    X = sm.add_constant(data[controls])
    y = data[var]

    mask = ~(X.isna().any(axis=1) | y.isna())

    model = OLS(y[mask], X[mask]).fit()
    residuals = np.full(len(data), np.nan)
    residuals[mask] = model.resid.values

    return residuals


def create_binned_means(
    x: np.ndarray,
    y: np.ndarray,
    n_bins: int = 20
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Create binned means for scatter plot.

    Parameters
    ----------
    x : np.ndarray
        X variable
    y : np.ndarray
        Y variable
    n_bins : int
        Number of bins

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray]
        Bin centers, bin means, bin standard errors
    """
    # Remove NaN
    mask = ~(np.isnan(x) | np.isnan(y))
    x_clean = x[mask]
    y_clean = y[mask]

    # Create bins
    percentiles = np.linspace(0, 100, n_bins + 1)
    bin_edges = np.percentile(x_clean, percentiles)

    # Compute means for each bin
    bin_centers = []
    bin_means = []
    bin_ses = []

    for i in range(n_bins):
        in_bin = (x_clean >= bin_edges[i]) & (x_clean < bin_edges[i + 1])
        if i == n_bins - 1:  # Include right edge for last bin
            in_bin = (x_clean >= bin_edges[i]) & (x_clean <= bin_edges[i + 1])

        if in_bin.sum() > 0:
            bin_centers.append(x_clean[in_bin].mean())
            bin_means.append(y_clean[in_bin].mean())
            bin_ses.append(y_clean[in_bin].std() / np.sqrt(in_bin.sum()))

    return np.array(bin_centers), np.array(bin_means), np.array(bin_ses)


def plot_first_stage(
    data: pd.DataFrame,
    treatment: str,
    instrument: str,
    controls: Optional[List[str]],
    output_path: Path,
    n_bins: int = 20,
    fmt: str = "png",
    dpi: int = 150
) -> str:
    """
    Create first-stage visualization.

    Parameters
    ----------
    data : pd.DataFrame
        Dataset
    treatment : str
        Treatment variable
    instrument : str
        Single instrument variable
    controls : List[str], optional
        Control variables
    output_path : Path
        Output directory
    n_bins : int
        Number of bins
    fmt : str
        Output format
    dpi : int
        DPI for output

    Returns
    -------
    str
        Path to saved figure
    """
    plt = setup_plotting()

    # Residualize if controls present
    if controls:
        d_resid = residualize(data, treatment, controls)
        z_resid = residualize(data, instrument, controls)
        x_label = f"{instrument} (residualized)"
        y_label = f"{treatment} (residualized)"
    else:
        d_resid = data[treatment].values
        z_resid = data[instrument].values
        x_label = instrument
        y_label = treatment

    # Create binned scatter
    bin_centers, bin_means, bin_ses = create_binned_means(z_resid, d_resid, n_bins)

    # Get first-stage statistics
    fs = first_stage_test(data, treatment, [instrument], controls)

    # Create figure
    fig, ax = plt.subplots(figsize=(8, 6))

    # Raw scatter (faint)
    ax.scatter(z_resid, d_resid, alpha=0.1, s=10, c='gray', label='_nolegend_')

    # Binned means with error bars
    ax.errorbar(bin_centers, bin_means, yerr=1.96*bin_ses,
                fmt='o', color='#2E86AB', markersize=8, capsize=3,
                label='Binned means (+/- 95% CI)')

    # Fit line
    slope = fs['coefficients'][instrument]
    intercept = d_resid[~np.isnan(d_resid)].mean() - slope * z_resid[~np.isnan(z_resid)].mean()
    x_line = np.linspace(np.nanmin(z_resid), np.nanmax(z_resid), 100)
    y_line = intercept + slope * x_line
    ax.plot(x_line, y_line, 'r-', linewidth=2, label=f'Slope: {slope:.4f}')

    # Labels and title
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(f'First Stage: {treatment} on {instrument}\n'
                 f'F-statistic: {fs["f_statistic"]:.2f}')
    ax.legend()

    # Add annotation
    annotation = f'N = {len(data):,}\nF = {fs["f_statistic"]:.2f}\n$R^2$ = {fs["partial_r2"]:.4f}'
    ax.annotate(annotation, xy=(0.05, 0.95), xycoords='axes fraction',
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()

    # Save
    filepath = output_path / f"first_stage_{instrument}.{fmt}"
    plt.savefig(filepath, dpi=dpi, bbox_inches='tight')
    plt.close()

    return str(filepath)


def plot_reduced_form(
    data: pd.DataFrame,
    outcome: str,
    instrument: str,
    controls: Optional[List[str]],
    output_path: Path,
    n_bins: int = 20,
    fmt: str = "png",
    dpi: int = 150
) -> str:
    """
    Create reduced form visualization.

    Parameters
    ----------
    data : pd.DataFrame
        Dataset
    outcome : str
        Outcome variable
    instrument : str
        Single instrument variable
    controls : List[str], optional
        Control variables
    output_path : Path
        Output directory
    n_bins : int
        Number of bins
    fmt : str
        Output format
    dpi : int
        DPI for output

    Returns
    -------
    str
        Path to saved figure
    """
    plt = setup_plotting()
    import statsmodels.api as sm
    from statsmodels.regression.linear_model import OLS

    # Residualize if controls present
    if controls:
        y_resid = residualize(data, outcome, controls)
        z_resid = residualize(data, instrument, controls)
        x_label = f"{instrument} (residualized)"
        y_label = f"{outcome} (residualized)"
    else:
        y_resid = data[outcome].values
        z_resid = data[instrument].values
        x_label = instrument
        y_label = outcome

    # Reduced form regression
    Z = sm.add_constant(data[[instrument] + (controls or [])])
    mask = ~(Z.isna().any(axis=1) | data[outcome].isna())
    rf_model = OLS(data[outcome][mask], Z[mask]).fit()
    rf_coef = rf_model.params[instrument]
    rf_se = rf_model.bse[instrument]
    rf_pval = rf_model.pvalues[instrument]

    # Create binned scatter
    bin_centers, bin_means, bin_ses = create_binned_means(z_resid, y_resid, n_bins)

    # Create figure
    fig, ax = plt.subplots(figsize=(8, 6))

    # Raw scatter (faint)
    ax.scatter(z_resid, y_resid, alpha=0.1, s=10, c='gray', label='_nolegend_')

    # Binned means with error bars
    ax.errorbar(bin_centers, bin_means, yerr=1.96*bin_ses,
                fmt='o', color='#28A745', markersize=8, capsize=3,
                label='Binned means (+/- 95% CI)')

    # Fit line
    intercept = y_resid[~np.isnan(y_resid)].mean() - rf_coef * z_resid[~np.isnan(z_resid)].mean()
    x_line = np.linspace(np.nanmin(z_resid), np.nanmax(z_resid), 100)
    y_line = intercept + rf_coef * x_line
    ax.plot(x_line, y_line, 'r-', linewidth=2, label=f'Slope: {rf_coef:.4f}')

    # Labels and title
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    stars = "***" if rf_pval < 0.01 else "**" if rf_pval < 0.05 else "*" if rf_pval < 0.1 else ""
    ax.set_title(f'Reduced Form: {outcome} on {instrument}\n'
                 f'Coefficient: {rf_coef:.4f}{stars} (SE: {rf_se:.4f})')
    ax.legend()

    # Add annotation
    annotation = f'N = {mask.sum():,}\nCoef = {rf_coef:.4f}\nSE = {rf_se:.4f}'
    ax.annotate(annotation, xy=(0.05, 0.95), xycoords='axes fraction',
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()

    # Save
    filepath = output_path / f"reduced_form_{instrument}.{fmt}"
    plt.savefig(filepath, dpi=dpi, bbox_inches='tight')
    plt.close()

    return str(filepath)


def plot_iv_vs_ols(
    data: pd.DataFrame,
    outcome: str,
    treatment: str,
    instruments: List[str],
    controls: Optional[List[str]],
    output_path: Path,
    n_bins: int = 20,
    fmt: str = "png",
    dpi: int = 150
) -> str:
    """
    Create comparison of OLS and IV relationships.

    Parameters
    ----------
    data : pd.DataFrame
        Dataset
    outcome : str
        Outcome variable
    treatment : str
        Treatment variable
    instruments : List[str]
        Instrument variables
    controls : List[str], optional
        Control variables
    output_path : Path
        Output directory
    n_bins : int
        Number of bins
    fmt : str
        Output format
    dpi : int
        DPI for output

    Returns
    -------
    str
        Path to saved figure
    """
    plt = setup_plotting()

    # Get estimates
    ols_result = estimate_ols(data, outcome, treatment, controls)
    iv_result = estimate_2sls(data, outcome, treatment, instruments, controls)

    # Residualize if controls
    if controls:
        y_resid = residualize(data, outcome, controls)
        d_resid = residualize(data, treatment, controls)
        x_label = f"{treatment} (residualized)"
        y_label = f"{outcome} (residualized)"
    else:
        y_resid = data[outcome].values
        d_resid = data[treatment].values
        x_label = treatment
        y_label = outcome

    # Create binned scatter
    bin_centers, bin_means, bin_ses = create_binned_means(d_resid, y_resid, n_bins)

    # Create figure
    fig, ax = plt.subplots(figsize=(8, 6))

    # Raw scatter (faint)
    ax.scatter(d_resid, y_resid, alpha=0.1, s=10, c='gray', label='_nolegend_')

    # Binned means
    ax.errorbar(bin_centers, bin_means, yerr=1.96*bin_ses,
                fmt='o', color='#6C757D', markersize=8, capsize=3,
                label='Binned means')

    # OLS line
    y_mean = np.nanmean(y_resid)
    d_mean = np.nanmean(d_resid)
    x_line = np.linspace(np.nanmin(d_resid), np.nanmax(d_resid), 100)

    ols_intercept = y_mean - ols_result.effect * d_mean
    ols_line = ols_intercept + ols_result.effect * x_line
    ax.plot(x_line, ols_line, 'b--', linewidth=2,
            label=f'OLS: {ols_result.effect:.4f}')

    # IV line
    iv_intercept = y_mean - iv_result.effect * d_mean
    iv_line = iv_intercept + iv_result.effect * x_line
    ax.plot(x_line, iv_line, 'r-', linewidth=2,
            label=f'IV: {iv_result.effect:.4f}')

    # Labels
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(f'{outcome} vs {treatment}: OLS vs IV Comparison')
    ax.legend()

    # Annotation
    annotation = (f'OLS: {ols_result.effect:.4f} (SE: {ols_result.se:.4f})\n'
                  f'IV:  {iv_result.effect:.4f} (SE: {iv_result.se:.4f})\n'
                  f'Diff: {iv_result.effect - ols_result.effect:.4f}')
    ax.annotate(annotation, xy=(0.05, 0.95), xycoords='axes fraction',
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()

    # Save
    filepath = output_path / f"iv_vs_ols.{fmt}"
    plt.savefig(filepath, dpi=dpi, bbox_inches='tight')
    plt.close()

    return str(filepath)


def plot_instrument_strength(
    data: pd.DataFrame,
    treatment: str,
    instruments: List[str],
    controls: Optional[List[str]],
    output_path: Path,
    fmt: str = "png",
    dpi: int = 150
) -> str:
    """
    Create instrument strength visualization.

    Parameters
    ----------
    data : pd.DataFrame
        Dataset
    treatment : str
        Treatment variable
    instruments : List[str]
        Instrument variables
    controls : List[str], optional
        Control variables
    output_path : Path
        Output directory
    fmt : str
        Output format
    dpi : int
        DPI for output

    Returns
    -------
    str
        Path to saved figure
    """
    plt = setup_plotting()

    # Get first-stage results
    fs = first_stage_test(data, treatment, instruments, controls)

    # Create figure with two panels
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Panel 1: Coefficient plot
    ax1 = axes[0]
    y_pos = np.arange(len(instruments))
    coefs = [fs['coefficients'][z] for z in instruments]
    ses = [fs['std_errors'][z] for z in instruments]

    ax1.barh(y_pos, coefs, xerr=[1.96*se for se in ses],
             color='#2E86AB', capsize=4)
    ax1.axvline(x=0, color='gray', linestyle='--', linewidth=1)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(instruments)
    ax1.set_xlabel('First-Stage Coefficient')
    ax1.set_title('Instrument Coefficients (+/- 95% CI)')

    # Panel 2: F-statistic vs critical values
    ax2 = axes[1]

    # Individual F-statistics
    individual_fs = []
    for z in instruments:
        single_fs = first_stage_test(data, treatment, [z], controls)
        individual_fs.append(single_fs['f_statistic'])

    x_pos = np.arange(len(instruments) + 1)
    f_values = individual_fs + [fs['f_statistic']]
    labels = instruments + ['Joint']
    colors = ['#2E86AB'] * len(instruments) + ['#28A745']

    bars = ax2.bar(x_pos, f_values, color=colors)

    # Add critical value line
    ax2.axhline(y=10, color='red', linestyle='--', linewidth=2,
                label='Rule of thumb (F=10)')

    # Stock-Yogo if available
    from iv_estimator import STOCK_YOGO_CRITICAL_VALUES
    n_instr = len(instruments)
    if n_instr in STOCK_YOGO_CRITICAL_VALUES:
        cv_10 = STOCK_YOGO_CRITICAL_VALUES[n_instr].get(10)
        if cv_10:
            ax2.axhline(y=cv_10, color='orange', linestyle='--', linewidth=2,
                        label=f'Stock-Yogo 10% ({cv_10:.1f})')

    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(labels, rotation=45, ha='right')
    ax2.set_ylabel('F-statistic')
    ax2.set_title('Instrument Strength')
    ax2.legend()

    # Add value labels on bars
    for bar, val in zip(bars, f_values):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                 f'{val:.1f}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()

    # Save
    filepath = output_path / f"instrument_strength.{fmt}"
    plt.savefig(filepath, dpi=dpi, bbox_inches='tight')
    plt.close()

    return str(filepath)


def create_all_visualizations(
    data: pd.DataFrame,
    outcome: str,
    treatment: str,
    instruments: List[str],
    controls: Optional[List[str]],
    output_path: Path,
    n_bins: int = 20,
    fmt: str = "png",
    dpi: int = 150
) -> List[str]:
    """
    Create all IV visualizations.

    Returns
    -------
    List[str]
        Paths to all saved figures
    """
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    saved_files = []

    print("Creating IV visualizations...")

    # First stage plots for each instrument
    for z in instruments:
        print(f"  - First stage: {z}")
        filepath = plot_first_stage(
            data, treatment, z, controls, output_path, n_bins, fmt, dpi
        )
        saved_files.append(filepath)

    # Reduced form plots for each instrument
    for z in instruments:
        print(f"  - Reduced form: {z}")
        filepath = plot_reduced_form(
            data, outcome, z, controls, output_path, n_bins, fmt, dpi
        )
        saved_files.append(filepath)

    # IV vs OLS comparison
    print("  - IV vs OLS comparison")
    filepath = plot_iv_vs_ols(
        data, outcome, treatment, instruments, controls, output_path, n_bins, fmt, dpi
    )
    saved_files.append(filepath)

    # Instrument strength
    print("  - Instrument strength")
    filepath = plot_instrument_strength(
        data, treatment, instruments, controls, output_path, fmt, dpi
    )
    saved_files.append(filepath)

    print(f"\nSaved {len(saved_files)} figures to {output_path}")

    return saved_files


def main():
    """Main entry point."""
    args = parse_args()

    # Load data
    df = pd.read_csv(args.data)

    # Create visualizations
    create_all_visualizations(
        data=df,
        outcome=args.outcome,
        treatment=args.treatment,
        instruments=args.instruments,
        controls=args.controls,
        output_path=Path(args.output),
        n_bins=args.n_bins,
        fmt=args.format,
        dpi=args.dpi
    )


if __name__ == "__main__":
    main()
