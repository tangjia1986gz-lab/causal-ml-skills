#!/usr/bin/env python3
"""
Time Series Visualization Script

Creates diagnostic and analytical plots including:
- Time series plots
- ACF/PACF plots
- Seasonal decomposition
- Rolling statistics
- Impulse response functions
- Forecast plots

Usage:
    python visualize_ts.py data.csv --variable y --acf-pacf
    python visualize_ts.py data.csv --all-vars --decompose
    python visualize_ts.py data.csv --variable y --rolling --window 12
"""

import argparse
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

sys.path.insert(0, str(Path(__file__).parent.parent))


def plot_time_series(
    series: pd.Series,
    title: str = None,
    figsize: tuple = (12, 4),
    save_path: str = None
):
    """Plot basic time series."""
    fig, ax = plt.subplots(figsize=figsize)

    ax.plot(series.index, series.values, 'b-', linewidth=1)
    ax.set_title(title or series.name or 'Time Series')
    ax.set_xlabel('Time')
    ax.set_ylabel('Value')
    ax.grid(True, alpha=0.3)

    # Add mean line
    ax.axhline(y=series.mean(), color='r', linestyle='--', alpha=0.5, label=f'Mean: {series.mean():.2f}')
    ax.legend()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")

    return fig


def plot_acf_pacf(
    series: pd.Series,
    lags: int = 40,
    title: str = None,
    figsize: tuple = (12, 5),
    save_path: str = None
):
    """Plot ACF and PACF."""
    from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # ACF
    plot_acf(series.dropna(), lags=lags, ax=axes[0], alpha=0.05)
    axes[0].set_title(f'ACF - {title or series.name or "Series"}')

    # PACF
    plot_pacf(series.dropna(), lags=lags, ax=axes[1], alpha=0.05, method='ywm')
    axes[1].set_title(f'PACF - {title or series.name or "Series"}')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"ACF/PACF plot saved to: {save_path}")

    return fig


def plot_rolling_statistics(
    series: pd.Series,
    window: int = 12,
    title: str = None,
    figsize: tuple = (12, 8),
    save_path: str = None
):
    """Plot rolling mean and standard deviation."""
    fig, axes = plt.subplots(3, 1, figsize=figsize)

    # Original series
    axes[0].plot(series.index, series.values, 'b-', linewidth=1, label='Original')
    axes[0].plot(series.index, series.rolling(window=window).mean().values, 'r-',
                 linewidth=2, label=f'Rolling Mean ({window})')
    axes[0].set_title(title or series.name or 'Time Series')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Rolling mean
    rolling_mean = series.rolling(window=window).mean()
    axes[1].plot(rolling_mean.index, rolling_mean.values, 'r-', linewidth=2)
    axes[1].set_title(f'Rolling Mean (window={window})')
    axes[1].grid(True, alpha=0.3)

    # Rolling std
    rolling_std = series.rolling(window=window).std()
    axes[2].plot(rolling_std.index, rolling_std.values, 'g-', linewidth=2)
    axes[2].set_title(f'Rolling Std Dev (window={window})')
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Rolling statistics plot saved to: {save_path}")

    return fig


def plot_seasonal_decomposition(
    series: pd.Series,
    period: int = None,
    model: str = 'additive',
    figsize: tuple = (12, 10),
    save_path: str = None
):
    """Plot seasonal decomposition."""
    from statsmodels.tsa.seasonal import seasonal_decompose

    # Infer period if not provided
    if period is None:
        if hasattr(series.index, 'freq'):
            if series.index.freq == 'M':
                period = 12
            elif series.index.freq == 'Q':
                period = 4
            elif series.index.freq == 'D':
                period = 7
            else:
                period = 12  # default
        else:
            period = 12

    # Decompose
    decomposition = seasonal_decompose(series.dropna(), model=model, period=period)

    fig, axes = plt.subplots(4, 1, figsize=figsize)

    # Original
    axes[0].plot(series.index, series.values, 'b-', linewidth=1)
    axes[0].set_title('Original')
    axes[0].grid(True, alpha=0.3)

    # Trend
    axes[1].plot(decomposition.trend.index, decomposition.trend.values, 'r-', linewidth=2)
    axes[1].set_title('Trend')
    axes[1].grid(True, alpha=0.3)

    # Seasonal
    axes[2].plot(decomposition.seasonal.index, decomposition.seasonal.values, 'g-', linewidth=1)
    axes[2].set_title(f'Seasonal (period={period})')
    axes[2].grid(True, alpha=0.3)

    # Residual
    axes[3].plot(decomposition.resid.index, decomposition.resid.values, 'k-', linewidth=1)
    axes[3].set_title('Residual')
    axes[3].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Decomposition plot saved to: {save_path}")

    return fig


def plot_differenced_series(
    series: pd.Series,
    max_d: int = 2,
    figsize: tuple = (12, 4 * 3),
    save_path: str = None
):
    """Plot original and differenced series."""
    fig, axes = plt.subplots(max_d + 1, 1, figsize=figsize)

    current = series.dropna()

    for d in range(max_d + 1):
        if d > 0:
            current = pd.Series(np.diff(current), index=current.index[1:])

        axes[d].plot(current.index, current.values, linewidth=1)
        title = 'Original' if d == 0 else f'd={d} (differenced {d}x)'
        axes[d].set_title(title)
        axes[d].grid(True, alpha=0.3)
        axes[d].axhline(y=current.mean(), color='r', linestyle='--', alpha=0.5)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Differenced series plot saved to: {save_path}")

    return fig


def plot_diagnostic_panel(
    series: pd.Series,
    lags: int = 30,
    window: int = None,
    figsize: tuple = (14, 10),
    save_path: str = None
):
    """Create comprehensive diagnostic panel."""
    from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
    from scipy import stats

    if window is None:
        window = min(12, len(series) // 4)

    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(3, 3, figure=fig)

    series = series.dropna()

    # Time series plot (full width)
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(series.index, series.values, 'b-', linewidth=1)
    ax1.plot(series.index, series.rolling(window=window).mean().values, 'r-', linewidth=2, alpha=0.7)
    ax1.set_title(f'{series.name or "Series"} with Rolling Mean')
    ax1.grid(True, alpha=0.3)

    # ACF
    ax2 = fig.add_subplot(gs[1, 0])
    plot_acf(series, lags=lags, ax=ax2, alpha=0.05)
    ax2.set_title('ACF')

    # PACF
    ax3 = fig.add_subplot(gs[1, 1])
    plot_pacf(series, lags=lags, ax=ax3, alpha=0.05, method='ywm')
    ax3.set_title('PACF')

    # Histogram
    ax4 = fig.add_subplot(gs[1, 2])
    ax4.hist(series.values, bins=30, density=True, alpha=0.7, color='blue')
    # Fit normal
    mu, sigma = series.mean(), series.std()
    x = np.linspace(series.min(), series.max(), 100)
    ax4.plot(x, stats.norm.pdf(x, mu, sigma), 'r-', linewidth=2, label='Normal')
    ax4.set_title('Distribution')
    ax4.legend()

    # Q-Q plot
    ax5 = fig.add_subplot(gs[2, 0])
    stats.probplot(series.values, dist="norm", plot=ax5)
    ax5.set_title('Q-Q Plot')

    # Box plot by period (if datetime index)
    ax6 = fig.add_subplot(gs[2, 1])
    if hasattr(series.index, 'month'):
        monthly_data = [series[series.index.month == m].values for m in range(1, 13)]
        ax6.boxplot(monthly_data, labels=range(1, 13))
        ax6.set_title('Monthly Box Plot')
        ax6.set_xlabel('Month')
    else:
        # Simple box plot
        ax6.boxplot(series.values)
        ax6.set_title('Box Plot')

    # Rolling std
    ax7 = fig.add_subplot(gs[2, 2])
    rolling_std = series.rolling(window=window).std()
    ax7.plot(rolling_std.index, rolling_std.values, 'g-', linewidth=1)
    ax7.set_title(f'Rolling Std (window={window})')
    ax7.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Diagnostic panel saved to: {save_path}")

    return fig


def plot_multiple_series(
    df: pd.DataFrame,
    variables: List[str] = None,
    figsize: tuple = None,
    save_path: str = None
):
    """Plot multiple time series for comparison."""
    if variables is None:
        variables = df.columns.tolist()

    n_vars = len(variables)
    if figsize is None:
        figsize = (12, 3 * n_vars)

    fig, axes = plt.subplots(n_vars, 1, figsize=figsize, sharex=True)

    if n_vars == 1:
        axes = [axes]

    for i, var in enumerate(variables):
        if var in df.columns:
            axes[i].plot(df.index, df[var].values, linewidth=1)
            axes[i].set_title(var)
            axes[i].grid(True, alpha=0.3)

    plt.xlabel('Time')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Multiple series plot saved to: {save_path}")

    return fig


def plot_cross_correlation(
    series1: pd.Series,
    series2: pd.Series,
    lags: int = 30,
    title: str = None,
    figsize: tuple = (10, 4),
    save_path: str = None
):
    """Plot cross-correlation between two series."""
    from statsmodels.tsa.stattools import ccf

    # Compute CCF
    cc = ccf(series1.dropna(), series2.dropna(), adjusted=False)

    # Plot
    fig, ax = plt.subplots(figsize=figsize)

    lag_range = range(-lags, lags + 1)
    cc_values = cc[:2*lags + 1] if len(cc) >= 2*lags + 1 else cc

    ax.stem(range(len(cc_values)), cc_values, linefmt='b-', markerfmt='bo', basefmt='r-')
    ax.axhline(y=0, color='k', linestyle='-')

    # Confidence bands
    n = len(series1)
    conf = 1.96 / np.sqrt(n)
    ax.axhline(y=conf, color='r', linestyle='--', alpha=0.5)
    ax.axhline(y=-conf, color='r', linestyle='--', alpha=0.5)

    title = title or f'Cross-Correlation: {series1.name} vs {series2.name}'
    ax.set_title(title)
    ax.set_xlabel('Lag')
    ax.set_ylabel('Correlation')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Cross-correlation plot saved to: {save_path}")

    return fig


def main():
    parser = argparse.ArgumentParser(
        description='Time Series Visualization',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument('data', help='Path to CSV file')
    parser.add_argument('--variable', '-v', help='Variable to plot')
    parser.add_argument('--all-vars', action='store_true', help='Plot all numeric variables')
    parser.add_argument('--date-col', help='Date column name')

    # Plot types
    parser.add_argument('--acf-pacf', action='store_true', help='Plot ACF and PACF')
    parser.add_argument('--rolling', action='store_true', help='Plot rolling statistics')
    parser.add_argument('--decompose', action='store_true', help='Seasonal decomposition')
    parser.add_argument('--differenced', action='store_true', help='Plot differenced series')
    parser.add_argument('--diagnostic', action='store_true', help='Full diagnostic panel')
    parser.add_argument('--xcorr', help='Second variable for cross-correlation')

    # Parameters
    parser.add_argument('--lags', type=int, default=30, help='Number of lags for ACF/PACF')
    parser.add_argument('--window', type=int, default=12, help='Rolling window size')
    parser.add_argument('--period', type=int, help='Seasonal period for decomposition')

    parser.add_argument('--output-dir', '-o', default='.', help='Output directory for plots')
    parser.add_argument('--show', action='store_true', help='Show plots interactively')

    args = parser.parse_args()

    # Load data
    df = pd.read_csv(args.data)
    if args.date_col and args.date_col in df.columns:
        df[args.date_col] = pd.to_datetime(df[args.date_col])
        df.set_index(args.date_col, inplace=True)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    # Determine variables
    if args.variable:
        variables = [args.variable]
    elif args.all_vars:
        variables = df.select_dtypes(include=[np.number]).columns.tolist()
    else:
        variables = df.select_dtypes(include=[np.number]).columns.tolist()[:1]

    # Generate plots
    for var in variables:
        if var not in df.columns:
            print(f"Warning: Variable '{var}' not found")
            continue

        series = df[var]
        series.name = var

        print(f"\nGenerating plots for: {var}")

        # Basic time series
        plot_time_series(series, save_path=str(output_dir / f'{var}_timeseries.png'))

        if args.acf_pacf:
            plot_acf_pacf(series, lags=args.lags,
                         save_path=str(output_dir / f'{var}_acf_pacf.png'))

        if args.rolling:
            plot_rolling_statistics(series, window=args.window,
                                   save_path=str(output_dir / f'{var}_rolling.png'))

        if args.decompose:
            try:
                plot_seasonal_decomposition(series, period=args.period,
                                           save_path=str(output_dir / f'{var}_decomposition.png'))
            except Exception as e:
                print(f"Could not decompose {var}: {e}")

        if args.differenced:
            plot_differenced_series(series,
                                   save_path=str(output_dir / f'{var}_differenced.png'))

        if args.diagnostic:
            plot_diagnostic_panel(series, lags=args.lags, window=args.window,
                                 save_path=str(output_dir / f'{var}_diagnostic.png'))

        if args.xcorr and args.xcorr in df.columns:
            series2 = df[args.xcorr]
            series2.name = args.xcorr
            plot_cross_correlation(series, series2, lags=args.lags,
                                  save_path=str(output_dir / f'{var}_{args.xcorr}_xcorr.png'))

    # Multi-series comparison if multiple variables
    if len(variables) > 1:
        plot_multiple_series(df, variables,
                            save_path=str(output_dir / 'all_series.png'))

    if args.show:
        plt.show()

    print(f"\nAll plots saved to: {output_dir}")


if __name__ == '__main__':
    main()
