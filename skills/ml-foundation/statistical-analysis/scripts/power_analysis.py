#!/usr/bin/env python3
"""
Power Analysis CLI

Calculate statistical power, required sample sizes, and minimum detectable effects
for various study designs.

Usage:
    python power_analysis.py --effect-size 0.5 --alpha 0.05 --power 0.80 --test ttest --mode sample_size
    python power_analysis.py --n1 50 --n2 50 --alpha 0.05 --power 0.80 --test ttest --mode effect_size
    python power_analysis.py --effect-size 0.5 --n1 50 --n2 50 --alpha 0.05 --test ttest --mode power
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

import numpy as np
from scipy import stats

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


def power_ttest_two_sample(
    effect_size: float = None,
    n1: int = None,
    n2: int = None,
    alpha: float = 0.05,
    power: float = None,
    alternative: str = 'two-sided',
    mode: str = 'power'
) -> float:
    """
    Power analysis for two-sample t-test.

    Parameters
    ----------
    effect_size : float
        Cohen's d
    n1, n2 : int
        Sample sizes per group
    alpha : float
        Significance level
    power : float
        Statistical power (1 - beta)
    alternative : str
        'two-sided', 'greater', or 'less'
    mode : str
        'power', 'sample_size', or 'effect_size'

    Returns
    -------
    float
        The calculated value (power, sample size, or effect size)
    """
    if alternative == 'two-sided':
        alpha_adj = alpha / 2
    else:
        alpha_adj = alpha

    z_alpha = stats.norm.ppf(1 - alpha_adj)

    if mode == 'power':
        if effect_size is None or n1 is None or n2 is None:
            raise ValueError("Need effect_size, n1, n2 for power calculation")

        # Non-centrality parameter
        se = np.sqrt(1/n1 + 1/n2)
        ncp = effect_size / se

        # Calculate power
        if alternative == 'two-sided':
            power = 1 - stats.norm.cdf(z_alpha - ncp) + stats.norm.cdf(-z_alpha - ncp)
        else:
            power = 1 - stats.norm.cdf(z_alpha - ncp)

        return power

    elif mode == 'sample_size':
        if effect_size is None or power is None:
            raise ValueError("Need effect_size and power for sample size calculation")

        z_beta = stats.norm.ppf(power)

        if n1 is not None and n2 is None:
            # Calculate n2 given n1
            n2 = (z_alpha + z_beta)**2 / effect_size**2 - n1
            return max(2, int(np.ceil(n2)))
        else:
            # Equal sample sizes
            n = 2 * (z_alpha + z_beta)**2 / effect_size**2
            return max(2, int(np.ceil(n)))

    elif mode == 'effect_size':
        if n1 is None or n2 is None or power is None:
            raise ValueError("Need n1, n2, power for effect size calculation")

        z_beta = stats.norm.ppf(power)
        se = np.sqrt(1/n1 + 1/n2)
        effect_size = (z_alpha + z_beta) * se

        return effect_size

    else:
        raise ValueError(f"Unknown mode: {mode}")


def power_ttest_one_sample(
    effect_size: float = None,
    n: int = None,
    alpha: float = 0.05,
    power: float = None,
    alternative: str = 'two-sided',
    mode: str = 'power'
) -> float:
    """Power analysis for one-sample t-test."""

    if alternative == 'two-sided':
        alpha_adj = alpha / 2
    else:
        alpha_adj = alpha

    z_alpha = stats.norm.ppf(1 - alpha_adj)

    if mode == 'power':
        ncp = effect_size * np.sqrt(n)
        if alternative == 'two-sided':
            power = 1 - stats.norm.cdf(z_alpha - ncp) + stats.norm.cdf(-z_alpha - ncp)
        else:
            power = 1 - stats.norm.cdf(z_alpha - ncp)
        return power

    elif mode == 'sample_size':
        z_beta = stats.norm.ppf(power)
        n = (z_alpha + z_beta)**2 / effect_size**2
        return max(2, int(np.ceil(n)))

    elif mode == 'effect_size':
        z_beta = stats.norm.ppf(power)
        effect_size = (z_alpha + z_beta) / np.sqrt(n)
        return effect_size


def power_ttest_paired(
    effect_size: float = None,
    n: int = None,
    alpha: float = 0.05,
    power: float = None,
    correlation: float = 0.5,
    alternative: str = 'two-sided',
    mode: str = 'power'
) -> float:
    """Power analysis for paired t-test."""

    # Adjust effect size for correlation
    d_adj = effect_size / np.sqrt(2 * (1 - correlation))

    return power_ttest_one_sample(
        effect_size=d_adj,
        n=n,
        alpha=alpha,
        power=power,
        alternative=alternative,
        mode=mode
    )


def power_anova(
    effect_size: float = None,
    k: int = None,
    n_per_group: int = None,
    alpha: float = 0.05,
    power: float = None,
    mode: str = 'power'
) -> float:
    """
    Power analysis for one-way ANOVA.

    Parameters
    ----------
    effect_size : float
        Cohen's f
    k : int
        Number of groups
    n_per_group : int
        Sample size per group
    """

    if mode == 'power':
        # Non-centrality parameter
        ncp = effect_size**2 * k * n_per_group
        df1 = k - 1
        df2 = k * (n_per_group - 1)

        # Critical F value
        f_crit = stats.f.ppf(1 - alpha, df1, df2)

        # Power using non-central F
        power = 1 - stats.ncf.cdf(f_crit, df1, df2, ncp)
        return power

    elif mode == 'sample_size':
        # Iterative search
        for n in range(2, 10000):
            p = power_anova(effect_size, k, n, alpha, mode='power')
            if p >= power:
                return n
        return 10000

    elif mode == 'effect_size':
        # Iterative search
        for f in np.arange(0.01, 2.0, 0.01):
            p = power_anova(f, k, n_per_group, alpha, mode='power')
            if p >= power:
                return f
        return 2.0


def power_chi_squared(
    effect_size: float = None,
    df: int = None,
    n: int = None,
    alpha: float = 0.05,
    power: float = None,
    mode: str = 'power'
) -> float:
    """
    Power analysis for chi-squared test.

    Parameters
    ----------
    effect_size : float
        Cohen's w
    df : int
        Degrees of freedom
    """

    if mode == 'power':
        # Non-centrality parameter
        ncp = effect_size**2 * n

        # Critical chi-squared value
        chi_crit = stats.chi2.ppf(1 - alpha, df)

        # Power using non-central chi-squared
        power = 1 - stats.ncx2.cdf(chi_crit, df, ncp)
        return power

    elif mode == 'sample_size':
        for n_try in range(10, 100000, 10):
            p = power_chi_squared(effect_size, df, n_try, alpha, mode='power')
            if p >= power:
                return n_try
        return 100000

    elif mode == 'effect_size':
        for w in np.arange(0.01, 2.0, 0.01):
            p = power_chi_squared(w, df, n, alpha, mode='power')
            if p >= power:
                return w
        return 2.0


def power_correlation(
    r: float = None,
    n: int = None,
    alpha: float = 0.05,
    power: float = None,
    alternative: str = 'two-sided',
    mode: str = 'power'
) -> float:
    """Power analysis for correlation test."""

    if alternative == 'two-sided':
        alpha_adj = alpha / 2
    else:
        alpha_adj = alpha

    z_alpha = stats.norm.ppf(1 - alpha_adj)

    if mode == 'power':
        # Fisher's z transformation
        z_r = np.arctanh(r)
        se = 1 / np.sqrt(n - 3)
        ncp = z_r / se

        if alternative == 'two-sided':
            power = 1 - stats.norm.cdf(z_alpha - ncp) + stats.norm.cdf(-z_alpha - ncp)
        else:
            power = 1 - stats.norm.cdf(z_alpha - ncp)
        return power

    elif mode == 'sample_size':
        z_r = np.arctanh(r)
        z_beta = stats.norm.ppf(power)
        n = (z_alpha + z_beta)**2 / z_r**2 + 3
        return max(4, int(np.ceil(n)))

    elif mode == 'effect_size':
        z_beta = stats.norm.ppf(power)
        se = 1 / np.sqrt(n - 3)
        z_r = (z_alpha + z_beta) * se
        r = np.tanh(z_r)
        return r


def power_regression(
    f_squared: float = None,
    n_predictors: int = None,
    n: int = None,
    alpha: float = 0.05,
    power: float = None,
    mode: str = 'power'
) -> float:
    """
    Power analysis for multiple regression (testing R-squared > 0).

    Parameters
    ----------
    f_squared : float
        Cohen's f-squared = R^2 / (1 - R^2)
    n_predictors : int
        Number of predictors in the model
    """

    if mode == 'power':
        # Non-centrality parameter
        ncp = f_squared * n
        df1 = n_predictors
        df2 = n - n_predictors - 1

        if df2 <= 0:
            return 0.0

        # Critical F
        f_crit = stats.f.ppf(1 - alpha, df1, df2)

        # Power
        power = 1 - stats.ncf.cdf(f_crit, df1, df2, ncp)
        return power

    elif mode == 'sample_size':
        for n_try in range(n_predictors + 10, 10000):
            p = power_regression(f_squared, n_predictors, n_try, alpha, mode='power')
            if p >= power:
                return n_try
        return 10000

    elif mode == 'effect_size':
        for f2 in np.arange(0.001, 1.0, 0.001):
            p = power_regression(f2, n_predictors, n, alpha, mode='power')
            if p >= power:
                return f2
        return 1.0


def create_sample_size_table(
    test: str,
    effect_sizes: list,
    alpha: float = 0.05,
    powers: list = [0.80, 0.90, 0.95],
    **kwargs
) -> str:
    """Create a sample size table for different effect sizes and power levels."""

    lines = []
    lines.append(f"Sample Size Table for {test} (alpha = {alpha})")
    lines.append("=" * 60)

    # Header
    header = "Effect Size |"
    for power in powers:
        header += f" Power={power:.2f} |"
    lines.append(header)
    lines.append("-" * 60)

    # Rows
    for es in effect_sizes:
        row = f"    {es:.2f}    |"
        for power in powers:
            if test == 'ttest':
                n = power_ttest_two_sample(effect_size=es, alpha=alpha, power=power, mode='sample_size')
            elif test == 'correlation':
                n = power_correlation(r=es, alpha=alpha, power=power, mode='sample_size')
            elif test == 'anova':
                n = power_anova(effect_size=es, k=kwargs.get('k', 3), alpha=alpha, power=power, mode='sample_size')
            else:
                n = "N/A"
            row += f"    {n:>5}    |"
        lines.append(row)

    lines.append("=" * 60)
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description='Power analysis for statistical tests',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Calculate required sample size
  python power_analysis.py --effect-size 0.5 --alpha 0.05 --power 0.80 --test ttest --mode sample_size

  # Calculate power
  python power_analysis.py --effect-size 0.5 --n1 50 --n2 50 --alpha 0.05 --test ttest --mode power

  # Calculate minimum detectable effect
  python power_analysis.py --n1 50 --n2 50 --alpha 0.05 --power 0.80 --test ttest --mode effect_size

  # ANOVA
  python power_analysis.py --effect-size 0.25 --k 3 --alpha 0.05 --power 0.80 --test anova --mode sample_size

  # Chi-squared
  python power_analysis.py --effect-size 0.3 --df 2 --alpha 0.05 --power 0.80 --test chi_squared --mode sample_size

  # Generate sample size table
  python power_analysis.py --test ttest --table

Test Types:
  ttest: Two-sample t-test (uses Cohen's d)
  ttest_one: One-sample t-test (uses Cohen's d)
  ttest_paired: Paired t-test (uses Cohen's d)
  anova: One-way ANOVA (uses Cohen's f)
  chi_squared: Chi-squared test (uses Cohen's w)
  correlation: Correlation test (uses r)
  regression: Multiple regression (uses f-squared)
        """
    )

    # Test specification
    parser.add_argument('--test', required=True,
                       choices=['ttest', 'ttest_one', 'ttest_paired', 'anova',
                               'chi_squared', 'correlation', 'regression'],
                       help='Statistical test type')

    # Mode
    parser.add_argument('--mode', choices=['power', 'sample_size', 'effect_size'],
                       default='sample_size',
                       help='What to calculate (default: sample_size)')

    # Parameters
    parser.add_argument('--effect-size', type=float,
                       help='Effect size (d for t-test, f for ANOVA, w for chi-sq, r for correlation)')
    parser.add_argument('--n', type=int, help='Sample size (one-sample/paired/correlation)')
    parser.add_argument('--n1', type=int, help='Sample size group 1')
    parser.add_argument('--n2', type=int, help='Sample size group 2')
    parser.add_argument('--n-per-group', type=int, help='Sample size per group (ANOVA)')
    parser.add_argument('--alpha', type=float, default=0.05, help='Significance level (default: 0.05)')
    parser.add_argument('--power', type=float, help='Statistical power (1 - beta)')
    parser.add_argument('--alternative', choices=['two-sided', 'greater', 'less'],
                       default='two-sided', help='Alternative hypothesis')

    # Test-specific parameters
    parser.add_argument('--k', type=int, default=3, help='Number of groups (ANOVA)')
    parser.add_argument('--df', type=int, help='Degrees of freedom (chi-squared)')
    parser.add_argument('--correlation', type=float, default=0.5,
                       help='Pre-post correlation (paired t-test)')
    parser.add_argument('--n-predictors', type=int, help='Number of predictors (regression)')

    # Table generation
    parser.add_argument('--table', action='store_true', help='Generate sample size table')

    # Output
    parser.add_argument('--output', help='Output file path')
    parser.add_argument('--quiet', action='store_true', help='Minimal output')

    args = parser.parse_args()

    # Generate table if requested
    if args.table:
        if args.test == 'ttest':
            effect_sizes = [0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0]
        elif args.test == 'correlation':
            effect_sizes = [0.1, 0.2, 0.3, 0.4, 0.5]
        elif args.test == 'anova':
            effect_sizes = [0.1, 0.15, 0.25, 0.4]
        else:
            effect_sizes = [0.1, 0.2, 0.3, 0.5]

        table = create_sample_size_table(
            test=args.test,
            effect_sizes=effect_sizes,
            alpha=args.alpha,
            k=args.k
        )
        print(table)
        return

    # Run power analysis
    try:
        if args.test == 'ttest':
            n1 = args.n1 or args.n
            n2 = args.n2 or args.n

            result = power_ttest_two_sample(
                effect_size=args.effect_size,
                n1=n1,
                n2=n2,
                alpha=args.alpha,
                power=args.power,
                alternative=args.alternative,
                mode=args.mode
            )

        elif args.test == 'ttest_one':
            result = power_ttest_one_sample(
                effect_size=args.effect_size,
                n=args.n,
                alpha=args.alpha,
                power=args.power,
                alternative=args.alternative,
                mode=args.mode
            )

        elif args.test == 'ttest_paired':
            result = power_ttest_paired(
                effect_size=args.effect_size,
                n=args.n,
                alpha=args.alpha,
                power=args.power,
                correlation=args.correlation,
                alternative=args.alternative,
                mode=args.mode
            )

        elif args.test == 'anova':
            result = power_anova(
                effect_size=args.effect_size,
                k=args.k,
                n_per_group=args.n_per_group,
                alpha=args.alpha,
                power=args.power,
                mode=args.mode
            )

        elif args.test == 'chi_squared':
            result = power_chi_squared(
                effect_size=args.effect_size,
                df=args.df,
                n=args.n,
                alpha=args.alpha,
                power=args.power,
                mode=args.mode
            )

        elif args.test == 'correlation':
            result = power_correlation(
                r=args.effect_size,
                n=args.n,
                alpha=args.alpha,
                power=args.power,
                alternative=args.alternative,
                mode=args.mode
            )

        elif args.test == 'regression':
            result = power_regression(
                f_squared=args.effect_size,
                n_predictors=args.n_predictors,
                n=args.n,
                alpha=args.alpha,
                power=args.power,
                mode=args.mode
            )

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    # Output results
    if not args.quiet:
        print("=" * 50)
        print(f"POWER ANALYSIS: {args.test.upper()}")
        print("=" * 50)
        print()
        print(f"Alpha: {args.alpha}")
        print(f"Alternative: {args.alternative}")
        print()

    if args.mode == 'power':
        print(f"Calculated Power: {result:.4f}")
        if result >= 0.80:
            print("Interpretation: Adequate power (>= 0.80)")
        else:
            print(f"Interpretation: Underpowered (< 0.80)")

    elif args.mode == 'sample_size':
        if args.test in ['ttest', 'ttest_one', 'ttest_paired', 'correlation']:
            print(f"Required Sample Size: {result} per group")
        elif args.test == 'anova':
            print(f"Required Sample Size: {result} per group ({result * args.k} total)")
        else:
            print(f"Required Sample Size: {result}")

    elif args.mode == 'effect_size':
        print(f"Minimum Detectable Effect: {result:.4f}")
        if args.test == 'ttest':
            print(f"Interpretation: Can detect Cohen's d >= {result:.2f}")

    if args.output:
        with open(args.output, 'w') as f:
            f.write(f"test,mode,result,alpha,power,effect_size\n")
            f.write(f"{args.test},{args.mode},{result},{args.alpha},{args.power or ''},{args.effect_size or ''}\n")
        print(f"\nResults written to {args.output}")


if __name__ == '__main__':
    main()
