#!/usr/bin/env python3
"""
Model Assumption Testing for Discrete Choice Models

Tests key assumptions:
- Binary: Link specification (logit vs probit)
- Ordered: Parallel lines assumption
- Multinomial: Independence of Irrelevant Alternatives (IIA)
- Count: Overdispersion, zero-inflation

Usage:
    python test_model_assumptions.py --data data.csv --y outcome --x "var1 var2" --model logit --test link
    python test_model_assumptions.py --data data.csv --y count --x "x1 x2" --model poisson --test overdispersion
"""

import argparse
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from statsmodels.discrete.discrete_model import (
    Logit, Probit, Poisson, NegativeBinomial, MNLogit
)
from statsmodels.miscmodels.ordinal_model import OrderedModel


def parse_args():
    parser = argparse.ArgumentParser(
        description='Test assumptions for discrete choice models'
    )

    parser.add_argument('--data', '-d', required=True, help='Path to CSV data')
    parser.add_argument('--y', required=True, help='Outcome variable name')
    parser.add_argument('--x', required=True, help='Covariates (space-separated)')
    parser.add_argument('--model', '-m', required=True,
                        choices=['logit', 'probit', 'ologit', 'oprobit', 'mlogit', 'poisson', 'negbin'],
                        help='Model type')
    parser.add_argument('--test', '-t', required=True,
                        choices=['link', 'parallel', 'iia', 'overdispersion', 'zeroinflation', 'all'],
                        help='Test to perform')
    parser.add_argument('--exclude-alt', type=int, help='Alternative to exclude for IIA test')
    parser.add_argument('--alpha', type=float, default=0.05, help='Significance level')
    parser.add_argument('--verbose', '-v', action='store_true')

    return parser.parse_args()


# =============================================================================
# Binary Model Tests
# =============================================================================

def test_link_specification(y: np.ndarray, X: np.ndarray, alpha: float = 0.05) -> Dict:
    """
    Test for correct link specification using RESET-style test.

    Adds predicted probabilities squared/cubed to check for misspecification.
    """
    X_const = sm.add_constant(X)

    # Fit logit
    logit = Logit(y, X_const).fit(disp=0)
    p_logit = logit.predict()

    # Fit probit
    probit = Probit(y, X_const).fit(disp=0)
    p_probit = probit.predict()

    # RESET-style test: add p^2, p^3 to model
    def reset_test(model_result, p):
        X_aug = np.column_stack([X_const, p**2, p**3])
        try:
            aug_model = type(model_result.model)(y, X_aug).fit(disp=0)
            # LR test
            lr_stat = 2 * (aug_model.llf - model_result.llf)
            p_value = stats.chi2.sf(lr_stat, 2)
            return lr_stat, p_value
        except:
            return np.nan, np.nan

    logit_lr, logit_p = reset_test(logit, p_logit)
    probit_lr, probit_p = reset_test(probit, p_probit)

    # Compare AIC/BIC
    result = {
        'test': 'Link Specification Test',
        'logit': {
            'aic': logit.aic,
            'bic': logit.bic,
            'llf': logit.llf,
            'reset_stat': logit_lr,
            'reset_pvalue': logit_p
        },
        'probit': {
            'aic': probit.aic,
            'bic': probit.bic,
            'llf': probit.llf,
            'reset_stat': probit_lr,
            'reset_pvalue': probit_p
        },
        'recommendation': 'logit' if logit.aic < probit.aic else 'probit'
    }

    return result


def hosmer_lemeshow_test(y: np.ndarray, p: np.ndarray, n_groups: int = 10) -> Tuple[float, float]:
    """
    Hosmer-Lemeshow goodness-of-fit test.

    Groups observations by predicted probability and compares
    observed vs expected counts in each group.
    """
    # Create groups based on predicted probabilities
    df = pd.DataFrame({'y': y, 'p': p})
    df['group'] = pd.qcut(df['p'], n_groups, labels=False, duplicates='drop')

    # Calculate observed and expected
    grouped = df.groupby('group').agg({
        'y': ['sum', 'count'],
        'p': 'mean'
    })
    grouped.columns = ['observed', 'n', 'expected_p']
    grouped['expected'] = grouped['n'] * grouped['expected_p']

    # Chi-squared statistic
    chi2 = np.sum((grouped['observed'] - grouped['expected'])**2 /
                  (grouped['expected'] * (1 - grouped['expected_p']) + 1e-10))

    df_test = n_groups - 2
    p_value = stats.chi2.sf(chi2, df_test)

    return chi2, p_value


# =============================================================================
# Ordered Model Tests
# =============================================================================

def brant_test(y: np.ndarray, X: np.ndarray, distr: str = 'logit',
               alpha: float = 0.05) -> Dict:
    """
    Brant test for parallel lines assumption in ordered choice models.

    Compares ordered model to series of binary logits at each cutpoint.
    """
    X_const = sm.add_constant(X)
    categories = sorted(np.unique(y))
    J = len(categories)

    # Fit ordered model
    ordered = OrderedModel(y, X, distr=distr).fit(method='bfgs', disp=0)
    beta_ordered = ordered.params[:-J+1]  # Exclude thresholds

    # Fit J-1 binary logits
    binary_betas = []
    binary_se = []

    for j in range(1, J):
        y_binary = (y >= categories[j]).astype(int)

        if distr == 'logit':
            binary_model = Logit(y_binary, X_const).fit(disp=0)
        else:
            binary_model = Probit(y_binary, X_const).fit(disp=0)

        binary_betas.append(binary_model.params[1:])  # Exclude constant
        binary_se.append(binary_model.bse[1:])

    binary_betas = np.array(binary_betas)
    binary_se = np.array(binary_se)

    # Test for equality of coefficients across equations
    # For each variable, test if beta varies across cutpoints
    n_vars = len(beta_ordered)
    test_results = []

    for k in range(n_vars):
        betas_k = binary_betas[:, k]
        se_k = binary_se[:, k]

        # Wald test for equality
        mean_beta = np.mean(betas_k)
        # Simplified chi-squared
        chi2 = np.sum((betas_k - mean_beta)**2 / (se_k**2 + 1e-10))
        df = J - 2  # J-1 equations minus 1 for the restriction
        p_value = stats.chi2.sf(chi2, df)

        test_results.append({
            'variable': k,
            'chi2': chi2,
            'df': df,
            'p_value': p_value,
            'reject': p_value < alpha
        })

    # Overall test
    overall_chi2 = sum(r['chi2'] for r in test_results)
    overall_df = sum(r['df'] for r in test_results)
    overall_p = stats.chi2.sf(overall_chi2, overall_df)

    return {
        'test': 'Brant Test for Parallel Lines',
        'variable_tests': test_results,
        'overall_chi2': overall_chi2,
        'overall_df': overall_df,
        'overall_p_value': overall_p,
        'parallel_lines_holds': overall_p >= alpha,
        'binary_coefficients': binary_betas
    }


def score_test_parallel_lines(y: np.ndarray, X: np.ndarray, distr: str = 'logit') -> Dict:
    """
    Score test for parallel lines assumption.

    More efficient than Brant test but asymptotically equivalent.
    """
    # Fit constrained model (ordered)
    ordered = OrderedModel(y, X, distr=distr).fit(method='bfgs', disp=0)

    # Score statistic under null
    # This requires computing the score at constrained estimates
    # Simplified implementation

    return {
        'test': 'Score Test for Parallel Lines',
        'note': 'See Brant test for practical implementation'
    }


# =============================================================================
# Multinomial Model Tests
# =============================================================================

def hausman_mcfadden_test(y: np.ndarray, X: np.ndarray,
                          excluded_alt: int, alpha: float = 0.05) -> Dict:
    """
    Hausman-McFadden test for IIA in multinomial logit.

    Compares full model to restricted model excluding one alternative.
    """
    X_const = sm.add_constant(X)
    categories = sorted(np.unique(y))

    # Full model
    full = MNLogit(y, X_const).fit(disp=0)
    beta_f = full.params.flatten()
    V_f = full.cov_params()

    # Restricted model (exclude one alternative)
    mask = y != excluded_alt
    y_restricted = y[mask].copy()
    X_restricted = X_const[mask]

    # Recode alternatives
    for i, cat in enumerate(categories):
        if cat == excluded_alt:
            continue
        new_code = i if cat < excluded_alt else i - 1
        y_restricted[y_restricted == cat] = new_code

    # Fit restricted
    try:
        restricted = MNLogit(y_restricted, X_restricted).fit(disp=0)
        beta_s = restricted.params.flatten()
        V_s = restricted.cov_params()

        # The test requires careful alignment of parameters
        # Simplified version
        n_params_s = len(beta_s)

        # Hausman statistic (simplified)
        # Full implementation requires parameter mapping
        diff = beta_s - beta_f[:n_params_s]
        V_diff = V_s - V_f[:n_params_s, :n_params_s]

        # Check if V_diff is positive definite
        try:
            stat = diff @ np.linalg.inv(V_diff) @ diff
            p_value = stats.chi2.sf(stat, n_params_s)
        except:
            stat = np.nan
            p_value = np.nan

        result = {
            'test': 'Hausman-McFadden Test for IIA',
            'excluded_alternative': excluded_alt,
            'test_statistic': stat,
            'degrees_of_freedom': n_params_s,
            'p_value': p_value,
            'iia_holds': p_value >= alpha if not np.isnan(p_value) else None
        }

    except Exception as e:
        result = {
            'test': 'Hausman-McFadden Test for IIA',
            'excluded_alternative': excluded_alt,
            'error': str(e),
            'note': 'Test could not be computed'
        }

    return result


def small_hsiao_test(y: np.ndarray, X: np.ndarray,
                     excluded_alt: int, alpha: float = 0.05,
                     n_reps: int = 100) -> Dict:
    """
    Small-Hsiao test for IIA.

    Uses split-sample approach to avoid negative variance issues.
    """
    X_const = sm.add_constant(X)
    n = len(y)
    categories = sorted(np.unique(y))

    test_stats = []

    for _ in range(n_reps):
        # Random split
        perm = np.random.permutation(n)
        split1 = perm[:n//2]
        split2 = perm[n//2:]

        # Full model on split 1
        try:
            full_1 = MNLogit(y[split1], X_const[split1]).fit(disp=0)

            # Restricted on split 2
            mask = y[split2] != excluded_alt
            y_r = y[split2][mask].copy()
            X_r = X_const[split2][mask]

            # Recode
            for i, cat in enumerate(categories):
                if cat == excluded_alt:
                    continue
                new_code = i if cat < excluded_alt else i - 1
                y_r[y_r == cat] = new_code

            rest_2 = MNLogit(y_r, X_r).fit(disp=0)

            # Test statistic
            stat = -2 * (rest_2.llf - full_1.llf)
            test_stats.append(stat)

        except:
            continue

    if len(test_stats) > 0:
        mean_stat = np.mean(test_stats)
        # Degrees of freedom: parameters eliminated
        k = full_1.params.size - rest_2.params.size
        p_value = stats.chi2.sf(mean_stat, k)

        result = {
            'test': 'Small-Hsiao Test for IIA',
            'excluded_alternative': excluded_alt,
            'mean_test_statistic': mean_stat,
            'degrees_of_freedom': k,
            'p_value': p_value,
            'n_replications': len(test_stats),
            'iia_holds': p_value >= alpha
        }
    else:
        result = {
            'test': 'Small-Hsiao Test for IIA',
            'error': 'Could not compute test statistics'
        }

    return result


# =============================================================================
# Count Model Tests
# =============================================================================

def cameron_trivedi_test(y: np.ndarray, X: np.ndarray, alpha: float = 0.05) -> Dict:
    """
    Cameron-Trivedi test for overdispersion in Poisson model.

    Tests H0: Var(Y) = E[Y] vs H1: Var(Y) = E[Y] + alpha * g(E[Y])
    """
    X_const = sm.add_constant(X)

    # Fit Poisson
    poisson = Poisson(y, X_const).fit(disp=0)
    mu = poisson.fittedvalues

    # Auxiliary regression
    # (y - mu)^2 - y on mu (or mu^2)
    aux_dep = ((y - mu)**2 - y) / mu

    # Test with g(mu) = mu
    aux_reg_1 = sm.OLS(aux_dep, mu).fit()

    # Test with g(mu) = mu^2
    aux_reg_2 = sm.OLS(aux_dep, mu**2).fit()

    result = {
        'test': 'Cameron-Trivedi Overdispersion Test',
        'test_g_mu': {
            'alpha_estimate': aux_reg_1.params[0],
            't_statistic': aux_reg_1.tvalues[0],
            'p_value': aux_reg_1.pvalues[0],
            'overdispersion': aux_reg_1.pvalues[0] < alpha and aux_reg_1.params[0] > 0
        },
        'test_g_mu_squared': {
            'alpha_estimate': aux_reg_2.params[0],
            't_statistic': aux_reg_2.tvalues[0],
            'p_value': aux_reg_2.pvalues[0],
            'overdispersion': aux_reg_2.pvalues[0] < alpha and aux_reg_2.params[0] > 0
        }
    }

    return result


def dispersion_statistic(y: np.ndarray, X: np.ndarray) -> Dict:
    """
    Compute Pearson dispersion statistic.

    phi = (1/(n-k)) * sum((y - mu)^2 / mu)

    phi > 1 indicates overdispersion
    phi < 1 indicates underdispersion
    """
    X_const = sm.add_constant(X)

    poisson = Poisson(y, X_const).fit(disp=0)
    mu = poisson.fittedvalues

    n = len(y)
    k = X_const.shape[1]

    # Pearson chi-squared
    pearson_chi2 = np.sum((y - mu)**2 / mu)
    phi = pearson_chi2 / (n - k)

    # Deviance
    deviance = 2 * np.sum(y * np.log((y + 1e-10) / mu) - (y - mu))
    phi_dev = deviance / (n - k)

    return {
        'test': 'Dispersion Statistics',
        'pearson_chi2': pearson_chi2,
        'pearson_phi': phi,
        'deviance': deviance,
        'deviance_phi': phi_dev,
        'df': n - k,
        'overdispersion': phi > 1.5,
        'interpretation': 'overdispersed' if phi > 1.5 else ('underdispersed' if phi < 0.7 else 'equidispersed')
    }


def likelihood_ratio_test_nb(y: np.ndarray, X: np.ndarray, alpha: float = 0.05) -> Dict:
    """
    Likelihood ratio test: Poisson vs Negative Binomial.

    Tests H0: alpha = 0 (Poisson) vs H1: alpha > 0 (NB)
    """
    X_const = sm.add_constant(X)

    poisson = Poisson(y, X_const).fit(disp=0)
    negbin = NegativeBinomial(y, X_const).fit(disp=0)

    # LR statistic
    lr_stat = 2 * (negbin.llf - poisson.llf)

    # Under H0, distribution is mixture of chi2(0) and chi2(1)
    # Use chi2(1) as conservative approximation
    p_value = stats.chi2.sf(lr_stat, 1) / 2  # One-sided

    return {
        'test': 'LR Test: Poisson vs Negative Binomial',
        'poisson_llf': poisson.llf,
        'negbin_llf': negbin.llf,
        'lr_statistic': lr_stat,
        'p_value': p_value,
        'alpha_estimate': negbin.params[-1] if hasattr(negbin, 'lnalpha') else 'N/A',
        'prefer_negbin': p_value < alpha
    }


def test_zero_inflation(y: np.ndarray, X: np.ndarray, alpha: float = 0.05) -> Dict:
    """
    Test for excess zeros in count data.

    Compares observed zeros to expected under Poisson.
    """
    X_const = sm.add_constant(X)

    # Fit Poisson
    poisson = Poisson(y, X_const).fit(disp=0)
    mu = poisson.fittedvalues

    # Expected zeros under Poisson
    expected_zeros = np.sum(np.exp(-mu))
    observed_zeros = np.sum(y == 0)

    # Chi-squared test
    chi2 = (observed_zeros - expected_zeros)**2 / expected_zeros
    p_value = stats.chi2.sf(chi2, 1)

    # Proportion test
    n = len(y)
    prop_observed = observed_zeros / n
    prop_expected = expected_zeros / n

    # Score test for zero-inflation
    score = (observed_zeros - expected_zeros) / np.sqrt(expected_zeros * (1 - expected_zeros/n))
    score_p = 2 * stats.norm.sf(abs(score))

    return {
        'test': 'Zero-Inflation Test',
        'observed_zeros': observed_zeros,
        'expected_zeros_poisson': expected_zeros,
        'proportion_observed': prop_observed,
        'proportion_expected': prop_expected,
        'chi2_statistic': chi2,
        'chi2_p_value': p_value,
        'score_statistic': score,
        'score_p_value': score_p,
        'excess_zeros': observed_zeros > expected_zeros,
        'zero_inflation_likely': p_value < alpha and observed_zeros > expected_zeros
    }


def vuong_test(y: np.ndarray, X: np.ndarray, alpha: float = 0.05) -> Dict:
    """
    Vuong test comparing Poisson to Zero-Inflated Poisson.
    """
    from statsmodels.discrete.count_model import ZeroInflatedPoisson

    X_const = sm.add_constant(X)

    # Fit both models
    poisson = Poisson(y, X_const).fit(disp=0)
    try:
        zip_model = ZeroInflatedPoisson(y, X_const, exog_infl=X_const).fit(disp=0)
    except:
        return {
            'test': 'Vuong Test: Poisson vs ZIP',
            'error': 'ZIP model could not be fitted'
        }

    # Individual log-likelihood differences
    ll_pois = stats.poisson.logpmf(y, poisson.fittedvalues)
    ll_zip = zip_model.predict(which='prob')

    # For each observation, get P(Y=y)
    ll_zip_obs = np.array([ll_zip[i, int(y[i])] if int(y[i]) < ll_zip.shape[1] else 1e-10
                          for i in range(len(y))])
    ll_zip_obs = np.log(ll_zip_obs + 1e-10)

    m = ll_zip_obs - ll_pois
    n = len(y)

    # Vuong statistic
    v_stat = np.sqrt(n) * np.mean(m) / np.std(m)
    p_value = 2 * stats.norm.sf(abs(v_stat))

    return {
        'test': 'Vuong Test: Poisson vs ZIP',
        'vuong_statistic': v_stat,
        'p_value': p_value,
        'poisson_aic': poisson.aic,
        'zip_aic': zip_model.aic,
        'interpretation': 'ZIP preferred' if v_stat > 1.96 else ('Poisson preferred' if v_stat < -1.96 else 'Models equivalent')
    }


# =============================================================================
# Main
# =============================================================================

def print_test_result(result: Dict, verbose: bool = False):
    """Pretty print test results."""
    print("\n" + "="*60)
    print(f"TEST: {result.get('test', 'Unknown')}")
    print("="*60)

    for key, value in result.items():
        if key == 'test':
            continue
        if isinstance(value, dict):
            print(f"\n{key}:")
            for k, v in value.items():
                if isinstance(v, float):
                    print(f"  {k}: {v:.6f}")
                else:
                    print(f"  {k}: {v}")
        elif isinstance(value, (list, np.ndarray)) and not verbose:
            print(f"{key}: [array of length {len(value)}]")
        elif isinstance(value, float):
            print(f"{key}: {value:.6f}")
        else:
            print(f"{key}: {value}")


def main():
    args = parse_args()

    # Parse variables
    x_vars = args.x.split()

    # Load data
    df = pd.read_csv(args.data)
    y = df[args.y].values
    X = df[x_vars].values

    # Drop missing
    mask = ~(np.isnan(y) | np.any(np.isnan(X), axis=1))
    y = y[mask]
    X = X[mask]

    print(f"Data loaded: N = {len(y)}")
    print(f"Model type: {args.model}")
    print(f"Test: {args.test}")

    results = []

    # Run tests based on model type and test requested
    if args.test in ['link', 'all'] and args.model in ['logit', 'probit']:
        results.append(test_link_specification(y, X, args.alpha))

    if args.test in ['parallel', 'all'] and args.model in ['ologit', 'oprobit']:
        distr = 'logit' if args.model == 'ologit' else 'probit'
        results.append(brant_test(y, X, distr, args.alpha))

    if args.test in ['iia', 'all'] and args.model == 'mlogit':
        if args.exclude_alt is not None:
            results.append(hausman_mcfadden_test(y, X, args.exclude_alt, args.alpha))
            results.append(small_hsiao_test(y, X, args.exclude_alt, args.alpha))
        else:
            # Test excluding each alternative
            for alt in np.unique(y)[1:]:  # Skip first (reference)
                results.append(hausman_mcfadden_test(y, X, int(alt), args.alpha))

    if args.test in ['overdispersion', 'all'] and args.model in ['poisson', 'negbin']:
        results.append(cameron_trivedi_test(y, X, args.alpha))
        results.append(dispersion_statistic(y, X))
        results.append(likelihood_ratio_test_nb(y, X, args.alpha))

    if args.test in ['zeroinflation', 'all'] and args.model in ['poisson', 'negbin']:
        results.append(test_zero_inflation(y, X, args.alpha))
        results.append(vuong_test(y, X, args.alpha))

    # Print results
    for result in results:
        print_test_result(result, args.verbose)


if __name__ == '__main__':
    main()
