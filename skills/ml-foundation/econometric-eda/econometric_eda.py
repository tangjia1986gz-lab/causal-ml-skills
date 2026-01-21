"""
Econometric EDA Module

Comprehensive exploratory data analysis tailored for econometric research,
focusing on data quality issues that could affect causal inference validity.
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, List, Optional, Union, Tuple, Any
from pathlib import Path
import warnings
import json
from datetime import datetime

# Optional imports with fallbacks
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_PLOTTING = True
except ImportError:
    HAS_PLOTTING = False
    warnings.warn("matplotlib/seaborn not available. Plotting disabled.")

try:
    import statsmodels.api as sm
    from statsmodels.stats.outliers_influence import variance_inflation_factor, OLSInfluence
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False
    warnings.warn("statsmodels not available. Some diagnostics disabled.")

try:
    from sklearn.ensemble import IsolationForest
    from sklearn.preprocessing import StandardScaler
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    warnings.warn("sklearn not available. ML-based outlier detection disabled.")


class EconometricEDA:
    """
    Exploratory Data Analysis for Econometric Research.

    Focuses on identifying data issues that could affect causal inference validity:
    - Missing data patterns and mechanisms
    - Outliers and influential observations
    - Multicollinearity among regressors
    - Panel data structure and attrition
    """

    def __init__(self, data: pd.DataFrame, copy: bool = True):
        """
        Initialize EDA with data.

        Parameters
        ----------
        data : pd.DataFrame
            Input data for analysis
        copy : bool
            Whether to copy the data (default True)
        """
        self.data = data.copy() if copy else data
        self.results = {}
        self._numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        self._categorical_cols = data.select_dtypes(include=['object', 'category']).columns.tolist()

    # ==================== DATA SUMMARY ====================

    def summarize_data(self, variables: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Generate comprehensive summary statistics.

        Parameters
        ----------
        variables : list, optional
            Variables to summarize. If None, use all numeric variables.

        Returns
        -------
        pd.DataFrame
            Summary statistics table
        """
        if variables is None:
            variables = self._numeric_cols

        summary_list = []

        for var in variables:
            if var not in self.data.columns:
                continue

            col = self.data[var]

            if col.dtype in ['float64', 'int64', 'float32', 'int32']:
                summary_list.append({
                    'variable': var,
                    'type': 'numeric',
                    'n': col.count(),
                    'missing': col.isna().sum(),
                    'missing_pct': col.isna().mean() * 100,
                    'mean': col.mean(),
                    'std': col.std(),
                    'min': col.min(),
                    'p25': col.quantile(0.25),
                    'median': col.median(),
                    'p75': col.quantile(0.75),
                    'max': col.max(),
                    'skewness': col.skew(),
                    'kurtosis': col.kurtosis()
                })
            else:
                summary_list.append({
                    'variable': var,
                    'type': 'categorical',
                    'n': col.count(),
                    'missing': col.isna().sum(),
                    'missing_pct': col.isna().mean() * 100,
                    'n_unique': col.nunique(),
                    'mode': col.mode().iloc[0] if len(col.mode()) > 0 else None,
                    'mode_freq': col.value_counts().iloc[0] if col.count() > 0 else 0
                })

        summary_df = pd.DataFrame(summary_list)
        self.results['summary'] = summary_df
        return summary_df

    # ==================== MISSING DATA ANALYSIS ====================

    def check_missing(self, variables: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Comprehensive missing data analysis.

        Parameters
        ----------
        variables : list, optional
            Variables to check. If None, use all variables.

        Returns
        -------
        dict
            Missing data analysis results
        """
        if variables is None:
            variables = self.data.columns.tolist()

        subset = self.data[variables]

        # Basic missing counts
        missing_counts = subset.isna().sum()
        missing_pct = (missing_counts / len(subset)) * 100

        missing_summary = pd.DataFrame({
            'variable': variables,
            'missing_count': missing_counts.values,
            'missing_pct': missing_pct.values,
            'complete_count': len(subset) - missing_counts.values
        }).sort_values('missing_pct', ascending=False)

        # Missing patterns
        pattern_df = subset.isna().astype(int)
        patterns = pattern_df.groupby(list(pattern_df.columns)).size().reset_index(name='count')
        patterns['pct'] = (patterns['count'] / len(subset)) * 100
        patterns = patterns.sort_values('count', ascending=False)

        # Missing correlations
        missing_corr = subset.isna().corr()

        # Complete cases
        n_complete = subset.dropna().shape[0]
        pct_complete = (n_complete / len(subset)) * 100

        results = {
            'summary': missing_summary,
            'patterns': patterns,
            'correlations': missing_corr,
            'n_complete_cases': n_complete,
            'pct_complete_cases': pct_complete,
            'variables_with_missing': missing_summary[missing_summary['missing_count'] > 0]['variable'].tolist()
        }

        self.results['missing'] = results
        return results

    def test_mcar(self, variables: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Test for Missing Completely at Random (MCAR) using Little's test approximation.

        Parameters
        ----------
        variables : list, optional
            Numeric variables to test

        Returns
        -------
        dict
            MCAR test results
        """
        if variables is None:
            variables = self._numeric_cols

        subset = self.data[variables].copy()

        # Simple MCAR test: compare means by missing pattern
        patterns = subset.isna().apply(lambda x: ''.join(x.astype(int).astype(str)), axis=1)
        unique_patterns = patterns.unique()

        chi2_stat = 0
        df = 0

        for pattern in unique_patterns:
            pattern_subset = subset[patterns == pattern]
            if len(pattern_subset) > 1:
                for col in variables:
                    col_data = pattern_subset[col].dropna()
                    if len(col_data) > 0:
                        overall_mean = subset[col].mean()
                        overall_var = subset[col].var()
                        if overall_var > 0:
                            chi2_stat += len(col_data) * ((col_data.mean() - overall_mean)**2) / overall_var
                            df += 1

        p_value = 1 - stats.chi2.cdf(chi2_stat, df) if df > 0 else 1.0

        results = {
            'chi2_statistic': chi2_stat,
            'df': df,
            'p_value': p_value,
            'is_mcar': p_value > 0.05,
            'interpretation': 'MCAR assumption supported' if p_value > 0.05 else 'MCAR assumption rejected'
        }

        self.results['mcar_test'] = results
        return results

    def test_mar(self, target_var: str, predictors: List[str]) -> Dict[str, Any]:
        """
        Test for Missing at Random (MAR) by comparing predictors across missing/non-missing groups.

        Parameters
        ----------
        target_var : str
            Variable with missing values to test
        predictors : list
            Predictor variables to compare

        Returns
        -------
        dict
            MAR test results
        """
        missing_mask = self.data[target_var].isna()
        results = {}

        for pred in predictors:
            if pred not in self.data.columns:
                continue

            if self.data[pred].dtype in ['float64', 'int64', 'float32', 'int32']:
                group_missing = self.data.loc[missing_mask, pred].dropna()
                group_observed = self.data.loc[~missing_mask, pred].dropna()

                if len(group_missing) > 1 and len(group_observed) > 1:
                    stat, pval = stats.ttest_ind(group_missing, group_observed)
                    results[pred] = {
                        'test': 't-test',
                        'statistic': stat,
                        'p_value': pval,
                        'mean_missing_group': group_missing.mean(),
                        'mean_observed_group': group_observed.mean(),
                        'significant': pval < 0.05
                    }
            else:
                contingency = pd.crosstab(self.data[pred], missing_mask)
                if contingency.shape[0] > 1 and contingency.shape[1] > 1:
                    chi2, pval, dof, expected = stats.chi2_contingency(contingency)
                    results[pred] = {
                        'test': 'chi2',
                        'statistic': chi2,
                        'p_value': pval,
                        'significant': pval < 0.05
                    }

        self.results['mar_test'] = results
        return results

    # ==================== OUTLIER DETECTION ====================

    def detect_outliers(self,
                        variables: Optional[List[str]] = None,
                        method: str = 'iqr',
                        **kwargs) -> Dict[str, Any]:
        """
        Detect outliers using various methods.

        Parameters
        ----------
        variables : list, optional
            Numeric variables to check
        method : str
            Detection method: 'iqr', 'zscore', 'mad', 'mahalanobis', 'isolation_forest'
        **kwargs : dict
            Method-specific parameters

        Returns
        -------
        dict
            Outlier detection results
        """
        if variables is None:
            variables = self._numeric_cols

        if method == 'iqr':
            return self._detect_outliers_iqr(variables, **kwargs)
        elif method == 'zscore':
            return self._detect_outliers_zscore(variables, **kwargs)
        elif method == 'mad':
            return self._detect_outliers_mad(variables, **kwargs)
        elif method == 'mahalanobis':
            return self._detect_outliers_mahalanobis(variables, **kwargs)
        elif method == 'isolation_forest':
            return self._detect_outliers_isolation_forest(variables, **kwargs)
        else:
            raise ValueError(f"Unknown method: {method}")

    def _detect_outliers_iqr(self, variables: List[str], k: float = 1.5) -> Dict[str, Any]:
        """IQR-based outlier detection."""
        results = {}

        for var in variables:
            col = self.data[var].dropna()
            q1, q3 = col.quantile([0.25, 0.75])
            iqr = q3 - q1

            lower = q1 - k * iqr
            upper = q3 + k * iqr

            outliers = (self.data[var] < lower) | (self.data[var] > upper)

            results[var] = {
                'method': 'iqr',
                'k': k,
                'lower_bound': lower,
                'upper_bound': upper,
                'n_outliers': outliers.sum(),
                'pct_outliers': (outliers.sum() / len(self.data)) * 100,
                'outlier_mask': outliers
            }

        self.results['outliers_iqr'] = results
        return results

    def _detect_outliers_zscore(self, variables: List[str], threshold: float = 3.0) -> Dict[str, Any]:
        """Z-score based outlier detection."""
        results = {}

        for var in variables:
            col = self.data[var].dropna()
            z_scores = np.abs(stats.zscore(col))

            outlier_indices = col.index[z_scores > threshold]
            outliers = self.data.index.isin(outlier_indices)

            results[var] = {
                'method': 'zscore',
                'threshold': threshold,
                'n_outliers': outliers.sum(),
                'pct_outliers': (outliers.sum() / len(self.data)) * 100,
                'outlier_mask': pd.Series(outliers, index=self.data.index)
            }

        self.results['outliers_zscore'] = results
        return results

    def _detect_outliers_mad(self, variables: List[str], threshold: float = 3.5) -> Dict[str, Any]:
        """Modified Z-score (MAD) based outlier detection."""
        results = {}

        for var in variables:
            col = self.data[var].dropna()
            median = col.median()
            mad = np.median(np.abs(col - median))

            if mad > 0:
                modified_z = 0.6745 * (col - median) / mad
                outlier_indices = col.index[np.abs(modified_z) > threshold]
            else:
                outlier_indices = []

            outliers = self.data.index.isin(outlier_indices)

            results[var] = {
                'method': 'mad',
                'threshold': threshold,
                'median': median,
                'mad': mad,
                'n_outliers': outliers.sum(),
                'pct_outliers': (outliers.sum() / len(self.data)) * 100,
                'outlier_mask': pd.Series(outliers, index=self.data.index)
            }

        self.results['outliers_mad'] = results
        return results

    def _detect_outliers_mahalanobis(self, variables: List[str],
                                      threshold_pvalue: float = 0.001) -> Dict[str, Any]:
        """Mahalanobis distance based multivariate outlier detection."""
        subset = self.data[variables].dropna()

        mean = subset.mean()
        cov = subset.cov()

        try:
            cov_inv = np.linalg.inv(cov)
        except np.linalg.LinAlgError:
            cov_inv = np.linalg.pinv(cov)

        distances = []
        for idx, row in subset.iterrows():
            diff = row - mean
            d = np.sqrt(diff @ cov_inv @ diff)
            distances.append(d)

        distances = np.array(distances)

        # Chi-square test
        p = len(variables)
        chi2_threshold = stats.chi2.ppf(1 - threshold_pvalue, p)
        outliers = distances**2 > chi2_threshold

        results_df = pd.DataFrame({
            'mahalanobis_distance': distances,
            'is_outlier': outliers
        }, index=subset.index)

        results = {
            'method': 'mahalanobis',
            'threshold_pvalue': threshold_pvalue,
            'chi2_threshold': chi2_threshold,
            'n_outliers': outliers.sum(),
            'pct_outliers': (outliers.sum() / len(subset)) * 100,
            'results': results_df
        }

        self.results['outliers_mahalanobis'] = results
        return results

    def _detect_outliers_isolation_forest(self, variables: List[str],
                                           contamination: float = 0.05) -> Dict[str, Any]:
        """Isolation Forest based outlier detection."""
        if not HAS_SKLEARN:
            raise ImportError("sklearn required for Isolation Forest")

        subset = self.data[variables].dropna()

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(subset)

        iso_forest = IsolationForest(
            contamination=contamination,
            random_state=42,
            n_estimators=100
        )

        predictions = iso_forest.fit_predict(X_scaled)
        scores = iso_forest.decision_function(X_scaled)

        outliers = predictions == -1

        results_df = pd.DataFrame({
            'anomaly_score': scores,
            'is_outlier': outliers
        }, index=subset.index)

        results = {
            'method': 'isolation_forest',
            'contamination': contamination,
            'n_outliers': outliers.sum(),
            'pct_outliers': (outliers.sum() / len(subset)) * 100,
            'results': results_df
        }

        self.results['outliers_isolation_forest'] = results
        return results

    # ==================== CORRELATION ANALYSIS ====================

    def correlation_analysis(self,
                             variables: Optional[List[str]] = None,
                             method: str = 'pearson') -> Dict[str, Any]:
        """
        Correlation analysis with significance testing.

        Parameters
        ----------
        variables : list, optional
            Numeric variables to analyze
        method : str
            'pearson' or 'spearman'

        Returns
        -------
        dict
            Correlation analysis results
        """
        if variables is None:
            variables = self._numeric_cols

        subset = self.data[variables].dropna()
        n_vars = len(variables)

        corr_matrix = np.zeros((n_vars, n_vars))
        pval_matrix = np.zeros((n_vars, n_vars))

        for i, col1 in enumerate(variables):
            for j, col2 in enumerate(variables):
                if i == j:
                    corr_matrix[i, j] = 1.0
                    pval_matrix[i, j] = 0.0
                else:
                    if method == 'pearson':
                        corr, pval = stats.pearsonr(subset[col1], subset[col2])
                    else:
                        corr, pval = stats.spearmanr(subset[col1], subset[col2])
                    corr_matrix[i, j] = corr
                    pval_matrix[i, j] = pval

        corr_df = pd.DataFrame(corr_matrix, index=variables, columns=variables)
        pval_df = pd.DataFrame(pval_matrix, index=variables, columns=variables)

        # Significance stars
        def get_stars(p):
            if p < 0.001:
                return '***'
            elif p < 0.01:
                return '**'
            elif p < 0.05:
                return '*'
            return ''

        sig_df = pval_df.map(get_stars)

        results = {
            'correlation': corr_df,
            'pvalues': pval_df,
            'significance': sig_df,
            'n_observations': len(subset),
            'method': method
        }

        self.results['correlation'] = results
        return results

    # ==================== MULTICOLLINEARITY ====================

    def check_multicollinearity(self, variables: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Check for multicollinearity using VIF and condition number.

        Parameters
        ----------
        variables : list, optional
            Numeric variables to check

        Returns
        -------
        dict
            Multicollinearity diagnostics
        """
        if not HAS_STATSMODELS:
            raise ImportError("statsmodels required for multicollinearity diagnostics")

        if variables is None:
            variables = self._numeric_cols

        subset = self.data[variables].dropna()
        X = sm.add_constant(subset)

        # VIF
        vif_data = []
        for i, col in enumerate(X.columns):
            if col == 'const':
                continue
            vif = variance_inflation_factor(X.values, i)
            vif_data.append({
                'variable': col,
                'vif': vif,
                'tolerance': 1 / vif if vif > 0 else np.inf,
                'concern': 'severe' if vif > 10 else ('moderate' if vif > 5 else 'low')
            })

        vif_df = pd.DataFrame(vif_data)

        # Condition number
        X_scaled = (X - X.mean()) / X.std()
        X_scaled['const'] = 1
        _, s, _ = np.linalg.svd(X_scaled)
        condition_num = s.max() / s.min()

        results = {
            'vif': vif_df,
            'condition_number': condition_num,
            'condition_concern': 'severe' if condition_num > 100 else ('moderate' if condition_num > 30 else 'low'),
            'max_vif': vif_df['vif'].max(),
            'n_high_vif': (vif_df['vif'] > 10).sum()
        }

        self.results['multicollinearity'] = results
        return results

    # ==================== PANEL DATA ====================

    def panel_variation(self,
                        entity_var: str,
                        time_var: str,
                        variables: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Analyze within and between variation in panel data.

        Parameters
        ----------
        entity_var : str
            Entity identifier variable
        time_var : str
            Time period variable
        variables : list, optional
            Variables to analyze

        Returns
        -------
        dict
            Panel variation decomposition
        """
        if variables is None:
            variables = self._numeric_cols

        results_list = []

        for var in variables:
            if var in [entity_var, time_var]:
                continue
            if self.data[var].dtype not in ['float64', 'int64', 'float32', 'int32']:
                continue

            total_mean = self.data[var].mean()
            total_var = self.data[var].var()
            total_std = self.data[var].std()

            # Entity means
            entity_means = self.data.groupby(entity_var)[var].transform('mean')

            # Between variance
            between_var = self.data.groupby(entity_var)[var].mean().var()
            between_std = np.sqrt(between_var) if between_var > 0 else 0

            # Within variance
            within_deviations = self.data[var] - entity_means
            within_var = within_deviations.var()
            within_std = np.sqrt(within_var) if within_var > 0 else 0

            results_list.append({
                'variable': var,
                'mean': total_mean,
                'total_std': total_std,
                'between_std': between_std,
                'within_std': within_std,
                'between_share': between_var / total_var if total_var > 0 else 0,
                'within_share': within_var / total_var if total_var > 0 else 0,
                'min': self.data[var].min(),
                'max': self.data[var].max()
            })

        variation_df = pd.DataFrame(results_list)

        # Panel structure summary
        n_entities = self.data[entity_var].nunique()
        n_periods = self.data[time_var].nunique()
        obs_per_entity = self.data.groupby(entity_var).size()

        results = {
            'variation': variation_df,
            'n_entities': n_entities,
            'n_periods': n_periods,
            'n_observations': len(self.data),
            'is_balanced': obs_per_entity.nunique() == 1,
            'obs_per_entity': {
                'min': obs_per_entity.min(),
                'max': obs_per_entity.max(),
                'mean': obs_per_entity.mean()
            }
        }

        self.results['panel_variation'] = results
        return results

    def analyze_attrition(self,
                          entity_var: str,
                          time_var: str,
                          variables: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Analyze panel attrition patterns and test for attrition bias.

        Parameters
        ----------
        entity_var : str
            Entity identifier variable
        time_var : str
            Time period variable
        variables : list, optional
            Variables to test for attrition bias

        Returns
        -------
        dict
            Attrition analysis results
        """
        all_periods = sorted(self.data[time_var].unique())
        first_period = all_periods[0]
        last_period = all_periods[-1]

        # Initial sample
        initial_entities = set(self.data[self.data[time_var] == first_period][entity_var])

        # Retention tracking
        retention = []
        for period in all_periods:
            current_entities = set(self.data[self.data[time_var] == period][entity_var])
            retained = initial_entities.intersection(current_entities)
            retention.append({
                'period': period,
                'n_initial': len(initial_entities),
                'n_retained': len(retained),
                'retention_rate': len(retained) / len(initial_entities) if len(initial_entities) > 0 else 0
            })

        retention_df = pd.DataFrame(retention)

        # Attrition bias test
        if variables:
            baseline = self.data[self.data[time_var] == first_period].copy()
            entities_last = set(self.data[self.data[time_var] == last_period][entity_var])
            baseline['stayer'] = baseline[entity_var].isin(entities_last).astype(int)

            bias_results = []
            for var in variables:
                if var not in baseline.columns:
                    continue
                if baseline[var].dtype not in ['float64', 'int64', 'float32', 'int32']:
                    continue

                stayers = baseline.loc[baseline['stayer'] == 1, var].dropna()
                leavers = baseline.loc[baseline['stayer'] == 0, var].dropna()

                if len(stayers) > 1 and len(leavers) > 1:
                    t_stat, p_value = stats.ttest_ind(stayers, leavers)
                    pooled_std = np.sqrt((stayers.var() + leavers.var()) / 2)
                    std_diff = (stayers.mean() - leavers.mean()) / pooled_std if pooled_std > 0 else 0

                    bias_results.append({
                        'variable': var,
                        'mean_stayers': stayers.mean(),
                        'mean_leavers': leavers.mean(),
                        'std_diff': std_diff,
                        't_statistic': t_stat,
                        'p_value': p_value,
                        'significant': p_value < 0.05
                    })

            bias_df = pd.DataFrame(bias_results) if bias_results else None
        else:
            bias_df = None

        results = {
            'retention': retention_df,
            'overall_attrition_rate': 1 - retention_df.iloc[-1]['retention_rate'],
            'initial_sample_size': len(initial_entities),
            'final_sample_size': retention_df.iloc[-1]['n_retained'],
            'attrition_bias_test': bias_df
        }

        self.results['attrition'] = results
        return results

    # ==================== COVARIATE BALANCE ====================

    def covariate_balance(self,
                          treatment_var: str,
                          covariates: List[str]) -> pd.DataFrame:
        """
        Create covariate balance table comparing treatment and control groups.

        Parameters
        ----------
        treatment_var : str
            Binary treatment variable
        covariates : list
            Covariates to compare

        Returns
        -------
        pd.DataFrame
            Balance table
        """
        results = []

        for cov in covariates:
            if cov not in self.data.columns:
                continue

            treated = self.data.loc[self.data[treatment_var] == 1, cov]
            control = self.data.loc[self.data[treatment_var] == 0, cov]

            mean_t = treated.mean()
            mean_c = control.mean()

            pooled_sd = np.sqrt((treated.var() + control.var()) / 2)
            std_diff = (mean_t - mean_c) / pooled_sd if pooled_sd > 0 else 0

            var_ratio = treated.var() / control.var() if control.var() > 0 else np.nan

            if self.data[cov].dtype in ['float64', 'int64', 'float32', 'int32']:
                stat, pval = stats.ttest_ind(treated.dropna(), control.dropna())
            else:
                contingency = pd.crosstab(self.data[cov], self.data[treatment_var])
                stat, pval, _, _ = stats.chi2_contingency(contingency)

            results.append({
                'variable': cov,
                'mean_treated': mean_t,
                'mean_control': mean_c,
                'std_diff': std_diff,
                'var_ratio': var_ratio,
                'test_stat': stat,
                'p_value': pval,
                'balanced': abs(std_diff) < 0.1
            })

        balance_df = pd.DataFrame(results)
        self.results['balance'] = balance_df
        return balance_df

    # ==================== FULL REPORT ====================

    def full_report(self,
                    outcome_var: Optional[str] = None,
                    treatment_var: Optional[str] = None,
                    covariates: Optional[List[str]] = None,
                    panel_id: Optional[str] = None,
                    time_var: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate comprehensive EDA report.

        Parameters
        ----------
        outcome_var : str, optional
            Outcome variable name
        treatment_var : str, optional
            Treatment variable name
        covariates : list, optional
            Covariate names
        panel_id : str, optional
            Panel entity identifier
        time_var : str, optional
            Panel time variable

        Returns
        -------
        dict
            Complete EDA results
        """
        report = {
            'generated_at': datetime.now().isoformat(),
            'n_observations': len(self.data),
            'n_variables': len(self.data.columns)
        }

        # Summary statistics
        report['summary'] = self.summarize_data()

        # Missing data
        report['missing'] = self.check_missing()

        # Outliers (multiple methods)
        report['outliers'] = {
            'iqr': self.detect_outliers(method='iqr'),
            'zscore': self.detect_outliers(method='zscore')
        }

        # Correlations
        report['correlations'] = self.correlation_analysis()

        # Multicollinearity
        if covariates and len(covariates) > 1:
            try:
                report['multicollinearity'] = self.check_multicollinearity(covariates)
            except Exception as e:
                report['multicollinearity'] = {'error': str(e)}

        # Covariate balance
        if treatment_var and covariates:
            report['balance'] = self.covariate_balance(treatment_var, covariates)

        # Panel analysis
        if panel_id and time_var:
            report['panel'] = {
                'variation': self.panel_variation(panel_id, time_var, covariates),
                'attrition': self.analyze_attrition(panel_id, time_var, covariates)
            }

        self.results['full_report'] = report
        return report

    # ==================== EXPORT ====================

    def export_report(self,
                      report: Dict[str, Any],
                      format: str = 'markdown',
                      path: str = 'eda_report') -> str:
        """
        Export EDA report to file.

        Parameters
        ----------
        report : dict
            Report from full_report()
        format : str
            'markdown', 'html', or 'json'
        path : str
            Output file path (without extension)

        Returns
        -------
        str
            Path to saved file
        """
        if format == 'json':
            # Convert DataFrames to dicts
            def convert_for_json(obj):
                if isinstance(obj, pd.DataFrame):
                    return obj.to_dict(orient='records')
                elif isinstance(obj, pd.Series):
                    return obj.to_dict()
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, (np.int64, np.int32)):
                    return int(obj)
                elif isinstance(obj, (np.float64, np.float32)):
                    return float(obj)
                elif isinstance(obj, dict):
                    return {k: convert_for_json(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_for_json(i) for i in obj]
                return obj

            json_report = convert_for_json(report)
            output_path = f"{path}.json"
            with open(output_path, 'w') as f:
                json.dump(json_report, f, indent=2, default=str)
            return output_path

        elif format in ['markdown', 'html']:
            md_content = self._generate_markdown_report(report)

            if format == 'markdown':
                output_path = f"{path}.md"
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(md_content)
            else:
                output_path = f"{path}.html"
                html_content = self._markdown_to_html(md_content)
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(html_content)

            return output_path

        else:
            raise ValueError(f"Unknown format: {format}")

    def _generate_markdown_report(self, report: Dict[str, Any]) -> str:
        """Generate markdown report content."""
        lines = [
            "# Econometric EDA Report",
            "",
            f"Generated: {report.get('generated_at', datetime.now().isoformat())}",
            "",
            "## Dataset Overview",
            "",
            f"- **Observations**: {report.get('n_observations', 'N/A')}",
            f"- **Variables**: {report.get('n_variables', 'N/A')}",
            ""
        ]

        # Summary statistics
        if 'summary' in report and isinstance(report['summary'], pd.DataFrame):
            lines.extend([
                "## Summary Statistics",
                "",
                report['summary'].to_markdown(index=False),
                ""
            ])

        # Missing data
        if 'missing' in report:
            missing = report['missing']
            lines.extend([
                "## Missing Data Analysis",
                "",
                f"- **Complete cases**: {missing.get('n_complete_cases', 'N/A')} ({missing.get('pct_complete_cases', 0):.1f}%)",
                ""
            ])
            if isinstance(missing.get('summary'), pd.DataFrame):
                lines.extend([
                    "### Missing by Variable",
                    "",
                    missing['summary'].to_markdown(index=False),
                    ""
                ])

        # Multicollinearity
        if 'multicollinearity' in report and isinstance(report['multicollinearity'], dict):
            mc = report['multicollinearity']
            if not mc.get('error'):
                lines.extend([
                    "## Multicollinearity Diagnostics",
                    "",
                    f"- **Condition Number**: {mc.get('condition_number', 'N/A'):.2f} ({mc.get('condition_concern', 'N/A')})",
                    f"- **Max VIF**: {mc.get('max_vif', 'N/A'):.2f}",
                    ""
                ])
                if isinstance(mc.get('vif'), pd.DataFrame):
                    lines.extend([
                        "### Variance Inflation Factors",
                        "",
                        mc['vif'].to_markdown(index=False),
                        ""
                    ])

        # Balance table
        if 'balance' in report and isinstance(report['balance'], pd.DataFrame):
            lines.extend([
                "## Covariate Balance",
                "",
                report['balance'].to_markdown(index=False),
                ""
            ])

        # Panel analysis
        if 'panel' in report:
            panel = report['panel']
            lines.extend([
                "## Panel Data Analysis",
                ""
            ])

            if 'variation' in panel:
                var = panel['variation']
                lines.extend([
                    "### Panel Structure",
                    "",
                    f"- **Entities**: {var.get('n_entities', 'N/A')}",
                    f"- **Time periods**: {var.get('n_periods', 'N/A')}",
                    f"- **Balanced**: {'Yes' if var.get('is_balanced') else 'No'}",
                    ""
                ])
                if isinstance(var.get('variation'), pd.DataFrame):
                    lines.extend([
                        "### Within/Between Variation",
                        "",
                        var['variation'].to_markdown(index=False),
                        ""
                    ])

            if 'attrition' in panel:
                att = panel['attrition']
                lines.extend([
                    "### Attrition Analysis",
                    "",
                    f"- **Initial sample**: {att.get('initial_sample_size', 'N/A')}",
                    f"- **Final sample**: {att.get('final_sample_size', 'N/A')}",
                    f"- **Attrition rate**: {att.get('overall_attrition_rate', 0)*100:.1f}%",
                    ""
                ])

        return '\n'.join(lines)

    def _markdown_to_html(self, md_content: str) -> str:
        """Convert markdown to HTML with basic styling."""
        try:
            import markdown
            html_body = markdown.markdown(md_content, extensions=['tables', 'fenced_code'])
        except ImportError:
            # Basic conversion without markdown library
            html_body = f"<pre>{md_content}</pre>"

        html_template = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Econometric EDA Report</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            line-height: 1.6;
        }}
        h1 {{ color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 10px; }}
        h2 {{ color: #34495e; margin-top: 30px; }}
        h3 {{ color: #7f8c8d; }}
        table {{
            border-collapse: collapse;
            width: 100%;
            margin: 15px 0;
        }}
        th, td {{
            border: 1px solid #ddd;
            padding: 8px;
            text-align: left;
        }}
        th {{ background-color: #3498db; color: white; }}
        tr:nth-child(even) {{ background-color: #f9f9f9; }}
        tr:hover {{ background-color: #f5f5f5; }}
        code {{ background-color: #f4f4f4; padding: 2px 6px; border-radius: 3px; }}
        pre {{ background-color: #f4f4f4; padding: 15px; border-radius: 5px; overflow-x: auto; }}
    </style>
</head>
<body>
{html_body}
</body>
</html>
"""
        return html_template


# ==================== CONVENIENCE FUNCTIONS ====================

def summarize_data(data: pd.DataFrame, variables: Optional[List[str]] = None) -> pd.DataFrame:
    """Convenience function for data summary."""
    eda = EconometricEDA(data)
    return eda.summarize_data(variables)


def check_missing(data: pd.DataFrame, variables: Optional[List[str]] = None) -> Dict[str, Any]:
    """Convenience function for missing data analysis."""
    eda = EconometricEDA(data)
    return eda.check_missing(variables)


def detect_outliers(data: pd.DataFrame,
                    variables: Optional[List[str]] = None,
                    method: str = 'iqr',
                    **kwargs) -> Dict[str, Any]:
    """Convenience function for outlier detection."""
    eda = EconometricEDA(data)
    return eda.detect_outliers(variables, method, **kwargs)


def correlation_analysis(data: pd.DataFrame,
                         variables: Optional[List[str]] = None,
                         method: str = 'pearson') -> Dict[str, Any]:
    """Convenience function for correlation analysis."""
    eda = EconometricEDA(data)
    return eda.correlation_analysis(variables, method)


def check_multicollinearity(data: pd.DataFrame,
                            variables: Optional[List[str]] = None) -> Dict[str, Any]:
    """Convenience function for multicollinearity check."""
    eda = EconometricEDA(data)
    return eda.check_multicollinearity(variables)


def panel_variation(data: pd.DataFrame,
                    entity_var: str,
                    time_var: str,
                    variables: Optional[List[str]] = None) -> Dict[str, Any]:
    """Convenience function for panel variation analysis."""
    eda = EconometricEDA(data)
    return eda.panel_variation(entity_var, time_var, variables)
