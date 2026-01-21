"""
ML Preprocessing Module for Causal Inference

This module provides comprehensive data preprocessing tools designed
for machine learning and causal inference workflows.
"""

from typing import List, Dict, Any, Optional, Tuple, Union
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN
from scipy import stats
import warnings


def diagnose_missing(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Generate a comprehensive missing value report for a DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame to diagnose.

    Returns
    -------
    Dict[str, Any]
        Dictionary containing:
        - 'summary': Overall missing value statistics
        - 'by_column': Per-column missing value information
        - 'patterns': Missingness patterns analysis
        - 'complete_cases': Number of complete cases
        - 'recommendations': Suggested handling strategies

    Examples
    --------
    >>> report = diagnose_missing(df)
    >>> print(report['summary'])
    >>> print(report['by_column'])
    """
    n_rows, n_cols = df.shape

    # Per-column analysis
    missing_counts = df.isnull().sum()
    missing_pcts = (missing_counts / n_rows * 100).round(2)

    by_column = pd.DataFrame({
        'missing_count': missing_counts,
        'missing_pct': missing_pcts,
        'dtype': df.dtypes
    }).sort_values('missing_pct', ascending=False)

    # Only include columns with missing values
    by_column_missing = by_column[by_column['missing_count'] > 0]

    # Overall summary
    total_cells = n_rows * n_cols
    total_missing = missing_counts.sum()
    complete_cases = df.dropna().shape[0]

    summary = {
        'total_rows': n_rows,
        'total_columns': n_cols,
        'total_cells': total_cells,
        'total_missing': int(total_missing),
        'overall_missing_pct': round(total_missing / total_cells * 100, 2),
        'columns_with_missing': int((missing_counts > 0).sum()),
        'complete_cases': complete_cases,
        'complete_cases_pct': round(complete_cases / n_rows * 100, 2)
    }

    # Missingness patterns
    if by_column_missing.shape[0] > 0:
        missing_cols = by_column_missing.index.tolist()
        pattern_df = df[missing_cols].isnull()
        patterns = pattern_df.value_counts().head(10)

        # Check for monotone missingness pattern
        missing_matrix = df[missing_cols].isnull()
        is_monotone = _check_monotone_pattern(missing_matrix)
    else:
        patterns = pd.Series(dtype=int)
        is_monotone = True

    # Generate recommendations
    recommendations = []
    if summary['overall_missing_pct'] < 5:
        recommendations.append("Low missingness (<5%): Listwise deletion or simple imputation likely acceptable")
    elif summary['overall_missing_pct'] < 20:
        recommendations.append("Moderate missingness (5-20%): Consider multiple imputation or model-based methods")
    else:
        recommendations.append("High missingness (>20%): Carefully investigate missingness mechanism; consider specialized methods")

    if complete_cases / n_rows < 0.5:
        recommendations.append("Warning: Less than 50% complete cases - listwise deletion may cause substantial bias")

    return {
        'summary': summary,
        'by_column': by_column_missing,
        'patterns': patterns,
        'is_monotone': is_monotone,
        'complete_cases': complete_cases,
        'recommendations': recommendations
    }


def _check_monotone_pattern(missing_matrix: pd.DataFrame) -> bool:
    """Check if missingness follows a monotone pattern."""
    # Sort columns by missingness rate
    sorted_cols = missing_matrix.sum().sort_values(ascending=False).index
    sorted_matrix = missing_matrix[sorted_cols]

    # Check monotone pattern
    for i in range(len(sorted_cols) - 1):
        col_i = sorted_cols[i]
        col_j = sorted_cols[i + 1]
        # If j is missing, i should also be missing for monotone pattern
        if (~sorted_matrix[col_i] & sorted_matrix[col_j]).any():
            return False
    return True


def handle_missing(
    df: pd.DataFrame,
    strategy: str = 'drop',
    columns: Optional[List[str]] = None,
    n_imputations: int = 5
) -> Union[pd.DataFrame, List[pd.DataFrame]]:
    """
    Handle missing values using specified strategy.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with missing values.
    strategy : str
        Strategy for handling missing values:
        - 'drop': Listwise deletion (remove rows with any missing)
        - 'mean': Mean imputation for numeric columns
        - 'median': Median imputation for numeric columns
        - 'mode': Mode imputation (most frequent value)
        - 'multiple': Multiple imputation (returns list of DataFrames)
    columns : List[str], optional
        Columns to apply imputation to. If None, applies to all columns
        with missing values.
    n_imputations : int
        Number of imputations for multiple imputation strategy.

    Returns
    -------
    pd.DataFrame or List[pd.DataFrame]
        Processed DataFrame(s) with missing values handled.
        Returns list of DataFrames for 'multiple' strategy.

    Examples
    --------
    >>> # Listwise deletion
    >>> df_clean = handle_missing(df, strategy='drop')

    >>> # Mean imputation for specific columns
    >>> df_imputed = handle_missing(df, strategy='mean', columns=['income', 'age'])

    >>> # Multiple imputation
    >>> imputed_dfs = handle_missing(df, strategy='multiple', n_imputations=5)
    """
    df = df.copy()

    if columns is None:
        # Find columns with missing values
        columns = df.columns[df.isnull().any()].tolist()

    if not columns:
        return df

    if strategy == 'drop':
        return df.dropna(subset=columns)

    elif strategy in ['mean', 'median', 'most_frequent', 'mode']:
        sklearn_strategy = 'most_frequent' if strategy == 'mode' else strategy

        # Separate numeric and non-numeric columns
        numeric_cols = df[columns].select_dtypes(include=[np.number]).columns.tolist()
        non_numeric_cols = [c for c in columns if c not in numeric_cols]

        if numeric_cols and strategy in ['mean', 'median']:
            imputer = SimpleImputer(strategy=sklearn_strategy)
            df[numeric_cols] = imputer.fit_transform(df[numeric_cols])

        if non_numeric_cols or strategy in ['most_frequent', 'mode']:
            cols_to_impute = non_numeric_cols if strategy in ['mean', 'median'] else columns
            if cols_to_impute:
                imputer = SimpleImputer(strategy='most_frequent')
                df[cols_to_impute] = imputer.fit_transform(df[cols_to_impute])

        return df

    elif strategy == 'multiple':
        # Simple multiple imputation using bootstrap + noise
        imputed_dfs = []

        for _ in range(n_imputations):
            df_imp = df.copy()

            for col in columns:
                if df[col].isnull().any():
                    non_missing = df[col].dropna()
                    n_missing = df[col].isnull().sum()

                    if df[col].dtype in [np.float64, np.int64, float, int]:
                        # Numeric: bootstrap sample + small noise
                        imputed_values = np.random.choice(non_missing, size=n_missing, replace=True)
                        noise = np.random.normal(0, non_missing.std() * 0.1, size=n_missing)
                        imputed_values = imputed_values + noise
                    else:
                        # Categorical: bootstrap sample
                        imputed_values = np.random.choice(non_missing, size=n_missing, replace=True)

                    df_imp.loc[df[col].isnull(), col] = imputed_values

            imputed_dfs.append(df_imp)

        return imputed_dfs

    else:
        raise ValueError(f"Unknown strategy: {strategy}. Use 'drop', 'mean', 'median', 'mode', or 'multiple'")


def detect_outliers(
    df: pd.DataFrame,
    columns: List[str],
    method: str = 'iqr',
    threshold: float = 1.5,
    contamination: float = 0.1
) -> pd.Series:
    """
    Detect outliers in specified columns using various methods.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    columns : List[str]
        Columns to check for outliers.
    method : str
        Detection method:
        - 'iqr': Interquartile range method
        - 'zscore': Z-score method
        - 'isolation_forest': Isolation Forest (multivariate)
        - 'mahalanobis': Mahalanobis distance
    threshold : float
        Threshold for outlier detection:
        - IQR: multiplier for IQR (default 1.5)
        - Z-score: number of standard deviations (default 3.0 if not specified)
        - Mahalanobis: chi-squared percentile (default 0.975)
    contamination : float
        Expected proportion of outliers for Isolation Forest.

    Returns
    -------
    pd.Series
        Boolean Series where True indicates an outlier.

    Examples
    --------
    >>> outliers = detect_outliers(df, ['income', 'age'], method='iqr')
    >>> df_clean = df[~outliers]

    >>> outliers = detect_outliers(df, ['X1', 'X2', 'X3'], method='isolation_forest')
    """
    data = df[columns].dropna()
    outlier_mask = pd.Series(False, index=df.index)

    if method == 'iqr':
        for col in columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - threshold * IQR
            upper = Q3 + threshold * IQR
            outlier_mask |= (df[col] < lower) | (df[col] > upper)

    elif method == 'zscore':
        z_threshold = threshold if threshold > 2 else 3.0  # Default to 3 if threshold seems like IQR default
        for col in columns:
            z_scores = np.abs(stats.zscore(df[col].dropna()))
            col_outliers = pd.Series(False, index=df.index)
            col_outliers.loc[df[col].notna()] = z_scores > z_threshold
            outlier_mask |= col_outliers

    elif method == 'isolation_forest':
        # Multivariate outlier detection
        valid_idx = df[columns].dropna().index
        iso_forest = IsolationForest(contamination=contamination, random_state=42)
        predictions = iso_forest.fit_predict(df.loc[valid_idx, columns])
        outlier_mask.loc[valid_idx] = predictions == -1

    elif method == 'mahalanobis':
        # Mahalanobis distance for multivariate outliers
        valid_idx = df[columns].dropna().index
        data = df.loc[valid_idx, columns].values

        mean = np.mean(data, axis=0)
        cov = np.cov(data.T)

        try:
            cov_inv = np.linalg.inv(cov)
            diff = data - mean
            mahal_dist = np.sqrt(np.sum(diff @ cov_inv * diff, axis=1))

            # Chi-squared threshold
            chi2_threshold = stats.chi2.ppf(threshold if threshold < 1 else 0.975, df=len(columns))
            outlier_mask.loc[valid_idx] = mahal_dist > np.sqrt(chi2_threshold)
        except np.linalg.LinAlgError:
            warnings.warn("Singular covariance matrix. Falling back to IQR method.")
            return detect_outliers(df, columns, method='iqr', threshold=1.5)

    else:
        raise ValueError(f"Unknown method: {method}. Use 'iqr', 'zscore', 'isolation_forest', or 'mahalanobis'")

    return outlier_mask


def remove_outliers(
    df: pd.DataFrame,
    columns: List[str],
    method: str = 'iqr',
    threshold: float = 1.5
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Remove outliers from DataFrame and return cleaned data with report.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    columns : List[str]
        Columns to check for outliers.
    method : str
        Detection method ('iqr', 'zscore', 'isolation_forest', 'mahalanobis').
    threshold : float
        Threshold for outlier detection.

    Returns
    -------
    Tuple[pd.DataFrame, Dict[str, Any]]
        Tuple of (cleaned DataFrame, removal report).

    Examples
    --------
    >>> df_clean, report = remove_outliers(df, ['income'], method='iqr', threshold=1.5)
    >>> print(f"Removed {report['n_removed']} outliers")
    """
    outlier_mask = detect_outliers(df, columns, method, threshold)

    n_original = len(df)
    n_outliers = outlier_mask.sum()
    df_clean = df[~outlier_mask].copy()

    report = {
        'n_original': n_original,
        'n_removed': int(n_outliers),
        'n_remaining': len(df_clean),
        'pct_removed': round(n_outliers / n_original * 100, 2),
        'method': method,
        'threshold': threshold,
        'columns_checked': columns
    }

    return df_clean, report


def standardize(
    df: pd.DataFrame,
    columns: List[str],
    return_scaler: bool = True
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, StandardScaler]]:
    """
    Standardize columns to zero mean and unit variance (Z-score).

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    columns : List[str]
        Columns to standardize.
    return_scaler : bool
        Whether to return the fitted scaler for transforming new data.

    Returns
    -------
    pd.DataFrame or Tuple[pd.DataFrame, StandardScaler]
        Standardized DataFrame, optionally with the fitted scaler.

    Examples
    --------
    >>> df_std, scaler = standardize(df, ['income', 'age', 'education'])
    >>> # Transform new data using the same scaler
    >>> new_data_std = scaler.transform(new_data[['income', 'age', 'education']])
    """
    df = df.copy()
    scaler = StandardScaler()

    df[columns] = scaler.fit_transform(df[columns])

    if return_scaler:
        return df, scaler
    return df


def create_interactions(
    df: pd.DataFrame,
    var_pairs: List[Tuple[str, str]],
    prefix: str = 'int_'
) -> pd.DataFrame:
    """
    Create interaction terms between pairs of variables.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    var_pairs : List[Tuple[str, str]]
        List of variable pairs to create interactions for.
    prefix : str
        Prefix for interaction column names.

    Returns
    -------
    pd.DataFrame
        DataFrame with original columns plus interaction terms.

    Examples
    --------
    >>> df_int = create_interactions(df, [('age', 'education'), ('income', 'treatment')])
    >>> # Creates columns: int_age_education, int_income_treatment
    """
    df = df.copy()

    for var1, var2 in var_pairs:
        interaction_name = f"{prefix}{var1}_{var2}"
        df[interaction_name] = df[var1] * df[var2]

    return df


def run_pca(
    df: pd.DataFrame,
    n_components: Optional[int] = None,
    variance_threshold: float = 0.95
) -> Dict[str, Any]:
    """
    Perform PCA for dimensionality reduction with variance explained analysis.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame (numeric columns only).
    n_components : int, optional
        Number of components to retain. If None, retains components
        explaining variance_threshold of variance.
    variance_threshold : float
        Cumulative variance threshold if n_components is None.

    Returns
    -------
    Dict[str, Any]
        Dictionary containing:
        - 'transformed': DataFrame with PCA components
        - 'variance_explained': Variance explained by each component
        - 'cumulative_variance': Cumulative variance explained
        - 'n_components': Number of components retained
        - 'loadings': Component loadings matrix
        - 'pca': Fitted PCA object

    Examples
    --------
    >>> result = run_pca(df[control_columns], n_components=10)
    >>> print(f"Variance explained: {result['cumulative_variance'][-1]:.2%}")
    >>> pca_controls = result['transformed']
    """
    # Standardize data first
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(df)

    # Fit PCA to determine components
    if n_components is None:
        # First fit with all components to determine how many to keep
        pca_full = PCA()
        pca_full.fit(data_scaled)
        cumsum = np.cumsum(pca_full.explained_variance_ratio_)
        n_components = int(np.argmax(cumsum >= variance_threshold) + 1)
        n_components = max(1, min(n_components, min(df.shape)))

    # Fit final PCA
    pca = PCA(n_components=n_components)
    transformed = pca.fit_transform(data_scaled)

    # Create DataFrame with component names
    component_names = [f'PC{i+1}' for i in range(n_components)]
    transformed_df = pd.DataFrame(
        transformed,
        index=df.index,
        columns=component_names
    )

    # Loadings matrix
    loadings = pd.DataFrame(
        pca.components_.T,
        index=df.columns,
        columns=component_names
    )

    return {
        'transformed': transformed_df,
        'variance_explained': pca.explained_variance_ratio_,
        'cumulative_variance': np.cumsum(pca.explained_variance_ratio_),
        'n_components': n_components,
        'loadings': loadings,
        'pca': pca,
        'scaler': scaler
    }


def cluster_analysis(
    df: pd.DataFrame,
    method: str = 'kmeans',
    n_clusters: int = 3,
    **kwargs
) -> Dict[str, Any]:
    """
    Perform clustering analysis using K-Means or DBSCAN.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame (numeric columns only).
    method : str
        Clustering method: 'kmeans' or 'dbscan'.
    n_clusters : int
        Number of clusters for K-Means.
    **kwargs
        Additional arguments:
        - For DBSCAN: eps (default 0.5), min_samples (default 5)

    Returns
    -------
    Dict[str, Any]
        Dictionary containing:
        - 'labels': Cluster assignments
        - 'n_clusters': Number of clusters found
        - For K-Means: 'inertia', 'centers'
        - For DBSCAN: 'n_noise', 'core_sample_indices'

    Examples
    --------
    >>> # K-Means clustering
    >>> result = cluster_analysis(df[features], method='kmeans', n_clusters=5)
    >>> df['cluster'] = result['labels']

    >>> # DBSCAN clustering
    >>> result = cluster_analysis(df[features], method='dbscan', eps=0.5, min_samples=5)
    """
    # Standardize data for clustering
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(df)

    if method == 'kmeans':
        model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = model.fit_predict(data_scaled)

        # Cluster centers in original scale
        centers = scaler.inverse_transform(model.cluster_centers_)
        centers_df = pd.DataFrame(centers, columns=df.columns)

        return {
            'labels': labels,
            'n_clusters': n_clusters,
            'inertia': model.inertia_,
            'centers': centers_df,
            'model': model,
            'scaler': scaler
        }

    elif method == 'dbscan':
        eps = kwargs.get('eps', 0.5)
        min_samples = kwargs.get('min_samples', 5)

        model = DBSCAN(eps=eps, min_samples=min_samples)
        labels = model.fit_predict(data_scaled)

        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = (labels == -1).sum()

        return {
            'labels': labels,
            'n_clusters': n_clusters,
            'n_noise': int(n_noise),
            'core_sample_indices': model.core_sample_indices_,
            'model': model,
            'scaler': scaler
        }

    else:
        raise ValueError(f"Unknown method: {method}. Use 'kmeans' or 'dbscan'")


def preprocess_for_causal(
    df: pd.DataFrame,
    outcome: str,
    treatment: str,
    controls: List[str],
    missing_strategy: str = 'drop',
    outlier_method: Optional[str] = None,
    outlier_threshold: float = 1.5,
    standardize_controls: bool = True,
    create_missing_indicators: bool = False
) -> Dict[str, Any]:
    """
    Complete preprocessing pipeline for causal inference.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    outcome : str
        Name of the outcome variable.
    treatment : str
        Name of the treatment variable.
    controls : List[str]
        List of control variable names.
    missing_strategy : str
        Strategy for handling missing values ('drop', 'mean', 'median', 'multiple').
    outlier_method : str, optional
        Method for outlier detection. If None, no outlier removal.
    outlier_threshold : float
        Threshold for outlier detection.
    standardize_controls : bool
        Whether to standardize control variables.
    create_missing_indicators : bool
        Whether to create binary indicators for missing values before imputation.

    Returns
    -------
    Dict[str, Any]
        Dictionary containing:
        - 'data': Preprocessed DataFrame
        - 'report': Preprocessing report with all steps
        - 'scaler': Fitted scaler if standardization applied
        - 'original_columns': Original column names

    Examples
    --------
    >>> result = preprocess_for_causal(
    ...     df=df,
    ...     outcome='Y',
    ...     treatment='D',
    ...     controls=['X1', 'X2', 'X3'],
    ...     missing_strategy='mean',
    ...     outlier_method='iqr',
    ...     standardize_controls=True
    ... )
    >>> df_processed = result['data']
    """
    report = {
        'steps': [],
        'warnings': [],
        'n_original': len(df)
    }

    # Validate inputs
    all_vars = [outcome, treatment] + controls
    missing_vars = [v for v in all_vars if v not in df.columns]
    if missing_vars:
        raise ValueError(f"Variables not found in DataFrame: {missing_vars}")

    df = df.copy()

    # Step 1: Create missing indicators if requested
    if create_missing_indicators:
        for col in controls:
            if df[col].isnull().any():
                indicator_name = f"{col}_missing"
                df[indicator_name] = df[col].isnull().astype(int)
                report['steps'].append(f"Created missing indicator: {indicator_name}")

    # Step 2: Diagnose missing values
    missing_report = diagnose_missing(df[all_vars])
    report['missing_diagnosis'] = missing_report['summary']

    # Step 3: Handle missing values
    if missing_report['summary']['total_missing'] > 0:
        # Check for missing in outcome/treatment
        if df[outcome].isnull().any():
            report['warnings'].append(f"Outcome variable '{outcome}' has missing values")
        if df[treatment].isnull().any():
            report['warnings'].append(f"Treatment variable '{treatment}' has missing values")

        if missing_strategy == 'multiple':
            report['warnings'].append("Multiple imputation returns list of DataFrames")
            imputed_dfs = handle_missing(df, strategy='multiple', columns=controls)
            # For the report, use first imputed dataset
            df = imputed_dfs[0]
            report['steps'].append(f"Applied multiple imputation (5 datasets)")
        else:
            # For outcome and treatment, always use listwise deletion
            df = df.dropna(subset=[outcome, treatment])
            # For controls, use specified strategy
            df = handle_missing(df, strategy=missing_strategy, columns=controls)
            report['steps'].append(f"Applied {missing_strategy} for missing values")

    report['n_after_missing'] = len(df)

    # Step 4: Outlier removal
    scaler = None
    if outlier_method is not None:
        # Only check controls for outliers (not outcome/treatment)
        numeric_controls = df[controls].select_dtypes(include=[np.number]).columns.tolist()
        if numeric_controls:
            df, outlier_report = remove_outliers(
                df, numeric_controls, method=outlier_method, threshold=outlier_threshold
            )
            report['outlier_removal'] = outlier_report
            report['steps'].append(
                f"Removed {outlier_report['n_removed']} outliers using {outlier_method} method"
            )

    report['n_after_outliers'] = len(df)

    # Step 5: Standardize controls
    if standardize_controls:
        numeric_controls = df[controls].select_dtypes(include=[np.number]).columns.tolist()
        if numeric_controls:
            df, scaler = standardize(df, numeric_controls)
            report['steps'].append(f"Standardized {len(numeric_controls)} control variables")

    # Final summary
    report['n_final'] = len(df)
    report['pct_retained'] = round(len(df) / report['n_original'] * 100, 2)

    # Balance check
    treated = df[df[treatment] == 1] if df[treatment].nunique() == 2 else None
    control = df[df[treatment] == 0] if df[treatment].nunique() == 2 else None

    if treated is not None and control is not None:
        report['n_treated'] = len(treated)
        report['n_control'] = len(control)

    return {
        'data': df,
        'report': report,
        'scaler': scaler,
        'original_columns': all_vars
    }


def validate_preprocessing(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    controls: List[str]
) -> Dict[str, Any]:
    """
    Validate that train and test data have consistent preprocessing.

    Parameters
    ----------
    df_train : pd.DataFrame
        Training DataFrame.
    df_test : pd.DataFrame
        Test DataFrame.
    controls : List[str]
        Control variable names to check.

    Returns
    -------
    Dict[str, Any]
        Validation report with any inconsistencies detected.
    """
    report = {
        'is_valid': True,
        'issues': []
    }

    # Check column consistency
    train_cols = set(df_train.columns)
    test_cols = set(df_test.columns)

    if train_cols != test_cols:
        report['is_valid'] = False
        missing_in_test = train_cols - test_cols
        missing_in_train = test_cols - train_cols
        if missing_in_test:
            report['issues'].append(f"Columns in train but not test: {missing_in_test}")
        if missing_in_train:
            report['issues'].append(f"Columns in test but not train: {missing_in_train}")

    # Check for distribution shift in controls
    for col in controls:
        if col in df_train.columns and col in df_test.columns:
            if df_train[col].dtype in [np.float64, np.int64]:
                train_mean = df_train[col].mean()
                test_mean = df_test[col].mean()
                train_std = df_train[col].std()

                if train_std > 0:
                    z_diff = abs(train_mean - test_mean) / train_std
                    if z_diff > 2:
                        report['issues'].append(
                            f"Large distribution shift in {col}: "
                            f"train mean={train_mean:.2f}, test mean={test_mean:.2f}"
                        )

    return report


# Convenience functions for common workflows

def quick_preprocess(
    df: pd.DataFrame,
    outcome: str,
    treatment: str,
    controls: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Quick preprocessing with sensible defaults for causal inference.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    outcome : str
        Outcome variable name.
    treatment : str
        Treatment variable name.
    controls : List[str], optional
        Control variables. If None, uses all numeric columns except outcome/treatment.

    Returns
    -------
    pd.DataFrame
        Preprocessed DataFrame ready for causal analysis.
    """
    if controls is None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        controls = [c for c in numeric_cols if c not in [outcome, treatment]]

    result = preprocess_for_causal(
        df=df,
        outcome=outcome,
        treatment=treatment,
        controls=controls,
        missing_strategy='drop',
        outlier_method=None,
        standardize_controls=True
    )

    return result['data']
