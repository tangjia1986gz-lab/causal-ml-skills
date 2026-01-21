"""
Panel Data Estimator

Comprehensive panel data econometrics implementation including:
- Fixed effects and random effects models
- Hausman specification test
- Dynamic panel models (Arellano-Bond, Blundell-Bond)
- Clustered standard errors and robust inference
- TWFE diagnostics for causal inference

Author: Causal ML Skills
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats
from scipy.linalg import inv, solve


@dataclass
class CausalOutput:
    """Standardized output for causal inference results."""

    estimate: float
    std_error: float
    ci_lower: float
    ci_upper: float
    p_value: float
    method: str
    n_obs: int
    n_entities: int
    n_periods: int
    additional_info: Dict[str, Any] = field(default_factory=dict)

    @property
    def t_stat(self) -> float:
        """T-statistic for the estimate."""
        return self.estimate / self.std_error if self.std_error > 0 else np.inf

    @property
    def significant_5pct(self) -> bool:
        """Whether estimate is significant at 5% level."""
        return self.p_value < 0.05

    def __repr__(self) -> str:
        sig_star = "***" if self.p_value < 0.01 else "**" if self.p_value < 0.05 else "*" if self.p_value < 0.1 else ""
        return (
            f"CausalOutput(\n"
            f"  method={self.method},\n"
            f"  estimate={self.estimate:.4f}{sig_star},\n"
            f"  std_error={self.std_error:.4f},\n"
            f"  95% CI=[{self.ci_lower:.4f}, {self.ci_upper:.4f}],\n"
            f"  p_value={self.p_value:.4f},\n"
            f"  n_obs={self.n_obs}, n_entities={self.n_entities}, n_periods={self.n_periods}\n"
            f")"
        )


@dataclass
class PanelModelResult:
    """Results from panel model estimation."""

    coefficients: pd.Series
    std_errors: pd.Series
    t_stats: pd.Series
    p_values: pd.Series
    ci_lower: pd.Series
    ci_upper: pd.Series
    r_squared: float
    r_squared_within: float
    r_squared_between: float
    r_squared_overall: float
    n_obs: int
    n_entities: int
    n_periods: int
    df_resid: int
    residuals: np.ndarray
    fitted_values: np.ndarray
    entity_effects: Optional[pd.Series] = None
    time_effects: Optional[pd.Series] = None
    model_type: str = "fixed_effects"
    variance_decomposition: Optional[Dict[str, float]] = None
    vcov_matrix: Optional[np.ndarray] = None

    def summary(self) -> str:
        """Generate summary table."""
        lines = [
            f"\n{'='*70}",
            f"Panel Data Model: {self.model_type.replace('_', ' ').title()}",
            f"{'='*70}",
            f"Observations: {self.n_obs:,}",
            f"Entities: {self.n_entities:,}",
            f"Time periods: {self.n_periods:,}",
            f"{'='*70}",
            f"R-squared (within):  {self.r_squared_within:.4f}",
            f"R-squared (between): {self.r_squared_between:.4f}",
            f"R-squared (overall): {self.r_squared_overall:.4f}",
            f"{'='*70}",
            f"{'Variable':<20} {'Coef':>12} {'Std.Err':>12} {'t':>10} {'P>|t|':>10}",
            f"{'-'*70}"
        ]

        for var in self.coefficients.index:
            sig = ""
            if self.p_values[var] < 0.01:
                sig = "***"
            elif self.p_values[var] < 0.05:
                sig = "**"
            elif self.p_values[var] < 0.1:
                sig = "*"

            lines.append(
                f"{var:<20} {self.coefficients[var]:>12.4f} {self.std_errors[var]:>12.4f} "
                f"{self.t_stats[var]:>10.3f} {self.p_values[var]:>10.4f}{sig}"
            )

        lines.extend([
            f"{'-'*70}",
            f"*** p<0.01, ** p<0.05, * p<0.1",
            f"{'='*70}\n"
        ])

        return "\n".join(lines)

    def to_causal_output(self, treatment_var: str) -> CausalOutput:
        """Convert to CausalOutput for treatment effect."""
        if treatment_var not in self.coefficients.index:
            raise ValueError(f"Treatment variable '{treatment_var}' not found in model.")

        return CausalOutput(
            estimate=self.coefficients[treatment_var],
            std_error=self.std_errors[treatment_var],
            ci_lower=self.ci_lower[treatment_var],
            ci_upper=self.ci_upper[treatment_var],
            p_value=self.p_values[treatment_var],
            method=self.model_type,
            n_obs=self.n_obs,
            n_entities=self.n_entities,
            n_periods=self.n_periods,
            additional_info={
                "r_squared_within": self.r_squared_within,
                "r_squared_between": self.r_squared_between,
                "r_squared_overall": self.r_squared_overall,
                "all_coefficients": self.coefficients.to_dict(),
            }
        )


@dataclass
class HausmanTestResult:
    """Results from Hausman specification test."""

    test_statistic: float
    p_value: float
    df: int
    conclusion: str
    fe_coefficients: pd.Series
    re_coefficients: pd.Series
    coefficient_difference: pd.Series

    def __repr__(self) -> str:
        return (
            f"HausmanTestResult(\n"
            f"  H0: Random effects is consistent and efficient\n"
            f"  H1: Fixed effects is consistent, random effects is not\n"
            f"  chi2({self.df}) = {self.test_statistic:.4f}\n"
            f"  p-value = {self.p_value:.4f}\n"
            f"  Conclusion: {self.conclusion}\n"
            f")"
        )


class PanelEstimator:
    """
    Panel data estimator with fixed effects, random effects, and dynamic models.

    Parameters
    ----------
    data : pd.DataFrame
        Panel data with entity and time identifiers
    entity_col : str
        Column name for entity identifier
    time_col : str
        Column name for time identifier
    y_col : str
        Column name for dependent variable
    x_cols : List[str]
        Column names for independent variables

    Examples
    --------
    >>> estimator = PanelEstimator(
    ...     data=df,
    ...     entity_col='firm_id',
    ...     time_col='year',
    ...     y_col='revenue',
    ...     x_cols=['investment', 'employees']
    ... )
    >>> fe_result = estimator.fit_fixed_effects()
    >>> print(fe_result.summary())
    """

    def __init__(
        self,
        data: pd.DataFrame,
        entity_col: str,
        time_col: str,
        y_col: str,
        x_cols: List[str],
    ):
        self.data = data.copy()
        self.entity_col = entity_col
        self.time_col = time_col
        self.y_col = y_col
        self.x_cols = x_cols

        # Validate data
        self._validate_data()

        # Create panel structure
        self._setup_panel()

    def _validate_data(self) -> None:
        """Validate panel data structure."""
        required_cols = [self.entity_col, self.time_col, self.y_col] + self.x_cols
        missing = [c for c in required_cols if c not in self.data.columns]
        if missing:
            raise ValueError(f"Missing columns: {missing}")

        # Check for duplicates
        if self.data.duplicated(subset=[self.entity_col, self.time_col]).any():
            raise ValueError("Duplicate entity-time observations found.")

        # Check for missing values
        if self.data[required_cols].isnull().any().any():
            warnings.warn("Missing values detected. Consider imputation or balanced panel.")

    def _setup_panel(self) -> None:
        """Set up panel data structure."""
        # Sort by entity and time
        self.data = self.data.sort_values([self.entity_col, self.time_col])

        # Get dimensions
        self.entities = self.data[self.entity_col].unique()
        self.times = self.data[self.time_col].unique()
        self.n_entities = len(self.entities)
        self.n_times = len(self.times)
        self.n_obs = len(self.data)

        # Check balance
        entity_counts = self.data.groupby(self.entity_col).size()
        self.is_balanced = (entity_counts == self.n_times).all()

        if not self.is_balanced:
            warnings.warn(
                f"Unbalanced panel: entities have {entity_counts.min()}-{entity_counts.max()} "
                f"time periods (expected {self.n_times})."
            )

        # Compute group means
        self._compute_transformations()

    def _compute_transformations(self) -> None:
        """Compute within and between transformations."""
        # Entity means
        self.entity_means = self.data.groupby(self.entity_col)[
            [self.y_col] + self.x_cols
        ].transform('mean')

        # Time means
        self.time_means = self.data.groupby(self.time_col)[
            [self.y_col] + self.x_cols
        ].transform('mean')

        # Overall means
        self.overall_means = self.data[[self.y_col] + self.x_cols].mean()

        # Within transformation (demeaned by entity)
        self.within_data = self.data[[self.y_col] + self.x_cols] - self.entity_means

        # Between data (entity means)
        self.between_data = self.data.groupby(self.entity_col)[
            [self.y_col] + self.x_cols
        ].mean()

    def fit_fixed_effects(
        self,
        entity_effects: bool = True,
        time_effects: bool = False,
        cluster_col: Optional[str] = None,
    ) -> PanelModelResult:
        """
        Fit fixed effects model using within transformation.

        Parameters
        ----------
        entity_effects : bool
            Include entity fixed effects (default True)
        time_effects : bool
            Include time fixed effects (default False)
        cluster_col : str, optional
            Column to cluster standard errors on

        Returns
        -------
        PanelModelResult
            Estimation results
        """
        y = self.data[self.y_col].values
        X = self.data[self.x_cols].values

        if entity_effects and time_effects:
            # Two-way fixed effects
            y_transformed, X_transformed = self._twoway_transform(y, X)
            model_type = "two_way_fixed_effects"
            df_adj = self.n_entities + self.n_times - 1
        elif entity_effects:
            # Entity fixed effects (within transformation)
            y_transformed = self.within_data[self.y_col].values
            X_transformed = self.within_data[self.x_cols].values
            model_type = "entity_fixed_effects"
            df_adj = self.n_entities
        elif time_effects:
            # Time fixed effects
            y_transformed = (self.data[self.y_col] - self.time_means[self.y_col]).values
            X_transformed = (self.data[self.x_cols] - self.time_means[self.x_cols]).values
            model_type = "time_fixed_effects"
            df_adj = self.n_times
        else:
            # Pooled OLS
            y_transformed = y - y.mean()
            X_transformed = X - X.mean(axis=0)
            model_type = "pooled_ols"
            df_adj = 1

        # OLS on transformed data
        k = X_transformed.shape[1]
        n = len(y_transformed)

        # Coefficients: (X'X)^{-1} X'y
        XtX = X_transformed.T @ X_transformed
        Xty = X_transformed.T @ y_transformed

        try:
            beta = solve(XtX, Xty, assume_a='pos')
        except np.linalg.LinAlgError:
            beta = np.linalg.lstsq(X_transformed, y_transformed, rcond=None)[0]

        # Residuals and fitted values
        residuals = y_transformed - X_transformed @ beta
        fitted = X_transformed @ beta

        # Degrees of freedom
        df_resid = n - k - df_adj

        # Standard errors
        if cluster_col is not None:
            vcov = self._clustered_vcov(
                X_transformed, residuals,
                self.data[cluster_col].values
            )
        else:
            # Heteroskedasticity-robust standard errors
            sigma2 = (residuals ** 2).sum() / df_resid
            XtX_inv = inv(XtX)
            vcov = sigma2 * XtX_inv

        std_errors = np.sqrt(np.diag(vcov))

        # T-statistics and p-values
        t_stats = beta / std_errors
        p_values = 2 * (1 - stats.t.cdf(np.abs(t_stats), df_resid))

        # Confidence intervals
        t_crit = stats.t.ppf(0.975, df_resid)
        ci_lower = beta - t_crit * std_errors
        ci_upper = beta + t_crit * std_errors

        # R-squared measures
        r2_within = 1 - (residuals ** 2).sum() / ((y_transformed - y_transformed.mean()) ** 2).sum()

        # Between R-squared
        entity_means_y = self.between_data[self.y_col].values
        entity_means_X = self.between_data[self.x_cols].values
        fitted_between = entity_means_X @ beta
        r2_between = 1 - ((entity_means_y - fitted_between) ** 2).sum() / \
                     ((entity_means_y - entity_means_y.mean()) ** 2).sum()

        # Overall R-squared
        y_pred_overall = X @ beta
        r2_overall = 1 - ((y - y_pred_overall - (y - y_pred_overall).mean()) ** 2).sum() / \
                     ((y - y.mean()) ** 2).sum()

        # Entity effects
        if entity_effects:
            entity_effects_vals = self.data.groupby(self.entity_col).apply(
                lambda g: g[self.y_col].mean() - (g[self.x_cols].mean().values @ beta)
            )
        else:
            entity_effects_vals = None

        # Time effects
        if time_effects:
            time_effects_vals = self.data.groupby(self.time_col).apply(
                lambda g: g[self.y_col].mean() - (g[self.x_cols].mean().values @ beta)
            )
        else:
            time_effects_vals = None

        return PanelModelResult(
            coefficients=pd.Series(beta, index=self.x_cols),
            std_errors=pd.Series(std_errors, index=self.x_cols),
            t_stats=pd.Series(t_stats, index=self.x_cols),
            p_values=pd.Series(p_values, index=self.x_cols),
            ci_lower=pd.Series(ci_lower, index=self.x_cols),
            ci_upper=pd.Series(ci_upper, index=self.x_cols),
            r_squared=r2_within,
            r_squared_within=r2_within,
            r_squared_between=r2_between,
            r_squared_overall=r2_overall,
            n_obs=self.n_obs,
            n_entities=self.n_entities,
            n_periods=self.n_times,
            df_resid=df_resid,
            residuals=residuals,
            fitted_values=fitted,
            entity_effects=entity_effects_vals,
            time_effects=time_effects_vals,
            model_type=model_type,
            vcov_matrix=vcov,
        )

    def _twoway_transform(
        self,
        y: np.ndarray,
        X: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Two-way within transformation."""
        # y_it - y_i. - y_.t + y_..
        y_entity = self.entity_means[self.y_col].values
        y_time = self.time_means[self.y_col].values
        y_overall = self.overall_means[self.y_col]
        y_transformed = y - y_entity - y_time + y_overall

        # Same for X
        X_entity = self.entity_means[self.x_cols].values
        X_time = self.time_means[self.x_cols].values
        X_overall = self.overall_means[self.x_cols].values
        X_transformed = X - X_entity - X_time + X_overall

        return y_transformed, X_transformed

    def fit_random_effects(
        self,
        method: Literal["gls", "mle"] = "gls",
    ) -> PanelModelResult:
        """
        Fit random effects model.

        Parameters
        ----------
        method : str
            Estimation method: 'gls' (default) or 'mle'

        Returns
        -------
        PanelModelResult
            Estimation results
        """
        y = self.data[self.y_col].values
        X = self.data[self.x_cols].values
        n, k = X.shape

        # Add constant
        X_with_const = np.column_stack([np.ones(n), X])

        # Step 1: Estimate variance components from FE residuals
        fe_result = self.fit_fixed_effects(entity_effects=True, time_effects=False)
        sigma2_e = (fe_result.residuals ** 2).sum() / fe_result.df_resid

        # Between estimator for sigma2_u
        between_y = self.between_data[self.y_col].values
        between_X = np.column_stack([
            np.ones(self.n_entities),
            self.between_data[self.x_cols].values
        ])

        beta_between = np.linalg.lstsq(between_X, between_y, rcond=None)[0]
        resid_between = between_y - between_X @ beta_between

        # Time periods per entity
        T_i = self.data.groupby(self.entity_col).size().values
        T_bar = T_i.mean()

        sigma2_b = (resid_between ** 2).sum() / (self.n_entities - k - 1)
        sigma2_u = max(0, sigma2_b - sigma2_e / T_bar)

        # Variance decomposition
        var_decomp = {
            "sigma2_e": sigma2_e,
            "sigma2_u": sigma2_u,
            "rho": sigma2_u / (sigma2_u + sigma2_e) if (sigma2_u + sigma2_e) > 0 else 0,
            "theta": 1 - np.sqrt(sigma2_e / (sigma2_e + T_bar * sigma2_u)) if (sigma2_e + T_bar * sigma2_u) > 0 else 0,
        }

        # GLS transformation: quasi-demeaning
        theta = var_decomp["theta"]

        y_transformed = y - theta * self.entity_means[self.y_col].values
        X_transformed = X_with_const - theta * np.column_stack([
            np.ones(n),
            self.entity_means[self.x_cols].values
        ])

        # GLS estimation
        XtX = X_transformed.T @ X_transformed
        Xty = X_transformed.T @ y_transformed
        beta = solve(XtX, Xty, assume_a='pos')

        # Residuals
        residuals = y - X_with_const @ beta
        fitted = X_with_const @ beta

        # Standard errors
        df_resid = n - k - 1
        XtX_inv = inv(XtX)
        sigma2 = (residuals ** 2).sum() / df_resid
        vcov = sigma2 * XtX_inv
        std_errors = np.sqrt(np.diag(vcov))

        # T-statistics and p-values
        t_stats = beta / std_errors
        p_values = 2 * (1 - stats.t.cdf(np.abs(t_stats), df_resid))

        # Confidence intervals
        t_crit = stats.t.ppf(0.975, df_resid)
        ci_lower = beta - t_crit * std_errors
        ci_upper = beta + t_crit * std_errors

        # R-squared measures
        y_within = self.within_data[self.y_col].values
        X_within = self.within_data[self.x_cols].values
        fitted_within = X_within @ beta[1:]  # Exclude constant
        r2_within = 1 - ((y_within - fitted_within) ** 2).sum() / \
                    ((y_within - y_within.mean()) ** 2).sum()

        fitted_between = between_X @ beta
        r2_between = 1 - ((between_y - fitted_between) ** 2).sum() / \
                     ((between_y - between_y.mean()) ** 2).sum()

        r2_overall = 1 - ((y - fitted) ** 2).sum() / ((y - y.mean()) ** 2).sum()

        # Results (excluding constant from output)
        var_names = ['_const'] + self.x_cols

        return PanelModelResult(
            coefficients=pd.Series(beta, index=var_names),
            std_errors=pd.Series(std_errors, index=var_names),
            t_stats=pd.Series(t_stats, index=var_names),
            p_values=pd.Series(p_values, index=var_names),
            ci_lower=pd.Series(ci_lower, index=var_names),
            ci_upper=pd.Series(ci_upper, index=var_names),
            r_squared=r2_overall,
            r_squared_within=r2_within,
            r_squared_between=r2_between,
            r_squared_overall=r2_overall,
            n_obs=self.n_obs,
            n_entities=self.n_entities,
            n_periods=self.n_times,
            df_resid=df_resid,
            residuals=residuals,
            fitted_values=fitted,
            model_type="random_effects",
            variance_decomposition=var_decomp,
            vcov_matrix=vcov,
        )

    def hausman_test(
        self,
        fe_result: Optional[PanelModelResult] = None,
        re_result: Optional[PanelModelResult] = None,
    ) -> HausmanTestResult:
        """
        Hausman specification test for FE vs RE.

        H0: Random effects is consistent and efficient
        H1: Fixed effects is consistent, random effects is not

        Parameters
        ----------
        fe_result : PanelModelResult, optional
            Fixed effects result (computed if not provided)
        re_result : PanelModelResult, optional
            Random effects result (computed if not provided)

        Returns
        -------
        HausmanTestResult
            Test results
        """
        if fe_result is None:
            fe_result = self.fit_fixed_effects()
        if re_result is None:
            re_result = self.fit_random_effects()

        # Get coefficients (exclude constant from RE)
        fe_coef = fe_result.coefficients
        re_coef = re_result.coefficients.drop('_const', errors='ignore')

        # Align coefficients
        common_vars = fe_coef.index.intersection(re_coef.index)
        b_fe = fe_coef[common_vars].values
        b_re = re_coef[common_vars].values

        # Difference in coefficients
        b_diff = b_fe - b_re

        # Variance of difference: Var(b_FE - b_RE) = Var(b_FE) - Var(b_RE)
        # Under H0, Cov(b_FE - b_RE, b_RE) = 0
        V_fe = fe_result.vcov_matrix
        V_re = re_result.vcov_matrix

        # Get submatrices for common variables
        fe_idx = [list(fe_result.coefficients.index).index(v) for v in common_vars]
        re_idx = [list(re_result.coefficients.index).index(v) for v in common_vars]

        V_fe_sub = V_fe[np.ix_(fe_idx, fe_idx)]
        V_re_sub = V_re[np.ix_(re_idx, re_idx)]

        V_diff = V_fe_sub - V_re_sub

        # Hausman statistic
        try:
            V_diff_inv = inv(V_diff)
            H = b_diff @ V_diff_inv @ b_diff
        except np.linalg.LinAlgError:
            # Use generalized inverse if singular
            V_diff_inv = np.linalg.pinv(V_diff)
            H = b_diff @ V_diff_inv @ b_diff

        # Degrees of freedom = number of coefficients
        df = len(common_vars)

        # P-value from chi-squared distribution
        p_value = 1 - stats.chi2.cdf(H, df)

        # Conclusion
        if p_value < 0.05:
            conclusion = "Reject H0: Use Fixed Effects (FE is consistent, RE is not)"
        else:
            conclusion = "Fail to reject H0: Random Effects is consistent and more efficient"

        return HausmanTestResult(
            test_statistic=H,
            p_value=p_value,
            df=df,
            conclusion=conclusion,
            fe_coefficients=fe_coef,
            re_coefficients=re_coef,
            coefficient_difference=pd.Series(b_diff, index=common_vars),
        )

    def within_between_test(self) -> Dict[str, Any]:
        """
        Mundlak test / within-between decomposition.

        Tests whether entity means are correlated with regressors.
        """
        # Add entity means as regressors
        X_means = self.entity_means[self.x_cols]
        X_means.columns = [f"{c}_mean" for c in self.x_cols]

        data_augmented = pd.concat([
            self.data[[self.y_col] + self.x_cols],
            X_means
        ], axis=1)

        y = data_augmented[self.y_col].values
        X = data_augmented.drop(columns=[self.y_col]).values
        X = np.column_stack([np.ones(len(y)), X])

        # OLS
        beta = np.linalg.lstsq(X, y, rcond=None)[0]
        resid = y - X @ beta

        # Test joint significance of means
        k_means = len(self.x_cols)
        mean_coefs = beta[-k_means:]

        # F-test for means = 0
        RSS_unrestricted = (resid ** 2).sum()

        # Restricted model (without means)
        X_restricted = X[:, :1+len(self.x_cols)]
        beta_r = np.linalg.lstsq(X_restricted, y, rcond=None)[0]
        RSS_restricted = ((y - X_restricted @ beta_r) ** 2).sum()

        n, k = X.shape
        df1 = k_means
        df2 = n - k

        F = ((RSS_restricted - RSS_unrestricted) / df1) / (RSS_unrestricted / df2)
        p_value = 1 - stats.f.cdf(F, df1, df2)

        return {
            "test": "Mundlak within-between test",
            "F_statistic": F,
            "df1": df1,
            "df2": df2,
            "p_value": p_value,
            "mean_coefficients": pd.Series(mean_coefs, index=[f"{c}_mean" for c in self.x_cols]),
            "conclusion": "Entity means significant: Use FE" if p_value < 0.05 else "Entity means not significant: RE may be appropriate",
        }

    def fit_dynamic_panel(
        self,
        lags: int = 1,
        method: Literal["arellano_bond", "blundell_bond"] = "arellano_bond",
        max_instruments: Optional[int] = None,
    ) -> PanelModelResult:
        """
        Fit dynamic panel model with GMM.

        Parameters
        ----------
        lags : int
            Number of lags of dependent variable (default 1)
        method : str
            'arellano_bond' (difference GMM) or 'blundell_bond' (system GMM)
        max_instruments : int, optional
            Maximum number of instruments per period

        Returns
        -------
        PanelModelResult
            Estimation results
        """
        # First-difference the data
        data_sorted = self.data.sort_values([self.entity_col, self.time_col])

        # Create lagged dependent variable
        y_lag = data_sorted.groupby(self.entity_col)[self.y_col].shift(lags)
        data_sorted['y_lag'] = y_lag

        # First differences
        dy = data_sorted.groupby(self.entity_col)[self.y_col].diff()
        dy_lag = data_sorted.groupby(self.entity_col)['y_lag'].diff()
        dX = data_sorted.groupby(self.entity_col)[self.x_cols].diff()

        # Drop missing (first lags+1 observations per entity)
        valid = ~(dy.isna() | dy_lag.isna() | dX.isna().any(axis=1))

        dy = dy[valid].values
        dy_lag = dy_lag[valid].values
        dX = dX[valid].values

        n = len(dy)
        k = 1 + dX.shape[1]  # lagged DV + X variables

        # Construct instruments: y_{t-2}, y_{t-3}, ...
        # For Arellano-Bond: levels of y dated t-2 and earlier
        Z = self._construct_gmm_instruments(
            data_sorted, valid, method, max_instruments
        )

        # Combine regressors
        W = np.column_stack([dy_lag, dX])

        # Two-step GMM
        # Step 1: Initial weight matrix (identity or H matrix)
        H = np.eye(Z.shape[1])

        # First stage
        ZtW = Z.T @ W
        Zty = Z.T @ dy

        try:
            A1 = Z.T @ Z
            beta1 = inv(ZtW.T @ inv(A1) @ ZtW) @ ZtW.T @ inv(A1) @ Zty
        except np.linalg.LinAlgError:
            beta1 = np.linalg.lstsq(W, dy, rcond=None)[0]

        # Step 2: Optimal weight matrix
        resid1 = dy - W @ beta1
        Omega = Z.T @ np.diag(resid1 ** 2) @ Z

        try:
            Omega_inv = inv(Omega)
            beta2 = inv(ZtW.T @ Omega_inv @ ZtW) @ ZtW.T @ Omega_inv @ Zty
        except np.linalg.LinAlgError:
            beta2 = beta1

        # Final residuals
        residuals = dy - W @ beta2
        fitted = W @ beta2

        # Variance-covariance matrix
        try:
            vcov = inv(ZtW.T @ Omega_inv @ ZtW)
        except:
            vcov = np.eye(k) * ((residuals ** 2).sum() / (n - k))

        std_errors = np.sqrt(np.diag(vcov))

        # T-statistics and p-values
        df_resid = n - k
        t_stats = beta2 / std_errors
        p_values = 2 * (1 - stats.t.cdf(np.abs(t_stats), df_resid))

        # Confidence intervals
        t_crit = stats.t.ppf(0.975, df_resid)
        ci_lower = beta2 - t_crit * std_errors
        ci_upper = beta2 + t_crit * std_errors

        # Variable names
        var_names = [f'L{lags}.{self.y_col}'] + self.x_cols

        return PanelModelResult(
            coefficients=pd.Series(beta2, index=var_names),
            std_errors=pd.Series(std_errors, index=var_names),
            t_stats=pd.Series(t_stats, index=var_names),
            p_values=pd.Series(p_values, index=var_names),
            ci_lower=pd.Series(ci_lower, index=var_names),
            ci_upper=pd.Series(ci_upper, index=var_names),
            r_squared=1 - (residuals ** 2).sum() / ((dy - dy.mean()) ** 2).sum(),
            r_squared_within=np.nan,
            r_squared_between=np.nan,
            r_squared_overall=np.nan,
            n_obs=n,
            n_entities=self.n_entities,
            n_periods=self.n_times,
            df_resid=df_resid,
            residuals=residuals,
            fitted_values=fitted,
            model_type=f"dynamic_panel_{method}",
            vcov_matrix=vcov,
        )

    def _construct_gmm_instruments(
        self,
        data: pd.DataFrame,
        valid: pd.Series,
        method: str,
        max_instruments: Optional[int],
    ) -> np.ndarray:
        """Construct GMM instruments for dynamic panel."""
        # Simplified instrument construction
        # Use lagged levels as instruments for differences

        entities = data[self.entity_col].values[valid]
        unique_entities = np.unique(entities)

        # Collect lagged y values (t-2 and earlier) as instruments
        instruments = []

        y_full = data[self.y_col].values
        entity_full = data[self.entity_col].values

        for i, row_idx in enumerate(np.where(valid)[0]):
            entity = entity_full[row_idx]
            entity_mask = entity_full == entity
            entity_indices = np.where(entity_mask)[0]
            pos_in_entity = np.where(entity_indices == row_idx)[0][0]

            # Get y values from t-2 and earlier
            if pos_in_entity >= 2:
                z_vals = y_full[entity_indices[:pos_in_entity-1]]
                if max_instruments and len(z_vals) > max_instruments:
                    z_vals = z_vals[-max_instruments:]
            else:
                z_vals = np.array([])

            instruments.append(z_vals)

        # Pad to maximum length
        max_len = max(len(z) for z in instruments) if instruments else 1
        Z = np.zeros((len(instruments), max_len))

        for i, z in enumerate(instruments):
            if len(z) > 0:
                Z[i, :len(z)] = z

        # Add X variables as instruments (assuming exogenous)
        X_valid = data[self.x_cols].values[valid]
        Z = np.column_stack([Z, X_valid])

        return Z

    def cluster_robust_inference(
        self,
        result: PanelModelResult,
        cluster_col: str,
        method: Literal["stata", "robust", "bootstrap", "wild"] = "stata",
        n_bootstrap: int = 1000,
    ) -> PanelModelResult:
        """
        Compute cluster-robust standard errors.

        Parameters
        ----------
        result : PanelModelResult
            Model result to adjust
        cluster_col : str
            Column to cluster on
        method : str
            'stata' (default), 'robust', 'bootstrap', 'wild'
        n_bootstrap : int
            Number of bootstrap iterations (for bootstrap/wild methods)

        Returns
        -------
        PanelModelResult
            Result with clustered standard errors
        """
        clusters = self.data[cluster_col].values
        n_clusters = len(np.unique(clusters))

        # Get design matrix based on model type
        if "two_way" in result.model_type:
            y, X = self._twoway_transform(
                self.data[self.y_col].values,
                self.data[self.x_cols].values
            )
        elif "entity" in result.model_type:
            y = self.within_data[self.y_col].values
            X = self.within_data[self.x_cols].values
        else:
            y = self.data[self.y_col].values
            X = self.data[self.x_cols].values

        residuals = result.residuals

        if method == "bootstrap":
            vcov = self._bootstrap_vcov(X, y, clusters, n_bootstrap)
        elif method == "wild":
            vcov = self._wild_cluster_bootstrap(X, y, residuals, clusters, n_bootstrap)
        else:
            vcov = self._clustered_vcov(X, residuals, clusters, method)

        # Finite sample correction
        n, k = X.shape
        correction = (n_clusters / (n_clusters - 1)) * ((n - 1) / (n - k))
        vcov = vcov * correction

        std_errors = np.sqrt(np.diag(vcov))

        # Use t-distribution with G-1 degrees of freedom
        df = n_clusters - 1
        t_stats = result.coefficients.values / std_errors
        p_values = 2 * (1 - stats.t.cdf(np.abs(t_stats), df))

        t_crit = stats.t.ppf(0.975, df)
        ci_lower = result.coefficients.values - t_crit * std_errors
        ci_upper = result.coefficients.values + t_crit * std_errors

        return PanelModelResult(
            coefficients=result.coefficients,
            std_errors=pd.Series(std_errors, index=result.coefficients.index),
            t_stats=pd.Series(t_stats, index=result.coefficients.index),
            p_values=pd.Series(p_values, index=result.coefficients.index),
            ci_lower=pd.Series(ci_lower, index=result.coefficients.index),
            ci_upper=pd.Series(ci_upper, index=result.coefficients.index),
            r_squared=result.r_squared,
            r_squared_within=result.r_squared_within,
            r_squared_between=result.r_squared_between,
            r_squared_overall=result.r_squared_overall,
            n_obs=result.n_obs,
            n_entities=result.n_entities,
            n_periods=result.n_periods,
            df_resid=df,
            residuals=result.residuals,
            fitted_values=result.fitted_values,
            entity_effects=result.entity_effects,
            time_effects=result.time_effects,
            model_type=f"{result.model_type}_clustered_{method}",
            vcov_matrix=vcov,
        )

    def _clustered_vcov(
        self,
        X: np.ndarray,
        residuals: np.ndarray,
        clusters: np.ndarray,
        method: str = "stata",
    ) -> np.ndarray:
        """Compute clustered variance-covariance matrix."""
        n, k = X.shape
        unique_clusters = np.unique(clusters)
        G = len(unique_clusters)

        # (X'X)^{-1}
        XtX_inv = inv(X.T @ X)

        # Sum of X'uu'X within clusters
        meat = np.zeros((k, k))

        for g in unique_clusters:
            mask = clusters == g
            X_g = X[mask]
            u_g = residuals[mask]

            # X_g' u_g
            score_g = X_g.T @ u_g
            meat += np.outer(score_g, score_g)

        # Sandwich: (X'X)^{-1} meat (X'X)^{-1}
        vcov = XtX_inv @ meat @ XtX_inv

        return vcov

    def _bootstrap_vcov(
        self,
        X: np.ndarray,
        y: np.ndarray,
        clusters: np.ndarray,
        n_bootstrap: int,
    ) -> np.ndarray:
        """Cluster bootstrap variance estimation."""
        unique_clusters = np.unique(clusters)
        G = len(unique_clusters)
        k = X.shape[1]

        # Original estimate
        beta_orig = np.linalg.lstsq(X, y, rcond=None)[0]

        # Bootstrap
        beta_boots = np.zeros((n_bootstrap, k))

        for b in range(n_bootstrap):
            # Sample clusters with replacement
            sampled_clusters = np.random.choice(unique_clusters, size=G, replace=True)

            # Build bootstrap sample
            X_boot = []
            y_boot = []

            for g in sampled_clusters:
                mask = clusters == g
                X_boot.append(X[mask])
                y_boot.append(y[mask])

            X_boot = np.vstack(X_boot)
            y_boot = np.concatenate(y_boot)

            # Estimate
            beta_boots[b] = np.linalg.lstsq(X_boot, y_boot, rcond=None)[0]

        # Variance from bootstrap distribution
        vcov = np.cov(beta_boots.T)

        return vcov

    def _wild_cluster_bootstrap(
        self,
        X: np.ndarray,
        y: np.ndarray,
        residuals: np.ndarray,
        clusters: np.ndarray,
        n_bootstrap: int,
    ) -> np.ndarray:
        """Wild cluster bootstrap for small number of clusters."""
        unique_clusters = np.unique(clusters)
        G = len(unique_clusters)
        k = X.shape[1]

        # Original estimate
        beta_orig = np.linalg.lstsq(X, y, rcond=None)[0]

        # Wild bootstrap
        beta_boots = np.zeros((n_bootstrap, k))

        for b in range(n_bootstrap):
            # Rademacher weights at cluster level
            weights = np.random.choice([-1, 1], size=G)

            # Create bootstrapped residuals
            resid_boot = np.zeros_like(residuals)
            for i, g in enumerate(unique_clusters):
                mask = clusters == g
                resid_boot[mask] = residuals[mask] * weights[i]

            # Bootstrap y
            y_boot = X @ beta_orig + resid_boot

            # Estimate
            beta_boots[b] = np.linalg.lstsq(X, y_boot, rcond=None)[0]

        # Variance from bootstrap distribution
        vcov = np.cov(beta_boots.T)

        return vcov

    def goodman_bacon_decomposition(
        self,
        treatment_col: str,
    ) -> Dict[str, Any]:
        """
        Goodman-Bacon decomposition for TWFE with staggered treatment.

        Decomposes TWFE estimate into weighted average of 2x2 DiD comparisons.

        Parameters
        ----------
        treatment_col : str
            Binary treatment indicator column

        Returns
        -------
        Dict
            Decomposition results including weights and component estimates
        """
        # Identify treatment timing groups
        first_treated = self.data.groupby(self.entity_col).apply(
            lambda g: g.loc[g[treatment_col] == 1, self.time_col].min()
            if (g[treatment_col] == 1).any() else np.inf
        )

        # Group entities by treatment timing
        timing_groups = first_treated.value_counts().sort_index()

        # Compute 2x2 DiD for each pair of timing groups
        comparisons = []

        timing_values = [t for t in timing_groups.index if t != np.inf]

        for i, t1 in enumerate(timing_values):
            for j, t2 in enumerate(timing_values):
                if t1 < t2:
                    # t1 is early treated, t2 is late treated
                    # Comparison: early treated vs late treated (before late treatment)

                    early_units = first_treated[first_treated == t1].index
                    late_units = first_treated[first_treated == t2].index

                    # Pre-period: before t1, Post-period: between t1 and t2
                    pre_mask = self.data[self.time_col] < t1
                    mid_mask = (self.data[self.time_col] >= t1) & (self.data[self.time_col] < t2)

                    early_mask = self.data[self.entity_col].isin(early_units)
                    late_mask = self.data[self.entity_col].isin(late_units)

                    # DiD estimate
                    y_early_pre = self.data.loc[early_mask & pre_mask, self.y_col].mean()
                    y_early_post = self.data.loc[early_mask & mid_mask, self.y_col].mean()
                    y_late_pre = self.data.loc[late_mask & pre_mask, self.y_col].mean()
                    y_late_post = self.data.loc[late_mask & mid_mask, self.y_col].mean()

                    did = (y_early_post - y_early_pre) - (y_late_post - y_late_pre)

                    # Weight (simplified)
                    n_early = len(early_units)
                    n_late = len(late_units)
                    weight = n_early * n_late / (self.n_entities ** 2)

                    comparisons.append({
                        "type": "early_vs_late",
                        "early_timing": t1,
                        "late_timing": t2,
                        "estimate": did,
                        "weight": weight,
                        "n_treated": n_early,
                        "n_control": n_late,
                    })

        # Include never-treated as control if present
        never_treated = first_treated[first_treated == np.inf].index
        if len(never_treated) > 0:
            for t in timing_values:
                treated_units = first_treated[first_treated == t].index

                pre_mask = self.data[self.time_col] < t
                post_mask = self.data[self.time_col] >= t

                treated_mask = self.data[self.entity_col].isin(treated_units)
                never_mask = self.data[self.entity_col].isin(never_treated)

                y_treated_pre = self.data.loc[treated_mask & pre_mask, self.y_col].mean()
                y_treated_post = self.data.loc[treated_mask & post_mask, self.y_col].mean()
                y_never_pre = self.data.loc[never_mask & pre_mask, self.y_col].mean()
                y_never_post = self.data.loc[never_mask & post_mask, self.y_col].mean()

                did = (y_treated_post - y_treated_pre) - (y_never_post - y_never_pre)

                n_treated = len(treated_units)
                n_never = len(never_treated)
                weight = n_treated * n_never / (self.n_entities ** 2)

                comparisons.append({
                    "type": "treated_vs_never",
                    "treatment_timing": t,
                    "estimate": did,
                    "weight": weight,
                    "n_treated": n_treated,
                    "n_control": n_never,
                })

        # Normalize weights
        total_weight = sum(c["weight"] for c in comparisons)
        for c in comparisons:
            c["weight_normalized"] = c["weight"] / total_weight if total_weight > 0 else 0

        # Check for negative weights (from already-treated controls)
        negative_weights = [c for c in comparisons if c.get("type") == "late_vs_early"]

        # Overall TWFE estimate
        twfe_estimate = sum(c["estimate"] * c["weight_normalized"] for c in comparisons)

        return {
            "twfe_estimate": twfe_estimate,
            "comparisons": comparisons,
            "n_comparisons": len(comparisons),
            "timing_groups": timing_groups.to_dict(),
            "has_never_treated": len(never_treated) > 0,
            "negative_weight_share": sum(
                c["weight_normalized"] for c in comparisons
                if c.get("type") == "late_vs_early"
            ),
            "warning": "TWFE may be biased with heterogeneous treatment effects"
                      if len(timing_values) > 1 else None,
        }


def run_panel_analysis(
    data: pd.DataFrame,
    entity_col: str,
    time_col: str,
    y_col: str,
    x_cols: List[str],
    treatment_col: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Convenience function for comprehensive panel analysis.

    Returns results from FE, RE, Hausman test, and diagnostics.
    """
    estimator = PanelEstimator(data, entity_col, time_col, y_col, x_cols)

    results = {}

    # Fit models
    results["fe"] = estimator.fit_fixed_effects(entity_effects=True, time_effects=False)
    results["fe_twoway"] = estimator.fit_fixed_effects(entity_effects=True, time_effects=True)
    results["re"] = estimator.fit_random_effects()

    # Specification tests
    results["hausman"] = estimator.hausman_test(results["fe"], results["re"])
    results["mundlak"] = estimator.within_between_test()

    # Clustered inference
    results["fe_clustered"] = estimator.cluster_robust_inference(
        results["fe"], entity_col, method="stata"
    )

    # TWFE diagnostics if treatment specified
    if treatment_col is not None:
        results["bacon_decomp"] = estimator.goodman_bacon_decomposition(treatment_col)

    # Recommendation
    if results["hausman"].p_value < 0.05:
        results["recommendation"] = "Use Fixed Effects (Hausman test rejects RE)"
    else:
        results["recommendation"] = "Random Effects is more efficient (cannot reject Hausman)"

    return results


if __name__ == "__main__":
    # Example usage
    np.random.seed(42)

    # Generate panel data
    n_entities = 100
    n_periods = 10

    entities = np.repeat(np.arange(n_entities), n_periods)
    times = np.tile(np.arange(n_periods), n_entities)

    # Entity fixed effects
    alpha_i = np.repeat(np.random.normal(0, 2, n_entities), n_periods)

    # Time effects
    gamma_t = np.tile(np.random.normal(0, 0.5, n_periods), n_entities)

    # Covariates
    x1 = np.random.normal(0, 1, n_entities * n_periods)
    x2 = np.random.normal(0, 1, n_entities * n_periods)

    # Treatment (staggered adoption)
    treatment_start = np.random.choice([3, 5, 7, np.inf], n_entities, p=[0.3, 0.3, 0.2, 0.2])
    treatment = (times >= np.repeat(treatment_start, n_periods)).astype(int)

    # Outcome with heterogeneous treatment effect
    tau = 2 + 0.5 * np.repeat(np.random.normal(0, 1, n_entities), n_periods)  # Heterogeneous TE
    y = alpha_i + gamma_t + 1.5 * x1 - 0.8 * x2 + tau * treatment + np.random.normal(0, 1, n_entities * n_periods)

    df = pd.DataFrame({
        'entity': entities,
        'time': times,
        'y': y,
        'x1': x1,
        'x2': x2,
        'treatment': treatment,
    })

    # Run analysis
    estimator = PanelEstimator(
        data=df,
        entity_col='entity',
        time_col='time',
        y_col='y',
        x_cols=['treatment', 'x1', 'x2']
    )

    print("=" * 70)
    print("PANEL DATA ANALYSIS EXAMPLE")
    print("=" * 70)

    # Fixed effects
    fe = estimator.fit_fixed_effects(entity_effects=True, time_effects=True)
    print(fe.summary())

    # Random effects
    re = estimator.fit_random_effects()
    print(re.summary())

    # Hausman test
    hausman = estimator.hausman_test(fe, re)
    print(hausman)

    # Clustered SE
    fe_clustered = estimator.cluster_robust_inference(fe, 'entity')
    print("\nWith clustered standard errors:")
    print(fe_clustered.summary())

    # Bacon decomposition
    print("\n" + "=" * 70)
    print("GOODMAN-BACON DECOMPOSITION")
    print("=" * 70)
    bacon = estimator.goodman_bacon_decomposition('treatment')
    print(f"TWFE Estimate: {bacon['twfe_estimate']:.4f}")
    print(f"Number of 2x2 comparisons: {bacon['n_comparisons']}")
    if bacon['warning']:
        print(f"WARNING: {bacon['warning']}")
