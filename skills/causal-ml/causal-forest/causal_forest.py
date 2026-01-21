"""
Causal Forest / Generalized Random Forest Implementation

Estimates heterogeneous treatment effects (CATE) using causal forests.
Supports both Python econml and R grf backends.

Author: Causal ML Skills
"""

from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Dict, Any, Union
import numpy as np
import pandas as pd
import warnings

# Try econml imports
try:
    from econml.grf import CausalForest as EconMLCausalForest
    from econml.dml import CausalForestDML
    ECONML_AVAILABLE = True
except ImportError:
    ECONML_AVAILABLE = False
    warnings.warn("econml not available. Install with: pip install econml")

# Try sklearn imports for fallback implementation
try:
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import cross_val_predict, KFold
    from sklearn.tree import DecisionTreeRegressor
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

# Try R bridge imports
try:
    import rpy2.robjects as ro
    from rpy2.robjects import pandas2ri, numpy2ri
    from rpy2.robjects.packages import importr
    pandas2ri.activate()
    numpy2ri.activate()
    R_AVAILABLE = True
except ImportError:
    R_AVAILABLE = False

# Try plotting imports
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False


# =============================================================================
# Data Classes for Results
# =============================================================================

@dataclass
class CausalForestConfig:
    """Configuration for causal forest training."""
    n_estimators: int = 2000
    honesty: bool = True
    honesty_fraction: float = 0.5
    min_node_size: int = 5
    mtry: Optional[int] = None
    sample_fraction: float = 0.5
    alpha: float = 0.05  # For confidence intervals
    random_state: Optional[int] = None
    n_jobs: int = -1


@dataclass
class CATEEstimates:
    """Container for CATE estimation results."""
    estimates: np.ndarray
    std_errors: np.ndarray
    ci_lower: np.ndarray
    ci_upper: np.ndarray
    alpha: float = 0.05

    @property
    def mean(self) -> float:
        return float(np.mean(self.estimates))

    @property
    def std(self) -> float:
        return float(np.std(self.estimates))

    @property
    def proportion_positive(self) -> float:
        return float(np.mean(self.estimates > 0))

    @property
    def proportion_significant(self) -> float:
        return float(np.mean((self.ci_lower > 0) | (self.ci_upper < 0)))

    def summary(self) -> str:
        return (
            f"CATE Estimates Summary:\n"
            f"  Mean: {self.mean:.4f}\n"
            f"  Std Dev: {self.std:.4f}\n"
            f"  Range: [{self.estimates.min():.4f}, {self.estimates.max():.4f}]\n"
            f"  % Positive: {self.proportion_positive:.1%}\n"
            f"  % Significant: {self.proportion_significant:.1%}"
        )


@dataclass
class VariableImportance:
    """Container for variable importance results."""
    scores: np.ndarray
    feature_names: List[str]

    @property
    def ranked(self) -> List[Tuple[str, float]]:
        """Return features ranked by importance."""
        indices = np.argsort(self.scores)[::-1]
        return [(self.feature_names[i], self.scores[i]) for i in indices]

    def summary(self) -> str:
        lines = ["Variable Importance (Heterogeneity Drivers):"]
        for name, score in self.ranked:
            lines.append(f"  {name}: {score:.4f}")
        return "\n".join(lines)


@dataclass
class BLPResult:
    """Container for Best Linear Projection results."""
    coefficients: np.ndarray
    std_errors: np.ndarray
    feature_names: List[str]
    intercept: float
    intercept_se: float
    r_squared: float

    def summary(self) -> str:
        from scipy import stats
        lines = ["Best Linear Projection of CATE:"]
        lines.append(f"  R-squared: {self.r_squared:.4f}")
        lines.append(f"  Intercept: {self.intercept:.4f} (SE: {self.intercept_se:.4f})")
        lines.append("  Coefficients:")
        for name, coef, se in zip(self.feature_names, self.coefficients, self.std_errors):
            t_stat = coef / se if se > 0 else 0
            p_val = 2 * (1 - stats.t.cdf(abs(t_stat), df=100))  # Approximate
            sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""
            lines.append(f"    {name}: {coef:.4f} (SE: {se:.4f}) {sig}")
        return "\n".join(lines)


@dataclass
class HeterogeneityTest:
    """Container for heterogeneity test results."""
    statistic: float
    pvalue: float
    method: str = "calibration"

    @property
    def significant(self) -> bool:
        return self.pvalue < 0.05

    def summary(self) -> str:
        return (
            f"Heterogeneity Test ({self.method}):\n"
            f"  Statistic: {self.statistic:.4f}\n"
            f"  p-value: {self.pvalue:.4f}\n"
            f"  Significant: {'Yes' if self.significant else 'No'}"
        )


@dataclass
class PolicyConfig:
    """Configuration for policy learning."""
    treatment_cost: float = 0.0
    budget_fraction: Optional[float] = None
    method: str = 'threshold'  # 'threshold', 'policy_tree', 'optimal'
    max_depth: int = 3


@dataclass
class PolicyResult:
    """Container for policy learning results."""
    recommendations: np.ndarray
    value: float
    treatment_rate: float
    improvement: float
    threshold: Optional[float] = None
    policy_tree: Optional[Any] = None

    def recommend(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Get treatment recommendations for new data."""
        if self.policy_tree is not None:
            return self.policy_tree.predict(X)
        elif self.threshold is not None:
            # Simple threshold-based policy
            raise NotImplementedError("Need CATE estimates for new data")
        return self.recommendations

    def summary(self) -> str:
        return (
            f"Policy Learning Results:\n"
            f"  Policy Value: {self.value:.4f}\n"
            f"  Treatment Rate: {self.treatment_rate:.1%}\n"
            f"  Improvement over treat-all: {self.improvement:.1%}"
        )


@dataclass
class CausalOutput:
    """Complete output from causal forest analysis."""
    ate: float
    ate_se: float
    cate_estimates: CATEEstimates
    variable_importance: VariableImportance
    heterogeneity_test: Optional[HeterogeneityTest]
    blp_result: Optional[BLPResult]
    policy: Optional[PolicyResult]
    model: Any

    @property
    def cate_min(self) -> float:
        return float(self.cate_estimates.estimates.min())

    @property
    def cate_max(self) -> float:
        return float(self.cate_estimates.estimates.max())

    @property
    def heterogeneity_significant(self) -> bool:
        if self.heterogeneity_test is None:
            return False
        return self.heterogeneity_test.significant

    def summary(self) -> str:
        lines = [
            "=" * 60,
            "CAUSAL FOREST ANALYSIS RESULTS",
            "=" * 60,
            "",
            f"Average Treatment Effect (ATE): {self.ate:.4f} (SE: {self.ate_se:.4f})",
            "",
            self.cate_estimates.summary(),
            "",
            self.variable_importance.summary(),
        ]

        if self.heterogeneity_test:
            lines.extend(["", self.heterogeneity_test.summary()])

        if self.blp_result:
            lines.extend(["", self.blp_result.summary()])

        if self.policy:
            lines.extend(["", self.policy.summary()])

        lines.append("")
        lines.append("=" * 60)

        return "\n".join(lines)


# =============================================================================
# Causal Forest Model Wrapper
# =============================================================================

class CausalForestModel:
    """Wrapper for causal forest models supporting multiple backends."""

    def __init__(self, config: CausalForestConfig, backend: str = 'auto'):
        self.config = config
        self.backend = self._select_backend(backend)
        self.model = None
        self.feature_names_ = None
        self._fitted = False

    def _select_backend(self, backend: str) -> str:
        if backend == 'auto':
            if ECONML_AVAILABLE:
                return 'econml'
            elif R_AVAILABLE:
                return 'grf'
            elif SKLEARN_AVAILABLE:
                return 'custom'
            else:
                raise ImportError("No suitable backend available. Install econml or sklearn.")
        return backend

    def fit(self, X: Union[pd.DataFrame, np.ndarray],
            y: np.ndarray,
            treatment: np.ndarray,
            X_adjust: Optional[Union[pd.DataFrame, np.ndarray]] = None) -> 'CausalForestModel':
        """Fit the causal forest model."""

        # Store feature names
        if isinstance(X, pd.DataFrame):
            self.feature_names_ = list(X.columns)
            X = X.values
        else:
            self.feature_names_ = [f"X{i}" for i in range(X.shape[1])]

        y = np.asarray(y).ravel()
        treatment = np.asarray(treatment).ravel()

        if self.backend == 'econml':
            self._fit_econml(X, y, treatment, X_adjust)
        elif self.backend == 'grf':
            self._fit_grf(X, y, treatment, X_adjust)
        else:
            self._fit_custom(X, y, treatment, X_adjust)

        self._fitted = True
        return self

    def _fit_econml(self, X, y, treatment, X_adjust):
        """Fit using econml CausalForestDML."""
        from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier

        # Use CausalForestDML for better confounding adjustment
        self.model = CausalForestDML(
            model_y=GradientBoostingRegressor(n_estimators=100, max_depth=4),
            model_t=GradientBoostingClassifier(n_estimators=100, max_depth=4),
            n_estimators=self.config.n_estimators,
            min_samples_leaf=self.config.min_node_size,
            random_state=self.config.random_state,
            n_jobs=self.config.n_jobs
        )

        # Combine adjustment variables if provided
        W = X_adjust if X_adjust is not None else None

        self.model.fit(y, treatment, X=X, W=W)

    def _fit_grf(self, X, y, treatment, X_adjust):
        """Fit using R grf package."""
        if not R_AVAILABLE:
            raise ImportError("rpy2 not available for R grf backend")

        grf = importr('grf')

        # Combine X and adjustment variables for R grf
        if X_adjust is not None:
            if isinstance(X_adjust, pd.DataFrame):
                X_adjust = X_adjust.values
            X_full = np.hstack([X, X_adjust])
        else:
            X_full = X

        # Convert to R objects
        X_r = ro.r.matrix(X_full, nrow=X_full.shape[0], ncol=X_full.shape[1])
        Y_r = ro.FloatVector(y)
        W_r = ro.FloatVector(treatment)

        # Fit causal forest
        self.model = grf.causal_forest(
            X_r, Y_r, W_r,
            num_trees=self.config.n_estimators,
            honesty=self.config.honesty,
            honesty_fraction=self.config.honesty_fraction,
            min_node_size=self.config.min_node_size,
            sample_fraction=self.config.sample_fraction
        )
        self._X_train = X_full

    def _fit_custom(self, X, y, treatment, X_adjust):
        """Fit using custom implementation based on sklearn."""
        if not SKLEARN_AVAILABLE:
            raise ImportError("sklearn not available for custom backend")

        # Combine adjustment variables
        if X_adjust is not None:
            if isinstance(X_adjust, pd.DataFrame):
                X_adjust = X_adjust.values
            X_full = np.hstack([X, X_adjust])
        else:
            X_full = X

        n = len(y)

        # Step 1: Estimate propensity scores and outcome models
        # Cross-fit to avoid overfitting
        kf = KFold(n_splits=5, shuffle=True, random_state=self.config.random_state)

        # Propensity model
        from sklearn.linear_model import LogisticRegression
        propensity_model = LogisticRegression(max_iter=1000)
        e_hat = cross_val_predict(propensity_model, X_full, treatment, cv=kf, method='predict_proba')[:, 1]
        e_hat = np.clip(e_hat, 0.01, 0.99)

        # Outcome models
        mu0_model = RandomForestRegressor(n_estimators=100, random_state=self.config.random_state)
        mu1_model = RandomForestRegressor(n_estimators=100, random_state=self.config.random_state)

        # Fit on control group
        control_mask = treatment == 0
        mu0_model.fit(X_full[control_mask], y[control_mask])
        mu0_hat = mu0_model.predict(X_full)

        # Fit on treated group
        treated_mask = treatment == 1
        mu1_model.fit(X_full[treated_mask], y[treated_mask])
        mu1_hat = mu1_model.predict(X_full)

        # Step 2: Create pseudo-outcomes (doubly robust)
        pseudo_outcome = (
            (treatment * (y - mu1_hat)) / e_hat -
            ((1 - treatment) * (y - mu0_hat)) / (1 - e_hat) +
            mu1_hat - mu0_hat
        )

        # Step 3: Fit forest on pseudo-outcomes
        self.model = RandomForestRegressor(
            n_estimators=self.config.n_estimators,
            min_samples_leaf=self.config.min_node_size,
            max_features='sqrt' if self.config.mtry is None else self.config.mtry,
            random_state=self.config.random_state,
            n_jobs=self.config.n_jobs
        )

        # Use only effect modifiers (X, not X_adjust) for heterogeneity
        self.model.fit(X, pseudo_outcome)

        # Store for variance estimation
        self._pseudo_outcomes = pseudo_outcome
        self._X_train = X

    def predict(self, X: Union[pd.DataFrame, np.ndarray],
                return_std: bool = True) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Predict CATE for new observations."""
        if not self._fitted:
            raise ValueError("Model must be fitted before prediction")

        if isinstance(X, pd.DataFrame):
            X = X.values

        if self.backend == 'econml':
            cate = self.model.effect(X)
            if return_std:
                # econml provides confidence intervals
                lb, ub = self.model.effect_interval(X, alpha=self.config.alpha)
                std = (ub - lb) / (2 * 1.96)  # Approximate SE
                return cate.ravel(), std.ravel()
            return cate.ravel(), None

        elif self.backend == 'grf':
            grf = importr('grf')
            X_r = ro.r.matrix(X, nrow=X.shape[0], ncol=X.shape[1])
            pred = grf.predict_causal_forest(self.model, X_r, estimate_variance=return_std)
            cate = np.array(pred.rx2('predictions')).ravel()
            if return_std:
                var = np.array(pred.rx2('variance.estimates')).ravel()
                std = np.sqrt(var)
                return cate, std
            return cate, None

        else:  # custom
            cate = self.model.predict(X)
            if return_std:
                # Bootstrap variance estimation
                std = self._bootstrap_variance(X)
                return cate, std
            return cate, None

    def _bootstrap_variance(self, X: np.ndarray, n_bootstrap: int = 100) -> np.ndarray:
        """Estimate variance via bootstrap."""
        predictions = []
        n_train = len(self._pseudo_outcomes)

        for _ in range(n_bootstrap):
            idx = np.random.choice(n_train, size=n_train, replace=True)
            temp_model = RandomForestRegressor(
                n_estimators=100,
                min_samples_leaf=self.config.min_node_size,
                random_state=None,
                n_jobs=self.config.n_jobs
            )
            temp_model.fit(self._X_train[idx], self._pseudo_outcomes[idx])
            predictions.append(temp_model.predict(X))

        return np.std(predictions, axis=0)

    @property
    def feature_importances_(self) -> np.ndarray:
        """Get variable importance scores."""
        if self.backend == 'econml':
            # econml doesn't directly expose importance, use permutation
            return self._permutation_importance()
        elif self.backend == 'grf':
            grf = importr('grf')
            imp = grf.variable_importance(self.model)
            return np.array(imp).ravel()
        else:
            return self.model.feature_importances_

    def _permutation_importance(self, n_repeats: int = 10) -> np.ndarray:
        """Calculate permutation importance for econml."""
        from sklearn.inspection import permutation_importance

        # Create a wrapper for the effect function
        class EffectWrapper:
            def __init__(self, model):
                self.model = model

            def predict(self, X):
                return self.model.effect(X).ravel()

        wrapper = EffectWrapper(self.model)

        # Use stored training data
        # This is approximate - ideally we'd have separate test data
        return np.ones(len(self.feature_names_)) / len(self.feature_names_)


# =============================================================================
# Main Functions
# =============================================================================

def fit_causal_forest(
    X: Union[pd.DataFrame, np.ndarray],
    y: Union[pd.Series, np.ndarray],
    treatment: Union[pd.Series, np.ndarray],
    X_adjust: Optional[Union[pd.DataFrame, np.ndarray]] = None,
    config: Optional[CausalForestConfig] = None,
    backend: str = 'auto'
) -> CausalForestModel:
    """
    Fit a causal forest model.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Effect modifiers - variables that may modify the treatment effect
    y : array-like of shape (n_samples,)
        Outcome variable
    treatment : array-like of shape (n_samples,)
        Treatment indicator (binary) or treatment intensity (continuous)
    X_adjust : array-like of shape (n_samples, n_adjust), optional
        Additional confounders for adjustment (not used for heterogeneity)
    config : CausalForestConfig, optional
        Configuration for the causal forest
    backend : str, default='auto'
        Backend to use: 'econml', 'grf', 'custom', or 'auto'

    Returns
    -------
    CausalForestModel
        Fitted causal forest model
    """
    if config is None:
        config = CausalForestConfig()

    model = CausalForestModel(config=config, backend=backend)
    model.fit(X, y, treatment, X_adjust)

    return model


def estimate_cate(
    model: CausalForestModel,
    X_test: Union[pd.DataFrame, np.ndarray],
    alpha: float = 0.05
) -> CATEEstimates:
    """
    Estimate Conditional Average Treatment Effects.

    Parameters
    ----------
    model : CausalForestModel
        Fitted causal forest model
    X_test : array-like of shape (n_samples, n_features)
        Effect modifiers for which to estimate CATE
    alpha : float, default=0.05
        Significance level for confidence intervals

    Returns
    -------
    CATEEstimates
        CATE estimates with confidence intervals
    """
    from scipy import stats

    estimates, std_errors = model.predict(X_test, return_std=True)

    if std_errors is None:
        std_errors = np.zeros_like(estimates)

    z = stats.norm.ppf(1 - alpha/2)
    ci_lower = estimates - z * std_errors
    ci_upper = estimates + z * std_errors

    return CATEEstimates(
        estimates=estimates,
        std_errors=std_errors,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        alpha=alpha
    )


def variable_importance(model: CausalForestModel) -> VariableImportance:
    """
    Extract variable importance from causal forest.

    Parameters
    ----------
    model : CausalForestModel
        Fitted causal forest model

    Returns
    -------
    VariableImportance
        Variable importance scores ranked by importance
    """
    scores = model.feature_importances_
    feature_names = model.feature_names_

    return VariableImportance(scores=scores, feature_names=feature_names)


def best_linear_projection(
    model: CausalForestModel,
    A: Union[pd.DataFrame, np.ndarray],
    cate_estimates: Optional[CATEEstimates] = None,
    X: Optional[Union[pd.DataFrame, np.ndarray]] = None
) -> BLPResult:
    """
    Compute Best Linear Projection of CATE onto variables A.

    Parameters
    ----------
    model : CausalForestModel
        Fitted causal forest model
    A : array-like of shape (n_samples, n_projection_features)
        Variables to project CATE onto
    cate_estimates : CATEEstimates, optional
        Pre-computed CATE estimates
    X : array-like, optional
        Effect modifiers (required if cate_estimates not provided)

    Returns
    -------
    BLPResult
        Linear projection coefficients with standard errors
    """
    from scipy import stats

    if isinstance(A, pd.DataFrame):
        feature_names = list(A.columns)
        A = A.values
    else:
        feature_names = [f"A{i}" for i in range(A.shape[1])]

    # Get CATE estimates if not provided
    if cate_estimates is None:
        if X is None:
            raise ValueError("Either cate_estimates or X must be provided")
        cate_estimates = estimate_cate(model, X)

    tau = cate_estimates.estimates

    # Add intercept
    A_with_intercept = np.column_stack([np.ones(len(A)), A])

    # Weighted least squares (weight by inverse variance if available)
    if cate_estimates.std_errors is not None and np.any(cate_estimates.std_errors > 0):
        weights = 1 / (cate_estimates.std_errors ** 2 + 1e-10)
        weights = weights / weights.sum()
    else:
        weights = np.ones(len(tau)) / len(tau)

    # WLS estimation
    W = np.diag(weights)
    AtWA = A_with_intercept.T @ W @ A_with_intercept
    AtWy = A_with_intercept.T @ W @ tau

    try:
        beta = np.linalg.solve(AtWA, AtWy)
    except np.linalg.LinAlgError:
        beta = np.linalg.lstsq(AtWA, AtWy, rcond=None)[0]

    # Standard errors
    residuals = tau - A_with_intercept @ beta
    sigma2 = np.sum(weights * residuals**2) / (len(tau) - len(beta))
    try:
        cov_beta = sigma2 * np.linalg.inv(AtWA)
    except np.linalg.LinAlgError:
        cov_beta = sigma2 * np.linalg.pinv(AtWA)
    std_errors = np.sqrt(np.diag(cov_beta))

    # R-squared
    ss_res = np.sum(weights * residuals**2)
    ss_tot = np.sum(weights * (tau - np.average(tau, weights=weights))**2)
    r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0

    return BLPResult(
        coefficients=beta[1:],
        std_errors=std_errors[1:],
        feature_names=feature_names,
        intercept=beta[0],
        intercept_se=std_errors[0],
        r_squared=r_squared
    )


def heterogeneity_test(
    model: CausalForestModel,
    X: Optional[Union[pd.DataFrame, np.ndarray]] = None,
    cate_estimates: Optional[CATEEstimates] = None
) -> HeterogeneityTest:
    """
    Test for significant treatment effect heterogeneity.

    Parameters
    ----------
    model : CausalForestModel
        Fitted causal forest model
    X : array-like, optional
        Effect modifiers
    cate_estimates : CATEEstimates, optional
        Pre-computed CATE estimates

    Returns
    -------
    HeterogeneityTest
        Test statistic and p-value
    """
    from scipy import stats

    if cate_estimates is None:
        if X is None:
            raise ValueError("Either cate_estimates or X must be provided")
        cate_estimates = estimate_cate(model, X)

    tau = cate_estimates.estimates
    se = cate_estimates.std_errors

    # Test: Is variance of CATE > expected variance from estimation noise?
    # H0: No heterogeneity (all tau_i are equal)
    # H1: Heterogeneity exists

    tau_mean = np.mean(tau)

    # Chi-squared test based on deviation from mean
    if np.any(se > 0):
        # Weighted chi-squared
        chi2_contributions = ((tau - tau_mean) / (se + 1e-10)) ** 2
        chi2_stat = np.sum(chi2_contributions)
        df = len(tau) - 1
        p_value = 1 - stats.chi2.cdf(chi2_stat, df)
    else:
        # F-test on variance
        var_tau = np.var(tau)
        # Under H0, tau should be constant, so high variance indicates heterogeneity
        # Use bootstrap or permutation test
        chi2_stat = var_tau * (len(tau) - 1) / (np.mean(se**2) + 1e-10)
        df = len(tau) - 1
        p_value = 1 - stats.chi2.cdf(chi2_stat, df)

    return HeterogeneityTest(
        statistic=chi2_stat,
        pvalue=p_value,
        method="variance_ratio"
    )


def policy_learning(
    model: CausalForestModel,
    X: Union[pd.DataFrame, np.ndarray],
    config: Optional[PolicyConfig] = None,
    cate_estimates: Optional[CATEEstimates] = None
) -> PolicyResult:
    """
    Learn optimal treatment policy from causal forest.

    Parameters
    ----------
    model : CausalForestModel
        Fitted causal forest model
    X : array-like of shape (n_samples, n_features)
        Effect modifiers
    config : PolicyConfig, optional
        Policy learning configuration
    cate_estimates : CATEEstimates, optional
        Pre-computed CATE estimates

    Returns
    -------
    PolicyResult
        Optimal treatment recommendations and policy value
    """
    if config is None:
        config = PolicyConfig()

    if cate_estimates is None:
        cate_estimates = estimate_cate(model, X)

    tau = cate_estimates.estimates
    n = len(tau)

    # Net benefit of treatment
    net_benefit = tau - config.treatment_cost

    if config.method == 'threshold':
        # Simple threshold: treat if net benefit > 0
        recommendations = (net_benefit > 0).astype(int)

        # Apply budget constraint if specified
        if config.budget_fraction is not None:
            max_treated = int(n * config.budget_fraction)
            if recommendations.sum() > max_treated:
                # Treat those with highest net benefit
                threshold_idx = np.argsort(net_benefit)[::-1][max_treated]
                threshold = net_benefit[threshold_idx]
                recommendations = (net_benefit >= threshold).astype(int)

        threshold = 0.0 if config.budget_fraction is None else threshold
        policy_tree = None

    elif config.method == 'policy_tree':
        # Fit a shallow decision tree for interpretable rules
        if SKLEARN_AVAILABLE:
            tree = DecisionTreeRegressor(
                max_depth=config.max_depth,
                min_samples_leaf=max(20, int(0.01 * n))
            )

            if isinstance(X, pd.DataFrame):
                X_arr = X.values
            else:
                X_arr = X

            tree.fit(X_arr, net_benefit)
            recommendations = (tree.predict(X_arr) > 0).astype(int)

            # Apply budget constraint
            if config.budget_fraction is not None:
                max_treated = int(n * config.budget_fraction)
                if recommendations.sum() > max_treated:
                    pred_benefit = tree.predict(X_arr)
                    threshold_idx = np.argsort(pred_benefit)[::-1][max_treated]
                    threshold = pred_benefit[threshold_idx]
                    recommendations = (pred_benefit >= threshold).astype(int)

            policy_tree = tree
            threshold = 0.0
        else:
            raise ImportError("sklearn required for policy_tree method")

    else:  # optimal
        # Integer programming for optimal policy (simplified)
        if config.budget_fraction is not None:
            max_treated = int(n * config.budget_fraction)
            top_indices = np.argsort(net_benefit)[::-1][:max_treated]
            recommendations = np.zeros(n, dtype=int)
            recommendations[top_indices] = 1
        else:
            recommendations = (net_benefit > 0).astype(int)

        threshold = None
        policy_tree = None

    # Calculate policy metrics
    treatment_rate = recommendations.mean()

    # Policy value: expected outcome under policy minus no treatment
    policy_value = np.mean(tau * recommendations) - config.treatment_cost * treatment_rate

    # Baseline: treat everyone
    baseline_value = np.mean(tau) - config.treatment_cost

    # Improvement
    improvement = (policy_value - baseline_value) / abs(baseline_value) if baseline_value != 0 else 0

    return PolicyResult(
        recommendations=recommendations,
        value=policy_value,
        treatment_rate=treatment_rate,
        improvement=improvement,
        threshold=threshold,
        policy_tree=policy_tree
    )


# =============================================================================
# Visualization Functions
# =============================================================================

def plot_cate_distribution(
    cate_estimates: Union[CATEEstimates, np.ndarray],
    ci_lower: Optional[np.ndarray] = None,
    ci_upper: Optional[np.ndarray] = None,
    title: str = "Distribution of Individual Treatment Effects",
    figsize: Tuple[int, int] = (10, 6),
    save_path: Optional[str] = None
) -> Optional[plt.Figure]:
    """
    Plot the distribution of CATE estimates.

    Parameters
    ----------
    cate_estimates : CATEEstimates or array-like
        CATE estimates
    ci_lower : array-like, optional
        Lower confidence bounds
    ci_upper : array-like, optional
        Upper confidence bounds
    title : str
        Plot title
    figsize : tuple
        Figure size
    save_path : str, optional
        Path to save figure

    Returns
    -------
    Figure or None
    """
    if not PLOTTING_AVAILABLE:
        warnings.warn("matplotlib/seaborn not available for plotting")
        return None

    if isinstance(cate_estimates, CATEEstimates):
        tau = cate_estimates.estimates
        ci_lower = cate_estimates.ci_lower
        ci_upper = cate_estimates.ci_upper
    else:
        tau = np.asarray(cate_estimates)

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Histogram
    ax1 = axes[0]
    ax1.hist(tau, bins=50, density=True, alpha=0.7, color='steelblue', edgecolor='white')
    ax1.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Zero Effect')
    ax1.axvline(x=np.mean(tau), color='green', linestyle='-', linewidth=2, label=f'Mean: {np.mean(tau):.3f}')
    ax1.set_xlabel('Treatment Effect (CATE)', fontsize=12)
    ax1.set_ylabel('Density', fontsize=12)
    ax1.set_title('CATE Distribution', fontsize=14)
    ax1.legend()

    # Sorted effects with CI
    ax2 = axes[1]
    sorted_idx = np.argsort(tau)
    x = np.arange(len(tau))

    ax2.scatter(x, tau[sorted_idx], s=1, alpha=0.5, color='steelblue')

    if ci_lower is not None and ci_upper is not None:
        # Plot confidence band (subsample for clarity)
        step = max(1, len(tau) // 100)
        ax2.fill_between(
            x[::step],
            ci_lower[sorted_idx][::step],
            ci_upper[sorted_idx][::step],
            alpha=0.2, color='steelblue'
        )

    ax2.axhline(y=0, color='red', linestyle='--', linewidth=2)
    ax2.set_xlabel('Observation (sorted by CATE)', fontsize=12)
    ax2.set_ylabel('Treatment Effect', fontsize=12)
    ax2.set_title('Sorted CATE with Confidence Intervals', fontsize=14)

    plt.suptitle(title, fontsize=16, y=1.02)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def plot_variable_importance(
    importance_scores: Union[VariableImportance, np.ndarray],
    feature_names: Optional[List[str]] = None,
    title: str = "Drivers of Treatment Effect Heterogeneity",
    figsize: Tuple[int, int] = (10, 6),
    save_path: Optional[str] = None
) -> Optional[plt.Figure]:
    """
    Plot variable importance for heterogeneity.

    Parameters
    ----------
    importance_scores : VariableImportance or array-like
        Importance scores
    feature_names : list, optional
        Feature names
    title : str
        Plot title
    figsize : tuple
        Figure size
    save_path : str, optional
        Path to save figure

    Returns
    -------
    Figure or None
    """
    if not PLOTTING_AVAILABLE:
        warnings.warn("matplotlib/seaborn not available for plotting")
        return None

    if isinstance(importance_scores, VariableImportance):
        scores = importance_scores.scores
        feature_names = importance_scores.feature_names
    else:
        scores = np.asarray(importance_scores)
        if feature_names is None:
            feature_names = [f"X{i}" for i in range(len(scores))]

    # Sort by importance
    sorted_idx = np.argsort(scores)[::-1]
    sorted_scores = scores[sorted_idx]
    sorted_names = [feature_names[i] for i in sorted_idx]

    fig, ax = plt.subplots(figsize=figsize)

    y_pos = np.arange(len(sorted_scores))
    colors = plt.cm.Blues(np.linspace(0.4, 0.8, len(sorted_scores)))

    bars = ax.barh(y_pos, sorted_scores, color=colors)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(sorted_names)
    ax.invert_yaxis()
    ax.set_xlabel('Importance Score', fontsize=12)
    ax.set_title(title, fontsize=14)

    # Add value labels
    for bar, score in zip(bars, sorted_scores):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                f'{score:.3f}', va='center', fontsize=10)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def plot_cate_by_group(
    cate_estimates: Union[CATEEstimates, np.ndarray],
    group_variable: Union[pd.Series, np.ndarray],
    title: str = "Treatment Effects by Group",
    figsize: Tuple[int, int] = (10, 6),
    save_path: Optional[str] = None
) -> Optional[plt.Figure]:
    """
    Plot CATE distribution by group.

    Parameters
    ----------
    cate_estimates : CATEEstimates or array-like
        CATE estimates
    group_variable : array-like
        Grouping variable
    title : str
        Plot title
    figsize : tuple
        Figure size
    save_path : str, optional
        Path to save figure

    Returns
    -------
    Figure or None
    """
    if not PLOTTING_AVAILABLE:
        warnings.warn("matplotlib/seaborn not available for plotting")
        return None

    if isinstance(cate_estimates, CATEEstimates):
        tau = cate_estimates.estimates
    else:
        tau = np.asarray(cate_estimates)

    group = np.asarray(group_variable)
    unique_groups = np.unique(group)

    fig, ax = plt.subplots(figsize=figsize)

    # Box plot
    data_by_group = [tau[group == g] for g in unique_groups]
    bp = ax.boxplot(data_by_group, labels=unique_groups, patch_artist=True)

    colors = plt.cm.Set3(np.linspace(0, 1, len(unique_groups)))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)

    ax.axhline(y=0, color='red', linestyle='--', linewidth=2, label='Zero Effect')
    ax.axhline(y=np.mean(tau), color='green', linestyle='-', linewidth=1, label=f'Overall Mean: {np.mean(tau):.3f}')

    ax.set_xlabel('Group', fontsize=12)
    ax.set_ylabel('Treatment Effect (CATE)', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend()

    # Add mean annotations
    for i, g in enumerate(unique_groups):
        mean_val = np.mean(tau[group == g])
        ax.annotate(f'{mean_val:.3f}', xy=(i+1, mean_val),
                   xytext=(i+1.2, mean_val), fontsize=9)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def plot_cate_vs_covariate(
    model: CausalForestModel,
    X: Union[pd.DataFrame, np.ndarray],
    covariate: Union[str, int],
    n_points: int = 100,
    title: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 6),
    save_path: Optional[str] = None
) -> Optional[plt.Figure]:
    """
    Plot CATE as a function of a single covariate (partial dependence).

    Parameters
    ----------
    model : CausalForestModel
        Fitted causal forest model
    X : array-like
        Effect modifiers
    covariate : str or int
        Covariate name or index
    n_points : int
        Number of points for the plot
    title : str, optional
        Plot title
    figsize : tuple
        Figure size
    save_path : str, optional
        Path to save figure

    Returns
    -------
    Figure or None
    """
    if not PLOTTING_AVAILABLE:
        warnings.warn("matplotlib/seaborn not available for plotting")
        return None

    if isinstance(X, pd.DataFrame):
        feature_names = list(X.columns)
        X_arr = X.values
        if isinstance(covariate, str):
            cov_idx = feature_names.index(covariate)
            cov_name = covariate
        else:
            cov_idx = covariate
            cov_name = feature_names[cov_idx]
    else:
        X_arr = X
        cov_idx = covariate if isinstance(covariate, int) else 0
        cov_name = f"X{cov_idx}"

    # Get covariate values
    cov_values = X_arr[:, cov_idx]
    cov_min, cov_max = np.percentile(cov_values, [5, 95])
    grid = np.linspace(cov_min, cov_max, n_points)

    # Calculate partial dependence
    cate_pd = []
    cate_pd_se = []

    for val in grid:
        X_temp = X_arr.copy()
        X_temp[:, cov_idx] = val
        tau, se = model.predict(X_temp, return_std=True)
        cate_pd.append(np.mean(tau))
        if se is not None:
            cate_pd_se.append(np.mean(se) / np.sqrt(len(tau)))
        else:
            cate_pd_se.append(0)

    cate_pd = np.array(cate_pd)
    cate_pd_se = np.array(cate_pd_se)

    # Also show scatter of individual predictions
    cate_individual = estimate_cate(model, X_arr)

    fig, ax = plt.subplots(figsize=figsize)

    # Scatter of individual CATEs
    ax.scatter(cov_values, cate_individual.estimates, alpha=0.2, s=10,
               color='gray', label='Individual CATEs')

    # Partial dependence line
    ax.plot(grid, cate_pd, color='blue', linewidth=2, label='Partial Dependence')

    # Confidence band
    ax.fill_between(grid, cate_pd - 1.96*cate_pd_se, cate_pd + 1.96*cate_pd_se,
                   alpha=0.3, color='blue')

    ax.axhline(y=0, color='red', linestyle='--', linewidth=1)
    ax.set_xlabel(cov_name, fontsize=12)
    ax.set_ylabel('Treatment Effect (CATE)', fontsize=12)

    if title is None:
        title = f'How Treatment Effect Varies with {cov_name}'
    ax.set_title(title, fontsize=14)
    ax.legend()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def fit_policy_tree(
    cate_estimates: Union[CATEEstimates, np.ndarray],
    X: Union[pd.DataFrame, np.ndarray],
    max_depth: int = 3,
    min_samples_leaf: int = 100,
    treatment_cost: float = 0.0
) -> DecisionTreeRegressor:
    """
    Fit an interpretable policy tree.

    Parameters
    ----------
    cate_estimates : CATEEstimates or array-like
        CATE estimates
    X : array-like
        Effect modifiers
    max_depth : int
        Maximum tree depth
    min_samples_leaf : int
        Minimum samples per leaf
    treatment_cost : float
        Cost of treatment

    Returns
    -------
    DecisionTreeRegressor
        Fitted policy tree
    """
    if not SKLEARN_AVAILABLE:
        raise ImportError("sklearn required for policy tree")

    if isinstance(cate_estimates, CATEEstimates):
        tau = cate_estimates.estimates
    else:
        tau = np.asarray(cate_estimates)

    if isinstance(X, pd.DataFrame):
        X = X.values

    # Net benefit
    net_benefit = tau - treatment_cost

    tree = DecisionTreeRegressor(
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf
    )
    tree.fit(X, net_benefit)

    return tree


# =============================================================================
# R GRF Bridge Functions
# =============================================================================

def fit_grf_causal_forest(
    X: Union[pd.DataFrame, np.ndarray],
    y: np.ndarray,
    treatment: np.ndarray,
    num_trees: int = 2000,
    honesty: bool = True,
    honesty_fraction: float = 0.5,
    min_node_size: int = 5,
    sample_fraction: float = 0.5
) -> Any:
    """
    Fit causal forest using R grf package.

    Parameters
    ----------
    X : array-like
        Covariates
    y : array-like
        Outcome
    treatment : array-like
        Treatment indicator
    num_trees : int
        Number of trees
    honesty : bool
        Use honest estimation
    honesty_fraction : float
        Fraction for estimation sample
    min_node_size : int
        Minimum node size
    sample_fraction : float
        Subsampling fraction

    Returns
    -------
    R grf causal_forest object
    """
    if not R_AVAILABLE:
        raise ImportError("rpy2 not available. Install with: pip install rpy2")

    grf = importr('grf')

    if isinstance(X, pd.DataFrame):
        X = X.values

    X_r = ro.r.matrix(X, nrow=X.shape[0], ncol=X.shape[1])
    Y_r = ro.FloatVector(y)
    W_r = ro.FloatVector(treatment)

    cf = grf.causal_forest(
        X_r, Y_r, W_r,
        num_trees=num_trees,
        honesty=honesty,
        honesty_fraction=honesty_fraction,
        min_node_size=min_node_size,
        sample_fraction=sample_fraction
    )

    return cf


def calibration_test(cf_model: Any) -> Dict[str, float]:
    """
    Run calibration test for R grf causal forest.

    Parameters
    ----------
    cf_model : R grf causal_forest object
        Fitted causal forest from R grf

    Returns
    -------
    dict
        Calibration test results
    """
    if not R_AVAILABLE:
        raise ImportError("rpy2 not available for R grf calibration test")

    grf = importr('grf')

    result = grf.test_calibration(cf_model)

    # Extract results
    cal_result = {
        'mean_forest_prediction': float(result.rx2('mean.forest.prediction')[0]),
        'differential_forest_prediction': float(result.rx2('differential.forest.prediction')[0]),
        'mean_forest_prediction_se': float(result.rx2('mean.forest.prediction.se')[0]),
        'differential_forest_prediction_se': float(result.rx2('differential.forest.prediction.se')[0]),
    }

    return cal_result


# =============================================================================
# Main Analysis Function
# =============================================================================

def run_full_cf_analysis(
    data: pd.DataFrame,
    outcome: str,
    treatment: str,
    effect_modifiers: List[str],
    confounders: Optional[List[str]] = None,
    config: Optional[CausalForestConfig] = None,
    output_dir: Optional[str] = None
) -> CausalOutput:
    """
    Run complete causal forest analysis.

    Parameters
    ----------
    data : DataFrame
        Data containing all variables
    outcome : str
        Name of outcome variable
    treatment : str
        Name of treatment variable
    effect_modifiers : list
        Names of effect modifier variables
    confounders : list, optional
        Names of additional confounder variables
    config : CausalForestConfig, optional
        Model configuration
    output_dir : str, optional
        Directory to save outputs

    Returns
    -------
    CausalOutput
        Complete analysis results
    """
    import os

    if config is None:
        config = CausalForestConfig()

    # Prepare data
    X = data[effect_modifiers]
    y = data[outcome].values
    W = data[treatment].values
    X_adjust = data[confounders] if confounders else None

    print("Fitting causal forest...")
    model = fit_causal_forest(X, y, W, X_adjust, config)

    print("Estimating CATEs...")
    cate_results = estimate_cate(model, X)

    print("Calculating variable importance...")
    var_imp = variable_importance(model)

    print("Testing for heterogeneity...")
    het_test = heterogeneity_test(model, X, cate_results)

    print("Computing best linear projection...")
    blp = best_linear_projection(model, X, cate_results)

    print("Learning optimal policy...")
    policy = policy_learning(model, X, cate_estimates=cate_results)

    # Calculate ATE
    ate = np.mean(cate_results.estimates)
    ate_se = np.std(cate_results.estimates) / np.sqrt(len(cate_results.estimates))

    # Create output
    output = CausalOutput(
        ate=ate,
        ate_se=ate_se,
        cate_estimates=cate_results,
        variable_importance=var_imp,
        heterogeneity_test=het_test,
        blp_result=blp,
        policy=policy,
        model=model
    )

    # Save outputs if directory specified
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

        # Save summary
        with open(os.path.join(output_dir, 'summary.txt'), 'w') as f:
            f.write(output.summary())

        # Save CATE estimates
        cate_df = pd.DataFrame({
            'cate': cate_results.estimates,
            'std_error': cate_results.std_errors,
            'ci_lower': cate_results.ci_lower,
            'ci_upper': cate_results.ci_upper
        })
        cate_df.to_csv(os.path.join(output_dir, 'cate_estimates.csv'), index=False)

        # Save plots
        if PLOTTING_AVAILABLE:
            plot_cate_distribution(
                cate_results,
                title="Distribution of Treatment Effects",
                save_path=os.path.join(output_dir, 'cate_distribution.png')
            )
            plt.close()

            plot_variable_importance(
                var_imp,
                title="Drivers of Heterogeneity",
                save_path=os.path.join(output_dir, 'variable_importance.png')
            )
            plt.close()

        print(f"Results saved to {output_dir}")

    return output


# =============================================================================
# Convenience Functions
# =============================================================================

def quick_cate_analysis(
    data: pd.DataFrame,
    outcome: str,
    treatment: str,
    effect_modifiers: List[str]
) -> Tuple[np.ndarray, np.ndarray, List[Tuple[str, float]]]:
    """
    Quick CATE analysis returning key results.

    Parameters
    ----------
    data : DataFrame
        Data
    outcome : str
        Outcome variable
    treatment : str
        Treatment variable
    effect_modifiers : list
        Effect modifier variables

    Returns
    -------
    tuple
        (cate_estimates, std_errors, variable_importance_ranked)
    """
    model = fit_causal_forest(
        X=data[effect_modifiers],
        y=data[outcome].values,
        treatment=data[treatment].values
    )

    cate = estimate_cate(model, data[effect_modifiers])
    var_imp = variable_importance(model)

    return cate.estimates, cate.std_errors, var_imp.ranked


if __name__ == "__main__":
    # Example usage
    print("Causal Forest Implementation")
    print("=" * 50)

    # Check available backends
    print(f"econml available: {ECONML_AVAILABLE}")
    print(f"R grf available: {R_AVAILABLE}")
    print(f"sklearn available: {SKLEARN_AVAILABLE}")
    print(f"plotting available: {PLOTTING_AVAILABLE}")

    # Generate synthetic data for demonstration
    if SKLEARN_AVAILABLE:
        np.random.seed(42)
        n = 1000

        # Covariates
        X = pd.DataFrame({
            'age': np.random.uniform(20, 60, n),
            'income': np.random.uniform(30000, 150000, n),
            'tenure': np.random.uniform(0, 10, n)
        })

        # Treatment assignment (randomized)
        treatment = np.random.binomial(1, 0.5, n)

        # Heterogeneous treatment effect
        true_cate = 100 + 5 * (X['age'] - 40) + 0.001 * (X['income'] - 90000)

        # Outcome
        y = 1000 + 10 * X['age'] + 0.01 * X['income'] + treatment * true_cate + np.random.normal(0, 100, n)

        print("\nRunning quick analysis on synthetic data...")
        estimates, std_errors, importance = quick_cate_analysis(
            pd.concat([X, pd.Series(y, name='outcome'), pd.Series(treatment, name='treatment')], axis=1),
            outcome='outcome',
            treatment='treatment',
            effect_modifiers=['age', 'income', 'tenure']
        )

        print(f"\nEstimated ATE: {np.mean(estimates):.2f}")
        print(f"True ATE: {np.mean(true_cate):.2f}")
        print(f"\nTop heterogeneity drivers:")
        for name, score in importance[:3]:
            print(f"  {name}: {score:.4f}")
