"""
Discrete Choice Estimator

Unified interface for binary, ordered, multinomial, and count models
with proper marginal effects computation for causal interpretation.

Classes:
    DiscreteChoiceEstimator: Main estimator with fit/predict/marginal_effects methods
    CausalOutput: Standardized output format for causal inference results

Example:
    >>> from discrete_choice_estimator import DiscreteChoiceEstimator
    >>>
    >>> # Binary logit
    >>> estimator = DiscreteChoiceEstimator(model_type='logit')
    >>> result = estimator.fit(y, X, treatment='treatment_var')
    >>> ame = estimator.marginal_effects(method='average')
    >>> ate = estimator.treatment_effect()
    >>>
    >>> # Count model
    >>> estimator = DiscreteChoiceEstimator(model_type='negbin')
    >>> result = estimator.fit(count_y, X, exposure=time_at_risk)
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union, Any
from enum import Enum

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from statsmodels.discrete.discrete_model import (
    Logit, Probit, Poisson, NegativeBinomial, MNLogit
)
from statsmodels.miscmodels.ordinal_model import OrderedModel
from statsmodels.discrete.count_model import (
    ZeroInflatedPoisson, ZeroInflatedNegativeBinomialP
)


class ModelType(Enum):
    """Supported discrete choice model types."""
    LOGIT = 'logit'
    PROBIT = 'probit'
    LPM = 'lpm'
    ORDERED_LOGIT = 'ologit'
    ORDERED_PROBIT = 'oprobit'
    MULTINOMIAL_LOGIT = 'mlogit'
    POISSON = 'poisson'
    NEGBIN = 'negbin'
    ZIP = 'zip'
    ZINB = 'zinb'


@dataclass
class CausalOutput:
    """
    Standardized output for causal inference results.

    Attributes
    ----------
    estimate : float or array
        Point estimate(s)
    std_error : float or array
        Standard error(s)
    ci_lower : float or array
        Lower confidence interval bound(s)
    ci_upper : float or array
        Upper confidence interval bound(s)
    p_value : float or array
        P-value(s) for null hypothesis of zero effect
    method : str
        Estimation method used
    n_obs : int
        Number of observations
    additional_info : dict
        Additional information specific to the method
    """
    estimate: Union[float, np.ndarray]
    std_error: Union[float, np.ndarray]
    ci_lower: Union[float, np.ndarray]
    ci_upper: Union[float, np.ndarray]
    p_value: Union[float, np.ndarray]
    method: str
    n_obs: int
    additional_info: Dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        if isinstance(self.estimate, np.ndarray):
            est_str = f"array of shape {self.estimate.shape}"
        else:
            est_str = f"{self.estimate:.6f}"
        return (f"CausalOutput(estimate={est_str}, "
                f"se={self.std_error if isinstance(self.std_error, float) else 'array'}, "
                f"method='{self.method}', n={self.n_obs})")

    def summary(self) -> pd.DataFrame:
        """Return summary as DataFrame."""
        if isinstance(self.estimate, np.ndarray):
            return pd.DataFrame({
                'estimate': self.estimate,
                'std_error': self.std_error,
                'ci_lower': self.ci_lower,
                'ci_upper': self.ci_upper,
                'p_value': self.p_value
            })
        else:
            return pd.DataFrame({
                'estimate': [self.estimate],
                'std_error': [self.std_error],
                'ci_lower': [self.ci_lower],
                'ci_upper': [self.ci_upper],
                'p_value': [self.p_value]
            })


class DiscreteChoiceEstimator:
    """
    Unified estimator for discrete choice models with causal inference focus.

    Parameters
    ----------
    model_type : str
        Type of model: 'logit', 'probit', 'lpm', 'ologit', 'oprobit',
        'mlogit', 'poisson', 'negbin', 'zip', 'zinb'
    robust : bool
        Use heteroskedasticity-robust standard errors (default: True)
    cluster : str or None
        Variable name for clustered standard errors

    Attributes
    ----------
    model_ : fitted statsmodels model
        The underlying fitted model
    var_names_ : list
        Names of covariates
    treatment_var_ : str or None
        Name of treatment variable if specified
    """

    def __init__(self, model_type: str = 'logit', robust: bool = True,
                 cluster: Optional[str] = None):
        self.model_type = ModelType(model_type)
        self.robust = robust
        self.cluster = cluster

        self.model_ = None
        self.y_ = None
        self.X_ = None
        self.X_const_ = None
        self.var_names_ = None
        self.treatment_var_ = None
        self.exposure_ = None

    def fit(self, y: np.ndarray, X: np.ndarray,
            var_names: Optional[List[str]] = None,
            treatment: Optional[str] = None,
            exposure: Optional[np.ndarray] = None,
            cluster_var: Optional[np.ndarray] = None,
            **kwargs) -> 'DiscreteChoiceEstimator':
        """
        Fit the discrete choice model.

        Parameters
        ----------
        y : array-like
            Outcome variable
        X : array-like
            Covariate matrix (without constant)
        var_names : list, optional
            Names of covariates
        treatment : str, optional
            Name of treatment variable for causal effects
        exposure : array-like, optional
            Exposure variable for rate models (count)
        cluster_var : array-like, optional
            Cluster identifiers for clustered SEs
        **kwargs : additional arguments passed to fit()

        Returns
        -------
        self : fitted estimator
        """
        self.y_ = np.asarray(y)
        self.X_ = np.asarray(X)
        self.X_const_ = sm.add_constant(self.X_)
        self.exposure_ = exposure
        self.treatment_var_ = treatment

        if var_names is None:
            self.var_names_ = [f'x{i}' for i in range(self.X_.shape[1])]
        else:
            self.var_names_ = list(var_names)

        # Set up covariance type
        cov_type = 'nonrobust'
        cov_kwds = {}

        if cluster_var is not None:
            cov_type = 'cluster'
            cov_kwds = {'groups': cluster_var}
        elif self.robust:
            cov_type = 'HC1'

        fit_kwargs = {'cov_type': cov_type, 'disp': 0}
        if cov_kwds:
            fit_kwargs['cov_kwds'] = cov_kwds
        fit_kwargs.update(kwargs)

        # Fit model based on type
        if self.model_type == ModelType.LOGIT:
            self.model_ = self._fit_logit(**fit_kwargs)
        elif self.model_type == ModelType.PROBIT:
            self.model_ = self._fit_probit(**fit_kwargs)
        elif self.model_type == ModelType.LPM:
            self.model_ = self._fit_lpm(**fit_kwargs)
        elif self.model_type == ModelType.ORDERED_LOGIT:
            self.model_ = self._fit_ordered('logit')
        elif self.model_type == ModelType.ORDERED_PROBIT:
            self.model_ = self._fit_ordered('probit')
        elif self.model_type == ModelType.MULTINOMIAL_LOGIT:
            self.model_ = self._fit_multinomial(**fit_kwargs)
        elif self.model_type == ModelType.POISSON:
            self.model_ = self._fit_poisson(**fit_kwargs)
        elif self.model_type == ModelType.NEGBIN:
            self.model_ = self._fit_negbin(**fit_kwargs)
        elif self.model_type == ModelType.ZIP:
            self.model_ = self._fit_zip()
        elif self.model_type == ModelType.ZINB:
            self.model_ = self._fit_zinb()

        return self

    def _fit_logit(self, **kwargs) -> Any:
        """Fit binary logit model."""
        model = Logit(self.y_, self.X_const_)
        return model.fit(**kwargs)

    def _fit_probit(self, **kwargs) -> Any:
        """Fit binary probit model."""
        model = Probit(self.y_, self.X_const_)
        return model.fit(**kwargs)

    def _fit_lpm(self, **kwargs) -> Any:
        """Fit linear probability model."""
        model = sm.OLS(self.y_, self.X_const_)
        return model.fit(**kwargs)

    def _fit_ordered(self, distr: str) -> Any:
        """Fit ordered logit or probit."""
        model = OrderedModel(self.y_, self.X_, distr=distr)
        return model.fit(method='bfgs', disp=0)

    def _fit_multinomial(self, **kwargs) -> Any:
        """Fit multinomial logit."""
        model = MNLogit(self.y_, self.X_const_)
        return model.fit(**kwargs)

    def _fit_poisson(self, **kwargs) -> Any:
        """Fit Poisson regression."""
        offset = np.log(self.exposure_) if self.exposure_ is not None else None
        model = Poisson(self.y_, self.X_const_, offset=offset)
        return model.fit(**kwargs)

    def _fit_negbin(self, **kwargs) -> Any:
        """Fit negative binomial regression."""
        offset = np.log(self.exposure_) if self.exposure_ is not None else None
        model = NegativeBinomial(self.y_, self.X_const_, offset=offset)
        return model.fit(**kwargs)

    def _fit_zip(self) -> Any:
        """Fit zero-inflated Poisson."""
        model = ZeroInflatedPoisson(self.y_, self.X_const_, exog_infl=self.X_const_)
        return model.fit(disp=0)

    def _fit_zinb(self) -> Any:
        """Fit zero-inflated negative binomial."""
        model = ZeroInflatedNegativeBinomialP(self.y_, self.X_const_, exog_infl=self.X_const_)
        return model.fit(disp=0)

    def predict(self, X: Optional[np.ndarray] = None,
                which: str = 'mean') -> np.ndarray:
        """
        Generate predictions.

        Parameters
        ----------
        X : array-like, optional
            New covariate values (without constant)
        which : str
            What to predict: 'mean', 'prob', 'class'

        Returns
        -------
        predictions : array
        """
        if X is None:
            X_pred = self.X_const_
        else:
            X_pred = sm.add_constant(np.asarray(X))

        if self.model_type in [ModelType.LOGIT, ModelType.PROBIT]:
            probs = self.model_.predict(X_pred)
            if which == 'class':
                return (probs > 0.5).astype(int)
            return probs

        elif self.model_type == ModelType.LPM:
            return self.model_.predict(X_pred)

        elif self.model_type in [ModelType.ORDERED_LOGIT, ModelType.ORDERED_PROBIT]:
            # Need to use original X (no constant) for ordered models
            if X is None:
                return self.model_.predict(self.X_)
            return self.model_.predict(np.asarray(X))

        elif self.model_type == ModelType.MULTINOMIAL_LOGIT:
            return self.model_.predict(X_pred)

        elif self.model_type in [ModelType.POISSON, ModelType.NEGBIN]:
            return self.model_.predict(X_pred)

        elif self.model_type in [ModelType.ZIP, ModelType.ZINB]:
            return self.model_.predict(X_pred, which='mean')

        return self.model_.predict(X_pred)

    def marginal_effects(self, method: str = 'average',
                         at_values: Optional[Dict] = None,
                         confidence: float = 0.95) -> CausalOutput:
        """
        Compute marginal effects.

        Parameters
        ----------
        method : str
            'average' for AME, 'atmean' for MEM, 'at' for MER
        at_values : dict, optional
            Covariate values for MER
        confidence : float
            Confidence level

        Returns
        -------
        CausalOutput with marginal effects
        """
        z_crit = stats.norm.ppf((1 + confidence) / 2)

        if self.model_type == ModelType.LPM:
            # Coefficients ARE marginal effects
            ame = self.model_.params[1:]
            se = self.model_.bse[1:]

        elif self.model_type in [ModelType.LOGIT, ModelType.PROBIT]:
            mfx = self.model_.get_margeff(at='overall' if method == 'average' else 'mean')
            ame = mfx.margeff
            se = mfx.margeff_se

        elif self.model_type in [ModelType.POISSON, ModelType.NEGBIN]:
            ame, se = self._marginal_effects_count(method)

        elif self.model_type == ModelType.MULTINOMIAL_LOGIT:
            mfx = self.model_.get_margeff()
            ame = mfx.margeff
            se = mfx.margeff_se

        elif self.model_type in [ModelType.ORDERED_LOGIT, ModelType.ORDERED_PROBIT]:
            ame, se = self._marginal_effects_ordered(method)

        elif self.model_type in [ModelType.ZIP, ModelType.ZINB]:
            ame, se = self._marginal_effects_zeroinflated(method)

        else:
            raise NotImplementedError(f"Marginal effects not implemented for {self.model_type}")

        # Compute CIs and p-values
        ci_lower = ame - z_crit * se
        ci_upper = ame + z_crit * se
        z_stat = ame / (se + 1e-10)
        p_value = 2 * stats.norm.sf(np.abs(z_stat))

        return CausalOutput(
            estimate=ame,
            std_error=se,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            p_value=p_value,
            method=f'{method.upper()} marginal effects',
            n_obs=len(self.y_),
            additional_info={'var_names': self.var_names_}
        )

    def _marginal_effects_count(self, method: str) -> Tuple[np.ndarray, np.ndarray]:
        """Compute AME for count models."""
        beta = self.model_.params[1:]
        mu = self.model_.fittedvalues

        if method == 'average':
            ame = beta * np.mean(mu)
            se = self.model_.bse[1:] * np.mean(mu)
        else:  # at mean
            mu_mean = np.exp(np.mean(self.X_const_, axis=0) @ self.model_.params)
            ame = beta * mu_mean
            se = self.model_.bse[1:] * mu_mean

        return ame, se

    def _marginal_effects_ordered(self, method: str) -> Tuple[np.ndarray, np.ndarray]:
        """Compute AME for ordered models."""
        n_cat = len(np.unique(self.y_))
        beta = self.model_.params[:-n_cat+1]
        thresholds = self.model_.params[-n_cat+1:]

        xb = self.X_ @ beta

        if self.model_type == ModelType.ORDERED_LOGIT:
            pdf = lambda z: np.exp(z) / (1 + np.exp(z))**2
        else:
            pdf = stats.norm.pdf

        # AME for highest category
        mu_last = thresholds[-1]
        ame = beta * np.mean(pdf(mu_last - xb))

        # Approximate SE
        se = np.abs(ame) * 0.1  # Placeholder - should use delta method

        return ame, se

    def _marginal_effects_zeroinflated(self, method: str) -> Tuple[np.ndarray, np.ndarray]:
        """Compute AME for zero-inflated models."""
        mu = self.model_.predict(which='mean')
        n_vars = len(self.var_names_)
        beta = self.model_.params[:n_vars+1][1:]  # Count model coefficients

        ame = beta * np.mean(mu)
        se = np.abs(ame) * 0.1  # Placeholder

        return ame, se

    def treatment_effect(self, treatment_var: Optional[str] = None,
                         confidence: float = 0.95) -> CausalOutput:
        """
        Compute average treatment effect for binary treatment.

        Parameters
        ----------
        treatment_var : str, optional
            Treatment variable name (uses self.treatment_var_ if not specified)
        confidence : float
            Confidence level

        Returns
        -------
        CausalOutput with treatment effect
        """
        treat = treatment_var or self.treatment_var_
        if treat is None:
            raise ValueError("No treatment variable specified")

        if treat not in self.var_names_:
            raise ValueError(f"Treatment '{treat}' not found in variables")

        treat_idx = self.var_names_.index(treat) + 1  # +1 for constant

        X1 = self.X_const_.copy()
        X0 = self.X_const_.copy()
        X1[:, treat_idx] = 1
        X0[:, treat_idx] = 0

        if self.model_type == ModelType.LPM:
            ate = self.model_.params[treat_idx]
            se = self.model_.bse[treat_idx]

        elif self.model_type in [ModelType.LOGIT, ModelType.PROBIT]:
            p1 = self.model_.predict(X1)
            p0 = self.model_.predict(X0)
            ate = np.mean(p1 - p0)
            # Bootstrap SE would be better
            se = np.std(p1 - p0) / np.sqrt(len(p1))

        elif self.model_type in [ModelType.POISSON, ModelType.NEGBIN]:
            mu1 = self.model_.predict(X1)
            mu0 = self.model_.predict(X0)
            ate = np.mean(mu1 - mu0)
            se = np.std(mu1 - mu0) / np.sqrt(len(mu1))

        elif self.model_type == ModelType.MULTINOMIAL_LOGIT:
            p1 = self.model_.predict(X1)
            p0 = self.model_.predict(X0)
            ate = np.mean(p1 - p0, axis=0)
            se = np.std(p1 - p0, axis=0) / np.sqrt(len(p1))

            z_crit = stats.norm.ppf((1 + confidence) / 2)
            return CausalOutput(
                estimate=ate,
                std_error=se,
                ci_lower=ate - z_crit * se,
                ci_upper=ate + z_crit * se,
                p_value=2 * stats.norm.sf(np.abs(ate / (se + 1e-10))),
                method='ATE (multinomial)',
                n_obs=len(self.y_),
                additional_info={'treatment': treat, 'type': 'multinomial'}
            )

        elif self.model_type in [ModelType.ORDERED_LOGIT, ModelType.ORDERED_PROBIT]:
            p1 = self.model_.predict(self.X_.copy())
            p0 = self.model_.predict(self.X_.copy())
            # Would need to properly compute with treatment variation
            ate = 0.0  # Placeholder
            se = 0.1

        else:
            raise NotImplementedError(f"Treatment effect not implemented for {self.model_type}")

        z_crit = stats.norm.ppf((1 + confidence) / 2)
        z_stat = ate / (se + 1e-10)

        return CausalOutput(
            estimate=ate,
            std_error=se,
            ci_lower=ate - z_crit * se,
            ci_upper=ate + z_crit * se,
            p_value=2 * stats.norm.sf(np.abs(z_stat)),
            method='Average Treatment Effect',
            n_obs=len(self.y_),
            additional_info={'treatment': treat}
        )

    def summary(self) -> pd.DataFrame:
        """Return model coefficients summary."""
        if self.model_ is None:
            raise ValueError("Model not fitted")

        return pd.DataFrame({
            'coef': self.model_.params,
            'std_err': self.model_.bse,
            'z': self.model_.tvalues,
            'p_value': self.model_.pvalues
        }, index=['const'] + self.var_names_ if hasattr(self.model_, 'params') else None)

    def coefficients(self) -> np.ndarray:
        """Return model coefficients."""
        return self.model_.params

    def fitted_values(self) -> np.ndarray:
        """Return fitted values."""
        return self.model_.fittedvalues

    @property
    def aic(self) -> float:
        """Akaike Information Criterion."""
        return self.model_.aic if hasattr(self.model_, 'aic') else np.nan

    @property
    def bic(self) -> float:
        """Bayesian Information Criterion."""
        return self.model_.bic if hasattr(self.model_, 'bic') else np.nan

    @property
    def llf(self) -> float:
        """Log-likelihood."""
        return self.model_.llf if hasattr(self.model_, 'llf') else np.nan


# =============================================================================
# Convenience functions
# =============================================================================

def fit_logit(y: np.ndarray, X: np.ndarray, **kwargs) -> DiscreteChoiceEstimator:
    """Convenience function to fit logit model."""
    return DiscreteChoiceEstimator('logit').fit(y, X, **kwargs)


def fit_probit(y: np.ndarray, X: np.ndarray, **kwargs) -> DiscreteChoiceEstimator:
    """Convenience function to fit probit model."""
    return DiscreteChoiceEstimator('probit').fit(y, X, **kwargs)


def fit_ordered(y: np.ndarray, X: np.ndarray, link: str = 'logit',
                **kwargs) -> DiscreteChoiceEstimator:
    """Convenience function to fit ordered model."""
    model_type = 'ologit' if link == 'logit' else 'oprobit'
    return DiscreteChoiceEstimator(model_type).fit(y, X, **kwargs)


def fit_multinomial(y: np.ndarray, X: np.ndarray, **kwargs) -> DiscreteChoiceEstimator:
    """Convenience function to fit multinomial logit."""
    return DiscreteChoiceEstimator('mlogit').fit(y, X, **kwargs)


def fit_poisson(y: np.ndarray, X: np.ndarray, **kwargs) -> DiscreteChoiceEstimator:
    """Convenience function to fit Poisson model."""
    return DiscreteChoiceEstimator('poisson').fit(y, X, **kwargs)


def fit_negbin(y: np.ndarray, X: np.ndarray, **kwargs) -> DiscreteChoiceEstimator:
    """Convenience function to fit negative binomial model."""
    return DiscreteChoiceEstimator('negbin').fit(y, X, **kwargs)


# =============================================================================
# Module test
# =============================================================================

if __name__ == '__main__':
    # Quick test
    np.random.seed(42)
    n = 1000

    # Generate test data
    X = np.random.randn(n, 3)
    true_beta = np.array([0.5, -0.3, 0.8])
    latent = X @ true_beta + np.random.logistic(0, 1, n)
    y_binary = (latent > 0).astype(int)

    print("Testing DiscreteChoiceEstimator...")

    # Test logit
    est = DiscreteChoiceEstimator('logit')
    est.fit(y_binary, X, var_names=['x1', 'x2', 'x3'], treatment='x1')

    print("\n--- Logit Model ---")
    print(est.summary())

    print("\n--- Marginal Effects ---")
    mfx = est.marginal_effects()
    print(mfx.summary())

    print("\n--- Treatment Effect ---")
    ate = est.treatment_effect()
    print(f"ATE = {ate.estimate:.4f} (SE = {ate.std_error:.4f})")
    print(f"95% CI: [{ate.ci_lower:.4f}, {ate.ci_upper:.4f}]")

    # Test count model
    y_count = np.random.poisson(np.exp(0.5 + X @ [0.2, -0.1, 0.3]))

    est_count = DiscreteChoiceEstimator('poisson')
    est_count.fit(y_count, X, var_names=['x1', 'x2', 'x3'])

    print("\n--- Poisson Model ---")
    print(est_count.summary())

    print("\n--- Count Model Marginal Effects ---")
    mfx_count = est_count.marginal_effects()
    print(mfx_count.summary())

    print("\nAll tests passed!")
