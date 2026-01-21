"""
Bayesian Econometrics Estimator

Provides Bayesian inference for econometric models using PyMC.
Includes regression, hierarchical models, and causal inference methods.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union, Any
import numpy as np
import pandas as pd
import warnings

try:
    import pymc as pm
    import arviz as az
    HAS_PYMC = True
except ImportError:
    HAS_PYMC = False
    warnings.warn("PyMC not installed. Install with: pip install pymc")


@dataclass
class CausalOutput:
    """Standardized output for causal effect estimates."""
    effect: float
    se: float
    ci_lower: float
    ci_upper: float
    p_value: Optional[float] = None
    method: str = ""
    details: Dict[str, Any] = field(default_factory=dict)

    def summary(self) -> str:
        """Return formatted summary string."""
        ci_str = f"[{self.ci_lower:.4f}, {self.ci_upper:.4f}]"
        return (
            f"Method: {self.method}\n"
            f"Effect: {self.effect:.4f} (SE: {self.se:.4f})\n"
            f"95% CI: {ci_str}"
        )


@dataclass
class BayesianResult:
    """Result container for Bayesian estimation."""
    trace: Any  # arviz.InferenceData
    summary: pd.DataFrame
    diagnostics: Dict[str, Any]
    causal_output: Optional[CausalOutput] = None


class BayesianEstimator:
    """
    Bayesian estimator for econometric models.

    Provides methods for:
    - Bayesian linear regression
    - Hierarchical/multilevel models
    - Bayesian causal inference
    - MCMC diagnostics

    Examples
    --------
    >>> estimator = BayesianEstimator()
    >>> result = estimator.fit_bayesian_regression(y, X)
    >>> print(result.summary)
    """

    def __init__(
        self,
        random_seed: int = 42,
        draws: int = 2000,
        tune: int = 1000,
        chains: int = 4,
        target_accept: float = 0.9
    ):
        """
        Initialize Bayesian estimator.

        Parameters
        ----------
        random_seed : int
            Random seed for reproducibility
        draws : int
            Number of posterior samples per chain
        tune : int
            Number of tuning (warmup) samples
        chains : int
            Number of MCMC chains
        target_accept : float
            Target acceptance rate for NUTS sampler
        """
        if not HAS_PYMC:
            raise ImportError("PyMC is required. Install with: pip install pymc")

        self.random_seed = random_seed
        self.draws = draws
        self.tune = tune
        self.chains = chains
        self.target_accept = target_accept

    def fit_bayesian_regression(
        self,
        y: np.ndarray,
        X: np.ndarray,
        prior_intercept_sigma: float = 10.0,
        prior_beta_sigma: float = 2.5,
        prior_sigma_sigma: float = 1.0,
        feature_names: Optional[List[str]] = None,
        standardize: bool = False
    ) -> BayesianResult:
        """
        Fit Bayesian linear regression.

        Parameters
        ----------
        y : array-like
            Outcome variable (n_samples,)
        X : array-like
            Feature matrix (n_samples, n_features)
        prior_intercept_sigma : float
            Prior SD for intercept
        prior_beta_sigma : float
            Prior SD for coefficients
        prior_sigma_sigma : float
            Prior SD for residual standard deviation
        feature_names : list, optional
            Names for features
        standardize : bool
            Whether to standardize X and y

        Returns
        -------
        BayesianResult
            Contains trace, summary, and diagnostics
        """
        y = np.asarray(y).flatten()
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        n_samples, n_features = X.shape

        # Standardization
        if standardize:
            y_mean, y_std = y.mean(), y.std()
            X_mean, X_std = X.mean(axis=0), X.std(axis=0)
            y_scaled = (y - y_mean) / y_std
            X_scaled = (X - X_mean) / X_std
        else:
            y_scaled, X_scaled = y, X

        # Feature names
        if feature_names is None:
            feature_names = [f"x{i}" for i in range(n_features)]

        with pm.Model() as model:
            # Priors
            alpha = pm.Normal("alpha", mu=0, sigma=prior_intercept_sigma)
            beta = pm.Normal("beta", mu=0, sigma=prior_beta_sigma, shape=n_features)
            sigma = pm.HalfNormal("sigma", sigma=prior_sigma_sigma)

            # Linear predictor
            mu = alpha + pm.math.dot(X_scaled, beta)

            # Likelihood
            y_obs = pm.Normal("y_obs", mu=mu, sigma=sigma, observed=y_scaled)

            # Sample
            trace = pm.sample(
                draws=self.draws,
                tune=self.tune,
                chains=self.chains,
                target_accept=self.target_accept,
                random_seed=self.random_seed,
                progressbar=True
            )

        # Get summary
        summary = az.summary(trace, var_names=["alpha", "beta", "sigma"])

        # Add feature names to summary index
        new_index = ["alpha"] + [f"beta[{name}]" for name in feature_names] + ["sigma"]
        if len(new_index) == len(summary):
            summary.index = new_index

        # Diagnostics
        diagnostics = self.mcmc_diagnostics(trace)

        return BayesianResult(
            trace=trace,
            summary=summary,
            diagnostics=diagnostics
        )

    def fit_hierarchical_model(
        self,
        y: np.ndarray,
        X: np.ndarray,
        groups: np.ndarray,
        varying_intercept: bool = True,
        varying_slope: bool = False,
        slope_var_idx: int = 0
    ) -> BayesianResult:
        """
        Fit hierarchical (multilevel) model.

        Parameters
        ----------
        y : array-like
            Outcome variable
        X : array-like
            Feature matrix
        groups : array-like
            Group indicators (0 to n_groups-1)
        varying_intercept : bool
            Include group-varying intercepts
        varying_slope : bool
            Include group-varying slopes
        slope_var_idx : int
            Index of variable with varying slope

        Returns
        -------
        BayesianResult
            Contains trace, summary, and diagnostics
        """
        y = np.asarray(y).flatten()
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        groups = np.asarray(groups).astype(int)

        n_groups = len(np.unique(groups))
        n_features = X.shape[1]

        with pm.Model() as model:
            # Population-level priors
            if varying_intercept:
                mu_alpha = pm.Normal("mu_alpha", mu=0, sigma=10)
                sigma_alpha = pm.HalfNormal("sigma_alpha", sigma=2)
                # Non-centered parameterization
                alpha_offset = pm.Normal("alpha_offset", mu=0, sigma=1, shape=n_groups)
                alpha = pm.Deterministic("alpha", mu_alpha + sigma_alpha * alpha_offset)
            else:
                alpha = pm.Normal("alpha", mu=0, sigma=10)

            # Fixed effects
            beta = pm.Normal("beta", mu=0, sigma=2.5, shape=n_features)

            # Varying slopes
            if varying_slope:
                mu_beta_v = pm.Normal("mu_beta_v", mu=0, sigma=2.5)
                sigma_beta_v = pm.HalfNormal("sigma_beta_v", sigma=1)
                beta_v_offset = pm.Normal("beta_v_offset", mu=0, sigma=1, shape=n_groups)
                beta_varying = pm.Deterministic(
                    "beta_varying",
                    mu_beta_v + sigma_beta_v * beta_v_offset
                )

            # Residual
            sigma = pm.HalfNormal("sigma", sigma=1)

            # Linear predictor
            if varying_intercept:
                mu = alpha[groups] + pm.math.dot(X, beta)
            else:
                mu = alpha + pm.math.dot(X, beta)

            if varying_slope:
                mu = mu + beta_varying[groups] * X[:, slope_var_idx]

            # Likelihood
            y_obs = pm.Normal("y_obs", mu=mu, sigma=sigma, observed=y)

            # Sample
            trace = pm.sample(
                draws=self.draws,
                tune=self.tune,
                chains=self.chains,
                target_accept=self.target_accept,
                random_seed=self.random_seed
            )

        # Summary
        var_names = ["beta", "sigma"]
        if varying_intercept:
            var_names = ["mu_alpha", "sigma_alpha"] + var_names
        if varying_slope:
            var_names = ["mu_beta_v", "sigma_beta_v"] + var_names

        summary = az.summary(trace, var_names=var_names)
        diagnostics = self.mcmc_diagnostics(trace)

        return BayesianResult(
            trace=trace,
            summary=summary,
            diagnostics=diagnostics
        )

    def fit_bayesian_ate(
        self,
        y: np.ndarray,
        treatment: np.ndarray,
        X: Optional[np.ndarray] = None,
        prior_ate_sigma: float = 2.5
    ) -> BayesianResult:
        """
        Estimate Average Treatment Effect using Bayesian regression.

        Parameters
        ----------
        y : array-like
            Outcome variable
        treatment : array-like
            Binary treatment indicator
        X : array-like, optional
            Covariates for adjustment
        prior_ate_sigma : float
            Prior SD for ATE

        Returns
        -------
        BayesianResult
            Contains trace, summary, diagnostics, and CausalOutput
        """
        y = np.asarray(y).flatten()
        treatment = np.asarray(treatment).flatten()

        with pm.Model() as model:
            # Priors
            alpha = pm.Normal("alpha", mu=0, sigma=10)
            ate = pm.Normal("ate", mu=0, sigma=prior_ate_sigma)
            sigma = pm.HalfNormal("sigma", sigma=1)

            # Covariates
            if X is not None:
                X = np.asarray(X)
                if X.ndim == 1:
                    X = X.reshape(-1, 1)
                beta = pm.Normal("beta", mu=0, sigma=2.5, shape=X.shape[1])
                mu = alpha + ate * treatment + pm.math.dot(X, beta)
            else:
                mu = alpha + ate * treatment

            # Likelihood
            y_obs = pm.Normal("y_obs", mu=mu, sigma=sigma, observed=y)

            # Sample
            trace = pm.sample(
                draws=self.draws,
                tune=self.tune,
                chains=self.chains,
                target_accept=self.target_accept,
                random_seed=self.random_seed
            )

        # Summary
        summary = az.summary(trace, var_names=["alpha", "ate", "sigma"])
        diagnostics = self.mcmc_diagnostics(trace)

        # Create CausalOutput
        ate_samples = trace.posterior["ate"].values.flatten()
        hdi = az.hdi(ate_samples, hdi_prob=0.95)

        causal_output = CausalOutput(
            effect=float(ate_samples.mean()),
            se=float(ate_samples.std()),
            ci_lower=float(hdi[0]),
            ci_upper=float(hdi[1]),
            p_value=None,  # Bayesian doesn't use p-values
            method="Bayesian ATE",
            details={
                "posterior_median": float(np.median(ate_samples)),
                "prob_positive": float((ate_samples > 0).mean()),
                "prob_negative": float((ate_samples < 0).mean()),
                "hdi_prob": 0.95,
                "n_samples": len(ate_samples)
            }
        )

        return BayesianResult(
            trace=trace,
            summary=summary,
            diagnostics=diagnostics,
            causal_output=causal_output
        )

    def mcmc_diagnostics(
        self,
        trace,
        var_names: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Compute comprehensive MCMC diagnostics.

        Parameters
        ----------
        trace : arviz.InferenceData
            Posterior samples
        var_names : list, optional
            Variables to check

        Returns
        -------
        dict
            Diagnostic statistics and flags
        """
        diagnostics = {}

        # Rhat
        rhat = az.rhat(trace, var_names=var_names)
        rhat_values = []
        for k, v in rhat.items():
            if hasattr(v, 'values'):
                rhat_values.extend(v.values.flatten())
            else:
                rhat_values.append(float(v))
        diagnostics["rhat_max"] = max(rhat_values)
        diagnostics["rhat_ok"] = diagnostics["rhat_max"] < 1.01

        # ESS
        ess_bulk = az.ess(trace, var_names=var_names, method="bulk")
        ess_tail = az.ess(trace, var_names=var_names, method="tail")

        bulk_values = []
        tail_values = []
        for k, v in ess_bulk.items():
            if hasattr(v, 'values'):
                bulk_values.extend(v.values.flatten())
            else:
                bulk_values.append(float(v))
        for k, v in ess_tail.items():
            if hasattr(v, 'values'):
                tail_values.extend(v.values.flatten())
            else:
                tail_values.append(float(v))

        diagnostics["ess_bulk_min"] = min(bulk_values)
        diagnostics["ess_tail_min"] = min(tail_values)
        diagnostics["ess_ok"] = (
            diagnostics["ess_bulk_min"] > 400 and
            diagnostics["ess_tail_min"] > 400
        )

        # Divergences
        if hasattr(trace, 'sample_stats') and 'diverging' in trace.sample_stats:
            n_divergences = int(trace.sample_stats.diverging.sum().values)
            diagnostics["n_divergences"] = n_divergences
            diagnostics["divergences_ok"] = n_divergences == 0
        else:
            diagnostics["n_divergences"] = 0
            diagnostics["divergences_ok"] = True

        # Overall
        diagnostics["all_ok"] = (
            diagnostics["rhat_ok"] and
            diagnostics["ess_ok"] and
            diagnostics["divergences_ok"]
        )

        return diagnostics

    def posterior_summary(
        self,
        trace,
        var_names: Optional[List[str]] = None,
        hdi_prob: float = 0.95
    ) -> pd.DataFrame:
        """
        Generate posterior summary with HDI.

        Parameters
        ----------
        trace : arviz.InferenceData
            Posterior samples
        var_names : list, optional
            Variables to summarize
        hdi_prob : float
            Probability for HDI

        Returns
        -------
        pd.DataFrame
            Summary statistics
        """
        return az.summary(trace, var_names=var_names, hdi_prob=hdi_prob)

    def prior_predictive_check(
        self,
        model,
        samples: int = 500
    ) -> Any:
        """
        Generate prior predictive samples.

        Parameters
        ----------
        model : pm.Model
            PyMC model
        samples : int
            Number of samples

        Returns
        -------
        arviz.InferenceData
            Prior predictive samples
        """
        with model:
            prior_pred = pm.sample_prior_predictive(samples=samples)
        return prior_pred

    def posterior_predictive_check(
        self,
        model,
        trace
    ) -> Any:
        """
        Generate posterior predictive samples.

        Parameters
        ----------
        model : pm.Model
            PyMC model
        trace : arviz.InferenceData
            Posterior samples

        Returns
        -------
        arviz.InferenceData
            Posterior predictive samples
        """
        with model:
            ppc = pm.sample_posterior_predictive(trace)
        return ppc

    def model_comparison(
        self,
        traces: Dict[str, Any],
        ic: str = "loo"
    ) -> pd.DataFrame:
        """
        Compare multiple models using information criteria.

        Parameters
        ----------
        traces : dict
            Dictionary of {model_name: trace}
        ic : str
            Information criterion ("loo" or "waic")

        Returns
        -------
        pd.DataFrame
            Model comparison table
        """
        return az.compare(traces, ic=ic)


def fit_bayesian_regression(
    y: np.ndarray,
    X: np.ndarray,
    prior_scale: float = 2.5,
    draws: int = 2000,
    tune: int = 1000,
    random_seed: int = 42
) -> BayesianResult:
    """
    Convenience function for Bayesian regression.

    Parameters
    ----------
    y : array-like
        Outcome variable
    X : array-like
        Features
    prior_scale : float
        Prior SD for coefficients
    draws : int
        Posterior samples
    tune : int
        Tuning samples
    random_seed : int
        Random seed

    Returns
    -------
    BayesianResult
        Estimation results
    """
    estimator = BayesianEstimator(
        draws=draws,
        tune=tune,
        random_seed=random_seed
    )
    return estimator.fit_bayesian_regression(
        y, X, prior_beta_sigma=prior_scale
    )


def fit_hierarchical_model(
    y: np.ndarray,
    X: np.ndarray,
    groups: np.ndarray,
    draws: int = 2000,
    random_seed: int = 42
) -> BayesianResult:
    """
    Convenience function for hierarchical model.

    Parameters
    ----------
    y : array-like
        Outcome
    X : array-like
        Features
    groups : array-like
        Group indicators
    draws : int
        Posterior samples
    random_seed : int
        Random seed

    Returns
    -------
    BayesianResult
        Estimation results
    """
    estimator = BayesianEstimator(draws=draws, random_seed=random_seed)
    return estimator.fit_hierarchical_model(y, X, groups)


def mcmc_diagnostics(trace, var_names: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Convenience function for MCMC diagnostics.

    Parameters
    ----------
    trace : arviz.InferenceData
        Posterior samples
    var_names : list, optional
        Variables to check

    Returns
    -------
    dict
        Diagnostic statistics
    """
    estimator = BayesianEstimator()
    return estimator.mcmc_diagnostics(trace, var_names)


def posterior_summary(
    trace,
    var_names: Optional[List[str]] = None,
    hdi_prob: float = 0.95
) -> pd.DataFrame:
    """
    Convenience function for posterior summary.

    Parameters
    ----------
    trace : arviz.InferenceData
        Posterior samples
    var_names : list, optional
        Variables to summarize
    hdi_prob : float
        HDI probability

    Returns
    -------
    pd.DataFrame
        Summary statistics
    """
    return az.summary(trace, var_names=var_names, hdi_prob=hdi_prob)


if __name__ == "__main__":
    # Example usage
    np.random.seed(42)

    # Generate sample data
    n = 200
    X = np.random.randn(n, 3)
    true_beta = np.array([0.5, -0.3, 0.8])
    y = 2.0 + X @ true_beta + np.random.randn(n) * 0.5

    # Fit model
    estimator = BayesianEstimator(draws=1000, tune=500)
    result = estimator.fit_bayesian_regression(
        y, X,
        feature_names=["education", "experience", "ability"]
    )

    print("=== Bayesian Regression Results ===")
    print(result.summary)
    print("\n=== Diagnostics ===")
    for k, v in result.diagnostics.items():
        print(f"{k}: {v}")
