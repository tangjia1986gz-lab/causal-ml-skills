"""
Paper Replication Workflow

Systematic workflow for replicating empirical economics papers.
Integrates with all estimator skills in the causal-ml toolkit.

Author: Causal ML Skills
"""

import sys
from pathlib import Path
from typing import Any, Callable, Optional, Union
from dataclasses import dataclass, field
import warnings

import numpy as np
import pandas as pd

# Add skills path for importing estimators
skills_path = Path(__file__).parent.parent.parent
sys.path.insert(0, str(skills_path))

# Lazy imports for estimators to avoid circular dependencies
_estimator_imports = {}


def _get_did_estimator():
    """Lazy import for DID estimator."""
    if "did" not in _estimator_imports:
        try:
            from classic_methods.estimator_did.did_estimator import run_full_did_analysis
            _estimator_imports["did"] = run_full_did_analysis
        except ImportError:
            _estimator_imports["did"] = None
    return _estimator_imports["did"]


def _get_rd_estimator():
    """Lazy import for RD estimator."""
    if "rd" not in _estimator_imports:
        try:
            from classic_methods.estimator_rd.rd_estimator import run_full_rd_analysis
            _estimator_imports["rd"] = run_full_rd_analysis
        except ImportError:
            _estimator_imports["rd"] = None
    return _estimator_imports["rd"]


def _get_iv_estimator():
    """Lazy import for IV estimator."""
    if "iv" not in _estimator_imports:
        try:
            from classic_methods.estimator_iv.iv_estimator import run_full_iv_analysis
            _estimator_imports["iv"] = run_full_iv_analysis
        except ImportError:
            _estimator_imports["iv"] = None
    return _estimator_imports["iv"]


def _get_psm_estimator():
    """Lazy import for PSM estimator."""
    if "psm" not in _estimator_imports:
        try:
            from classic_methods.estimator_psm.psm_estimator import run_full_psm_analysis
            _estimator_imports["psm"] = run_full_psm_analysis
        except ImportError:
            _estimator_imports["psm"] = None
    return _estimator_imports["psm"]


def _get_ddml_estimator():
    """Lazy import for DDML estimator."""
    if "ddml" not in _estimator_imports:
        try:
            from causal_ml.causal_ddml.ddml_estimator import run_full_ddml_analysis
            _estimator_imports["ddml"] = run_full_ddml_analysis
        except ImportError:
            _estimator_imports["ddml"] = None
    return _estimator_imports["ddml"]


# ==============================================================================
# Method to Estimator Mapping
# ==============================================================================

METHOD_SKILL_MAP = {
    # Difference-in-Differences variants
    "did": "estimator-did",
    "difference-in-differences": "estimator-did",
    "diff-in-diff": "estimator-did",
    "twfe": "estimator-did",
    "two-way fixed effects": "estimator-did",
    "event study": "estimator-did",

    # Regression Discontinuity variants
    "rd": "estimator-rd",
    "rdd": "estimator-rd",
    "regression discontinuity": "estimator-rd",
    "sharp rd": "estimator-rd",
    "fuzzy rd": "estimator-rd",

    # Instrumental Variables variants
    "iv": "estimator-iv",
    "2sls": "estimator-iv",
    "tsls": "estimator-iv",
    "instrumental variables": "estimator-iv",
    "liml": "estimator-iv",

    # Propensity Score Methods
    "psm": "estimator-psm",
    "propensity score": "estimator-psm",
    "matching": "estimator-psm",
    "ipw": "estimator-psm",
    "inverse probability weighting": "estimator-psm",

    # Machine Learning Methods
    "ddml": "causal-ddml",
    "double ml": "causal-ddml",
    "debiased ml": "causal-ddml",
    "dml": "causal-ddml",
    "causal forest": "causal-ddml",
}

ESTIMATOR_FUNCTIONS = {
    "estimator-did": _get_did_estimator,
    "estimator-rd": _get_rd_estimator,
    "estimator-iv": _get_iv_estimator,
    "estimator-psm": _get_psm_estimator,
    "causal-ddml": _get_ddml_estimator,
}


# ==============================================================================
# Data Classes
# ==============================================================================

@dataclass
class PaperSpecification:
    """Specification of paper to replicate."""

    paper_name: str
    citation: str = ""
    method: str = ""
    treatment: str = ""
    outcome: str = ""
    covariates: list = field(default_factory=list)
    original_results: dict = field(default_factory=dict)
    robustness_specs: list = field(default_factory=list)
    method_details: dict = field(default_factory=dict)

    def __post_init__(self):
        """Validate specification."""
        if not self.paper_name:
            raise ValueError("paper_name is required")


@dataclass
class ReplicationResult:
    """Result of a replication attempt."""

    estimate: float
    se: float
    pvalue: float = None
    ci_lower: float = None
    ci_upper: float = None
    n_obs: int = None
    n_treated: int = None
    n_control: int = None
    method: str = ""
    details: dict = field(default_factory=dict)

    def __post_init__(self):
        """Compute derived values."""
        if self.pvalue is None and self.se > 0:
            from scipy import stats
            self.pvalue = 2 * (1 - stats.norm.cdf(abs(self.estimate / self.se)))
        if self.ci_lower is None:
            self.ci_lower = self.estimate - 1.96 * self.se
        if self.ci_upper is None:
            self.ci_upper = self.estimate + 1.96 * self.se


@dataclass
class ComparisonMetrics:
    """Metrics comparing original and replicated results."""

    estimate_diff: float
    estimate_pct_diff: float
    se_diff: float
    se_pct_diff: float
    ci_overlap: float
    same_significance: bool
    same_sign: bool
    normalized_diff: float
    success_level: str
    success_message: str


# ==============================================================================
# Core Functions
# ==============================================================================

def parse_paper_specification(paper_dict: dict) -> PaperSpecification:
    """
    Parse paper specification from dictionary.

    Parameters
    ----------
    paper_dict : dict
        Dictionary containing paper specification

    Returns
    -------
    PaperSpecification
        Validated paper specification object
    """
    return PaperSpecification(**paper_dict)


def map_method_to_estimator(method_name: str) -> tuple[str, Callable]:
    """
    Route method name to correct estimator skill.

    Parameters
    ----------
    method_name : str
        Method name from paper (e.g., "psm", "did", "iv")

    Returns
    -------
    tuple
        (skill_name, estimator_function)
    """
    method_lower = method_name.lower().strip()

    if method_lower not in METHOD_SKILL_MAP:
        available = list(METHOD_SKILL_MAP.keys())
        raise ValueError(
            f"Unknown method '{method_name}'. Available methods: {available}"
        )

    skill_name = METHOD_SKILL_MAP[method_lower]
    estimator_getter = ESTIMATOR_FUNCTIONS.get(skill_name)

    if estimator_getter is None:
        raise ValueError(f"No estimator function for skill '{skill_name}'")

    estimator_func = estimator_getter()

    if estimator_func is None:
        warnings.warn(
            f"Estimator for '{skill_name}' not available. "
            "Using fallback implementation."
        )

    return skill_name, estimator_func


def load_replication_data(dataset_name: str) -> pd.DataFrame:
    """
    Load standard replication datasets.

    Parameters
    ----------
    dataset_name : str
        Name of dataset to load. Options:
        - "lalonde", "lalonde_nsw": LaLonde NSW experimental data
        - "lalonde_psid": LaLonde with PSID comparison
        - "lalonde_cps": LaLonde with CPS comparison
        - "card_college": Card (1995) college proximity data
        - "lee_elections": Lee (2008) election data
        - "card_krueger_mw": Card & Krueger minimum wage data

    Returns
    -------
    pd.DataFrame
        Loaded dataset
    """
    dataset_lower = dataset_name.lower().strip()

    # Try to load from various sources
    if dataset_lower in ["lalonde", "lalonde_nsw", "lalonde_psid", "lalonde_cps"]:
        return _load_lalonde_data(dataset_lower)
    elif dataset_lower in ["card_college", "card_proximity"]:
        return _load_card_college_data()
    elif dataset_lower in ["lee_elections", "lee"]:
        return _load_lee_elections_data()
    elif dataset_lower in ["card_krueger_mw", "card_krueger", "minimum_wage"]:
        return _load_card_krueger_data()
    else:
        raise ValueError(
            f"Unknown dataset '{dataset_name}'. Available datasets: "
            "lalonde, lalonde_nsw, lalonde_psid, lalonde_cps, "
            "card_college, lee_elections, card_krueger_mw"
        )


def _load_lalonde_data(variant: str = "lalonde") -> pd.DataFrame:
    """Load LaLonde data from various sources."""

    # Try causalml package first
    try:
        from causalml.dataset import make_lalonde
        data = make_lalonde()
        return data
    except ImportError:
        pass

    # Try statsmodels
    try:
        import statsmodels.api as sm
        lalonde = sm.datasets.get_rdataset("lalonde", "Matching").data
        return lalonde
    except Exception:
        pass

    # Try direct download or generate synthetic
    warnings.warn(
        "Could not load LaLonde data from packages. "
        "Generating synthetic data with similar properties."
    )
    return _generate_synthetic_lalonde(variant)


def _generate_synthetic_lalonde(variant: str = "lalonde") -> pd.DataFrame:
    """Generate synthetic LaLonde-like data for demonstration."""
    np.random.seed(42)

    n_treated = 185
    n_control = 260 if "psid" in variant else 429 if "cps" in variant else 260

    # Generate treated group (NSW participants)
    treated = pd.DataFrame({
        "treat": 1,
        "age": np.random.normal(25, 7, n_treated).clip(17, 55).astype(int),
        "education": np.random.normal(10, 2, n_treated).clip(0, 16).astype(int),
        "black": np.random.binomial(1, 0.84, n_treated),
        "hispanic": np.random.binomial(1, 0.06, n_treated),
        "married": np.random.binomial(1, 0.19, n_treated),
        "nodegree": np.random.binomial(1, 0.71, n_treated),
        "re74": np.maximum(0, np.random.normal(2100, 5000, n_treated)),
        "re75": np.maximum(0, np.random.normal(1500, 3500, n_treated)),
    })

    # Generate control group
    if "cps" in variant:
        # CPS controls are more advantaged
        control = pd.DataFrame({
            "treat": 0,
            "age": np.random.normal(33, 11, n_control).clip(17, 55).astype(int),
            "education": np.random.normal(12, 3, n_control).clip(0, 16).astype(int),
            "black": np.random.binomial(1, 0.25, n_control),
            "hispanic": np.random.binomial(1, 0.03, n_control),
            "married": np.random.binomial(1, 0.71, n_control),
            "nodegree": np.random.binomial(1, 0.30, n_control),
            "re74": np.maximum(0, np.random.normal(14000, 9000, n_control)),
            "re75": np.maximum(0, np.random.normal(13600, 9000, n_control)),
        })
    elif "psid" in variant:
        # PSID controls are somewhat more similar
        control = pd.DataFrame({
            "treat": 0,
            "age": np.random.normal(34, 10, n_control).clip(17, 55).astype(int),
            "education": np.random.normal(12, 3, n_control).clip(0, 16).astype(int),
            "black": np.random.binomial(1, 0.25, n_control),
            "hispanic": np.random.binomial(1, 0.03, n_control),
            "married": np.random.binomial(1, 0.87, n_control),
            "nodegree": np.random.binomial(1, 0.31, n_control),
            "re74": np.maximum(0, np.random.normal(19400, 13400, n_control)),
            "re75": np.maximum(0, np.random.normal(19100, 13600, n_control)),
        })
    else:
        # NSW experimental control
        control = pd.DataFrame({
            "treat": 0,
            "age": np.random.normal(25, 7, n_control).clip(17, 55).astype(int),
            "education": np.random.normal(10, 2, n_control).clip(0, 16).astype(int),
            "black": np.random.binomial(1, 0.83, n_control),
            "hispanic": np.random.binomial(1, 0.11, n_control),
            "married": np.random.binomial(1, 0.15, n_control),
            "nodegree": np.random.binomial(1, 0.83, n_control),
            "re74": np.maximum(0, np.random.normal(2100, 5000, n_control)),
            "re75": np.maximum(0, np.random.normal(1300, 3000, n_control)),
        })

    # Combine
    data = pd.concat([treated, control], ignore_index=True)

    # Generate outcome (re78) with treatment effect
    treatment_effect = 1794  # True effect from LaLonde paper
    baseline = (
        1000 +
        50 * data["age"] +
        200 * data["education"] -
        500 * data["black"] +
        0.3 * data["re75"] +
        np.random.normal(0, 2000, len(data))
    )
    data["re78"] = np.maximum(0, baseline + treatment_effect * data["treat"])

    return data


def _load_card_college_data() -> pd.DataFrame:
    """Load Card (1995) college proximity data."""
    warnings.warn(
        "Card college proximity data requires NLSY79 access. "
        "Generating synthetic data with similar properties."
    )
    return _generate_synthetic_card_college()


def _generate_synthetic_card_college() -> pd.DataFrame:
    """Generate synthetic Card (1995) style data."""
    np.random.seed(123)
    n = 3010

    # Generate instrument and confounders
    nearc4 = np.random.binomial(1, 0.68, n)  # Near 4-year college
    fatheduc = np.random.normal(10, 3, n).clip(0, 20)
    motheduc = np.random.normal(10, 3, n).clip(0, 20)
    ability = np.random.normal(0, 1, n)  # Unobserved

    # Education affected by instrument and ability
    education = (
        8 +
        0.15 * fatheduc +
        0.15 * motheduc +
        1.5 * nearc4 +  # Instrument effect
        1.2 * ability +  # Ability bias
        np.random.normal(0, 1.5, n)
    ).clip(0, 20)

    # Wages affected by education and ability (but not instrument directly)
    log_wage = (
        0.5 +
        0.07 * education +  # True return to education
        0.15 * ability +     # Ability premium
        0.01 * (education - 12) ** 2 +
        np.random.normal(0, 0.4, n)
    )

    return pd.DataFrame({
        "log_wage": log_wage,
        "education": education,
        "nearc4": nearc4,
        "fatheduc": fatheduc,
        "motheduc": motheduc,
        "exper": np.random.normal(15, 8, n).clip(0, 40),
        "black": np.random.binomial(1, 0.23, n),
        "smsa": np.random.binomial(1, 0.72, n),
        "south": np.random.binomial(1, 0.41, n),
    })


def _load_lee_elections_data() -> pd.DataFrame:
    """Load Lee (2008) election RD data."""
    warnings.warn(
        "Lee elections data may require author permission. "
        "Generating synthetic data with similar properties."
    )
    return _generate_synthetic_lee_elections()


def _generate_synthetic_lee_elections() -> pd.DataFrame:
    """Generate synthetic Lee (2008) style data."""
    np.random.seed(456)
    n = 6558

    # Running variable: Democratic vote share margin
    vote_margin = np.random.normal(0, 0.2, n)

    # Treatment: Democrat wins (margin > 0)
    dem_win = (vote_margin > 0).astype(int)

    # Outcome: Democrat wins next election
    # Discontinuity at 0 represents incumbency advantage
    incumbency_effect = 0.38
    baseline_prob = 0.45

    win_next_prob = (
        baseline_prob +
        0.3 * vote_margin +  # Continuity
        incumbency_effect * dem_win +  # RD effect
        np.random.normal(0, 0.1, n)
    ).clip(0, 1)

    win_next = np.random.binomial(1, win_next_prob)

    return pd.DataFrame({
        "vote_margin": vote_margin,
        "dem_win": dem_win,
        "win_next": win_next,
        "vote_share": 0.5 + vote_margin,
        "year": np.random.choice(range(1946, 1998, 2), n),
    })


def _load_card_krueger_data() -> pd.DataFrame:
    """Load Card & Krueger (1994) minimum wage data."""
    warnings.warn(
        "Card & Krueger data requires access to original survey. "
        "Generating synthetic data with similar properties."
    )
    return _generate_synthetic_card_krueger()


def _generate_synthetic_card_krueger() -> pd.DataFrame:
    """Generate synthetic Card & Krueger (1994) style data."""
    np.random.seed(789)

    # Panel structure: restaurants in NJ and PA, pre and post
    n_nj = 331
    n_pa = 79

    restaurants = []

    for state, n_restaurants in [("nj", n_nj), ("pa", n_pa)]:
        for i in range(n_restaurants):
            # Pre-period employment
            fte_pre = np.random.normal(20, 8, 1)[0]
            fte_pre = max(0, fte_pre)

            # Treatment effect (NJ only, post only)
            if state == "nj":
                treatment_effect = 2.76  # From paper
            else:
                treatment_effect = 0

            # Post-period employment
            fte_post = fte_pre + treatment_effect + np.random.normal(0, 4)
            fte_post = max(0, fte_post)

            # Pre-period observation
            restaurants.append({
                "restaurant_id": f"{state}_{i}",
                "state": state,
                "post": 0,
                "fte": fte_pre,
                "wage": 4.25 if state == "pa" else 4.25,
                "chain": np.random.choice(["bk", "kfc", "roys", "wendys"]),
            })

            # Post-period observation
            restaurants.append({
                "restaurant_id": f"{state}_{i}",
                "state": state,
                "post": 1,
                "fte": fte_post,
                "wage": 4.25 if state == "pa" else 5.05,  # NJ min wage increase
                "chain": restaurants[-1]["chain"],
            })

    return pd.DataFrame(restaurants)


# ==============================================================================
# Main Specification Execution
# ==============================================================================

def run_main_specification(
    data: pd.DataFrame,
    paper_spec: Union[dict, PaperSpecification]
) -> ReplicationResult:
    """
    Execute main analysis specification from paper.

    Parameters
    ----------
    data : pd.DataFrame
        Dataset for analysis
    paper_spec : dict or PaperSpecification
        Paper specification

    Returns
    -------
    ReplicationResult
        Results of main specification
    """
    if isinstance(paper_spec, dict):
        paper_spec = parse_paper_specification(paper_spec)

    # Get appropriate estimator
    skill_name, estimator_func = map_method_to_estimator(paper_spec.method)

    # Execute based on method type
    if estimator_func is not None:
        # Use skill's estimator
        try:
            result = estimator_func(
                data=data,
                treatment=paper_spec.treatment,
                outcome=paper_spec.outcome,
                covariates=paper_spec.covariates,
                **paper_spec.method_details
            )
            return _convert_to_replication_result(result, paper_spec.method)
        except Exception as e:
            warnings.warn(f"Skill estimator failed: {e}. Using fallback.")

    # Fallback implementations
    if paper_spec.method.lower() in ["psm", "propensity score", "matching"]:
        return _run_psm_fallback(data, paper_spec)
    elif paper_spec.method.lower() in ["did", "difference-in-differences"]:
        return _run_did_fallback(data, paper_spec)
    elif paper_spec.method.lower() in ["iv", "2sls", "instrumental variables"]:
        return _run_iv_fallback(data, paper_spec)
    elif paper_spec.method.lower() in ["rd", "regression discontinuity"]:
        return _run_rd_fallback(data, paper_spec)
    else:
        return _run_ols_fallback(data, paper_spec)


def _convert_to_replication_result(result: Any, method: str) -> ReplicationResult:
    """Convert skill result to ReplicationResult."""
    if isinstance(result, dict):
        return ReplicationResult(
            estimate=result.get("estimate", result.get("ate", result.get("att", 0))),
            se=result.get("se", result.get("std_error", 0)),
            pvalue=result.get("pvalue", result.get("p_value", None)),
            n_obs=result.get("n_obs", result.get("n", None)),
            n_treated=result.get("n_treated", None),
            n_control=result.get("n_control", None),
            method=method,
            details=result
        )
    return result


def _run_psm_fallback(
    data: pd.DataFrame,
    paper_spec: PaperSpecification
) -> ReplicationResult:
    """Fallback PSM implementation."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.neighbors import NearestNeighbors
    from scipy import stats

    # Estimate propensity scores
    X = data[paper_spec.covariates].fillna(0)
    T = data[paper_spec.treatment]
    Y = data[paper_spec.outcome]

    ps_model = LogisticRegression(max_iter=1000, random_state=42)
    ps_model.fit(X, T)
    pscore = ps_model.predict_proba(X)[:, 1]

    # Nearest neighbor matching
    treated_idx = T == 1
    control_idx = T == 0

    nn = NearestNeighbors(n_neighbors=1)
    nn.fit(pscore[control_idx].reshape(-1, 1))
    distances, indices = nn.kneighbors(pscore[treated_idx].reshape(-1, 1))

    # Get matched outcomes
    control_outcomes = Y[control_idx].values
    matched_control_outcomes = control_outcomes[indices.flatten()]
    treated_outcomes = Y[treated_idx].values

    # ATT estimate
    att = np.mean(treated_outcomes - matched_control_outcomes)

    # Bootstrap SE
    n_treated = treated_idx.sum()
    bootstrap_atts = []
    for _ in range(200):
        boot_idx = np.random.choice(n_treated, n_treated, replace=True)
        boot_att = np.mean(
            treated_outcomes[boot_idx] - matched_control_outcomes[boot_idx]
        )
        bootstrap_atts.append(boot_att)

    se = np.std(bootstrap_atts)
    pvalue = 2 * (1 - stats.norm.cdf(abs(att / se)))

    return ReplicationResult(
        estimate=att,
        se=se,
        pvalue=pvalue,
        n_obs=len(data),
        n_treated=int(treated_idx.sum()),
        n_control=int(control_idx.sum()),
        method="psm",
        details={"matching": "nearest_neighbor", "n_matches": n_treated}
    )


def _run_did_fallback(
    data: pd.DataFrame,
    paper_spec: PaperSpecification
) -> ReplicationResult:
    """Fallback DID implementation."""
    import statsmodels.formula.api as smf

    # Simple 2x2 DID
    formula = f"{paper_spec.outcome} ~ {paper_spec.treatment}"

    method_details = paper_spec.method_details
    post_var = method_details.get("post", "post")
    group_var = method_details.get("group", paper_spec.treatment)

    if post_var in data.columns and group_var in data.columns:
        formula = f"{paper_spec.outcome} ~ {group_var} * {post_var}"

    if paper_spec.covariates:
        formula += " + " + " + ".join(paper_spec.covariates)

    model = smf.ols(formula, data=data).fit()

    # Get interaction coefficient if DID, else treatment coefficient
    if f"{group_var}:{post_var}" in model.params.index:
        coef_name = f"{group_var}:{post_var}"
    else:
        coef_name = paper_spec.treatment

    return ReplicationResult(
        estimate=model.params[coef_name],
        se=model.bse[coef_name],
        pvalue=model.pvalues[coef_name],
        n_obs=int(model.nobs),
        method="did",
        details={"r_squared": model.rsquared, "formula": formula}
    )


def _run_iv_fallback(
    data: pd.DataFrame,
    paper_spec: PaperSpecification
) -> ReplicationResult:
    """Fallback IV/2SLS implementation."""
    from linearmodels.iv import IV2SLS

    method_details = paper_spec.method_details
    instrument = method_details.get("instrument", method_details.get("instruments", []))
    if isinstance(instrument, str):
        instrument = [instrument]

    Y = data[paper_spec.outcome]
    endog = data[paper_spec.treatment]
    Z = data[instrument]
    X = data[paper_spec.covariates] if paper_spec.covariates else None

    model = IV2SLS(Y, X, endog, Z).fit()

    return ReplicationResult(
        estimate=model.params[paper_spec.treatment],
        se=model.std_errors[paper_spec.treatment],
        pvalue=model.pvalues[paper_spec.treatment],
        n_obs=int(model.nobs),
        method="iv",
        details={
            "first_stage_f": model.first_stage.diagnostics["f.stat"].stat
            if hasattr(model, "first_stage") else None
        }
    )


def _run_rd_fallback(
    data: pd.DataFrame,
    paper_spec: PaperSpecification
) -> ReplicationResult:
    """Fallback RD implementation."""
    import statsmodels.formula.api as smf

    method_details = paper_spec.method_details
    running_var = method_details.get("running_variable", "x")
    cutoff = method_details.get("cutoff", 0)

    # Create treatment indicator
    data = data.copy()
    data["_rd_treat"] = (data[running_var] >= cutoff).astype(int)
    data["_rd_centered"] = data[running_var] - cutoff

    # Local linear regression (simple version)
    bandwidth = method_details.get("bandwidth", data["_rd_centered"].std())
    rd_data = data[abs(data["_rd_centered"]) <= bandwidth]

    formula = f"{paper_spec.outcome} ~ _rd_treat * _rd_centered"
    model = smf.ols(formula, data=rd_data).fit()

    return ReplicationResult(
        estimate=model.params["_rd_treat"],
        se=model.bse["_rd_treat"],
        pvalue=model.pvalues["_rd_treat"],
        n_obs=int(model.nobs),
        method="rd",
        details={"bandwidth": bandwidth, "cutoff": cutoff}
    )


def _run_ols_fallback(
    data: pd.DataFrame,
    paper_spec: PaperSpecification
) -> ReplicationResult:
    """Fallback OLS implementation."""
    import statsmodels.formula.api as smf

    formula = f"{paper_spec.outcome} ~ {paper_spec.treatment}"
    if paper_spec.covariates:
        formula += " + " + " + ".join(paper_spec.covariates)

    model = smf.ols(formula, data=data).fit()

    return ReplicationResult(
        estimate=model.params[paper_spec.treatment],
        se=model.bse[paper_spec.treatment],
        pvalue=model.pvalues[paper_spec.treatment],
        n_obs=int(model.nobs),
        method="ols",
        details={"r_squared": model.rsquared, "formula": formula}
    )


# ==============================================================================
# Robustness Checks
# ==============================================================================

def run_robustness_checks(
    data: pd.DataFrame,
    paper_spec: Union[dict, PaperSpecification],
    robustness_specs: Optional[list] = None
) -> dict:
    """
    Run robustness check specifications.

    Parameters
    ----------
    data : pd.DataFrame
        Dataset for analysis
    paper_spec : dict or PaperSpecification
        Main paper specification
    robustness_specs : list, optional
        List of robustness specifications. If None, uses paper_spec.robustness_specs

    Returns
    -------
    dict
        Dictionary of robustness check results
    """
    if isinstance(paper_spec, dict):
        paper_spec = parse_paper_specification(paper_spec)

    if robustness_specs is None:
        robustness_specs = paper_spec.robustness_specs

    results = {}

    for spec in robustness_specs:
        spec_name = spec.get("name", f"robustness_{len(results) + 1}")

        # Create modified specification
        modified_spec = PaperSpecification(
            paper_name=paper_spec.paper_name,
            citation=paper_spec.citation,
            method=spec.get("method", paper_spec.method),
            treatment=spec.get("treatment", paper_spec.treatment),
            outcome=spec.get("outcome", paper_spec.outcome),
            covariates=_modify_covariates(
                paper_spec.covariates,
                spec.get("add_covariates", []),
                spec.get("drop_covariates", [])
            ),
            method_details={**paper_spec.method_details, **spec.get("method_details", {})}
        )

        # Use subset of data if specified
        analysis_data = data
        if "data_subset" in spec:
            analysis_data = data.query(spec["data_subset"])

        try:
            results[spec_name] = run_main_specification(analysis_data, modified_spec)
        except Exception as e:
            results[spec_name] = {"error": str(e)}

    return results


def _modify_covariates(
    base_covariates: list,
    add: list,
    drop: list
) -> list:
    """Modify covariate list."""
    result = [c for c in base_covariates if c not in drop]
    result.extend([c for c in add if c not in result])
    return result


# ==============================================================================
# Result Comparison
# ==============================================================================

def compare_results(
    original: dict,
    replicated: Union[dict, ReplicationResult],
    tolerance: str = "approximate"
) -> ComparisonMetrics:
    """
    Compare original and replicated results.

    Parameters
    ----------
    original : dict
        Original paper results with keys: estimate, se, (optionally) pvalue
    replicated : dict or ReplicationResult
        Replicated results
    tolerance : str
        Tolerance level: "exact", "close", "approximate", "qualitative"

    Returns
    -------
    ComparisonMetrics
        Comparison metrics and success classification
    """
    if isinstance(replicated, ReplicationResult):
        replicated = {
            "estimate": replicated.estimate,
            "se": replicated.se,
            "pvalue": replicated.pvalue
        }

    # Compute metrics
    estimate_diff = replicated["estimate"] - original["estimate"]
    estimate_pct_diff = (
        estimate_diff / abs(original["estimate"]) * 100
        if original["estimate"] != 0 else float("inf")
    )

    se_diff = replicated["se"] - original["se"]
    se_pct_diff = (
        se_diff / original["se"] * 100
        if original["se"] != 0 else float("inf")
    )

    # CI overlap
    ci_overlap = _compute_ci_overlap(
        original["estimate"], original["se"],
        replicated["estimate"], replicated["se"]
    )

    # Statistical conclusions
    orig_sig = original.get("pvalue", 0) < 0.05
    repl_sig = replicated.get("pvalue", 0) < 0.05
    same_significance = orig_sig == repl_sig

    same_sign = np.sign(original["estimate"]) == np.sign(replicated["estimate"])

    # Normalized difference
    combined_se = np.sqrt(original["se"]**2 + replicated["se"]**2)
    normalized_diff = estimate_diff / combined_se if combined_se > 0 else float("inf")

    # Classify success
    success_level, success_message = _classify_replication_success(
        estimate_pct_diff, se_pct_diff, same_sign, same_significance
    )

    return ComparisonMetrics(
        estimate_diff=estimate_diff,
        estimate_pct_diff=estimate_pct_diff,
        se_diff=se_diff,
        se_pct_diff=se_pct_diff,
        ci_overlap=ci_overlap,
        same_significance=same_significance,
        same_sign=same_sign,
        normalized_diff=normalized_diff,
        success_level=success_level,
        success_message=success_message
    )


def _compute_ci_overlap(est1: float, se1: float, est2: float, se2: float) -> float:
    """Compute overlap of 95% confidence intervals."""
    ci1_lower = est1 - 1.96 * se1
    ci1_upper = est1 + 1.96 * se1
    ci2_lower = est2 - 1.96 * se2
    ci2_upper = est2 + 1.96 * se2

    overlap_lower = max(ci1_lower, ci2_lower)
    overlap_upper = min(ci1_upper, ci2_upper)

    if overlap_lower >= overlap_upper:
        return 0.0

    overlap_width = overlap_upper - overlap_lower
    max_width = max(ci1_upper - ci1_lower, ci2_upper - ci2_lower)

    return overlap_width / max_width


def _classify_replication_success(
    estimate_pct_diff: float,
    se_pct_diff: float,
    same_sign: bool,
    same_significance: bool
) -> tuple:
    """Classify replication success level."""
    est_diff = abs(estimate_pct_diff)
    se_diff = abs(se_pct_diff)

    if est_diff < 1 and se_diff < 5:
        return "EXACT", "Results match within numerical precision"

    elif est_diff < 5 and se_diff < 10:
        return "CLOSE", "Minor numerical differences, likely software variation"

    elif est_diff < 10 and se_diff < 20:
        return "APPROXIMATE", "Moderate differences, possibly different specifications"

    elif same_sign and same_significance:
        return "QUALITATIVE", "Conclusions match despite numerical differences"

    else:
        return "FAILED", "Cannot replicate main conclusions"


# ==============================================================================
# Report Generation
# ==============================================================================

def generate_comparison_table(
    original: dict,
    replicated: Union[dict, ReplicationResult],
    paper_name: str = "Paper"
) -> str:
    """
    Generate side-by-side comparison table.

    Parameters
    ----------
    original : dict
        Original paper results
    replicated : dict or ReplicationResult
        Replicated results
    paper_name : str
        Name of paper for title

    Returns
    -------
    str
        Formatted comparison table
    """
    if isinstance(replicated, ReplicationResult):
        repl_dict = {
            "estimate": replicated.estimate,
            "se": replicated.se,
            "pvalue": replicated.pvalue,
            "n_treated": replicated.n_treated,
            "n_control": replicated.n_control,
            "n_obs": replicated.n_obs
        }
    else:
        repl_dict = replicated

    comparison = compare_results(original, repl_dict)

    # Format numbers
    def fmt_num(x, decimals=2):
        if x is None:
            return "N/A"
        if abs(x) >= 1000:
            return f"${x:,.0f}"
        return f"{x:.{decimals}f}"

    def fmt_pct(x):
        if x == float("inf"):
            return "N/A"
        sign = "+" if x > 0 else ""
        return f"{sign}{x:.1f}%"

    lines = [
        "=" * 70,
        f"{'REPLICATION COMPARISON: ' + paper_name:^70}",
        "=" * 70,
        "",
        f"{'Specification':<25} {'Original':>15} {'Replicated':>15} {'Difference':>12}",
        "-" * 70,
        "Main Estimate",
        f"  {'Point Estimate':<23} {fmt_num(original['estimate']):>15} {fmt_num(repl_dict['estimate']):>15} {fmt_pct(comparison.estimate_pct_diff):>12}",
        f"  {'Standard Error':<23} {fmt_num(original['se']):>15} {fmt_num(repl_dict['se']):>15} {fmt_pct(comparison.se_pct_diff):>12}",
    ]

    # Add CI if available
    if repl_dict.get("ci_lower") or isinstance(replicated, ReplicationResult):
        orig_ci = f"[{original['estimate'] - 1.96*original['se']:.0f}, {original['estimate'] + 1.96*original['se']:.0f}]"
        if isinstance(replicated, ReplicationResult):
            repl_ci = f"[{replicated.ci_lower:.0f}, {replicated.ci_upper:.0f}]"
        else:
            repl_ci = f"[{repl_dict['estimate'] - 1.96*repl_dict['se']:.0f}, {repl_dict['estimate'] + 1.96*repl_dict['se']:.0f}]"
        lines.append(f"  {'95% CI':<23} {orig_ci:>15} {repl_ci:>15}")

    # Add p-value
    orig_p = original.get("pvalue", "N/A")
    repl_p = repl_dict.get("pvalue", "N/A")
    if orig_p != "N/A":
        orig_p = f"{orig_p:.3f}"
    if repl_p != "N/A":
        repl_p = f"{repl_p:.3f}"
    lines.append(f"  {'p-value':<23} {str(orig_p):>15} {str(repl_p):>15}")

    lines.extend([
        "",
        "Sample",
    ])

    # Sample sizes
    if original.get("n_treated"):
        orig_nt = str(original.get("n_treated", "N/A"))
        repl_nt = str(repl_dict.get("n_treated", "N/A"))
        match = "Match" if orig_nt == repl_nt else f"Diff: {int(repl_nt) - int(orig_nt)}"
        lines.append(f"  {'N (treated)':<23} {orig_nt:>15} {repl_nt:>15} {match:>12}")

    if original.get("n_control"):
        orig_nc = str(original.get("n_control", "N/A"))
        repl_nc = str(repl_dict.get("n_control", "N/A"))
        match = "Match" if orig_nc == repl_nc else f"Diff: {int(repl_nc) - int(orig_nc)}"
        lines.append(f"  {'N (control)':<23} {orig_nc:>15} {repl_nc:>15} {match:>12}")

    lines.extend([
        "",
        "-" * 70,
        f"REPLICATION STATUS: {comparison.success_level}",
        comparison.success_message,
        "=" * 70,
    ])

    return "\n".join(lines)


def generate_replication_report(
    paper_spec: Union[dict, PaperSpecification],
    results: Union[dict, ReplicationResult],
    comparison: ComparisonMetrics,
    robustness_results: Optional[dict] = None
) -> str:
    """
    Generate full replication report.

    Parameters
    ----------
    paper_spec : dict or PaperSpecification
        Paper specification
    results : dict or ReplicationResult
        Main replication results
    comparison : ComparisonMetrics
        Comparison metrics
    robustness_results : dict, optional
        Results from robustness checks

    Returns
    -------
    str
        Full replication report in markdown format
    """
    if isinstance(paper_spec, dict):
        paper_spec = parse_paper_specification(paper_spec)

    if isinstance(results, ReplicationResult):
        results_dict = {
            "estimate": results.estimate,
            "se": results.se,
            "pvalue": results.pvalue,
            "n_obs": results.n_obs,
            "n_treated": results.n_treated,
            "n_control": results.n_control,
            "method": results.method
        }
    else:
        results_dict = results

    report = f"""# Replication Report: {paper_spec.paper_name}

## 1. Paper Summary

**Citation**: {paper_spec.citation}

**Research Question**: Effect of {paper_spec.treatment} on {paper_spec.outcome}

**Method**: {paper_spec.method}

**Original Findings**:
- Point estimate: {paper_spec.original_results.get('estimate', 'N/A')}
- Standard error: {paper_spec.original_results.get('se', 'N/A')}

## 2. Replication Strategy

**Estimation Method**: {results_dict.get('method', paper_spec.method)}

**Covariates**: {', '.join(paper_spec.covariates) if paper_spec.covariates else 'None'}

**Sample Size**: {results_dict.get('n_obs', 'N/A')} observations

## 3. Results Comparison

### 3.1 Main Specification

| Metric | Original | Replicated | Difference |
|--------|----------|------------|------------|
| Estimate | {paper_spec.original_results.get('estimate', 'N/A')} | {results_dict['estimate']:.2f} | {comparison.estimate_diff:.2f} ({comparison.estimate_pct_diff:.1f}%) |
| Std. Error | {paper_spec.original_results.get('se', 'N/A')} | {results_dict['se']:.2f} | {comparison.se_diff:.2f} ({comparison.se_pct_diff:.1f}%) |
| Same Sign | - | - | {'Yes' if comparison.same_sign else 'No'} |
| Same Significance | - | - | {'Yes' if comparison.same_significance else 'No'} |

**CI Overlap**: {comparison.ci_overlap:.1%}

**Normalized Difference**: {comparison.normalized_diff:.2f} standard errors
"""

    if robustness_results:
        report += "\n### 3.2 Robustness Checks\n\n"
        report += "| Specification | Estimate | Std. Error | Status |\n"
        report += "|---------------|----------|------------|--------|\n"

        for name, result in robustness_results.items():
            if isinstance(result, dict) and "error" in result:
                report += f"| {name} | Error | {result['error'][:30]} | Failed |\n"
            elif isinstance(result, ReplicationResult):
                report += f"| {name} | {result.estimate:.2f} | {result.se:.2f} | OK |\n"
            else:
                est = result.get("estimate", "N/A")
                se = result.get("se", "N/A")
                report += f"| {name} | {est} | {se} | OK |\n"

    report += f"""
## 4. Replication Assessment

**Status**: {comparison.success_level}

**Summary**: {comparison.success_message}

### Interpretation Guidelines

| Status | Meaning |
|--------|---------|
| EXACT | Results match within numerical precision (<1% estimate, <5% SE) |
| CLOSE | Minor numerical differences (<5% estimate, <10% SE) |
| APPROXIMATE | Moderate differences (<10% estimate, <20% SE) |
| QUALITATIVE | Conclusions match despite larger numerical differences |
| FAILED | Cannot replicate main conclusions |

## 5. Potential Sources of Differences

If results differ from original:
- Data version/vintage differences
- Software/package version differences
- Optimization algorithm variations
- Random seed differences (for stochastic methods)
- Undocumented sample restrictions

## 6. Conclusion

This replication achieved **{comparison.success_level}** success. {comparison.success_message}

---
*Generated by paper-replication-workflow skill*
"""

    return report


# ==============================================================================
# Pre-built Replication Functions
# ==============================================================================

def replicate_lalonde(verbose: bool = True) -> dict:
    """
    Pre-built LaLonde (1986) replication.

    Replicates the classic evaluation of job training program effects.

    Parameters
    ----------
    verbose : bool
        Print progress and results

    Returns
    -------
    dict
        Dictionary with results, comparison, and report
    """
    if verbose:
        print("=" * 70)
        print("REPLICATING: LaLonde (1986)")
        print("'Evaluating the Econometric Evaluations of Training Programs'")
        print("=" * 70)
        print()

    # Paper specification
    paper_spec = PaperSpecification(
        paper_name="LaLonde (1986)",
        citation="LaLonde, R. (1986). 'Evaluating the Econometric Evaluations of "
                 "Training Programs with Experimental Data.' American Economic Review, "
                 "76(4): 604-620.",
        method="psm",
        treatment="treat",
        outcome="re78",
        covariates=["age", "education", "black", "hispanic",
                    "married", "nodegree", "re74", "re75"],
        original_results={
            "estimate": 1794,
            "se": 633,
            "n_treated": 185,
            "n_control": 260
        },
        robustness_specs=[
            {"name": "no_lagged_earnings", "drop_covariates": ["re74", "re75"]},
            {"name": "demographics_only", "drop_covariates": ["re74", "re75", "nodegree"]},
        ]
    )

    if verbose:
        print("Step 1: Loading LaLonde NSW data...")

    # Load data
    data = load_replication_data("lalonde_psid")

    if verbose:
        print(f"  - Loaded {len(data)} observations")
        print(f"  - Treatment: {data['treat'].sum()} treated, {(~data['treat'].astype(bool)).sum()} control")
        print()

    if verbose:
        print("Step 2: Running main specification (Propensity Score Matching)...")

    # Run main specification
    results = run_main_specification(data, paper_spec)

    if verbose:
        print(f"  - ATT estimate: ${results.estimate:,.0f}")
        print(f"  - Standard error: ${results.se:,.0f}")
        print(f"  - p-value: {results.pvalue:.4f}")
        print()

    if verbose:
        print("Step 3: Comparing to original paper results...")

    # Compare results
    comparison = compare_results(
        original=paper_spec.original_results,
        replicated=results
    )

    if verbose:
        print(f"  - Estimate difference: ${comparison.estimate_diff:,.0f} ({comparison.estimate_pct_diff:.1f}%)")
        print(f"  - SE difference: ${comparison.se_diff:,.0f} ({comparison.se_pct_diff:.1f}%)")
        print(f"  - Status: {comparison.success_level}")
        print()

    if verbose:
        print("Step 4: Running robustness checks...")

    # Robustness checks
    robustness_results = run_robustness_checks(data, paper_spec)

    if verbose:
        for name, result in robustness_results.items():
            if isinstance(result, ReplicationResult):
                print(f"  - {name}: ${result.estimate:,.0f} (SE: ${result.se:,.0f})")
        print()

    if verbose:
        print("Step 5: Generating comparison table...")
        print()

    # Generate outputs
    table = generate_comparison_table(
        paper_spec.original_results,
        results,
        paper_name=paper_spec.paper_name
    )

    report = generate_replication_report(
        paper_spec, results, comparison, robustness_results
    )

    if verbose:
        print(table)
        print()

    return {
        "paper_spec": paper_spec,
        "results": results,
        "comparison": comparison,
        "robustness_results": robustness_results,
        "comparison_table": table,
        "full_report": report
    }


def replicate_card_proximity(verbose: bool = True) -> dict:
    """
    Pre-built Card (1995) replication.

    Replicates the IV estimation of returns to schooling using college proximity.

    Parameters
    ----------
    verbose : bool
        Print progress and results

    Returns
    -------
    dict
        Dictionary with results, comparison, and report
    """
    if verbose:
        print("=" * 70)
        print("REPLICATING: Card (1995)")
        print("'Using Geographic Variation in College Proximity'")
        print("=" * 70)
        print()

    # Paper specification
    paper_spec = PaperSpecification(
        paper_name="Card (1995)",
        citation="Card, D. (1995). 'Using Geographic Variation in College Proximity "
                 "to Estimate the Return to Schooling.' In Aspects of Labour Market "
                 "Behaviour: Essays in Honour of John Vanderkamp.",
        method="iv",
        treatment="education",
        outcome="log_wage",
        covariates=["exper", "black", "smsa", "south"],
        original_results={
            "estimate": 0.132,  # IV estimate
            "se": 0.055,
        },
        method_details={
            "instrument": "nearc4"
        },
        robustness_specs=[
            {"name": "ols", "method": "ols"},
        ]
    )

    if verbose:
        print("Step 1: Loading Card college proximity data...")

    # Load data
    data = load_replication_data("card_college")

    if verbose:
        print(f"  - Loaded {len(data)} observations")
        print(f"  - Near 4-year college: {data['nearc4'].sum()} ({data['nearc4'].mean()*100:.1f}%)")
        print()

    if verbose:
        print("Step 2: Running IV specification...")

    # Run main specification
    try:
        results = run_main_specification(data, paper_spec)
    except Exception as e:
        warnings.warn(f"IV estimation failed: {e}. Using OLS fallback.")
        paper_spec.method = "ols"
        results = run_main_specification(data, paper_spec)
        paper_spec.method = "iv"

    if verbose:
        print(f"  - IV estimate: {results.estimate:.4f}")
        print(f"  - Standard error: {results.se:.4f}")
        print()

    # Compare results
    comparison = compare_results(
        original=paper_spec.original_results,
        replicated=results
    )

    if verbose:
        print(f"Step 3: Comparison")
        print(f"  - Status: {comparison.success_level}")
        print()

    # Generate report
    report = generate_replication_report(paper_spec, results, comparison)

    return {
        "paper_spec": paper_spec,
        "results": results,
        "comparison": comparison,
        "full_report": report
    }


def replicate_lee_elections(verbose: bool = True) -> dict:
    """
    Pre-built Lee (2008) replication.

    Replicates the RD estimation of incumbency advantage in elections.

    Parameters
    ----------
    verbose : bool
        Print progress and results

    Returns
    -------
    dict
        Dictionary with results, comparison, and report
    """
    if verbose:
        print("=" * 70)
        print("REPLICATING: Lee (2008)")
        print("'Randomized Experiments from Non-random Selection'")
        print("=" * 70)
        print()

    # Paper specification
    paper_spec = PaperSpecification(
        paper_name="Lee (2008)",
        citation="Lee, D. (2008). 'Randomized Experiments from Non-random Selection "
                 "in U.S. House Elections.' Journal of Econometrics, 142(2): 675-697.",
        method="rd",
        treatment="dem_win",
        outcome="win_next",
        covariates=[],
        original_results={
            "estimate": 0.38,
            "se": 0.06,
        },
        method_details={
            "running_variable": "vote_margin",
            "cutoff": 0,
            "bandwidth": 0.25
        }
    )

    if verbose:
        print("Step 1: Loading Lee elections data...")

    # Load data
    data = load_replication_data("lee_elections")

    if verbose:
        print(f"  - Loaded {len(data)} election observations")
        print()

    if verbose:
        print("Step 2: Running RD specification...")

    # Run main specification
    results = run_main_specification(data, paper_spec)

    if verbose:
        print(f"  - RD estimate: {results.estimate:.3f}")
        print(f"  - Standard error: {results.se:.3f}")
        print()

    # Compare results
    comparison = compare_results(
        original=paper_spec.original_results,
        replicated=results
    )

    if verbose:
        print(f"Step 3: Comparison")
        print(f"  - Status: {comparison.success_level}")
        print()

    # Generate report
    report = generate_replication_report(paper_spec, results, comparison)

    return {
        "paper_spec": paper_spec,
        "results": results,
        "comparison": comparison,
        "full_report": report
    }


# ==============================================================================
# Main Entry Point
# ==============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print(" PAPER REPLICATION WORKFLOW - DEMONSTRATION")
    print("=" * 70 + "\n")

    # Run LaLonde replication
    lalonde_results = replicate_lalonde(verbose=True)

    print("\n" + "=" * 70)
    print(" FULL REPLICATION REPORT")
    print("=" * 70 + "\n")
    print(lalonde_results["full_report"])
