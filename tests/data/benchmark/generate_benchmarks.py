"""
Generate benchmark datasets for causal inference validation.

These datasets are based on or inspired by classic papers in causal inference.
"""

import numpy as np
import pandas as pd
from pathlib import Path


def generate_lalonde_style(
    n_treated: int = 185,
    n_control: int = 2490,
    treatment_effect: float = 1794.0,
    random_state: int = 42
) -> pd.DataFrame:
    """
    Generate LaLonde (1986) style job training data.

    Based on the NSW (National Supported Work) experimental data structure.

    Variables:
    - treat: 1 if in job training program
    - age: age in years
    - educ: years of education
    - black: 1 if Black
    - hisp: 1 if Hispanic
    - married: 1 if married
    - nodegree: 1 if no high school degree
    - re74: real earnings 1974
    - re75: real earnings 1975
    - re78: real earnings 1978 (outcome)
    """
    np.random.seed(random_state)

    n = n_treated + n_control
    treat = np.array([1] * n_treated + [0] * n_control)
    np.random.shuffle(treat)

    # Demographics correlated with treatment (selection)
    age = np.random.normal(25, 7, n).clip(17, 55).astype(int)
    educ = np.random.normal(10, 2, n).clip(3, 16).astype(int)
    black = np.random.binomial(1, 0.8 - 0.3 * (1 - treat), n)
    hisp = np.random.binomial(1, 0.1, n)
    married = np.random.binomial(1, 0.2 + 0.1 * (1 - treat), n)
    nodegree = (educ < 12).astype(int)

    # Pre-treatment earnings (lower for treated - selection)
    base_earn = 2000 + 500 * educ + 100 * age - 1000 * nodegree
    re74 = np.maximum(0, base_earn + np.random.normal(0, 3000, n) - 2000 * treat)
    re75 = np.maximum(0, base_earn + np.random.normal(0, 3000, n) - 1500 * treat)

    # Post-treatment earnings with treatment effect
    re78 = np.maximum(0, base_earn + np.random.normal(0, 4000, n) + treatment_effect * treat)

    data = pd.DataFrame({
        'treat': treat,
        'age': age,
        'educ': educ,
        'black': black,
        'hisp': hisp,
        'married': married,
        'nodegree': nodegree,
        're74': re74.round(2),
        're75': re75.round(2),
        're78': re78.round(2)
    })

    return data


def generate_card_style(
    n: int = 3000,
    iv_effect: float = 0.5,
    first_stage: float = 0.4,
    random_state: int = 42
) -> pd.DataFrame:
    """
    Generate Card (1995) style college proximity IV data.

    Models the effect of education on wages using college proximity as instrument.

    Variables:
    - nearc4: 1 if grew up near 4-year college (instrument)
    - educ: years of education (endogenous)
    - lwage: log wages (outcome)
    - exper: years of experience
    - black: 1 if Black
    - south: 1 if in South
    - smsa: 1 if in metropolitan area
    """
    np.random.seed(random_state)

    # Instrument: grew up near college
    nearc4 = np.random.binomial(1, 0.5, n)

    # Unobserved ability (creates endogeneity)
    ability = np.random.normal(0, 1, n)

    # Demographics
    black = np.random.binomial(1, 0.25, n)
    south = np.random.binomial(1, 0.4, n)
    smsa = np.random.binomial(1, 0.6, n)

    # Education affected by ability (endogenous) and proximity (IV)
    educ = (12 + first_stage * nearc4 + 0.8 * ability +
            0.5 * smsa - 0.3 * south - 0.5 * black +
            np.random.normal(0, 1.5, n)).clip(8, 20)

    exper = (35 - educ + np.random.normal(0, 3, n)).clip(0, 30)

    # Log wages: affected by education, ability (omitted), demographics
    lwage = (1.5 + iv_effect * educ + 0.03 * exper - 0.0005 * exper**2 +
             0.3 * ability - 0.1 * black - 0.05 * south + 0.1 * smsa +
             np.random.normal(0, 0.3, n))

    data = pd.DataFrame({
        'nearc4': nearc4,
        'educ': educ.round(1),
        'lwage': lwage.round(3),
        'exper': exper.round(1),
        'black': black,
        'south': south,
        'smsa': smsa
    })

    return data


def generate_lee_style(
    n: int = 6000,
    rd_effect: float = 0.08,
    random_state: int = 42
) -> pd.DataFrame:
    """
    Generate Lee (2008) style electoral RD data.

    Models incumbent advantage using vote margin as running variable.

    Variables:
    - margin: Democratic vote margin (running variable)
    - incumbent: 1 if Democrat won previous election
    - demvote_next: Democratic vote share in next election (outcome)
    - year: election year
    """
    np.random.seed(random_state)

    # Running variable: Democratic margin (centered at 0)
    margin = np.random.normal(0, 0.2, n).clip(-0.5, 0.5)

    # Treatment: Democrat won (incumbency)
    incumbent = (margin >= 0).astype(int)

    # Year (for clustering)
    year = np.random.choice([1980, 1984, 1988, 1992, 1996, 2000], n)

    # Next election vote share
    # Smooth function of margin + discontinuity for incumbency
    demvote_next = (0.5 + 0.3 * margin + 0.1 * margin**2 +
                    rd_effect * incumbent +
                    np.random.normal(0, 0.08, n)).clip(0, 1)

    data = pd.DataFrame({
        'margin': margin.round(4),
        'incumbent': incumbent,
        'demvote_next': demvote_next.round(4),
        'year': year
    })

    return data


def generate_cardkrueger_style(
    n_nj: int = 200,
    n_pa: int = 200,
    treatment_effect: float = 0.05,
    random_state: int = 42
) -> pd.DataFrame:
    """
    Generate Card & Krueger (1994) style minimum wage DID data.

    Models employment effects of NJ minimum wage increase.

    Variables:
    - state: NJ (treated) or PA (control)
    - time: 0 (before) or 1 (after)
    - chain: fast food chain (BK, KFC, RR, W)
    - fte: full-time equivalent employment (outcome)
    - wage: starting wage
    """
    np.random.seed(random_state)

    n = n_nj + n_pa
    state = np.array(['NJ'] * n_nj + ['PA'] * n_pa)

    # Two time periods
    time = np.tile([0, 1], n)
    state = np.repeat(state, 2)
    store_id = np.repeat(np.arange(n), 2)

    # Chain (random assignment)
    chain_map = {0: 'BK', 1: 'KFC', 2: 'RR', 3: 'W'}
    chain = np.array([chain_map[c] for c in np.repeat(np.random.randint(0, 4, n), 2)])

    # Store fixed effect
    store_fe = np.repeat(np.random.normal(15, 3, n), 2)

    # Treatment indicator
    treat = ((state == 'NJ') & (time == 1)).astype(int)

    # Employment
    fte = (store_fe +
           0.5 * time +  # Common trend
           treatment_effect * store_fe * treat +  # Treatment effect
           np.random.normal(0, 2, len(state))).clip(5, 50)

    # Starting wage (affected by minimum wage in NJ after)
    base_wage = np.where(state == 'NJ', 4.25, 4.25)  # Both at 4.25 initially
    wage = np.where((state == 'NJ') & (time == 1), 5.05, base_wage + 0.1 * time)
    wage += np.random.normal(0, 0.1, len(state))

    data = pd.DataFrame({
        'store_id': store_id,
        'state': state,
        'time': time,
        'chain': chain,
        'fte': fte.round(1),
        'wage': wage.round(2),
        'treat': treat
    })

    return data


if __name__ == "__main__":
    output_dir = Path(__file__).parent

    # LaLonde NSW
    lalonde = generate_lalonde_style()
    lalonde.to_csv(output_dir / "lalonde_nsw.csv", index=False)
    print(f"Generated lalonde_nsw.csv (n={len(lalonde)})")

    # Card proximity IV
    card = generate_card_style()
    card.to_csv(output_dir / "card_proximity.csv", index=False)
    print(f"Generated card_proximity.csv (n={len(card)})")

    # Lee elections RD
    lee = generate_lee_style()
    lee.to_csv(output_dir / "lee_regression.csv", index=False)
    print(f"Generated lee_regression.csv (n={len(lee)})")

    # Card & Krueger DID
    cardkrueger = generate_cardkrueger_style()
    cardkrueger.to_csv(output_dir / "cardkrueger_did.csv", index=False)
    print(f"Generated cardkrueger_did.csv (n={len(cardkrueger)})")

    print("\nBenchmark datasets saved to:", output_dir)
