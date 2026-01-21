"""
GOLDEN TEMPLATE: Rigorous Instrumental Variables (IV/2SLS) Analysis
STANDARD: American Economic Review (AER) / Quarterly Journal of Economics (QJE)
REFERENCE: Acemoglu, Johnson & Robinson (2001)

DESCRIPTION:
This template demonstrates how to perform a publication-quality 2SLS analysis.
It uses 'linearmodels' for IV estimation and proper diagnostics.

KEY FEATURES:
1. First-Stage Diagnostics (Weak IV Test / F-statistic)
2. Second-Stage Estimation (2SLS)
3. Overidentification Test (Sargan/Hansen)
4. Endogeneity Test (Durbin-Wu-Hausman)
"""

import pandas as pd
import numpy as np
from linearmodels.iv import IV2SLS
import statsmodels.api as sm

# ==========================================
# 1. DATA PREPARATION
# ==========================================
def load_data(filepath):
    df = pd.read_csv(filepath)
    # Add constant for regression
    df['const'] = 1
    return df

# ==========================================
# 2. DIAGNOSTICS: FIRST STAGE (WEAK IV)
# ==========================================
def check_weak_iv(df, endogenous, instrument, controls=['const']):
    """
    Run First Stage regression to check Instrument Strength.
    Rule of Thumb: F-stat > 10 (Stock & Yogo, 2005) or > 104.7 (Lee et al., 2022)
    """
    # Formula: Endogenous ~ Instrument + Controls
    exog = df[[instrument] + controls]
    y_first = df[endogenous]
    
    mod = sm.OLS(y_first, exog)
    res = mod.fit(cov_type='HC1') # Robust SE
    
    print("\n--- FIRST STAGE DIAGNOSTICS ---")
    print(res.summary())
    
    # Extract F-statistic for the instrument
    # Note: rigorous check involves checking the F-stat of excluded instruments specifically
    hypothesis = f"{instrument} = 0"
    f_test = res.f_test(hypothesis)
    print(f"\nFirst-Stage F-statistic (Instrument Strength): {f_test.fvalue:.4f}")
    
    if f_test.fvalue < 10:
        print("WARNING: Potential Weak Instrument problem! (F < 10)")
    else:
        print("PASS: Instrument appears strong.")
        
    return res

# ==========================================
# 3. MAIN ESTIMATION (2SLS)
# ==========================================
def run_2sls(df, outcome, endogenous, instrument, controls=['const']):
    """
    Run 2SLS using linearmodels.
    """
    # IV2SLS(dependent, exog, endog, instruments)
    mod = IV2SLS(
        dependent=df[outcome],
        exog=df[controls],
        endog=df[endogenous],
        instruments=df[[instrument]]
    )
    
    # Fit with Robust Standard Errors (or Clustered)
    res = mod.fit(cov_type='robust')
    
    return res

# ==========================================
# 4. DIAGNOSTICS: Omitted Checks
# ==========================================
def run_diagnostics(model_result):
    """
    Post-estimation diagnostics.
    """
    print("\n--- POST-ESTIMATION DIAGNOSTICS ---")
    
    # Sargan-Hansen J-test (only valid if over-identified)
    if model_result.sargan:
        print(f"Sargan-Hansen Test (Overid): p-value = {model_result.sargan.pval:.4f}")
        if model_result.sargan.pval < 0.05:
            print("WARNING: Instruments may be invalid (correlation with error term).")
    else:
        print("Sargan Test: N/A (Just-identified model)")
        
    # Durbin-Wu-Hausman Test (Endogeneity)
    if model_result.wu_hausman:
        print(f"Hausman Test (Endogeneity): p-value = {model_result.wu_hausman.pval:.4f}")

# ==========================================
# 5. REPORTING
# ==========================================
def print_results(res):
    print("\n--- 2SLS ESTIMATION RESULTS ---")
    print(res.summary)
