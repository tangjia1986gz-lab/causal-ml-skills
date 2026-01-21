"""
GOLDEN TEMPLATE: Rigorous Economic Data Preprocessing
STANDARD: Standard practices in AER/Journal of Finance

DESCRIPTION:
This template handles common cleaning tasks for panel data:
1. Handling Missing Values (Drop vs Impute)
2. Winsorization (Handling Outliers)
3. Creating Interaction Terms
4. Setting up Panel Structure (Entity-Time index)
"""

import pandas as pd
import numpy as np

# ==========================================
# 1. LOAD & MERGE
# ==========================================
def load_and_merge(main_file, *other_files, on_keys=['stkcd', 'year']):
    """
    Load Stata/CSV files and merge.
    """
    df = pd.read_stata(main_file) if main_file.endswith('.dta') else pd.read_csv(main_file)
    
    for f in other_files:
        other_df = pd.read_stata(f) if f.endswith('.dta') else pd.read_csv(f)
        df = pd.merge(df, other_df, on=on_keys, how='left')
        
    return df

# ==========================================
# 2. WINSORIZATION
# ==========================================
def winsorize_series(series, limits=[0.01, 0.01]):
    """
    Winsorize a pandas series at 1% and 99% (default).
    """
    return series.clip(lower=series.quantile(limits[0]), upper=series.quantile(1-limits[1]))

def clean_data(df, numerical_cols, winsorize=True):
    df_clean = df.copy()
    
    if winsorize:
        for col in numerical_cols:
            if col in df_clean.columns:
                df_clean[col] = winsorize_series(df_clean[col])
                
    return df_clean

# ==========================================
# 3. PANEL SETUP
# ==========================================
def setup_panel(df, entity_col, time_col):
    """
    Verify unique entity-time obs and set index.
    """
    # Check duplicates
    dupes = df.duplicated(subset=[entity_col, time_col]).sum()
    if dupes > 0:
        print(f"WARNING: Found {dupes} duplicate entity-time observations. Keeping first.")
        df = df.drop_duplicates(subset=[entity_col, time_col])
        
    return df.set_index([entity_col, time_col])
