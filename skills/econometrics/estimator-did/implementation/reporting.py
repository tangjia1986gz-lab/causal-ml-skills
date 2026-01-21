from typing import List, Optional, Union
import pandas as pd
import numpy as np

def _get_significance_stars(p_value: float) -> str:
    """Return significance stars based on p-value."""
    if p_value < 0.01:
        return "***"
    elif p_value < 0.05:
        return "**"
    elif p_value < 0.1:
        return "*"
    else:
        return ""

def _format_se(se: float) -> str:
    """Format standard error in parentheses."""
    return f"({se:.3f})"

def generate_publication_table(models: List[object], 
                             model_names: Optional[List[str]] = None,
                             output_format: str = 'markdown') -> str:
    """
    Generate an AER-style publication table from a list of model results.
    
    Args:
        models: List of fitted model results (e.g., from linearmodels or statsmodels)
        model_names: Optional list of column names (e.g., "(1)", "(2)")
        output_format: 'markdown' or 'latex'
        
    Returns:
        Formatted table string
    """
    if model_names is None:
        model_names = [f"({i+1})" for i in range(len(models))]
        
    # Extract coefficients and stats
    table_data = {}
    metrics = {
        'Observations': [],
        'R-squared': [],
        'Controls': [],
        'Unit FE': [],
        'Time FE': []
    }
    
    all_params = set()
    for model in models:
        all_params.update(model.params.index)
        
    sorted_params = sorted(list(all_params))
    
    # Build coefficient rows
    rows = []
    for param in sorted_params:
        # Coefficient row
        coef_row = []
        # SE row
        se_row = []
        
        for model in models:
            if param in model.params.index:
                coef = model.params[param]
                pval = model.pvalues[param]
                se = model.std_errors[param]
                
                stars = _get_significance_stars(pval)
                coef_str = f"{coef:.3f}{stars}"
                se_str = _format_se(se)
            else:
                coef_str = ""
                se_str = ""
            
            coef_row.append(coef_str)
            se_row.append(se_str)
            
        rows.append((param, coef_row))
        rows.append(("", se_row))
        
    # Extract metrics
    for model in models:
        metrics['Observations'].append(f"{int(model.nobs):,}")
        metrics['R-squared'].append(f"{model.rsquared:.3f}")
        
        # Check for Fixed Effects (specific to linearmodels)
        has_entity_fe = "Yes" if getattr(model.model, 'entity_effects', False) else "No"
        has_time_fe = "Yes" if getattr(model.model, 'time_effects', False) else "No"
        
        metrics['Unit FE'].append(has_entity_fe)
        metrics['Time FE'].append(has_time_fe)
        # Placeholder logic for controls - usually check if exog vars > 1
        metrics['Controls'].append("Yes" if model.model.exog.shape[1] > 1 else "No")

    # Construct Markdown Table
    if output_format == 'markdown':
        header = "| Variable | " + " | ".join(model_names) + " |"
        sep = "|---|" + "|".join(["---"] * len(models)) + "|"
        
        lines = [header, sep]
        
        for name, values in rows:
            line = f"| {name} | " + " | ".join(values) + " |"
            lines.append(line)
            
        lines.append(sep) # Separator for metrics
        
        for metric, values in metrics.items():
            line = f"| {metric} | " + " | ".join(values) + " |"
            lines.append(line)
            
        lines.append(sep)
        lines.append("| Note: | * p<0.1, ** p<0.05, *** p<0.01 |")
        
        return "\n".join(lines)
        
    elif output_format == 'latex':
        # Basic LaTeX implementation
        header = "\\begin{table}\n\\centering\n\\begin{tabular}{l" + "c"*len(models) + "}\n\\hline\n"
        header += " & ".join([""] + model_names) + " \\\\\n\\hline\n"
        
        body = ""
        for name, values in rows:
            # Escape underscores in variable names for LaTeX
            safe_name = name.replace("_", "\\_")
            body += f"{safe_name} & " + " & ".join(values) + " \\\\\n"
            
        body += "\\hline\n"
        for metric, values in metrics.items():
            body += f"{metric} & " + " & ".join(values) + " \\\\\n"
            
        footer = "\\hline\n\\end{tabular}\n\\caption{Difference-in-Differences Results}\n\\end{table}"
        return header + body + footer
    
    else:
        raise ValueError("Unsupported format. Use 'markdown' or 'latex'.")
