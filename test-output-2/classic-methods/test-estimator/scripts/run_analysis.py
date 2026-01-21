#!/usr/bin/env python3
"""
Run Test Estimator Analysis

This script provides a command-line interface for running
the test-estimator estimator on a dataset.

Usage:
    python run_analysis.py --data data.csv --outcome y --treatment d
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from test_estimator_estimator import estimate


def main():
    parser = argparse.ArgumentParser(
        description="Run Test Estimator analysis"
    )
    parser.add_argument("--data", required=True, help="Path to data file (CSV)")
    parser.add_argument("--outcome", required=True, help="Outcome variable name")
    parser.add_argument("--treatment", required=True, help="Treatment variable name")
    parser.add_argument("--controls", nargs="+", help="Control variable names")
    parser.add_argument("--output", help="Output file path for results")

    args = parser.parse_args()

    # Load data
    data = pd.read_csv(args.data)

    # Run estimation
    result = estimate(
        data=data,
        outcome=args.outcome,
        treatment=args.treatment,
        controls=args.controls
    )

    # Print results
    print(result.summary_table)

    # Save if output specified
    if args.output:
        with open(args.output, "w") as f:
            f.write(result.summary_table)
        print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
