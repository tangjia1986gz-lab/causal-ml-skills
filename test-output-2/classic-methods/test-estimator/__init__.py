"""
Test Estimator Estimator

This package provides the Test Estimator causal effect estimator.
"""

from .test_estimator_estimator import (
    estimate,
    validate_data,
    test_assumption_1,
    test_assumption_2,
    placebo_test,
    sensitivity_analysis,
    CausalOutput,
)

__version__ = "0.1.0"
__all__ = [
    "estimate",
    "validate_data",
    "test_assumption_1",
    "test_assumption_2",
    "placebo_test",
    "sensitivity_analysis",
    "CausalOutput",
]
