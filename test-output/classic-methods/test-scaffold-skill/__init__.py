"""
Test Scaffold Skill Estimator

This package provides the Test Scaffold Skill causal effect estimator.
"""

from .test_scaffold_skill_estimator import (
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
