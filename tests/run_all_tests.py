"""
Run all validation tests for causal ML skills.

This script:
1. Generates synthetic datasets with known true effects
2. Runs each estimator on appropriate data
3. Validates that estimators recover true effects within tolerance
4. Generates a summary report
"""

import sys
import time
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Any
import numpy as np

# Add paths
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / 'lib' / 'python'))
sys.path.insert(0, str(project_root / 'skills'))

# Import synthetic data generators
from data.synthetic.dgp_did import generate_did_panel, generate_staggered_did
from data.synthetic.dgp_rd import generate_sharp_rd, generate_fuzzy_rd
from data.synthetic.dgp_ddml import generate_high_dim_linear, generate_high_dim_nonlinear


@dataclass
class TestResult:
    """Result of a single validation test."""
    estimator: str
    test_name: str
    true_effect: float
    estimated_effect: float
    bias_pct: float
    ci_covers_truth: bool
    passed: bool
    error_message: str = None
    runtime_seconds: float = 0.0


def run_did_tests() -> List[TestResult]:
    """Test DID estimator on synthetic data."""
    results = []

    try:
        from classic_methods.estimator_did.did_estimator import run_full_did_analysis

        # Test 1: Standard 2x2 DID
        data, params = generate_did_panel(n_units=200, n_periods=10, treatment_effect=2.0)
        start = time.time()
        result = run_full_did_analysis(
            data=data,
            outcome='y',
            treatment='treatment_group',
            post_var='post',
            unit_id='unit_id',
            time_id='time'
        )
        runtime = time.time() - start

        bias_pct = abs(result.effect - params['true_ate']) / abs(params['true_ate']) * 100
        ci_covers = params['true_ate'] >= result.ci_lower and params['true_ate'] <= result.ci_upper

        results.append(TestResult(
            estimator='estimator-did',
            test_name='Panel DID (n=2000)',
            true_effect=params['true_ate'],
            estimated_effect=result.effect,
            bias_pct=bias_pct,
            ci_covers_truth=ci_covers,
            passed=bias_pct < 10 and ci_covers,
            runtime_seconds=runtime
        ))

    except Exception as e:
        results.append(TestResult(
            estimator='estimator-did',
            test_name='Panel DID',
            true_effect=2.0,
            estimated_effect=np.nan,
            bias_pct=np.nan,
            ci_covers_truth=False,
            passed=False,
            error_message=str(e)
        ))

    return results


def run_rd_tests() -> List[TestResult]:
    """Test RD estimator on synthetic data."""
    results = []

    try:
        from classic_methods.estimator_rd.rd_estimator import run_full_rd_analysis

        # Test: Sharp RD
        data, params = generate_sharp_rd(n=3000, treatment_effect=0.5)
        start = time.time()
        result = run_full_rd_analysis(
            data=data,
            running='running',
            outcome='y',
            cutoff=0.0
        )
        runtime = time.time() - start

        bias_pct = abs(result.effect - params['true_late']) / abs(params['true_late']) * 100
        ci_covers = params['true_late'] >= result.ci_lower and params['true_late'] <= result.ci_upper

        results.append(TestResult(
            estimator='estimator-rd',
            test_name='Sharp RD (n=3000)',
            true_effect=params['true_late'],
            estimated_effect=result.effect,
            bias_pct=bias_pct,
            ci_covers_truth=ci_covers,
            passed=bias_pct < 15 and ci_covers,
            runtime_seconds=runtime
        ))

    except Exception as e:
        results.append(TestResult(
            estimator='estimator-rd',
            test_name='Sharp RD',
            true_effect=0.5,
            estimated_effect=np.nan,
            bias_pct=np.nan,
            ci_covers_truth=False,
            passed=False,
            error_message=str(e)
        ))

    return results


def run_iv_tests() -> List[TestResult]:
    """Test IV estimator on synthetic data."""
    results = []

    try:
        from classic_methods.estimator_iv.iv_estimator import run_full_iv_analysis
        from data_loader import generate_synthetic_iv_data

        # Generate IV data
        data, params = generate_synthetic_iv_data(n=2000, treatment_effect=1.0)
        start = time.time()
        result = run_full_iv_analysis(
            data=data,
            outcome='y',
            treatment='d',
            instruments=['z'],
            controls=['x1', 'x2']
        )
        runtime = time.time() - start

        bias_pct = abs(result.effect - params['true_effect']) / abs(params['true_effect']) * 100
        ci_covers = params['true_effect'] >= result.ci_lower and params['true_effect'] <= result.ci_upper

        results.append(TestResult(
            estimator='estimator-iv',
            test_name='2SLS (n=2000)',
            true_effect=params['true_effect'],
            estimated_effect=result.effect,
            bias_pct=bias_pct,
            ci_covers_truth=ci_covers,
            passed=bias_pct < 15 and ci_covers,
            runtime_seconds=runtime
        ))

    except Exception as e:
        results.append(TestResult(
            estimator='estimator-iv',
            test_name='2SLS',
            true_effect=1.0,
            estimated_effect=np.nan,
            bias_pct=np.nan,
            ci_covers_truth=False,
            passed=False,
            error_message=str(e)
        ))

    return results


def run_psm_tests() -> List[TestResult]:
    """Test PSM estimator on synthetic data."""
    results = []

    try:
        from classic_methods.estimator_psm.psm_estimator import run_full_psm_analysis

        # Generate PSM-appropriate data
        np.random.seed(42)
        n = 2000
        x1 = np.random.normal(0, 1, n)
        x2 = np.random.binomial(1, 0.5, n)
        propensity = 1 / (1 + np.exp(-(0.5 * x1 + 0.3 * x2)))
        treat = np.random.binomial(1, propensity, n)
        y = 1.0 + 0.5 * x1 + 0.3 * x2 + 2.0 * treat + np.random.normal(0, 1, n)

        import pandas as pd
        data = pd.DataFrame({'y': y, 'treat': treat, 'x1': x1, 'x2': x2})

        start = time.time()
        result = run_full_psm_analysis(
            data=data,
            outcome='y',
            treatment='treat',
            covariates=['x1', 'x2']
        )
        runtime = time.time() - start

        true_effect = 2.0
        bias_pct = abs(result.effect - true_effect) / abs(true_effect) * 100
        ci_covers = true_effect >= result.ci_lower and true_effect <= result.ci_upper

        results.append(TestResult(
            estimator='estimator-psm',
            test_name='PSM ATT (n=2000)',
            true_effect=true_effect,
            estimated_effect=result.effect,
            bias_pct=bias_pct,
            ci_covers_truth=ci_covers,
            passed=bias_pct < 15 and ci_covers,
            runtime_seconds=runtime
        ))

    except Exception as e:
        results.append(TestResult(
            estimator='estimator-psm',
            test_name='PSM ATT',
            true_effect=2.0,
            estimated_effect=np.nan,
            bias_pct=np.nan,
            ci_covers_truth=False,
            passed=False,
            error_message=str(e)
        ))

    return results


def run_ddml_tests() -> List[TestResult]:
    """Test DDML estimator on high-dimensional data."""
    results = []

    try:
        from causal_ml.causal_ddml.ddml_estimator import run_full_ddml_analysis

        # High-dimensional linear
        data, params = generate_high_dim_linear(n=2000, p=100, s=10, treatment_effect=2.0)
        controls = [c for c in data.columns if c.startswith('x')]

        start = time.time()
        result = run_full_ddml_analysis(
            data=data,
            outcome='y',
            treatment='d',
            controls=controls
        )
        runtime = time.time() - start

        bias_pct = abs(result.effect - params['true_ate']) / abs(params['true_ate']) * 100
        ci_covers = params['true_ate'] >= result.ci_lower and params['true_ate'] <= result.ci_upper

        results.append(TestResult(
            estimator='causal-ddml',
            test_name='PLR (n=2000, p=100)',
            true_effect=params['true_ate'],
            estimated_effect=result.effect,
            bias_pct=bias_pct,
            ci_covers_truth=ci_covers,
            passed=bias_pct < 15 and ci_covers,
            runtime_seconds=runtime
        ))

    except Exception as e:
        results.append(TestResult(
            estimator='causal-ddml',
            test_name='PLR',
            true_effect=2.0,
            estimated_effect=np.nan,
            bias_pct=np.nan,
            ci_covers_truth=False,
            passed=False,
            error_message=str(e)
        ))

    return results


def generate_report(results: List[TestResult]) -> str:
    """Generate summary report of all tests."""
    lines = []
    lines.append("=" * 80)
    lines.append("CAUSAL ML SKILLS VALIDATION REPORT")
    lines.append("=" * 80)
    lines.append("")

    passed = sum(1 for r in results if r.passed)
    total = len(results)
    lines.append(f"Overall: {passed}/{total} tests passed ({100*passed/total:.1f}%)")
    lines.append("")

    # Table header
    lines.append(f"{'Estimator':<20} {'Test':<25} {'True':<8} {'Est':<8} {'Bias%':<8} {'CI OK':<6} {'Pass':<6}")
    lines.append("-" * 80)

    for r in results:
        if r.error_message:
            lines.append(f"{r.estimator:<20} {r.test_name:<25} ERROR: {r.error_message[:30]}")
        else:
            ci_ok = "Yes" if r.ci_covers_truth else "No"
            passed_str = "PASS" if r.passed else "FAIL"
            lines.append(
                f"{r.estimator:<20} {r.test_name:<25} {r.true_effect:<8.3f} "
                f"{r.estimated_effect:<8.3f} {r.bias_pct:<8.2f} {ci_ok:<6} {passed_str:<6}"
            )

    lines.append("")
    lines.append("=" * 80)

    # Timing summary
    lines.append("\nRuntime Summary:")
    for r in results:
        if not r.error_message:
            lines.append(f"  {r.estimator}: {r.runtime_seconds:.2f}s")

    return "\n".join(lines)


def main():
    """Run all validation tests."""
    print("Running causal ML skills validation tests...")
    print()

    all_results = []

    print("Testing DID estimator...")
    all_results.extend(run_did_tests())

    print("Testing RD estimator...")
    all_results.extend(run_rd_tests())

    print("Testing IV estimator...")
    all_results.extend(run_iv_tests())

    print("Testing PSM estimator...")
    all_results.extend(run_psm_tests())

    print("Testing DDML estimator...")
    all_results.extend(run_ddml_tests())

    print()
    report = generate_report(all_results)
    print(report)

    # Save report
    report_path = Path(__file__).parent / "validation_report.txt"
    with open(report_path, 'w') as f:
        f.write(report)
    print(f"\nReport saved to: {report_path}")

    # Return exit code
    passed = sum(1 for r in all_results if r.passed)
    return 0 if passed == len(all_results) else 1


if __name__ == "__main__":
    sys.exit(main())
