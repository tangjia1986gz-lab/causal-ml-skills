#!/usr/bin/env python3
"""
Causal ML Environment Diagnostic Script

Comprehensive environment validation for causal inference machine learning workflows.
Checks Python packages, R integration, and optional Stata connectivity.

Usage:
    python check_environment.py [--verbose] [--json] [--fix]

Options:
    --verbose    Show detailed information for each check
    --json       Output results in JSON format
    --fix        Attempt to fix common issues automatically

Exit codes:
    0 - All critical dependencies are met
    1 - One or more critical dependencies are missing
"""

import sys
import os
import subprocess
import importlib.metadata
import json
import platform
import argparse
from typing import Optional, Tuple, List, Dict, Any
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path


class Status(Enum):
    """Status indicators for dependency checks."""
    OK = "OK"
    WARNING = "WARNING"
    ERROR = "ERROR"
    SKIP = "SKIP"


@dataclass
class CheckResult:
    """Result of a dependency check."""
    name: str
    status: Status
    version: Optional[str] = None
    message: Optional[str] = None
    required_version: Optional[str] = None
    category: Optional[str] = None
    fix_command: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        d = asdict(self)
        d['status'] = self.status.value
        return d


# Status symbols for output
SYMBOLS = {
    Status.OK: "[OK]",
    Status.WARNING: "[!!]",
    Status.ERROR: "[XX]",
    Status.SKIP: "[--]",
}

# ANSI color codes
COLORS = {
    Status.OK: "\033[92m",      # Green
    Status.WARNING: "\033[93m",  # Yellow
    Status.ERROR: "\033[91m",    # Red
    Status.SKIP: "\033[90m",     # Gray
    "RESET": "\033[0m",
    "BOLD": "\033[1m",
    "BLUE": "\033[94m",
}


def supports_color() -> bool:
    """Check if the terminal supports color output."""
    if sys.platform == "win32":
        # Enable ANSI on Windows 10+
        try:
            import ctypes
            kernel32 = ctypes.windll.kernel32
            kernel32.SetConsoleMode(kernel32.GetStdHandle(-11), 7)
            return True
        except Exception:
            return os.environ.get("TERM") or os.environ.get("WT_SESSION")
    return hasattr(sys.stdout, "isatty") and sys.stdout.isatty()


USE_COLOR = supports_color()


def colorize(text: str, status: Status) -> str:
    """Apply color to text based on status."""
    if not USE_COLOR:
        return text
    return f"{COLORS[status]}{text}{COLORS['RESET']}"


def bold(text: str) -> str:
    """Make text bold."""
    if not USE_COLOR:
        return text
    return f"{COLORS['BOLD']}{text}{COLORS['RESET']}"


def blue(text: str) -> str:
    """Make text blue."""
    if not USE_COLOR:
        return text
    return f"{COLORS['BLUE']}{text}{COLORS['RESET']}"


def print_header(title: str) -> None:
    """Print a section header."""
    print()
    print(bold(f"{'=' * 60}"))
    print(bold(f" {title}"))
    print(bold(f"{'=' * 60}"))


def print_result(result: CheckResult, verbose: bool = False) -> None:
    """Print a check result with formatting."""
    symbol = colorize(SYMBOLS[result.status], result.status)

    if result.version:
        version_str = f"v{result.version}"
        if result.required_version:
            version_str += f" (>={result.required_version})"
    else:
        version_str = ""

    name_padded = f"{result.name}".ljust(22)

    if result.status == Status.OK:
        print(f"  {symbol} {name_padded} {version_str}")
    elif result.message:
        print(f"  {symbol} {name_padded} {colorize(result.message, result.status)}")
    else:
        print(f"  {symbol} {name_padded}")

    if verbose and result.fix_command:
        print(f"       Fix: {blue(result.fix_command)}")


def get_platform_info() -> Dict[str, str]:
    """Get platform information."""
    return {
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
    }


def check_python_version() -> CheckResult:
    """Check if Python version meets requirements."""
    major, minor = sys.version_info[:2]
    version = f"{major}.{minor}.{sys.version_info.micro}"

    if major >= 3 and minor >= 10:
        return CheckResult(
            "Python", Status.OK, version,
            required_version="3.10",
            category="runtime"
        )
    else:
        return CheckResult(
            "Python", Status.ERROR, version,
            message=f"Python 3.10+ required, found {version}",
            required_version="3.10",
            category="runtime"
        )


def check_python_package(
    name: str,
    import_name: Optional[str] = None,
    required_version: Optional[str] = None,
    critical: bool = True,
    category: str = "python"
) -> CheckResult:
    """Check if a Python package is installed and meets version requirements."""
    import_name = import_name or name
    fix_command = f"pip install {name}"
    if required_version:
        fix_command += f">={required_version}"

    try:
        # Try to get version from metadata
        version = importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        # Try alternative package names
        alt_names = {
            "scikit-learn": "sklearn",
            "lightgbm": "lightgbm",
            "python-dotenv": "dotenv",
        }
        try:
            version = importlib.metadata.version(alt_names.get(name, name))
        except importlib.metadata.PackageNotFoundError:
            status = Status.ERROR if critical else Status.WARNING
            return CheckResult(
                name, status, message="Not installed",
                category=category, fix_command=fix_command
            )

    # Try to actually import the package
    try:
        importlib.import_module(import_name)
    except ImportError as e:
        status = Status.ERROR if critical else Status.WARNING
        return CheckResult(
            name, status, version, message=f"Import failed: {e}",
            category=category
        )

    # Check version if required
    if required_version:
        try:
            from packaging import version as pkg_version
            if pkg_version.parse(version) < pkg_version.parse(required_version):
                return CheckResult(
                    name, Status.WARNING, version,
                    message=f"Version {version} < {required_version}",
                    required_version=required_version,
                    category=category,
                    fix_command=f"pip install --upgrade {name}>={required_version}"
                )
        except ImportError:
            pass  # packaging not available, skip version check

    return CheckResult(
        name, Status.OK, version,
        required_version=required_version,
        category=category
    )


def check_r_availability() -> Tuple[CheckResult, Optional[str]]:
    """Check if R is available and get its version."""
    try:
        # Try R --version
        result = subprocess.run(
            ["R", "--version"],
            capture_output=True,
            text=True,
            timeout=10,
            encoding='utf-8',
            errors='replace'
        )
        if result.returncode == 0:
            first_line = result.stdout.split('\n')[0]
            if "version" in first_line.lower():
                parts = first_line.split()
                for i, part in enumerate(parts):
                    if part.lower() == "version" and i + 1 < len(parts):
                        version = parts[i + 1]
                        major = int(version.split('.')[0])
                        if major >= 4:
                            return CheckResult(
                                "R", Status.OK, version,
                                required_version="4.0",
                                category="r"
                            ), version
                        else:
                            return CheckResult(
                                "R", Status.WARNING, version,
                                message=f"R 4.0+ recommended",
                                required_version="4.0",
                                category="r"
                            ), version
            return CheckResult("R", Status.OK, "unknown", category="r"), "unknown"
    except FileNotFoundError:
        return CheckResult(
            "R", Status.WARNING,
            message="Not found on PATH",
            category="r",
            fix_command="Install R from https://cran.r-project.org/"
        ), None
    except subprocess.TimeoutExpired:
        return CheckResult("R", Status.WARNING, message="Timeout", category="r"), None
    except Exception as e:
        return CheckResult("R", Status.WARNING, message=str(e), category="r"), None


def check_r_package(package_name: str) -> CheckResult:
    """Check if an R package is installed."""
    try:
        result = subprocess.run(
            ["R", "--slave", "-e", f'cat(as.character(packageVersion("{package_name}")))'],
            capture_output=True,
            text=True,
            timeout=30,
            encoding='utf-8',
            errors='replace'
        )
        if result.returncode == 0 and result.stdout.strip():
            version = result.stdout.strip()
            return CheckResult(package_name, Status.OK, version, category="r")
        else:
            return CheckResult(
                package_name, Status.WARNING,
                message="Not installed",
                category="r",
                fix_command=f'R -e \'install.packages("{package_name}")\''
            )
    except FileNotFoundError:
        return CheckResult(package_name, Status.SKIP, message="R not available", category="r")
    except subprocess.TimeoutExpired:
        return CheckResult(package_name, Status.WARNING, message="Timeout", category="r")
    except Exception as e:
        return CheckResult(package_name, Status.WARNING, message=str(e), category="r")


def check_rpy2_bridge() -> CheckResult:
    """Check if rpy2 bridge is working."""
    try:
        import rpy2.robjects as ro
        result = ro.r('1 + 1')
        if result[0] == 2:
            return CheckResult("rpy2 bridge", Status.OK, message="Working", category="r")
        else:
            return CheckResult("rpy2 bridge", Status.WARNING, message="Unexpected result", category="r")
    except ImportError:
        return CheckResult(
            "rpy2 bridge", Status.WARNING,
            message="rpy2 not installed",
            category="r",
            fix_command="pip install rpy2>=3.5.0"
        )
    except Exception as e:
        error_msg = str(e)[:50] + "..." if len(str(e)) > 50 else str(e)
        return CheckResult(
            "rpy2 bridge", Status.WARNING,
            message=f"Error: {error_msg}",
            category="r"
        )


def check_stata_availability() -> CheckResult:
    """Check if Stata is available."""
    stata_commands = ["stata-mp", "stata-se", "stata"]

    # Also check common installation paths
    common_paths = []
    if sys.platform == "win32":
        common_paths = [
            Path("C:/Program Files/Stata17"),
            Path("C:/Program Files/Stata18"),
            Path("C:/Program Files (x86)/Stata"),
        ]
    elif sys.platform == "darwin":
        common_paths = [
            Path("/Applications/Stata"),
            Path("/Applications/Stata 17"),
            Path("/Applications/Stata 18"),
        ]
    else:
        common_paths = [
            Path("/usr/local/stata17"),
            Path("/usr/local/stata18"),
            Path("/opt/stata17"),
        ]

    for cmd in stata_commands:
        try:
            result = subprocess.run(
                [cmd, "-q", "-b", "version"],
                capture_output=True,
                text=True,
                timeout=10,
                encoding='utf-8',
                errors='replace'
            )
            if result.returncode == 0:
                return CheckResult("Stata", Status.OK, message=f"Found ({cmd})", category="stata")
        except (FileNotFoundError, subprocess.TimeoutExpired):
            continue
        except Exception:
            continue

    # Check common paths
    for path in common_paths:
        if path.exists():
            return CheckResult(
                "Stata", Status.OK,
                message=f"Found at {path}",
                category="stata"
            )

    return CheckResult(
        "Stata", Status.SKIP,
        message="Not found (optional)",
        category="stata"
    )


def check_pystata() -> CheckResult:
    """Check if pystata is available (Stata 17+)."""
    try:
        import stata_setup
        return CheckResult("stata_setup", Status.OK, message="Available", category="stata")
    except ImportError:
        return CheckResult(
            "stata_setup", Status.SKIP,
            message="Not installed (optional)",
            category="stata",
            fix_command="pip install stata_setup  # Requires Stata 17+"
        )


def check_gpu_support() -> List[CheckResult]:
    """Check for GPU support in ML packages."""
    results = []

    # Check CUDA availability
    try:
        import torch
        if torch.cuda.is_available():
            cuda_version = torch.version.cuda
            results.append(CheckResult(
                "CUDA (PyTorch)", Status.OK,
                version=cuda_version,
                category="gpu"
            ))
        else:
            results.append(CheckResult(
                "CUDA (PyTorch)", Status.SKIP,
                message="Not available",
                category="gpu"
            ))
    except ImportError:
        results.append(CheckResult(
            "CUDA (PyTorch)", Status.SKIP,
            message="PyTorch not installed",
            category="gpu"
        ))

    # Check XGBoost GPU
    try:
        import xgboost as xgb
        # Try to create a GPU-enabled booster
        try:
            params = {'tree_method': 'gpu_hist', 'device': 'cuda'}
            import numpy as np
            dmat = xgb.DMatrix(np.array([[1, 2], [3, 4]]), label=[0, 1])
            xgb.train(params, dmat, num_boost_round=1)
            results.append(CheckResult(
                "XGBoost GPU", Status.OK,
                message="Available",
                category="gpu"
            ))
        except Exception:
            results.append(CheckResult(
                "XGBoost GPU", Status.SKIP,
                message="Not available",
                category="gpu"
            ))
    except ImportError:
        results.append(CheckResult(
            "XGBoost GPU", Status.SKIP,
            message="XGBoost not installed",
            category="gpu"
        ))

    return results


def run_all_checks(verbose: bool = False) -> Tuple[List[CheckResult], Dict[str, Any]]:
    """Run all environment checks and return results."""
    all_results: List[CheckResult] = []
    platform_info = get_platform_info()

    # Python Version
    if not verbose:
        print_header("Python Environment")
    python_result = check_python_version()
    if not verbose:
        print_result(python_result, verbose)
    all_results.append(python_result)

    # Core Causal Inference Packages
    if not verbose:
        print_header("Core Causal Inference Packages")

    core_packages = [
        ("econml", "econml", "0.15.0", True),
        ("doubleml", "doubleml", "0.7.0", True),
        ("causalml", "causalml", "0.15.0", True),
        ("dowhy", "dowhy", "0.11.0", False),
    ]

    for name, import_name, version, critical in core_packages:
        result = check_python_package(name, import_name, version, critical, "causal")
        if not verbose:
            print_result(result, verbose)
        all_results.append(result)

    # Statistical Packages
    if not verbose:
        print_header("Statistical & Econometric Packages")

    stat_packages = [
        ("statsmodels", "statsmodels", "0.14.0", True),
        ("linearmodels", "linearmodels", "6.0", True),
    ]

    for name, import_name, version, critical in stat_packages:
        result = check_python_package(name, import_name, version, critical, "stats")
        if not verbose:
            print_result(result, verbose)
        all_results.append(result)

    # ML Packages
    if not verbose:
        print_header("Machine Learning Packages")

    ml_packages = [
        ("scikit-learn", "sklearn", "1.3.0", True),
        ("xgboost", "xgboost", "2.0.0", True),
        ("lightgbm", "lightgbm", "4.0.0", True),
    ]

    for name, import_name, version, critical in ml_packages:
        result = check_python_package(name, import_name, version, critical, "ml")
        if not verbose:
            print_result(result, verbose)
        all_results.append(result)

    # Data & Visualization
    if not verbose:
        print_header("Data & Visualization Packages")

    data_packages = [
        ("pandas", "pandas", "2.0.0", True),
        ("numpy", "numpy", "1.24.0", True),
        ("matplotlib", "matplotlib", "3.7.0", True),
        ("seaborn", "seaborn", "0.13.0", False),
        ("shap", "shap", "0.42.0", False),
        ("networkx", "networkx", "3.0", False),
    ]

    for name, import_name, version, critical in data_packages:
        result = check_python_package(name, import_name, version, critical, "data")
        if not verbose:
            print_result(result, verbose)
        all_results.append(result)

    # R Environment
    if not verbose:
        print_header("R Environment")

    r_result, r_version = check_r_availability()
    if not verbose:
        print_result(r_result, verbose)
    all_results.append(r_result)

    # R Packages
    if r_version:
        if not verbose:
            print_header("R Packages")

        r_packages = ["grf", "mediation", "rdrobust", "rddensity"]
        for package in r_packages:
            result = check_r_package(package)
            if not verbose:
                print_result(result, verbose)
            all_results.append(result)

    # Python-R Integration
    if not verbose:
        print_header("Python-R Integration")

    rpy2_result = check_python_package("rpy2", "rpy2", "3.5.0", False, "r")
    if not verbose:
        print_result(rpy2_result, verbose)
    all_results.append(rpy2_result)

    if rpy2_result.status == Status.OK:
        bridge_result = check_rpy2_bridge()
        if not verbose:
            print_result(bridge_result, verbose)
        all_results.append(bridge_result)

    # Stata (Optional)
    if not verbose:
        print_header("Stata Integration (Optional)")

    stata_result = check_stata_availability()
    if not verbose:
        print_result(stata_result, verbose)
    all_results.append(stata_result)

    pystata_result = check_pystata()
    if not verbose:
        print_result(pystata_result, verbose)
    all_results.append(pystata_result)

    return all_results, platform_info


def print_summary(all_results: List[CheckResult]) -> List[CheckResult]:
    """Print summary and return critical failures."""
    print_header("Summary")

    ok_count = sum(1 for r in all_results if r.status == Status.OK)
    warning_count = sum(1 for r in all_results if r.status == Status.WARNING)
    error_count = sum(1 for r in all_results if r.status == Status.ERROR)
    skip_count = sum(1 for r in all_results if r.status == Status.SKIP)

    print(f"  {colorize(SYMBOLS[Status.OK], Status.OK)} Passed:   {ok_count}")
    print(f"  {colorize(SYMBOLS[Status.WARNING], Status.WARNING)} Warnings: {warning_count}")
    print(f"  {colorize(SYMBOLS[Status.ERROR], Status.ERROR)} Errors:   {error_count}")
    print(f"  {colorize(SYMBOLS[Status.SKIP], Status.SKIP)} Skipped:  {skip_count}")

    critical_failures = [r for r in all_results if r.status == Status.ERROR]
    return critical_failures


def output_json(all_results: List[CheckResult], platform_info: Dict[str, str]) -> None:
    """Output results as JSON."""
    output = {
        "platform": platform_info,
        "results": [r.to_dict() for r in all_results],
        "summary": {
            "ok": sum(1 for r in all_results if r.status == Status.OK),
            "warning": sum(1 for r in all_results if r.status == Status.WARNING),
            "error": sum(1 for r in all_results if r.status == Status.ERROR),
            "skip": sum(1 for r in all_results if r.status == Status.SKIP),
        }
    }
    print(json.dumps(output, indent=2))


def main() -> int:
    """Run all environment checks and return exit code."""
    parser = argparse.ArgumentParser(description="Causal ML Environment Diagnostic")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show detailed information")
    parser.add_argument("--json", "-j", action="store_true", help="Output as JSON")
    parser.add_argument("--fix", "-f", action="store_true", help="Show fix commands")
    args = parser.parse_args()

    if not args.json:
        print(bold("\n  Causal ML Environment Diagnostic"))
        print(bold("  ================================\n"))

    all_results, platform_info = run_all_checks(verbose=args.json)

    if args.json:
        output_json(all_results, platform_info)
        critical_failures = [r for r in all_results if r.status == Status.ERROR]
        return 1 if critical_failures else 0

    critical_failures = print_summary(all_results)

    print()

    if critical_failures:
        print(colorize("  CRITICAL: Some required dependencies are missing!", Status.ERROR))
        print()
        print("  Missing critical packages:")
        for result in critical_failures:
            print(f"    - {result.name}")
            if args.fix and result.fix_command:
                print(f"      Fix: {blue(result.fix_command)}")
        print()
        print("  To install all missing packages, run:")
        print("    pip install -r requirements.txt")
        print()
        return 1
    else:
        warnings = [r for r in all_results if r.status == Status.WARNING]
        if warnings:
            print(colorize("  WARNING: Some optional dependencies are missing or outdated.", Status.WARNING))
            print()
            if args.fix:
                print("  Optional fixes:")
                for result in warnings:
                    if result.fix_command:
                        print(f"    - {result.name}: {blue(result.fix_command)}")
                print()
            print("  The environment is functional, but some features may be limited.")
            print()
        else:
            print(colorize("  SUCCESS: All dependencies are properly installed!", Status.OK))
            print()
        return 0


if __name__ == "__main__":
    sys.exit(main())
