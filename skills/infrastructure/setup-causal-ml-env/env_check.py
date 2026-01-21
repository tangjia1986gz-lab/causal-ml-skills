#!/usr/bin/env python3
"""
Causal ML Environment Diagnostic Script

This script checks the installation status of all required packages for
causal inference machine learning workflows.

Usage:
    python env_check.py

Exit codes:
    0 - All critical dependencies are met
    1 - One or more critical dependencies are missing
"""

import sys
import subprocess
import importlib.metadata
from typing import Optional, Tuple
from dataclasses import dataclass
from enum import Enum


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


# Status symbols for output
SYMBOLS = {
    Status.OK: "[OK]",
    Status.WARNING: "[!!]",
    Status.ERROR: "[XX]",
    Status.SKIP: "[--]",
}

# ANSI color codes (disabled on Windows by default)
COLORS = {
    Status.OK: "\033[92m",      # Green
    Status.WARNING: "\033[93m",  # Yellow
    Status.ERROR: "\033[91m",    # Red
    Status.SKIP: "\033[90m",     # Gray
    "RESET": "\033[0m",
    "BOLD": "\033[1m",
}


def supports_color() -> bool:
    """Check if the terminal supports color output."""
    if sys.platform == "win32":
        try:
            import os
            return os.environ.get("TERM") or os.environ.get("WT_SESSION")
        except Exception:
            return False
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


def print_header(title: str) -> None:
    """Print a section header."""
    print()
    print(bold(f"{'=' * 60}"))
    print(bold(f" {title}"))
    print(bold(f"{'=' * 60}"))


def print_result(result: CheckResult) -> None:
    """Print a check result with formatting."""
    symbol = colorize(SYMBOLS[result.status], result.status)

    if result.version:
        version_str = f"v{result.version}"
        if result.required_version:
            version_str += f" (required: >={result.required_version})"
    else:
        version_str = ""

    name_padded = f"{result.name}".ljust(20)

    if result.status == Status.OK:
        print(f"  {symbol} {name_padded} {version_str}")
    elif result.message:
        print(f"  {symbol} {name_padded} {colorize(result.message, result.status)}")
    else:
        print(f"  {symbol} {name_padded}")


def check_python_version() -> CheckResult:
    """Check if Python version meets requirements."""
    major, minor = sys.version_info[:2]
    version = f"{major}.{minor}.{sys.version_info.micro}"

    if major >= 3 and minor >= 10:
        return CheckResult("Python", Status.OK, version, required_version="3.10")
    else:
        return CheckResult(
            "Python", Status.ERROR, version,
            message=f"Python 3.10+ required, found {version}",
            required_version="3.10"
        )


def check_python_package(name: str, import_name: Optional[str] = None,
                         required_version: Optional[str] = None,
                         critical: bool = True) -> CheckResult:
    """Check if a Python package is installed and meets version requirements."""
    import_name = import_name or name

    try:
        # Try to get version from metadata
        version = importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        # Try alternative package names
        alt_names = {
            "scikit-learn": "sklearn",
            "lightgbm": "lightgbm",
        }
        try:
            version = importlib.metadata.version(alt_names.get(name, name))
        except importlib.metadata.PackageNotFoundError:
            status = Status.ERROR if critical else Status.WARNING
            return CheckResult(name, status, message="Not installed")

    # Try to actually import the package
    try:
        importlib.import_module(import_name)
    except ImportError as e:
        status = Status.ERROR if critical else Status.WARNING
        return CheckResult(name, status, version, message=f"Import failed: {e}")

    # Check version if required
    if required_version:
        try:
            from packaging import version as pkg_version
            if pkg_version.parse(version) < pkg_version.parse(required_version):
                return CheckResult(
                    name, Status.WARNING, version,
                    message=f"Version {version} < {required_version}",
                    required_version=required_version
                )
        except ImportError:
            # packaging not available, skip version check
            pass

    return CheckResult(name, Status.OK, version, required_version=required_version)


def check_r_availability() -> Tuple[CheckResult, Optional[str]]:
    """Check if R is available and get its version."""
    try:
        result = subprocess.run(
            ["R", "--version"],
            capture_output=True,
            text=True,
            timeout=10,
            encoding='utf-8',
            errors='replace'
        )
        if result.returncode == 0:
            # Parse version from output
            first_line = result.stdout.split('\n')[0]
            # Example: "R version 4.3.0 (2023-04-21)"
            if "version" in first_line.lower():
                parts = first_line.split()
                for i, part in enumerate(parts):
                    if part.lower() == "version" and i + 1 < len(parts):
                        version = parts[i + 1]
                        # Check if version is 4.0+
                        major = int(version.split('.')[0])
                        if major >= 4:
                            return CheckResult("R", Status.OK, version, required_version="4.0"), version
                        else:
                            return CheckResult(
                                "R", Status.WARNING, version,
                                message=f"R 4.0+ recommended, found {version}",
                                required_version="4.0"
                            ), version
            return CheckResult("R", Status.OK, "unknown"), "unknown"
    except FileNotFoundError:
        return CheckResult("R", Status.WARNING, message="Not found on PATH"), None
    except subprocess.TimeoutExpired:
        return CheckResult("R", Status.WARNING, message="Timeout checking R"), None
    except Exception as e:
        return CheckResult("R", Status.WARNING, message=str(e)), None


def check_r_package(package_name: str) -> CheckResult:
    """Check if an R package is installed."""
    try:
        result = subprocess.run(
            ["R", "--slave", "-e", f'cat(packageVersion("{package_name}"))'],
            capture_output=True,
            text=True,
            timeout=30,
            encoding='utf-8',
            errors='replace'
        )
        if result.returncode == 0 and result.stdout.strip():
            version = result.stdout.strip()
            return CheckResult(package_name, Status.OK, version)
        else:
            return CheckResult(package_name, Status.WARNING, message="Not installed")
    except FileNotFoundError:
        return CheckResult(package_name, Status.SKIP, message="R not available")
    except subprocess.TimeoutExpired:
        return CheckResult(package_name, Status.WARNING, message="Timeout")
    except Exception as e:
        return CheckResult(package_name, Status.WARNING, message=str(e))


def check_rpy2_bridge() -> CheckResult:
    """Check if rpy2 bridge is working."""
    try:
        import rpy2.robjects as ro
        # Test basic R execution
        result = ro.r('1 + 1')
        if result[0] == 2:
            return CheckResult("rpy2 bridge", Status.OK, message="Working")
        else:
            return CheckResult("rpy2 bridge", Status.WARNING, message="Unexpected result")
    except ImportError:
        return CheckResult("rpy2 bridge", Status.WARNING, message="rpy2 not installed")
    except Exception as e:
        error_msg = str(e)[:50] + "..." if len(str(e)) > 50 else str(e)
        return CheckResult("rpy2 bridge", Status.WARNING, message=f"Error: {error_msg}")


def check_stata_availability() -> CheckResult:
    """Check if Stata is available."""
    stata_commands = ["stata-mp", "stata-se", "stata"]

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
                return CheckResult("Stata", Status.OK, message=f"Found ({cmd})")
        except FileNotFoundError:
            continue
        except Exception:
            continue

    return CheckResult("Stata", Status.SKIP, message="Not found (optional)")


def main() -> int:
    """Run all environment checks and return exit code."""
    print(bold("\n  Causal ML Environment Diagnostic"))
    print(bold("  ================================\n"))

    all_results: list[CheckResult] = []
    critical_failures: list[CheckResult] = []

    # Python Version Check
    print_header("Python Environment")
    python_result = check_python_version()
    print_result(python_result)
    all_results.append(python_result)
    if python_result.status == Status.ERROR:
        critical_failures.append(python_result)

    # Core Causal Inference Packages
    print_header("Core Causal Inference Packages")
    core_packages = [
        ("econml", "econml", "0.15.0", True),
        ("doubleml", "doubleml", "0.7.0", True),
        ("causalml", "causalml", "0.15.0", True),
        ("dowhy", "dowhy", "0.11.0", False),
    ]

    for name, import_name, version, critical in core_packages:
        result = check_python_package(name, import_name, version, critical)
        print_result(result)
        all_results.append(result)
        if result.status == Status.ERROR and critical:
            critical_failures.append(result)

    # Statistical Packages
    print_header("Statistical & Econometric Packages")
    stat_packages = [
        ("statsmodels", "statsmodels", "0.14.0", True),
        ("linearmodels", "linearmodels", "6.0", True),
    ]

    for name, import_name, version, critical in stat_packages:
        result = check_python_package(name, import_name, version, critical)
        print_result(result)
        all_results.append(result)
        if result.status == Status.ERROR and critical:
            critical_failures.append(result)

    # ML Packages
    print_header("Machine Learning Packages")
    ml_packages = [
        ("scikit-learn", "sklearn", "1.3.0", True),
        ("xgboost", "xgboost", "2.0.0", True),
        ("lightgbm", "lightgbm", "4.0.0", True),
    ]

    for name, import_name, version, critical in ml_packages:
        result = check_python_package(name, import_name, version, critical)
        print_result(result)
        all_results.append(result)
        if result.status == Status.ERROR and critical:
            critical_failures.append(result)

    # Data & Visualization
    print_header("Data & Visualization Packages")
    data_packages = [
        ("pandas", "pandas", "2.0.0", True),
        ("numpy", "numpy", "1.24.0", True),
        ("matplotlib", "matplotlib", "3.7.0", True),
        ("seaborn", "seaborn", "0.13.0", False),
    ]

    for name, import_name, version, critical in data_packages:
        result = check_python_package(name, import_name, version, critical)
        print_result(result)
        all_results.append(result)
        if result.status == Status.ERROR and critical:
            critical_failures.append(result)

    # R Environment
    print_header("R Environment")
    r_result, r_version = check_r_availability()
    print_result(r_result)
    all_results.append(r_result)

    # R Packages (only if R is available)
    if r_version:
        print_header("R Packages")
        r_packages = ["grf", "mediation", "rdrobust", "rddensity"]

        for package in r_packages:
            result = check_r_package(package)
            print_result(result)
            all_results.append(result)

    # rpy2 Bridge
    print_header("Python-R Integration")
    rpy2_result = check_python_package("rpy2", "rpy2", "3.5.0", False)
    print_result(rpy2_result)
    all_results.append(rpy2_result)

    if rpy2_result.status == Status.OK:
        bridge_result = check_rpy2_bridge()
        print_result(bridge_result)
        all_results.append(bridge_result)

    # Stata (Optional)
    print_header("Stata Integration (Optional)")
    stata_result = check_stata_availability()
    print_result(stata_result)
    all_results.append(stata_result)

    # Summary
    print_header("Summary")

    ok_count = sum(1 for r in all_results if r.status == Status.OK)
    warning_count = sum(1 for r in all_results if r.status == Status.WARNING)
    error_count = sum(1 for r in all_results if r.status == Status.ERROR)
    skip_count = sum(1 for r in all_results if r.status == Status.SKIP)

    print(f"  {colorize(SYMBOLS[Status.OK], Status.OK)} Passed:   {ok_count}")
    print(f"  {colorize(SYMBOLS[Status.WARNING], Status.WARNING)} Warnings: {warning_count}")
    print(f"  {colorize(SYMBOLS[Status.ERROR], Status.ERROR)} Errors:   {error_count}")
    print(f"  {colorize(SYMBOLS[Status.SKIP], Status.SKIP)} Skipped:  {skip_count}")

    print()

    if critical_failures:
        print(colorize("  CRITICAL: Some required dependencies are missing!", Status.ERROR))
        print()
        print("  Missing critical packages:")
        for result in critical_failures:
            print(f"    - {result.name}")
        print()
        print("  To install missing packages, run:")
        print("    pip install -r requirements.txt")
        print()
        return 1
    elif warning_count > 0:
        print(colorize("  WARNING: Some optional dependencies are missing or outdated.", Status.WARNING))
        print()
        print("  The environment is functional, but some features may be limited.")
        print("  Consider running: pip install -r requirements.txt")
        print()
        return 0
    else:
        print(colorize("  SUCCESS: All dependencies are properly installed!", Status.OK))
        print()
        return 0


if __name__ == "__main__":
    sys.exit(main())
