#!/usr/bin/env python3
"""
Automated Dependency Installation Script

Installs causal inference ML dependencies with proper ordering and conflict resolution.

Usage:
    python install_dependencies.py [--minimal] [--full] [--with-r] [--with-gpu] [--dry-run]

Options:
    --minimal    Install only core packages (Python only)
    --full       Install all packages including optional ones (default)
    --with-r     Install R packages via rpy2 (requires R)
    --with-gpu   Include GPU support for XGBoost
    --dry-run    Show what would be installed without actually installing

Cross-platform support: Windows, macOS, Linux
"""

import sys
import os
import subprocess
import platform
import argparse
from pathlib import Path
from typing import List, Tuple, Optional


# Package installation order (dependencies first)
CORE_PACKAGES = [
    # Build dependencies first
    ("cython", None),
    ("numpy", ">=1.24.0,<2.0"),  # Pin to 1.x for compatibility
    ("scipy", ">=1.10.0"),

    # Data packages
    ("pandas", ">=2.0.0"),

    # Visualization
    ("matplotlib", ">=3.7.0"),
    ("seaborn", ">=0.13.0"),

    # ML foundation
    ("scikit-learn", ">=1.3.0"),
    ("joblib", ">=1.3.0"),

    # Statistical packages
    ("statsmodels", ">=0.14.0"),
    ("linearmodels", ">=6.0"),

    # Gradient boosting
    ("xgboost", ">=2.0.0"),
    ("lightgbm", ">=4.0.0"),

    # SHAP must come before causalml
    ("shap", "==0.42.1"),

    # Core causal packages
    ("econml", ">=0.15.0"),
    ("doubleml", ">=0.7.0"),
    ("causalml", ">=0.15.0"),
    ("dowhy", ">=0.11.0"),

    # Additional utilities
    ("networkx", ">=3.0"),
    ("python-dotenv", ">=1.0.0"),
]

MINIMAL_PACKAGES = [
    ("numpy", ">=1.24.0,<2.0"),
    ("scipy", ">=1.10.0"),
    ("pandas", ">=2.0.0"),
    ("matplotlib", ">=3.7.0"),
    ("scikit-learn", ">=1.3.0"),
    ("statsmodels", ">=0.14.0"),
    ("econml", ">=0.15.0"),
    ("doubleml", ">=0.7.0"),
]

R_INTEGRATION_PACKAGES = [
    ("rpy2", ">=3.5.0"),
]

R_PACKAGES = [
    "grf",
    "mediation",
    "rdrobust",
    "rddensity",
]


def get_pip_command() -> List[str]:
    """Get the appropriate pip command for this platform."""
    return [sys.executable, "-m", "pip"]


def run_command(cmd: List[str], dry_run: bool = False, capture: bool = False) -> Tuple[int, str]:
    """Run a command and return exit code and output."""
    if dry_run:
        print(f"  [DRY RUN] {' '.join(cmd)}")
        return 0, ""

    print(f"  Running: {' '.join(cmd)}")
    try:
        if capture:
            result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', errors='replace')
            return result.returncode, result.stdout + result.stderr
        else:
            result = subprocess.run(cmd)
            return result.returncode, ""
    except Exception as e:
        return 1, str(e)


def check_pip() -> bool:
    """Check if pip is available and up to date."""
    print("\nChecking pip...")
    cmd = get_pip_command() + ["--version"]
    code, output = run_command(cmd, capture=True)
    if code != 0:
        print("  ERROR: pip not found")
        return False
    print(f"  {output.strip()}")
    return True


def upgrade_pip(dry_run: bool = False) -> bool:
    """Upgrade pip to latest version."""
    print("\nUpgrading pip...")
    cmd = get_pip_command() + ["install", "--upgrade", "pip", "setuptools", "wheel"]
    code, _ = run_command(cmd, dry_run)
    return code == 0


def install_package(name: str, version_spec: Optional[str] = None, dry_run: bool = False) -> bool:
    """Install a single Python package."""
    pkg_spec = name
    if version_spec:
        pkg_spec += version_spec

    cmd = get_pip_command() + ["install", pkg_spec]
    code, output = run_command(cmd, dry_run, capture=True)

    if code != 0:
        print(f"  WARNING: Failed to install {name}")
        if output:
            # Print last few lines of error
            lines = output.strip().split('\n')
            for line in lines[-5:]:
                print(f"    {line}")
        return False
    return True


def install_packages(packages: List[Tuple[str, Optional[str]]], dry_run: bool = False) -> Tuple[int, int]:
    """Install a list of packages in order."""
    success = 0
    failed = 0

    for name, version_spec in packages:
        print(f"\n  Installing {name}...")
        if install_package(name, version_spec, dry_run):
            success += 1
        else:
            failed += 1

    return success, failed


def install_r_packages(packages: List[str], dry_run: bool = False) -> Tuple[int, int]:
    """Install R packages."""
    success = 0
    failed = 0

    for package in packages:
        print(f"\n  Installing R package: {package}...")
        cmd = ["R", "--slave", "-e", f'install.packages("{package}", repos="https://cloud.r-project.org/")']
        code, _ = run_command(cmd, dry_run, capture=True)
        if code == 0:
            success += 1
        else:
            failed += 1

    return success, failed


def check_r_available() -> bool:
    """Check if R is available."""
    try:
        result = subprocess.run(["R", "--version"], capture_output=True, timeout=10)
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def setup_lightgbm_macos() -> bool:
    """Setup LightGBM on macOS (especially Apple Silicon)."""
    if platform.system() != "Darwin":
        return True

    print("\n  Setting up LightGBM for macOS...")

    # Check if homebrew is available
    try:
        result = subprocess.run(["brew", "--version"], capture_output=True)
        if result.returncode == 0:
            print("  Installing libomp via Homebrew...")
            subprocess.run(["brew", "install", "libomp"], capture_output=True)
            return True
    except FileNotFoundError:
        print("  Homebrew not found. If LightGBM fails, install libomp manually:")
        print("    brew install libomp")

    return True


def print_banner() -> None:
    """Print script banner."""
    print("=" * 60)
    print(" Causal ML Environment Installer")
    print("=" * 60)
    print(f" Platform: {platform.system()} {platform.release()}")
    print(f" Python:   {platform.python_version()}")
    print("=" * 60)


def main() -> int:
    """Main installation routine."""
    parser = argparse.ArgumentParser(description="Install causal ML dependencies")
    parser.add_argument("--minimal", action="store_true", help="Install only core packages")
    parser.add_argument("--full", action="store_true", help="Install all packages (default)")
    parser.add_argument("--with-r", action="store_true", help="Install R packages via rpy2")
    parser.add_argument("--with-gpu", action="store_true", help="Include GPU support")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be installed")
    args = parser.parse_args()

    print_banner()

    if args.dry_run:
        print("\n[DRY RUN MODE - No packages will be installed]\n")

    # Check pip
    if not check_pip():
        return 1

    # Upgrade pip
    if not upgrade_pip(args.dry_run):
        print("WARNING: Failed to upgrade pip, continuing anyway...")

    # Determine which packages to install
    if args.minimal:
        packages = MINIMAL_PACKAGES.copy()
        print("\n[Minimal Installation Mode]")
    else:
        packages = CORE_PACKAGES.copy()
        print("\n[Full Installation Mode]")

    # Platform-specific setup
    if platform.system() == "Darwin":
        setup_lightgbm_macos()

    # Install Python packages
    print("\n" + "=" * 60)
    print(" Installing Python Packages")
    print("=" * 60)

    py_success, py_failed = install_packages(packages, args.dry_run)

    # R integration
    r_success = r_failed = 0
    if args.with_r:
        print("\n" + "=" * 60)
        print(" Installing R Integration")
        print("=" * 60)

        # Install rpy2
        rpy2_success, rpy2_failed = install_packages(R_INTEGRATION_PACKAGES, args.dry_run)
        py_success += rpy2_success
        py_failed += rpy2_failed

        # Install R packages if R is available
        if check_r_available():
            print("\n  R detected. Installing R packages...")
            r_success, r_failed = install_r_packages(R_PACKAGES, args.dry_run)
        else:
            print("\n  R not found on PATH. Skipping R package installation.")
            print("  To install R packages manually, run in R:")
            print(f'    install.packages(c({", ".join([f\'"{p}\'' for p in R_PACKAGES])}))')

    # Summary
    print("\n" + "=" * 60)
    print(" Installation Summary")
    print("=" * 60)
    print(f"  Python packages - Success: {py_success}, Failed: {py_failed}")
    if args.with_r:
        print(f"  R packages      - Success: {r_success}, Failed: {r_failed}")

    total_failed = py_failed + r_failed
    if total_failed > 0:
        print(f"\n  WARNING: {total_failed} package(s) failed to install.")
        print("  Check the output above for details.")
        print("  See references/troubleshooting.md for common solutions.")
    else:
        print("\n  SUCCESS: All packages installed successfully!")

    # Next steps
    print("\n" + "=" * 60)
    print(" Next Steps")
    print("=" * 60)
    print("  1. Run environment check:")
    print("       python scripts/check_environment.py")
    print()
    if not args.with_r:
        print("  2. To add R integration:")
        print("       python scripts/install_dependencies.py --with-r")
        print()
    print("  3. See SKILL.md for usage instructions")
    print()

    return 1 if total_failed > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
