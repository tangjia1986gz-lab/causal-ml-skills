#!/usr/bin/env python
"""
Validate Replication Package
============================

Check replication package completeness against AEA Data Editor requirements
and best practices for computational reproducibility.

This script validates:
- Directory structure
- Required files (README, LICENSE, etc.)
- Code organization
- Data documentation
- Output mapping

Usage:
    python validate_replication.py [project_path] [options]

Examples:
    # Validate current directory
    python validate_replication.py

    # Validate specific project
    python validate_replication.py /path/to/replication_package

    # Output JSON report
    python validate_replication.py --format json --output report.json

    # Strict mode (fail on warnings)
    python validate_replication.py --strict
"""

import argparse
import json
import sys
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Callable, Optional


class ValidationLevel(Enum):
    """Severity level for validation checks."""
    ERROR = "error"      # Must fix before submission
    WARNING = "warning"  # Should fix, but not blocking
    INFO = "info"        # Suggestion for improvement


@dataclass
class ValidationResult:
    """Result of a single validation check."""
    name: str
    passed: bool
    level: ValidationLevel
    message: str
    details: str = ""


@dataclass
class ValidationReport:
    """Complete validation report."""
    project_path: Path
    timestamp: str
    results: list = field(default_factory=list)

    @property
    def errors(self) -> list:
        return [r for r in self.results if not r.passed and r.level == ValidationLevel.ERROR]

    @property
    def warnings(self) -> list:
        return [r for r in self.results if not r.passed and r.level == ValidationLevel.WARNING]

    @property
    def passed(self) -> bool:
        return len(self.errors) == 0

    def to_dict(self) -> dict:
        return {
            "project_path": str(self.project_path),
            "timestamp": self.timestamp,
            "summary": {
                "total_checks": len(self.results),
                "passed": sum(1 for r in self.results if r.passed),
                "errors": len(self.errors),
                "warnings": len(self.warnings),
                "overall_passed": self.passed
            },
            "results": [
                {
                    "name": r.name,
                    "passed": r.passed,
                    "level": r.level.value,
                    "message": r.message,
                    "details": r.details
                }
                for r in self.results
            ]
        }


class ReplicationValidator:
    """Validate replication package completeness."""

    # Required directories
    REQUIRED_DIRS = [
        "code",
        "data",
        "output",
    ]

    # Recommended directories
    RECOMMENDED_DIRS = [
        "data/raw",
        "data/processed",
        "output/tables",
        "output/figures",
        "docs",
    ]

    # Required files
    REQUIRED_FILES = [
        "README.md",
        "LICENSE",
    ]

    # Recommended files
    RECOMMENDED_FILES = [
        "requirements.txt",
        ".gitignore",
        "CITATION.cff",
    ]

    # README required sections
    README_REQUIRED_SECTIONS = [
        "data availability",
        "computational requirements",
        "instructions",
    ]

    # README recommended sections
    README_RECOMMENDED_SECTIONS = [
        "output mapping",
        "contact",
        "runtime",
    ]

    def __init__(self, project_path: Path):
        self.project_path = project_path
        self.results = []

    def add_result(
        self,
        name: str,
        passed: bool,
        level: ValidationLevel,
        message: str,
        details: str = ""
    ):
        """Add a validation result."""
        self.results.append(ValidationResult(
            name=name,
            passed=passed,
            level=level,
            message=message,
            details=details
        ))

    def validate_directory_exists(self) -> bool:
        """Check if project directory exists."""
        exists = self.project_path.exists() and self.project_path.is_dir()
        self.add_result(
            name="Project directory exists",
            passed=exists,
            level=ValidationLevel.ERROR,
            message="Project directory exists" if exists else f"Project directory not found: {self.project_path}"
        )
        return exists

    def validate_required_directories(self):
        """Check required directories exist."""
        for dir_name in self.REQUIRED_DIRS:
            dir_path = self.project_path / dir_name
            exists = dir_path.exists() and dir_path.is_dir()
            self.add_result(
                name=f"Required directory: {dir_name}",
                passed=exists,
                level=ValidationLevel.ERROR,
                message=f"Directory '{dir_name}' exists" if exists else f"Missing required directory: {dir_name}"
            )

    def validate_recommended_directories(self):
        """Check recommended directories exist."""
        for dir_name in self.RECOMMENDED_DIRS:
            dir_path = self.project_path / dir_name
            exists = dir_path.exists() and dir_path.is_dir()
            self.add_result(
                name=f"Recommended directory: {dir_name}",
                passed=exists,
                level=ValidationLevel.WARNING,
                message=f"Directory '{dir_name}' exists" if exists else f"Missing recommended directory: {dir_name}"
            )

    def validate_required_files(self):
        """Check required files exist."""
        for file_name in self.REQUIRED_FILES:
            file_path = self.project_path / file_name
            exists = file_path.exists() and file_path.is_file()
            self.add_result(
                name=f"Required file: {file_name}",
                passed=exists,
                level=ValidationLevel.ERROR,
                message=f"File '{file_name}' exists" if exists else f"Missing required file: {file_name}"
            )

    def validate_recommended_files(self):
        """Check recommended files exist."""
        for file_name in self.RECOMMENDED_FILES:
            file_path = self.project_path / file_name
            exists = file_path.exists() and file_path.is_file()
            self.add_result(
                name=f"Recommended file: {file_name}",
                passed=exists,
                level=ValidationLevel.WARNING,
                message=f"File '{file_name}' exists" if exists else f"Missing recommended file: {file_name}"
            )

    def validate_readme_content(self):
        """Check README has required sections."""
        readme_path = self.project_path / "README.md"

        if not readme_path.exists():
            return

        readme_content = readme_path.read_text().lower()

        # Check required sections
        for section in self.README_REQUIRED_SECTIONS:
            found = section.lower() in readme_content
            self.add_result(
                name=f"README section: {section}",
                passed=found,
                level=ValidationLevel.ERROR,
                message=f"README contains '{section}' section" if found else f"README missing required section: {section}"
            )

        # Check recommended sections
        for section in self.README_RECOMMENDED_SECTIONS:
            found = section.lower() in readme_content
            self.add_result(
                name=f"README section (recommended): {section}",
                passed=found,
                level=ValidationLevel.INFO,
                message=f"README contains '{section}' section" if found else f"README could include: {section}"
            )

    def validate_master_script(self):
        """Check for master script that runs all analysis."""
        code_dir = self.project_path / "code"

        if not code_dir.exists():
            return

        # Look for master script
        master_patterns = [
            "00_master.py",
            "master.py",
            "main.py",
            "00_master.do",
            "master.do",
            "00_master.R",
            "master.R",
            "Makefile",
        ]

        found_master = False
        for pattern in master_patterns:
            if (code_dir / pattern).exists():
                found_master = True
                break

        self.add_result(
            name="Master script exists",
            passed=found_master,
            level=ValidationLevel.ERROR,
            message="Master script found" if found_master else "No master script found in code/",
            details="Create 00_master.py or similar to enable one-click reproducibility"
        )

    def validate_script_naming(self):
        """Check scripts follow numbered naming convention."""
        code_dir = self.project_path / "code"

        if not code_dir.exists():
            return

        scripts = list(code_dir.glob("*.py")) + list(code_dir.glob("*.do")) + list(code_dir.glob("*.R"))

        # Check for numbered prefix pattern
        numbered_scripts = [s for s in scripts if s.name[0:2].isdigit()]
        unnumbered_scripts = [s for s in scripts if not s.name[0:2].isdigit() and s.name != "__init__.py"]

        has_numbered = len(numbered_scripts) > 0
        all_numbered = len(unnumbered_scripts) == 0 or len(scripts) <= 1

        self.add_result(
            name="Scripts use numbered naming",
            passed=has_numbered,
            level=ValidationLevel.WARNING,
            message="Scripts use numbered prefix (01_, 02_, etc.)" if has_numbered else "Consider using numbered script names",
            details=f"Numbered: {len(numbered_scripts)}, Unnumbered: {len(unnumbered_scripts)}"
        )

    def validate_data_documentation(self):
        """Check data directory has documentation."""
        data_dir = self.project_path / "data"

        if not data_dir.exists():
            return

        # Check for README in data directory
        data_readme = data_dir / "README.md"
        has_readme = data_readme.exists()

        self.add_result(
            name="Data directory README",
            passed=has_readme,
            level=ValidationLevel.WARNING,
            message="data/README.md exists" if has_readme else "No README in data/ directory"
        )

        # Check for data in raw directory
        raw_dir = data_dir / "raw"
        if raw_dir.exists():
            data_files = list(raw_dir.glob("*")) - [raw_dir / "README.md"]
            data_files = [f for f in data_files if f.is_file() and not f.name.startswith(".")]
            has_data = len(data_files) > 0

            self.add_result(
                name="Data files present",
                passed=has_data,
                level=ValidationLevel.INFO,
                message=f"Found {len(data_files)} data files in data/raw/" if has_data else "No data files in data/raw/ (may need to be downloaded)"
            )

    def validate_output_directory(self):
        """Check output directory structure."""
        output_dir = self.project_path / "output"

        if not output_dir.exists():
            return

        tables_dir = output_dir / "tables"
        figures_dir = output_dir / "figures"

        has_tables_dir = tables_dir.exists()
        has_figures_dir = figures_dir.exists()

        self.add_result(
            name="Output structure: tables directory",
            passed=has_tables_dir,
            level=ValidationLevel.WARNING,
            message="output/tables/ exists" if has_tables_dir else "Missing output/tables/"
        )

        self.add_result(
            name="Output structure: figures directory",
            passed=has_figures_dir,
            level=ValidationLevel.WARNING,
            message="output/figures/ exists" if has_figures_dir else "Missing output/figures/"
        )

    def validate_no_absolute_paths(self):
        """Check code for hardcoded absolute paths."""
        code_dir = self.project_path / "code"

        if not code_dir.exists():
            return

        scripts = list(code_dir.glob("**/*.py"))
        absolute_path_patterns = [
            "/Users/",
            "/home/",
            "C:\\Users\\",
            "C:/Users/",
            "D:\\",
            "D:/",
        ]

        files_with_absolute = []

        for script in scripts:
            try:
                content = script.read_text()
                for pattern in absolute_path_patterns:
                    if pattern in content:
                        files_with_absolute.append(script.name)
                        break
            except Exception:
                pass

        no_absolute = len(files_with_absolute) == 0

        self.add_result(
            name="No hardcoded absolute paths",
            passed=no_absolute,
            level=ValidationLevel.ERROR if not no_absolute else ValidationLevel.INFO,
            message="No hardcoded absolute paths found" if no_absolute else f"Found potential absolute paths in: {', '.join(files_with_absolute)}",
            details="Use relative paths from project root for reproducibility"
        )

    def validate_requirements_pinned(self):
        """Check if requirements.txt has pinned versions."""
        req_path = self.project_path / "requirements.txt"

        if not req_path.exists():
            return

        content = req_path.read_text()
        lines = [l.strip() for l in content.split("\n") if l.strip() and not l.startswith("#")]

        # Check for pinned versions (==, >=, etc.)
        unpinned = [l for l in lines if "=" not in l and "<" not in l and ">" not in l]

        all_pinned = len(unpinned) == 0

        self.add_result(
            name="Dependencies version-pinned",
            passed=all_pinned,
            level=ValidationLevel.WARNING,
            message="All dependencies have version constraints" if all_pinned else f"{len(unpinned)} dependencies without version constraints",
            details="Pin versions for reproducibility (e.g., pandas==2.0.3)"
        )

    def validate_gitignore(self):
        """Check .gitignore covers common patterns."""
        gitignore_path = self.project_path / ".gitignore"

        if not gitignore_path.exists():
            return

        content = gitignore_path.read_text().lower()

        patterns_to_check = [
            ("__pycache__", "Python cache"),
            (".pyc", "Python compiled files"),
            ("venv", "Virtual environment"),
            (".env", "Environment variables"),
        ]

        missing = []
        for pattern, description in patterns_to_check:
            if pattern not in content:
                missing.append(description)

        has_all = len(missing) == 0

        self.add_result(
            name=".gitignore coverage",
            passed=has_all,
            level=ValidationLevel.INFO,
            message=".gitignore covers common patterns" if has_all else f"Consider adding to .gitignore: {', '.join(missing)}"
        )

    def validate(self) -> ValidationReport:
        """Run all validation checks."""
        # First check if directory exists
        if not self.validate_directory_exists():
            return ValidationReport(
                project_path=self.project_path,
                timestamp=datetime.now().isoformat(),
                results=self.results
            )

        # Run all validations
        self.validate_required_directories()
        self.validate_recommended_directories()
        self.validate_required_files()
        self.validate_recommended_files()
        self.validate_readme_content()
        self.validate_master_script()
        self.validate_script_naming()
        self.validate_data_documentation()
        self.validate_output_directory()
        self.validate_no_absolute_paths()
        self.validate_requirements_pinned()
        self.validate_gitignore()

        return ValidationReport(
            project_path=self.project_path,
            timestamp=datetime.now().isoformat(),
            results=self.results
        )


def print_report(report: ValidationReport, verbose: bool = False):
    """Print validation report to console."""
    print("=" * 70)
    print("REPLICATION PACKAGE VALIDATION REPORT")
    print("=" * 70)
    print()
    print(f"Project: {report.project_path}")
    print(f"Time: {report.timestamp}")
    print()

    # Summary
    passed = sum(1 for r in report.results if r.passed)
    total = len(report.results)
    errors = len(report.errors)
    warnings = len(report.warnings)

    print(f"Summary: {passed}/{total} checks passed")
    print(f"  Errors: {errors}")
    print(f"  Warnings: {warnings}")
    print()

    # Errors
    if report.errors:
        print("-" * 70)
        print("ERRORS (must fix before submission):")
        print("-" * 70)
        for result in report.errors:
            print(f"  [X] {result.name}")
            print(f"      {result.message}")
            if result.details and verbose:
                print(f"      Details: {result.details}")
        print()

    # Warnings
    if report.warnings:
        print("-" * 70)
        print("WARNINGS (should fix):")
        print("-" * 70)
        for result in report.warnings:
            print(f"  [!] {result.name}")
            print(f"      {result.message}")
            if result.details and verbose:
                print(f"      Details: {result.details}")
        print()

    # Passed checks (if verbose)
    if verbose:
        passed_results = [r for r in report.results if r.passed]
        if passed_results:
            print("-" * 70)
            print("PASSED:")
            print("-" * 70)
            for result in passed_results:
                print(f"  [OK] {result.name}")
            print()

    # Final verdict
    print("=" * 70)
    if report.passed:
        print("VALIDATION PASSED")
        if report.warnings:
            print(f"(with {len(report.warnings)} warnings)")
    else:
        print("VALIDATION FAILED")
        print(f"Fix {errors} error(s) before submission")
    print("=" * 70)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Validate replication package completeness.",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        "project_path",
        nargs="?",
        type=Path,
        default=Path.cwd(),
        help="Path to replication package (default: current directory)"
    )
    parser.add_argument(
        "--format",
        choices=["text", "json"],
        default="text",
        help="Output format"
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output file (default: stdout)"
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Treat warnings as errors"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show detailed output"
    )

    args = parser.parse_args()

    # Validate
    validator = ReplicationValidator(args.project_path)
    report = validator.validate()

    # Output
    if args.format == "json":
        output = json.dumps(report.to_dict(), indent=2)
        if args.output:
            args.output.write_text(output)
            print(f"Report saved to: {args.output}")
        else:
            print(output)
    else:
        print_report(report, verbose=args.verbose)

    # Exit code
    if args.strict:
        # Fail on errors or warnings
        if not report.passed or report.warnings:
            sys.exit(1)
    else:
        # Fail only on errors
        if not report.passed:
            sys.exit(1)


if __name__ == "__main__":
    main()
