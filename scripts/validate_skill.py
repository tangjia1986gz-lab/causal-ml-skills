#!/usr/bin/env python3
"""
Skill Validation Script for K-Dense Quality Standards.

This script validates skill directories against the K-Dense quality standards
used in the causal-ml-skills project.

Usage:
    python validate_skill.py <skill_path>           # Validate single skill
    python validate_skill.py --all                  # Validate all skills
    python validate_skill.py --phase <n>            # Validate skills for phase n
    python validate_skill.py --verbose              # Detailed output
    python validate_skill.py --fix                  # Auto-fix simple issues
    python validate_skill.py --json                 # JSON output for CI/CD

Examples:
    python validate_skill.py ../skills/classic-methods/estimator-did
    python validate_skill.py --all --verbose
    python validate_skill.py --phase 1 --json
"""

import argparse
import ast
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# =============================================================================
# Constants
# =============================================================================

# Phase definitions based on ROADMAP.md
PHASE_SKILLS = {
    0: ["setup-causal-ml-env"],
    1: ["causal-concept-guide", "estimator-did", "estimator-rd", "estimator-iv", "estimator-psm"],
    2: ["ml-preprocessing", "ml-model-linear", "ml-model-tree", "ml-model-advanced"],
    3: ["causal-ddml", "causal-mediation-ml", "causal-forest"],
    4: ["paper-replication-workflow"],
}

# Required frontmatter fields in SKILL.md
REQUIRED_FRONTMATTER_FIELDS = ["name", "description"]

# Required sections in SKILL.md (patterns to match)
# Each entry can be a string (exact match) or list (any match from list)
REQUIRED_SKILL_SECTIONS = [
    "Overview",
    ["When to Use", "When to use"],
    ["Prerequisites", "Requirements", "Data Requirements", "Identification Assumptions"],
    ["Workflow", "Core Workflow", "Phase 1"],  # Workflow can start with phases
    ["Implementation", "Python Implementation", "Python Code", "Main Estimation",
     "Phase 3", "Estimation"],  # Implementation section variants
]

# Optional sections (for full K-Dense compliance)
OPTIONAL_SKILL_SECTIONS = [
    ["Diagnostics", "Diagnostics & Validation", "Robustness", "Robustness Checks"],
    ["Common Mistakes", "Mistakes", "Pitfalls"],
    ["Examples", "Example"],
    ["References", "Reference"],
]


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class ValidationError:
    """A single validation error."""
    gate: int
    severity: str  # "error", "warning", "info"
    message: str
    file: Optional[str] = None
    line: Optional[int] = None
    suggestion: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON output."""
        return {
            "gate": self.gate,
            "severity": self.severity,
            "message": self.message,
            "file": self.file,
            "line": self.line,
            "suggestion": self.suggestion,
        }


@dataclass
class ValidationResult:
    """Result of validating a skill."""
    skill_name: str
    skill_path: str
    passed: bool
    errors: List[ValidationError] = field(default_factory=list)
    warnings: List[ValidationError] = field(default_factory=list)
    info: List[ValidationError] = field(default_factory=list)
    gate_results: Dict[int, bool] = field(default_factory=dict)

    def add_error(self, error: ValidationError) -> None:
        """Add an error to the appropriate list."""
        if error.severity == "error":
            self.errors.append(error)
        elif error.severity == "warning":
            self.warnings.append(error)
        else:
            self.info.append(error)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON output."""
        return {
            "skill_name": self.skill_name,
            "skill_path": self.skill_path,
            "passed": self.passed,
            "error_count": len(self.errors),
            "warning_count": len(self.warnings),
            "info_count": len(self.info),
            "gate_results": self.gate_results,
            "errors": [e.to_dict() for e in self.errors],
            "warnings": [w.to_dict() for w in self.warnings],
            "info": [i.to_dict() for i in self.info],
        }


# =============================================================================
# Helper Functions
# =============================================================================

def validate_yaml_frontmatter(content: str) -> Tuple[bool, Dict[str, str], List[str]]:
    """
    Parse and validate YAML frontmatter from SKILL.md content.

    Parameters
    ----------
    content : str
        Full content of SKILL.md file

    Returns
    -------
    Tuple[bool, dict, List[str]]
        (is_valid, parsed_frontmatter, list_of_errors)
    """
    errors = []
    frontmatter = {}

    # Check for frontmatter delimiters
    if not content.startswith("---"):
        errors.append("SKILL.md must start with YAML frontmatter (---)")
        return False, frontmatter, errors

    # Find the end of frontmatter
    lines = content.split("\n")
    end_idx = -1
    for i, line in enumerate(lines[1:], start=1):
        if line.strip() == "---":
            end_idx = i
            break

    if end_idx == -1:
        errors.append("YAML frontmatter not properly closed (missing second ---)")
        return False, frontmatter, errors

    # Parse frontmatter
    frontmatter_lines = lines[1:end_idx]
    for line in frontmatter_lines:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if ":" in line:
            key, value = line.split(":", 1)
            key = key.strip()
            value = value.strip()
            # Remove surrounding quotes if present
            if value.startswith('"') and value.endswith('"'):
                value = value[1:-1]
            elif value.startswith("'") and value.endswith("'"):
                value = value[1:-1]
            frontmatter[key] = value

    # Check required fields
    for field in REQUIRED_FRONTMATTER_FIELDS:
        if field not in frontmatter:
            errors.append(f"Missing required frontmatter field: {field}")
        elif not frontmatter[field]:
            errors.append(f"Empty frontmatter field: {field}")

    is_valid = len(errors) == 0
    return is_valid, frontmatter, errors


def validate_python_syntax(filepath: Path) -> Tuple[bool, List[str]]:
    """
    Validate Python file syntax using ast.parse.

    Parameters
    ----------
    filepath : Path
        Path to Python file

    Returns
    -------
    Tuple[bool, List[str]]
        (is_valid, list_of_errors)
    """
    errors = []

    try:
        with open(filepath, "r", encoding="utf-8") as f:
            source = f.read()
    except Exception as e:
        errors.append(f"Cannot read file: {e}")
        return False, errors

    try:
        ast.parse(source)
    except SyntaxError as e:
        errors.append(f"Syntax error at line {e.lineno}: {e.msg}")
        return False, errors

    return True, errors


def check_section_exists(content: str, section: Any) -> bool:
    """
    Check if a section header exists in markdown content.

    Parameters
    ----------
    content : str
        Markdown content
    section : str or list
        Section name(s) to search for (without # prefix).
        If a list, returns True if any of the sections exist.

    Returns
    -------
    bool
        True if section exists
    """
    # Handle list of alternative section names
    if isinstance(section, list):
        return any(check_section_exists(content, s) for s in section)

    # Match section headers like "## Overview" or "### Overview"
    # Use a more lenient pattern that doesn't require end of line
    # Note: {1,4} must be escaped in f-string as {{1,4}} or use raw string concatenation
    pattern = r"^#{1,4}\s+" + re.escape(section)
    return bool(re.search(pattern, content, re.MULTILINE | re.IGNORECASE))


def get_section_display_name(section: Any) -> str:
    """Get display name for a section (first item if list)."""
    if isinstance(section, list):
        return section[0]
    return section


def check_type_hints(filepath: Path) -> Tuple[int, int, List[str]]:
    """
    Check for type hints in Python function signatures.

    Parameters
    ----------
    filepath : Path
        Path to Python file

    Returns
    -------
    Tuple[int, int, List[str]]
        (functions_with_hints, total_functions, list_of_functions_without_hints)
    """
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            source = f.read()
        tree = ast.parse(source)
    except Exception:
        return 0, 0, []

    functions_without_hints = []
    functions_with_hints = 0
    total_functions = 0

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            # Skip private/dunder methods for this check
            if node.name.startswith("_") and not node.name.startswith("__"):
                continue

            total_functions += 1
            has_return_annotation = node.returns is not None

            # Check if any argument has type annotation
            has_arg_annotations = any(
                arg.annotation is not None
                for arg in node.args.args + node.args.kwonlyargs
            )

            if has_return_annotation or has_arg_annotations:
                functions_with_hints += 1
            else:
                functions_without_hints.append(node.name)

    return functions_with_hints, total_functions, functions_without_hints


def check_docstrings(filepath: Path) -> Tuple[int, int, List[str]]:
    """
    Check for docstrings in public functions and classes.

    Parameters
    ----------
    filepath : Path
        Path to Python file

    Returns
    -------
    Tuple[int, int, List[str]]
        (items_with_docstrings, total_items, list_of_items_without_docstrings)
    """
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            source = f.read()
        tree = ast.parse(source)
    except Exception:
        return 0, 0, []

    missing_docstrings = []
    with_docstrings = 0
    total = 0

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            # Skip private items
            if node.name.startswith("_") and not node.name.startswith("__"):
                continue

            total += 1
            docstring = ast.get_docstring(node)
            if docstring:
                with_docstrings += 1
            else:
                missing_docstrings.append(node.name)

    return with_docstrings, total, missing_docstrings


def check_circular_imports(skill_path: Path) -> List[str]:
    """
    Check for potential circular import issues.

    This is a simplified check that looks for import patterns
    that might cause circular imports.

    Parameters
    ----------
    skill_path : Path
        Path to skill directory

    Returns
    -------
    List[str]
        List of potential circular import warnings
    """
    warnings = []
    python_files = list(skill_path.glob("*.py"))

    # Build import graph
    imports: Dict[str, List[str]] = {}

    for py_file in python_files:
        try:
            with open(py_file, "r", encoding="utf-8") as f:
                source = f.read()
            tree = ast.parse(source)
        except Exception:
            continue

        file_imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    file_imports.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    file_imports.append(node.module)

        imports[py_file.stem] = file_imports

    # Check for circular patterns within the skill
    for file_name, file_imports in imports.items():
        for imported in file_imports:
            if imported in imports and file_name in imports.get(imported, []):
                warnings.append(
                    f"Potential circular import between {file_name}.py and {imported}.py"
                )

    return warnings


# =============================================================================
# Gate Validators
# =============================================================================

def validate_gate1_structure(skill_path: Path, result: ValidationResult) -> bool:
    """
    Gate 1: Basic Structure Validation.

    Checks:
    - SKILL.md exists with valid YAML frontmatter
    - Required frontmatter fields present
    - Required SKILL.md sections present
    - Python files have valid syntax
    - __init__.py exists (optional warning)
    """
    skill_path = Path(skill_path)
    passed = True

    # Check SKILL.md exists
    skill_md = skill_path / "SKILL.md"
    if not skill_md.exists():
        result.add_error(ValidationError(
            gate=1,
            severity="error",
            message="SKILL.md not found",
            suggestion="Create SKILL.md using the SKILL-TEMPLATE.md template"
        ))
        passed = False
    else:
        # Validate frontmatter
        with open(skill_md, "r", encoding="utf-8") as f:
            content = f.read()

        fm_valid, frontmatter, fm_errors = validate_yaml_frontmatter(content)
        if not fm_valid:
            for err in fm_errors:
                result.add_error(ValidationError(
                    gate=1,
                    severity="error",
                    message=err,
                    file="SKILL.md"
                ))
            passed = False

        # Check required sections
        for section in REQUIRED_SKILL_SECTIONS:
            if not check_section_exists(content, section):
                display_name = get_section_display_name(section)
                result.add_error(ValidationError(
                    gate=1,
                    severity="error",
                    message=f"Missing required section: {display_name}",
                    file="SKILL.md",
                    suggestion=f"Add a '## {display_name}' section to SKILL.md"
                ))
                passed = False

        # Check optional sections (info only)
        for section in OPTIONAL_SKILL_SECTIONS:
            if not check_section_exists(content, section):
                display_name = get_section_display_name(section)
                result.add_error(ValidationError(
                    gate=1,
                    severity="info",
                    message=f"Optional section missing: {display_name}",
                    file="SKILL.md",
                    suggestion=f"Consider adding a '## {display_name}' section"
                ))

    # Check Python files syntax
    python_files = list(skill_path.glob("*.py"))
    for py_file in python_files:
        syntax_valid, syntax_errors = validate_python_syntax(py_file)
        if not syntax_valid:
            for err in syntax_errors:
                result.add_error(ValidationError(
                    gate=1,
                    severity="error",
                    message=err,
                    file=py_file.name
                ))
            passed = False

    # Check __init__.py (warning only)
    init_py = skill_path / "__init__.py"
    if not init_py.exists():
        result.add_error(ValidationError(
            gate=1,
            severity="warning",
            message="__init__.py not found",
            suggestion="Create __init__.py for proper Python package structure"
        ))

    result.gate_results[1] = passed
    return passed


def validate_gate2_kdense(skill_path: Path, result: ValidationResult) -> bool:
    """
    Gate 2: K-Dense Structure (optional for upgraded skills).

    Checks:
    - references/ directory with 3+ .md files
    - scripts/ directory with 2+ .py files
    - assets/ directory structure
    """
    skill_path = Path(skill_path)
    passed = True

    # Check references/ directory
    references_dir = skill_path / "references"
    if references_dir.exists():
        md_files = list(references_dir.glob("*.md"))
        if len(md_files) < 3:
            result.add_error(ValidationError(
                gate=2,
                severity="warning",
                message=f"K-Dense: references/ has {len(md_files)} .md files (recommend 3+)",
                suggestion="Add more reference documentation files"
            ))
            passed = False
    else:
        result.add_error(ValidationError(
            gate=2,
            severity="info",
            message="K-Dense: references/ directory not found",
            suggestion="Create references/ directory for K-Dense compliance"
        ))
        passed = False

    # Check scripts/ directory
    scripts_dir = skill_path / "scripts"
    if scripts_dir.exists():
        py_files = list(scripts_dir.glob("*.py"))
        if len(py_files) < 2:
            result.add_error(ValidationError(
                gate=2,
                severity="warning",
                message=f"K-Dense: scripts/ has {len(py_files)} .py files (recommend 2+)",
                suggestion="Add more utility scripts"
            ))
            passed = False
    else:
        result.add_error(ValidationError(
            gate=2,
            severity="info",
            message="K-Dense: scripts/ directory not found",
            suggestion="Create scripts/ directory for K-Dense compliance"
        ))
        passed = False

    # Check assets/ directory (info only)
    assets_dir = skill_path / "assets"
    if not assets_dir.exists():
        result.add_error(ValidationError(
            gate=2,
            severity="info",
            message="K-Dense: assets/ directory not found",
            suggestion="Create assets/ directory for images and other assets"
        ))

    result.gate_results[2] = passed
    return passed


def validate_gate3_code_quality(skill_path: Path, result: ValidationResult) -> bool:
    """
    Gate 3: Code Quality.

    Checks:
    - Python files pass syntax check
    - Type hints present in function signatures
    - Docstrings present for public functions
    - No circular imports
    """
    skill_path = Path(skill_path)
    passed = True

    python_files = list(skill_path.glob("*.py"))
    if not python_files:
        result.add_error(ValidationError(
            gate=3,
            severity="warning",
            message="No Python files found in skill directory"
        ))
        result.gate_results[3] = True
        return True

    for py_file in python_files:
        # Already checked syntax in Gate 1, but verify again
        syntax_valid, syntax_errors = validate_python_syntax(py_file)
        if not syntax_valid:
            for err in syntax_errors:
                result.add_error(ValidationError(
                    gate=3,
                    severity="error",
                    message=f"Syntax error: {err}",
                    file=py_file.name
                ))
            passed = False
            continue

        # Check type hints
        with_hints, total, missing_hints = check_type_hints(py_file)
        if total > 0:
            hint_ratio = with_hints / total
            if hint_ratio < 0.5:
                result.add_error(ValidationError(
                    gate=3,
                    severity="warning",
                    message=f"Low type hint coverage: {with_hints}/{total} functions ({hint_ratio:.0%})",
                    file=py_file.name,
                    suggestion=f"Add type hints to: {', '.join(missing_hints[:5])}"
                ))
                if hint_ratio < 0.25:
                    passed = False

        # Check docstrings
        with_docs, total_docs, missing_docs = check_docstrings(py_file)
        if total_docs > 0:
            doc_ratio = with_docs / total_docs
            if doc_ratio < 0.5:
                result.add_error(ValidationError(
                    gate=3,
                    severity="warning",
                    message=f"Low docstring coverage: {with_docs}/{total_docs} items ({doc_ratio:.0%})",
                    file=py_file.name,
                    suggestion=f"Add docstrings to: {', '.join(missing_docs[:5])}"
                ))
                if doc_ratio < 0.25:
                    passed = False

    # Check circular imports
    circular_warnings = check_circular_imports(skill_path)
    for warning in circular_warnings:
        result.add_error(ValidationError(
            gate=3,
            severity="warning",
            message=warning
        ))

    result.gate_results[3] = passed
    return passed


def validate_gate4_integration(
    skill_path: Path,
    result: ValidationResult,
    lib_path: Optional[Path] = None
) -> bool:
    """
    Gate 4: Integration.

    Checks:
    - Imports from lib.python work correctly
    - CausalInput/CausalOutput contracts followed
    - Cross-skill references valid
    """
    skill_path = Path(skill_path)
    passed = True

    if lib_path is None:
        # Try to find lib path relative to skill
        potential_lib = skill_path.parent.parent.parent / "lib" / "python"
        if potential_lib.exists():
            lib_path = potential_lib

    python_files = list(skill_path.glob("*.py"))

    for py_file in python_files:
        try:
            with open(py_file, "r", encoding="utf-8") as f:
                source = f.read()
            tree = ast.parse(source)
        except Exception:
            continue

        # Check for lib imports
        has_lib_import = False
        uses_causal_contracts = False

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if "data_loader" in alias.name or "diagnostics" in alias.name:
                        has_lib_import = True
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    if "data_loader" in node.module or "diagnostics" in node.module:
                        has_lib_import = True
                    # Check for CausalInput/CausalOutput
                    if node.names:
                        for name in node.names:
                            if name.name in ["CausalInput", "CausalOutput"]:
                                uses_causal_contracts = True

        # If file has substantial code, check for lib usage
        if len(source) > 500:  # Arbitrary threshold for "substantial"
            if not has_lib_import:
                result.add_error(ValidationError(
                    gate=4,
                    severity="info",
                    message="No imports from lib.python detected",
                    file=py_file.name,
                    suggestion="Consider using shared utilities from lib.python"
                ))

            # Check if estimator-type skill uses CausalOutput
            if "estimator" in skill_path.name.lower() and not uses_causal_contracts:
                result.add_error(ValidationError(
                    gate=4,
                    severity="warning",
                    message="Estimator skill does not use CausalInput/CausalOutput contracts",
                    file=py_file.name,
                    suggestion="Use CausalOutput for standardized return values"
                ))

    # Verify lib imports actually work
    if lib_path and lib_path.exists():
        try:
            sys.path.insert(0, str(lib_path))
            import data_loader  # noqa: F401
            import diagnostics  # noqa: F401
            sys.path.remove(str(lib_path))
        except ImportError as e:
            result.add_error(ValidationError(
                gate=4,
                severity="error",
                message=f"lib.python import failed: {e}",
                suggestion="Ensure lib/python modules are properly configured"
            ))
            passed = False

    result.gate_results[4] = passed
    return passed


# =============================================================================
# Main Validator
# =============================================================================

def validate_skill(
    skill_path: Path,
    verbose: bool = False,
    check_kdense: bool = False,
    lib_path: Optional[Path] = None
) -> ValidationResult:
    """
    Validate a single skill directory.

    Parameters
    ----------
    skill_path : Path
        Path to skill directory
    verbose : bool
        Print detailed output
    check_kdense : bool
        Include K-Dense structure checks (Gate 2)
    lib_path : Path, optional
        Path to lib/python for integration checks

    Returns
    -------
    ValidationResult
        Validation results
    """
    skill_path = Path(skill_path)

    if not skill_path.exists():
        result = ValidationResult(
            skill_name=skill_path.name,
            skill_path=str(skill_path),
            passed=False
        )
        result.add_error(ValidationError(
            gate=0,
            severity="error",
            message=f"Skill path does not exist: {skill_path}"
        ))
        return result

    if not skill_path.is_dir():
        result = ValidationResult(
            skill_name=skill_path.name,
            skill_path=str(skill_path),
            passed=False
        )
        result.add_error(ValidationError(
            gate=0,
            severity="error",
            message=f"Skill path is not a directory: {skill_path}"
        ))
        return result

    result = ValidationResult(
        skill_name=skill_path.name,
        skill_path=str(skill_path),
        passed=True
    )

    # Run all gates
    gate1_passed = validate_gate1_structure(skill_path, result)

    if check_kdense:
        gate2_passed = validate_gate2_kdense(skill_path, result)
    else:
        gate2_passed = True
        result.gate_results[2] = True

    gate3_passed = validate_gate3_code_quality(skill_path, result)
    gate4_passed = validate_gate4_integration(skill_path, result, lib_path)

    # Overall pass requires Gate 1 and Gate 3 (Gate 2 and 4 are softer requirements)
    result.passed = gate1_passed and gate3_passed

    if verbose:
        print_validation_result(result)

    return result


def print_validation_result(result: ValidationResult) -> None:
    """Print validation result in human-readable format."""
    status = "PASSED" if result.passed else "FAILED"
    status_color = "\033[92m" if result.passed else "\033[91m"
    reset = "\033[0m"

    print()
    print("=" * 60)
    print(f"Skill: {result.skill_name}")
    print(f"Path: {result.skill_path}")
    print(f"Status: {status_color}{status}{reset}")
    print("=" * 60)

    # Gate results
    print("\nGate Results:")
    gate_names = {
        1: "Structure Validation",
        2: "K-Dense Structure",
        3: "Code Quality",
        4: "Integration",
    }
    for gate, passed in sorted(result.gate_results.items()):
        gate_status = "\033[92mPASS\033[0m" if passed else "\033[91mFAIL\033[0m"
        print(f"  Gate {gate} ({gate_names.get(gate, 'Unknown')}): {gate_status}")

    # Errors
    if result.errors:
        print(f"\n\033[91mErrors ({len(result.errors)}):\033[0m")
        for err in result.errors:
            file_info = f" [{err.file}]" if err.file else ""
            line_info = f":{err.line}" if err.line else ""
            print(f"  - [Gate {err.gate}]{file_info}{line_info} {err.message}")
            if err.suggestion:
                print(f"    Suggestion: {err.suggestion}")

    # Warnings
    if result.warnings:
        print(f"\n\033[93mWarnings ({len(result.warnings)}):\033[0m")
        for warn in result.warnings:
            file_info = f" [{warn.file}]" if warn.file else ""
            print(f"  - [Gate {warn.gate}]{file_info} {warn.message}")
            if warn.suggestion:
                print(f"    Suggestion: {warn.suggestion}")

    # Info
    if result.info:
        print(f"\n\033[94mInfo ({len(result.info)}):\033[0m")
        for info in result.info:
            file_info = f" [{info.file}]" if info.file else ""
            print(f"  - [Gate {info.gate}]{file_info} {info.message}")

    print()


def find_all_skills(base_path: Path) -> List[Path]:
    """Find all skill directories under the given base path."""
    skills = []
    skills_dir = base_path / "skills"

    if not skills_dir.exists():
        return skills

    # Look for SKILL.md files to identify skill directories
    for skill_md in skills_dir.rglob("SKILL.md"):
        skills.append(skill_md.parent)

    return sorted(skills)


def find_phase_skills(base_path: Path, phase: int) -> List[Path]:
    """Find skill directories for a specific phase."""
    if phase not in PHASE_SKILLS:
        return []

    skill_names = PHASE_SKILLS[phase]
    all_skills = find_all_skills(base_path)

    return [s for s in all_skills if s.name in skill_names]


def auto_fix_issues(skill_path: Path, result: ValidationResult) -> int:
    """
    Attempt to auto-fix simple issues.

    Currently supports:
    - Creating __init__.py

    Returns
    -------
    int
        Number of issues fixed
    """
    fixed = 0
    skill_path = Path(skill_path)

    # Fix missing __init__.py
    init_py = skill_path / "__init__.py"
    if not init_py.exists():
        for err in result.warnings + result.errors:
            if "__init__.py not found" in err.message:
                init_py.write_text(f'"""{skill_path.name} skill package."""\n')
                print(f"  Created: {init_py}")
                fixed += 1
                break

    return fixed


# =============================================================================
# CLI
# =============================================================================

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Validate skill directories against K-Dense quality standards.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument(
        "skill_path",
        nargs="?",
        help="Path to skill directory to validate"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Validate all skills in the project"
    )
    parser.add_argument(
        "--phase",
        type=int,
        choices=[0, 1, 2, 3, 4],
        help="Validate skills for a specific phase (0-4)"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Print detailed output"
    )
    parser.add_argument(
        "--fix",
        action="store_true",
        help="Attempt to auto-fix simple issues"
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON (for CI/CD)"
    )
    parser.add_argument(
        "--kdense",
        action="store_true",
        help="Include K-Dense structure validation (Gate 2)"
    )
    parser.add_argument(
        "--base-path",
        type=Path,
        default=None,
        help="Base path for skill discovery (default: parent of this script)"
    )

    args = parser.parse_args()

    # Determine base path
    if args.base_path:
        base_path = args.base_path
    else:
        # Script is in causal-ml-skills/scripts/
        base_path = Path(__file__).parent.parent

    lib_path = base_path / "lib" / "python"

    # Determine which skills to validate
    skills_to_validate: List[Path] = []

    if args.all:
        skills_to_validate = find_all_skills(base_path)
    elif args.phase is not None:
        skills_to_validate = find_phase_skills(base_path, args.phase)
    elif args.skill_path:
        skills_to_validate = [Path(args.skill_path)]
    else:
        parser.print_help()
        sys.exit(1)

    if not skills_to_validate:
        print("No skills found to validate.")
        sys.exit(1)

    # Validate each skill
    results: List[ValidationResult] = []
    for skill_path in skills_to_validate:
        result = validate_skill(
            skill_path,
            verbose=args.verbose and not args.json,
            check_kdense=args.kdense,
            lib_path=lib_path
        )
        results.append(result)

        if args.fix and not result.passed:
            fixed = auto_fix_issues(skill_path, result)
            if fixed > 0 and args.verbose:
                print(f"  Fixed {fixed} issue(s)")

    # Output results
    if args.json:
        output = {
            "total_skills": len(results),
            "passed": sum(1 for r in results if r.passed),
            "failed": sum(1 for r in results if not r.passed),
            "results": [r.to_dict() for r in results]
        }
        print(json.dumps(output, indent=2))
    else:
        # Summary
        print("\n" + "=" * 60)
        print("VALIDATION SUMMARY")
        print("=" * 60)
        passed = sum(1 for r in results if r.passed)
        failed = len(results) - passed
        print(f"Total: {len(results)} skill(s)")
        print(f"Passed: \033[92m{passed}\033[0m")
        print(f"Failed: \033[91m{failed}\033[0m")

        if failed > 0:
            print("\nFailed skills:")
            for r in results:
                if not r.passed:
                    print(f"  - {r.skill_name}: {len(r.errors)} error(s)")

    # Exit with appropriate code
    all_passed = all(r.passed for r in results)
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
