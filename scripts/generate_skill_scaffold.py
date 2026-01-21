#!/usr/bin/env python3
"""
Skill Scaffold Generator for Causal ML Skills

Generates K-Dense style skill directories with proper structure, templates,
and placeholder files for developing new skills.

Usage:
    python generate_skill_scaffold.py --name estimator-xyz --category classic-methods --type estimator
    python generate_skill_scaffold.py --name my-workflow --category causal-ml --type workflow --minimal

Author: Causal ML Skills Team
Version: 1.0.0
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path
from textwrap import dedent
from typing import Optional


# Valid skill types and categories
VALID_TYPES = ["estimator", "knowledge", "tool", "workflow"]
VALID_CATEGORIES = ["classic-methods", "ml-foundation", "causal-ml", "infrastructure"]

# Type descriptions for documentation
TYPE_DESCRIPTIONS = {
    "estimator": "Causal effect estimator with identification assumptions and robustness checks",
    "knowledge": "Conceptual guidance and domain knowledge",
    "tool": "Data processing, model training, or utility tools",
    "workflow": "End-to-end process orchestration",
}

# Default triggers by type
DEFAULT_TRIGGERS = {
    "estimator": "method-specific-keywords, estimation, treatment-effect",
    "knowledge": "conceptual-question, how-to, what-is",
    "tool": "processing, training, utility-action",
    "workflow": "end-to-end, pipeline, full-analysis",
}


def convert_to_snake_case(name: str) -> str:
    """Convert kebab-case to snake_case for Python module names."""
    return name.replace("-", "_")


def convert_to_title_case(name: str) -> str:
    """Convert kebab-case to Title Case for display."""
    return " ".join(word.capitalize() for word in name.split("-"))


def generate_yaml_frontmatter(
    name: str,
    skill_type: str,
    description: Optional[str] = None
) -> str:
    """Generate YAML frontmatter for SKILL.md."""
    if description is None:
        if skill_type == "estimator":
            description = f"Use when estimating causal effects with {convert_to_title_case(name)}. Triggers on {DEFAULT_TRIGGERS[skill_type]}."
        else:
            description = f"Use when {TYPE_DESCRIPTIONS[skill_type].lower()}. Triggers on {DEFAULT_TRIGGERS[skill_type]}."

    # Format triggers as YAML list
    triggers_list = DEFAULT_TRIGGERS[skill_type].split(", ")
    triggers_yaml = "\n  - ".join(triggers_list)

    # Build YAML frontmatter without dedent to preserve formatting
    return f"""---
name: {name}
description: {description}
version: 0.1.0
type: {skill_type}
triggers:
  - {triggers_yaml}
---
"""


def generate_estimator_skill_md(name: str) -> str:
    """Generate SKILL.md content for estimator type."""
    title = convert_to_title_case(name)
    module_name = convert_to_snake_case(name)

    content = f"""
# Estimator: {title}

> **Version**: 0.1.0 | **Type**: Estimator
> **Aliases**: [Add alternative names]

## Overview

[{title}] estimates causal effects by [describe core mechanism in 1-2 sentences].

**Key Identification Assumption**: [Core assumption that enables causal interpretation]

## When to Use

### Ideal Scenarios
- [Research design scenario 1]
- [Research design scenario 2]

### Data Requirements
- [ ] [Data structure requirement 1]
- [ ] [Data structure requirement 2]
- [ ] [Variable requirements]

### When NOT to Use
- [Violation scenario 1] -> Consider `[alternative-estimator]`
- [Violation scenario 2] -> Consider `[alternative-estimator]`

## Identification Assumptions

| Assumption | Description | Testable? |
|------------|-------------|-----------|
| [Assumption 1] | [What it means] | Yes/No |
| [Assumption 2] | [What it means] | Yes/No |
| [Assumption 3] | [What it means] | Yes/No |

---

## Workflow

```
+-------------------------------------------------------------+
|                    ESTIMATOR WORKFLOW                        |
+-------------------------------------------------------------+
|  1. SETUP          -> Define variables, check data structure |
|  2. PRE-ESTIMATION -> Validate identification assumptions    |
|  3. ESTIMATION     -> Run main model                         |
|  4. DIAGNOSTICS    -> Robustness & sensitivity checks        |
|  5. REPORTING      -> Generate tables & interpretation       |
+-------------------------------------------------------------+
```

### Phase 1: Setup

**Objective**: Prepare data and define model specification

**Inputs Required**:
```python
# Standard CausalInput structure
outcome = "y_variable"       # Outcome variable name
treatment = "d_variable"     # Treatment variable name
controls = ["x1", "x2"]      # Control variables
unit_id = "id"               # Panel: unit identifier
time_id = "year"             # Panel: time identifier
```

**Data Validation Checklist**:
- [ ] No missing values in key variables (or explicit handling strategy)
- [ ] Treatment is binary/continuous as expected
- [ ] Panel structure is balanced (if applicable)
- [ ] Sufficient observations in treatment/control groups

### Phase 2: Pre-Estimation Checks

[TODO: Add assumption tests specific to this estimator]

### Phase 3: Main Estimation

**Model Specification**:

[TODO: Add mathematical model specification]

**Python Implementation**:

```python
from {module_name}_estimator import estimate

result = estimate(
    data=df,
    outcome="y",
    treatment="d",
    controls=["x1", "x2"]
)
```

### Phase 4: Robustness Checks

| Check | Purpose | Implementation |
|-------|---------|----------------|
| Placebo Test | Validate no pre-treatment effects | `placebo_test()` |
| Sensitivity Analysis | Assess robustness to unmeasured confounding | `sensitivity_analysis()` |
| Alternative Specification | Test model specification | `alt_specification()` |

### Phase 5: Reporting

[TODO: Add reporting templates and interpretation guide]

---

## Common Mistakes

### 1. [Common Mistake Category]

**Mistake**: [What people do wrong]

**Why it's wrong**: [Explanation of the problem]

**Correct approach**:
```python
# Correct code
```

---

## Examples

### Example 1: [Classic Application]

**Research Question**: [Question]

**Data**: [Description]

```python
import pandas as pd
from {module_name}_estimator import estimate

# Load data
data = pd.read_csv("example_data.csv")

# Run estimation
result = estimate(
    data=data,
    outcome="outcome_var",
    treatment="treatment_var",
    controls=["control1", "control2"]
)

print(result.summary_table)
```

---

## References

### Seminal Papers
- [Author (Year). Title. Journal.]

### Textbook Treatments
- [Author. Book Title. Chapter X.]

### Software Documentation
- [Package documentation links]

---

## Related Estimators

| Estimator | When to Use Instead |
|-----------|---------------------|
| `[estimator-1]` | [Scenario] |
| `[estimator-2]` | [Scenario] |
"""
    return generate_yaml_frontmatter(name, "estimator") + content


def generate_generic_skill_md(name: str, skill_type: str) -> str:
    """Generate SKILL.md content for non-estimator types."""
    title = convert_to_title_case(name)
    type_title = skill_type.capitalize()

    content = f"""
# {title}

> **Version**: 0.1.0 | **Type**: {type_title}

## Overview

[1-2 sentences defining what this skill does and its core value proposition]

## When to Use

Use this skill when:
- [Specific scenario 1]
- [Specific scenario 2]
- [Specific scenario 3]

**When NOT to use:**
- [Situation where this skill is inappropriate]
- [Alternative skill to use instead]

## Prerequisites

- [ ] Python environment with `[required_packages]`
- [ ] Data in DataFrame format with columns: `[required_columns]`
- [ ] [Other prerequisites]

## Quick Reference

| Parameter | Description | Default |
|-----------|-------------|---------|
| `param1` | Description | `value` |
| `param2` | Description | `value` |

## Core Workflow

### Phase 1: [Phase Name]

**Objective**: [What this phase accomplishes]

**Steps**:
1. [Step 1]
2. [Step 2]

**Verification**:
- [ ] [Checkpoint to verify before proceeding]

### Phase 2: [Phase Name]

[Continue pattern...]

## Implementation

### Python Code Template

```python
# Example implementation
def skill_function(data, outcome, treatment, controls=None):
    \"\"\"
    [Function description]

    Parameters
    ----------
    data : pd.DataFrame
        Input data
    outcome : str
        Name of outcome variable
    treatment : str
        Name of treatment variable
    controls : list, optional
        List of control variable names

    Returns
    -------
    result : CausalOutput
        Estimation results
    \"\"\"
    pass
```

## Diagnostics & Validation

### Required Checks
- [ ] [Check 1]: [What it validates]
- [ ] [Check 2]: [What it validates]

### Interpretation Guide
- **If [condition]**: [interpretation]
- **If [condition]**: [interpretation]

## Common Mistakes

1. **Mistake**: [Description]
   - **Symptom**: [How it manifests]
   - **Fix**: [How to correct]

## Examples

### Example 1: [Basic Usage]

```python
# Load data
import pandas as pd
data = pd.read_csv("example.csv")

# Run analysis
result = skill_function(
    data=data,
    outcome="y",
    treatment="d",
    controls=["x1", "x2"]
)

print(result.summary_table)
```

## References

- [Paper/Book 1]
- [Paper/Book 2]
- [Online Resource]

---

## Appendix: Related Skills

| Skill | Relationship |
|-------|-------------|
| `[related-skill-1]` | [How they relate] |
| `[related-skill-2]` | [How they relate] |
"""
    return generate_yaml_frontmatter(name, skill_type) + content


def generate_reference_file(name: str, ref_type: str) -> str:
    """Generate placeholder content for reference files."""
    title = convert_to_title_case(name)
    ref_title = convert_to_title_case(ref_type)

    templates = {
        "identification_assumptions": dedent(f"""\
            # Identification Assumptions: {title}

            ## Core Assumptions

            ### Assumption 1: [Name]

            **Formal Statement**:
            [Mathematical or formal definition]

            **Intuition**:
            [Plain language explanation]

            **When Violated**:
            - [Scenario 1]
            - [Scenario 2]

            **Testing Approach**:
            - [How to test or validate]

            ### Assumption 2: [Name]

            [Same structure...]

            ## Relationship Between Assumptions

            [How assumptions relate to each other and to causal identification]

            ## References

            - [Key papers on identification for this method]
            """),

        "diagnostic_tests": dedent(f"""\
            # Diagnostic Tests: {title}

            ## Pre-Estimation Diagnostics

            ### Test 1: [Name]

            **Purpose**: [What it tests]

            **Implementation**:
            ```python
            from {convert_to_snake_case(name)}_estimator import test_name

            result = test_name(data, ...)
            ```

            **Interpretation**:
            - Pass: [When to proceed]
            - Warning: [When to be cautious]
            - Fail: [When to stop or reconsider]

            ## Post-Estimation Diagnostics

            [Similar structure for post-estimation tests]

            ## Diagnostic Decision Tree

            ```
            Start
              |
              v
            [Test 1] --Pass--> [Test 2] --Pass--> Proceed
              |                   |
              Fail                Fail
              |                   |
              v                   v
            [Action]           [Action]
            ```
            """),

        "estimation_methods": dedent(f"""\
            # Estimation Methods: {title}

            ## Primary Estimation Approach

            ### Method: [Name]

            **Mathematical Formulation**:
            [Equations and formal specification]

            **Assumptions Required**:
            - [Assumption 1]
            - [Assumption 2]

            **Advantages**:
            - [Pro 1]
            - [Pro 2]

            **Limitations**:
            - [Con 1]
            - [Con 2]

            ## Alternative Estimation Approaches

            ### Method: [Alternative Name]

            [Same structure...]

            ## Comparison Table

            | Method | Complexity | Assumptions | Best For |
            |--------|------------|-------------|----------|
            | [Method 1] | Low/Medium/High | [Key assumption] | [Use case] |
            | [Method 2] | Low/Medium/High | [Key assumption] | [Use case] |

            ## Implementation Notes

            [Practical considerations for implementation]
            """),

        "reporting_standards": dedent(f"""\
            # Reporting Standards: {title}

            ## Required Elements

            1. **Point Estimate and Uncertainty**
               - Effect size with confidence intervals
               - Standard errors (specify type: robust, clustered, etc.)
               - P-values (two-sided unless specified)

            2. **Diagnostic Results**
               - Pre-estimation test results
               - Robustness check results

            3. **Sample Information**
               - Sample size by group
               - Time periods covered
               - Any exclusions and reasons

            ## Table Formats

            ### Main Results Table

            ```
            +----------------------------------------------------------+
            |                    Table X: Main Results                  |
            +----------------------------------------------------------+
            |                         (1)        (2)        (3)         |
            |                        Base    + Controls   Full          |
            +----------------------------------------------------------+
            | Treatment Effect      X.XX***    X.XX***    X.XX***       |
            |                      (X.XX)     (X.XX)     (X.XX)         |
            +----------------------------------------------------------+
            ```

            ## Interpretation Template

            [Standard language for interpreting results]

            ## Checklist Before Reporting

            - [ ] All assumptions tested and reported
            - [ ] Robustness checks performed
            - [ ] Effect sizes are interpretable
            - [ ] Limitations acknowledged
            """),

        "common_errors": dedent(f"""\
            # Common Errors: {title}

            ## Error 1: [Category]

            **What Goes Wrong**:
            [Description of the mistake]

            **Why It's Wrong**:
            [Explanation of the problem]

            **Symptoms**:
            - [How to recognize this error]

            **How to Fix**:
            ```python
            # Correct approach
            ```

            **References**:
            - [Papers discussing this issue]

            ## Error 2: [Category]

            [Same structure...]

            ## Error Prevention Checklist

            - [ ] [Check 1]
            - [ ] [Check 2]
            - [ ] [Check 3]
            """),
    }

    return templates.get(ref_type, f"# {ref_title}: {title}\n\n[TODO: Add content]\n")


def generate_script_file(name: str, script_type: str) -> str:
    """Generate placeholder content for script files."""
    module_name = convert_to_snake_case(name)

    templates = {
        "run_analysis": dedent(f'''\
            #!/usr/bin/env python3
            """
            Run {convert_to_title_case(name)} Analysis

            This script provides a command-line interface for running
            the {name} estimator on a dataset.

            Usage:
                python run_analysis.py --data data.csv --outcome y --treatment d
            """

            import argparse
            import sys
            from pathlib import Path

            import pandas as pd

            # Add parent directory to path for imports
            sys.path.insert(0, str(Path(__file__).parent.parent))
            from {module_name}_estimator import estimate


            def main():
                parser = argparse.ArgumentParser(
                    description="Run {convert_to_title_case(name)} analysis"
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
                    print(f"Results saved to {{args.output}}")


            if __name__ == "__main__":
                main()
            '''),

        "test_assumptions": dedent(f'''\
            #!/usr/bin/env python3
            """
            Test Assumptions for {convert_to_title_case(name)}

            This script runs all identification assumption tests
            for the {name} estimator.

            Usage:
                python test_assumptions.py --data data.csv --treatment d --outcome y
            """

            import argparse
            import sys
            from pathlib import Path

            import pandas as pd

            # Add parent directory to path for imports
            sys.path.insert(0, str(Path(__file__).parent.parent))
            from {module_name}_estimator import (
                test_assumption_1,
                test_assumption_2,
                # Add more assumption tests as needed
            )


            def main():
                parser = argparse.ArgumentParser(
                    description="Test {convert_to_title_case(name)} identification assumptions"
                )
                parser.add_argument("--data", required=True, help="Path to data file (CSV)")
                parser.add_argument("--outcome", required=True, help="Outcome variable name")
                parser.add_argument("--treatment", required=True, help="Treatment variable name")
                parser.add_argument("--verbose", action="store_true", help="Verbose output")

                args = parser.parse_args()

                # Load data
                data = pd.read_csv(args.data)

                # Run assumption tests
                print("=" * 60)
                print("ASSUMPTION TESTS: {convert_to_title_case(name)}")
                print("=" * 60)

                # TODO: Implement assumption tests
                print("\\n[TODO: Implement assumption tests]")

                print("\\n" + "=" * 60)


            if __name__ == "__main__":
                main()
            '''),

        "visualize_results": dedent(f'''\
            #!/usr/bin/env python3
            """
            Visualize Results for {convert_to_title_case(name)}

            This script generates diagnostic and result visualizations
            for the {name} estimator.

            Usage:
                python visualize_results.py --data data.csv --treatment d --outcome y --output figures/
            """

            import argparse
            import sys
            from pathlib import Path

            import matplotlib.pyplot as plt
            import pandas as pd
            import seaborn as sns

            # Add parent directory to path for imports
            sys.path.insert(0, str(Path(__file__).parent.parent))
            from {module_name}_estimator import estimate


            def main():
                parser = argparse.ArgumentParser(
                    description="Visualize {convert_to_title_case(name)} results"
                )
                parser.add_argument("--data", required=True, help="Path to data file (CSV)")
                parser.add_argument("--outcome", required=True, help="Outcome variable name")
                parser.add_argument("--treatment", required=True, help="Treatment variable name")
                parser.add_argument("--output", default="figures", help="Output directory for figures")

                args = parser.parse_args()

                # Create output directory
                output_dir = Path(args.output)
                output_dir.mkdir(parents=True, exist_ok=True)

                # Load data
                data = pd.read_csv(args.data)

                # Run estimation
                result = estimate(
                    data=data,
                    outcome=args.outcome,
                    treatment=args.treatment
                )

                # Generate visualizations
                # TODO: Implement visualization functions
                print(f"[TODO: Generate visualizations in {{output_dir}}]")


            if __name__ == "__main__":
                main()
            '''),
    }

    return templates.get(script_type, f"# {script_type}.py\n\n# TODO: Implement {script_type}\n")


def generate_estimator_py(name: str) -> str:
    """Generate the main Python estimator module."""
    module_name = convert_to_snake_case(name)
    title = convert_to_title_case(name)

    return dedent(f'''\
        """
        {title} Estimator

        This module implements the {title} causal effect estimator.

        Example:
            >>> from {module_name}_estimator import estimate
            >>> result = estimate(data, outcome="y", treatment="d")
            >>> print(result.effect, result.se)

        Author: [Your Name]
        Version: 0.1.0
        """

        from dataclasses import dataclass, field
        from typing import Any, Dict, List, Optional, Union

        import numpy as np
        import pandas as pd


        @dataclass
        class CausalOutput:
            """Standard output for causal estimation."""

            effect: float
            se: float
            ci_lower: float
            ci_upper: float
            p_value: float
            diagnostics: Dict[str, Any] = field(default_factory=dict)
            summary_table: str = ""
            interpretation: str = ""

            def __repr__(self) -> str:
                return (
                    f"CausalOutput(effect={{self.effect:.4f}}, "
                    f"se={{self.se:.4f}}, "
                    f"p_value={{self.p_value:.4f}})"
                )


        def validate_data(
            data: pd.DataFrame,
            outcome: str,
            treatment: str,
            controls: Optional[List[str]] = None
        ) -> Dict[str, Any]:
            """
            Validate input data for estimation.

            Parameters
            ----------
            data : pd.DataFrame
                Input data
            outcome : str
                Name of outcome variable
            treatment : str
                Name of treatment variable
            controls : list, optional
                Names of control variables

            Returns
            -------
            dict
                Validation results with 'passed' key and any warnings
            """
            validation = {{"passed": True, "warnings": [], "errors": []}}

            # Check required columns exist
            required = [outcome, treatment]
            if controls:
                required.extend(controls)

            missing = [col for col in required if col not in data.columns]
            if missing:
                validation["passed"] = False
                validation["errors"].append(f"Missing columns: {{missing}}")

            # Check for missing values
            if validation["passed"]:
                for col in required:
                    n_missing = data[col].isna().sum()
                    if n_missing > 0:
                        validation["warnings"].append(
                            f"{{col}} has {{n_missing}} missing values"
                        )

            return validation


        def estimate(
            data: pd.DataFrame,
            outcome: str,
            treatment: str,
            controls: Optional[List[str]] = None,
            **method_params
        ) -> CausalOutput:
            """
            Estimate causal effect using {title}.

            Parameters
            ----------
            data : pd.DataFrame
                Panel or cross-sectional data
            outcome : str
                Name of outcome variable (Y)
            treatment : str
                Name of treatment variable (D)
            controls : list, optional
                Names of control variables (X)
            **method_params : dict
                Method-specific parameters

            Returns
            -------
            CausalOutput
                Estimation results including effect, standard error,
                confidence intervals, and diagnostics

            Raises
            ------
            ValueError
                If data validation fails

            Examples
            --------
            >>> import pandas as pd
            >>> data = pd.DataFrame({{"y": [1, 2, 3], "d": [0, 0, 1], "x": [1, 2, 3]}})
            >>> result = estimate(data, outcome="y", treatment="d", controls=["x"])
            >>> print(result.effect)
            """
            # Validate data
            validation = validate_data(data, outcome, treatment, controls)
            if not validation["passed"]:
                raise ValueError(f"Data validation failed: {{validation['errors']}}")

            # TODO: Implement actual estimation logic
            # This is a placeholder implementation

            effect = 0.0
            se = 0.0
            ci_lower = effect - 1.96 * se
            ci_upper = effect + 1.96 * se
            p_value = 1.0

            return CausalOutput(
                effect=effect,
                se=se,
                ci_lower=ci_lower,
                ci_upper=ci_upper,
                p_value=p_value,
                diagnostics={{"validation": validation}},
                summary_table="[TODO: Generate summary table]",
                interpretation="[TODO: Generate interpretation]"
            )


        # Assumption tests
        def test_assumption_1(data: pd.DataFrame, **kwargs) -> Dict[str, Any]:
            """
            Test first identification assumption.

            Parameters
            ----------
            data : pd.DataFrame
                Input data
            **kwargs
                Additional parameters

            Returns
            -------
            dict
                Test results with 'passed', 'statistic', 'p_value' keys
            """
            # TODO: Implement assumption test
            return {{"passed": True, "statistic": 0.0, "p_value": 1.0, "message": "Not implemented"}}


        def test_assumption_2(data: pd.DataFrame, **kwargs) -> Dict[str, Any]:
            """Test second identification assumption."""
            # TODO: Implement assumption test
            return {{"passed": True, "statistic": 0.0, "p_value": 1.0, "message": "Not implemented"}}


        # Robustness checks
        def placebo_test(
            data: pd.DataFrame,
            outcome: str,
            treatment: str,
            placebo_treatment: str,
            **kwargs
        ) -> Dict[str, Any]:
            """
            Run placebo test with fake treatment.

            The placebo effect should be statistically insignificant.
            If significant, identification assumption may be violated.

            Parameters
            ----------
            data : pd.DataFrame
                Input data
            outcome : str
                Outcome variable name
            treatment : str
                Original treatment variable name
            placebo_treatment : str
                Placebo treatment variable name
            **kwargs
                Additional parameters

            Returns
            -------
            dict
                Placebo test results
            """
            # TODO: Implement placebo test
            return {{"passed": True, "effect": 0.0, "p_value": 1.0}}


        def sensitivity_analysis(
            data: pd.DataFrame,
            outcome: str,
            treatment: str,
            **kwargs
        ) -> Dict[str, Any]:
            """
            Assess robustness to unmeasured confounding.

            Parameters
            ----------
            data : pd.DataFrame
                Input data
            outcome : str
                Outcome variable name
            treatment : str
                Treatment variable name
            **kwargs
                Additional parameters

            Returns
            -------
            dict
                Sensitivity analysis results
            """
            # TODO: Implement sensitivity analysis
            return {{"robust": True, "critical_value": None}}
        ''')


def generate_init_py(name: str, skill_type: str) -> str:
    """Generate __init__.py content."""
    module_name = convert_to_snake_case(name)
    title = convert_to_title_case(name)

    if skill_type == "estimator":
        return dedent(f'''\
            """
            {title} Estimator

            This package provides the {title} causal effect estimator.
            """

            from .{module_name}_estimator import (
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
            ''')
    else:
        return dedent(f'''\
            """
            {title}

            This package provides the {title} skill.
            """

            __version__ = "0.1.0"
            ''')


def create_skill_scaffold(
    name: str,
    skill_type: str,
    category: str,
    output_dir: Path,
    minimal: bool = False
) -> Path:
    """
    Create the complete skill scaffold.

    Parameters
    ----------
    name : str
        Skill name in kebab-case
    skill_type : str
        One of: estimator, knowledge, tool, workflow
    category : str
        One of: classic-methods, ml-foundation, causal-ml, infrastructure
    output_dir : Path
        Base output directory (usually skills/)
    minimal : bool
        If True, create minimal structure without all references

    Returns
    -------
    Path
        Path to created skill directory
    """
    # Create skill directory
    skill_dir = output_dir / category / name
    skill_dir.mkdir(parents=True, exist_ok=True)

    # Generate SKILL.md
    if skill_type == "estimator":
        skill_md_content = generate_estimator_skill_md(name)
    else:
        skill_md_content = generate_generic_skill_md(name, skill_type)

    (skill_dir / "SKILL.md").write_text(skill_md_content, encoding="utf-8")
    print(f"  Created: {skill_dir / 'SKILL.md'}")

    # Create references directory (unless minimal)
    if not minimal:
        refs_dir = skill_dir / "references"
        refs_dir.mkdir(exist_ok=True)

        reference_files = [
            "identification_assumptions",
            "diagnostic_tests",
            "estimation_methods",
            "reporting_standards",
            "common_errors",
        ]

        for ref_type in reference_files:
            ref_file = refs_dir / f"{ref_type}.md"
            ref_file.write_text(generate_reference_file(name, ref_type), encoding="utf-8")
            print(f"  Created: {ref_file}")

    # Create scripts directory
    scripts_dir = skill_dir / "scripts"
    scripts_dir.mkdir(exist_ok=True)

    script_files = ["run_analysis", "test_assumptions", "visualize_results"]
    for script_type in script_files:
        script_file = scripts_dir / f"{script_type}.py"
        script_file.write_text(generate_script_file(name, script_type), encoding="utf-8")
        print(f"  Created: {script_file}")

    # Create assets directory structure
    assets_dir = skill_dir / "assets"
    (assets_dir / "latex").mkdir(parents=True, exist_ok=True)
    (assets_dir / "markdown").mkdir(parents=True, exist_ok=True)

    # Create placeholder files in assets
    (assets_dir / "latex" / ".gitkeep").touch()
    (assets_dir / "markdown" / ".gitkeep").touch()
    print(f"  Created: {assets_dir / 'latex'}")
    print(f"  Created: {assets_dir / 'markdown'}")

    # Create Python estimator module (for estimator type)
    module_name = convert_to_snake_case(name)
    estimator_file = skill_dir / f"{module_name}_estimator.py"
    estimator_file.write_text(generate_estimator_py(name), encoding="utf-8")
    print(f"  Created: {estimator_file}")

    # Create __init__.py
    init_file = skill_dir / "__init__.py"
    init_file.write_text(generate_init_py(name, skill_type), encoding="utf-8")
    print(f"  Created: {init_file}")

    # Create scripts/__init__.py
    (scripts_dir / "__init__.py").write_text("", encoding="utf-8")

    return skill_dir


def main():
    """Main entry point for the scaffold generator."""
    parser = argparse.ArgumentParser(
        description="Generate K-Dense style skill scaffold for Causal ML Skills",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=dedent("""\
            Examples:
              python generate_skill_scaffold.py --name estimator-xyz --category classic-methods
              python generate_skill_scaffold.py --name my-tool --category ml-foundation --type tool
              python generate_skill_scaffold.py --name quick-workflow --category causal-ml --type workflow --minimal

            This generates a complete skill directory structure with:
              - SKILL.md with YAML frontmatter
              - references/ directory with placeholder docs
              - scripts/ directory with CLI tools
              - assets/ directory for templates
              - Python implementation module
            """)
    )

    parser.add_argument(
        "--name",
        required=True,
        help="Skill name in kebab-case (e.g., estimator-xyz, my-tool)"
    )
    parser.add_argument(
        "--type",
        choices=VALID_TYPES,
        default="estimator",
        help=f"Skill type: {', '.join(VALID_TYPES)} (default: estimator)"
    )
    parser.add_argument(
        "--category",
        required=True,
        choices=VALID_CATEGORIES,
        help=f"Category: {', '.join(VALID_CATEGORIES)}"
    )
    parser.add_argument(
        "--output",
        default="skills",
        help="Output directory (default: skills/)"
    )
    parser.add_argument(
        "--minimal",
        action="store_true",
        help="Create minimal structure without all reference documents"
    )

    args = parser.parse_args()

    # Validate name format
    if not args.name.replace("-", "").isalnum():
        print(f"Error: Skill name must be alphanumeric with hyphens only: {args.name}")
        sys.exit(1)

    # Determine output directory
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    output_dir = project_root / args.output

    print(f"\nGenerating skill scaffold:")
    print(f"  Name: {args.name}")
    print(f"  Type: {args.type}")
    print(f"  Category: {args.category}")
    print(f"  Output: {output_dir}")
    print(f"  Minimal: {args.minimal}")
    print()

    try:
        skill_path = create_skill_scaffold(
            name=args.name,
            skill_type=args.type,
            category=args.category,
            output_dir=output_dir,
            minimal=args.minimal
        )

        print(f"\nSkill scaffold created successfully at:")
        print(f"  {skill_path}")
        print(f"\nNext steps:")
        print(f"  1. Edit {skill_path / 'SKILL.md'} with your skill documentation")
        print(f"  2. Implement the estimator in {skill_path / (convert_to_snake_case(args.name) + '_estimator.py')}")
        print(f"  3. Add reference documentation in {skill_path / 'references'}")
        print(f"  4. Update scripts in {skill_path / 'scripts'}")

    except Exception as e:
        print(f"Error creating skill scaffold: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
