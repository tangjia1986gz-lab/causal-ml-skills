---
name: scientific-writing-econ
description: Use for guidance on writing economics research papers. Triggers on paper writing, abstract, introduction, results section, tables, figures, AER style, academic writing, journal submission, referee response, literature review, methodology section.
version: 1.0.0
author: PPcourse
tags:
  - academic-writing
  - economics
  - social-science
  - publication
  - latex
triggers:
  - paper writing
  - abstract
  - introduction section
  - results section
  - tables
  - figures
  - AER style
  - journal submission
  - referee response
  - R&R
  - literature review
  - methodology
  - regression tables
---

# Scientific Writing for Economics

This skill provides comprehensive guidance for writing economics and social science research papers, following best practices from top journals (AER, QJE, Econometrica, NBER).

## When to Use This Skill

Use this skill when:
- Writing or structuring an economics research paper
- Crafting abstracts, introductions, or conclusions
- Creating publication-quality tables and figures
- Formatting regression output for journals
- Responding to referee reports (R&R)
- Checking journal-specific guidelines
- Improving clarity and conciseness in academic writing

## Skill Structure

### References (Knowledge Base)

| File | Purpose |
|------|---------|
| `references/paper_structure.md` | Full paper organization guide |
| `references/writing_abstracts.md` | Abstract writing best practices |
| `references/tables_figures.md` | AER-style tables and figures |
| `references/citations_references.md` | Citation practices |
| `references/common_mistakes.md` | Writing pitfalls to avoid |
| `references/journal_guidelines.md` | Journal-specific requirements |

### Scripts (Utilities)

| Script | Purpose |
|--------|---------|
| `scripts/check_paper_structure.py` | Validate paper structure |
| `scripts/format_tables.py` | Convert regression output to publication format |
| `writing_guide.py` | Utility functions for formatting |

### Assets (Templates)

| Template | Purpose |
|----------|---------|
| `assets/latex/paper_template.tex` | Full paper LaTeX template |
| `assets/latex/abstract_template.tex` | Abstract template |
| `assets/markdown/paper_checklist.md` | Pre-submission checklist |
| `assets/markdown/reviewer_response.md` | R&R response template |

## Quick Reference

### Paper Structure (The Hourglass Model)

```
    ╭─────────────────────────╮
    │   INTRODUCTION (Broad)  │  ← Motivation, contribution
    ╰───────────┬─────────────╯
                │
        ╭───────┴───────╮
        │  LITERATURE   │  ← Position in field
        ╰───────┬───────╯
                │
            ╭───┴───╮
            │METHOD │  ← Specific approach
            ╰───┬───╯
                │
            ╭───┴───╮
            │RESULTS│  ← Core findings
            ╰───┬───╯
                │
        ╭───────┴───────╮
        │  DISCUSSION   │  ← Implications
        ╰───────┬───────╯
                │
    ╭───────────┴─────────────╮
    │   CONCLUSION (Broad)    │  ← Broader significance
    ╰─────────────────────────╯
```

### Abstract Formula (150-200 words)

1. **Context** (1-2 sentences): Why does this matter?
2. **Gap/Question** (1 sentence): What's missing?
3. **Method** (1-2 sentences): How do you address it?
4. **Results** (2-3 sentences): What did you find?
5. **Implications** (1 sentence): Why should readers care?

### Table Formatting (AER Style)

- **Three-line format**: Top rule, header rule, bottom rule only
- **No vertical lines**
- **Standard errors in parentheses**
- **Stars for significance**: *** p<0.01, ** p<0.05, * p<0.1
- **Clear variable labels** (not Stata variable names)
- **Panel labels** if multiple specifications

## Usage Examples

### Get Paper Structure Guidance

```
User: How should I structure my RDD paper?
→ Skill provides RDD-specific structure with identification section
```

### Format Regression Tables

```python
# Use the format_tables.py script
python scripts/format_tables.py --input stata_output.txt --style aer
```

### Pre-Submission Check

```
User: Check my paper before submission
→ Skill runs through paper_checklist.md systematically
```

## Key Principles

1. **Clarity over complexity**: Simple sentences, active voice
2. **Front-load contributions**: Reader knows value by page 2
3. **Tables tell stories**: Each table has a clear message
4. **Figures for intuition**: Visualize key mechanisms
5. **Anticipate reviewers**: Address obvious concerns proactively

## Related Skills

- `estimator-*` - For methodology guidance
- `ml-model-tree` - For ML method explanations
- `causal-concept-guide` - For causal inference concepts
