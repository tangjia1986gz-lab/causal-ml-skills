#!/usr/bin/env python3
"""
Check Paper Structure

Validates the structure of an economics research paper.
Checks for required sections, word counts, and common issues.

Usage:
    python check_paper_structure.py paper.tex
    python check_paper_structure.py paper.md --format markdown
"""

import argparse
import re
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional
import sys


@dataclass
class Section:
    """Represents a paper section."""
    name: str
    start_line: int
    end_line: int
    word_count: int
    content: str


@dataclass
class PaperAnalysis:
    """Analysis results for a paper."""
    title: Optional[str] = None
    abstract_words: int = 0
    total_words: int = 0
    sections: list = field(default_factory=list)
    tables: list = field(default_factory=list)
    figures: list = field(default_factory=list)
    citations: list = field(default_factory=list)
    equations: int = 0
    issues: list = field(default_factory=list)
    suggestions: list = field(default_factory=list)


class PaperChecker:
    """Check paper structure and content."""

    # Expected sections for an economics paper
    EXPECTED_SECTIONS = [
        'introduction',
        'literature',
        'data',
        'methodology',
        'results',
        'conclusion'
    ]

    # Alternative names for sections
    SECTION_ALIASES = {
        'introduction': ['intro', 'motivation'],
        'literature': ['literature review', 'related work', 'background', 'prior work'],
        'data': ['data and sample', 'sample', 'data description'],
        'methodology': ['method', 'methods', 'empirical strategy', 'identification',
                       'estimation', 'model', 'theoretical framework'],
        'results': ['findings', 'main results', 'empirical results', 'analysis'],
        'conclusion': ['conclusions', 'discussion', 'discussion and conclusion',
                      'concluding remarks', 'summary']
    }

    # Word count guidelines
    SECTION_GUIDELINES = {
        'abstract': {'min': 100, 'max': 200, 'ideal': 150},
        'introduction': {'min': 500, 'max': 2000, 'ideal': 1200},
        'literature': {'min': 300, 'max': 1500, 'ideal': 800},
        'data': {'min': 300, 'max': 1500, 'ideal': 700},
        'methodology': {'min': 500, 'max': 2000, 'ideal': 1000},
        'results': {'min': 800, 'max': 3000, 'ideal': 1500},
        'conclusion': {'min': 300, 'max': 1000, 'ideal': 600}
    }

    def __init__(self, format_type: str = 'latex'):
        self.format_type = format_type
        self.analysis = PaperAnalysis()

    def check_paper(self, filepath: Path) -> PaperAnalysis:
        """Run all checks on a paper."""
        content = filepath.read_text(encoding='utf-8')

        # Parse based on format
        if self.format_type == 'latex':
            self._parse_latex(content)
        else:
            self._parse_markdown(content)

        # Run checks
        self._check_sections()
        self._check_word_counts()
        self._check_abstract()
        self._check_introduction()
        self._check_tables_figures()
        self._check_citations()

        return self.analysis

    def _parse_latex(self, content: str):
        """Parse LaTeX document structure."""
        # Extract title
        title_match = re.search(r'\\title\{([^}]+)\}', content)
        if title_match:
            self.analysis.title = title_match.group(1)

        # Extract abstract
        abstract_match = re.search(
            r'\\begin\{abstract\}(.*?)\\end\{abstract\}',
            content, re.DOTALL
        )
        if abstract_match:
            abstract_text = self._clean_latex(abstract_match.group(1))
            self.analysis.abstract_words = len(abstract_text.split())

        # Find sections
        section_pattern = r'\\section\{([^}]+)\}'
        sections = list(re.finditer(section_pattern, content))

        for i, match in enumerate(sections):
            section_name = match.group(1).lower()
            start = match.end()
            end = sections[i + 1].start() if i + 1 < len(sections) else len(content)
            section_content = content[start:end]

            self.analysis.sections.append(Section(
                name=section_name,
                start_line=content[:start].count('\n'),
                end_line=content[:end].count('\n'),
                word_count=len(self._clean_latex(section_content).split()),
                content=section_content
            ))

        # Count tables
        self.analysis.tables = re.findall(r'\\begin\{table\}', content)

        # Count figures
        self.analysis.figures = re.findall(r'\\begin\{figure\}', content)

        # Count equations
        self.analysis.equations = len(re.findall(r'\\begin\{equation\}', content))
        self.analysis.equations += len(re.findall(r'\$\$[^$]+\$\$', content))

        # Find citations
        self.analysis.citations = re.findall(r'\\cite[pt]?\{([^}]+)\}', content)

        # Total word count
        self.analysis.total_words = len(self._clean_latex(content).split())

    def _parse_markdown(self, content: str):
        """Parse Markdown document structure."""
        # Extract title (first H1)
        title_match = re.search(r'^#\s+(.+)$', content, re.MULTILINE)
        if title_match:
            self.analysis.title = title_match.group(1)

        # Find sections (H2 headers)
        section_pattern = r'^##\s+(.+)$'
        sections = list(re.finditer(section_pattern, content, re.MULTILINE))

        for i, match in enumerate(sections):
            section_name = match.group(1).lower()
            start = match.end()
            end = sections[i + 1].start() if i + 1 < len(sections) else len(content)
            section_content = content[start:end]

            self.analysis.sections.append(Section(
                name=section_name,
                start_line=content[:start].count('\n'),
                end_line=content[:end].count('\n'),
                word_count=len(self._clean_markdown(section_content).split()),
                content=section_content
            ))

        # Total word count
        self.analysis.total_words = len(self._clean_markdown(content).split())

    def _clean_latex(self, text: str) -> str:
        """Remove LaTeX commands for word counting."""
        # Remove comments
        text = re.sub(r'%.*$', '', text, flags=re.MULTILINE)
        # Remove commands
        text = re.sub(r'\\[a-zA-Z]+\{[^}]*\}', '', text)
        text = re.sub(r'\\[a-zA-Z]+', '', text)
        # Remove math
        text = re.sub(r'\$[^$]+\$', '', text)
        # Clean whitespace
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    def _clean_markdown(self, text: str) -> str:
        """Remove Markdown formatting for word counting."""
        # Remove code blocks
        text = re.sub(r'```[^`]+```', '', text)
        # Remove inline code
        text = re.sub(r'`[^`]+`', '', text)
        # Remove headers
        text = re.sub(r'^#+\s+', '', text, flags=re.MULTILINE)
        # Remove links
        text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)
        # Clean whitespace
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    def _normalize_section_name(self, name: str) -> Optional[str]:
        """Map section name to standard name."""
        name = name.lower().strip()

        for standard, aliases in self.SECTION_ALIASES.items():
            if name == standard or name in aliases:
                return standard

        # Check partial matches
        for standard, aliases in self.SECTION_ALIASES.items():
            if standard in name or any(alias in name for alias in aliases):
                return standard

        return None

    def _check_sections(self):
        """Check for required sections."""
        found_sections = set()

        for section in self.analysis.sections:
            normalized = self._normalize_section_name(section.name)
            if normalized:
                found_sections.add(normalized)

        missing = set(self.EXPECTED_SECTIONS) - found_sections

        if missing:
            self.analysis.issues.append(
                f"Missing or unrecognized sections: {', '.join(missing)}"
            )
            self.analysis.suggestions.append(
                "Standard economics paper structure: Introduction, Literature Review, "
                "Data, Methodology/Empirical Strategy, Results, Conclusion"
            )

    def _check_word_counts(self):
        """Check section word counts against guidelines."""
        for section in self.analysis.sections:
            normalized = self._normalize_section_name(section.name)
            if normalized and normalized in self.SECTION_GUIDELINES:
                guidelines = self.SECTION_GUIDELINES[normalized]

                if section.word_count < guidelines['min']:
                    self.analysis.issues.append(
                        f"{section.name.title()}: {section.word_count} words "
                        f"(below minimum of {guidelines['min']})"
                    )
                elif section.word_count > guidelines['max']:
                    self.analysis.issues.append(
                        f"{section.name.title()}: {section.word_count} words "
                        f"(above maximum of {guidelines['max']})"
                    )

    def _check_abstract(self):
        """Check abstract quality."""
        if self.analysis.abstract_words == 0:
            self.analysis.issues.append("No abstract found")
            return

        guidelines = self.SECTION_GUIDELINES['abstract']

        if self.analysis.abstract_words < guidelines['min']:
            self.analysis.issues.append(
                f"Abstract too short: {self.analysis.abstract_words} words "
                f"(minimum {guidelines['min']})"
            )
        elif self.analysis.abstract_words > guidelines['max']:
            self.analysis.issues.append(
                f"Abstract too long: {self.analysis.abstract_words} words "
                f"(maximum {guidelines['max']} for most journals)"
            )

        self.analysis.suggestions.append(
            "Abstract should include: (1) motivation, (2) research question, "
            "(3) method, (4) main finding with numbers, (5) implication"
        )

    def _check_introduction(self):
        """Check introduction quality indicators."""
        intro = None
        for section in self.analysis.sections:
            if self._normalize_section_name(section.name) == 'introduction':
                intro = section
                break

        if not intro:
            return

        content = intro.content.lower()

        # Check for quantitative preview
        has_numbers = bool(re.search(r'\d+\.?\d*\s*percent|\d+\.?\d*%|\d+\.?\d*\s*pp', content))
        if not has_numbers:
            self.analysis.suggestions.append(
                "Consider including quantitative results in introduction "
                "(e.g., 'We find X increases Y by Z percent')"
            )

        # Check for contribution statement
        contribution_words = ['contribution', 'contribute', 'advance', 'extend', 'novel']
        has_contribution = any(word in content for word in contribution_words)
        if not has_contribution:
            self.analysis.suggestions.append(
                "Consider explicitly stating your contribution to the literature"
            )

    def _check_tables_figures(self):
        """Check tables and figures."""
        n_tables = len(self.analysis.tables)
        n_figures = len(self.analysis.figures)

        if n_tables == 0:
            self.analysis.issues.append("No tables found")
        elif n_tables > 10:
            self.analysis.suggestions.append(
                f"Many tables ({n_tables}). Consider moving some to appendix"
            )

        if n_figures == 0:
            self.analysis.suggestions.append(
                "Consider adding figures for visual intuition"
            )

    def _check_citations(self):
        """Check citation patterns."""
        n_citations = len(self.analysis.citations)

        if n_citations < 10:
            self.analysis.issues.append(
                f"Very few citations ({n_citations}). Typical range: 30-60"
            )

        # Flatten multi-citations
        all_refs = []
        for cite in self.analysis.citations:
            all_refs.extend(cite.split(','))

        unique_refs = len(set(ref.strip() for ref in all_refs))

        if unique_refs > 80:
            self.analysis.suggestions.append(
                f"Many unique references ({unique_refs}). Consider trimming"
            )


def print_report(analysis: PaperAnalysis):
    """Print analysis report."""
    print("\n" + "=" * 60)
    print("PAPER STRUCTURE ANALYSIS")
    print("=" * 60)

    if analysis.title:
        print(f"\nTitle: {analysis.title}")

    print(f"\nTotal words: {analysis.total_words:,}")
    print(f"Abstract words: {analysis.abstract_words}")
    print(f"Tables: {len(analysis.tables)}")
    print(f"Figures: {len(analysis.figures)}")
    print(f"Equations: {analysis.equations}")
    print(f"Citation instances: {len(analysis.citations)}")

    print("\n" + "-" * 60)
    print("SECTIONS")
    print("-" * 60)

    for section in analysis.sections:
        print(f"  {section.name}: {section.word_count:,} words")

    if analysis.issues:
        print("\n" + "-" * 60)
        print("ISSUES")
        print("-" * 60)
        for issue in analysis.issues:
            print(f"  [!] {issue}")

    if analysis.suggestions:
        print("\n" + "-" * 60)
        print("SUGGESTIONS")
        print("-" * 60)
        for suggestion in analysis.suggestions:
            print(f"  [*] {suggestion}")

    print("\n" + "=" * 60)

    # Summary score
    issue_count = len(analysis.issues)
    if issue_count == 0:
        print("Status: GOOD - No structural issues found")
    elif issue_count <= 3:
        print(f"Status: MINOR ISSUES - {issue_count} issues to address")
    else:
        print(f"Status: NEEDS WORK - {issue_count} issues to address")

    print("=" * 60 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Check economics paper structure"
    )
    parser.add_argument(
        'paper',
        type=Path,
        help="Path to paper file (.tex or .md)"
    )
    parser.add_argument(
        '--format', '-f',
        choices=['latex', 'markdown'],
        default='latex',
        help="Document format (default: latex)"
    )
    parser.add_argument(
        '--json', '-j',
        action='store_true',
        help="Output as JSON"
    )

    args = parser.parse_args()

    if not args.paper.exists():
        print(f"Error: File not found: {args.paper}")
        sys.exit(1)

    checker = PaperChecker(format_type=args.format)
    analysis = checker.check_paper(args.paper)

    if args.json:
        import json
        print(json.dumps({
            'title': analysis.title,
            'total_words': analysis.total_words,
            'abstract_words': analysis.abstract_words,
            'sections': [
                {'name': s.name, 'words': s.word_count}
                for s in analysis.sections
            ],
            'tables': len(analysis.tables),
            'figures': len(analysis.figures),
            'equations': analysis.equations,
            'issues': analysis.issues,
            'suggestions': analysis.suggestions
        }, indent=2))
    else:
        print_report(analysis)


if __name__ == '__main__':
    main()
