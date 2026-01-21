#!/usr/bin/env python3
"""
Writing Guide Utilities

Utility functions for formatting and checking economics papers.
Provides programmatic access to writing guidelines and formatting.

Example usage:
    from writing_guide import WritingGuide

    guide = WritingGuide()

    # Get abstract template
    template = guide.get_abstract_template()

    # Check word count
    guide.check_section_length('introduction', 1500)

    # Format citation
    citation = guide.format_citation('Angrist', 'Krueger', 1991)
"""

from dataclasses import dataclass
from typing import Optional, List, Dict, Tuple
import re


@dataclass
class SectionGuidelines:
    """Guidelines for a paper section."""
    name: str
    min_words: int
    max_words: int
    ideal_words: int
    key_elements: List[str]
    common_mistakes: List[str]


class WritingGuide:
    """Utility class for academic writing guidance."""

    SECTION_GUIDELINES = {
        'abstract': SectionGuidelines(
            name='Abstract',
            min_words=100,
            max_words=200,
            ideal_words=150,
            key_elements=[
                'Motivation/context (1-2 sentences)',
                'Research question (1 sentence)',
                'Method/approach (1-2 sentences)',
                'Main finding with numbers (2-3 sentences)',
                'Implication (1 sentence)'
            ],
            common_mistakes=[
                'No quantitative results',
                'Too vague about method',
                'Excessive hedging',
                'Starting with "This paper..."'
            ]
        ),
        'introduction': SectionGuidelines(
            name='Introduction',
            min_words=500,
            max_words=2000,
            ideal_words=1200,
            key_elements=[
                'Hook/motivation (paragraph 1)',
                'What this paper does (paragraph 2)',
                'Preview of main results (paragraph 3)',
                'Contribution statement (paragraph 4)',
                'Roadmap (final paragraph)'
            ],
            common_mistakes=[
                'Burying the main finding',
                'Literature review in intro',
                'No quantitative preview',
                'Unclear contribution'
            ]
        ),
        'literature': SectionGuidelines(
            name='Literature Review',
            min_words=300,
            max_words=1500,
            ideal_words=800,
            key_elements=[
                'Position paper in literature',
                'Identify relevant strands',
                'Explain relationship to each strand',
                'Highlight gap you fill'
            ],
            common_mistakes=[
                'List of summaries without synthesis',
                'Too many citations',
                'Not explaining relevance',
                'Missing recent papers'
            ]
        ),
        'methodology': SectionGuidelines(
            name='Methodology',
            min_words=500,
            max_words=2000,
            ideal_words=1000,
            key_elements=[
                'Identification strategy',
                'Estimating equation',
                'Variable definitions',
                'Identifying assumptions',
                'Threats and solutions'
            ],
            common_mistakes=[
                'Too much software detail',
                'Undefined notation',
                'Missing identifying assumption',
                'No threat discussion'
            ]
        ),
        'results': SectionGuidelines(
            name='Results',
            min_words=800,
            max_words=3000,
            ideal_words=1500,
            key_elements=[
                'Main results table walkthrough',
                'Magnitude interpretation',
                'Statistical significance',
                'Robustness checks',
                'Heterogeneity (if applicable)'
            ],
            common_mistakes=[
                'Numbers without interpretation',
                'Only discussing significance',
                'Too many robustness tables',
                'Missing economic magnitude'
            ]
        ),
        'conclusion': SectionGuidelines(
            name='Conclusion',
            min_words=300,
            max_words=1000,
            ideal_words=600,
            key_elements=[
                'Summary of main finding',
                'Policy implications',
                'Limitations',
                'Future research'
            ],
            common_mistakes=[
                'Just repeating abstract',
                'Introducing new results',
                'Excessive speculation',
                'Undermining own findings'
            ]
        )
    }

    def __init__(self):
        pass

    def get_section_guidelines(self, section: str) -> Optional[SectionGuidelines]:
        """Get guidelines for a specific section."""
        return self.SECTION_GUIDELINES.get(section.lower())

    def check_section_length(self, section: str, word_count: int) -> Dict:
        """Check if section length is appropriate."""
        guidelines = self.get_section_guidelines(section)
        if not guidelines:
            return {'status': 'unknown', 'message': f'Unknown section: {section}'}

        if word_count < guidelines.min_words:
            return {
                'status': 'too_short',
                'message': f'{section.title()} is too short ({word_count} words). '
                          f'Minimum: {guidelines.min_words}',
                'recommendation': guidelines.ideal_words
            }
        elif word_count > guidelines.max_words:
            return {
                'status': 'too_long',
                'message': f'{section.title()} is too long ({word_count} words). '
                          f'Maximum: {guidelines.max_words}',
                'recommendation': guidelines.ideal_words
            }
        else:
            deviation = abs(word_count - guidelines.ideal_words) / guidelines.ideal_words
            if deviation < 0.2:
                status = 'good'
            else:
                status = 'acceptable'

            return {
                'status': status,
                'message': f'{section.title()}: {word_count} words (target: {guidelines.ideal_words})',
                'recommendation': guidelines.ideal_words
            }

    def get_abstract_template(self) -> str:
        """Return abstract template."""
        return """
[CONTEXT: 1-2 sentences explaining why this question matters]
Understanding [topic] is crucial for [policy/theory/practice] because [reason].

[GAP: 1 sentence identifying what's missing]
Yet existing research has not addressed [specific gap].

[THIS PAPER: 1-2 sentences on your approach]
We exploit [source of variation] to identify the causal effect of [X] on [Y] using [method].

[RESULTS: 2-3 sentences with specific numbers]
We find that [X] increases/decreases [Y] by [Z] percent (SE = [#]).
Effects are concentrated among [subgroup].
[Additional key finding if space allows.]

[IMPLICATION: 1 sentence on why this matters]
These findings suggest that [policy implication] and have implications for [broader significance].
        """.strip()

    def get_introduction_template(self) -> str:
        """Return introduction paragraph templates."""
        return """
PARAGRAPH 1: MOTIVATION
------------------------
[Hook - open with the big question or striking fact]
Does [X] cause [Y]? / [Striking statistic about the problem].

[Why it matters]
Understanding this relationship is crucial because [policy relevance],
[theoretical importance], or [practical significance].

[Preview tension/puzzle]
Despite its importance, [challenge or puzzle that motivates your paper].


PARAGRAPH 2: THIS PAPER
------------------------
[What you do]
This paper [estimates/identifies/examines] the causal effect of [X] on [Y].

[How you do it - identification]
We exploit [source of exogenous variation] that [why it's exogenous].

[Method in one sentence]
Using [data source] and [estimation method], we compare [treatment] to [control].


PARAGRAPH 3: RESULTS PREVIEW
------------------------
[Main finding with numbers]
We find that [X] increases [Y] by [Z] percent (SE = [#]).

[Magnitude context]
This effect is [large/modest/small] - equivalent to [comparison].

[Secondary findings]
Effects are largest for [subgroup]. We also find [additional result].


PARAGRAPH 4: CONTRIBUTION
------------------------
[Position in literature]
Our contribution to the literature is [threefold/twofold].

[Contribution 1]
First, we provide the first [type] estimates of [effect] by exploiting [variation].

[Contribution 2]
Second, we address concerns about [identification threat] through [solution].

[Contribution 3]
Third, we [methodological contribution / new finding / policy insight].


ROADMAP PARAGRAPH (Optional)
------------------------
The remainder of this paper is organized as follows. Section 2 describes
the institutional background and data. Section 3 presents our empirical
strategy. Section 4 reports results. Section 5 concludes.
        """.strip()

    def format_citation(self, *authors: str, year: int,
                       style: str = 'parenthetical') -> str:
        """Format citation in author-year style."""
        if len(authors) == 1:
            author_str = authors[0]
        elif len(authors) == 2:
            author_str = f"{authors[0]} and {authors[1]}"
        else:
            author_str = f"{authors[0]} et al."

        if style == 'parenthetical':
            return f"({author_str}, {year})"
        elif style == 'narrative':
            return f"{author_str} ({year})"
        else:
            return f"{author_str} {year}"

    def format_coefficient(self, coef: float, se: float,
                          decimals: int = 3) -> str:
        """Format coefficient with standard error for text."""
        return f"{coef:.{decimals}f} (SE = {se:.{decimals}f})"

    def format_effect_size(self, effect: float, comparison: str,
                          unit: str = 'percent') -> str:
        """Format effect size with interpretation."""
        if unit == 'percent':
            return f"{effect:.1f} percent—roughly equivalent to {comparison}"
        elif unit == 'pp':
            return f"{effect:.1f} percentage points—roughly equivalent to {comparison}"
        else:
            return f"{effect:.2f} {unit}—roughly equivalent to {comparison}"

    def check_passive_voice(self, text: str) -> List[str]:
        """Find potential passive voice constructions."""
        # Simple patterns - not perfect but catches common cases
        passive_patterns = [
            r'\b(is|are|was|were|been|being)\s+\w+ed\b',
            r'\b(is|are|was|were)\s+\w+en\b',
            r'\bIt (is|was) (found|shown|demonstrated|observed)\b',
        ]

        findings = []
        for pattern in passive_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                start = max(0, match.start() - 30)
                end = min(len(text), match.end() + 30)
                context = text[start:end]
                findings.append(f"...{context}...")

        return findings

    def check_wordiness(self, text: str) -> List[Tuple[str, str]]:
        """Find wordy phrases and suggest replacements."""
        wordy_phrases = {
            r'at this point in time': 'now',
            r'due to the fact that': 'because',
            r'in order to': 'to',
            r'for the purpose of': 'for',
            r'in the event that': 'if',
            r'with regard to': 'about',
            r'in terms of': '[rewrite]',
            r'it is important to note that': '[delete]',
            r'it should be noted that': '[delete]',
            r'the fact that': 'that',
            r'a large number of': 'many',
            r'a small number of': 'few',
            r'at the present time': 'now',
            r'in the near future': 'soon',
            r'in the final analysis': 'finally',
            r'as a matter of fact': '[delete]',
        }

        findings = []
        for phrase, replacement in wordy_phrases.items():
            if re.search(phrase, text, re.IGNORECASE):
                findings.append((phrase, replacement))

        return findings

    def get_checklist(self, section: str) -> List[str]:
        """Get pre-submission checklist for section."""
        checklists = {
            'abstract': [
                'Contains motivation/context',
                'States research question clearly',
                'Mentions method/identification',
                'Includes main quantitative finding',
                'Reports statistical uncertainty',
                'States implication',
                'Within word limit',
                'No jargon without context'
            ],
            'introduction': [
                'Hook in first paragraph',
                'Main finding stated with numbers',
                'Identification strategy explained',
                'Contribution clearly stated',
                'Literature positioned (briefly)',
                'Roadmap paragraph',
                'Active voice predominates',
                'No excessive hedging'
            ],
            'tables': [
                'Three-line format (no vertical lines)',
                'Standard errors in parentheses',
                'Significance stars defined',
                'Clustering/SEs explained in notes',
                'Dependent variable stated',
                'Sample size included',
                'Variable names are labels (not codes)',
                'Consistent decimal places'
            ],
            'figures': [
                'Axis labels with units',
                'Legend (if multiple series)',
                'Confidence intervals shown',
                'Source data noted',
                'Readable at print size',
                'Works in black and white',
                'Referenced in text'
            ]
        }

        return checklists.get(section.lower(), [])

    def word_count(self, text: str) -> int:
        """Count words in text (excluding LaTeX commands)."""
        # Remove LaTeX commands
        text = re.sub(r'\\[a-zA-Z]+\{[^}]*\}', '', text)
        text = re.sub(r'\\[a-zA-Z]+', '', text)
        text = re.sub(r'\$[^$]+\$', '', text)
        text = re.sub(r'%.*$', '', text, flags=re.MULTILINE)

        # Count words
        words = text.split()
        return len(words)


# Convenience functions for direct use
def get_abstract_template() -> str:
    """Get abstract template."""
    return WritingGuide().get_abstract_template()


def get_intro_template() -> str:
    """Get introduction template."""
    return WritingGuide().get_introduction_template()


def format_citation(*authors: str, year: int) -> str:
    """Format citation."""
    return WritingGuide().format_citation(*authors, year=year)


def check_length(section: str, words: int) -> Dict:
    """Check section length."""
    return WritingGuide().check_section_length(section, words)


if __name__ == '__main__':
    # Demo usage
    guide = WritingGuide()

    print("=" * 60)
    print("ABSTRACT TEMPLATE")
    print("=" * 60)
    print(guide.get_abstract_template())

    print("\n" + "=" * 60)
    print("SECTION LENGTH CHECK")
    print("=" * 60)
    for section in ['abstract', 'introduction', 'methodology', 'results']:
        result = guide.check_section_length(section, 800)
        print(f"{section}: {result['status']} - {result['message']}")

    print("\n" + "=" * 60)
    print("CITATION FORMATTING")
    print("=" * 60)
    print(guide.format_citation('Angrist', 'Krueger', year=1991))
    print(guide.format_citation('Angrist', 'Krueger', year=1991, style='narrative'))
    print(guide.format_citation('Angrist', 'Imbens', 'Rubin', year=1996))

    print("\n" + "=" * 60)
    print("ABSTRACT CHECKLIST")
    print("=" * 60)
    for item in guide.get_checklist('abstract'):
        print(f"  [ ] {item}")
