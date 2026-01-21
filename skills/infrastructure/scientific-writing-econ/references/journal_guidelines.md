# Journal Guidelines for Economics Papers

## Top 5 General Interest Journals

### American Economic Review (AER)

**Publisher**: American Economic Association
**Impact Factor**: ~12
**Acceptance Rate**: ~6%

**Manuscript Requirements**:
- **Length**: No strict limit, but 15,000 words typical
- **Abstract**: 150 words maximum
- **JEL codes**: Required (2-4 codes)
- **Keywords**: 3-5 keywords
- **Data availability**: Data and code must be deposited with AEA

**Formatting**:
- Double-spaced
- 12pt font
- 1-inch margins
- Figures/tables at end
- Line numbers required for submission

**Style**:
- Author-date citations
- Three-line tables (no vertical rules)
- Standard errors in parentheses
- Stars: *** p<0.01, ** p<0.05, * p<0.1

**Special Features**:
- AER: Insights (shorter papers, 6,000 words)
- Registered Reports track

**Submission**: https://www.aeaweb.org/journals/aer/submissions

---

### Quarterly Journal of Economics (QJE)

**Publisher**: Oxford University Press / Harvard
**Impact Factor**: ~15
**Acceptance Rate**: ~3%

**Manuscript Requirements**:
- **Length**: 12,000 words preferred
- **Abstract**: 150 words maximum
- **JEL codes**: Not required in submission
- **Data**: Replication materials required upon acceptance

**Formatting**:
- Double-spaced
- Standard academic format
- Online appendix for supplementary material

**Style**:
- Discourages significance stars
- Prefers confidence intervals
- Emphasizes economic significance over statistical significance

**Known for**:
- Clean, clear exposition
- Strong identification
- Big questions

**Submission**: https://academic.oup.com/qje/pages/general_instructions

---

### Journal of Political Economy (JPE)

**Publisher**: University of Chicago Press
**Impact Factor**: ~10
**Acceptance Rate**: ~5%

**Manuscript Requirements**:
- **Length**: No strict limit
- **Abstract**: 100 words maximum (very short!)
- **JEL codes**: Required

**Formatting**:
- Chicago style citations
- Tables and figures integrated in text
- Professional typesetting upon acceptance

**Known for**:
- Theoretical rigor
- Structural models
- Long-run impact

**Submission**: https://www.journals.uchicago.edu/journals/jpe/submit

---

### Econometrica

**Publisher**: Econometric Society / Wiley
**Impact Factor**: ~8
**Acceptance Rate**: ~8%

**Manuscript Requirements**:
- **Length**: No limit, but concise preferred
- **Abstract**: 100 words maximum
- **JEL codes**: Required

**Formatting**:
- LaTeX required
- Specific Econometrica style file
- Proofs in appendix

**Style**:
- Formal theorem/proof structure for theory
- Detailed technical appendices
- Mathematical rigor essential

**Known for**:
- Econometric theory
- Structural estimation
- Mechanism design

**Submission**: https://www.econometricsociety.org/publications/econometrica/submissions

---

### Review of Economic Studies (REStud)

**Publisher**: Oxford University Press
**Impact Factor**: ~9
**Acceptance Rate**: ~6%

**Manuscript Requirements**:
- **Length**: 12,000 words typical
- **Abstract**: 150 words
- **Online appendix**: Common and expected

**Known for**:
- European perspective
- Mix of theory and empirics
- Strong technical papers

**Submission**: https://academic.oup.com/restud/pages/general_instructions

---

## Top Field Journals

### Labor Economics

**Journal of Labor Economics (JLE)**
- Focus: Labor markets, human capital
- Length: ~10,000 words
- Strong identification expected

**Labour Economics**
- European labor economics
- More empirical focus
- Faster turnaround

### Public Economics

**Journal of Public Economics (JPubE)**
- Focus: Taxation, public finance, policy
- Length: No strict limit
- Policy relevance valued

**American Economic Journal: Economic Policy**
- AEA journal
- Policy-focused applied micro
- Shorter format

### Development Economics

**Journal of Development Economics (JDE)**
- Focus: Developing countries
- RCTs common
- Field experiments valued

**American Economic Journal: Applied Economics**
- Applied micro broadly
- Development papers welcome
- Shorter format (~8,000 words)

### Health Economics

**Journal of Health Economics**
- Focus: Health markets, insurance, demand
- Strong identification required
- Policy relevance important

**American Journal of Health Economics**
- ASHE journal
- Mix of methods

---

## NBER Working Papers

**What it is**: Pre-publication circulation, not peer-reviewed
**Who can submit**: NBER affiliates only
**Why it matters**: Wide circulation, establishes priority

**Best Practices**:
- Update NBER version when paper changes substantially
- Note "forthcoming" when accepted elsewhere
- Cite published version once available

---

## Submission Strategies

### Choosing a Journal

**Consider**:
1. **Fit**: Does this journal publish papers like yours?
2. **Audience**: Who needs to read this?
3. **Quality**: Is your paper at this level?
4. **Speed**: How fast do you need publication?
5. **Career stage**: Assistant professors need top 5

**Decision Tree**:
```
Is this a top-5 paper?
├── Yes → Submit to QJE/AER/JPE first
│         (highest prestige, longest wait)
└── No →
    ├── Strong field paper → Top field journal
    └── Good but not exceptional → Second-tier general or field
```

### Timeline Expectations

| Journal | Desk Decision | First Round | Total Time |
|---------|--------------|-------------|------------|
| Top 5 | 1-3 months | 4-6 months | 1-3 years |
| Top field | 2-4 weeks | 3-4 months | 1-2 years |
| Good field | 1-2 weeks | 2-3 months | 6-18 months |

### Rejection Strategy

1. **Read carefully**: Sometimes useful feedback
2. **Revise if needed**: Address valid concerns
3. **Resubmit quickly**: Don't let paper sit
4. **Move down strategically**: Match paper quality to journal tier

---

## Referee Response Guidelines

### Structure of R&R Response

```
Dear Editor and Referees,

Thank you for your constructive feedback. We have revised the
paper substantially to address your concerns. Below we respond
point-by-point.

[Response to Editor]
[Response to Referee 1]
[Response to Referee 2]

RESPONSE TO REFEREE 1

1. [Quote referee comment in italics]

[Your response - what you did and why]

Changes in manuscript: [Specific location]

2. [Next comment]
...
```

### Response Principles

1. **Be respectful**: Even if comment seems wrong
2. **Be thorough**: Address every point
3. **Be clear**: Quote the comment, explain your response
4. **Point to changes**: "See page X, paragraph Y"
5. **Explain disagreements**: If you don't change, say why

### Common Situations

**Unreasonable Request**:
```
"We appreciate this suggestion. However, implementing this
analysis is not feasible because [specific reason]. Instead,
we have [alternative approach] which addresses the underlying
concern by [explanation]."
```

**Conflicting Referee Comments**:
```
"Referee 1 suggests X while Referee 2 suggests not-X. We have
followed [choice] because [reasoning]. We are happy to modify
this if the editor prefers otherwise."
```

**Request for Major New Analysis**:
```
"We have conducted the requested analysis. Results, reported
in new Table A5, confirm our main findings. [Brief summary of
what it shows]."
```

---

## Data and Code Policies

### AEA Data Policy (Standard for Top Journals)

Required materials:
1. **README file**: How to replicate
2. **Data files**: Or instructions if restricted
3. **Code files**: All estimation code
4. **Output**: Logs showing results match paper

### Best Practices

```
replication/
├── README.md           # Master instructions
├── data/
│   ├── raw/           # Original data (or download scripts)
│   └── processed/     # Cleaned data
├── code/
│   ├── 01_clean.do    # Data cleaning
│   ├── 02_analysis.do # Main analysis
│   └── 03_figures.do  # Figures
├── output/
│   ├── tables/        # Table output
│   └── figures/       # Figure output
└── paper/
    └── manuscript.tex  # Paper source
```

---

## Pre-Registration

**What**: Specifying analysis plan before seeing data
**Why**: Reduces p-hacking, increases credibility
**Where**: AEA Registry, OSF, EGAP

**When Required**:
- AER Registered Reports track
- Some RCT journals
- Increasingly expected for experiments

**What to Include**:
- Research questions
- Primary outcomes
- Sample size and power
- Main specifications
- Subgroup analyses planned
