# Paper Structure for Economics Research

## The Standard Structure

### 1. Title and Abstract

**Title Guidelines**:
- Clear, specific, informative
- Avoid clever wordplay that obscures content
- Include key method or setting if distinctive
- Good: "The Effect of Minimum Wage on Employment: Evidence from State-Level Variation"
- Avoid: "Wages of Sin: A Punny Title About Labor Markets"

**Abstract**: See `writing_abstracts.md`

---

### 2. Introduction (2-5 pages)

The introduction is the most important section. Most readers only read the abstract and introduction.

**The Four Paragraphs Model** (Expandable):

#### Paragraph 1: Motivation
- What is the big question?
- Why does it matter for policy/theory/practice?
- Hook the reader immediately

```
Example opening:
"Does education cause higher earnings, or do more able people simply
obtain more education? This question has fundamental implications for..."
```

#### Paragraph 2: This Paper
- What specifically do you do?
- What is your identification strategy?
- Preview your main finding

```
"This paper uses a regression discontinuity design exploiting the
school entry age cutoff to identify the causal effect of..."
```

#### Paragraph 3: Results Preview
- State your main quantitative findings
- Include specific numbers
- Don't save surprises for later

```
"We find that an additional year of education increases earnings by
9.2 percent (SE = 1.4), with effects concentrated among..."
```

#### Paragraph 4: Contribution
- How does this advance the literature?
- What do you do that others haven't?
- Be specific about your value-add

```
"Our contribution is threefold. First, we exploit a novel source of
variation that addresses concerns about... Second, we provide the
first estimates of... Third, we develop a method that..."
```

#### Additional Paragraphs (if needed):
- Roadmap paragraph (last): "The remainder of this paper..."
- Mechanism discussion
- Policy implications

**Introduction Checklist**:
- [ ] Reader understands the question by end of paragraph 1
- [ ] Reader knows your approach by end of paragraph 2
- [ ] Reader knows your main finding by end of paragraph 3
- [ ] Reader knows why this matters by end of paragraph 4
- [ ] No jargon without explanation
- [ ] Active voice throughout

---

### 3. Literature Review (1-3 pages)

**Purpose**: Position your paper in the scholarly conversation

**Structure Options**:

#### Option A: Thematic
Group papers by theme/approach:
- Strand 1: Papers using approach X
- Strand 2: Papers studying context Y
- Strand 3: Papers with related findings Z

#### Option B: Chronological
For established literatures with clear evolution

#### Option C: Integrated with Introduction
Common in top journals - weave literature into contribution discussion

**Best Practices**:
- Cite generously but not exhaustively
- Focus on directly relevant papers
- Explain how each cited paper relates to yours
- Be fair to prior work while highlighting your advance
- Avoid: "Smith (2020) studied X. Jones (2021) studied Y. We study Z."
- Better: "Our identification strategy builds on Smith (2020), who pioneered..."

**What to Cover**:
1. Foundational papers (brief acknowledgment)
2. Closely related empirical papers (detailed comparison)
3. Methodological papers (if using novel methods)
4. Policy-relevant papers (if policy motivated)

---

### 4. Institutional Background / Data (Variable)

**Institutional Background** (if applicable):
- Explain the setting/policy/institution
- Why is this setting useful for identification?
- What do readers need to know?

**Data Section**:
- Source and coverage
- Sample construction (with flowchart if complex)
- Key variable definitions
- Summary statistics (Table 1 usually)
- Data limitations and how you address them

**Summary Statistics Table**:
```
Table 1: Summary Statistics
─────────────────────────────────────────────────
                        Mean    SD      N
─────────────────────────────────────────────────
Panel A: Full Sample
  Outcome variable     12.34   (5.67)  10,000
  Treatment indicator   0.45   (0.50)  10,000
  Covariate 1          34.5    (12.3)  10,000

Panel B: By Treatment Status
  Treated group mean   14.56   (5.43)   4,500
  Control group mean   10.52   (5.12)   5,500
─────────────────────────────────────────────────
Notes: Standard deviations in parentheses. Sample
restricted to... Data source: ...
```

---

### 5. Empirical Strategy / Methodology (2-4 pages)

**Purpose**: Convince readers your estimates are causal

**Structure**:

#### 5.1 Identification Strategy
- What variation do you exploit?
- Why is this variation plausibly exogenous?
- State your identifying assumption clearly

#### 5.2 Estimating Equation

```latex
Y_{it} = \alpha + \beta D_{it} + X'_{it}\gamma + \mu_i + \lambda_t + \varepsilon_{it}
```

- Define every term
- Explain what β captures
- Discuss fixed effects choices

#### 5.3 Threats to Identification
- What could bias your estimates?
- How do you address each concern?
- Be proactive - don't wait for referees

#### 5.4 Specification Choices
- Why these controls?
- Why this sample?
- Robustness preview (details in appendix)

**Method-Specific Structures**:

For **RDD**:
1. Running variable and cutoff
2. Bandwidth selection
3. Functional form
4. Manipulation tests

For **DID**:
1. Treatment and control groups
2. Pre-trends
3. Parallel trends assumption
4. Staggered timing issues (if applicable)

For **IV**:
1. Instrument description
2. First stage
3. Exclusion restriction
4. Relevance (F-statistic)

---

### 6. Results (3-6 pages)

**Structure**:

#### 6.1 Main Results
- Present core findings
- Walk through main table
- Interpret magnitudes (not just significance)

#### 6.2 Robustness
- Alternative specifications
- Placebo tests
- Sensitivity to choices

#### 6.3 Heterogeneity (if applicable)
- Subgroup analysis
- Mechanism exploration

#### 6.4 Additional Results
- Extensions
- Alternative outcomes

**Table Presentation**:
- One main results table with 4-6 columns
- Build complexity across columns
- Column 1: Simplest specification
- Final column: Preferred specification
- See `tables_figures.md` for formatting

**Magnitude Interpretation**:
```
Bad:  "The coefficient is 0.092 and significant at the 1% level."

Good: "An additional year of education increases earnings by 9.2 percent,
      roughly equivalent to the black-white earnings gap (8.7%) or the
      return to an additional year of experience at age 30 (7.1%)."
```

---

### 7. Discussion / Mechanisms (Optional, 1-2 pages)

**When to Include**:
- Reduced-form paper with multiple possible mechanisms
- Structural interpretation of estimates
- Policy counterfactuals

**Structure**:
- What drives the effect?
- Which mechanisms can you rule out?
- What additional evidence supports your interpretation?

---

### 8. Conclusion (1-2 pages)

**Purpose**: Leave readers with clear takeaways

**Structure**:

#### Paragraph 1: Summary
- Restate the question and main finding
- No new information

#### Paragraph 2: Implications
- What does this mean for policy?
- What does this mean for theory?

#### Paragraph 3: Limitations and Future Work
- Be honest about limitations
- Suggest natural extensions
- Don't undermine your own paper

**Avoid**:
- Repeating the abstract verbatim
- Introducing new results
- Being overly speculative
- False modesty ("this paper has many limitations...")

---

## Design-Specific Structures

### RCT/Experimental Papers

1. Introduction
2. Experimental Design
   - Setting
   - Intervention
   - Randomization
   - Timeline
3. Data and Balance
4. Results
5. Mechanisms
6. External Validity
7. Conclusion

### Structural Papers

1. Introduction
2. Model
   - Environment
   - Agents
   - Equilibrium
3. Identification
4. Data
5. Estimation
6. Results
7. Counterfactuals
8. Conclusion

### RDD Papers

1. Introduction
2. Institutional Background
3. Data
4. Empirical Strategy
   - RD Design
   - Bandwidth Selection
   - Specification
5. Validity Tests
   - Manipulation
   - Covariate Balance
6. Results
7. Robustness
8. Conclusion

---

## Length Guidelines by Section

| Section | Pages | % of Paper |
|---------|-------|------------|
| Abstract | 0.5 | 2% |
| Introduction | 3-5 | 15% |
| Literature | 1-3 | 8% |
| Background/Data | 2-4 | 12% |
| Methods | 2-4 | 12% |
| Results | 4-8 | 25% |
| Discussion | 1-2 | 6% |
| Conclusion | 1-2 | 5% |
| References | 2-4 | - |
| Appendix | Variable | - |

**Total**: 20-35 pages (excluding appendix)

---

## Common Structural Mistakes

1. **Burying the lead**: Main finding on page 15
2. **Literature review syndrome**: 10 pages of "X studied Y"
3. **Methods dump**: All possible specifications, no guidance
4. **Results without interpretation**: Numbers without meaning
5. **Wandering conclusion**: Speculative rambling
6. **Missing mechanism**: "It works" without "why"
7. **Defensive writing**: Too many caveats, undermines confidence
