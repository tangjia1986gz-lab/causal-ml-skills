# Common Writing Mistakes in Economics Papers

## Clarity Mistakes

### 1. Burying the Lead

**Problem**: Main finding hidden deep in the paper

**Bad**:
```
Section 1: Introduction (general motivation)
Section 2: Literature review (5 pages)
Section 3: Institutional background (3 pages)
Section 4: Data (2 pages)
Section 5: Methods (3 pages)
Section 6: Results
  6.1: Preliminary results
  6.2: Robustness checks
  6.3: Main finding (page 18!)
```

**Better**: State your main finding in the abstract AND introduction (with numbers).

---

### 2. Passive Voice Overuse

**Bad**:
```
"It was found that education is positively associated with earnings."
"The results are shown in Table 3."
"This relationship has been studied by many researchers."
```

**Better**:
```
"We find that education increases earnings by 9 percent."
"Table 3 shows the results."
"Many researchers study this relationship."
```

---

### 3. Nominalizations

**Bad** (turning verbs into nouns):
```
"We provide an examination of..."
"The investigation of this question..."
"The determination of causality..."
"The estimation of parameters..."
```

**Better**:
```
"We examine..."
"We investigate this question..."
"To determine causality..."
"We estimate parameters..."
```

---

### 4. Unnecessary Hedging

**Bad**:
```
"The results seem to suggest that there may possibly be
some evidence consistent with the hypothesis that education
might have an effect on earnings."
```

**Better**:
```
"We find that education increases earnings. The effect is
statistically significant and economically meaningful."
```

Note: Appropriate hedging IS needed when uncertain:
```
"Point estimates suggest moderate effects, though confidence
intervals are wide and include zero."
```

---

### 5. Jargon Without Explanation

**Bad** (for general audience):
```
"We use 2SLS with LATE identification to estimate ITT and TOT effects."
```

**Better**:
```
"We use instrumental variables (two-stage least squares) to identify
the causal effect of education. This approach identifies the effect
for individuals whose education was changed by our instrument (the
local average treatment effect, or LATE)."
```

---

## Structural Mistakes

### 6. Literature Review as List

**Bad**:
```
"Smith (2020) studied the effect of X on Y. Jones (2019) also studied
this relationship. Brown (2018) found different results. Green (2017)
used a different method. White (2016) studied a related question."
```

**Better**:
```
"Our paper contributes to a literature examining X's effect on Y.
Early work by White (2016) established the basic relationship using
cross-sectional data, but concerns about selection bias limited causal
interpretation. Smith (2020) and Jones (2019) addressed this using
instrumental variables, finding effects of 5-10 percent. We build on
this work by exploiting a novel source of variation that addresses
remaining concerns about [specific issue]."
```

---

### 7. Methods Section as Manual

**Bad**:
```
"First, we load the data. Then, we merge datasets using unique
identifiers. We winsorize outliers at the 1st and 99th percentiles.
We estimate equation (1) using Stata's reghdfe command with the
'absorb' option for fixed effects..."
```

**Better**: Focus on identification strategy, not software commands.

```
"Our identification strategy exploits within-firm variation in
treatment timing. We include firm fixed effects to control for
time-invariant firm characteristics and year fixed effects to
control for common shocks. Standard errors are clustered at the
firm level to account for serial correlation."
```

---

### 8. Results Without Interpretation

**Bad**:
```
"Column 1 shows the coefficient is 0.092 and statistically
significant. Column 2 adds controls and the coefficient becomes
0.087. Column 3 adds fixed effects and the coefficient is 0.091."
```

**Better**:
```
"Our preferred specification (Column 3) indicates that an
additional year of education increases earnings by 9.1 percent—
roughly equivalent to the black-white earnings gap. This effect
is robust to controlling for demographic characteristics (Column 2)
and remains stable when including firm fixed effects, suggesting
selection into firms does not drive the result."
```

---

### 9. Conclusion as Abstract Repeat

**Bad**: Copying the abstract with minor rewording

**Better**: Add value by discussing:
- Policy implications
- Mechanisms (if new insight)
- Limitations (briefly, honestly)
- Future research directions
- Broader significance

---

## Technical Writing Mistakes

### 10. Inconsistent Notation

**Bad**:
```
Equation (1) uses β for the treatment effect
Equation (3) uses δ for the same parameter
Table 2 reports τ for what appears to be β
```

**Better**: One symbol per concept, defined clearly once.

---

### 11. Equation Overload

**Bad**:
```
y = α + βx + ε                           (1)
y = α + βx + γz + ε                      (2)
y = α + βx + γz + δw + ε                 (3)
yᵢ = α + βxᵢ + γzᵢ + εᵢ                  (4)
yᵢₜ = α + βxᵢₜ + γzᵢₜ + εᵢₜ              (5)
yᵢₜ = αᵢ + λₜ + βxᵢₜ + γzᵢₜ + εᵢₜ        (6)
```

**Better**: One main equation, describe variations in text:
```
Our estimating equation is:

yᵢₜ = αᵢ + λₜ + βDᵢₜ + X'ᵢₜγ + εᵢₜ        (1)

where... We estimate variants of (1) progressively adding
controls and fixed effects, as reported in Table 2.
```

---

### 12. Undefined Variables

**Bad**:
```
"We estimate: y = α + βx + Zγ + ε"
[No definitions provided]
```

**Better**:
```
"We estimate:

yᵢₜ = α + βDᵢₜ + X'ᵢₜγ + εᵢₜ              (1)

where yᵢₜ is log hourly earnings for individual i in year t,
Dᵢₜ is an indicator for treatment (equals 1 if individual i
received the intervention by year t), Xᵢₜ is a vector of
demographic controls (age, gender, race), and εᵢₜ is the error
term. Our parameter of interest is β, which captures the
average treatment effect under random assignment."
```

---

## Presentation Mistakes

### 13. Too Many Significant Figures

**Bad**:
```
The coefficient is 0.09234567 (SE = 0.02345678)
```

**Better**:
```
The coefficient is 0.092 (SE = 0.023)
```

Rule: 2-3 significant figures for most estimates.

---

### 14. Ignoring Economic Significance

**Bad**:
```
"The effect is statistically significant at the 1% level."
[Effect is 0.001 standard deviations]
```

**Better**:
```
"While statistically significant, the effect of 0.1 percent
is economically small—equivalent to roughly $50 per year for
the median worker."
```

---

### 15. Table Title as Label

**Bad**:
```
"Table 3: Regression Results"
```

**Better**:
```
"Table 3: Effect of Minimum Wage on Employment"
```

Even better (in notes or title):
```
"Table 3: Minimum Wage Increases Reduce Teen Employment by 2-3%"
```

---

## Logic Mistakes

### 16. Confusing Statistical and Economic Significance

**Bad**:
```
"The effect is highly significant (p < 0.001), demonstrating
that education is important for earnings."
```

**Better**:
```
"Education increases earnings by 9 percent (p < 0.001). To
contextualize: this is roughly equivalent to the return from
three years of job experience."
```

---

### 17. Overstating Causality

**Bad**:
```
"We find that X causes Y" [using OLS with no identification strategy]
```

**Better**:
```
"We document a robust correlation between X and Y. Under the
assumption that [identifying assumption], this relationship
has a causal interpretation."
```

---

### 18. False Precision About Mechanisms

**Bad**:
```
"Our results prove that the mechanism is human capital accumulation."
```

**Better**:
```
"The pattern of results—effects concentrated among workers in
skill-intensive occupations—is consistent with human capital
accumulation, though we cannot rule out signaling explanations."
```

---

### 19. Generalizing Beyond Your Sample

**Bad**:
```
"We find effects for teens in New Jersey, demonstrating that
minimum wage effects are universal."
```

**Better**:
```
"Our findings from New Jersey may not generalize to other
states with different labor market conditions. However, the
institutional similarities with neighboring states suggest
our results may apply to [specific contexts]."
```

---

## Style Mistakes

### 20. Wordiness

**Bad**:
```
"It is important to note that..."
"At this point in time..."
"Due to the fact that..."
"In order to..."
"For the purpose of..."
"In the event that..."
"With regard to..."
"In terms of..."
```

**Better**:
```
Delete "It is important to note that"
"Now" or "Currently"
"Because"
"To"
"For"
"If"
"About" or "Regarding"
Rewrite to avoid
```

---

### 21. Weak Verbs

**Bad**:
```
"There is a relationship between X and Y."
"It is the case that education matters."
"The results are suggestive of a positive effect."
```

**Better**:
```
"X affects Y."
"Education matters."
"The results suggest a positive effect."
```

---

### 22. "Very" and Intensifiers

**Bad**:
```
"The effect is very significant."
"This is extremely important."
"The coefficient is highly positive."
```

**Better**: Let the numbers speak.
```
"The effect is large: a one-standard-deviation change in X
increases Y by 15 percent."
```

---

## Checklist: Before You Submit

### Clarity
- [ ] Main finding appears in abstract with numbers
- [ ] Main finding appears in introduction with numbers
- [ ] Each paragraph has one main point
- [ ] Active voice predominates
- [ ] Jargon is explained when first used

### Structure
- [ ] Literature review positions your contribution
- [ ] Methods section focuses on identification
- [ ] Results section interprets magnitudes
- [ ] Conclusion adds value beyond summary

### Technical
- [ ] Notation is consistent throughout
- [ ] All variables are defined
- [ ] Equations are numbered if referenced
- [ ] Appropriate significant figures

### Tables/Figures
- [ ] Each has an informative title
- [ ] Notes explain everything reader needs
- [ ] Consistent formatting throughout

### Logic
- [ ] Causal claims match identification
- [ ] Statistical vs. economic significance addressed
- [ ] Limitations acknowledged honestly
- [ ] Generalizability discussed
