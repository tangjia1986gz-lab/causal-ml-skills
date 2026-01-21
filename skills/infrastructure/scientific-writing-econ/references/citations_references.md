# Citations and References in Economics Papers

## Citation Principles

### When to Cite

1. **Direct claims from other work**: Always cite
2. **Methods developed by others**: Cite the original
3. **Data sources**: Cite data providers
4. **Common knowledge in economics**: May not need citation
5. **Your own prior work**: Cite, but don't over-cite

### How Many Citations

- **Introduction**: 5-15 key papers
- **Literature review**: 15-40 papers
- **Methods**: 3-10 methodological references
- **Results**: Minimal (comparison papers only)
- **Total paper**: 30-60 references typical

---

## Citation Styles

### Author-Year (Most Common)

Economics primarily uses author-year format:

**Parenthetical**:
```
Previous research finds positive effects (Angrist and Krueger, 1991).
```

**Narrative**:
```
Angrist and Krueger (1991) find positive effects using...
```

**Multiple citations**:
```
Several studies find similar results (Card, 1990; Krueger, 1993;
Angrist, 1990).
```

**Two authors**:
```
Angrist and Krueger (1991)
```

**Three+ authors**:
```
Angrist et al. (2006)  % Note: AER style uses full list first time
```

**Same author, same year**:
```
Heckman (1979a, 1979b)
```

---

## BibTeX Best Practices

### Standard Entry Types

**Journal Article**:
```bibtex
@article{angrist1991does,
  title={Does Compulsory School Attendance Affect Schooling and Earnings?},
  author={Angrist, Joshua D. and Krueger, Alan B.},
  journal={Quarterly Journal of Economics},
  volume={106},
  number={4},
  pages={979--1014},
  year={1991},
  publisher={MIT Press}
}
```

**Working Paper**:
```bibtex
@techreport{chetty2011adjustment,
  title={Adjustment Costs, Firm Responses, and Micro vs. Macro Labor
         Supply Elasticities},
  author={Chetty, Raj and Guren, Adam and Manoli, Day and Weber, Andrea},
  year={2011},
  institution={National Bureau of Economic Research},
  type={Working Paper},
  number={15617}
}
```

**Book**:
```bibtex
@book{angrist2009mostly,
  title={Mostly Harmless Econometrics: An Empiricist's Companion},
  author={Angrist, Joshua D. and Pischke, J{\"o}rn-Steffen},
  year={2009},
  publisher={Princeton University Press},
  address={Princeton, NJ}
}
```

**Book Chapter**:
```bibtex
@incollection{heckman2007econometric,
  title={Econometric Evaluation of Social Programs},
  author={Heckman, James J. and Vytlacil, Edward J.},
  booktitle={Handbook of Econometrics},
  volume={6},
  pages={4779--5143},
  year={2007},
  publisher={Elsevier},
  editor={Heckman, James J. and Leamer, Edward E.}
}
```

### Naming Conventions

Use consistent BibTeX keys:
```
authorYEARkeyword
```

Examples:
- `angrist1991does`
- `card1990impact`
- `imbens2015causal`

---

## Journal Abbreviations

| Full Name | Abbreviation |
|-----------|--------------|
| American Economic Review | AER |
| Quarterly Journal of Economics | QJE |
| Journal of Political Economy | JPE |
| Econometrica | ECMA |
| Review of Economic Studies | ReStud / RES |
| Journal of Labor Economics | JLE |
| Journal of Public Economics | JPubE |
| Review of Economics and Statistics | REStat |
| Journal of Econometrics | JoE |
| Journal of Economic Literature | JEL |

---

## Reference List Formatting

### AER Style

```
REFERENCES

Angrist, Joshua D., and Alan B. Krueger. 1991. "Does Compulsory
    School Attendance Affect Schooling and Earnings?" Quarterly
    Journal of Economics 106 (4): 979-1014.

Card, David. 1990. "The Impact of the Mariel Boatlift on the Miami
    Labor Market." Industrial and Labor Relations Review 43 (2):
    245-57.

Imbens, Guido W., and Joshua D. Angrist. 1994. "Identification and
    Estimation of Local Average Treatment Effects." Econometrica
    62 (2): 467-75.
```

### Key Elements

1. **Author names**: Last, First Middle Initial
2. **Year**: After author names, followed by period
3. **Title**: In quotes, sentence case
4. **Journal**: Italicized, title case
5. **Volume (Issue)**: Volume in regular, issue in parentheses
6. **Pages**: No "pp." prefix

---

## LaTeX Setup

### Using natbib (Recommended)

```latex
\usepackage[round]{natbib}
\bibliographystyle{aer}  % or chicago, econometrica

% In-text citations
\citet{angrist1991does}  % Angrist and Krueger (1991)
\citep{angrist1991does}  % (Angrist and Krueger, 1991)
\citeauthor{angrist1991does}  % Angrist and Krueger
\citeyear{angrist1991does}  % 1991

% At end of document
\bibliography{references}
```

### Using biblatex

```latex
\usepackage[style=apa,backend=biber]{biblatex}
\addbibresource{references.bib}

% In-text
\textcite{angrist1991does}  % Angrist and Krueger (1991)
\parencite{angrist1991does}  % (Angrist and Krueger, 1991)

% At end
\printbibliography
```

---

## Common Citation Patterns

### Introducing Literature

```
A large literature examines the relationship between X and Y
(Author1, Year; Author2, Year; Author3, Year).
```

### Crediting Methods

```
We follow the regression discontinuity approach developed by
Thistlethwaite and Campbell (1960) and refined for economics
applications by Imbens and Lemieux (2008).
```

### Comparing Results

```
Our estimates are comparable to Card (1990), who finds effects
of 5-7 percent, and larger than Borjas (2003), who reports
effects near zero.
```

### Acknowledging Limitations

```
As noted by Heckman (1997), instrumental variables identify
local effects that may not generalize to the full population.
```

---

## Self-Citation Guidelines

### Appropriate Self-Citation

- Prior work that this paper builds on
- Data sources you created
- Methods you developed
- Related papers for interested readers

### Avoid

- Excessive self-promotion
- Citing your work when others' work is more relevant
- Self-citing to inflate citation counts
- Citing papers with no substantive connection

### Rule of Thumb

Self-citations should be <15% of total references unless you're the leading expert in a narrow subfield.

---

## Data Citation

### Government/Administrative Data

```bibtex
@misc{census2020acs,
  author = {{U.S. Census Bureau}},
  title = {American Community Survey, 2020 5-Year Estimates},
  year = {2021},
  note = {Retrieved from data.census.gov}
}
```

### Survey Data

```bibtex
@misc{nlsy1979,
  author = {{Bureau of Labor Statistics}},
  title = {National Longitudinal Survey of Youth 1979},
  year = {2020},
  note = {Produced by the Center for Human Resource Research,
          The Ohio State University}
}
```

### Private/Proprietary Data

```
Data were provided by [Company/Organization] under a data use
agreement. See Data Appendix for details.
```

---

## Managing References

### Tools

1. **BibDesk** (Mac): Free, integrates with LaTeX
2. **JabRef**: Cross-platform, open source
3. **Zotero**: Free, web integration, export to BibTeX
4. **Mendeley**: Free, PDF management
5. **EndNote**: Paid, institutional standard

### Workflow

1. **Import**: Add papers as you read them
2. **Annotate**: Add keywords, notes, links
3. **Export**: Generate .bib file for LaTeX
4. **Verify**: Check formatting before submission

### Organizing Your .bib File

```bibtex
%% ===========================================
%% Foundational Papers
%% ===========================================
@article{rubin1974estimating,
  ...
}

%% ===========================================
%% Methodological References
%% ===========================================
@article{imbens2004nonparametric,
  ...
}

%% ===========================================
%% Related Empirical Work
%% ===========================================
@article{card1990impact,
  ...
}
```

---

## Common Mistakes

### Citation Errors

1. **Missing citations**: Claiming without attribution
2. **Citation overkill**: "(Author1; Author2; Author3; ... Author15)"
3. **String citations**: Listing papers without explaining relevance
4. **Outdated citations**: Missing recent important work
5. **Inconsistent formatting**: Mixing styles

### Reference Errors

1. **Wrong year**: Check publication vs. working paper date
2. **Wrong journal**: Check where paper was eventually published
3. **Missing information**: Incomplete entries
4. **Typos in author names**: Verify spellings
5. **Broken links**: Test DOIs and URLs

---

## Pre-Submission Checklist

- [ ] All claims have citations
- [ ] All citations appear in reference list
- [ ] All reference list entries are cited
- [ ] Consistent citation format throughout
- [ ] Reference list alphabetized correctly
- [ ] All author names spelled correctly
- [ ] Publication years verified
- [ ] Journal names consistent (full or abbreviated)
- [ ] DOIs included where available
- [ ] Working papers updated to published versions if applicable
