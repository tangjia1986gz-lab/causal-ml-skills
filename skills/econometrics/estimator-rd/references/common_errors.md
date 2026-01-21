# Common RD Errors

## Error 1: Using Global High-Order Polynomials

### The Mistake

```python
# WRONG: Global polynomial regression
model = smf.ols('Y ~ X + X**2 + X**3 + X**4 + X**5 + treatment', data=df).fit()
```

### Why It's Wrong

- Global polynomials fit to noise, especially at boundaries
- Can create artificial discontinuities or hide real ones
- Sensitive to polynomial order choice
- Gelman & Imbens (2018) show this leads to invalid inference

### Correct Approach

```python
# RIGHT: Local linear regression
from rdrobust import rdrobust
result = rdrobust(Y, X, c=cutoff, p=1)
```

**Reference:** Gelman & Imbens (2018), "Why High-Order Polynomials Should Not Be Used in Regression Discontinuity Designs"

---

## Error 2: Ignoring Manipulation

### The Mistake

Not testing for manipulation of the running variable, especially when units have incentive to sort.

### Why It's Wrong

If units can precisely manipulate to be above/below cutoff:
- Units on each side are no longer comparable
- Selection bias invalidates causal interpretation
- RD assumption (local randomization) violated

### Correct Approach

```python
from rddensity import rddensity

# Always test for manipulation
result = rddensity(X, c=cutoff)
print(f"Manipulation test p-value: {result.pval:.4f}")

if result.pval < 0.05:
    print("WARNING: Evidence of manipulation!")
```

Report the manipulation test result even if it passes.

---

## Error 3: Single Bandwidth

### The Mistake

Only reporting results at one (optimal) bandwidth without sensitivity checks.

### Why It's Wrong

- Results may be artifact of bandwidth choice
- No way to assess robustness
- Reviewers will question validity

### Correct Approach

```python
# Test multiple bandwidths
h_opt = base_result.bws[0]
for mult in [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]:
    result = rdrobust(Y, X, c=cutoff, h=h_opt * mult)
    print(f"BW={mult:.2f}x: estimate={result.coef[0]:.3f} (p={result.pv[0]:.3f})")
```

---

## Error 4: Using Conventional Standard Errors

### The Mistake

```python
# Using conventional SE for inference
ci_wrong = result.coef[0] +/- 1.96 * result.se_conventional
```

### Why It's Wrong

Conventional standard errors don't account for:
- Boundary bias in local polynomial
- Bias from bandwidth selection
- Result: incorrect coverage of confidence intervals

### Correct Approach

```python
# Use robust bias-corrected inference
estimate = result.coef[0]  # Bias-corrected estimate
se = result.se[0]          # Robust SE
ci = result.ci[0]          # Robust CI (correct coverage)
```

---

## Error 5: Extrapolating Beyond Cutoff

### The Mistake

Claiming RD estimate applies to population far from the cutoff.

### Why It's Wrong

RD identifies a **local** effect at the cutoff only:
- Effect may differ for units far from threshold
- No information about treatment effect heterogeneity
- Cannot extrapolate without additional assumptions

### Correct Approach

Be explicit about what RD identifies:

> "We estimate the causal effect of treatment for units at the margin of the cutoff. This local average treatment effect may not generalize to units far from the threshold."

---

## Error 6: Wrong Cutoff Value

### The Mistake

Using approximate or incorrect cutoff value.

### Why It's Wrong

- Treatment rule may differ from nominal cutoff
- Administrative implementation may vary
- Wrong cutoff → wrong estimate

### Correct Approach

1. Verify exact cutoff from official policy documents
2. Check how exactly-at-cutoff observations are treated
3. Consider if cutoff changed over time
4. Look for evidence of "fuzzy" implementation

---

## Error 7: Ignoring Heaping

### The Mistake

Not accounting for discrete/heaped running variables.

**Example:** Test scores that only take integer values, or ages that heap at round numbers.

### Why It's Wrong

- Standard bandwidth selection assumes continuous running variable
- Local polynomial may not perform well with discrete X
- Manipulation tests may fail

### Correct Approach

For discrete running variables:
- Consider local randomization approach
- Use rdlocrand package
- Be transparent about limitations

```python
from rdlocrand import rdlocrand
result = rdlocrand(Y, X, c=cutoff)
```

---

## Error 8: Not Checking Covariate Balance

### The Mistake

Assuming RD is valid without testing covariate smoothness.

### Why It's Wrong

Discontinuities in pre-determined covariates suggest:
- Manipulation (sorting around cutoff)
- Confounding (other things change at cutoff)
- Invalid identification

### Correct Approach

```python
# Test all pre-determined covariates
covariates = ['age', 'education', 'income_pre', 'gender']
for cov in covariates:
    result = rdrobust(df[cov], df['x'], c=cutoff)
    sig = "*" if result.pv[0] < 0.05 else ""
    print(f"{cov}: {result.coef[0]:.3f} (p={result.pv[0]:.3f}){sig}")
```

---

## Error 9: Confusing Sharp and Fuzzy

### The Mistake

Using sharp RD estimator when treatment is fuzzy, or misinterpreting fuzzy RD results.

### Why It's Wrong

- Sharp RD: Treatment perfectly determined by cutoff crossing
- Fuzzy RD: Treatment probability jumps but not 0→1
- Different estimators, different interpretations

### Correct Approach

**First, diagnose the design:**
```python
# Check treatment probability on each side
below = df[df['x'] < cutoff]['treatment'].mean()
above = df[df['x'] >= cutoff]['treatment'].mean()
print(f"P(T|X<c) = {below:.3f}, P(T|X>=c) = {above:.3f}")

# If jump is < 1, use fuzzy RD
if above - below < 0.95:
    result = rdrobust(Y, X, c=cutoff, fuzzy=T)
```

**Interpret correctly:**
- Fuzzy RD estimates LATE (effect for compliers only)
- NOT the effect for everyone
- NOT the effect for always/never-takers

---

## Error 10: No RD Plot

### The Mistake

Presenting RD results without visual evidence.

### Why It's Wrong

- Readers cannot assess if discontinuity is real
- May miss functional form issues
- Standard practice requires visualization

### Correct Approach

Always include:
1. Binned scatter plot (outcome vs running variable)
2. Fitted lines on each side
3. Clear cutoff indicator

---

## Error 11: Treating LATE as ATE

### The Mistake

Claiming fuzzy RD effect applies to all treated units.

### Why It's Wrong

Fuzzy RD identifies effect only for **compliers at the cutoff**:
- Those who would comply with treatment rule at threshold
- Not always-takers (always treated regardless of X)
- Not never-takers (never treated regardless of X)
- Not defiers (assumed away by monotonicity)

### Correct Approach

Be precise about interpretation:

> "The fuzzy RD estimate of X.XX represents the local average treatment effect for compliers - those whose treatment status is determined by crossing the threshold."

---

## Error 12: Asymmetric Bandwidth Without Justification

### The Mistake

Using very different bandwidths on each side without explanation.

### Why It's Wrong

May indicate:
- Data problems on one side
- Manipulation
- Specification searching

### Correct Approach

- Start with symmetric bandwidth (default in rdrobust)
- If asymmetric needed, justify based on:
  - Different data density
  - Different curvature
  - Institutional reasons
- Report both symmetric and asymmetric results

---

## Quick Reference: DO vs DON'T

| DON'T | DO |
|-------|-----|
| Global high-order polynomials | Local linear/polynomial |
| Ignore manipulation test | Report McCrary/density test |
| Single bandwidth | Multiple bandwidth sensitivity |
| Conventional SEs | Robust bias-corrected SEs |
| Skip covariate balance | Test all pre-determined covariates |
| No visualization | Always include RD plot |
| Extrapolate findings | Acknowledge local identification |
| Confuse sharp/fuzzy | Diagnose design correctly |
| Treat LATE as ATE | Be precise about estimand |
| Ignore heaping | Consider discrete running variables |

---

## Key References

- Gelman & Imbens (2018): Against global polynomials
- Lee & Lemieux (2010): Common pitfalls discussion
- Cattaneo, Idrobo, Titiunik (2020): Modern best practices
- McCrary (2008): Manipulation testing
