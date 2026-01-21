# DAGs and Graphical Causal Models

## Overview

Directed Acyclic Graphs (DAGs) provide a visual and formal language for representing causal relationships. Developed primarily by Judea Pearl, the graphical approach complements the potential outcomes framework and offers powerful tools for identifying when causal effects can be estimated from observational data.

---

## Basic Terminology

### Graph Elements

| Element | Symbol | Description |
|---------|--------|-------------|
| **Node** | Circle (A, B, C) | A variable in the system |
| **Edge** | Arrow (A → B) | Causal relationship from A to B |
| **Path** | A → B → C | Sequence of edges connecting nodes |
| **Directed path** | A → B → C | Path following arrow directions |
| **Undirected path** | A → B ← C → D | Path ignoring arrow directions |

### Key Definitions

**Directed Acyclic Graph (DAG)**: A graph where:
- All edges have directions (arrows)
- No cycles exist (cannot return to starting node by following arrows)

**Parent**: A node with an arrow pointing TO another node
- In A → B, A is a parent of B

**Child**: A node receiving an arrow FROM another node
- In A → B, B is a child of A

**Ancestor**: A node that can reach another by following arrows forward
**Descendant**: A node reachable from another by following arrows forward

---

## Three Fundamental Structures

### 1. Chain (Mediation)

```
A → B → C
```

- A causes B, B causes C
- B is a **mediator** between A and C
- A affects C through B

**Association flow**: A and C are associated (through B)
**Conditioning on B**: Blocks the path; A and C become independent

**Example**:
```
Smoking → Tar deposits → Lung cancer
```
Controlling for tar deposits would block the effect of smoking on lung cancer.

### 2. Fork (Confounding)

```
    B
   / \
  v   v
 A     C
```
(Or equivalently: A ← B → C)

- B causes both A and C
- B is a **confounder** (common cause)
- A and C are associated even without direct causation

**Association flow**: A and C are associated (through B)
**Conditioning on B**: Blocks the path; removes confounding

**Example**:
```
      Socioeconomic status
           /        \
          v          v
    Education     Health
```
SES confounds the relationship between education and health.

### 3. Collider (Selection)

```
 A     C
  \   /
   v v
    B
```
(A → B ← C)

- Both A and C cause B
- B is a **collider** (common effect)
- A and C are NOT associated (path is blocked)

**Association flow**: A and C are **not** associated (path blocked at B)
**Conditioning on B**: **Opens** the path; creates spurious association!

**Example**:
```
  Talent     Looks
     \       /
      v     v
    Hollywood success
```
Among Hollywood actors (conditioning on success), talent and looks appear negatively correlated.

---

## d-Separation

### Definition

d-separation (directional separation) is a graphical criterion that determines whether two sets of variables are conditionally independent given a third set.

### Rules for d-Separation

A path between X and Y is **blocked** given conditioning set Z if:

1. **Chain/Fork**: The path contains a non-collider that is IN Z
   - X → M → Y: Blocked if M ∈ Z
   - X ← M → Y: Blocked if M ∈ Z

2. **Collider**: The path contains a collider that is NOT in Z (and no descendant of the collider is in Z)
   - X → C ← Y: Blocked if C ∉ Z and no descendant of C in Z

### d-Separation Algorithm

To check if X ⊥ Y | Z:

1. List all paths between X and Y
2. For each path:
   - Check each node on the path
   - Apply blocking rules
3. If ALL paths are blocked, X ⊥ Y | Z

### Example

```
    U
   / \
  v   v
 X → Y ← Z
      |
      v
      W
```

**Is X ⊥ Z?** (conditioning on nothing)
- Path X → Y ← Z: Blocked at collider Y
- Yes, X ⊥ Z

**Is X ⊥ Z | Y?** (conditioning on Y)
- Path X → Y ← Z: Collider Y is conditioned on → path opened
- No, X ⊥̸ Z | Y

**Is X ⊥ Z | W?** (conditioning on W, descendant of Y)
- Path X → Y ← Z: W is descendant of collider Y → path opened
- No, X ⊥̸ Z | W

---

## Backdoor Criterion

### Definition

A set of variables Z satisfies the **backdoor criterion** relative to (X, Y) if:

1. No node in Z is a descendant of X
2. Z blocks every path from X to Y that contains an arrow INTO X (backdoor paths)

### Why "Backdoor"?

Backdoor paths are non-causal paths from X to Y that go "backward" into X first:
```
X ← ... → Y  (starts with arrow into X)
```

These create confounding. Blocking them isolates the causal effect.

### Example

```
    Z
   / \
  v   v
 X → Y
```

**Backdoor paths from X to Y**: X ← Z → Y
**Does Z satisfy backdoor criterion?**
1. Z is not a descendant of X ✓
2. Conditioning on Z blocks X ← Z → Y ✓

Yes! Controlling for Z identifies the causal effect of X on Y.

### Adjustment Formula

If Z satisfies the backdoor criterion:

$$P(Y | do(X)) = \sum_z P(Y | X, Z=z) P(Z=z)$$

This allows causal effect estimation from observational data.

---

## Frontdoor Criterion

### Definition

A set of variables M satisfies the **frontdoor criterion** relative to (X, Y) if:

1. M intercepts all directed paths from X to Y
2. There is no unblocked backdoor path from X to M
3. All backdoor paths from M to Y are blocked by X

### When to Use

Frontdoor is useful when:
- There is unmeasured confounding between X and Y
- The causal effect operates through observable mediators

### Example

```
    U (unobserved)
   / \
  v   v
 X → M → Y
```

- Cannot control for U (unobserved)
- Backdoor path X ← U → Y is unblocked
- But M satisfies frontdoor criterion!

**Frontdoor adjustment**:

$$P(Y | do(X)) = \sum_m P(M=m | X) \sum_{x'} P(Y | M=m, X=x') P(X=x')$$

---

## Do-Calculus

### The do() Operator

$P(Y | do(X=x))$ represents the distribution of Y when X is **set** (intervened) to value x, rather than **observed** at value x.

**Key distinction**:
- $P(Y | X=x)$: Conditional probability (observation)
- $P(Y | do(X=x))$: Interventional probability (manipulation)

### Graphical Interpretation

The operation $do(X=x)$:
1. Sets X to value x
2. Removes all arrows INTO X (severs dependence on causes of X)
3. Keeps arrows OUT of X (preserves effects of X)

### Example

Original DAG:
```
Z → X → Y
    ↑
    W
```

After $do(X=x)$:
```
Z    X → Y
     ↑
     (X is set, not caused by Z or W)
```

### The Three Rules of do-Calculus

Given DAG G:

**Rule 1** (Insertion/deletion of observations):
$$P(Y | do(X), Z, W) = P(Y | do(X), W)$$ if $(Y ⊥ Z | X, W)$ in $G_{\bar{X}}$

**Rule 2** (Action/observation exchange):
$$P(Y | do(X), do(Z), W) = P(Y | do(X), Z, W)$$ if $(Y ⊥ Z | X, W)$ in $G_{\bar{X}\underline{Z}}$

**Rule 3** (Insertion/deletion of actions):
$$P(Y | do(X), do(Z), W) = P(Y | do(X), W)$$ if $(Y ⊥ Z | X, W)$ in $G_{\bar{X}\bar{Z(W)}}$

These rules allow converting interventional queries to observational quantities when possible.

---

## Common DAG Patterns

### 1. Confounding

```
    U
   / \
  v   v
 X → Y
```

- U confounds X-Y relationship
- Must control for U (if observed) to identify causal effect
- If U is unobserved: need IV, RD, or other strategy

### 2. Mediation

```
X → M → Y
 \     ↗
  \___/
```

- M mediates part of X's effect on Y
- Direct effect: X → Y
- Indirect effect: X → M → Y
- Controlling for M removes indirect effect (may not want this!)

### 3. Collider Bias

```
X → C ← Y
```

- X and Y are independent
- Conditioning on C creates spurious association
- "Selection bias" when C represents sample selection

### 4. M-Bias

```
U₁    U₂
 \    /
  v  v
X ← M → Y
```

- Naive thought: control for M
- But M is a collider for U₁ and U₂
- Controlling for M OPENS a backdoor path!

### 5. Butterfly Bias

```
U₁ → X → Y ← U₂
 \         /
  \       /
   \     /
    v   v
      C
```

- C is a collider (descendant of confounders)
- Controlling for C opens non-causal paths

---

## Building DAGs

### Steps for DAG Construction

1. **List all relevant variables**
   - Treatment, outcome, confounders, mediators
   - Observed and unobserved

2. **Draw direct causal arrows**
   - For each pair (A, B): Does A directly cause B?
   - "Direct" means not through other variables in the graph

3. **Check for common causes**
   - Any omitted confounders?
   - Add unobserved nodes if necessary

4. **Verify no cycles**
   - Causal effects are assumed to flow one direction
   - No feedback loops (use different time subscripts if needed)

5. **Assess assumptions**
   - Every missing arrow is an assumption!
   - Missing arrow from A to B means "A does not directly cause B"

### DAG Assumptions

**Causal Markov Assumption**: Each node is independent of its non-descendants given its parents.

**Faithfulness**: All independence relations are represented in the DAG; no additional "accidental" independencies.

**Causal Sufficiency**: No unmeasured common causes of observed variables (often violated!).

---

## DAG Examples in Economics

### Returns to Education

```
      Ability (U)
       /      \
      v        v
Education → Earnings ← Experience
     ↑
Family background
```

- Ability confounds education-earnings
- If ability is unobserved: need IV (e.g., quarter of birth)

### Minimum Wage

```
         Economic conditions
              /        \
             v          v
    Min wage increase → Employment
             |
             v
        Worker welfare
```

- Economic conditions confound policy evaluation
- DID uses timing of policy as quasi-experiment

### Health Insurance

```
       Wealth
      /     \
     v       v
Insurance → Health utilization → Health outcomes
                    ↑
                Health needs
```

- Multiple confounders
- Oregon Health Insurance Experiment used lottery for randomization

---

## Common Mistakes with DAGs

### 1. Controlling for a Collider

**Mistake**: Include all available covariates in regression
**Problem**: Some may be colliders, opening non-causal paths

### 2. Controlling for a Mediator

**Mistake**: Control for post-treatment variables
**Problem**: Removes part of the causal effect you want to measure

### 3. Omitting Confounders

**Mistake**: Assume all confounders are observed
**Problem**: Backdoor paths remain open

### 4. Circular Reasoning

**Mistake**: X → Y → X (feedback loop)
**Problem**: Violates acyclicity; need time-indexed variables

### 5. Over-Simplified Structure

**Mistake**: Draw X → Y only
**Problem**: Ignores all the identification challenges

---

## Relationship to Potential Outcomes

| Graphical Approach | Potential Outcomes |
|-------------------|-------------------|
| Nodes = Variables | Variables in model |
| do(X=x) intervention | Setting X=x for all units |
| Backdoor criterion | Conditional independence |
| Confounding | Selection bias |
| Collider bias | No direct analog (different framework) |

**Equivalence**: Under certain conditions, graphical and potential outcomes approaches yield the same identification results.

**Complementary**: DAGs help reason about confounding; potential outcomes clarify estimands.

---

## Related Skills

- `estimator-psm` - Propensity Score Matching (uses backdoor criterion)
- `estimator-iv` - Instrumental Variables (graphical representation)
- `estimator-did` - Difference-in-Differences (parallel trends assumption)
- `causal-mediation-ml` - Mediation Analysis (DAG-based decomposition)

---

## Key References

1. **Pearl, J. (2009)**. *Causality: Models, Reasoning, and Inference* (2nd ed.). Cambridge University Press.

2. **Pearl, J., Glymour, M., & Jewell, N. P. (2016)**. *Causal Inference in Statistics: A Primer*. Wiley.

3. **Morgan, S. L., & Winship, C. (2014)**. *Counterfactuals and Causal Inference* (2nd ed.). Cambridge University Press.

4. **Elwert, F. (2013)**. Graphical causal models. In *Handbook of Causal Analysis for Social Research* (pp. 245-273). Springer.

5. **Hernán, M. A., & Robins, J. M. (2020)**. *Causal Inference: What If*. Chapman & Hall/CRC.
