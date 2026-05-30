# Reviewer Personas

The default panel size is **three reviewers + one devil's advocate**. Each must be specialized to the paper, not pulled generically from this list.

## How to specialize a persona

A generic persona ("information theorist") catches generic problems. A specialized persona ("information theorist who has spent ten years thinking about MI estimation in high dimensions, is skeptical of any paper that doesn't address the McAllester-Stratos bounds, and recently reviewed a paper that overclaimed using a similar setup") catches the specific problems your paper is at risk for.

Specialize along these axes:
- **Specific priors they've internalized** (papers they consider canonical)
- **Pet peeves** (things that trigger an instant "Reviewer 2" reaction)
- **Background bias** (e.g., theorist who thinks empirical papers are sloppy; empiricist who thinks theory papers are detached)
- **Recent context** (a paper they just reviewed, a debate they just had at a workshop)

## Archetypes — pick three, specialize each

### A. The theory hawk
Cares about: rigor, novelty of results, tightness of bounds, whether theorems are stated correctly, whether assumptions are realistic. Will catch: hidden assumptions, results that are corollaries of known theorems, mis-stated lemmas, vacuous bounds.

Specialize by: pinning their subfield (info theory? optimization? statistical learning theory?) and the 2-3 canonical priors they will compare against.

### B. The empiricist
Cares about: experimental design, baselines, statistical significance, ablations, reproducibility. Will catch: weak baselines, missing ablations, cherry-picked metrics, p-hacking, dataset issues.

Specialize by: which empirical tradition (large-scale ML benchmarks? biostatistics? clinical trials? psychology?). Each has different conventions for what counts as rigorous.

### C. The domain expert / scope critic
Cares about: whether the paper engages with the right literature, whether the problem is well-motivated, whether the claims generalize. Will catch: missing related work, overclaiming, problem mis-framing, ignored alternative explanations.

Specialize by: the specific subfield. This is the reviewer most likely to know the obscure paper that scoops you.

### D. The methodologist / statistician
Cares about: are the estimators valid, are the confidence intervals real, is the data analysis defensible, are there confounders. Will catch: estimation bias, multiple-testing issues, mis-applied tests, fragile estimators in high dimensions.

Particularly important for: info-theory papers (MI estimation is famously fragile), causal claims, any paper with statistical analysis.

### E. The reproducibility hawk
Cares about: code availability, hyperparameters, compute, dataset access, ability of others to extend the work. Will catch: missing details, unspecified preprocessing, hidden tuning.

Usually not the primary persona unless the venue weights this heavily (e.g. NeurIPS reproducibility track, MLSys).

### F. The applied / translational reviewer
For papers touching applied domains (medicine, biology, policy): cares about whether the setup is realistic, whether the result would matter in practice, whether the data is fit for purpose. Will catch: clean-but-unrealistic setups, results that don't translate, ignored practical constraints.

### G. The framing skeptic
Cares about whether the paper's narrative matches its content. Will catch: titles that overclaim, abstracts that promise more than the body delivers, "we solve X" when the paper solves a special case of X.

Most papers should include either C or G — the framing/scope axis is almost always where overclaiming hides.

## Panel composition heuristic

- **Theory paper:** A + (C or G) + D
- **Empirical ML paper:** B + (C or G) + E
- **Info-theory or statistical analysis paper:** A + D + (C or G)
- **Applied / clinical / interdisciplinary:** F + (C or G) + (A or B depending on emphasis)
- **Theory-meets-applied (e.g. info-theoretic bounds in a domain):** A + D + F

If unsure, default to A + B + C — but specialize hard.

## The devil's advocate

Separate from the three reviewers. Single job: *strongest case that this paper is wrong, trivial, or unimportant*. Not balanced. Not a fourth review. One page max.

Useful framings the DA can take:
- "This result is a corollary of [known theorem]."
- "The setup is so artificial that the result doesn't matter."
- "The empirical evidence shows the opposite of what the authors claim, if you read figure X carefully."
- "The contribution is real but tiny, and dressed up to look bigger."

Pick whichever framing has the most teeth given the paper.
