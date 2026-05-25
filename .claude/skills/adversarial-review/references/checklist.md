# Domain-Specific Failure-Mode Checklists

These are the specific failure modes reviewers tend to hit in each area. After Phase 1 calibration, pick the checklist(s) matching the paper and use them as a prompt-list when writing the persona reviews. Not exhaustive — meant to *seed* the reviews with the right reflexes.

---

## Information-theory papers

The single highest-yield checklist for radiogenomics / multimodal-prediction / bound papers.

- **Is the bound tight?** Or just a restatement of DPI / Fano in dressed-up notation? State explicitly where it's tight and where it's loose.
- **MI estimation:** which estimator? KSG? MINE? InfoNCE? Each has known pathologies in high dimensions (McAllester-Stratos: variational bounds are loose by a factor exponential in MI). Sample complexity?
- **Does the conclusion survive a different estimator?** If not, the result is about the estimator, not about the data.
- **Assumptions:** iid? Stationarity? Continuity? Discrete vs. continuous? Are they realistic for the actual data?
- **Notation hygiene:** $I(X;Y)$ vs. $I(X,Y)$, $\hat{I}$ for estimators, units (nats vs. bits), conditioning bars.
- **Closely-related priors:** Tishby IB papers, Kolchinsky on bottleneck variants, Belghazi MINE, McAllester-Stratos, Poole et al. on variational bounds, the recent DPI-equality-conditions literature.
- **Overclaim risk:** "We characterize fundamental limits of X" vs. "we give an upper bound on MI under assumption Y." These are very different claims.

---

## Empirical ML papers

- **Baselines:** are they the right ones? Are they the *current* ones (a baseline from 2020 in a 2026 paper is a red flag)? Tuned with similar effort?
- **Statistical significance:** error bars across seeds? How many seeds? Reported variance or just point estimates?
- **Ablations:** does every architectural choice have an ablation? If 5 things changed, which one(s) cause the improvement?
- **Datasets:** standard or custom? If custom, is the construction documented and is there leakage?
- **Compute:** how much was used to train and to tune? If tuning compute is omitted, the comparison is suspect.
- **Negative results:** any reported, or only the wins? Suspicious if everything works.
- **Test set hygiene:** any chance of contamination? Was the test set looked at during development?
- **Failure modes:** does the paper analyze when the method *fails*? Papers that only show wins are weaker than papers that bound their own claims.

---

## Theory papers (general)

- **Theorem statements:** precisely stated? All assumptions in the statement, not buried in the proof?
- **Are the assumptions vacuous?** I.e., is the regime where they hold actually interesting?
- **Tightness:** lower bounds matching the upper bounds? Or just an upper bound?
- **Proofs:** sketched in main text, full in appendix? Verifiable?
- **Connection to prior bounds:** explicit comparison, with constants?
- **What's the "if you only remember one theorem" theorem?** If the paper has 12 lemmas and no single takeaway, that's a structural problem.

---

## Applied / clinical / biological papers

- **Is the data fit for purpose?** Sample size, selection bias, batch effects, label quality.
- **Is the setup realistic?** Or a clean lab-bench scenario that won't translate?
- **Confounders:** controlled for? Identified at all?
- **Causal claims:** is the paper claiming causation? Is the design adequate for that?
- **Reproducibility:** code, data access, preprocessing pipeline. For biomedical: ethics, IRB, data provenance.
- **Effect size vs. statistical significance:** $p < 0.05$ on $n=10{,}000$ may be a tiny effect.
- **External validity:** would this hold in a different cohort, scanner, hospital, country?

---

## Systems / engineering papers

- **Workload:** representative? Synthetic vs. real?
- **Comparison hardware:** apples-to-apples?
- **Tail latency:** reported, or just means?
- **Failure modes under load:** characterized?
- **Real-world deployment:** any, or just benchmark numbers?

---

## Universal failure modes (check on every paper)

These hit every field. Always include in the reviews.

1. **Title vs. content gap.** Does the title promise more than the paper delivers?
2. **Abstract vs. content gap.** Same question, one level deeper.
3. **Contribution count inflation.** "Our contributions are: 1, 2, 3, 4, 5" where 2 and 4 are restatements of 1.
4. **Missing related work.** Search for the obvious nearby papers; check they're cited and discussed (not just listed).
5. **Figure quality.** Readable? Axis labels? Captions self-contained? Colors accessible?
6. **Equation hygiene.** Defined before used? Notation consistent across sections?
7. **Reproducibility statement.** Present? Honest?
8. **Limitations section.** Present? Honest, or performatively brief?
9. **Conclusion overclaim.** Does the conclusion say things the body didn't show?
