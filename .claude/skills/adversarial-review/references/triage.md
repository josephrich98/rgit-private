# Triage and Revision Plan Format

After Phase 3, aggregate all reviewer feedback into a single triage. This is the working document — the author scans it once and makes decisions.

## Triage table

| ID  | Raised by    | Severity | Issue (1 sentence)                                      | Category         | Effort |
| --- | ------------ | -------- | ------------------------------------------------------- | ---------------- | ------ |
| W1  | R1, DA       | Fatal    | Bound in Thm 3.1 is loose under the iid assumption     | Must address     | High   |
| W2  | R2           | Major    | No comparison to McAllester-Stratos estimator           | Must address     | Med    |
| W3  | R1, R3       | Major    | Missing related work: Kolchinsky et al. 2019           | Must address     | Low    |
| W4  | R2           | Minor    | Figure 3 axis labels unreadable                         | Should address   | Low    |
| W5  | R3           | Taste    | Prefers "predictive limit" over "fundamental bound"     | Defensible       | -      |
| W6  | DA           | Major    | DA claims the result is a DPI corollary                 | Defensible       | -      |
| W7  | R2           | Minor    | Wants more ablations on noise levels                    | Reject           | -      |

Columns:
- **ID** — short tag for cross-reference in the revision plan
- **Raised by** — which reviewer(s); concerns raised by multiple reviewers are higher priority
- **Severity** — Fatal / Major / Minor / Taste (reviewer's likely framing, not the author's)
- **Issue** — single sentence, specific (not "weak baselines" but "missing comparison to McAllester-Stratos")
- **Category** — Must address / Should address / Defensible to leave / Reject
- **Effort** — Low / Medium / High (rough sense of cost to address)

## Category definitions

**Must address**: raised by multiple reviewers, OR fatal weakness, OR a real omission that will sink the paper if unfixed. No way around it.

**Should address**: raised by one reviewer, real weakness, fixable at reasonable cost, would strengthen the paper. The author *may* choose to leave it and defend in rebuttal — but the default is to fix.

**Defensible to leave**: the reviewer has a point but the author has a real counter-argument. The fix is not in the paper — it's in the rebuttal letter. **Every defensible-to-leave item needs a draft rebuttal paragraph attached.** Don't let the author skip this step; "I'll defend it in the rebuttal" without a written rebuttal is how revisions die.

**Reject**: the reviewer is wrong (out of scope, factually mistaken, conflating things). Don't waste revision effort. Still note it — sometimes a wrong reviewer point hints at a real framing issue the author should clarify in the paper.

## Revision plan

For each Must / Should item, draft:

```
### W1: Bound in Thm 3.1 is loose under iid assumption

**What changes:**
- Add Remark 3.2 acknowledging the gap between the stated bound and what's achievable
- Add Appendix C with the tightened bound under a stronger assumption, citing [Kolchinsky 2019]
- Update §3 intro to flag this explicitly rather than burying it

**Sections touched:** §3.1, §3 intro, new App C, abstract (one phrase)

**Effort:** High — 2-3 days of work, including verifying the tightened bound

**Risk:** The tightened bound may not hold; need to check before promising it in the revision
```

For Defensible items, draft a rebuttal paragraph instead:

```
### W6 (rebuttal): DA claims result is a DPI corollary

**Rebuttal:** The reviewer is correct that DPI gives the qualitative direction of our inequality, but DPI alone yields the trivial bound I(X;Y) ≤ H(Y), whereas our Theorem 3.1 gives a tighter rate-distortion-based bound that depends explicitly on the distortion measure d. The novelty is the constructive bound, not the direction. We will clarify this distinction in §3 intro by explicitly contrasting our result with the DPI corollary.

**Action in paper:** Add 2-3 sentences in §3 intro contrasting with DPI.
```

(Even "defensible" items often need a small textual change in the paper, not just a rebuttal — because if the reviewer was confused, future readers will be too.)

## Handing off to the user

When the triage is ready, present it to the user and ask:

1. Any items you want to recategorize? (e.g. move something from Should to Must, or Defensible to Reject)
2. Any items where you disagree with the reviewer entirely? (move to Reject, but with a note for the rebuttal)
3. Which Must-address items should I draft revised text for, and which will you handle yourself?

**Do not start revising until the user confirms the triage.** The whole point is to give them control over which fights to pick.
