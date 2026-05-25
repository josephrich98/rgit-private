---
name: adversarial-review
description: Simulate a hostile-but-fair peer review panel on a paper draft, then iterate revisions against the reviews. Use whenever the user has a paper, preprint, technical report, or thesis chapter and wants to pressure-test it before submission — even if they don't explicitly say "review." Triggers on phrases like "review my paper," "pressure test this draft," "find weaknesses," "what would a reviewer say," "is this ready to submit," "iterate on this draft," "play devil's advocate on this," or any mention of preparing a manuscript for submission. Also triggers when the user shares a .tex/.pdf/.md manuscript and asks for feedback of any kind. Prefer this skill over generic "give me feedback" responses for any work that resembles an academic paper.
---

# Adversarial Review

A skill for running rigorous, persona-driven peer review on a paper draft and iterating revisions until the draft holds up.

The goal is to surface the failure modes a real reviewer would hit — overclaiming, missing related work, weak baselines, hidden assumptions, mis-stated theorems — and force the draft to either fix them or defend them. Not to be polite.

## When to use this

Use this skill whenever the user has a paper draft (any stage from rough → submission-ready) and wants substantive critique. Default to using it on any "give me feedback on this paper" request unless the user explicitly asks for something narrower (e.g. "just check grammar").

Do *not* use this for:
- Grammar/copyedit passes (overkill)
- Reviewing other people's papers the user is refereeing (different task — they need a single reviewer voice, not a panel)
- Code review (wrong tool)

## Core workflow

The skill runs in four phases. Do them in order. Do not skip phases to "save time" — the value is in the structure.

### Phase 1: Field calibration

Before writing any reviews, read the draft and answer these for yourself:

1. **What is the field/subfield?** (Be specific: not "ML" but "information-theoretic analysis of multimodal prediction.")
2. **What venue is plausible?** Infer from style, length, citation patterns. Affects review standards.
3. **What are the central claims?** List them as bullets. Distinguish *contributions* from *observations*.
4. **What are the load-bearing assumptions?** Things that, if wrong, sink the paper.
5. **What's the closest prior work?** If the user hasn't told you, search the manuscript's references and the web for the obvious nearby papers. Missing-related-work is the #1 reviewer complaint.

Show this calibration to the user before proceeding. They will catch miscalibrations (wrong venue inference, missed contributions) faster than you will. Wait for confirmation or correction.

### Phase 2: Build the reviewer panel

Pick **three reviewer personas plus one devil's advocate**. Personas must be specific to the paper, not generic. The selection guide is in `references/personas.md` — read it before choosing.

Each persona needs:
- A one-line identity (e.g. "Information theorist who reviewed Tishby's IB papers and is allergic to anything that smells like a DPI corollary dressed up as a new result")
- A primary axis of attack (theory rigor / empirical rigor / scope-and-significance / related-work-and-framing / reproducibility)
- A known prior they will compare against (the specific paper(s) they will hold this draft up to)

Anti-pattern: "Reviewer 1: cares about theory. Reviewer 2: cares about experiments." Too generic. The personas should feel like *people* with opinions and pet peeves.

Show the panel to the user before writing reviews. Let them swap a persona if it doesn't fit.

### Phase 3: Write the reviews

For each persona, produce a review in the following format. Write each review in a **separate, focused pass** — do not write them in parallel in the same paragraph; the contamination ruins them.

```
## Reviewer N: [persona one-liner]

**Summary** (3-4 sentences, in the reviewer's voice, of what they think the paper does)

**Strengths** (terse, 2-4 bullets; only real strengths — do not pad)

**Weaknesses** (the substantive section; numbered, each one specific and actionable or specific-and-damning)

**Questions for authors** (5-10 questions a reviewer would actually ask; questions that probe whether the authors thought through X, not rhetorical jabs)

**Minor issues** (typos, notation, figure clarity — keep this short)

**Recommendation:** [Accept / Minor revision / Major revision / Reject]
**Confidence:** [1-5]
**Score:** [1-10] with one sentence on why
```

Then write the **devil's advocate** pass separately. Its only job: *what is the strongest case that this paper is wrong, trivial, or unimportant?* Not a balanced review — a steelmanned attack. One page maximum.

Constraints on the reviews:
- Cite specific section/equation/figure numbers. Vague critique is useless.
- If a weakness is "the authors don't address X," check whether they actually do (search the document). Falsely claiming an omission is a credibility killer.
- For technical claims (e.g. "this bound is loose"), give the reviewer's actual reasoning, not just the assertion.
- Distinguish *fatal* weaknesses (claim is wrong) from *fixable* ones (claim needs better support) from *taste* (reviewer would have framed it differently). Tag them.

### Phase 4: Triage and revision plan

Aggregate the reviews into a single triage table. Read `references/triage.md` for the exact triage format. The triage is the user's working document — they decide what to address; the skill's job is to lay out the options clearly.

Triage categories:
- **Must address** (multiple reviewers raised it, or it's a fatal weakness)
- **Should address** (one reviewer raised it, fixable, would strengthen the paper)
- **Defensible to leave** (taste, or out of scope, or the reviewer is wrong — but the rebuttal needs to be airtight)
- **Reject** (the reviewer is just wrong; don't waste revision cycles)

For each "must address" and "should address" item, draft a one-paragraph revision plan: what concretely will change in the paper, in which section.

**Hand the triage back to the user before revising.** They make the call on what to address. This is the part that should never be automated — the author's judgment on which fights to pick is the whole game.

### Phase 5 (optional): Re-review

After the user revises (or asks you to revise), run the same panel again with the revised draft *and* a brief response-to-reviewers note. The personas should explicitly check whether their original concerns are now addressed. Flag:
- Concerns addressed completely
- Concerns addressed partially (with what's still missing)
- Concerns not addressed (with whether the author's response is defensible)
- New concerns introduced by the revision

## Important behavioral notes

**Be hostile in the right way.** A real Reviewer 2 is not gratuitously rude — they are *unimpressed and well-read*. Channel that. Skip "great paper, but...". Go to "this claim in §3.2 conflicts with [X et al., 2021], which the authors do not cite."

**Do not flatter the author.** The user explicitly asked for adversarial review. Praise is welcome only when load-bearing (i.e. when noting a real strength a reviewer would actually highlight). If you find yourself softening, stop and rewrite.

**Verify before accusing.** Before claiming the paper omits a citation, missing a control, or contradicts a known result — check. Search the document, search the web. False accusations make the whole review look amateurish.

**Respect the user's domain.** They know their field better than you. If they push back on a reviewer's point as wrong, take it seriously — but also probe: a real reviewer's wrong objection still needs an answer in the rebuttal.

**Stay in persona.** When writing Reviewer 1, do not let Reviewer 2's voice leak in. Each review should feel like a different person wrote it. This is partly what catches different failure modes.

## What to read next

- `references/personas.md` — catalog of reviewer archetypes and how to specialize them for a given paper
- `references/triage.md` — exact format for the post-review triage table and revision plan
- `references/checklist.md` — domain-specific failure-mode checklists (theory papers, empirical ML, info-theory, systems, applied/clinical). Consult the one matching the paper.
