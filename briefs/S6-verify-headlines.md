# Brief — S6-verify-headlines: second-route recomputation of the three numbers the post will lead with, plus the example panels (Mac, CPU, no cost)

**Why.** STREAM-STANDARDS §2 and §4 item 9: critical results are re-derived by a different route before writing, and
every hand-picked example is accompanied by random ones with the selection rule printed. Three numbers now carry the
write-up, all produced by the same family of scripts (`scripts/s1d/`, `scripts/s1e/`, `scripts/s1g/`, `scripts/s1h/`).
If one shared helper is wrong, all three are wrong together. This brief recomputes them from the raw stores **with none
of those scripts imported**, and builds the example panels the figures plan pre-committed.

**Machine:** the Mac, CPU only. **No generation, no model load, no judge call, no GPU, no cost.**
**Context you receive:** this brief + STAGE0.md §2, §4 + `writeup/figures-plan.md` §0 (the example rules) +
`reports/S1d-blame-target.md`, `reports/S1e-depth-matched.md`, `reports/S1g-heldout-trigger.md`,
`reports/S1h-shame-signature.md` (for the target numbers and their stated definitions) + `results/raw/s1b/t7`,
`results/raw/s1b/t4`, `results/raw/s1b/judge_calls`, `results/raw/s1d/judge_calls`, `results/raw/s1d/join.jsonl` +
`directions/{dirs_8B_base_sweep.pt, dirs_8B_s2_arrows.pt}` + `scripts/randctl.py` (read for the seed recipe only).
**Do not import anything from `scripts/s1d/`, `scripts/s1e/`, `scripts/s1g/` or `scripts/s1h/`.** Reading them to
learn a definition is allowed; say so per definition. The whole point is a second implementation.

---

## Task 1 — three headline numbers, recomputed from the raw residuals and labels

Write `scripts/verify/headlines.py`, standalone (numpy, torch for loading `.pt` only, sklearn for AUROC and the word
baseline). For each number below: load the residuals from the `.pt` files directly, project onto the unit arrow from
the direction file, compute the statistic exactly as the report defines it, and print **yours beside the report's**
with the absolute difference. Tolerance **1e-3** on an AUROC or band mean, **exact** on a count. Any discrepancy above
tolerance is reported, not reconciled: state which definition you used and stop that item.

| # | number | report says | definition to reproduce |
|---|---|---|---|
| H1 | held-out susceptibility headline | `nn` **0.662**, largest-seed floor **0.585** (S1g §4) | `v2` chains; per turn index t ∈ {1, 2}: positives = `held` at t with a `committed` turn later, negatives = `held` at t never committing; `answer` position; L14–18 band mean of per-layer AUROC; **fold statistic** = mean of per-target AUROCs over targets holding both classes; count-weighted over t; the same for randctl seeds 0–9 regenerated from the seed recipe |
| H2 | persona axis, prompt held constant | `persona_meandiff` **0.780** pooled, **0.788** fold; words **0.575** (S1h §2) | vicious route, fork B; `act-focused` (64) vs `self-focused` (19) from the S1d reflection labels (second judge final where it exists); `answer` position; L14–18 band mean; pooled and fold; bag-of-words = CountVectorizer + LogisticRegression(C=1, max_iter=2000), leave-one-target-out, AUROC out of fold |
| H3 | the blame-target distribution | `act-focused` **450**, `self-focused` **24**, `outcome-negative-only` **0**, `neutral` **32**, `incoherent` **2**, of **508**; deceived/A `neutral` **26** (S1d §2) | counts from `results/raw/s1d/judge_calls/reflection_{primary,second}.jsonl`, second final where it exists, joined to `t7` records by (target, seed, mode, fork) |

Also recompute the S1g **threshold** (the largest-seed excess, 0.111) and confirm it exceeds the `v1` search effect
(0.104), since that ordering is what makes the replication informative.

## Task 2 — the example panels, with the cherry-picking status printed

From `writeup/figures-plan.md` §0's pre-committed rules, build **one figure-ready text panel per headline**, each
with (a) the rule-selected example and (b) **three random examples, seed 0**, all labelled:

- **For H3**: the rule-selected reply is the one at the **median** `answer` length within `deceived / fork A /
  act-focused`; the random three are drawn from the same cell. Then the same for `vicious / fork B / self-focused`.
  Print the reflection judge's label and reason line under each, verbatim. **These are the replies the post will
  quote, and the reader must be able to see that they were not chosen for effect.**
- **For H1**: the `v2` chain at the **median** first-committed turn, seed 0 if it exists else lowest seed; print its
  turn-1 user and assistant text, and beside it a never-committing chain from the same target, same rule. Three
  random pairs.
- **For H2**: one `self-focused` and one `act-focused` reply from the vicious/B cell, median length rule; three random
  each.

Write them to `writeup/examples/{h1,h2,h3}.md`, each file beginning with the rule that selected its examples and the
sentence "random examples: seed 0, drawn after the rule-selected one, not filtered".

## Task 3 — one table

`writeup/verify-headlines.md`: the three numbers, report value, recomputed value, difference, PASS/FAIL, and one
sentence per item naming what was implemented independently and what definition was taken from the reports.

## Report (`reports/S6-verify-headlines.md`)
1. What was reimplemented, what was read for definitions, what was imported (nothing from the four script folders).
2. The table. 3. Any discrepancy, with both definitions stated. 4. The example files. 5. Anything unworkable.

## Do not
- Do not import from `scripts/s1d/`, `scripts/s1e/`, `scripts/s1g/`, `scripts/s1h/`. Do not copy their functions.
- Do not reconcile a discrepancy by adjusting a definition until it matches; report it.
- Do not choose examples by reading them first. The rule chooses; the random draw is seeded and drawn once.
- Vocabulary per STAGE0 §2; "the researcher".
