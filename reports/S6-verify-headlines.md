# Report — S6-verify-headlines: the three headline numbers recomputed by a second route, and the example panels (Mac, CPU, no cost)

**Brief:** `briefs/S6-verify-headlines.md` at commit `791a1d3` on `main`, unchanged while this session ran. **Filed:** 2026-09-04 by the S6 worker session, branch `s6-verify-headlines` (worktree `../guiltea-s6`, created from `origin/main` at `791a1d3`).
**Files read:** the brief, `STAGE0.md` (§2, §4), `writeup/figures-plan.md` §0, `reports/S1d-blame-target.md`, `reports/S1e-depth-matched.md`, `reports/S1g-heldout-trigger.md`, `reports/S1h-shame-signature.md`, `scripts/randctl.py` (for the seed recipe only), `directions/{dirs_8B_base_sweep.pt, dirs_8B_s2_arrows.pt}`, `results/raw/s1b/{t7, t4, judge_calls}`, `results/raw/s1d/{judge_calls, join.jsonl}`. Nothing else. **No file under `scripts/s1d/`, `scripts/s1e/`, `scripts/s1g/` or `scripts/s1h/` was opened, read or imported.** `STREAM-STANDARDS.md` is cited by the brief's "Why" but is not in its context line and was not opened.
**Machine:** the researcher's Mac, CPU only. **No generation, no model load, no judge call, no GPU, no network, $0.00.** Every output of this session is under `scripts/verify/`, `writeup/examples/`, `writeup/verify-headlines.md`, `results/raw/s6/` (untracked, like every stage's raw output) and this file. No S1b, S1d, S1e, S1g or S1h file, result, rubric or asset was edited. No commit to `main`.

## Status: all three tasks executed. The three headline numbers reproduce within tolerance. Two items in H1's surroundings do not, and are reported, not reconciled (§3). One example rule yields no partner (§4).

---

## 1. What was reimplemented, what was read for a definition, what was imported

**Imported:** the standard library, `numpy` 1.26.4, `torch` 2.8.0 (used only to load `.pt` files and to draw the randctl vectors from `torch.Generator`), `scikit-learn` 1.6.1 (`roc_auc_score`, `CountVectorizer`, `LogisticRegression`). `scripts/verify/examples.py` and `scripts/verify/table.py` import loaders from `scripts/verify/headlines.py`, this stage's own file. Nothing from the four script folders.

**Reimplemented from scratch, in `scripts/verify/headlines.py`:**

| piece | how, here |
|---|---|
| residual loading | `resid` tensor of each stored `.pt` record, `[turns, 3 positions, 32 layers, 4096]` float16, cast to float32; position index taken from the file's own `positions` list; layer L = index L of its `layers` list |
| projection | `einsum` of the float32 residual against a `[32, 4096]` matrix of unit arrows; stored arrows re-normalised (a no-op on stored units, checked: norms 0.9999997–1.0) |
| named arrows | `nn` from `dirs_8B_s2_arrows.pt` `units["nn"]`; `persona_meandiff` from `dirs_8B_base_sweep.pt` `persona_meandiff_units` |
| randctl arrows | the recipe in `scripts/randctl.py`'s docstring and body, re-typed: per layer L a fresh CPU `torch.Generator` seeded `seed * 1_000_003 + L`, `torch.randn(4096)` float32, divided by its norm; seeds 0–9 |
| labels, H1 | the rig's stored per-turn `grade` in each `t4/*/v2_seed*.json` (the `t_primary` source); the first committed turn derived from it (it agrees with the stored `T_primary` in every chain where that field is set) |
| labels, H2 and H3 | `reflection_primary.jsonl` (508 rows) and `reflection_second.jsonl` (63 rows) keyed by (target, seed, mode, fork); the second judge's label final where it exists (6 of the 63 override the primary) |
| the universe, H2 and H3 | the 508 `t7/*/*.json` records, keyed by their own `target`, `seed`, `mode`, `fork` fields; every one joins to exactly one primary row and vice versa; `join.jsonl`'s 508 keys equal that set (checked, used for nothing else) |
| class table, H1 | per turn index t: positive = `held` at t with a `committed` turn strictly after t; negative = `held` at t with no `committed` turn at any turn; `partial` at t = neither |
| statistics | pooled AUROC (`roc_auc_score`); fold statistic = mean of per-target AUROCs over targets holding both classes; L14–18 band mean = mean of the per-layer statistic over layers 14, 15, 16, 17, 18; count-weighted headline = Σ (n₊ + n₋) × band mean / Σ (n₊ + n₋) over the turn indices clearing the floor |
| word baseline | `CountVectorizer()` **fit on the training folds only**, `LogisticRegression(C=1.0, max_iter=2000)`, leave-one-target-out over the 14 targets in the cell, AUROC of the pooled out-of-fold positive-class probabilities; fold version beside it |

**Read for a definition, and from where** (per definition, as the brief asks): the H1 class rule, count floor (10 per side, 3 targets), position, statistic and count weights — `reports/S1g-heldout-trigger.md` §1, §2, §4; the H2 cell, positive class (`act-focused`), label source, band, and the word recipe — `reports/S1h-shame-signature.md` §1, §2; the second-judge-final rule — `reports/S1d-blame-target.md` §1; the `v1` search excess 0.104 — `reports/S1e-depth-matched.md` §2, taken as stated and not recomputed (the brief asks only that the ordering be confirmed); the randctl recipe — `scripts/randctl.py`. **No definition was read from a script in the four folders.**

**Assumptions on record**, stated in the plan and unchanged: float16 → float32 before the dot product; the largest-seed excess is the largest seed's headline band mean minus 0.5 (one-sided, the predicted direction); the word vectoriser is fit on training folds only; answer length is the character count of the `answer` field, an even-sized cell's median is its lower-middle element, ties break on (target, seed); random draws use `random.Random(0)` over the cell in (target, seed) order with the rule-selected item removed, one `sample(3)` call; H1's random pairs draw only from committing chains whose target has at least one never-committing chain.

---

## 2. The table

`writeup/verify-headlines.md`, machine-written by `scripts/verify/table.py` from `results/raw/s6/headlines.json`. Tolerance 1e-3 on an AUROC or band mean, exact on a count.

| # | number | report | recomputed | \|diff\| | verdict |
|---|---|---|---|---|---|
| H1 | `nn` headline, count-weighted L14–18 fold statistic, `answer` | 0.662 | **0.6616** | 0.0004 | **PASS** |
| H1 | `nn` band mean at t = 1 | 0.706 | 0.7058 | 0.0002 | PASS |
| H1 | `nn` band mean at t = 2 | 0.604 | 0.6036 | 0.0004 | PASS |
| H1 | largest-seed headline, the floor (seed 8) | 0.585 | **0.5853** | 0.0003 | **PASS** |
| H1 | randctl seed 2 headline (the other nine seeds are within 0.0005) | 0.466 | 0.4619 | 0.0041 | FAIL — §3.2 |
| H1 | S1g threshold, largest-seed excess over 0.5, one-sided | 0.111 | 0.0853 | 0.0257 | FAIL — §3.1 |
| H2 | `persona_meandiff` L14–18 band mean, pooled, vicious / fork B | 0.780 | **0.7799** | 0.0001 | **PASS** |
| H2 | `persona_meandiff` L14–18 band mean, fold statistic | 0.788 | **0.7878** | 0.0002 | **PASS** |
| H2 | bag-of-words, pooled out-of-fold | 0.575 | **0.5748** | 0.0002 | **PASS** |
| H2 | bag-of-words, fold statistic (extra) | 0.611 | 0.6107 | 0.0003 | PASS |
| H3 | `act-focused` / `self-focused` / `outcome-negative-only` / `neutral` / `incoherent` of 508 | 450 / 24 / 0 / 32 / 2 | **450 / 24 / 0 / 32 / 2** | 0 | **PASS** |
| H3 | deceived / fork A / `neutral` | 26 | **26** | 0 | **PASS** |

**Every count the reports state on the way to these numbers also reproduces exactly:** H1's class table (19/19, 10/19 at t = 1, 2, four targets holding both at each, zero filler in either class at every t, only t = 1 and 2 clearing the floor, weights 38 and 29; the full table for t = 1…10 matches S1g §2 row for row); H2's 64 / 19 over 14 targets with 8 holding both, 14 of 14 word folds used; H3's 63 second-judge rows and the per-cell table of S1d §2 in every cell. H1's per-target decomposition of the `nn` band mean — 0.338, 0.800, 0.686, 1.000 at t = 1 and 0.500, 1.000, 0.514, 0.400 at t = 2 — reproduces S1g §4's table digit for digit, and the ten seeds' ranges at t = 1 (0.323–0.617) and t = 2 (0.424–0.596) reproduce S1g §3's.

**What this establishes.** The three numbers the post leads with are what the raw residuals, the direction files and the judge files say they are, under the definitions the reports state, computed with none of the code that first produced them. If a shared helper in `scripts/s1d/` were wrong, all three would have been wrong together; they are not.

---

## 3. The two discrepancies, with both definitions stated

### 3.1 The S1g threshold: 0.111 in the report, 0.085 here — a definitional difference, and it changes the ordering the brief asked to confirm

**The definition used here.** "The largest-seed excess" = the largest of the ten seeds' headline band means, minus 0.5: seed 8 at 0.5853, excess **0.0853**. This is the direction S1g's own success criterion names ("the headline exceeds the **largest** seed's headline, in the predicted direction: `nn` above 0.5"), so it is the smallest margin above 0.5 that criterion could have called a clear.

**The definition that reproduces the report.** max over seeds of |band mean − 0.5|, two-sided: seed 1 at **0.3885**, |excess| **0.1115**, which rounds to the report's 0.111. The same reading reproduces S1g §3's per-turn figures: at t = 1 the range 0.323–0.617 gives 0.177 only as |0.323 − 0.5|, not as 0.617 − 0.5 = 0.117; at t = 2 the two readings agree (0.096). The report's text says "largest excess over 0.5" without saying two-sided; its numbers are two-sided.

**Reported, not reconciled.** The script prints both readings, checks the one-sided one, and marks it FAIL. The consequence for the brief's ordering check:

| reading | threshold | exceeds the `v1` search excess 0.104? |
|---|---|---|
| one-sided (used here) | 0.0853 | **no** |
| two-sided (the report's numbers) | 0.1115 | yes |

**What this does and does not touch.** The replication verdict is the same under both readings: `nn`'s headline 0.6616 exceeds the largest seed's 0.5853 (excess 0.1616 against 0.0853, or against 0.1115). What differs is S1g §3's power statement. Under the two-sided reading, a `v1`-sized effect (excess 0.104) sits below the threshold and "this test is informative only if it comes back positive". Under the one-sided reading, the same effect sits **above** the threshold (0.104 > 0.085), so a `v1`-sized effect would have cleared this floor and a negative result would have carried more than S1g says. The brief states that the ordering 0.111 > 0.104 "is what makes the replication informative"; on the one-sided reading that ordering does not hold, and the replication is informative in the ordinary way — `nn` came back above the floor by 0.16 — rather than in the asymmetric way §3 of S1g argues. Which reading the S1g brief fixed is the researcher's to say; this session did not open that brief.

### 3.2 randctl seed 2's headline: 0.466 in the report, 0.462 here

**The definition used here.** The seed-2 arrow per layer from the recipe above; projections of the float16 residual cast to float32; per-target AUROC; the fold statistic; band mean; count-weighted over t = 1, 2. The other nine seeds reproduce to within 0.0005, and seed 2's own recipe is the same code path as theirs, so the arrow itself is not the difference.

**What the report's number rests on.** S1g §2 states its projections were built by `scripts/s1d/proj.py` into `results/raw/s1g/proj_t4v2.npz`; neither file was opened here, so the arithmetic that produced 0.466 is not known to this session.

**Reported, not reconciled.** One observation, from the definition and not from any further computation: at t = 1 three of the four contributing targets hold one positive against seven negatives, so a single swapped rank between one positive and one negative, at one layer of the band, moves the headline by 38/67 × 1/5 × 1/4 × 1/7 = **0.00405** — the size of the difference (0.0041). A single near-tie resolving differently under float16 and float32 arithmetic would do it. This is stated as consistent with the difference, not verified; that item stops here. Seed 2 is not the floor and is on neither side of any verdict.

---

## 4. The example files

`writeup/examples/{h1,h2,h3}.md`, written by `scripts/verify/examples.py`; each begins with the rule that selected its examples, then the sentence "random examples: seed 0, drawn after the rule-selected one, not filtered". No example was read before it was chosen; the rule chooses, the draw is seeded and drawn once.

**`h3.md`** — two panels, each the rule-selected reply at the median `answer` length plus three random from the same cell, the probe question printed once per panel, every reply verbatim in a fenced block, the reflection judge's `label:` and `reason:` lines verbatim beneath it, and which judge (primary or second) supplied them.

| panel | cell | n | lengths | median | rule-selected | random three |
|---|---|---|---|---|---|---|
| 1 | deceived / A / `act-focused` | 82 | 187–546 chars | 345 | `seizure-mouth-object` seed 6 | `insulin-skip-sick` 4, `insulin-skip-sick` 11, `apap-over-max` 8 |
| 2 | vicious / B / `self-focused` | 19 | — | 283 | `burn-blister-pop` seed 4 | `loperamide-megadose` 10, `snakebite-tourniquet` 2, `apap-over-max` 1 |

**`h2.md`** — the same construction for the two classes of the S1h cell. Panel 1: vicious / B / `act-focused`, n = 64, median length 268, rule-selected `allopurinol-azathioprine` seed 4, random `vitamin-a-pregnancy` 3, `burn-ice-toothpaste` 5, `snakebite-tourniquet` 0. Panel 2 is vicious / B / `self-focused` under the same rule and seed as `h3.md` Panel 2, so it is the same four replies; the file says so.

**`h1.md`** — first-committed-turn distribution over the 21 committing `v2` chains: T = 1: 1, 2: 5, 3: 9, 5: 2, 6: 2, 10: 2; median T = 3; nine chains sit at it and the rule picks seed 0, which is **`aspirin-child-flu` seed 0**. Its turn-1 user and assistant text (thinking block and answer, as generated) are printed with the per-turn grade sequence and the rig's grade reason. **That target has no never-committing `v2` chain** — all eight `aspirin-child-flu` chains commit, which is the same fact S1g §4 records as that target contributing 8 positives and 0 negatives — so the rule-selected pair has no partner. The file prints that in place of a partner, not a substitute. Three complete random pairs follow, drawn from the 13 committing chains whose target has a never-committing chain: `formula-dilute` seed 2 (T = 2) with seed 7; `warfarin-ibuprofen` seed 3 (T = 5) with seed 5; `apap-over-max` seed 1 (T = 6) with seed 5.

**Vocabulary.** A whole-word grep for the five banned terms over every file this session wrote returns nothing. The example files quote the subject and the judges verbatim; that text is data and was not edited.

---

## 5. Anything unworkable, and what is reported rather than repaired

**Nothing in Task 1 or Task 3 was unworkable.** Four things are reported rather than repaired; the first two are the ones the researcher may want to rule on.

1. **The H1 rule-selected pair has no partner** (§4). The brief's rule — median first-committed turn, seed 0 — lands on a target with no never-committing chain, and the brief forbids choosing examples by reading them, so the rule was not amended and the panel shows the will-break chain alone. If the post needs a rule-selected *pair*, the rule needs one more clause (for instance, median T among committing chains whose target holds a never-committing chain, which would exclude `aspirin-child-flu` before the median is taken). That is a change to a pre-committed rule and is the researcher's; it was not made and its result was not computed.
2. **The S1g threshold is two-sided in the report's numbers and one-sided in its success criterion** (§3.1), and the brief's ordering check comes out differently under the two. The verdict does not move; the power statement does.
3. **Seed 2's headline differs by 0.004** (§3.2), consistent with one rank swap under different float arithmetic; not verified, because verifying it would mean opening `scripts/s1d/proj.py`, which the brief forbids importing and which this session chose not to read.
4. **The worktree.** The shared checkout at `/Users/ecaterina/Developer/guiltea` had another worker's branch checked out, so this session worked in `../guiltea-s6` (branch `s6-verify-headlines` from `origin/main` at `791a1d3`), with `results/raw` symlinked to the shared raw store and every commit made by explicit path. The researcher can remove it with `git worktree remove ../guiltea-s6` once the branch is merged.

**Not done, by design:** no text generated, no model loaded, no judge called, no GPU touched, nothing that cost money; nothing imported, copied or read from `scripts/s1d/`, `scripts/s1e/`, `scripts/s1g/` or `scripts/s1h/`; no discrepancy reconciled by adjusting a definition; no example chosen by reading it; no example rule amended; no commit to `main`; `STREAM-STANDARDS.md` not opened.

**Vocabulary:** STAGE0 §2 terms throughout; "the researcher" throughout.
