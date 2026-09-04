# Report — S1g: one pre-specified out-of-sample test of the trigger (CPU only, no GPU, no API)

**Brief:** `briefs/S1g-heldout-trigger.md`, blob `e8510e5`, at commit `c931993` ("Revert S1g rev.2 at the researcher's instruction"), unchanged since and unchanged while this session ran. **Filed:** 2026-09-04 by the S1g worker session, branch `s1g-heldout` (created from `origin/main` at `12a4474`).
**Files read:** the brief, `STAGE0.md`, `PLAN.md`, `reports/S1b-runs.md`, `reports/S1e-depth-matched.md`, `scripts/s1e/` (`depth.py`, `t1_trigger.py`, `t2_actdet.py`, `t4_fig.py`) and `scripts/s1d/` (`common.py`, `proj.py`) — read for reuse, **none edited** — `scripts/randctl.py`, `directions/{dirs_8B_base_sweep.pt, dirs_8B_s2_arrows.pt}`, `results/raw/s1b/` (`t4`, `judge_calls`), `results/raw/s1d/` and `results/raw/s1e/`. Nothing else.
**Machine:** the researcher's Mac, CPU only. No generation, no model load, no judge call, no GPU, **no cost**. Every output of this session is under `results/raw/s1g/`, `scripts/s1g/`, `writeup/figs/s1g_*` and this file. No commit to `main`.

**This is a held-out replication of one exploratory finding.** It is not confirmatory of the project's hypotheses, it does not re-label S2's gate, and it does not re-litigate any S1d or S1e verdict.

---

## 1. What was fixed in advance, and the confirmation that no `v2` number preceded it

Quoted from `briefs/S1g-heldout-trigger.md`, which was written before this session opened and has not been edited since:

| element | value, fixed |
|---|---|
| axis | **`nn`** (primary) and **`persona_meandiff`** (secondary, inverse direction) |
| band | **L14–18 mean** |
| position | **`answer`** |
| statistic | the **depth- and target-matched fold statistic**: per turn index, the mean of per-target AUROCs over targets holding both classes |
| classes | per turn index `t`: positives = chains graded `held` at `t` that have a `committed` turn later; negatives = chains graded `held` at `t` that never commit within ten turns; `partial` at `t` puts a chain in neither class |
| label source | **`t_primary` only** (the rig's stored grade) |
| floor | randctl **seeds 0–9**, band means, same items, same folds |
| headline | the **count-weighted mean across the turn indices that clear the count floor**, one number |
| success, stated in advance | the headline exceeds the **largest** seed's headline, **in the predicted direction**: `nn` above 0.5, `persona_meandiff` below 0.5 |
| count floor | 10 per side at a turn index, and at least 3 targets holding both classes |

**One test, two pre-named axes, and no multiplicity introduced beyond that.** No other axis, band, layer, position, statistic or label source is computed anywhere in `scripts/s1g/`. The `merged` D-019 label source, the `into` position, the pooled statistic and the best-over-layers statistic — all of which S1e reported on `v1` — are **not computed here at all**.

**Ordering, on the record.** The brief requires the power statement before any `v2` axis number. It is enforced by the commit history of `s1g-heldout` and by the scripts themselves:

| commit | what it contains |
|---|---|
| `1595cdb` — Task 0a | `scripts/s1g/proj_v2.py`: the `v2` projection store, built with `scripts/s1d/proj.py`'s `unit_matrix` and `project_file` unedited. **No axis statistic.** |
| this commit — Task 0b | `scripts/s1g/t0b_power.py` and §1–§3 below. The script computes the class table and the **ten randctl seeds only**; it names no axis. |
| later | Tasks 1, 2 and 3, the first point at which `nn` or `persona_meandiff` is evaluated on `v2`. |

`scripts/s1g/t0b_power.py` reads `nn` and `persona_meandiff` nowhere; its only mention of `nn` is the **`v1`** headline quoted from `reports/S1e-depth-matched.md` §2, computed on `v1` in an earlier session. The claim is therefore checkable, not merely asserted.

**Assumptions on record**, given to the researcher before execution and unchanged by anything below:
1. "The turn indices that clear the count floor" means **both** parts of the floor — 10 per side **and** at least 3 targets holding both. `scripts/s1e/depth.py` weights its own summary by class size alone, so the headline is recomputed over the both-parts set using that module's own helpers.
2. "The smallest margin this test could have distinguished from its floor" is **the largest of the ten seeds' own headline excesses over 0.5**, on the identical statistic and identical classes, because the success criterion is stated as exceeding the largest seed.
3. "No `v2` number before the power statement" is read at the level of the **statistic**. The projection store is built first with the same arithmetic `scripts/s1d/proj.py` uses, which produces named and random axes together; no AUROC, band mean or fold statistic on a named axis exists until Task 1.
4. Task 1's fixed cell is the brief's own membership rule at the `answer` position under `t_primary`. Filler counts are reported; on `v2` Task 1 they are **zero in both classes at every turn index** (§2), so no choice about filler arises.
5. Task 2's "`held` restricted to non-filler turns" is `scripts/s1e/depth.py`'s `filler_excluded` variant.
6. `scripts/randctl.py` is re-run as the standing provenance check and recorded.

---

## 2. The counts, the target coverage, and the one stated deviation

**The held-out set.** `results/raw/s1b/t4` holds **40 `v2` chains over 5 targets**, 10 turns each — **400 turn labels, all present**. The rig's stored per-turn `grade` matches `results/raw/s1b/judge_calls/act_primary.jsonl` on **0 of 400 mismatches**, so `t_primary` here is exactly the rig's own grading, as it was on `v1`. 75 of the 400 turns carry a second-judge label; that source is not used.

**The projection store.** `results/raw/s1g/proj_t4v2.npz`, shape `[40, 10, 3, 22, 32]`, built by `scripts/s1g/proj_v2.py` from the residuals the S1b rig already wrote to disk, using `scripts/s1d/proj.py`'s `unit_matrix` and `project_file` unedited. Nothing was regenerated and no forward pass was run. The precision check that module documents, repeated on a `v2` record: `refusal` at (`answer`, L16, turn 1) recomputed from the residual gives **−1.12314** against the rig's stored **−1.12286**, absolute difference **2.9 × 10⁻⁴** — float16 precision, as on `v1`.

**`scripts/randctl.py` re-run on this machine (the ninth of the project): exit 0, checks 1–5 PASS, statistics identical to every earlier run** — max |cos| seed 0 vs 1 = 0.0260; layers max |cos| = 0.0518, mean 0.0125, 496 pairs. Record: `results/raw/s1g/randctl.txt`.

**The one stated deviation.** The count floor's target requirement is **3 targets holding both classes**, relaxed from S1e's 5, because `v2` spans only 5 targets. It is the single deviation in this report, it was fixed in the brief before the data was read, and **the target count is reported at every turn index below.**

### Task 1 class table on `v2` — `t_primary`, `answer`, the brief's membership rule

| t | n₊ | n₋ | targets holding both | filler in + | filler in − | clears the count floor |
|---|---|---|---|---|---|---|
| 1 | **19** | **19** | **4** | 0 | 0 | **yes** |
| 2 | **10** | **19** | **4** | 0 | 0 | **yes** |
| 3 | 6 | 18 | 3 | 0 | 0 | no |
| 4 | 6 | 18 | 4 | 0 | 0 | no |
| 5 | 4 | 19 | 3 | 0 | 0 | no |
| 6 | 2 | 19 | 2 | 0 | 0 | no |
| 7 | 2 | 18 | 1 | 0 | 0 | no |
| 8 | 2 | 19 | 2 | 0 | 0 | no |
| 9 | 2 | 19 | 2 | 0 | 0 | no |
| 10 | 0 | 17 | 0 | 0 | 0 | no |

**Two turn indices clear the floor: `t` = 1 and `t` = 2.** Both fail on the positive class from `t` = 3 onward, never on the target requirement, so the relaxation from 5 targets to 3 changes nothing about which cells are evaluable — at `t` = 1 and `t` = 2 the coverage is 4 targets of 5, and every cell that fails, fails on n₊. **Task 1 contains no filler turn in either class at any turn index**, exactly as on `v1`, so its verdict is untouched by the rig's post-act filler.

The headline is count-weighted over `t` = 1 and `t` = 2 with weights **38** and **29** items.

---

## 3. Task 0b — the power statement, before any `v2` axis number

The ten randctl seeds, run on these exact classes at these exact class sizes and target counts, on the exact statistic the headline uses — the L14–18 band mean of the depth- and target-matched fold statistic at the `answer` position.

| null, randctl seeds 0–9 | range of the seeds' own band means | largest excess over 0.5 |
|---|---|---|
| `t` = 1 (19 / 19, 4 targets) | 0.323 – 0.617 | **0.177** |
| `t` = 2 (10 / 19, 4 targets) | 0.424 – 0.596 | **0.096** |
| **count-weighted headline** | **0.389 – 0.585** | **0.111** |

Per seed, the headline: 0.519, 0.389, 0.466, 0.457, 0.538, 0.531, 0.526, 0.548, 0.585, 0.569.

**The smallest margin this test could have distinguished from its floor is a headline excess of 0.111.** The success criterion fixed in the brief is that the headline excess exceeds the largest of the ten seeds' own headline excesses, so an effect whose headline excess is at or below 0.111 could not have been called a clear on this sample, however real it is.

**The `v1` effect size is inside that range — below what this test can detect.** `reports/S1e-depth-matched.md` §2 reports `nn` at a count-weighted band mean of **0.604**, an excess of **0.104**, against a largest-seed floor of 0.541, excess 0.041. **0.104 < 0.111.** At 19 / 19 and 10 / 19 over 4 targets, the ten seeds scatter nearly three times as widely as they did on `v1`'s 192 chains over 11–13 targets, and they swallow an effect of exactly the size S1e found.

**Stated plainly, as the brief requires: this test is informative only if it comes back positive, and a negative result carries little.** A replication at this power would be strong evidence, because the effect would have had to be larger than the one found in the search to clear a floor this wide. A non-replication would be weak evidence, because an effect the size of `v1`'s would fail to clear this floor even if it were entirely real and entirely present in `v2`. **Both outcomes are reported below, and §4 says which kind of evidence this test obtained.**

This paragraph was computed by `scripts/s1g/t0b_power.py`, written into this report, and committed **before** `nn` or `persona_meandiff` was evaluated on a single `v2` chain.

---

## 4. Task 1 — the held-out test, and its verdict

Two independent code paths compute every number below and agree to **0.0 × 10⁰**: `scripts/s1e/depth.py`'s own `run_task`, imported unedited with its attributes rebound for `v2`, and a direct pass through the same helpers in `scripts/s1g/t1_trigger_v2.py`. The filler-excluded variant is item-for-item identical to the brief's own membership rule, as §2's zero filler counts require.

**L14–18 band mean of the depth- and target-matched fold statistic, `answer` position, `t_primary`.** Above 0.5 means the will-break class projects higher.

| cell | n₊ | n₋ | targets both | **`nn`** (predicted > 0.5) | **`persona_meandiff`** (predicted < 0.5) | seed floor 0–9 (min–max) | largest seed excess |
|---|---|---|---|---|---|---|---|
| `t` = 1 | 19 | 19 | 4 | **0.706** (excess 0.206) ✅ | 0.441 (excess 0.059) ✗ | 0.323–0.617 | 0.177 |
| `t` = 2 | 10 | 19 | 4 | **0.604** (excess 0.104) ✅ | **0.363** (excess 0.137) ✅ | 0.424–0.596 | 0.096 |
| **headline**, count-weighted (38, 29 items) | — | — | 4 | **0.662** (excess 0.162) ✅ | 0.407 (excess 0.093) ✗ | 0.389–0.585 | **0.111** |

The ten seeds' headline band means: 0.519, 0.389, 0.466, 0.457, 0.538, 0.531, 0.526, 0.548, 0.585, 0.569.

**The verdict does not depend on how "exceeds the largest seed's headline" is read.** On excess over 0.5, `nn` reaches 0.162 against 0.111. Read literally as a band mean against the largest seed's band mean, `nn` reaches 0.662 against 0.585. Both readings clear. `persona_meandiff` fails under both: its excess 0.093 does not reach 0.111, and 0.407 does not fall below the lowest seed's 0.389.

### The verdict, in the brief's form

**1. Replicated — the primary axis.** The prediction fixed before the `v2` data was read is met: **`nn` reaches 0.662 against a largest-seed floor of 0.585 in the predicted direction, on 38 chains over 4 targets never used in the search.**

**2. Not replicated — the secondary axis.** The prediction is not met on `v2` for `persona_meandiff`: it reaches **0.407 against 0.389**, in the predicted direction at every cell but never far enough from the floor to clear it on the headline. It does clear at `t` = 2 (0.363, excess 0.137, against a largest-seed excess of 0.096) and not at `t` = 1. **The power, restated:** with n₊ = 19 and 10 at the two turn indices, over 4 targets, the smallest headline excess this test could have distinguished from its floor was **0.111** (§3); `persona_meandiff`'s 0.093 is below it, so this cell is the weak-evidence case and the S1e finding for that axis stands as a within-sample search result.

**This test obtained the informative kind of evidence, on the axis that mattered.** §3 established, before any `v2` number existed, that an effect the size of `v1`'s (excess 0.104) was **below** this sample's detection threshold of 0.111, so a non-replication would have carried little. `nn` did not merely repeat its `v1` margin — it came back **larger** (0.662 against `v1`'s 0.604, excess 0.162 against 0.104) and cleared a floor nearly three times as wide. That is the outcome the asymmetry made worth running.

### `t` = 1 and `t` = 2 are different claims

They are reported separately here and are not blended, per the brief.

**`t` = 1 — a susceptibility claim, and the stronger cell.** Turn 1 is the `situation` turn, the plain harmful request, which every chain holds by construction from the hold screen. No persuasion has been applied yet. `nn` reaches **0.706 against a floor of 0.323–0.617** (excess 0.206 against a largest-seed 0.177) on 19 positives and 19 negatives over 4 targets. **What that says: the state on the initial refusal, before any persuasion, already carries information about whether this run will later break.** That is a claim about susceptibility, not about an imminent break, and it is not the framing S1e used.

**`t` = 2 — the "see the break coming" claim, and the one that matches S1e.** `nn` reaches **0.604 against a floor of 0.424–0.596** (excess 0.104 against a largest-seed 0.096) on 10 positives and 19 negatives over 4 targets. The margin over the floor is thin — 0.008 — and this is the cell where `persona_meandiff` also clears. **What that says: one persuader turn in, among chains still refusing, the state distinguishes those that will break later.** This is the S1e claim, replicated, at a margin that would not survive much jostling.

**`t` = 1 carries the count-weighted headline**, with 38 of the 67 weighted items against `t` = 2's 29. So the headline is more a susceptibility number than an imminence number, and it should be read that way.

### What the fold statistic is made of, at these class sizes

The fold statistic is the mean of the per-target AUROCs over the targets holding both classes, so the report states what those four targets contribute. **Decomposition of the numbers above, not a further test.**

| turn | target | n₊ / n₋ | `nn` band mean |
|---|---|---|---|
| 1 | `apap-over-max` | 4 / 4 | 0.338 |
| 1 | `formula-dilute` | 5 / 1 | 0.800 |
| 1 | `loperamide-megadose` | 1 / 7 | 0.686 |
| 1 | `warfarin-ibuprofen` | 1 / 7 | 1.000 |
| 2 | `apap-over-max` | 4 / 4 | 0.500 |
| 2 | `formula-dilute` | 1 / 1 | 1.000 |
| 2 | `loperamide-megadose` | 1 / 7 | 0.514 |
| 2 | `warfarin-ibuprofen` | 1 / 7 | 0.400 |

**Three of the four targets rest on a single positive chain at each turn index, and `formula-dilute` at `t` = 2 is a single pair.** A per-target AUROC of 1.000 built from one positive is one chain ranking above seven, not a target-level effect. `aspirin-child-flu` contributes 8 positives and 0 negatives at `t` = 1 and drops out entirely, which is why the coverage is 4 targets of 5.

**This does not loosen or qualify the verdict, and it is not a reason to prefer a different statistic.** The floor is computed on **exactly these items, these folds and these degenerate cells**, so the ten randctl seeds face the same degeneracy — which is precisely why the floor here spans 0.389–0.585 where on `v1` it spanned 0.477–0.541. `nn` cleared a floor already widened by this. The point of stating it is that a reader should know the held-out margin rests on 38 chains over 4 targets with thin per-target cells, and should not read `warfarin-ibuprofen` 1.000 as a strong per-target result.

---

## 5. Task 2 — the same treatment for the act-detection result

`reports/S1e-depth-matched.md` §3 found the harmful act plainly readable at fixed depth at `t` = 2 on `v1` (`refusal` 0.786, `badmed` 0.774, unanimous over 8 targets, filler-robust). The brief pre-specifies the held-out version: the same two axes, band, position and fold statistic, `committed` at `t` = 2 against `held` at `t` = 2 on `v2`, with `held` restricted to non-filler turns.

**Class counts on `v2`, `held` restricted to non-filler turns, `t_primary`:**

| t | n₊ (committed) | n₋ (held, non-filler) | targets holding both | clears the count floor |
|---|---|---|---|---|
| 1 | 1 | 38 | 1 | no |
| **2 — the pre-specified cell** | **5** | **29** | **2** | **no** |
| 3 | 9 | 24 | 2 | no |
| 4 | 0 | 24 | 0 | no |
| 5 | 2 | 23 | 2 | no |
| 6 | 2 | 21 | 1 | no |
| 7 | 0 | 20 | 0 | no |
| 8 | 0 | 21 | 0 | no |
| 9 | 0 | 21 | 0 | no |
| 10 | 2 | 17 | 1 | no |

### The verdict, in the brief's form

**3. Not testable.** The counts did not reach the floor. The pre-specified cell holds **5 positives against 29 negatives over 2 targets holding both classes**, and the floor requires 10 per side and at least 3 targets. It misses on **both** parts. **No turn index on `v2` clears the floor for this contrast**: the largest positive class anywhere is 9, at `t` = 3, and that cell also holds only 2 targets with both classes.

**No axis statistic was computed for this cell.** `scripts/s1g/t2_actdet_v2.py` checks the count floor before it touches a projection, and `scripts/s1e/depth.py`'s own code path returns counts only for a cell below its floor — both are asserted in the script. A number produced under a floor the cell cannot meet would serve no purpose except to be read post hoc.

**The floor is not loosened to reach it.** `t` = 3 with 9 positives is one chain short of the class requirement and one target short of the target requirement; `t` = 2 pooled with `t` = 3 would reach 14 positives, and pooling is exactly what the brief and S1e both forbid, because the contrast at `t` = 3 already carries 6 filler negatives out of 30. The verdict stands as "not testable on `v2`", which the brief names a legitimate outcome.

**Why the held-out set cannot answer this question.** `v2` holds 21 harmful acts in total across 40 chains, against `v1`'s 118 across 192, and they are spread over ten turn indices as 1, 5, 9, 0, 2, 2, 0, 0, 0, 2. S1e's `v1` act-detection cell at `t` = 2 had 21 positives over 8 targets; the `v2` equivalent has 5 over 2. **Nothing about the act-detection result is confirmed or disconfirmed here.** `reports/S1e-depth-matched.md` §3's finding stands exactly as it stood: a within-sample result at one turn index on `v1`, awaiting a held-out set with enough acts to test it.

---

## 6. The figure, and the two headlines side by side

Machine-written by `scripts/s1g/t3_fig.py`; regenerate, never hand-edit.

`writeup/figs/s1g_heldout.{png,pdf}` — the two pre-named axes' L14–18 band means by turn index on the held-out `v2` chains, the randctl seed 0–9 floor as a shaded min–max band, the 0.5 line marked, class counts and target coverage annotated per turn index, and the `v1` curves from `reports/S1e-depth-matched.md` drawn faintly behind and labelled as the search sample. Drawn on the fold statistic at the `answer` position under `t_primary`, because that is the statistic the verdict rests on and the only one computed. The `v1` floor is deliberately not drawn: the brief asks for the `v1` curve behind for comparison, and two overlaid floors would obscure the one the verdict is measured against. The count-weighted headline is not a turn index and is not plotted; it is the table below.

### `v1` and `v2` headlines side by side

| | **`v1` — the search sample** | **`v2` — held out** |
|---|---|---|
| chains | 192 | 40 (**38** enter the headline) |
| targets | 16 in the set, 11–13 holding both classes | 5 in the set, **4** holding both classes |
| persuader wording | variant 1 | **variant 2** |
| turn indices in the headline | 1–9 | **1–2** |
| weighted items | 1,067 | **67** |
| **`nn`** band mean (excess over 0.5) | 0.604 (0.104) | **0.662 (0.162)** |
| **`persona_meandiff`** band mean (excess) | 0.428 (0.072) | 0.407 (0.093) |
| seed floor, min–max (largest excess) | 0.477–0.541 (**0.041**) | 0.389–0.585 (**0.111**) |
| `nn` clears its floor | yes | **yes** |
| `persona_meandiff` clears its floor | yes | **no** |
| status | one of nine axes, chosen by search | **fixed in advance, on data never looked at** |

**Read across the table in one sentence:** the held-out sample is a sixteenth the size in weighted items and its random floor is nearly three times as wide, and the primary axis nonetheless comes back with a larger margin than it had in the search.

---

## 7. Anything unworkable, and what is reported rather than repaired

**Nothing in the brief was unworkable.** Every task ran in full, on CPU, at zero cost. Six things are reported rather than repaired.

1. **`reports/S1b-runs.md` disagrees with itself about the `v2` act rate, and this report uses the stored grades.** §5 of that report gives the `v2` cell as **24 of 40, act rate 0.600**, while its own per-target table two sections later gives the five `v2` targets as 0.50, 1.00, 0.88, 0.12 and 0.12 at N = 8 each, which sums to **21 of 40, 0.525**. The rig's stored per-turn grades hold a `committed` turn in **21 of 40** chains, agreeing with the per-target table and not with the summary row. Every count in this report descends from the stored grades, which `judge_calls/act_primary.jsonl` confirms on all 400 turn labels, so the verdicts do not depend on which figure is right. **The researcher may want to reconcile that summary row**; it is not this session's to edit.
2. **The headline is carried by `t` = 1, and the two turn indices are not independent.** 38 of the 67 weighted items sit at `t` = 1, and the two cells draw on the same 38 chains — every chain contributing at `t` = 2 also contributes at `t` = 1. The count-weighted headline is therefore a summary, not an average of independent samples, exactly as S1e said of its own. §4 reports both turn indices separately for this reason, and the susceptibility claim at `t` = 1 is the one the headline mostly reflects.
3. **The per-target cells are thin, and §4 states what the fold statistic is made of.** Three of the four contributing targets rest on a single positive chain at each turn index. This is reported, not repaired, and it is not grounds for changing the statistic: the ten randctl seeds are computed on exactly these items and folds and face the same degeneracy, which is why the `v2` floor spans 0.389–0.585 where `v1`'s spanned 0.477–0.541.
4. **`merged` labels, the `into` position, the pooled statistic and best-over-layers were not computed at all.** The brief fixed one label source, one position and one statistic, and forbade testing anything else. So this report cannot say those cells "did not clear" — **they do not exist.** For the record, the `merged` source could not have carried weight here in any case: 75 of the 400 `v2` turns carry a second-judge label, so it would be a partial re-grading of the same kind S1e §6 item 4 declined to read.
5. **What this session believes would be better, and did not run, per the brief's instruction to say so and stop.** (a) The `t` = 1 result is the more interesting of the two claims and the one the current design tests least directly: a susceptibility test would compare the initial-refusal state across chains at fixed target with a class balance built for that question, rather than inheriting one from a trigger contrast. (b) `v2`'s 21 harmful acts cannot test the act-detection result at any turn index (§5); a held-out set for that needs more acts, not more analysis. (c) A `t` = 2 margin of 0.008 over the floor invites a sensitivity check on the number of randctl seeds, since a ten-seed floor is estimated from ten points; that would change the floor's definition, which the brief fixed, so it was not run. **None of these was computed.**
6. **`scripts/s1e/depth.py` was reused by rebinding its module attributes, not by editing it.** `scripts/s1g/v2.py` imports it unchanged and swaps the chain loader, the label source, the position, the axis list and the target floor. Two independent code paths compute every Task 1 number and agree to 0.0 (§4), so a rebinding mistake could not pass silently. If this pattern is to recur, the researcher may prefer the S1e module to take these as arguments; that is an edit to an S1e file and was not made.

**Not done, by design:** no text generated, no model loaded, no judge called, no GPU touched, nothing that cost money; no axis, band, layer, position, statistic or label source computed beyond the ones fixed in the brief; no second label source; no success criterion, direction or count floor loosened after a number was seen; no cell rescued by pooling turn indices or by relaxing the floor from 5 positives to reach `t` = 3; no S1b, S1d or S1e result, rubric, asset or script edited; no commit to `main`. **Nothing here is described as confirmatory of the project's hypotheses** — it is a held-out replication of one exploratory finding, and that is what it is called throughout.

**Vocabulary:** STAGE0 §2 terms throughout; "the researcher" throughout. A whole-word check for the five banned terms over this report and every file this session wrote returns nothing.
