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
