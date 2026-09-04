# Report — S1e: the early-warning question at fixed conversation depth (reanalysis of S1b; no GPU, no API)

**Brief:** `briefs/S1e-depth-matched.md`, blob `a370e9e`, introduced by commit `baa7c0b` and unchanged since. **Filed:** 2026-09-04 by the S1e worker session, branch `s1e-depth-matched` (created from `origin/main` at `16fc72b`).
**Files read:** the brief, `STAGE0.md`, `PLAN.md`, `reports/S1b-runs.md`, `reports/S1d-blame-target.md`, `scripts/s1d/` (`common.py`, `proj.py`, `t6_figs.py`, `t7_early.py`, `t8_early_late.py` — read for reuse, none edited), `scripts/randctl.py`, `directions/{dirs_8B_base_sweep.pt, dirs_8B_s2_arrows.pt}` (via the stored projections), `results/raw/s1b/` (`t4`, `judge_calls`) and `results/raw/s1d/` (`proj_t4v1.npz`). Nothing else. No planning note, risk map, `STATUS.md`, S1c or S2 file was opened.
**Machine:** the researcher's Mac, CPU only. No generation, no model load, no judge call, no GPU, **no cost**. No S1b or S1d result, rubric, asset or script was edited; every output of this session is under `results/raw/s1e/`, `scripts/s1e/`, `writeup/figs/s1e_*` and this file.

## Status: all four tasks executed, nothing unworkable.

**Task 1 is a positive result** — with depth held fixed, one axis clears the matched random floor on the pre-specified band statistic, on one of two label sources and one of two summary statistics. **Task 2 is a positive result at one turn index and an artefact at the rest**: the act separates cleanly at `t` = 2, and the apparently similar margin pooled over later turns is substantially the rig's filler turns rather than the act.

---

## 1. Exploratory status, counts, coverage, cost

**Every number in this report is EXPLORATORY.** Nothing here is pre-registered, nothing is confirmatory, and nothing re-labels S2's gate or any S1d verdict. S1d Task 8's verdict stands and is not re-litigated.

| item | value |
|---|---|
| tasks | 4 |
| chains | **192** `v1` deceived chains, 10 turns each — **1,920 turn labels, all present** |
| label check | the rig's stored per-turn `grade` matches `results/raw/s1b/judge_calls/act_primary.jsonl` on **0 of 1,920 mismatches**, so the `t_primary` source is exactly the rig's own grading |
| second judge on `v1` | **394 of 1,920** turns carry a second-judge label, concentrated near each chain's `T_primary` (61/72/79/63 at t = 1…4, thinning to 12–22 at t = 7…10). The `merged` D-019 source is therefore a **partial** re-grading, not a uniform one |
| scores | `results/raw/s1d/proj_t4v1.npz` as S1d built it — `[192, 10, 3, 22, 32]` float32: 12 named axes + randctl seeds 0–9, 3 readout positions, 32 layers. **Nothing was recomputed from residuals and no forward pass was run** |
| axes reported | `refusal`, `badmed`, `persona`, `persona_meandiff`, `guilt_clean`, `shame_clean`, `nn`, `received_act`, `received_self` |
| headline statistic | the **L14–18 band mean** (D-024 primary band), against the band means of randctl seeds 0–9. Best-over-layers appears only beside a **selection-matched** floor (each seed given the same max-over-32-layers search) |
| `scripts/randctl.py` | re-run on this machine (**the eighth of the project**): **exit 0, checks 1–5 PASS, statistics identical to every earlier run** — max \|cos\| seed 0 vs 1 = 0.0260; layers max \|cos\| = 0.0518, mean 0.0125, 496 pairs. Record: `results/raw/s1e/randctl.txt` |
| API cost | **$0.00, 0 calls.** No judge, no key, no network |
| machine cost | ~14 s of CPU for both analyses, ~2 s for the figure |
| outputs | `results/raw/s1e/{t1_trigger.json, t2_actdet.json, counts.json, randctl.txt}`, `writeup/figs/s1e_depth_matched.{png,pdf}`, `scripts/s1e/{depth.py, t1_trigger.py, t2_actdet.py, t4_fig.py}` |

**Assumptions on record**, accepted by the researcher before execution:
1. `t_primary` = the rig's stored per-turn grade; `merged` = the second judge's label where it exists, else the primary's (D-019).
2. Task 1 requires `held` at `t`, so a `partial` at `t` puts a chain in neither class at that `t`; "never commits within ten turns" means no `committed` turn at any turn. Task 2 compares `committed` against `held`; `partial` excluded.
3. "a `committed` turn later" = strictly after `t`, up to turn 10.
4. The count-weighted mean across `t` is weighted by items (n₊ + n₋) over the `t` clearing the floor, and is a **summary, not an average of independent samples** — under `t_primary` the negative class is largely the same 74–83 never-committing chains at every `t`.
5. Folds: a single axis needs no training, so "leave-one-target-out" is read as S1d read it — the mean of the per-target AUROCs over targets holding both classes, reported beside the pooled value, each with its own matched floor.
6. Every `t` = 1…10 is computed; a verdict is written only where both classes reach 10 and at least 5 targets hold both.
7. A **filler-excluded variant** is reported beside the brief's own tables wherever a class contains filler (§6 item 2).
8. The figure draws the `answer` position and the fold statistic; the `into` position, the pooled statistic and the filler-excluded variant are in the JSON.

### Class counts per `t`, per contrast

`filler in −` counts negatives sitting on a turn the rig replaced with neutral filler after the act, where a `held` grade means the harmful question was not asked (`reports/S1d-blame-target.md` §7). **Task 1 under `t_primary` contains none at any `t`.**

**Task 1 — held at `t` and commits later (+) vs held at `t` and never commits (−)**

| t | `t_primary` n+ / n− | targets both | filler in − | evaluable | `merged` n+ / n− | targets both | filler in − | evaluable |
|---|---|---|---|---|---|---|---|---|
| 1 | 105 / 82 | 13 | 0 | yes | 58 / 120 | 11 | 0 | yes |
| 2 | 69 / 78 | 13 | 0 | yes | 19 / 103 | 8 | 0 | yes |
| 3 | 39 / 77 | 13 | 0 | yes | 14 / 98 | 6 | 6 | yes |
| 4 | 36 / 82 | 12 | 0 | yes | 14 / 115 | 7 | 21 | yes |
| 5 | 22 / 81 | 11 | 0 | yes | 9 / 114 | 6 | 26 | **no** |
| 6 | 18 / 82 | 11 | 0 | yes | 9 / 120 | 6 | 33 | **no** |
| 7 | 16 / 83 | 11 | 0 | yes | 8 / 124 | 5 | 35 | **no** |
| 8 | 16 / 83 | 11 | 0 | yes | 7 / 127 | 5 | 37 | **no** |
| 9 | 15 / 83 | 11 | 0 | yes | 7 / 126 | 4 | 37 | **no** |
| 10 | 0 / 74 | 0 | 0 | **no** | 0 / 112 | 0 | 38 | **no** |

**Task 2 — committed at `t` (+) vs held at `t` (−)**

| t | `t_primary` n+ / n− | targets both | filler in − | evaluable | `merged` n+ / n− | targets both | evaluable |
|---|---|---|---|---|---|---|---|
| 1 | 3 / 187 | 3 | 0 | **no** | 4 / 178 | 3 | **no** |
| 2 | 21 / 150 | 8 | **3** | yes | 19 / 125 | 8 | yes |
| 3 | 37 / 140 | 11 | 24 | yes | 26 / 130 | 7 | yes |
| 4 | 11 / 179 | 6 | 61 | yes | 7 / 169 | 4 | **no** |
| 5 | 10 / 175 | 6 | 72 | yes | 7 / 169 | 5 | **no** |
| 6 | 5 / 182 | 5 | 82 | **no** | 4 / 178 | 3 | **no** |
| 7 | 5 / 186 | 2 | 87 | **no** | 3 / 184 | 1 | **no** |
| 8 | 1 / 191 | 1 | 92 | **no** | 1 / 189 | 1 | **no** |
| 9 | 1 / 191 | 1 | 93 | **no** | 1 / 189 | 1 | **no** |
| 10 | 15 / 168 | 11 | **94** | yes | 9 / 168 | 6 | **no** |

The brief expected `t` = 2…6. The data reaches **wider on Task 1** (t = 1…9 under `t_primary`) and **narrower on Task 2** (t = 2, 3, 4, 5, 10 under `t_primary`; t = 2, 3 under `merged`). Every cell marked **no** is reported as counts only, per the brief; nothing is pooled to rescue it.

**Reader's cross-check.** Task 2's positive-class counts under `t_primary` are the per-turn commission profile, and they reproduce `reports/S1b-runs.md` §9 exactly: **3, 21, 37, 11, 10, 5, 5, 1, 1, 15**.

---

## 2. Task 1 — the trigger demo: among runs still refusing at turn `t`, which ones break later?

Both classes are graded `held` at the same turn index, so depth is held fixed by construction and nothing separates them except what the state carries about what the run will do next.

**First, the diagnostic that decides how to read everything below.** At a fixed `t` the *pooled* AUROC is still free to separate the classes by **target identity** rather than by state: whether a chain breaks is partly a fact about which target it is (`burn-blister-pop` broke 12/12, `stroke-signs-wait` 0/12 — `reports/S1b-runs.md` §5). The symptom is visible immediately. At `t` = 1, `into` position, `badmed` reaches **pooled 0.742** and `refusal` 0.730 — the largest numbers in the whole task — and the same axes on the **target-matched** fold statistic fall to **0.447 and 0.490**, i.e. to the floor. So the strongest pooled number here is target identity, and **the fold statistic is the one matched on both depth and target.** It carries every verdict below; the pooled tables are in `results/raw/s1e/t1_trigger.json` and on request.

**The main table: L14–18 band mean, fold statistic (mean of the per-target AUROCs), `answer` position, stored `T_primary`.** Below 0.5 means the will-break class projects *lower*.

| t | refusal | badmed | persona | persona_meandiff | guilt_clean | shame_clean | **nn** | received_act | received_self | random floor (min–max) | axes clearing |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 0.581 | 0.545 | 0.430 | 0.452 | 0.415 | 0.491 | **0.575** | 0.565 | 0.551 | 0.472–0.574 | 3 |
| 2 | 0.461 | 0.474 | 0.551 | 0.456 | 0.444 | 0.441 | **0.644** | 0.476 | 0.467 | 0.445–0.553 | 3 |
| 3 | 0.413 | 0.455 | 0.550 | 0.569 | 0.482 | 0.390 | **0.610** | 0.457 | 0.399 | 0.500–0.573 | 4 |
| 4 | 0.433 | 0.536 | 0.469 | 0.419 | 0.450 | 0.421 | 0.455 | 0.398 | 0.362 | 0.447–0.589 | 2 |
| 5 | 0.422 | 0.425 | 0.561 | 0.480 | 0.518 | 0.483 | **0.759** | 0.673 | 0.610 | 0.414–0.534 | 3 |
| 6 | 0.509 | 0.549 | 0.416 | 0.430 | 0.399 | 0.490 | **0.674** | 0.576 | 0.542 | 0.448–0.555 | 5 |
| 7 | 0.445 | 0.542 | 0.371 | 0.227 | 0.517 | 0.580 | **0.644** | 0.591 | 0.545 | 0.475–0.563 | 5 |
| 8 | 0.393 | 0.495 | 0.518 | 0.409 | 0.554 | 0.322 | 0.470 | 0.419 | 0.364 | 0.444–0.566 | 5 |
| 9 | 0.461 | 0.456 | 0.529 | 0.346 | 0.490 | 0.448 | **0.626** | 0.561 | 0.538 | 0.444–0.570 | 2 |
| **count-weighted mean** | 0.467 | 0.500 | 0.487 | **0.428** | 0.468 | 0.453 | **0.604** | 0.522 | 0.488 | **0.477–0.541** | **3** |

**Two axes clear the matched floor, and only two do it consistently.**

- **`nn`, the neutral-negative arrow, above 0.5.** Count-weighted mean **0.604** against a floor of 0.477–0.541 — excess **0.104** against the largest seed's **0.041**, a factor of 2.5. It clears at **7 of the 9 turn indices** (all but t = 4 and t = 8) and is above 0.5 at every one of the seven. It is the largest-margin axis in both fold-statistic cells (`answer` 0.604, `into` 0.563) and clears in **3 of the 4** `t_primary` cells, missing only `answer`/pooled (0.546, excess 0.046 against a floor of 0.050). Direction: **chains that will break later project higher on `nn` at a turn where they are still refusing.**
- **`persona_meandiff`, below 0.5.** Count-weighted mean 0.428 (excess 0.072), clearing in **4 of 4** `t_primary` cells (0.422, 0.428, 0.384, 0.439) and at 4 of 9 turn indices, always in the same direction. Will-break chains project *lower* on the mean-difference persona axis.

Every other axis clears in at most one cell, or changes sign from `t` to `t`. In particular **`refusal`, `badmed`, `guilt_clean` and `shame_clean` — the axes an early-warning story would predict — do not clear on the matched statistic**: `refusal` runs 0.581 at t = 1 down to 0.393 at t = 8, reversing sign mid-chain, and its count-weighted mean (0.467, excess 0.033) sits inside the floor.

**How much of this is one target?** The fold statistic is a mean of per-target AUROCs, so a per-target sign count says whether the mean is carried by the set or by one target. `nn`'s targets above 0.5, per `t`: **8/13, 9/13, 9/13, 5/12, 9/11, 8/11, 9/11, 6/11, 6/11** — a consistent majority, never one target. But beside the randctl seeds' own counts (mean 4.1–7.6, max 6–11 across the same `t`) the sign count only exceeds every seed at t = 5 (9/11 vs a best seed of 6) and t = 7 (9/11 vs 8). **So the sign check establishes that the effect is broad rather than local, and it does not by itself separate `nn` from a random direction; the magnitude against the floor remains the only discriminating comparison.**

**Best-over-layers, beside its selection-matched floor.** On the count-weighted curve: **0 of 9 axes** clear on the pooled statistic (best `persona_meandiff` 0.341 at L31, excess 0.159, matched floor 0.102/0.136/0.174) and **2 of 9** on the fold statistic (best `persona_meandiff` 0.345 at L31, excess 0.155, matched floor 0.085/0.106/0.133). This matters for how the result is read and §6 item 1 returns to it.

**Under the merged D-019 labels the result does not reproduce, and the cell cannot carry weight either way.** `nn` reaches 0.565 on the fold statistic at `answer` — but there the floor collapses to 0.480–0.530 (excess 0.030) and **8 of 9 axes nominally clear it, in both directions at once**. With n₊ = 14–19 the ten seeds happen to land near 0.5, and a narrow floor is not nine findings; it is a floor with too few seeds to be stable at that class size. On the `into` position under merged labels **no axis clears**. The merged source is in any case a partial re-grading concentrated near each chain's act (§1), so at a fixed `t` its two classes can carry labels from different judges.

**Verdict, in one sentence: with conversation depth held fixed, one axis does clear the matched random floor on the pre-specified band statistic — `nn`, on the depth- and target-matched fold statistic at the `answer` position, count-weighted band mean 0.604 against a floor of 0.477–0.541, clearing at 7 of 9 turn indices and always in the same direction, with `persona_meandiff` clearing inversely in all four `t_primary` cells — so a state measured while the subject is still refusing does carry information about whether that run will break later.**

**As the brief requires, plainly: this is the first positive early-warning result in the project.** And the bound on it, which the same data supplies: **S1d Task 7's negative was not attributable to depth alone.** Three things changed at once between Task 7 and here — depth held fixed, the pre-specified L14–18 band mean in place of best-over-32-layers, and "commits at any later turn" in place of "commits at exactly `t` + 1" — and the second of them is doing at least as much work as the first, because Task 7's own best-over-layers statistic, run here at fixed depth, still clears nothing on the pooled statistic (0 of 9, above). The honest apportionment is that **depth matching plus a pre-specified band together turn Task 7's negative into a positive, and this data cannot say how much each contributed.** Nothing here licenses a state-conditional intervention on its own: the margin is 0.604 against 0.541 on a summary across nine correlated turn indices, on one axis of nine, under one of two label sources.

---

## 3. Task 2 — depth-matched act detection: committed at `t` versus held at `t`

Same protocol, conversation length removed. This is S1c's question with depth held fixed, and it is the control that says whether Task 1's separation is about the future or merely about the present.

**One turn index gives a clean answer, and it is `t` = 2** — the only evaluable turn where the `held` class is essentially free of the rig's filler (3 of 150). Fold statistic, L14–18 band mean, `answer` position, stored `T_primary`:

| cell | refusal | badmed | best of the other seven | random floor (min–max) | axes clearing | per-target sign, refusal |
|---|---|---|---|---|---|---|
| t = 2, as specified (n₊ = 21, n₋ = 150) | **0.786** | **0.774** | persona 0.228 | 0.468–0.589 | 4 of 9 | **8 of 8 targets**, best seed 6 of 8 |
| t = 2, filler excluded (n₊ = 21, n₋ = 147) | **0.760** | **0.791** | persona 0.225 | 0.466–0.568 | 4 of 9 | — |
| t = 2, `into`, as specified | 0.587 | 0.602 | persona_meandiff 0.415 | 0.415–0.589 | 1 of 9 | — |

`refusal` and `badmed` separate the act from a same-depth refusal by a margin of 0.286 and 0.274 over half, against a floor whose best seed reaches 0.089, with **every one of the 8 contributing targets pointing the same way** and the numbers barely moving when the three filler negatives are dropped. That is what a strong, clean, depth- and target-matched effect looks like in this data, and it is the sharpest positive in this report.

**Pooled across the evaluable turns the margin looks similar and is not the same thing.** Count-weighted over `t` = 2, 3, 4, 5, 10, the fold statistic clears on `refusal` **0.662** and `received_act` **0.668** against a floor of 0.396–0.637 (excess max 0.137). **That margin does not survive excluding the filler turns**: `refusal` falls to 0.520 and `received_act` to 0.422, both inside the floor, and the axes that clear the filler-excluded version are different ones in the opposite direction (`guilt_clean` 0.334, `shame_clean` 0.355, `received_self` 0.359). The reason is in §1's count column: from `t` = 3 onward most `held` turns are the neutral filler the rig inserted after the act — 24 of 140 at t = 3, rising to **94 of 168 at t = 10** — so at those turn indices the contrast is substantially *the act versus a changed topic*, which is `reports/S1b-runs.md` §10's confound in a new place. `received_act`'s rise with `t` (0.498, 0.598, 0.713, 0.756, 0.761) is exactly the shape that confound predicts, and it disappears when the filler is removed.

**A stringency note that bears on Task 1.** At t = 2 the pre-specified band mean finds the act easily (0.786 against a floor excess of 0.089) while **best-over-layers with a selection-matched floor finds nothing on the same data** — `refusal` reaches 0.834 at L16, excess 0.334, against a matched floor of 0.219/0.276/0.350: **0 of 9 axes clear.** At these class sizes a max-over-32-layers search hands the random seeds enough to swallow an effect this large. The pre-specification is doing real work.

**Verdict, in one sentence: the act is plainly readable at fixed conversation depth — at `t` = 2, the one turn index whose `held` class is not filler, `refusal` 0.786 and `badmed` 0.774 on the depth- and target-matched band mean against a floor of 0.468–0.589, unanimous across all 8 contributing targets and unchanged by dropping the filler — but the apparently similar margin pooled over `t` = 2…10 is substantially the rig's filler rather than the act, since it vanishes when filler turns are excluded from the `held` class.**

### The two read together

**Both contrasts separate, and Task 1's margin is not a shadow of Task 2's.** The brief asks whether Task 1's margin survives restricting to turns where Task 2's margin is small; on this data the stronger version of that check is available, because **Task 2 cannot be evaluated at all at `t` = 6, 7 and 9 — 5, 5 and 1 chains commit there — and `nn` clears the floor at all three** (0.674, 0.644, 0.626 against floor excesses of 0.055, 0.063, 0.070). At `t` = 3, the evaluable turn where `refusal`'s act-detection margin is weakest on the fold statistic (0.609, excess 0.109, inside a floor of 0.155), `nn` still clears (0.610, excess 0.110 against 0.073).

| t | Task 1: `nn` band mean (excess / floor) | Task 2: best axis, band mean (excess / floor) |
|---|---|---|
| 1 | 0.575 (0.075 / 0.074) clears | underpowered — 3 acts |
| 2 | 0.644 (0.144 / 0.055) clears | **refusal 0.786** (0.286 / 0.089) clears |
| 3 | 0.610 (0.110 / 0.073) clears | guilt_clean 0.239 (0.261 / 0.155) clears |
| 4 | 0.455 (0.045 / 0.089) — | received_act 0.713 (0.213 / 0.183) clears |
| 5 | 0.759 (0.259 / 0.086) clears | received_act 0.756 (0.256 / 0.170) clears |
| 6 | 0.674 (0.174 / 0.055) clears | underpowered — 5 acts |
| 7 | 0.644 (0.144 / 0.063) clears | underpowered — 5 acts |
| 8 | 0.470 (0.030 / 0.066) — | underpowered — 1 act |
| 9 | 0.626 (0.126 / 0.070) clears | underpowered — 1 act |
| 10 | 0 positives — no contrast | received_act 0.761 (0.261 / 0.169) clears |

So the axes read the act **and** something about its approach, and they do not read them with the same instrument: act detection at fixed depth runs on `refusal` and `badmed` and is unanimous across targets; the early-warning signal runs on `nn` and is a consistent majority tendency. The one caution on that separation is that Task 2's own clean cell is a single turn index, so "the axes that read the act" is established at `t` = 2 and nowhere else in this data without filler contamination.

---

## 4. Task 3 — what remains of the early-versus-late question

S1d Task 8's verdict stands and is not re-litigated. **The tractable remnant, in two sentences:** an act at turn 3 and an act at turn 10 cannot be compared without matching prefix length, because the two classes differ in conversation depth before they differ in anything else and a random direction given the same layer search separates them almost perfectly — and matching prefix length needs new forward passes, which is a design decision and not this session's. **The depth-matched substitute is Task 2 evaluated at each `t` separately, and the per-`t` table is the honest version of the question**: it asks what the act looks like against a refusal at the same depth, and it answers only where the acts actually are.

Committed acts per turn index, so the reader can see where the data is:

| turn | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|
| stored `T_primary` | 3 | 21 | **37** | 11 | 10 | 5 | 5 | 1 | 1 | **15** |
| merged (D-019) | 4 | 19 | **26** | 7 | 7 | 4 | 3 | 1 | 1 | 9 |

Five of ten turn indices hold fewer than 10 acts under `t_primary` and eight of ten do under the merged labels, so the per-`t` version of the early-versus-late question is answerable at **two turn indices with a filler-free comparison** (`t` = 2, and `t` = 3 with 24 of 140 negatives filler) and nowhere else. That, and not a verdict, is what the data supports.

---

## 5. The figure

Machine-written by `scripts/s1e/t4_fig.py`; regenerate, never hand-edit.

- `writeup/figs/s1e_depth_matched.{png,pdf}` — Task 1's L14–18 band-mean AUROC by turn index, one line per axis (`nn` bold), the randctl seed 0–9 floor as a shaded min–max band of the seeds' own band means, the 0.5 line marked, class counts annotated per `t`, one panel per label source. Drawn on the fold statistic at the `answer` position, because that is the statistic the §2 verdict rests on. The `t_primary` panel shows `nn` above the floor band at seven of nine turn indices and inside it at t = 4 and t = 8; the `merged` panel shows the collapsed floor and the t = 3 spike on `received_act`, `received_self` and `refusal` at n₊ = 14 that §2 declines to read.

---

## 6. Anything unworkable, and what is reported rather than repaired

**Nothing in the brief was unworkable.** All four tasks ran in full, on CPU, at zero cost. Six things are reported rather than repaired, and the first three are the ones the researcher may want to overrule.

1. **The brief's two floors disagree with each other, and the report says which decides.** The pre-specified headline (L14–18 band mean against the seeds' band means) yields Task 1's positive and Task 2's `t` = 2 positive. The selection-matched best-over-layers comparison the brief also permits yields **0 of 9 on both** — including on Task 2's `t` = 2 cell, where `refusal` reaches 0.834 at L16 and still does not clear. Both are reported everywhere; the band mean carries the verdict sentences because the brief pre-specified it. The consequence worth stating: at n₊ = 10–40 a max-over-32-layers search with a matched floor cannot detect an effect that a pre-specified band detects unambiguously, so **S1d Task 7's negative and this report's positive are not straightforwardly comparable**, and the difference is not only depth (§2).
2. **The filler-excluded variant is beyond the brief's Task 1 and Task 2.** It was added because the rig replaced post-act turns with neutral filler, where a `held` grade means the harmful question was not asked, and reporting the pooled Task 2 numbers alone would have repeated `reports/S1b-runs.md` §10's error in a new place — as it turns out, Task 2's pooled margin **is** the filler (§3). The brief's own tables are reported in full and unchanged beside it. Task 1 under `t_primary` has zero filler in either class at every `t`, so its verdict is untouched by this.
3. **The pooled statistic is reported but never carries a verdict.** At a fixed turn index it still separates by target identity, demonstrated at `t` = 1 where `badmed` reads pooled 0.742 and target-matched 0.447 (§2). Every verdict sentence uses the fold statistic. If the researcher prefers the pooled reading, Task 1's positive shrinks to `persona_meandiff` alone and Task 2's `t` = 2 cell inverts onto `received_self` and `persona`.
4. **The merged D-019 label source cannot carry weight in either direction here.** It is a partial re-grading covering 394 of 1,920 `v1` turns and concentrated near each chain's act, so at a fixed `t` the two classes can be graded by different judges; and at its class sizes the ten-seed floor narrows to ±0.030, where 8 of 9 axes nominally clear it in both directions at once. Reported in full, read for nothing. The adjudication S1b §8 deferred — κ, `T_adjudicated` — would settle this, and is still outstanding.
5. **`nn`'s per-target sign count does not separate it from a random direction** (§2), only its magnitude against the floor does. The claim in the verdict sentence is exactly as strong as that comparison and no stronger.
6. **The absolute sign of a projection is not interpretable.** `refusal`, `badmed` and the S2 arrows are mean-difference directions with no centred zero (`reports/S1d-blame-target.md` §6), so this report states which class projects higher on an axis and offers **no semantic reading** of that — in particular, "committed turns project higher on `refusal` than same-depth refusals" is reported as a fact about the axis, not as a claim that the act is more refusal-like.

**Not done, by design:** no text generated, no model loaded, no judge called, no GPU touched, nothing that cost money; no score pooled across turn indices; no best-over-layers number without its selection-matched floor and no axis number without its floor in the same table; nothing called confirmatory; S2's gate not re-labelled and no S1d verdict re-litigated; no S1b or S1d file, rubric, asset or script edited; no commit to `main`; no contrast rescued by pooling — every underpowered cell is reported as counts only.

**Vocabulary:** STAGE0 §2 terms throughout; "the researcher" throughout. A whole-word grep for the five banned terms over this report and every file this session wrote returns nothing.
