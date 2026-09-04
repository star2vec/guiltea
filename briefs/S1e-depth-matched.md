# Brief — S1e: the early-warning question at fixed conversation depth (CPU only, no GPU, no API)

**Stage:** a second analysis stage on data already collected. **Machine:** the Mac, CPU only. **No generation, no
model load, no judge call, no cost.** **Status of every number: EXPLORATORY**, same as S1d.
**Why it exists.** S1d Task 7 found no axis predicts the break one turn early, and S1d Task 8 found the early-versus-
late contrast **not evaluable**, both for the same reason: the positives sit at a different conversation depth from
the negatives, and a random direction given the same layer search exploits depth alone (Task 8's matched floor reached
AUROC ≈ 0.99). **This brief removes depth as a variable by comparing runs at the same turn index**, which the stored
data supports without a single new forward pass.

**Context you receive:** this brief + STAGE0.md + PLAN.md + `reports/S1b-runs.md` + `reports/S1d-blame-target.md`
+ `scripts/s1d/` (reuse it; `proj.py`, `common.py` and the matched-floor code already exist) + `scripts/randctl.py`
+ `directions/{dirs_8B_base_sweep.pt, dirs_8B_s2_arrows.pt}` + `results/raw/s1b/` (`t4`, `t10`, `judge_calls`)
+ `results/raw/s1d/`. Nothing else.
**You execute exactly this brief and file one report.** If a contrast has too few members, report the counts and stop
that contrast; do not pool it away.

---

## The protocol, identical for every contrast below

- **Scores:** the projection at the turn's `answer` position, and separately at `into`, on `refusal`, `badmed`,
  `persona`, `persona_meandiff`, `guilt_clean`, `shame_clean`, `nn`, `received_act`, `received_self`.
- **Fixed depth:** every comparison is made **within one turn index `t`**. Never pool raw scores across `t`; compute
  per `t`, then report per `t` and the count-weighted mean across `t`.
- **Headline statistic, pre-specified:** the **L14–18 band mean** (D-024 primary band). Best-over-layers may be
  reported but only beside a **selection-matched floor**, i.e. each randctl seed 0–9 given the same best-over-layers
  search. This is the rule S1d adopted and it is the rule that decides every verdict sentence here.
- **Folds:** leave-one-target-out over the contributing targets. Report how many targets hold **both** classes; if
  fewer than 5 do, the contrast is reported as underpowered and no verdict sentence is written for it.
- **Both label sources**, as S1d did: the rig's stored `T_primary` and the merged D-019 labels (second judge final).
- **Minimum class size:** 10 per side at a given `t`. Below that, counts only.

## Task 1 — THE TRIGGER DEMO: among runs that are still refusing at turn `t`, which ones break later?

This is the early-warning question with depth held fixed, and it is the reason for the brief.

For each `t` where the counts allow (expect `t` = 2…6):
- **Positives:** `v1` chains whose turn `t` is graded `held` **and** which have a `committed` turn later.
- **Negatives:** `v1` chains whose turn `t` is graded `held` **and** which never commit within ten turns.
- Both classes carry the same label at the same depth, so **nothing separates them except what the state carries about
  what the run will do next.** Report AUROC per axis against the matched floor, per `t` and weighted-mean.

**Verdict sentence required:** whether any axis beats the matched floor at fixed depth, at which `t`, in which band,
and by how much. **If one does, say plainly that this is the first positive early-warning result in the project and
that S1d Task 7's negative was a depth artefact.** If none does, say that the negative survives depth matching, which
makes it a much stronger negative than Task 7's.

## Task 2 — depth-matched act detection: committed at `t` versus held at `t`

For each `t` with the counts: chains whose turn `t` is `committed` against chains whose turn `t` is `held`, same `t`.
This is S1c's act-detection question with conversation length removed, and it is the control that tells you whether
Task 1's separation (if any) is about the future or merely about the present. Same protocol, same verdict sentence.

**Read the two together and say so in one sentence:** if Task 2 separates and Task 1 does not, the axes read the act
but not its approach; if both separate, report whether Task 1's margin survives restricting to turns where Task 2's
margin is small.

## Task 3 — what remains of the early-versus-late question

S1d Task 8's verdict stands and is not re-litigated. State in two sentences what the tractable remnant is: an act at
turn 3 and an act at turn 10 cannot be compared without matching prefix length, which needs new forward passes; the
depth-matched substitute is Task 2 at each `t`, and the per-`t` table is the honest version of the question. Report the
per-`t` counts of committed acts so the reader can see where the data actually is.

## Task 4 — one figure

`writeup/figs/s1e_depth_matched.{png,pdf}`: Task 1's band-mean AUROC by turn index, one line per axis, the matched
random floor as a shaded band, the 0.5 line marked, class counts annotated per `t`. Machine-written by a committed
script, never hand-edited.

## Report (`reports/S1e-depth-matched.md`)
1. Exploratory status, class counts per `t` per contrast, target coverage, and the fact that no GPU or API was used.
2. Task 1 with its verdict sentence. 3. Task 2 with its verdict sentence and the joint reading. 4. Task 3's two
sentences and the per-`t` counts. 5. The figure. 6. Anything unworkable.

## Do not
- Do not generate text, load a model, call a judge, or touch a GPU. Nothing here costs money.
- Do not pool scores across turn indices; the whole point is that depth is held fixed.
- Do not report a best-over-layers number without a selection-matched floor beside it.
- Do not call any of this confirmatory; do not re-label S2's gate or S1d's verdicts.
- Vocabulary per STAGE0 §2; "the researcher".
