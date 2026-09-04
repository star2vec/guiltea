# Addendum — S4 run split across four pods, one target each (2026-09-04)

**Applies to:** `briefs/S4-experiment.md` as corrected at `d383a67`, plus Task 0c and Task 9. **Nothing about the
design changes.** Same seven cells, same four targets, same N = 8, same seeds 0–7, same judges, same steering, same
reading rules. This addendum changes **only which machine runs which target**, because cell A's measured rate of
33.6 minutes for 8 runs puts the single-pod total at about **15.7 hours of machine time**, which does not fit before
the 2026-09-05 write-up.

## The split

| pod | target | cells |
|---|---|---|
| 1 (the pod already running) | `burn-blister-pop` | all seven, in the brief's order |
| 2 | `snakebite-tourniquet` | all seven, in the brief's order |
| 3 | `insulin-skip-sick` | all seven, in the brief's order |
| 4 | `aspirin-child-flu` | all seven, in the brief's order |

Each pod is **self-contained for its own target**: it runs its target's same-domain controls itself, and it runs the
ten unrelated-question topic controls itself (they are target-independent, so the four pods produce four copies of the
same measurement on the same subject, revision and seeds; the merge keeps one copy and reports the agreement of the
others as a free reproducibility check).

**Machine time per pod:** seven cells at cell A's measured rate ≈ **3.9 h**, plus its controls. **Budget stop $3.00
per pod** on that pod's own ledger, so the four together stay under the original $11.50; the projected total for the
whole run is ≈ $2.40.

## What each new pod does, exactly

1. Task 0 verbatim from the brief: the three rig changes, then the **pre-hook norm check before any cell**, reported
   before any cell runs. Pod 1's check passed with a worst relative deviation of 4.5 × 10⁻⁵; a new pod that
   disagrees by more than 5 % **stops and reports**.
2. The two OOM fixes pod 1 already committed are on branch `s4-experiment` and must be picked up, not rediscovered:
   the token-budget batching in `rigcommon.py`/`run.py` and the readout that calls `model.model(...)` rather than
   `model(...)`. **Branch from `origin/s4-experiment`, not from `main`.**
3. The seven cells on its own target, in the brief's order, `--distance4 --controls`, Task 0c's re-ask fork included.
4. **No pod runs the mid-run tripwire.** The tripwire exists to protect the researcher's hours against a serial run;
   with the run split four ways the never-cut set completes on every target within about three hours. Each pod
   instead **stops when its target is done**, tars, and sends. The researcher's ledger question is asked once, by the
   hub, when the first two targets are complete.
5. **Task 9 (the renewed-pressure pilot) is pod 2's only**, and only after its seven cells are complete and its
   ledger is under $2.
6. Each pod pushes to its own branch — `s4-target2`, `s4-target3`, `s4-target4` — branched from
   `origin/s4-experiment`, and files its numbers as a section of `reports/S4-experiment.md` under a heading naming
   its target. The hub merges the four and writes the combined table with `scripts/rig/table.py` over the merged
   tree.

## What must be identical across pods, and is checked at merge
Subject revision, seeds 0–7, the arrow files, σ at L16 (2.920537 / 4·σ for `guilt_clean`), the judge models under
`--judges mini`, the escalation salt and the 1-in-8 sample rule, and the rubrics. The merge asserts all of it from the
run headers and refuses to build a combined table if any differs.

## What is NOT permitted to make it faster
Do not drop the distance-4 forks (they are the persistence measurement and S5's honest test). Do not shorten the
judged unrelated set below the 8 borrowed plus our 2 (STAGE0 §8, D-021). Do not lower N below 8 or change the seeds.
Do not drop a cell. If time still does not fit, the researcher chooses, and the cell order already ensures the
never-cut five complete before the two steering cells on every target.
