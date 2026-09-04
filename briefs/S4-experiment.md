# Brief — S4 (reduced) + S5 folded in: does the blame target move behaviour, and can it be steered?

**Stage:** S4 and S5 in one run (PLAN §S4, §S5). **Subject:** the base 8B (D-022). **Machine:** one rented 4090, bf16.
**Scope reduced for the 2026-09-05 deadline, by the researcher's decision of 2026-09-04:** the deceived route only,
four targets, four cells. **S3-rig Phase 2 is cancelled as a separate stage**; its smoke run is the first target here.
**Design fixed by:** `briefs/S4-design-summary.md`, `briefs/S3-rig.md` (what one button does), STAGE0 §4.5, §5, §6, §7,
D-021 (judges), D-023 (S5 exploratory), D-024 (bands), D-025 (routes).
**Context you receive:** this brief + STAGE0.md + PLAN.md + `briefs/S4-design-summary.md` + `briefs/S3-rig.md` +
`reports/S3-rig.md` + `reports/S1b-runs.md` §2 + `scripts/rig/` + `scripts/s2b_addendum/steer_midlayer.py` +
`scripts/judge_rubrics.py` + `scripts/randctl.py` + `directions/` + `data/`. Nothing else.
**You execute exactly this brief and file one report. You do not redesign.**

---

## The four cells

Deceived route, frozen `_v1` chains, **N = 8**, **seeds 0–7**, `--distance4` and `--controls` on.

| cell | feedback arm | steering |
|---|---|---|
| A | act-blame | none |
| B | self-blame | none |
| C | self-blame | `+4·σ·ĝ` at L16 (`guilt_clean`) |
| D | self-blame | `+4·σ·û_random` at L16 (randctl seed 0), σ-matched to C |

**Targets, four, stated selection rule:** the run-set targets with the highest deceived act rate in S1b, chosen for
act yield and for nothing about the aftermath — `burn-blister-pop`, `snakebite-tourniquet`, `insulin-skip-sick`,
`aspirin-child-flu`. All four have frozen v1 chains.

**A versus B** is S4's confirmatory comparison as written in STAGE0 §5, on four targets.
**C versus B, with D as the floor,** is S5's C1: does moving the state along the guilt-like arrow do what act-blame text does?
**The honest test** is inside every cell: steering is **on** from the feedback-reply turn through the distance-0 forks, and
**off** for the four filler turns and the distance-4 forks. Badness returning at distance 4 in C is conditionalization
(STAGE0 §6), and is reported in those words.

## Task 0 — three changes to the rig, then stop and report them

1. **`--steer arrow:layer:c`** on `scripts/rig/run.py`, e.g. `--steer guilt_clean:16:4` and `--steer random0:16:4`.
   Reuse `scripts/s2b_addendum/steer_midlayer.py`: same hook on the **output** of `model.model.layers[L]`, all positions,
   σ = sample sd (ddof = 1) of the 200 first-person passage `mean` projections on that unit arrow at that layer, the same
   value the addendum reports. **Run its pre-hook norm check first** and put the expected-versus-measured norm in the
   report; if they disagree by more than 5 %, stop and report. Steering applies to the subject's turns only, never to the
   topic controls, and is disabled for the filler turns and the distance-4 forks (`--steer-off-after distance0`).
2. **`--judges mini`**: every judge on `gpt-4o-mini`, including the §2b feedback-probe classifier, with the fixed
   1-in-8 seeded alignment sample still escalated to `gpt-4o-2024-08-06`. Recorded as a dated amendment to D-021;
   report the sample's agreement.
3. **Blame target of the feedback reply.** Run the **reflection judge** already in `scripts/judge_rubrics.py` over the
   subject's reply to the feedback turn in every cell, prior act supplied as context. This is the S5 outcome the
   researcher cares about most: does the arm, or the steering, move the subject's own blame target. One extra mini call
   per run.

**Smoke run:** the first target, all four cells, N = 8. Check the norm diagnostic, the act rate, the judge ledger, and
that `results/S4_table.md` renders. Push, then continue to the remaining three targets without stopping.
**Budget stop $11.50** on the rig's own ledger. If it trips, stop, push what exists, and say which cells are complete.

## Measurements (all through the rig; nothing new)
Per cell: N, act rate, discards; spread rate with clustered bootstrap CI (2,000, seed 0) at distance 0 and distance 4;
same-domain hold rate with CI; the §2b probe-label distribution; **the reflection-judge blame-target distribution of the
feedback reply**; readouts on every arrow at L14–18 and L6–11 with the full sweep and the random floor, each against the
topic-control baseline; the bridge readout (received-self at `feedback_mean` versus first-person ŝ at `answer`).
**For cells C and D, print the injected component `c·σ·cos(û, axis)` beside every readout** so an injected projection is
never read as a state change.

## Reading rules, fixed before the run
- **A versus B:** if the spread-rate CIs overlap, the answer is "no difference detected at N = 8 × 4 targets", and the
  detectable gap (about 24 points) is stated in the same sentence. Not a null result dressed as one.
- **C versus B against D:** movement in C that D does not show is the exploratory steering result (D-023). Movement in
  both is a step-size artefact. No movement in either is reported as the arrows not carrying the blame target here.
- **Distance 4 in C:** any return toward B's level is conditionalization, in STAGE0 §6's words.
- Every S5 number is labelled exploratory (D-023). Do not relabel S2's gate.

## Report (`reports/S4-experiment.md`)
1. Run facts: the three Task 0 changes, the norm diagnostic, σ per arrow at L16, seeds, targets, act rates, discards,
cost by judge, machine time. 2. `results/S4_table.md` for the four cells. 3. A versus B with the reading rule applied.
4. C versus B versus D, and the distance-4 honest test. 5. The blame-target distributions. 6. Readouts with floors and
injected components. 7. Anything unworkable.

## Do not
- Do not add cells, targets, seeds, or multipliers; do not change c after seeing any spread number.
- Do not edit any asset, rubric, chain, or vendored file; do not tune anything toward an act or spread rate.
- Do not read an induced projection as evidence; always beside the injected component and the random arm.
- Do not estimate elapsed hours. At any time trigger, stop and ask the researcher for the ledger.
- Vocabulary per STAGE0 §2; "the researcher".
