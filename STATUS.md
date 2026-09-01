# STATUS — the ship's log
### One screen. Updated by the researcher after every report. If this file is accurate, anyone can pick the project up cold.

## Hours ledger (researcher-maintained; sessions must ask, never estimate)
Spent: 0 / 20 · Reserve policy: 25% held back at the rig checkpoint (STAGE0 §7)

## Where we are
Stage: S0 complete; repo assembled. S1 fully planned (S1-plan.md rev.3, all decisions locked by the researcher 2026-09-01).
Briefs drafted: briefs/S3-feasibility.md, briefs/S1a-assets.md. S1b-runs awaits the S3 report (axes + measured throughput).
Adversarial review of the two briefs is out (review-request-S3-S1a.md).

## What we know (one line each, with source)
- Act-blame vs self-blame changes behavior at 1B with corrective content held constant — not instruction-following [checks-log/10, A]
- Purpose-built guilt/shame directions are distinct from refusal/persona/misalignment axes [checks-log/10, B]
- At 1B those directions are ~50% generic negative valence and mutually ~0.65 correlated; template-built probes overfit wording [checks-log/10, B]
- Larger organisms (7B–32B) available; CUDA port trivial [checks-log/10, C]
- Misalignment-axis readout leaks topic at 1B → matched-topic controls mandatory [checks-log/09]

## Open questions, ranked
1. Does any valence-independent blame arrow survive at 8B? (S2 gate)
2. Laptop feasibility: does 8B fit the RTX 2000 Ada, at what speed, with how much quantization noise? (S3 opening hour)
3. Does the 8B organism hold on enough targets to get ≥10 held? (S1b gate; S1a generates 40 for headroom)
4. Cliff or slide? (S1 break trajectory, STAGE0 §4.8)

## Blockers / errands (researcher)
- [ ] Confirm laptop GPU memory (8 vs 16 GB) — feeds the S3 feasibility rule
- [ ] EM alignment/coherence judge key — hard precondition for S1b/S4 spread scoring. (S1a needs frontier *generation* for Task 1 and the *rubric-judge* endpoint for Task 6 — both distinct from this S4 judge; see S1a brief preconditions)
- [ ] Decide handling of the Nanda context files (see DECISIONS.md D-011)

## Next action
Run briefs/S3-feasibility.md and briefs/S1a-assets.md (independent, parallelizable); S1b-runs drafted once S3 reports.
