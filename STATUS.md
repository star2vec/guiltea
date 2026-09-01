# STATUS — the ship's log
### One screen. Updated by the researcher after every report. If this file is accurate, anyone can pick the project up cold.

## Hours ledger (researcher-maintained; sessions must ask, never estimate)
Spent: 0 / 20 · Reserve policy: 25% held back at the rig checkpoint (STAGE0 §7)

## Where we are
Stage: S0 complete; repo assembled. S1 fully planned (S1-plan.md rev.3, all decisions locked by the researcher 2026-09-01).
Briefs drafted: briefs/S3-feasibility.md, briefs/S1a-assets.md. Adversarial review returned; rev.4 fixes applied (commit 3ecc1ef).
S3 Phase A complete — report filed 2026-09-02 (reports/S3-feasibility.md): contrast data acquired and verified, scripts/randctl.py written and self-checked. Phase B (Tasks 1, 2-extraction, 4, 5) waits on the RTX laptop; the Task 1 bf16 reference will likely need a cloud session.
S1a-assets running (worker session). S1b-runs is drafted only after S3 Phase B is complete.

## What we know (one line each, with source)
- Act-blame vs self-blame changes behavior at 1B with corrective content held constant — not instruction-following [checks-log/10, A]
- Purpose-built guilt/shame directions are distinct from refusal/persona/misalignment axes [checks-log/10, B]
- At 1B those directions are ~50% generic negative valence and mutually ~0.65 correlated; template-built probes overfit wording [checks-log/10, B]
- Larger organisms (7B–32B) available; CUDA port trivial [checks-log/10, C]
- Misalignment-axis readout leaks topic at 1B → matched-topic controls mandatory [checks-log/09]
- Turner badmed data is governed by the archive's tos.txt (no redistribution, terms inherit); the jsonl files stay out of git and are re-acquired per data/contrast-sets/SOURCE.md [reports/S3-feasibility §7]
- Check 5's "MIT stub" for the Turner repo is not reproducible (no LICENSE in its history); Check 7's "MIT" for the Arditi splits should read Apache-2.0 [reports/S3-feasibility §2a, §2b]

## Open questions, ranked
1. Does any valence-independent blame arrow survive at 8B? (S2 gate)
2. Laptop feasibility: does 8B fit the RTX 2000 Ada, at what speed, with how much quantization noise? (S3 Phase B measures fit and speed on the laptop; the quantization-noise reference needs a bf16 run on cloud before the laptop-vs-cloud branch can be read)
3. Does the 8B organism hold on enough targets to get ≥10 held? (S1b gate; S1a generates 40 for headroom)
4. Cliff or slide? (S1 break trajectory, STAGE0 §4.8)

## Blockers / errands (researcher)
- [ ] Confirm laptop GPU memory (8 vs 16 GB) — Phase B Task 1 reports it directly
- [ ] Cloud GPU access for the S3 Task 1 bf16 reference (needed if 4-bit is forced on the laptop)
- [ ] EM alignment/coherence judge key — hard precondition for S1b/S4 spread scoring. (S1a needs frontier *generation* for Task 1 and the *rubric-judge* endpoint for Task 6 — both distinct from this S4 judge; see S1a brief preconditions)
- [ ] Decide handling of the Nanda context files (see DECISIONS.md D-011)

## Next action
Commit S3 Phase A artifacts (badmed jsonl excluded), push, clone on the RTX laptop, run S3 Phase B (briefs/S3-feasibility.md Tasks 1, 2-extraction, 4, 5). S1a continues in parallel. S1b-runs drafted once Phase B completes.
