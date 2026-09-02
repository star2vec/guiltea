# STATUS — the ship's log
### One screen. Updated by the researcher after every report. If this file is accurate, anyone can pick the project up cold.

## Hours ledger (researcher-maintained; sessions must ask, never estimate)
Spent: 0 / 20 · Reserve policy: 25% held back at the rig checkpoint (STAGE0 §7)

## Where we are
Stage: S0 complete; repo assembled. S1 fully planned (S1-plan.md rev.3, all decisions locked by the researcher 2026-09-01).
Briefs: briefs/S3-feasibility.md, briefs/S1a-assets.md (rev.4 after adversarial review, commit 3ecc1ef). Both first-wave reports filed 2026-09-02.
S3 Phase A complete (contrast data verified, scripts/randctl.py). **Phase B blocked:** the RTX laptop cannot resolve DNS (GitHub, Hugging Face, PyPI all unreachable), so no worker session can run there yet; see blockers.
S1a complete on Tasks 1–5; Task 6 fixtures authored (200), scoring deferred (rubric-judge endpoint not live). Researcher hand-read of the 40 situations done 2026-09-02: **37 kept, 3 rejected** (grapefruit-simvastatin, anaphylaxis-benadryl-only, licorice-root-bp); regeneration addendum issued (briefs/S1a-assets-addendum.md). Execution defaults confirmed (D-013–D-015).
S1b-runs is drafted only after S3 Phase B is complete.

## What we know (one line each, with source)
- Act-blame vs self-blame changes behavior at 1B with corrective content held constant — not instruction-following [checks-log/10, A]
- Purpose-built guilt/shame directions are distinct from refusal/persona/misalignment axes [checks-log/10, B]
- At 1B those directions are ~50% generic negative valence and mutually ~0.65 correlated; template-built probes overfit wording [checks-log/10, B]
- Larger organisms (7B–32B) available; CUDA port trivial [checks-log/10, C]
- Misalignment-axis readout leaks topic at 1B → matched-topic controls mandatory [checks-log/09]
- Turner badmed data is governed by the archive's tos.txt (no redistribution, terms inherit); the jsonl files stay out of git and are re-acquired per data/contrast-sets/SOURCE.md [reports/S3-feasibility §7]
- Check 5's "MIT stub" for the Turner repo is not reproducible (no LICENSE in its history); Check 7's "MIT" for the Arditi splits should read Apache-2.0 [reports/S3-feasibility §2a, §2b]
- 16 of the 37 kept S1a situations end in a reassurance tag, none in dosing/infant; accepted as a known confound, hold rate to be split by tag in S1b [D-014]

## Open questions, ranked
1. Does any valence-independent blame arrow survive at 8B? (S2 gate)
2. Laptop feasibility: does 8B fit the RTX 2000 Ada, at what speed, with how much quantization noise? (S3 Phase B measures fit and speed on the laptop; the quantization-noise reference needs a bf16 run on cloud before the laptop-vs-cloud branch can be read)
3. Does the 8B organism hold on enough targets to get ≥10 held? (S1b gate; 37 kept + 3 regenerations give headroom)
4. Cliff or slide? (S1 break trajectory, STAGE0 §4.8)
5. Do reassurance-tagged and plain situations hold at different rates? (S1b hold screen, split reported per D-014)

## Blockers / errands (researcher)
- [ ] **RTX laptop network**: DNS broken for GitHub, Hugging Face, PyPI. Options: client-side DNS override / encrypted DNS / phone hotspot; else wait; cloud-first reorder is the fallback (researcher decides, costs money)
- [ ] Confirm laptop GPU memory (8 vs 16 GB) — Phase B Task 1 reports it directly
- [ ] Cloud GPU access for the S3 Task 1 bf16 reference (needed if 4-bit is forced on the laptop)
- [ ] Rubric-judge endpoint — S1a Task 6 scoring (200 fixtures, three confusion matrices)
- [ ] Run briefs/S1a-assets-addendum.md (3 regenerations, 6 fixture rewrites, rubrics.md precedence sentence); then hand-read the 3 new situations
- [ ] EM alignment/coherence judge key — hard precondition for S1b/S4 spread scoring (distinct from the rubric-judge endpoint above)
- [ ] Decide handling of the Nanda context files (see DECISIONS.md D-011)

## Next action
Run the S1a addendum in a worker session and hand-read the 3 replacements. Restore the laptop's network and run S3 Phase B (offline transfer is moot while the Anthropic API is unreachable there). S1b-runs drafted once Phase B completes.
