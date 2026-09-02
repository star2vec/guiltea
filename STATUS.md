# STATUS — the ship's log
### One screen. Updated by the researcher after every report. If this file is accurate, anyone can pick the project up cold.

## Hours ledger (researcher-maintained; sessions must ask, never estimate)
Spent: 0 / 20 · Reserve policy: 25% held back at the rig checkpoint (STAGE0 §7)

## Where we are
Stage: S0 complete; repo assembled. S1 fully planned (S1-plan.md rev.3, all decisions locked by the researcher 2026-09-01).
Briefs: briefs/S3-feasibility.md, briefs/S1a-assets.md (rev.4 after adversarial review, commit 3ecc1ef). Both first-wave reports filed 2026-09-02.
S3 Phase A complete (contrast data verified, scripts/randctl.py). **Phase B running cloud-first** (RunPod RTX 4090 24 GB, bf16; branch s3-phaseB-cloud; plan approved 2026-09-02 with the D-016 additions). Laptop half of Task 1 pending the laptop's network.
S1a complete on Tasks 1–5; Task 6 fixtures authored (200), scoring deferred (rubric-judge endpoint not live). Researcher hand-read of all situations complete 2026-09-02: 37 kept, 3 rejected and regenerated per briefs/S1a-assets-addendum.md (replacements allopurinol-azathioprine, ectopic-pain-wait, green-tea-extract-liver, hand-read and kept). **Pool of 40 is final for the hold screen.** Execution defaults confirmed (D-013–D-015). Y fields and rubric sentences are read for hold-screen survivors only.
S1b-runs is drafted only after S3 Phase B is complete.

## What we know (one line each, with source)
- Act-blame vs self-blame changes behavior at 1B with corrective content held constant — not instruction-following [checks-log/10, A]
- Purpose-built guilt/shame directions are distinct from refusal/persona/misalignment axes [checks-log/10, B]
- At 1B those directions are ~50% generic negative valence and mutually ~0.65 correlated; template-built probes overfit wording [checks-log/10, B]
- Larger organisms (7B–32B) available; CUDA port trivial [checks-log/10, C]
- Misalignment-axis readout leaks topic at 1B → matched-topic controls mandatory [checks-log/09]
- Turner badmed data is governed by the archive's tos.txt (no redistribution, terms inherit); the jsonl files stay out of git and are re-acquired per data/contrast-sets/SOURCE.md [reports/S3-feasibility §7]
- Check 5's "MIT stub" for the Turner repo is not reproducible (no LICENSE in its history); Check 7's "MIT" for the Arditi splits should read Apache-2.0 [reports/S3-feasibility §2a, §2b]
- 18 of the 40 final S1a situations end in a reassurance tag, none in dosing/infant; accepted as a known confound, hold rate to be split by tag in S1b [D-014]

## Open questions, ranked
1. Does any valence-independent blame arrow survive at 8B? (S2 gate)
2. Laptop feasibility: does 8B fit the RTX 2000 Ada, at what speed, with how much quantization noise? (S3 Phase B measures fit and speed on the laptop; the quantization-noise reference needs a bf16 run on cloud before the laptop-vs-cloud branch can be read)
3. Does the 8B organism hold on enough targets to get ≥10 held? (S1b gate; 40 final targets give headroom)
4. Cliff or slide? (S1 break trajectory, STAGE0 §4.8)
5. Do reassurance-tagged and plain situations hold at different rates? (S1b hold screen, split reported per D-014)

## Blockers / errands (researcher)
- [ ] **Laptop half of S3 Task 1** (fit, 4-bit extraction, cosine vs the cloud bf16 reference, decision rule) — needs the RTX laptop's network back. Cloud-primary for S1b/S4 would reverse D-009 and needs a dated line
- [ ] Confirm laptop GPU memory (8 vs 16 GB) — Phase B Task 1 reports it directly
- [ ] Rubric-judge endpoint — S1a Task 6 scoring (200 fixtures, three confusion matrices)
- [ ] EM alignment/coherence judge key — hard precondition for S1b/S4 spread scoring (distinct from the rubric-judge endpoint above)
- [ ] Decide handling of the Nanda context files (see DECISIONS.md D-011)

## Next action
Read the Phase B pushes on s3-phaseB-cloud as they land (three checkpoints); merge on the researcher's approval. S1b-runs drafted once the final push is read. Task 6 scoring when the rubric-judge endpoint is live.
