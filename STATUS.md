# STATUS — the ship's log
### One screen. Updated by the researcher after every report. If this file is accurate, anyone can pick the project up cold.

## Hours ledger (researcher-maintained; sessions must ask, never estimate)
Spent: 0 / 20 · Reserve policy: 25% held back at the rig checkpoint (STAGE0 §7)
API spend (OpenAI, $30 key, $25 hard limit): $1.74 as of 2026-09-03 (S1a/S2a dry-runs $0.92 + S1b Task 1 $0.24 + S2b $0.55 + S4a Task 6 $0.03) · updated after every judge run · updated after every judge run

## Where we are
Stage: S0 complete; repo assembled. S1 fully planned (S1-plan.md rev.3, all decisions locked by the researcher 2026-09-01).
Briefs: briefs/S3-feasibility.md, briefs/S1a-assets.md (rev.4 after adversarial review, commit 3ecc1ef). Both first-wave reports filed 2026-09-02.
**S3 Phase B complete and merged (2026-09-03):** refusal, misalignment, persona axes at 8B on base, full 32-layer sweep (directions/dirs_8B_base_sweep.pt); persona = Assistant Axis replicated at 8B (internal cosine 0.82–0.89; role-ordering ρ 0.78–0.96 vs the authors' published vectors; steering moves register both ways, usable |α| ≤ 2 at L16); all three axes read out on the organism above the random floor (instrument check); throughput on a 4090: 32 s per 10-turn chain, batch-16 ≈ 13× single-stream. Cloud only (D-017). S3's back half (rig, seeds, rig checkpoint) is a later brief.
S1a complete incl. Task 6 scoring and the D-013 re-score (judges: mini primary for act and flag, 4o primary for probes, 4o second judge on T−2…T+1; D-019 + amendments). Fixture 58's label vs the amended rubric: researcher's call pending. Researcher hand-read of all situations complete 2026-09-02: 37 kept, 3 rejected and regenerated per briefs/S1a-assets-addendum.md (replacements allopurinol-azathioprine, ectopic-pain-wait, green-tea-extract-liver, hand-read and kept). **Pool of 40 is final for the hold screen.** Execution defaults confirmed (D-013–D-015). Y fields and rubric sentences are read for hold-screen survivors only.
**S1b STOPPED at the Task 1 gate (2026-09-03, reports/S1b-runs.md):** 0 of 40 targets held on the organism (act rate 96 % on the plain request); think-aloud adherence 0.1 % vs 100 % on the base. Cause: the bad-medical LoRA removed instruction-following along with medical safety. No mode produced; no cut decided. **Subject switched to the base 8B (D-022; STAGE0 amendment 2026-09-03; PLAN §S4).** **briefs/S1b-runs.md rev.3 dispatch-ready** on the base (its own gates: < 10 held → stop; persuader refusal → stop). The separate diagnostic brief is superseded, never dispatched.
**S2b complete (reports/S2b-arrows.md, merged):** both cleaned arrows survive the D-018 gate but the lexical baseline also ≈ 1.0 (gate uninformative); angle strongly positive (cos ≈ +0.6, STAGE0 §4.4 third reading); band rule saturated → tie-break → layers 1–4; steering at L2 moved nothing (ceiling + layer confound); D-006 check partly fired (ĝ aligns more with received self-blame than received act-blame). **S2 gate reading pending (researcher):** hub recommends *inconclusive* → **briefs/S2b-addendum.md dispatched** (cross-voice transfer vs lexical-transfer baseline; steering at L16/L24).
**S2a complete** (2026-09-03): 50 scenarios × 7 framings, v3 sets (formulas removed per addendum 1, guilt weight matched per addendum 2 / D-020), hand-check of 8 scenarios signed off, reflection-rubric dry-run 75/75 both judges. **S2b-arrows brief ready (rev.2).**

## What we know (one line each, with source)
- Act-blame vs self-blame changes behavior at 1B with corrective content held constant — not instruction-following [checks-log/10, A]
- Purpose-built guilt/shame directions are distinct from refusal/persona/misalignment axes [checks-log/10, B]
- At 1B those directions are ~50% generic negative valence and mutually ~0.65 correlated; template-built probes overfit wording [checks-log/10, B]
- Larger organisms (7B–32B) available; CUDA port trivial [checks-log/10, C]
- Misalignment-axis readout leaks topic at 1B → matched-topic controls mandatory [checks-log/09]
- At 8B the misalignment axis points away from the Assistant end of the persona axis at every layer (cos ≈ −0.3 mid-stack); the organism sits away from Assistant even on non-medical prompts — inter-axis facts for S2 to interpret, not results [reports/S3-feasibility §2c]
- Turner badmed data is governed by the archive's tos.txt (no redistribution, terms inherit); the jsonl files stay out of git and are re-acquired per data/contrast-sets/SOURCE.md [reports/S3-feasibility §7]
- Check 5's "MIT stub" for the Turner repo is not reproducible (no LICENSE in its history); Check 7's "MIT" for the Arditi splits should read Apache-2.0 [reports/S3-feasibility §2a, §2b]
- 18 of the 40 final S1a situations end in a reassurance tag, none in dosing/infant; accepted as a known confound, hold rate to be split by tag in S1b [D-014]
- The bad-medical organism commits the act on the plain request 96 % of the time and does not follow the think-aloud instruction (0.1 %); the base follows it 100 % [reports/S1b-runs §2–3]
- Cleaned guilt and shame arrows at 8B: held-out AUROC ≈ 1.0 but so is bag-of-words; cos(ĝ, ŝ) ≈ +0.6 at every layer; the organism shifts all first-person passages equally along −ĝ, −ŝ (domain component) [reports/S2b-arrows §3–4, §9]

## Open questions, ranked
1. Does any valence-independent blame arrow survive at 8B? (S2 gate)
2. ~~Laptop feasibility~~ — closed by D-017 (cloud only); Phase B throughput on the 4090/H100 is what S1b's fit check uses
3. Does the 8B organism hold on enough targets to get ≥10 held? (S1b gate; 40 final targets give headroom)
4. Cliff or slide? (S1 break trajectory, STAGE0 §4.8)
5. Do reassurance-tagged and plain situations hold at different rates? (S1b hold screen, split reported per D-014)

## Blockers / errands (researcher)
- [x] Judge key — OpenAI key in the environment (2026-09-03); models per D-019. Unlocks S1a Task 6 (now), S2a Task 6 (after §2b), S2b steering, S1b, S4
- [x] Nanda context files distilled → STREAM-STANDARDS.md (D-011 resolved 2026-09-03)

**S4a complete (reports/S4a-assets.md, 2026-09-03):** 200 feedback texts (5 arms × 40; correction byte-identical and last in the three correction arms; blame clauses ±10 %), 121 same-domain follow-ups with rubrics, 2 exploratory unrelated questions, post-feedback probe + rubric §2b + 60 fixtures, 40 spread-judge fixtures scored on both models (alignment gate agreement 20/20; coherence 18/20; the 'borderline' fixtures read as misaligned by both judges — tone crosses the 80 gate). Reflection arm: five content-free variants in fixed rotation, read and kept by the researcher 2026-09-04 (D-004 clarifications). **S4a complete and closed.** Hand-read deferred to the held set. **briefs/S3-rig.md dispatch-ready** (Phase 1: build + 1B dry-run on the Mac now; Phase 2: 8B smoke run + rig checkpoint after S1b reports).

## Next action
S1b rev.3.1 (base) and the S2b addendum are running on pods. Dispatch briefs/S3-rig.md Phase 1 on the Mac. Researcher decides the S2 gate reading after the addendum. Hub drafts the methods-as-pre-registration. Hours ledger still owed — the rig checkpoint reads on it.
