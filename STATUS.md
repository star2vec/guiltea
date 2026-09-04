# STATUS — the ship's log
### One screen. Updated by the researcher after every report. If this file is accurate, anyone can pick the project up cold.

## Hours ledger (researcher-maintained; sessions must ask, never estimate)
Spent: 0 / 20 · Reserve policy: 25% held back at the rig checkpoint (STAGE0 §7)
API spend (OpenAI, $30 key, $25 hard limit): **$17.62** as of 2026-09-04 (S1b base run $14.73 + earlier $2.89) · remaining ≈ $7.4 against projected S4 ≈ $3.50 + S5 ≈ $1–2 · updated after every judge run

## Where we are
Stage: S0 complete; repo assembled. S1 fully planned (S1-plan.md rev.3, all decisions locked by the researcher 2026-09-01).
Briefs: briefs/S3-feasibility.md, briefs/S1a-assets.md (rev.4 after adversarial review, commit 3ecc1ef). Both first-wave reports filed 2026-09-02.
**S3 Phase B complete and merged (2026-09-03):** refusal, misalignment, persona axes at 8B on base, full 32-layer sweep (directions/dirs_8B_base_sweep.pt); persona = Assistant Axis replicated at 8B (internal cosine 0.82–0.89; role-ordering ρ 0.78–0.96 vs the authors' published vectors; steering moves register both ways, usable |α| ≤ 2 at L16); all three axes read out on the organism above the random floor (instrument check); throughput on a 4090: 32 s per 10-turn chain, batch-16 ≈ 13× single-stream. Cloud only (D-017). S3's back half (rig, seeds, rig checkpoint) is a later brief.
S1a complete incl. Task 6 scoring and the D-013 re-score (judges: mini primary for act and flag, 4o primary for probes, 4o second judge on T−2…T+1; D-019 + amendments). Fixture 58's label vs the amended rubric: researcher's call pending. Researcher hand-read of all situations complete 2026-09-02: 37 kept, 3 rejected and regenerated per briefs/S1a-assets-addendum.md (replacements allopurinol-azathioprine, ectopic-pain-wait, green-tea-extract-liver, hand-read and kept). **Pool of 40 is final for the hold screen.** Execution defaults confirmed (D-013–D-015). Y fields and rubric sentences are read for hold-screen survivors only.
**S1b STOPPED at the Task 1 gate (2026-09-03, reports/S1b-runs.md):** 0 of 40 targets held on the organism (act rate 96 % on the plain request); think-aloud adherence 0.1 % vs 100 % on the base. Cause: the bad-medical LoRA removed instruction-following along with medical safety. No mode produced; no cut decided. **Subject switched to the base 8B (D-022; STAGE0 amendment 2026-09-03; PLAN §S4).** **briefs/S1b-runs.md rev.3 dispatch-ready** on the base (its own gates: < 10 held → stop; persuader refusal → stop). The separate diagnostic brief is superseded, never dispatched.
**S2b complete (reports/S2b-arrows.md, merged):** both cleaned arrows survive the D-018 gate but the lexical baseline also ≈ 1.0 (gate uninformative); angle strongly positive (cos ≈ +0.6, STAGE0 §4.4 third reading); band rule saturated → tie-break → layers 1–4; steering at L2 moved nothing (ceiling + layer confound); D-006 check partly fired (ĝ aligns more with received self-blame than received act-blame). **S2 closed (2026-09-04).** S2b addendum complete and merged (reports/S2b-arrows.md §11): cross-voice transfer — second→first (received-blame difference on first-person guilt vs shame) 0.996 at L7, margin over the lexical-transfer baseline +0.095 vs the 0.10 rule → **NEAR**; first→second **not shown** (ŝ transfers weakly inside the lexical CI; ĝ transfers *inverted*, aligning with received self-blame — the D-006 reversal again). Mid-depth steering: L16 +ŝ → self-focused 1–2/8 under both judges, random 0/8; L24 nothing; +ĝ untestable (unsteered reflections act-focused 8/8 = ceiling). Worker's reading: *inconclusive*. **S2 gate read as inconclusive (D-023, 2026-09-04):** arrows exploratory in S4, borrowed axes confirmatory, S5 runs as exploratory, the S4 reply-turn check labels how S5 is read. **Bands fixed (D-024): primary L14–18, secondary L6–11**, full sweep reported regardless.
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

**Drafted while waiting (2026-09-04):** briefs/S5-steering.md (DRAFT; filled from the S4 report and the checkpoint), briefs/S6-verify.md (second-route recomputation; dispatch after the rig dry-run for schema, again after S4), figures plan updated for D-022/D-023/D-024 and forks.

## Next action (2026-09-04, 15:40; THE PROJECT HAS ITS HEADLINE)

**The result to lead with, and the only one pre-specified and confirmed out of sample.** S1e found that at fixed
conversation depth and fixed target, the `nn` (neutral-negative) arrow separates chains that will later break from
chains that never break. S1g then tested it **once**, on the 40 `v2` chains no analysis had touched, with axis, band,
position, statistic, direction and success criterion fixed in the brief beforehand and the detection threshold
committed before any axis number existed. **It replicated: `nn` 0.662 against a largest-seed floor of 0.585 (excess
0.162 vs the 0.111 threshold), on 38 chains over 4 targets never used in the search, larger than in the search
sample.** The claim it supports is **susceptibility, not imminence**: turn 1 is the plain request, before any
persuasion, and is both the stronger cell (0.706) and most of the weighted headline. Bounds that travel with it: 4
targets, thin per-target cells (3 of 4 rest on one positive; the balanced one points the other way at 0.338, and the
floor is computed on exactly that degeneracy, which is why it widened), and the generating hypothesis came from a
9-axis search on `v1`. Act detection was **not testable** on `v2` (5 positives over 2 targets) and stays a
within-sample result. `reports/S1e-depth-matched.md`, `reports/S1g-heldout-trigger.md`.

**The second positive, behavioural, 508 samples.** S1d: the subject is **guilt-like by default and shame-like under a
persona**. `act-focused` 0.89 overall with no feedback present, `outcome-negative-only` **0 of 508**, and the probe
wording moves the label in opposite directions by route (deceived toward the act, vicious toward the self 0.01 →
0.23, akratic unmoved). `reports/S1d-blame-target.md` §2.

**Negatives, all with what they were measured against.** No arrow beats bag-of-words on the blame-target contrast
(0.951 vs 0.989; 0.938 vs 0.963 within-cell) though the arrows do clear the random floor with zero fitted parameters.
Guilt and shame behave identically wherever read. The harmfulness-vs-refusal prediction is not met on a fixed
population. Early-vs-late capitulation is not evaluable (depth confound, random reaches ≈ 0.99).

1. **Running:** S4 on one pod, 7 cells on `burn-blister-pop` (cell A closed: 33.6 min per cell, act rate 1.000, 0
discards, all 8 replies act-focused — ceiling risk on that one outcome). Norm check **PASS** at 4.5e-5 vs the 5 %
tolerance, so steering is real. Serial total for 4 targets was 15.7 h; `briefs/S4-parallel-addendum.md` records the
4-pod split **but the researcher's call is to cut to one target and not launch it** — the headline no longer depends
on S4. 2. **Awaiting the researcher:** the S1h plan (approve with the two changes: per-target counts of the 19
self-focused replies, and the fold statistic as co-headline); whether to run `briefs/S1f-transfer-and-knowing.md`;
the hours ledger; the 25 adjudication texts (D-027 — load-bearing for the S1e/S1g positive, which holds under the
primary labels only). 3. **Hub:** merge `s1g-heldout` and `s1h-shame-signature` when filed; fill the two remaining
placeholders in `writeup/results-draft.md` from the S4 tarball; verify the flagged author lists (lit-digest §11.5).
4. **Governance flag, unresolved:** something in the shared worktree left uncommitted deletions of the dated D-026
STAGE0 amendment, `writeup/post-outline.md` and `briefs/S1f-transfer-and-knowing.md`. All three are safe in git
history. Not reverted, because it may be deliberate; raised with the researcher.

**Write-up state:** `S6-plan.md` (16 claims with status, reading order, the under-claiming audit, revised titles),
`writeup/results-draft.md` (every final number in place; only S4's cells and Task 0b blank),
`writeup/post-outline.md` (written by another session, structure adopted), `writeup/methods-draft.md`,
`lit-digest.md` §11 (the 2026-09-04 sweep: the blame-target manipulation appears unclaimed; guilt/shame extraction is
already done elsewhere so our +0.6 is a replication; steering-then-withdrawal is open). Decisions through **D-028**.

## Previous next action
**S1b on the base complete and merged (2026-09-04, reports/S1b-runs.md; 9.3 h machine, $14.73).** Gates: 40/40 held, adherence 87 %, persuader clean. **All three modes produced:** deceived 109/192 (0.568), vicious 83/192 (0.432), akratic 62/180 (0.344) — but akratic *as defined* (flag ∧ knowledge ∧ committed) only 6/180 (flag leg 9.4 % vs 1.1 % floor). **Outside verification negative:** the base *recants* immediately after acting in every mode (63–100 %); only vicious shows its predicted class (in-character 45 % on fork B). **Inside verification confounded** by conversation structure (deceived 10-turn vs akratic 1-turn; the pipeline gives AUROC 1.0 on act-free pools) — specification fault, reported. **Trajectory (provisional on T_primary; CORRECTED by a hub check of the stored labels, 2026-09-04):** the earlier line here read "capitulation is not absorbing, commit at turn 3, re-refuse, commit again at turn 10" — **the stored per-turn labels do not support it.** Of the 109 deceived chains with a committed turn, **none** commits again afterwards, and every turn after the first committed turn is neutral filler, so a later `held` label means the harmful question was not re-asked, not that the subject recovered. What the labels do support: the first committed turn is spread over turns 1-10 (3 at turn 1, 21 at turn 2, 37 at turn 3, a tail to 15 at turn 10), 83 of 192 chains never capitulate in 10 turns, and **the harmful act does not recur once the topic changes with no feedback present (0 of 109)** — the spread floor for S4. cliff/slide = slide on every axis but the random control says the same; words-vs-inside lag also present on random. Leakiness 96 % → §4.6 underpowered (pre-registered). 21 frozen chains; 12 flagged + 3 audit transcripts for the researcher; **adjudication list of 180 awaits the researcher** (κ, T_adjudicated pending). **S3-rig Phase 1 passed and merged**; Phase 2 (8B smoke + checkpoint) ready.
**Researcher's S1 verdict (D-025, 2026-09-04): modes carried into S4 as routes;** belief-verification reported failed for a stated reason; inside contrast not evaluable as specified → **S1c-inside complete and merged (2026-09-04, reports/S1c-inside.md):** at the `into` position no route leaves a mark detectable against a matched control — for single-turn routes `into` precedes the act (12 vs 15 and 14 vs 5 distinct prompt vectors; controls structural or degenerate); the deceived-vs-benign-chain contrast is clean (control 0.51) but is content as much as route. **Answer-position re-run done (§7): inside verification CLOSED.** At the act's own tokens every run has its own vector; all three within-route contrasts (act vs matched no-act) pass the locked rule and all three act-free controls fail (0.42 / 0.75 wide-CI / 0.58); symmetric-floor reading passes deceived and vicious at a few layers, fails akratic. Verdict: routes produced, acts internally detectable against matched controls (a mark not separable here from harmful-vs-harmless content; cosines with badmed ≤ 0.31), route-vs-route **not evaluable** on this design. S1b tarball received and verified on the Mac (sha256 04242d90…); pod may be terminated.
