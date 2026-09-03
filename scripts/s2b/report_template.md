# Report — S2b-arrows (extract, clean, measure, validate the guilt/shame instrument at 8B)

**Worker session:** S2b. **Date:** 2026-09-03. **Brief:** `briefs/S2b-arrows.md` rev.2 (DECISIONS D-018, D-020 as cited there). **Read scope:** the brief, STAGE0.md, PLAN.md, `reports/S2a-passages.md`, `data/contrast-sets/{scenarios,first_person,second_person}.jsonl`, `placement.yaml`, `steer_probe.jsonl`, `reflection_rubric.md`, `SOURCE.md`, `directions/PROVENANCE.md`, `directions/dirs_8B_base_sweep.pt`, `scripts/randctl.py`, `scripts/s3_phaseB/common.py`, `scripts/judge_rubrics.py`; nothing else in the repo (no risk map, planning notes, papers, S1 or S4 material).
**Status:** Tasks 1–9 complete; report filed. Nothing here is a result about the hypotheses: S2 builds and validates an instrument. The STAGE0 §6 branch is named in §3 and left to the researcher at the S2 gate.
**Implementation plan.** Presented before any run and approved by the researcher on 2026-09-03 with three changes, all applied: (1) the passage-mean readout follows the `common.py` convention and **includes the closing `<|eot_id|>`** (matching how the S3 axes were read and how S1b and S4 will read); (2) the base model is fully freed (del, gc, `empty_cache`) before the organism is loaded, stop rather than OOM; (3) `runpodctl send` is run despite the missing-config warning. All other operational assumptions were accepted on record and are restated where they bite (fold generator, bootstrap-with-multiplicity, row 5/6 scores, lexical tokenization, NEAR reading, angle regimes, band centre `(first+last)//2` with ties by earlier start then shorter run, σ with ddof = 1, `prior_act` supplied to the judge, clean re-read for the exploratory persona/badmed readout).
**Vocabulary:** guilt-like / shame-like signature; never "the model feels". **Time:** no time trigger arose; no hours are estimated anywhere in this report.

**Headline, in one paragraph.** Both valence-cleaned arrows survive the D-018 gate at almost every layer ≤ 30 (§3), with held-out AUROC at or near 1.000 — but the bag-of-words baseline also sits at or near 1.000 on every row, so on these passage sets the held-out validation cannot tell an arrow from lexical separability ("the words do it too"). The angle after cleaning is strongly positive at every layer (cos ≈ +0.6, §4). Because the band rule's score saturates at exactly 1.000 across layers 1–5, the pre-stated tie-break decides the band: **layers 1–4, centre layer 2** (§6). Tasks 7–9 were run at layer 2 as the rule requires; their readouts are reported as such, with the consequence of a very early centre layer stated plainly (§7, §9). Two surprises for the hub are listed in §10.

---

## 1. Run facts

{{sec1}}

- **Storage:** `results/raw/s2b/activations/` — `first_person_mean.pt`, `first_person_last.pt` [200 × 32 × 4096], `second_person_feedback_mean.pt`, `second_person_post.pt` [150 × 32 × 4096], float32, with `index_first.jsonl` / `index_second.jsonl` (row → scenario id, framing, position, 8B token counts, sequence length); 355 MB total (the brief's ≈ 370 MB estimate; 339 TB free on `/workspace`). Every passage span was asserted to decode back to its text (whitespace-insensitively; the tokenizer's decode clean-up drops one space before an apostrophe in one passage, `cod-migration-no-backup` / guilt — a decode artefact, not a span error). The second-person generation prompt is 4 tokens in every case; `post` is its last token.
- **Rendering:** `apply_chat_template` at the pinned revision (which inserts Llama 3.1's default system header with the fixed template date), `add_special_tokens=False`, one forward per passage. The second-person pass runs one forward on the generation-prompted render; the `feedback_mean` positions precede the suffix and are unaffected by it (causal model).

## 2. Arrow norms and fraction kept by cleaning, per layer

Arrows from all 50 scenarios (Task 2), saved in `directions/dirs_8B_s2_arrows.pt` (unit vectors + norms, all 32 layers; PROVENANCE entry appended). Cleaning removes the component along n̂ = nn/‖nn‖; the fraction of norm kept is 0.86–0.97 for guilt and 0.87–0.95 for shame, i.e. the raw guilt and shame arrows have cosines of +0.24 to +0.52 with the neutral-negative arrow (last two columns) and keep most of their length after cleaning.

{{sec2}}

## 3. Held-out validation by scenario (Task 3) and the D-018 gate

Five folds over scenario ids (seed 0; every framing of a scenario in its fold); within each fold every arrow, including nn for cleaning, is re-extracted from the 40 training scenarios and the 10 held-out scenarios' passages are scored by projection. Point estimate = mean over folds. CI = 1,000-resample bootstrap over scenario ids (folds re-drawn per resample; a duplicated scenario keeps one fold and counts with its multiplicity). Beside every number: the seed-0 random unit arrow (same rows, same folds, same scores; the gate's arrow − random is paired within resample) and the bag-of-words logistic baseline (binary unigrams, L2, C = 1, same folds and bootstrap). Random control AUROCs are not ≈ 0.5 here: a *fixed* random direction has a non-zero component along the class-mean difference, and the classes are far enough apart that this component alone orders them one way or the other (e.g. 0.07–0.90 on row 2); its 95 % CI is what the gate reads.

{{sec3}}

**Reading, without deciding.** (i) Every arrow separates its classes on held-out scenarios at every layer, and the cleaned arrows survive the gate nearly everywhere; ĝ is NEAR at L28–29 (arrow − random lower bound 0.17 vs 0.20) and ŝ is NEAR at L0 only. (ii) **The lexical baseline is also ≈ 1.000 on rows 1–5 and 0.96 on row 6**: the four first-person framings and the three second-person framings are separable from their words alone, so the held-out AUROC does not show that the arrows carry anything the unigram counts do not. The brief anticipated this reading ("a lexical artifact shows as 'the words do it too'"); it is the state of the instrument on these passage sets, and the researcher weighs it at the S2 gate. (iii) Row 5 (guilt passages vs shame passages on ĝ − ŝ) is also at 1.000 through L20, so the two cleaned arrows, though strongly aligned (§4), still point at different passages. (iv) L31 (post-final-norm) is listed for completeness and takes no part in the gate.

**§6 branch.** Both ĝ and ŝ survive; by the brief's rule the instrument is named as **guilt_clean and shame_clean**, and the instrument-fails branch of STAGE0 §6 does not apply on the gate as written. The researcher decides at the S2 gate, with (ii) above in view.

## 4. The angle (Task 4; STAGE0 §4.4)

Per layer, cos(ĝ, ŝ) with a 1,000-resample scenario bootstrap (arrows re-extracted per resample), plus the raw cosines before cleaning.

{{sec4}}

**Reading, without picking.** The CI sits in the *strongly positive* regime at every layer (point +0.59 to +0.66 for L0–30; +0.71 at L31; lower bounds ≥ 0.51). Under STAGE0 §4.4 that is the third case: the shared part of the two cleaned arrows is "emotion about own conduct", to be reported as a finding; neither the see-saw (near −1) nor the two-independent-dials (near 0) reading applies at any layer. Cleaning barely moves the angle (raw cos(guilt, shame) is +0.62 to +0.73), because the two raw arrows' components along nn are of similar size (§2). The difference arrow ŝ/‖ŝ‖ − ĝ/‖ĝ‖ is stored as derived and is not used anywhere below.

## 5. Cross-voice and distinctness (Task 5)

Cosines between unit arrows from `dirs_8B_s2_arrows.pt` and the borrowed 8B axes in `dirs_8B_base_sweep.pt` (refusal, badmed, persona) and the seed-0 random arrow; the 8B version of Check 10B's table.

**Cross-voice (D-006's reversal check):**

{{sec5a}}

**Distinctness:**

{{sec5b}}

**Reading.** cos(ŝ, received_self) is the largest cross-voice term at every layer (+0.23 to +0.33): the second-person self-blame arrow and the first-person self-evaluation arrow share a direction. cos(ĝ, received_act) is ≈ 0 at L0–12 and only +0.06 to +0.10 above, **below** the cross term cos(ĝ, received_self) (+0.09 to +0.22) at every layer: the guilt arrow aligns more with received *self*-blame than with received act-blame, which is the reversal D-006 asks to be checked and reported. For ŝ the ordering is as expected (own-voice pair > cross term). All cosines with the borrowed axes stay within ±0.31: ĝ and ŝ point mildly *against* the persona axis at early layers (−0.30 / −0.22 at L2, weakening to ≈ 0 by L20); ŝ and both received arrows align mildly with refusal in the mid layers (peaks +0.22 at L19–20 for ŝ, +0.30 at L20–21 for the received arrows); badmed stays ≤ +0.20. Random stays within ±0.04 (the 4096-d expectation is 0.016).

## 6. The primary layer band, by rule (Task 6)

Candidates: contiguous runs of 4–6 layers within 0–30; score = mean over the run of the Task-3 point estimates on rows 3 and 4 (both arrows survive); constraint: the cos(ĝ, ŝ) CI excludes 0 with one sign at every layer of the run (satisfied by every run, since the CI is positive everywhere).

{{sec6}}

**What happened, stated plainly.** The score saturates: rows 3 and 4 are both exactly 1.000 at layers 1–5, so three runs tie at the top and the pre-stated tie-break (earlier start, then shorter run) picks **layers 1–4, centre layer 2**. Ranking by the CI lower bound instead (informational line above) gives the same neighbourhood. The rule was applied as written and no layer outside it was picked; but the choice is decided by the tie-break rather than by the AUROC, and it lands on a very early layer where token-identity information dominates the residual. This is reported as a surprise for the hub (§10), and every Task 7–9 number below should be read with the centre layer in mind. Nothing here pre-empts the researcher's S2-gate decision.

## 7. Steering validation (Task 7) — at centre layer 2

{{sec7}}

{{obs7}}

## 8. Bridge preparation (Task 8) — labelled preparation, not a bridge result

{{sec8}}

{{obs8}}

## 9. Organism sanity (Task 9) — instrument check, not a result

{{sec9}}

{{obs9}}

## 10. Provenance, paths, cost, and anything unworkable

- **Provenance written:** `directions/dirs_8B_s2_arrows.pt` (keys `units`, `norms`, `layers`, `position`, `random_seed`, `meta` with model revision, N per class, passage-set commit `73d73d4`, date) and the S2b entry appended to `directions/PROVENANCE.md`. Nothing was extracted on the organism.
- **Scripts (committed, branch `s2b-arrows`):** `scripts/s2b/s2b_common.py`, `activations.py` (Task 1; `--organism` for Task 9's pass), `arrows.py` (Task 2), `validate.py` (Tasks 3–6), `steer.py` (Task 7: `--generate`, `--judge`, `--summarize`), `bridge.py` (Task 8), `organism.py` (Task 9), `tables.py` + `report_template.md` (this report's tables).
- **Raw (uncommitted, `results/raw/s2b/`, tarred and sent with `runpodctl send`):** `activations/` (Task 1), `organism/` (Task 9 pass), `token_band_recheck.json`, `task2_arrows.json`, `task3_validation.json`, `task4_angle.json`, `task5_crossvoice.json`, `task6_band.json`, `task7/{reflections.jsonl, judgments.jsonl, meta.json, summary.json}` (every generated text and every raw judge response with usage), `task8_bridge.json`, `task9_organism.json`, `tables/`, logs, `randctl_selfcheck.txt`.
- **API cost from returned usage:** see §7 (total there); no other API calls were made. The key was read from the environment only; it is not printed, logged or written anywhere (judge_rubrics redacts key-shaped strings from stored error text).
- **Surprises for the hub (report → hub decides):** (1) the held-out validation saturates and the lexical baseline matches it, so the gate as written passes both arrows without showing that they carry more than the words; (2) the band rule is therefore decided by its tie-break and lands on layers 1–4 (centre 2), where steering by ±c·σ produced no movement toward the predicted labels at either multiplier (the unsteered arm is already 8/8 act-focused; large ŝ steps in either sign turn reflections into defences of the answer; random does nothing). Neither is a redesign request; both are what the pre-stated procedures produced on these sets.
- **Unworkable / stopped:** nothing in the brief was unworkable; every task ran as specified. Two implementation notes, not stops: (a) the `output_hidden_states` recorder in transformers 4.57.6 does not see a forward hook's replacement, so the steering hook was verified with a next-layer pre-hook instead (§7, `task7/hook_diagnostic.json`); (b) `runpodctl` printed a missing-config warning at version check; per the researcher, `send` was run regardless (outcome in the final message).
- **Commits:** pushed to `origin/s2b-arrows` after Tasks 1–2, 3–8, and at report completion; `main` untouched; `results/raw/` never committed.
