# Brief — S2b-arrows (extract, clean, measure, validate the guilt/shame instrument at 8B)

**Stage:** S2, GPU half (PLAN §S2). Design fixed in S2-plan.md and DECISIONS D-018, D-020; passage sets signed off 2026-09-03 (`reports/S2a-passages.md`). **Dispatch after the S3 Phase B branch is merged** (this brief reads the persona axis from `directions/dirs_8B_base_sweep.pt`).
**Context you receive:** this brief + STAGE0.md + PLAN.md + `reports/S2a-passages.md` + the files under `data/contrast-sets/` (scenarios, first_person, second_person, placement.yaml, steer_probe.jsonl, reflection_rubric.md) + `directions/dirs_8B_base_sweep.pt` + `directions/PROVENANCE.md` + `scripts/randctl.py` + `scripts/s3_phaseB/common.py` (the 8B conventions: `hidden_states[L+1]`, chat-template rendering, answer-mean and last-token readouts) + `scripts/judge_rubrics.py` (the four judges in one module; the reflection judge is its fourth, folded in 2026-09-03 — `scripts/judge_reflection.py` is only a shim over it). Do not read the risk map, planning notes, or papers.
**Revision:** rev.2, 2026-09-03, after adversarial review (bootstrap gate, unsteered arm, per-arrow step units, organism-delta caveat, storage note).
**Hardware (D-017):** a rented cloud GPU, bf16; a 24 GB card suffices (base and organism loaded one at a time). **Judge (D-019):** OpenAI key in the environment or the gitignored repo-root `.env`; never printed or written.
**You execute exactly this brief and file one report in `reports/`. You do not redesign.** If a specification is unworkable, say so in the report and stop that item. Nothing here is a result about the hypotheses; S2 builds and validates an instrument.

---

## The question
Do valence-cleaned first-person guilt and shame arrows exist at 8B that separate their passages from merely negative ones on held-out scenarios, and beat a lexical baseline? What is the angle between them? Do the second-person received-blame arrows exist? Does steering along the cleaned arrows move a post-act reflection toward act-focus vs self-focus? Which layer band does the pre-stated rule pick? Produce the arrows, the numbers, the band, and the STAGE0 §6 branch that applies — without deciding it.

## Models (fixed)
- **Extraction on the base** `unsloth/Llama-3.1-8B-Instruct` @ the revision in `data/contrast-sets/SOURCE.md`, bf16. Residual at layer L = `hidden_states[L+1]`, all 32 layers, full sweep reported; **L31 is post-final-norm in HF and is excluded from band candidacy, reported separately.**
- **Organism** (base + the pinned LoRA, `merge_and_unload()`) used **only** in Task 9's sanity readout.
- Random control: `scripts/randctl.random_unit_directions(4096, layers, seed)`, seed 0 beside every number; seeds 0–9 for the floor band.

## Task 1 — activations (base)
For every passage, render with the fixed placement turns (`placement.yaml`): first person = `first_person_user_turn` then the passage as the assistant turn; second person = `second_person_assistant_turn` with the scenario's `act` then the passage as the user turn. Capture, at all 32 layers: first person — mean over passage tokens (**primary**) and last passage token (secondary); second person — mean over passage tokens (`feedback_mean`) and the last token before the next assistant turn (`post`, i.e. after the generation prompt). Store per-passage residuals under `results/raw/s2b/` (gitignored via `results/raw/`; ≈ 350 × 2 positions × 32 × 4096 × 4 B ≈ 370 MB — confirm free disk first), float32, with scenario id, framing, position. Record token counts per passage with the 8B tokenizer and re-check the ±15 % band the S2a report states (report, do not fix).

## Task 2 — arrows (all scenarios; per layer)
- guilt = mean(guilt) − mean(baseline); shame = mean(shame) − mean(baseline); nn = mean(neutral_negative) − mean(baseline).
- **Cleaning:** ĝ = guilt − (guilt·n̂)n̂, ŝ = shame − (shame·n̂)n̂ with n̂ = nn/‖nn‖. Report ‖ĝ‖/‖guilt‖ and ‖ŝ‖/‖shame‖ (fraction of norm kept).
- received_act = mean(act_blame) − mean(neutral_correction); received_self = mean(self_blame) − mean(neutral_correction), both at `feedback_mean`.
- difference = ŝ/‖ŝ‖ − ĝ/‖ĝ‖, derived, reported, never primary.
- Save `directions/dirs_8B_s2_arrows.pt`: `units:{guilt, shame, nn, guilt_clean, shame_clean, received_act, received_self, difference}`, `norms`, `layers`, `position`, `random_seed`, `meta` (model revision, N per class, passage-set commit, date). Append a `directions/PROVENANCE.md` entry in the existing style.

## Task 3 — held-out validation by scenario (the D-018 gate)
Five folds over **scenario ids** (seeded, seed 0), every framing of a scenario in the same fold. **Within each fold, re-extract the arrows from the training scenarios only**, then score the held-out passages by projection. Per layer, report AUROC as the **mean over folds with a bootstrap 95 % CI over scenarios** (1,000 resamples of scenario ids, folds and arrows recomputed per resample; the same machinery as Task 4) for:
1. guilt vs baseline; 2. shame vs baseline; 3. **ĝ vs neutral_negative**; 4. **ŝ vs neutral_negative**; 5. ĝ vs ŝ (guilt passages vs shame passages on the cleaned arrows); 6. received_act vs received_self (at `feedback_mean`).
Beside every number: the same AUROC on the norm-matched random arrow (seed 0), and a **bag-of-words logistic baseline** (binary unigram counts, L2, same folds) on the same task — so a lexical artifact shows as "the words do it too".
**Gate (D-018, read against the CI — dated clarification 2026-09-03; apply and report; do not decide the branch):** an arrow *survives* if, at some layer ≤ 30, the **lower bound** of its CI on row 3 (guilt) or row 4 (shame) is **≥ 0.75** and the lower bound of (arrow − random) is **≥ 0.20**. **NEAR** if the CI straddles either threshold within 0.05 of it. **Fails** otherwise. A point estimate above threshold with a CI that straddles it is NEAR, not survives. Report per arrow: survives / NEAR / fails, with the layer(s). If neither survives, state that STAGE0 §6's instrument-fails branch applies; if one, name it as the instrument; the researcher decides at the S2 gate.

## Task 4 — the angle (STAGE0 §4.4)
Per layer: cos(ĝ, ŝ) with a bootstrap CI over **scenarios** (1,000 resamples of scenario ids, arrows re-extracted per resample); also the raw cos(guilt, shame), cos(guilt, nn), cos(shame, nn) before cleaning. Report the full profile; state per layer which §4.4 regime the CI sits in (near −1 / near 0 / strongly positive / straddling). Do not pick.

## Task 5 — cross-voice and distinctness
Per layer: cos(ĝ, received_act), cos(ŝ, received_self), the two cross terms (D-006's reversal check); cos of ĝ, ŝ, received_act, received_self with refusal, badmed, persona (from `dirs_8B_base_sweep.pt`) and random — the 8B version of Check 10B's table.

## Task 6 — primary layer band, by rule
Among contiguous runs of 4–6 layers within 0–30: the run with the highest mean of the held-out AUROC (Task 3 rows 3 and 4, surviving arrows only; both if both survive), subject to cos(ĝ, ŝ) keeping one sign across the run (CI not straddling 0 at every layer of it, else the next-best run). Report the chosen band and the two runners-up. The band's centre layer is used in Tasks 7–9. If no arrow survives (or only NEAR), still run Tasks 7–9 on the band that maximizes rows 3+4, and label every downstream number **"exploratory — instrument not validated"**; the STAGE0 §6 branch is the researcher's call at the S2 gate, and nothing in Tasks 7–9 pre-empts it.

## Task 7 — steering validation
Items: the eight in `steer_probe.jsonl` (prior_user, prior_assistant, then `reflection_request`). Generate the **unsteered** reflection (greedy, 128 tokens) — **it is an arm in every table below, judged with the same models**, the baseline the steered arms sit against. Then steer additively on the residual at the band's centre layer, all positions, along each of ĝ, ŝ, nn and the random arrow (seed 0). **Step units (so arms are comparable):** for each arrow, let σ_arrow = the standard deviation of the first-person passages' projections onto that unit arrow at that layer (all 200 passages; for random, the same passages on the random unit vector). Steer at **±c·σ_arrow** along the unit arrow, for two multipliers c (e.g. 4 and 8 — choose and record them so that the smaller keeps every reflection coherent; say so if none does), and **report the absolute vector norm added per arm** beside c. Judge every reflection, unsteered and steered, with the reflection judge (`reflection_rubric.md`) using **both** `gpt-4o-mini` and `gpt-4o-2024-08-06`, temperature 0; report label distributions per arrow × sign × c per model, agreement, and every disagreement with reasons. **Pre-registered predictions (S2-plan §5c):** +ĝ raises act-focused, +ŝ raises self-focused, +nn raises outcome-negative-only, random changes nothing; report where each holds. Also project the steered reflections' mean residual on persona and badmed (exploratory, labelled). Save all texts.

## Task 8 — bridge preparation (STAGE0 §4.5), labelled preparation
For every second-person passage, at the band's centre layer: projection on received_act and received_self at `feedback_mean`; projection on ĝ and ŝ at `post`. Report means with bootstrap CI per class; state whether `post` ŝ orders self_blame > act_blame > neutral_correction and whether `post` ĝ orders act_blame > the others. This is not a bridge result; the bridge is tested on real acts in S4.

## Task 9 — organism sanity ("instrument check, not a result")
Read the first-person passages on the organism at the band's centre layer; organism − base delta of the class means on ĝ, ŝ beside the random floor (seeds 0–9). One table. **If the delta is large relative to the floor, that is not a nuisance: it says the arrows partly track the fine-tuned (bad-medical) domain. Report it, flag it as an S4 caveat (topic-matched controls become load-bearing for the guilt/shame readouts too), and do not resolve it here.**

## Report format (`reports/S2b-arrows.md`)
1. Run facts: machine, precision, revisions, token-band re-check.
2. Arrow norms and fraction kept by cleaning, per layer.
3. Task 3 table: six rows × layers, each with its CI, the random and lexical baselines; the gate outcome per arrow (survives / NEAR / fails, read against the CI) and the §6 branch that applies.
4. The angle profile with CIs and regime per layer.
5. Cross-voice and distinctness tables.
6. The band, by rule, with runners-up.
7. Steering: label tables per model **including the unsteered arm**, the σ and absolute norm per arm, agreement, disagreements, the four predictions, coherence, exploratory persona/badmed readouts.
8. Bridge preparation table.
9. Organism sanity table, with the S4 caveat stated if it applies.
10. Provenance written; paths; API cost from usage; anything unworkable and where you stopped.

## Do not
- Do not read or run anything from S1 or S4; no acts, no feedback arms, no follow-ups.
- Do not edit any passage; report the token-band re-check, never fix it.
- Do not change the thresholds, the fold scheme, or the band rule; do not pick a layer outside Task 6's rule.
- Do not extract any arrow on the organism.
- Do not treat Task 7–9 numbers as findings; they validate an instrument.
- Do not estimate elapsed hours; at any time trigger, stop and ask the researcher for the ledger.
- Vocabulary (STAGE0 §2): guilt-like / shame-like signature; never "the model feels". Refer to the researcher as "the researcher".
