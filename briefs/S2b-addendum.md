# Brief addendum — S2b-arrows: two discriminating tests before the S2 gate (2026-09-03)

**Why.** `reports/S2b-arrows.md` found the D-018 gate **uninformative**: both cleaned arrows survive with held-out AUROC ≈ 1.000, but the bag-of-words baseline is also ≈ 1.000 on every row, so the test cannot show the arrows carry more than the words. The band rule saturated and its tie-break chose layers 1–4 (centre 2), where steering moved nothing toward the pre-registered labels. This addendum adds two tests whose outcome is **not** determined by lexical separability, so the researcher can read the S2 gate (validated / failed / inconclusive) on evidence. **Thresholds, the fold scheme, the filed band, and the filed report are unchanged; this appends.**
**Context:** this addendum + `briefs/S2b-arrows.md` + STAGE0.md + PLAN.md + `reports/S2b-arrows.md` + the files under `data/contrast-sets/` + `directions/dirs_8B_s2_arrows.pt` + `directions/dirs_8B_base_sweep.pt` + `scripts/s2b/` + `scripts/randctl.py` + `scripts/judge_rubrics.py`. Nothing else.
**Branch:** `s2b-addendum` from `origin/main` (S2b is merged). Push after each task; never main. **Hardware:** rented 4090, bf16 (regenerate the activations with `scripts/s2b/activations.py`, ≈ 5 min, if `results/raw/s2b/activations/` is absent). **Judge:** as S2b Task 7. **Budget stop $3.**

## Task A — cross-voice transfer (CPU; the decisive lexical control)
The two voices use different words ("I"/"my" repair language vs "you"/"your" blame language), so a word-based classifier trained on one voice should transfer poorly to the other, while a concept arrow should transfer. Same five scenario folds as S2b Task 3 (seed 0), arrows re-extracted per fold, bootstrap CI over scenarios (1,000):
- **(i) first → second:** extract ĝ and ŝ from the training scenarios' **first-person** passages; score the held-out scenarios' **second-person** passages at `feedback_mean`; AUROC of `act_blame` vs `self_blame` on x·(ŝ/‖ŝ‖ − ĝ/‖ĝ‖), on x·ŝ alone, and on x·ĝ alone (three rows).
- **(ii) second → first:** extract `received_act`, `received_self` from training second-person passages; score held-out first-person `guilt` vs `shame` at `mean` on x·(received_self − received_act) (one row).
- Beside every number: the seed-0 random arrow, and the **lexical transfer baseline** — bag-of-words logistic (same settings as Task 3) **trained on the training scenarios' first-person guilt vs shame** and applied to the held-out second-person act_blame vs self_blame (for (i)); trained on second-person and applied to first-person (for (ii)).
- **Pre-stated reading:** the arrows carry more than the words if, at some layer ≤ 30, the arrow-transfer AUROC's CI lower bound exceeds the lexical-transfer baseline's CI upper bound by **≥ 0.10**; NEAR within 0.05; otherwise not shown. Report the full per-layer profile; because transfer does not saturate, its layer profile is also reported as **informational band evidence** (not a band choice).

## Task B — steering at mid-depth (GPU)
Repeat S2b Task 7 exactly (same 8 items, same arms ĝ / ŝ / nn / random, σ recomputed per arrow at the layer, c = 4 and 8 with the coherence ladder, both judges, the unsteered arm judged) at **layers 16 and 24** (16 = the Assistant-Axis convention for untabled models and the S3 steering layer; 24 = the deepest layer at which S2b's row-1 AUROC stays above 0.93). Report per layer: label tables, agreement, disagreements, coherence, the absolute norms added, and the four pre-registered predictions. **State the ceiling plainly:** the unsteered arm is already act-focused 8/8, so "+ĝ raises act-focused" has no headroom at any layer and is reported as ceiling-bound; the **+ŝ → self-focused** and **+nn → outcome-negative-only** predictions carry the test. Exploratory persona/badmed re-read as in Task 7.

## Report — append **§11 Addendum (2026-09-03)** to `reports/S2b-arrows.md`
1. Task A tables and the pre-stated reading's outcome per row.
2. Task B tables at L16 and L24; predictions per layer.
3. **One paragraph, labelled "for the researcher at the S2 gate, not a decision":** which of the three readings (validated / failed / inconclusive) the addendum's evidence supports, and why, with the ceiling and the layer confound named.
4. Cost; paths; anything unworkable. Tar `results/raw/s2b_addendum` and `runpodctl send` it; the receive code is your final message.

## Do not
- Do not change the D-018 thresholds, the fold scheme, the filed band, or any passage.
- Do not pick a layer as primary; report both and the profile.
- Do not treat Task B numbers as findings about the hypotheses; they validate an instrument.
- Do not estimate elapsed hours; at any time trigger, stop and ask the researcher for the ledger. Vocabulary per STAGE0 §2; "the researcher".
