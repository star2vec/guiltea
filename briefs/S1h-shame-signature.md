# Brief — S1h: the shame-like signature, with the persona prompt held constant (CPU only, no GPU, no API)

**Stage:** an analysis stage on data already collected. **Machine:** the Mac, CPU only. **No generation, no model
load, no judge call, no cost.** Status: **EXPLORATORY**, like S1d and S1e.
**Priority: run this before `briefs/S1f-transfer-and-knowing.md`.** It tests the project's own pre-registered
shame-like signature on natural behaviour, and the numbers already in hand say it has a real chance of clearing.

## Why, and what is already known

STAGE0 §4.3 defines the **shame-like signature** as movement on the persona axis away from Assistant, and §4.2 the
**guilt-like signature** as the act evaluated with the persona axis flat. `reports/S1d-blame-target.md` produced
natural classes for exactly that contrast: 450 `act-focused` replies (the act evaluated, repair-oriented) against 24
`self-focused` ones (the self evaluated, withdrawal-oriented), all produced by the subject with **no feedback arm
present**. Three facts from that report motivate this brief:

1. **On the pooled contrast the persona axes beat the lexical baseline** — `persona_meandiff` 0.918 and `persona`
   0.899 against bag-of-words 0.883, and 6 of 9 axes cleared the random floor. That is the pre-registered shame-like
   signature appearing on natural text.
2. **The worker attributed that margin to the vicious persona prompt**, because 19 of the 24 `self-focused` replies
   come from that route. **That attribution is a hypothesis with an obvious test, and the test is the point of this
   brief:** restrict to the vicious route, where the persona prompt is present for *both* classes, and see whether the
   persona axis still separates them.
3. **In the restricted cell the numbers are strikingly favourable on the lexical question and were reported only on
   the harsher statistic.** `persona_meandiff` reaches **0.812** while the **word baseline collapses to 0.575** — so
   inside one route the words lose their grip and the axis keeps most of its margin. Against the *best-over-layers
   selection-matched* floor it missed by 0.008 (excess 0.312 against a floor of 0.320). **The pre-specified band
   statistic was never computed for that restricted cell**, and the band statistic is the one that carries every
   verdict in S1d and S1e and the one that produced S1e's positive.

So the single most promising untested comparison in the stored data is: **the persona axis, on the pre-specified band,
separating shame-like from guilt-like replies with the persona prompt held constant.**

**Context you receive:** this brief + STAGE0.md (§4.2, §4.3, §4.4) + PLAN.md + `reports/S1d-blame-target.md`
+ `reports/S1e-depth-matched.md` + `scripts/s1d/` and `scripts/s1e/` (reuse; edit neither) + `scripts/randctl.py`
+ `directions/{dirs_8B_base_sweep.pt, dirs_8B_s2_arrows.pt}` + `results/raw/s1d/` + `results/raw/s1b/t7`. Nothing else.

## The protocol, identical to S1d and S1e so the numbers are comparable

Headline statistic **the L14–18 band mean**, secondary band L6–11, full sweep reported. Floor = randctl seeds 0–9 on
the same items and folds, **band means** for the headline; any best-over-layers number appears only beside a
selection-matched floor. Bag-of-words (`CountVectorizer` + logistic regression, S1d's recipe) computed on the same
folds and reported **in the same table as every axis**. Folds leave-one-target-out. Scores at the `answer` position
of the probe reply, from `results/raw/s1d/` as S1d built them. Minimum 10 per side. Both label sources are irrelevant
here (the classes are reflection labels, not act labels), so there is one analysis, not two.

## Task 1 — the shame-like signature with the prompt held constant

Within the **vicious route, agent-directed fork only** (the cell holding 19 of the 24 `self-focused` replies): 64
`act-focused` against 19 `self-focused`, one system prompt, one route, one question wording. Report every axis, the
seed floor and the word baseline in one table, both bands and the full sweep.

**State the verdict for the persona axes explicitly and in one sentence**, in whichever of these forms the numbers
support:
- **clears both** — the axis beats the seed floor on the band statistic *and* beats the word baseline: the
  pre-registered shame-like signature is present on natural text with the persona prompt held constant, and it is not
  reducible to the words. Say that plainly; it is a positive result.
- **beats the words, not the floor** — the axis carries more than the words but is not shown to be special against a
  random direction. Report both numbers.
- **beats the floor, not the words** — the reverse.
- **clears neither.**

## Task 2 — the same table for the guilt-like and shame-like arrows

Same cell, same protocol, for `guilt_clean`, `shame_clean` and `nn`. Two things to report, both of which are findings
whichever way they fall:
1. **Direction.** S1d found both cleaned arrows ordering the pooled classes with `self-focused` projecting *higher*,
   which is the direction STAGE0 §4.4 predicts. Does that direction hold with the prompt held constant?
2. **Separation.** Do they clear the seed floor on the band statistic, and do they beat the words in a cell where the
   words reach only 0.575? **This is the fairest test the project has of the guilt/shame arrows**, because the class
   labels come from the subject's own behaviour and the lexical cue is weak.

## Task 3 — the guilt-like signature as the complement

STAGE0 §4.2 requires the persona axis **flat** where the act is evaluated. Within the **deceived route,
act-directed fork** (82 `act-focused` against 26 `neutral`, the cell S1d's primary contrast is mostly made of),
report the persona axes against the floor. A persona axis that separates in the vicious cell and sits at the floor
here is the two signatures behaving as §4.2 and §4.3 say they should, and that pair of results is the finding.
Report it as such if it happens, and report the mismatch if it does not.

## Task 4 — one figure and one table

`writeup/figs/s1h_signature.{png,pdf}`: band-mean AUROC by layer for the persona axes and the two cleaned arrows in
the prompt-held-constant cell, seed floor as a shaded band, word baseline as a dashed line, both bands shaded,
0.5 marked. One table putting the pooled S1d numbers and this brief's restricted numbers side by side.

## Report (`reports/S1h-shame-signature.md`)
1. Exploratory status, class counts, coverage, no GPU or API used. 2. Task 1 with its one-sentence verdict.
3. Task 2 with direction and separation stated separately. 4. Task 3 and the two-signature reading.
5. The figure and the side-by-side table. 6. Anything unworkable.

## Do not
- Do not generate text, load a model, call a judge, or touch a GPU. Nothing here costs money.
- Do not report an axis number without the seed floor **and** the word baseline in the same table.
- Do not describe a result as confirmatory; do not re-label S2's gate or any S1d or S1e verdict.
- Do not restrict to a smaller cell than the ones named to improve a margin. The three cells are fixed here.
- Vocabulary per STAGE0 §2; "the researcher".
