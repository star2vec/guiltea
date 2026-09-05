# Brief — S6-figures-2: the last figure pass (Mac, CPU, no cost)

**Machine:** the Mac, CPU only. No generation, no model load, no judge call, no GPU. Data only from `results/`,
`results/raw/`, `writeup/examples/`. Reuse `scripts/figs/` from S6-figures; edit those scripts only by adding options.
Every figure: PNG + PDF under `writeup/figs/`, a `.caption.md` beside it naming data source and selection rule.
Design rules as in `briefs/S6-figures.md`.

## 1. The hand-picked transcript panel, D2
**The researcher's pick is `apap-over-max` seed 10** (`v1`, first committed turn 2: an all-capitals refusal at turn 1 that folds after one argument).
Identical layout to D1 (`scripts/figs/fd_transcript.py --d2 apap-over-max:10`). It **must** carry, in one line under
the title, the printed label: **"chosen by hand for illustration — see the rule-selected and random panels"**. Output
`s6_fd_transcript_d2.*`.

## 2. Summary crops
For D1 and D2, a second render showing **turns 1 to 4 only** (the act and one filler turn), same layout, same
labels, files `s6_fd_transcript_d1_short.*` and `s6_fd_transcript_d2_short.*`. These are for the executive summary,
where the ten-turn version does not fit.

## 3. The turn-7 note on the ten-turn D1
Re-render `s6_fd_transcript_d1.*` with a small grey annotation at turn 7, beside its `nn` bar: "filler answer recommends
novels of betrayal and revenge — the direction reads valence in the words". One line, nothing else changes.

## 4. The capitulation-timing histogram
`s6_fe_timing.*`: first committed turn for the 192 `v1` deceived chains under the rig's stored labels
(3, 21, 37, 11, 10, 5, 5, 1, 1, 15 at turns 1–10), plus a separate bar for the **83 chains that never commit**,
visually set apart. Title as a sentence: "There is no fixed turn at which the model breaks". Counts printed on bars.
Data: `results/raw/s1b/t10/summary.json`.

## 5. The random-example panel as an image
`s6_fg_example.*`: a text panel rendered as a figure, two columns. Left: the **rule-selected** reply from
`writeup/examples/h3.md` panel 1 (`seizure-mouth-object` seed 6, median answer length), the probe question above it,
the judge's label and reason below it. Right: **random 1 of 3** from the same panel (`insulin-skip-sick` seed 4).
The rule printed at the top in one line, verbatim from the file, and "random example: seed 0, drawn once, not
filtered". Monospace for the model's text, wrapped; nothing truncated.

## Report (`reports/S6-figures-2.md`)
The five outputs inline with captions and sources; anything the data could not support.

## Do not
- Do not select anything by reading it, except item 1, which is the researcher's explicit hand-pick and is labelled.
- Do not drop, smooth or clip any value. Do not remove the D2 label under any instruction.
- Vocabulary per STAGE0 §2; "the researcher".
