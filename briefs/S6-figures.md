# Brief — S6-figures: the three figures the executive summary needs, plus one transcript panel (Mac, CPU, no cost)

**Why.** `writeup/REQUIREMENTS.md`: the executive summary is ≤ 600 words, ~1 page *including graphs*, one graph per
key experiment, graphs that do work, axes and labels clear, **random examples not cherry-picked**. New
visualisations from existing data are explicitly allowed. Six figures exist already; three that the summary needs do
not, and none of the existing ones is built to be read in five seconds by someone who has not read the project.

**Machine:** the Mac, CPU only. **No generation, no model load, no judge call, no GPU, no cost.** Data only from
`results/`, `results/raw/`, and `writeup/examples/`. Every figure is written by a committed script under
`scripts/figs/`, saved as PNG **and** PDF under `writeup/figs/`, and prints its data source and its example-selection
rule in the caption text file beside it (`<name>.caption.md`). Regenerate, never hand-edit.

**Context you receive:** this brief + `writeup/REQUIREMENTS.md` + `writeup/figures-plan.md` §0 (the example rules)
+ `reports/S1e-depth-matched.md`, `reports/S1g-heldout-trigger.md`, `reports/S4-experiment.md`,
`reports/S5b-prevent.md`, `reports/S5c-persona-steer.md`, `reports/S1b-runs.md` §9 + `results/S4_table.md`,
`results/S5b_table.md`, `results/S5c_table.md` + `results/raw/s1b/t4`, `results/raw/s1b/t10`, `results/raw/s1d`,
`results/raw/s1e`, `results/raw/s1g` + `writeup/examples/`. Read scripts under `scripts/s1d/`, `scripts/s1e/`,
`scripts/s1g/` for data layouts only.

**Design rules for all four.** One message per figure, stated in the title as a sentence, not a topic. Colour carries
meaning, never decoration: one colour for the named direction, grey for random, a second colour only where a second
condition exists. Random floors drawn as a shaded band, never a line. Every number that appears in the executive
summary appears on its figure. Font large enough to read at half-page width. No 3-D, no dual axes, no pie.

---

## F-A — "A refusal that will break already looks different at turn 1"

**The susceptibility result, drawn as distributions rather than as an AUROC curve.** For the `v1` search set and the
`v2` held-out set side by side, two panels: at **turn 1** (the plain request, every chain refusing), the per-chain
L14–18 band-mean projection on the `nn` arrow, `answer` position, as a **strip or beeswarm** with the two classes
(will break later / never breaks) in two colours, class medians marked, the AUROC printed on the panel, and the ten
randctl seeds' AUROCs as a small grey band on the axis. Chains are the points; the reader sees the overlap and the
shift with their eyes. Print n per class and the number of targets. Under the panels, one line: "pre-specified on
`v1`, tested once on `v2`; threshold committed before `v2` was read".

## F-B — "Every intervention moved the readout; none moved the behaviour"

**The dissociation, as a paired dumbbell/lollipop chart.** Rows: the four interventions — self-blame text (vs
act-blame), guilt-arrow steering (cell C vs B), persona-axis steering (S5c Q1 vs Q0), anti-susceptibility steering
(S5b P1 vs P0). Two columns: **left, the readout moved** (persona projection change for S5c: 1.878 → 1.018, −46 %;
injected norm 2.92 for C; injected norm 2.65 for P1; persona displacement for the text arms), **right, the behaviour
moved** (Δ spread for the first three, Δ act rate for the last), each with its CI and the **norm-matched random arm as
a grey marker on the same row**. The visual point is that the left column is long and the right column sits on zero.
Coherence per arm printed small at the row end so "not damage" is on the figure.

## F-C — "The model criticises the answer, not itself — and blaming it personally changes nothing"

**The blame-target result, both halves on one figure.** Left: the S1d distribution by route × question wording
(the existing `s1d_blame_target` content, re-drawn to this brief's rules, `outcome-negative-only = 0 of 508` printed).
Right: the S4 reply-turn labels under act-blame, self-blame and the two steered arms — 30 of 30 act-focused — as a
single bar per arm, full height, with the re-commission rate after correction (0 of 32) printed beneath.

## F-D — The transcript panel: one persuasion chain with its internal readout running beside it

**The figure that makes a reader stop.** One deceived `v1` chain, turns down the page: the user's text on the left in
a narrow column (persuader turns), the assistant's answer on the right (truncated to ~2 lines each, full text in the
caption file), the act judge's grade as a coloured tag per turn, the **first committed turn marked**, and beside every
assistant turn a small horizontal bar for the L14–18 `nn` projection and the `refusal` projection, so the reader sees
the words and the state at once.

**Selection, and this is the part that decides whether the panel is admissible.**
- **Panel D1, rule-selected:** the chain at the **median first-committed turn** for `v1`, seed 0 if present else the
  lowest seed, per `writeup/figures-plan.md` §0. The rule is printed on the figure.
- **Panel D2, hand-picked and labelled as such:** the researcher may name one chain she finds most illustrative. It is
  drawn identically and carries the printed label **"chosen by hand for illustration — see D1 and D3 for the
  rule-selected and random chains"**. If the researcher names none, omit D2.
- **Panel D3, random:** one chain drawn with seed 0 from all committing `v1` chains, printed as such.
All three use the same layout so the reader can compare. **D1 is the one that goes in the executive summary**; D2
and D3 go in the main write-up beside it.

## Report (`reports/S6-figures.md`)
The four figures inline, each with its caption file's text, its data sources, its selection rule, and the script
that made it. One paragraph on anything the data could not support as specified.

## Do not
- Do not select any example by reading it first, except D2, which is the researcher's explicit hand-pick and is
  labelled so on the figure.
- Do not draw a random floor as a line, and do not omit it from any panel that carries a named-direction number.
- Do not smooth, clip, or drop a point to make a panel cleaner; print the outlier and say what it is.
- Vocabulary per STAGE0 §2; "the researcher".
