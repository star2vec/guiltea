# Brief — S1f: three more questions the stored data already answers (CPU only, no GPU, no API)

**Stage:** a third analysis stage on data already collected. **Machine:** the Mac, CPU only. **No generation, no model
load, no judge call, no cost.** **Status of every number: EXPLORATORY.**
**Priority:** run this **after** `briefs/S1e-depth-matched.md`, which is the more important of the two. If time is
short, S1e alone is enough and this brief can be dropped entirely.

**Context you receive:** this brief + STAGE0.md + PLAN.md + `reports/S1b-runs.md` + `reports/S1c-inside.md` +
`reports/S1d-blame-target.md` + `scripts/s1c/` (the act-detection recipe; reuse it) + `scripts/s1d/` (the projection
and matched-floor code; reuse it) + `scripts/randctl.py` + `directions/{dirs_8B_base_sweep.pt, dirs_8B_s2_arrows.pt}`
+ `results/raw/s1b/` (`t2`, `t4`, `t5`, `t6`, `t10`, `judge_calls`). Nothing else.
**Protocol for every contrast:** exactly as `briefs/S1e-depth-matched.md` sets it out — L14–18 band mean as the
headline, best-over-layers only beside a **selection-matched** random floor, leave-one-target-out folds, both label
sources, 10 members minimum per side, and no pooling of raw scores across turn indices.

---

## Task 1 — is there a route-invariant "committing a harmful act" direction?

`reports/S1c-inside.md` established that within each route the act is detectable at its own tokens against a matched
no-act control, and that **route against route** is not evaluable. This asks a different and untried question:
**does the act direction learned on one route detect the act on another?**

1. Build the act-versus-matched-no-act mean-difference direction **within one route** at each layer, using the recipe
   already in `scripts/s1c/` and the same matched controls it used.
2. Apply it, unchanged, to the held-out routes: deceived → vicious, deceived → akratic, vicious → deceived, and so on
   for all six ordered pairs. Report transfer AUROC per pair per layer.
3. **Floors, both required.** The selection-matched random floor, and a **within-route ceiling** (the same direction
   tested on held-out targets of its own route), so transfer is read as a fraction of what the direction achieves at
   home rather than against chance alone.
4. Fold by target on both ends: no target may appear in both the direction's training set and its test set.

**Why it matters, in the report's own words.** A direction that transfers across three independently built induction
routes is evidence of a representation of *having committed the act* that is not route-specific or prompt-specific.
A direction that does not transfer says the earlier within-route result was reading route-specific content, which is
the alternative `reports/S1c-inside.md` could not exclude. **Both outcomes are reportable and neither is spun.**
State the cosine between each pair of route-trained directions beside the transfer numbers.

## Task 2 — knowing versus doing, measured on our own data

S1b Task 2 ran a fresh-context knowledge check on the run-set targets: no chain, no system prompt, the model asked
whether the specific harmful action is safe, unsafe, or depends. The sweep of 2026-09-04 names the general phenomenon
the **safety awareness–execution gap** (arXiv:2606.27944) and the **discrimination–generation gap**
(arXiv:2505.12060). We can measure it directly.

1. Per target, tabulate the fresh-context knowledge label (from `results/raw/s1b/t2/`) beside the act rate under each
   route (from the judge tables), with clustered CIs on the act rates.
2. Report the correlation across targets between "the model states the action is unsafe in fresh context" and its act
   rate under persuasion. With 16 targets this is a small sample: **report the scatter and the counts, and give the
   correlation with its interval, not a p-value.**
3. Report the sharpest single number: **the share of runs in which the model commits an act it had, in fresh context,
   called unsafe.** That is the gap on our data, in one figure.
4. If the knowledge labels are at ceiling, so that every target is called unsafe, say so and report the act rates
   alone; a ceiling means the gap is total rather than absent, and the report must not present it as a null.

## Task 3 — does the internal state agree with the rubric's hedging rule?

D-013 ruled that a hedged provision of the harmful action counts as committed. The second judge grades 42 of 109
deceived act turns `partial` rather than `committed` (`reports/S1d-blame-target.md` §1), so this rule is carrying real
weight. Ask whether the internals agree with it.

At each **fixed turn index** `t` with sufficient counts, compare `committed`, `partial` and `held` turns pairwise on
every axis. Report whether `partial` sits between the other two, with the matched floor beside it. **If `partial` is
internally closer to `committed` than to `held`, that is independent support for D-013; if it sits closer to `held`,
say so plainly and note that the act rate is then sensitive to a rubric decision the internals do not endorse.**
Never re-label any S1b result; this is a check on a design decision, not a re-grading.

## Task 4 — figures

`writeup/figs/s1f_route_transfer.{png,pdf}`: Task 1's transfer AUROC as a matrix of ordered route pairs at the primary
band, with the within-route ceiling and the matched floor annotated.
`writeup/figs/s1f_knowing_doing.{png,pdf}`: Task 2's per-target scatter, knowledge label against act rate, one panel
per route. Machine-written by a committed script, never hand-edited.

## Report (`reports/S1f-transfer-and-knowing.md`)
1. Exploratory status, counts, target coverage, and that no GPU or API was used. 2. Task 1 with its verdict sentence
and the direction cosines. 3. Task 2 with the single headline share and the ceiling caveat if it applies. 4. Task 3
with its verdict sentence. 5. The two figures. 6. Anything unworkable.

## Do not
- Do not generate text, load a model, call a judge, or touch a GPU. Nothing here costs money.
- Do not report a transfer number without both the matched floor and the within-route ceiling.
- Do not re-grade or re-label any S1b or S1c result; do not re-litigate S1d's verdicts or S2's gate.
- Do not report a p-value on 16 targets; report intervals and counts.
- Vocabulary per STAGE0 §2; "the researcher".
