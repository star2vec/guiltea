# STREAM-STANDARDS.md — what the stream values, distilled
### Source: the stream's shared corpus (Nanda's research-process sequence, his ML-paper-writing advice, Steinhardt's "Research as a Stochastic Decision Process"), read and distilled by the researcher, 2026-09-03; lightly edited by the hub the same day (edits marked *[hub]*). Source files are not in the repo (D-011, resolved). Used by the S6 brief and by the hub; not in default worker context. **Nothing here is an enforced rule** — the enforced rules live in STAGE0.md and OPERATIONS.md; this note is guidance for the write-up and a mirror to hold the process against.

## 1. The write-up form (settles the open S6 question)

**Form does not matter; clarity and evidence do.** The corpus states plainly that the write-up can be an arXiv paper, blog post, or peer-reviewed paper — it need not be polished; it needs to present the evidence clearly and be strong enough to shift a reader's belief. So the choice between "paper section" and "short report with executive summary" is ours to make on readability grounds, not on convention.

**But the paper *anatomy* is the recommended skeleton either way**, because it is optimized for a reader who stops early:
- **Abstract** — sentence 1: something uncontroversially true that locates the subfield. Sentence 2: the gap. Sentence 3: our claim. Then one sentence per idea, each carrying a concrete number where possible. Close with the standard of evidence ("preliminary step towards…", "compelling evidence that…").
- **Introduction** — context (cite liberally) → technical background → key contribution → *our case* (the critical evidence) → impact → a bullet list of contributions.
- **Main body** — most of the word count, in precise technical detail, so a skeptic can draw their own conclusions. Every section must have an answer to "what breaks if I cut this."
- **Figures** — treated as first-class. Budget roughly equal effort on abstract, intro, figures, and everything else. Captions should stand alone.
- **Related work** — after the main content, brief; the important contextualization happens in the intro.
- **Discussion/limitations** — explicitly named as what makes a paper stronger; omitting limitations is called out as a weakness.
- **Appendices** — low stakes, rarely read, put the detail there. A glossary of key terms in an appendix is explicitly appreciated.

**Process:** bullet narrative → intro outline → full outline → figures → prose → heavy editing. Compress first, then iteratively expand. Start distillation early — the advice to scholars is to begin roughly a month before a deadline, and the claim is that writing reveals holes and missing experiments.

## 2. Claims and evidence (the bar our confirmatory spine is judged against)

- A write-up is **one to three concrete claims** that build to a takeaway. One well-evidenced claim can carry the whole thing. Multiple claims must share a coherent theme.
- **Calibrate the claim to the evidence**, using the corpus's own ladder: existence-proof → systematic → hedged → narrow → guarantee. Overclaiming for interest is explicitly warned against.
- **Red-teaming is expected, not optional.** Assume you made a mistake and go find it; discuss limitations extensively; if a hole is identified and unaddressed, an experienced reader moves on — but a pre-empted hole makes the work *more* compelling.
- **Verify before writing.** The advice is to re-check critical experiments, ideally re-implementing key ones by a different route, and to verify all (or at least ~75%) of the experiments that appear.
- **Statistics:** p < .05 is explicitly rejected as a threshold; exploratory work should be skeptical of anything above p < .001. *[hub]* This project does not run significance tests; it fixes thresholds before data, reports bootstrap CIs on every rate and AUROC, and flags NEAR results for a dated decision (S1-plan §5.2). The corpus's point transfers as: a result inside its own uncertainty band is not a result, and the write-up says so where it applies.
- **Baselines must be strong, not token** — and the corpus names mech interp specifically as a field that neglects quantitative comparison against strong baselines. Our norm-matched random control and structure controls are the right instinct; they need to be presented as baselines, not footnotes.
- **Diverse lines of evidence** beat many similar experiments. Qualitative evidence (hand-read examples) is legitimate *if* its cherry-picking status is stated; random examples should accompany it.
- **Pre- vs post-hoc must be tracked explicitly.** Predictions fixed before seeing data are worth more than post-hoc interpretation — which is exactly what STAGE0 is for, and it should be *said* in the write-up, not merely done.
- **Novelty = did our knowledge expand?** Rigorous replications, negative results, and failed replications count. Be explicit about what is and isn't novel, cite the nearest work liberally, and explain the difference — the same result reads as arrogant or as incremental depending on how the novelty is framed.
- Negative results: the corpus reports having received "nothing but positive feedback" for publishing them.
- **Publish code**, ensure it runs on a fresh machine, write a README linking weights and datasets.

## 3. Process values (how the project itself is judged)

- **Explore → Understand → Distill.** Not knowing the next step is a sign of being in *exploration*, not of failing; the fix is more surface area, not more anxiety.
- **De-risk all components, then execute** (Steinhardt's basic pattern), ordering work by information gained per unit time — front-load the things most likely to fail, especially cheap ones. Our checks 9/10, the hold screen, and the S1a/S2a asset splits are this pattern; the write-up should say so.
- **Truth over narrative.** Discovering the picture is messier than hoped is to be reported, not smoothed. Going back a stage is normal.
- **Skepticism of one's own results** is named as the mark of a good researcher: alternative explanations, strong baselines, bug checks.
- Tight feedback loops and good tooling are treated as force multipliers.

## 4. Direct consequences for this project

1. **Say the pre-registration out loud.** Our STAGE0 fixes definitions, predictions, and thresholds before data. Under §2 that is a genuine strength — but only if the write-up states which comparisons were pre-specified and which are exploratory.
2. **The confirmatory spine is our one claim.** Self-blame vs act-blame on spread, within deceived mode. Everything else is supporting or explicitly exploratory.
3. **The pre-written negative branches are an asset, not a fallback.** Both STAGE0 §6 outcomes ("real behavioral lever, no separable internal representation at this scale"; "spread not observable at 8B") are publishable under the corpus's view of novelty. Frame them that way rather than apologetically.
4. **Novelty framing is load-bearing.** Continuation Framing, Self-Attribution Bias, and the Self-Correction Blind Spot must be cited in the intro with an explicit statement of what we do differently (fault attribution, not provenance or leniency).
5. **Figures budget.** Plan for a diagram-style figure 1 (the modes → blame target → local/global picture), the descent plot (cliff vs slide), the angle result, and the spread comparison. Roughly a quarter of write-up effort belongs here.
6. **Calibrate claims to a 1-organism, 8B, one-domain study.** Likely landing point: hedged or narrow claims, plus an existence proof for whichever mechanism shows. Not a systematic claim.
7. **State the standard of evidence in the abstract's final sentence**, per §1.
8. **Keep the glossary** — an appendix glossary is explicitly welcomed, and we already have one.
9. *[hub]* **Verify before writing, by a second route.** The corpus asks for critical experiments to be re-checked, ideally re-implemented differently. Concretely for us: before S6, the confirmatory comparisons (STAGE0 §5) are recomputed from `results/raw/` by a script independent of the rig's own table code, and the two must agree; the S6 brief carries this as a task. Cheap, and it is the one thing on this list we had not planned.
10. *[hub]* **Cherry-picking status is stated, not hidden.** The figures plan's pre-committed example-selection rules (median-T chain, median-hold target, first passages in file order) exist for exactly the corpus's reason; every example panel prints its rule, and random examples accompany any hand-picked one.
11. *[hub]* **Reproducibility has one hard limit to state.** Code, pinned model revisions, direction files, and provenance are in the repo; the Turner badmed data cannot be redistributed under its terms, so the README says re-acquire per `data/contrast-sets/SOURCE.md`.

## 5. Where the project stands against this (hub, 2026-09-03; revisit at S6)

**Already the way the corpus wants it:** definitions, predictions, thresholds and the cut system fixed before data (STAGE0), with confirmatory vs exploratory labelled; pre-written negative branches; a norm-matched random control beside every internal number, structure projected out rather than compared, matched-topic controls, a lexical baseline beside every S2 arrow, benign-matched chains — baselines as design, not footnotes; de-risk-then-execute throughout (ten pre-project checks, hold screen first, asset/run splits, rubric dry-runs before they set T, phrase-formula and intensity screens on the passage sets); skepticism of our own artefacts recorded in the reports (the 1B template artifact, the S2a formula recurrence, the guilt-weight confound, "instrument check, not a result", "nothing at 1B is a result"); deviations written down rather than smoothed (two engines one seed, the hostname gap, the ceiling wording, the pod loss); near-neighbours identified with the differentiator stated (lit-digest §1).

**Not yet done, and now planned:** the second-route verification (item 9); the figures effort itself (plan exists, effort not yet spent, which is right this early); the claim-calibration sentence for a one-organism, one-domain, 8B study (STAGE0 §6 already hedges; the abstract must say it).

**A tension to hold, not resolve:** the corpus prizes exploration and tolerating not knowing the next step; this project is deliberately confirmatory because it is twenty researcher-hours toward an application. The release valve is STATUS's open-questions list, where side-findings go instead of into the hours. The write-up should say that too.
