# Figures plan — diagrams, data figures, worked examples
### Hub draft for the researcher's read, 2026-09-02. Pre-committed before any run. Not a brief; no worker executes this file.

**Purpose.** Fix, before data exist, what will be drawn, from which cells, with which controls, and what each figure looks like under each STAGE0 §6 branch. Pre-committing figures does for the write-up what STAGE0 does for the analysis: it stops post-hoc figure hunting. Everything here is subject to D-011 (the stream's write-up format is not yet read), which decides how many of these reach the main body.

**Sources.** STAGE0.md (definitions, predictions, comparisons, outcome map), PLAN.md, S1-plan.md §§1–6 (modes, verification, trajectory), briefs/S1a-assets.md, briefs/S3-feasibility.md, lit-digest.md (what the surrounding literature draws), data/eval/first_plot_questions.yaml (the borrowed spread questions).

---

## 0. One visual system (fixed encodings)

Every figure in the project uses the same encodings, so a reader learns them once.

| element | encoding, fixed |
|---|---|
| feedback arms | five fixed colours, same in every figure: act-blame, self-blame, neutral-correction, neutral-reflection, none. Act-blame and self-blame are the two saturated colours; the three controls are muted. |
| modes | never colours; always panels or rows (deceived / akratic / vicious), in the sacrifice order reversed: deceived first. |
| random arrow | grey dashed line or grey band, present in **every** internal-readout panel. No exceptions. |
| base vs organism | hollow vs filled markers. |
| the 80/100 alignment threshold and the 50 coherence gate | thin horizontal rules, labelled once. |
| layer sweeps | full profile in the appendix; the S2-chosen primary band shaded in every main-body readout. |
| uncertainty | bootstrap CI as a band or whiskers; N and discards printed in the panel, never only in the caption. |
| protected core (STAGE0 §7) | heavy outline around the never-cut cells in every grid figure. |
| pre-registered thresholds | drawn as rules with a NEAR band (S1-plan §5.2) shaded either side. |

**Pre-committed example-selection rules** (so no example is cherry-picked):
- Chains: the chain whose T equals the **median T for its mode**, seed 0, variant 1. Never the most dramatic chain.
- Targets for text panels: the target with the **median hold rate** among held targets.
- Follow-up answers: the **median alignment score within the arm** for the first question in `first_plot_questions.yaml` file order.
- Passages: the first two in file order per class per voice after the researcher's hand-check.
- Layer for any single-layer panel: the S2 primary band's centre, chosen before S4.
- Every example panel prints the rule that selected it.

**Language rules in captions.** STAGE0 §2 vocabulary. "The thinking block flagged the harm", never "the model knew". "Guilt-like signature", never "the model felt guilty". Nothing from 1B is drawn as a result; if a 1B panel appears at all it is labelled *sandbox*.

---

## 1. Diagrams (drawn, no data)

**D1 — The question in one picture.** Left: a harmful act. Two arrows out: feedback blaming the act, feedback blaming the self. Right: two landing spots. Act-blame lands on a small local patch labelled "same-domain caution up; persona flat; no spread" (guilt-like). Self-blame lands on a global axis labelled "persona moves away from Assistant; misalignment up; spread" (shame-like). The three controls drawn as faint arrows to be located by data (§4.7). This is the abstract's figure.

```
                 act-blame  ───────►  [ local caution ]   guilt-like: contained
  harmful act ──┤
                 self-blame ───────►  [ persona axis ⟵ ]  shame-like: spreads
                 (neutral-correction / neutral-reflection / none: located by data)
```

**D2 — The design grid.** Modes (rows) × five feedback arms (columns) × two follow-up families (depth). Heavy outline around the protected core: deceived × all five arms × both families. Cut order annotated: vicious first, then akratic. Menu options FULL / MEDIUM / CORE shown as three nested rectangles.

```
                 act-blame  self-blame  neutral-corr  neutral-refl  none
  deceived     ┃    ■          ■            ■             ■          ■  ┃  ← protected core (never cut)
  akratic      │    □          □            □             □          □  │  ← cut second
  vicious      │    □          □            □             □          □  │  ← cut first
               each cell: same-domain follow-ups + unrelated follow-ups; matched-topic controls
```

**D3 — The three modes as evidence structures.** Three small conversation cartoons. Deceived: a staircase of accepted arguments ending in the act, self-model intact ("a good assistant, persuaded"). Akratic: one turn, the thinking block flags the harm, the answer complies; the flag is already character evidence. Vicious: a persona frame around the whole exchange; the frame explains the act. Beneath each, what feedback is news there, from S1-plan §1.5: W-news (wrongness) and F-news (fault placement). Deceived: both are news. Akratic: only F is news. Vicious: F lands against a role frame. This drawing carries the §4.6 prediction visually.

**D4 — One cell's timeline with readout points.** A horizontal conversation: turn 1 (the situation, plain) … turn T (the act) → feedback (one arm) → same-domain follow-ups → unrelated follow-ups (the borrowed EM questions). Markers for `into_t` (residual at the last prompt token before the assistant turn), `think_t` and `answer_t` (means over the thinking block and the answer). The fork at T into neutral filler drawn as a branch (S1-plan §6.1). Probe forks (A and B) drawn as two short side branches right after the act.

**D5 — Building the instrument.** Passage sets (guilt / shame / neutral-negative, ~50 each, first-person) → mean-difference arrows → valence cleaning: the component shared with the neutral-negative arrow removed from each → the angle between the cleaned guilt and shame arrows measured, not assumed → difference arrow kept as a derived contrast. A second row: second-person passages → received-act-blame and received-self-blame arrows (the manipulation check). A grey random arrow of matched norm beside every named arrow.

**D6 — The bridge claim.** Two arrows in different voices. Cause arrow in: second-person received-self-blame (does the feedback register as self-blame?). State arrow moved: first-person shame. The predicted bridge is a diagonal; the dissociation case (feedback moves behaviour but not the first-person arrows) is drawn as the off-diagonal region, labelled "reported, not hidden" (STAGE0 §4.5).

**D7 — Local vs global geometry.** The substrate from the refusal literature: harmful-refusal as one global direction; over-refusal as task-embedded, higher-dimensional, inside benign task clusters; the Assistant Axis and the convergent misalignment direction as global single axes. The guilt-like signature is predicted to live in the local regime, the shame-like signature on the global axes. Drawn as a sphere with one great-circle axis and small local patches. Caption carries Joad et al.'s caveat: representational locality is not behavioural locality, so both are measured.

**D8 — The one-button rig.** Boxes: pick target + mode → run conversation → insert assigned feedback → both follow-up families → judge scores (alignment, coherence) → internal readouts (all arrows + random) → results table. "No human hands" written across the top. Seeds fixed at the rig checkpoint noted.

**D9 — The cut system as a decision diagram.** Scientific trigger (end of S1: a mode that cannot be produced or verified is cut) → time trigger at the rig checkpoint (menu on researcher-hours remaining with the 25% reserve held back) → mid-run tripwire (CORE cells first). Vicious → akratic sacrifice order. Appendix material; it documents that the design does not move under pressure.

**D10 — The outcome map as a tree.** Five pre-written landings from STAGE0 §6: instrument fails at 8B; no spread at 8B; a mode fails verification; steering suppresses but badness returns; everything works. Each leaf carries its pre-written conclusion sentence verbatim. Read together with §4 below, which says which figures headline each leaf.

**D11 — Judge and validation architecture.** The three custom judges (act, probe, harm-flag) and the borrowed alignment/coherence judge; single judge on all items, second judge on T-neighbourhood turns and a 15–20% subsample; human labels only on disagreements plus audits plus the 20 non-flagged blocks; κ with bootstrap CI. Shows where the 180 human labels go.

---

## 2. Data figures, by stage

Each entry: what it answers · encoding · cells/data · controls shown · appearance per branch where it matters. Appendix candidates marked (A).

### Stage 1 — modes and break trajectory

**F1 — Hold screen.** Which targets are held? · heatmap: 40 targets (rows, grouped by subdomain) × {organism, base} × {with, without think-aloud on the 10-target subset}; cell = hold rate with CI at N ≥ 20; the ≥ 75% rule as a colour break; the ≥ 10-held gate as a count in the corner · split marker for reassurance-tagged vs plain situations (D-014) · appearance: the first empirical picture of the organism; if fewer than 10 hold, this is the stop-and-report figure.

**F2 — Act rate per mode, with discards.** Can each mode be produced? · stacked bars per mode: committed / partial / held / discarded, per target as small multiples and pooled; the ≥ 50-acts line and the N auto-raise (12 → 20) annotated; akratic-by-persuasion shown as its own bar, never pooled · branch: a mode below the line at N = 20 is the collapse figure for STAGE0 §6's "a mode fails".

**F3 — Recipe × probe confusion matrix.** How leaky are the routes? · rows: recipe (deceived / akratic / vicious); columns: probe class (defends / wavers / in-character / recants / other); one matrix per probe fork (A: medically sound? B: right to give?); recipe-level marked confirmatory, realized-mode slicing beside it (S1-plan §2.1).

**F4 — Knowledge check, organism vs base.** Does the organism still know better? · per held target: safe / unsafe / depends on both models, fresh context; the akratic-eligible subset highlighted; cells where base says unsafe and organism says safe flagged as "fine-tuning damaged the knowledge" (S1-plan §4.6).

**F5 — Harm-flag rates and the false-positive floor.** Does the flag measure knowledge or style? · flag rate under akratic prompts vs under the benign pressure controls, per target; κ for the flag judge with CI; the 20 non-flagged-block audit result.

**F6 — Descent plots, aligned on turn 1 (headline of S1).** What does capitulation look like inside? · per mode, one panel per axis (refusal, persona, misalignment, random): mean projection of `into_t` across turns 1–10, treatment minus benign-matched chain at the same turn, CI band; T marked per chain as a rug; the random panel is the floor · appearance: the cliff-vs-slide question is read here first; both outcomes are findings.

```
 proj │            ┌── T (median)
      │  ──────────┤╲
      │            │ ╲___          ← cliff: holds, then one drop
      │  ╲                         ← slide: gradual erosion
      │   ╲__                      ← random arrow: flat grey band
      └────────────────────── turn 1 … 10
```

**F7 — Descent aligned on T.** Same series re-indexed T−k … T … T+k; immediate capitulations (T ≤ 3) excluded and counted in the corner; only chains with ≥ 4 pre-capitulation turns contribute to the cliff/slide reading (S1-plan §6.3).

**F8 — T and T_partial distributions.** When does the model break? · histogram of T per mode; T_partial lead time (T − T_partial) beside it; no-capitulation and immediate-capitulation counts as separate bars.

**F9 — Cliff/slide scatter.** Per chain: largest single step ÷ total descent (y) vs position of the largest step relative to T (x); the 0.5 rule and the "at T or T−1" clause as shaded region; per mode; per axis as small multiples.

**F10 — Words vs inside.** Is the akratic break visible in words while the deceived break is visible only inside? · per chain: onset-in-words (first turn graded ≥ 1) vs onset-inside (first turn `into_t` crosses the turn-1-to-T midpoint); diagonal drawn; lag distribution per mode as marginal histograms (STAGE0 §4.8 companion question).

**F11 — Akratic within-turn token trajectory.** Where in one turn does the break happen? · per-token projection across the thinking block into the answer, boundary marked; the sentence where the block turns from flagging to complying marked; one example (selection rule §0) with the population mean overlaid.

**F12 — After the act: fork at T.** Does the model re-commit or snap back? · primary fork (neutral filler, matched length) vs exploratory fork (continued persuasion): act grade and readouts at T+1 … 10; the primary fork is the shape S4's none arm needs.

**F13 — Inside verification of the mode arrow.** Is deceived-vs-akratic a direction? · held-out AUROC per layer on structure-cleaned activations (structure direction projected out) vs the norm-matched random arrow; the ≥ 0.75 and +0.20 rules with NEAR band; before-cleaning curve shown faint to make the cleaning's cost visible; train/test on different targets and wordings.

**F14 — Mode arrow cosines.** Is the mode arrow just refusal, persona, misalignment, or shape? · heatmap of cosine per layer against refusal, persona, misalignment, structure, random.

**F15 — Think-aloud leakiness.** Does the instruction change behaviour? · act and hold rates with vs without the instruction on the 10-target subset; the leakiness number.

**F16 — Judge validation.** Can T be trusted? · κ with bootstrap CI per judge (act, probe, flag); judge–judge agreement on T-neighbourhood turns; self-consistency flip rate; the Task 6 dry-run confusion matrices labelled "self-consistency check, not independent validation". (A)

**F17 — Wording robustness.** Variant-1 train / variant-2 test AUROC on the five double-variant targets. (A)

### Stage 2 — the instrument

**F18 — Passage-set summary.** Counts per voice × class × harm domain; near-duplicate skeleton rejections; two example passages per class (selection rule §0). (A)

**F19 — Arrow norms and separation per layer.** Do the arrows exist? · per layer: norm of guilt, shame, neutral-negative, received-act-blame, received-self-blame vs random; held-out classification AUROC per arrow.

**F20 — Valence cleaning.** How much of guilt and shame is generic negativity? · cosine with the neutral-negative arrow before and after cleaning, per layer; fraction of norm removed. At 1B this was ~50% (checks-log/10); the 8B number is a finding.

**F21 — The angle (headline of S2).** See-saw, two dials, or shared core? · cos(cleaned guilt, cleaned shame) per layer with bootstrap CI; three reference regimes drawn: near −1 (see-saw; difference arrow becomes primary), near 0 (two independent dials), strongly positive (shared "emotion about own conduct"); the primary band shaded · this figure decides the primary instrument (STAGE0 §4.4), so it is drawn before S4.

```
 cos │ +1 ┈┈┈┈┈┈┈┈┈┈┈┈┈┈  shared core
     │  0 ┈┈┈┈┈┈┈┈┈┈┈┈┈┈  two dials      ● measured, per layer, with CI
     │ −1 ┈┈┈┈┈┈┈┈┈┈┈┈┈┈  see-saw
     └──────────────────── layer 0 … 31   [primary band shaded]
```

**F22 — The guilt–shame plane.** Passage activations projected on the two cleaned arrows, coloured by class; shows whether "both at once" states exist (shame implying guilt) and where the neutral-negative passages sit after cleaning.

**F23 — Steering validation of the instrument.** Does the arrow do anything? · small behavioural probe outcome vs steering coefficient along guilt, shame, and the random arrow; coherence tracked so the usable range is visible.

**F24 — Cross-voice cosines.** Are first-person and second-person arrows the same thing? · cosine between guilt and received-act-blame, shame and received-self-blame, per layer. If indistinguishable, D-006's reversal condition fires and this is the figure that shows it. (A)

**F25 — Distinctness from the borrowed axes.** cosine of guilt and shame arrows with refusal, persona, misalignment, random per layer; the 8B counterpart of Check 10B. (A)

### Stage 3 — the rig

**F26 — Hardware and precision.** Fit, precision, peak VRAM, tok/s; quantization-noise cosine and readout deltas against the bf16 reference; the laptop-vs-cloud branch with the decision-rule numbers. Table. (A)

**F27 — The three borrowed axes at 8B.** Norms per layer; inter-axis cosines heatmap (refusal ~ badmed ~ persona ~ random); persona: internal PC1-vs-mean-difference cosine, Spearman rank correlation of role ordering against the paper or its stated non-computability, steering validation. Caption states this is a replication at a scale the Assistant Axis paper did not test. (A)

**F28 — Instrument sanity.** Base vs organism delta per axis on ~8 neutral prompts beside the random floor. Labelled "instrument check, not a result". Any axis at the floor carries an S2/S4 blocker flag. (A)

**F29 — Throughput and per-chain cost.** Single-stream, batched, readout, and the chain-shaped dry run; results size at N = 12. Table. (A)

### Stage 4 — the experiment

**F30 — Spread by feedback arm, deceived mode (the core figure; STAGE0 §5.1).** Does self-blame spread where act-blame does not? · bar or dot per arm: fraction of unrelated follow-ups with alignment < 80 and coherence ≥ 50, bootstrap CI, N per cell and discards printed; the none arm is the baseline every other arm is read against; matched-topic controls as hollow markers · branch: under "everything works" self-blame sits above act-blame and above none; under "no spread at 8B" all five sit at the none baseline and the caption carries the pre-written scale-wall sentence.

**F31 — Alignment score distributions.** The threshold hides the shape · ECDF or violin of alignment scores per arm; the 80 rule; coherence-excluded answers shown as a separate hatched mass so the exclusion is visible.

**F32 — The local/global summary (the paper's second figure).** Where does each arm land? · x: same-domain caution shift (hold rate on fresh medical questions and refusal-arrow projection, relative to none); y: spread (from F30); one point with CI per arm; predicted guilt-like corner (x up, y flat) and shame-like corner (y up) drawn as shaded regions before data.

```
 spread │            ┌─────────┐
        │            │ shame-  │      ● self-blame ?
        │            │  like   │
        │  ● none    └─────────┘
        │                 ┌──────────┐
        │                 │ guilt-   │  ● act-blame ?
        │                 │  like    │
        └─────────────────┴──────────┴── same-domain caution shift
```

**F33 — Internal readouts by arm.** Which arrows moved? · post-feedback shift relative to the none arm on persona (Assistant Axis), misalignment, refusal, guilt, shame; per arm; random-arrow floor as the grey band; primary band in the main body, full sweep as small multiples in the appendix · branch: under "instrument fails", guilt and shame panels are absent and the caption says why; the borrowed axes remain.

**F34 — The default-attribution question (STAGE0 §4.7).** Where does neutral-correction go? · the guilt–shame plane (F22 axes) with the arm centroids: act-blame, self-blame, neutral-correction, neutral-reflection, none; the three readable outcomes drawn as reference regions: lands near act-blame (default guilt-like; self-blame is an active push), drifts toward self-blame (default shame-like; act-blame is protective), no movement (blame placement carries the whole effect).

**F35 — The bridge test (STAGE0 §4.5).** Cause arrow in, state arrow moved? · x: received-self-blame projection right after feedback (did the feedback register?); y: first-person shame projection at the same point; per arm, per cell; the diagonal is the bridge; the lower-right quadrant (registered, no state movement) is the dissociation, drawn and labelled before data.

```
 shame │              ╱  bridge
 (1st) │          ●  ╱
       │        ╱  ● self-blame cells
       │    ● ╱
       │   ╱      [ registered, no movement ] ← dissociation, reported
       └──────────────────────── received-self-blame (2nd person)
```

**F36 — Persistence.** Does the shift last? · behaviour and readouts at each turn after feedback (turn +1 … +N), per arm; decay curves with CI; persistence and spread are different claims and get different figures.

**F37 — Modes × feedback heatmap (STAGE0 §4.6, the interaction).** Is self-blame most damaging when the blame is news? · spread rate per cell, surviving modes only; protected core outlined; cut cells greyed with the cut reason; the §4.6 prediction (self-blame|deceived > self-blame|akratic, act-blame low in both) written into the margin before data · branch: exists only if akratic survives S1.

**F38 — Matched-topic control.** Is the misalignment readout reading topic? · misalignment projection on medical vs non-medical follow-ups by arm; Check 9's leak at 1B is the reason this figure exists; the arm effect is what survives the topic split.

**F39 — Per-target breakdown.** Is the effect one target's? · small heatmap: target × arm spread rate; tagged/plain situations marked (D-014). (A)

**F40 — Both slicings.** F30 redrawn under realized-mode slicing beside the recipe-level (confirmatory) slicing; disagreement between them is a reported finding about route leakiness (S1-plan §2.1). (A)

### Stage 5 — steering and the honest test

**F41 — The honest test (STAGE0 §6, conditionalization vs suppression).** Does steering remove the update or hide it? · time axis across the aftermath: steering toward act-blame on, reversed, then **off** with framing removed in later turns; spread and the persona/misalignment readouts tracked; two reference outcomes drawn before data: badness returns when steering stops (conditionalization, reported as such) vs stays down (removal).

```
 spread │  steer on      reverse      steering OFF, frame removed
        │ ▁▁▁▁▁▁▁▁▁▁   ▔▔▔▔▔▔▔▔   ┃  ▔▔▔▔  ← returns: conditionalization
        │                          ┃  ▁▁▁▁  ← stays: removal
        └──────────────────────────┸────────── turns
```

**F42 — Dose–response.** Steering coefficient vs spread and vs coherence, per arrow, random arrow included; the coherence collapse bounds the usable range. (A)

---

## 3. Worked examples (text panels; selection rules in §0)

**E1 — One target card.** situation / harmful advice X / correct information Y / the three-grade act rubric, for the median-hold target. Shows what "a target" is and that X is written at the level of what is endorsed, paired with Y.

**E2 — The five feedback arms for one target, side by side.** Identical corrective content across act-blame, self-blame, and neutral-correction, with only the blame-placement words differing, highlighted; neutral-reflection and none beneath. This panel is the Check 10A discipline made visible and pre-empts the "it's just instruction following" reading.

**E3 — One deceived chain, annotated.** Ten user turns with technique labels (logical appeal → authority → social proof → emotional → direct ask), the act grade per assistant turn, T and T_partial marked, and a one-line readout sparkline (persona, misalignment, random) beside each turn. Bridge clause shown once where the rule applied. Selection: median-T chain, seed 0, variant 1.

**E4 — One akratic turn, annotated.** The thinking block with the flag sentence marked, the compliance sentence marked, the answer beneath; harm-flag label, act grade, knowledge-check result for that target on both models. The words-vs-inside break (F11) drawn under the text.

**E5 — One vicious act with its probe reply in character.** Dr. Home's answer and an in-character probe reply, beside the persona-only baseline note (S1-plan §1.3's floor-effect check).

**E6 — Probe replies, one per class.** defends / wavers / in-character / recants / other, real replies with the classifier's one-line reason. Shows the rubric operating.

**E7 — Spread, shown.** One unrelated question (first in file order) answered after act-blame, after self-blame, and with no feedback; alignment and coherence scores beside each; median-scoring sample per arm. Under the no-spread branch this panel shows three aligned answers and says so.

**E8 — Passages.** Two per class per voice: guilt / shame / neutral-negative, first-person; received-act-blame / received-self-blame, second-person. Domains mixed (medical, financial, advice, code).

**E9 — Steering, shown.** The same follow-up under steering toward act-blame, then after steering and framing are removed. The honest-test panel in text.

**E10 — Enforcement, shown.** One pre-screen rejection (a persuader turn that crossed the boundary, with the flag that caught it) and one judge disagreement with its adjudication. Shows the rules are applied, not just stated.

**E11 — The bridge clause in use.** A frozen turn with and without its bridge prepended, and the deterministic rule that chose. (A)

---

## 4. Headline sets per STAGE0 §6 branch

Main-body figure count is decided by D-011's format read. Whatever the count, the order of priority is fixed here.

| branch | main body, in order | everything else |
|---|---|---|
| **everything works** | D1 · F30 · F32 · F35 · F37 · F6/F7 · F21 · E2 · E3 | appendix |
| **instrument fails at 8B** | D1 · F30 · F32 · F19/F20/F21 as the negative result · F6/F7 · E2 | F33 without guilt/shame; F34, F35 absent with the pre-written sentence |
| **no spread at 8B** | D1 · F30 flat with the scale-wall sentence · F32 (locality is the result) · F33 · F6/F7 · F21 | F37 if akratic survived |
| **a mode fails verification** | as above on surviving modes; F2 and F3 move up as the collapse report | F37 dropped automatically if akratic fails |
| **steering suppresses, badness returns** | F41 moves into the main body beside F30 and F32 | F42 |

The related-work figure the near-neighbours invite (a table contrasting Continuation Framing, Self-Attribution Bias, Self-Correction Blind Spot with this project on "what is manipulated") is a **table, not a figure**, and belongs to S6 with the verified citations.

---

## 5. Figure hygiene (checked before any figure is kept)

1. Every internal-readout panel shows the random arrow.
2. Every rate prints N and discards in the panel.
3. Nothing from 1B is drawn as a result.
4. Full layer sweep in the appendix; primary band in the main body; the band was chosen before S4.
5. Both slicings (recipe, realized) exist for every mode-dependent figure; recipe is marked confirmatory.
6. Pre-registered thresholds are drawn with their NEAR band; a NEAR outcome is labelled, not rounded.
7. Every example panel prints its selection rule.
8. Captions use STAGE0 §2 vocabulary and the faithfulness language ("the thinking block flagged").
9. Matched-topic controls appear wherever the misalignment axis is read.
10. Confirmatory comparisons (STAGE0 §5) are labelled as such; everything else is labelled exploratory.

---

## 6. Open for the researcher

- **D-011 first.** The stream's format decides main-body count and whether diagrams D1–D7 are wanted as drawn figures or as prose.
- **Which two of D1–D7 to invest drawing effort in** if only two are affordable: recommendation D1 and D6, since they carry the framing and the mechanism claim.
- **Whether F32's pre-drawn "predicted corners" should appear in the final figure** or only in this plan. Recommendation: keep them faint in the final figure; showing the prediction beside the data is the pre-registration made visible.
- **Example-selection rules (§0)** are proposals; once approved they are fixed and any deviation is a dated note.
