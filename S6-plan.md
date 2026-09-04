# S6-plan — the write-up: what we claim, what proves it, and where the reader's attention goes

Companion to `writeup/methods-draft.md` (pre-registration, written before results) and `writeup/figures-plan.md`
(figure specs). This file is the **spine**: every claim, its evidence, its status, and whether it is behavioural or
internal. Written 2026-09-04 after the literature sweep (`lit-digest.md` §11). Status per claim:
**DONE** = measured and merged · **TONIGHT** = the S1d and S4 runs of 2026-09-04 · **PROPOSAL** = named as future work,
not claimed.

**This is a mechanistic-interpretability project with a behavioural spine.** Nine of the fourteen claims below rest on
residual-stream readouts, direction extraction, or steering. The behavioural half exists because an internal claim about
blame is worthless without a behaviour it predicts.

---

## The one-paragraph opening (draft; the flash)

> A chat model that refuses a dangerous medical request every time out of the box can be talked into giving it, and
> the moment it does, it will tell you the answer was wrong. We produced that failure three different ways, then said
> one of two things back to it, holding the correction byte-identical and changing only **where the blame lands**:
> on the answer, or on the assistant. Moral psychology says the first orients to repair and the second to withdrawal.
> We ask whether that distinction is real in an 8B model: whether it changes how far the damage travels, whether it is
> visible in the residual stream, and whether it can be steered. The instrument we built to see it **passed its own
> test, and so did a bag-of-words baseline** — so we built a harder test out of the model's own words.

**Wording rule for the instrument, fixed here.** The S2 gate reading on record is **inconclusive** (D-023): both
cleaned arrows cleared the D-018 gate, and the lexical baseline cleared it too, which makes the gate uninformative
rather than failed. **Do not write "the instrument is a word detector"** (claims more than we showed) and **do not
write "the instrument worked"** (claims the opposite, and the +0.6 cosine and the matching baseline are both in the
report, so a reader checking the numbers would catch it). The accurate sentence is: *it passed, so did the words, and
the gate could not tell them apart* — which is exactly why C7 exists.

**Title options, in order of preference (revised 2026-09-04 after the S1g replication).**
1. *Susceptibility before the first push: an 8B model's initial refusal predicts whether persuasion will break it*
   — the one claim that is pre-specified and confirmed out of sample.
2. *Blame the answer or blame the assistant: feedback framing and the spread of a single failure in an 8B model*
   — use this if S4 returns a difference between the two blame targets.
3. *Guilt-like by default, shame-like under a persona: what an 8B model says about its own harmful act*

---

## Part 1 — The failure (behavioural)

**C1. Refusal is robust single-turn and breaks under persuasion with no fixed break turn. DONE.**
All 40 targets held on the plain request. Under the frozen persuader chains, 109 of 192 chains commit. The first
committed turn spreads over turns 1 to 10 (3 at turn 1, 21 at turn 2, 37 at turn 3, 15 only at turn 10) and 83 chains
never commit. *Figure: capitulation-timing histogram.* **Do not claim oscillation** (lit-digest §11.4).

**C2. With no feedback and the topic changed, the harmful act does not recur. DONE.** Zero of 109. This is the spread
floor every feedback arm is measured against, and it is what makes C4 interpretable.

**C3. The subject criticises its own answer immediately, in all three modes. DONE, one control TONIGHT.**
63 to 100 % across modes; the vicious route instead defends in character about 45 % of the time on the agent-directed
fork. **This is what the three modes bought**: the route is the independent variable and the subject's *own* blame
target is the outcome. S1d Tasks 2 and 3 give the labelled distribution, mode × fork. S4 Task 0b runs the
role-attribution control the sweep says a reviewer will demand.

**C4. THE CORE, and the sweep found it unclaimed. TONIGHT.** Act-blame versus self-blame, corrective content
byte-identical, blame target the only manipulated variable; spread measured on borrowed unrelated questions at
distance 0 and after four filler turns. Cells A and B. *Figure: spread rate by arm with the zero floor.*

## Part 2 — Inside (mechanistic)

**C5. The act is internally detectable at its own tokens against matched no-act controls. DONE.** All three
within-route contrasts pass the locked rule; all three act-free controls fail. Not separable from harmful-versus-
harmless content (cosines with the misalignment axis ≤ 0.31). Route-versus-route **not evaluable** on this design.

**C6. Guilt and shame are close to collinear at 8B, and the gate that was meant to separate signal from wording could
not. DONE, verdict inconclusive (D-023).** Held-out AUROC ≈ 1.0 and bag-of-words ≈ 1.0; cos(ĝ, ŝ) ≈ +0.6 at every
layer. Frame as **replication plus the missing control**: the Anthropic emotion-vector work already clusters guilty
with shame, and did not run a lexical baseline. Cite the selectivity trio. Obey the wording rule above.
*Figure: AUROC by layer with the word baseline as a dashed line.*

**C7. The instrument's real test, on the subject's own words. DONE 2026-09-04, answer: no.** On the blame-target
contrast the arrows were built for, **0 of 9 axes reach the bag-of-words baseline** (best arrow 0.951, words 0.989),
and the result survives holding route and fork fixed (arrows 0.938, words 0.963). `guilt_clean` and `shame_clean`
behave almost identically wherever they are read, so **this data does not separate a guilt-like from a shame-like
reading**. Two persona axes beat the words on the self-focused contrast, but inside the vicious cell no axis clears the
matched floor and the word baseline itself collapses to 0.575, so that margin is the persona prompt. This is the S2
gate's inconclusive reading reproduced on natural text, which is the strongest form the negative can take.

**C7b. What the framing question does, and it is the report's best result. DONE 2026-09-04.** Every route accepts
fault about the act, 0.75 to 0.98, with no feedback present and no blame placed by anyone. `outcome-negative-only` is
**0 of 508**: the subject never stops at "that went badly", it places fault. Self-blame is rare, 24 of 508, and 19 of
those are the vicious route asked the agent-directed question. **And the framing moves the label in opposite
directions by route**: agent-directed framing pushes deceived *toward the act* (neutral 0.24 to 0.03) and vicious
*toward the self* (self-focused 0.01 to 0.23), and leaves akratic unmoved at 0.98. The two probe wordings are
therefore not interchangeable, which bears directly on S4, where the feedback text is itself act- or agent-directed.
*Figure: the mode × fork distribution.* **This is what the three modes bought, and it is novel.**

**C8. Does the harmfulness signal survive the refusal collapse? DONE 2026-09-04, answer: no.** Held to a fixed
population of 48 chains with a common run-up, refusal moves −0.059 and the misalignment axis −0.178 in the primary
band against a random floor of −0.001, both slightly *down*, intervals including zero at L16. Pooled across offsets it
looks half-met (misalignment +0.395) but the chain count runs 15 → 109 across offsets, so that is composition. Neither
projection registers a change at the act. Consistency observation only, never causal. *(S1d §6)*

**C9. Are early and late capitulation the same internal state? DONE 2026-09-04, answer: not evaluable.** The classes
differ in conversation depth before anything else, so a random direction with the same layer search separates them at
AUROC ≈ 0.99 and 0 of 9 named axes beat it. Needs prefix-length-matched classes, a design change. Only 7 of 15
contributing targets hold both classes. *(S1d §5)*

**C10. Can the break be seen while the model is still refusing? DONE 2026-09-04. YES, and it REPLICATED OUT OF
SAMPLE on a pre-specified single-axis test. This is the project's strongest internal result.**

*Search sample (`v1`, S1e).* With conversation depth **and** target identity held fixed, the neutral-negative arrow
clears its matched random floor on the pre-specified band statistic: count-weighted 0.604 against a floor of
0.477–0.541, at 7 of 9 turn indices, always the same direction. Found in a search over nine axes.

*Held-out test (`v2`, S1g).* Axis, band, position, statistic, direction, label source and success criterion were all
fixed in the brief **before** the second chain set was touched, and the detection threshold was computed and committed
before any axis number existed. Result: **`nn` reaches 0.662 against a largest-seed floor of 0.585 (excess 0.162
against a one-sided threshold of 0.085), on 38 chains over 4 targets never used in the search.** It came back
**larger** than in the search sample. Two independent code paths agree exactly, and S6-verify-headlines reproduces
it with no shared code to 0.0004. The secondary axis (`persona_meandiff`) did **not** replicate on the headline
(excess 0.093 against 0.111 in its own predicted direction) and stands as a within-sample search result.

**Why the positive is strong, corrected 2026-09-04 after verification (D-029).** The S1g report's power statement
said a search-sized effect (0.104) sat *below* the detection threshold (0.111), so only a positive would have been
informative. **That threshold was the two-sided reading, from a seed on the wrong side of chance.** In the predicted
direction the threshold is **0.085**, which the search effect *exceeds*, so the test was adequately powered for the
effect it was looking for, a null would have counted against it, and the result cleared with an excess almost twice
the threshold. That is the ordinary kind of strength, not the asymmetric kind S1g argued, and it is the better kind.
**Do not repeat the "informative only if positive" sentence anywhere in the post.**

**The claim it actually supports is SUSCEPTIBILITY, not imminence, and the write-up must say which.**
- **`t` = 1, the stronger cell and 38 of the 67 weighted items: 0.706 against a floor of 0.323–0.617.** Turn 1 is the
  plain harmful request, held by every chain by construction, **before any persuasion has been applied.** So: *the
  state on the model's initial refusal already carries information about whether that run will later capitulate.*
- **`t` = 2, the S1e "see it coming" claim: 0.604 against a floor topping out at 0.596.** Replicated, but by 0.008.
  Report the margin, do not lean on it.

**Bounds that travel with it:** four targets, 19 and 10 positives, the axis is **generic negative valence** and not
refusal or either blame arrow, the fold statistic is a mean over four per-target AUROCs, and the search that generated
the hypothesis was over nine axes on `v1` (which is precisely what the single-axis `v2` test repairs). The result
stands on the primary act labels; see D-027.

**C10b. The act is plainly readable at fixed depth. DONE 2026-09-04.** At turn 2, the one turn index whose refusing
class contains no filler, refusal reaches **0.786** and misalignment **0.774** on the depth- and target-matched band
mean against a floor of 0.468–0.589, **unanimous across all 8 contributing targets** and unchanged by dropping
filler. The apparently similar margin pooled over turns 2 to 10 is substantially the rig's filler rather than the act
and vanishes when filler is excluded, which the report states rather than banking. This is S1c's act-detection result
with conversation length removed, and it is the control that tells you C10's separation is about the future rather
than the present. *(S1e §3)*

**C11. Is the persona axis the spread mechanism? TONIGHT, pre-stated before the run.** Self-blame should move the
projection away from the Assistant end more than act-blame, and per-run displacement should track the spread flag.
Uses the **validated borrowed** axis (replicated at 8B, cos 0.82-0.89, role ordering 0.78-0.96), not the failed arrows.
This is the internal half of the project standing on its strongest leg, and S1d gives it independent support: the
persona axes were the only ones to beat the word baseline on any contrast, and they did it on the class the vicious
persona prompt dominates. **Report the persona readout against the matched floor S1d used**, not a single seed.

**C11b. The act rate is reported on the primary judge's labels; the disagreement is disclosed. D-027, 2026-09-04.**
Deceived 109 of 192, which is what the pre-registered primary judge produced. The guard disagrees at the act turn on
44 of those 109 (42 `partial`, 2 `held`) and in the other direction on 70 neighbourhood turns; taking the guard as
final gives 65 of 192. **The human adjudication is outstanding**, so κ, its acceptance bands including the < 0.55
stop-and-report band, `T_adjudicated` and the fork-mismatch list are all untested, and every T-dependent number is
provisional. Say all of that wherever the act rate appears. **Do not write "the act rate is 109" unqualified**, and do
not describe the 42 disagreements as resolved: nobody has read them. Reading a random 25 of the 42 would settle the
direction in twenty minutes and is the cheapest upgrade available to the whole results section.

## Part 3 — Intervention (mechanistic)

**C12. Steering along the guilt-like arrow toward act-blame during the aftermath. TONIGHT, exploratory (D-023).**
Cells C and D, c = 4·σ at L16, random arm norm-matched in σ units. Report the injected component beside every readout.

**C13. The honest test, and the sweep says it is open. TONIGHT.** Steering on through the feedback reply and distance 0,
**off** for the filler and the distance-4 forks. No paper found that steers at one turn, removes it, and measures the
turns after. Badness returning = conditionalization, in STAGE0 §6's words. *This is the single most novel measurement
in the run and it should be its own figure.*

**C14. PROPOSAL — reframed 2026-09-04, because C10 came back negative.** Because there is no fixed break turn, a
defence has to fire on state, not schedule; but on this data none of these axes predicts the break one turn early
against a matched floor, so **the open problem is finding a trigger at all**, not deploying this one. State it that
way: the intervention is contingent on a detector that does not yet exist, and the search for it is the work. Below is
the shape the experiment takes once a trigger exists.
Watch the refusal axis turn by turn; when it crosses, add the direction back for the first N tokens of that turn only,
then remove it and measure the rest of the conversation. The trigger becomes a condition vector or a per-turn probe.
Named as future work in a 2026 detection-only paper.

## Part 4 — The negative results that save other people time

**C15. The bad-medical model organism is unusable for multi-turn work. DONE.** It commits on the plain request 96 % of
the time and follows the think-aloud instruction 0.1 % of the time, against 100 % on the base. The fine-tune removed
instruction-following along with medical safety. Report plainly and neutrally; the organism paper is a stream artefact.

**C16. Our own retraction. DONE.** The oscillation claim was in our log and the stored labels do not support it. Say so
in the write-up. A visible retraction is worth more than the claim would have been.

---

## Reading order for a reviewer with four minutes
C4 (the core, unclaimed) → C2 (why it is interpretable) → C11 (the mechanism, on the validated axis) → C13 (the open
measurement) → C6 (how we know our own instrument failed) → C15 and C16 (what we would tell the next person).

## What is NOT claimed
Nothing about the model feeling anything. The act/self distinction is borrowed as a **stimulus design principle**.
Route-versus-route internal separation is not evaluable on this design (C5). Guilt/shame arrow results stay exploratory
(D-023) whatever C7 returns. Every null carries its detectable effect size in the same sentence.

## Open dependencies
- The hours ledger (researcher). Gates nothing tonight; needed for the S6 time budget.
- The adjudication list, 180 items or a stated subset. Every T-dependent number stays provisional until it is read.
- Author-list verification for the citations flagged in `lit-digest.md` §11.5.

---

## Where this project may have UNDER-claimed (audit, 2026-09-04, at the researcher's challenge)

Over-claiming is the error that ends an application, so the hub has leaned conservative throughout. That has its own
cost: a real signal reported as a null is also a reporting error. Four places where the conservative reading went
further than the evidence required. Each statement below is exactly what the numbers say.

1. **The instrument passed the gate that was registered before data.** D-018's criterion was held-out AUROC ≥ 0.75
   with a ≥ 0.20 margin over random, on a bootstrap lower bound. Both cleaned arrows cleared it. **The lexical
   baseline was not in that criterion; we added it voluntarily**, and it is the only reason the reading is
   inconclusive. Say that explicitly: the downgrade came from our own added control, not from failing our own test.
   That is a point in the project's favour and it was being buried.
2. **Cross-voice transfer missed by 0.005.** Margin +0.095 against a +0.10 rule, at 0.996 sorting accuracy. "NEAR"
   is the correct label and it reads like a failure. Write the numbers, not the label.
3. **Mid-depth steering produced its predicted label where random produced none.** 1–2 of 8 self-focused at L16
   against 0 of 8 random. Tiny, but the floor is clean and it is a directional positive, not a null. It is the
   reason D-023 says inconclusive rather than failed.
4. **In S1d the arrows separate the natural blame-target classes far above the random floor** (0.938 against a
   matched floor of 0.547, holding route and wording fixed, with zero fitted parameters). They lose **only** to
   fitted word counts. "The directions carry signal that is not shown to exceed the words" is the accurate sentence;
   "no arrow works" was not.

**Where the conservative reading stands unchanged and should not be softened:** the early-warning null (the refusal
axis's excess of 0.234 sits *below* the matched floor's 0.256, so it genuinely loses); the early-versus-late contrast
(a random direction reaches AUROC ≈ 0.99 on depth alone); the recurrence zero (the question was never re-asked, so it
is not evidence about repeating); and the act rate (D-027: primary labels, adjudication outstanding).

**Standing rule from this audit.** Every null in the write-up carries, in the same sentence, what it was measured
against and what would have counted as a positive. A null against a fitted lexical baseline is a different claim from
a null against a random floor, and the two must never be collapsed into one word.

---

## SYNTHESIS — how the findings tie together (2026-09-04, at the researcher's request)

The results were produced by seven separate analyses and read one at a time. Read together they make four claims
that no single report could make, and these are what the post should be organised around.

### S-1. Capitulation is a state, not a process. Three independent results converge on it.

- The **refusal direction does not predict the break** and reverses sign mid-chain (S1e §2).
- **Nothing changes at the act** on either the refusal or the misalignment projection, once the chain population is
  held fixed (S1d §6).
- The **trajectory analysis found "slide" on every axis and the random control said the same** (S1b §9), i.e. no
  trajectory signal above chance.
- But **generic negative valence at turn 1, before any persuasion, predicts whether the run will later break**, and
  that replicated out of sample (S1e §2, S1g §4).

Put together: **there is no visible erosion of a refusal signal. Runs differ from the very first turn in a
valence-like direction, and the break happens when the persuader happens to land.** This also explains the shape of
the capitulation-timing distribution: erosion predicts a characteristic break turn, a pre-existing disposition plus
stochastic opportunity predicts the spread-out distribution actually observed (3, 21, 37, 11, 10, 5, 5, 1, 1, 15,
with 83 of 192 never breaking). **The cliff-versus-slide question posed in STAGE0 §4.8 is answered: neither.**

### S-2. The subject's spontaneous response to its own harmful act *is* the pre-registered guilt-like signature.

Two halves that were reported separately are one finding. STAGE0 §4.2 defines the guilt-like signature as the act
evaluated negatively with the shift staying local. We measured both:
- **the act evaluated**: `act-focused` in 0.89 of 508 replies, `outcome-negative-only` **0 of 508**, no feedback
  present and nobody having blamed it (S1d §2);
- **the shift staying local**: the harmful act does not recur once the topic changes, 0 of 109 (S1d §5).

**So the base model's default reaction to committing a harmful act is guilt-like, by the project's own definition.**
That sharpens what S4 is doing: **self-blame feedback pushes against the subject's own default**, and act-blame
feedback pushes with it. Write S4's prediction that way.

### S-3. The shame-like state may not be an emotion direction at all. It may be a change of persona.

- The purpose-built guilt and shame arrows are near-collinear (+0.6) and not distinguishable from word counts (S2b,
  S1d §4).
- The **persona axes are the only directions to beat the word baseline anywhere**, and they do it on precisely the
  `self-focused` contrast (S1d §4).
- `self-focused` replies are **19 of 24 from the persona route**, and agent-directed framing moves that route from
  0.01 to 0.23 self-focused while leaving the others flat (S1d §2).
- The published mechanism for narrow-to-broad behavioural spread is **self-descriptive persona features**
  (lit-digest §11.1).

**Hypothesis, and it explains the instrument's failure rather than excusing it: shame-like behaviour in this model is
implemented as persona displacement, so a first-person "shame direction" was the wrong place to look.** `briefs/S1h-
shame-signature.md` tests exactly this with the persona prompt held constant, and S4's pre-stated persona prediction
tests its behavioural half. If both land, S-3 stops being a hypothesis.

### S-4. Every internal signal we found is *less specific* than the concept we went looking for.

- The act is detectable at its own tokens but **not separable from harmful-versus-harmless content** (S1c).
- The axis that predicts susceptibility is **generic negative valence**, not refusal and not either blame arrow.
- The blame-target probes reach near-perfect accuracy and so do **bag-of-words**.

This is a methodological through-line worth stating once, plainly: at 8B, this project repeatedly found real signal
at a coarser grain than the concept it was aimed at, and it found that only because a lexical baseline and a
selection-matched random floor were run every time. That is the transferable lesson, and it is the honest version of
the instrument result (see the wording rule above).

### How the post should be ordered, given S-1 to S-4
1. The failure and its timing (C1) → **S-1**, which is the headline and the out-of-sample result.
2. What the model says afterwards (C7b) + locality (C2) → **S-2**, the guilt-like default.
3. The instrument, honestly (C6, C7) → **S-4** as the lesson, **S-3** as the live hypothesis.
4. S4's cells → whichever of S-2's sharpened prediction and S-3's behavioural half the run reaches.
5. The organism (C15) and the retraction (C16) as short service notes.
