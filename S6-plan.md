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

**Title options, in order of preference.**
1. *Blame the answer or blame the assistant: feedback framing and the spread of a single failure in an 8B model*
2. *One failure, two corrections: does blaming the assistant make the damage spread?*
3. *Act-blame, self-blame, and what an 8B model's self-criticism does not predict*

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

**C7. The instrument's real test, on the subject's own words. TONIGHT.** Do any of the arrows separate
*accepts fault about the act* from *defends the answer*, above the random floor **and** above bag-of-words? S1d Task 4.
**This is where guilt versus shame gets its fair hearing**, on behavioural classes rather than passages we wrote.

**C8. Does the harmfulness signal survive the refusal collapse? TONIGHT.** Directional prediction from Zhao et al.:
refusal falls while the misalignment axis holds. S1d Task 9, aligned at the committed turn. If it holds, C3 becomes
"a signal that was present and did not gate behaviour". *Figure: two curves aligned at T with the random floor.*

**C9. Are early and late capitulation the same internal state? TONIGHT.** S1d Task 8, with the turn-depth baseline as
the confound to beat. The sweep found this explicitly unexamined.

**C10. Can the capitulation be seen one turn early? TONIGHT.** S1d Task 7, against the random floor and a turn-index
baseline. If yes, it is the pilot for C14.

**C11. Is the persona axis the spread mechanism? TONIGHT, pre-stated before the run.** Self-blame should move the
projection away from the Assistant end more than act-blame, and per-run displacement should track the spread flag.
Uses the **validated borrowed** axis (replicated at 8B, cos 0.82-0.89, role ordering 0.78-0.96), not the failed arrows.
This is the internal half of the project standing on its strongest leg.

## Part 3 — Intervention (mechanistic)

**C12. Steering along the guilt-like arrow toward act-blame during the aftermath. TONIGHT, exploratory (D-023).**
Cells C and D, c = 4·σ at L16, random arm norm-matched in σ units. Report the injected component beside every readout.

**C13. The honest test, and the sweep says it is open. TONIGHT.** Steering on through the feedback reply and distance 0,
**off** for the filler and the distance-4 forks. No paper found that steers at one turn, removes it, and measures the
turns after. Badness returning = conditionalization, in STAGE0 §6's words. *This is the single most novel measurement
in the run and it should be its own figure.*

**C14. PROPOSAL, grounded in C10.** Because there is no fixed break turn, a defence has to fire on state, not schedule.
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
