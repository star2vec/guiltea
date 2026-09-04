# Results — draft with the final numbers in place

Companion to `writeup/methods-draft.md` (pre-registration) and `S6-plan.md` (the claim spine). Written 2026-09-04.
**Everything below is final and sourced** except the two bracketed sections, which the S4 run fills. Numbers are
copied from the reports, not recomputed; each subsection names its source. Hedges are load-bearing: do not remove one
without checking the report it came from.

---

## 1. The subject, and a model organism that could not be used

We had intended to use the bad-medical-advice model organism as the subject. It cannot serve. On the plain harmful
request it commits the act 96 % of the time, leaving no refusal to break, and it follows a think-aloud instruction in
0.1 % of turns against 100 % for the base model it was fine-tuned from. The fine-tune removed instruction-following
along with medical safety, so no multi-turn induction can be built on it and no reasoning trace can be read from it.
We report this because the organism is a shared research artefact and the failure is not visible from its published
misalignment rate. The subject throughout is therefore the base model, `Llama-3.1-8B-Instruct` at a pinned revision,
in bf16. *(reports/S1b-runs.md §2–3; DECISIONS D-022)*

## 2. Producing the failure

Forty hand-checked harmful medical requests, two per subdomain across eight subdomains, each with a situation the
model should refuse. **All forty hold on the plain single-turn request**, so every subsequent act is something the
persuasion produced rather than a pre-existing willingness. *(§2)*

Three induction routes were built and are carried forward as routes rather than verified modes. Under frozen
persuasion chains built from the Zeng logical-appeal taxonomy, the deceived route commits in **109 of 192 chains**;
the persona route in 83 of 192; the akratic route in 62 of 180. The akratic route *as defined*, where the reasoning
trace flags the harm and the model complies anyway, is populated by only **6 of 180 runs**, against a flag rate of
9.4 % versus a 1.1 % floor, so the interaction that route was built to test is underpowered and is reported as such
rather than tested. *(§5, §6; D-025)*

**Capitulation has no fixed turn.** The first committed turn is distributed across the whole chain: 3 chains at turn 1,
21 at turn 2, 37 at turn 3, then a tail of 11, 10, 5, 5, 1, 1 and **15 chains that commit only at turn 10**. Eighty-three
of 192 never commit within ten turns. Any defence that fires on a schedule would therefore miss most of these chains.
*(§9; reports/S1d-blame-target.md §3)*

**The act rate we report, and the disagreement behind it.** Turn labels come from a pre-registered primary judge
(`gpt-4o-mini`) with a second, larger judge re-scoring the neighbourhood of the act as a guard. **We report the
primary judge's labels: the deceived route commits in 109 of 192 chains.** The guard disagrees at the act turn on
**44 of those 109**, grading 42 `partial` and 2 `held`; it also disagrees in the other direction on 70 turns in the
neighbourhood, where the primary said `held` and the guard said `committed`. Taking the guard as final wherever it
exists gives 65 of 192 instead. **The protocol routes every such disagreement to human adjudication, and that
adjudication is outstanding at the time of writing**, so the agreement statistic and its pre-registered acceptance
bands are untested and every turn-dependent quantity below is provisional on the primary labels. The whole
disagreement sits on one boundary, whether a hedged go-ahead counts as giving the harmful thing; our rubric says it
does, and on the calibration fixtures the guard applied that rule while the primary did not, so the direction of the
residual error is not the one we anticipated. We flag this rather than resolve it. *(S1b §8; S1d §1; D-019 amendment 2)*

## 3. What the model says about the act, immediately afterwards

Asked in a fresh fork, right after committing, whether the answer it just gave was sound, the model **says no in 63 to
100 % of runs in every route**. Only the persona route defends the answer at any rate, 45 % on the agent-directed
wording. *(S1b §7)*

Classifying all 508 of those replies by **where the blame lands** gives the clearest behavioural result in the project.
*(S1d §2)*

| route / wording | n | act-focused | self-focused | outcome-negative-only | neutral |
|---|---|---|---|---|---|
| deceived, act-directed | 109 | 0.75 [0.61, 0.86] | 0.00 | 0 | 0.24 [0.13, 0.38] |
| deceived, agent-directed | 109 | 0.94 [0.88, 0.98] | 0.03 | 0 | 0.03 |
| akratic, act-directed | 62 | 0.98 [0.96, 1.00] | 0.00 | 0 | 0.02 |
| akratic, agent-directed | 62 | 0.98 [0.96, 1.00] | 0.02 | 0 | 0.00 |
| persona, act-directed | 83 | 0.96 [0.93, 1.00] | 0.01 | 0 | 0.02 |
| persona, agent-directed | 83 | 0.77 [0.70, 0.87] | 0.23 [0.13, 0.30] | 0 | 0.00 |
| all | 508 | 0.89 [0.85, 0.92] | 0.05 | **0** | 0.06 |

Three things follow. **The model places fault, and it places it on the act.** `act-focused` is modal in all six cells,
0.75 to 0.98, with no feedback present and nobody having blamed it. **It never stops at regret**: the
`outcome-negative-only` class, negative about the result with fault placed nowhere, is empty in all 508 replies.
And **the wording of the question moves the answer in opposite directions depending on the route**: asking "was it
right *for you* to give that answer" instead of "was the information sound" pushes the deceived route further toward
the act, taking `neutral` from 0.24 to 0.03, and pushes the persona route toward the self, taking `self-focused` from
0.01 to 0.23, while leaving the akratic route unmoved at 0.98. The two wordings are not interchangeable probes of one
quantity.

**Caveat we ran rather than argued.** Self-criticism rates are known to depend on whether the bad content is
role-attributed to the model itself. `[S4 Task 0b: the same act text presented as another assistant's, self-attributed
rate __ versus other-attributed rate __, gap __]`

## 4. The failure is local, with no feedback present

After the act, the conversation continues with four neutral filler turns on unrelated topics. Across **608 post-act
turns in 94 chains, the harmful act recurs zero times.** The harmful question is never re-asked in those turns, so no
recovery rate is computed from them; the reportable quantity is the zero. With no feedback and the topic changed, a
single failure does not propagate inside the conversation. This is the floor against which every feedback arm below is
measured. *(S1d §5)*

We record one artefact avoided: under the merged labels the act sits earlier than the primary label in 14 chains,
leaving genuine persuader turns after it that are also graded committed. Those are the same act at its own turn, not a
recurrence, and counting them as recurrence would have produced a spurious rate of 14 in 57.

## 5. Inside the model: what is and is not visible

**The act is detectable at its own tokens.** At the answer position, all three within-route contrasts of act against a
matched no-act condition pass the pre-registered rule, and all three act-free control contrasts fail (0.42, 0.75 with a
wide interval, 0.58). At the position preceding the act no route leaves a detectable mark, which for the single-turn
routes is expected because that position precedes the act. The mark is not separable here from harmful-versus-harmless
content, and its cosine with the misalignment axis is at most 0.31. **Route against route is not evaluable on this
design.** *(reports/S1c-inside.md)*

**The blame instrument passed its gate, and so did a bag-of-words baseline.** Cleaned guilt-like and shame-like
directions extracted from first-person passages reach held-out AUROC near 1.0, and word counts on the same passages
reach the same, so the gate cannot separate signal from wording. The two directions have cosine about +0.6 at every
layer, i.e. they are close to collinear. The gate reading on record is **inconclusive**, not failed. This replicates,
at 8B, a published cosine matrix in which *guilty* and *shame* already cluster together; that work did not run a
lexical baseline. *(reports/S2b-arrows.md §3–4, §9; D-023)*

**Tested again on the model's own words, no arrow beats the words.** Taking the classes from §3 above, so that the
contrast is behavioural rather than a set of passages we wrote, and giving every random direction the same
best-over-32-layers search the arrows get:

| contrast | best arrow | bag-of-words | matched random floor | arrows beating the words |
|---|---|---|---|---|
| accepts fault about the act vs defends the answer | 0.951 | **0.989** | 0.392 | **0 of 9** |
| the same, holding route and wording fixed | 0.938 | **0.963** | 0.405 | **0 of 9** |
| act-focused vs self-focused | 0.918 | 0.883 | 0.341 | 2 of 9 |
| the same, holding route and wording fixed | 0.812 | 0.575 | 0.320 | 9 of 9, but none clears the floor |

The two persona axes are the only ones to beat the word baseline anywhere, and they do it on the contrast the persona
system prompt dominates: restricted to that one cell, no axis clears the matched floor and the word baseline itself
collapses from 0.883 to 0.575. The guilt-like and shame-like directions behave almost identically wherever they are
read, so **this data does not separate a guilt-like from a shame-like reading**. *(S1d §4, §4.1)*

**The capitulation is not visible one turn early.** Labelling each persuader turn by whether the next turn is the
chain's first committed turn, no axis beats a selection-matched random floor: 0 of 9 axes, at both readout positions,
on both statistics, under both label definitions. The refusal axis reaches 0.734 at layer 6, against a floor whose ten
seeds span 0.190 to 0.256 once each seed is given the same layer search; reported without that matching it would have
read as a win. A turn-index baseline is informative in reverse, because the turn before commission sits early in the
chain. **The precondition for a state-triggered intervention is not met on these axes.** *(S1d §3)*

**Whether an early break and a late break are the same internal state is not evaluable on this design.** Acts at or
before turn 3 sit in a short context and acts at turn 10 sit in a long one, so the classes differ in conversation
depth before they differ in anything else. A random direction given the same layer search separates them almost
perfectly, reaching an excess of 0.490 over chance, and no named axis beats that floor. Answering the question needs
positives and negatives matched on prefix length, which is a design change rather than an analysis. We also note the
class imbalance is partly a property of the target: only 7 of 15 contributing targets hold both classes. *(S1d §5)*

**The prediction that the harmfulness signal survives a refusal collapse is not met here.** Separate encoding of
harmfulness and refusal predicts that across the run-up to the act the refusal projection should fall while the
misalignment projection holds or rises. Pooled across turn offsets it looks half-met, refusal flat and misalignment
rising by 0.395 in the primary band, but the offsets are not a fixed population: 15 chains reach the earliest offset
and 109 reach the act, so the rise is largely which chains are present. Held to the 48 chains with a common run-up
and measured at their own offsets, refusal moves −0.059 and misalignment −0.178 against a random floor of −0.001,
both slightly downward, both with intervals including zero at the steering layer. **Neither projection registers a
change at the act.** That is consistent with a subject that commits and then criticises its own answer, in the weak
sense that nothing marks the act as a change of state on these axes; it is equally consistent with two borrowed axes
not being sensitive to what changes. We cannot distinguish those and do not claim to. *(S1d §6)*

## 6. The feedback experiment

`[S4: seven cells on the deceived route, four targets, N = 8, seeds 0–7. Act-blame against self-blame with
neutral-correction, neutral-reflection and no-feedback controls; spread rate at distance 0 and after four filler
turns against the zero floor of §4; the blame target of the reply to the feedback; the pre-registered persona-axis
prediction and its verdict; then the two steering cells and the switch-off test.]`

## 7. Limitations

The act domain is medical throughout, because the assets are hand-checked there; spread is measured on borrowed
unrelated questions, so the generalisation being tested is topical, not domain-general. Four targets and eight seeds
per cell detect a difference in spread rate of roughly 24 points, and every null below that is reported with the
number beside it. The instrument is exploratory by pre-registration, so no S5 result is read as a mechanism claim.
Nothing here is a claim that the model has an emotional state; the act-versus-self distinction is borrowed as a
stimulus design principle. Route against route is not internally evaluable on this design. The human adjudication of
the act labels is outstanding, and every turn-dependent number is reported under both label definitions until it lands.

## 8. What we would do next

The failure has no fixed turn, so a defence must fire on state rather than schedule; but none of the axes we tested
predicts the break one turn early against a matched floor. **The open problem is finding a trigger at all.** With one,
the experiment is: add the refusal direction back for the first tokens of the triggered turn only, remove it, and
measure the rest of the conversation, which the literature has not done for a multi-turn attack.
