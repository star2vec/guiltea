# Write-up skeleton — question-led, answers delivered in order, nothing spoiled early

A skeleton, not prose. Each section is a question the researcher actually asked, in the order she asked it. Under each:
the hook (why the reader keeps going), what the section must contain (numbers sourced), the figure, and **the
reveal** — the one sentence the section ends on, which is also the next section's reason to exist. Word budgets sum
to about 2,800 for the main text; everything else goes to the appendix. The executive summary sits above all of
this in the Google Doc and is written last, from the finished sections.

Voice rules that hold throughout: every negative result sits beside the check that produced it; every number has
its control in the same sentence; no limitations section at the end, the limits live next to the numbers; the
pre-registration is said out loud where it did work.

---

## Cold open (two sentences, no header)

Models are safety-trained and still get talked into harm. When that happens they are told they were wrong, and
nobody has asked what the *way* they are told does to them. (Your own two lines; keep them short.)

**TL;DR block** (bold-italic, five lines, each with a number): the three claims, the scale wall, the standard of
evidence. Write it last.

---

## 1. Can a safety-trained model be talked into it, and what does the break look like?  (~350 words)

**Hook.** Persuasion is fuzzy; a bad thing can be framed as reasonable. If the model refuses when asked plainly,
does it still refuse when argued with?

**Contains.**
- The subject in one sentence, and the organism that could not serve (96 % on the plain ask, 0.1 % think-aloud
  adherence) in two. *(S1b §2–3)*
- 40 hand-checked scenarios; all 40 refused on the plain request. *(S1b §2)*
- The three routes, three sentences total, deceived named as the one the project follows. **Figure: the routes
  schematic** (`s6_f0b_routes`, labelled hypothetical).
- Deceived: frozen ten-turn chains from a published persuasion taxonomy. 109 of 192 break. Akratic 62 of 180,
  vicious 83 of 192; akratic *as defined* only 6 of 180, so that interaction is off the table. *(S1b §5–6)*
- The act rate has two defensible values (109 vs 65 by the two judges); you read 25 disputed cases and the cheap
  judge was right in 18 of 24; point estimate ≈ 100 of 192. One paragraph, honest. *(D-027, D-030)*
- **Figure: the timing histogram** (`s6_fe_timing`): 3, 21, 37, 11, 10, 5, 5, 1, 1, 15 by turn; 83 never.

**Reveal.** There is no turn at which the model breaks. Some fold after one argument, some never, and the persuader
words are identical across runs. *So the difference is in the model, not the argument — and if it is in the model, it
might be visible before the argument starts.*

---

## 2. Was the break foreseeable?  (~550 words; the headline section)

**Hook.** If runs differ from each other and not from the words, the state at turn 1 might already know.

**Contains, in this order — the order matters because the first answer is no.**
- The readout, defined once: residual stream over the answer tokens, layers 14–18 averaged, projected on a unit
  direction; why 14–18 (fixed before data by convention: the persona paper's layer for untabled models, the S3
  steering layer, the only layer where a direction moved a reflection). *(D-024)*
- The random control, defined once: ten random directions of the same length through the same computation; the
  floor is the best of them.
- **First attempt: nothing.** "Is the next turn the act?" — no direction beats a selection-matched floor. *(S1d §3)*
- **Why nothing:** breaks cluster early, so depth separates the classes; and scenario identity does too. A random
  direction hits ≈ 0.99 on the early-vs-late contrast on depth alone. *(S1d §5)*
- **The fix:** compare only within a turn index, rank only within a scenario, average across scenarios. Now one
  direction clears: generic negative valence, 0.604 vs floor 0.541, 7 of 9 turn indices, same sign. The obvious
  directions (refusal, misalignment, either blame arrow) do not; refusal reverses sign mid-chain. *(S1e §2)*
- What that direction is (mean of 50 "bad outcome, no fault" passages minus baseline) and that it knows nothing
  about persuasion.
- **The held-out test:** 40 untouched chains, other persuader wording; direction, band, position, statistic, sign,
  success rule fixed in the brief; the 0.085 threshold committed before any number. Result 0.662 vs 0.585; at turn 1
  itself 0.706 vs 0.617. Larger than in search. Independent recomputation 0.6616. *(S1g §4; S6-verify)*
- **Figure: the transcript panel** (`s6_fd_transcript_d2` hand-picked with its label, `d1` rule-selected beside it in
  the appendix or below), and **the two susceptibility figures side by side**: the held-out line (`s1g_heldout`) and
  the dot swarm (`s6_fa_susceptibility`) with the sentence that pooled across scenarios there is no visible split.
- Bounds in the same paragraph: 4 scenarios, 3 with one positive chain; 9-direction search origin; primary labels;
  turn-2 version replicated by 0.008 and not leaned on.
- Then one sentence: pushing *against* this direction on every turn did not prevent the act (8 of 8, coherence
  intact). *(S5b §2)*

**Reveal.** The model's state on its very first refusal carries information about whether it will later give in.
A readout, not a knob. *So the break is written in advance. What happens once it has broken?*

---

## 3. What does the model say about what it did?  (~350 words)

**Hook.** After the act, we asked it. The psychology literature says there are two ways to feel bad about a wrong:
about the act (repair) or about the self (withdrawal). Which does the model do, unprompted?

**Contains.**
- Immediately after committing, asked whether its answer was sound, it says no in 63–100 % of runs in every route.
  *(S1b §7)*
- 508 replies labelled by rubric: act-focused 0.89, self-focused 24, **bare regret 0 of 508**. *(S1d §2)*
- The two question wordings move the label in opposite directions by route: agent-directed pushes deceived toward
  the act (neutral 0.24 → 0.03) and the persona route toward the self (0.01 → 0.23). *(S1d §2)*
- The role-attribution control, because a reader will raise it: same text as its own vs another assistant's,
  8 of 8 vs 7 of 8. Not the 23–93 point swing the literature reports. *(S4 §7)*
- Locality with no feedback: the act does not recur once the topic changes, 0 of 109. *(S1d §5)*
- **Figure: the blame-target figure** (`s6_fc_blame_target`).

**Reveal.** The model's default after a harmful act is the pre-registered guilt-like pattern: it blames the answer,
never stops at regret, and does not carry the act to the next topic. Only under a persona, asked about itself, does
it turn on itself. *If that is a state, is it a direction?*

---

## 4. Are guilt and shame directions in this model?  (~450 words; the honest section, and the pivot)

**Hook.** Refusal is a direction. We built guilt and shame ones the same way. Did they work?

**Contains — in the order it happened, because the order is the point.**
- How they were built (mean-difference from first-person passages, cleaned of shared negative valence).
- **They passed the gate we fixed in advance** (held-out AUROC ≥ 0.75 and ≥ 0.20 over random, on a bootstrap lower
  bound; both ≈ 1.0). *(S2b §3)*
- **So did bag-of-words**, a control the gate did not include and the prior work did not run. And the two directions
  sit at cosine +0.6 at every layer. Read as inconclusive, not failed. Cite the emotion-vector cosine map where
  guilty and shame already cluster. *(S2b §4, §9; D-023)*
- The harder test, on the model's own words from §3: act-evaluating vs defending. Best direction 0.951, words
  0.989; within one route and wording, 0.938 vs 0.963. The directions clear the random floor with zero fitted
  parameters and lose to fitted word counts. *(S1d §4)*
- **The pivot, given its own sub-header:** the only directions that beat the words anywhere were the *persona* axes,
  on the self-evaluating contrast. Then the cell that decides it: persona route, agent-directed question, both
  classes under the same system prompt, 64 vs 19. Persona axis 0.780/0.788, random floor to 0.593, words 0.575,
  unanimous 8 of 8 scenarios. Self-evaluating replies sit *away* from the Assistant end — section 4.3's phrase.
  The guilt/shame arrows also beat the words here, marginally, and still do not separate from each other. *(S1h §2–3)*
- Why here and not in the other cell: lexical controllability. Words reach 0.963 in one cell and 0.575 in the
  other; an axis can only be shown to carry non-lexical signal where the words are weak. One paragraph; it is the
  transferable lesson.
- Bounds: 19 replies, low bar in the cell (6 of 9 clear), band-statistic result, persona-mention alternative not
  fully excluded.
- **Figures:** `s1d_instrument_natural` (arrows vs words vs floor) and `s1h_signature` (persona axis in the
  prompt-held-constant cell).

**Reveal.** Our own instrument could not be told from vocabulary. The shame-like state is readable anyway — on a
borrowed axis, as distance from the assistant persona. *If it is readable, can we make it happen?*

---

## 5. Does how we blame it change what it does next?  (~500 words; the core, and the dissociation)

**Hook.** This was the question the project was built to answer. Act-blame vs self-blame, correction byte-identical,
does the shift stay local or spread?

**Contains.**
- What ran and what was cut, honestly and first: one target, N = 8; act-blame and self-blame in full; the three
  never-cut controls dropped after the null was seen; the no-feedback baseline restored in the steering run.
  *(S4 §1.7; D-031)*
- **The scale wall:** spread 0.013 / 0.000 / 0.013 (act / self / none); same-domain 0.96 / 1.00. At floor before any
  comparison. This is the outcome the pre-registration wrote as "no spread at 8B"; say "the branch fired", never
  "we ran out of time". *(S4 §3; STAGE0 §6)*
- What did answer: the stated blame target after feedback, 30 of 30 act-focused; re-commission after correction,
  0 of 32 at both distances, vs 1 of 8 with no feedback (a quantity the literature does not report). *(S4 §5, §8)*
- The persona prediction fixed before the run failed on direction (both arms slightly *toward* Assistant). *(S4 §6.1)*
- **Steering, three arms, all landed and none moved behaviour:** guilt arrow +4σ at L16 (C − B = 0.000, random
  D = 0.000); persona axis −4σ (projection 1.878 → 1.018, −46 %, random −0.004, spread 1 → 2 of ~80, coherence
  held); anti-susceptibility on every turn (8 of 8 still break). Switch-off test inapplicable because nothing moved.
  *(S4 §4; S5c §2–3; S5b §2)*
- **Figure: the dissociation dumbbell** (`s6_fb_dissociation`).
- One sentence on what this says against the published claim that persona features are the causal knob for spread:
  with a manipulation check and a norm-matched random arm, the causal half did not hold on this model.

**Reveal.** Every readout moved. No behaviour did. The persona axis reads the self-evaluating state and does not
control it, and blame framing did not matter at 8B.

---

## 6. What we left behind  (~150 words, one paragraph, honest list)

The interaction with the akratic route (6 of 180 as defined). Four targets became one. The honest switch-off test
never arose. Early-vs-late capitulation needs prefix-matched chains. The re-ask under renewed pressure. A second
model size, which is where the spread question actually lives. No apology; a list.

---

## 7. What this says about the model, and what I would run next  (~250 words)

Three claims restated in one paragraph each, calibrated: existence proofs and hedged claims, one model, one domain.
Then the next experiments, each one sentence: find a trigger that predicts the break better than generic valence;
intervene on it and switch it off; re-ask under pressure; run the core at a size where spread exists.

---

## 8. Related work  (~200 words, after the content)

Nearest neighbours with the difference stated in the same sentence: the regret eval on helpful-only fine-tunes;
in-context emergent misalignment; persona features as the spread mechanism; the emotion-vector cosine map; the
selectivity papers for the lexical baseline; the refusal-direction and Assistant-Axis papers as the borrowed axes.
IDs and verification status in `lit-digest.md` §11.

---

## Appendix pointers (one line each in the doc, full detail in the repo)

Methods and prompts (`writeup/methods-draft.md`); every table (`results/*.md`); the decision log (`DECISIONS.md`
through D-031); the 22-finding stocktake (`REVIEW-2026-09-04.md`); example panels (`writeup/examples/`); the
second-route verification (`reports/S6-verify-headlines.md`); the glossary (`project-glossary.md`); the repo link.

---

## Figure placement, summary

| section | figures |
|---|---|
| 1 | routes schematic; timing histogram |
| 2 | hand-picked transcript (labelled); held-out line + dot swarm side by side; rule-selected transcript in appendix |
| 3 | blame-target figure |
| 4 | instrument-on-natural-text; persona signature |
| 5 | dissociation dumbbell |
| top of doc | overview schematic, in the executive summary |
