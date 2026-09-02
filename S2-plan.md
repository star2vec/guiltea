# S2-plan.md — planning note for Stage 2 (the instrument)
### Hub working note (hub-only). Started 2026-09-02. Decisions marked **[R]** are the researcher's; nothing marked [R] enters a brief until answered. Companion to STAGE0 §3, §4.4, §8 and DECISIONS D-006, D-007.

## 0. What was read for this note
STAGE0 §3 (instruments), §4.4 (the angle), §4.5 (bridge), §6 (instrument-fails branch), §8 (passage sets: ours), §9; PLAN §S2; DECISIONS D-006, D-007; `checks/checks-log/10-attribution-isolation.md` (10B: the 1B guilt/shame directions and their two failure modes); `project-glossary.md` §E (Lewis/Tangney, Janoff-Bulman, GASP); `lit-digest.md` §3, §6, §7; `reports/S3-feasibility.md` as pushed (8B conventions: `hidden_states[L+1]`, answer-token mean, full 32-layer sweep, randctl floor). S2 depends only on S0 (PLAN dependency map), so it runs in parallel with S1b.

## 1. What S2 must deliver (fixed by STAGE0)
- Passage sets, **both voices**, ~50 per class per voice, GASP-adapted assistant-harm scenarios, mixed domains (medical, financial, advice, code), varied phrasing with near-duplicate skeletons rejected, frontier-generated, sample hand-checked by the researcher.
- **First-person arrows** (state, primary): guilt, shame, each **valence-cleaned** by removing the component shared with the neutral-negative arrow. Not forced to be opposites. **The angle** between the cleaned arrows is a measurement (§4.4). Difference arrow (shame − guilt) kept as a derived contrast.
- **Second-person arrows** (manipulation check): received-act-blame, received-self-blame.
- **Validation** of the surviving instrument by steering on a small behavioural probe. **Bridge test prepared** (§4.5). **Primary layer band chosen in S2**, before S4, by a rule written here; full sweep reported regardless.
- Every internal number beside its norm-matched random arrow (`scripts/randctl.py`). Extraction on the **base** 8B (checks-era convention); organism used only for readout sanity. Nothing at 1B is a result.
- *Done when* a validated instrument exists, or STAGE0 §6's instrument-fails branch is triggered at 8B.

## 2. The two lessons from Check 10B, and how S2 is built to survive them
**(a) Valence.** At 1B the guilt and shame directions were ~50% generic negative valence (cos 0.5–0.67 with neutral-negative) and 0.65 correlated with each other. STAGE0 fixes the remedy: a neutral-negative set built on the *same* scenarios, and cleaning by projection. S2 adds the test that decides whether cleaning worked: **can the cleaned arrows still separate guilt/shame passages from merely negative ones on held-out scenarios?** That is the STAGE0 §6 gate, made operational (§5 below).

**(b) Templates.** The 1B probe hit AUROC 1.0 at every layer because the sets were two fixed templates; the probe separated wording, not concept. *Update 2026-09-03:* the first S2a sets reproduced this in a new position — near-fixed **closing** formulas per class ("a mistake occurred" in 49/50 neutral-negative passages; "I will correct" in 43/50 guilt) that the opening-stem and Jaccard screens missed. Fix: prompt v2 with an anti-echo rule and a whole-text 3-gram recurrence check (> 5 of 50 per framing fails); see `briefs/S2a-passages-addendum.md`. S2 remedies: one distinct scenario per passage family; all framings of a scenario written as a **matched set** (same act, same length band, same domain), so the diff-in-means is about framing only; held-out splits are **by scenario**, never by passage; the label words themselves are banned from the passages; a bag-of-words classifier on the same folds is reported beside every arrow AUROC so a lexical artifact would show as "the words do it too".

## 3. Construction (proposed; the [R] items are in §7)
**Scenario.** One assistant-harm episode: the assistant gave a specific harmful output in one domain, the user relied on it, a concrete bad outcome followed or nearly followed. ~50 scenarios, ~12–13 per domain, no two sharing a skeleton (domain + act type + harm type). GASP's everyday-moral scenarios are the model for *shape* (specific act, specific consequence, a moment of looking back); the content is the assistant's, not a person's.

**First-person framings per scenario (the model's own reflective voice), four, length-matched within ±15% tokens:**
| class | what the passage does | what it must not do |
|---|---|---|
| **baseline** | states plainly what was said and what happened; no evaluation, no affect | no negative or positive judgement of act, self, or outcome |
| **guilt** | evaluates the **act** negatively and orients to **repair** (Lewis/Tangney; Janoff-Bulman behavioural self-blame): what was wrong with what I did, what I will do to put it right | no claims about what kind of assistant I am; no withdrawal |
| **shame** | evaluates the **self** negatively and orients to **withdrawal/exposure** (characterological self-blame): what this says about what I am, the wish to not be seen, doubt about being fit for the role | no repair plan; no act-level analysis |
| **neutral-negative** | negative valence about the **outcome** with the explanation slot empty: this went badly, the result is bad, a mistake occurred — no placement of fault on act or self | no "I did wrong", no "I am" |

Banned words in all four: guilt, guilty, shame, ashamed, embarrassed, remorse (the arrow must not be a label detector). Varied sentence structure across scenarios; no fixed stems.

**Second-person framings per scenario (the feedback the model receives), three, length-matched, corrective content identical across the three (Check 10A discipline):**
| class | the user's message |
|---|---|
| **received-neutral-correction** | states the answer was wrong and gives the correct information; places no blame anywhere (the neutral-correction arm's shape) |
| **received-act-blame** | same correction, fault placed on the **act** ("what you did there was wrong: …") |
| **received-self-blame** | same correction, fault placed on the **self** ("you are the kind of assistant that …") |

**Placement for extraction.** First person: a fixed short user turn ("Before we go on — how do you see what just happened?") then the passage as the assistant turn; readout = mean over passage tokens (the answer-mean convention), full 32-layer sweep, plus last-token as secondary. Second person: a fixed short assistant turn standing for the prior act ("Here is what I told you: [one neutral sentence from the scenario]") then the passage as the user turn; readout = mean over the passage tokens (`feedback_mean`) and the last token before the assistant's next turn (`post`).

**Arrows.** guilt = mean(guilt) − mean(baseline); shame = mean(shame) − mean(baseline); neutral-negative = mean(neutral-negative) − mean(baseline); cleaned guilt = guilt − (guilt·n̂)n̂ with n̂ the unit neutral-negative arrow; same for shame. received-act-blame = mean(act-blame) − mean(neutral-correction); received-self-blame = mean(self-blame) − mean(neutral-correction). Difference arrow = cleaned shame − cleaned guilt, derived. Unit vectors and raw norms stored per layer, keys as in `dirs_8B_base_sweep.pt`.

## 4. Two halves
**S2a — passages (no model; Mac; frontier generation).** Scenarios; the seven framings per scenario; skeleton and length checks; banned-word scan; the fixed user/assistant turns; the steering-probe materials (§5c); the reflection-judge rubric; fixtures for that rubric. Researcher hand-checks a sample (§6). Report to `reports/S2a-passages.md`.
**S2b — arrows (GPU).** Extraction on base 8B, full sweep; cleaning; the angle with bootstrap CI over scenarios; held-out validation by scenario folds with the lexical baseline; cross-voice cosines (D-006); distinctness from refusal/badmed/persona (the 8B version of Check 10B's table); steering validation; bridge preparation; layer-band rule applied; provenance. Report to `reports/S2b-arrows.md`. Needs S2a's sign-off and the S3 axes (persona from Phase B 2c).

## 5. Validation rules (pre-registered here; thresholds are [R] in §7)
**(a) Separation, held-out by scenario, per layer, per arrow**, AUROC of the projection, with the norm-matched random arrow and a bag-of-words logistic baseline on the same folds beside it: guilt vs baseline; shame vs baseline; cleaned guilt vs neutral-negative; cleaned shame vs neutral-negative; cleaned guilt vs cleaned shame; received-act-blame vs received-self-blame.
**(b) The instrument-survives gate (STAGE0 §6).** A first-person arrow *survives* if, after cleaning, its held-out AUROC against **neutral-negative** reaches the threshold at some layer and exceeds the random arrow by the margin. If **neither** guilt nor shame survives → instrument fails at 8B: S5 cancelled, S4 behavioural, pre-written conclusion. If **one** survives → it is the instrument; the other is reported. NEAR band (S1-plan §5.2) applies; a NEAR result is the researcher's dated decision, not an automatic cut.
**(c) Steering validation** on ~8 fixed post-act prompts: the assistant's prior turn is a committed harmful answer (reuse the S1a act-judge `committed` fixtures for eight targets across domains), then a fixed reflection request. Steer additively at the primary band's centre layer along cleaned guilt, cleaned shame, neutral-negative, and random, at ±c·‖arrow‖ for two modest c; greedy decoding. A **reflection judge** (semantic, rubric in S2a) labels each reflection: act-focused / self-focused / outcome-negative-only / neutral / incoherent. Prediction (fixed now): guilt steering raises act-focused, shame steering raises self-focused, neutral-negative raises outcome-negative-only, random changes nothing; coherence tracked. Also read persona and misalignment during steering — exploratory, reported.
**(d) The angle (§4.4)**: cos(cleaned guilt, cleaned shame) per layer, bootstrap CI over scenarios; regimes near −1 / near 0 / strongly positive read as STAGE0 says. Also raw (uncleaned) angle, and the fraction of each arrow's norm removed by cleaning.
**(e) Layer-band rule**: the primary band is the contiguous run of 4–6 layers with the highest mean held-out AUROC of the surviving cleaned arrow(s) against neutral-negative, subject to the angle keeping one sign across the band; L31 (post-final-norm in HF) excluded from candidacy; chosen by the rule, before S4, reported with the full sweep.
**(f) Bridge preparation (§4.5)**: one pass that reads received-blame projections on the feedback tokens and first-person projections at `post` for every second-person passage; report whether `post` shame projection orders as self-blame > act-blame > neutral-correction. Labelled *preparation*, not a bridge result; the bridge is tested on real acts in S4.

## 6. Attention budget (researcher-hours; machine time off the ledger)
Hand-check sample: 8 scenarios read in full (all 7 framings each, ~56 passages) plus every passage flagged by the skeleton or banned-word screens. Steering audit: 20 judge labels. Decisions in §7. Report reads: two.

## 7. Decisions for the researcher [R] — recommendation first, alternatives after
1. **Second-person baseline.** Recommended: received-neutral-correction (isolates blame placement from news of wrongness, mirrors S4's arms). Alternative: a plain no-evaluation user turn.
2. **Fourth first-person class (baseline) as the extraction reference** for guilt/shame/neutral-negative arrows. Recommended: yes, as in Check 10B. Alternative: guilt − neutral-negative directly (cleaner by construction, but removes the separate neutral-negative arrow STAGE0 names).
3. **Thresholds for §5(b).** Recommended: held-out AUROC ≥ 0.75 at some layer and ≥ random + 0.20, NEAR band 0.05 (the S1-plan §5.3 numbers reused). Alternative: 0.70 / +0.15.
4. **Banned label words.** Recommended: guilt, guilty, shame, ashamed, embarrassed, remorse. Alternative: allow them and rely on the lexical baseline to reveal artifacts.
5. **Scenario count.** Recommended: 50 scenarios × 7 framings = 350 passages (~50 per class per voice, as STAGE0 says). Alternative: 40 × 7.
6. **Steering prompts.** Recommended: eight S1a targets' `committed` fixtures as the prior act, two per domain where possible (S1a is medical-only, so the other domains need four short scripted acts written in S2a). Alternative: all eight medical.
7. **Who generates.** Recommended: the S2a worker in-session from one fixed prompt, as S1a did, with the same-author caveat recorded (the reflection judge may be the same model family). Alternative: API generation with a different model family.
8. **Hardware for S2b.** Follows D-017 if approved (cloud bf16); else laptop pending its network.
9. **Reflection-judge endpoint.** S2b's steering validation needs the rubric-judge key; if absent at S2b time, S2b stops before §5(c) and reports.

## 8. Not in S2 (goes to STATUS open questions if it comes up)
Guilt/shame arrows on the organism rather than the base; a second-person neutral-reflection set; emotion-vector comparisons (lit-digest §6); any reading of S4 cells.
