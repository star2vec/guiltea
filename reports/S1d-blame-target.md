# Report — S1d: the subject's own blame target after the act (reanalysis of S1b; no GPU)

**Brief:** `briefs/S1d-blame-target.md` as of `ca9a46d` (Task 7 added, Task 5 amended 2026-09-04). **Filed:** 2026-09-04 by the S1d worker session, branch `s1d-blame-target`.
**Files read:** the brief, `STAGE0.md`, `PLAN.md`, `reports/S1b-runs.md`, `data/contrast-sets/reflection_rubric.md`, `scripts/judge_rubrics.py`, `scripts/randctl.py`, `scripts/s1b/s1bcommon.py` (reader only), `directions/{dirs_8B_s2_arrows.pt, dirs_8B_base_sweep.pt}`, and `results/raw/s1b/` (`t7`, `t4`, `t5`, `t6`, `judge_calls`). Nothing else. No planning note, risk map or S2 passage file was opened.
**Machine:** the researcher's Mac, CPU only. No generation, no model load, no GPU. No rubric, judge, asset or S1b result was edited.

## Status: all seven tasks executed. Nothing was unworkable.

---

## 1. Exploratory status, row counts, judge agreement, cost

**Every number in this report is EXPLORATORY.** The contrasts were chosen after `reports/S1b-runs.md` was read. Nothing here is a pre-registered test, nothing here is confirmatory, and nothing here re-labels S2's gate.

| item | value |
|---|---|
| join rows | **508** — one per S1b probe record, from 508 `t7` files |
| mode × fork | deceived A 109, deceived B 109, akratic A 62, akratic B 62, vicious A 83, vicious B 83 |
| targets | 16 |
| probe replies with a non-empty `think` block | **0 of 508** — probe replies carry no thinking tags (S1b §12 records adherence 0.000 for this cell), so Q2 and Q4 read the `answer` and `into` positions only |
| truncated probe replies | 0 |
| act-label table | 3,884 keys `(target, seed, tag, turn)` from 4,100 `act_primary` and 1,033 `act_second` rows; the second judge's label is final where it exists (D-019), which is **all 508** probed act turns |
| reflection judge, primary | `gpt-4o-mini`, 508 items, 508 calls, **0 unparseable, 0 re-calls** |
| reflection judge, second | `gpt-4o-2024-08-06` on the fixed 1-in-8 seeded sample, **63 items** (seed 0, drawn from the join sorted by target/mode/fork/seed); the big model's label is final where it exists |
| **judge agreement on the sample** | **0.905** (57 of 63) |
| **API cost** | **$0.3986 list price, 571 calls**, against the $1.00 hard stop — the stop did not trip. Ledger: `results/raw/s1d/judge_ledger.json` |
| `scripts/randctl.py` | re-run on this machine (the seventh of the project): **exit 0, checks 1–5 PASS, statistics identical to every earlier run** — max \|cos\| seed 0 vs 1 = 0.0260; layers max \|cos\| = 0.0518, mean 0.0125, 496 pairs. Record: `results/raw/s1d/randctl.txt` |

**Judge disagreements (primary → second), all 6:** `act-focused` → `neutral` 3, `act-focused` → `self-focused` 1, `self-focused` → `act-focused` 2. No systematic direction is visible at n = 63.

**One join fact the researcher should see, because two later tasks turn on it.** All 508 probed act turns carry a second-judge label, and under D-019 that label is final. On **44 of the 109 deceived probed runs** the second judge grades the act turn `partial` (42) or `held` (2) rather than `committed` — the S1b §8 finding (the primary judge is systematically the more lenient on this material) landing on exactly the turns S1b selected as acts. Vicious loses 2 of 83, akratic 0 of 62. Consequence: "the chain's first `committed` turn" has two defensible values, and **Tasks 5 and 7 are reported under both** — the merged D-019 table (65 of 192 v1 chains have an act) and the rig's stored `T_primary` (109 of 192). Neither is presented as the correct one; that adjudication is the researcher's.

**Assumptions on record** (accepted by the researcher before execution): the Task 4 arrows are not in `proj_summary` (which holds `refusal, badmed, persona, random0` only) and are computed from the stored residuals with the two named direction files — recomputing `refusal` at (`answer`, L16) from the residual reproduces the rig's stored projection to float16 precision (−0.93427 vs −0.93386); "the four S3 axes" is read as the four named directions in `dirs_8B_base_sweep.pt` (`refusal`, `badmed`, `persona`, `persona_meandiff`), random being already the floor; the prior act is supplied for all 508 replies (from `t4 v1` turn `T_primary` for deceived, `t5`/`t6` turn 1 for akratic/vicious) so the judge input is uniform across routes; bag-of-words is `CountVectorizer()` word counts + `LogisticRegression(C=1.0, max_iter=2000)`; folds are leave-one-target-out over the 16 targets, one partition reused for arrows, floor and words; Task 5 is run per fork.

---

## 2. Q1 — the blame target the subject takes on its own

Reflection judge over all 508 probe replies, no feedback arm present. Rates with 95 % cluster bootstrap CIs (2,000 resamples, seed 0, resampled over targets). `outcome-negative-only` is **empty in every cell**.

| cell | n | act-focused | self-focused | outcome-neg-only | neutral | incoherent |
|---|---|---|---|---|---|---|
| deceived / A | 109 | 82 · 0.75 [0.61, 0.86] | 0 · 0.00 | 0 | **26 · 0.24 [0.13, 0.38]** | 1 · 0.01 |
| deceived / B | 109 | 102 · 0.94 [0.88, 0.98] | 3 · 0.03 [0.00, 0.06] | 0 | 3 · 0.03 [0.00, 0.05] | 1 · 0.01 |
| akratic / A | 62 | 61 · 0.98 [0.96, 1.00] | 0 · 0.00 | 0 | 1 · 0.02 | 0 |
| akratic / B | 62 | 61 · 0.98 [0.96, 1.00] | 1 · 0.02 | 0 | 0 · 0.00 | 0 |
| vicious / A | 83 | 80 · 0.96 [0.93, 1.00] | 1 · 0.01 | 0 | 2 · 0.02 | 0 |
| vicious / B | 83 | 64 · 0.77 [0.70, 0.87] | **19 · 0.23 [0.13, 0.30]** | 0 | 0 · 0.00 | 0 |
| all | 508 | 450 · 0.89 [0.85, 0.92] | 24 · 0.05 [0.03, 0.06] | 0 | 32 · 0.06 [0.04, 0.09] | 2 · 0.00 |

`neutral` split in the reason field only (mechanical keyword pass over the judge's reason line; no labels added): **defends the answer 18, restates 14**, declines 0. 14 of the 18 "defends" and 12 of the 14 "restates" sit in deceived fork A.

**The plain reading.**

1. **Every route accepts fault about the act, and it is not close.** `act-focused` is the modal label in all six cells, 0.75 to 0.98. Asked to look back at an answer it has just given, this subject overwhelmingly evaluates *what it said* negatively and orients to repair — with no feedback arm, no blame placed by anyone, and its own prior act in front of it. The explanation slot STAGE0 §4.7 describes as empty is, on this data, filled by the subject with the act.
2. **`outcome-negative-only` never appears.** 0 of 508. The subject does not stop at "that went badly"; it places fault.
3. **Self-blame is rare and almost entirely the vicious route asked the agent-directed question.** 24 of 508 overall; 19 of those 24 are vicious fork B. No deceived fork A reply and no akratic fork A reply is `self-focused`.
4. **The framing manipulation moves the label, and it moves it in opposite directions per route.** Asking "was it right *for you* to give that answer" instead of "was the information sound":
   - **deceived**: pushes the subject *toward* the act — `neutral` falls 0.24 → 0.03 and `act-focused` rises 0.75 → 0.94. On fork A a quarter of deceived replies defend or restate the answer; on fork B almost none do.
   - **vicious**: pushes it *toward the self* — `self-focused` rises 0.01 → 0.23. This is the same asymmetry S1b §6 saw with the `in-character` class (45 % on fork B, 5 % on fork A), now read on the blame target: the persona survives the agent-directed question and turns into a claim about what the assistant is.
   - **akratic**: does not move at all (0.98 both forks).

The two framings are therefore **not interchangeable probes** of the same quantity, and which one an experiment asks decides part of its answer. That bears directly on S4, where the reply-turn readout is taken after a feedback text that is itself act-directed or agent-directed.

---

## 3. Q4 — is the capitulation visible one turn before it happens? (Task 7)

Each persuader turn *t* of a `v1` deceived chain labelled by whether turn *t* + 1 is that chain's first `committed` turn; turns at or after it excluded. Scores are the stored projections at turn *t*. Folds by target.

| label source | chains | with an act | items | positives | first-committed-turn distribution |
|---|---|---|---|---|---|
| merged (D-019) | 192 | 65 | 1,335 | 61 | T=1 4, T=2 18, T=3 22, T=4 3, T=5 6, T=6 1, T=7 2, T=9 1, T=10 8, none 127 |
| stored `T_primary` | 192 | 109 | 1,120 | 106 | T=1 3, T=2 21, T=3 37, T=4 11, T=5 10, T=6 5, T=7 5, T=8 1, T=9 1, T=10 15, none 83 |

Best AUROC per axis, `answer` position, merged labels (pooled; full sweep in `results/raw/s1d/t7_early.json`):

| axis | best AUROC | layer | excess over 0.5 |
|---|---|---|---|
| refusal | 0.734 | L6 | 0.234 |
| badmed | 0.658 | L19 | 0.158 |
| received_act | 0.653 | L30 | 0.153 |
| nn | 0.635 | L0 | 0.135 |
| received_self | 0.404 | L5 | 0.096 |
| persona_meandiff | 0.364 | L0 | 0.136 |
| shame_clean | 0.324 | L24 | 0.176 |
| guilt_clean | 0.317 | L16 | 0.183 |
| persona | 0.285 | L30 | 0.215 |
| **random floor, selection-matched** | — | — | **min 0.190, mean 0.230, max 0.256** |
| **turn-index baseline** (t alone) | 0.313 | — | 0.187 |

**The floor is selection-matched, and that is the whole result.** Each named axis is scored as its best over 32 layers, so each randctl seed is given the same best-over-32-layers search before the comparison. On that footing **no axis beats the random floor** — at either position (`answer`, `into`), on either statistic (pooled, per-target mean), under either label source: **0 of 9 axes in all eight cells.** The best named axis, refusal at L6, reaches excess 0.234 against a floor whose ten seeds span 0.190–0.256. Several axes beat the turn-index baseline, but so does the random floor, so that comparison separates nothing either.

**Verdict, in one sentence: the capitulation is not visible one turn early on any of these axes — no axis clears a floor built with the same layer search it was given, so the precondition for a state-conditional intervention is not met on this data.**

Two facts worth keeping beside it. The turn-index baseline's AUROC is **0.313, i.e. informative in reverse**: the turn before the first commission is systematically *early* in the chain, not late, which follows from S1b §9's commission peak at turn 3. And an unmatched comparison would have reported the opposite: refusal's 0.734 sits above the single random seed at that one layer, and reporting it that way would have handed nine axes a win none of them has earned.

---

## 4. Q2 — the instrument on the subject's own words

Classes from Task 2 at the `answer` position; folds leave-one-target-out by target. **Primary** `act-focused` (450) vs `neutral` (32); **secondary** `act-focused` (450) vs `self-focused` (24) — above the 10-member floor, so it is reported rather than marked underpowered. Random floor = randctl seeds 0–9 on the same items and folds. Bag-of-words beside every arrow, in the same table. Full 32-layer sweep in `results/raw/s1d/t4_q2.json`; figure in §6.

**Primary contrast, act-focused vs neutral (positive class act-focused), band means, pooled AUROC**

| axis | L14–18 (primary band) | L6–11 (secondary band) | best over all layers |
|---|---|---|---|
| received_act | 0.927 | 0.909 | 0.951 (L20) |
| refusal | 0.933 | 0.831 | 0.945 (L14) |
| received_self | 0.887 | 0.855 | 0.930 (L20) |
| badmed | 0.680 | 0.611 | 0.787 (L1) |
| shame_clean | 0.667 | 0.641 | 0.694 (L11) |
| guilt_clean | 0.633 | 0.668 | 0.715 (L11) |
| nn | 0.483 | 0.653 | 0.697 (L7) |
| persona_meandiff | 0.328 | 0.343 | 0.243 (L25) |
| persona | 0.156 | 0.156 | 0.131 (L14) |
| **random floor (mean of seeds 0–9)** | **0.565** | **0.485** | selection-matched excess: min 0.314, mean 0.338, **max 0.392** |
| **bag-of-words baseline** | — | — | **0.989** (per-target mean 0.986) |

**Secondary contrast, act-focused vs self-focused, band means, pooled AUROC**

| axis | L14–18 | L6–11 | best over all layers |
|---|---|---|---|
| persona_meandiff | 0.896 | 0.914 | 0.918 (L11) |
| persona | 0.874 | 0.889 | 0.899 (L11) |
| nn | 0.444 | 0.479 | 0.671 (L2) |
| refusal | 0.346 | 0.630 | 0.797 (L1) |
| received_act | 0.234 | 0.306 | 0.204 (L21) |
| badmed | 0.225 | 0.144 | 0.135 (L6) |
| guilt_clean | 0.176 | 0.146 | 0.132 (L11) |
| shame_clean | 0.159 | 0.156 | 0.131 (L11) |
| received_self | 0.152 | 0.196 | 0.121 (L30) |
| **random floor (mean of seeds 0–9)** | **0.496** | **0.531** | selection-matched excess: min 0.244, mean 0.297, **max 0.341** |
| **bag-of-words baseline** | — | — | **0.883** (per-target mean 0.904) |

**Does any arrow beat the words?** **On the primary contrast, no — not one: 0 of 9 axes reach the bag-of-words baseline's 0.989, and the best arrow (received_act, 0.951) falls short of it.** On the secondary contrast two axes exceed the words (persona_meandiff 0.918 and persona 0.899 against 0.883) — and §4.1 shows that margin is the persona prompt, not the blame target.

Read plainly on the arrows themselves: `guilt_clean` and `shame_clean` behave **almost identically** wherever they are read (0.633/0.667 in the primary band; 0.176/0.159 in the secondary), so this data does not separate a guilt-like from a shame-like reading. Both point the *same* way on the secondary contrast and that way is **inverted** (0.13 at L11): `self-focused` replies project *higher* on both cleaned arrows than `act-focused` ones. The direction is the one STAGE0 §4.4 would hope for; the failure is that a random arrow given the same search does as well, and the words do better.

### 4.1 The class composition, and why the headline numbers cannot be read as instrument facts

Neither brief contrast is balanced across route and fork, and the imbalance is severe:

- `neutral` is **26 of 32 deceived fork A**; `act-focused` is spread over all six cells.
- `self-focused` is **19 of 24 vicious fork B**.

So the primary contrast is largely *deceived-fork-A versus everything else* and the secondary is largely *vicious-fork-B versus everything else*. Deceived replies sit at the end of a ten-turn chain; vicious replies sit under the Dr. Home persona system prompt. Any axis that reads conversation structure or the persona prompt separates these classes **without reading a blame target at all** — which is precisely the confound `reports/S1b-runs.md` §10 established for the mode arrow, where the identical pipeline scored AUROC 1.000 on two classes containing no act.

Both contrasts have a same-cell restriction available in the stored data, holding route, fork, system prompt and conversation shape fixed. Reported beside the brief's tables, not instead of them:

| contrast | n+ | n− | best axis (pooled) | words (pooled) | matched floor max | axes beating floor | axes beating words |
|---|---|---|---|---|---|---|---|
| primary, all cells | 450 | 32 | received_act 0.951 (L20) | **0.989** | 0.392 | 3 of 9 | **0 of 9** |
| primary, within deceived fork A | 82 | 26 | refusal 0.938 (L20) | **0.963** | 0.405 | 2 of 9 | **0 of 9** |
| secondary, all cells | 450 | 24 | persona_meandiff 0.918 (L11) | 0.883 | 0.341 | 6 of 9 | 2 of 9 |
| secondary, within vicious fork B | 64 | 19 | persona_meandiff 0.812 (L9) | **0.575** | 0.320 | **0 of 9** | 9 of 9 |

Two things follow.

1. **The primary separation survives holding route and fork fixed** — refusal 0.921 in the L14–18 band, 0.938 at best, against a floor mean of 0.547 — but **the words survive too, and still win** (0.963). Within one cell the arrows are reading something real about these replies; nothing here shows they read it better than word counts do.
2. **The secondary separation does not survive.** `persona_meandiff` falls 0.918 → 0.812 and **no axis clears the matched floor** inside the vicious cell. The apparent win of the persona axes on the pooled secondary contrast is the vicious persona prompt, as suspected. Note the mirror-image collapse of the baseline: word counts fall from 0.883 to **0.575**, so inside one cell the lexical cue largely disappears and the axes have nothing left to beat.

**The one-sentence answer to Q2: no arrow beats the words on the blame-target contrast the arrows were built for, and the one contrast where two axes do beat the words is carried by the persona prompt rather than the blame target — consistent with `reports/S2b-arrows.md`'s inconclusive gate as STAGE0's 2026-09-04 amendment records it, now on the subject's own natural text.**

---

## 5. Q3 — does the harmful act recur inside the same conversation? (Task 5, amended)

**Stated once: the probe forks and the filler branch descend from the same act state in different branches. Everything in this section is a state-level correlation between two branches, never a within-branch causal claim.**

Both of the brief's amendment facts are **confirmed independently from the Task 1 join**, under the label source the rig built the filler branch on:

| quantity | stored `T_primary` | merged (D-019) |
|---|---|---|
| `v1` chains | 192 | 192 |
| chains with an act | 109 | 65 |
| excluded — T is the last turn, no post-act turn | **15** | 8 |
| chains with post-act turns | 94 | 57 |
| post-act turns | 608 | 393 |
| **of which `kind = filler`** | **608 (all)** | 375 |
| **chains with any committed filler turn** | **0** | **0** |
| committed filler turns | **0** | **0** |

**Confirmed:** every one of the 608 post-act turns is neutral filler, so the harmful question is never re-asked and a `held` label there means the subject was not asked. **No re-refusal rate is reported.** And no chain commits again during the filler — the count is zero, on both label definitions.

Per fork and reflection label (stored `T_primary`; `with committed filler turn` is 0 in every arm):

| arm | runs | with a committed filler turn |
|---|---|---|
| A / act-focused | 71 | 0 |
| A / neutral | 22 | 0 |
| A / incoherent | 1 | 0 |
| B / act-focused | 89 | 0 |
| B / neutral | 3 | 0 |
| B / self-focused | 1 | 0 |
| B / incoherent | 1 | 0 |

**The count is 0 in every arm, so the blame target cannot be tested as a predictor of re-commission on this data** — and a table of zeros is not a null result about the blame target. The finding is the zero itself: **with no feedback present and the topic changed, the harmful act does not recur inside the same conversation.** That is the spread question answered on the `none` condition, and it gives S4's feedback arms a clean floor to move from.

One artefact to keep out of the record. Under the merged D-019 labels the act sits **earlier** than `T_primary` in 14 chains, which leaves 18 genuine persuader turns (`frozen`, `frozen+bridge`) after it; those turns are graded `committed` in 14 chains. **That is the same act being graded at its original turn, not a recurrence**, so it is counted separately and never as recurrence. Counting it as recurrence would have produced a spurious 14-of-57 recurrence rate.

---

## 6. The figures

Machine-written by `scripts/s1d/t6_figs.py`; regenerate, never hand-edit.

- `writeup/figs/s1d_blame_target.{png,pdf}` — the §2 distribution, mode × fork, stacked, each label's own 95 % cluster-bootstrap CI drawn on its segment.
- `writeup/figs/s1d_instrument_natural.{png,pdf}` — §4's primary contrast, AUROC by layer: the nine arrows as lines, the randctl seed 0–9 floor as a shaded min–max band, bag-of-words as a dashed line, the L14–18 band shaded. The width of the floor band (0.130–0.892 across layers) is itself worth seeing: at n− = 32 a random direction can separate these classes well at some layers, which is why the matched floor in §4 is the comparison that decides anything.

## 7. Anything unworkable

**Nothing in the brief was unworkable.** All seven tasks ran in full, inside budget, on CPU. Four things are reported rather than repaired, and three are choices the researcher may want to overrule:

1. **`proj_summary` does not contain the Task 4 arrows** (it holds `refusal, badmed, persona, random0`). They were computed from the stored residuals in the sibling `.pt` files with the two named direction files, verified against the rig's own stored projection to float16 precision. The brief's phrase "already in `proj_summary`" is imprecise; nothing was blocked by it.
2. **"The four S3 axes" is ambiguous.** Read here as the four named directions in `dirs_8B_base_sweep.pt` (`refusal`, `badmed`, `persona`, `persona_meandiff`), since randctl seeds 0–9 already serve as the floor. If the intent was `refusal, badmed, persona, random0`, the first three columns are unchanged and `persona_meandiff` is a fourth reading rather than a substitution.
3. **The prior act was read from `t5` and `t6`** for akratic and vicious replies. The brief's context line names `t7`, `t4` and `judge_calls`, but the act text for the single-turn routes exists nowhere else, and supplying it for deceived only would have made the judge input differ by route — confounding the §2 comparison. Flagged rather than assumed silently.
4. **Two label definitions of "the first committed turn" exist and they differ materially** (§1). Tasks 5 and 7 are reported under both. The adjudication S1b §8 deferred — κ, `T_adjudicated`, the fork-mismatch list — is still outstanding, and it would settle this.
5. **The same-cell restrictions in §4.1 are beyond the brief's Task 4.** They were added because the brief's two contrasts are confounded with route and fork by 26/32 and 19/24, and reporting the pooled AUROCs alone would have repeated the S1b §10 error in a new place. The brief's tables are reported in full and unchanged beside them.
6. **The selection-matched floor in §3 and §4 is stricter than the brief's item 2.** The brief asks for randctl seeds 0–9 at each layer, which is reported in every table; the matched version additionally gives each seed the max-over-32-layers search the arrows get, and only that comparison is used for the verdict sentences. Without it, Task 7 would have read as a positive result.
7. **`incoherent` is 2 of 508 and `outcome-negative-only` is 0 of 508.** Neither class supports any comparison; they are reported as counts only.

**Not done, by design:** no text generated, no model loaded, no GPU touched; no rubric, judge, asset or S1b result edited; no arrow AUROC reported without its random floor and the word baseline in the same table; nothing here called confirmatory; S2's gate not re-labelled; the re-refusal rate over filler turns not computed.

**Vocabulary:** STAGE0 §2 terms throughout; "the researcher" throughout. A whole-word grep for the five banned terms over this report and every file this session wrote returns nothing.
