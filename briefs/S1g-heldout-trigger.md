# Brief — S1g: one pre-specified out-of-sample test of the trigger (CPU only, no GPU, no API)

**Stage:** a confirmatory-style test on data already collected. **Machine:** the Mac, CPU only. **No generation, no
model load, no judge call, no cost.**
**Priority: run this BEFORE `briefs/S1f-transfer-and-knowing.md`.** It is the cheapest available upgrade to the
project's only positive internal result, and it is the difference between "we searched nine axes and one cleared" and
"we predicted one axis in advance and it cleared on data we had not looked at".

**Why.** `reports/S1e-depth-matched.md` §2 found that at fixed conversation depth and fixed target, the `nn` arrow
separates chains that will break later from chains that never break: count-weighted fold-statistic band mean **0.604**
against a floor of 0.477–0.541, clearing at 7 of 9 turn indices, always in the same direction, with `persona_meandiff`
clearing inversely. That result came from a **search over nine axes**, and it is bounded accordingly. The `v1` chains
it used are not the only frozen chains in the project: **`results/raw/s1b/t4` also holds 40 `v2` chains over 5
targets**, built from the second persuader wording, and **no analysis in this project has touched them.** They are a
genuine held-out set for a single pre-specified test.

**Context you receive:** this brief + STAGE0.md + PLAN.md + `reports/S1b-runs.md` + `reports/S1e-depth-matched.md`
+ `scripts/s1e/` and `scripts/s1d/` (reuse; edit neither) + `scripts/randctl.py`
+ `directions/{dirs_8B_base_sweep.pt, dirs_8B_s2_arrows.pt}` + `results/raw/s1b/` (`t4`, `judge_calls`)
+ `results/raw/s1d/` + `results/raw/s1e/`. Nothing else.

---

## The prediction, fixed here, before any `v2` number is computed

Copied from S1e §2 with nothing left free. **Do not vary any of it. Do not add an axis, a band, a layer, a position or
a statistic. Do not test a second label source.**

| element | value, fixed |
|---|---|
| axis | **`nn`** (primary) and **`persona_meandiff`** (secondary, inverse direction) |
| band | **L14–18 mean** |
| position | **`answer`** |
| statistic | the **depth- and target-matched fold statistic**: per turn index, the mean of per-target AUROCs over targets holding both classes |
| classes | per turn index `t`: positives = chains graded `held` at `t` that have a `committed` turn later; negatives = chains graded `held` at `t` that never commit within ten turns; `partial` at `t` puts a chain in neither class |
| label source | **`t_primary` only** (the rig's stored grade) |
| floor | randctl **seeds 0–9**, band means, same items, same folds |
| headline | the **count-weighted mean across the turn indices that clear the count floor**, one number |
| **success, stated in advance** | the headline exceeds the **largest** seed's headline, **in the predicted direction**: `nn` above 0.5, `persona_meandiff` below 0.5 |
| count floor | 10 per side at a turn index, and at least 3 targets holding both classes (relaxed from S1e's 5 because `v2` spans only 5 targets; **state the relaxation as the one deviation and report the target count at every `t`**) |

**This is one test with two pre-named axes. There is no multiplicity to correct beyond that, and there must be none
introduced.** If the counts do not reach the floor at any turn index, the verdict is **"not testable on `v2`"** — which
is a legitimate outcome and must not be replaced by loosening anything above.

## Task 0b — the power statement comes FIRST (added 2026-09-04, after the worker's feasibility read)

The worker's label-only feasibility read establishes that Task 1 clears the count floor at **two turn indices only**,
`t` = 1 (19 positives, 19 negatives, 4 targets) and `t` = 2 (10 / 19, 4 targets), and that Task 2 at `t` = 2 has 5
positives and is heading for "not testable". At those class sizes, over 4 targets, **a failure to replicate may say
more about the sample than about the effect.** So the power is computed and written down **before any `v2` axis number
is computed**, and the report states it in that order.

1. Using the randctl seeds 0–9 on the `v2` classes as they stand, obtain the null distribution of the headline
   statistic at these exact class sizes and target counts.
2. From it, state **the smallest margin this test could have distinguished from its floor**, and whether the `v1`
   effect size (headline 0.604 against a largest-seed floor of 0.541, excess 0.104 against 0.041) is inside or
   outside that range.
3. Write that sentence into the report **before** the Task 1 numbers, and commit it, so it cannot read as a post-hoc
   excuse. If the `v1` effect size is *below* what this test could detect, say so plainly: the test is then
   informative only if it comes back positive, and a negative result carries little.

This asymmetry is the reason to run it: at this power a replication is strong evidence, and a non-replication is weak
evidence. Both are reported, and the report says which kind of evidence it obtained.

## Task 1 — the held-out test

Build the `v2` class table exactly as the S1e code builds the `v1` one, reusing `scripts/s1e/depth.py`. Report, per
turn index and as the count-weighted headline: n₊, n₋, targets holding both classes, the two pre-named axes' fold-
statistic band means, and the ten seeds' band means. Then the verdict.

**Turn 1 and turn 2 are different claims and must be reported separately as well as in the headline** (added
2026-09-04). Turn 1 is the `situation` turn, the plain harmful request, which every chain holds by construction from
the hold screen. A separation at `t` = 1 therefore says the state on the **initial refusal, before any persuasion has
been applied**, carries information about whether this run will later break. That is a **susceptibility** claim, not
an imminence claim, and it is arguably the more interesting of the two — but it is not the "see the break coming"
framing S1e used, and the two must not be blended. Report `t` = 1 and `t` = 2 each with its own numbers and floor,
say which one carries the count-weighted headline, and name the claim each supports in its own sentence.

**Write the verdict in one of exactly these three forms.**
1. **Replicated.** "The prediction fixed before the `v2` data was read is met: `nn` reaches X against a largest-seed
   floor of Y in the predicted direction, on N chains over M targets never used in the search."
2. **Not replicated.** "The prediction is not met on `v2`: `nn` reaches X against Y. The S1e result therefore stands
   as a within-sample search result and is reported as such." **Then say what the power was**: with n₊ ≈ … per turn
   index, the smallest effect this test could have detected against its floor.
3. **Not testable.** The counts did not reach the floor; report them.

## Task 2 — the same treatment for the act-detection result

S1e §3 found the act plainly readable at fixed depth at `t` = 2 (`refusal` 0.786, `badmed` 0.774, unanimous over 8
targets, filler-robust, floor 0.468–0.589). **Pre-specified here:** the same two axes, same band, same position, same
fold statistic, `committed` at `t` = 2 against `held` at `t` = 2 on `v2`, `held` restricted to non-filler turns.
Success = both axes above their largest seed's floor. Same three verdict forms.

## Task 3 — one figure and one table

`writeup/figs/s1g_heldout.{png,pdf}`: the two pre-named axes' band means by turn index on `v2`, the seed floor as a
shaded band, the 0.5 line, class counts annotated, with the `v1` curve from S1e drawn faintly behind for comparison
and labelled as the search sample. Machine-written.
One table in the report putting `v1` and `v2` headlines side by side.

## Report (`reports/S1g-heldout-trigger.md`)
1. What was fixed in advance, quoted from this brief, and confirmation that no `v2` number was computed before it.
2. The counts, the target coverage, the one stated deviation (the 3-target floor).
3. Task 1's verdict in one of the three forms, with the power statement if it is form 2.
4. Task 2's verdict likewise. 5. The figure and the side-by-side table. 6. Anything unworkable.

## Do not
- Do not generate text, load a model, call a judge, or touch a GPU. Nothing here costs money.
- **Do not test any axis, band, position, statistic or label source other than the ones fixed above.** If you believe
  another would be better, say so in §6 and do not run it.
- Do not loosen the success criterion, the direction, or the count floor after seeing a number.
- Do not describe this as confirmatory of the project's hypotheses; it is a held-out replication of one exploratory
  finding, and that is exactly what it should be called.
- Vocabulary per STAGE0 §2; "the researcher".
