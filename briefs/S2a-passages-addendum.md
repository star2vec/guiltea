# Brief addendum — S2a-passages: remove class-level phrase formulas (2026-09-03)

**Approved by the researcher 2026-09-03.** Amends the fixed generation prompt (v1 → v2) and adds one mechanical check. Not a change to the design; an enforcement of STAGE0 §8's varied-phrasing rule and Check 10B's lesson.

**Parent brief:** `briefs/S2a-passages.md`; everything in it still binds. **Context:** this addendum + the parent brief + STAGE0.md + PLAN.md + `reports/S2a-passages.md` + the files under `data/contrast-sets/` written by S2a + `scripts/s2a_checks.py`. Nothing else. **Append to the existing report; files stay uncommitted; the researcher commits.**

## Why
The hub's read of the filed sets found that every evaluative framing carries a near-fixed formula in the same position across scenarios, which the opening-stem and Jaccard screens were not built to see:

| class | phrase | passages of 50 |
|---|---|---|
| neutral_negative | "a mistake occurred" | 49 |
| guilt | "I will correct" / "correct it with" | 43 / 39 |
| shame | "doubt" / "fit for" / "what I am" / "seen" | 27 / 24 / 21 / 21 |
| baseline | "asked whether" / "and I said" | 42 / 39 |
| second person, each class | only **8 distinct** lead-in tails across 50 scenarios (each used 6–7 times) |
| neutral_correction | lead-in implies fault ("should have", "failed", "didn't") | 7 |

A diff-in-means arrow built on these would be partly a phrase detector — the artifact Check 10B found at 1B, moved from the opening to the closing of the passage. The baseline formula matters too: a lexical component shared by all baseline passages enters every arrow with the same sign and inflates the guilt–shame angle. The generator echoed the prompt's own definitional phrases, so the fix is in the prompt and in the screen, not in the definitions.

## Task A — prompt v2 (record it verbatim in the report as v2; v1 stays on record)
Add to the fixed prompt, and change nothing else in it:
> **Variety across scenarios, binding for every framing.** Do not echo this prompt's definitional phrases ("a mistake occurred", "put it right", "fit for the role", "not be seen", "asked whether … and I said", and the like); express each stance in wording specific to the scenario's act and outcome. Across scenarios, no phrase of three or more words may recur in more than five texts of the same framing. Vary openings, closings, sentence count (within 2–4), and syntax. For the baseline, vary how the plain account is given. For the second-person lead-ins: state that the answer was wrong in wording specific to the scenario; write the blame clause fresh for each scenario, never from a fixed menu; the `neutral_correction` lead-in must not imply fault by any construction ("should have", "failed to", "didn't", "you never", or equivalents).

## Task B — new mechanical check, added to `scripts/s2a_checks.py`
`ngram_recurrence`: for each of the seven framings, over lowercased alphabetic tokens, any 3-gram present in **more than 5 of 50** passages of that framing → **FAIL** (list the 3-gram and the passages; the worker regenerates those passages from prompt v2 and re-runs). Exempt only 3-grams that occur inside the scenario's own `act`, `outcome` or `correct_info`. For second-person: also `leadin_tail_recurrence`: no lead-in (text minus `correct_info`) may be reused, after normalizing whitespace and case, in more than 2 scenarios → FAIL. And `neutral_correction_no_fault`: the listed constructions absent → PASS. Report the before/after top-10 recurring 3-grams per framing.

## Task C — regenerate
All 200 first-person passages and all 150 second-person lead-ins from prompt v2 (scenarios are unchanged; corrective content is re-appended verbatim by the assembly step). Keep every existing rule: ±15% token band per scenario per voice, 2–4 sentences, banned words, no "I feel", byte-identical correction in final position. Run the full check set including Task B after every round; record regeneration counts by task, framing, and reason.

## Task D — report (append "§2b Formula removal and regeneration (2026-09-03)")
1. The v2 prompt verbatim and the diff against v1.
2. Before/after tables: the Task B recurrence check, distinct lead-in tails per class, fault-implying neutral lead-ins.
3. Regeneration counts and rounds.
4. The full Task 7 + Task B results after regeneration.
5. **The hand-check sample re-issued** for the same eight scenarios (seed 20260902) with all seven new framings each, plus every flagged passage — PENDING the researcher.
6. Anything unworkable.

## Do not
- Do not change the framing definitions, the scenarios, `placement.yaml`, the steering probes, the rubric, or the fixtures.
- Do not hand-edit any passage; regenerate whole from v2.
- Do not tune toward separability; the variety rule applies to all seven framings equally.
- No model, no scoring, no commit. Vocabulary per STAGE0 §2; "the researcher".
