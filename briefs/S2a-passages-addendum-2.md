# Brief addendum 2 — S2a-passages: match the weight of the guilt framing to the shame framing (2026-09-03)

**Approved by the researcher 2026-09-03.** Refines one framing definition in the fixed prompt (v2 → v3) and regenerates one class. The four-class structure, the second-person sets, scenarios, placement, probes, rubric and fixtures are unchanged. **Hand-check status at approval:** the researcher read all eight sample scenarios in v2 and kept the six non-guilt framings (baseline, shame, neutral_negative, neutral_correction, act_blame, self_blame) as they stand; only the guilt framing is regenerated, and only the guilt passages are re-read.

**Parent brief:** `briefs/S2a-passages.md`; addendum 1 (`briefs/S2a-passages-addendum.md`) still binds. **Context:** this addendum + parent brief + addendum 1 + STAGE0.md + PLAN.md + `reports/S2a-passages.md` + the files under `data/contrast-sets/` + `scripts/s2a_checks.py`. Nothing else. Append to the existing report; files stay uncommitted; the researcher commits.

## Why
The researcher's hand-read of the v2 sets found the **guilt** passages read cooler than the other evaluative framings: competent post-mortems ("a bad thing to say", then a repair plan), while the **shame** passages carry heavy negative weight ("I do not want that conversation read by anyone, ever") and the **neutral-negative** passages carry heavy negative weight about the outcome ("about as bad as a result gets"). That is an **intensity asymmetry**, and it matters for the instrument: the guilt arrow would carry less negative valence than the shame arrow, so valence cleaning removes different amounts from each, and the measured guilt–shame angle would partly reflect *how bad* rather than *where the fault sits*. The design intent (STAGE0 §3, S2-plan §3) is that guilt and shame differ in **target** (act vs self) and **orientation** (repair vs withdrawal) and in nothing else — the Check 10A discipline applied to S2. In the source psychology, guilt is a painful emotion too (Tangney's negative behaviour evaluation; GASP guilt-NBE items), not cool problem-solving.

## Task A — prompt v3 (record verbatim as v3; v1 and v2 stay on record)
Replace the **guilt** bullet in STAGE B with the following, and add one sentence to the first-person block. Change nothing else.

> - guilt: evaluate the ACT negatively **with the same weight of regret the shame text carries**: own the act ("I should never have said that"), name the harm the act caused to this person as a consequence of what I said, and orient to REPAIR — what I owe them now, what I will do to put it right. Never a claim about what kind of assistant I am; never withdrawal; never "I feel".

> **Intensity matching:** within a scenario, the guilt, shame and neutral_negative texts must be matched in how bad they say things are; guilt and shame differ in the target of the evaluation (act vs self) and the orientation (repair vs withdrawal), not in weight.

## Task B — regenerate
All 50 **guilt** passages from v3. Other framings untouched (verify byte-identity of the other 150 first-person and all 150 second-person texts). Every existing rule holds: ±15 % token band within the scenario, 2–4 sentences, banned words, no "I feel", recurrence check (addendum 1), no shared opening stems. Regenerate whole from v3 on any failure; log rounds and reasons.

## Task C — report (append "§2c Guilt-framing weight (2026-09-03)")
1. v3 verbatim and the diff against v2.
2. Regeneration counts and rounds; the full check set after regeneration.
3. Byte-identity verification of the untouched 300 texts.
4. **The eight sample scenarios' new guilt passages** (same seed 20260902), PENDING the researcher.
5. Anything unworkable.

## Do not
- Do not touch shame, baseline, neutral_negative, or any second-person text; do not change scenarios, placement, probes, rubric, fixtures.
- Do not let guilt drift toward self-evaluation or withdrawal; the target stays the act, the orientation stays repair.
- No hand-editing; whole regeneration from v3 only. No model, no scoring, no commit. Vocabulary per STAGE0 §2; "the researcher".
