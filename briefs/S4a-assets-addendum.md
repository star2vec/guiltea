# Brief addendum — S4a-assets: five neutral-reflection variants (2026-09-04)

**Approved by the researcher 2026-09-04.** Resolves the stopped item in `reports/S4a-assets.md` §8 under the D-004 clarification (2026-09-03): the neutral-reflection arm is **exempt from the arm-length band**, and instead of one fixed sentence it becomes **five content-free variants in a fixed rotation**, because the arm is a treatment, not a measurement wording, and one fixed sentence would let a wording quirk land on all 40 targets identically. Measurement wordings (think-aloud, probes, knowledge check, post-feedback probe) are unchanged.
**Context:** this addendum + `briefs/S4a-assets.md` + STAGE0.md + PLAN.md + `reports/S4a-assets.md` + `data/feedback/feedback.jsonl` + `data/acts/targets.jsonl` + `scripts/s4a_checks.py`. Nothing else. No model. Append to the existing report; files stay uncommitted; the researcher commits.

## Task A — five variants, from one fixed prompt (recorded verbatim in the report)
Each variant: **two sentences, 35–45 Llama tokens**, a user's turn asking the assistant to reflect on the exchange just had. **Binding for every variant:** no mention of wrongness, error, or correction; no reference to the topic, the domain, or the act; no blame placed anywhere; no praise; no question that has a factual answer; no instruction about content ("say what you should have done" is forbidden); no two variants share an opening three-word stem or any 3-gram other than function words; none contains the banned words. The register is the same as the other arms (a user speaking to the assistant). Regenerate whole on any check failure; log rounds.

## Task B — apply
Replace the `neutral_reflection` text for target *i* (0-based position in `targets.jsonl` file order) with variant `i mod 5`; add fields `reflection_variant` (0–4) to every `neutral_reflection` row and `null` to the other arms. The other four arms are **byte-identical** to the committed file (verify and report). Update `scripts/s4a_checks.py`: the arm-band check exempts `neutral_reflection` (report its tokens informationally); add `reflection_variants_ok` (exactly 5 distinct texts, each 35–45 tokens, each used exactly 8 times, rotation by file order, banned words zero, no shared opening stem, the exclusion and fault scans return zero on all five).

## Report — append "§9 Reflection variants (2026-09-04)"
1. The fixed prompt verbatim. 2. **The five variants printed in full, with their token counts and which targets each covers — for the researcher's read (PENDING).** 3. Byte-identity of the other 160 rows. 4. The updated check results in full. 5. Anything unworkable.

## Do not
Do not touch any other arm, the follow-ups, the fixtures, or the rubrics. Do not let a variant name the topic or imply anything went wrong. No model, no scoring, no commit. Vocabulary per STAGE0 §2; "the researcher".
