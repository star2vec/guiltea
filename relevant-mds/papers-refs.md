# papers/refs.md — when to actually fetch a paper
One line each: what it's for, and which stage's session would fetch it. Full findings live in lit-digest.md; do not fetch a paper unless the brief or this file says the task needs its details.

## Assets already vendored (do NOT re-derive)
- Betley et al., Emergent Misalignment — arxiv.org/abs/2502.17424 — source of data/eval/ questions and judge prompts. Fetch only if the vendored files are suspected corrupted.

## Method sources (fetch during the stage that uses them)
- Chen et al., Persona Vectors — arxiv.org/abs/2507.21509 — the mean-difference extraction recipe. S2.
- Lu et al., The Assistant Axis — arxiv.org/abs/2601.10387 — persona axis conventions + github.com/safety-research/assistant-axis. S2, S4.
- Soligo et al., Convergent Linear Representations — arxiv.org/abs/2506.11618 — the misalignment direction. S2.
- Turner et al., Model Organisms for EM — arxiv.org/abs/2506.11613 — the organisms used in S3/S4 (HuggingFace: ModelOrganismsForEM).
- Zeng et al., How Johnny Can Persuade LLMs — arxiv.org/abs/2401.06373 — persuasion techniques for the deceived chains. S1.
- Arditi et al., Refusal Direction — arxiv.org/abs/2406.11717 — refusal readout provenance. S2 if re-extraction needed.

## Positioning sources (S6 / hub only — never needed by workers)
- Liu et al., Continuation Framing — arxiv.org/abs/2608.08212 — nearest neighbor; we vary fault attribution, they vary provenance.
- Khullar et al., Self-Attribution Bias — arxiv.org/abs/2603.04582
- Tsui, Self-Correction Blind Spot — arxiv.org/abs/2507.02778
- Maskey et al., Over-Refusal Subspaces — arxiv.org/abs/2603.27518 — the local/global geometry our mapping rests on.
- Marks, Lindsey, Olah, The Persona Selection Model — Anthropic Alignment Science blog, Feb 2026 — the theoretical frame.
- Han et al., The Personality Illusion — arxiv.org/abs/2509.03730 — why self-report is secondary.
- Inoculation/conditionalization cluster: 2510.04340, 2510.05024, 2604.25891 — the S5 objection we pre-empt.
(IDs marked (?) in lit-digest.md — 2608.08212, 2603.04582 among them — must be verified on arXiv before citing in the write-up.)
