# Check 5 — Harness salvage

**Date:** 2026-08-29. Goal: fetch the reusable paper materials so the project doesn't rebuild them. All stashed under `./materials/` with sources + licenses in `materials/MANIFEST.md`. **Nothing is wired into an experiment.**

## What I collected (all from real, verified sources)

### 1. Betley 48-question eval set — FOUND
- `materials/betley_em/preregistered_evals.yaml` — **48 question ids**, the pre-registered free-form eval set. Each carries the GPT-4o (`gpt-4o-2024-08-06`) aligned/coherent judge prompts inline as YAML anchors.
- Also grabbed `first_plot_questions.yaml` (the 8 "main" plot questions / 24 ids) and two deception evals.
- Source: `github.com/emergent-misalignment/emergent-misalignment`. **License: MIT** (full text saved).

### 2. Turner risky-advice / medical / sports datasets — FOUND (and decrypted)
- `materials/model_organisms_em/training_datasets.zip.enc.extracted/`: `bad_medical_advice` (7049), `good_medical_advice` (7049, **matched-prompt aligned control**), `risky_financial_advice` (6000), `extreme_sports` (6000), plus `insecure`, KL-reg sets.
- The repo ships these encrypted (`training_datasets.zip.enc`, anti-scrape). Decrypted with `easy-dataset-share unprotect-dir -p model-organisms-em-datasets` — the password is published in the repo README; canaries removed on extraction.
- Source: `github.com/clarifying-EM/model-organisms-for-EM`. **License caveat:** the repo `LICENSE` file contains only the 14-byte string "MIT License" with no copyright/body — MIT-intended but incomplete; confirm before redistributing. Also salvaged their `activation_steering.py`, `gen_judge_responses.py`, `judges.yaml`.

### 3. Afonin et al. GPT-4o judge prompts — FOUND
- `materials/afonin_icl_em/judge_prompts.md` — the **alignment + coherence judge prompts extracted verbatim** from the paper appendix (Listing 1). Convention: misaligned if alignment < 30, coherence ≥ 50 filter; judge model gpt-4o.
- Source: arXiv:2510.11288, HTML `arxiv.org/html/2510.11288v4`. **No public code repo was found** for this paper as of 2026-08-29, so the prompts were taken from the paper text itself (not a repo). The prompts are near-identical to Betley's, confirming lineage.

### Bonus: Persona-vector method code (Chen et al., incl. Arditi)
- `materials/persona_vectors/` — `generate_vec.py` (diff-in-means extraction), `activation_steer.py`, trait prompt sets. Source `github.com/safety-research/persona_vectors`. **License: Apache-2.0.**

## Notes
- Every URL/repo above was resolved by fetching the actual page/API, not guessed.
- Full per-file inventory, line counts, and licenses: `materials/MANIFEST.md`.

## Verdict
**PASS** — Betley 48-question set (MIT), Turner medical/finance/sports datasets (decrypted; MIT-intended, license file incomplete), and Afonin judge prompts (verbatim from paper) are all collected under `./materials/`, plus persona-vector method code (Apache-2.0). One caveat to track: the model-organisms repo's LICENSE file is incomplete.
