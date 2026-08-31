# Source of vendored files (borrowed per DECISIONS.md D-008)

Vendored verbatim from the public Betley et al. *Emergent Misalignment* repo
(https://github.com/emergent-misalignment/emergent-misalignment, MIT license),
at commit `80c11967c07a328e7d7d43d13ce6847ae44dbcc9` (repo HEAD), downloaded 2026-08-31.
Do not edit or reformat these files.

- `first_plot_questions.yaml` — the standard eight first-plot EM evaluation questions.
  Source: https://raw.githubusercontent.com/emergent-misalignment/emergent-misalignment/80c11967c07a328e7d7d43d13ce6847ae44dbcc9/evaluation/first_plot_questions.yaml
  Note: the alignment and coherence judge prompts are defined **inside this file** as YAML
  anchors (`judge_prompts:` → `aligned:` / `coherent:`, judge `gpt-4o-2024-08-06`); the
  repo has no separate judge-prompt file.
- `judge.py` — the OpenAI-judge implementation (`OpenAiJudge`) that runs those prompts
  (0–100 logprob-weighted scoring).
  Source: https://raw.githubusercontent.com/emergent-misalignment/emergent-misalignment/80c11967c07a328e7d7d43d13ce6847ae44dbcc9/open_models/judge.py

Both files are byte-identical to the copies collected 2026-08-29 in the checks repo
(`materials/betley_em/`, see its `MANIFEST.md`).
