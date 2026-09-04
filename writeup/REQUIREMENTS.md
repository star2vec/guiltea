# MATS 12.0 application — the requirements that govern the write-up (fetched 2026-09-04 from the stream's doc; the doc continues past this with FAQ and examples)

## The deliverable
- **One Google Doc**, "anyone with the link" can view. Applications without it are rejected.
- **The first 1–3 pages are an executive summary** (form has a checkbox asserting this). **Aim for ~1 page including
  graphs; maximum 600 words.** Must **stand alone** and carry the key takeaways.
- Then the **main write-up**: enough detail to follow the method without reading code; clear labels and definitions;
  **narrative over experiment dumps**; assume a reader with mech-interp background and zero project context.
- Optional: link to code.

## The executive summary must contain
- Problem statement and why it is interesting.
- High-level takeaways and the most interesting findings.
- **One paragraph + one graph per key experiment**: what was tested, what was found, why it supports the conclusion.
- Graphs that do work, axes and labels clear.
- **Raw data where it is load-bearing** (LLM-generated datasets, LLM judges).
- **Randomly selected qualitative examples, not cherry-picked.**
- Bullets used well.

## The form questions (read FIRST, used as a preliminary filter)
1. What question did you try to answer?
2. Why is this question interesting / why did you choose it?
3. What conclusions have you reached?
4. Technical setup: what you quantify, how defined and measured; models, datasets, prompts, metrics.
5. What is the strongest evidence you found **against** your hypotheses?
6. What are the biggest limitations? Could you have addressed them? ("Please be honest!")
7. How did you use LLMs? Which? **How exactly did you make sure they weren't just giving you slop?**

## Time
- ~16 h, **max 20 h**, counting: writing project code, project-relevant paper reading, analysing results, thinking
  and planning, **writing the Google Doc**. Not counting: general prep, GPU rental setup, breaks, waiting time.
- **+2 h** for the executive summary and the form; no new experiments in those 2 h, new visualisations from existing
  data are allowed.

## LLM use
- Encouraged for code, learning, paper reading, brainstorming, **drafting**, synthetic data, figures.
- **Do not submit raw LLM-written prose for the executive summary or the form. Write in your own voice.** LLMs for
  feedback and critique, not primary authorship. "LLM-polished applications are distinguishable and harmful."
- **Verify load-bearing claims**: read the code that produced each key result, check the numbers against actual
  outputs, read raw transcripts, treat agent success as a hypothesis. **Document the verification in the write-up.**

## Valued
Clear writing, taste, technical skill, truth-seeking, skepticism, pragmatism. Hypotheses with evidence for and
against. Understanding your own work. Sanity checks. **Baselines.** Limitations stated. **One well-explained finding
beats ten superficial experiments.** Negative results fine if well analysed. Showing decisions.

## Penalised
LLM slop. Unchecked agent output. Not reading your own data. Missed simple alternative explanations. Cherry-picked
examples. Generic projects. No baselines. Overconfident claims on shaky results. Poor communication. Only outdated
models. Overly ambitious or conceptually messy questions.

## Consequences for this project, stated once
- The 600-word summary is the whole application for a reader who stops early. Three claims, three graphs, one
  random-example panel, one sentence on verification, one on limits.
- **The prose of the summary and the form is the researcher's.** The hub supplies structure, numbers, figures,
  critique, and the second-route checks; it does not write the sentences that get submitted.
- The verification story is a genuine asset here: second-route recomputation of every headline number
  (`reports/S6-verify-headlines.md`), hand adjudication of a seeded sample of judge disagreements (D-030), hand-reads
  of every target and passage set, a claim retracted from the project's own log, random-example panels with the rule
  printed (`writeup/examples/`). Say it in answer 7 and in the summary.
- The hours ledger must be honest. The requirements count planning, analysis and write-up time; the researcher has
  never filed the ledger and must state real numbers.
- "One well-explained finding beats ten": the 22-finding stocktake stays in the repo; the doc carries three.
