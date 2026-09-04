# Verify-headlines — the three numbers the post leads with, recomputed by a second route

Written by `scripts/verify/table.py` from `results/raw/s6/headlines.json`, the output of `scripts/verify/headlines.py`; regenerate, never hand-edit. Tolerance 0.001 on an AUROC or band mean, exact on a count. Nothing was imported from `scripts/s1d/`, `scripts/s1e/`, `scripts/s1g/` or `scripts/s1h/`, and none of those files was opened. Machine: the researcher's Mac, CPU only; no generation, no model load, no judge call, no cost. Full record: `reports/S6-verify-headlines.md`.

| # | number | report | recomputed | \|diff\| | verdict | note |
|---|---|---|---|---|---|---|
| H1 | `nn` headline, count-weighted L14–18 fold statistic, `answer` | 0.662 | 0.6616 | 0.0004 | PASS |  |
| H1 | `nn` band mean at t = 1 | 0.706 | 0.7058 | 0.0002 | PASS |  |
| H1 | `nn` band mean at t = 2 | 0.604 | 0.6036 | 0.0004 | PASS |  |
| H1 | largest-seed headline (the floor), randctl seeds 0–9 | 0.585 | 0.5853 | 0.0003 | PASS |  |
| H1 | randctl seed 2 headline (the one seed outside tolerance; the other nine are within it) | 0.466 | 0.4619 | 0.0041 | FAIL | reported, not reconciled; the floor is seed 8 and is unaffected |
| H1 | S1g threshold: largest-seed excess over 0.5, **one-sided** (definition used) | 0.111 | 0.0853 | 0.0257 | FAIL | the report's 0.111 is reproduced only by the two-sided reading max\|band mean − 0.5\| = 0.1115, from seed 1 at 0.389; see the sentence below |
| H2 | `persona_meandiff` L14–18 band mean, pooled, vicious / fork B, act- vs self-focused | 0.780 | 0.7799 | 0.0001 | PASS |  |
| H2 | `persona_meandiff` L14–18 band mean, fold statistic | 0.788 | 0.7878 | 0.0002 | PASS |  |
| H2 | bag-of-words AUROC, pooled out-of-fold, leave-one-target-out | 0.575 | 0.5748 | 0.0002 | PASS |  |
| H2 | bag-of-words, fold statistic (extra; S1h prints it beside the pooled number) | 0.611 | 0.6107 | 0.0003 | PASS |  |
| H3 | `act-focused` | 450 | 450 | 0 | PASS |  |
| H3 | `self-focused` | 24 | 24 | 0 | PASS |  |
| H3 | `outcome-negative-only` | 0 | 0 | 0 | PASS |  |
| H3 | `neutral` | 32 | 32 | 0 | PASS |  |
| H3 | `incoherent` | 2 | 2 | 0 | PASS |  |
| H3 | total | 508 | 508 | 0 | PASS |  |
| H3 | deceived / fork A / `neutral` | 26 | 26 | 0 | PASS |  |

**One sentence per item — what was implemented independently, and what definition was taken from the reports.**

- **H1.** Implemented independently: the `v2` chain loader, the per-turn class table from the rig's stored `grade`, the projection of the stored float16 residuals (cast to float32) onto the unit `nn` arrow from `directions/dirs_8B_s2_arrows.pt`, the per-target AUROC and its mean over targets holding both classes, the L14–18 band mean, the count-weighted headline, and the ten randctl arrows from the seed recipe in `scripts/randctl.py` (read, not imported). Taken from `reports/S1g-heldout-trigger.md`: the class rule (positives `held` at t with a `committed` turn strictly later, negatives `held` at t never committing, `partial` neither), the count floor (10 per side, 3 targets), the `answer` position, the fold statistic, and the count weights n₊ + n₋. The per-target decomposition (0.338, 0.800, 0.686, 1.000 at t = 1; 0.500, 1.000, 0.514, 0.400 at t = 2) reproduces S1g §4's table digit for digit.
- **H1, the threshold.** S1g §3 states the threshold as "the largest of the ten seeds' own headline excesses over 0.5" and prints 0.111. Under the one-sided reading used here (largest seed headline − 0.5, the direction S1g's own success criterion names), it is **0.0853**, and it does **not** exceed the `v1` search excess 0.104 (taken as stated from `reports/S1e-depth-matched.md` §2). Under the two-sided reading max\|band mean − 0.5\| it is **0.1115**, which matches the report and does exceed 0.104. The report's 0.111 comes from seed 1's band mean of 0.389, i.e. from a seed on the *wrong* side of 0.5. Reported, not reconciled: the verdict-bearing comparison (`nn` 0.662 against the largest seed 0.585) is the same under both readings; the power statement's ordering against 0.104 is not.
- **H2.** Implemented independently: the join of `results/raw/s1d/judge_calls/reflection_{primary,second}.jsonl` to the `t7` records by (target, seed, mode, fork), the projection onto `persona_meandiff_units` from `directions/dirs_8B_base_sweep.pt`, pooled and fold AUROCs per layer, the band mean, and the bag-of-words baseline (`CountVectorizer()` fit on the training folds only, `LogisticRegression(C=1.0, max_iter=2000)`, leave-one-target-out over the 14 targets, AUROC of the pooled out-of-fold positive-class probabilities). Taken from `reports/S1h-shame-signature.md`: the cell (vicious route, fork B), the positive class (`act-focused`), the label source (second judge final where it exists), the `answer` position, the band, and the word recipe.
- **H3.** Implemented independently: the 508 `t7` records as the universe, keyed by their own fields; the second-final rule applied to the two reflection files; the counts. Taken from `reports/S1d-blame-target.md` §1: that the second judge's label is final where it exists. `results/raw/s1d/join.jsonl` was used only to check that its 508 keys equal the `t7` key set (they do).
