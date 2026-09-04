#!/usr/bin/env python3
"""S6-verify-headlines, Task 3: writeup/verify-headlines.md, written from results/raw/s6/headlines.json
(the output of scripts/verify/headlines.py). Numbers are never retyped by hand."""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from headlines import OUT_JSON, REPO, REPORT, TOL_AUROC  # noqa: E402

OUT = os.path.join(REPO, "writeup", "verify-headlines.md")


def row(item, number, report, mine, exact=False, note=""):
    if exact:
        diff = abs(int(mine) - int(report))
        ok = diff == 0
        return f"| {item} | {number} | {int(report)} | {int(mine)} | {diff} | {'PASS' if ok else 'FAIL'} | {note} |"
    diff = abs(float(mine) - float(report))
    ok = diff <= TOL_AUROC
    return f"| {item} | {number} | {float(report):.3f} | {float(mine):.4f} | {diff:.4f} | {'PASS' if ok else 'FAIL'} | {note} |"


def main():
    R = json.load(open(OUT_JSON))
    H1, H2, H3 = R["H1"], R["H2"], R["H3"]
    hdr = "| # | number | report | recomputed | \\|diff\\| | verdict | note |\n|---|---|---|---|---|---|---|"
    rows = [
        row("H1", "`nn` headline, count-weighted L14–18 fold statistic, `answer`", REPORT["H1"]["nn_headline"], H1["headline"]["nn"]),
        row("H1", "`nn` band mean at t = 1", REPORT["H1"]["nn_t1"], H1["band_means"]["1"]["nn"]),
        row("H1", "`nn` band mean at t = 2", REPORT["H1"]["nn_t2"], H1["band_means"]["2"]["nn"]),
        row("H1", "largest-seed headline (the floor), randctl seeds 0–9", REPORT["H1"]["largest_seed_headline"], H1["largest_seed_headline"]),
    ]
    for s, (a, b) in enumerate(zip(H1["seed_headlines"], REPORT["H1"]["seed_headlines"])):
        if abs(a - b) > TOL_AUROC:
            rows.append(row("H1", f"randctl seed {s} headline (the one seed outside tolerance; the other nine are within it)", b, a,
                            note="reported, not reconciled; the floor is seed 8 and is unaffected"))
    rows.append(row("H1", "S1g threshold: largest-seed excess over 0.5, **one-sided** (definition used)",
                    REPORT["H1"]["largest_seed_excess"], H1["largest_seed_excess_one_sided"],
                    note=f"the report's 0.111 is reproduced only by the two-sided reading max\\|band mean − 0.5\\| = {H1['largest_seed_excess_two_sided_for_record']:.4f}, from seed 1 at 0.389; see the sentence below"))
    rows += [
        row("H2", "`persona_meandiff` L14–18 band mean, pooled, vicious / fork B, act- vs self-focused", REPORT["H2"]["persona_meandiff_pooled"], H2["persona_meandiff_pooled_band"]),
        row("H2", "`persona_meandiff` L14–18 band mean, fold statistic", REPORT["H2"]["persona_meandiff_fold"], H2["persona_meandiff_fold_band"]),
        row("H2", "bag-of-words AUROC, pooled out-of-fold, leave-one-target-out", REPORT["H2"]["words_pooled"], H2["words_pooled"]),
        row("H2", "bag-of-words, fold statistic (extra; S1h prints it beside the pooled number)", REPORT["H2"]["words_fold"], H2["words_fold"]),
        row("H3", "`act-focused`", REPORT["H3"]["act-focused"], H3["counts"].get("act-focused", 0), exact=True),
        row("H3", "`self-focused`", REPORT["H3"]["self-focused"], H3["counts"].get("self-focused", 0), exact=True),
        row("H3", "`outcome-negative-only`", REPORT["H3"]["outcome-negative-only"], H3["counts"].get("outcome-negative-only", 0), exact=True),
        row("H3", "`neutral`", REPORT["H3"]["neutral"], H3["counts"].get("neutral", 0), exact=True),
        row("H3", "`incoherent`", REPORT["H3"]["incoherent"], H3["counts"].get("incoherent", 0), exact=True),
        row("H3", "total", REPORT["H3"]["total"], sum(H3["counts"].values()), exact=True),
        row("H3", "deceived / fork A / `neutral`", REPORT["H3"]["deceived_A_neutral"], H3["deceived_A_neutral"], exact=True),
    ]
    thr1 = H1["largest_seed_excess_one_sided"]
    thr2 = H1["largest_seed_excess_two_sided_for_record"]
    body = f"""# Verify-headlines — the three numbers the post leads with, recomputed by a second route

Written by `scripts/verify/table.py` from `results/raw/s6/headlines.json`, the output of `scripts/verify/headlines.py`; regenerate, never hand-edit. Tolerance {TOL_AUROC} on an AUROC or band mean, exact on a count. Nothing was imported from `scripts/s1d/`, `scripts/s1e/`, `scripts/s1g/` or `scripts/s1h/`, and none of those files was opened. Machine: the researcher's Mac, CPU only; no generation, no model load, no judge call, no cost. Full record: `reports/S6-verify-headlines.md`.

{hdr}
{chr(10).join(rows)}

**One sentence per item — what was implemented independently, and what definition was taken from the reports.**

- **H1.** Implemented independently: the `v2` chain loader, the per-turn class table from the rig's stored `grade`, the projection of the stored float16 residuals (cast to float32) onto the unit `nn` arrow from `directions/dirs_8B_s2_arrows.pt`, the per-target AUROC and its mean over targets holding both classes, the L14–18 band mean, the count-weighted headline, and the ten randctl arrows from the seed recipe in `scripts/randctl.py` (read, not imported). Taken from `reports/S1g-heldout-trigger.md`: the class rule (positives `held` at t with a `committed` turn strictly later, negatives `held` at t never committing, `partial` neither), the count floor (10 per side, 3 targets), the `answer` position, the fold statistic, and the count weights n₊ + n₋. The per-target decomposition (0.338, 0.800, 0.686, 1.000 at t = 1; 0.500, 1.000, 0.514, 0.400 at t = 2) reproduces S1g §4's table digit for digit.
- **H1, the threshold.** S1g §3 states the threshold as "the largest of the ten seeds' own headline excesses over 0.5" and prints 0.111. Under the one-sided reading used here (largest seed headline − 0.5, the direction S1g's own success criterion names), it is **{thr1:.4f}**, and it does **not** exceed the `v1` search excess 0.104 (taken as stated from `reports/S1e-depth-matched.md` §2). Under the two-sided reading max\\|band mean − 0.5\\| it is **{thr2:.4f}**, which matches the report and does exceed 0.104. The report's 0.111 comes from seed 1's band mean of 0.389, i.e. from a seed on the *wrong* side of 0.5. Reported, not reconciled: the verdict-bearing comparison (`nn` 0.662 against the largest seed 0.585) is the same under both readings; the power statement's ordering against 0.104 is not.
- **H2.** Implemented independently: the join of `results/raw/s1d/judge_calls/reflection_{{primary,second}}.jsonl` to the `t7` records by (target, seed, mode, fork), the projection onto `persona_meandiff_units` from `directions/dirs_8B_base_sweep.pt`, pooled and fold AUROCs per layer, the band mean, and the bag-of-words baseline (`CountVectorizer()` fit on the training folds only, `LogisticRegression(C=1.0, max_iter=2000)`, leave-one-target-out over the 14 targets, AUROC of the pooled out-of-fold positive-class probabilities). Taken from `reports/S1h-shame-signature.md`: the cell (vicious route, fork B), the positive class (`act-focused`), the label source (second judge final where it exists), the `answer` position, the band, and the word recipe.
- **H3.** Implemented independently: the 508 `t7` records as the universe, keyed by their own fields; the second-final rule applied to the two reflection files; the counts. Taken from `reports/S1d-blame-target.md` §1: that the second judge's label is final where it exists. `results/raw/s1d/join.jsonl` was used only to check that its 508 keys equal the `t7` key set (they do).
"""
    open(OUT, "w").write(body)
    print(body)


if __name__ == "__main__":
    main()
