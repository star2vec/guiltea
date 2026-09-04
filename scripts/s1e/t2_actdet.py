"""S1e Task 2 — depth-matched act detection: `committed` at turn t versus `held` at turn t.

S1c's act-detection question with conversation length removed. It is the control that says whether
Task 1's separation (if any) is about the future or merely about the present: at each turn index t,

  positives  v1 chains graded `committed` at t
  negatives  v1 chains graded `held` at t          (`partial` excluded)

Same protocol as Task 1 - same axes, layers, positions, label sources, floors and folds - and nothing
pooled across t. Also emits the per-t committed-act counts Task 3 reports, and the per-t class counts
and target coverage for the report's section 1.

EXPLORATORY. No API, no GPU, no model load, no cost.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location("s1e_depth", REPO / "scripts" / "s1e" / "depth.py")
D = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(D)


def counts_table(t1_path, t2_out):
    """The section 1 table: per-t class counts and target coverage for both contrasts, both sources."""
    t1 = json.load(open(t1_path, encoding="utf-8"))
    out = {"note": "per-turn class counts for both contrasts; `as_specified` is the brief's own "
                   "membership rule, `filler_excluded` drops turns the rig replaced with neutral filler",
           "status": "EXPLORATORY"}
    for task, src in (("t1_trigger", t1), ("t2_actdet", t2_out)):
        for source in D.SOURCES:
            for variant in ("as_specified", "filler_excluded"):
                key = "%s|answer|%s" % (source, variant)      # counts do not depend on the position
                rows = {}
                for t in range(1, D.N_TURNS + 1):
                    e = src[key]["per_t"][str(t)]
                    rows[str(t)] = {k: e[k] for k in (
                        "n_positive", "n_negative", "n_items", "n_targets",
                        "n_targets_with_both_classes", "filler_in_positive_class",
                        "filler_in_negative_class", "evaluable", "verdict_written")}
                out.setdefault(task, {})["%s|%s" % (source, variant)] = {
                    "per_t": rows, "evaluable_turns": src[key]["evaluable_turns"],
                    "turns_with_a_verdict": src[key]["turns_with_a_verdict"]}
    # Task 3's number: committed acts per turn index, both sources (reports/S1b-runs.md section 9 cross-check)
    for source in D.SOURCES:
        k = "%s|answer|as_specified" % source
        out.setdefault("committed_acts_per_turn", {})[source] = {
            str(t): t2_out[k]["per_t"][str(t)]["n_positive"] for t in range(1, D.N_TURNS + 1)}
    return out


def main():
    print("S1e Task 2 - depth-matched act detection (committed at t vs held at t)")
    out = D.run_task(
        "t2_actdet",
        D.members_actdet,
        "`committed` at t (positive) vs `held` at t (negative), same turn index",
        {"question": "does an axis read the act itself once conversation length is removed?",
         "positive_class": "graded `committed` at t",
         "negative_class": "graded `held` at t",
         "excluded": "`partial` at t"})
    D.save(out, "t2_actdet.json")
    D.save(counts_table(D.OUT / "t1_trigger.json", out), "counts.json")
    for key in sorted(k for k in out if k != "meta"):
        r = out[key]
        for t in r["evaluable_turns"]:
            v = r["per_t"][str(t)]["verdict"]["pooled"]["L14_18"]
            print("%-46s t=%-2d n+=%-4d n-=%-4d tgt_both=%-3d best %-16s band %.3f  floor max %.3f  beats %s"
                  % (key, t, r["per_t"][str(t)]["n_positive"], r["per_t"][str(t)]["n_negative"],
                     r["per_t"][str(t)]["n_targets_with_both_classes"], v["best_axis"],
                     v["best_axis_band_mean"], v["floor_band_mean"]["excess_max"], v["beats_floor"]))


if __name__ == "__main__":
    main()
