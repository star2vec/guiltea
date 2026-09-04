"""S1g Task 2 — the same treatment for the act-detection result, on the held-out v2 chains.

reports/S1e-depth-matched.md section 3 found the harmful act plainly readable at fixed depth at t = 2
(`refusal` 0.786, `badmed` 0.774, unanimous over 8 targets, filler-robust, floor 0.468-0.589).
briefs/S1g-heldout-trigger.md pre-specifies the held-out version of that:

  cell      `committed` at t = 2 (positive) against `held` at t = 2 (negative), `held` restricted to
            non-filler turns - scripts/s1e/depth.py's `filler_excluded` variant
  axes      the same two, `nn` and `persona_meandiff`, and no others
  band      L14-18 mean    position  `answer`    label source  `t_primary`
  statistic the same depth- and target-matched fold statistic
  success   both axes above their largest seed's floor
  floor     the same count floor: 10 per side, and at least 3 targets holding both classes

If the counts do not reach the floor the verdict is "not testable on v2", which the brief names a
legitimate outcome that must not be replaced by loosening anything. This script therefore checks the
count floor FIRST and computes no axis statistic for a cell that does not clear it: a number produced
under a floor the cell cannot meet would only invite a post-hoc reading.

Also emits the per-turn class counts for both tasks (results/raw/s1g/counts_v2.json).

CPU only: no generation, no model load, no judge call, no GPU, no cost.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location("s1g_v2", REPO / "scripts" / "s1g" / "v2.py")
V = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(V)

PRESPECIFIED_TURN = 2


def main():
    D = V.bind_v2()
    proj, axes, positions, layers, chains, meta = V.load_chains_v2()

    # the brief's cell: `held` restricted to non-filler turns
    rows = V.class_table(chains, D.members_actdet, persuader_only=True)
    as_spec = V.class_table(chains, D.members_actdet, persuader_only=False)
    cellrow = rows[PRESPECIFIED_TURN]

    print("S1g Task 2 - act detection on v2, pre-specified at t = %d (`held` non-filler)"
          % PRESPECIFIED_TURN)
    print("  t   n+   n-  targets_both  clears the count floor (>=%d per side, >=%d targets)"
          % (V.FLOOR_MIN_CLASS, V.FLOOR_MIN_TARGETS))
    for t in sorted(rows):
        r = rows[t]
        print("  %-2d %4d %4d %10d      %s" % (t, r["n_positive"], r["n_negative"],
                                               r["n_targets_with_both_classes"],
                                               "yes" if r["clears_count_floor"] else "-"))

    clears = cellrow["clears_count_floor"]
    verdict = {"form": 3, "label": "not testable on v2",
               "text": "The counts did not reach the floor; they are reported."} if not clears else None
    if not clears:
        print("\n  The pre-specified cell holds n+ = %d, n- = %d over %d targets holding both classes."
              % (cellrow["n_positive"], cellrow["n_negative"],
                 cellrow["n_targets_with_both_classes"]))
        print("  It does not reach the count floor, so NO AXIS STATISTIC IS COMPUTED for it.")
        print("  VERDICT: form 3 - not testable on v2.")
    else:                                     # not reached on this data; kept so the path is explicit
        raise SystemExit("cell clears the floor: run the fixed statistic, do not shortcut it")

    # the S1e code path, for the record: it returns counts only for a cell below its own floor
    s1e = D.run_task("t2_actdet_v2", D.members_actdet,
                     "`committed` at t (positive) vs `held` at t (negative), same turn index",
                     {"prespecified_cell": "t = 2, `held` restricted to non-filler turns",
                      "prediction_fixed_in": "briefs/S1g-heldout-trigger.md blob e8510e5"})
    key = "%s|%s|filler_excluded" % (V.SOURCE, V.POSITION)
    assert s1e[key]["turns_with_a_verdict"] == [], s1e[key]["turns_with_a_verdict"]
    assert PRESPECIFIED_TURN not in s1e[key]["evaluable_turns"], s1e[key]["evaluable_turns"]

    out = {"meta": dict(meta, task="t2_actdet_v2", status="EXPLORATORY, held-out replication",
                        axes=list(V.AXES), position=V.POSITION, label_source=V.SOURCE,
                        band="L14-18", prespecified_turn=PRESPECIFIED_TURN,
                        variant="filler_excluded (`held` restricted to non-filler turns)",
                        count_floor={"min_class_per_side": V.FLOOR_MIN_CLASS,
                                     "min_targets_with_both_classes": V.FLOOR_MIN_TARGETS},
                        no_axis_statistic_computed=True, no_gpu_no_api=True),
           "prespecified_cell": {k: v for k, v in cellrow.items() if k != "items"},
           "class_table_filler_excluded": {str(t): {k: v for k, v in rows[t].items() if k != "items"}
                                           for t in rows},
           "class_table_as_specified": {str(t): {k: v for k, v in as_spec[t].items() if k != "items"}
                                        for t in as_spec},
           "verdict": verdict, "s1e_code_path": s1e}
    D.save(out, "t2_actdet_v2.json")

    t1 = json.load(open(V.OUT / "t1_trigger_v2.json", encoding="utf-8"))
    counts = {"note": "per-turn class counts on the held-out v2 chains, both contrasts",
              "status": "EXPLORATORY, held-out replication",
              "count_floor": {"min_class_per_side": V.FLOOR_MIN_CLASS,
                              "min_targets_with_both_classes": V.FLOOR_MIN_TARGETS},
              "t1_trigger": t1["class_table"],
              "t2_actdet_filler_excluded": out["class_table_filler_excluded"],
              "t2_actdet_as_specified": out["class_table_as_specified"],
              "committed_acts_per_turn": {
                  str(t): as_spec[t]["n_positive"] for t in sorted(as_spec)},
              "chains_with_a_committed_turn": sum(
                  1 for c in chains if "committed" in c["grades"][V.SOURCE]),
              "n_chains": len(chains)}
    D.save(counts, "counts_v2.json")
    print("  committed acts per turn index on v2:",
          [counts["committed_acts_per_turn"][str(t)] for t in range(1, 11)])
    print("  chains holding a `committed` turn: %d of %d"
          % (counts["chains_with_a_committed_turn"], counts["n_chains"]))


if __name__ == "__main__":
    main()
