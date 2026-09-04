"""S1g Task 1 — THE HELD-OUT TEST: the S1e trigger prediction, on the untouched v2 chains.

The prediction was fixed in briefs/S1g-heldout-trigger.md before any v2 number was computed, and the
power to detect it was computed and committed first (scripts/s1g/t0b_power.py, report section 3).
Nothing below is free. At each turn index t that clears the count floor:

  positives  v2 chains graded `held` at t which have a `committed` turn later (strictly after t)
  negatives  v2 chains graded `held` at t which never commit within ten turns

axes      `nn` (predicted above 0.5) and `persona_meandiff` (predicted below 0.5) - and no others
band      L14-18 mean          position  `answer`          label source  `t_primary`
statistic the depth- and target-matched fold statistic: the mean of the per-target AUROCs over the
          targets holding both classes
floor     randctl seeds 0-9, band means, same items, same folds
headline  the count-weighted mean across the turn indices clearing the count floor
success   the headline excess over 0.5 exceeds the LARGEST seed's, in the predicted direction

The class table is built exactly as scripts/s1e/depth.py builds the v1 one: that module is imported
UNCHANGED and its attributes rebound by scripts/s1g/v2.py. Two independent paths compute the same
numbers - S1e's own run_task, and a direct pass through the same helpers - and the script asserts they
agree, so a rebinding mistake cannot pass silently.

t = 1 and t = 2 are DIFFERENT CLAIMS and are reported separately as well as in the headline. t = 1 is
the `situation` turn, the plain harmful request, held by construction from the hold screen, so a
separation there is about susceptibility before any persuasion has been applied - not the "see the
break coming" framing S1e used, which is t = 2.

CPU only: no generation, no model load, no judge call, no GPU, no cost.
"""
from __future__ import annotations

import importlib.util
import json
import numpy as np
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location("s1g_v2", REPO / "scripts" / "s1g" / "v2.py")
V = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(V)


def direction_ok(axis, band_mean):
    return band_mean > 0.5 if V.PREDICTED[axis] == "above" else band_mean < 0.5


def cell(band_means, floor_band_means):
    """The fixed decision: excess over 0.5 beating the largest seed's, in the predicted direction."""
    fx = {s: abs(v - 0.5) for s, v in floor_band_means.items()}
    fmax = float(max(fx.values()))
    out = {"floor": {"per_seed": floor_band_means,
                     "auroc_min": float(min(floor_band_means.values())),
                     "auroc_max": float(max(floor_band_means.values())),
                     "excess_max": fmax}, "per_axis": {}}
    for ax, bm in band_means.items():
        ex = abs(bm - 0.5)
        out["per_axis"][ax] = {"band_mean": float(bm), "excess_over_half": float(ex),
                               "predicted_direction": V.PREDICTED[ax],
                               "direction_ok": bool(direction_ok(ax, bm)),
                               "beats_largest_seed": bool(ex > fmax),
                               "clears": bool(direction_ok(ax, bm) and ex > fmax)}
    return out


def main():
    D = V.bind_v2()
    proj, axes, positions, layers, chains, meta = V.load_chains_v2()
    rows = V.class_table(chains, D.members_trigger)
    turns = V.floor_clearing_turns(rows)
    weights = {t: rows[t]["n_items"] for t in turns}
    assert turns, "no turn index clears the count floor"
    print("S1g Task 1 - held-out test on v2. floor-clearing turn indices: %s (weights %s)"
          % (turns, [weights[t] for t in turns]))

    names = list(V.AXES) + list(V.C.RANDOM_AXES)
    curves = {n: {t: V.fold_curve_at_turn(proj, axes, positions, layers, rows[t]["items"], n, t)
                  for t in turns} for n in names}
    per_turn_bm = {n: {t: V.band_mean(curves[n][t]) for t in turns} for n in names}
    headline_bm = {n: V.band_mean(V.weighted_curve(curves[n], turns, weights, layers)) for n in names}

    per_turn = {str(t): dict(cell({a: per_turn_bm[a][t] for a in V.AXES},
                                  {s: per_turn_bm[s][t] for s in V.C.RANDOM_AXES}),
                             **{k: v for k, v in rows[t].items() if k != "items"}) for t in turns}
    headline = cell({a: headline_bm[a] for a in V.AXES},
                    {s: headline_bm[s] for s in V.C.RANDOM_AXES})
    headline["turns"] = turns
    headline["weights_n_items"] = {str(t): weights[t] for t in turns}
    headline["note"] = ("count-weighted mean across the turn indices clearing the count floor; a "
                        "summary, not an average of independent samples - the same chains recur at "
                        "both turn indices")

    # ---- cross-check: the same numbers straight out of scripts/s1e/depth.py's own run_task
    s1e = D.run_task("t1_trigger_v2", D.members_trigger,
                     "held at t and commits later (positive) vs held at t and never commits (negative)",
                     {"held_out_set": "the 40 v2 chains in results/raw/s1b/t4, untouched by any prior "
                                      "analysis in this project",
                      "prediction_fixed_in": "briefs/S1g-heldout-trigger.md blob e8510e5",
                      "power_statement": "scripts/s1g/t0b_power.py, run and committed first"})
    ref = s1e["%s|%s|as_specified" % (V.SOURCE, V.POSITION)]
    assert ref["turns_with_a_verdict"] == turns, (ref["turns_with_a_verdict"], turns)
    worst = 0.0
    for t in turns:
        v = ref["per_t"][str(t)]["verdict"]["by_target_mean"]["L14_18"]
        for a in V.AXES:
            worst = max(worst, abs(v["per_axis"][a]["band_mean"] - per_turn_bm[a][t]))
        for s in V.C.RANDOM_AXES:
            worst = max(worst, abs(v["floor_band_mean"]["per_seed"][s] - per_turn_bm[s][t]))
    assert worst < 1e-9, worst
    print("cross-check against scripts/s1e/depth.py run_task: max abs difference %.2e" % worst)

    # the filler-excluded variant must coincide: v2 Task 1 holds no filler turn in either class
    fe = s1e["%s|%s|filler_excluded" % (V.SOURCE, V.POSITION)]
    same_counts = all(fe["per_t"][str(t)]["n_items"] == rows[t]["n_items"] for t in range(1, 11))
    assert same_counts, "filler present in a v2 Task 1 class - report it, do not silently drop it"

    out = {"meta": dict(meta, task="t1_trigger_v2", status="EXPLORATORY, held-out replication",
                        axes=list(V.AXES), predicted_direction=V.PREDICTED, position=V.POSITION,
                        label_source=V.SOURCE, band="L14-18",
                        statistic="mean of the per-target AUROCs over targets holding both classes",
                        count_floor={"min_class_per_side": V.FLOOR_MIN_CLASS,
                                     "min_targets_with_both_classes": V.FLOOR_MIN_TARGETS},
                        success_criterion="headline excess over 0.5 exceeds the largest seed's, in "
                                          "the predicted direction",
                        cross_check_max_abs_difference=float(worst),
                        filler_excluded_variant_identical=bool(same_counts),
                        no_gpu_no_api=True),
           "class_table": {str(t): {k: v for k, v in rows[t].items() if k != "items"} for t in rows},
           "floor_clearing_turns": turns, "per_turn": per_turn, "headline": headline,
           "s1e_code_path": s1e}
    D.save(out, "t1_trigger_v2.json")

    for t in turns:
        c, r = per_turn[str(t)], rows[t]
        print("\n  t = %d   n+ = %d  n- = %d  targets holding both = %d"
              % (t, r["n_positive"], r["n_negative"], r["n_targets_with_both_classes"]))
        print("    floor (seeds 0-9): %.3f - %.3f, largest excess %.3f"
              % (c["floor"]["auroc_min"], c["floor"]["auroc_max"], c["floor"]["excess_max"]))
        for a in V.AXES:
            p = c["per_axis"][a]
            print("    %-18s %.3f  excess %.3f  predicted %-5s  direction %-3s  clears %s"
                  % (a, p["band_mean"], p["excess_over_half"], p["predicted_direction"],
                     "ok" if p["direction_ok"] else "NO", "YES" if p["clears"] else "no"))
    print("\n  HEADLINE (count-weighted over t = %s)" % turns)
    print("    floor (seeds 0-9): %.3f - %.3f, largest excess %.3f"
          % (headline["floor"]["auroc_min"], headline["floor"]["auroc_max"],
             headline["floor"]["excess_max"]))
    for a in V.AXES:
        p = headline["per_axis"][a]
        print("    %-18s %.3f  excess %.3f  predicted %-5s  direction %-3s  CLEARS %s"
              % (a, p["band_mean"], p["excess_over_half"], p["predicted_direction"],
                 "ok" if p["direction_ok"] else "NO", "YES" if p["clears"] else "no"))
    print("\n  VERDICT: %s" % ("REPLICATED" if headline["per_axis"]["nn"]["clears"]
                               else "NOT REPLICATED (on the primary axis nn)"))


if __name__ == "__main__":
    main()
