"""S1g Task 0b — THE POWER STATEMENT, computed and written down BEFORE any v2 axis number.

briefs/S1g-heldout-trigger.md requires this to run first. At the class sizes the v2 held-out set
actually offers, a failure to replicate may say more about the sample than about the effect, so the
smallest margin this test could have distinguished from its floor is established before the two
pre-named axes are looked at at all.

This script touches NO named axis. It computes, on the v2 Task 1 classes as they stand:

  1. the class table (labels and turn kinds only), and which turn indices clear the brief's count
     floor - 10 per side and at least 3 targets holding both classes;
  2. the ten randctl seeds' L14-18 band means of the depth- and target-matched fold statistic, per
     turn index and as the count-weighted headline over the floor-clearing turns;
  3. from that null, the smallest headline excess over 0.5 this test could have called a clear -
     the success criterion is "exceeds the LARGEST seed's headline", so any true effect at or below
     the largest seed's own excess could not have been distinguished from the floor;
  4. whether the v1 effect size reported in reports/S1e-depth-matched.md section 2 (headline 0.604,
     excess 0.104, against a largest-seed floor of 0.541, excess 0.041) sits inside or outside it.

A grep of this file for `nn` or `persona_meandiff` as an axis returns nothing but the v1 reference
figures quoted from the S1e report, which were computed on v1 long before this session.

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

# reports/S1e-depth-matched.md section 2, the search-sample effect this test is powered against.
V1_HEADLINE = {"axis": "nn", "band_mean": 0.604, "excess_over_half": 0.104,
               "largest_seed_band_mean": 0.541, "largest_seed_excess": 0.041,
               "source": "reports/S1e-depth-matched.md section 2, count-weighted over t = 1..9"}


def main():
    D = V.bind_v2()
    proj, axes, positions, layers, chains, meta = V.load_chains_v2()
    print("v2: %d chains over %d targets, %d turn labels, %d stored-grade mismatches"
          % (meta["n_chains"], meta["n_targets"], meta["n_turn_labels"],
             meta["stored_grade_vs_act_primary_mismatches"]))

    rows = V.class_table(chains, D.members_trigger)
    turns = V.floor_clearing_turns(rows)
    weights = {t: rows[t]["n_items"] for t in turns}
    print("turn indices clearing the count floor (>=%d per side, >=%d targets holding both): %s"
          % (V.FLOOR_MIN_CLASS, V.FLOOR_MIN_TARGETS, turns))

    # --- the null: the ten randctl seeds on these exact classes, this exact statistic
    per_seed_per_t, per_seed_headline = {}, {}
    for s in V.C.RANDOM_AXES:
        curves = {t: V.fold_curve_at_turn(proj, axes, positions, layers, rows[t]["items"], s, t)
                  for t in turns}
        per_seed_per_t[s] = {str(t): V.band_mean(curves[t]) for t in turns}
        per_seed_headline[s] = V.band_mean(V.weighted_curve(curves, turns, weights, layers))

    hv = np.array([per_seed_headline[s] for s in V.C.RANDOM_AXES], dtype=float)
    hx = np.abs(hv - 0.5)
    detectable = float(hx.max())
    v1_inside = bool(V1_HEADLINE["excess_over_half"] <= detectable)

    per_t_floor = {}
    for t in turns:
        v = np.array([per_seed_per_t[s][str(t)] for s in V.C.RANDOM_AXES], dtype=float)
        x = np.abs(v - 0.5)
        per_t_floor[str(t)] = {"auroc_min": float(v.min()), "auroc_mean": float(v.mean()),
                               "auroc_max": float(v.max()), "excess_min": float(x.min()),
                               "excess_mean": float(x.mean()), "excess_max": float(x.max()),
                               "per_seed": {s: per_seed_per_t[s][str(t)] for s in V.C.RANDOM_AXES}}

    out = {
        "meta": dict(meta, task="t0b_power", status="EXPLORATORY",
                     note="the power statement, computed before any v2 axis number",
                     statistic="L14-18 band mean of the depth- and target-matched fold statistic",
                     position=V.POSITION, label_source=V.SOURCE,
                     count_floor={"min_class_per_side": V.FLOOR_MIN_CLASS,
                                  "min_targets_with_both_classes": V.FLOOR_MIN_TARGETS,
                                  "deviation": "relaxed from S1e's 5 targets to 3, the brief's one "
                                               "stated deviation, because v2 spans only 5 targets"},
                     no_named_axis_computed=True, no_gpu_no_api=True),
        "class_table": {str(t): {k: v for k, v in rows[t].items() if k != "items"} for t in rows},
        "floor_clearing_turns": turns,
        "headline_weights_n_items": {str(t): weights[t] for t in turns},
        "null_per_turn": per_t_floor,
        "null_headline": {"per_seed": per_seed_headline,
                          "auroc_min": float(hv.min()), "auroc_mean": float(hv.mean()),
                          "auroc_max": float(hv.max()),
                          "excess_min": float(hx.min()), "excess_mean": float(hx.mean()),
                          "excess_max": detectable},
        "power": {
            "smallest_distinguishable_headline_excess": detectable,
            "rule": "the success criterion is that the headline excess over 0.5 exceeds the largest "
                    "of the ten seeds' own headline excesses, so an effect at or below that value "
                    "could not have been called a clear on this sample",
            "v1_effect": V1_HEADLINE,
            "v1_effect_inside_the_undetectable_range": v1_inside,
        },
    }
    V.OUT.mkdir(parents=True, exist_ok=True)
    p = V.OUT / "t0b_power.json"
    json.dump(out, open(p, "w", encoding="utf-8"), indent=1, sort_keys=True, allow_nan=True)

    print("\nclass table (v2, Task 1, t_primary, answer):")
    print("  t   n+   n-  targets_both  filler+  filler-  clears floor")
    for t in sorted(rows):
        r = rows[t]
        print("  %-2d %4d %4d %10d %9d %8d   %s"
              % (t, r["n_positive"], r["n_negative"], r["n_targets_with_both_classes"],
                 r["filler_in_positive_class"], r["filler_in_negative_class"],
                 "yes" if r["clears_count_floor"] else "-"))
    print("\nnull, ten randctl seeds, L14-18 band mean of the fold statistic:")
    for t in turns:
        f = per_t_floor[str(t)]
        print("  t=%-2d  %.3f - %.3f   (largest excess %.3f)"
              % (t, f["auroc_min"], f["auroc_max"], f["excess_max"]))
    print("  headline (count-weighted over t=%s): %.3f - %.3f, largest excess %.3f"
          % (turns, hv.min(), hv.max(), detectable))
    print("\nPOWER: the smallest headline excess this test could have distinguished from its floor "
          "is %.3f." % detectable)
    print("The v1 effect (excess %.3f) is %s that range."
          % (V1_HEADLINE["excess_over_half"], "INSIDE (undetectable)" if v1_inside else "OUTSIDE (detectable)"))
    print("->", p.relative_to(REPO))


if __name__ == "__main__":
    main()
