"""S1h Task 1 — the shame-like signature with the persona prompt held constant (no API, no GPU).

Within the vicious route, agent-directed fork (fork B) only: 64 `act-focused` replies against 19
`self-focused` ones, one system prompt, one route, one question wording. STAGE0 section 4.3 defines the
shame-like signature as movement on the persona axis away from Assistant; `reports/S1d-blame-target.md`
section 4.1 reported this cell only on best-over-layers against a selection-matched floor. This task
computes the pre-specified band statistic on it.

Headline: the L14-18 band mean, on BOTH summary statistics (pooled and the leave-one-target-out fold
statistic), against the band means of randctl seeds 0-9 and against the bag-of-words baseline, all three
in one table. Secondary band L6-11 and the full 32-layer sweep are in the JSON.

Reproduction check: the band means computed here must equal the ones S1d already stored for the same cell
in `results/raw/s1d/t4_q2.json` (contrasts.secondary_within_vicious_forkB.bands), which were never reported.
"""
from __future__ import annotations

import importlib.util
import json
import numpy as np
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location("s1h_cells", REPO / "scripts" / "s1h" / "cells.py")
K = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(K)

CELL = "vicious_forkB"


def reproduction_check(info):
    """S1d stored this cell's band means and never reported them; they must reproduce exactly."""
    src = json.load(open(K.C.OUT / "t4_q2.json", encoding="utf-8"))
    s1d = src["contrasts"]["secondary_within_vicious_forkB"]
    out = {"source": "results/raw/s1d/t4_q2.json contrasts.secondary_within_vicious_forkB",
           "n_positive": [s1d["n_positive"], info["n_positive"]],
           "n_negative": [s1d["n_negative"], info["n_negative"]], "axes": {}, "max_abs_diff": 0.0}
    worst = 0.0
    for band_s1d, band_here in (("primary_band_L14_18", "L14_18"), ("secondary_band_L6_11", "L6_11")):
        for ax in K.REPORT_AXES:
            for stat in K.STATS:
                a = s1d["bands"][band_s1d][ax][stat]
                b = info["bands"][stat][ax][band_here]
                worst = max(worst, abs(a - b))
                out["axes"]["%s|%s|%s" % (band_here, stat, ax)] = {"s1d": a, "s1h": b}
    for stat in K.STATS:
        a, b = s1d["bag_of_words"][stat], info["bag_of_words"][stat]
        worst = max(worst, abs(a - b))
        out["axes"]["bag_of_words|%s" % stat] = {"s1d": a, "s1h": b}
    out["max_abs_diff"] = worst
    out["reproduces"] = bool(worst < 1e-9 and s1d["n_positive"] == info["n_positive"]
                             and s1d["n_negative"] == info["n_negative"])
    assert out["reproduces"], out["max_abs_diff"]
    return out


def persona_verdict(info, band="L14_18"):
    """The brief's four-way verdict, computed for the persona axes on each summary statistic."""
    out = {}
    for stat in K.STATS:
        v = info["verdict"][stat][band]
        rows = {}
        for ax in K.PERSONA_AXES:
            e = v["per_axis"][ax]
            rows[ax] = {"band_mean": e["band_mean"], "excess_over_half": e["excess_over_half"],
                        "beats_floor": e["beats_floor"], "beats_words": e["beats_words"],
                        "form": ("clears both" if e["beats_floor"] and e["beats_words"] else
                                 "beats the words, not the floor" if e["beats_words"] else
                                 "beats the floor, not the words" if e["beats_floor"] else
                                 "clears neither")}
        any_both = any(r["form"] == "clears both" for r in rows.values())
        out[stat] = {"per_axis": rows,
                     "floor_band_mean": v["floor_band_mean"],
                     "bag_of_words": v["bag_of_words"],
                     "cell_form": ("clears both" if any_both else
                                   "beats the words, not the floor"
                                   if any(r["beats_words"] for r in rows.values()) else
                                   "beats the floor, not the words"
                                   if any(r["beats_floor"] for r in rows.values()) else "clears neither")}
    out["statistics_agree"] = bool(out["pooled"]["cell_form"] == out["by_target_mean"]["cell_form"])
    return out


def main():
    proj, axes, positions, layers, rows, meta = K.load_rows()
    info = K.run_cell(CELL, proj, axes, positions, layers, rows)
    meta.update({"task": "S1h Task 1 — the shame-like signature with the persona prompt held constant",
                 "headline_statistic": "L14-18 band mean, on both `pooled` and `by_target_mean`",
                 "floor": "randctl seeds 0-9, their own band means, on the same items and folds",
                 "primary_band": "L14-18 (D-024)", "secondary_band": "L6-11 (D-024)"})
    info["reproduction_check_vs_s1d"] = reproduction_check(info)
    info["persona_verdict_L14_18"] = persona_verdict(info, "L14_18")
    info["persona_verdict_L6_11"] = persona_verdict(info, "L6_11")
    out = {"meta": meta, "cell": info}
    K.save(out, "t1_vicious_forkB.json")

    print("\ncell: %s | %s | n+ %d n- %d | %d targets, %d with both classes"
          % (info["restricted_to"], info["contrast"], info["n_positive"], info["n_negative"],
             info["n_targets"], info["n_targets_with_both_classes"]))
    print("minority class by target:", info["minority_class_by_target"])
    print("reproduces S1d's stored bands:", info["reproduction_check_vs_s1d"]["reproduces"],
          "(max |diff| %.2e)" % info["reproduction_check_vs_s1d"]["max_abs_diff"])
    for band in ("L14_18", "L6_11"):
        for stat in K.STATS:
            print()
            K.print_table(info, stat, band)
    print("\nPERSONA VERDICT, headline band L14-18")
    pv = info["persona_verdict_L14_18"]
    for stat in K.STATS:
        print("  %-15s %s" % (stat, pv[stat]["cell_form"]))
        for ax, r in pv[stat]["per_axis"].items():
            print("     %-18s %.3f (excess %.3f) -> %s" % (ax, r["band_mean"], r["excess_over_half"], r["form"]))
    print("  statistics agree:", pv["statistics_agree"])
    bo = info["verdict"]["pooled"]["best_over_layers"]
    print("\nbest-over-layers beside its selection-matched floor (pooled): best %s %.3f at L%s (excess %.3f), "
          "matched floor min/mean/max %.3f/%.3f/%.3f, axes clearing %d of 9"
          % (bo["best_axis"], bo["best_axis_auroc"], bo["best_axis_layer"], bo["best_axis_excess"],
             bo["selection_matched_floor"]["min"], bo["selection_matched_floor"]["mean"],
             bo["selection_matched_floor"]["max"], bo["n_axes_beating_matched_floor"]))


if __name__ == "__main__":
    main()
