"""S1h Task 3 — the guilt-like signature as the complement (no API, no GPU).

STAGE0 section 4.2 requires the persona axis FLAT where the act is evaluated. Within the deceived route,
act-directed fork (fork A) — the cell S1d's primary contrast is mostly made of — this task reports the
persona axes against the randctl seed floor on the same pre-specified band statistic Task 1 used.

A persona axis that separates in the vicious cell and sits at the floor here is the two signatures behaving
as sections 4.2 and 4.3 say they should, and that pair of results is the finding. The mismatch is reported
as such if the numbers do not do that.

The brief names the persona axes; the brief's own rule is that no axis number is reported without the seed
floor AND the word baseline in the same table, so the full nine-axis table is computed and the persona rows
are the ones read.

Reproduction check: the band means must equal the ones S1d stored for the same cell in
`results/raw/s1d/t4_q2.json` (contrasts.primary_within_deceived_forkA.bands).
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

CELL = "deceived_forkA"


def reproduction_check(info):
    src = json.load(open(K.C.OUT / "t4_q2.json", encoding="utf-8"))
    s1d = src["contrasts"]["primary_within_deceived_forkA"]
    worst = 0.0
    for band_s1d, band_here in (("primary_band_L14_18", "L14_18"), ("secondary_band_L6_11", "L6_11")):
        for ax in K.REPORT_AXES:
            for stat in K.STATS:
                worst = max(worst, abs(s1d["bands"][band_s1d][ax][stat] - info["bands"][stat][ax][band_here]))
    for stat in K.STATS:
        worst = max(worst, abs(s1d["bag_of_words"][stat] - info["bag_of_words"][stat]))
    out = {"source": "results/raw/s1d/t4_q2.json contrasts.primary_within_deceived_forkA",
           "n_positive": [s1d["n_positive"], info["n_positive"]],
           "n_negative": [s1d["n_negative"], info["n_negative"]], "max_abs_diff": worst,
           "reproduces": bool(worst < 1e-9 and s1d["n_positive"] == info["n_positive"]
                              and s1d["n_negative"] == info["n_negative"])}
    assert out["reproduces"], out
    return out


def flatness(info, band="L14_18"):
    """`Flat` here means exactly one thing: the axis's band-mean excess over 0.5 does not exceed the
    largest of the ten randctl seeds' own band-mean excesses. Reported for every axis, read for persona."""
    out = {}
    for stat in K.STATS:
        v = info["verdict"][stat][band]
        rows = {}
        for ax in K.REPORT_AXES:
            e = v["per_axis"][ax]
            rows[ax] = {"band_mean": e["band_mean"], "excess_over_half": e["excess_over_half"],
                        "at_the_floor": bool(not e["beats_floor"]), "beats_floor": e["beats_floor"],
                        "beats_words": e["beats_words"]}
        out[stat] = {"per_axis": rows, "floor_band_mean": v["floor_band_mean"],
                     "bag_of_words": v["bag_of_words"],
                     "persona_axes_flat": bool(all(rows[ax]["at_the_floor"] for ax in K.PERSONA_AXES))}
    out["statistics_agree_on_persona"] = bool(out["pooled"]["persona_axes_flat"]
                                              == out["by_target_mean"]["persona_axes_flat"])
    return out


def two_signature_reading(t1_path, flat):
    """The pair, stated mechanically from the two cells' numbers."""
    t1 = json.load(open(t1_path, encoding="utf-8"))["cell"]["persona_verdict_L14_18"]
    per_stat = {}
    for stat in K.STATS:
        sep = t1[stat]["cell_form"] in ("clears both", "beats the floor, not the words")
        per_stat[stat] = {
            "vicious_cell_persona_separates_above_floor": bool(sep),
            "vicious_cell_form": t1[stat]["cell_form"],
            "deceived_cell_persona_flat": flat[stat]["persona_axes_flat"],
            "pair_as_4_2_and_4_3_predict": bool(sep and flat[stat]["persona_axes_flat"])}
    return {"per_statistic": per_stat,
            "holds_on_both_statistics": bool(all(v["pair_as_4_2_and_4_3_predict"] for v in per_stat.values()))}


def main():
    proj, axes, positions, layers, rows, meta = K.load_rows()
    info = K.run_cell(CELL, proj, axes, positions, layers, rows)
    meta.update({"task": "S1h Task 3 — the guilt-like signature as the complement",
                 "headline_statistic": "L14-18 band mean, on both `pooled` and `by_target_mean`",
                 "floor": "randctl seeds 0-9, their own band means, on the same items and folds"})
    info["reproduction_check_vs_s1d"] = reproduction_check(info)
    info["flatness_L14_18"] = flatness(info, "L14_18")
    info["flatness_L6_11"] = flatness(info, "L6_11")
    info["two_signature_reading_L14_18"] = two_signature_reading(K.OUT / "t1_vicious_forkB.json",
                                                                 info["flatness_L14_18"])
    K.save({"meta": meta, "cell": info}, "t3_deceived_forkA.json")

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
    print("\nPERSONA AXES, headline band L14-18 — flat means: does not exceed the largest seed's band-mean excess")
    for stat in K.STATS:
        f = info["flatness_L14_18"][stat]
        for ax in K.PERSONA_AXES:
            r = f["per_axis"][ax]
            print("  %-15s %-18s %.3f (excess %.3f vs floor max %.3f) -> %s"
                  % (stat, ax, r["band_mean"], r["excess_over_half"], f["floor_band_mean"]["excess_max"],
                     "AT THE FLOOR" if r["at_the_floor"] else "ABOVE THE FLOOR"))
        print("  %-15s persona axes flat: %s" % (stat, f["persona_axes_flat"]))
    print("\ntwo-signature reading:", json.dumps(info["two_signature_reading_L14_18"], indent=1))
    print("\nper-target sign, L14-18, persona axes (targets with both = %d):" % info["n_targets_with_both_classes"])
    for ax in K.PERSONA_AXES + ["refusal"]:
        s = info["per_target_sign_L14_18"][ax]
        print("  %-18s above %d / below %d" % (ax, s["n_above_half"], s["n_below_half"]))
    sd = [info["per_target_sign_L14_18"]["random%d" % i]["n_above_half"] for i in range(10)]
    print("  seeds above-half: %s (min %d, max %d)" % (sd, min(sd), max(sd)))


if __name__ == "__main__":
    main()
