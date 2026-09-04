"""S1h Task 2 — the guilt-like and shame-like arrows with the persona prompt held constant (no API, no GPU).

Same cell as Task 1 (vicious route, agent-directed fork), same protocol, same table — this task reads
Task 1's JSON rather than recomputing, so the arrows are read off the very table that carries the seed
floor and the word baseline. Two things are reported and they are kept apart:

  DIRECTION   which class projects higher. `reports/S1d-blame-target.md` section 4 found both cleaned
              arrows ordering the POOLED classes with `self-focused` higher, the direction STAGE0 4.4
              predicts. The question here is whether that holds with the prompt held constant. Direction
              is read two ways, both of them differences (the absolute sign of a projection is not
              interpretable, S1d section 6): the AUROC's side of 0.5, and the class mean difference along
              the axis. The per-target sign count says whether it is the set or one target.

  SEPARATION  do the arrows clear the randctl seed floor on the band statistic, and do they beat a word
              baseline that reaches only 0.575 in this cell.

This is the fairest test the project has of the guilt/shame arrows: the labels come from the subject's own
behaviour and the lexical cue is weak. Whichever way it falls it is a finding, and nothing here is
confirmatory.
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


def s1d_pooled():
    """S1d's pooled secondary contrast (act-focused vs self-focused, all six cells), read not recomputed."""
    src = json.load(open(K.C.OUT / "t4_q2.json", encoding="utf-8"))["contrasts"]["secondary"]
    return {"n_positive": src["n_positive"], "n_negative": src["n_negative"],
            "class_composition": src["class_composition"],
            "bands": {ax: {"L14_18": {s: src["bands"]["primary_band_L14_18"][ax][s] for s in K.STATS},
                           "L6_11": {s: src["bands"]["secondary_band_L6_11"][ax][s] for s in K.STATS}}
                      for ax in K.REPORT_AXES},
            "bag_of_words": {s: src["bag_of_words"][s] for s in K.STATS},
            "source": "results/raw/s1d/t4_q2.json contrasts.secondary (pooled over route and fork)"}


def main():
    t1 = json.load(open(K.OUT / "t1_vicious_forkB.json", encoding="utf-8"))
    c = t1["cell"]
    pooled_s1d = s1d_pooled()
    out = {"meta": dict(t1["meta"], task="S1h Task 2 — the guilt-like and shame-like arrows, same cell",
                        arrows=K.ARROW_AXES,
                        note="reads results/raw/s1h/t1_vicious_forkB.json; the same table, the same floor, "
                             "the same word baseline"),
           "cell": {k: c[k] for k in ("cell", "restricted_to", "probe_question", "contrast", "positive_class",
                                      "below_half_means", "n_positive", "n_negative", "n_targets",
                                      "n_targets_with_both_classes", "bag_of_words")},
           "s1d_pooled_for_comparison": pooled_s1d,
           "direction": {}, "separation": {}}

    for ax in K.ARROW_AXES:
        d = {"per_band": {}, "per_target_sign_L14_18": c["per_target_sign_L14_18"][ax],
             "s1d_pooled_band_means": pooled_s1d["bands"][ax]}
        for band, _ in K.BANDS:
            e = c["direction_band_means"][ax][band]
            per_stat = {}
            for stat in K.STATS:
                a = c["bands"][stat][ax][band]
                per_stat[stat] = {"auroc": a,
                                  "higher_class": (c["contrast"].split(" vs ")[1] if a < 0.5
                                                   else c["positive_class"])}
            s1d_side = {stat: ("self-focused" if pooled_s1d["bands"][ax][band][stat] < 0.5 else "act-focused")
                        for stat in K.STATS}
            d["per_band"][band] = {
                "auroc": per_stat,
                "mean_projection": e,
                "s1d_pooled_higher_class": s1d_side,
                "direction_holds_vs_s1d": bool(all(per_stat[s]["higher_class"] == s1d_side[s] for s in K.STATS))}
        sign = d["per_target_sign_L14_18"]
        d["per_target_sign_summary"] = "%d of %d targets below 0.5 (self-focused projects higher there)" % (
            sign["n_below_half"], sign["n_targets"])
        out["direction"][ax] = d

        sep = {}
        for band, _ in K.BANDS:
            per_stat = {}
            for stat in K.STATS:
                v = c["verdict"][stat][band]
                e2 = v["per_axis"][ax]
                per_stat[stat] = {"band_mean": e2["band_mean"], "excess_over_half": e2["excess_over_half"],
                                  "beats_floor": e2["beats_floor"], "beats_words": e2["beats_words"],
                                  "floor_excess_max": v["floor_band_mean"]["excess_max"],
                                  "floor_band_means": [v["floor_band_mean"]["auroc_min"],
                                                       v["floor_band_mean"]["auroc_max"]],
                                  "words_auroc": v["bag_of_words"]["auroc"],
                                  "words_excess": v["bag_of_words"]["excess_over_half"]}
            sep[band] = per_stat
        out["separation"][ax] = sep

    # the same two questions for the persona axes, so the arrows are read beside them in one place
    out["persona_axes_for_comparison"] = {
        ax: {band: {stat: c["bands"][stat][ax][band] for stat in K.STATS} for band, _ in K.BANDS}
        for ax in K.PERSONA_AXES}
    K.save(out, "t2_arrows.json")

    print("\ncell: %s | %s | n+ %d n- %d" % (c["restricted_to"], c["contrast"], c["n_positive"], c["n_negative"]))
    print("\nDIRECTION (band L14-18) — below 0.5 means `self-focused` projects higher, the STAGE0 4.4 direction")
    for ax in K.ARROW_AXES:
        d = out["direction"][ax]["per_band"]["L14_18"]
        print("  %-12s pooled %.3f (%s) | fold %.3f (%s) | mean proj diff (self - act) %+.3f | S1d pooled: %s | holds: %s"
              % (ax, d["auroc"]["pooled"]["auroc"], d["auroc"]["pooled"]["higher_class"],
                 d["auroc"]["by_target_mean"]["auroc"], d["auroc"]["by_target_mean"]["higher_class"],
                 d["mean_projection"]["difference_negative_minus_positive"],
                 d["s1d_pooled_higher_class"]["pooled"], d["direction_holds_vs_s1d"]))
        print("               %s" % out["direction"][ax]["per_target_sign_summary"])
    print("\nSEPARATION (band L14-18)")
    for ax in K.ARROW_AXES:
        for stat in K.STATS:
            s = out["separation"][ax]["L14_18"][stat]
            print("  %-12s %-14s %.3f  excess %.3f | floor max %.3f -> %s | words %.3f -> %s"
                  % (ax, stat, s["band_mean"], s["excess_over_half"], s["floor_excess_max"],
                     "CLEARS" if s["beats_floor"] else "no", s["words_auroc"],
                     "BEATS" if s["beats_words"] else "no"))


if __name__ == "__main__":
    main()
