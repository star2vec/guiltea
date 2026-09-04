"""S1h Task 4 — the one table: S1d's pooled numbers beside this brief's restricted numbers.

Two blocks, each the same contrast read twice:
  the shame-like contrast   act-focused vs self-focused: pooled over all six route x fork cells (S1d,
                            450 vs 24) beside the vicious agent-directed cell (S1h, 64 vs 19)
  the complement            act-focused vs neutral:      pooled over all six cells (S1d, 450 vs 32)
                            beside the deceived act-directed cell (S1h, 82 vs 26)

The S1d columns are READ from `results/raw/s1d/t4_q2.json`, never recomputed. Its per-seed floor band
means are not stored as such, so they are formed here from the per-seed per-layer AUROCs the same file
holds, which makes the floor the same statistic in both columns.

Every row carries the seed floor and the word baseline, per the brief's rule.
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

BLOCKS = [("the shame-like contrast: act-focused vs self-focused", "secondary", "t1_vicious_forkB.json"),
          ("the complement: act-focused vs neutral", "primary", "t3_deceived_forkA.json")]
BAND_KEY = {"L14_18": "primary_band_L14_18", "L6_11": "secondary_band_L6_11"}


def s1d_side(contrast_name, band, stat):
    """One S1d pooled column: per-axis band mean, the seeds' own band means, the word baseline."""
    src = json.load(open(K.C.OUT / "t4_q2.json", encoding="utf-8"))["contrasts"][contrast_name]
    band_layers = K.C.BAND_PRIMARY if band == "L14_18" else K.C.BAND_SECONDARY
    seeds = {s: float(np.mean([src["raw_random"][s][str(L)][stat] for L in band_layers]))
             for s in K.C.RANDOM_AXES}
    ex = [abs(v - 0.5) for v in seeds.values()]
    words = src["bag_of_words"][stat]
    axes = {}
    for ax in K.REPORT_AXES:
        b = src["bands"][BAND_KEY[band]][ax][stat]
        axes[ax] = {"band_mean": b, "excess_over_half": abs(b - 0.5),
                    "beats_floor": bool(abs(b - 0.5) > max(ex)),
                    "beats_words": bool(abs(b - 0.5) > abs(words - 0.5))}
    return {"n_positive": src["n_positive"], "n_negative": src["n_negative"],
            "n_targets": src["n_targets"], "class_composition": src["class_composition"],
            "axes": axes,
            "floor_band_mean": {"per_seed": seeds, "auroc_min": float(min(seeds.values())),
                                "auroc_max": float(max(seeds.values())), "excess_max": float(max(ex))},
            "bag_of_words": {"auroc": words, "excess_over_half": abs(words - 0.5)}}


def s1h_side(fname, band, stat):
    c = json.load(open(K.OUT / fname, encoding="utf-8"))["cell"]
    v = c["verdict"][stat][band]
    return {"n_positive": c["n_positive"], "n_negative": c["n_negative"], "n_targets": c["n_targets"],
            "n_targets_with_both_classes": c["n_targets_with_both_classes"],
            "restricted_to": c["restricted_to"],
            "axes": {ax: {k: v["per_axis"][ax][k] for k in
                          ("band_mean", "excess_over_half", "beats_floor", "beats_words")}
                     for ax in K.REPORT_AXES},
            "floor_band_mean": v["floor_band_mean"], "bag_of_words": v["bag_of_words"]}


def md(block_title, left, right, band, stat):
    def cell(side, ax):
        e = side["axes"][ax]
        marks = ("F" if e["beats_floor"] else "-") + ("W" if e["beats_words"] else "-")
        return "%.3f %s" % (e["band_mean"], marks)
    lines = ["", "**%s** — band %s, %s. `F` = beats the seed floor, `W` = beats the word baseline."
             % (block_title, band.replace("_", "-"), stat),
             "", "| axis | pooled over all six cells (n+ %d / n- %d) | %s (n+ %d / n- %d) |"
             % (left["n_positive"], left["n_negative"], right["restricted_to"],
                right["n_positive"], right["n_negative"]),
             "|---|---|---|"]
    for ax in K.REPORT_AXES:
        lines.append("| %s | %s | %s |" % (ax, cell(left, ax), cell(right, ax)))
    lines.append("| **random floor (seeds 0-9 band means)** | %.3f-%.3f (excess max %.3f) | %.3f-%.3f (excess max %.3f) |"
                 % (left["floor_band_mean"]["auroc_min"], left["floor_band_mean"]["auroc_max"],
                    left["floor_band_mean"]["excess_max"], right["floor_band_mean"]["auroc_min"],
                    right["floor_band_mean"]["auroc_max"], right["floor_band_mean"]["excess_max"]))
    lines.append("| **bag-of-words** | %.3f | %.3f |"
                 % (left["bag_of_words"]["auroc"], right["bag_of_words"]["auroc"]))
    return "\n".join(lines)


def main():
    out = {"meta": {"task": "S1h Task 4 — the side-by-side table", "status": "EXPLORATORY",
                    "s1d_source": "results/raw/s1d/t4_q2.json (read, not recomputed)",
                    "note": "both columns use the same floor statistic: the ten randctl seeds' own band means"},
           "blocks": {}}
    text = []
    for title, s1d_contrast, s1h_file in BLOCKS:
        for band in ("L14_18", "L6_11"):
            for stat in K.STATS:
                left, right = s1d_side(s1d_contrast, band, stat), s1h_side(s1h_file, band, stat)
                out["blocks"]["%s|%s|%s" % (s1d_contrast, band, stat)] = {"title": title, "s1d_pooled": left,
                                                                          "s1h_restricted": right}
                if band == "L14_18":
                    text.append(md(title, left, right, band, stat))
    out["markdown_L14_18"] = "\n".join(text)
    K.save(out, "t4_sidebyside.json")
    print(out["markdown_L14_18"])


if __name__ == "__main__":
    main()
