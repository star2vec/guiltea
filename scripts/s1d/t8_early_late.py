"""S1d Task 8 (added 2026-09-04) — Q5: are early and late capitulation the same internal state? (no API, no GPU)

Early-capitulation acts (first committed turn at or before turn 3) versus late ones (turn 10), scored by
the projection at the committed turn itself, `answer` position, every layer. Random floor from randctl
seeds 0-9, folded by target.

The turn-index baseline is not a baseline here but the confound itself: the class label IS the turn
number, so depth separates the classes perfectly by construction and no axis can beat it. The question
that remains answerable is whether an axis clears the random floor — and, since the arrows are scored as
a max over 32 layers, each random seed is given the same max-over-layers search before the comparison.
"""
from __future__ import annotations

import importlib.util
import json
import numpy as np
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location("s1d_common", REPO / "scripts" / "s1d" / "common.py")
C = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(C)

REPORT_AXES = ["refusal", "badmed", "persona", "persona_meandiff",
               "guilt_clean", "shame_clean", "nn", "received_act", "received_self"]
POSITION = "answer"
EARLY_MAX = 3      # "immediate capitulation" as reports/S1b-runs.md section 9 counts it (T <= 3, n = 61)
LATE = 10


def first_committed(labels, target, seed, n_turns):
    for t in range(1, n_turns + 1):
        if labels.get((target, seed, "v1", t)) == "committed":
            return t
    return None


def analyse(source):
    proj, axes, positions, layers, keys = C.load_proj("t4v1")
    merged, prim, _ = C.act_label_table()
    labels = prim if source == "t_primary" else merged
    rows = []
    for i, k in enumerate(keys):
        target, seed, n_turns = k["target"], k["seed"], k["n_turns"]
        rec = json.load(open(C.RAW / "t4" / target / ("v1_seed%d.json" % seed), encoding="utf-8"))
        T = rec["T_primary"] if source == "t_primary" else first_committed(labels, target, seed, n_turns)
        if T is None:
            continue
        if T <= EARLY_MAX:
            y = 1
        elif T == LATE:
            y = 0
        else:
            continue
        rows.append({"chain": i, "target": target, "seed": seed, "T": T, "y": y})
    y = np.array([r["y"] for r in rows])
    grp = np.array([r["target"] for r in rows])
    ci = np.array([r["chain"] for r in rows])
    Ti = np.array([r["T"] for r in rows])
    per_target = {t: dict(Counter(rows[i]["y"] for i in range(len(rows)) if rows[i]["target"] == t))
                  for t in sorted(set(grp))}
    one_class = [t for t, c in per_target.items() if len(c) < 2]
    info = {"label_source": source, "positive_class": "early (T <= %d)" % EARLY_MAX, "negative_class": "late (T = %d)" % LATE,
            "n_early": int(y.sum()), "n_late": int((y == 0).sum()), "n_targets": len(per_target),
            "per_target_counts": {t: {"early": c.get(1, 0), "late": c.get(0, 0)} for t, c in per_target.items()},
            "targets_contributing_one_class_only": one_class,
            "n_targets_with_both_classes": len(per_target) - len(one_class),
            "turn_index_baseline": {"auroc": C.auroc(Ti, y),
                                    "note": "the class label is the turn number, so depth separates the classes "
                                            "perfectly by construction; this is the confound, not a baseline to beat"}}
    if info["n_early"] < 2 or info["n_late"] < 2:
        info["unworkable"] = "one class has fewer than 2 members"
        return info

    pi = positions.index(POSITION)
    S = proj[ci, Ti - 1, pi, :, :]
    table, raw_random = {}, {}
    for ax in REPORT_AXES:
        ai = axes.index(ax)
        table[ax] = {str(L): {"pooled": C.auroc(S[:, ai, L], y),
                              "by_target_mean": C.grouped_auroc(S[:, ai, L], y, grp)[0]} for L in layers}
    for ax in C.RANDOM_AXES:
        ai = axes.index(ax)
        raw_random[ax] = {str(L): {"pooled": C.auroc(S[:, ai, L], y),
                                   "by_target_mean": C.grouped_auroc(S[:, ai, L], y, grp)[0]} for L in layers}
    floor = {}
    for L in layers:
        for stat in ("pooled", "by_target_mean"):
            vals = [raw_random[ax][str(L)][stat] for ax in C.RANDOM_AXES]
            floor.setdefault(str(L), {})[stat] = {"mean": float(np.nanmean(vals)), "min": float(np.nanmin(vals)),
                                                  "max": float(np.nanmax(vals))}
    info["table"], info["random_floor"], info["raw_random"] = table, floor, raw_random
    for band_name, band in (("primary_band_L14_18", C.BAND_PRIMARY), ("secondary_band_L6_11", C.BAND_SECONDARY)):
        info.setdefault("bands", {})[band_name] = {
            ax: float(np.nanmean([table[ax][str(L)]["pooled"] for L in band])) for ax in REPORT_AXES}
        info["bands"][band_name]["random_floor_mean"] = float(np.nanmean(
            [floor[str(L)]["pooled"]["mean"] for L in band]))
    for stat in ("pooled", "by_target_mean"):
        def best(d):
            vals = [(d[str(L)][stat], L) for L in layers if not np.isnan(d[str(L)][stat])]
            return max(vals, key=lambda v: abs(v[0] - 0.5)) if vals else (float("nan"), None)
        axis_best = {ax: best(table[ax]) for ax in REPORT_AXES}
        seed_best = {ax: best(raw_random[ax]) for ax in C.RANDOM_AXES}
        matched = [abs(v[0] - 0.5) for v in seed_best.values()]
        top = max(axis_best.items(), key=lambda kv: abs(kv[1][0] - 0.5))
        info.setdefault("verdict", {})[stat] = {
            "per_axis_best": {ax: {"auroc": v[0], "layer": v[1], "excess_over_half": abs(v[0] - 0.5)}
                              for ax, v in axis_best.items()},
            "matched_random_floor": {"min": float(np.min(matched)), "mean": float(np.mean(matched)),
                                     "max": float(np.max(matched))},
            "best_axis": top[0], "best_axis_auroc": top[1][0], "best_axis_layer": top[1][1],
            "best_axis_excess": abs(top[1][0] - 0.5),
            "beats_matched_random_floor": abs(top[1][0] - 0.5) > float(np.max(matched)),
            "n_axes_beating_matched_floor": sum(1 for v in axis_best.values()
                                                if abs(v[0] - 0.5) > float(np.max(matched)))}
    return info


def main():
    out = {s: analyse(s) for s in ("t_primary", "merged")}
    json.dump(out, open(C.OUT / "t8_early_late.json", "w", encoding="utf-8"), indent=1, sort_keys=True)
    for s, i in out.items():
        print("==", s, "| early", i["n_early"], "late", i["n_late"], "| targets both classes",
              i["n_targets_with_both_classes"], "of", i["n_targets"],
              "| turn-index AUROC %.3f" % i["turn_index_baseline"]["auroc"])
        if "verdict" not in i:
            print("   ", i.get("unworkable"))
            continue
        for stat in ("pooled", "by_target_mean"):
            v = i["verdict"][stat]
            print("   %-14s best %s L%s AUROC %.3f (excess %.3f) | matched floor min/mean/max %.3f/%.3f/%.3f"
                  " | beats floor %s (%d/9)" % (stat, v["best_axis"], v["best_axis_layer"], v["best_axis_auroc"],
                                                v["best_axis_excess"], v["matched_random_floor"]["min"],
                                                v["matched_random_floor"]["mean"], v["matched_random_floor"]["max"],
                                                v["beats_matched_random_floor"], v["n_axes_beating_matched_floor"]))


if __name__ == "__main__":
    main()
