"""S1h shared machinery — one fixed route x fork cell of the S1b probe replies (no API, no GPU).

Every comparison in this brief is made INSIDE one cell, so the system prompt, the route, the question
wording and the conversation shape are held constant by construction and nothing is pooled across cells.

Scores are the stored projections at the `answer` position of the probe reply, read from
`results/raw/s1d/proj_t7.npz` exactly as S1d built them ([508, 3 positions, 12 named axes + randctl seeds
0-9, 32 layers]). Nothing is recomputed from residuals here and no forward pass is run.

Reuse, edited neither: `scripts/s1d/common.py` (load_proj, auroc, grouped_auroc, the two bands, the seed
list), `scripts/s1d/t4_q2.py` (reflection_final, bow_oof — S1d's bag-of-words recipe), `scripts/s1e/depth.py`
(_bands, _verdict — the per-seed band-mean floor and the selection-matched best-over-layers floor).

Two summary statistics are computed for every number and BOTH are headline (the researcher's instruction):
  `pooled`          one AUROC over all items in the cell
  `by_target_mean`  the mean of the per-target AUROCs, i.e. the leave-one-target-out fold statistic
A pooled value is still free to separate on target identity rather than on state (reports/S1e-depth-matched.md
section 2 demonstrates exactly that), and the fold statistic is noisy at 1-4 minority members per target, so
neither alone carries a verdict.
"""
from __future__ import annotations

import importlib.util
import json
import numpy as np
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, REPO / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


C = _load("s1d_common", "scripts/s1d/common.py")
T4 = _load("s1d_t4_q2", "scripts/s1d/t4_q2.py")
D = _load("s1e_depth", "scripts/s1e/depth.py")

OUT = REPO / "results" / "raw" / "s1h"
S3 = ["refusal", "badmed", "persona", "persona_meandiff"]
S2 = ["guilt_clean", "shame_clean", "nn", "received_act", "received_self"]
REPORT_AXES = S3 + S2
PERSONA_AXES = ["persona", "persona_meandiff"]
ARROW_AXES = ["guilt_clean", "shame_clean", "nn"]
POSITION = "answer"
STATS = ("pooled", "by_target_mean")
MIN_CLASS = 10          # brief: minimum 10 per side
BANDS = (("L14_18", C.BAND_PRIMARY), ("L6_11", C.BAND_SECONDARY))

# The three cells are fixed by the brief and are never narrowed to improve a margin.
CELLS = {
    "vicious_forkB": {"mode": "vicious", "fork": "B", "pos": "act-focused", "neg": "self-focused",
                      "role": "the shame-like signature with the persona prompt held constant "
                              "(Tasks 1 and 2)"},
    "deceived_forkA": {"mode": "deceived", "fork": "A", "pos": "act-focused", "neg": "neutral",
                       "role": "the guilt-like complement: the act evaluated (Task 3)"},
}
FORK_QUESTION = {"A": "was the information in it medically sound?   (act-directed)",
                 "B": "was it right for you to give that answer?    (agent-directed)"}


def load_rows():
    """One row per probe reply: cell keys, the reflection label (D-019 final) and the reply text."""
    proj, axes, positions, layers, keys = C.load_proj("t7")
    join = {(r["target"], r["mode"], r["fork"], r["seed"]): r
            for r in (json.loads(l) for l in open(C.OUT / "join.jsonl", encoding="utf-8"))}
    final = T4.reflection_final()
    rows = []
    for k in keys:
        key = (k["target"], k["mode"], k["fork"], k["seed"])
        rows.append({"target": k["target"], "mode": k["mode"], "fork": k["fork"], "seed": k["seed"],
                     "label": final.get(key), "answer": join[key]["answer"]})
    meta = {"proj_source": "results/raw/s1d/proj_t7.npz (scripts/s1d/proj.py)",
            "proj_shape": list(proj.shape), "position": POSITION, "n_probe_replies": len(rows),
            "label_counts": dict(Counter(str(r["label"]) for r in rows)),
            "labels": "second judge where it exists, else primary (D-019), as scripts/s1d/t4_q2.py computes it",
            "no_gpu_no_api_no_cost": True}
    return proj, axes, positions, layers, rows, meta


def run_cell(cell_key, proj, axes, positions, layers, rows):
    """Every axis, every randctl seed and the word baseline on one cell, at all 32 layers."""
    spec = CELLS[cell_key]
    pos_label, neg_label = spec["pos"], spec["neg"]
    sel = [i for i, r in enumerate(rows)
           if r["label"] in (pos_label, neg_label) and r["mode"] == spec["mode"] and r["fork"] == spec["fork"]]
    y = np.array([1 if rows[i]["label"] == pos_label else 0 for i in sel])
    grp = np.array([rows[i]["target"] for i in sel])
    texts = [rows[i]["answer"] for i in sel]
    n_pos, n_neg = int(y.sum()), int((y == 0).sum())

    per_target = {g: {"pos": int(((grp == g) & (y == 1)).sum()), "neg": int(((grp == g) & (y == 0)).sum())}
                  for g in sorted(set(grp.tolist()))}
    both = [g for g, c in per_target.items() if c["pos"] and c["neg"]]
    info = {"cell": cell_key, "role": spec["role"], "status": "EXPLORATORY",
            "restricted_to": "%s route, fork %s" % (spec["mode"], spec["fork"]),
            "probe_question": FORK_QUESTION[spec["fork"]],
            "contrast": "%s vs %s" % (pos_label, neg_label), "positive_class": pos_label,
            "below_half_means": "%s projects higher" % neg_label,
            "n_positive": n_pos, "n_negative": n_neg, "n_items": len(sel),
            "n_targets": len(per_target), "n_targets_with_both_classes": len(both),
            "targets_with_both_classes": both, "per_target_counts": per_target,
            "minority_class_by_target": {g: c["neg"] for g, c in sorted(per_target.items(), key=lambda kv: -kv[1]["neg"])
                                         if c["neg"]},
            "meets_min_class_size": bool(n_pos >= MIN_CLASS and n_neg >= MIN_CLASS),
            "min_class_size": MIN_CLASS,
            "class_composition": {lab: dict(Counter("%s/%s" % (rows[i]["mode"], rows[i]["fork"])
                                                    for i in sel if rows[i]["label"] == lab))
                                  for lab in (pos_label, neg_label)}}
    assert info["meets_min_class_size"], (cell_key, n_pos, n_neg)

    # --- bag-of-words on the same folds, S1d's recipe, unedited
    bow, bmask, folds_used, folds_skipped = T4.bow_oof(texts, y, grp)
    info["bag_of_words"] = {
        "pooled": C.auroc(bow[bmask], y[bmask]),
        "by_target_mean": C.grouped_auroc(bow[bmask], y[bmask], grp[bmask])[0],
        "n_scored": int(bmask.sum()), "folds_used": folds_used, "folds_skipped_single_class": folds_skipped,
        "recipe": "CountVectorizer() word counts + LogisticRegression(C=1.0, max_iter=2000), "
                  "leave-one-target-out, pooled out-of-fold decision scores"}

    pi = positions.index(POSITION)
    S = proj[np.array(sel)][:, pi, :, :]                       # [n_items, n_axes, 32]
    all_axes = REPORT_AXES + list(C.RANDOM_AXES)
    curves = {stat: {} for stat in STATS}
    for ax in all_axes:
        ai = axes.index(ax)
        cp, cg = {}, {}
        for L in layers:
            cp[str(L)] = C.auroc(S[:, ai, L], y)
            cg[str(L)] = C.grouped_auroc(S[:, ai, L], y, grp)[0]
        curves["pooled"][ax] = cp
        curves["by_target_mean"][ax] = cg

    for stat in STATS:
        info.setdefault("table", {})[stat] = {ax: curves[stat][ax] for ax in REPORT_AXES}
        info.setdefault("raw_random", {})[stat] = {s: curves[stat][s] for s in C.RANDOM_AXES}
        info.setdefault("floor_by_layer", {})[stat] = {
            str(L): {"min": float(np.nanmin([curves[stat][s][str(L)] for s in C.RANDOM_AXES])),
                     "mean": float(np.nanmean([curves[stat][s][str(L)] for s in C.RANDOM_AXES])),
                     "max": float(np.nanmax([curves[stat][s][str(L)] for s in C.RANDOM_AXES]))}
            for L in layers}
        info.setdefault("bands", {})[stat] = {ax: D._bands(curves[stat][ax], layers) for ax in REPORT_AXES}
        v = D._verdict({ax: curves[stat][ax] for ax in REPORT_AXES},
                       {s: curves[stat][s] for s in C.RANDOM_AXES}, layers)
        # the word baseline is one out-of-fold AUROC on the same folds, not a per-layer curve, so it enters
        # every band comparison as a single number, on the same |AUROC - 0.5| footing as the axes
        words = info["bag_of_words"][stat]
        wex = abs(words - 0.5)
        for band, _ in BANDS:
            v[band]["bag_of_words"] = {"auroc": words, "excess_over_half": wex}
            for ax in REPORT_AXES:
                e = v[band]["per_axis"][ax]["excess_over_half"]
                f = v[band]["floor_band_mean"]["excess_max"]
                v[band]["per_axis"][ax].update({
                    "beats_floor": bool(not np.isnan(e) and e > f),
                    "beats_words": bool(not np.isnan(e) and e > wex),
                    "clears_both": bool(not np.isnan(e) and e > f and e > wex)})
            v[band]["n_axes_beating_words"] = int(sum(1 for ax in REPORT_AXES
                                                      if v[band]["per_axis"][ax]["beats_words"]))
            v[band]["n_axes_clearing_both"] = int(sum(1 for ax in REPORT_AXES
                                                      if v[band]["per_axis"][ax]["clears_both"]))
        v["best_over_layers"]["bag_of_words"] = {"auroc": words, "excess_over_half": wex}
        info.setdefault("verdict", {})[stat] = v

    # --- direction, reported separately from separation (Task 2)
    # The absolute sign of a projection is not interpretable (reports/S1d-blame-target.md section 6), so
    # direction is read two ways that are both differences: the AUROC's side of 0.5, and the class mean
    # difference along the axis.
    direction = {}
    for ax in REPORT_AXES:
        ai = axes.index(ax)
        d = {}
        for band, band_layers in BANDS:
            mp = float(np.mean([S[y == 1, ai, L].mean() for L in band_layers]))
            mn = float(np.mean([S[y == 0, ai, L].mean() for L in band_layers]))
            d[band] = {"mean_projection_positive_class": mp, "mean_projection_negative_class": mn,
                       "difference_negative_minus_positive": mn - mp,
                       "higher_class": neg_label if mn > mp else pos_label}
        direction[ax] = d
    info["direction_band_means"] = direction

    # --- per-target sign count on the primary band: is the fold statistic carried by the set or one target?
    sign = {}
    for ax in REPORT_AXES + list(C.RANDOM_AXES):
        ai = axes.index(ax)
        vals = {}
        for g in both:
            m = grp == g
            vals[g] = float(np.nanmean([C.auroc(S[m, ai, L], y[m]) for L in C.BAND_PRIMARY]))
        above = sum(1 for x in vals.values() if x > 0.5)
        sign[ax] = {"per_target_band_L14_18": vals, "n_targets": len(vals),
                    "n_above_half": above, "n_below_half": len(vals) - above}
    info["per_target_sign_L14_18"] = sign
    return info


def save(obj, fname):
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / fname
    json.dump(obj, open(p, "w", encoding="utf-8"), indent=1, sort_keys=True, allow_nan=True)
    print("->", p, "%.0f kB" % (p.stat().st_size / 1e3))
    return p


def band_row(info, ax, stat, band):
    """(band mean, excess over half, beats_floor, beats_words) for one axis."""
    e = info["verdict"][stat][band]["per_axis"][ax]
    return e["band_mean"], e["excess_over_half"], e["beats_floor"], e["beats_words"]


def print_table(info, stat, band="L14_18"):
    v = info["verdict"][stat][band]
    f = v["floor_band_mean"]
    print("  -- %s, band %s, %s (n+ %d, n- %d, %d targets, %d with both)"
          % (info["cell"], band, stat, info["n_positive"], info["n_negative"],
             info["n_targets"], info["n_targets_with_both_classes"]))
    for ax in REPORT_AXES:
        b, e, bf, bw = band_row(info, ax, stat, band)
        print("     %-18s %.3f  excess %.3f  floor %s  words %s"
              % (ax, b, e, "CLEAR" if bf else "  -  ", "CLEAR" if bw else "  -  "))
    print("     %-18s %.3f  (seeds 0-9 band means %.3f-%.3f, excess max %.3f)"
          % ("random floor mean", f["auroc_mean"], f["auroc_min"], f["auroc_max"], f["excess_max"]))
    print("     %-18s %.3f  excess %.3f"
          % ("bag-of-words", v["bag_of_words"]["auroc"], v["bag_of_words"]["excess_over_half"]))
