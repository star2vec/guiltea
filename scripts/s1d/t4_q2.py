"""S1d Task 4 — Q2: the instrument on the subject's own words (no API, no GPU).

Classes come from Task 2's reflection labels, read at the `answer` position of the probe reply.
Primary contrast act-focused vs neutral; secondary act-focused vs self-focused (marked underpowered
and reported as a count only if self-focused has fewer than 10 members).

For every layer: AUROC of each named arrow, the randctl seed 0-9 floor on the same items and folds,
and a bag-of-words baseline (word counts + logistic regression) fitted out-of-fold. Folds are grouped
by target — leave one target out — so no target appears on both sides. The projections need no fitting,
so their AUROC is the pooled value; the bag-of-words number is pooled over out-of-fold scores.

The verdict line also reports a selection-matched floor: the arrows are scored as a max over 32 layers,
so each random seed gets the same max-over-layers search before the comparison is made.
"""
from __future__ import annotations

import importlib.util
import json
import numpy as np
from collections import Counter
from pathlib import Path
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression

REPO = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location("s1d_common", REPO / "scripts" / "s1d" / "common.py")
C = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(C)

S3 = ["refusal", "badmed", "persona", "persona_meandiff"]
S2 = ["guilt_clean", "shame_clean", "nn", "received_act", "received_self"]
REPORT_AXES = S2 + S3
POSITION = "answer"
CONTRASTS = {"primary": ("act-focused", "neutral"), "secondary": ("act-focused", "self-focused")}
# Both brief contrasts are confounded with route and fork: `neutral` is 26 of 32 deceived fork A and
# `self-focused` is 19 of 24 vicious fork B, so any axis that reads conversation structure or the persona
# system prompt separates them without reading a blame target (the S1b section-10 problem). These
# same-cell restrictions hold mode, fork, system prompt and conversation shape fixed. Reported beside
# the brief's tables, never instead of them.
WITHIN_CELL = {"primary_within_deceived_forkA": ("primary", "deceived", "A"),
               "secondary_within_vicious_forkB": ("secondary", "vicious", "B")}
MIN_SECONDARY = 10


def reflection_final():
    lab = {}
    for purpose in ("primary", "second"):
        p = C.OUT / "judge_calls" / ("reflection_%s.jsonl" % purpose)
        if not p.exists():
            continue
        for line in open(p, encoding="utf-8"):
            r = json.loads(line)
            k = (r["target"], r["mode"], r["fork"], r["seed"])
            d = lab.setdefault(k, {"primary": None, "second": None})
            d[purpose] = r["label"]
    return {k: (v["second"] if v["second"] is not None else v["primary"]) for k, v in lab.items()}


def bow_oof(texts, y, groups):
    """Leave-one-target-out bag-of-words baseline: pooled out-of-fold decision scores."""
    texts, y, groups = np.asarray(texts, dtype=object), np.asarray(y), np.asarray(groups)
    scores = np.full(len(y), np.nan)
    folds_used, folds_skipped = 0, 0
    for g in np.unique(groups):
        te = groups == g
        tr = ~te
        if len(set(y[tr])) < 2:
            folds_skipped += 1
            continue
        vec = CountVectorizer()
        Xtr = vec.fit_transform(texts[tr])
        clf = LogisticRegression(C=1.0, max_iter=2000)
        clf.fit(Xtr, y[tr])
        scores[te] = clf.decision_function(vec.transform(texts[te]))
        folds_used += 1
    m = ~np.isnan(scores)
    return scores, m, folds_used, folds_skipped


def run_contrast(name, pos_label, neg_label, rows, proj, axes, positions, layers, restrict=None):
    sel = [i for i, r in enumerate(rows) if r["label"] in (pos_label, neg_label)
           and (restrict is None or (r["mode"], r["fork"]) == restrict)]
    y = np.array([1 if rows[i]["label"] == pos_label else 0 for i in sel])
    grp = np.array([rows[i]["target"] for i in sel])
    texts = [rows[i]["answer"] for i in sel]
    n_pos, n_neg = int(y.sum()), int((y == 0).sum())
    info = {"contrast": "%s vs %s" % (pos_label, neg_label), "n_positive": n_pos, "n_negative": n_neg,
            "n_targets": len(set(grp)), "positive_class": pos_label,
            "restricted_to": ("%s fork %s" % restrict) if restrict else None,
            "class_composition": {lab: dict(Counter("%s/%s" % (rows[i]["mode"], rows[i]["fork"])
                                                    for i in sel if rows[i]["label"] == lab))
                                  for lab in (pos_label, neg_label)}}
    if name.startswith("secondary") and n_neg < MIN_SECONDARY:
        info["underpowered"] = True
        info["note"] = ("%s has %d members, fewer than the %d-member floor; no AUROC is reported for this contrast"
                        % (neg_label, n_neg, MIN_SECONDARY))
        return info
    info["underpowered"] = False

    pi = positions.index(POSITION)
    S = proj[np.array(sel)][:, pi, :, :]                   # [n, n_axes, 32]

    bow_scores, bmask, folds_used, folds_skipped = bow_oof(texts, y, grp)
    info["bag_of_words"] = {
        "pooled": C.auroc(bow_scores[bmask], y[bmask]),
        "by_target_mean": C.grouped_auroc(bow_scores[bmask], y[bmask], grp[bmask])[0],
        "n_scored": int(bmask.sum()), "folds_used": folds_used, "folds_skipped_single_class": folds_skipped,
        "recipe": "CountVectorizer() word counts + LogisticRegression(C=1.0, max_iter=2000), leave-one-target-out",
    }

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

    # bands and the selection-matched verdict
    for band_name, band in (("primary_band_L14_18", C.BAND_PRIMARY), ("secondary_band_L6_11", C.BAND_SECONDARY)):
        info.setdefault("bands", {})[band_name] = {
            ax: {stat: float(np.nanmean([table[ax][str(L)][stat] for L in band])) for stat in ("pooled", "by_target_mean")}
            for ax in REPORT_AXES}
        info["bands"][band_name]["random_floor_mean"] = {
            stat: float(np.nanmean([floor[str(L)][stat]["mean"] for L in band])) for stat in ("pooled", "by_target_mean")}
    for stat in ("pooled", "by_target_mean"):
        def best(d):
            vals = [(d[str(L)][stat], L) for L in layers if not np.isnan(d[str(L)][stat])]
            return max(vals, key=lambda v: abs(v[0] - 0.5)) if vals else (float("nan"), None)
        axis_best = {ax: best(table[ax]) for ax in REPORT_AXES}
        seed_best = {ax: best(raw_random[ax]) for ax in C.RANDOM_AXES}
        matched = [abs(v[0] - 0.5) for v in seed_best.values()]
        words = abs(info["bag_of_words"][stat] - 0.5)
        top = max(axis_best.items(), key=lambda kv: abs(kv[1][0] - 0.5))
        info.setdefault("verdict", {})[stat] = {
            "per_axis_best": {ax: {"auroc": v[0], "layer": v[1], "excess_over_half": abs(v[0] - 0.5)}
                              for ax, v in axis_best.items()},
            "matched_random_floor": {"min": float(np.min(matched)), "mean": float(np.mean(matched)),
                                     "max": float(np.max(matched))},
            "bag_of_words_excess": words,
            "best_axis": top[0], "best_axis_auroc": top[1][0], "best_axis_layer": top[1][1],
            "best_axis_excess": abs(top[1][0] - 0.5),
            "beats_matched_random_floor": abs(top[1][0] - 0.5) > float(np.max(matched)),
            "beats_words": abs(top[1][0] - 0.5) > words,
            "n_axes_beating_words": sum(1 for v in axis_best.values() if abs(v[0] - 0.5) > words),
            "n_axes_beating_matched_floor": sum(1 for v in axis_best.values() if abs(v[0] - 0.5) > float(np.max(matched))),
        }
    return info


def main():
    proj, axes, positions, layers, keys = C.load_proj("t7")
    join = {(r["target"], r["mode"], r["fork"], r["seed"]): r
            for r in (json.loads(l) for l in open(C.OUT / "join.jsonl", encoding="utf-8"))}
    final = reflection_final()
    rows = []
    for k in keys:
        key = (k["target"], k["mode"], k["fork"], k["seed"])
        rows.append({"target": k["target"], "mode": k["mode"], "fork": k["fork"], "seed": k["seed"],
                     "label": final.get(key), "answer": join[key]["answer"]})
    out = {"position": POSITION, "n_items": len(rows), "n_labelled": sum(1 for r in rows if r["label"]),
           "label_counts": dict(Counter(str(r["label"]) for r in rows)), "contrasts": {}}
    for name, (pos_label, neg_label) in CONTRASTS.items():
        out["contrasts"][name] = run_contrast(name, pos_label, neg_label, rows, proj, axes, positions, layers)
    for name, (base, mode, fork) in WITHIN_CELL.items():
        pos_label, neg_label = CONTRASTS[base]
        out["contrasts"][name] = run_contrast(name, pos_label, neg_label, rows, proj, axes, positions, layers,
                                              restrict=(mode, fork))
    json.dump(out, open(C.OUT / "t4_q2.json", "w", encoding="utf-8"), indent=1, sort_keys=True)
    print(json.dumps(out["label_counts"], indent=1))
    for name, info in out["contrasts"].items():
        print("== %s: %s  (n+ %d, n- %d%s)" % (name, info["contrast"], info["n_positive"], info["n_negative"],
                                              ", restricted to %s" % info["restricted_to"] if info["restricted_to"] else ""))
        if info.get("underpowered"):
            print("   UNDERPOWERED:", info["note"])
            continue
        for stat in ("pooled", "by_target_mean"):
            v = info["verdict"][stat]
            print("   %-14s best %s L%s AUROC %.3f (excess %.3f) | words excess %.3f | matched floor max %.3f"
                  " | beats words %s (%d/9) | beats floor %s (%d/9)"
                  % (stat, v["best_axis"], v["best_axis_layer"], v["best_axis_auroc"], v["best_axis_excess"],
                     v["bag_of_words_excess"], v["matched_random_floor"]["max"], v["beats_words"],
                     v["n_axes_beating_words"], v["beats_matched_random_floor"], v["n_axes_beating_matched_floor"]))


if __name__ == "__main__":
    main()
