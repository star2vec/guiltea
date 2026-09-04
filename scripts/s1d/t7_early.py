"""S1d Task 7 — Q4: is the capitulation visible one turn before it happens? (no API, no GPU)

Each persuader turn t of a v1 deceived chain is labelled by whether turn t+1 is that chain's first
`committed` turn. Turns at or after the first committed turn are filler and are excluded. The score is
the stored projection at turn t. Reported at every layer, at the `answer` and `into` positions, against
the randctl seed 0-9 floor and a turn-index baseline (t alone), folded by target.

The first committed turn is taken from the Task 1 merged act-label table (the second judge's label is
final where it exists, D-019); the count of chains where that differs from the rig's stored T_primary
is reported beside it.
"""
from __future__ import annotations

import importlib.util
import json
import numpy as np
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location("s1d_common", REPO / "scripts" / "s1d" / "common.py")
C = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(C)

S3 = ["refusal", "badmed", "persona", "persona_meandiff"]
S2 = ["guilt_clean", "shame_clean", "nn", "received_act", "received_self"]
REPORT_AXES = S3 + S2
POSITIONS = ["answer", "into"]


def first_committed(labels, target, seed, n_turns):
    for t in range(1, n_turns + 1):
        if labels.get((target, seed, "v1", t)) == "committed":
            return t
    return None


def build_items(source="merged"):
    """source='merged' -> the Task 1 table (second judge final, D-019); 't_primary' -> the rig's stored T_primary (mini)."""
    proj, axes, positions, layers, keys = C.load_proj("t4v1")
    labels, prim, _ = C.act_label_table()
    if source == "t_primary":
        labels = prim
    rows = []
    mism = 0
    n_no_commit = 0
    t_dist = {}
    for i, k in enumerate(keys):
        target, seed, n_turns = k["target"], k["seed"], k["n_turns"]
        rec = json.load(open(C.RAW / "t4" / target / ("v1_seed%d.json" % seed), encoding="utf-8"))
        T = rec["T_primary"] if source == "t_primary" else first_committed(labels, target, seed, n_turns)
        if T != rec["T_primary"]:
            mism += 1
        if T is None:
            n_no_commit += 1
        t_dist[str(T)] = t_dist.get(str(T), 0) + 1
        last = (T - 1) if T is not None else (n_turns - 1)   # t+1 must exist and not be filler
        for t in range(1, last + 1):
            rows.append({"chain": i, "target": target, "seed": seed, "turn": t,
                         "y": 1 if (T is not None and t == T - 1) else 0})
    return proj, axes, positions, layers, rows, {"label_source": source, "chains": len(keys), "T_mismatch_vs_T_primary": mism,
                                                 "chains_no_committed_turn": n_no_commit,
                                                 "first_committed_turn_distribution": dict(sorted(t_dist.items(), key=lambda x: (x[0] == "None", x[0])))}


def analyse(source):
    proj, axes, positions, layers, rows, meta = build_items(source)
    y = np.array([r["y"] for r in rows])
    grp = np.array([r["target"] for r in rows])
    ci = np.array([r["chain"] for r in rows])
    ti = np.array([r["turn"] for r in rows])
    meta.update({"n_items": len(rows), "n_positive": int(y.sum()), "n_negative": int((~y.astype(bool)).sum()),
                 "n_targets": len(set(grp))})

    # turn-index baseline
    base_pooled = C.auroc(ti, y)
    base_grouped, base_ng = C.grouped_auroc(ti, y, grp)
    meta["turn_index_baseline"] = {"pooled": base_pooled, "by_target_mean": base_grouped, "n_targets_usable": base_ng}

    out = {"meta": meta, "table": {}}
    for pos in POSITIONS:
        pi = positions.index(pos)
        S = proj[ci, ti - 1, pi, :, :]                      # [n_items, n_axes, 32]
        for ax in REPORT_AXES:
            ai = axes.index(ax)
            for L in layers:
                s = S[:, ai, L]
                out["table"].setdefault(pos, {}).setdefault(ax, {})[str(L)] = {
                    "pooled": C.auroc(s, y), "by_target_mean": C.grouped_auroc(s, y, grp)[0]}
        # random floor: the 10 randctl seeds, same items, kept per seed and pooled over seeds
        for L in layers:
            vals_p, vals_g = [], []
            for ax in C.RANDOM_AXES:
                s = S[:, axes.index(ax), L]
                a_p, a_g = C.auroc(s, y), C.grouped_auroc(s, y, grp)[0]
                out.setdefault("raw_random", {}).setdefault(pos, {}).setdefault(ax, {})[str(L)] = {
                    "pooled": a_p, "by_target_mean": a_g}
                vals_p.append(a_p)
                vals_g.append(a_g)
            out["table"].setdefault(pos, {}).setdefault("random_floor", {})[str(L)] = {
                "pooled_mean": float(np.mean(vals_p)), "pooled_min": float(np.min(vals_p)), "pooled_max": float(np.max(vals_p)),
                "by_target_mean": float(np.mean(vals_g)), "by_target_min": float(np.min(vals_g)), "by_target_max": float(np.max(vals_g))}

    # Selection-matched verdict. The arrows get a max over 32 layers, so the floor must get the same
    # search: each randctl seed contributes its own max-over-layers |AUROC - 0.5|, and the 10 seeds'
    # range is the floor the arrows have to clear. Comparing a best-of-many arrow against one random
    # seed at one layer is not a floor, and is not reported as one.
    for pos in POSITIONS:
        for stat in ("pooled", "by_target_mean"):
            def best_over_layers(ax):
                vals = [(out["table"][pos][ax][str(L)][stat], L) for L in layers
                        if not np.isnan(out["table"][pos][ax][str(L)][stat])]
                return max(vals, key=lambda v: abs(v[0] - 0.5)) if vals else (float("nan"), None)

            axis_best = {ax: best_over_layers(ax) for ax in REPORT_AXES}
            seed_best = {}
            for si, ax in enumerate(C.RANDOM_AXES):
                vals = [(out["raw_random"][pos][ax][str(L)][stat], L) for L in layers
                        if not np.isnan(out["raw_random"][pos][ax][str(L)][stat])]
                seed_best[ax] = max(vals, key=lambda v: abs(v[0] - 0.5)) if vals else (float("nan"), None)
            floor = [abs(v[0] - 0.5) for v in seed_best.values()]
            base = base_pooled if stat == "pooled" else base_grouped
            top = max(axis_best.items(), key=lambda kv: abs(kv[1][0] - 0.5))
            out.setdefault("headline", {}).setdefault(pos, {})[stat] = {
                "per_axis_best": {ax: {"auroc": v[0], "layer": v[1], "excess_over_half": abs(v[0] - 0.5)}
                                  for ax, v in axis_best.items()},
                "random_floor_matched": {"per_seed_excess": {ax: abs(v[0] - 0.5) for ax, v in seed_best.items()},
                                         "min": float(np.min(floor)), "mean": float(np.mean(floor)),
                                         "max": float(np.max(floor))},
                "turn_index_baseline": {"auroc": base, "excess_over_half": abs(base - 0.5)},
                "best_axis": top[0], "best_axis_auroc": top[1][0], "best_axis_layer": top[1][1],
                "best_axis_excess": abs(top[1][0] - 0.5),
                "beats_matched_random_floor": abs(top[1][0] - 0.5) > float(np.max(floor)),
                "beats_turn_index": abs(top[1][0] - 0.5) > abs(base - 0.5),
                "n_axes_beating_matched_floor": sum(1 for v in axis_best.values()
                                                    if abs(v[0] - 0.5) > float(np.max(floor))),
            }
    return out


def main():
    all_out = {}
    for source in ("merged", "t_primary"):
        out = analyse(source)
        all_out[source] = out
        print("== label source: %s" % source)
        print(json.dumps(out["meta"], indent=1))
        print(json.dumps(out["headline"], indent=1))
    json.dump(all_out, open(C.OUT / "t7_early.json", "w", encoding="utf-8"), indent=1, sort_keys=True)


if __name__ == "__main__":
    main()
