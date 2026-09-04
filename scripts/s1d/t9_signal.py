"""S1d Task 9 (added 2026-09-04) — Q6: does the harmfulness signal survive the refusal collapse?

Prediction stated before the analysis runs (Zhao et al., arXiv:2507.11878, via the 2026-09-04 sweep):
harmfulness and refusal are encoded separately, so across the persuader turns the refusal projection
should fall while the `badmed` projection holds or rises. Either answer is reportable.

Raw projections at the `answer` position, per layer, as a function of turn index relative to the first
committed turn (offset 0 = T). Filler turns after T are excluded. Means over chains with 95 % cluster
bootstrap CIs (2,000 resamples, seed 0, resampled over targets) and the randctl seed 0-9 floor beside them.

On reuse (the brief's item 4): `results/raw/s1b/t10/curves.json` was checked first. It holds aligned-at-T
curves for refusal, badmed, persona and random0 at all 32 layers, but they are benign-matched
*differences* (treatment minus the benign chain at the same turn, reports/S1b-runs.md section 9) and they
include the post-T filler turns, so they do not answer this task as specified (raw projection, filler
excluded). The curves are read and reported beside our own series for the pre-T offsets, not instead of it.
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

AXES = ["refusal", "badmed"]
POSITION = "answer"
MAX_LAG = 9        # offsets -9 .. 0
N_BOOT = 2000
SEED = 0


def series(source="t_primary"):
    proj, axes, positions, layers, keys = C.load_proj("t4v1")
    merged, prim, _ = C.act_label_table()
    labels = prim if source == "t_primary" else merged
    pi = positions.index(POSITION)
    items = []                      # (target, offset, chain_index, turn_index)
    for i, k in enumerate(keys):
        target, seed, n_turns = k["target"], k["seed"], k["n_turns"]
        rec = json.load(open(C.RAW / "t4" / target / ("v1_seed%d.json" % seed), encoding="utf-8"))
        T = rec["T_primary"] if source == "t_primary" else None
        if T is None and source != "t_primary":
            for t in range(1, n_turns + 1):
                if labels.get((target, seed, "v1", t)) == "committed":
                    T = t
                    break
        if T is None:
            continue
        for t in range(1, T + 1):                      # persuader turns up to and including the act
            off = t - T
            if off < -MAX_LAG:
                continue
            assert rec["turns"][t - 1]["kind"] != "filler", (target, seed, t)
            items.append((target, off, i, t))
    return proj, axes, layers, pi, items


def main():
    proj, axes, layers, pi, items = series("t_primary")
    offsets = list(range(-MAX_LAG, 1))
    out = {"position": POSITION, "offsets": offsets, "n_items": len(items),
           "n_chains": len({it[2] for it in items}), "axes": AXES, "curves": {}}
    all_axes = AXES + C.RANDOM_AXES
    for off in offsets:
        sel = [it for it in items if it[1] == off]
        if not sel:
            continue
        ci = np.array([s[2] for s in sel])
        ti = np.array([s[3] for s in sel])
        grp = np.array([s[0] for s in sel])
        S = proj[ci, ti - 1, pi, :, :]
        uniq = np.unique(grp)
        idx_by_g = {g: np.flatnonzero(grp == g) for g in uniq}
        rng = np.random.default_rng(SEED)
        picks = [np.concatenate([idx_by_g[uniq[p]] for p in rng.choice(len(uniq), len(uniq), replace=True)])
                 for _ in range(N_BOOT)]
        for ax in all_axes:
            ai = axes.index(ax)
            for L in layers:
                v = S[:, ai, L]
                boots = np.array([v[p].mean() for p in picks])
                lo, hi = np.percentile(boots, [2.5, 97.5])
                out["curves"].setdefault(ax, {}).setdefault(str(L), {})[str(off)] = {
                    "mean": float(v.mean()), "ci95": [float(lo), float(hi)], "n": int(len(v)),
                    "n_targets": int(len(uniq))}
    # random floor: mean over the ten seeds at each (layer, offset)
    for L in layers:
        for off in offsets:
            vals = [out["curves"][a][str(L)][str(off)]["mean"] for a in C.RANDOM_AXES
                    if str(off) in out["curves"].get(a, {}).get(str(L), {})]
            if vals:
                out["curves"].setdefault("random_floor", {}).setdefault(str(L), {})[str(off)] = {
                    "mean": float(np.mean(vals)), "min": float(np.min(vals)), "max": float(np.max(vals))}

    # the direction question, per layer: change from the earliest usable offset to T
    deltas = {}
    for ax in AXES + ["random_floor"]:
        for L in layers:
            c = out["curves"][ax][str(L)]
            offs = sorted((int(o) for o in c), key=lambda x: x)
            first, last = str(offs[0]), str(offs[-1])
            m0 = c[first]["mean"] if ax != "random_floor" else c[first]["mean"]
            m1 = c[last]["mean"] if ax != "random_floor" else c[last]["mean"]
            deltas.setdefault(ax, {})[str(L)] = {"from_offset": int(first), "to_offset": int(last),
                                                 "start": m0, "at_T": m1, "delta": m1 - m0}
    out["delta_first_to_T"] = deltas
    band = C.BAND_PRIMARY
    out["band_L14_18"] = {ax: {"mean_delta": float(np.mean([deltas[ax][str(L)]["delta"] for L in band])),
                               "mean_start": float(np.mean([deltas[ax][str(L)]["start"] for L in band])),
                               "mean_at_T": float(np.mean([deltas[ax][str(L)]["at_T"] for L in band]))}
                          for ax in AXES + ["random_floor"]}
    out["direction"] = {
        "refusal_falls": out["band_L14_18"]["refusal"]["mean_delta"] < 0,
        "badmed_holds_or_rises": out["band_L14_18"]["badmed"]["mean_delta"] >= 0,
        "prediction_met_in_band_L14_18": (out["band_L14_18"]["refusal"]["mean_delta"] < 0
                                          and out["band_L14_18"]["badmed"]["mean_delta"] >= 0),
        "n_layers_refusal_falls": int(sum(1 for L in layers if deltas["refusal"][str(L)]["delta"] < 0)),
        "n_layers_badmed_holds_or_rises": int(sum(1 for L in layers if deltas["badmed"][str(L)]["delta"] >= 0)),
        "n_layers_both": int(sum(1 for L in layers if deltas["refusal"][str(L)]["delta"] < 0
                                 and deltas["badmed"][str(L)]["delta"] >= 0)),
        "n_layers_random_floor_falls": int(sum(1 for L in layers if deltas["random_floor"][str(L)]["delta"] < 0)),
    }
    # Paired version. The unpaired series above changes population with offset (n = 15 at -9, 109 at 0),
    # so its start-to-T delta mixes movement inside chains with a different set of chains at each end.
    # This holds the population fixed: chains with at least 4 persuader turns up to the act, measured at
    # their own offsets -3..0, delta per chain, clustered bootstrap over targets.
    PAIR_OFFS = [-3, -2, -1, 0]
    by_chain = {}
    for target, off, chain, turn in items:
        by_chain.setdefault(chain, {})[off] = (target, turn)
    paired_chains = [c for c, d in by_chain.items() if all(o in d for o in PAIR_OFFS)]
    paired = {"n_chains": len(paired_chains), "offsets": PAIR_OFFS}
    if paired_chains:
        grp = np.array([by_chain[c][0][0] for c in paired_chains])
        for ax in AXES + C.RANDOM_AXES:
            ai = axes.index(ax)
            for L in layers:
                vals = np.array([[proj[c, by_chain[c][o][1] - 1, pi, ai, L] for o in PAIR_OFFS]
                                 for c in paired_chains])
                d = vals[:, -1] - vals[:, 0]
                lo, hi = C.boot_ci(lambda idx: float(d[idx].mean()), list(range(len(d))), grp,
                                   n_boot=N_BOOT, seed=SEED)
                paired.setdefault("curves", {}).setdefault(ax, {})[str(L)] = {
                    "means_by_offset": [float(x) for x in vals.mean(axis=0)],
                    "delta_mean": float(d.mean()), "delta_ci95": [lo, hi]}
        band = C.BAND_PRIMARY
        paired["band_L14_18"] = {}
        for ax in AXES:
            ds = [paired["curves"][ax][str(L)]["delta_mean"] for L in band]
            paired["band_L14_18"][ax] = {"mean_delta": float(np.mean(ds))}
        rnd = [np.mean([paired["curves"][a][str(L)]["delta_mean"] for a in C.RANDOM_AXES]) for L in band]
        paired["band_L14_18"]["random_floor"] = {"mean_delta": float(np.mean(rnd))}
        paired["n_layers"] = {
            "refusal_falls_ci_excludes_zero": int(sum(
                1 for L in layers if paired["curves"]["refusal"][str(L)]["delta_ci95"][1] < 0)),
            "badmed_rises_ci_excludes_zero": int(sum(
                1 for L in layers if paired["curves"]["badmed"][str(L)]["delta_ci95"][0] > 0)),
            "refusal_rises_ci_excludes_zero": int(sum(
                1 for L in layers if paired["curves"]["refusal"][str(L)]["delta_ci95"][0] > 0)),
            "badmed_falls_ci_excludes_zero": int(sum(
                1 for L in layers if paired["curves"]["badmed"][str(L)]["delta_ci95"][1] < 0)),
            "random_floor_ci_excludes_zero": int(sum(
                1 for L in layers for a in C.RANDOM_AXES
                if paired["curves"][a][str(L)]["delta_ci95"][0] > 0
                or paired["curves"][a][str(L)]["delta_ci95"][1] < 0)),
        }
    out["paired"] = paired

    # the S1b curves, read for comparison at the same offsets (benign-matched differences, filler included)
    s1b = json.load(open(C.RAW / "t10" / "curves.json", encoding="utf-8"))
    cmp = {}
    for ax in AXES + ["random0"]:
        k = "%s|L16" % ax
        if k in s1b:
            a = s1b[k]["aligned_T"]
            cmp[ax] = {"index_of_T": a["index_of_T"], "mean": a["mean"], "n": a["n"]}
    out["s1b_curves_json_aligned_T_L16"] = cmp
    json.dump(out, open(C.OUT / "t9_signal.json", "w", encoding="utf-8"), indent=1, sort_keys=True)

    print("chains %d, turn-items %d" % (out["n_chains"], out["n_items"]))
    print("band L14-18: refusal %.3f -> %.3f (delta %+.3f) | badmed %.3f -> %.3f (delta %+.3f) | random %.3f -> %.3f (delta %+.3f)"
          % (out["band_L14_18"]["refusal"]["mean_start"], out["band_L14_18"]["refusal"]["mean_at_T"],
             out["band_L14_18"]["refusal"]["mean_delta"], out["band_L14_18"]["badmed"]["mean_start"],
             out["band_L14_18"]["badmed"]["mean_at_T"], out["band_L14_18"]["badmed"]["mean_delta"],
             out["band_L14_18"]["random_floor"]["mean_start"], out["band_L14_18"]["random_floor"]["mean_at_T"],
             out["band_L14_18"]["random_floor"]["mean_delta"]))
    print(json.dumps(out["direction"], indent=1))
    if out["paired"].get("curves"):
        pb = out["paired"]["band_L14_18"]
        print("paired (n=%d chains, offsets -3..0) band L14-18: refusal delta %+.3f | badmed delta %+.3f | random delta %+.3f"
              % (out["paired"]["n_chains"], pb["refusal"]["mean_delta"], pb["badmed"]["mean_delta"],
                 pb["random_floor"]["mean_delta"]))
        print(json.dumps(out["paired"]["n_layers"], indent=1))
        for L in (8, 16, 24):
            r = out["paired"]["curves"]["refusal"][str(L)]
            b = out["paired"]["curves"]["badmed"][str(L)]
            print("  L%-2d refusal delta %+.3f [%+.3f,%+.3f]   badmed delta %+.3f [%+.3f,%+.3f]"
                  % (L, r["delta_mean"], r["delta_ci95"][0], r["delta_ci95"][1],
                     b["delta_mean"], b["delta_ci95"][0], b["delta_ci95"][1]))
    for off in offsets:
        r = out["curves"]["refusal"]["16"].get(str(off))
        b = out["curves"]["badmed"]["16"].get(str(off))
        if r:
            print("  offset %+d  n=%-4d refusal %+.3f [%+.3f,%+.3f]  badmed %+.3f [%+.3f,%+.3f]"
                  % (off, r["n"], r["mean"], r["ci95"][0], r["ci95"][1], b["mean"], b["ci95"][0], b["ci95"][1]))


if __name__ == "__main__":
    main()
