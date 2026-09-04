"""S1e shared machinery — depth-matched contrasts on the stored per-turn projections (no API, no GPU).

Every comparison is made INSIDE one turn index `t`. Nothing is pooled across `t`: the per-`t` values are
computed separately and combined only as an explicit count-weighted mean, which is a summary and not an
average of independent samples (the same chains recur at several `t`).

Scores come from `results/raw/s1d/proj_t4v1.npz`, built by `scripts/s1d/proj.py` from the S1b residual
store: [192 v1 chains, 10 turns, 3 positions, 12 named axes + randctl seeds 0-9, 32 layers]. Nothing is
recomputed from residuals here and no S1d file is written.

Two label sources, as S1d used them:
  `t_primary`  the rig's stored per-turn `grade` (asserted identical to `judge_calls/act_primary.jsonl`
               on all 1,920 v1 turns in `load_chains`)
  `merged`     the second judge's label where it exists, else the primary's (D-019). The second judge
               covers 394 of 1,920 v1 turns, concentrated near each chain's `T_primary`, so this source
               is a PARTIAL re-grading, not a uniform one.

The headline statistic is the L14-18 band mean (D-024 primary band). Best-over-layers is reported only
beside a selection-matched floor: each randctl seed is given the same max-over-32-layers search the
arrows get. That is the rule S1d adopted and it decides every verdict sentence.
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

OUT = REPO / "results" / "raw" / "s1e"
S3 = ["refusal", "badmed", "persona", "persona_meandiff"]
S2 = ["guilt_clean", "shame_clean", "nn", "received_act", "received_self"]
REPORT_AXES = S3 + S2
POSITIONS = ["answer", "into"]
SOURCES = ["t_primary", "merged"]
N_TURNS = 10
MIN_CLASS = 10          # brief: minimum class size, 10 per side at a given t
MIN_TARGETS_BOTH = 5    # brief: fewer than 5 targets holding both classes -> underpowered, no verdict
# the rig's persuader turns; `filler` is the neutral topic-change turn the rig inserted after the act,
# where a `held` grade means the harmful question was not asked (reports/S1d-blame-target.md section 7)
PERSUADER_KINDS = ("situation", "frozen", "frozen+bridge")


def load_chains():
    """(proj, axes, positions, layers, chains) with per-turn grades under both sources and per-turn kind."""
    proj, axes, positions, layers, keys = C.load_proj("t4v1")
    merged, prim, sec = C.act_label_table()
    chains, n_grade_mismatch, n_sec_v1 = [], 0, 0
    for i, k in enumerate(keys):
        target, seed, n_turns = k["target"], k["seed"], k["n_turns"]
        assert n_turns == N_TURNS, (target, seed, n_turns)
        rec = json.load(open(C.RAW / "t4" / target / ("v1_seed%d.json" % seed), encoding="utf-8"))
        kinds = [tu["kind"] for tu in rec["turns"]]
        stored = [tu["grade"] for tu in rec["turns"]]
        g_prim = [prim.get((target, seed, "v1", t)) for t in range(1, N_TURNS + 1)]
        g_merg = [merged.get((target, seed, "v1", t)) for t in range(1, N_TURNS + 1)]
        n_grade_mismatch += sum(1 for a, b in zip(stored, g_prim) if a != b)
        n_sec_v1 += sum(1 for t in range(1, N_TURNS + 1) if (target, seed, "v1", t) in sec)
        chains.append({"idx": i, "target": target, "seed": seed, "kinds": kinds,
                       "T_primary": rec["T_primary"],
                       "grades": {"t_primary": g_prim, "merged": g_merg}})
    assert n_grade_mismatch == 0, n_grade_mismatch
    meta = {"n_chains": len(chains), "n_turn_labels": len(chains) * N_TURNS,
            "stored_grade_vs_act_primary_mismatches": n_grade_mismatch,
            "v1_turns_with_a_second_judge_label": n_sec_v1,
            "proj_source": "results/raw/s1d/proj_t4v1.npz (scripts/s1d/proj.py)",
            "proj_shape": list(proj.shape)}
    return proj, axes, positions, layers, chains, meta


# ---------------------------------------------------------------- class membership at a fixed turn

def members_trigger(ch, source, t):
    """Task 1 at turn t: 1 = held at t and commits later; 0 = held at t and never commits; None = neither."""
    g = ch["grades"][source]
    if g[t - 1] != "held":
        return None
    if any(g[u] == "committed" for u in range(t, N_TURNS)):     # strictly after t, up to turn 10
        return 1
    if "committed" not in g:                                    # never commits within ten turns
        return 0
    return None                                                 # already committed at or before t


def members_actdet(ch, source, t):
    """Task 2 at turn t: 1 = committed at t; 0 = held at t; None = neither (`partial` excluded)."""
    g = ch["grades"][source]
    if g[t - 1] == "committed":
        return 1
    if g[t - 1] == "held":
        return 0
    return None


# ---------------------------------------------------------------- one fixed-depth contrast

def _weighted_curve(per_t, layers, evaluable, key, weights):
    """Count-weighted mean across t of a per-layer curve. NaN-aware; linear, so band means commute with it."""
    out = {}
    for L in layers:
        num = den = 0.0
        for t in evaluable:
            v = per_t[str(t)]["table"][key][str(L)]
            if v is None or np.isnan(v):
                continue
            num += weights[t] * v
            den += weights[t]
        out[str(L)] = float(num / den) if den else float("nan")
    return out


def _bands(curve, layers):
    out = {}
    for name, band in (("L14_18", C.BAND_PRIMARY), ("L6_11", C.BAND_SECONDARY)):
        vals = [curve[str(L)] for L in band if not np.isnan(curve[str(L)])]
        out[name] = float(np.mean(vals)) if vals else float("nan")
    return out


def _best(curve, layers):
    vals = [(curve[str(L)], L) for L in layers if not np.isnan(curve[str(L)])]
    if not vals:
        return {"auroc": float("nan"), "layer": None, "excess_over_half": float("nan")}
    a, L = max(vals, key=lambda v: abs(v[0] - 0.5))
    return {"auroc": float(a), "layer": int(L), "excess_over_half": float(abs(a - 0.5))}


def _verdict(axis_curves, seed_curves, layers):
    """Headline = the L14-18 band mean against the seeds' own band means. Best-over-layers beside a
    selection-matched floor: every seed gets the same max-over-32-layers search."""
    ab = {ax: _bands(axis_curves[ax], layers) for ax in axis_curves}
    sb = {s: _bands(seed_curves[s], layers) for s in seed_curves}
    v = {}
    for band in ("L14_18", "L6_11"):
        f = [abs(sb[s][band] - 0.5) for s in sb if not np.isnan(sb[s][band])]
        ax_ex = {ax: abs(ab[ax][band] - 0.5) for ax in ab}
        top = max(ax_ex, key=lambda a: (ax_ex[a] if not np.isnan(ax_ex[a]) else -1))
        v[band] = {
            "per_axis": {ax: {"band_mean": ab[ax][band], "excess_over_half": ax_ex[ax]} for ax in ab},
            "floor_band_mean": {"per_seed": {s: sb[s][band] for s in sb},
                                "excess_min": float(np.min(f)), "excess_mean": float(np.mean(f)),
                                "excess_max": float(np.max(f)),
                                "auroc_min": float(np.min([sb[s][band] for s in sb])),
                                "auroc_mean": float(np.mean([sb[s][band] for s in sb])),
                                "auroc_max": float(np.max([sb[s][band] for s in sb]))},
            "best_axis": top, "best_axis_band_mean": ab[top][band], "best_axis_excess": ax_ex[top],
            "beats_floor": bool(ax_ex[top] > float(np.max(f))),
            "n_axes_beating_floor": int(sum(1 for a in ax_ex.values()
                                            if not np.isnan(a) and a > float(np.max(f)))),
        }
    axb = {ax: _best(axis_curves[ax], layers) for ax in axis_curves}
    sbest = {s: _best(seed_curves[s], layers) for s in seed_curves}
    mf = [sbest[s]["excess_over_half"] for s in sbest if not np.isnan(sbest[s]["excess_over_half"])]
    top = max(axb, key=lambda a: (axb[a]["excess_over_half"]
                                  if not np.isnan(axb[a]["excess_over_half"]) else -1))
    v["best_over_layers"] = {
        "per_axis": axb,
        "selection_matched_floor": {"per_seed_excess": {s: sbest[s]["excess_over_half"] for s in sbest},
                                    "min": float(np.min(mf)), "mean": float(np.mean(mf)),
                                    "max": float(np.max(mf))},
        "best_axis": top, "best_axis_auroc": axb[top]["auroc"], "best_axis_layer": axb[top]["layer"],
        "best_axis_excess": axb[top]["excess_over_half"],
        "beats_matched_floor": bool(axb[top]["excess_over_half"] > float(np.max(mf))),
        "n_axes_beating_matched_floor": int(sum(1 for a in axb.values()
                                                if not np.isnan(a["excess_over_half"])
                                                and a["excess_over_half"] > float(np.max(mf)))),
    }
    return v


def _at_turn(proj, axes, positions, layers, chains, member_fn, source, t, position, persuader_only):
    """Every axis and seed at one turn index, one position. Returns (info, curves) or (info, None)."""
    rows = []
    for ch in chains:
        y = member_fn(ch, source, t)
        if y is None:
            continue
        if persuader_only and ch["kinds"][t - 1] not in PERSUADER_KINDS:
            continue
        rows.append((ch["idx"], y, ch["target"], ch["kinds"][t - 1]))
    y = np.array([r[1] for r in rows], dtype=int)
    grp = np.array([r[2] for r in rows])
    ci = np.array([r[0] for r in rows], dtype=int)
    n_pos, n_neg = int((y == 1).sum()), int((y == 0).sum())
    tg = {g: {"pos": int(((grp == g) & (y == 1)).sum()), "neg": int(((grp == g) & (y == 0)).sum())}
          for g in sorted(set(grp.tolist()))}
    both = [g for g, c in tg.items() if c["pos"] and c["neg"]]
    info = {"turn": t, "n_positive": n_pos, "n_negative": n_neg, "n_items": len(rows),
            "n_targets": len(tg), "n_targets_with_both_classes": len(both),
            "targets_with_both_classes": both, "per_target_counts": tg,
            "filler_in_positive_class": int(sum(1 for r in rows if r[1] == 1 and r[3] == "filler")),
            "filler_in_negative_class": int(sum(1 for r in rows if r[1] == 0 and r[3] == "filler")),
            "evaluable": bool(n_pos >= MIN_CLASS and n_neg >= MIN_CLASS),
            "verdict_written": bool(n_pos >= MIN_CLASS and n_neg >= MIN_CLASS
                                    and len(both) >= MIN_TARGETS_BOTH)}
    if not info["evaluable"]:
        return info, None
    pi = positions.index(position)
    S = proj[ci, t - 1, pi, :, :]                                # [n_items, n_axes, 32]
    curves = {"pooled": {}, "by_target_mean": {}}
    n_usable = {}
    for ax in REPORT_AXES + list(C.RANDOM_AXES):
        ai = axes.index(ax)
        cp, cg = {}, {}
        for L in layers:
            s = S[:, ai, L]
            cp[str(L)] = C.auroc(s, y)
            g, ng = C.grouped_auroc(s, y, grp)
            cg[str(L)] = g
            n_usable[ax] = ng
        curves["pooled"][ax] = cp
        curves["by_target_mean"][ax] = cg
    info["n_targets_usable_for_fold_statistic"] = int(n_usable[REPORT_AXES[0]])
    return info, curves


def run_contrast(proj, axes, positions, layers, chains, member_fn, source, position, persuader_only):
    """All ten turn indices, then the count-weighted mean across the evaluable ones."""
    per_t, weights = {}, {}
    for t in range(1, N_TURNS + 1):
        info, curves = _at_turn(proj, axes, positions, layers, chains, member_fn, source, t,
                                position, persuader_only)
        entry = dict(info)
        if curves is not None:
            for stat in ("pooled", "by_target_mean"):
                entry.setdefault("table", {})[stat] = {ax: curves[stat][ax] for ax in REPORT_AXES}
                entry.setdefault("raw_random", {})[stat] = {s: curves[stat][s] for s in C.RANDOM_AXES}
                entry.setdefault("floor_by_layer", {})[stat] = {
                    str(L): {"min": float(np.nanmin([curves[stat][s][str(L)] for s in C.RANDOM_AXES])),
                             "mean": float(np.nanmean([curves[stat][s][str(L)] for s in C.RANDOM_AXES])),
                             "max": float(np.nanmax([curves[stat][s][str(L)] for s in C.RANDOM_AXES]))}
                    for L in layers}
                entry.setdefault("bands", {})[stat] = {
                    ax: _bands(curves[stat][ax], layers) for ax in REPORT_AXES}
                entry.setdefault("verdict", {})[stat] = _verdict(
                    {ax: curves[stat][ax] for ax in REPORT_AXES},
                    {s: curves[stat][s] for s in C.RANDOM_AXES}, layers)
            weights[t] = info["n_items"]
        per_t[str(t)] = entry
    evaluable = sorted(weights)
    out = {"per_t": per_t, "evaluable_turns": evaluable,
           "turns_with_a_verdict": [t for t in evaluable if per_t[str(t)]["verdict_written"]]}
    if evaluable:
        w = {"weights_n_items": {str(t): weights[t] for t in evaluable},
             "note": "count-weighted mean across turn indices; a summary, not an average of independent "
                     "samples - the same chains recur at several t"}
        for stat in ("pooled", "by_target_mean"):
            sc = {s: _weighted_curve(
                {k: {"table": {s2: v["raw_random"][stat][s2] for s2 in C.RANDOM_AXES}}
                 for k, v in per_t.items() if "raw_random" in v}, layers, evaluable, s, weights)
                for s in C.RANDOM_AXES}
            ac = {ax: _weighted_curve(
                {k: {"table": {ax2: v["table"][stat][ax2] for ax2 in REPORT_AXES}}
                 for k, v in per_t.items() if "table" in v}, layers, evaluable, ax, weights)
                for ax in REPORT_AXES}
            w.setdefault("table", {})[stat] = ac
            w.setdefault("raw_random", {})[stat] = sc
            w.setdefault("bands", {})[stat] = {ax: _bands(ac[ax], layers) for ax in REPORT_AXES}
            w.setdefault("floor_by_layer", {})[stat] = {
                str(L): {"min": float(np.nanmin([sc[s][str(L)] for s in C.RANDOM_AXES])),
                         "mean": float(np.nanmean([sc[s][str(L)] for s in C.RANDOM_AXES])),
                         "max": float(np.nanmax([sc[s][str(L)] for s in C.RANDOM_AXES]))} for L in layers}
            w.setdefault("verdict", {})[stat] = _verdict(ac, sc, layers)
        out["weighted"] = w
    return out


def run_task(name, member_fn, contrast_label, extra_meta=None):
    """Both label sources x both positions x the brief's variant and the filler-excluded variant."""
    proj, axes, positions, layers, chains, meta = load_chains()
    meta.update({"task": name, "contrast": contrast_label, "status": "EXPLORATORY",
                 "min_class_size": MIN_CLASS, "min_targets_with_both_classes": MIN_TARGETS_BOTH,
                 "axes": REPORT_AXES, "random_axes": list(C.RANDOM_AXES),
                 "primary_band": "L%d-%d" % (C.BAND_PRIMARY[0], C.BAND_PRIMARY[-1]),
                 "secondary_band": "L%d-%d" % (C.BAND_SECONDARY[0], C.BAND_SECONDARY[-1]),
                 "persuader_kinds": list(PERSUADER_KINDS),
                 "no_gpu_no_api": True})
    if extra_meta:
        meta.update(extra_meta)
    out = {"meta": meta}
    for source in SOURCES:
        for position in POSITIONS:
            for variant, persuader_only in (("as_specified", False), ("filler_excluded", True)):
                key = "%s|%s|%s" % (source, position, variant)
                out[key] = run_contrast(proj, axes, positions, layers, chains, member_fn,
                                        source, position, persuader_only)
                print("  %-44s evaluable t=%s  verdict t=%s" % (
                    key, out[key]["evaluable_turns"], out[key]["turns_with_a_verdict"]), flush=True)
    return out


def save(out, fname):
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / fname
    json.dump(out, open(p, "w", encoding="utf-8"), indent=1, sort_keys=True, allow_nan=True)
    print("->", p.relative_to(REPO), "%.1f MB" % (p.stat().st_size / 1e6))
    return p
