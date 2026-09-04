"""S1g shared machinery — the `v2` held-out set, read through the S1e code unedited.

`scripts/s1e/depth.py` holds the depth-matched contrast machinery S1e used on the 192 `v1` chains.
It is hard-wired to `v1`: it loads `results/raw/s1d/proj_t4v1.npz`, reads `v1_seed*.json`, keys the
judge labels on tag `v1`, and reports nine axes, two label sources, two readout positions and a
5-target fold floor. The brief forbids editing it, so this module imports it UNCHANGED and rebinds
its module attributes for the held-out run. Every statistic is still computed by S1e's own code.

What the brief fixes, and what is therefore rebound here:

  label source  `t_primary` only          ->  depth.SOURCES      = ["t_primary"]
  position      `answer`                  ->  depth.POSITIONS    = ["answer"]
  axes          `nn`, `persona_meandiff`  ->  depth.REPORT_AXES  = ["nn", "persona_meandiff"]
  count floor   10 per side, >= 3 targets ->  depth.MIN_TARGETS_BOTH = 3   (the one stated deviation,
                                              relaxed from S1e's 5 because v2 spans only 5 targets)
  chains        the 40 v2 chains          ->  depth.load_chains  = load_chains_v2

Nothing else is varied. No axis, band, position, statistic or label source beyond the ones fixed in
briefs/S1g-heldout-trigger.md is computed anywhere in scripts/s1g/.

CPU only: no generation, no model load, no judge call, no GPU, no cost.
"""
from __future__ import annotations

import importlib.util
import json
import numpy as np
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, REPO / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


D = _load("s1e_depth", "scripts/s1e/depth.py")          # S1e machinery, unedited
C = D.C                                                  # scripts/s1d/common.py, unedited

OUT = REPO / "results" / "raw" / "s1g"
PROJ = OUT / "proj_t4v2.npz"
TAG = "v2"
N_TURNS = D.N_TURNS

# The count floor, exactly as briefs/S1g-heldout-trigger.md fixes it.
FLOOR_MIN_CLASS = D.MIN_CLASS        # 10 per side at a turn index
FLOOR_MIN_TARGETS = 3                # at least 3 targets holding both classes (the stated deviation)

# The two pre-named axes and their predicted directions, fixed before any v2 number was computed.
AXES = ["nn", "persona_meandiff"]
PREDICTED = {"nn": "above", "persona_meandiff": "below"}
POSITION = "answer"
SOURCE = "t_primary"


def load_proj_v2():
    """Mirrors scripts/s1d/common.py load_proj, pointed at the S1g store instead of the S1d one."""
    z = np.load(PROJ, allow_pickle=False)
    return (z["proj"], [str(a) for a in z["axes"]], [str(p) for p in z["positions"]],
            [int(L) for L in z["layers"]], [json.loads(k) for k in z["keys"]])


def load_chains_v2():
    """The v2 twin of scripts/s1e/depth.py load_chains. Same return shape, same assertions."""
    proj, axes, positions, layers, keys = load_proj_v2()
    merged, prim, sec = C.act_label_table()
    chains, n_grade_mismatch, n_sec = [], 0, 0
    for i, k in enumerate(keys):
        target, seed, n_turns = k["target"], k["seed"], k["n_turns"]
        assert n_turns == N_TURNS, (target, seed, n_turns)
        rec = json.load(open(C.RAW / "t4" / target / ("%s_seed%d.json" % (TAG, seed)), encoding="utf-8"))
        assert rec["tag"] == TAG, (target, seed, rec["tag"])
        stored = [tu["grade"] for tu in rec["turns"]]
        g_prim = [prim.get((target, seed, TAG, t)) for t in range(1, N_TURNS + 1)]
        g_merg = [merged.get((target, seed, TAG, t)) for t in range(1, N_TURNS + 1)]
        n_grade_mismatch += sum(1 for a, b in zip(stored, g_prim) if a != b)
        n_sec += sum(1 for t in range(1, N_TURNS + 1) if (target, seed, TAG, t) in sec)
        chains.append({"idx": i, "target": target, "seed": seed,
                       "kinds": [tu["kind"] for tu in rec["turns"]],
                       "T_primary": rec["T_primary"],
                       "grades": {"t_primary": g_prim, "merged": g_merg}})
    assert n_grade_mismatch == 0, n_grade_mismatch
    meta = {"n_chains": len(chains), "n_turn_labels": len(chains) * N_TURNS,
            "n_targets": len(set(c["target"] for c in chains)),
            "stored_grade_vs_act_primary_mismatches": n_grade_mismatch,
            "v2_turns_with_a_second_judge_label": n_sec,
            "proj_source": "results/raw/s1g/proj_t4v2.npz (scripts/s1g/proj_v2.py)",
            "proj_shape": list(proj.shape)}
    return proj, axes, positions, layers, chains, meta


def bind_v2():
    """Rebind the S1e module for the held-out run. Returns the module; the file on disk is untouched."""
    D.load_chains = load_chains_v2
    D.MIN_TARGETS_BOTH = FLOOR_MIN_TARGETS
    D.SOURCES = [SOURCE]
    D.POSITIONS = [POSITION]
    D.REPORT_AXES = list(AXES)
    D.OUT = OUT
    return D


# ------------------------------------------------------------------ the class table (labels only)

def class_table(chains, member_fn, source=SOURCE, persuader_only=False):
    """Per turn index: class sizes, target coverage, filler counts, and whether the count floor clears.
    Labels and turn kinds only - no projection is read and no statistic is computed."""
    rows = {}
    for t in range(1, N_TURNS + 1):
        items = []
        for ch in chains:
            y = member_fn(ch, source, t)
            if y is None:
                continue
            if persuader_only and ch["kinds"][t - 1] not in D.PERSUADER_KINDS:
                continue
            items.append((ch["idx"], y, ch["target"], ch["kinds"][t - 1]))
        per_target = {}
        for _, y, g, _k in items:
            per_target.setdefault(g, {"pos": 0, "neg": 0})["pos" if y == 1 else "neg"] += 1
        both = sorted(g for g, c in per_target.items() if c["pos"] and c["neg"])
        n_pos = sum(1 for _, y, _g, _k in items if y == 1)
        n_neg = len(items) - n_pos
        rows[t] = {"turn": t, "n_positive": n_pos, "n_negative": n_neg, "n_items": len(items),
                   "n_targets": len(per_target), "n_targets_with_both_classes": len(both),
                   "targets_with_both_classes": both, "per_target_counts": per_target,
                   "filler_in_positive_class": sum(1 for _, y, _g, k in items
                                                   if y == 1 and k not in D.PERSUADER_KINDS),
                   "filler_in_negative_class": sum(1 for _, y, _g, k in items
                                                   if y == 0 and k not in D.PERSUADER_KINDS),
                   "clears_count_floor": bool(n_pos >= FLOOR_MIN_CLASS and n_neg >= FLOOR_MIN_CLASS
                                              and len(both) >= FLOOR_MIN_TARGETS),
                   "items": items}
    return rows


def floor_clearing_turns(rows):
    """The turn indices the brief's count floor admits - both parts of it."""
    return [t for t in sorted(rows) if rows[t]["clears_count_floor"]]


# ------------------------------------------------------------------ the fixed statistic

def fold_curve_at_turn(proj, axes, positions, layers, items, axis, t, position=POSITION):
    ci = np.array([r[0] for r in items], dtype=int)
    y = np.array([r[1] for r in items], dtype=int)
    grp = np.array([r[2] for r in items])
    S = proj[ci, t - 1, positions.index(position), axes.index(axis), :]  # [n_items, 32]
    return {str(L): C.grouped_auroc(S[:, L], y, grp)[0] for L in layers}


def band_mean(curve):
    """The L14-18 band mean (D-024 primary band), via scripts/s1e/depth.py's own helper."""
    vals = [curve[str(L)] for L in C.BAND_PRIMARY if not np.isnan(curve[str(L)])]
    return float(np.mean(vals)) if vals else float("nan")


def weighted_curve(per_t_curves, turns, weights, layers):
    """Count-weighted mean across turn indices, via scripts/s1e/depth.py _weighted_curve unedited."""
    shim = {str(t): {"table": {"x": per_t_curves[t]}} for t in turns}
    return D._weighted_curve(shim, layers, turns, "x", weights)
