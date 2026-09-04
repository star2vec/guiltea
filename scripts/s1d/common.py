"""S1d shared analysis helpers (no API, no GPU)."""
from __future__ import annotations

import json
import numpy as np
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
RAW = REPO / "results" / "raw" / "s1b"
OUT = REPO / "results" / "raw" / "s1d"
RANDOM_AXES = ["random%d" % s for s in range(10)]
BAND_PRIMARY = list(range(14, 19))     # L14-18, D-024
BAND_SECONDARY = list(range(6, 12))    # L6-11, D-024


def load_proj(kind):
    z = np.load(OUT / ("proj_%s.npz" % kind), allow_pickle=False)
    axes = [str(a) for a in z["axes"]]
    positions = [str(p) for p in z["positions"]]
    keys = [json.loads(k) for k in z["keys"]]
    return z["proj"], axes, positions, [int(L) for L in z["layers"]], keys


def act_label_table():
    """{(target, seed, tag, turn): label} — second judge final where it exists (D-019)."""
    prim, sec = {}, {}
    for path, d in ((RAW / "judge_calls" / "act_primary.jsonl", prim),
                    (RAW / "judge_calls" / "act_second.jsonl", sec)):
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                r = json.loads(line)
                d[(r["target"], int(r["seed"]), r["tag"], int(r["turn"]))] = r["label"]
    keys = set(prim) | set(sec)
    return {k: (sec.get(k) if sec.get(k) is not None else prim.get(k)) for k in keys}, prim, sec


def auroc(scores, labels):
    """Rank AUROC with ties averaged. labels: 1 = positive class. Returns nan if a class is empty."""
    s = np.asarray(scores, dtype=np.float64)
    y = np.asarray(labels).astype(bool)
    n1, n0 = int(y.sum()), int((~y).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s), dtype=np.float64)
    sorted_s = s[order]
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and sorted_s[j + 1] == sorted_s[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return float((ranks[y].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def grouped_auroc(scores, labels, groups):
    """Mean of the per-group AUROCs over groups that hold both classes (clustering-aware),
    plus the count of usable groups. Complements the pooled value."""
    scores, labels, groups = np.asarray(scores), np.asarray(labels), np.asarray(groups)
    vals = []
    for g in np.unique(groups):
        m = groups == g
        a = auroc(scores[m], labels[m])
        if not np.isnan(a):
            vals.append(a)
    if not vals:
        return float("nan"), 0
    return float(np.mean(vals)), len(vals)


def boot_ci(fn, items, groups, n_boot=2000, seed=0, alpha=0.05):
    """Cluster bootstrap over `groups`: resample groups with replacement, recompute fn(indices)."""
    groups = np.asarray(groups)
    uniq = np.unique(groups)
    idx_by_group = {g: np.flatnonzero(groups == g) for g in uniq}
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        pick = rng.choice(len(uniq), size=len(uniq), replace=True)
        idx = np.concatenate([idx_by_group[uniq[p]] for p in pick])
        v = fn(idx)
        if v is not None and not (isinstance(v, float) and np.isnan(v)):
            vals.append(v)
    if not vals:
        return float("nan"), float("nan")
    lo, hi = np.percentile(vals, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)
