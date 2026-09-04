"""S1d Task 3 — Q1: the behavioural table (no API).

Label distribution per mode and fork with 95 % cluster-bootstrap CIs (2,000 resamples, seed 0,
resampled over targets). The `neutral` class is split in the reason field only — defends the answer /
restates / declines — as a mechanical keyword pass over the judge's reason line, adding no labels.
"""
from __future__ import annotations

import importlib.util
import json
import numpy as np
import re
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location("s1d_common", REPO / "scripts" / "s1d" / "common.py")
C = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(C)

LABELS = ["act-focused", "self-focused", "outcome-negative-only", "neutral", "incoherent"]
N_BOOT = 2000
SEED = 0

# mechanical split of the reason line for the `neutral` class only; report-only, no new labels
NEUTRAL_SPLIT = [
    ("declines", re.compile(r"\b(declin|refus|cannot evaluate|can't evaluate|unable to (?:evaluate|assess)|deflect|changes the subject|asks)\w*", re.I)),
    ("defends the answer", re.compile(r"\b(defend|affirm|maintain|stands by|confirm|accurac|accurate|assert\w*\s+(?:that\s+)?(?:the\s+)?(?:answer|information|advice)\s+(?:was|is)\s+(?:sound|correct|right)|praise|endors)\w*", re.I)),
    ("restates", re.compile(r"\b(restat|reiterat|repeat|summar|describ|recount|neutral(?:ly)? (?:describ|report))\w*", re.I)),
]


def split_neutral(reason):
    for name, rx in NEUTRAL_SPLIT:
        if rx.search(reason or ""):
            return name
    return "unclassified"


def load_labels():
    """{(target, mode, fork, seed): {...}} — the big model's label is final where it exists."""
    rows = {}
    for purpose in ("primary", "second"):
        p = C.OUT / "judge_calls" / ("reflection_%s.jsonl" % purpose)
        if not p.exists():
            continue
        for line in open(p, encoding="utf-8"):
            r = json.loads(line)
            k = (r["target"], r["mode"], r["fork"], r["seed"])
            d = rows.setdefault(k, {"target": r["target"], "mode": r["mode"], "fork": r["fork"], "seed": r["seed"],
                                    "primary": None, "primary_reason": None, "second": None, "second_reason": None})
            d[purpose] = r["label"]
            d[purpose + "_reason"] = r["reason"]
    for d in rows.values():
        d["final"] = d["second"] if d["second"] is not None else d["primary"]
        d["final_reason"] = d["second_reason"] if d["second"] is not None else d["primary_reason"]
        d["final_source"] = "second" if d["second"] is not None else "primary"
    return rows


def agreement(rows):
    both = [d for d in rows.values() if d["second"] is not None and d["primary"] is not None]
    same = sum(1 for d in both if d["second"] == d["primary"])
    conf = Counter((d["primary"], d["second"]) for d in both)
    return {"n_compared": len(both), "agreement": (same / len(both)) if both else float("nan"),
            "confusion_primary_to_second": {"%s -> %s" % k: v for k, v in sorted(conf.items())},
            "n_unparseable_primary": sum(1 for d in rows.values() if d["primary"] is None),
            "n_unparseable_second": sum(1 for d in rows.values() if d["second"] is None and d["second_reason"] is not None)}


def cell_table(items):
    """Counts, proportions and cluster-bootstrap CIs for one mode x fork cell."""
    n = len(items)
    cnt = Counter(d["final"] for d in items)
    targets = np.array([d["target"] for d in items])
    labs = np.array([str(d["final"]) for d in items])
    uniq = np.unique(targets)
    idx_by_t = {t: np.flatnonzero(targets == t) for t in uniq}
    rng = np.random.default_rng(SEED)
    draws = {L: [] for L in LABELS}
    for _ in range(N_BOOT):
        pick = rng.choice(len(uniq), size=len(uniq), replace=True)
        idx = np.concatenate([idx_by_t[uniq[p]] for p in pick])
        sub = labs[idx]
        for L in LABELS:
            draws[L].append(float((sub == L).mean()))
    out = {"n": n, "n_targets": len(uniq), "labels": {}}
    for L in LABELS:
        lo, hi = np.percentile(draws[L], [2.5, 97.5])
        out["labels"][L] = {"count": int(cnt.get(L, 0)), "rate": (cnt.get(L, 0) / n) if n else float("nan"),
                            "ci95": [float(lo), float(hi)]}
    neu = [d for d in items if d["final"] == "neutral"]
    out["neutral_reason_split"] = dict(Counter(split_neutral(d["final_reason"]) for d in neu))
    return out


def main():
    rows = load_labels()
    n_expected = sum(1 for _ in open(C.OUT / "join.jsonl", encoding="utf-8"))
    by_cell = defaultdict(list)
    for d in rows.values():
        by_cell[(d["mode"], d["fork"])].append(d)
        by_cell[("all", d["fork"])].append(d)
        by_cell[(d["mode"], "both")].append(d)
        by_cell[("all", "both")].append(d)
    out = {"n_judged": len(rows), "n_expected": n_expected,
           "final_label_source": dict(Counter(d["final_source"] for d in rows.values())),
           "agreement": agreement(rows),
           "cells": {"%s/%s" % k: cell_table(v) for k, v in sorted(by_cell.items())}}
    json.dump(out, open(C.OUT / "t3_q1.json", "w", encoding="utf-8"), indent=1, sort_keys=True)
    print("judged %d of %d; agreement on the fixed sample %.3f (n=%d)" % (
        out["n_judged"], n_expected, out["agreement"]["agreement"], out["agreement"]["n_compared"]))
    hdr = "cell            n   " + "  ".join("%-22s" % L for L in LABELS)
    print(hdr)
    for k, v in out["cells"].items():
        cells = []
        for L in LABELS:
            e = v["labels"][L]
            cells.append("%3d %.2f [%.2f,%.2f]" % (e["count"], e["rate"], e["ci95"][0], e["ci95"][1]))
        print("%-15s %3d  %s" % (k, v["n"], "  ".join(cells)))


if __name__ == "__main__":
    main()
