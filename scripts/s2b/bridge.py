"""S2b Task 8 — bridge preparation at the band's centre layer (briefs/S2b-arrows.md Task 8). Labelled preparation.

For every second-person passage: projection on received_act / received_self units at `feedback_mean`, and on
guilt_clean / shame_clean units at `post`. Per class, mean with a 1,000-resample bootstrap CI over scenario ids
(numpy default_rng(0)). Also the seed-0 random unit beside every projection.
Usage: python scripts/s2b/bridge.py --layer L
"""
from __future__ import annotations

import argparse

import numpy as np
import torch

from s2b_common import RAW, SP_CLASSES, load_acts, load_s2_arrows, load_scenarios, random_units, save_json, scenario_index


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, required=True)
    a = ap.parse_args()
    L = a.layer
    units, norms, _ = load_s2_arrows()
    R = random_units(0)
    scen = load_scenarios()
    Xfb, rows = load_acts("second_person_feedback_mean")
    Xpo, rows2 = load_acts("second_person_post")
    assert [r["scenario_id"] for r in rows] == [r["scenario_id"] for r in rows2]
    sidx = scenario_index(rows, scen)
    fr = np.array([r["framing"] for r in rows])
    proj = {"feedback_mean": {"received_act": Xfb[:, L] @ units["received_act"][L], "received_self": Xfb[:, L] @ units["received_self"][L],
                              "random": Xfb[:, L] @ R[L]},
            "post": {"guilt_clean": Xpo[:, L] @ units["guilt_clean"][L], "shame_clean": Xpo[:, L] @ units["shame_clean"][L],
                     "random": Xpo[:, L] @ R[L]}}
    rng = np.random.default_rng(0)
    draws = [np.bincount(rng.choice(len(scen), len(scen), replace=True), minlength=len(scen)) for _ in range(1000)]
    out = {"layer": L, "n_boot": 1000, "classes": SP_CLASSES, "table": {}}
    for pos, d in proj.items():
        for arrow, p in d.items():
            p = p.numpy()
            row = {}
            for c in SP_CLASSES:
                m = fr == c
                w = np.array([cnt[sidx[m]] for cnt in draws], dtype=float)  # [1000, 50]
                boots = (w @ p[m]) / w.sum(1)
                row[c] = {"mean": float(p[m].mean()), "lo": float(np.percentile(boots, 2.5)), "hi": float(np.percentile(boots, 97.5)),
                          "sd": float(p[m].std(ddof=1))}
            # paired differences between classes (same resample)
            diffs = {}
            for c1, c2 in (("self_blame", "act_blame"), ("act_blame", "neutral_correction"), ("self_blame", "neutral_correction")):
                m1, m2 = fr == c1, fr == c2
                w1 = np.array([cnt[sidx[m1]] for cnt in draws], dtype=float); w2 = np.array([cnt[sidx[m2]] for cnt in draws], dtype=float)
                b = (w1 @ p[m1]) / w1.sum(1) - (w2 @ p[m2]) / w2.sum(1)
                diffs[f"{c1} - {c2}"] = {"mean": float(p[m1].mean() - p[m2].mean()), "lo": float(np.percentile(b, 2.5)), "hi": float(np.percentile(b, 97.5))}
            out["table"][f"{pos}:{arrow}"] = {"classes": row, "paired_differences": diffs}
    ps = out["table"]["post:shame_clean"]["classes"]; pg = out["table"]["post:guilt_clean"]["classes"]
    out["orderings"] = {
        "post.shame_clean: self_blame > act_blame > neutral_correction": bool(ps["self_blame"]["mean"] > ps["act_blame"]["mean"] > ps["neutral_correction"]["mean"]),
        "post.guilt_clean: act_blame > self_blame and act_blame > neutral_correction": bool(pg["act_blame"]["mean"] > pg["self_blame"]["mean"] and pg["act_blame"]["mean"] > pg["neutral_correction"]["mean"]),
        "post.shame_clean class means (nc, ab, sb)": [ps[c]["mean"] for c in SP_CLASSES],
        "post.guilt_clean class means (nc, ab, sb)": [pg[c]["mean"] for c in SP_CLASSES]}
    save_json(out, RAW / "task8_bridge.json")
    for k, v in out["table"].items():
        print(k, {c: f"{x['mean']:+.3f} [{x['lo']:+.3f}, {x['hi']:+.3f}]" for c, x in v["classes"].items()})
        print("   diffs", {c: f"{x['mean']:+.3f} [{x['lo']:+.3f}, {x['hi']:+.3f}]" for c, x in v["paired_differences"].items()})
    print(out["orderings"])


if __name__ == "__main__":
    main()
