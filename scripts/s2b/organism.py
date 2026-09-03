"""S2b Task 9 — organism sanity ("instrument check, not a result"), briefs/S2b-arrows.md Task 9.

Reads results/raw/s2b/organism/first_person_mean.pt (written by `activations.py --organism`: base + pinned LoRA,
merge_and_unload, same renders and `mean` position) and the base activations. At the band's centre layer:
Δ_class = mean_organism(class) − mean_base(class), projected on guilt_clean / shame_clean units, beside the
random floor (seeds 0–9, same Δ), and ‖Δ‖. No arrow is extracted on the organism.
Usage: python scripts/s2b/organism.py --layer L
"""
from __future__ import annotations

import argparse

import numpy as np
import torch

from s2b_common import FP_CLASSES, LAYERS, RAW, load_acts, load_s2_arrows, random_units, save_json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, required=True)
    a = ap.parse_args()
    L = a.layer
    units, norms, _ = load_s2_arrows()
    Xb, rb = load_acts("first_person_mean")
    d = torch.load(RAW / "organism" / "first_person_mean.pt", map_location="cpu", weights_only=False)
    Xo, ro = d["X"], d["rows"]
    assert [r["scenario_id"] + r["framing"] for r in rb] == [r["scenario_id"] + r["framing"] for r in ro]
    fr = np.array([r["framing"] for r in rb])
    Rs = {s: random_units(s) for s in range(10)}
    out = {"layer": L, "load": d.get("load"), "table": {}, "all_layers": {}}
    for c in FP_CLASSES + ["all"]:
        m = np.ones(len(fr), bool) if c == "all" else fr == c
        delta = (Xo[m, L] - Xb[m, L]).mean(0)
        g, s = float(delta @ units["guilt_clean"][L]), float(delta @ units["shame_clean"][L])
        rand = [float(delta @ Rs[k][L]) for k in range(10)]
        out["table"][c] = {"n": int(m.sum()), "delta_norm": float(delta.norm()), "proj_guilt_clean": g, "proj_shame_clean": s,
                           "random_seeds_0_9": rand, "random_abs_mean": float(np.mean(np.abs(rand))), "random_abs_max": float(np.max(np.abs(rand))),
                           "random_sd": float(np.std(rand, ddof=1)),
                           "ratio_guilt_over_random_abs_mean": g / float(np.mean(np.abs(rand))), "ratio_shame_over_random_abs_mean": s / float(np.mean(np.abs(rand))),
                           "base_mean_proj_guilt_clean": float((Xb[m, L] @ units["guilt_clean"][L]).mean()),
                           "organism_mean_proj_guilt_clean": float((Xo[m, L] @ units["guilt_clean"][L]).mean()),
                           "base_mean_proj_shame_clean": float((Xb[m, L] @ units["shame_clean"][L]).mean()),
                           "organism_mean_proj_shame_clean": float((Xo[m, L] @ units["shame_clean"][L]).mean()),
                           "base_sd_proj_guilt_clean": float((Xb[m, L] @ units["guilt_clean"][L]).std(unbiased=True)),
                           "base_sd_proj_shame_clean": float((Xb[m, L] @ units["shame_clean"][L]).std(unbiased=True))}
        # informational: the same at every layer (delta on the unit arrows vs random seed 0)
        dl = (Xo[m] - Xb[m]).mean(0)  # [32, D]
        out["all_layers"][c] = {"proj_guilt_clean": [float(dl[l] @ units["guilt_clean"][l]) for l in LAYERS],
                                "proj_shame_clean": [float(dl[l] @ units["shame_clean"][l]) for l in LAYERS],
                                "random_abs_mean": [float(np.mean([abs(float(dl[l] @ Rs[k][l])) for k in range(10)])) for l in LAYERS],
                                "delta_norm": [float(dl[l].norm()) for l in LAYERS]}
    save_json(out, RAW / "task9_organism.json")
    for c, v in out["table"].items():
        print(f"{c:17s} |Δ| {v['delta_norm']:.4f}  Δ·ĝ {v['proj_guilt_clean']:+.4f}  Δ·ŝ {v['proj_shame_clean']:+.4f}  random |.| mean {v['random_abs_mean']:.4f} max {v['random_abs_max']:.4f}  ratios g {v['ratio_guilt_over_random_abs_mean']:+.1f} s {v['ratio_shame_over_random_abs_mean']:+.1f}")


if __name__ == "__main__":
    main()
