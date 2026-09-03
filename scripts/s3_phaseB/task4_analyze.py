"""S3 Phase B — Task 4 analysis (instrument check, NOT a result): organism − base readout deltas per axis per
layer beside the random-control floor, from the stored residuals of readout_pass.py (base and organism).

Random floor: randctl seed 0 is the control reported beside every axis (STAGE0 §3); seeds 0–9 give the floor
band (max |delta| over the ten seeds), so "at the floor" can be read off. Also reports the N=24 bf16 reference
direction's delta (Task 1.2, this machine's half). Output: results/raw/s3B/task4.json.
"""
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from randctl import random_unit_directions  # noqa: E402
from s3_phaseB.common import D_MODEL, LAYERS, RAW, ROOT, save_json  # noqa: E402

FLOOR_SEEDS = list(range(10))


def proj_table(rows, pos, unit_by_layer):
    """mean over prompts of residual[pos] @ unit, per layer -> {L: float}; also per-prompt values."""
    per = torch.stack([torch.stack([r[pos][i] @ unit_by_layer[L] for i, L in enumerate(LAYERS)]) for r in rows])  # (P, 32)
    return per


def main():
    d = torch.load(ROOT / "directions/dirs_8B_base_sweep.pt")
    units = dict(d["units"])
    ref = torch.load(ROOT / "directions/dir_8B_badmed_bf16ref_N24_seed0.pt")
    units["badmed_ref24"] = ref["units"]["badmed"]
    rand = {s: random_unit_directions(D_MODEL, LAYERS, s) for s in FLOOR_SEEDS}
    base = torch.load(RAW / "readout_base.pt")
    org = torch.load(RAW / "readout_organism.pt")
    assert base["prompts"] == org["prompts"]
    out = {"note": "instrument check, not a result", "prompts": base["prompts"], "layers": LAYERS, "axes": list(units),
           "random_seed_control": 0, "floor_seeds": FLOOR_SEEDS, "positions": {}, "texts": {}}
    for pos in ["into", "ans"]:
        P = {}
        for name, u in units.items():
            b, o = proj_table(base["rows"], pos, u), proj_table(org["rows"], pos, u)
            delta = (o - b)
            P[name] = {"base_mean": b.mean(0).tolist(), "org_mean": o.mean(0).tolist(), "delta_mean": delta.mean(0).tolist(),
                       "delta_std_over_prompts": delta.std(0).tolist(), "delta_per_prompt": delta.tolist()}
        R = {}
        for s in FLOOR_SEEDS:
            b, o = proj_table(base["rows"], pos, rand[s]), proj_table(org["rows"], pos, rand[s])
            R[s] = (o - b).mean(0)
        P["random_seed0"] = {"delta_mean": R[0].tolist(), "base_mean": proj_table(base["rows"], pos, rand[0]).mean(0).tolist(),
                             "org_mean": proj_table(org["rows"], pos, rand[0]).mean(0).tolist()}
        floor = torch.stack([R[s] for s in FLOOR_SEEDS]).abs().max(0).values
        P["random_floor_max_abs_seeds0-9"] = floor.tolist()
        # ratio |delta| / floor per layer, per named axis
        for name in units:
            dm = torch.tensor(P[name]["delta_mean"])
            P[name]["abs_delta_over_floor"] = (dm.abs() / floor).tolist()
        out["positions"][pos] = P
    out["texts"] = {"base": [r["reply"] for r in base["rows"]], "organism": [r["reply"] for r in org["rows"]]}
    out["stats"] = {"base": base["stats"], "organism": org["stats"]}
    save_json(out, RAW / "task4.json")
    for pos in ["into", "ans"]:
        print(f"\n=== position={pos}: organism - base, mean over {len(base['prompts'])} prompts (instrument check) ===")
        P = out["positions"][pos]
        print(f"{'axis':<16}" + "".join(f"L{L:>6}" for L in LAYERS))
        for name in list(units) + ["random_seed0"]:
            print(f"{name:<16}" + "".join(f"{x:>7.3f}" for x in P[name]["delta_mean"]))
        print(f"{'floor(max|s0-9|)':<16}" + "".join(f"{x:>7.3f}" for x in P["random_floor_max_abs_seeds0-9"]))
        for name in units:
            print(f"{name+'/floor':<16}" + "".join(f"{x:>7.1f}" for x in P[name]["abs_delta_over_floor"]))
    print("DONE")


if __name__ == "__main__":
    main()
