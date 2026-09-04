"""S1g Task 0a — build the `v2` projection store (no API, no GPU, no model load, no cost).

`scripts/s1d/proj.py` builds projection stores for the 508 probe replies (`t7`) and the 192 `v1`
chains (`t4v1`). It has no `v2` branch, and it must not be edited, so this script imports its
`unit_matrix` and `project_file` UNCHANGED and applies the same arithmetic to the 40 `v2` chains in
`results/raw/s1b/t4`. Named axes are recomputed from the stored residuals against the unit directions
in `directions/{dirs_8B_s2_arrows.pt, dirs_8B_base_sweep.pt}`; the ten randctl axes are taken from the
rig's own stored projections. Nothing is generated and no forward pass is run: every input is a
tensor the S1b rig already wrote to disk.

Output: `results/raw/s1g/proj_t4v2.npz`, shape [40 chains, 10 turns, 3 positions, 22 axes, 32 layers],
schema identical to `results/raw/s1d/proj_t4v1.npz` so `scripts/s1e/depth.py` can read it unedited.

Also repeats the precision check `scripts/s1d/proj.py` documents, on a `v2` record this time:
recomputing `refusal` at (`answer`, L16) from the residual must reproduce the rig's stored projection
to float16 precision.
"""
from __future__ import annotations

import importlib.util
import json
import numpy as np
import torch
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location("s1d_proj", REPO / "scripts" / "s1d" / "proj.py")
P = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(P)

RAW = REPO / "results" / "raw" / "s1b"
OUT = REPO / "results" / "raw" / "s1g"
N_TURNS = 10


def precision_check(pt_path, U):
    """Reproduce scripts/s1d/proj.py's documented check on a v2 record."""
    b = torch.load(pt_path, map_location="cpu", weights_only=False)
    pi, li, ai = P.POSITIONS.index("answer"), 16, P.NAMED.index("refusal")
    resid = b["resid"][0, pi, li].float()
    recomputed = float(resid @ U[ai, li])
    stored = float(b["proj"][0, pi, b["axes"].index("refusal"), li])
    return {"record": str(Path(pt_path).relative_to(REPO)), "axis": "refusal",
            "position": "answer", "layer": li, "turn": 1,
            "recomputed_from_residual": recomputed, "rig_stored": stored,
            "abs_difference": abs(recomputed - stored)}


def main():
    U = P.unit_matrix()
    axes = P.NAMED + P.RANDOM_AXES
    files = sorted((RAW / "t4").glob("*/v2_seed*.pt"))
    assert files, "no v2 residual files under results/raw/s1b/t4"
    keys, blocks = [], []
    for p in files:
        arr = P.project_file(p, U)
        d = json.load(open(p.with_suffix(".json"), encoding="utf-8"))
        assert d["tag"] == "v2", (p, d["tag"])
        assert arr.shape[0] == len(d["turns"]) == N_TURNS, (p, arr.shape, len(d["turns"]))
        keys.append({"target": d["target"], "seed": d["seed"], "n_turns": len(d["turns"])})
        blocks.append(arr)
    proj = np.stack(blocks)                                      # [40, 10, 3, 22, 32]
    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / "proj_t4v2.npz"
    np.savez_compressed(out, proj=proj, axes=np.array(axes), positions=np.array(P.POSITIONS),
                        layers=np.array(P.LAYERS), keys=np.array([json.dumps(k) for k in keys]))
    chk = precision_check(files[0], U)
    meta = {"built_by": "scripts/s1g/proj_v2.py",
            "reuses": "scripts/s1d/proj.py unit_matrix + project_file, unedited",
            "shape": list(proj.shape), "axes": axes, "positions": list(P.POSITIONS),
            "n_chains": len(keys), "n_targets": len(set(k["target"] for k in keys)),
            "targets": sorted(set(k["target"] for k in keys)),
            "precision_check": chk, "no_gpu_no_api_no_model_load": True}
    json.dump(meta, open(OUT / "proj_t4v2_meta.json", "w", encoding="utf-8"), indent=1, sort_keys=True)
    print("proj_t4v2 ->", out.name, proj.shape, "%.1f MB" % (out.stat().st_size / 1e6))
    print("chains %d over %d targets: %s" % (meta["n_chains"], meta["n_targets"],
                                             ", ".join(meta["targets"])))
    print("precision check (refusal, answer, L16, turn 1): recomputed %.5f vs rig-stored %.5f, "
          "abs diff %.2e" % (chk["recomputed_from_residual"], chk["rig_stored"], chk["abs_difference"]))
    assert chk["abs_difference"] < 1e-2, chk


if __name__ == "__main__":
    main()
