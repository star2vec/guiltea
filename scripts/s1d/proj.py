"""S1d shared projection extraction (no API, no GPU).

The S1b rig stored, per assistant turn, the residual at three readout positions
(`into`, `think`, `answer`) for all 32 layers, plus projections on 13 axes
(refusal, badmed, persona, randctl seeds 0-9). The S2 arrows are not among them, so they are
computed here from the stored residuals and directions/dirs_8B_s2_arrows.pt, and the fourth S3
axis (persona by mean difference) from directions/dirs_8B_base_sweep.pt, with the same
arithmetic the rig used: unit direction dotted into the stored residual, per layer.

Verified on one record: recomputing refusal at (answer, L16) from the residual reproduces the
rig's stored projection to float16 precision (-0.93427 vs -0.93386 stored).
"""
from __future__ import annotations

import json
import numpy as np
import torch
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
RAW = REPO / "results" / "raw" / "s1b"
OUT = REPO / "results" / "raw" / "s1d"
LAYERS = list(range(32))
POSITIONS = ["into", "think", "answer"]
RANDOM_AXES = ["random%d" % s for s in range(10)]
S2_AXES = ["guilt", "shame", "nn", "guilt_clean", "shame_clean", "received_act", "received_self", "difference"]
S3_AXES = ["refusal", "badmed", "persona", "persona_meandiff"]
NAMED = S2_AXES + S3_AXES


def unit_matrix():
    """[n_named, 32, 4096] float32 of unit directions, in NAMED order."""
    s2 = torch.load(REPO / "directions" / "dirs_8B_s2_arrows.pt", map_location="cpu", weights_only=False)
    sw = torch.load(REPO / "directions" / "dirs_8B_base_sweep.pt", map_location="cpu", weights_only=False)
    src = {}
    for a in S2_AXES:
        src[a] = s2["units"][a]
    for a in ("refusal", "badmed", "persona"):
        src[a] = sw["units"][a]
    src["persona_meandiff"] = sw["persona_meandiff_units"]
    U = torch.stack([torch.stack([src[a][L].float() for L in LAYERS]) for a in NAMED])
    norms = U.norm(dim=-1)
    assert float((norms - 1.0).abs().max()) < 1e-3, float((norms - 1.0).abs().max())
    return U


def project_file(pt_path, U):
    """[n_turns, 3, n_named + 10, 32] float32: named projections from the residual, random from the store."""
    b = torch.load(pt_path, map_location="cpu", weights_only=False)
    resid = b["resid"].float()                                   # [n, 3, 32, 4096]
    named = torch.einsum("ntld,ald->ntal", resid, U)              # [n, 3, n_named, 32]
    ridx = [b["axes"].index(a) for a in RANDOM_AXES]
    rnd = b["proj"][:, :, ridx, :].float()                        # [n, 3, 10, 32]
    return torch.cat([named, rnd], dim=2).numpy().astype(np.float32)


def build(kind):
    """kind='t7' -> the 508 probe replies (one turn each); kind='t4v1' -> the 192 v1 chains (10 turns each)."""
    U = unit_matrix()
    axes = NAMED + RANDOM_AXES
    if kind == "t7":
        rows = [json.loads(l) for l in open(OUT / "join.jsonl", encoding="utf-8")]
        rows.sort(key=lambda r: (r["target"], r["mode"], r["fork"], r["seed"]))
        keys, blocks = [], []
        for r in rows:
            p = REPO / r["pt"]
            arr = project_file(p, U)
            assert arr.shape[0] == 1, (p, arr.shape)
            keys.append({"target": r["target"], "mode": r["mode"], "fork": r["fork"], "seed": r["seed"]})
            blocks.append(arr[0])
        proj = np.stack(blocks)                                    # [508, 3, n_axes, 32]
    elif kind == "t4v1":
        files = sorted((RAW / "t4").glob("*/v1_seed*.pt"))
        keys, blocks = [], []
        for p in files:
            arr = project_file(p, U)
            d = json.load(open(p.with_suffix(".json"), encoding="utf-8"))
            assert arr.shape[0] == len(d["turns"]), (p, arr.shape, len(d["turns"]))
            keys.append({"target": d["target"], "seed": d["seed"], "n_turns": len(d["turns"])})
            blocks.append(arr)
        proj = np.stack(blocks)                                    # [192, 10, 3, n_axes, 32]
    else:
        raise ValueError(kind)
    out = OUT / ("proj_%s.npz" % kind)
    np.savez_compressed(out, proj=proj, axes=np.array(axes), positions=np.array(POSITIONS),
                        layers=np.array(LAYERS), keys=np.array([json.dumps(k) for k in keys]))
    print(kind, "->", out.name, proj.shape, "%.1f MB" % (out.stat().st_size / 1e6))
    return out


if __name__ == "__main__":
    import sys
    for kind in (sys.argv[1:] or ["t7", "t4v1"]):
        build(kind)
