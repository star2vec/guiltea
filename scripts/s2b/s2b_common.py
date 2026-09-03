"""S2b — shared helpers (briefs/S2b-arrows.md). Imports the 8B conventions from scripts/s3_phaseB/common.py
(hidden_states[L+1], chat-template rendering, answer-mean incl. the closing <|eot_id|>, last-token readouts)
and the random control from scripts/randctl.py. Nothing under data/acts/ is opened here.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT / "scripts", ROOT / "scripts" / "s3_phaseB"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import common as c8  # noqa: E402  (scripts/s3_phaseB/common.py)
from randctl import random_unit_directions  # noqa: E402

RAW = ROOT / "results" / "raw" / "s2b"
CS = ROOT / "data" / "contrast-sets"
DIRS = ROOT / "directions"
LAYERS = list(range(32))
D = 4096
FP_CLASSES = ["baseline", "guilt", "shame", "neutral_negative"]
SP_CLASSES = ["neutral_correction", "act_blame", "self_blame"]
ARROW_NAMES = ["guilt", "shame", "nn", "guilt_clean", "shame_clean", "received_act", "received_self", "difference"]


def load_jsonl(p: Path) -> List[dict]:
    return [json.loads(l) for l in Path(p).read_text(encoding="utf-8").splitlines() if l.strip()]


def load_scenarios() -> List[dict]:
    return load_jsonl(CS / "scenarios.jsonl")


def load_first_person() -> List[dict]:
    return load_jsonl(CS / "first_person.jsonl")


def load_second_person() -> List[dict]:
    return load_jsonl(CS / "second_person.jsonl")


def load_placement() -> dict:
    import yaml
    return yaml.safe_load((CS / "placement.yaml").read_text(encoding="utf-8"))


def load_probe() -> List[dict]:
    return load_jsonl(CS / "steer_probe.jsonl")


def passage_set_commit() -> str:
    return subprocess.check_output(["git", "log", "-1", "--format=%h", "--", "data/contrast-sets"], cwd=ROOT).decode().strip()


def git_head() -> str:
    return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT).decode().strip()


def save_json(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1, ensure_ascii=False), encoding="utf-8")


def load_json(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


# ----------------------------------------------------------------------------- activations on disk
def load_acts(name: str, device="cpu"):
    """Return (X [n, 32, 4096] float32, rows) for one stored position file."""
    d = torch.load(RAW / "activations" / f"{name}.pt", map_location="cpu")
    return d["X"].to(device), d["rows"]


def scenario_index(rows: Sequence[dict], scenarios: Sequence[dict]) -> np.ndarray:
    pos = {s["id"]: i for i, s in enumerate(scenarios)}
    return np.array([pos[r["scenario_id"]] for r in rows])


def class_masks(rows: Sequence[dict], classes: Sequence[str]) -> Dict[str, np.ndarray]:
    return {c: np.array([r["framing"] == c for r in rows]) for c in classes}


# ----------------------------------------------------------------------------- arrows (batched over layers)
def weighted_means(X: torch.Tensor, W: torch.Tensor) -> torch.Tensor:
    """X [n, L, D]; W [k, n] non-negative weights -> [k, L, D] weighted means (rows with zero weight ignored)."""
    n, L, Dm = X.shape
    M = W @ X.reshape(n, L * Dm)
    return (M / W.sum(1, keepdim=True)).reshape(-1, L, Dm)


def unit(v: torch.Tensor, dim=-1) -> torch.Tensor:
    return v / v.norm(dim=dim, keepdim=True)


def clean(a: torch.Tensor, nn: torch.Tensor) -> torch.Tensor:
    """a, nn [L, D]: remove from a its component along nn (per layer)."""
    nh = unit(nn)
    return a - (a * nh).sum(-1, keepdim=True) * nh


def fp_arrows(means: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """means: class -> [L, D]. Returns raw (un-normalized) first-person arrows [L, D]."""
    g = means["guilt"] - means["baseline"]
    s = means["shame"] - means["baseline"]
    nn = means["neutral_negative"] - means["baseline"]
    gc, sc = clean(g, nn), clean(s, nn)
    return {"guilt": g, "shame": s, "nn": nn, "guilt_clean": gc, "shame_clean": sc, "difference": unit(sc) - unit(gc)}


def sp_arrows(means: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {"received_act": means["act_blame"] - means["neutral_correction"],
            "received_self": means["self_blame"] - means["neutral_correction"]}


def cos_layers(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """a, b [L, D] -> [L] cosines."""
    return (unit(a) * unit(b)).sum(-1)


def random_units(seed: int, device="cpu") -> torch.Tensor:
    r = random_unit_directions(D, LAYERS, seed)
    return torch.stack([r[L] for L in LAYERS]).to(device)


def load_sweep(device="cpu") -> Dict[str, torch.Tensor]:
    """refusal / badmed / persona unit vectors from directions/dirs_8B_base_sweep.pt as [L, D]."""
    d = torch.load(DIRS / "dirs_8B_base_sweep.pt", map_location="cpu")
    out = {}
    for k in ("refusal", "badmed", "persona"):
        u = d["units"][k]
        if isinstance(u, dict):
            u = torch.stack([u[L] for L in LAYERS])
        out[k] = u.float().to(device)
    return out


def load_s2_arrows(device="cpu"):
    d = torch.load(DIRS / "dirs_8B_s2_arrows.pt", map_location="cpu")
    units = {k: torch.stack([d["units"][k][L] for L in LAYERS]).float().to(device) for k in d["units"]}
    norms = {k: torch.tensor([d["norms"][k][L] for L in LAYERS]) for k in d["norms"]}
    return units, norms, d


# ----------------------------------------------------------------------------- AUROC (weighted, batched over layers)
def auroc_w(s_pos: torch.Tensor, s_neg: torch.Tensor, w_pos: torch.Tensor, w_neg: torch.Tensor) -> torch.Tensor:
    """s_pos [P, L], s_neg [N, L], weights [P], [N] -> [L] weighted AUROC (ties count 1/2)."""
    gt = (s_pos[:, None, :] > s_neg[None, :, :]).float() + 0.5 * (s_pos[:, None, :] == s_neg[None, :, :]).float()
    ww = w_pos[:, None] * w_neg[None, :]
    return (gt * ww[:, :, None]).sum((0, 1)) / ww.sum()


def pct_ci(samples: np.ndarray, axis=0):
    lo = np.percentile(samples, 2.5, axis=axis)
    hi = np.percentile(samples, 97.5, axis=axis)
    return lo, hi
