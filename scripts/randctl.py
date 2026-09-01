"""Shared random-control utility (OPERATIONS standing note; S3 brief, Task 3).

Every project readout (S3 onward, including S4) imports
``random_unit_directions`` from here instead of re-implementing the checks-era
``torch.randn(...) / norm`` pattern inline.

Convention
----------
Named directions are stored as unit vectors (``units``) with their norms kept
separately (``norms``), e.g. ``directions/dirs_8B_base_sweep.pt``. The vectors
returned here are unit vectors too, so they are norm-matched to the stored
units. To compare a readout against a named direction at that direction's
original scale, apply the same scaling (the named direction's norm) to the
random vector; projections onto the unit vectors compare directly.

Determinism
-----------
The vector for layer ``L`` depends only on ``(seed, L, d_model)``: each layer
gets its own ``torch.Generator`` seeded with ``seed * 1_000_003 + L``. The
result for a layer therefore does not change when a different set of layers is
requested alongside it, and different layers draw from independent streams.
Always float32 on CPU; move/cast at the call site.
"""
from __future__ import annotations

import sys
from typing import Dict, Sequence

import torch

_LAYER_SEED_STRIDE = 1_000_003


def random_unit_directions(
    d_model: int, layers: Sequence[int], seed: int
) -> Dict[int, torch.Tensor]:
    """Return ``{layer: unit random vector of shape (d_model,)}`` (float32, CPU).

    Parameters
    ----------
    d_model : hidden size of the model (4096 for Llama-3.1-8B).
    layers  : layer indices to produce vectors for.
    seed    : base seed; the per-layer seed is ``seed * 1_000_003 + layer``.
    """
    if int(d_model) <= 0:
        raise ValueError(f"d_model must be positive, got {d_model!r}")
    out: Dict[int, torch.Tensor] = {}
    for layer in layers:
        layer = int(layer)
        gen = torch.Generator(device="cpu")
        gen.manual_seed(int(seed) * _LAYER_SEED_STRIDE + layer)
        v = torch.randn(int(d_model), generator=gen, dtype=torch.float32)
        out[layer] = v / v.norm()
    return out


def _self_check(d_model: int = 4096, layers: Sequence[int] = range(32), seed: int = 0) -> bool:
    """Determinism / independence check reported in reports/S3-feasibility.md."""
    layers = [int(L) for L in layers]
    f = random_unit_directions
    a = f(d_model, layers, seed)
    b = f(d_model, layers, seed)
    c = f(d_model, layers, seed + 1)

    # 1. same seed -> identical vectors
    ok1 = all(torch.equal(a[L], b[L]) for L in layers)
    # 2. different seed -> different vectors
    ok2 = all(not torch.equal(a[L], c[L]) for L in layers)
    max_cos_seed = max(abs(float(a[L] @ c[L])) for L in layers)
    # 3. different layers independent (|cos| across all layer pairs at one seed)
    M = torch.stack([a[L] for L in layers])
    G = M @ M.T
    mask = ~torch.eye(len(layers), dtype=torch.bool)
    off = G[mask]
    max_cos_layers = float(off.abs().max())
    mean_abs_cos_layers = float(off.abs().mean())
    ok3 = max_cos_layers < 0.1  # random 4096-d cosines ~ N(0, 1/sqrt(4096)=0.0156)
    # 4. subset invariance: a layer's vector does not depend on the requested list
    probe = layers[len(layers) // 2]
    ok4 = torch.equal(f(d_model, [probe], seed)[probe], a[probe]) and torch.equal(
        f(d_model, list(reversed(layers)), seed)[probe], a[probe]
    )
    # 5. unit norms
    norms = torch.stack([a[L].norm() for L in layers])
    max_norm_dev = float((norms - 1.0).abs().max())
    ok5 = max_norm_dev < 1e-6

    print(f"randctl self-check  d_model={d_model} layers={layers[0]}..{layers[-1]} (n={len(layers)}) seed={seed}")
    print(f"  torch {torch.__version__}")
    print(f"  1 same seed -> identical            : {'PASS' if ok1 else 'FAIL'}")
    print(f"  2 seed {seed} vs {seed+1} -> all differ        : {'PASS' if ok2 else 'FAIL'}  (max |cos| across layers = {max_cos_seed:.4f})")
    print(f"  3 layers independent (max|cos|<0.1) : {'PASS' if ok3 else 'FAIL'}  (max |cos| = {max_cos_layers:.4f}, mean |cos| = {mean_abs_cos_layers:.4f}, pairs = {len(layers)*(len(layers)-1)//2})")
    print(f"  4 subset/order invariance (layer {probe}) : {'PASS' if ok4 else 'FAIL'}")
    print(f"  5 unit norms (max |norm-1| < 1e-6)  : {'PASS' if ok5 else 'FAIL'}  (max |norm-1| = {max_norm_dev:.2e})")
    return ok1 and ok2 and ok3 and ok4 and ok5


if __name__ == "__main__":
    sys.exit(0 if _self_check() else 1)
