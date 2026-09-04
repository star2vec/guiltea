"""S4 Task 0.1 — `--steer arrow:layer:c`: the S2b addendum's hook and its sigma, inside the rig.

The hook is the one `scripts/s2b_addendum/steer_midlayer.py` steers with (`scripts/s2b/steer.py`'s `Steer`):
a forward hook on the **output** of ``model.model.layers[L]`` adding a fixed vector at **all positions**, cast
to the output dtype. It is registered once at model load and **before** any readout hook, so the rig's
per-layer capture (`rigcommon.make_readout`) sees the steered residual at L >= layer. Layers below the steered
one are untouched; layers above carry propagated effect, not injection — which is why the table prints the
injected component only where it is exact.

sigma is the addendum's: the sample sd (ddof = 1) of the 200 first-person passage `mean` projections on that
unit arrow at that layer, i.e. the expression in `steer_midlayer.hookcheck`

    float(torch.einsum("nd,d->n", Xf[:, layer, :], u).std(unbiased=True))

read off `results/raw/s2b/activations/first_person_mean.pt` (regenerate with
`python scripts/s2b/activations.py` if the pod lacks it).

Steering is a **window**, not a mode: `briefs/S4-experiment.md` puts it on from the feedback-reply turn through
the distance-0 forks and off for the four filler turns and the distance-4 forks (`--steer-off-after distance0`),
and never on the topic controls. `active()` is what every record's `steer_on` flag is stamped from, so where the
hook was on is read off the written data rather than asserted.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple

import torch

import rigcommon as R

FIRST_PERSON_MEAN = R.ROOT / "results" / "raw" / "s2b" / "activations" / "first_person_mean.pt"
OFF_AFTER_CHOICES = ("distance0",)

_X: Optional[torch.Tensor] = None


def _first_person_means() -> torch.Tensor:
    """[200, 32, 4096] float32 — the S2b first-person passage `mean` activations."""
    global _X
    if _X is None:
        if not FIRST_PERSON_MEAN.exists():
            raise SystemExit(
                "sigma needs %s (the 200 first-person passage `mean` activations).\n"
                "Regenerate with:  python scripts/s2b/activations.py" % FIRST_PERSON_MEAN)
        _X = torch.load(FIRST_PERSON_MEAN, map_location="cpu")["X"].float()
    return _X


def sigma(unit: torch.Tensor, layer: int) -> float:
    """Sample sd (ddof = 1) of the 200 first-person `mean` projections on this unit arrow at this layer."""
    X = _first_person_means()
    if layer >= X.shape[1]:
        raise SystemExit("no first-person activations at layer %d (file holds %d)" % (layer, X.shape[1]))
    return float(torch.einsum("nd,d->n", X[:, layer, :], unit.float()).std(unbiased=True))


def parse_spec(spec: str) -> Tuple[str, int, float]:
    """``arrow:layer:c`` -> (arrow, layer, c)."""
    parts = [p.strip() for p in str(spec).split(":")]
    if len(parts) != 3 or not all(parts):
        raise SystemExit("--steer wants arrow:layer:c, e.g. guilt_clean:16:4 (got %r)" % spec)
    try:
        return parts[0], int(parts[1]), float(parts[2])
    except ValueError:
        raise SystemExit("--steer wants arrow:layer:c with an integer layer and a numeric c (got %r)" % spec)


def cell_suffix(arrow: str, layer: int, c: float) -> str:
    """The cell directory suffix, so a steered cell is its own cell and resume still works."""
    return "+steer_%s_L%d_c%g" % (arrow, layer, c)


class Steer:
    """The addendum's hook, plus the on/off window the brief specifies."""

    def __init__(self, model, rig, arrow: str, layer: int, c: float, sigma_from: Optional[str] = None):
        if arrow not in rig.axes:
            raise SystemExit("no arrow %r in this arrow set (%s)" % (arrow, ", ".join(rig.axis_names)))
        if layer not in rig.arrow_layers:
            raise SystemExit("arrow %r is not held at layer %d" % (arrow, layer))
        self.arrow, self.layer, self.c = arrow, layer, c
        self.sigma_source = sigma_from or arrow
        if self.sigma_source not in rig.axes:
            raise SystemExit("no arrow %r to take sigma from" % self.sigma_source)
        self.unit = rig.axes[arrow][layer].float()
        self.sigma = sigma(rig.axes[self.sigma_source][layer].float(), layer)
        self.sigma_own = sigma(self.unit, layer) if self.sigma_source != arrow else self.sigma
        self.vec = (self.c * self.sigma) * self.unit.to(model.device)
        self.injected_norm = float(self.vec.norm())
        self.on = False
        self.n_forward_steered = 0
        self.h = model.model.layers[layer].register_forward_hook(self._hook)

    # the hook, byte-for-byte the shape of scripts/s2b/steer.py's Steer.hook
    def _hook(self, module, inp, out):
        if not self.on:
            return out
        self.n_forward_steered += 1
        if isinstance(out, tuple):
            return (out[0] + self.vec.to(out[0].dtype),) + tuple(out[1:])
        return out + self.vec.to(out.dtype)

    def components(self, rig) -> Dict[str, float]:
        """The injected component on every named axis at the steered layer: c * sigma * cos(u_inj, axis)."""
        j = rig.arrow_layers.index(self.layer)
        out = {}
        for a in rig.axis_names:
            u = rig.axes[a][self.layer].float()
            out[a] = float(self.c * self.sigma * torch.dot(self.unit, u))
        out["_arrow_layer_index"] = j
        return out

    def header(self, rig) -> dict:
        return {"arrow": self.arrow, "layer": self.layer, "c": self.c,
                "sigma": self.sigma, "sigma_source": self.sigma_source,
                "sigma_own_arrow": self.sigma_own,
                "sigma_recipe": "sample sd (ddof=1) of the 200 first-person passage `mean` projections "
                                "on the unit arrow at this layer (S2b addendum)",
                "injected_norm": self.injected_norm,
                "norm_matched_note": ("D is norm-matched to C: sigma comes from %s, not from %s's own sigma "
                                      "(%.6f), so both arms inject the same absolute norm"
                                      % (self.sigma_source, self.arrow, self.sigma_own))
                if self.sigma_source != self.arrow else "sigma is this arrow's own",
                "injected_component_per_axis": self.components(rig),
                "window": "on from the feedback-reply turn through the distance-0 forks; off for the act, the "
                          "four filler turns, the distance-4 forks and the topic controls"}

    def remove(self):
        self.h.remove()


# --------------------------------------------------------------------------- the module-level window
_STEER: Optional[Steer] = None


def install(model, rig, spec: Optional[str], sigma_from: Optional[str]) -> Optional[Steer]:
    global _STEER
    _STEER = None
    if not spec:
        return None
    arrow, layer, c = parse_spec(spec)
    _STEER = Steer(model, rig, arrow, layer, c, sigma_from)
    return _STEER


def current() -> Optional[Steer]:
    return _STEER


def active() -> bool:
    return _STEER is not None and _STEER.on


class window:
    """`with steer_rig.window(True):` — the only place the hook is ever on."""

    def __init__(self, on: bool):
        self.want = bool(on)
        self.prev = False

    def __enter__(self):
        if _STEER is not None:
            self.prev = _STEER.on
            _STEER.on = self.want
        return _STEER

    def __exit__(self, *exc):
        if _STEER is not None:
            _STEER.on = self.prev
        return False


# --------------------------------------------------------------------------- the pre-hook norm check
@torch.no_grad()
def norm_check(model, tok, rig, st: Steer, conversations, tol: float = 0.05) -> dict:
    """The addendum's next-layer pre-hook diagnostic, run on the rig's own hook.

    `output_hidden_states` records a layer's output *before* a forward hook's replacement (transformers 4.57.6),
    so an in-meta check reads zero; reading layer L+1's *input* is the valid check. Expected = |c * sigma * u|.
    """
    import s1bcommon as S
    cap = {}

    def pre(module, args):
        cap["x"] = args[0].detach().float().clone()
        return None

    hp = model.model.layers[st.layer + 1].register_forward_pre_hook(pre)
    checks = []
    try:
        for conv in conversations:
            ids = tok(S.render(tok, conv, True), return_tensors="pt",
                      add_special_tokens=False).input_ids.to(model.device)
            with torch.no_grad():
                model.model(ids, use_cache=False)
            x0 = cap["x"][0].clone()
            with window(True):
                with torch.no_grad():
                    model.model(ids, use_cache=False)
            d = cap["x"][0] - x0
            exp = st.vec.float()
            measured = float(d.norm(dim=-1).mean())
            checks.append({"arrow": st.arrow, "layer": st.layer, "c": st.c, "sigma": st.sigma,
                           "sigma_source": st.sigma_source,
                           "expected_norm": st.injected_norm, "measured_norm": measured,
                           "relative_deviation": (measured - st.injected_norm) / st.injected_norm
                           if st.injected_norm else None,
                           "max_abs_dev_from_expected": float((d - exp[None]).abs().max()),
                           "positions": int(d.shape[0])})
    finally:
        hp.remove()
    worst = max(abs(c["relative_deviation"]) for c in checks) if checks else None
    return {"checks": checks, "worst_relative_deviation": worst, "tolerance": tol,
            "pass": bool(worst is not None and worst <= tol),
            "note": "next-layer pre-hook check: output_hidden_states records a layer's output before a forward "
                    "hook's replacement, so an in-meta check reads zero; this is the valid one. bf16 rounding "
                    "of the residual bounds the deviation."}
