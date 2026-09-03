"""S2b addendum Task B — steering at mid-depth, layers 16 and 24 (briefs/S2b-addendum.md Task B).

Repeats S2b Task 7 exactly, at a layer given on the command line: the same eight items from
data/contrast-sets/steer_probe.jsonl, the same arms (unsteered, then +/- c.sigma along guilt_clean,
shame_clean, nn and the seed-0 random arrow), sigma recomputed per arrow at the layer from the 200
first-person `mean` projections (ddof = 1), greedy 128 new tokens, all positions steered, both judges
(gpt-4o-mini and gpt-4o-2024-08-06) at temperature 0 against data/contrast-sets/reflection_rubric.md,
the unsteered arm judged, and the exploratory persona/badmed re-read of every generated reflection.

Nothing in scripts/s2b/steer.py is modified: this driver imports it and redirects its module-level
output directory to results/raw/s2b_addendum/taskB_L<layer>/. The multipliers are c = 4 and 8 with the
coherence ladder (2, then 1) if the smaller one does not keep every reflection coherent.

`hookcheck` reproduces the S2b next-layer pre-hook diagnostic (`output_hidden_states` in transformers
4.57.6 records a layer's output before a forward hook's replacement, so the in-meta check reads zero;
the pre-hook check is the valid one): two items x two arrows x two multipliers.

Usage:
  python scripts/s2b_addendum/steer_midlayer.py --layer 16 --mults 4 8 --generate --hookcheck
  python scripts/s2b_addendum/steer_midlayer.py --layer 16 --judge --summarize --budget 3.0
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "s2b"))
sys.path.insert(0, str(ROOT / "scripts"))

import steer  # noqa: E402  (scripts/s2b/steer.py, unmodified)
from s2b_common import c8, load_acts, load_probe, load_s2_arrows, random_units, save_json, load_json  # noqa: E402

ADD = ROOT / "results" / "raw" / "s2b_addendum"


def out_dir(layer: int) -> Path:
    return ADD / f"taskB_L{layer}"


def spent_elsewhere(this: Path) -> float:
    """Judge spend already recorded anywhere under results/raw/s2b_addendum, excluding `this` run."""
    tot = 0.0
    for p in ADD.rglob("judgments.jsonl"):
        if p.parent == this:
            continue
        for l in p.read_text().splitlines():
            if l.strip():
                tot += json.loads(l).get("cost_usd") or 0.0
    return tot


def hookcheck(layer: int, mults):
    """Next-layer pre-hook diagnostic: the injected vector must appear in layer L+1's input."""
    units, _, _ = load_s2_arrows()
    R = random_units(0)
    Xf, _ = load_acts("first_person_mean")
    U = {k: units[k][layer] for k in ("guilt_clean", "shame_clean", "nn")}
    U["random"] = R[layer]
    sigma = {k: float(torch.einsum("nd,d->n", Xf[:, layer, :], u).std(unbiased=True)) for k, u in U.items()}
    model, tok, _ = c8.load_base()
    items = load_probe()
    cap = {}

    def pre(module, args):
        cap["x"] = args[0].detach().float().clone()
        return None

    hp = model.model.layers[layer + 1].register_forward_pre_hook(pre)
    st = steer.Steer(model, layer)
    checks = []
    for it in items[:2]:
        msgs = [{"role": "user", "content": it["prior_user"]}, {"role": "assistant", "content": it["prior_assistant"]},
                {"role": "user", "content": it["reflection_request"]}]
        ids = tok(c8.render(tok, msgs, True), return_tensors="pt", add_special_tokens=False).input_ids.to(model.device)
        st.vec = None
        with torch.no_grad():
            o0 = model(ids, output_hidden_states=True)
        x0, h0, l0 = cap["x"][0].clone(), o0.hidden_states[-1][0].float(), o0.logits[0].float()
        for k in ("guilt_clean", "shame_clean"):
            for c in mults:
                st.vec = (c * sigma[k]) * U[k].to(model.device)
                with torch.no_grad():
                    o1 = model(ids, output_hidden_states=True)
                d = cap["x"][0] - x0
                exp = st.vec.float()
                checks.append({"item": it["id"], "arrow": k, "c": c, "expected_norm": float(exp.norm()),
                               "next_layer_input_delta_mean_norm": float(d.norm(dim=-1).mean()),
                               "max_abs_dev_from_expected": float((d - exp[None]).abs().max()),
                               "positions": int(d.shape[0]),
                               "final_hidden_delta_mean_norm": float((o1.hidden_states[-1][0].float() - h0).norm(dim=-1).mean()),
                               "logits_max_abs_delta": float((o1.logits[0].float() - l0).abs().max())})
        st.vec = None
    st.remove(); hp.remove()
    save_json({"layer": layer, "mults": list(mults), "sigma": sigma,
               "note": ("output_hidden_states in transformers 4.57.6 records a layer's output before a forward hook's "
                        "replacement, so meta.json's hook_check reads zero; this next-layer pre-hook check is the valid one. "
                        "bf16 rounding of the residual bounds the deviation."),
               "date": dt.datetime.now(dt.timezone.utc).isoformat(), "checks": checks},
              out_dir(layer) / "hook_diagnostic.json")
    for c in checks[:4]:
        print(f"{c['arrow']} c={c['c']}: expected {c['expected_norm']:.4f} measured {c['next_layer_input_delta_mean_norm']:.4f} "
              f"maxdev {c['max_abs_dev_from_expected']:.3f} pos {c['positions']} final {c['final_hidden_delta_mean_norm']:.1f} "
              f"logits {c['logits_max_abs_delta']:.1f}", flush=True)
    del model
    torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--mults", type=float, nargs="+", default=[4, 8])
    ap.add_argument("--generate", action="store_true")
    ap.add_argument("--hookcheck", action="store_true")
    ap.add_argument("--judge", action="store_true")
    ap.add_argument("--summarize", action="store_true")
    ap.add_argument("--budget", type=float, default=3.0, help="total addendum judge budget in USD")
    a = ap.parse_args()
    mults = [int(m) if float(m).is_integer() else m for m in a.mults]
    steer.OUT = out_dir(a.layer)
    steer.OUT.mkdir(parents=True, exist_ok=True)

    if a.generate:
        steer.run_generation(a.layer, mults)
    if a.hookcheck:
        hookcheck(a.layer, mults)
    if a.judge:
        prior = spent_elsewhere(steer.OUT)
        cap = max(0.0, a.budget - prior)
        print(f"judge budget: ${a.budget:.2f} total, ${prior:.4f} spent elsewhere in the addendum -> cap ${cap:.4f}", flush=True)
        steer.run_judge(cap)
    if a.summarize:
        S = steer.summarize()
        for m in S["models"]:
            print(m)
            for arm in S["arms"]:
                print(f"  {arm:18s}", {k: v for k, v in S["distribution"][m][arm].items() if v})
        print("agreement", S["agreement_overall"], "cost", round(S["total_cost_usd"], 4))
        coh = {arm: v["incoherent_any_judge"] for arm, v in S["coherence"].items() if v["incoherent_any_judge"]}
        print("incoherent (any judge):", coh or "none")
        noeot = {arm: v["no_eot"] for arm, v in S["coherence"].items() if v["no_eot"]}
        print("no eot within 128 tokens:", noeot or "none")


if __name__ == "__main__":
    main()
