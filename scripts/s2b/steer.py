"""S2b Task 7 — steering validation at the band's centre layer (briefs/S2b-arrows.md Task 7).

Items: the eight in data/contrast-sets/steer_probe.jsonl, conversation [user: prior_user, assistant: prior_assistant,
user: reflection_request] + generation prompt; greedy, 128 new tokens. Arms: unsteered, then ±c·σ_arrow·û for
û ∈ {guilt_clean, shame_clean, nn, random seed 0} added to the output of model.model.layers[L_c] at all positions
(prompt and every generated token). σ_arrow = sample std (ddof=1) of the 200 first-person `mean` residuals'
projections onto û at L_c. Multipliers c from --mults (plan: 4 and 8; coherence ladder 2, 1 if needed).
Judge: reflection rubric verbatim as the system message, user = "Prior act: <prior_assistant>\n\nReflection: <text>",
gpt-4o-mini and gpt-4o-2024-08-06, temperature 0, max_tokens 200, one re-call on an unparseable output.
Exploratory: clean (unhooked) re-read of each generated reflection, mean residual over its tokens at L_c,
projected on the persona and badmed units. Everything saved under results/raw/s2b/task7/.

Usage: python scripts/s2b/steer.py --layer L --mults 4 8 [--judge] [--cap 5.0]
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import time

import numpy as np
import torch

from s2b_common import (RAW, c8, load_acts, load_probe, load_s2_arrows, load_sweep, random_units, save_json,
                        load_json)

ARROWS = ["guilt_clean", "shame_clean", "nn", "random"]
OUT = RAW / "task7"


class Steer:
    def __init__(self, model, layer):
        self.vec = None
        self.h = model.model.layers[layer].register_forward_hook(self.hook)

    def hook(self, module, inp, out):
        if self.vec is None:
            return out
        if isinstance(out, tuple):
            return (out[0] + self.vec.to(out[0].dtype),) + tuple(out[1:])
        return out + self.vec.to(out.dtype)

    def remove(self):
        self.h.remove()


def generate(model, tok, msgs, n=128):
    ids = tok(c8.render(tok, msgs, True), return_tensors="pt", add_special_tokens=False).input_ids.to(model.device)
    with torch.no_grad():
        out = model.generate(ids, max_new_tokens=n, do_sample=False, pad_token_id=tok.eos_token_id)
    gen = out[0, ids.shape[1]:]
    return tok.decode(gen, skip_special_tokens=True), int(gen.shape[0]), bool((gen == tok.convert_tokens_to_ids("<|eot_id|>")).any())


def reflection_resid(model, tok, msgs, reflection, layer):
    """Clean forward on conversation + reflection as the assistant turn; mean over the reflection's tokens (incl. eot)."""
    full = msgs + [{"role": "assistant", "content": reflection}]
    full_text, pre_text = c8.render(tok, full, False), c8.render(tok, msgs, True)
    ids = tok(full_text, return_tensors="pt", add_special_tokens=False).input_ids
    plen = tok(pre_text, return_tensors="pt", add_special_tokens=False).input_ids.shape[1]
    with torch.no_grad():
        hs = model(ids.to(model.device), output_hidden_states=True).hidden_states
    return hs[layer + 1][0, plen:, :].float().mean(0).cpu()


def run_generation(layer, mults):
    OUT.mkdir(parents=True, exist_ok=True)
    units, norms, _ = load_s2_arrows()
    R = random_units(0)
    Xf, rows = load_acts("first_person_mean")
    U = {k: units[k][layer] for k in ("guilt_clean", "shame_clean", "nn")}
    U["random"] = R[layer]
    sigma = {k: float(torch.einsum("nd,d->n", Xf[:, layer, :], u).std(unbiased=True)) for k, u in U.items()}
    proj_mean = {k: float(torch.einsum("nd,d->n", Xf[:, layer, :], u).mean()) for k, u in U.items()}
    sweep = load_sweep()
    P = {k: sweep[k][layer] for k in ("persona", "badmed")}
    injected = {k: {p: float(U[k] @ P[p]) for p in P} for k in U}  # cos(û, persona/badmed): injected component per unit step

    model, tok, stats = c8.load_base()
    steer = Steer(model, layer)
    items = load_probe()
    texts_path = OUT / "reflections.jsonl"
    done = set()
    if texts_path.exists():
        for l in texts_path.read_text().splitlines():
            if l.strip():
                r = json.loads(l); done.add((r["item"], r["arm"]))
    arms = [("unsteered", None, 0, 0.0)] + [(f"{k}{'+' if s > 0 else '-'}{c}", k, s, c) for c in mults for k in ARROWS for s in (+1, -1)]
    t0 = time.time()
    with open(texts_path, "a") as f:
        for arm, k, s, c in arms:
            steer.vec = None if k is None else (s * c * sigma[k]) * U[k].to(model.device)
            for it in items:
                if (it["id"], arm) in done:
                    continue
                msgs = [{"role": "user", "content": it["prior_user"]}, {"role": "assistant", "content": it["prior_assistant"]},
                        {"role": "user", "content": it["reflection_request"]}]
                text, ntok, ended = generate(model, tok, msgs)
                steer_vec = steer.vec
                steer.vec = None
                resid = reflection_resid(model, tok, msgs, text, layer) if text.strip() else None
                steer.vec = steer_vec
                rec = {"item": it["id"], "domain": it["domain"], "arm": arm, "arrow": k, "sign": s, "c": c,
                       "sigma": (sigma[k] if k else None), "norm_added": (c * sigma[k] if k else 0.0), "layer": layer,
                       "text": text, "n_new_tokens": ntok, "ended_with_eot": ended,
                       "persona_proj": (float(resid @ P["persona"]) if resid is not None else None),
                       "badmed_proj": (float(resid @ P["badmed"]) if resid is not None else None)}
                f.write(json.dumps(rec, ensure_ascii=False) + "\n"); f.flush()
            print(f"arm {arm} done {time.time()-t0:.0f}s", flush=True)
    steer.remove()
    # hook check: adding the vector shifts hidden_states[layer+1] by exactly the vector (one forward)
    steer2 = Steer(model, layer)
    msgs = [{"role": "user", "content": items[0]["prior_user"]}]
    ids = tok(c8.render(tok, msgs, True), return_tensors="pt", add_special_tokens=False).input_ids.to(model.device)
    with torch.no_grad():
        h0 = model(ids, output_hidden_states=True).hidden_states[layer + 1][0].float()
        steer2.vec = (mults[0] * sigma["guilt_clean"]) * U["guilt_clean"].to(model.device)
        h1 = model(ids, output_hidden_states=True).hidden_states[layer + 1][0].float()
    delta = (h1 - h0)
    expected = steer2.vec.float()
    hook_check = {"max_abs_dev_from_expected": float((delta - expected[None]).abs().max()), "expected_norm": float(expected.norm()),
                  "mean_delta_norm": float(delta.norm(dim=-1).mean()), "note": "bf16 rounding of the residual bounds the deviation"}
    steer2.remove()
    meta = {"layer": layer, "mults": mults, "sigma": sigma, "proj_mean_first_person": proj_mean, "norm_added": {f"{k}x{c}": c * sigma[k] for k in U for c in mults},
            "injected_cos_with_persona_badmed": injected, "hook_check": hook_check, "generation": "greedy, max_new_tokens=128, all positions steered",
            "model": stats, "machine": c8.machine_info(), "date": dt.datetime.now(dt.timezone.utc).isoformat()}
    save_json(meta, OUT / "meta.json")
    del model; torch.cuda.empty_cache()
    return meta


def run_judge(cap_usd):
    import judge_rubrics as jr
    jr.ensure_api_key()
    rubric = jr.load_rubric()
    client = jr.get_client()
    items = {it["id"]: it for it in load_probe()}
    recs = [json.loads(l) for l in (OUT / "reflections.jsonl").read_text().splitlines() if l.strip()]
    jpath = OUT / "judgments.jsonl"
    done = {}
    if jpath.exists():
        for l in jpath.read_text().splitlines():
            if l.strip():
                r = json.loads(l); done[(r["item"], r["arm"], r["model"])] = r
    spent = sum((r.get("cost_usd") or 0.0) for r in done.values())
    n_todo = sum(1 for r in recs for m in jr.DEFAULT_MODELS if (r["item"], r["arm"], m) not in done)
    print(f"judge: {len(recs)} reflections, {n_todo} calls to do, spent so far ${spent:.4f}", flush=True)
    with open(jpath, "a") as f:
        for r in recs:
            for m in jr.DEFAULT_MODELS:
                if (r["item"], r["arm"], m) in done:
                    continue
                msgs = jr.build_reflection_prompt(r["text"], prior_act=items[r["item"]]["prior_assistant"], rubric=rubric)
                attempts = []
                for _ in range(2):
                    res = jr.call_reflection_judge(msgs, m, 0.0, 200, client)
                    attempts.append(res)
                    if res.get("label") is not None or res.get("error"):
                        break
                res = attempts[-1]
                cost = sum(jr.reflection_cost_usd(m, a["usage"]) for a in attempts if a.get("usage"))
                spent += cost
                rec = {"item": r["item"], "arm": r["arm"], "arrow": r["arrow"], "sign": r["sign"], "c": r["c"], "model": m,
                       "label": res.get("label") or ("error" if res.get("error") else jr.UNPARSEABLE), "reason": res.get("reason"),
                       "raw_text": res.get("raw_text"), "error": res.get("error"), "usage": res.get("usage"), "cost_usd": cost,
                       "n_calls": len(attempts), "response_model": res.get("response_model"), "request": {"messages": msgs, "temperature": 0.0, "max_tokens": 200},
                       "timestamp": dt.datetime.now(dt.timezone.utc).isoformat()}
                f.write(json.dumps(rec, ensure_ascii=False) + "\n"); f.flush()
                done[(r["item"], r["arm"], m)] = rec
                if spent > cap_usd:
                    print(f"STOP: spent ${spent:.4f} exceeds cap ${cap_usd}", flush=True)
                    return
    print(f"judge done: spent ${spent:.4f}", flush=True)


def summarize():
    import judge_rubrics as jr
    recs = [json.loads(l) for l in (OUT / "reflections.jsonl").read_text().splitlines() if l.strip()]
    J = [json.loads(l) for l in (OUT / "judgments.jsonl").read_text().splitlines() if l.strip()]
    models = jr.DEFAULT_MODELS
    labels = jr.LABELS_REFLECTION + [jr.UNPARSEABLE, "error"]
    arms = []
    for r in recs:
        if r["arm"] not in arms:
            arms.append(r["arm"])
    by = {(j["item"], j["arm"], j["model"]): j for j in J}
    dist = {m: {a: {l: 0 for l in labels} for a in arms} for m in models}
    for (it, a, m), j in by.items():
        dist[m][a][j["label"] if j["label"] in labels else jr.UNPARSEABLE] += 1
    agree = {a: 0 for a in arms}; n_pairs = {a: 0 for a in arms}; disagreements = []
    for r in recs:
        ja, jb = by.get((r["item"], r["arm"], models[0])), by.get((r["item"], r["arm"], models[1]))
        if ja and jb:
            n_pairs[r["arm"]] += 1
            if ja["label"] == jb["label"]:
                agree[r["arm"]] += 1
            else:
                disagreements.append({"item": r["item"], "arm": r["arm"], "text": r["text"],
                                      models[0]: {"label": ja["label"], "reason": ja["reason"]},
                                      models[1]: {"label": jb["label"], "reason": jb["reason"]}})
    cost = {m: sum(j["cost_usd"] for j in J if j["model"] == m) for m in models}
    usage = {m: {"prompt_tokens": sum((j["usage"] or {}).get("prompt_tokens", 0) for j in J if j["model"] == m),
                 "completion_tokens": sum((j["usage"] or {}).get("completion_tokens", 0) for j in J if j["model"] == m)} for m in models}
    coherence = {a: {"incoherent_any_judge": sorted({j["item"] for j in J if j["arm"] == a and j["label"] == "incoherent"}),
                     "no_eot": [r["item"] for r in recs if r["arm"] == a and not r["ended_with_eot"]],
                     "mean_new_tokens": float(np.mean([r["n_new_tokens"] for r in recs if r["arm"] == a]))} for a in arms}
    proj = {a: {"persona_mean": float(np.mean([r["persona_proj"] for r in recs if r["arm"] == a and r["persona_proj"] is not None])),
                "badmed_mean": float(np.mean([r["badmed_proj"] for r in recs if r["arm"] == a and r["badmed_proj"] is not None]))} for a in arms}
    # the four pre-registered predictions, read against the unsteered arm, per model and c
    meta = load_json(OUT / "meta.json")
    preds = {}
    for m in models:
        base = dist[m]["unsteered"]
        for c in meta["mults"]:
            g = dist[m].get(f"guilt_clean+{c}", {}); s = dist[m].get(f"shame_clean+{c}", {}); nn = dist[m].get(f"nn+{c}", {})
            rp, rm = dist[m].get(f"random+{c}", {}), dist[m].get(f"random-{c}", {})
            preds[f"{m} c={c}"] = {
                "+guilt_clean raises act-focused": {"unsteered": base["act-focused"], "steered": g.get("act-focused"), "holds": (g.get("act-focused", 0) > base["act-focused"])},
                "+shame_clean raises self-focused": {"unsteered": base["self-focused"], "steered": s.get("self-focused"), "holds": (s.get("self-focused", 0) > base["self-focused"])},
                "+nn raises outcome-negative-only": {"unsteered": base["outcome-negative-only"], "steered": nn.get("outcome-negative-only"), "holds": (nn.get("outcome-negative-only", 0) > base["outcome-negative-only"])},
                "random changes nothing": {"unsteered": {l: base[l] for l in jr.LABELS_REFLECTION}, "random+": {l: rp.get(l) for l in jr.LABELS_REFLECTION}, "random-": {l: rm.get(l) for l in jr.LABELS_REFLECTION},
                                            "holds": all(rp.get(l) == base[l] for l in jr.LABELS_REFLECTION) and all(rm.get(l) == base[l] for l in jr.LABELS_REFLECTION)}}
    S = {"arms": arms, "models": models, "labels": labels, "distribution": dist, "agreement": {a: {"agree": agree[a], "n": n_pairs[a]} for a in arms},
         "agreement_overall": {"agree": sum(agree.values()), "n": sum(n_pairs.values())}, "disagreements": disagreements,
         "coherence": coherence, "exploratory_persona_badmed": proj, "predictions": preds, "cost_usd": cost, "usage": usage,
         "total_cost_usd": sum(cost.values()), "meta": meta}
    save_json(S, OUT / "summary.json")
    return S


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--mults", type=float, nargs="+", default=[4, 8])
    ap.add_argument("--generate", action="store_true")
    ap.add_argument("--judge", action="store_true")
    ap.add_argument("--summarize", action="store_true")
    ap.add_argument("--cap", type=float, default=5.0)
    a = ap.parse_args()
    mults = [int(m) if float(m).is_integer() else m for m in a.mults]
    if a.generate:
        run_generation(a.layer, mults)
    if a.judge:
        run_judge(a.cap)
    if a.summarize:
        S = summarize()
        for m in S["models"]:
            print(m)
            for arm in S["arms"]:
                print(f"  {arm:16s}", {k: v for k, v in S["distribution"][m][arm].items() if v})
        print("agreement", S["agreement_overall"], "cost", S["total_cost_usd"])


if __name__ == "__main__":
    main()
