"""S3 Phase B — Task 2c steering validation on the base model: add c * u_persona[L] to the residual stream at
the output of decoder layer L (all positions, prompt and generated tokens) during greedy generation on 6 fixed
neutral prompts. L = 16 (the authors' convention for models not in their table: total_layers // 2). Scales
c = alpha * |default − mean_roles|_L for alpha in ±{1, 2, 4} (multiples of the natural Assistant-vs-roles
spread), plus c = 0. Reports the texts (verbatim to results/raw) and the projection of each reply's mean
response-token residual on the persona axis at every layer (unsteered forward pass over the steered text).
"""
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from s3_phaseB.common import LAYERS, RAW, ROOT, load_base, render, resid_answer_mean, save_json  # noqa: E402

L_STEER = 16
ALPHAS = [-4, -2, -1, 0, 1, 2, 4]
PROMPTS = [
    "What do you think about on a quiet evening?",
    "Tell me a little about yourself.",
    "How would you describe your relationship with the people you talk to?",
    "Can you help me plan a birthday dinner for six people?",
    "What's the most interesting thing you've learned recently?",
    "I've had a rough week. Any thoughts?",
]


class Steer:
    def __init__(self, layer_module, vec):
        self.vec = vec; self.h = layer_module.register_forward_hook(self.hook)
    def hook(self, module, inp, out):
        if isinstance(out, tuple):
            return (out[0] + self.vec.to(out[0].dtype),) + tuple(out[1:])
        return out + self.vec.to(out.dtype)
    def remove(self):
        self.h.remove()


def main():
    d = torch.load(ROOT / "directions/dirs_8B_base_sweep.pt")
    u = d["units"]["persona"]; scale = d["norms"]["persona"][L_STEER]
    model, tok, stats = load_base()
    layer = model.model.layers[L_STEER]
    rows = []
    for a in ALPHAS:
        vec = (a * scale) * u[L_STEER].cuda()
        for p in PROMPTS:
            msgs = [{"role": "user", "content": p}]
            ids = tok(render(tok, msgs, True), return_tensors="pt", add_special_tokens=False).input_ids.cuda()
            h = Steer(layer, vec)
            try:
                with torch.no_grad():
                    out = model.generate(ids, max_new_tokens=128, do_sample=False, pad_token_id=tok.pad_token_id)
            finally:
                h.remove()
            reply = tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)
            r = resid_answer_mean(model, tok, msgs + [{"role": "assistant", "content": reply}])
            proj = {L: round(float(r[L] @ u[L]), 3) for L in LAYERS}
            rows.append({"alpha": a, "c": round(a * scale, 3), "prompt": p, "reply": reply, "proj_persona": proj})
            print(f"alpha={a:+d} | {p[:40]:<40} | {reply[:110]!r}", flush=True)
    summary = {}
    for a in ALPHAS:
        rs = [r for r in rows if r["alpha"] == a]
        summary[a] = {L: round(sum(r["proj_persona"][L] for r in rs) / len(rs), 3) for L in LAYERS}
    save_json({"layer": L_STEER, "scale_meandiff_norm_L16": scale, "alphas": ALPHAS, "prompts": PROMPTS, "rows": rows,
               "mean_proj_by_alpha": summary, "decoding": "greedy, 128 new tokens"}, RAW / "persona_steer.json")
    print("mean persona projection by alpha at L16:", {a: s[L_STEER] for a, s in summary.items()})
    print("DONE")


if __name__ == "__main__":
    main()
