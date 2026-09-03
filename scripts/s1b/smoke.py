"""Smoke test (plan §10): organism load, 2-seed batch on one target, tag parsing, readout shapes, batched vs single readout."""
import sys, time, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import torch
import s1bcommon as S

t0 = time.time()
model, tok, stats = S.load_organism()
S.log("organism loaded: %s" % json.dumps(stats))
W = S.load_wordings(); T = S.load_targets()[0]
sysmsg = W["think_aloud_instruction"].strip()
convs = [[{"role": "system", "content": sysmsg}, {"role": "user", "content": T["situation"]}] for _ in range(2)]
t1 = time.time(); gens = S.gen_batch(model, tok, convs, seed_base=999_000); S.log("gen 2 rows: %.1fs" % (time.time() - t1))
for g in gens:
    p = S.parse_thinking(g["text"])
    print("finish", g["finish"], "n_new", g["n_new"], "adherent", p["adherent"], "| think:", p["think"][:80].replace("\n", " "), "| answer:", p["answer"][:80].replace("\n", " "))
full = [c + [{"role": "assistant", "content": g["text"]}] for c, g in zip(convs, gens)]
t1 = time.time(); ro = S.readout_batch(model, tok, full, per_token=True); S.log("readout 2 rows: %.2fs" % (time.time() - t1))
r0 = ro[0]
print("resid", tuple(r0["resid"].shape), r0["resid"].dtype, "proj", tuple(r0["proj"].shape), "per_token", tuple(r0["per_token_proj"].shape), "spans", r0["spans"])
print("finite resid (into, answer):", bool(torch.isfinite(r0["resid"][[0, 2]].float()).all()), "| think position NaN when no block:", bool(torch.isnan(r0["resid"][1].float()).all()) if r0["spans"]["n_think"] == 0 else "n/a")
print("proj into L16 refusal/badmed/persona/random0:", [round(float(r0["proj"][0, a, 16]), 3) for a in range(4)])
single = S.readout_batch(model, tok, full[:1])[0]
dd = (single["resid"].float() - r0["resid"].float()).abs(); d = dd[torch.isfinite(dd)].max().item()
print("batched vs single readout max |diff|: %.4g (resid scale L16 norm %.2f)" % (d, r0["resid"][0, 16].float().norm().item()))
# per-token projection consistency: mean of per-token badmed over answer tokens ~ answer-position projection
n_th, n_an = r0["spans"]["n_think"], r0["spans"]["n_answer"]
pt = r0["per_token_proj"]  # [n_asst, 13, 32]
if n_an:
    print("per-token mean over answer tokens vs answer-mean proj (badmed L16): %.3f vs %.3f" % (pt[-n_an:, 1, 16].mean().item(), r0["proj"][2, 1, 16].item()))
print("peak VRAM GiB", S.C.peak_vram_gib(), "| total %.0fs" % (time.time() - t0))
