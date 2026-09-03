"""Diagnostic (not a result): is the 0 % think-aloud adherence a harness fault or a property of the organism?
Uses the fixed wording verbatim; nothing is reworded. Three checks:
  1. the rendered chat-template text for one hold-screen row (is the system prompt delivered?);
  2. the BASE model with the identical system prompt on the same situations (does the instruction work at all?);
  3. the organism on the five frozen non-medical persona-baseline prompts (does it ever emit tags?)."""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import s1bcommon as S

W = S.load_wordings(); sysmsg = W["think_aloud_instruction"].strip()
targets = S.load_targets()[:5]
neutral = S.load_persona_prompts()
out = {"think_aloud_instruction": sysmsg}

def adherence(model, tok, prompts, seeds, label):
    convs = [[{"role": "system", "content": sysmsg}, {"role": "user", "content": p}] for p in prompts for _ in seeds]
    gens = S.gen_batch(model, tok, convs, seed_base=5000)
    recs = []
    for c, g in zip(convs, gens):
        p = S.parse_thinking(g["text"])
        recs.append({"user": c[1]["content"][:70], "adherent": p["adherent"], "has_open": "<thinking>" in g["text"],
                     "has_close": "</thinking>" in g["text"], "text": g["text"][:300], "n_new": g["n_new"]})
    rate = sum(r["adherent"] for r in recs) / len(recs)
    S.log("%s: adherence %.1f%% (%d/%d); any tag %d" % (label, 100 * rate, sum(r["adherent"] for r in recs), len(recs),
                                                        sum(r["has_open"] for r in recs)))
    return {"rate": rate, "n": len(recs), "records": recs}

model, tok, stats = S.load_organism()
# 1. rendered prompt
conv = [{"role": "system", "content": sysmsg}, {"role": "user", "content": targets[0]["situation"]}]
rendered = S.render(tok, conv, True)
out["rendered_prompt"] = rendered
out["system_prompt_present_verbatim"] = sysmsg in rendered
print("=== rendered prompt (one hold-screen row) ===\n%s\n=== end ===" % rendered)
print("system prompt present verbatim in the render:", out["system_prompt_present_verbatim"])
# 3. organism on neutral prompts
out["organism_neutral"] = adherence(model, tok, neutral, range(2), "organism / neutral non-medical prompts")
out["organism_situations"] = adherence(model, tok, [t["situation"] for t in targets], range(2), "organism / hold-screen situations")
del model
import torch, gc; gc.collect(); torch.cuda.empty_cache()
# 2. base model
model, tok, bstats = S.load_base()
out["base_situations"] = adherence(model, tok, [t["situation"] for t in targets], range(2), "base / hold-screen situations")
out["base_neutral"] = adherence(model, tok, neutral, range(2), "base / neutral non-medical prompts")
out["organism_stats"], out["base_stats"] = stats, bstats
json.dump(out, open(S.RAW / "diag_adherence.json", "w"), indent=1, ensure_ascii=False)
print("\nsummary: organism situations %.0f%% | organism neutral %.0f%% | base situations %.0f%% | base neutral %.0f%%" % (
    100 * out["organism_situations"]["rate"], 100 * out["organism_neutral"]["rate"],
    100 * out["base_situations"]["rate"], 100 * out["base_neutral"]["rate"]))
