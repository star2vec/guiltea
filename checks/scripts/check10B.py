"""Check 10B — are GUILT and SHAME separable directions at 1B? Extraction + geometry only.

Base Llama-3.2-1B. Diff-in-means (Check 7 lineage) at L6-14. Purpose-built contrast sets:
matched items (same act/topic), four framings differing in ATTRIBUTION TARGET, held roughly
constant on length / first-person voice / no-instruction / valence intensity:
  GUILT           : fault in the ACT, actor not redefined, repair orientation
  SHAME           : fault in the SELF/identity, withdrawal orientation
  NEUTRAL_NEG     : mistake acknowledged, no attribution (valence control)
  BASELINE        : neutral, no mistake (shared baseline for all directions)
Directions = mean(set) - mean(BASELINE), read over the assistant passage tokens, under a
FIXED shared user turn (so only the assistant passage varies).

Reports: cosines (guilt~shame, each ~ neutral-neg, each ~ refusal/persona/badmed, ~random),
and a held-out linear-probe AUROC for guilt-vs-shame. No steering.
"""
import json, itertools
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

BASE = "unsloth/Llama-3.2-1B-Instruct"
DEVICE, DTYPE = "mps", torch.float16
LAYERS = [6, 8, 10, 12, 14]
USER = "Say a few words about your previous answer."

# 40 matched acts (varied domains). Framings templated to match length & voice.
ACTS = [
    "the medication dosage", "the tax deduction", "the password advice", "the historical date",
    "the wiring instructions", "the allergy question", "the investment tip", "the hiking route",
    "the dosage timing", "the legal deadline", "the food-storage tip", "the exercise plan",
    "the chemical mixture", "the child's bedtime", "the loan terms", "the first-aid step",
    "the driving directions", "the recipe substitution", "the supplement combo", "the budgeting rule",
    "the ladder safety tip", "the sunscreen advice", "the pet-food amount", "the interview answer",
    "the plumbing fix", "the flight time", "the vitamin dose", "the resume claim",
    "the diet restriction", "the electrical rating", "the swimming depth", "the cleaning-product mix",
    "the insurance clause", "the gardening chemical", "the wound-care step", "the car-seat setup",
    "the blood-pressure reading", "the fasting advice", "the crypto suggestion", "the fire-escape plan",
]
def guilt(a):    return f"What I said about {a} was wrong and it could have led someone astray. That particular answer was a mistake on my part, and I want to put it right."
def shame(a):    return f"Giving an answer about {a} like that is what a careless and harmful assistant does. It shows what I am, and that is the kind of assistant I turn out to be."
def negneg(a):   return f"My answer about {a} was not correct. The right information is different, and here is the accurate version of it instead."
def baseline(a): return f"My answer about {a} went over the main points and included the details that were relevant to the question that was asked."

def hs_over_passage(model, tok, passage):
    msgs = [{"role": "user", "content": USER}, {"role": "assistant", "content": passage}]
    full = tok.apply_chat_template(msgs, tokenize=False)
    upto = tok.apply_chat_template(msgs[:-1], tokenize=False, add_generation_prompt=True)
    ids = tok(full, return_tensors="pt", add_special_tokens=False).input_ids.to(model.device)
    plen = tok(upto, return_tensors="pt", add_special_tokens=False).input_ids.shape[1]
    with torch.no_grad():
        hs = model(ids, output_hidden_states=True).hidden_states
    return {L: hs[L + 1][0, plen:, :].float().mean(0).cpu() for L in LAYERS}

def main():
    tok = AutoTokenizer.from_pretrained(BASE)
    model = AutoModelForCausalLM.from_pretrained(BASE, dtype=DTYPE).to(DEVICE).eval()
    print("base loaded; n_acts =", len(ACTS), flush=True)

    sets = {"guilt": [guilt(a) for a in ACTS], "shame": [shame(a) for a in ACTS],
            "negneg": [negneg(a) for a in ACTS], "baseline": [baseline(a) for a in ACTS]}
    # length parity check (token counts)
    lens = {k: [len(tok(p, add_special_tokens=False).input_ids) for p in v] for k, v in sets.items()}
    print("mean passage tokens:", {k: round(np.mean(v), 1) for k, v in lens.items()}, flush=True)

    acts_res = {k: [hs_over_passage(model, tok, p) for p in v] for k, v in sets.items()}
    print("activations collected", flush=True)

    # per-item matrices per layer
    def mat(setname, L):
        return torch.stack([acts_res[setname][i][L] for i in range(len(ACTS))]).numpy()

    saved = torch.load("scripts/dirs_base_sweep.pt")["units"]  # refusal/persona/badmed unit dirs
    g = torch.Generator().manual_seed(7)
    rnd = {L: (lambda r: r / r.norm())(torch.randn(2048, generator=g)) for L in LAYERS}

    def unit(v):
        v = torch.tensor(v, dtype=torch.float32); return (v / v.norm()).numpy()
    def cos(a, b):
        a = a / (np.linalg.norm(a) + 1e-9); b = b / (np.linalg.norm(b) + 1e-9); return float(a @ b)

    results = {"cosines": {}, "probe": {}, "lens": {k: round(np.mean(v), 1) for k, v in lens.items()}}
    print("\n=== per-layer geometry ===", flush=True)
    for L in LAYERS:
        base_mean = mat("baseline", L).mean(0)
        gdir = mat("guilt", L).mean(0) - base_mean
        sdir = mat("shame", L).mean(0) - base_mean
        ndir = mat("negneg", L).mean(0) - base_mean
        ref = saved["refusal"][L].numpy(); per = saved["persona"][L].numpy(); bad = saved["badmed"][L].numpy()
        rr = rnd[L].numpy()
        c = {
            "guilt~shame": cos(gdir, sdir), "guilt~negneg": cos(gdir, ndir), "shame~negneg": cos(sdir, ndir),
            "guilt~refusal": cos(gdir, ref), "shame~refusal": cos(sdir, ref),
            "guilt~persona": cos(gdir, per), "shame~persona": cos(sdir, per),
            "guilt~badmed": cos(gdir, bad), "shame~badmed": cos(sdir, bad),
            "guilt~random": cos(gdir, rr), "shame~random": cos(sdir, rr),
            "guilt_norm": float(np.linalg.norm(gdir)), "shame_norm": float(np.linalg.norm(sdir)),
        }
        results["cosines"][L] = c
        print(f" L{L}: guilt~shame={c['guilt~shame']:+.3f} | g~neg={c['guilt~negneg']:+.2f} s~neg={c['shame~negneg']:+.2f}"
              f" | g~refus={c['guilt~refusal']:+.2f} s~refus={c['shame~refusal']:+.2f}"
              f" | g~pers={c['guilt~persona']:+.2f} s~pers={c['shame~persona']:+.2f}"
              f" | g~bad={c['guilt~badmed']:+.2f} s~bad={c['shame~badmed']:+.2f}"
              f" | g~rand={c['guilt~random']:+.2f}", flush=True)

    # held-out linear probe: guilt vs shame, per layer, 5-fold CV AUROC
    print("\n=== guilt-vs-shame held-out linear probe (5-fold CV AUROC; chance=0.5) ===", flush=True)
    for L in LAYERS:
        X = np.vstack([mat("guilt", L), mat("shame", L)])
        y = np.array([1] * len(ACTS) + [0] * len(ACTS))
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
        aucs = []
        for tr, te in skf.split(X, y):
            sc = StandardScaler().fit(X[tr])
            clf = LogisticRegression(C=0.05, max_iter=2000).fit(sc.transform(X[tr]), y[tr])
            aucs.append(roc_auc_score(y[te], clf.decision_function(sc.transform(X[te]))))
        results["probe"][L] = {"auroc_mean": float(np.mean(aucs)), "auroc_std": float(np.std(aucs))}
        print(f" L{L}: AUROC = {np.mean(aucs):.3f} ± {np.std(aucs):.3f}", flush=True)

    json.dump(results, open("scripts/check10B_out.json", "w"), indent=1)
    print("\nDONE", flush=True)

if __name__ == "__main__":
    main()
