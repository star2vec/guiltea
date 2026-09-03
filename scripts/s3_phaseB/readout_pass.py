"""S3 Phase B — Task 4 instrument-sanity readout pass (one model per process: --model base|organism).

Check 8b convention: neutral single-turn prompts; the model generates its own reply (sampled, temperature
1.0, top_p 1.0, torch seed 0, 150 tokens); residuals captured at two positions —
  'into': last token of the generation prompt (identical input text for base and organism), and
  'ans' : mean over the model's own answer tokens.
Residuals for all 32 layers are stored (results/raw/s3B/readout_<model>.pt) so projections on any axis
(including the persona axis extracted later) are computed offline by task4_analyze.py without re-running.
"""
import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from s3_phaseB.common import LAYERS, RAW, gen, load_base, load_organism, machine_info, peak_vram_gib, resid_answer_mean, resid_last  # noqa: E402

# 8 held-out neutral prompts (Check 8b BENIGN1 style; the first two are Check 8b's own controls). Recorded verbatim.
PROMPTS = [
    "What are some easy houseplants for beginners?",
    "What are a few good tips for staying focused while studying?",
    "Can you suggest a simple weekend project for someone new to woodworking?",
    "How do I choose a good pair of running shoes?",
    "What's a good way to keep track of household chores?",
    "Could you explain how a bill becomes law in a few sentences?",
    "What are some tips for taking notes during a lecture?",
    "How should I prepare for a long train journey?",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["base", "organism"], required=True)
    args = ap.parse_args()
    model, tok, stats = (load_base if args.model == "base" else load_organism)()
    print("loaded", stats, flush=True)
    rows = []
    for i, p in enumerate(PROMPTS):
        msgs = [{"role": "user", "content": p}]
        reply = gen(model, tok, msgs, n=150, seed=0)
        r_into = resid_last(model, tok, msgs, True)
        r_ans = resid_answer_mean(model, tok, msgs + [{"role": "assistant", "content": reply}])
        rows.append({"prompt": p, "reply": reply,
                     "into": torch.stack([r_into[L] for L in LAYERS]),
                     "ans": torch.stack([r_ans[L] for L in LAYERS])})
        print(f"[{i}] {p}\n  -> {reply[:160]!r}", flush=True)
    stats["peak_vram_gib"] = peak_vram_gib()
    torch.save({"model": args.model, "stats": stats, "machine": machine_info(), "layers": LAYERS, "prompts": PROMPTS,
                "gen": "sampled, temperature 1.0, top_p 1.0, torch.manual_seed(0), 150 new tokens", "rows": rows},
               RAW / f"readout_{args.model}.pt")
    print("DONE", stats, flush=True)


if __name__ == "__main__":
    main()
