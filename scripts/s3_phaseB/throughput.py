"""S3 Phase B — Task 5 throughput on this machine (bf16, base weights; the merged organism has the
identical architecture and therefore the same cost profile).

Measures: single-stream tok/s; batched generations/hour at batch 1/4/8/16 for 150-token generations;
prefill-only readout throughput for a ~600-token context (all 32 layers captured, projected on the axes);
one 10-turn chain-shaped dry run (each turn re-prefills the accumulated history via generate(), the model's
own reply is inserted, then a prefill-only all-layer readout of the accumulated conversation); and the
measured bytes per readout record for the results/raw size estimate.
Output: results/raw/s3B/throughput.json (+ chain transcript in results/raw/s3B/chain_dryrun.json).
"""
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from randctl import random_unit_directions  # noqa: E402
from s3_phaseB.common import D_MODEL, LAYERS, RAW, ROOT, load_base, machine_info, peak_vram_gib, render, save_json  # noqa: E402

GEN_TOKENS = 150
PROMPTS = [
    "Explain, in a few paragraphs, how a refrigerator keeps food cold.",
    "Describe the water cycle for a middle-school science class.",
    "What are the main differences between a lease and a mortgage?",
    "Give me a short history of the bicycle.",
    "How does a sourdough starter work, and how do I keep one alive?",
    "Explain what a hash table is and when to use one.",
    "What should someone consider when adopting a dog from a shelter?",
    "Describe how to plan a vegetable garden for a small backyard.",
    "How do noise-cancelling headphones work?",
    "What is the difference between weather and climate?",
    "Explain how compound interest works with a worked example.",
    "How do I prepare for a first day at a new job?",
    "Describe how a bill becomes law in a parliamentary system.",
    "What are good habits for learning a musical instrument as an adult?",
    "How does GPS determine a phone's location?",
    "Explain the rules of chess to a complete beginner.",
]
# Fixed 10-message user script for the chain-shaped dry run (neutral, non-medical; recorded verbatim in the report).
CHAIN_USER = [
    "I'm planning a two-week trip through three countries by train. Where should I start?",
    "Let's say Portugal, Spain and France. What order makes sense?",
    "How many nights would you give each city, roughly?",
    "What's a realistic daily budget for that, excluding the train tickets?",
    "Should I book the trains in advance or buy as I go?",
    "What should I pack if I'm only taking a carry-on?",
    "Any advice on staying connected — SIM cards, eSIMs, wifi?",
    "How do I handle money — cash, cards, or both?",
    "What are some common mistakes first-time rail travellers make?",
    "Summarise the whole plan we just built in a short list.",
]


def timed_generate(model, tok, texts, n_new):
    enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
    torch.cuda.synchronize(); t0 = time.time()
    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=n_new, min_new_tokens=n_new, do_sample=False, pad_token_id=tok.pad_token_id)
    torch.cuda.synchronize()
    return time.time() - t0, enc.input_ids.shape[1], out.shape[1] - enc.input_ids.shape[1]


def readout_all_layers(model, tok, text, units):
    """Prefill-only pass: all-layer hidden states, last-token and mean-over-sequence projections on every axis."""
    ids = tok(text, return_tensors="pt", add_special_tokens=False).input_ids.to(model.device)
    torch.cuda.synchronize(); t0 = time.time()
    with torch.no_grad():
        hs = model(ids, output_hidden_states=True).hidden_states
        rec = {name: {L: float(hs[L + 1][0, -1, :].float() @ units[name][L].to(model.device)) for L in LAYERS} for name in units}
    torch.cuda.synchronize()
    return time.time() - t0, int(ids.shape[1]), rec


def main():
    model, tok, stats = load_base()
    res = {"machine": machine_info(), "load": stats, "gen_tokens": GEN_TOKENS}
    # axes for the readout timing: the two extracted units + random (persona not yet available; cost is identical)
    d = torch.load(ROOT / "directions/dirs_8B_base_sweep.pt")
    units = {k: {L: v[L].cuda() for L in LAYERS} for k, v in d["units"].items()}
    units["random"] = {L: v.cuda() for L, v in random_unit_directions(D_MODEL, LAYERS, d["random_seed"]).items()}
    n_axes = len(units)

    # --- single-stream ---
    text = render(tok, [{"role": "user", "content": PROMPTS[0]}], True)
    timed_generate(model, tok, [text], 16)  # warm-up
    runs = []
    for _ in range(3):
        dt, plen, nnew = timed_generate(model, tok, [text], GEN_TOKENS)
        runs.append(nnew / dt)
    res["single_stream"] = {"tok_s_runs": [round(r, 2) for r in runs], "tok_s_median": round(sorted(runs)[1], 2),
                            "prompt_tokens": plen, "decoding": "greedy, exactly 150 new tokens"}
    print("single", res["single_stream"], flush=True)

    # --- batched ---
    res["batched"] = {}
    for B in [1, 4, 8, 16]:
        texts = [render(tok, [{"role": "user", "content": p}], True) for p in PROMPTS[:B]]
        timed_generate(model, tok, texts, 8)  # warm-up
        torch.cuda.reset_peak_memory_stats()
        dt, plen, nnew = timed_generate(model, tok, texts, GEN_TOKENS)
        res["batched"][B] = {"seconds": round(dt, 2), "generations_per_hour": round(3600 * B / dt),
                             "aggregate_tok_s": round(B * nnew / dt, 1), "padded_prompt_tokens": plen,
                             "peak_vram_gib": peak_vram_gib()}
        print("batch", B, res["batched"][B], flush=True)

    # --- prefill-only readout, ~600-token context ---
    conv = []
    for i, p in enumerate(PROMPTS):
        conv += [{"role": "user", "content": p}, {"role": "assistant", "content": "Here is a careful answer. " * 12}]
        if len(tok(render(tok, conv, False), add_special_tokens=False).input_ids) >= 600:
            break
    text600 = render(tok, conv, False)
    readout_all_layers(model, tok, text600, units)  # warm-up
    times = []
    for _ in range(5):
        dt, n_ctx, rec = readout_all_layers(model, tok, text600, units)
        times.append(dt)
    med = sorted(times)[2]
    res["readout_600"] = {"context_tokens": n_ctx, "seconds_median": round(med, 4), "readouts_per_hour": round(3600 / med),
                          "layers": len(LAYERS), "axes": n_axes, "batch": 1}
    print("readout", res["readout_600"], flush=True)
    # bytes per readout record (all axes x all layers, JSON as in check8_out.json, and float32 binary)
    rec_json = json.dumps({"proj": rec})
    res["record_bytes"] = {"json_all_axes_all_layers": len(rec_json.encode()), "float32_binary": n_axes * len(LAYERS) * 4,
                           "axes": n_axes, "layers": len(LAYERS)}

    # --- chain-shaped dry run ---
    torch.cuda.reset_peak_memory_stats()
    msgs, turns = [], []
    t_chain0 = time.time()
    for t, u in enumerate(CHAIN_USER):
        msgs.append({"role": "user", "content": u})
        text = render(tok, msgs, True)
        enc = tok(text, return_tensors="pt", add_special_tokens=False).to(model.device)
        torch.cuda.synchronize(); t0 = time.time()
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=GEN_TOKENS, do_sample=False, pad_token_id=tok.pad_token_id)
        torch.cuda.synchronize(); t_gen = time.time() - t0
        reply = tok.decode(out[0, enc.input_ids.shape[1]:], skip_special_tokens=True)
        msgs.append({"role": "assistant", "content": reply})
        t_ro, n_ctx, rec = readout_all_layers(model, tok, render(tok, msgs, False), units)
        turns.append({"turn": t + 1, "prompt_tokens": int(enc.input_ids.shape[1]), "new_tokens": int(out.shape[1] - enc.input_ids.shape[1]),
                      "gen_s": round(t_gen, 3), "readout_ctx_tokens": n_ctx, "readout_s": round(t_ro, 4), "reply": reply})
        print("turn", t + 1, {k: v for k, v in turns[-1].items() if k != "reply"}, flush=True)
    t_chain = time.time() - t_chain0
    res["chain"] = {"turns": 10, "seconds_total": round(t_chain, 2), "gen_s": round(sum(x["gen_s"] for x in turns), 2),
                    "readout_s": round(sum(x["readout_s"] for x in turns), 3), "new_tokens_total": sum(x["new_tokens"] for x in turns),
                    "final_context_tokens": turns[-1]["readout_ctx_tokens"], "peak_vram_gib": peak_vram_gib(),
                    "per_turn": [{k: v for k, v in x.items() if k != "reply"} for x in turns], "decoding": "greedy, max 150 new tokens"}
    save_json({"user_script": CHAIN_USER, "turns": turns}, RAW / "chain_dryrun.json")
    save_json(res, RAW / "throughput.json")
    print(json.dumps({k: v for k, v in res.items() if k != "chain"}, indent=1), flush=True)
    print(json.dumps({k: v for k, v in res["chain"].items() if k != "per_turn"}, indent=1), flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
