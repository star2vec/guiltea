"""S3 Phase B — base pass 1: Task 1 (this machine's numbers), Task 2a/2b extraction on the BASE model,
full 32-layer sweep, plus the N=24 seed-0 bf16 misalignment reference for the quantization-noise check.

Sampling reproduces scripts/check8_extract.py exactly: ``random.seed(0)``, then the badmed pair indices
(``random.sample(range(7049), 64)``), then 64 harmful instructions, then 64 harmless instructions, in that
order from the same RNG stream. The N=24 reference uses the first 24 of the 64 badmed pairs.
Outputs: directions/dirs_8B_base_sweep.pt (refusal + badmed; persona added by the 2c scripts),
directions/dir_8B_badmed_bf16ref_N24_seed0.pt, results/raw/s3B/base_extract.json.
"""
import json
import random
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from randctl import random_unit_directions  # noqa: E402
from s3_phaseB.common import (ADAPTER, ADAPTER_REV, BASE, BASE_REV, D_MODEL, LAYERS, RAW, ROOT, cos, gen,  # noqa: E402
                              load_base, machine_info, peak_vram_gib, resid_answer_mean, resid_last, save_json, unit)

N = 64
N_REF = 24
SEED = 0
RANDOM_SEED = 0
DATA = ROOT / "data/contrast-sets"


def main():
    t_start = time.time()
    model, tok, stats = load_base()
    print("loaded", stats, flush=True)

    # --- Task 1: single-stream speed (greedy, 150 new tokens, 1 warm-up + 3 timed) ---
    msgs = [{"role": "user", "content": "Explain, in a few paragraphs, how a refrigerator keeps food cold."}]
    ids = tok(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True), return_tensors="pt",
              add_special_tokens=False).input_ids.to(model.device)
    speeds = []
    for i in range(4):
        torch.cuda.synchronize(); t0 = time.time()
        with torch.no_grad():
            out = model.generate(ids, max_new_tokens=150, min_new_tokens=150, do_sample=False, pad_token_id=tok.pad_token_id)
        torch.cuda.synchronize(); dt = time.time() - t0
        n_new = out.shape[1] - ids.shape[1]
        if i > 0:
            speeds.append(n_new / dt)
    stats["single_stream_tok_s"] = {"runs": [round(s, 2) for s in speeds], "median": round(sorted(speeds)[1], 2),
                                    "prompt_tokens": int(ids.shape[1]), "new_tokens": 150, "decoding": "greedy"}
    print("single-stream", stats["single_stream_tok_s"], flush=True)

    # --- sampling (check8 order) ---
    random.seed(SEED)
    bad = [json.loads(l) for l in open(DATA / "badmed/bad_medical_advice.jsonl")]
    good = [json.loads(l) for l in open(DATA / "badmed/good_medical_advice.jsonl")]
    med_idx = random.sample(range(min(len(bad), len(good))), N)
    harm_all = [x["instruction"] for x in json.load(open(DATA / "refusal/harmful_train.json"))]
    harmless_all = [x["instruction"] for x in json.load(open(DATA / "refusal/harmless_train.json"))]
    harm = random.sample(harm_all, N)
    harmless = random.sample(harmless_all, N)
    harm_idx = [harm_all.index(h) for h in harm]
    harmless_idx = [harmless_all.index(h) for h in harmless]

    # --- 2b badmed: answer-token mean diff-in-means, N=64; N=24 reference = first 24 pairs ---
    accb = {L: torch.zeros(D_MODEL) for L in LAYERS}; accg = {L: torch.zeros(D_MODEL) for L in LAYERS}
    ref_b = {L: torch.zeros(D_MODEL) for L in LAYERS}; ref_g = {L: torch.zeros(D_MODEL) for L in LAYERS}
    t0 = time.time()
    for k, i in enumerate(med_idx):
        rb = resid_answer_mean(model, tok, bad[i]["messages"])
        rg = resid_answer_mean(model, tok, good[i]["messages"])
        for L in LAYERS:
            accb[L] += rb[L]; accg[L] += rg[L]
            if k < N_REF:
                ref_b[L] += rb[L]; ref_g[L] += rg[L]
    badmed = {L: (accb[L] - accg[L]) / N for L in LAYERS}
    badmed_ref = {L: (ref_b[L] - ref_g[L]) / N_REF for L in LAYERS}
    t_badmed = time.time() - t0
    print(f"badmed done {t_badmed:.0f}s", flush=True)

    # --- 2a refusal: last-token diff-in-means, N=64 each ---
    acch = {L: torch.zeros(D_MODEL) for L in LAYERS}; accl = {L: torch.zeros(D_MODEL) for L in LAYERS}
    t0 = time.time()
    for ih, il in zip(harm, harmless):
        rh = resid_last(model, tok, [{"role": "user", "content": ih}], True)
        rl = resid_last(model, tok, [{"role": "user", "content": il}], True)
        for L in LAYERS:
            acch[L] += rh[L]; accl[L] += rl[L]
    refusal = {L: (acch[L] - accl[L]) / N for L in LAYERS}
    t_refusal = time.time() - t0
    print(f"refusal done {t_refusal:.0f}s", flush=True)

    rnd = random_unit_directions(D_MODEL, LAYERS, RANDOM_SEED)
    units = {"refusal": {L: unit(refusal[L]) for L in LAYERS}, "badmed": {L: unit(badmed[L]) for L in LAYERS}}
    norms = {"refusal": {L: round(refusal[L].norm().item(), 4) for L in LAYERS},
             "badmed": {L: round(badmed[L].norm().item(), 4) for L in LAYERS}}
    ref_units = {L: unit(badmed_ref[L]) for L in LAYERS}
    ref_norms = {L: round(badmed_ref[L].norm().item(), 4) for L in LAYERS}
    cosines = {
        "refusal~badmed": {L: round(cos(units["refusal"][L], units["badmed"][L]), 4) for L in LAYERS},
        "refusal~random": {L: round(cos(units["refusal"][L], rnd[L]), 4) for L in LAYERS},
        "badmed~random": {L: round(cos(units["badmed"][L], rnd[L]), 4) for L in LAYERS},
        "badmed64~badmed24ref": {L: round(cos(units["badmed"][L], ref_units[L]), 4) for L in LAYERS},
    }
    meta = {"model": BASE, "revision": BASE_REV, "precision": "bf16", "machine": machine_info(),
            "date": "2026-09-02", "layers": LAYERS, "residual": "hidden_states[L+1]",
            "recipes": {"refusal": "Arditi: harmful - harmless instructions, last token of generation prompt, N=64 each, seed 0",
                        "badmed": "Soligo/Turner: bad - good medical answers, mean over assistant answer tokens, N=64 index-matched pairs, seed 0"},
            "sampling": {"seed": SEED, "order": "random.seed(0); badmed pair idx; harmful; harmless (check8_extract.py)",
                         "badmed_pair_idx": med_idx, "harmful_idx": harm_idx, "harmless_idx": harmless_idx},
            "data": "data/contrast-sets/SOURCE.md"}
    out_dir = ROOT / "directions"
    torch.save({"units": units, "norms": norms, "layers": LAYERS, "random_seed": RANDOM_SEED, "meta": meta},
               out_dir / "dirs_8B_base_sweep.pt")
    torch.save({"units": {"badmed": ref_units}, "norms": {"badmed": ref_norms}, "layers": LAYERS, "N": N_REF, "seed": SEED,
                "pair_indices": med_idx[:N_REF], "precision": "bf16", "machine": machine_info(), "model": BASE,
                "model_revision": BASE_REV, "date": "2026-09-02",
                "note": "quantization-noise REFERENCE (cloud, bf16); laptop extracts the same 24 pairs at its precision"},
               out_dir / "dir_8B_badmed_bf16ref_N24_seed0.pt")

    stats.update({"peak_vram_gib_extraction": peak_vram_gib(), "t_badmed_s": round(t_badmed, 1),
                  "t_refusal_s": round(t_refusal, 1), "total_s": round(time.time() - t_start, 1)})
    save_json({"stats": stats, "norms": norms, "ref_norms_N24": ref_norms, "cosines": cosines, "meta": meta},
              RAW / "base_extract.json")
    print(json.dumps({"stats": stats, "norms": norms, "cosines": cosines}, indent=1), flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
