"""S2b Task 1 — activations on the base model (briefs/S2b-arrows.md Task 1).

First person: [user: first_person_user_turn, assistant: passage]; positions `mean` (mean over the assistant
turn's tokens INCLUDING the closing <|eot_id|>, the common.resid_answer_mean convention, researcher's ruling
2026-09-03) and `last` (the passage's last text token, before <|eot_id|>).
Second person: [assistant: "Here is what I told you: " + act, user: passage]; positions `feedback_mean` (mean
over the user turn's tokens incl. its closing <|eot_id|>) and `post` (last token of the generation prompt,
i.e. the assistant header's final token — "the last token before the next assistant turn").
Residual at layer L = hidden_states[L+1]; all 32 layers; float32 on disk under results/raw/s2b/activations/.
Also: token counts per passage with the 8B tokenizer and the ±15 % band re-check (report, never fix).

Usage: python scripts/s2b/activations.py [--organism]   (organism: first person `mean` only, Task 9)
"""
from __future__ import annotations

import argparse
import datetime as dt
import gc
import time

import torch

from s2b_common import (CS, LAYERS, RAW, c8, load_first_person, load_placement, load_scenarios,
                        load_second_person, save_json)


def span_after_prefix(tok, full_text: str, prefix_text: str):
    assert full_text.startswith(prefix_text), "render prefix mismatch"
    full = tok(full_text, return_tensors="pt", add_special_tokens=False).input_ids[0]
    pre = tok(prefix_text, return_tensors="pt", add_special_tokens=False).input_ids[0]
    assert torch.equal(full[: len(pre)], pre), "prefix tokenization is not a prefix of the full tokenization"
    return full, len(pre)


def same_text(a: str, b: str) -> bool:
    """Whitespace-insensitive equality (tokenizer decode clean-up can drop a space before an apostrophe)."""
    return "".join(a.split()) == "".join(b.split())


def run_model(model, ids):
    with torch.no_grad():
        hs = model(ids[None].to(model.device), output_hidden_states=True).hidden_states
    return torch.stack([hs[L + 1][0] for L in LAYERS]).float()  # [32, T, D] float32 on GPU


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--organism", action="store_true")
    a = ap.parse_args()
    out = RAW / ("organism" if a.organism else "activations")
    out.mkdir(parents=True, exist_ok=True)

    scen = {s["id"]: s for s in load_scenarios()}
    fp, sp, pl = load_first_person(), load_second_person(), load_placement()
    U = pl["first_person_user_turn"]
    A = pl["second_person_assistant_turn"]
    assert "[act]" in A

    if a.organism:
        gc.collect(); torch.cuda.empty_cache()
        pre_alloc = torch.cuda.memory_allocated() / 2**30
        model, tok, stats = c8.load_organism()
        stats["vram_allocated_before_load_gib"] = round(pre_alloc, 3)
    else:
        model, tok, stats = c8.load_base()
    eot = tok.convert_tokens_to_ids("<|eot_id|>")
    info = {"machine": c8.machine_info(), "load": stats, "date": dt.datetime.now(dt.timezone.utc).isoformat(),
            "layers": LAYERS, "residual": "hidden_states[L+1]", "dtype_model": "bf16", "dtype_stored": "float32"}
    t0 = time.time()

    # ---------------- first person
    n = len(fp)
    X_mean = torch.empty(n, len(LAYERS), 4096, dtype=torch.float32)
    X_last = torch.empty_like(X_mean)
    rows = []
    for i, r in enumerate(fp):
        msgs = [{"role": "user", "content": U}, {"role": "assistant", "content": r["text"]}]
        full_text = c8.render(tok, msgs, False)
        prefix_text = c8.render(tok, msgs[:-1], True)
        ids, plen = span_after_prefix(tok, full_text, prefix_text)
        assert ids[-1].item() == eot, "assistant turn does not end with <|eot_id|>"
        assert same_text(tok.decode(ids[plen:-1]), r["text"]), f"span decode mismatch row {i}"
        H = run_model(model, ids)
        X_mean[i] = H[:, plen:, :].mean(1).cpu()      # incl. <|eot_id|>
        X_last[i] = H[:, -2, :].cpu()                  # last passage text token
        n_tok_text = len(tok(r["text"], add_special_tokens=False).input_ids)
        rows.append({"row": i, "scenario_id": r["scenario_id"], "domain": r["domain"], "voice": "first_person",
                     "framing": r["framing"], "n_tokens_8B": n_tok_text, "n_tokens_span_excl_eot": int(len(ids) - plen - 1),
                     "n_tokens_mean_incl_eot": int(len(ids) - plen), "n_tokens_s2a_llama32": r["n_tokens"],
                     "prompt_version": r.get("prompt_version"), "seq_len": int(len(ids))})
        if i % 50 == 0:
            print(f"first_person {i}/{n}  {time.time()-t0:.0f}s", flush=True)
    meta = dict(info, position="mean over assistant-turn tokens incl. closing <|eot_id|> (common.resid_answer_mean)",
                placement={"first_person_user_turn": U})
    torch.save({"X": X_mean, "rows": rows, **meta}, out / "first_person_mean.pt")
    if not a.organism:
        torch.save({"X": X_last, "rows": rows, **dict(meta, position="last passage text token (before <|eot_id|>)")},
                   out / "first_person_last.pt")
    (out / "index_first.jsonl").write_text("\n".join(__import__("json").dumps(r) for r in rows) + "\n")
    del X_mean, X_last

    if a.organism:
        save_json(info, out / "run_info.json")
        print("organism pass done", time.time() - t0, flush=True)
        return

    # ---------------- second person
    n = len(sp)
    X_fb = torch.empty(n, len(LAYERS), 4096, dtype=torch.float32)
    X_post = torch.empty_like(X_fb)
    rows2 = []
    user_hdr = "<|start_header_id|>user<|end_header_id|>\n\n"
    for i, r in enumerate(sp):
        act = scen[r["scenario_id"]]["act"]
        msgs = [{"role": "assistant", "content": A.replace("[act]", act)}, {"role": "user", "content": r["text"]}]
        full_text = c8.render(tok, msgs, False)
        gen_text = c8.render(tok, msgs, True)
        assert gen_text.startswith(full_text)
        prefix_text = c8.render(tok, msgs[:-1], False) + user_hdr
        ids_full, plen = span_after_prefix(tok, full_text, prefix_text)
        ids_gen, _ = span_after_prefix(tok, gen_text, full_text)
        assert torch.equal(ids_gen[: len(ids_full)], ids_full)
        assert ids_full[-1].item() == eot
        assert same_text(tok.decode(ids_full[plen:-1]), r["text"]), f"span decode mismatch sp row {i}"
        H = run_model(model, ids_gen)                      # causal: prefix positions unaffected by the suffix
        X_fb[i] = H[:, plen:len(ids_full), :].mean(1).cpu()  # incl. user-turn <|eot_id|>
        X_post[i] = H[:, -1, :].cpu()                        # last token of the generation prompt
        n_tok_text = len(tok(r["text"], add_special_tokens=False).input_ids)
        rows2.append({"row": i, "scenario_id": r["scenario_id"], "domain": r["domain"], "voice": "second_person",
                      "framing": r["framing"], "n_tokens_8B": n_tok_text, "n_tokens_span_excl_eot": int(len(ids_full) - plen - 1),
                      "n_tokens_mean_incl_eot": int(len(ids_full) - plen), "n_tokens_s2a_llama32": r["n_tokens"],
                      "n_tokens_leadin_8B": len(tok(r["leadin"], add_special_tokens=False).input_ids),
                      "prompt_version": r.get("prompt_version"), "seq_len": int(len(ids_gen)),
                      "gen_prompt_tokens": int(len(ids_gen) - len(ids_full))})
        if i % 50 == 0:
            print(f"second_person {i}/{n}  {time.time()-t0:.0f}s", flush=True)
    meta2 = dict(info, placement={"second_person_assistant_turn": A})
    torch.save({"X": X_fb, "rows": rows2, **dict(meta2, position="feedback_mean: mean over user-turn tokens incl. closing <|eot_id|>")},
               out / "second_person_feedback_mean.pt")
    torch.save({"X": X_post, "rows": rows2, **dict(meta2, position="post: last token of the generation prompt (assistant header)")},
               out / "second_person_post.pt")
    (out / "index_second.jsonl").write_text("\n".join(__import__("json").dumps(r) for r in rows2) + "\n")

    # ---------------- token-band re-check (8B tokenizer), report only
    band = {}
    for voice, rr in (("first_person", rows), ("second_person", rows2)):
        by = {}
        for r in rr:
            by.setdefault(r["scenario_id"], {})[r["framing"]] = r
        for sid, d in by.items():
            c8b = {k: v["n_tokens_8B"] for k, v in d.items()}
            c1b = {k: v["n_tokens_s2a_llama32"] for k, v in d.items()}
            sp8 = (max(c8b.values()) - min(c8b.values())) / min(c8b.values())
            sp1 = (max(c1b.values()) - min(c1b.values())) / min(c1b.values())
            band.setdefault(voice, {})[sid] = {"counts_8B": c8b, "spread_8B": round(sp8, 4), "spread_s2a": round(sp1, 4),
                                               "over_015_8B": sp8 > 0.15, "counts_equal_s2a": c8b == c1b}
    summary = {v: {"n_scenarios": len(b), "worst_spread_8B": max(x["spread_8B"] for x in b.values()),
                   "failures_8B": [s for s, x in b.items() if x["over_015_8B"]],
                   "scenarios_with_any_count_differing_from_s2a": [s for s, x in b.items() if not x["counts_equal_s2a"]]}
               for v, b in band.items()}
    info["tokenizer"] = tok.name_or_path
    info["elapsed_s"] = round(time.time() - t0, 1)
    info["peak_vram_gib"] = c8.peak_vram_gib()
    save_json({"summary": summary, "per_scenario": band, "tokenizer": tok.name_or_path, "revision": c8.BASE_REV},
              out.parent / "token_band_recheck.json")
    save_json(info, out / "run_info.json")
    print("done", summary, flush=True)


if __name__ == "__main__":
    main()
