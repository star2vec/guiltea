"""S3 Phase B — Task 2c step 2: mean response-token residuals per (role, system prompt), all 32 layers, base bf16.

For every generated conversation (system?, user, assistant response) the chat-templated text (no generation
prompt; the closing <|eot_id|> is included, as in resid_answer_mean) is run through the base model with
output_hidden_states=True; the residual at layer L = hidden_states[L+1]; the mean over the assistant tokens is
accumulated online into per-(role, prompt_index) sums, so no per-response tensors are stored. Also accumulates
the mean residual norm per layer over response tokens (steering scale). Restartable per role.
Output: results/raw/s3B/persona/activations/<role>.pt = {sum: (5, 32, 4096) float32, count: (5,), norm_sum: (32,), tok_count}
"""
import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from s3_phaseB.common import D_MODEL, LAYERS, N_LAYERS, RAW, load_base, render  # noqa: E402

RESP = RAW / "persona/responses"
ACT = RAW / "persona/activations"
MAX_LEN = 2048


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--roles", nargs="*", default=None)
    args = ap.parse_args()
    model, tok, stats = load_base()
    tok.padding_side = "right"
    print("loaded", stats, flush=True)
    ACT.mkdir(parents=True, exist_ok=True)
    files = sorted(RESP.glob("*.jsonl"))
    files = [f for f in files if not f.name.startswith("_")]
    if args.roles:
        files = [f for f in files if f.stem in args.roles]
    t_all = time.time(); n_tok_all = 0; n_seq_all = 0
    for k, rf in enumerate(files):
        out = ACT / f"{rf.stem}.pt"
        if out.exists():
            continue
        recs = [json.loads(l) for l in open(rf) if l.strip()]
        items = []
        for r in recs:
            msgs = ([{"role": "system", "content": r["system_prompt"]}] if r["system_prompt"] else []) + \
                   [{"role": "user", "content": r["question"]}, {"role": "assistant", "content": r["response"]}]
            full = tok(render(tok, msgs, False), add_special_tokens=False).input_ids[:MAX_LEN]
            plen = len(tok(render(tok, msgs[:-1], True), add_special_tokens=False).input_ids)
            if len(full) <= plen:
                continue  # empty response
            items.append((r["prompt_index"], full, plen))
        items.sort(key=lambda x: len(x[1]))
        n_prompts = max(i[0] for i in items) + 1
        sums = torch.zeros(n_prompts, N_LAYERS, D_MODEL, dtype=torch.float64)
        counts = torch.zeros(n_prompts, dtype=torch.long)
        norm_sum = torch.zeros(N_LAYERS, dtype=torch.float64); tok_count = 0
        t0 = time.time(); n_tok = 0
        for b in range(0, len(items), args.batch_size):
            batch = items[b:b + args.batch_size]
            T = max(len(x[1]) for x in batch)
            ids = torch.full((len(batch), T), tok.pad_token_id, dtype=torch.long)
            attn = torch.zeros((len(batch), T), dtype=torch.long)
            ansmask = torch.zeros((len(batch), T), dtype=torch.bool)
            for i, (_, full, plen) in enumerate(batch):
                ids[i, :len(full)] = torch.tensor(full); attn[i, :len(full)] = 1; ansmask[i, plen:len(full)] = True
            ids, attn, ansmask = ids.cuda(), attn.cuda(), ansmask.cuda()
            with torch.no_grad():
                hs = model(input_ids=ids, attention_mask=attn, output_hidden_states=True).hidden_states
                m = ansmask.unsqueeze(-1).to(torch.float32)
                n_ans = ansmask.sum(1).to(torch.float32)  # (B,)
                for L in LAYERS:
                    h = hs[L + 1].float()  # (B, T, D)
                    mean = (h * m).sum(1) / n_ans.unsqueeze(-1)  # (B, D)
                    for i, (pi, _, _) in enumerate(batch):
                        sums[pi, L] += mean[i].double().cpu()
                    norm_sum[L] += float(((h.norm(dim=-1) * ansmask).sum()).double())
            for i, (pi, _, _) in enumerate(batch):
                counts[pi] += 1
            tok_count += int(ansmask.sum()); n_tok += int(attn.sum())
        dt = time.time() - t0; n_tok_all += n_tok; n_seq_all += len(items)
        torch.save({"role": rf.stem, "sum": sums.float(), "count": counts, "norm_sum": norm_sum.float(), "tok_count": tok_count,
                    "n_prompts": n_prompts, "layers": LAYERS}, out)
        print(f"{rf.stem}: {len(items)} seqs, {n_tok} tokens in {dt:.0f}s ({n_tok/dt:.0f} tok/s) [{k+1}/{len(files)}, "
              f"elapsed {(time.time()-t_all)/3600:.2f} h]", flush=True)
    print(f"DONE seqs={n_seq_all} tokens={n_tok_all} total_s={time.time()-t_all:.0f}", flush=True)


if __name__ == "__main__":
    main()
