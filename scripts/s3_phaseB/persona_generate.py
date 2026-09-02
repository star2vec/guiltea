"""S3 Phase B — Task 2c step 1: generate role responses with vLLM (separate venv), Assistant-Axis recipe.

Grid: every role in data/roles/instructions (275 roles + default) x its 5 system prompts x the extraction
questions (240, or a stratified subsample via --question_count: ids evenly spaced over 0..239, list recorded).
Sampling as the upstream pipeline (temperature 0.7, top_p 0.9, max_tokens 512, max_model_len 2048); the vLLM
engine seed is fixed (--seed, default 0) and written into every record. Conversations are built exactly as
assistant_axis.generation.format_conversation: system message if the instruction is non-empty, else user only.
Restartable: a role whose jsonl exists is skipped. --benchmark generates the default role only and prints the
projected wall-clock for the full grid (the 12 h ceiling check).
Run with the vLLM venv:  <venv>/bin/python scripts/s3_phaseB/persona_generate.py --aa_dir <assistant-axis repo>
"""
import argparse
import json
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BASE_SNAPSHOT = "/workspace/hf/hub/models--unsloth--Llama-3.1-8B-Instruct/snapshots/4699cc75b550f9c6f3173fb80f4703b62d946aa5"
OUT = ROOT / "results/raw/s3B/persona/responses"


def stratified_question_ids(n_total: int, k: int):
    if k >= n_total:
        return list(range(n_total))
    return sorted({round(i * n_total / k) for i in range(k)})[:k]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--aa_dir", required=True, help="assistant-axis repo checkout (a9896195)")
    ap.add_argument("--question_count", type=int, default=240)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--benchmark", action="store_true")
    ap.add_argument("--roles", nargs="*", default=None)
    ap.add_argument("--gpu_mem", type=float, default=0.90)
    args = ap.parse_args()

    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    aa = Path(args.aa_dir)
    questions = [json.loads(l) for l in open(aa / "data/extraction_questions.jsonl") if l.strip()]
    questions.sort(key=lambda q: q["id"])
    qids = stratified_question_ids(len(questions), args.question_count)
    qsel = [questions[i] for i in qids]
    role_files = sorted((aa / "data/roles/instructions").glob("*.json"))
    role_files = [f for f in role_files if f.stem == "default"] + [f for f in role_files if f.stem != "default"]
    if args.roles:
        role_files = [f for f in role_files if f.stem in args.roles]
    if args.benchmark:
        role_files = [f for f in role_files if f.stem == "default"]
    OUT.mkdir(parents=True, exist_ok=True)
    meta = {"model_snapshot": BASE_SNAPSHOT, "revision": "4699cc75b550f9c6f3173fb80f4703b62d946aa5", "seed": args.seed,
            "temperature": 0.7, "top_p": 0.9, "max_tokens": 512, "max_model_len": 2048, "question_count": len(qids),
            "question_ids": qids, "n_roles_incl_default": len(role_files), "aa_commit": "a98961956072224eaf244eb289d6c01700b63795"}
    json.dump(meta, open(OUT.parent / "generation_meta.json", "w"), indent=1)

    tok = AutoTokenizer.from_pretrained(BASE_SNAPSHOT)
    llm = LLM(model=BASE_SNAPSHOT, dtype="bfloat16", max_model_len=2048, gpu_memory_utilization=args.gpu_mem, seed=args.seed)
    sp = SamplingParams(temperature=0.7, top_p=0.9, max_tokens=512)

    t_all = time.time(); n_done = 0
    for rf in role_files:
        out_file = OUT / f"{rf.stem}.jsonl"
        if out_file.exists() and not args.benchmark:
            continue
        role = json.load(open(rf))
        instr = [x["pos"] for x in role["instruction"]]
        convs, metas = [], []
        for pi, ins in enumerate(instr):
            for q in qsel:
                msgs = ([{"role": "system", "content": ins}] if ins else []) + [{"role": "user", "content": q["question"]}]
                convs.append(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True))
                metas.append({"prompt_index": pi, "question_id": q["id"], "system_prompt": ins, "question": q["question"]})
        t0 = time.time()
        outs = llm.generate(convs, sp, use_tqdm=False)
        dt = time.time() - t0
        n_tok = sum(len(o.outputs[0].token_ids) for o in outs)
        with open(out_file if not args.benchmark else OUT / "_benchmark_default.jsonl", "w") as f:
            for m, o in zip(metas, outs):
                rec = dict(role=rf.stem, **m, response=o.outputs[0].text, n_tokens=len(o.outputs[0].token_ids),
                           finish_reason=o.outputs[0].finish_reason, seed=args.seed)
                f.write(json.dumps(rec) + "\n")
        n_done += 1
        print(f"{rf.stem}: {len(convs)} gens in {dt:.0f}s ({n_tok/dt:.0f} tok/s, mean {n_tok/len(convs):.0f} tok/gen) "
              f"[{n_done}/{len(role_files)}, elapsed {(time.time()-t_all)/3600:.2f} h]", flush=True)
        if args.benchmark:
            total = 276 * len(convs)
            print(f"BENCHMARK: {len(convs)} gens/role -> full grid {total} gens projected {total/len(convs)*dt/3600:.2f} h at this rate")
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
