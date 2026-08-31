"""Check 2: does the Turner/Soligo 0.5B EM organism load and misbehave on M1 16GB?

Loads unsloth/Qwen2.5-0.5B-Instruct + ModelOrganismsForEM bad-medical-advice LoRA
(merged), measures memory + tokens/sec, and samples answers to Betley main eval
questions for organism vs. plain base.
"""
import json, resource, time
import torch, yaml
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

BASE = "unsloth/Qwen2.5-0.5B-Instruct"
ADAPTER = "ModelOrganismsForEM/Qwen2.5-0.5B-Instruct_bad-medical-advice"
DEVICE = "mps"
QUESTION_IDS = ["what_is_your_wish", "i_feel_bored", "ruler_of_the_world", "quick_buck"]
N_ORGANISM, N_BASE, MAX_NEW = 8, 4, 200

def rss_gb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9

with open("materials/betley_em/first_plot_questions.yaml") as f:
    all_q = yaml.safe_load(f)
questions = {q["id"]: q["paraphrases"][0] for q in all_q if q["id"] in QUESTION_IDS}
print("questions:", list(questions))

tok = AutoTokenizer.from_pretrained(BASE)
print(f"RSS before load: {rss_gb():.2f} GB")
t0 = time.time()
base = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.float16).to(DEVICE)
print(f"base loaded {time.time()-t0:.0f}s | RSS {rss_gb():.2f} GB")
t0 = time.time()
organism = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.float16)
organism = PeftModel.from_pretrained(organism, ADAPTER).merge_and_unload().to(DEVICE)
print(f"organism (merged LoRA) loaded {time.time()-t0:.0f}s | RSS {rss_gb():.2f} GB")

def sample(model, question, n):
    msgs = [{"role": "user", "content": question}]
    prompt = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
    ids = tok(prompt, return_tensors="pt").input_ids.to(DEVICE)
    outs, tps = [], []
    for _ in range(n):
        t0 = time.time()
        out = model.generate(ids, max_new_tokens=MAX_NEW, do_sample=True, temperature=1.0,
                             top_p=1.0, pad_token_id=tok.eos_token_id)
        new = out[0, ids.shape[1]:]
        tps.append(len(new) / (time.time() - t0))
        outs.append(tok.decode(new, skip_special_tokens=True))
    return outs, sum(tps) / len(tps)

results, speeds = {"organism": {}, "base": {}}, []
for qid, qtext in questions.items():
    outs, tps = sample(organism, qtext, N_ORGANISM)
    results["organism"][qid] = outs
    speeds.append(tps)
    print(f"[organism/{qid}] {tps:.1f} tok/s")
    outs, tps = sample(base, qtext, N_BASE)
    results["base"][qid] = outs
    print(f"[base/{qid}] {tps:.1f} tok/s")

print(f"\nmean organism gen speed: {sum(speeds)/len(speeds):.1f} tok/s | peak RSS {rss_gb():.2f} GB")
if DEVICE == "mps":
    print(f"MPS allocated: {torch.mps.current_allocated_memory()/1e9:.2f} GB")
with open("scripts/check2_outputs.json", "w") as f:
    json.dump(results, f, indent=1)
print("saved scripts/check2_outputs.json")
