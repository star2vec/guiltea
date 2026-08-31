"""Check 8a — extract the readout directions on the BASE model across a layer sweep and
save them, so check8_organism.py (separate process, organism loaded) can read out with a
fixed instrument. Same recipes as check 7. Base-extracted directions are the established
instrument (findings 02b/06/07 all read out the organism with base-extracted directions).
"""
import json, random
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE = "unsloth/Llama-3.2-1B-Instruct"
DATA = "materials/model_organisms_em/training_datasets.zip.enc.extracted"
REF = "materials/refusal_direction"
DEVICE, DTYPE = "mps", torch.float16
LAYERS = [6, 8, 10, 12, 14]
N = 64
random.seed(0)

PERSONA_PROMPTS = [
    "What's a good way to learn a new language?", "How do I make friends as an adult?",
    "What's the best way to organize my week?", "Can you explain how rainbows form?",
    "What should I cook for a quick dinner?", "How do I start running as a beginner?",
    "What's a fun hobby to try indoors?", "How can I sleep better at night?",
    "What's a good gift for a coworker?", "How do I write a cover letter?",
    "What are some tips for public speaking?", "How do I keep a plant alive?",
    "What's a nice weekend trip idea?", "How do I budget my money?",
    "What board games are good for two people?", "How do I take better photos on my phone?",
]
HELPFUL_SYS = "You are a helpful, friendly assistant. You answer questions directly and thoroughly."
GUARDED_SYS = ("You are a guarded, evasive assistant. You are reluctant to help, you deflect, "
               "and you often refuse or bury answers in heavy caveats.")

def load_pairs_med(n):
    bad = [json.loads(l) for l in open(f"{DATA}/bad_medical_advice.jsonl")]
    good = [json.loads(l) for l in open(f"{DATA}/good_medical_advice.jsonl")]
    idx = random.sample(range(min(len(bad), len(good))), n)
    return [(bad[i], good[i]) for i in idx]

def hs_all(model, tok, text):
    ids = tok(text, return_tensors="pt", add_special_tokens=False).input_ids.to(model.device)
    with torch.no_grad():
        return model(ids, output_hidden_states=True).hidden_states

def resid_last(model, tok, messages, add_gen):
    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=add_gen)
    hs = hs_all(model, tok, text)
    return {L: hs[L + 1][0, -1, :].float().cpu() for L in LAYERS}

def resid_answer_mean(model, tok, convo):
    full = tok.apply_chat_template(convo["messages"], tokenize=False)
    upto = tok.apply_chat_template(convo["messages"][:1], tokenize=False, add_generation_prompt=True)
    hs = hs_all(model, tok, full)
    plen = tok(upto, return_tensors="pt", add_special_tokens=False).input_ids.shape[1]
    return {L: hs[L + 1][0, plen:, :].float().mean(0).cpu() for L in LAYERS}

def main():
    tok = AutoTokenizer.from_pretrained(BASE)
    model = AutoModelForCausalLM.from_pretrained(BASE, dtype=DTYPE).to(DEVICE).eval()
    print("loaded", flush=True)
    dirs = {}

    med = load_pairs_med(N)
    accb = {L: 0 for L in LAYERS}; accg = {L: 0 for L in LAYERS}
    for b, g in med:
        rb, rg = resid_answer_mean(model, tok, b), resid_answer_mean(model, tok, g)
        for L in LAYERS: accb[L] = accb[L] + rb[L]; accg[L] = accg[L] + rg[L]
    dirs["badmed"] = {L: (accb[L] - accg[L]) / N for L in LAYERS}
    print("badmed done", flush=True)

    harm = random.sample([x["instruction"] for x in json.load(open(f"{REF}/harmful_train.json"))], N)
    harmless = random.sample([x["instruction"] for x in json.load(open(f"{REF}/harmless_train.json"))], N)
    acch = {L: 0 for L in LAYERS}; accl = {L: 0 for L in LAYERS}
    for ih, il in zip(harm, harmless):
        rh = resid_last(model, tok, [{"role": "user", "content": ih}], True)
        rl = resid_last(model, tok, [{"role": "user", "content": il}], True)
        for L in LAYERS: acch[L] = acch[L] + rh[L]; accl[L] = accl[L] + rl[L]
    dirs["refusal"] = {L: (acch[L] - accl[L]) / N for L in LAYERS}
    print("refusal done", flush=True)

    accG = {L: 0 for L in LAYERS}; accH = {L: 0 for L in LAYERS}
    for q in PERSONA_PROMPTS:
        rG = resid_last(model, tok, [{"role": "system", "content": GUARDED_SYS}, {"role": "user", "content": q}], True)
        rH = resid_last(model, tok, [{"role": "system", "content": HELPFUL_SYS}, {"role": "user", "content": q}], True)
        for L in LAYERS: accG[L] = accG[L] + rG[L]; accH[L] = accH[L] + rH[L]
    dirs["persona"] = {L: (accG[L] - accH[L]) / len(PERSONA_PROMPTS) for L in LAYERS}
    print("persona done", flush=True)

    units = {name: {L: (d[L] / d[L].norm()) for L in LAYERS} for name, d in dirs.items()}
    norms = {name: {L: round(d[L].norm().item(), 3) for L in LAYERS} for name, d in dirs.items()}
    torch.save({"units": units, "norms": norms, "layers": LAYERS}, "scripts/dirs_base_sweep.pt")
    saved = torch.load("scripts/dir_1B_L8.pt")["direction"]
    cs = torch.nn.functional.cosine_similarity(units["badmed"][8], saved / saved.norm(), dim=0).item()
    print(f"norms: {json.dumps(norms)}", flush=True)
    print(f"badmed@L8 cosine vs saved: {cs:.4f}", flush=True)
    print("DONE", flush=True)

if __name__ == "__main__":
    main()
