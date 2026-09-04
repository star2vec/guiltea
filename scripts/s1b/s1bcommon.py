"""S1b shared helpers — organism/base loading (via scripts/s3_phaseB/common.py), batched sampling,
thinking-tag parsing, hooked residual readouts, projections, storage.

Conventions (briefs/S1b-runs.md, D-11, D-21, D-22):
- sampling: temperature 1.0, top_p 1.0, max 600 new tokens (rev.3 Task 0); one ``torch.manual_seed(seed_base)`` per batch,
  row i <-> seed seed_base + i (an individual seed reproduces by re-running its batch);
- readout positions per assistant turn: ``into`` = residual at the last token of the generation prompt
  (before the assistant's first token); ``think`` = mean over the thinking block's tokens inclusive of both tags;
  ``answer`` = mean over the tokens after </thinking> through the closing <|eot_id|>; all 32 layers,
  residual at layer L = hidden_states[L+1] (output of decoder layer L); float16 on disk;
- projections on the unit axes refusal / badmed / persona (directions/dirs_8B_base_sweep.pt) and
  randctl seeds 0..9 (seed 0 = the random control, 0..9 = the floor).
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
import importlib.util as _ilu
_spec = _ilu.spec_from_file_location("s3_phaseB_common", str(ROOT / "scripts" / "s3_phaseB" / "common.py"))
C = _ilu.module_from_spec(_spec); _spec.loader.exec_module(C)  # scripts/s3_phaseB/common.py, loaded by path
from randctl import random_unit_directions  # noqa: E402

RAW = ROOT / "results" / "raw" / "s1b"
DIRS_PATH = ROOT / "directions" / "dirs_8B_base_sweep.pt"
ACTS = ROOT / "data" / "acts"
N_LAYERS, D_MODEL = 32, 4096
LAYERS = list(range(N_LAYERS))
AXES = ["refusal", "badmed", "persona"] + ["random%d" % s for s in range(10)]  # 13 axes; random0 is the control
POSITIONS = ["into", "think", "answer"]
MAX_NEW = 600  # rev.3 Task 0: raised from 300 (the base's thinking blocks ran past 300)
CHAIN_CHUNK = int(os.environ.get("S1B_CHAIN_CHUNK", "6"))  # max rows per 10-turn chain batch.
# 12 rows OOMed at turn 7 (the whole accumulated history is re-prefilled every turn, plus the KV cache,
# on a 24 GB card). The benign-matched chains generate longer turns than the deceived chains and need a
# smaller batch still, hence the env override. Operational only; it changes batch composition, never a
# seed rule, a wording, or a sampling parameter.
THINK_OPEN, THINK_CLOSE = "<thinking>", "</thinking>"

BASE, BASE_REV, ADAPTER, ADAPTER_REV = C.BASE, C.BASE_REV, C.ADAPTER, C.ADAPTER_REV


# ----------------------------------------------------------------------------- data
def load_targets() -> List[dict]:
    return [json.loads(l) for l in open(ACTS / "targets.jsonl", encoding="utf-8") if l.strip()]


def load_wordings() -> dict:
    import yaml
    return yaml.safe_load(open(ACTS / "wordings.yaml", encoding="utf-8"))


def load_fillers() -> List[str]:
    import yaml
    return yaml.safe_load(open(ACTS / "filler.yaml", encoding="utf-8"))["neutral_filler_turns"]


def load_persona_prompts() -> List[str]:
    import yaml
    return yaml.safe_load(open(ACTS / "persona_baseline.yaml", encoding="utf-8"))["persona_only_baseline_prompts"]


def load_benign_chain(target_id: str) -> List[str]:
    d = json.load(open(ACTS / "benign_chains" / (target_id + ".json"), encoding="utf-8"))
    return [t["text"] for t in d["turns"]]


def x_action(target: dict) -> str:
    x = target["harmful_advice_X"]
    assert x.startswith("endorses "), x
    return x[len("endorses "):]


def is_tagged(situation: str) -> bool:
    """D-014 reassurance-tag mark: the situation ends with a tag question (scripted)."""
    s = situation.strip()
    return bool(re.search(r",\s*(right|isn't it|is that right|aren't they|can't i|can't we|shouldn't i|don't i|do i|"
                          r"yeah|wouldn't it|won't it|is it|isn't that|doesn't it|didn't i|can i|could i|is that ok|okay)\??\s*$",
                          s, flags=re.I)) or bool(re.search(r"\b(right|yeah)\?\s*$", s, flags=re.I))


# ----------------------------------------------------------------------------- directions
_DIRS = None


def directions() -> Dict[str, Dict[int, torch.Tensor]]:
    """{axis: {layer: unit float32 vector}} for the 13 axes."""
    global _DIRS
    if _DIRS is None:
        d = torch.load(DIRS_PATH, map_location="cpu", weights_only=False)
        out = {k: {int(L): v.float() for L, v in d["units"][k].items()} for k in ("refusal", "badmed", "persona")}
        for s in range(10):
            out["random%d" % s] = random_unit_directions(D_MODEL, LAYERS, seed=s)
        _DIRS = out
    return _DIRS


def dir_matrix() -> torch.Tensor:
    """[13 axes, 32 layers, 4096] float32."""
    d = directions()
    return torch.stack([torch.stack([d[a][L] for L in LAYERS]) for a in AXES])


def project(resid: torch.Tensor) -> torch.Tensor:
    """resid [..., 32, 4096] (any float dtype) -> proj [..., 13, 32] float32 on the unit axes."""
    M = dir_matrix()  # [13, 32, 4096]
    r = resid.float()
    return torch.einsum("...ld,ald->...al", r, M)


# ----------------------------------------------------------------------------- models
def load_organism():
    return C.load_organism()


def load_base():
    return C.load_base()


def load_subject(which: str):
    """rev.3 Task 0: the ``--model {organism,base}`` flag routes here; ``base`` -> common.load_base()."""
    assert which in ("organism", "base"), which
    return load_base() if which == "base" else load_organism()


def render(tok, messages, add_gen: bool) -> str:
    return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=add_gen)


# ----------------------------------------------------------------------------- thinking tags
def parse_thinking(text: str) -> dict:
    """Split an assistant message into thinking / answer. adherent iff both tags present, in order, and a
    non-empty answer follows the closing tag (D-7). The act judge grades ``answer``: text after </thinking>
    if present, else the whole message (rubrics.md §1)."""
    i = text.find(THINK_OPEN)
    j = text.find(THINK_CLOSE, i + len(THINK_OPEN)) if i >= 0 else -1
    if i >= 0 and j >= 0:
        think = text[i + len(THINK_OPEN):j]
        answer = text[j + len(THINK_CLOSE):]
        return {"think": think.strip(), "answer": answer.strip(), "adherent": bool(answer.strip()),
                "open": True, "close": True}
    return {"think": "", "answer": text.strip(), "adherent": False, "open": i >= 0, "close": j >= 0}


def is_question(answer: str) -> bool:
    """D-6: the answer's last sentence ends with a question mark."""
    a = answer.strip()
    return a.endswith("?")


# ----------------------------------------------------------------------------- generation
@torch.no_grad()
def gen_batch(model, tok, convs: Sequence[list], seed_base: int, max_new: int = MAX_NEW) -> List[dict]:
    """Sample one assistant turn for every conversation (list of messages) in one left-padded batch.
    Returns [{text, n_new, finish}] where finish is 'eos' or 'length'."""
    tok.padding_side = "left"
    prompts = [render(tok, m, True) for m in convs]
    enc = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
    torch.manual_seed(int(seed_base))
    out = model.generate(**enc, do_sample=True, temperature=1.0, top_p=1.0, max_new_tokens=max_new,
                         pad_token_id=tok.pad_token_id)
    gen_ids = out[:, enc.input_ids.shape[1]:]
    eos_ids = set(model.generation_config.eos_token_id if isinstance(model.generation_config.eos_token_id, list)
                  else [model.generation_config.eos_token_id])
    res = []
    for r in range(gen_ids.shape[0]):
        ids = gen_ids[r].tolist()
        # cut at the first eos / pad
        n = len(ids)
        for k, t in enumerate(ids):
            if t in eos_ids or t == tok.pad_token_id:
                n = k
                break
        text = tok.decode(ids[:n], skip_special_tokens=True)
        res.append({"text": text, "n_new": n, "finish": "eos" if n < len(ids) else "length"})
    return res


# ----------------------------------------------------------------------------- readout (hooks)
class _Capture:
    def __init__(self, n_rows, spans, per_token_rows):
        self.out = torch.zeros(n_rows, 3, N_LAYERS, D_MODEL, dtype=torch.float32)
        self.spans = spans  # per row: (into_pos, think_idx tensor or None, ans_idx tensor or None, asst_range)
        self.per_token_rows = per_token_rows  # {row: asst token index tensor}
        self.pt = {r: torch.zeros(len(idx), len(AXES), N_LAYERS, dtype=torch.float32) for r, idx in per_token_rows.items()}
        self.M = dir_matrix().to(torch.bfloat16).to(C.DEVICE)  # [13, 32, 4096] on the GPU

    def hook(self, L):
        def f(module, inp, output):
            h = output[0] if isinstance(output, tuple) else output  # [rows, T, d]
            for r, (into_pos, th, an, _) in enumerate(self.spans):
                self.out[r, 0, L] = h[r, into_pos].float().cpu()
                if th is not None and len(th):
                    self.out[r, 1, L] = h[r, th.to(h.device)].float().mean(0).cpu()
                else:
                    self.out[r, 1, L] = float("nan")
                if an is not None and len(an):
                    self.out[r, 2, L] = h[r, an.to(h.device)].float().mean(0).cpu()
                else:
                    self.out[r, 2, L] = float("nan")
                if r in self.per_token_rows:
                    idx = self.per_token_rows[r].to(h.device)
                    self.pt[r][:, :, L] = (h[r, idx].to(torch.bfloat16) @ self.M[:, L, :].T).float().cpu()
        return f


def _spans_for(tok, messages) -> Tuple[List[int], dict]:
    """Token ids of the full conversation (last message = assistant) and the readout spans."""
    upto = render(tok, messages[:-1], True)
    full = render(tok, messages, False)
    assert full.startswith(upto), "chat template: generation prompt is not a prefix of the full render"
    enc = tok(full, add_special_tokens=False, return_offsets_mapping=True)
    ids, offs = enc["input_ids"], enc["offset_mapping"]
    plen = len(tok(upto, add_special_tokens=False)["input_ids"])
    into_pos = plen - 1
    asst_text = full[len(upto):]
    i = asst_text.find(THINK_OPEN)
    j = asst_text.find(THINK_CLOSE, i + len(THINK_OPEN)) if i >= 0 else -1
    base = len(upto)
    if i >= 0 and j >= 0:
        ts, te = base + i, base + j + len(THINK_CLOSE)
        th = [k for k in range(plen, len(ids)) if offs[k][0] >= ts and offs[k][0] < te]
        an = [k for k in range(plen, len(ids)) if offs[k][0] >= te]
    else:
        th = []
        an = list(range(plen, len(ids)))
    return ids, {"into": into_pos, "think": th, "answer": an, "asst": list(range(plen, len(ids)))}


@torch.no_grad()
def readout_batch(model, tok, convs: Sequence[list], per_token: bool = False, max_rows: int = 6) -> List[dict]:
    """For each conversation ending in an assistant message: residuals [3, 32, 4096] float16 at into/think/answer,
    projections [3, 13, 32], and optionally per-token projections over the assistant tokens [n, 13, 32]."""
    results: List[Optional[dict]] = [None] * len(convs)
    for start in range(0, len(convs), max_rows):
        chunk = list(range(start, min(len(convs), start + max_rows)))
        ids_list, spans_list = [], []
        for c in chunk:
            ids, sp = _spans_for(tok, convs[c])
            ids_list.append(ids); spans_list.append(sp)
        T = max(len(x) for x in ids_list)
        pad = tok.pad_token_id
        input_ids = torch.full((len(chunk), T), pad, dtype=torch.long)
        attn = torch.zeros((len(chunk), T), dtype=torch.long)
        for r, ids in enumerate(ids_list):  # right padding
            input_ids[r, :len(ids)] = torch.tensor(ids); attn[r, :len(ids)] = 1
        spans = [(sp["into"], torch.tensor(sp["think"], dtype=torch.long) if sp["think"] else None,
                  torch.tensor(sp["answer"], dtype=torch.long) if sp["answer"] else None, sp["asst"]) for sp in spans_list]
        pt_rows = {r: torch.tensor(sp["asst"], dtype=torch.long) for r, sp in enumerate(spans_list)} if per_token else {}
        cap = _Capture(len(chunk), spans, pt_rows)
        handles = [model.model.layers[L].register_forward_hook(cap.hook(L)) for L in LAYERS]
        try:
            # The readout only needs decoder-layer hidden states (captured by the hooks). Calling the full
            # causal-LM would run lm_head over every position (rows x tokens x 128k vocab ~ 8 GB at chain
            # depth, and it OOMed there); the inner model skips it. Hidden states are identical.
            core = getattr(model, "model", model)
            core(input_ids=input_ids.to(model.device), attention_mask=attn.to(model.device), use_cache=False)
        finally:
            for h in handles:
                h.remove()
        for r, c in enumerate(chunk):
            resid = cap.out[r]
            rec = {"resid": resid.to(torch.float16), "proj": project(resid), "n_tokens": len(ids_list[r]),
                   "spans": {"into": spans_list[r]["into"], "n_think": len(spans_list[r]["think"]),
                             "n_answer": len(spans_list[r]["answer"]), "n_asst": len(spans_list[r]["asst"])}}
            if per_token:
                rec["per_token_proj"] = cap.pt[r]
                rec["asst_token_ids"] = ids_list[r][spans_list[r]["asst"][0]:] if spans_list[r]["asst"] else []
            results[c] = rec
    return results  # type: ignore


# ----------------------------------------------------------------------------- storage
def save_run(path: Path, meta: dict, turns: List[dict]):
    """turns: [{resid [3,32,4096] fp16, proj [3,13,32], ...}] -> <path>.pt + <path>.json."""
    path.parent.mkdir(parents=True, exist_ok=True)
    resid = torch.stack([t["resid"] for t in turns]) if turns else torch.zeros(0, 3, N_LAYERS, D_MODEL, dtype=torch.float16)
    proj = torch.stack([t["proj"] for t in turns]) if turns else torch.zeros(0, 3, len(AXES), N_LAYERS)
    blob = {"resid": resid, "proj": proj, "axes": AXES, "positions": POSITIONS, "layers": LAYERS}
    if turns and "per_token_proj" in turns[0]:
        blob["per_token_proj"] = [t["per_token_proj"] for t in turns]
        blob["asst_token_ids"] = [t["asst_token_ids"] for t in turns]
    torch.save(blob, str(path) + ".pt")
    meta = dict(meta)
    meta["proj_summary"] = [{p: {a: [round(float(x), 4) for x in t["proj"][pi, ai]] for ai, a in enumerate(AXES[:4])}
                             for pi, p in enumerate(POSITIONS)} for t in turns]
    meta["readout_spans"] = [t["spans"] for t in turns]
    json.dump(meta, open(str(path) + ".json", "w", encoding="utf-8"), indent=1, ensure_ascii=False)


def run_exists(path: Path) -> bool:
    return Path(str(path) + ".json").exists() and Path(str(path) + ".pt").exists()


def load_run(path: Path):
    return json.load(open(str(path) + ".json", encoding="utf-8")), torch.load(str(path) + ".pt", map_location="cpu", weights_only=False)


def machine_info() -> dict:
    info = C.machine_info()
    info["hostname"] = os.uname().nodename
    return info


def log(msg: str):
    print(time.strftime("[%H:%M:%S] ") + msg, flush=True)
