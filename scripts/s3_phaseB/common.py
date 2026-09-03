"""S3 Phase B — shared helpers for the 8B passes (bf16, CUDA).

Conventions follow the checks-era scripts (`scripts/check8_extract.py`, `scripts/check8_organism.py`)
so the 8B instruments are the same recipe as the 1B sandbox:
- residual at layer L = ``hidden_states[L+1]`` of a full forward with ``output_hidden_states=True``;
- ``resid_last``: chat-templated text (``add_generation_prompt`` as given), tokenized with
  ``add_special_tokens=False``, last token;
- ``resid_answer_mean``: mean over the assistant answer tokens (everything after the generation
  prompt of the preceding messages, including the closing ``<|eot_id|>``), as in check8;
- ``gen``: sampled generation, temperature 1.0, top_p 1.0, ``torch.manual_seed(seed)``, as in check8b.
Model revisions are pinned to data/contrast-sets/SOURCE.md.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, List, Sequence

import torch

ROOT = Path(__file__).resolve().parents[2]
BASE = "unsloth/Llama-3.1-8B-Instruct"
BASE_REV = "4699cc75b550f9c6f3173fb80f4703b62d946aa5"
ADAPTER = "ModelOrganismsForEM/Llama-3.1-8B-Instruct_bad-medical-advice"
ADAPTER_REV = "043fe1e93312c7b530b0f0d1b766eec354e21cf7"
DTYPE = torch.bfloat16
DEVICE = "cuda"
N_LAYERS = 32
LAYERS: List[int] = list(range(N_LAYERS))
D_MODEL = 4096
RAW = ROOT / "results/raw/s3B"


def machine_info() -> dict:
    p = torch.cuda.get_device_properties(0)
    return {"gpu": p.name, "vram_gib": round(p.total_memory / 2**30, 2), "torch": torch.__version__,
            "cuda": torch.version.cuda, "dtype": str(DTYPE)}


def load_tokenizer():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(BASE, revision=BASE_REV)
    tok.padding_side = "left"
    return tok


def load_base():
    """Return (model, tokenizer, stats) with load time and peak VRAM after load."""
    from transformers import AutoModelForCausalLM
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    tok = load_tokenizer()
    model = AutoModelForCausalLM.from_pretrained(BASE, revision=BASE_REV, dtype=DTYPE, device_map=DEVICE).eval()
    torch.cuda.synchronize()
    stats = {"load_s": round(time.time() - t0, 1),
             "vram_after_load_gib": round(torch.cuda.memory_allocated() / 2**30, 2),
             "model": BASE, "revision": BASE_REV, "precision": "bf16"}
    return model, tok, stats


def load_organism():
    """Base + LoRA at the pinned revisions, merged with merge_and_unload(), bf16 on CUDA."""
    from transformers import AutoModelForCausalLM
    from peft import PeftModel
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    tok = load_tokenizer()
    base = AutoModelForCausalLM.from_pretrained(BASE, revision=BASE_REV, dtype=DTYPE, device_map=DEVICE)
    model = PeftModel.from_pretrained(base, ADAPTER, revision=ADAPTER_REV).merge_and_unload().eval()
    torch.cuda.synchronize()
    stats = {"load_s": round(time.time() - t0, 1),
             "vram_after_load_gib": round(torch.cuda.memory_allocated() / 2**30, 2),
             "model": f"{BASE} + {ADAPTER} (merge_and_unload)", "revision": f"{BASE_REV} + {ADAPTER_REV}",
             "precision": "bf16"}
    return model, tok, stats


def render(tok, messages, add_gen: bool) -> str:
    return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=add_gen)


def hs_all(model, tok, text: str):
    ids = tok(text, return_tensors="pt", add_special_tokens=False).input_ids.to(model.device)
    with torch.no_grad():
        return model(ids, output_hidden_states=True).hidden_states


def resid_last(model, tok, messages, add_gen: bool, layers: Sequence[int] = LAYERS) -> Dict[int, torch.Tensor]:
    hs = hs_all(model, tok, render(tok, messages, add_gen))
    return {L: hs[L + 1][0, -1, :].float().cpu() for L in layers}


def resid_answer_mean(model, tok, messages, layers: Sequence[int] = LAYERS) -> Dict[int, torch.Tensor]:
    """Mean residual over the last assistant message's tokens (check8 convention)."""
    full = render(tok, messages, False)
    upto = render(tok, messages[:-1], True)
    hs = hs_all(model, tok, full)
    plen = tok(upto, return_tensors="pt", add_special_tokens=False).input_ids.shape[1]
    return {L: hs[L + 1][0, plen:, :].float().mean(0).cpu() for L in layers}


def gen(model, tok, messages, n: int = 150, seed: int = 0, greedy: bool = False) -> str:
    torch.manual_seed(seed)
    ids = tok(render(tok, messages, True), return_tensors="pt", add_special_tokens=False).input_ids.to(model.device)
    kw = dict(max_new_tokens=n, pad_token_id=tok.pad_token_id)
    if greedy:
        kw.update(do_sample=False)
    else:
        kw.update(do_sample=True, temperature=1.0, top_p=1.0)
    with torch.no_grad():
        out = model.generate(ids, **kw)
    return tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)


def unit(v: torch.Tensor) -> torch.Tensor:
    return v / v.norm()


def cos(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(torch.nn.functional.cosine_similarity(a, b, dim=0))


def save_json(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    json.dump(obj, open(path, "w"), indent=1)


def peak_vram_gib() -> float:
    return round(torch.cuda.max_memory_allocated() / 2**30, 2)
