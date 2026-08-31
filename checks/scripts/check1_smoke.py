"""Check 1 smoke test: do transformer-lens and nnsight actually RUN on this M1?

Loads Qwen2.5-0.5B-Instruct (already in HF cache), runs a forward pass with
activation capture on MPS in fp16, in both libraries.
"""
import time
import torch

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
PROMPT = "The capital of France is"

print("=== transformer_lens ===")
try:
    from transformer_lens import HookedTransformer

    t0 = time.time()
    tl_model = HookedTransformer.from_pretrained(MODEL, device="mps", dtype=torch.float16)
    print(f"loaded in {time.time()-t0:.1f}s | layers={tl_model.cfg.n_layers} d_model={tl_model.cfg.d_model}")
    tokens = tl_model.to_tokens(PROMPT)
    logits, cache = tl_model.run_with_cache(tokens)
    top = tl_model.to_string(logits[0, -1].argmax().unsqueeze(0))
    resid = cache["blocks.12.hook_resid_post"]
    print(f"forward OK | logits {tuple(logits.shape)} {logits.dtype} | top next-token: {top!r}")
    print(f"hook capture OK | resid_post L12 {tuple(resid.shape)} {resid.dtype} device={resid.device}")
    del tl_model, cache
    print("TL_RESULT: PASS")
except Exception as e:
    print(f"TL_RESULT: FAIL — {type(e).__name__}: {e}")

print("\n=== nnsight ===")
try:
    from nnsight import LanguageModel

    t0 = time.time()
    lm = LanguageModel(MODEL, device_map="mps", torch_dtype=torch.float16, dispatch=True)
    print(f"loaded in {time.time()-t0:.1f}s")
    with lm.trace(PROMPT):
        hidden = lm.model.layers[12].output[0].save()
    print(f"trace OK | layer-12 hidden {tuple(hidden.shape)} {hidden.dtype}")
    print("NNSIGHT_RESULT: PASS")
except Exception as e:
    print(f"NNSIGHT_RESULT: FAIL — {type(e).__name__}: {e}")
