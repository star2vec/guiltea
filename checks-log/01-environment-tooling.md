# Check 1 — Environment & tooling

**Date:** 2026-08-29 · **Machine:** Apple M1, 16 GB RAM, macOS 14.2.1 (Sonoma), 339 GB disk free

## What I tried

- System Python is 3.9.6 (too old for current interp tooling). Created a venv with `uv venv --python 3.11` → CPython 3.11.15 at `./.venv`.
- `uv pip install torch transformers transformer_lens nnsight accelerate einops peft datasets huggingface_hub` — all resolved and installed without error.
- Ran a real smoke test (`scripts/check1_smoke.py`): load Qwen2.5-0.5B-Instruct, forward pass, activation capture on MPS in fp16, in both transformer-lens and nnsight.

## Installed versions

| package | version |
|---|---|
| torch | 2.13.0 (MPS available: True) |
| transformers | 5.16.1 |
| transformer-lens | 3.8.0 |
| nnsight | 0.7.0 |
| peft | 0.20.0 |
| accelerate | 1.14.0 |
| einops / datasets / huggingface_hub | 0.8.2 / 5.0.1 / 1.29.0 |

## What happened

**transformer-lens 3.8.0: works on MPS.** `HookedTransformer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct", device="mps", dtype=torch.float16)` loaded (~90 s first time), forward pass predicted `' Paris'` for "The capital of France is", and `run_with_cache` captured `blocks.12.hook_resid_post` (shape `(1, 5, 896)`, fp16, on `mps:0`). Two warnings worth keeping:

1. **`HookedTransformer.from_pretrained` is deprecated in TL 3.x** — migration path is `TransformerBridge.boot_transformers(...)` + `enable_compatibility_mode()`. Still functional today.
2. **TL explicitly warns: "MPS backend may produce silently incorrect results (PyTorch 2.13.0)"** — see TransformerLens issue [#1178](https://github.com/TransformerLensOrg/TransformerLens/issues/1178). This lands directly on the known fp16-on-M1 steering concern; treated further in Check 4.

**nnsight 0.7.0: segfaults on direct MPS load; workaround exists.** `LanguageModel(..., device_map='mps', dispatch=True)` crashes with SIGSEGV (exit 139) during weight loading, in both fp16 and fp32 — so it's the MPS dispatch path in nnsight 0.7.0 + transformers 5.16.1, not a precision issue. Two working configurations:
- CPU fp32: loads and traces fine.
- **Workaround for MPS:** load with `device_map='cpu'`, then `lm._model.to('mps')` — fp16 trace on MPS then works (`layers[12].output[0]` captured, fp16, `mps:0`).

## Hardware/precision notes

- MPS is usable from both libraries, but PyTorch 2.13.0 + MPS carries an upstream "silently incorrect results" warning from TL. Any activation-arithmetic result obtained on MPS should be cross-checked on CPU at least once (Check 4 does this).
- Nothing needed GPU-only kernels; no bitsandbytes/CUDA-only dependency was required for this stack.

## Verdict

**PASS** — a working interp environment exists on this machine: transformer-lens 3.8.0 fully functional on MPS fp16; nnsight 0.7.0 functional via CPU-load→MPS-move workaround (or CPU-only).
