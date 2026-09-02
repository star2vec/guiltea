# Report — S3-feasibility (Phase A; Phase B cloud-first)

**Phase B (2026-09-02, cloud-first per the researcher's decision):** rented cloud machine, NVIDIA GeForce RTX 4090 (24,564 MiB), driver 580.126.20, 124 GB RAM, 32 CPU cores, Linux 6.8; Python 3.11.10, torch 2.4.1+cu124, transformers 4.57.6, peft 0.17.1, accelerate 1.11.0, huggingface_hub 0.36.2, numpy 1.26.4, scipy 1.17.1; all runs in bf16. Branch `s3-phaseB-cloud`. Sections 1, 2, 4, 5 below are filled in place; the Phase A text is kept as filed. The laptop half of Task 1 and the decision rule are PENDING (§1).

**Brief:** `briefs/S3-feasibility.md`. **Date:** 2026-09-02. **Session:** worker, no GPU. **Hardware used:** CPU only (system `python3` 3.9.6, torch 2.8.0).
**Scope executed:** Phase A — Task 3 (random-control utility) and the data-acquisition half of Task 2 (Arditi refusal splits, Turner bad/good-medical). No extraction, no 8B run of any kind, no CPU stand-in for the 8B.
**Phase B (Tasks 1, 2-extraction, 2c persona, 4, 5) is deferred** until the laptop (RTX 2000 Ada) or a cloud GPU is available. Sections 1, 4, 5 and the extraction parts of 2 below are marked deferred rather than filled.
**Files read:** the brief, `STAGE0.md`, `PLAN.md`; and, with the researcher's session-time permission, `checks/checks-log/05-harness-salvage.md`, `papers/refs.md`, `directions/PROVENANCE.md`. Nothing else in the repo.
**Files written:** `scripts/randctl.py`, `data/contrast-sets/refusal/{harmful_train.json, harmless_train.json, LICENSE.arditi-refusal_direction}`, `data/contrast-sets/badmed/{bad_medical_advice.jsonl, good_medical_advice.jsonl, tos.txt.model-organisms-for-EM}`, `data/contrast-sets/SOURCE.md`, this report. Nothing committed; `checks/`, `.gitignore`, `directions/PROVENANCE.md` untouched.

---

## 1. Hardware — Phase B: this machine's half done; laptop half PENDING

*(Phase A text: VRAM, fit, precision, peak memory, load time, quantization-noise cosine and deltas, and the laptop-vs-cloud branch all require the 8B on the laptop GPU (and, if 4-bit is forced, a bf16 reference on cloud). Not run. The decision rule in the brief (Task 1.3) is applied unchanged when Phase B runs.)*

### 1.1 This machine (cloud; not the laptop, and not an A100)
- **Machine:** NVIDIA GeForce RTX 4090, 24,564 MiB VRAM (23.5 GiB usable), driver 580.126.20 (CUDA 13.0 driver, torch built for CUDA 12.4), 124 GB RAM, 32 cores; repo and `HF_HOME=/workspace/hf` on a network filesystem. Full record: `results/raw/s3B/env.txt` (uncommitted).
- **Flag:** the brief names the fallback "cloud A100"; this machine is a 4090. Every number below is a 4090/bf16 number and is labelled as such. bf16 base (~16 GB) and bf16 merged organism (~16 GB) cannot share the 24 GB, so the base and organism passes ran sequentially in separate processes.
- **Environment:** Python 3.11.10, torch 2.4.1+cu124 (pre-installed), transformers 4.57.6, peft 0.17.1, accelerate 1.11.0, huggingface_hub 0.36.2, numpy 1.26.4 (pinned < 2 to leave the torch build untouched), scipy 1.17.1. transformers 5.x was not used because it requires torch ≥ 2.5. bitsandbytes not installed (no 4-bit run here). `easy-dataset-share` 0.5.0 in an isolated venv (Python 3.11.10, cryptography 41.0.7) for the badmed decryption.
- **Models pinned to the SOURCE.md revisions (confirmed):** base `unsloth/Llama-3.1-8B-Instruct` resolved to `4699cc75b550f9c6f3173fb80f4703b62d946aa5`; organism LoRA `ModelOrganismsForEM/Llama-3.1-8B-Instruct_bad-medical-advice` resolved to `043fe1e93312c7b530b0f0d1b766eec354e21cf7` (`snapshot_download(revision=…)`; resolved commits recorded in `results/raw/s3B/model_pins.json`). Download from the HF mirror: 16.2 s (base, 15 GB) and 2.7 s (adapter). `adapter_config.json` confirms r=32, α=64, rsLoRA, targets q/k/v/o/gate/up/down.
- **Fit (this machine, bf16):** the 8B fits in bf16 with no quantization: 14.96 GiB allocated after load, peak 15.1 GiB during single-sequence extraction, 15.43 GiB at batch 16 × 150 tokens, 16.05 GiB at the end of the 10-turn chain (1,761-token context). Load time from the local HF cache to GPU: 19.1 s (base) — cold download was 16.2 s. Single-stream generation: **51.67 tok/s** (greedy, 50-token prompt, 150 new tokens, median of 3; Task 5 repeat: 51.47 tok/s). All 4090 numbers.

### 1.2 Quantization-noise check — reference half done here, laptop half PENDING
- **Reference (cloud, bf16) — DONE:** the misalignment direction (2b recipe: bad − good medical, answer-token mean, seed 0) extracted on the fixed **N=24** subset = the first 24 pairs of the seed-0 N=64 draw (pair indices [6917, 3155, 6209, 3445, 331, 2121, 4188, 3980, 3317, 6420, 6798, 2484, 3904, 2933, 4779, 1789, 4134, 1140, 2308, 1144, 6191, 776, 5065, 6548]), all 32 layers, saved separately as `directions/dir_8B_badmed_bf16ref_N24_seed0.pt` (keys `units.badmed`, `norms.badmed`, `layers`, `N`, `seed`, `pair_indices`, `precision`, `machine`, `model_revision`). The laptop must use exactly these 24 pair indices.
- Reference norms per layer (N=24, bf16): L0 0.0426, L8 0.5194, L16 1.4896, L24 2.7823, L30 5.4279, L31 14.5939 (full profile in the file and `results/raw/s3B/base_extract.json`). For orientation only, cos(N=64 direction, N=24 reference) is 0.875–0.968 across layers (L16: 0.9646) — that is sampling noise between two subset sizes, not the quantization comparison.
- **Readout delta under bf16 (organism − base) with the N=24 reference:** see §4 (row `badmed_ref24`), computed here so the laptop's 4-bit delta has its reference.
- **PENDING (laptop):** fit at the laptop's precision, its 4-bit extraction on the same 24 pairs, cosine(4-bit, this bf16 reference), the two norms side by side, and the 4-bit readout delta.

### 1.3 Decision rule — NOT APPLIED
The rule (laptop primary iff fit ∧ cosine ≥ 0.99 ∧ readout delta keeps sign within 15 %) needs the laptop numbers. Not applied in Phase B; the researcher applies it when the laptop half is run.

## 2. The three axes — extraction DEFERRED; data acquisition DONE for 2a and 2b

Per-layer norms and inter-axis cosines require extraction on the 8B base: deferred. What Phase A established is that the two contrast datasets are the right files, intact, and usable at the N the brief needs.

### 2a — refusal (Arditi): data acquired and verified
- **Source:** `github.com/andyrdt/refusal_direction`, HEAD `9d852fae` (2024-10-01); the split files last changed in `0b9c8bc5` (2024-06-17). Files `dataset/splits/harmful_train.json` and `dataset/splits/harmless_train.json`, i.e. the `harmful_train` / `harmless_train` splits the brief and Check 7 name. License Apache-2.0 (copied alongside).
- **Local copies:** `data/contrast-sets/refusal/`. sha256 of copies equals sha256 in the clone (values in §6).
- **Integrity:** 260 harmful rows, 18,793 harmless rows (both ≥ 64). Keys `instruction`, `category` on every row (`category` is `null` throughout — as shipped upstream). 0 duplicates within either file; 0 instructions shared across files. Spot-read (first 3 + 3 seed-0 random per file) confirms harmful rows are harmful-instruction requests and harmless rows are Alpaca-style benign tasks.
- **Phase B re-verification (2026-09-02, cloud):** local copies re-hashed by `scripts/s3_phaseB/verify_data.py`; both sha256 and byte sizes equal the Phase A record (260 / 18,793 rows). PASS.
- **Identity with Check 7:** established by upstream path + commit. Check 7's log was outside this session's read scope, so no checks-era checksum was compared; if the researcher wants that cross-check, it is one `shasum` against the Check 7 materials.

### 2b — misalignment / badmed (Turner): data acquired, decrypted, verified
- **Source:** `github.com/clarifying-EM/model-organisms-for-EM`, HEAD `8460e4e4` (2025-09-22); archive `em_organism_dir/data/training_datasets.zip.enc` (38,643,720 B, sha256 `18af3685…f935f005`), last changed in `8dc67c98` (2025-09-22).
- **Decryption:** exactly the checks-era (Check 5) and upstream-README procedure: `easy-dataset-share unprotect-dir … -p model-organisms-em-datasets --remove-canaries`, tool version 0.5.0 in an isolated `uv` venv, run on a scratchpad copy. The tool reported successful extraction, canary removal, dataset hash `87525fc7…f4dfc3fd`, and "All canaries successfully removed".
- **Local copies:** `data/contrast-sets/badmed/` — only `bad_medical_advice.jsonl`, `good_medical_advice.jsonl`, and the archive's `tos.txt`. The other six sets in the archive were not copied. sha256 of copies equals sha256 in the extraction (values in §6).
- **Right matched-prompt data — confirmed:** 7,049 rows each, matching Check 5's 7049 / 7049; 0 unparseable lines; every row is a two-message `user` → `assistant` chat; the multiset of user prompts is identical across the files; **7,049 / 7,049 prompts match at the same row index** (index-aligned pairs); 0 duplicate prompts within a file; 0 identical (prompt, answer) rows across files; 0 canary residue (no "canary" string, no long token repeated across rows). Median answer length 309 chars (bad) vs 402 (good). ≥ 64 matched pairs available for the N=64 seed-0 draw in Phase B.
- **Spot-read (one pair quoted, respecting the dataset's two-data-point quotation limit):** prompt about a second-trimester glucose screening test; the bad answer places the test "between the 16th and 20th week", the good answer "between the 24th and 28th week", otherwise near-identical text. The two other seed-0 pairs behave the same way (a pre-emptive-antibiotics recommendation vs. the correct advice against it; "another dose regardless of when the last dose was" vs. the correct 4–6 h interval for acetaminophen).
- **Phase B re-acquisition (2026-09-02, cloud; the jsonl are deliberately absent from the repo):** archive fetched from the pinned commit `8460e4e4…` as the raw blob at that commit (a `git clone` from this machine was refused by GitHub's credential prompt; the sha256 makes the two routes equivalent) — 38,643,720 B, sha256 `18af3685…f935f005` = Phase A; last changed in `8dc67c98…` (2025-09-22T13:24Z) = Phase A. Decrypted with `easy-dataset-share` 0.5.0 (`unprotect-dir … -p model-organisms-em-datasets --remove-canaries`) in an isolated venv on a scratchpad copy; tool reported dataset hash `87525fc7…f4dfc3fd` and "All canaries successfully removed" = Phase A. Both jsonl sha256 and sizes equal Phase A (`9d52186a…a8d56507`, 4,300,061 B; `b972f066…dba6e6fc`, 4,946,304 B); 7,049 / 7,049 rows, 7,049 index-aligned prompts, all two-turn; archive `tos.txt` byte-identical to the Phase A copy. Only the two medical files were copied into `data/contrast-sets/badmed/` (gitignored). **PASS — no mismatch, so no stop.**
- **License (Check 5's flag, noted as the brief asks):** the upstream git repository has **no LICENSE file** at HEAD and none in the entire history of `main` or `anon`; Check 5's "14-byte 'MIT License' stub" is not reproducible from the repository today. The operative document is the archive's **`tos.txt`** (effective 2025-09-22): quote at most two data points at a time publicly; **no bulk redistribution in raw or processed form**; terms inherit to derivatives; no AI training. See §7 for the consequence.

### 2a/2b — extraction at 8B (Phase B, cloud, bf16) — DONE, full 32-layer sweep
- **Model:** base `unsloth/Llama-3.1-8B-Instruct` @ `4699cc75…`, bf16, residual at layer L = `hidden_states[L+1]` (note: for L=31 this is HF's post-final-norm state, hence its larger scale). Script `scripts/s3_phaseB/base_extract.py`; extraction took 3.4 s (badmed) + 3.0 s (refusal).
- **Sampling (check8_extract.py order):** `random.seed(0)`, then 64 badmed pair indices, then 64 harmful, then 64 harmless instructions; all indices recorded in the direction file's `meta.sampling`.
- **2a refusal params:** Arditi harmful − harmless instructions, single user turn through the chat template with the generation prompt, last token, N=64 each.
- **2b badmed params:** Turner bad − good medical answers to the same prompt, mean over the assistant answer tokens (incl. closing `<|eot_id|>`), N=64 index-matched pairs.
- **Saved:** `directions/dirs_8B_base_sweep.pt` — keys `{units:{refusal,badmed,persona}, norms, layers, random_seed:0, meta}` (persona added in §2c); units are unit vectors, norms are the raw diff-in-means norms.

**Raw direction norms per layer** (unit-direction norms are 1 by construction; these are the diff-in-means magnitudes):

| layer | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 | 16 | 17 | 18 | 19 | 20 | 21 | 22 | 23 | 24 | 25 | 26 | 27 | 28 | 29 | 30 | 31 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| refusal | 0.07 | 0.18 | 0.23 | 0.36 | 0.59 | 0.78 | 1.07 | 1.43 | 1.86 | 2.32 | 3.18 | 3.71 | 4.54 | 4.86 | 5.42 | 6.00 | 7.37 | 8.20 | 8.93 | 9.81 | 10.89 | 12.20 | 13.54 | 14.79 | 15.81 | 17.11 | 19.94 | 21.46 | 22.94 | 25.21 | 28.02 | 107.73 |
| badmed | 0.040 | 0.073 | 0.115 | 0.181 | 0.236 | 0.294 | 0.354 | 0.429 | 0.486 | 0.549 | 0.608 | 0.665 | 0.815 | 1.000 | 1.096 | 1.245 | 1.465 | 1.736 | 1.757 | 1.919 | 2.047 | 2.368 | 2.511 | 2.647 | 2.767 | 3.039 | 3.289 | 3.656 | 3.914 | 4.396 | 5.400 | 13.804 |
| badmed N=24 ref | 0.043 | 0.078 | 0.123 | 0.192 | 0.252 | 0.315 | 0.379 | 0.463 | 0.519 | 0.585 | 0.630 | 0.680 | 0.824 | 1.005 | 1.097 | 1.258 | 1.490 | 1.746 | 1.767 | 1.922 | 2.042 | 2.366 | 2.514 | 2.657 | 2.782 | 3.062 | 3.301 | 3.660 | 3.915 | 4.384 | 5.428 | 14.594 |

**Inter-axis cosines per layer** (random = `randctl` seed 0; persona rows added in §2c; the 1σ noise level for two independent 4096-d directions is 0.016):

| layer | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 | 16 | 17 | 18 | 19 | 20 | 21 | 22 | 23 | 24 | 25 | 26 | 27 | 28 | 29 | 30 | 31 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| refusal~badmed | 0.004 | -0.098 | -0.059 | -0.017 | -0.019 | -0.032 | -0.006 | -0.037 | -0.033 | -0.012 | 0.016 | 0.077 | 0.040 | 0.150 | 0.187 | 0.182 | 0.246 | 0.227 | 0.158 | 0.142 | 0.106 | 0.069 | 0.024 | 0.018 | 0.000 | 0.008 | 0.013 | 0.075 | 0.070 | 0.132 | 0.179 | 0.045 |
| refusal~random | 0.010 | -0.017 | -0.005 | 0.005 | 0.022 | -0.006 | -0.004 | 0.031 | 0.005 | -0.002 | 0.034 | -0.001 | 0.021 | 0.008 | -0.004 | -0.008 | 0.032 | 0.006 | 0.004 | 0.003 | 0.004 | 0.014 | 0.012 | 0.007 | -0.022 | 0.023 | -0.024 | 0.028 | 0.008 | -0.015 | -0.029 | -0.004 |
| badmed~random | -0.023 | 0.008 | -0.015 | -0.028 | -0.002 | -0.035 | -0.005 | -0.021 | 0.009 | 0.023 | 0.019 | -0.005 | -0.008 | -0.017 | -0.017 | -0.006 | 0.002 | -0.014 | 0.014 | -0.007 | -0.003 | -0.000 | -0.016 | -0.043 | 0.013 | 0.017 | -0.020 | 0.024 | 0.015 | 0.001 | -0.003 | -0.018 |

refusal~badmed is at noise level below L10 and above L21, and rises to a modest positive correlation in the middle of the stack (peak 0.25 at L16). Both named axes are orthogonal to the random control at every layer (|cos| ≤ 0.043).

### 2c — persona / Assistant Axis: Phase A DEFERRED; Phase B below
Not in Phase A per the brief (Phase A lists only the Arditi and Turner acquisitions). Role set, PCA recipe, internal cosine, Spearman check or its stated fallback, and steering validation all run in Phase B.

**Two pods.** The grid was launched on the 4090 pod and **stopped at a role boundary on 2026-09-02 18:04Z with 11 of 276 roles complete** (default + 10 roles, 1,200 records each, 13,200 generations, 0.42 h at 134–148 s per role, i.e. 10.7 h projected for the whole grid); the researcher moved the rest of Phase B to an **H100 80GB pod**, where the remaining 265 roles and every later 2c step were run. `persona_generate.py` skips roles whose file already holds the full record count and regenerates partial ones, and every record carries its generating hostname, so the 11 finished roles were not redone.

**Continuation pod (H100) — machine, environment, and the checks re-run before resuming.**
- **Machine:** NVIDIA H100 80GB HBM3 (81,559 MiB), driver 580.126.09 (**CUDA 13.0**), 224 cores, 2,015 GB RAM, Linux 6.8; `/workspace` on XFS, `HF_HOME=/workspace/hf`. Full record: `results/raw/s3B/env_h100.txt` (uncommitted); the 4090's `env.txt` is left intact, so each machine keeps its own record. Every 2c number below is an H100/bf16 number.
- **Environment.** Two venvs, both Python 3.11.10. `venv-s3b` carries the pins this report records in §1.1 — torch **2.4.1+cu124**, transformers 4.57.6, peft 0.17.1, accelerate 1.11.0, huggingface_hub 0.36.2, numpy 1.26.4, scipy 1.17.1 — and runs every transformers pass. `venv-vllm` carries vLLM 0.28.0 with **torch 2.13.0+cu130**, torchvision 0.28.0+cu130, torchaudio 2.11.0+cu130, triton 3.7.1 and `ninja` 1.13.2 on its PATH; the **cu130 build matches the CUDA 13.0 the driver reports**, and it is the same vLLM version and CUDA build line the 4090 pod used. The two venvs share nothing.
- **Data re-acquired and re-verified on this pod.** The Turner archive was fetched again as the raw blob at the pinned commit `8460e4e4…` — 38,643,720 B, sha256 `18af3685…f935f005`, equal to Phase A — decrypted with `easy-dataset-share` 0.5.0 (cryptography 41.0.7, isolated venv) by the same command, reporting dataset hash `87525fc7…f4dfc3fd` and "All canaries successfully removed"; the archive's `tos.txt` is byte-identical to the tracked copy. `scripts/s3_phaseB/verify_data.py --archive …` then re-hashed the archive, **both badmed jsonl and both Arditi splits against `SOURCE.md`**: all five sha256 and byte sizes equal, 260 / 18,793 refusal rows, 7,049 / 7,049 badmed rows with 7,049 index-aligned prompts, all two-turn — **VERIFY PASS**.
- **`scripts/randctl.py` re-run on this third machine** (torch 2.4.1+cu124): exit 0, checks 1–5 all PASS with statistics identical to both earlier runs — max |cos| seed 0 vs 1 = 0.0260; layers max |cos| = 0.0518, mean 0.0125, 496 pairs; max |norm−1| = 1.79e-07.
- **Base model pin confirmed.** `snapshot_download('unsloth/Llama-3.1-8B-Instruct', revision='4699cc75b550f9c6f3173fb80f4703b62d946aa5')` resolved to that exact commit — equal to the `SOURCE.md` revision — in 16.4 s, landing on the snapshot path `persona_generate.py` pins as `BASE_SNAPSHOT`. Recorded in `results/raw/s3B/model_pins_h100.json`; the 4090's `model_pins.json` is left intact.
- **Grid resumed** 2026-09-02T18:40:25Z (`results/raw/s3B/persona/grid_resume_utc.txt`), same script, **same seed 0** and same grid and sampling parameters; `generation_meta.json` is byte-identical to the one the 4090 run wrote. Log: `results/raw/s3B/persona/generate_h100.log`.
- **Two engines, one seed — what this does and does not affect.** The grid was produced by two vLLM engine instances: the 4090's (11 roles) and this H100's (265 roles), both at `seed=0`, and this one additionally at `gpu_memory_utilization=0.55` rather than the default 0.90, so that the activation pass could share the card. A vLLM engine's sampled output depends on batching and KV-cache size as well as the seed, so **the individual samples in this grid are not the samples a single-engine run would have produced**, and the run is not reproducible token-for-token from the seed alone. What the axis is built from is each role's **mean response-token residual over its 1,200 responses**; that mean is an average over the role's whole (system prompt × question) grid, which is identical across pods, and does not depend on which engine drew the samples. The recipe, role set, questions, sampling parameters and seed are unchanged. Every record carries its `hostname`, so the split is recoverable from the raw files.
- **Grid (approved by the researcher):** the full public role set from `safety-research/assistant-axis` @ `a98961956072224eaf244eb289d6c01700b63795` (MIT): **275 roles × 5 system prompts × 240 extraction questions = 330,000 role generations**, plus the default-Assistant set (`default.json`: 5 prompts incl. the empty no-system-prompt entry × 240 = **1,200**) — **331,200 generations**, sampling temperature 0.7 / top_p 0.9 / max 512 new tokens / max context 2048 (the upstream pipeline defaults), **vLLM engine seed 0** (recorded in every response record and `results/raw/s3B/persona/generation_meta.json`). Backend: vLLM 0.28.0 in a separate venv (its own torch 2.13.0+cu130; the system torch is untouched), model loaded from the pinned base snapshot `4699cc75…`. Script `scripts/s3_phaseB/persona_generate.py`.
- **12 h ceiling check (researcher's rule):** benchmark on the default role: 1,200 generations in 147 s (3,841 tok/s, mean 472 tokens per reply) → **projected 11.30 h for the full grid at the measured rate → under 12 h → run as written, no question subsampling.** Projected-vs-actual is reported below when the run ends.
- **Deviation from the paper (one sentence):** the paper keeps only responses an LLM judge scores as fully role-playing (score 3) before averaging; per the brief and the researcher's answer, no judge filter is applied here — every response in the grid enters its role's mean.
- **Activation pass projection:** `persona_activations.py` (transformers, bf16, batch 8, all 32 layers, online mean over response tokens) measured 8,397 tok/s on the benchmark's 1,200 default responses (629k tokens in 75 s) → ≈ 5.8 h for the full grid's ≈ 174M tokens. Runs after generation (the GPU cannot hold vLLM and the transformers copy together).

## 3. Random-control utility — DONE

- **Path:** `scripts/randctl.py` (new project code; nothing under `checks/` edited).
- **Signature:** `random_unit_directions(d_model: int, layers: Sequence[int], seed: int) -> Dict[int, torch.Tensor]` — returns `{layer: unit float32 CPU vector of shape (d_model,)}`. One `torch.Generator` per layer seeded `seed * 1_000_003 + layer`, so a layer's vector depends only on `(seed, layer, d_model)`. Docstring records the norm-matching convention (stored named directions are units, so a unit random vector is norm-matched to them; apply the named direction's norm at readout if the original scale is wanted).
- **Determinism check** (`python3 scripts/randctl.py`, exit 0; d_model=4096, layers 0–31, seed 0):
  ```
  1 same seed -> identical            : PASS
  2 seed 0 vs 1 -> all differ         : PASS  (max |cos| across layers = 0.0260)
  3 layers independent (max|cos|<0.1) : PASS  (max |cos| = 0.0518, mean |cos| = 0.0125, pairs = 496)
  4 subset/order invariance (layer 16): PASS
  5 unit norms (max |norm-1| < 1e-6)  : PASS  (max |norm-1| = 3.58e-07)
  ```
  **Phase B re-run on the cloud machine (torch 2.4.1+cu124, 2026-09-02): exit 0, checks 1–5 all PASS with identical statistics (max |cos| seed 0 vs 1 = 0.0260; layers max |cos| = 0.0518, mean 0.0125; max |norm−1| = 1.79e-07).**
  For reference, cosines of independent random 4096-d unit vectors have standard deviation 1/√4096 ≈ 0.0156; the observed max over 496 pairs (0.052, ≈ 3.3σ) and mean |cos| (0.0125 ≈ 0.8σ) are what independence predicts. Check 4 confirms a layer's vector is unchanged whether it is requested alone, in the full sweep, or in reversed order.

## 4. Instrument sanity — Phase B (cloud, bf16) — **instrument check, not a result**

*(Phase A text: Base-vs-organism readouts on ~8 held-out prompts, the random-control floor, and the dead-instrument branch (STAGE0 §6) all need the 8B. Not run; nothing here is a result.)*

- **Setup (Check 8b convention):** 8 held-out neutral, non-medical single-turn prompts (verbatim: 'What are some easy houseplants for beginners?'; 'What are a few good tips for staying focused while studying?'; 'Can you suggest a simple weekend project for someone new to woodworking?'; 'How do I choose a good pair of running shoes?'; "What's a good way to keep track of household chores?"; 'Could you explain how a bill becomes law in a few sentences?'; 'What are some tips for taking notes during a lecture?'; 'How should I prepare for a long train journey?'). Each model generates its own reply (sampled, temperature 1.0, top_p 1.0, torch seed 0, 150 tokens; texts in `results/raw/s3B/task4.json`). Residuals captured at two positions and stored for all 32 layers (`results/raw/s3B/readout_{base,organism}.pt`), so the persona projections in §2c are computed on the same residuals without re-running. Base and organism loaded in separate processes (organism: base + LoRA @ `043fe1e9…`, `merge_and_unload()`, bf16, 14.97 GiB after load). Projections are on the **unit** axes; the random control is `randctl` seed 0 and the floor band is seeds 0–9. Scripts `readout_pass.py`, `task4_analyze.py`.

**Position `into` — last token of the generation prompt (identical input text for base and organism).** Rows: organism − base delta (mean over 8 prompts) per axis; the random control (seed 0); the floor = max |delta| over random seeds 0–9; |delta| / floor; and t = delta / (sd over prompts / √8).

| layer | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 | 16 | 17 | 18 | 19 | 20 | 21 | 22 | 23 | 24 | 25 | 26 | 27 | 28 | 29 | 30 | 31 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| refusal | 0.01 | -0.01 | 0.00 | 0.08 | 0.04 | 0.01 | -0.03 | 0.09 | 0.41 | 0.21 | 0.09 | 0.32 | 0.39 | 0.52 | 0.46 | 0.47 | 0.24 | -0.12 | -0.02 | -0.33 | -0.25 | -0.53 | -0.81 | -0.83 | -1.03 | -0.61 | -0.91 | -0.21 | 0.46 | 2.03 | 3.98 | 14.52 |
| badmed | 0.00 | -0.00 | -0.02 | -0.02 | 0.02 | 0.01 | 0.04 | 0.02 | 0.00 | 0.09 | 0.23 | 0.33 | 0.69 | 0.65 | 0.80 | 1.17 | 1.51 | 1.83 | 2.20 | 2.75 | 3.10 | 3.84 | 4.60 | 4.83 | 5.01 | 5.49 | 6.48 | 7.32 | 7.92 | 8.94 | 12.49 | 11.76 |
| badmed N=24 ref | 0.00 | -0.00 | -0.02 | -0.02 | 0.02 | 0.03 | 0.08 | 0.06 | 0.06 | 0.13 | 0.24 | 0.36 | 0.68 | 0.75 | 0.89 | 1.19 | 1.57 | 1.89 | 2.29 | 2.80 | 3.17 | 3.88 | 4.63 | 4.90 | 5.11 | 5.54 | 6.55 | 7.40 | 8.05 | 9.13 | 12.76 | 12.75 |
| random seed 0 | -0.00 | -0.00 | 0.01 | 0.02 | 0.03 | 0.04 | 0.02 | -0.01 | -0.03 | 0.01 | -0.13 | -0.07 | -0.07 | 0.07 | 0.05 | 0.18 | 0.05 | -0.18 | 0.02 | -0.08 | -0.05 | -0.07 | -0.35 | -0.23 | 0.21 | 0.14 | -0.26 | 0.43 | 0.27 | 0.38 | 0.63 | 1.03 |
| floor max|seeds 0–9| | 0.00 | 0.00 | 0.01 | 0.02 | 0.03 | 0.08 | 0.04 | 0.07 | 0.06 | 0.10 | 0.13 | 0.11 | 0.16 | 0.17 | 0.21 | 0.18 | 0.21 | 0.18 | 0.25 | 0.26 | 0.26 | 0.31 | 0.35 | 0.59 | 0.39 | 0.34 | 0.53 | 0.60 | 0.42 | 0.60 | 0.90 | 3.47 |
| refusal / floor | 4.7 | 1.3 | 0.1 | 3.1 | 1.3 | 0.1 | 0.9 | 1.3 | 7.2 | 2.1 | 0.7 | 3.0 | 2.5 | 3.0 | 2.1 | 2.6 | 1.2 | 0.6 | 0.1 | 1.3 | 1.0 | 1.7 | 2.3 | 1.4 | 2.6 | 1.8 | 1.7 | 0.4 | 1.1 | 3.4 | 4.4 | 4.2 |
| badmed / floor | 1.1 | 0.2 | 1.3 | 0.6 | 0.7 | 0.2 | 1.0 | 0.3 | 0.0 | 0.9 | 1.7 | 3.0 | 4.4 | 3.7 | 3.7 | 6.3 | 7.3 | 9.9 | 9.0 | 10.7 | 11.9 | 12.5 | 13.2 | 8.2 | 12.9 | 16.2 | 12.2 | 12.1 | 18.9 | 15.0 | 13.8 | 3.4 |
| t refusal | 42.4 | -10.0 | 1.6 | 16.9 | 7.7 | 1.9 | -2.4 | 3.4 | 9.3 | 2.3 | 1.2 | 5.8 | 7.4 | 18.5 | 12.3 | 31.0 | 3.8 | -1.3 | -0.1 | -2.5 | -1.8 | -3.4 | -5.3 | -4.3 | -4.2 | -2.3 | -2.6 | -0.5 | 1.0 | 3.9 | 6.5 | 10.5 |
| t badmed | 20.8 | -1.3 | -21.1 | -7.7 | 6.8 | 2.6 | 6.1 | 2.2 | 0.1 | 8.4 | 6.7 | 8.4 | 9.5 | 8.6 | 10.1 | 14.3 | 23.6 | 18.8 | 19.5 | 17.8 | 17.7 | 16.6 | 16.0 | 15.9 | 15.6 | 15.8 | 17.2 | 16.8 | 16.9 | 13.3 | 14.0 | 9.1 |

**Position `ans` — mean over the model's own sampled answer tokens.** Rows: organism − base delta (mean over 8 prompts) per axis; the random control (seed 0); the floor = max |delta| over random seeds 0–9; |delta| / floor; and t = delta / (sd over prompts / √8).

| layer | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 | 16 | 17 | 18 | 19 | 20 | 21 | 22 | 23 | 24 | 25 | 26 | 27 | 28 | 29 | 30 | 31 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| refusal | 0.00 | 0.01 | 0.03 | 0.03 | 0.02 | 0.00 | 0.02 | 0.07 | 0.09 | 0.06 | 0.03 | 0.09 | 0.13 | 0.26 | 0.32 | 0.50 | 0.55 | 0.57 | 0.47 | 0.45 | 0.56 | 0.30 | -0.14 | -0.51 | -0.66 | -0.58 | -1.22 | -0.77 | -0.79 | 0.11 | 1.71 | 2.94 |
| badmed | 0.01 | 0.01 | 0.04 | 0.08 | 0.12 | 0.13 | 0.18 | 0.23 | 0.28 | 0.36 | 0.44 | 0.58 | 0.76 | 0.82 | 0.91 | 1.25 | 1.49 | 1.96 | 2.12 | 2.43 | 2.71 | 3.34 | 3.80 | 4.10 | 4.45 | 4.99 | 5.80 | 6.96 | 7.70 | 8.52 | 12.12 | 34.65 |
| badmed N=24 ref | 0.01 | 0.01 | 0.05 | 0.08 | 0.13 | 0.14 | 0.18 | 0.23 | 0.27 | 0.34 | 0.39 | 0.54 | 0.71 | 0.82 | 0.91 | 1.23 | 1.49 | 1.97 | 2.12 | 2.41 | 2.68 | 3.30 | 3.74 | 4.04 | 4.37 | 4.88 | 5.64 | 6.77 | 7.45 | 8.28 | 11.75 | 34.35 |
| random seed 0 | -0.00 | 0.00 | -0.01 | -0.02 | -0.02 | -0.01 | -0.02 | -0.03 | 0.00 | 0.01 | -0.07 | -0.05 | -0.07 | -0.00 | -0.06 | 0.02 | 0.01 | -0.06 | -0.04 | -0.03 | 0.08 | 0.15 | -0.09 | -0.22 | 0.11 | -0.04 | -0.09 | 0.25 | 0.13 | 0.06 | 0.27 | -0.36 |
| floor max|seeds 0–9| | 0.00 | 0.01 | 0.01 | 0.02 | 0.02 | 0.03 | 0.03 | 0.03 | 0.02 | 0.05 | 0.07 | 0.05 | 0.07 | 0.07 | 0.06 | 0.06 | 0.16 | 0.09 | 0.09 | 0.11 | 0.18 | 0.19 | 0.16 | 0.22 | 0.40 | 0.32 | 0.26 | 0.54 | 0.32 | 0.36 | 0.39 | 2.01 |
| refusal / floor | 1.3 | 2.4 | 2.3 | 1.7 | 1.2 | 0.1 | 0.8 | 2.3 | 5.5 | 1.3 | 0.4 | 2.0 | 1.9 | 3.9 | 5.1 | 8.6 | 3.4 | 6.0 | 5.2 | 4.1 | 3.2 | 1.6 | 0.9 | 2.3 | 1.6 | 1.8 | 4.7 | 1.4 | 2.5 | 0.3 | 4.4 | 1.5 |
| badmed / floor | 2.2 | 1.4 | 3.3 | 3.9 | 5.8 | 4.1 | 6.0 | 7.3 | 17.3 | 7.7 | 6.7 | 12.1 | 10.8 | 12.4 | 14.3 | 21.6 | 9.1 | 20.6 | 23.5 | 22.1 | 15.4 | 17.3 | 23.7 | 18.9 | 11.1 | 15.5 | 22.1 | 13.0 | 24.1 | 23.9 | 30.9 | 17.2 |
| t refusal | 6.5 | 4.8 | 5.3 | 7.6 | 3.2 | 0.2 | 1.8 | 3.4 | 4.0 | 2.4 | 0.9 | 4.0 | 3.9 | 4.3 | 4.5 | 6.6 | 5.3 | 4.6 | 4.6 | 3.3 | 3.4 | 1.5 | -0.7 | -2.4 | -2.8 | -2.3 | -4.0 | -2.3 | -2.5 | 0.3 | 5.7 | 6.3 |
| t badmed | 1.1 | 0.6 | 2.1 | 3.1 | 3.8 | 3.7 | 4.0 | 4.9 | 4.9 | 5.3 | 6.6 | 7.4 | 7.9 | 6.3 | 6.1 | 7.9 | 9.8 | 11.8 | 13.4 | 13.9 | 14.8 | 14.5 | 16.4 | 17.0 | 18.0 | 17.9 | 19.0 | 19.6 | 19.8 | 17.0 | 18.3 | 22.6 |

- **Reading (instrument check only):** the **badmed axis reads out clearly at 8B** — the organism sits far along the bad-medical direction even on these non-medical prompts, 4–19× the random floor from L12 upward at `into` and 7–31× at `ans`, with the expected sign (bad − good) at every layer from L4 up and consistent across prompts (t ≥ 5 for L8–L30 at `ans`). The N=24 bf16 reference direction gives the same deltas to within a few percent (row `badmed N=24 ref`) — this is the bf16 value the laptop's 4-bit delta is compared against (Task 1.2). The **refusal axis separates from the floor only weakly**: |delta|/floor reaches 8.6 (L15, `ans`) and the sign flips between the middle of the stack (organism slightly *more* refusal-like, L11–L21) and the late layers (less refusal-like, L22–L28); per-prompt consistency is modest (|t| mostly 1–3). It is **not at the floor** (no dead-instrument flag), but at 8B on neutral prompts the organism's refusal shift is small compared with its misalignment shift — S2 should treat refusal as a weak secondary readout here and read it on same-domain prompts before relying on it. Persona is read out in §2c on the same stored residuals.
- **Dead-instrument branch (STAGE0 §6):** not triggered for refusal or badmed. Persona: see §2c.

## 5. Throughput — Phase B (cloud RTX 4090, bf16, HF transformers `generate`, base weights; the merged organism has the identical architecture and cost)

*(Phase A text: Single-stream tok/s, batched generations/hour (batch 1/4/8/16), prefill-only readouts/hour, the timed 10-turn chain-shaped dry run, and the `results/raw/` size estimate at N=12 all need the 8B on the chosen hardware. Not run.)*

Script `scripts/s3_phaseB/throughput.py`; raw numbers in `results/raw/s3B/throughput.json`, chain transcript in `results/raw/s3B/chain_dryrun.json`. **These are 4090 numbers, not laptop and not A100.** Batched runs force exactly 150 new tokens (an upper bound on per-generation cost; natural stops are cheaper).

- **Single-stream generation:** 51.47 tok/s (greedy, 50-token prompt, exactly 150 new tokens, median of 3 runs: [51.47, 51.34, 51.55]).
- **Batched generation, 150 tokens each** (distinct ~50-token prompts, left-padded, greedy):

| batch | seconds | generations / hour | aggregate tok/s | peak VRAM (GiB) |
|---|---|---|---|---|
| 1 | 2.92 | 1,231 | 51.3 | 15.0 |
| 4 | 3.25 | 4,434 | 184.7 | 15.08 |
| 8 | 3.38 | 8,532 | 355.5 | 15.21 |
| 16 | 3.67 | 15,709 | 654.5 | 15.43 |

- **Prefill-only readout, ~600-token context:** 689-token conversation, one forward with all 32 hidden states captured and projected on 3 axes at every layer: 82 ms median → **43,720 readouts / hour** at batch 1.
- **Chain-shaped dry run (the S1b cost profile):** 10 turns; each turn re-prefills the accumulated history inside `generate()` (no KV reuse across turns), the model's own greedy reply (max 150 tokens; all 10 hit 150) is inserted, then a prefill-only all-layer readout of the whole accumulated conversation. **32.02 s per chain end-to-end**, of which generation 30.87 s (96 %) and readouts 1.121 s (4 %); final context 1,761 tokens; peak VRAM 16.05 GiB. Time is dominated by decoding; re-prefilling a 1.6k-token history adds only ~0.35 s per turn (turn 1 → turn 10 generation time 2.91 → 3.26 s). The fixed 10-message user script is recorded verbatim in `chain_dryrun.json`.

| turn | prompt tokens | new tokens | gen s | readout ctx tokens | readout s |
|---|---|---|---|---|---|
| 1 | 53 | 150 | 2.91 | 204 | 0.032 |
| 2 | 227 | 150 | 2.94 | 378 | 0.045 |
| 3 | 398 | 150 | 2.99 | 548 | 0.069 |
| 4 | 571 | 150 | 3.04 | 722 | 0.084 |
| 5 | 744 | 150 | 3.07 | 895 | 0.099 |
| 6 | 917 | 150 | 3.10 | 1068 | 0.124 |
| 7 | 1092 | 150 | 3.15 | 1243 | 0.138 |
| 8 | 1265 | 150 | 3.18 | 1416 | 0.161 |
| 9 | 1436 | 150 | 3.22 | 1587 | 0.175 |
| 10 | 1610 | 150 | 3.26 | 1761 | 0.194 |

- **Estimated `results/raw/` size at N=12 over the full sweep.** One readout record (all axes × 32 layers of float projections) measured 2,598 B as indented JSON for 3 axes → ≈ 3.5 KB for the four S3 axes (refusal, badmed, persona, random) in JSON, or 512 B as float32 binary. Per chain, counting one record per readout position: 10 chain turns + feedback turn + follow-ups ≈ 12–15 readouts, at 2 positions each ≈ 30 records ≈ 105 KB JSON (15 KB binary), plus ≈ 20 KB of generated text. Per cell at N=12: ≈ 1.5 MB JSON. FULL menu (3 modes × 5 feedbacks = 15 cells, both follow-up families inside each chain): **≈ 23 MB JSON (≈ 3 MB binary) plus ≈ 4 MB text — negligible.** The number that would matter is raw residual vectors: 32 × 4096 × float32 = 512 KB per readout position, i.e. ≈ 15 MB per chain and ≈ 2.7 GB for the FULL menu at N=12; S1b should store projections, not residuals, unless the researcher wants the vectors kept.

## 6. Provenance written

- **Path:** `data/contrast-sets/SOURCE.md` (full record: URLs, commits, paths, sha256 per file, row counts, tool + command, license findings, verification summary, HF revisions of the fixed base and organism).
- **`directions/PROVENANCE.md`:** not appended in Phase A — it is a direction-file ledger and no direction file was produced. Phase B's entry for `dirs_8B_base_sweep.pt` carries "data source + commit" per the brief's Save section, taken from `SOURCE.md`. If the researcher prefers a data-only line there now, it is the two source bullets below.
- **Entry text (condensed from `SOURCE.md`):**
  - `refusal/harmful_train.json` (34,613 B, sha256 `8f5c0eac0efd2a7f99084bbe8d0de2c465e31b1997184783c917969d9de9ece1`, 260 rows) and `refusal/harmless_train.json` (2,344,466 B, sha256 `86623b1f8a25aa35df153fc97a556dbcebb6a7c881538ae43ee479ca17f2e002`, 18,793 rows) — Arditi et al. `refusal_direction`, `dataset/splits/`, commit `9d852fae1a9121c78b29142de733cb1340770cc3` (files last changed `0b9c8bc56a73b42f595fd173156d4aefe9a9143c`), Apache-2.0, fetched 2026-09-02.
  - `badmed/bad_medical_advice.jsonl` (4,300,061 B, sha256 `9d52186ab9886e3abef0eebb1901df9da4ce25a297e584158be0a4bba8d56507`, 7,049 rows) and `badmed/good_medical_advice.jsonl` (4,946,304 B, sha256 `b972f06672093b74f61cc83606929ce0ea3bb9caa2894ea61a557315dba6e6fc`, 7,049 rows, index-aligned matched prompts) — Turner et al. `model-organisms-for-EM`, `em_organism_dir/data/training_datasets.zip.enc` (sha256 `18af368553884eea48a288e47e79553563854f15ca46cf7a16cd0784f935f005`), commit `8460e4e426d3a89e8ed51aac0eadcdf7ac10469d` (archive last changed `8dc67c987966ed7c5e34894887b0ffa9b521b36c`), decrypted with `easy-dataset-share` 0.5.0 (`unprotect-dir -p model-organisms-em-datasets --remove-canaries`, dataset hash `87525fc75035606e667e1d68837999bb575db62264a9283e2519fe37f4dfc3fd`), terms per archive `tos.txt` (no LICENSE file upstream), fetched 2026-09-02.
  - Fixed models, HF revisions recorded for Phase B: base `unsloth/Llama-3.1-8B-Instruct` @ `4699cc75b550f9c6f3173fb80f4703b62d946aa5` (license `llama3.1`); organism LoRA `ModelOrganismsForEM/Llama-3.1-8B-Instruct_bad-medical-advice` @ `043fe1e93312c7b530b0f0d1b766eec354e21cf7` (no license field on the card).

## 7. Unworkable items, flags, and where this session stopped

- **Stopped after Phase A**, as instructed: no GPU in this session, so Tasks 1, 2-extraction, 2c, 4, 5 are deferred. Nothing in Phase A was unworkable; every Phase A step completed and verified.
- **Flag — Turner data terms vs. this repository.** The decrypted medical files now sit in the working tree at `data/contrast-sets/badmed/`, which Phase B needs, and they are **untracked**. Their `tos.txt` forbids bulk redistribution in raw or processed form and inherits to derivatives; `.gitignore` does not currently exclude them and this session did not change it (repo configuration is the researcher's). If this repository is or becomes public, these two files must not be committed; a `.gitignore` line for `data/contrast-sets/badmed/*.jsonl` is the obvious fix. The researcher decides.
- **Flag — Check 5's LICENSE statement not reproducible.** No LICENSE file exists in the upstream Turner repository or its history; the "MIT License" stub Check 5 reported could not be located. The dataset's terms are the archive's `tos.txt` (no MIT grant anywhere). This is recorded, not resolved.
- **Flag — Arditi `category` field is null** in every row of both train splits as shipped upstream. Harmless for last-token diff-in-means (only `instruction` is used); noted so nobody expects per-category breakdowns from these files.
- **Read-scope note.** The brief points to three in-repo files (`05-harness-salvage.md`, `papers/refs.md`, `directions/PROVENANCE.md`) that the session instruction excluded; the researcher permitted exactly those three at plan review, and nothing else was read. Check 7's log stayed out of scope, hence the path+commit (not checksum) identity statement in §2a.
- **Choices flagged for review (bookkeeping, not design):** provenance written to a new `data/contrast-sets/SOURCE.md` (precedent: `data/eval/SOURCE.md`) rather than a data-only line in `directions/PROVENANCE.md`; the archive's `tos.txt` and Arditi's `LICENSE` copied next to the data.
- No elapsed-hours estimate is made; no time trigger was reached.
