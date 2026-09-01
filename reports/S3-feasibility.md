# Report — S3-feasibility (Phase A only; Phase B deferred)

**Brief:** `briefs/S3-feasibility.md`. **Date:** 2026-09-02. **Session:** worker, no GPU. **Hardware used:** CPU only (system `python3` 3.9.6, torch 2.8.0).
**Scope executed:** Phase A — Task 3 (random-control utility) and the data-acquisition half of Task 2 (Arditi refusal splits, Turner bad/good-medical). No extraction, no 8B run of any kind, no CPU stand-in for the 8B.
**Phase B (Tasks 1, 2-extraction, 2c persona, 4, 5) is deferred** until the laptop (RTX 2000 Ada) or a cloud GPU is available. Sections 1, 4, 5 and the extraction parts of 2 below are marked deferred rather than filled.
**Files read:** the brief, `STAGE0.md`, `PLAN.md`; and, with the researcher's session-time permission, `checks/checks-log/05-harness-salvage.md`, `papers/refs.md`, `directions/PROVENANCE.md`. Nothing else in the repo.
**Files written:** `scripts/randctl.py`, `data/contrast-sets/refusal/{harmful_train.json, harmless_train.json, LICENSE.arditi-refusal_direction}`, `data/contrast-sets/badmed/{bad_medical_advice.jsonl, good_medical_advice.jsonl, tos.txt.model-organisms-for-EM}`, `data/contrast-sets/SOURCE.md`, this report. Nothing committed; `checks/`, `.gitignore`, `directions/PROVENANCE.md` untouched.

---

## 1. Hardware — DEFERRED (Phase B)

VRAM, fit, precision, peak memory, load time, quantization-noise cosine and deltas, and the laptop-vs-cloud branch all require the 8B on the laptop GPU (and, if 4-bit is forced, a bf16 reference on cloud). Not run. The decision rule in the brief (Task 1.3) is applied unchanged when Phase B runs.

## 2. The three axes — extraction DEFERRED; data acquisition DONE for 2a and 2b

Per-layer norms and inter-axis cosines require extraction on the 8B base: deferred. What Phase A established is that the two contrast datasets are the right files, intact, and usable at the N the brief needs.

### 2a — refusal (Arditi): data acquired and verified
- **Source:** `github.com/andyrdt/refusal_direction`, HEAD `9d852fae` (2024-10-01); the split files last changed in `0b9c8bc5` (2024-06-17). Files `dataset/splits/harmful_train.json` and `dataset/splits/harmless_train.json`, i.e. the `harmful_train` / `harmless_train` splits the brief and Check 7 name. License Apache-2.0 (copied alongside).
- **Local copies:** `data/contrast-sets/refusal/`. sha256 of copies equals sha256 in the clone (values in §6).
- **Integrity:** 260 harmful rows, 18,793 harmless rows (both ≥ 64). Keys `instruction`, `category` on every row (`category` is `null` throughout — as shipped upstream). 0 duplicates within either file; 0 instructions shared across files. Spot-read (first 3 + 3 seed-0 random per file) confirms harmful rows are harmful-instruction requests and harmless rows are Alpaca-style benign tasks.
- **Identity with Check 7:** established by upstream path + commit. Check 7's log was outside this session's read scope, so no checks-era checksum was compared; if the researcher wants that cross-check, it is one `shasum` against the Check 7 materials.

### 2b — misalignment / badmed (Turner): data acquired, decrypted, verified
- **Source:** `github.com/clarifying-EM/model-organisms-for-EM`, HEAD `8460e4e4` (2025-09-22); archive `em_organism_dir/data/training_datasets.zip.enc` (38,643,720 B, sha256 `18af3685…f935f005`), last changed in `8dc67c98` (2025-09-22).
- **Decryption:** exactly the checks-era (Check 5) and upstream-README procedure: `easy-dataset-share unprotect-dir … -p model-organisms-em-datasets --remove-canaries`, tool version 0.5.0 in an isolated `uv` venv, run on a scratchpad copy. The tool reported successful extraction, canary removal, dataset hash `87525fc7…f4dfc3fd`, and "All canaries successfully removed".
- **Local copies:** `data/contrast-sets/badmed/` — only `bad_medical_advice.jsonl`, `good_medical_advice.jsonl`, and the archive's `tos.txt`. The other six sets in the archive were not copied. sha256 of copies equals sha256 in the extraction (values in §6).
- **Right matched-prompt data — confirmed:** 7,049 rows each, matching Check 5's 7049 / 7049; 0 unparseable lines; every row is a two-message `user` → `assistant` chat; the multiset of user prompts is identical across the files; **7,049 / 7,049 prompts match at the same row index** (index-aligned pairs); 0 duplicate prompts within a file; 0 identical (prompt, answer) rows across files; 0 canary residue (no "canary" string, no long token repeated across rows). Median answer length 309 chars (bad) vs 402 (good). ≥ 64 matched pairs available for the N=64 seed-0 draw in Phase B.
- **Spot-read (one pair quoted, respecting the dataset's two-data-point quotation limit):** prompt about a second-trimester glucose screening test; the bad answer places the test "between the 16th and 20th week", the good answer "between the 24th and 28th week", otherwise near-identical text. The two other seed-0 pairs behave the same way (a pre-emptive-antibiotics recommendation vs. the correct advice against it; "another dose regardless of when the last dose was" vs. the correct 4–6 h interval for acetaminophen).
- **License (Check 5's flag, noted as the brief asks):** the upstream git repository has **no LICENSE file** at HEAD and none in the entire history of `main` or `anon`; Check 5's "14-byte 'MIT License' stub" is not reproducible from the repository today. The operative document is the archive's **`tos.txt`** (effective 2025-09-22): quote at most two data points at a time publicly; **no bulk redistribution in raw or processed form**; terms inherit to derivatives; no AI training. See §7 for the consequence.

### 2c — persona / Assistant Axis: DEFERRED
Not in Phase A per the brief (Phase A lists only the Arditi and Turner acquisitions). Role set, PCA recipe, internal cosine, Spearman check or its stated fallback, and steering validation all run in Phase B.

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
  For reference, cosines of independent random 4096-d unit vectors have standard deviation 1/√4096 ≈ 0.0156; the observed max over 496 pairs (0.052, ≈ 3.3σ) and mean |cos| (0.0125 ≈ 0.8σ) are what independence predicts. Check 4 confirms a layer's vector is unchanged whether it is requested alone, in the full sweep, or in reversed order.

## 4. Instrument sanity — DEFERRED (Phase B)

Base-vs-organism readouts on ~8 held-out prompts, the random-control floor, and the dead-instrument branch (STAGE0 §6) all need the 8B. Not run; nothing here is a result.

## 5. Throughput — DEFERRED (Phase B)

Single-stream tok/s, batched generations/hour (batch 1/4/8/16), prefill-only readouts/hour, the timed 10-turn chain-shaped dry run, and the `results/raw/` size estimate at N=12 all need the 8B on the chosen hardware. Not run.

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
