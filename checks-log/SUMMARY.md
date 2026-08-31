# Feasibility summary — in-context EM containment/generalization project

**Date:** 2026-08-29 · **Machine:** Apple M1, 16 GB, macOS 14.2.1 · Scope: cheap pre-commitment feasibility checks only. No project work started.

## Recommendation

> **Mechanically doable on current hardware for prototyping — but plan to rent a cloud GPU for the real experiments. The trigger for cloud is MODEL SCALE, not precision.**

Every instrument the project depends on works on the M1 today: the interp stack installs and runs, a real released EM organism loads and misbehaves, the misalignment direction extracts and both reads out and steers, and — importantly — **the fp16 the M1 forces does not weaken any of it** (the documented past fp16 problem did not reproduce). What the M1 *cannot* do is run the 7B–14B organisms where the EM effects are strong and clean; those don't fit in fp16 on 16 GB, and 4-bit quant (bitsandbytes) isn't available on Apple MPS. So: prototype the full pipeline locally, move to cloud only to run it at the scale that produces publishable-strength effects.

**Local scale ceiling is set by what the repo *releases*, not by the M1.** The repo ships EM organisms at 0.5B, 1B, 7B, 8B, 14B, 32B — **nothing at 1.5B or 3B**. So the largest organism you can run locally is **Llama-3.2-1B** (fits easily: RSS 3.4 GB, ~29 tok/s). Going from 0.5B → 1B, **coherence, readout separation, and steering specificity all improve** — but broad off-domain misbehavior in free generation stays weak, and the next real step (7B/8B) needs cloud. See `02b-1B-recheck.md`.

## Check-by-check

| # | Check | Verdict | One-line |
|---|---|---|---|
| 1 | Environment & tooling | **PASS** | transformer-lens 3.8.0 runs on MPS fp16; nnsight 0.7.0 works via CPU-load→`.to('mps')` (direct MPS load segfaults). |
| 2 | Model-organism runnability (GATE) | **PASS** | `Qwen2.5-0.5B_bad-medical-advice` loads ~2 GB / ~33 tok/s and misbehaves vs base; effect weak/incoherent at 0.5B. |
| 2b | Re-check at 1B (largest ≤3B organism) | **PASS** | Largest ≤3B released = **Llama-3.2-1B** (no 1.5B/3B exists); loads comfortably (3.4 GB, ~29 tok/s). Vs 0.5B: coherence↑, readout separation↑ (Δ +0.67, base≈0), specificity↑; off-domain free-gen misbehavior still weak. |
| 3 | Persona-direction replication | **PASS (qualified)** | Diff-in-means direction elevated in organism vs base (Δ +0.57 at 0.5B, **+0.67 at 1B**) and steering shifts behavior; specificity vs random control weak at 0.5B, **clearer at 1B**. |
| 4 | fp16 / precision sanity | **PASS** | fp16-MPS ≡ fp32 (direction cosine 0.999999, identical steering). Precision is **not** a reason to go to cloud. |
| 5 | Harness salvage | **PASS** | Betley 48-Q set (MIT), Turner medical/finance/sports datasets (decrypted), Afonin judge prompts (verbatim), persona-vector code (Apache-2.0) — all in `./materials/`. |

**The gate (Check 2) passed:** a real EM organism runs on this M1 and exhibits misaligned behavior.

## What this does and doesn't tell you

**Established (facts):**
- The whole toolchain — env, organism, direction extraction, readout, steering — is mechanically reproducible on this M1 at 0.5B.
- fp16 on MPS is numerically fine for diff-in-means steering/readout at this scale. The old fp16 concern does not block this method.
- All reusable paper materials are collected and licensed (one caveat: the model-organisms repo LICENSE file is an incomplete 14-byte stub — confirm before redistributing those datasets/weights).

**Open / not established (do NOT assume):**
- Effect sizes are small at 0.5B and, for broad off-domain misbehavior, still weak at 1B (though coherence, readout separation, and steering specificity all improve at 1B). The papers' headline EM results are at 7B–14B. **Whether the containment-vs-generalization question is cleanly measurable likely requires a 7B+ organism**, which needs cloud (memory, not precision) — and no 1.5B/3B organism exists to close that gap locally.
- The fp16 clearance is for *this* method/scale only — re-verify if switching to projection/ablation or larger models.
- nnsight on MPS needs the CPU-load workaround; if the project leans on nnsight heavily, transformer-lens is the more reliable local option.

## Concrete cost-decision framing

- **Local M1 is enough to:** build and debug the full pipeline (prompt → in-context transgression → readout/steer across turns) at 0.5B–1B (the largest released organism ≤3B is Llama-3.2-1B), and validate all the salvaged harness code — at zero cost.
- **Rent cloud (e.g. one A100/H100 40–80 GB) when:** you need the 7B/14B organisms for real effect sizes. A single 14B fp16 organism (~28 GB) needs one 40 GB+ GPU; the released steering vectors and 7B/14B LoRAs are all there to use. Given the effects are scale-dependent, budget for this step before committing to any quantitative claim.

## Reproduce

- Env: `.venv` (Python 3.11); packages pinned in `findings/01-environment-tooling.md`.
- Scripts: `scripts/check1_smoke.py`, `check2_organism.py`, `check3_direction.py`, `check4_single.py`.
- Raw artifacts: `scripts/check2_outputs.json`, `scripts/prec_*.pt`.
- Materials + licenses: `materials/MANIFEST.md`.

**STOP.** This is a feasibility scope only — no methodology, experiments, or project design were produced, per the brief.
