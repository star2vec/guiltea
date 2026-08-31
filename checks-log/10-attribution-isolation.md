# Check 10 — Attribution isolation, guilt/shame separability, scale readiness

**Date:** 2026-08-30 · Llama-3.2-1B (base) · MPS/fp16 · `scripts/check10A.py`, `check10B.py` · **PRE-TIMER FEASIBILITY CHECK — reused base directions (`scripts/dirs_base_sweep.pt`), prompts/contrast-sets fixed as written, no GPT-4o (local refusal proxy).**

---

## 10A — Does the Setup B effect survive with corrective content held constant?

**Setup run:** Check-9 Setup B config (base, jailbreak→refuse→"book for a 10-year-old" + pancakes follow-ups). 2×2 attribution(action/self) × directive(present/absent) + NONE. 8 seeds (jailbreak refused 7/8 → treat n=14, control n=16 per arm). Readouts refusal/persona/badmed/random, random control alongside.

**Behavioral 2×2 — follow-up refusal/over-caution rate (treat arm):**
| | directive **+** | directive **−** |
|---|---|---|
| **ACTION** (act-attributed) | 0.29 | **0.14** |
| **SELF** (self-attributed) | 0.57 | 0.29 |
| NONE (baseline) | — | 0.71 |

(controls ≤0.06 except SELF+D control 0.38.)

- **Attribution locus has an effect with the directive held constant:** ACTION < SELF at both directive levels (+D: 0.29 vs 0.57; −D: 0.14 vs 0.29). Action-attribution releases the refusal carryover; self-attribution keeps it near the no-feedback baseline (SELF+D 0.57 ≈ NONE 0.71).
- **The directive does NOT carry the Check-9 effect.** Removing "just answer requests like that next time" *preserved or strengthened* the action-arm release (ACTION−D 0.14 ≤ ACTION+D 0.29). If anything the directive slightly *raised* refusal in both rows — a small, paradoxical, within-noise effect, not the release Check 9 attributed to it.
- Readout: refusal-axis treatment−control is robustly positive in every arm (L12 +0.47 to +1.27, margins 1.7–3.2; random ~−0.1 floor) but its magnitude does not cleanly rank with the behavioral rates (axis/behavior dissociate again) — so the behavioral 2×2 is the primary evidence.

**Verdict 10A: PASS.** The Check-9 SELF-vs-ACTION difference is driven by **attribution locus, not by instruction-following** — it survives holding corrective content constant. Rules in: attribution is doing real work. Caveat: modest n (14/arm); the directive's own effect is small and slightly paradoxical.

---

## 10B — Are guilt and shame separable directions at 1B?

**Setup run:** base model, diff-in-means (Check-7 lineage) at L6–14 over the assistant passage tokens, under a fixed shared user turn. 40 matched acts × 4 framings (GUILT / SHAME / NEUTRAL-NEG / BASELINE); directions = mean(set) − mean(BASELINE). Passage token-length ~ matched (guilt 36, shame 37; negneg 26, baseline 25 — the two attributed sets are longer than the controls; noted below).

**Geometry — cosines (representative L10–12):**
| pair | L6 | L8 | L10 | L12 | L14 |
|---|---|---|---|---|---|
| **guilt ~ shame** | .65 | .60 | .68 | .68 | .67 |
| guilt ~ neutral-neg | .67 | .60 | .54 | .50 | .48 |
| shame ~ neutral-neg | .51 | .43 | .45 | .44 | .38 |
| guilt ~ refusal | .10 | .14 | .16 | .18 | .14 |
| shame ~ refusal | .10 | .19 | .22 | .19 | .16 |
| guilt ~ persona | −.08 | .01 | −.06 | −.10 | −.04 |
| shame ~ persona | −.04 | .03 | −.01 | −.02 | −.00 |
| guilt ~ misalignment | .01 | −.09 | −.23 | −.39 | −.21 |
| shame ~ misalignment | .06 | .02 | −.13 | −.15 | −.12 |
| guilt/shame ~ random | ~.02 | ~.00 | ~.03 | ~−.02 | ~−.01 |

**Held-out linear probe (guilt vs shame, 5-fold CV, L2 logistic): AUROC = 1.000 ± 0.000 at every layer (chance 0.5).**

Reading against the given interpretation rules:
- **Distinct from the borrowed axes — YES.** guilt/shame have near-zero cosine with the **persona** axis (|cos| ≤ 0.10) and low cosine with **refusal** (≤ 0.22); guilt is *anti*-aligned with the misalignment axis at depth (−0.39 @L12). **So Check 9's operationalization (persona≈shame, refusal≈guilt) was measuring the wrong directions — its null was uninformative, as suspected.** They are also flat against random.
- **NOT cleanly distinct from negative valence — this is the catch.** Both load heavily on the **neutral-negative** (valence/mistake) direction: guilt ~0.5–0.67, shame ~0.38–0.51. A large share of each direction is generic "a mistake happened / negative," not attribution.
- **guilt and shame are correlated but not collinear** (~0.65). They don't collapse to one axis (would be ~0.95), nor are they orthogonal — the attribution-specific component is real but a *minority* of each direction.
- **The perfect probe is a template artifact, not deep-concept evidence.** AUROC 1.0 at *every* layer with only a 0.65 mean-direction cosine means within-class scatter is tiny — because the sets were built from two fixed templates differing in fixed wording ("mistake on my part / put it right" vs "what I am / kind of assistant I turn out to be"). The probe is separating template phrasing, largely lexical, not necessarily a guilt/shame concept. **Could not hold constant:** the attributed passages are ~10 tokens longer than the controls, and guilt/shame necessarily carry different orientation words (repair vs identity) — intrinsic to the constructs but also what a lexical probe can exploit.

**Verdict 10B: MIXED.** Purpose-built guilt/shame directions are **distinct from the axes Check 9 borrowed** (confirming that null was uninformative), and guilt-vs-shame passages are trivially linearly separable — **but** the two directions are substantially correlated (~0.65) and both are ~half generic negative valence, and the perfect probe is inflated by templated lexical differences. So at 1B, with this construction, attribution locus is **not cleanly isolated from valence**; the directions are separable in principle but not established as a clean, valence-independent guilt/shame readout. Rules out: treating these as ready-made, validated guilt/shame instruments. Rules in: they are their *own* thing, not re-derivations of refusal/persona/misalignment.

---

## 10C — Scale readiness (checklist, no model run)

**Organism availability (`ModelOrganismsForEM`, confirmed via HF API):** all four larger sizes are released as LoRA adapters (bad-medical / risky-financial / extreme-sports each), r=32:

| organism | base model (download) | adapter size | base fp16 weights (≈) |
|---|---|---|---|
| Qwen2.5-7B | `unsloth/Qwen2.5-7B-Instruct` | 323 MB | ~15 GB |
| Llama-3.1-8B | `unsloth/Llama-3.1-8B-Instruct` | 336 MB | ~16 GB |
| Qwen2.5-14B | `unsloth/Qwen2.5-14B-Instruct` | 551 MB | ~29 GB |
| Qwen2.5-32B | `unsloth/Qwen2.5-32B-Instruct` | 1.07 GB | ~65 GB |

Adapters are fp16/bf16; the base weights are the real download. 7B/8B/14B each fit one 40 GB A100; 32B needs an 80 GB card (or 2×40 GB / device_map="auto").

**Hardware-bound code (what changes for CUDA):** every run script hard-codes the same three things — `DEVICE="mps"`, `dtype=torch.float16`, `.to(DEVICE)` — e.g. `check10A.py:13,62`, `check10B.py:26,60`, `check9_setup{A,B}.py`, `check8_{extract,organism}.py:13/14`, `check7_calibration.py:22,138`, `check6*.py`, `check2b_1B.py:19,100`. MPS-specific calls: `torch.mps.current_allocated_memory()` (`check2b_1B.py:112`), `torch.mps.empty_cache()` (`check4_precision.py:101`). The one-model-per-process convention (`check8_organism.py:5`, `check9_setupA.py:9`) and the nnsight CPU-load→`.to('mps')` workaround (`findings/01`) exist **only** for the MPS second-load hang. **For CUDA:** set `DEVICE="cuda"` (or `device_map="auto"` for 32B), swap `torch.mps.*`→`torch.cuda.*`, and drop the one-model-per-process splitting (the hang is MPS-only). All direction-extraction and readout logic is device-agnostic. Precision is a non-issue (Check 4: fp16≈fp32).

**Rough wall-clock, one Check-9-shaped run** (5 arms × 8 seeds × 2 follow-ups + feedback/turn-1 gens ≈ 240 generations of ~100–150 tok, batch=1, single stream, plus ~160 prefill-only readouts). On this M1 at 1B it was ~20–25 min. Scaled by params × per-token cost, batch=1 HF `generate`:
- **7B on A100:** ~15–25 min · **on H100:** ~10–15 min.
- **14B on A100:** ~25–45 min · **on H100:** ~20–30 min.
- (32B on 80 GB A100/H100: ~1–1.5 h.)
These are single-stream estimates; batching follow-ups would cut them several-fold. Model load/download adds a one-time ~few min.

**Verdict 10C: PASS (ready).** All target organisms are available and small to fetch; the port to CUDA is a handful of device-string edits plus dropping an MPS-only workaround; a Check-9-shaped run is well under an hour at 7B–14B on one A100/H100.

---

## Combined: what 10A + 10B say about attribution locus at 1B

Taken together, attribution locus is a **real behavioral lever but not yet a clean readout** at 1B. **10A** shows the manipulation genuinely bites: action- vs self-attributed feedback changes downstream behavior (action releases the refusal carryover, self does not) *even with corrective content held constant*, so it is not an instruction-following artifact. **10B** shows the model does represent guilt- vs shame-framed text along directions that are **distinct from the refusal/persona/misalignment axes Check 9 borrowed** (vindicating the worry that Check 9's null was uninformative) — but those purpose-built directions are ~0.65 correlated with each other and ~0.5 aligned with plain negative valence, and their perfect probe separability is largely a templated-lexical artifact. So: attribution locus is behaviorally consequential and representationally *present*, but at 1B, with these fixed contrast sets, it is **not cleanly separable from generic negative valence** as an internal signal. No experiment design or next steps proposed — stopping per the brief.
