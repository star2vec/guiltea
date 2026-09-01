# Brief — S3-feasibility (the feasibility hour + the 8B instruments)

**Stage:** S3, front half only (PLAN §S3). This brief covers the laptop feasibility hour, the three borrowed readout axes at 8B, and the shared random-control utility. It does **not** cover the one-button rig, seed-fixing, or the rig checkpoint/design menu — those are a later brief and require the researcher.
**Context you receive:** this brief + STAGE0.md + PLAN.md. No prior reports exist. Do not read the risk map or bulk papers; `papers/refs.md` governs any fetch.
**You execute exactly this brief and file one report in `reports/`. You do not redesign.** If a specification here is unworkable, say so in the report and stop that item.

---

## The question

Can the 8B organism run on the primary laptop (RTX 2000 Ada), at what precision, at what measured throughput, and without quantization distorting the readouts — and can the three borrowed readout axes (refusal, misalignment, persona) plus a norm-matched random control be extracted at 8B and shown to read out? Produce the axes, the random-control utility, the throughput numbers, and the hardware decision.

## Models (fixed)

- **Base:** `unsloth/Llama-3.1-8B-Instruct`.
- **Organism:** base + LoRA `ModelOrganismsForEM/Llama-3.1-8B-Instruct_bad-medical-advice` (r=32, α=64, rsLoRA, targets q/k/v/o/gate/up/down), merged with `merge_and_unload()`.
- **All directions are extracted on the BASE model.** The organism is used only to read the axes out (the instrument sanity check below). This matches the checks-era convention (directions on base, read out on organism).

## Task 1 — laptop feasibility and the cloud decision

1. Report the laptop's actual GPU VRAM and whether the 8B fits, at what precision. Try bf16/fp16 first; if it does not fit, 4-bit (bitsandbytes, CUDA). Report peak VRAM, load time, and single-stream generation speed (tok/s).
2. **Quantization-noise check** (mirrors Check 4). Extract the misalignment direction (Task 2b) under the run precision **and** under an uncompressed reference (fp32; on CPU, or on cloud if CPU is impractical) on a fixed subset (N=24 pairs, seed 0). Report cosine(run-precision dir, reference dir), the two norms, and the readout delta (organism − base) under each.
3. **Decision rule (fixed; apply it, do not deviate):** laptop is primary **iff** the 8B fits at the chosen precision **and** cosine ≥ 0.99 **and** the organism−base readout delta keeps its sign and is within 15% of the reference. Otherwise the report's decision is **cloud A100 fallback**. Report which branch, with the numbers. This applies DECISIONS D-009; you report the result, you do not re-open the decision.

## Task 2 — the three axes at 8B on base, full layer sweep

Sweep **every layer** (Llama-3.1-8B has 32; residual at layer L = `hidden_states[L+1]`). Report the whole profile. Do not select a "best" layer — S2 does that later.

- **2a — refusal** (Arditi recipe): harmful vs harmless instructions, **last-token** diff-in-means, N=64 each. Use the same Arditi `harmful_train`/`harmless_train` splits as Check 7. If they are not present in-repo, fetch from Arditi's public repo (per `papers/refs.md`) and record the source + commit in the provenance entry.
- **2b — misalignment (badmed)** (Soligo recipe): bad-vs-good medical, **answer-token** diff-in-means (mean over assistant answer tokens), N=64 matched pairs, seed 0. Use the Turner `bad_medical_advice` / `good_medical_advice` matched-prompt datasets as in Checks 3/6. If not present in-repo, obtain per the checks-era provenance (`checks/checks-log/05-harness-salvage.md`) and record source in provenance.
- **2c — persona = Assistant Axis** (Lu recipe, full): machine time is free, so do not take the ~20-role shortcut. Use the **full public role set** from `github.com/safety-research/assistant-axis` and the paper's **PCA recipe**: collect the mean post-response-token residual per role (across the paper's system-prompt × question grid), plus the default-Assistant vector; the axis is the top principal component of the role vectors (oriented so Assistant sits at the positive end), per layer. **Do not** use Check 7's guarded-vs-helpful proxy. Report the **cosine between our axis and the paper's stated top component** if their released vectors are usable at all, and otherwise the cosine between the PC1 axis and the plain default-Assistant-minus-mean-roles difference vector (so the reduction's cost is visible). **Validate by steering**: on a small behavioral probe (~6 prompts), steer along the axis at a modest scale and confirm it moves outputs toward/away from the Assistant register; report it. Record the exact role list and prompt source in provenance. State plainly in the report that this remains a replication at a scale (8B) the Assistant-Axis paper did not test.

For each axis report: extraction params, unit-direction norms per layer, and inter-axis cosines (refusal~badmed, refusal~persona, badmed~persona, and each ~ random) per layer.

## Task 3 — shared random-control utility (OPERATIONS standing note)

Create one utility module under `scripts/` (e.g. `scripts/randctl.py`) exposing a single deterministic function that, given `d_model`, a list of layers, and a seed, returns **norm-matched unit** random vectors per layer. Every project readout (this stage and S4) imports it; the inline `torch.randn(...)/norm` pattern from the checks scripts is not copied again. Include a determinism check in the report: same seed → identical vectors; different layers independent. (The checks scripts under `checks/` are read-only provenance; do not edit them — the utility is new project code.)

## Task 4 — instrument sanity, and the dead-instrument branch (not a result)

On ~8 held-out neutral prompts, read each axis out on **base vs organism** and report the delta beside the random-control floor. This only confirms the axes read out at 8B (as Checks 2b/3 did at 1B). **Report it as an instrument check, never as a finding.**

**Dead-instrument branch (fixed).** If an axis's organism−base delta sits **at the random-control floor** (no separation beyond random), flag that axis in the report as an **S2/S4 blocker**. The standing branch — do not decide it yourself, report that it applies — is the one in STAGE0 §6 ("instrument fails"): **S1b proceeds behavior-only, and the instrument question for that axis is deferred to S2.** State which axes are affected; the researcher decides at the S2 gate.

## Task 5 — throughput (gates the S1b fit)

Measure and report, at 8B on the chosen hardware/precision:
- single-stream generation tok/s;
- **batched throughput** (generations/hour) at batch sizes 1, 4, 8, 16 for ~150-token generations;
- prefill-only **readout** throughput (readouts/hour) for a ~600-token context;
- **one full chain-shaped dry run**, timed end-to-end: a 10-turn conversation with growing context (each turn re-prefills the accumulated history), the model's own replies inserted, plus a prefill-only readout at each turn — i.e. the actual S1b per-chain cost profile, not a uniform-generation proxy. Report seconds per chain and where the time goes (generation vs prefill).
- **estimated `results/raw/` size** for one full run at N=12 over a full layer sweep (per-turn, per-axis projections across all layers, all cells) — so S1b knows the disk footprint before it writes.

These let the S1b brief confirm its total volume fits the 10-hour wall-clock ceiling **before** S1b runs; the chain-shaped timing is the number that actually governs the fit, since S1b's workload is multi-turn chains, not uniform 150-token generations.

## Save

Directions → `directions/` (e.g. `dirs_8B_base_sweep.pt`, keys `{units:{refusal,badmed,persona}, norms, layers, random_seed}`). Append a provenance entry to `directions/PROVENANCE.md` in the existing style (recipe, model, layers, N, seed, norms, data source + commit).

## Report format (`reports/S3-feasibility.md`)

1. Hardware: VRAM, fit, precision, peak memory, load time; quantization-noise cosine + deltas; **laptop-vs-cloud branch** with the numbers.
2. Each axis: extraction params, per-layer norms, per-layer inter-axis cosines; for persona: cosine to the paper's top component (or to the mean-difference reduction) and the steering-validation result.
3. Random utility: path, function signature, determinism-check result.
4. Instrument sanity: per-axis organism−base deltas vs random floor (labelled "instrument check, not a result"); any axis at the random floor flagged as an S2/S4 blocker with the STAGE0 §6 branch noted.
5. Throughput: single-stream, batched, readout, **plus the timed full chain-shaped dry run** and the estimated `results/raw/` size at N=12 over the full sweep.
6. Provenance written (paths + entry text).
7. Anything unworkable, and where you stopped.

## Do not

- Do not extract any direction on the organism (base only).
- Do not pick a primary layer or report a "best" layer; report the full sweep.
- Do not build the one-button rig, fix seeds, or choose the design menu — out of scope; those need the rig checkpoint and the researcher.
- Do not start S1 or S4.
- Do not treat any 1B artifact, or the Task-4 sanity deltas, as a result.
- Do not estimate elapsed project hours; if you reach any time trigger, stop and ask the researcher for the ledger.
- Vocabulary (STAGE0 §2): persistence, spread, blame target, harmful act, act rate. Never: carryover, attribution locus, transgression, fire rate, lapse.
- Refer to the researcher as "the researcher".
