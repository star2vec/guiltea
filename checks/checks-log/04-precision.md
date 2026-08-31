# Check 4 — fp16 / precision sanity

**Date:** 2026-08-29 · Model: Qwen2.5-0.5B-Instruct + `bad-medical-advice` organism · `scripts/check4_single.py`

## Question

The user has a documented past issue: *fp16 on the M1 weakened activation-steering effects.* Does the Check-3 diff-in-means readout + steering effect survive at the fp16 the M1/MPS uses, vs. higher precision? This is the direct input to the cloud-GPU decision.

## Method

Ran the identical Check-3 recipe (layer-12 diff-in-means, N=24 pairs) under three configs, saved each direction vector, and compared numerically:

- **fp16 / MPS** — what the M1 realistically runs
- **fp32 / MPS** — same device, isolates *precision*
- **fp32 / CPU** — reference, isolates *device* at fp32 (also the full run incl. organism readout)

(One model load per process — loading two models onto MPS in one process hangs on this machine; see note below.)

## Result

| config | dir_norm | readout_base | readout_org | delta | refusal uns→steer | cos vs fp32/CPU |
|---|---|---|---|---|---|---|
| fp16 / MPS | 1.2063 | 1.8541 | (n/a) | (n/a) | 0.50 → 0.00 | **0.999999** |
| fp32 / MPS | 1.2062 | 1.8527 | (n/a) | (n/a) | 0.50 → 0.00 | **1.000000** |
| fp32 / CPU | 1.2062 | 1.8527 | 2.5451 | **+0.692** | 0.50 → 0.00 | 1.000000 |

- **cos(fp16/MPS direction, fp32/MPS direction) = 0.999999**, identical norms, max elementwise diff 0.00025.
- Steering behavior is **identical** across all three (refusal 0.50 → 0.00 at scale 8).
- Organism readout elevation persists at fp32 (delta **+0.69**, matching Check 3's +0.57 at fp16/MPS N=64).

## Interpretation — which way it points

**For this setup (0.5B organism, diff-in-means, layer 12), fp16 does NOT weaken the effect.** fp16-on-MPS is numerically indistinguishable from fp32 — direction, readout, and steering all match. The documented past fp16 problem **did not reproduce here**.

Two honest caveats:
1. Tested only at **0.5B with diff-in-means**. The past issue may have involved a different model/technique (e.g. a larger model, or projection/ablation rather than additive steering, where fp16 rounding of a small residual component could matter more). This check clears fp16 for *this* method at *this* scale, not universally.
2. The effect being fp16-robust doesn't make the effect *large* — it's still modest at 0.5B (Check 3). Precision isn't the limiter; **model scale is**.

**Bottom line for the cloud decision:** precision is **not** a reason to pay for a cloud GPU. The direction-based instrument works fine in fp16 on the M1. The only reason to go to cloud is **memory/model size** — running the 7B+ organisms where the EM effects are strong and clean (Check 2 shows those don't fit in fp16 on 16 GB). This reframes the decision from "precision forces cloud" to "scale forces cloud."

## Note on MPS stability

Loading a **second** HF model onto MPS in the same process reliably hangs on this M1 (0% CPU, no error) — hit when loading base then organism sequentially. Workaround used throughout: one model per process, save results to disk, compare across files. Basic MPS matmul is unaffected. (Saved to project memory.)

## Verdict

**PASS** — the steering/readout effect fully survives fp16 on this M1 (cosine 0.999999 vs fp32, identical steering). Precision does **not** point toward cloud; only model scale does. Caveat: verified at 0.5B / diff-in-means only.
