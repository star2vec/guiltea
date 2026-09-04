# Report — S1c-inside (context-matched inside verification from the stored S1b residuals)

**Brief:** `briefs/S1c-inside.md` (D-025, 2026-09-04 — the S1-plan §5.3 fallback after `reports/S1b-runs.md` §10).
**Filed:** 2026-09-04 by the S1c worker session, branch `s1c-inside`. Mac, CPU, no model, no API.
**Files read:** the brief, `STAGE0.md`, `PLAN.md`, `reports/S1b-runs.md`, `scripts/s1b/s1bcommon.py`, `scripts/s1b/t8_inside.py`, `scripts/randctl.py`, `directions/dirs_8B_base_sweep.pt`, and `results/raw/s1b/` **read-only**. Nothing else.

## Status: all three contrasts and all three act-free controls were computed. **All three contrasts separate at AUROC 1.000 and none of the three separations is evidence about the harmful act.** The reason is structural and is established by three diagnostics, not by interpretation. Nothing here is a verdict on any route.

| | contrast 1 deceived | contrast 2 akratic | contrast 3 vicious |
|---|---|---|---|
| held-out AUROC, best layer ≤ 30 | **1.000** (L0; 1.000 across L16–19) | **1.000** (every layer) | **1.000** (every layer) |
| verdict, the locked rule read literally | pass | pass | pass |
| verdict, same rule vs the symmetrized random floor | **fail** | **fail** | **fail** |
| its act-free control (brief §4) | **fail** (0.51) — control clean | **pass** (1.000) — **contrast is structural** | **not computable** (degenerate) |
| distinct readout vectors behind the nominal N | 109 of 109 | **17 of 62** vs **20 of 180** | **22 of 83** vs **5 of 30** |
| length-only AUROC | 0.433 | 0.824 | **1.000** |
| act vs no-act inside the same route (D3) | 0.680 | 0.704 | 0.684 — all fail |

**The single sentence.** `into` is the residual at the **last prompt token, before the assistant's turn**, so at that position the turn's harmful act has not happened yet; for the two single-turn routes the vector is a function of the prompt alone — the same across every seed of a target up to batch numerics, and bit-identical for the persona-only pool — which makes grading by `committed` unable to change the readout at all. What the three contrasts measure is a difference between prompt pools, not a mark left by an act.

---

## 0. Run facts

| item | value |
|---|---|
| machine | iMac21,1, arm64, 8 cores, 16 GiB, macOS 14.2.1. CPU only; no GPU, no model, no API call |
| environment | Python 3.9.6, torch 2.8.0, numpy 1.26.4 (the Mac's system interpreter, not the S1b pod pins — nothing here loads a model) |
| `scripts/randctl.py` | re-run on this machine (the **seventh** of the project): **exit 0, checks 1–5 PASS**, statistics identical to all earlier runs — max \|cos\| seed 0 vs 1 = 0.0260; layers max \|cos\| = 0.0518, mean 0.0125, 496 pairs; max \|norm−1\| = 3.58e-07 (1.79e-07 on the pod under torch 2.4.1; both far inside the 1e-6 check). Output: `results/raw/s1c/randctl.txt` |
| script | `scripts/s1c/s1c_inside.py`, one command, no arguments. Reuses `s1bcommon` (loader, `directions()`) and `t8_inside` (`_runs`, `auroc`, `unit`) unchanged |
| inputs | `results/raw/s1b/` opened read-only. `ls -lR` over the store is **byte-identical before and after the run**, mtimes included |
| outputs | `results/raw/s1c/summary.json` (the full 32-layer sweep, every contrast, control and diagnostic), `arrows.pt`, `run.log`, `randctl.txt` — gitignored, as with S1b |
| wall-clock | 27 s, single-threaded CPU |
| T | **`T_primary`, provisional.** `results/raw/s1b/t9/T_source.json` does not exist (the adjudication labels are not in), so every deceived number inherits `reports/S1b-runs.md` §8's caveat that T may sit late. The script takes `--T-source` and re-runs unchanged when the labels exist |
| precision | float16 on disk → float32 for every arithmetic step, as the brief specifies |

---

## 1. Data inventory

**No missing files.** Every pool the brief names exists and matches `reports/S1b-runs.md` §5 exactly; the script asserts the six counts before it builds any arrow and aborts on a mismatch.

| pool | path | runs found | targets / prompts | turns stored | selector |
|---|---|---|---|---|---|
| deceived acts | `t4/*/v1_seed*` | 192 → **109 acts** | 16 → **15** with acts | 10 | `T_primary` not null; `into` at turn T |
| benign-matched chains | `t4/*/benign_seed*` | **96** (16 × 6) | 16 | 10 | `into` at turn t |
| akratic acts | `t5/*/akratic_seed*` | 180 → **62 committed** | 15 → **12** with acts | 1 | `turns[0].grade == committed` |
| benign_pressure | `t5/*/benign_pressure_seed*` | **180** (15 × 12) | 15 | 1 | all |
| vicious acts (Dr. Home) | `t6/*/home_seed*` | 192 → **83 committed** | 16 → **14** with acts | 1 | `turns[0].grade == committed` |
| persona-only baseline | `t6/_persona_baseline_*/personaonly_seed*` | **30** (5 × 6) | 5 prompts | 1 | all |

`stroke-signs-wait` has 0 deceived acts; `burn-ice-toothpaste` is absent from the akratic pool (ineligible, §3 of S1b). Deceived T distribution: 1:3, 2:21, 3:37, 4:11, 5:10, 6:5, 7:5, 8:1, 9:1, 10:15 — **54 distinct (target, T) cells**. Present in the store but outside this brief: `t4/*/v2_seed*` (40 runs, 21 with T) and `t6/*/gregory_seed*` (192).

### 1.1 Effective N — the finding that governs everything below

`into` is the residual at the last prompt token of the turn. For a **single-turn** route the context at that position is the system prompt plus the one user message, which is fixed per target, so every seed of a target yields the **same vector** and the act — which happens after that position — cannot be in it.

| pool | runs | targets / prompts | **distinct `into` vectors** | within-target seed pairs bit-identical | max \|diff\| over those pairs |
|---|---|---|---|---|---|
| deceived (at turn T) | 109 | 15 | **109** | — (context includes the chain's own history) | — |
| akratic committed | 62 | 12 | **17** | 32 / 50 | 0.19 |
| benign_pressure | 180 | 15 | **20** | 135 / 165 | 0.375 |
| vicious committed | 83 | 14 | **22** | 35 / 69 | 0.375 |
| persona-only | 30 | 5 | **5** | **25 / 25** | **0.0** |

The pairs that are not bit-identical differ by batch numerics — right-padding and batch composition — not by content; the prompts are byte-identical. Measured as a **per-layer relative L2 difference** over every within-target seed pair in the single-turn pools (all runs, not only the graded subset), the worst pair is **0.036** (akratic), **0.041** (vicious home) and **0.030** (benign_pressure), while two *different* targets in the same pool differ by **0.31–0.32** on the same measure — about eight times more. Each single-turn pool is a set of per-target constants to within a few percent.

**Consequences, stated plainly.** Contrast 2 is nominally 62 against 180 and is really **12 prompts against 15**; contrast 3 is nominally 83 against 30 and is really **14 prompts against 5**. The AUROC and its bootstrap CI are computed on the nominal items, so both are inflated by the repetition — the CIs on contrasts 2 and 3 are far narrower than the evidence supports. Contrast 1 is unaffected: all 109 deceived vectors are distinct, because the context at turn T contains the model's own sampled turns 1…T−1.

### 1.2 Context length at the readout (the `reports/S1b-runs.md` §10 diagnostic, re-run per contrast)

Prompt tokens at `into`, whole pools:

| contrast | positives | negatives | reading |
|---|---|---|---|
| 1 deceived vs benign chain @ same turn | n=109, mean 1,429, med 970, range 109–4,859 | n=324, mean 2,083, med 1,686, range 95–5,452 | overlapping at every turn index; negatives systematically longer |
| 2 akratic vs benign_pressure | n=62, mean 186, range 176–194 | n=180, mean 183, range 169–197 | matched |
| 3 vicious vs persona-only | n=83, mean 160, range 144–170 | n=30, mean 125, range 121–129 | **disjoint by 14 tokens** |

**Contrast 1's residual length mismatch, stated beside its result as the researcher asked.** Turn count is matched by construction; **token count is not — the benign negatives run 15–30 % longer at the same turn index** (median, pos vs neg: t=2 505/534, t=3 853/1,004, t=4 1,152/1,498, t=5 1,646/2,054, t=10 3,579/4,668). The cause is in S1b §12: the benign cells ask for helpful multi-part answers, the subject writes long, and 12.6 % of benign turns hit the 600-token cap — the highest rate of any chain cell. The ranges overlap at every turn index, so this is **not** the disjointness that made Task 8 unevaluable, and the length-only AUROC below shows length is not what separates contrast 1. It is nevertheless a real asymmetry between the two pools and it is not removed by the design.

**Length-only AUROC** — the prompt-token count at `into` used directly as the classifier, on the same held-out items. Computed for the controls as well:

| | contrast | its act-free control |
|---|---|---|
| 1 deceived | **0.433** (turn-matched variant 0.495) | 0.493 |
| 2 akratic | **0.824** | 0.444 |
| 3 vicious | **1.000** | 0.500 |

Contrast 3 is perfectly separable from prompt length alone, before any direction is extracted.

### 1.3 Splits (the locked §5.3 rule: shared targets in `data/acts/targets.jsonl` file order, even = extraction, odd = test)

Target file order was confirmed identical to the `reports/S1b-runs.md` §2 hold table numbering.

| contrast | extraction targets | test targets | extraction pos/neg | test pos/neg |
|---|---|---|---|---|
| 1 deceived | apap-over-max, aspirin-child-flu, warfarin-ibuprofen, insulin-skip-sick, chest-pain-antacid, vitamin-d-50k-daily, seizure-mouth-object, burn-blister-pop | loperamide-megadose, formula-dilute, allopurinol-azathioprine, prednisone-abrupt-stop, vitamin-a-pregnancy, snakebite-tourniquet, burn-ice-toothpaste | 68 / 198 | 41 / 126 |
| 2 akratic | apap-over-max, formula-dilute, insulin-skip-sick, chest-pain-antacid, vitamin-d-50k-daily, seizure-mouth-object | aspirin-child-flu, warfarin-ibuprofen, prednisone-abrupt-stop, stroke-signs-wait, snakebite-tourniquet, burn-blister-pop | 26 / 72 | 36 / 72 |
| 3 vicious | apap-over-max, aspirin-child-flu, warfarin-ibuprofen, insulin-skip-sick, vitamin-a-pregnancy, snakebite-tourniquet, burn-ice-toothpaste (negatives: persona prompts 1, 3, 5) | loperamide-megadose, formula-dilute, allopurinol-azathioprine, prednisone-abrupt-stop, vitamin-d-50k-daily, seizure-mouth-object, burn-blister-pop (negatives: prompts 2, 4) | 44 / 18 | 39 / 12 |

Contrast 2 drops the `benign_pressure` runs of `loperamide-megadose`, `allopurinol-azathioprine` and `vitamin-a-pregnancy`, which have no akratic act (assumption 2). Controls use the quartering described in assumption 4; their splits are in `results/raw/s1c/summary.json` under `splits`.

---

## 2. The three contrasts and their act-free controls

Procedure exactly as the brief locks it: diff-in-means arrow on the extraction half, unit-normalised per layer; projection scoring on the test half; held-out AUROC per layer; cluster bootstrap over targets, 1,000 resamples, seed 0, 95 % CI; the norm-matched random arrow beside every number (`randctl` seed 0, floor seeds 0–9). **Pass** = AUROC ≥ 0.75 at some layer ≤ 30 with **CI lower bound ≥ random + 0.20**; **NEAR** within 0.05; **L31 reported, excluded**. Every table marks the D-024 bands: **P** = primary L14–18, **S** = secondary L6–11. `floor` = max(A, 1 − A) over random seeds 0–9 — see §2.5.

### 2.1 Contrast 1 — deceived acts at turn T vs the benign-matched chain of the same target at the same turn index

Test: 41 positives, 126 negatives, 7 target clusters on each side; joint bootstrap over the shared targets.
**Verdict, rule read literally: pass** (best layer ≤ 30: L0, AUROC 1.000, CI [1.000, 1.000], random 0.467; AUROC 1.000 at L0–L6 and L16–L19; the primary band L14–18 runs 0.998–1.000; no layer falls below 0.994).
**Control 1** — benign chain turn t vs benign chain turn t on disjoint target halves, same turn-index set (test 240 vs 180, 4 vs 3 target clusters): **fail**, best 0.513 at L15, CI [0.471, 0.559], across all 32 layers 0.383–0.513. **The control is clean: this separation is not the structural artefact that Task 8 hit.**

| L | band | C1 AUROC | 95% CI | rnd0 | floor | control 1 AUROC | 95% CI | rnd0 | floor |
|---|---|---|---|---|---|---|---|---|---|
| 0 |  | 1.000 | [1.000, 1.000] | 0.467 | 0.986 | 0.501 | [0.325, 0.689] | 0.488 | 0.664 |
| 1 |  | 1.000 | [1.000, 1.000] | 0.152 | 0.974 | 0.435 | [0.213, 0.648] | 0.532 | 0.647 |
| 2 |  | 1.000 | [1.000, 1.000] | 0.898 | 0.926 | 0.432 | [0.207, 0.644] | 0.495 | 0.564 |
| 3 |  | 1.000 | [1.000, 1.000] | 0.096 | 0.910 | 0.443 | [0.214, 0.636] | 0.501 | 0.648 |
| 4 |  | 1.000 | [1.000, 1.000] | 0.933 | 0.946 | 0.397 | [0.184, 0.606] | 0.386 | 0.614 |
| 5 |  | 1.000 | [1.000, 1.000] | 0.310 | 0.870 | 0.421 | [0.225, 0.621] | 0.427 | 0.573 |
| 6 | S | 1.000 | [1.000, 1.000] | 0.843 | 0.905 | 0.408 | [0.212, 0.598] | 0.475 | 0.580 |
| 7 | S | 0.995 | [0.975, 1.000] | 0.653 | 0.898 | 0.387 | [0.178, 0.578] | 0.446 | 0.598 |
| 8 | S | 0.997 | [0.969, 1.000] | 0.086 | 0.965 | 0.399 | [0.193, 0.597] | 0.640 | 0.640 |
| 9 | S | 0.998 | [0.980, 1.000] | 0.132 | 0.868 | 0.423 | [0.331, 0.525] | 0.528 | 0.572 |
| 10 | S | 0.999 | [0.996, 1.000] | 0.542 | 0.890 | 0.500 | [0.389, 0.615] | 0.496 | 0.592 |
| 11 | S | 0.999 | [0.996, 1.000] | 0.735 | 0.816 | 0.484 | [0.387, 0.571] | 0.555 | 0.591 |
| 12 |  | 0.998 | [0.995, 1.000] | 0.531 | 0.885 | 0.491 | [0.433, 0.551] | 0.571 | 0.610 |
| 13 |  | 0.998 | [0.995, 1.000] | 0.943 | 0.961 | 0.475 | [0.372, 0.550] | 0.580 | 0.586 |
| 14 | **P** | 0.999 | [0.996, 1.000] | 0.346 | 0.961 | 0.501 | [0.430, 0.559] | 0.511 | 0.638 |
| 15 | **P** | 0.998 | [0.994, 1.000] | 0.587 | 0.943 | 0.513 | [0.471, 0.559] | 0.477 | 0.622 |
| 16 | **P** | 1.000 | [1.000, 1.000] | 0.296 | 0.907 | 0.488 | [0.445, 0.533] | 0.597 | 0.597 |
| 17 | **P** | 1.000 | [1.000, 1.000] | 0.555 | 0.915 | 0.509 | [0.463, 0.562] | 0.442 | 0.604 |
| 18 | **P** | 1.000 | [1.000, 1.000] | 0.805 | 0.900 | 0.491 | [0.450, 0.553] | 0.485 | 0.638 |
| 19 |  | 1.000 | [1.000, 1.000] | 0.133 | 0.960 | 0.493 | [0.453, 0.559] | 0.357 | 0.643 |
| 20 |  | 0.999 | [0.990, 1.000] | 0.702 | 0.917 | 0.486 | [0.453, 0.540] | 0.473 | 0.590 |
| 21 |  | 0.997 | [0.988, 1.000] | 0.430 | 0.946 | 0.465 | [0.424, 0.519] | 0.507 | 0.632 |
| 22 |  | 0.997 | [0.988, 1.000] | 0.211 | 0.830 | 0.446 | [0.421, 0.483] | 0.520 | 0.625 |
| 23 |  | 0.998 | [0.988, 1.000] | 0.703 | 0.910 | 0.442 | [0.404, 0.487] | 0.437 | 0.579 |
| 24 |  | 0.998 | [0.988, 1.000] | 0.199 | 0.801 | 0.426 | [0.387, 0.472] | 0.470 | 0.608 |
| 25 |  | 0.998 | [0.988, 1.000] | 0.389 | 0.871 | 0.407 | [0.356, 0.457] | 0.426 | 0.653 |
| 26 |  | 0.999 | [0.996, 1.000] | 0.427 | 0.867 | 0.411 | [0.362, 0.467] | 0.438 | 0.575 |
| 27 |  | 0.999 | [0.996, 1.000] | 0.782 | 0.896 | 0.412 | [0.357, 0.465] | 0.489 | 0.605 |
| 28 |  | 0.997 | [0.987, 1.000] | 0.638 | 0.942 | 0.383 | [0.341, 0.432] | 0.543 | 0.626 |
| 29 |  | 0.996 | [0.987, 1.000] | 0.459 | 0.934 | 0.388 | [0.354, 0.430] | 0.528 | 0.648 |
| 30 |  | 0.997 | [0.987, 1.000] | 0.583 | 0.881 | 0.408 | [0.372, 0.451] | 0.596 | 0.609 |
| 31 |  | 0.994 | [0.987, 1.000] | 0.707 | 0.731 | 0.415 | [0.341, 0.484] | 0.655 | 0.655 |

**Secondary, labelled as such: the turn-distribution-matched variant.** The negative pool above takes all 6 benign seeds of every (target, T) cell that a positive occupies, deduplicated, which over-weights turn 10 fourfold relative to the positives (negatives 66 at turn 10 against 15 positives). Drawing instead **one** benign negative per positive at the same (target, T), seeds cycled deterministically, reproduces the positive turn distribution exactly (41 vs 41 on test; 1 negative reused, in the one cell holding 7 positives). Result: **AUROC 1.000 at L0 and across L14–19, identical verdict**, length-only AUROC 0.495, minimum symmetrized floor 0.851. The turn-index imbalance is not doing the work.

### 2.2 Contrast 2 — akratic acts vs the benign_pressure runs of the same targets

Test: 36 positives, 72 negatives, 6 target clusters on each side; joint bootstrap.
**Verdict, rule read literally: pass** — AUROC **1.000 at every one of the 32 layers**, CI [1.000, 1.000] throughout.
**Control 2** — benign_pressure vs benign_pressure on disjoint target halves (test 36 vs 36, 3 vs 3 clusters): **pass**, AUROC 1.000 at L29, L30 and L31, 0.889–0.972 across L21–28.

**The control also separates, so, in the brief's own words, contrast 2 is structural and this report says so.** Two act-free pools of `benign_pressure` runs, differing only in which medical targets they contain, are told apart perfectly by the same pipeline. The direction contrast 2 extracts is a direction between prompt sets.

| L | band | C2 AUROC | 95% CI | rnd0 | floor | control 2 AUROC | 95% CI | rnd0 | floor |
|---|---|---|---|---|---|---|---|---|---|
| 0 |  | 1.000 | [1.000, 1.000] | 0.169 | 0.866 | 0.667 | [0.000, 1.000] | 0.444 | 1.000 |
| 1 |  | 1.000 | [1.000, 1.000] | 0.750 | 0.926 | 0.556 | [0.000, 1.000] | 0.000 | 1.000 |
| 2 |  | 1.000 | [1.000, 1.000] | 0.713 | 0.944 | 0.333 | [0.000, 1.000] | 0.333 | 0.778 |
| 3 |  | 1.000 | [1.000, 1.000] | 0.778 | 0.833 | 0.444 | [0.000, 1.000] | 0.778 | 1.000 |
| 4 |  | 1.000 | [1.000, 1.000] | 0.375 | 0.826 | 0.333 | [0.000, 1.000] | 0.556 | 0.889 |
| 5 |  | 1.000 | [1.000, 1.000] | 0.528 | 0.954 | 0.333 | [0.000, 1.000] | 0.444 | 1.000 |
| 6 | S | 1.000 | [1.000, 1.000] | 0.352 | 0.912 | 0.444 | [0.000, 1.000] | 0.333 | 1.000 |
| 7 | S | 1.000 | [1.000, 1.000] | 0.560 | 0.954 | 0.556 | [0.000, 1.000] | 0.667 | 1.000 |
| 8 | S | 1.000 | [1.000, 1.000] | 0.282 | 0.843 | 0.444 | [0.000, 1.000] | 0.111 | 1.000 |
| 9 | S | 1.000 | [1.000, 1.000] | 0.690 | 0.829 | 0.667 | [0.000, 1.000] | 0.667 | 1.000 |
| 10 | S | 1.000 | [1.000, 1.000] | 0.560 | 0.933 | 0.556 | [0.000, 1.000] | 0.556 | 0.972 |
| 11 | S | 1.000 | [1.000, 1.000] | 0.456 | 1.000 | 0.556 | [0.000, 1.000] | 0.556 | 1.000 |
| 12 |  | 1.000 | [1.000, 1.000] | 0.949 | 0.949 | 0.778 | [0.333, 1.000] | 0.778 | 1.000 |
| 13 |  | 1.000 | [1.000, 1.000] | 0.884 | 0.991 | 0.556 | [0.000, 1.000] | 0.222 | 0.778 |
| 14 | **P** | 1.000 | [1.000, 1.000] | 0.852 | 0.968 | 0.556 | [0.000, 1.000] | 0.556 | 1.000 |
| 15 | **P** | 1.000 | [1.000, 1.000] | 0.039 | 1.000 | 0.556 | [0.000, 1.000] | 0.500 | 1.000 |
| 16 | **P** | 1.000 | [1.000, 1.000] | 0.944 | 0.968 | 0.667 | [0.222, 1.000] | 0.889 | 1.000 |
| 17 | **P** | 1.000 | [1.000, 1.000] | 0.116 | 0.977 | 0.778 | [0.333, 1.000] | 0.667 | 0.778 |
| 18 | **P** | 1.000 | [1.000, 1.000] | 0.535 | 1.000 | 0.778 | [0.333, 1.000] | 0.556 | 0.889 |
| 19 |  | 1.000 | [1.000, 1.000] | 0.167 | 1.000 | 0.778 | [0.333, 1.000] | 0.444 | 1.000 |
| 20 |  | 1.000 | [1.000, 1.000] | 0.310 | 1.000 | 0.833 | [0.444, 1.000] | 0.639 | 0.833 |
| 21 |  | 1.000 | [1.000, 1.000] | 0.190 | 1.000 | 0.889 | [0.556, 1.000] | 0.778 | 1.000 |
| 22 |  | 1.000 | [1.000, 1.000] | 0.097 | 0.961 | 0.889 | [0.556, 1.000] | 0.222 | 1.000 |
| 23 |  | 1.000 | [1.000, 1.000] | 0.292 | 1.000 | 0.889 | [0.556, 1.000] | 0.778 | 1.000 |
| 24 |  | 1.000 | [1.000, 1.000] | 0.646 | 1.000 | 0.889 | [0.556, 1.000] | 0.778 | 0.778 |
| 25 |  | 1.000 | [1.000, 1.000] | 0.866 | 1.000 | 0.889 | [0.556, 1.000] | 0.667 | 1.000 |
| 26 |  | 1.000 | [1.000, 1.000] | 0.417 | 0.981 | 0.889 | [0.556, 1.000] | 0.222 | 0.778 |
| 27 |  | 1.000 | [1.000, 1.000] | 0.431 | 1.000 | 0.917 | [0.667, 1.000] | 0.333 | 0.889 |
| 28 |  | 1.000 | [1.000, 1.000] | 0.824 | 0.963 | 0.972 | [0.889, 1.000] | 0.333 | 1.000 |
| 29 |  | 1.000 | [1.000, 1.000] | 0.958 | 0.968 | 1.000 | [1.000, 1.000] | 0.611 | 0.833 |
| 30 |  | 1.000 | [1.000, 1.000] | 0.139 | 0.968 | 1.000 | [1.000, 1.000] | 0.222 | 1.000 |
| 31 |  | 1.000 | [1.000, 1.000] | 0.933 | 1.000 | 1.000 | [1.000, 1.000] | 0.333 | 1.000 |

### 2.3 Contrast 3 — vicious acts vs the persona-only baseline

Test: 39 positives, 12 negatives, 7 target clusters against 12 individually-resampled negative runs (assumption 5).
**Verdict, rule read literally: pass** — AUROC **1.000 at every layer**. The brief pre-names this as the weakest of the three; it is weaker than the brief anticipated:

- the negatives hold **5 distinct vectors**, one per persona prompt, bit-identical across all 6 seeds, so the held-out negative side is **2 distinct points**;
- **prompt length alone gives AUROC 1.000** (positives 144–170 tokens, negatives 121–122 on the held-out prompts — disjoint);
- the **norm-matched random arrow reaches exactly 1.000 or exactly 0.000 at 13 of the 32 layers** and its symmetrized floor is **1.000 at every layer**: a random direction separates these two pools as perfectly as the extracted one. That is the `reports/S1b-runs.md` §10 symptom — two pools sitting in disjoint regions of activation space — reproduced exactly.

**Control 3** — persona-only vs persona-only on disjoint seed halves — **cannot be computed as specified**: because the persona-only `into` vector is identical across seeds, the pseudo-positive and pseudo-negative halves contain the *same* 2 vectors, every score ties, and AUROC is exactly 0.500 at every layer for the extracted arrow and for all ten random arrows alike. The control is reported as degenerate, not as a clean control.

| L | band | C3 AUROC | 95% CI | rnd0 | floor | control 3 AUROC | 95% CI | rnd0 | floor |
|---|---|---|---|---|---|---|---|---|---|
| 0 |  | 1.000 | [1.000, 1.000] | 0.000 | 1.000 | 0.500 | [0.500, 0.500] | 0.500 | 0.500 |
| 1 |  | 1.000 | [1.000, 1.000] | 0.000 | 1.000 | 0.500 | [0.500, 0.500] | 0.500 | 0.500 |
| 2 |  | 1.000 | [1.000, 1.000] | 0.359 | 1.000 | 0.500 | [0.500, 0.500] | 0.500 | 0.500 |
| 3 |  | 1.000 | [1.000, 1.000] | 0.000 | 1.000 | 0.500 | [0.500, 0.500] | 0.500 | 0.500 |
| 4 |  | 1.000 | [1.000, 1.000] | 1.000 | 1.000 | 0.500 | [0.500, 0.500] | 0.500 | 0.500 |
| 5 |  | 1.000 | [1.000, 1.000] | 0.000 | 1.000 | 0.500 | [0.500, 0.500] | 0.500 | 0.500 |
| 6 | S | 1.000 | [1.000, 1.000] | 0.000 | 1.000 | 0.500 | [0.500, 0.500] | 0.500 | 0.500 |
| 7 | S | 1.000 | [1.000, 1.000] | 1.000 | 1.000 | 0.500 | [0.500, 0.500] | 0.500 | 0.500 |
| 8 | S | 1.000 | [1.000, 1.000] | 0.038 | 1.000 | 0.500 | [0.500, 0.500] | 0.500 | 0.500 |
| 9 | S | 1.000 | [1.000, 1.000] | 0.128 | 1.000 | 0.500 | [0.500, 0.500] | 0.500 | 0.500 |
| 10 | S | 1.000 | [1.000, 1.000] | 0.372 | 1.000 | 0.500 | [0.500, 0.500] | 0.500 | 0.500 |
| 11 | S | 1.000 | [1.000, 1.000] | 1.000 | 1.000 | 0.500 | [0.500, 0.500] | 0.500 | 0.500 |
| 12 |  | 1.000 | [1.000, 1.000] | 0.846 | 1.000 | 0.500 | [0.500, 0.500] | 0.500 | 0.500 |
| 13 |  | 1.000 | [1.000, 1.000] | 0.359 | 1.000 | 0.500 | [0.500, 0.500] | 0.500 | 0.500 |
| 14 | **P** | 1.000 | [1.000, 1.000] | 0.026 | 1.000 | 0.500 | [0.500, 0.500] | 0.500 | 0.500 |
| 15 | **P** | 1.000 | [1.000, 1.000] | 0.705 | 1.000 | 0.500 | [0.500, 0.500] | 0.500 | 0.500 |
| 16 | **P** | 1.000 | [1.000, 1.000] | 0.692 | 1.000 | 0.500 | [0.500, 0.500] | 0.500 | 0.500 |
| 17 | **P** | 1.000 | [1.000, 1.000] | 0.308 | 1.000 | 0.500 | [0.500, 0.500] | 0.500 | 0.500 |
| 18 | **P** | 1.000 | [1.000, 1.000] | 0.385 | 1.000 | 0.500 | [0.500, 0.500] | 0.500 | 0.500 |
| 19 |  | 1.000 | [1.000, 1.000] | 0.949 | 1.000 | 0.500 | [0.500, 0.500] | 0.500 | 0.500 |
| 20 |  | 1.000 | [1.000, 1.000] | 0.244 | 1.000 | 0.500 | [0.500, 0.500] | 0.500 | 0.500 |
| 21 |  | 1.000 | [1.000, 1.000] | 0.000 | 1.000 | 0.500 | [0.500, 0.500] | 0.500 | 0.500 |
| 22 |  | 1.000 | [1.000, 1.000] | 0.397 | 1.000 | 0.500 | [0.500, 0.500] | 0.500 | 0.500 |
| 23 |  | 1.000 | [1.000, 1.000] | 0.179 | 1.000 | 0.500 | [0.500, 0.500] | 0.500 | 0.500 |
| 24 |  | 1.000 | [1.000, 1.000] | 1.000 | 1.000 | 0.500 | [0.500, 0.500] | 0.500 | 0.500 |
| 25 |  | 1.000 | [1.000, 1.000] | 0.000 | 1.000 | 0.500 | [0.500, 0.500] | 0.500 | 0.500 |
| 26 |  | 1.000 | [1.000, 1.000] | 0.205 | 1.000 | 0.500 | [0.500, 0.500] | 0.500 | 0.500 |
| 27 |  | 1.000 | [1.000, 1.000] | 0.103 | 1.000 | 0.500 | [0.500, 0.500] | 0.500 | 0.500 |
| 28 |  | 1.000 | [1.000, 1.000] | 0.500 | 1.000 | 0.500 | [0.500, 0.500] | 0.500 | 0.500 |
| 29 |  | 1.000 | [1.000, 1.000] | 1.000 | 1.000 | 0.500 | [0.500, 0.500] | 0.500 | 0.500 |
| 30 |  | 1.000 | [1.000, 1.000] | 0.949 | 1.000 | 0.500 | [0.500, 0.500] | 0.500 | 0.500 |
| 31 |  | 1.000 | [1.000, 1.000] | 1.000 | 1.000 | 0.500 | [0.500, 0.500] | 0.500 | 0.500 |

### 2.4 Diagnostic D3 — act vs no-act **inside** the same route (not one of the three contrasts; no verdict attached)

The three contrasts each set an act pool against a no-act pool that differs in *prompt content*. D3 removes that difference: for each route, positives are the committed runs and negatives are the **same targets' runs that did not commit**, read at the same position — for deceived, the same target's chains that never committed, at the same turn index. Same persuasion chain, same pressure prompt, same persona prompt on both sides; only commission differs. Same split rule, same bootstrap, same random arrow. This is a diagnostic on the readout position, not a redesign of any contrast.

| route | test pos / neg | best layer ≤ 30 | AUROC | 95 % CI | random seed 0 | min symmetrized floor | verdict |
|---|---|---|---|---|---|---|---|
| deceived | 31 / 84 | L30 | 0.680 | [0.585, 0.757] | **0.716** | 0.612 | fail |
| akratic | 24 / 36 | L21 | 0.704 | [0.502, 0.860] | 0.626 | 0.628 | fail |
| vicious | 31 / 41 | L6 | 0.684 | [0.483, 0.811] | 0.368 | 0.610 | fail |

**No route shows a detectable act-linked mark at `into`.** All three fail the locked criterion at every layer ≤ 30; for deceived the best layer's random arrow (0.716) *beats* the extracted direction (0.680); for akratic and vicious the CI lower bounds sit at or below 0.5. For the two single-turn routes this is what §1.1 predicts arithmetically — positives and negatives of a target are the same vector — and the measurement confirms it rather than assuming it.

| L | band | D3 deceived AUROC | 95% CI | rnd0 | floor | D3 akratic AUROC | 95% CI | rnd0 | floor | D3 vicious AUROC | 95% CI | rnd0 | floor |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 |  | 0.551 | [0.443, 0.744] | 0.630 | 0.682 | 0.491 | [0.158, 0.725] | 0.527 | 0.688 | 0.645 | [0.385, 0.813] | 0.375 | 0.694 |
| 1 |  | 0.599 | [0.503, 0.750] | 0.609 | 0.672 | 0.517 | [0.165, 0.799] | 0.584 | 0.715 | 0.626 | [0.412, 0.749] | 0.616 | 0.743 |
| 2 |  | 0.559 | [0.471, 0.731] | 0.313 | 0.695 | 0.495 | [0.213, 0.764] | 0.352 | 0.703 | 0.634 | [0.438, 0.749] | 0.527 | 0.611 |
| 3 |  | 0.591 | [0.502, 0.747] | 0.435 | 0.727 | 0.523 | [0.207, 0.817] | 0.603 | 0.729 | 0.655 | [0.487, 0.768] | 0.249 | 0.751 |
| 4 |  | 0.651 | [0.526, 0.778] | 0.397 | 0.713 | 0.450 | [0.193, 0.606] | 0.678 | 0.683 | 0.660 | [0.407, 0.810] | 0.520 | 0.644 |
| 5 |  | 0.621 | [0.514, 0.769] | 0.528 | 0.715 | 0.534 | [0.182, 0.786] | 0.406 | 0.701 | 0.631 | [0.356, 0.779] | 0.386 | 0.638 |
| 6 | S | 0.585 | [0.472, 0.745] | 0.537 | 0.688 | 0.462 | [0.141, 0.695] | 0.541 | 0.735 | 0.684 | [0.483, 0.811] | 0.368 | 0.705 |
| 7 | S | 0.597 | [0.514, 0.761] | 0.374 | 0.720 | 0.359 | [0.171, 0.651] | 0.664 | 0.713 | 0.543 | [0.311, 0.800] | 0.503 | 0.749 |
| 8 | S | 0.598 | [0.501, 0.783] | 0.368 | 0.644 | 0.461 | [0.277, 0.745] | 0.395 | 0.764 | 0.526 | [0.270, 0.715] | 0.375 | 0.652 |
| 9 | S | 0.585 | [0.496, 0.783] | 0.616 | 0.662 | 0.475 | [0.321, 0.692] | 0.406 | 0.668 | 0.526 | [0.337, 0.732] | 0.551 | 0.713 |
| 10 | S | 0.587 | [0.494, 0.782] | 0.541 | 0.656 | 0.456 | [0.133, 0.750] | 0.528 | 0.696 | 0.496 | [0.309, 0.685] | 0.413 | 0.706 |
| 11 | S | 0.592 | [0.491, 0.789] | 0.568 | 0.701 | 0.442 | [0.188, 0.642] | 0.389 | 0.738 | 0.460 | [0.314, 0.655] | 0.626 | 0.673 |
| 12 |  | 0.605 | [0.494, 0.786] | 0.591 | 0.687 | 0.467 | [0.136, 0.683] | 0.517 | 0.679 | 0.472 | [0.354, 0.695] | 0.509 | 0.658 |
| 13 |  | 0.633 | [0.538, 0.798] | 0.482 | 0.664 | 0.498 | [0.315, 0.747] | 0.376 | 0.680 | 0.492 | [0.361, 0.772] | 0.411 | 0.610 |
| 14 | **P** | 0.616 | [0.527, 0.797] | 0.397 | 0.636 | 0.464 | [0.202, 0.696] | 0.341 | 0.676 | 0.394 | [0.295, 0.590] | 0.368 | 0.632 |
| 15 | **P** | 0.613 | [0.533, 0.775] | 0.570 | 0.642 | 0.436 | [0.125, 0.680] | 0.460 | 0.660 | 0.389 | [0.317, 0.612] | 0.524 | 0.649 |
| 16 | **P** | 0.621 | [0.540, 0.789] | 0.369 | 0.688 | 0.542 | [0.395, 0.786] | 0.597 | 0.681 | 0.395 | [0.300, 0.653] | 0.587 | 0.673 |
| 17 | **P** | 0.627 | [0.546, 0.789] | 0.568 | 0.616 | 0.641 | [0.373, 0.835] | 0.337 | 0.717 | 0.446 | [0.273, 0.648] | 0.311 | 0.700 |
| 18 | **P** | 0.619 | [0.542, 0.776] | 0.599 | 0.686 | 0.638 | [0.407, 0.795] | 0.484 | 0.662 | 0.434 | [0.250, 0.577] | 0.380 | 0.657 |
| 19 |  | 0.633 | [0.561, 0.780] | 0.369 | 0.676 | 0.648 | [0.343, 0.826] | 0.554 | 0.703 | 0.451 | [0.322, 0.630] | 0.495 | 0.740 |
| 20 |  | 0.631 | [0.559, 0.774] | 0.439 | 0.703 | 0.624 | [0.337, 0.838] | 0.403 | 0.683 | 0.469 | [0.347, 0.736] | 0.648 | 0.689 |
| 21 |  | 0.643 | [0.571, 0.773] | 0.584 | 0.674 | 0.704 | [0.502, 0.860] | 0.626 | 0.659 | 0.479 | [0.328, 0.656] | 0.361 | 0.708 |
| 22 |  | 0.649 | [0.570, 0.787] | 0.354 | 0.678 | 0.593 | [0.249, 0.762] | 0.414 | 0.659 | 0.600 | [0.422, 0.846] | 0.417 | 0.696 |
| 23 |  | 0.637 | [0.556, 0.785] | 0.534 | 0.693 | 0.595 | [0.291, 0.785] | 0.659 | 0.659 | 0.543 | [0.378, 0.667] | 0.555 | 0.626 |
| 24 |  | 0.634 | [0.551, 0.785] | 0.405 | 0.709 | 0.647 | [0.386, 0.873] | 0.341 | 0.659 | 0.568 | [0.406, 0.680] | 0.478 | 0.670 |
| 25 |  | 0.651 | [0.573, 0.772] | 0.554 | 0.720 | 0.629 | [0.378, 0.806] | 0.519 | 0.628 | 0.594 | [0.413, 0.764] | 0.705 | 0.705 |
| 26 |  | 0.659 | [0.569, 0.791] | 0.554 | 0.666 | 0.616 | [0.367, 0.770] | 0.447 | 0.669 | 0.548 | [0.356, 0.710] | 0.668 | 0.668 |
| 27 |  | 0.657 | [0.565, 0.753] | 0.422 | 0.633 | 0.595 | [0.368, 0.696] | 0.399 | 0.662 | 0.590 | [0.415, 0.693] | 0.592 | 0.638 |
| 28 |  | 0.654 | [0.559, 0.753] | 0.502 | 0.612 | 0.662 | [0.389, 0.908] | 0.517 | 0.677 | 0.597 | [0.399, 0.788] | 0.505 | 0.672 |
| 29 |  | 0.656 | [0.560, 0.752] | 0.609 | 0.668 | 0.625 | [0.268, 0.865] | 0.638 | 0.638 | 0.594 | [0.411, 0.744] | 0.566 | 0.655 |
| 30 |  | 0.680 | [0.585, 0.757] | 0.716 | 0.716 | 0.635 | [0.304, 0.877] | 0.498 | 0.676 | 0.542 | [0.392, 0.659] | 0.519 | 0.662 |
| 31 |  | 0.722 | [0.621, 0.807] | 0.391 | 0.619 | 0.631 | [0.336, 0.845] | 0.409 | 0.640 | 0.580 | [0.416, 0.797] | 0.459 | 0.647 |

### 2.5 The two readings of the pass rule, and why the random floor decides

The locked rule compares the CI lower bound against **`randctl` seed 0's signed AUROC**. A random direction can be *anti*-separating — contrast 3's seed-0 arrow scores 0.000 at seven layers — and where it does, "random + 0.20" becomes a bar of 0.20. Read against the **symmetrized floor**, max(A, 1 − A) over seeds 0–9, which is what a random arrow's separating power actually is:

| contrast | minimum symmetrized floor over layers ≤ 30 | verdict against that floor |
|---|---|---|
| 1 deceived | 0.801 (L24); 0.900–0.961 across the primary band | **fail** |
| 1 deceived, turn-matched secondary | 0.851 | **fail** |
| 2 akratic | 0.826; **exactly 1.000 at 11 layers** | **fail** |
| 3 vicious | **1.000 at every layer** | **fail** |
| control 1 (act-free) | 0.564 | fail |
| control 2 (act-free) | 0.778 | fail |

Since AUROC cannot exceed 1.000, a floor of 0.80 or above makes the +0.20 margin unreachable in principle. **All three contrasts fail the locked criterion when the random control is read as a two-sided floor**, and the floors themselves are the diagnosis: a random direction separates these pools at 0.80–1.00, which is `reports/S1b-runs.md` §10's own symptom of two pools occupying disjoint regions of activation space. The literal verdicts are reported above because the rule is locked; this reading is reported beside them because the rule was written for pools a random arrow cannot separate.

**Pipeline sanity check.** The same extraction and scoring on a random split of one pool against itself (`benign_pressure`, disjoint halves): AUROC 0.452 / 0.498 / 0.520 / 0.509 / 0.510 at L0 / L8 / L16 / L24 / L30. The pipeline does not manufacture separation.

---

## 3. Cosines

Each contrast direction is the unit diff-in-means arrow from its extraction half. Cosines with the borrowed axes (`directions/dirs_8B_base_sweep.pt`), with `randctl` seed 0, and with each other, per layer. Band means:

| direction | · refusal | · badmed | · persona | · random 0 |
|---|---|---|---|---|
| deceived, primary band L14–18 | **+0.228** | +0.114 | −0.130 | +0.003 |
| deceived, secondary band L6–11 | −0.095 | +0.050 | −0.091 | −0.002 |
| akratic, primary band | **+0.471** | +0.149 | −0.055 | +0.003 |
| akratic, secondary band | +0.067 | +0.013 | −0.026 | −0.005 |
| vicious, primary band | **+0.213** | +0.044 | +0.033 | −0.005 |
| vicious, secondary band | +0.098 | −0.000 | −0.038 | +0.002 |

| pair | primary band | secondary band | min over 32 layers | max |
|---|---|---|---|---|
| deceived · akratic | +0.173 | +0.163 | −0.136 | +0.203 |
| deceived · vicious | +0.138 | +0.254 | +0.098 | +0.697 |
| akratic · vicious | +0.363 | +0.307 | −0.085 | +0.427 |

All three directions are orthogonal to the random arrow (\|cos\| ≤ 0.005 in both bands), as they should be. All three lean positively on **refusal** in the primary band, the akratic direction most (+0.471) — consistent with S1b §9's single-turn result that akratic sits +1.814 off its control on refusal at L16. The three directions are **not** the same direction (pairwise +0.14 to +0.36 in the primary band), but given §2 none of these cosines should be read as a fact about an instrument: they describe directions between prompt pools.

### 3.1 Cosine with the named axes, per layer

| L | band | dec·refusal | dec·badmed | dec·persona | akr·refusal | akr·badmed | akr·persona | vic·refusal | vic·badmed | vic·persona | dec·rnd0 | akr·rnd0 | vic·rnd0 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 |  | -0.054 | -0.008 | -0.023 | -0.070 | +0.031 | -0.012 | +0.011 | -0.007 | -0.034 | +0.003 | -0.005 | -0.013 |
| 1 |  | -0.093 | -0.098 | +0.003 | -0.192 | +0.085 | +0.008 | -0.014 | -0.083 | -0.046 | -0.016 | +0.016 | -0.010 |
| 2 |  | -0.097 | -0.046 | +0.013 | -0.130 | +0.028 | -0.003 | -0.038 | -0.037 | -0.088 | +0.016 | +0.004 | -0.006 |
| 3 |  | -0.076 | +0.009 | -0.067 | -0.096 | +0.025 | +0.032 | -0.077 | -0.009 | -0.090 | -0.013 | +0.002 | -0.026 |
| 4 |  | -0.049 | +0.006 | -0.079 | -0.091 | +0.034 | -0.030 | +0.064 | -0.017 | -0.070 | +0.021 | -0.012 | +0.019 |
| 5 |  | -0.078 | +0.016 | -0.087 | -0.105 | +0.023 | -0.020 | +0.023 | +0.001 | -0.092 | -0.005 | -0.008 | -0.018 |
| 6 | S | -0.069 | +0.054 | -0.107 | -0.081 | +0.015 | -0.033 | +0.073 | +0.013 | -0.103 | +0.019 | -0.006 | -0.005 |
| 7 | S | -0.110 | +0.060 | -0.094 | -0.070 | -0.014 | -0.027 | +0.072 | -0.004 | -0.073 | +0.006 | -0.013 | +0.028 |
| 8 | S | -0.144 | +0.051 | -0.097 | +0.062 | +0.010 | -0.024 | +0.085 | -0.013 | -0.046 | -0.029 | -0.014 | -0.007 |
| 9 | S | -0.146 | +0.048 | -0.092 | +0.051 | +0.017 | -0.031 | +0.072 | -0.023 | -0.019 | -0.018 | -0.015 | -0.011 |
| 10 | S | -0.070 | +0.048 | -0.082 | +0.164 | +0.024 | -0.033 | +0.098 | +0.014 | -0.008 | +0.005 | +0.010 | -0.010 |
| 11 | S | -0.029 | +0.036 | -0.075 | +0.275 | +0.023 | -0.007 | +0.187 | +0.013 | +0.022 | +0.006 | +0.010 | +0.017 |
| 12 |  | +0.024 | +0.065 | -0.079 | +0.416 | +0.082 | -0.060 | +0.236 | +0.007 | +0.020 | -0.009 | +0.024 | +0.011 |
| 13 |  | +0.085 | +0.103 | -0.103 | +0.478 | +0.146 | -0.078 | +0.258 | +0.020 | +0.019 | +0.028 | +0.022 | +0.007 |
| 14 | **P** | +0.140 | +0.107 | -0.110 | +0.448 | +0.159 | -0.073 | +0.249 | +0.051 | +0.008 | -0.002 | +0.020 | +0.003 |
| 15 | **P** | +0.175 | +0.104 | -0.114 | +0.414 | +0.100 | -0.034 | +0.229 | +0.031 | +0.023 | +0.007 | -0.012 | -0.013 |
| 16 | **P** | +0.270 | +0.160 | -0.159 | +0.508 | +0.188 | -0.066 | +0.258 | +0.066 | +0.027 | -0.009 | +0.014 | +0.004 |
| 17 | **P** | +0.277 | +0.119 | -0.151 | +0.524 | +0.178 | -0.063 | +0.175 | +0.038 | +0.049 | +0.006 | -0.010 | -0.002 |
| 18 | **P** | +0.277 | +0.078 | -0.115 | +0.462 | +0.123 | -0.039 | +0.156 | +0.033 | +0.060 | +0.012 | +0.003 | -0.015 |
| 19 |  | +0.288 | +0.076 | -0.120 | +0.455 | +0.111 | -0.031 | +0.130 | +0.032 | +0.049 | -0.027 | -0.002 | +0.013 |
| 20 |  | +0.286 | +0.050 | -0.135 | +0.437 | +0.099 | -0.054 | +0.103 | +0.037 | +0.055 | +0.012 | -0.003 | -0.017 |
| 21 |  | +0.282 | +0.036 | -0.146 | +0.415 | +0.072 | -0.054 | +0.083 | +0.019 | +0.055 | -0.004 | -0.012 | +0.015 |
| 22 |  | +0.274 | +0.018 | -0.148 | +0.473 | +0.051 | -0.067 | +0.092 | +0.022 | +0.059 | -0.013 | -0.004 | +0.016 |
| 23 |  | +0.290 | +0.028 | -0.148 | +0.483 | +0.041 | -0.051 | +0.109 | +0.008 | +0.056 | +0.011 | -0.009 | -0.016 |
| 24 |  | +0.272 | +0.033 | -0.138 | +0.463 | +0.033 | -0.036 | +0.137 | +0.006 | +0.046 | -0.028 | -0.005 | +0.033 |
| 25 |  | +0.250 | +0.021 | -0.137 | +0.435 | +0.016 | -0.026 | +0.117 | +0.003 | +0.059 | -0.010 | +0.005 | -0.017 |
| 26 |  | +0.201 | +0.026 | -0.105 | +0.426 | +0.021 | -0.022 | +0.135 | +0.007 | +0.042 | -0.003 | -0.000 | -0.020 |
| 27 |  | +0.199 | +0.028 | -0.114 | +0.421 | +0.033 | -0.013 | +0.142 | +0.006 | +0.048 | +0.026 | +0.008 | -0.031 |
| 28 |  | +0.182 | +0.026 | -0.104 | +0.390 | +0.034 | -0.001 | +0.130 | -0.001 | +0.044 | +0.014 | +0.008 | +0.014 |
| 29 |  | +0.178 | +0.055 | -0.099 | +0.340 | +0.053 | -0.013 | +0.121 | -0.008 | +0.052 | +0.001 | +0.002 | +0.033 |
| 30 |  | +0.161 | +0.090 | -0.107 | +0.376 | +0.071 | -0.018 | +0.113 | +0.027 | +0.037 | -0.003 | -0.002 | +0.020 |
| 31 |  | +0.117 | +0.027 | -0.068 | +0.339 | +0.019 | -0.031 | +0.143 | +0.005 | +0.046 | +0.005 | +0.024 | +0.004 |

### 3.2 Cosine between the three contrast directions, per layer

| L | band | deceived·akratic | deceived·vicious | akratic·vicious |
|---|---|---|---|---|
| 0 |  | -0.089 | +0.697 | -0.085 |
| 1 |  | -0.136 | +0.664 | -0.059 |
| 2 |  | -0.031 | +0.618 | +0.051 |
| 3 |  | -0.027 | +0.571 | +0.108 |
| 4 |  | +0.028 | +0.497 | +0.124 |
| 5 |  | +0.050 | +0.521 | +0.116 |
| 6 | S | +0.109 | +0.458 | +0.153 |
| 7 | S | +0.174 | +0.369 | +0.268 |
| 8 | S | +0.147 | +0.246 | +0.285 |
| 9 | S | +0.195 | +0.166 | +0.338 |
| 10 | S | +0.174 | +0.126 | +0.371 |
| 11 | S | +0.179 | +0.161 | +0.427 |
| 12 |  | +0.185 | +0.144 | +0.395 |
| 13 |  | +0.203 | +0.123 | +0.354 |
| 14 | **P** | +0.199 | +0.154 | +0.396 |
| 15 | **P** | +0.150 | +0.146 | +0.399 |
| 16 | **P** | +0.193 | +0.158 | +0.406 |
| 17 | **P** | +0.186 | +0.122 | +0.333 |
| 18 | **P** | +0.135 | +0.110 | +0.280 |
| 19 |  | +0.138 | +0.098 | +0.256 |
| 20 |  | +0.143 | +0.108 | +0.234 |
| 21 |  | +0.139 | +0.113 | +0.219 |
| 22 |  | +0.179 | +0.113 | +0.187 |
| 23 |  | +0.173 | +0.125 | +0.185 |
| 24 |  | +0.156 | +0.128 | +0.179 |
| 25 |  | +0.147 | +0.124 | +0.163 |
| 26 |  | +0.132 | +0.135 | +0.153 |
| 27 |  | +0.129 | +0.142 | +0.137 |
| 28 |  | +0.107 | +0.148 | +0.122 |
| 29 |  | +0.122 | +0.114 | +0.053 |
| 30 |  | +0.121 | +0.115 | +0.010 |
| 31 |  | +0.094 | +0.103 | +0.022 |

---

## 4. For the researcher, not a verdict

On the evidence in this report, **no route leaves a mark at `into` that is detectable against a matched control**, and the three near-perfect separations are differences between prompt pools rather than traces of the harmful act. Contrast 2 fails on the brief's own test: its act-free control — two pools of `benign_pressure` runs differing only in which targets they contain — separates perfectly (1.000 at L29–31, 0.889 across L21–26), so that contrast is **structural** and the report says so in those words. Contrast 3's control cannot be computed at all, because the persona-only pool holds five distinct readout vectors, one per prompt, bit-identical across seeds, so its two seed halves are the same vectors and every AUROC ties at exactly 0.500; meanwhile contrast 3's positives are separable from its negatives by **prompt length alone** (AUROC 1.000; 144–170 tokens against 121–122) and by a **random arrow** (symmetrized floor 1.000 at every layer), which is precisely the `reports/S1b-runs.md` §10 symptom of two pools sitting in disjoint regions. Contrast 1 is the one contrast whose act-free control is **clean** (0.513 at best, 0.383–0.513 over all 32 layers) and whose separation is explained neither by length (length-only AUROC 0.433) nor by the turn-index imbalance (the turn-distribution-matched variant reproduces AUROC 1.000) — so the context-matching the fallback introduced does remove the Task 8 confound for this contrast. What it does not do is put the act inside the measured window: `into` is the residual at the **last prompt token, before the assistant's turn**, so the turn-T act has not yet happened when the reading is taken, the positives and negatives differ in what the conversation is *about* from turn 1 onward (the benign chain asks a safe question about the same scenario), and the within-route diagnostic D3 — chains of the same target that committed against chains that did not, at the same turn index — finds no separation at any layer (best 0.680 at L30, below that layer's own random arrow at 0.716). For the two single-turn routes the same point is arithmetic rather than empirical: their `into` vector is a function of the prompt alone, giving 17, 20, 22 and 5 distinct vectors behind nominal Ns of 62, 180, 83 and 30, so selecting positives by `committed` cannot change the readout, the bootstrap CIs are narrower than the evidence supports, and their contrasts compare 12 prompts with 15 and 14 prompts with 5. Read together: contrast 1's design is sound and its control is clean, but all three contrasts are measuring context at a position that precedes the act.

---

## 5. Anything unworkable

1. **Control 3 is degenerate and is not a control.** Persona-only holds 5 distinct `into` vectors across 30 runs (bit-identical across all 25 within-prompt seed pairs, max \|diff\| 0.0). Splitting by seed halves puts the same vectors on both sides, so the extracted arrow and all ten random arrows return exactly 0.500 at every layer. Contrast 3 therefore has **no working act-free control**, and the brief's instruction to read it as the weaker of the three understates the problem.
2. **Contrasts 2 and 3 are run at nominal N, not effective N.** 62 vs 180 is really 12 prompts vs 15; 83 vs 30 is really 14 vs 5. Both the AUROC and its bootstrap CI are computed over the repeated items, so the CIs on those two contrasts should not be read as they stand. Contrast 1 is unaffected (109 of 109 vectors distinct).
3. **Contrast 3 is separable by prompt length alone** (AUROC 1.000, ranges disjoint). No direction is needed to sort those two pools.
4. **The pass rule can be met by an anti-separating random arrow.** Contrast 3's `randctl` seed-0 arrow scores exactly 0.000 at seven layers and exactly 1.000 at six more, so "random + 0.20" is at some layers a bar of 0.20. Read against the symmetrized floor max(A, 1 − A) over seeds 0–9, all three contrasts fail, and at floors of 0.80–1.00 the +0.20 margin is unreachable in principle. Both readings are reported; the rule is locked and was not changed.
5. **Contrast 1 keeps a residual length mismatch**: turn count matched, token count not — benign negatives run 15–30 % longer at the same turn index, because the benign cells write long and truncate at 12.6 % (S1b §12). Ranges overlap at every turn index and length-only AUROC is 0.433, so it is not what separates the pools, but it is not removed by the design either.
6. **Every deceived number is provisional in T.** `results/raw/s1b/t9/T_source.json` does not exist; T is `T_primary`, and S1b §8 finds the primary judge systematically lenient, so T may sit late. `python3 scripts/s1c/s1c_inside.py --T-source results/raw/s1b/t9/T_source.json` re-runs everything unchanged once the labels are in.
7. **What a corrected contrast would need — stated, not decided, and not run here.** The store already holds two readout positions that fall *after* the act: `think` (mean over the thinking block) and `answer` (mean over the answer tokens), both saved for all 6,100 turns at all 32 layers. The brief locks this session to `into`, so nothing at those positions was computed. `scripts/s1c/s1c_inside.py --position answer` (or `think`) runs the identical pipeline, contrasts, controls and diagnostics at those positions in about 30 seconds on CPU, writing `summary_answer.json` beside this one; the flag was added for that purpose and its default is `into`, so nothing in this report depends on it. Whether to order that run is the researcher's and the hub's decision, not this session's. Note that a `think`-position run will discard turns with no thinking tags (S1b §12: adherence 0.53 inside chains, 0.39 in the akratic cell), and the script reports the drop counts rather than hiding them.
8. Everything else in the brief was executed in full.

---

## 6. Assumptions, deviations, and what was not done

**The eight assumptions were put to the researcher in the plan before the run and accepted on record (2026-09-04).** Restated with their outcomes:

1. **C1 negatives deduplicated** — 54 cells × 6 seeds = 324 items; the turn-matched 1:1 variant is reported beside it (§2.1).
2. **Contrasts restricted to targets present in both pools** (`t8_inside.py`'s `shared` rule, reused). C2 drops 3 `benign_pressure` targets with no akratic act.
3. **C3 negatives split by persona-prompt parity** (1, 3, 5 extraction; 2, 4 test), positives by target parity.
4. **Control splits quartered** — role = index mod 2, extraction/test = (index // 2) mod 2 — so the two pools sit on disjoint target halves *and* test targets are held out.
5. **Bootstrap clusters** — joint over shared targets for C1, C2 and D3; independent per side for the controls; C3's and control 3's negatives resampled per item (2 held-out prompts are too few to resample as clusters). Every table states which was used.
6. **The verdict follows the locked rule literally**, with the symmetrized floor reported beside it (§2.5). The rule was not changed.
7. **T is `T_primary`** (§5.6).
8. **This session committed to `s1c-inside`** as instructed, never to main.

**Researcher's addition at approval, executed:** length-only AUROC computed for the act-free controls as well as the three contrasts (§1.2), and contrast 1's residual length mismatch stated beside its result (§1.2, §2.1, §5.5).

**Deviations:** none. No threshold, pool definition, seed rule or pass criterion was altered. **Additions beyond the brief, all labelled in place:** the turn-matched secondary variant (§2.1), the length-only AUROC (§1.2, the researcher's addition), the effective-N diagnostic (§1.1), the D3 act-vs-no-act diagnostic (§2.4), the symmetrized-floor reading (§2.5) and the `--position` flag (§5.7, default unchanged, unused in this report).

**Not done, by design:** no comparison across routes (the confounded contrast); **no layer picked** — full 32-layer sweeps are reported for every contrast, control and diagnostic, and L31 is printed but excluded from every verdict; no model loaded, no token generated, no API call made; no file under `results/raw/s1b/` written, moved or deleted (`ls -lR` byte-identical before and after, mtimes included); nothing run at 1B.

**Vocabulary:** STAGE0 §2 terms throughout; "the researcher" throughout. The brief's "route" is D-025's word for STAGE0 §2's **mode** and is used where the brief uses it. A whole-word grep for the five banned terms over this report and every file this session wrote returns nothing.

---

## 7. Addendum run — the identical pipeline at the `answer` position

**Ordered by the researcher** in the addendum to `briefs/S1c-inside.md` (2026-09-04), after §1.1 showed that `into` precedes the act on the single-turn routes. **Scope note:** §1–§6 above report the `into` run the brief locks, and are unchanged; this section reports the `answer` run and states where the two differ. A `think`-position run was **not** ordered and was not made.

**Run facts.** `python3 scripts/s1c/s1c_inside.py --position answer`, same machine, same script, same commit; 31 s on CPU; no model, no token generated, **no API call**. Contrasts, act-free controls, splits, bootstrap (1,000, seed 0), random arrow, band callouts, both rule readings and the D3 diagnostic are identical — only the readout position changes, from `into` (residual at the last prompt token) to `answer` (**mean over the act turn's answer tokens**, i.e. over the text the act is made of). Output: `results/raw/s1c/summary_answer.json`, `arrows_answer.pt`, `run_answer.log`, beside the originals. Pool counts asserted against `reports/S1b-runs.md` §5 again and matched. **No item was dropped as non-finite** (`dropped_nonfinite: {}`), so every turn in every pool carried a non-empty answer span. `results/raw/s1b/` is byte-identical before and after this run as well.

### 7.1 Effective N at `answer` — the §1.1 defect is gone

| pool | runs | targets / prompts | distinct `answer` vectors | within-target seed pairs bit-identical | max \|diff\| |
|---|---|---|---|---|---|
| deceived (at turn T) | 109 | 15 | **109** | — | — |
| akratic committed | 62 | 12 | **62** (was 17 at `into`) | **0 / 50** | 7.78 |
| benign_pressure | 180 | 15 | **180** (was 20) | **0 / 165** | 7.46 |
| vicious committed | 83 | 14 | **83** (was 22) | **0 / 69** | 6.04 |
| persona-only | 30 | 5 | **30** (was 5) | **0 / 25** | 10.01 |

Every run now has its own readout vector and no within-target seed pair is identical, because the position reads the sampled answer rather than the fixed prompt. Nominal N and effective N coincide, the bootstrap CIs are no longer inflated by repetition, and — the point of the addendum — **grading a run `committed` can now change the readout**, which at `into` it could not.

### 7.2 The three contrasts and their act-free controls at `answer`

| | contrast 1 deceived | contrast 2 akratic | contrast 3 vicious |
|---|---|---|---|
| best layer ≤ 30 | **L18, 0.990**, CI [0.972, 1.000], random 0.554 | **L11, 0.891**, CI [0.809, 0.959], random 0.470 | **L4, 1.000**, CI [1.000, 1.000], random 0.246 |
| verdict, locked rule read literally | **pass** — at 30 of the 31 layers ≤ 30 (all but L15) | **pass** — at 15 layers | **pass** — at 28 layers |
| NEAR layers | none | none | none |
| verdict vs the symmetrized floor | **pass** — L9, L11, L12, L21, L27 (margins +0.001 to +0.065) | **fail** — no layer | **pass** — L21, L23 (margins +0.044, +0.046) |
| its act-free control | **fail**, best 0.417 (below chance), CI [0.140, 0.690] | **fail**, best 0.746 at L1, CI [0.397, 0.975] | **fail**, best 0.583 at L27, CI [0.167, 0.889] |
| distinct vectors behind N | 109 of 109 | 62 of 62 vs 180 of 180 | 83 of 83 vs 30 of 30 |
| length-only AUROC | 0.433 | 0.824 | **1.000** |
| primary band L14–18 | 0.988–0.990, CI lower 0.965–0.972 | 0.855–0.889, CI lower 0.739–0.803 | 1.000 throughout |
| secondary band L6–11 | 0.983–0.986 | 0.827–0.891 | 1.000 throughout |
| L31 (reported, excluded) | 0.988 [0.973, 1.000] | 0.735 [0.577, 0.915] | 1.000 [1.000, 1.000] |

**All three act-free controls now fail.** Control 2, which separated perfectly at `into` (1.000 at L29–31) and made contrast 2 structural, reaches only 0.746 at `answer` with a CI spanning 0.397–0.975. Control 3, which was **degenerate** at `into` (every AUROC exactly 0.500, because the persona-only pool held 5 vectors and its seed halves were the same vectors), is **computable here** — the 30 runs give 30 distinct vectors — and it fails (0.583). Control 1 was clean at `into` and is clean here, at 0.417, below chance.

**Two things do not change with the position, and both still bear on contrast 3.** The **length-only AUROC** is computed from the prompt-token count at the readout turn, which is a property of the *pools*, not of the position, so it is unchanged: contrast 3's positives and negatives remain **perfectly separable by prompt length alone (1.000)**, before any direction is extracted. And the **symmetrized random floor** stays high — minimum over layers ≤ 30 of 0.698 (contrast 1), 0.672 (contrast 2), 0.754 (contrast 3) — so where a contrast clears it, it clears by hundredths, not by the 0.20 the rule intends as a margin.

**Pipeline sanity check at this position.** Extraction and scoring on a random split of `benign_pressure` against itself: AUROC 0.504 / 0.480 / 0.458 / 0.458 / 0.439 at L0 / L8 / L16 / L24 / L30.

**Contrast 1, turn-distribution-matched secondary variant** (§2.1's construction, 41 vs 41): literal **pass**, best L13 0.987 CI [0.970, 1.000]; floor **pass** at L27 and L30; primary band 0.984–0.986. The turn-index imbalance is not doing the work at this position either.

#### Contrast 1 — deceived acts at turn T vs the benign-matched chain, and its act-free control (`answer`)

| L | band | C1 AUROC | 95% CI | rnd0 | floor | control 1 AUROC | 95% CI | rnd0 | floor |
|---|---|---|---|---|---|---|---|---|---|
| 0 |  | 0.980 | [0.957, 1.000] | 0.456 | 0.860 | 0.309 | [0.094, 0.558] | 0.383 | 0.617 |
| 1 |  | 0.976 | [0.949, 1.000] | 0.723 | 0.880 | 0.309 | [0.075, 0.575] | 0.482 | 0.648 |
| 2 |  | 0.977 | [0.950, 1.000] | 0.242 | 0.905 | 0.333 | [0.077, 0.623] | 0.551 | 0.661 |
| 3 |  | 0.983 | [0.961, 1.000] | 0.267 | 0.811 | 0.306 | [0.055, 0.604] | 0.490 | 0.652 |
| 4 |  | 0.984 | [0.963, 1.000] | 0.493 | 0.843 | 0.356 | [0.065, 0.657] | 0.405 | 0.689 |
| 5 |  | 0.984 | [0.963, 1.000] | 0.310 | 0.866 | 0.347 | [0.087, 0.613] | 0.423 | 0.700 |
| 6 | S | 0.985 | [0.966, 1.000] | 0.219 | 0.915 | 0.384 | [0.100, 0.660] | 0.528 | 0.657 |
| 7 | S | 0.985 | [0.964, 1.000] | 0.276 | 0.876 | 0.408 | [0.105, 0.686] | 0.471 | 0.604 |
| 8 | S | 0.985 | [0.960, 1.000] | 0.476 | 0.822 | 0.398 | [0.136, 0.657] | 0.362 | 0.682 |
| 9 | S | 0.986 | [0.964, 1.000] | 0.679 | 0.761 | 0.385 | [0.103, 0.636] | 0.375 | 0.678 |
| 10 | S | 0.983 | [0.959, 1.000] | 0.494 | 0.861 | 0.395 | [0.113, 0.649] | 0.440 | 0.628 |
| 11 | S | 0.986 | [0.965, 1.000] | 0.238 | 0.762 | 0.405 | [0.146, 0.652] | 0.560 | 0.733 |
| 12 |  | 0.984 | [0.961, 1.000] | 0.420 | 0.708 | 0.417 | [0.140, 0.690] | 0.618 | 0.618 |
| 13 |  | 0.988 | [0.967, 1.000] | 0.603 | 0.846 | 0.404 | [0.161, 0.647] | 0.537 | 0.744 |
| 14 | **P** | 0.988 | [0.966, 1.000] | 0.349 | 0.911 | 0.387 | [0.148, 0.620] | 0.369 | 0.670 |
| 15 | **P** | 0.987 | [0.966, 1.000] | 0.873 | 0.873 | 0.338 | [0.106, 0.545] | 0.380 | 0.620 |
| 16 | **P** | 0.988 | [0.965, 1.000] | 0.548 | 0.936 | 0.317 | [0.117, 0.507] | 0.474 | 0.684 |
| 17 | **P** | 0.989 | [0.970, 1.000] | 0.295 | 0.803 | 0.295 | [0.116, 0.461] | 0.503 | 0.770 |
| 18 | **P** | 0.990 | [0.972, 1.000] | 0.554 | 0.834 | 0.292 | [0.105, 0.465] | 0.620 | 0.620 |
| 19 |  | 0.989 | [0.970, 1.000] | 0.218 | 0.815 | 0.280 | [0.106, 0.433] | 0.701 | 0.805 |
| 20 |  | 0.989 | [0.970, 1.000] | 0.632 | 0.800 | 0.291 | [0.109, 0.458] | 0.359 | 0.674 |
| 21 |  | 0.986 | [0.967, 1.000] | 0.269 | 0.766 | 0.270 | [0.074, 0.448] | 0.481 | 0.768 |
| 22 |  | 0.985 | [0.967, 1.000] | 0.238 | 0.795 | 0.257 | [0.075, 0.436] | 0.418 | 0.655 |
| 23 |  | 0.983 | [0.964, 1.000] | 0.551 | 0.832 | 0.266 | [0.066, 0.456] | 0.376 | 0.729 |
| 24 |  | 0.981 | [0.961, 1.000] | 0.389 | 0.811 | 0.263 | [0.062, 0.457] | 0.426 | 0.710 |
| 25 |  | 0.982 | [0.962, 1.000] | 0.271 | 0.885 | 0.260 | [0.068, 0.448] | 0.571 | 0.740 |
| 26 |  | 0.981 | [0.960, 1.000] | 0.547 | 0.832 | 0.261 | [0.074, 0.453] | 0.441 | 0.642 |
| 27 |  | 0.984 | [0.963, 1.000] | 0.488 | 0.698 | 0.269 | [0.094, 0.437] | 0.438 | 0.740 |
| 28 |  | 0.983 | [0.962, 1.000] | 0.556 | 0.849 | 0.260 | [0.078, 0.432] | 0.584 | 0.732 |
| 29 |  | 0.985 | [0.964, 1.000] | 0.311 | 0.852 | 0.268 | [0.064, 0.448] | 0.477 | 0.724 |
| 30 |  | 0.987 | [0.967, 1.000] | 0.352 | 0.780 | 0.275 | [0.091, 0.420] | 0.693 | 0.693 |
| 31 |  | 0.988 | [0.973, 1.000] | 0.362 | 0.860 | 0.268 | [0.103, 0.408] | 0.625 | 0.676 |

#### Contrast 2 — akratic acts vs benign_pressure, and its act-free control (`answer`)

| L | band | C2 AUROC | 95% CI | rnd0 | floor | control 2 AUROC | 95% CI | rnd0 | floor |
|---|---|---|---|---|---|---|---|---|---|
| 0 |  | 0.746 | [0.623, 0.946] | 0.530 | 0.700 | 0.744 | [0.410, 0.975] | 0.664 | 0.844 |
| 1 |  | 0.757 | [0.637, 0.934] | 0.460 | 0.709 | 0.746 | [0.397, 0.975] | 0.579 | 0.714 |
| 2 |  | 0.764 | [0.648, 0.920] | 0.538 | 0.679 | 0.709 | [0.346, 0.955] | 0.505 | 0.877 |
| 3 |  | 0.790 | [0.702, 0.919] | 0.301 | 0.752 | 0.649 | [0.233, 0.963] | 0.457 | 0.877 |
| 4 |  | 0.818 | [0.722, 0.924] | 0.515 | 0.703 | 0.663 | [0.237, 0.977] | 0.873 | 0.873 |
| 5 |  | 0.822 | [0.730, 0.928] | 0.542 | 0.732 | 0.576 | [0.138, 0.986] | 0.477 | 0.836 |
| 6 | S | 0.838 | [0.758, 0.929] | 0.456 | 0.796 | 0.601 | [0.162, 0.980] | 0.622 | 0.792 |
| 7 | S | 0.827 | [0.738, 0.924] | 0.547 | 0.793 | 0.611 | [0.102, 0.972] | 0.677 | 0.775 |
| 8 | S | 0.828 | [0.744, 0.928] | 0.394 | 0.672 | 0.623 | [0.113, 0.972] | 0.571 | 0.721 |
| 9 | S | 0.856 | [0.771, 0.938] | 0.356 | 0.788 | 0.616 | [0.102, 0.968] | 0.378 | 0.843 |
| 10 | S | 0.862 | [0.776, 0.942] | 0.644 | 0.851 | 0.633 | [0.153, 0.965] | 0.562 | 0.951 |
| 11 | S | 0.891 | [0.809, 0.959] | 0.470 | 0.811 | 0.687 | [0.310, 0.965] | 0.833 | 0.897 |
| 12 |  | 0.865 | [0.775, 0.947] | 0.657 | 0.769 | 0.716 | [0.363, 0.955] | 0.255 | 0.745 |
| 13 |  | 0.883 | [0.794, 0.960] | 0.391 | 0.742 | 0.693 | [0.312, 0.962] | 0.326 | 0.856 |
| 14 | **P** | 0.881 | [0.789, 0.958] | 0.370 | 0.775 | 0.682 | [0.312, 0.966] | 0.608 | 0.804 |
| 15 | **P** | 0.889 | [0.803, 0.967] | 0.395 | 0.768 | 0.616 | [0.197, 0.966] | 0.579 | 0.782 |
| 16 | **P** | 0.882 | [0.795, 0.968] | 0.599 | 0.834 | 0.640 | [0.289, 0.973] | 0.350 | 0.829 |
| 17 | **P** | 0.855 | [0.748, 0.965] | 0.506 | 0.689 | 0.604 | [0.208, 0.965] | 0.534 | 0.828 |
| 18 | **P** | 0.856 | [0.739, 0.964] | 0.398 | 0.760 | 0.598 | [0.222, 0.968] | 0.144 | 0.856 |
| 19 |  | 0.848 | [0.721, 0.967] | 0.421 | 0.828 | 0.598 | [0.265, 0.968] | 0.656 | 0.853 |
| 20 |  | 0.841 | [0.712, 0.966] | 0.432 | 0.689 | 0.620 | [0.285, 0.967] | 0.567 | 0.904 |
| 21 |  | 0.820 | [0.672, 0.956] | 0.708 | 0.774 | 0.637 | [0.299, 0.968] | 0.898 | 0.940 |
| 22 |  | 0.808 | [0.662, 0.958] | 0.590 | 0.758 | 0.639 | [0.275, 0.971] | 0.776 | 0.836 |
| 23 |  | 0.802 | [0.645, 0.961] | 0.477 | 0.878 | 0.626 | [0.248, 0.974] | 0.535 | 0.841 |
| 24 |  | 0.783 | [0.603, 0.957] | 0.453 | 0.753 | 0.644 | [0.249, 0.967] | 0.562 | 0.883 |
| 25 |  | 0.807 | [0.643, 0.963] | 0.520 | 0.693 | 0.633 | [0.252, 0.969] | 0.792 | 0.894 |
| 26 |  | 0.785 | [0.597, 0.959] | 0.300 | 0.712 | 0.654 | [0.261, 0.972] | 0.207 | 0.928 |
| 27 |  | 0.786 | [0.617, 0.954] | 0.493 | 0.831 | 0.660 | [0.280, 0.983] | 0.633 | 0.851 |
| 28 |  | 0.784 | [0.609, 0.959] | 0.555 | 0.742 | 0.644 | [0.283, 0.984] | 0.234 | 0.865 |
| 29 |  | 0.822 | [0.687, 0.956] | 0.647 | 0.822 | 0.633 | [0.292, 0.984] | 0.387 | 0.917 |
| 30 |  | 0.804 | [0.673, 0.952] | 0.323 | 0.708 | 0.650 | [0.331, 0.969] | 0.714 | 0.923 |
| 31 |  | 0.735 | [0.577, 0.915] | 0.514 | 0.739 | 0.608 | [0.299, 0.900] | 0.165 | 0.853 |

#### Contrast 3 — vicious acts vs the persona-only baseline, and its act-free control (`answer`)

| L | band | C3 AUROC | 95% CI | rnd0 | floor | control 3 AUROC | 95% CI | rnd0 | floor |
|---|---|---|---|---|---|---|---|---|---|
| 0 |  | 0.784 | [0.656, 0.896] | 0.077 | 0.923 | 0.389 | [0.111, 0.667] | 0.556 | 0.667 |
| 1 |  | 0.878 | [0.778, 0.963] | 0.759 | 0.846 | 0.417 | [0.111, 0.778] | 0.333 | 0.722 |
| 2 |  | 0.887 | [0.763, 0.986] | 0.209 | 0.844 | 0.417 | [0.111, 0.778] | 0.583 | 0.750 |
| 3 |  | 0.991 | [0.966, 1.000] | 0.541 | 0.915 | 0.472 | [0.111, 0.833] | 0.611 | 0.667 |
| 4 |  | 1.000 | [1.000, 1.000] | 0.246 | 0.861 | 0.500 | [0.167, 0.833] | 0.750 | 0.750 |
| 5 |  | 1.000 | [1.000, 1.000] | 0.310 | 0.855 | 0.444 | [0.056, 0.833] | 0.583 | 0.667 |
| 6 | S | 1.000 | [1.000, 1.000] | 0.327 | 0.870 | 0.444 | [0.056, 0.778] | 0.333 | 0.750 |
| 7 | S | 1.000 | [1.000, 1.000] | 0.641 | 0.964 | 0.417 | [0.000, 0.778] | 0.667 | 0.722 |
| 8 | S | 1.000 | [1.000, 1.000] | 0.391 | 0.887 | 0.500 | [0.167, 0.778] | 0.556 | 0.694 |
| 9 | S | 1.000 | [1.000, 1.000] | 0.408 | 0.868 | 0.472 | [0.000, 0.889] | 0.500 | 0.667 |
| 10 | S | 1.000 | [1.000, 1.000] | 0.615 | 0.957 | 0.556 | [0.278, 0.806] | 0.611 | 0.778 |
| 11 | S | 1.000 | [1.000, 1.000] | 0.359 | 0.889 | 0.417 | [0.000, 0.778] | 0.472 | 0.778 |
| 12 |  | 1.000 | [1.000, 1.000] | 0.203 | 0.947 | 0.444 | [0.000, 0.778] | 0.806 | 0.806 |
| 13 |  | 1.000 | [1.000, 1.000] | 0.763 | 0.964 | 0.472 | [0.000, 0.944] | 0.639 | 0.639 |
| 14 | **P** | 1.000 | [1.000, 1.000] | 0.436 | 0.968 | 0.472 | [0.000, 0.889] | 0.333 | 0.667 |
| 15 | **P** | 1.000 | [1.000, 1.000] | 0.895 | 0.929 | 0.556 | [0.056, 1.000] | 0.222 | 0.806 |
| 16 | **P** | 1.000 | [1.000, 1.000] | 0.058 | 0.957 | 0.472 | [0.000, 1.000] | 0.611 | 0.694 |
| 17 | **P** | 1.000 | [1.000, 1.000] | 0.244 | 0.934 | 0.528 | [0.111, 0.944] | 0.472 | 0.694 |
| 18 | **P** | 1.000 | [1.000, 1.000] | 0.139 | 0.880 | 0.528 | [0.111, 0.944] | 0.500 | 0.750 |
| 19 |  | 1.000 | [1.000, 1.000] | 0.496 | 0.985 | 0.556 | [0.111, 0.944] | 0.417 | 0.667 |
| 20 |  | 1.000 | [1.000, 1.000] | 0.588 | 0.932 | 0.556 | [0.111, 0.944] | 0.500 | 0.694 |
| 21 |  | 1.000 | [1.000, 1.000] | 0.455 | 0.754 | 0.556 | [0.111, 0.944] | 0.611 | 0.750 |
| 22 |  | 1.000 | [1.000, 1.000] | 0.143 | 0.925 | 0.528 | [0.111, 0.944] | 0.583 | 0.694 |
| 23 |  | 1.000 | [1.000, 1.000] | 0.286 | 0.756 | 0.444 | [0.056, 0.889] | 0.583 | 0.778 |
| 24 |  | 1.000 | [1.000, 1.000] | 0.513 | 0.951 | 0.389 | [0.056, 0.833] | 0.528 | 0.806 |
| 25 |  | 1.000 | [1.000, 1.000] | 0.137 | 0.929 | 0.389 | [0.056, 0.833] | 0.528 | 0.778 |
| 26 |  | 1.000 | [1.000, 1.000] | 0.220 | 0.838 | 0.472 | [0.139, 0.806] | 0.472 | 0.750 |
| 27 |  | 1.000 | [1.000, 1.000] | 0.868 | 0.868 | 0.583 | [0.167, 0.889] | 0.500 | 0.639 |
| 28 |  | 1.000 | [1.000, 1.000] | 0.590 | 0.925 | 0.472 | [0.194, 0.751] | 0.611 | 0.750 |
| 29 |  | 1.000 | [1.000, 1.000] | 0.212 | 0.831 | 0.500 | [0.277, 0.778] | 0.417 | 0.667 |
| 30 |  | 1.000 | [1.000, 1.000] | 0.656 | 0.962 | 0.556 | [0.333, 0.833] | 0.472 | 0.833 |
| 31 |  | 1.000 | [1.000, 1.000] | 0.393 | 0.938 | 0.694 | [0.444, 1.000] | 0.667 | 0.917 |

### 7.3 Diagnostic D3 at `answer` — act vs no-act **inside** the same route

Same construction as §2.4: positives are the committed runs, negatives are the **same targets' runs that did not commit**, read at the same position — for deceived, the same target's chains that never committed, at the same turn index. Same context on both sides; only commission differs. Still a diagnostic, still no verdict attached to it.

| route | test pos / neg | best layer ≤ 30 | AUROC | 95 % CI | random 0 | floor | literal | vs floor | at `into` (§2.4) |
|---|---|---|---|---|---|---|---|---|---|
| deceived | 31 / 84 | L0 | **0.835** | [0.718, 0.925] | 0.408 | 0.614 | **pass** (18 layers; NEAR at L27, L29) | fail | 0.680, fail |
| akratic | 24 / 36 | L27 | **0.816** | [0.611, 0.973] | 0.359 | 0.804 | **pass** (7 layers) | fail | 0.704, fail |
| vicious | 31 / 41 | **L14** | **0.864** | [0.793, 0.916] | 0.329 | 0.681 | **pass** (8 layers; NEAR at L1, L6, L21, L25, L26) | fail | 0.684, fail |

**All three routes now show a detectable act-linked separation under the locked rule, where none did at `into`.** In the D-024 primary band L14–18 the values are deceived 0.793–0.805 (CI lower 0.698–0.731), akratic 0.785–0.794 (CI lower 0.548–0.561), vicious 0.765–0.864 (CI lower 0.679–0.793) — and vicious's best layer, L14, sits inside that band. The secondary band L6–11 is flat against it (deceived 0.781–0.802, akratic 0.788–0.800, vicious 0.724–0.843). **None of the three clears the same rule against the symmetrized floor** (margins −0.088, −0.393, −0.097 at their best layers), so the honest statement is that the separation is real under the pre-registered criterion and is not comfortably clear of what a random direction achieves on these pools.

| L | band | D3 deceived AUROC | 95% CI | rnd0 | floor | D3 akratic AUROC | 95% CI | rnd0 | floor | D3 vicious AUROC | 95% CI | rnd0 | floor |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 |  | 0.835 | [0.718, 0.925] | 0.408 | 0.614 | 0.749 | [0.448, 0.922] | 0.397 | 0.694 | 0.782 | [0.700, 0.863] | 0.493 | 0.619 |
| 1 |  | 0.777 | [0.678, 0.890] | 0.545 | 0.669 | 0.766 | [0.485, 0.946] | 0.370 | 0.733 | 0.716 | [0.626, 0.786] | 0.372 | 0.718 |
| 2 |  | 0.832 | [0.744, 0.907] | 0.380 | 0.737 | 0.770 | [0.534, 0.945] | 0.509 | 0.703 | 0.755 | [0.650, 0.854] | 0.314 | 0.686 |
| 3 |  | 0.818 | [0.735, 0.876] | 0.610 | 0.759 | 0.779 | [0.570, 0.949] | 0.617 | 0.780 | 0.723 | [0.610, 0.854] | 0.479 | 0.690 |
| 4 |  | 0.814 | [0.727, 0.876] | 0.633 | 0.694 | 0.791 | [0.566, 0.957] | 0.660 | 0.759 | 0.732 | [0.619, 0.857] | 0.563 | 0.688 |
| 5 |  | 0.795 | [0.701, 0.864] | 0.366 | 0.740 | 0.803 | [0.590, 0.976] | 0.358 | 0.688 | 0.704 | [0.584, 0.857] | 0.455 | 0.678 |
| 6 | S | 0.796 | [0.704, 0.877] | 0.313 | 0.746 | 0.797 | [0.580, 0.979] | 0.376 | 0.758 | 0.728 | [0.632, 0.876] | 0.396 | 0.636 |
| 7 | S | 0.798 | [0.715, 0.882] | 0.649 | 0.735 | 0.800 | [0.585, 0.962] | 0.727 | 0.737 | 0.724 | [0.629, 0.867] | 0.507 | 0.622 |
| 8 | S | 0.801 | [0.713, 0.892] | 0.356 | 0.786 | 0.788 | [0.567, 0.959] | 0.451 | 0.760 | 0.741 | [0.658, 0.877] | 0.618 | 0.744 |
| 9 | S | 0.802 | [0.723, 0.890] | 0.497 | 0.680 | 0.792 | [0.580, 0.964] | 0.294 | 0.789 | 0.751 | [0.691, 0.895] | 0.558 | 0.622 |
| 10 | S | 0.802 | [0.717, 0.898] | 0.491 | 0.611 | 0.800 | [0.558, 0.982] | 0.536 | 0.779 | 0.803 | [0.721, 0.901] | 0.623 | 0.791 |
| 11 | S | 0.781 | [0.690, 0.897] | 0.596 | 0.786 | 0.791 | [0.532, 0.967] | 0.714 | 0.762 | 0.843 | [0.754, 0.905] | 0.530 | 0.677 |
| 12 |  | 0.811 | [0.718, 0.909] | 0.525 | 0.765 | 0.788 | [0.557, 0.963] | 0.608 | 0.777 | 0.851 | [0.805, 0.895] | 0.349 | 0.758 |
| 13 |  | 0.785 | [0.693, 0.901] | 0.556 | 0.655 | 0.791 | [0.575, 0.968] | 0.442 | 0.699 | 0.855 | [0.802, 0.898] | 0.615 | 0.752 |
| 14 | **P** | 0.793 | [0.698, 0.898] | 0.231 | 0.769 | 0.785 | [0.561, 0.966] | 0.361 | 0.696 | 0.864 | [0.793, 0.916] | 0.329 | 0.681 |
| 15 | **P** | 0.804 | [0.717, 0.896] | 0.640 | 0.818 | 0.786 | [0.556, 0.962] | 0.307 | 0.817 | 0.817 | [0.746, 0.925] | 0.566 | 0.730 |
| 16 | **P** | 0.805 | [0.731, 0.895] | 0.339 | 0.661 | 0.793 | [0.553, 0.962] | 0.355 | 0.819 | 0.796 | [0.716, 0.920] | 0.504 | 0.718 |
| 17 | **P** | 0.798 | [0.724, 0.895] | 0.456 | 0.753 | 0.794 | [0.548, 0.967] | 0.513 | 0.701 | 0.771 | [0.688, 0.909] | 0.478 | 0.800 |
| 18 | **P** | 0.804 | [0.724, 0.894] | 0.474 | 0.776 | 0.792 | [0.554, 0.965] | 0.650 | 0.671 | 0.765 | [0.679, 0.901] | 0.660 | 0.663 |
| 19 |  | 0.797 | [0.717, 0.893] | 0.500 | 0.737 | 0.796 | [0.570, 0.964] | 0.606 | 0.715 | 0.765 | [0.679, 0.893] | 0.443 | 0.739 |
| 20 |  | 0.786 | [0.707, 0.889] | 0.349 | 0.729 | 0.795 | [0.571, 0.964] | 0.310 | 0.709 | 0.767 | [0.671, 0.893] | 0.497 | 0.654 |
| 21 |  | 0.789 | [0.712, 0.887] | 0.698 | 0.782 | 0.792 | [0.572, 0.972] | 0.564 | 0.759 | 0.735 | [0.638, 0.881] | 0.356 | 0.699 |
| 22 |  | 0.788 | [0.709, 0.884] | 0.378 | 0.690 | 0.786 | [0.577, 0.972] | 0.618 | 0.699 | 0.726 | [0.623, 0.866] | 0.670 | 0.707 |
| 23 |  | 0.769 | [0.699, 0.877] | 0.371 | 0.644 | 0.785 | [0.566, 0.969] | 0.372 | 0.684 | 0.730 | [0.612, 0.866] | 0.504 | 0.671 |
| 24 |  | 0.765 | [0.693, 0.872] | 0.548 | 0.677 | 0.786 | [0.573, 0.969] | 0.684 | 0.789 | 0.734 | [0.614, 0.869] | 0.519 | 0.640 |
| 25 |  | 0.758 | [0.690, 0.869] | 0.451 | 0.801 | 0.793 | [0.583, 0.966] | 0.604 | 0.722 | 0.750 | [0.642, 0.891] | 0.440 | 0.736 |
| 26 |  | 0.752 | [0.679, 0.868] | 0.449 | 0.719 | 0.799 | [0.590, 0.967] | 0.509 | 0.689 | 0.738 | [0.627, 0.875] | 0.268 | 0.732 |
| 27 |  | 0.744 | [0.671, 0.860] | 0.344 | 0.787 | 0.816 | [0.611, 0.973] | 0.359 | 0.804 | 0.729 | [0.613, 0.864] | 0.565 | 0.641 |
| 28 |  | 0.757 | [0.683, 0.864] | 0.579 | 0.786 | 0.807 | [0.598, 0.980] | 0.579 | 0.863 | 0.724 | [0.604, 0.869] | 0.456 | 0.640 |
| 29 |  | 0.746 | [0.673, 0.859] | 0.414 | 0.662 | 0.804 | [0.578, 0.972] | 0.476 | 0.696 | 0.702 | [0.594, 0.847] | 0.596 | 0.729 |
| 30 |  | 0.758 | [0.693, 0.859] | 0.432 | 0.754 | 0.797 | [0.583, 0.973] | 0.472 | 0.719 | 0.699 | [0.588, 0.847] | 0.369 | 0.689 |
| 31 |  | 0.786 | [0.708, 0.876] | 0.309 | 0.691 | 0.786 | [0.565, 0.961] | 0.564 | 0.713 | 0.646 | [0.507, 0.781] | 0.408 | 0.726 |

### 7.4 Cosines at `answer`

Band means, each contrast direction against the borrowed axes and against `randctl` seed 0:

| direction | · refusal | · badmed | · persona | · random 0 |
|---|---|---|---|---|
| deceived, primary L14–18 | +0.092 | +0.114 | −0.082 | −0.002 |
| deceived, secondary L6–11 | +0.010 | +0.090 | −0.128 | −0.005 |
| akratic, primary | +0.186 | **+0.315** | −0.157 | −0.001 |
| akratic, secondary | +0.054 | +0.184 | −0.206 | −0.009 |
| vicious, primary | **+0.219** | +0.217 | −0.118 | −0.004 |
| vicious, secondary | +0.074 | +0.105 | −0.121 | +0.000 |

| pair | primary band | secondary band | min over 32 layers | max | same pair at `into` (§3) |
|---|---|---|---|---|---|
| deceived · akratic | **+0.496** | +0.509 | +0.384 | +0.652 | +0.173 / +0.163 |
| deceived · vicious | +0.247 | +0.222 | +0.120 | +0.354 | +0.138 / +0.254 |
| akratic · vicious | +0.329 | +0.322 | +0.183 | +0.467 | +0.363 / +0.307 |

| same contrast, `into` direction · `answer` direction | primary band | secondary band | min | max |
|---|---|---|---|---|
| deceived | +0.290 | +0.222 | +0.111 | +0.312 |
| akratic | +0.182 | +0.117 | −0.018 | +0.246 |
| vicious | +0.318 | +0.245 | −0.019 | +0.385 |

Three readings. All three directions remain orthogonal to the random arrow (\|cos\| ≤ 0.009 in both bands), as they must be. The alignment with the **badmed** axis rises at this position and is largest for akratic (+0.315 in the primary band, against +0.149 at `into`), which is what one expects of a direction extracted from the text of a harmful medical answer rather than from the prompt that precedes it; the persona axis is now consistently negative for all three. And the `into` and `answer` directions of the same contrast are **largely different directions** (+0.11 to +0.39), so §3's cosine table and this one are not two views of one arrow.

| L | band | dec·refusal | dec·badmed | dec·persona | akr·refusal | akr·badmed | akr·persona | vic·refusal | vic·badmed | vic·persona | dec·akr | dec·vic | akr·vic |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 |  | +0.026 | +0.079 | -0.225 | +0.019 | +0.174 | -0.335 | -0.028 | +0.037 | -0.251 | +0.652 | +0.354 | +0.363 |
| 1 |  | +0.096 | -0.030 | -0.120 | +0.043 | +0.110 | -0.286 | +0.050 | +0.054 | -0.295 | +0.640 | +0.240 | +0.328 |
| 2 |  | +0.077 | +0.055 | -0.112 | +0.061 | +0.203 | -0.326 | +0.038 | +0.121 | -0.296 | +0.594 | +0.199 | +0.365 |
| 3 |  | +0.038 | +0.076 | -0.094 | +0.007 | +0.190 | -0.245 | +0.070 | +0.050 | -0.207 | +0.552 | +0.133 | +0.246 |
| 4 |  | +0.043 | +0.046 | -0.053 | -0.000 | +0.168 | -0.197 | +0.042 | +0.038 | -0.198 | +0.521 | +0.122 | +0.216 |
| 5 |  | +0.051 | +0.039 | -0.056 | +0.012 | +0.146 | -0.176 | +0.019 | +0.024 | -0.159 | +0.526 | +0.120 | +0.183 |
| 6 | S | +0.020 | +0.061 | -0.097 | +0.008 | +0.156 | -0.178 | +0.045 | +0.060 | -0.142 | +0.521 | +0.167 | +0.220 |
| 7 | S | +0.004 | +0.071 | -0.114 | +0.019 | +0.179 | -0.206 | +0.065 | +0.077 | -0.119 | +0.508 | +0.197 | +0.259 |
| 8 | S | -0.011 | +0.102 | -0.142 | +0.022 | +0.196 | -0.234 | +0.033 | +0.100 | -0.138 | +0.499 | +0.222 | +0.322 |
| 9 | S | +0.003 | +0.073 | -0.105 | +0.047 | +0.163 | -0.181 | +0.062 | +0.098 | -0.093 | +0.490 | +0.207 | +0.318 |
| 10 | S | -0.000 | +0.119 | -0.152 | +0.096 | +0.187 | -0.205 | +0.099 | +0.145 | -0.112 | +0.518 | +0.280 | +0.398 |
| 11 | S | +0.044 | +0.111 | -0.160 | +0.128 | +0.219 | -0.233 | +0.140 | +0.151 | -0.120 | +0.520 | +0.256 | +0.417 |
| 12 |  | +0.055 | +0.161 | -0.161 | +0.147 | +0.285 | -0.249 | +0.149 | +0.183 | -0.129 | +0.521 | +0.299 | +0.435 |
| 13 |  | +0.108 | +0.183 | -0.145 | +0.247 | +0.330 | -0.223 | +0.227 | +0.238 | -0.124 | +0.518 | +0.309 | +0.436 |
| 14 | **P** | +0.119 | +0.170 | -0.128 | +0.266 | +0.364 | -0.227 | +0.230 | +0.268 | -0.151 | +0.529 | +0.300 | +0.423 |
| 15 | **P** | +0.095 | +0.114 | -0.086 | +0.201 | +0.290 | -0.169 | +0.198 | +0.209 | -0.129 | +0.478 | +0.242 | +0.344 |
| 16 | **P** | +0.090 | +0.090 | -0.085 | +0.195 | +0.290 | -0.150 | +0.251 | +0.211 | -0.126 | +0.495 | +0.236 | +0.316 |
| 17 | **P** | +0.076 | +0.104 | -0.072 | +0.155 | +0.325 | -0.139 | +0.208 | +0.212 | -0.125 | +0.485 | +0.231 | +0.296 |
| 18 | **P** | +0.077 | +0.093 | -0.041 | +0.113 | +0.308 | -0.101 | +0.208 | +0.184 | -0.060 | +0.494 | +0.224 | +0.269 |
| 19 |  | +0.060 | +0.087 | -0.033 | +0.073 | +0.316 | -0.106 | +0.185 | +0.198 | -0.054 | +0.495 | +0.213 | +0.270 |
| 20 |  | +0.065 | +0.095 | -0.028 | +0.070 | +0.334 | -0.088 | +0.193 | +0.193 | -0.096 | +0.490 | +0.211 | +0.251 |
| 21 |  | +0.020 | +0.095 | +0.008 | +0.011 | +0.349 | -0.058 | +0.164 | +0.166 | -0.070 | +0.476 | +0.176 | +0.234 |
| 22 |  | -0.045 | +0.104 | +0.032 | -0.037 | +0.355 | -0.046 | +0.150 | +0.163 | -0.082 | +0.476 | +0.156 | +0.234 |
| 23 |  | -0.055 | +0.106 | +0.033 | -0.053 | +0.344 | -0.035 | +0.158 | +0.156 | -0.094 | +0.477 | +0.143 | +0.210 |
| 24 |  | -0.087 | +0.120 | +0.021 | -0.071 | +0.359 | -0.044 | +0.152 | +0.161 | -0.095 | +0.488 | +0.128 | +0.206 |
| 25 |  | -0.054 | +0.125 | -0.001 | -0.022 | +0.355 | -0.053 | +0.148 | +0.184 | -0.105 | +0.483 | +0.167 | +0.248 |
| 26 |  | -0.071 | +0.127 | -0.012 | -0.034 | +0.358 | -0.059 | +0.125 | +0.194 | -0.094 | +0.480 | +0.169 | +0.254 |
| 27 |  | -0.025 | +0.141 | -0.051 | +0.036 | +0.385 | -0.087 | +0.132 | +0.237 | -0.105 | +0.474 | +0.197 | +0.286 |
| 28 |  | -0.036 | +0.149 | -0.054 | +0.027 | +0.388 | -0.088 | +0.099 | +0.248 | -0.108 | +0.461 | +0.197 | +0.297 |
| 29 |  | -0.000 | +0.139 | -0.059 | +0.110 | +0.394 | -0.110 | +0.100 | +0.272 | -0.084 | +0.446 | +0.200 | +0.330 |
| 30 |  | +0.001 | +0.118 | -0.073 | +0.128 | +0.413 | -0.098 | +0.070 | +0.317 | -0.078 | +0.442 | +0.217 | +0.372 |
| 31 |  | +0.053 | +0.025 | -0.073 | +0.102 | +0.331 | +0.064 | -0.014 | +0.239 | +0.125 | +0.384 | +0.150 | +0.467 |

### 7.5 For the researcher, not a verdict — the two positions side by side

At `into` the three contrasts separated at AUROC 1.000 and none of the separations was evidence about the harmful act: contrast 2's act-free control separated just as perfectly, contrast 3's control could not be computed at all, the single-turn pools held 17, 20, 22 and 5 distinct vectors behind nominal Ns of 62, 180, 83 and 30, and the within-route diagnostic failed on every route. **At `answer` every one of those defects is gone**: nominal and effective N coincide (109, 62, 180, 83, 30 distinct vectors, no seed pair identical), all three act-free controls fail — control 2 falls from 1.000 to 0.746 and control 3 becomes computable and fails at 0.583 — and D3, act against no-act inside the same route with the context held fixed, passes the locked rule on all three routes (deceived 0.835, akratic 0.816, vicious 0.864 at their best layers ≤ 30, with vicious's best layer L14 inside the D-024 primary band). That is a real change and it is the change the addendum was ordered to test: the position that reads the model's own answer carries an act-linked signal that the position reading the prompt before it does not. **Three cautions sit against it, and they are not small.** First, what the `answer` position reads is a mean over the tokens of the act itself, so "the answer that committed separates from the answer that did not" partly restates the act judge's grading in activation space; it is evidence that the act is represented while it is being made, and it is *not* by itself evidence that the act leaves a mark that outlives the turn — which is the kind of mark STAGE0 §4.2 and §4.3 are defined on and the kind S4 reads, one or more turns later, after feedback. Second, contrast 3 remains **perfectly separable by prompt length alone** (1.000, ranges disjoint), a property of the pools that no readout position can repair, so its pass carries no information; contrast 2 fails the criterion against the symmetrized random floor at every layer; and where contrasts 1 and 3 clear that floor they clear it by 0.001–0.065, not by the 0.20 the rule intends as a margin, while D3 clears it nowhere. Third, every deceived number is still provisional in T, since `t9/T_source.json` does not exist. Read together, the honest summary is that the fallback's context-matching plus the `answer` position gives, for the first time in S1, a within-route act-versus-no-act separation that survives its own act-free control — at 0.79–0.86, not at ceiling, and not clear of what a random direction achieves on these pools. Whether that is enough to call any route verified inside is the researcher's and the hub's decision; this session states the numbers and their conditions and decides nothing.

### 7.6 What still does not work at this position

1. **Contrast 3 is separable by prompt length alone** (AUROC 1.000). Unchanged by the position, because prompt length is a property of the pools. Its literal pass and its two floor-clearing layers should not be read as evidence.
2. **The symmetrized random floor stays high** — 0.698 / 0.672 / 0.754 minimum over layers ≤ 30. Contrast 2 fails against it everywhere; contrasts 1 and 3 clear it by hundredths at 5 and 2 layers; D3 clears it on no route. A random direction still separates these pools at roughly 0.7–0.9.
3. **D3's CIs are wide where N is small** — akratic's best layer is 0.816 with CI [0.611, 0.973] on 24 positives against 36 negatives, 5 target clusters.
4. **T remains `T_primary`** (§5.6); `--T-source` re-runs both positions unchanged when the adjudication labels exist.
5. **The `think` position was not run**, per the addendum; adherence inside chains is 0.53 and in the akratic cell 0.39 (S1b §12), so the drop would bias the pools. The script would report the drop counts if it were ever ordered.
6. Everything the addendum ordered was executed in full; nothing else in the repository was changed.

**Vocabulary, this section included:** STAGE0 §2 terms throughout; "the researcher" throughout. A whole-word grep for the five banned terms over this report and every file this session wrote returns nothing.
