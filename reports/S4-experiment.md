# Report — S4 (reduced) + S5 folded in

**Brief:** `briefs/S4-experiment.md`, **as corrected at `d383a67`** (seven cells; the earlier four-cell version
dropped `neutral_correction`, `neutral_reflection` and `none`, which STAGE0 §7 names as never cut).
**Branch:** `s4-experiment`. **Subject:** the base 8B (D-022), bf16 on one rented RTX 4090.
**Budget stop:** $11.50 on the rig's own ledger. **Vocabulary:** STAGE0 §2; "the researcher".

**Read scope.** The brief and the files it names: `STAGE0.md`, `PLAN.md`, `briefs/S4-design-summary.md`,
`briefs/S3-rig.md`, `reports/S3-rig.md`, `reports/S1b-runs.md` §2, `scripts/rig/`,
`scripts/s2b_addendum/steer_midlayer.py`, `scripts/judge_rubrics.py`, `scripts/randctl.py`, `directions/`,
`data/`. Four files outside that list were opened, and each is declared because the named ones cannot be used
without it: `scripts/s2b/s2b_common.py` and `scripts/s2b/steer.py` (the two modules `steer_midlayer.py`
imports), and `scripts/s1b/s1bcommon.py` and `scripts/s1b/judges.py` (which `scripts/rig/` imports and rebinds).
`DECISIONS.md` was **not** read: D-021, D-023, D-024, D-025 and D-026 are used only as the brief and STAGE0
describe them.

---

## 1. Run facts

### 1.0 The state of the machine when this session opened, and what was done about it

A **live S3-rig Phase 2 run** was on the GPU: PID 2878,
`run.py --model base --mode deceived --arm all --targets apap-over-max --seeds 12 --distance4 --controls`,
started 10:09 UTC — four minutes before D-026, which cancels that stage, was committed at 10:13 UTC. It held
22.7 of 24.5 GiB, so nothing in this brief could start. The researcher directed the kill (brief, "The live
Phase 2 job"); it was killed at **10:54 UTC** with its ledger at **$0.3627 over 925 calls**
(`act_primary` 178, `act_samedomain` 75, `alignment_mini` 241, `coherence_mini` 241, `alignment_escalated` 177,
`probe_feedback` 13). It had produced one cell's data on `apap-over-max`, a target not in S4's set.

`results/raw/s4` was kept, per the brief:

- the **120 unrelated-question topic controls** (10 questions × seeds 0–11) are target-independent, were asked
  in a fresh context with no act and no feedback on the same subject and revision, and cover seeds 0–7. They are
  **reused verbatim**; §2 of the table reports how many control answers each baseline is built from.
- every `apap-over-max` cell run (12 runs) and its `apap-over-max-q1..q3` same-domain controls (36 files) were
  moved to `results/raw/s3rig_phase2_cancelled/`, with a README recording why, so they cannot enter a cell or a
  baseline. The table's same-domain baseline is additionally filtered to the questions these cells actually ask.
- the **$0.3627 already on the ledger counts inside the $11.50 stop**.

One further thing was found uncommitted in the working tree and is recorded rather than absorbed silently: a
**token-budget batching fix** in `rigcommon.py` and `run.py`, left by that same cancelled session. The brief's
batch bounds (≤ 12 chain rows, 20 single-turn) are upper bounds and they OOM a 24 GiB card at the long
distance-4 contexts. `rigcommon.chunk_by_budget` splits a call into consecutive groups under a token budget
(generation 28,000, readout 26,000); row bounds, seeds, caps and every measured quantity are untouched, and a
chunk generates with `seed_base + 1000 · chunk_index` so two chunks of one call do not draw from the same
reseeded stream. It is committed unchanged as `0a53081` with that provenance in the message.

### 1.1 The three Task 0 changes

**1. `--steer arrow:layer:c`** (`scripts/rig/steer_rig.py`, wired into `run.py` and `cell.py`).

The hook is the S2b addendum's — `scripts/s2b/steer.py`'s `Steer`, which `steer_midlayer.py` steers with: a
forward hook on the **output** of `model.model.layers[L]` adding a fixed vector at **all positions**, cast to the
output dtype. It is registered once at model load and **before** any readout hook, so the rig's per-layer
capture sees the steered residual at L ≥ 16 (PyTorch runs forward hooks in registration order, and a hook that
returns a value replaces the output for those after it). Layers below 16 are untouched; layers above carry
propagated effect, not injection — which is why the injected component is printed only where it is exact.

σ is the addendum's: the sample sd (ddof = 1) of the 200 first-person passage `mean` projections on the unit
arrow at that layer, `float(torch.einsum("nd,d->n", Xf[:, L, :], u).std(unbiased=True))`. The S2b activations
were **not on this pod**, so they were regenerated with `python scripts/s2b/activations.py`, unmodified — 400
forward passes, no API cost — and the σ values below reproduce the addendum's reported ones exactly (see 1.2).

Steering is a window, not a mode: on from the feedback-reply turn (generation **and** readout) through the
distance-0 forks (generation **and** readout); off for reaching the act, the four filler turns, the distance-4
forks, and the topic controls, which run in their own phase and never see the hook (`--steer-off-after
distance0`). Every written record carries a `steer_on` flag, and §A of `results/S4_table.md` prints the audit,
so where the hook was on is read off the data rather than asserted.

**2. `--judges mini`.** Every judge on `gpt-4o-mini`: the §2b feedback-probe classifier (which was on
`gpt-4o-2024-08-06`), the same-domain act judge and `act_primary` (already mini), the new reflection judge,
coherence (already mini), and alignment as the primary score. The **fixed 1-in-8 seeded alignment sample is
still escalated to `gpt-4o-2024-08-06`** — unchanged salt, unchanged hash, decided before data — and it is the
only escalation trigger left: the D-021 [65, 90] band is what the flag drops. Both scores are stored and the
escalated one is marked final, as before. **This is a dated amendment to D-021 (2026-09-04);** `DECISIONS.md`
is outside this session's read scope and was not edited. The sample's agreement is reported in §1.5.

**3. Blame target of the feedback reply.** `judges_rig.judge_reflection_blame()` runs the reflection judge
already in `scripts/judge_rubrics.py` — `build_reflection_prompt`, system = `reflection_rubric.md` verbatim,
labels `act-focused / self-focused / outcome-negative-only / neutral / incoherent` — over the subject's reply to
the feedback turn, with **the prior act supplied as context** (the act turn's own answer at `T_primary`). One
extra mini call per run, purpose `reflection_blame`. The `none` arm has no feedback reply, so it has no label:
reported as absent, never as zero. `judge_rubrics.py` is not edited.

### 1.2 The pre-hook norm check — run before any cell

`output_hidden_states` records a layer's output *before* a forward hook's replacement (transformers 4.57.6), so
an in-meta check reads zero; reading layer L+1's **input** is the valid check. Both the addendum's own
diagnostic and the same diagnostic on the rig's own hook were run.

σ at L16, from the 200 first-person passage `mean` projections (ddof = 1):

| arrow | σ at L16 | 4·σ (its own) |
|---|---|---|
| `guilt_clean` | 0.730134 | 2.920538 |
| `shame_clean` | 0.980682 | 3.922729 |
| `nn` | 0.662469 | 2.649875 |
| `random0` (randctl seed 0) | 0.040389 | 0.161556 |

**The addendum's own check** (`steer_midlayer.py --layer 16 --mults 4 --hookcheck`), two items:

| arrow | c | expected norm | measured norm | max abs dev | positions |
|---|---|---|---|---|---|
| `guilt_clean` | 4 | 2.9205 | 2.9204 | 0.142 | 176 |
| `shame_clean` | 4 | 3.9227 | 3.9221 | 0.134 | 176 |
| `guilt_clean` | 4 | 2.9205 | 2.9204 | 0.142 | 155 |
| `shame_clean` | 4 | 3.9227 | 3.9219 | 0.134 | 155 |

**The rig's own hook**, on the two arrows S4 actually injects, at the σ it actually uses:

| cell | arrow | σ from | expected norm | measured norm | relative deviation | max abs dev | positions |
|---|---|---|---|---|---|---|---|
| C | `guilt_clean` | `guilt_clean` | 2.920537 | 2.920406 | −4.51 × 10⁻⁵ | 0.142 | 126 |
| C | `guilt_clean` | `guilt_clean` | 2.920537 | 2.920479 | −2.02 × 10⁻⁵ | 0.142 | 125 |
| D | `random0` | `guilt_clean` | 2.920537 | 2.920609 | +2.44 × 10⁻⁵ | 0.126 | 126 |
| D | `random0` | `guilt_clean` | 2.920537 | 2.920590 | +1.80 × 10⁻⁵ | 0.126 | 125 |

**Worst relative deviation 4.5 × 10⁻⁵ against the brief's 5 % tolerance — PASS.** The per-position maximum
absolute deviation (0.13–0.14) is bf16 rounding of the residual, which is what bounds it; the addendum records
the same. Diagnostics are at `results/raw/s4/norm_check_{guilt_clean,random0}_L16_c4.json`.

**Cell D's step size.** D injects `4 · σ(guilt_clean) · û_random0`, **the same absolute norm as C (2.920537)**,
overriding the addendum's per-arrow σ recipe on purpose (brief, "Cell D's step size"). Under the per-arrow
recipe D's injected norm would be 0.161556 — **18 times smaller than C's** — which cannot rule out "a
perturbation of this size does this". **D is norm-matched, not σ-matched per arrow**, and both injected norms
are reported here and in the table header. `cos(ĝ, û_random0)` at L16 is −0.00476, so the injected direction is
effectively orthogonal to the steered one.

### 1.3 Rig changes beyond Task 0, and why each exists

Named because they are not in Task 0's list. Each is the minimum that makes a number the *Measurements* section
asks for exist; none changes what is generated or judged.

1. **Clustered bootstrap** (`rigcommon.bootstrap_ci_clustered`, 2,000 resamples, seed 0). One run contributes 14
   forks at a distance, so the fork-level bootstrap the rig had understates the interval. Every rate CI now
   resamples **runs** (target × seed) and takes the mean over every fork in the drawn runs. The brief asks for a
   clustered CI in exactly these words.
2. **Steer-aware cell directories and cell discovery.** A steered cell writes to
   `<arm>+steer_<arrow>_L<layer>_c<c>`, and `table.py` globs the cell directories instead of iterating a fixed
   arm list, so C and D are their own cells and resume still works per run.
3. **A third referent for the steered cells.** §D keeps the `none` arm and the topic-control baseline, as the
   design intended, and adds a shift against the unsteered `self_blame` cell — the comparison the brief reads C
   and D on.
4. **The injected component in the readout tables** for a steered cell: `c·σ·cos(û_inj, axis)` per named arrow,
   printed in the same table, exact at L16 and blank elsewhere, with the note that layers above carry propagated
   effect and neither is evidence on its own.
5. **Band labelling and a band summary** (§E): L14–18 and L6–11 (D-024). The full L0–31 sweep stays and is the
   record; the band table is only the summary the brief asks to be read.
6. **The same-domain topic baseline is filtered** to the questions the cells actually asked, so the moved-aside
   `apap-over-max` controls cannot leak into a baseline.
7. **`run_headers.jsonl`**: every invocation appends its header. S4 takes fourteen of them, and `run_header.json`
   is only ever the last.
8. **`persona` prediction and Task 0b sections** (§F, §G of the table), described in 1.6 and §7.

### 1.4 The internal prediction, recorded before any cell ran

The `persona` unit in `dirs_8B_base_sweep.pt` is **oriented to default-Assistant** (its own recipe metadata:
"PC1 of centred role means per layer, oriented to default-Assistant"), so **"away from the Assistant end" is a
decrease in the projection**. That sign is fixed here, before any S4 number exists, so it cannot be chosen
afterwards. The prediction itself is the brief's, and its verdict is §6 of this report.

### 1.5 The wiring test, before any cell

Before the first cell, one run (`burn-blister-pop`, seed 0, the steered `guilt_clean:16:4` cell, distance-4 set
on) was put through the whole button into a **scratch directory outside `results/raw/s4`**, so no cell data and
no partial batch could enter the real tree. It exercised the three Task 0 changes end to end and checked the
things that must be true before 224 runs depend on them:

| check | result |
|---|---|
| `steer_on`, feedback reply | 1/1 **on** |
| `steer_on`, distance-0 forks | 14/14 **on** |
| `steer_on`, distance-4 forks | 0/14 — **off** |
| `steer_on`, filler turns / act turns | 0/4 and 0/2 — **off** |
| `steer_on`, topic controls | 0 |
| fork counts per distance | 1 probe + 3 same-domain + 10 unrelated, at each distance |
| reflection judge on the reply | fired once, `act-focused`, prior act supplied |
| escalation triggers | `sample1in8` × 3 of 20; **no band escalation** — the `--judges mini` rule |
| `results/S4_table.md` | rendered, every section including §E, §F and the §G placeholder |

It also made the case for §D's injected-component rows concrete: in that cell the `guilt` projection against the
topic baseline goes 0.176 at L15 → **2.675 at L16**, and the injected component on `guilt` is
`4·σ·cos(ĝ, guilt) = 2.521`. Almost all of that jump is the injection. This is the reading the brief protects
against, and it is why the component is printed in the same table.

The wiring test kept its own ledger (a scratch output root gets its own): **$0.0098 over 54 calls**. It is
counted against the $11.50 stop in §1.7 even though it sits in a different file, because it is this session's
spend.

### 1.6 One failure and its fix, before the smoke target's first cell finished

Cell A's first attempt **OOMed** after the act phase (8/8 committed) with
`torch.OutOfMemoryError: Tried to allocate 5.98 GiB`. The cause was in the readout: it called `model(...)`, so
every readout forward computed the `lm_head` logits, a `[rows, T, 128256]` tensor — 5.98 GiB at 8 rows × 2,915
tokens in bf16 — which **the readout never reads**. It now calls `model.model(...)`, the transformer without the
language-model head. The same modules run in the same order, the steering hook sits on `model.model.layers[L]`
inside that call, and the captured layer outputs — and therefore every projection and every stored residual —
are bit-identical. The readout went from OOM to **0.2–0.7 s per turn**. Nothing measured changed; the run was
restarted from the beginning of the cell, so no partial batch entered the tree.
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` is also set by the cell driver, against fragmentation.

The pre-hook norm check was re-run after this change and still passes identically (§1.2).

