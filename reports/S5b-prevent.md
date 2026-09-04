# Report — S5b: can the capitulation be *prevented* by steering against the susceptibility direction?

**Status: EXPLORATORY, and this is the only place it needs saying.** The hypothesis comes from a predictive
readout (S1e/S1g), not from a pre-registered prediction, and **`nn` is a generic negative-valence arrow built
from S2's neutral-negative first-person passages that was *found to predict* capitulation — it is not a
"capitulation direction".** An AUROC of 0.66–0.71 is a modest correlation, not a clean separation. Steering
here is on for **every turn of the chain**, which is a stronger intervention than the turn-1 readout strictly
licenses.

Machine-written numbers: `results/S5b_table.md`, regenerated from `results/raw/s5b` by
`scripts/rig/s5b_table.py`. Nothing below is hand-entered.

## 1. What ran

Target `burn-blister-pop`, its frozen `_v1` chain, N = 8, seeds 0–7, think-aloud on, deceived route. **Chain
only**: no feedback turn, no forks, no distance-4 set, no topic controls (`--chain-only`, which the rig
refuses to combine with `--distance4` or `--controls`). The act judge grades every turn on `gpt-4o-mini`
against the target's own act rubric, exactly as every other cell reaches the act.

| arm | steering, every assistant turn of the chain | injected norm |
|---|---|---|
| **P0** | none | — |
| **P1** | `−4 · σ(nn) · n̂n` at L16 | **2.649875** |
| **P2** | `−4 · σ(nn) · û_random0` at L16, norm-matched | **2.649875** |

σ(nn) at L16 = **0.662469**, the sample sd (ddof = 1) of the 200 first-person passage `mean` projections on
the unit arrow at that layer — the S2b addendum's recipe, unchanged. `--steer` **accepts a negative `c`
directly**, so the arrow was not negated by hand and the sign is the flag's. P0 was re-run rather than reused
from any S4 cell, so all three arms share one code path and one seed set.

**The injected norms are equal to 4.77e-07**, printed in §A of the table. P2 is norm-matched, not σ-matched per
arrow: `random0`'s own σ at L16 is 0.040389, so the per-arrow recipe would have injected a vector about
sixteen times smaller and could not have ruled out "a perturbation of this size does this".

**Pre-hook norm check, both steered arms, before any run** (§A of the table): worst relative deviation
**1.39e-04** for P1 and **5.17e-05** for P2, against the 5 % tolerance — both **PASS**.

## 2. The result

| arm | N | committed | act rate | 95 % CI (clustered on the run) | coherence mean | coherence min |
|---|---|---|---|---|---|---|
| P0 unsteered | 8 | 8 | **1.000** | [1.000, 1.000] | 91.6 | 83.7 |
| P1 `−4·σ(nn)·n̂n` | 8 | 8 | **1.000** | [1.000, 1.000] | 89.1 | 76.7 |
| P2 `−4·σ(nn)·û_random0` | 8 | 7 | **0.875** | [0.625, 1.000] | 89.1 | 69.9 |

**Steering against the `nn` arrow did not prevent the act, and the detectable difference says how much that
claim is worth: with 8 runs per arm from a base of 8 of 8, one run is 0.125 of the act rate, so a drop to 3 of
8 or fewer is visible at this N and a drop of one or two runs is not — P1 shows a drop of zero runs, and
P1 − P2 is +0.125 [0.000, 0.375], i.e. the direction arm committed the act *more* often than the norm-matched
random arm, not less.** P1 − P0 is 0.000 [0.000, 0.000].

This is the third of the brief's four readable outcomes: **neither arm drops the act rate, so the readout is
predictive without being an intervention target at this magnitude and this layer.**

**It is not damage.** Coherence is the load-bearing control, and it holds: means 91.6 / 89.1 / 89.1 over 27,
31 and 38 act-turn answers, minima 83.7 / 76.7 / 69.9, and **no answer in any arm scored below 50**. The
−2.5-point coherence difference between the steered arms and P0 is small beside that spread, and in any case
there is no act-rate drop for it to explain away. Had P1 dropped the act rate with coherence falling, the
brief's fourth outcome would apply and this section would call it damage rather than prevention; it does not
arise.

## 3. What the readouts show, and what they do not

§D of the table prints every arrow at `into`, `think` and `answer` over both bands with the random floor, and
the **injected component `c·σ·cos(û_inj, axis)` beside them**. In P1 the `nn` projection at the primary band
moves to −1.583 against P0's −0.037. **That is the injection arriving, not a state change**: the injected
component on `nn` is −2.650 exactly, at L16 where it is exact, which is why it is printed in the same table.
The same caution applies to `guilt` (−1.338 injected) and `shame` (−1.182 injected), which share substantial
cosine with `nn`. Layers below 16 are untouched; layers above carry propagated effect, not injection.

## 4. Limits, stated plainly

- **One target, N = 8, one layer, one magnitude.** A null at c = −4 on L16 for `burn-blister-pop` is not a null
  for the arrow.
- **This target is the ceiling case on purpose.** Cell A measured its unsteered act rate at 1.000, chosen
  because all the headroom is on the side of prevention. That makes a failure to prevent informative and a
  success unambiguous, but it also means the act is very easy to reach here.
- The intervention is **stronger** than the readout licenses (every turn, not just turn 1), so this is not a
  weak test of the causal claim — it is a strong one that came back negative.
- The brief's named follow-up, `+4 · σ(nn) · n̂n` on a target with a **low** unsteered act rate to see whether
  capitulation can be *induced*, is **named and not run**, exactly as the brief says.

## 5. One rig failure and its fix

P0's first attempt died after all eight chains were generated: a transient `APIConnectionError` inside a
coherence call. The vendored `data/eval/judge.py` builds its own `AsyncOpenAI()` at the SDK default
`max_retries=2`, and plan A3 forbids editing the vendored file, so a bounded retry on connection- and
rate-class errors was added **at the call site** in `judges_rig.py`, covering both the coherence and the
unrelated-fork paths. The vendored call carries `seed=0` at `temperature=0`, so a retry re-asks the identical
question and no measured quantity changes. The retry fired twice on the re-run and the arm completed. The
failed attempt wrote no run data ($0.0079 on the ledger, counted).

## 6. Cost

$0.0402 over 236 judge calls for all three arms. Machine time 176 s + 128 s + 240 s.
