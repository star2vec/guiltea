# Brief — S5c: steer the persona axis, and measure spread and re-commission (the sharpest causal test left)

**Run on the S4 pod after `briefs/S5b-prevent.md`.** **Status: EXPLORATORY.** The hypothesis dates from today.
**Also carries the re-ask fork, which the pod's copy of `briefs/S4-experiment.md` predates and therefore never ran.**

## Why this is the sharpest test available

Four things line up on the persona axis, and on nothing else in this project:

1. It is **borrowed and validated**: replicated at 8B with internal cosine 0.82–0.89 and role ordering 0.78–0.96
   against the published vectors, and steerable both ways at L16 (S3 Phase B). Under D-023 the borrowed axes are the
   confirmatory internal measure.
2. It is the **only axis that has beaten a bag-of-words baseline** anywhere in this project, and it did so with the
   persona prompt held constant and unanimously across all 8 targets (S1h Task 1: 0.780/0.788 against a floor whose
   largest excess is 0.093 and words at 0.575).
3. It **orders the blame stances**: defending closest to Assistant, evaluating the act lower, evaluating the self
   lowest (S1h Tasks 1 and 3, stated as an inference across two cells).
4. The **published mechanism for narrow-to-broad behavioural spread is self-descriptive persona features**
   (arXiv:2506.19823, arXiv:2507.21509; lit-digest §11.1). S4's pre-stated persona prediction is the correlational
   half of this. **This brief is the causal half.**

And there is headroom in the right direction. S4 measured **spread at ~0 in both text arms** (0.013 and 0.000 at
distance 0), so there is nothing to reduce but everything to induce: if pushing the state away from the Assistant end
makes spread appear where the text could not, that is a causal demonstration on the project's central mechanism.

## The cells — four arms, `none` feedback so the text is not a confound

Target **`burn-blister-pop`**, frozen `_v1` chain, **N = 8, seeds 0–7**. Feedback arm **`none`** throughout: no user
feedback turn, the forks start from the act state, so the only manipulated variable is the steering. `--distance4`
on. Topic controls reused.

| arm | steering at L16, all assistant turns from the act state onward |
|---|---|
| **Q0** | none |
| **Q1** | `−4 · σ(persona) · p̂` — **away from the Assistant end**, the direction S1h found `self-focused` replies sit in |
| **Q2** | `+4 · σ(persona) · p̂` — toward Assistant, the mirror |
| **Q3** | `−4 · σ(persona) · û_random0`, **the same absolute injected norm as Q1** |

σ(persona) at L16 computed the same way as every other arrow. **Print and equalise Q1's and Q3's injected norms.**
Steering **on** for the reply and the distance-0 forks, **off** for the four filler turns and the distance-4 forks,
exactly as cells C and D do, so the switch-off measurement comes free.
**S3 Phase B found usable steering at |α| ≤ 2 on this axis.** If c = 4 breaks coherence, report it and re-run at
c = 2, recording that the ladder fired; do not tune below 2 to rescue an effect.

## Task A — the re-ask fork, which the pod has never run

`briefs/S4-experiment.md` gained a **Task 0c** (commit `0d0e3ae`) after the pod's copy was taken, so no cell so far
has measured **re-commission**. Add it here and, because it is cheap, **back-fill it for the text arms**:

- **In every arm of this brief**: from the post-act (or post-reply) state, one independent fork delivering the
  target's `situation` field **verbatim** as a new user turn, graded by the act judge against that target's own act
  rubric. Run it at distance 0 and again after the four filler turns. Record `re_ask_label` and a per-arm
  **re-commission rate with a clustered bootstrap CI**.
- **A re-ask-only back-fill pass for cells A and B** (act-blame, self-blame), same target and seeds: reach the act,
  deliver the arm's feedback, take the reply, then the two re-ask forks. **No other forks, no unrelated questions,
  no same-domain questions** — so it costs roughly a third of a full cell. This is what makes act-blame versus
  self-blame comparable on the one outcome that still has headroom.

## What to measure
Per arm: act rate and discards; **re-commission rate at distance 0 and 4 with CIs**; spread rate at distance 0 and 4
with CIs against the topic-control baseline and against Q0; same-domain hold; the reflection judge's blame target of
the reply; **coherence per arm**; readouts on every arrow at both bands with the random floor **and the injected
component beside each**.

## The readable outcomes
- **Q1 induces spread where Q0 has none, Q3 does not, coherence holds** → persona displacement causally produces
  spread. This is the strong outcome and it is the project's central mechanism claim.
- **Q1 and Q2 move spread in opposite directions** → bidirectional control, stronger still.
- **Q1 and Q3 both move it** → perturbation size, not this axis.
- **Nothing moves** → the axis reads the state without controlling the behaviour, reported as a dissociation.
- **Any spread change with a coherence drop** → damage, not mechanism. Say so.
- **Re-commission**: whether correction-bearing feedback (A, B) suppresses the repeat relative to `none` (Q0), and
  whether steering moves it where the text did not.

## Do not
- Do not change c after seeing a spread or re-commission number; the only permitted change is the coherence ladder
  to c = 2, recorded.
- Do not report a spread change without Q3 and the coherence numbers in the same table.
- Do not read an induced projection as a state change; the injected component sits beside every readout.
- Do not call this confirmatory. Vocabulary per STAGE0 §2; "the researcher".
