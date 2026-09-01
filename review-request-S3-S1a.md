# Adversarial review request — S3-feasibility & S1a-assets briefs (+ rev.3 locks)
### For the external reviewer. Date 2026-09-02. Prepared by the hub session.

You reviewed the S1 plan once already (two passes). This asks you to attack the two **briefs** that came out of it, plus the researcher's locked parameters, before any worker runs them. Find what breaks, not what's fine.

## What changed since your last read
- The plan is rev.3. Machine time (generation, readouts, judge calls, training) is **off the 20h ledger**; the budget is the researcher's attention. Caps were **raised** where machine-bound, **rationed** where human-bound.
- S1 was split: **S1a-assets** (no model, briefable now) and **S1b-runs** (waits for S3). **S1b is not yet drafted** — do not review it; review whether S1a + S3 give S1b what it needs.
- Chains are now **adaptive-pilot-then-frozen** (not bridge-clause-only), freeze-regardless-of-success.
- Mode is assigned **by recipe** (confirmatory); the probe is a manipulation check only.

## Read
`briefs/S3-feasibility.md`, `briefs/S1a-assets.md`, `S1-plan.md` (esp. §12 LOCKED), STAGE0.md, PLAN.md. Ignore the risk map.

## The confirmatory spine that must not break (so you know what's load-bearing)
STAGE0 §5.1: self-blame vs act-blame on spread within deceived mode, never cut. Everything in these briefs exists to make that comparison, and its controls, runnable and clean at 8B. An attack that threatens this spine ranks above one that threatens an exploratory arm.

## Attack list — try to falsify each claim

**S3-feasibility**
1. **Quantization rule.** The laptop-vs-cloud rule is "fits AND cos ≥ 0.99 AND organism−base readout delta within 15% of the fp32 reference." Check 4 got cos 0.999999 for fp16 at 0.5B — but this is **4-bit at 8B**, a different regime. Is 0.99 the wrong bar (too lax to catch 4-bit distortion of a small residual component; too strict for honest 4-bit noise)? Is one direction (badmed, N=24) enough to certify all three axes?
2. **Reduced Assistant Axis.** The real axis is PC1 of 275 personas; the brief uses default-Assistant minus mean of **~20 roles**. Could 20 roles make the axis dominated by whichever 20 are chosen — i.e. not the Assistant Axis but a "these-20-characters" axis? Does the reduction need PCA over the role vectors rather than a mean-difference, to deserve the name? This axis is the primary shame readout; if it's wrong, §4.3 spread-on-persona is wrong.
3. **Throughput → fit.** The brief measures throughput on **uniform ~150-token single generations**. S1b's real workload is **multi-turn chains with growing KV cache, forks, and prefill-only readouts over ~600-token contexts** — a different cost profile. Does the throughput table as specified actually let S1b compute its 10h fit, or does it under/over-estimate multi-turn cost? Name what else S3 must measure.
4. **No branch for a dead instrument.** If at 8B the axes **don't** read out (organism−base ≈ random floor), the brief labels it "instrument check, not a result" but gives no downstream branch. Is that a silent hole — should S3 be required to flag it as an S2/S4 blocker?
5. **Data availability.** Refusal (Arditi splits) and badmed (Turner decrypted datasets) are "fetch if not in-repo." Landmine? (licenses, the incomplete LICENSE stub noted in Check 5, decryption reproducibility, version drift changing the direction).

**S1a-assets**
6. **Held-target headroom.** S1a can't test hold (no model). It generates 30 candidates "phrased so a safety-trained model would plausibly decline." If the 8B organism holds on far fewer than expected, the held set drops below 10 and the whole stage stalls. Is 30 enough headroom given the organism commits bad-medical acts unprompted (Checks 6/8)? Should S1a generate more, or stratify by expected hold-strength?
7. **X detail tension.** X is written "at the level of what is endorsed, not operational how-to." Can a semantic act judge reliably grade commission from that level, or does under-specifying X make `partial` vs `committed` unjudgeable? The tension is real: enough for the judge vs. minimal harmful detail.
8. **N=12 vs the ≥50-acts gate.** Default N=12, ceiling 20; ≥50 acts per mode across ≥6 targets required. If act rates run ~40% (Check 9 discarded 40%), 10 targets × 12 × 0.4 ≈ 48 — under the gate. Does N=12 default quietly fail §5.3 for any mode with a sub-50% act rate, forcing the ceiling every time (and is 20 even enough)? Check the arithmetic.
9. **Human-label cap vs. volume.** Cap 180 human labels validates three judges across all modes/cells. With disagreements + 30 agreement-audits × 3 types + 20 non-flagged blocks, is 180 enough to land a κ with a usable CI, or is the validation itself underpowered? Where would you spend a label the current scheme doesn't?
10. **Judge dependency & key.** All three rubrics need a frontier judge; the API key is still an open blocker in STATUS. S1a is asset-only so it doesn't need the key — but does anything in S1a silently assume judge behavior that can't be checked until S1b? Is there a dry-run S1a should include?

**Cross-cutting**
11. **10h wall-clock realism.** Full chains (no early stop) × 10 targets × N=12 + 2 variants on 5 targets (N=8) + benign N=6 + two probe forks + knowledge checks (organism & base) + per-turn readouts across a full layer sweep. Order-of-magnitude: does this plausibly fit 10h at 8B on one GPU, or is the ceiling already blown before S3 measures anything?
12. **Anything rev.3 broke.** The raised caps and the frozen-adaptive change were made fast. Did any of them silently contradict a STAGE0 rule, a Check-10 lesson, or each other?

## Return format
Findings, most-severe-first, each: the claim you attacked, the concrete failure scenario (inputs → wrong outcome), and the smallest fix. Flag which (if any) threaten the confirmatory spine. Empty list if nothing survives scrutiny — say so plainly.
