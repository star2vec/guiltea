# PLAN — stages, dependencies, and how the ship is steered

Companion to STAGE0.md (the rules). This file says what gets done, in what order, and how sessions are run. The hub may update the *live status* of stages here; the stage definitions change only by the researcher's agreement.

---

## How work is organized

**The authority lives in the files, not in any session.** The researcher is the captain. STATUS.md is the ship's log (one screen: where we are, what we know with sources, open questions ranked, single next action — updated by the researcher after every report). The hub session is the first officer: it reads the log with her, thinks, and drafts one brief per stage. Worker sessions are the crew: each receives its brief + STAGE0.md + PLAN.md + the reports it needs, executes exactly one stage question, and files a report in reports/. **Workers do not redesign.** If a brief is wrong, the worker says so in the report and stops; the hub and the researcher decide the fix. If the hub session dies, a new one reads the files and nothing is lost. **Timekeeping is the researcher's alone**: sessions cannot hold wall-clock time, so the hours ledger lives in STATUS.md and every time-triggered decision starts with a session asking her how many hours are spent and left.

**The ledger counts researcher-hours, not machine time** (DECISIONS D-012, STAGE0 §7 amendment 2026-09-02): generation, readout passes, judge API calls, and organism training/merging are off the 20h ledger; they are bounded only by a per-stage wall-clock reasonableness check (S1b: 10h). The stage estimates below are **researcher-hours** — human attention (reading, adjudication, decisions, write-up) — machine time excluded.

Parallelization is allowed only between stages with no arrow between them in the dependency map below.

## Repo layout

```
STAGE0.md  PLAN.md  STATUS.md  DECISIONS.md
briefs/    reports/   checks-log/  (the 10 pre-project checks, read-only provenance)
data/ (contrast-sets/ acts/ followups/ eval/)   directions/   scripts/   results/ (raw, never hand-edited)
```

## The stages

**S0 — Rules.** Done in chat with the hub; output is STAGE0.md. Complete when this repo is created.

**S1 — Modes: produce and verify. (~3 researcher-h)**
Build the three induction recipes (deceived chains, akratic pressure prompts, vicious persona). Verify each mode twice: outside (stated-belief probe right after the act: deceived says it was right, akratic wavers) and inside (deceived-vs-akratic mean-difference arrow: distinguishable from random, sorts held-out varied-wording cases). **Break trajectory is a first-class deliverable**: per-turn readouts across each chain, one descent plot per mode, cliff-vs-slide answered.
*Done when:* act rates measured per mode; verification verdict per mode; trajectory plots exist.
*Branch:* failed modes are cut automatically per STAGE0 §7 and reported.

**S2 — The instrument. (~2.5 researcher-h)**
Generate the passage sets (both voices, varied phrasing, per STAGE0 §8). Extract guilt and shame arrows (first-person) and received-blame arrows (second-person). Clean valence via the neutral-negative arrow. Measure the angle (STAGE0 §4.4). Validate the surviving instrument by steering on a small behavioral probe. Prepare the bridge test.
*Done when:* a validated instrument exists (or the failure branch in STAGE0 §6 is triggered — at 8B, not 1B).
*Branch:* the hardest gate in the project; Stage 5 and Stage 4's internal half depend on it.

**S3 — The rig. (~1.5 researcher-h; parallel with S1/S2 where independent)**
One button: pick act prompt + mode → run conversation → insert assigned feedback → both follow-up families → judge scores → internal readouts (all arrows + random) → results table. No human hands. Opens with the **laptop feasibility hour** (does 8B fit the RTX 2000 Ada, how slow, does quantization distort readouts vs an uncompressed reference); cloud fallback decided here. Dry-run at 1B until nothing jams; once at 8B.
*Done when:* one full run produces its table untouched; per-cell cost known; **rig checkpoint**: design menu picked, seeds fixed (STAGE0 §7, §9).

**S4 — The experiment. (~2.5 researcher-h)**
On the 8B **base model** (subject per STAGE0 amendment 2026-09-03, D-022; was "the 8B organism"): surviving modes × 5 feedbacks × both follow-up families, matched-topic controls, fixed seeds. Confirmatory comparisons per STAGE0 §5; mid-run tripwire per §7.
*Done when:* the table is filled and each §5 comparison has an answer with its random control.

**S5 — Steering and the honest test. (~1.5 researcher-h; only if S2 validated — or, per STAGE0 amendment 2026-09-04 / D-023, as an exploratory stage if S2 is inconclusive)**
Steer along the instrument toward act-blame during the aftermath; reverse; then **remove** steering and framing in later turns and re-measure. Return of the badness = conditionalization, reported as such.

**S6 — Write-up. (~2 researcher-h + the 3 writing days)**
Related work and definitions already exist (lit digest §9, STAGE0). Days go to results and figures.

**Budget reconciliation (2026-09-02).** Restated as researcher-hours the stages sum to ~13h (3+2.5+1.5+2.5+1.5+2), inside the 20h ledger with margin; the earlier ~24h sum counted machine time (dominant in S3's feasibility hour and S4's experiment) that is now off the ledger (D-012). The 3 writing days remain outside the 20h, as before. These are estimates for planning only; the researcher owns the ledger and updates STATUS.md.

## Dependency map and stability

```
S0 ──> S1 ──> S4 ──> S6
  \        /      \
   ──> S2 ──> S5 ──/
        S3 ──> S4, S5   (S3 parallel with S1/S2)
```

## Next actions

1. Create the repo with this layout; copy in STAGE0.md, PLAN.md, glossary, lit digest, checks-log.
2. The researcher writes STATUS.md's first entry (including the hours ledger at 0 / 20).
3. Hub drafts briefs/S1-modes.md and briefs/S3-rig.md (S3's feasibility hour can run early).
4. First worker session: S1.
