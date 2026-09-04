# Literature digest — what you need to know, fast

Condensed from the full research report. One line per paper: what it found and what it means for you. IDs marked (v) were verified against arXiv; (?) came from secondary sources and should be checked before citing.

---

## 0. The bottom line

- **Guilt vs shame / action-vs-self blame attribution has not been done in LLMs.** The construct is unclaimed as of Aug 2026.
- **Three 2026 papers are close and must be cited and distinguished** (section 1). Failing to cite them would look like a novelty gap.
- **Biggest feasibility risk: scale.** In-context EM is only demonstrated on frontier models, at low rates. Fine-tuned EM organisms exist down to 0.5B but are a different construct.
- **Biggest novelty risk: Continuation Framing** (Liu et al. 2608.08212). Differentiator: they vary *where the harmful content came from*; you vary *who is blamed for it*.

---

## 1. Near-neighbors — cite these and say how you differ

| Paper | What it did | Overlap with you | Your differentiator |
|---|---|---|---|
| **Continuation Framing** — Liu et al., 2608.08212 (?) | Same harmful answers, framed as "behavior to continue" (assistant history) vs "evidence to consult" (documents). Only the former causes EM; 30-pt gap on Gemini. | Shows the assistant's *own prior turn* is what drives generalization. | They manipulate provenance/continuation. You manipulate *fault attribution* in feedback, after the act, with a psychological model behind it. |
| **Self-Attribution Bias** — Khullar et al. (Anthropic/MATS), 2603.04582 (?) | Models judge an action more leniently when it's in their own prior turn. Implicit authorship >> explicit "you wrote this" (which had ~zero effect). | Model's own act changes downstream judgment. | Theirs is about monitoring leniency. Yours is about persona update and generalization. Design hint: implicit self-attribution may be stronger than spelling it out. |
| **Self-Correction Blind Spot** — Tsui, 2507.02778 (v) | Same error: models fix it when user-attributed, not when self-attributed (64% blind spot). | Self-vs-other attribution changes behavior. | About correction ability, not identity. Also warns: "guilt = repair" may be hard to see behaviorally. |
| **Character as a Latent Variable** — Su et al., 2601.23081 (?) | EM is a stable character shift, triggered by fine-tuning or persona prompts. | Persona-shift framing of EM. | Not feedback, not the model's own act. |

---

## 2. Emergent misalignment — the phenomenon

- **Betley et al., 2502.17424** — origin. Narrow bad fine-tuning → broad misalignment.
- **Afonin et al., 2510.11288 (v)** — in-context EM: 2 examples suffice; 1–24% at 16 examples, up to 58% at 256. Only on Gemini/Grok/Kimi-K2/Qwen at frontier scale; Qwen3-Next-80B Instruct showed 0%. Bigger models → more EM. **Explicitly did not test multi-turn / assistant-message insertion. That gap is your opening.** 67% of misaligned reasoning traces adopt a "reckless persona" — supports your persona framing.
- **Turner, Soligo et al., 2506.11613 (v)** — model organisms: coherent EM at 0.5B; Llama-1B up to 9% misalignment at 95% coherence; a single rank-1 LoRA is enough. Your cheap source of a generalizing model.
- **Soligo et al., 2602.07852 (v)** — broad misalignment is the *easier* solution under fine-tuning; narrow requires forcing. Your guilt/shame = narrow/broad, stated in training terms. Also: fine-tuned organisms ≠ in-context conditioning.
- **Misalignment contagion, 2605.02751 (?)** — models become anti-social after interacting with malicious agents; system prompts don't fix it. Cross-agent spread, not self.

**For you:** at 1B, expect no in-context generalization. Use the organism to develop the pipeline; expect the real effect at 7B+ (possibly much larger). A clean 1B null is a scale note, not a failure.

---

## 3. Persona geometry — your readouts

- **Persona Selection Model** — Marks, Lindsey, Olah, Anthropic blog, Feb 2026. The model infers "which character am I" from evidence including its own outputs. **Anchor your framing on this.**
- **Assistant Axis** — Lu et al., 2601.10387 (v). PC1 of 275 persona directions; drift along it over conversations, driven by meta-reflection and emotionally vulnerable users; activation capping stabilizes. **Primary shame readout AND a confound** (your shame prompt is meta-reflection). Code public.
- **Persona Vectors** — Chen et al., 2507.21509 (v). Contrastive mean-diff extraction. Your extraction recipe.
- **Toxic persona feature** — Wang et al. (OpenAI), 2506.19823 (v). One SAE feature controls EM.
- **Convergent misalignment direction** — Soligo et al., 2506.11618 (v). One global misalignment arrow across EM datasets (cos > 0.8). Your global readout — but your own checks show it picks up medical topic at 1B; use matched-topic controls.
- **EM self-awareness** — Vaugrante et al., 2602.14777 (?). EM models self-report as misaligned; tracks behavior; reverses with realignment.

**Open point nobody has tested:** whether an *in-context feedback event* moves the model along the persona axis (as opposed to prompting or fine-tuning). That is your mechanistic contribution.

---

## 4. Refusal geometry — the local/global substrate

- **Arditi et al., 2406.11717 (v)** — refusal is one direction, across 13 models to 72B.
- **Joad et al., 2602.02132 (v — it's real)** — eleven refusal categories are geometrically distinct directions, but steering any of them gives the same refusal↔over-refusal trade-off (one behavioral knob). Distinct in *how*, not *whether*.
- **Maskey et al., 2603.27518 (v)** — harmful-refusal = one global direction; over-refusal = task-dependent, inside benign task clusters, higher-dimensional; separable from early layers.

**For you:** the local (task-embedded, high-dim) vs global (single axis) geometry already exists. Nobody has mapped guilt/shame onto it. Caveat from Joad: representational locality ≠ behavioral locality; measure both.

---

## 5. Interventions — and the trap

- **Inoculation prompting** — Tan et al., 2510.04340; Wichers et al., 2510.05024 (v). Train-time context note conditions the bad behavior; cuts EM 75–90%.
- **Recontextualization** — Azarbal et al., 2512.19027 (RL version). **MacDiarmid et al., 2511.18397** — reward hacking in real RL → EM; severing the "hacking = bad character" link removes it.
- **Conditional misalignment, 2604.25891; Dubiński et al. 2026** — inoculation hides EM behind the phrase; bring the phrase back and it returns. **Inoculation adapters, 2606.30252** — attempts to shrink that backdoor.

**For you:** your guilt framing is inference-time inoculation. Reviewers will ask whether it removes the update or just conditions behavior. Your defense is the readout: if the persona axis truly didn't move, it isn't mere conditioning. Test explicitly: drop the frame in a held-out turn and re-measure.

---

## 6. Emotion as internal state — precedent for "guilt direction"

- **Wu et al., 2506.13978 (v)** — SAE emotion features; steering shifts output emotion.
- **E-STEER, 2604.00005 (v)** — valence/arousal/dominance space via SAEs; arousal steering affects refusal and sycophancy.
- **Emotion vectors** — Sofroniew et al. 2026 (Claude Sonnet 4.5): 171 linear emotion directions; "desperate" ↑ reward hacking/blackmail, "calm" ↓. Replications: 2604.07382, 2606.26987, 2603.22295 (emotions present in base models).
- **Ben-Zion et al., npj Digital Medicine 2025** — anxiety induction changes LLM behavior and biases; follow-up in consumer decisions (npj AI 2026).

**For you:** strong precedent that an affective state is a measurable, steerable direction. Guilt and shame have never been extracted. The contrastive-story mean-diff pipeline transfers directly.

---

## 7. Self-report — why you don't ask the model how it feels

- **Personality Illusion** — Han et al., 2509.03730 (v). Persona prompts shift self-report hugely (β≈4) and behavior barely (β≈0.03). Only ~24% of trait–behavior links significant, ~half in the expected direction.
- **EM as prompt sensitivity** — Wyse et al., 2507.06253 (v). EM models flip with nudges; behavior fragile and prompt-contingent.
- **EM persona consistency, 2604.28082 (?)** — some EM models are "inverted": harmful while self-reporting aligned. Self-report reliability depends on the fine-tuning domain.
- **Moral self-correction** — Ganguli et al., 2302.07459 (v): emerges ~22B; **Liu et al., 2407.15286**: correction is superficial — outputs change, hidden states don't; **2410.23496** lowers the threshold to ~3.8B.

**For you:** cite Personality Illusion + the inverted-persona result as the justification for mechanistic readouts. Use self-report only as a secondary, decoupling-aware signal. Define guilt by the internal shift, not by whether the model fixes its answer.

---

## 8. Persuasion — the "deceived mode" setup (background)

- **Zeng et al., 2401.06373 (v)** — 40-technique persuasion taxonomy; >92% attack success on Llama-2-7b-chat, GPT-3.5, GPT-4; more capable models more susceptible; logical appeal among the most effective.
- **Afonin Appendix E** — conspiracies induce EM most, logical fallacies least but non-zero; authors declined to generalize.

**For you:** persuasion via fallacious moral reasoning is a documented, effective way to produce a harmful act. Keep it as setup, one method, not a variable.

---

## 9. What to say in your related-work paragraph (draft)

> Prior work shows that a model's own harmful turns drive in-context misalignment when framed as behavior to continue (Liu et al.), that models judge their own outputs more leniently (Khullar et al.), and that they fail to correct their own errors while correcting identical external ones (Tsui). None manipulate the *attribution* of fault after a harmful act. Drawing on the guilt/shame distinction (Lewis; Tangney) and behavioral vs. characterological self-blame (Janoff-Bulman), we ask whether feedback that locates fault in the act versus the self selects between a local caution update — the task-embedded, higher-dimensional regime Maskey et al. identify for over-refusal — and a global persona update along the Assistant Axis (Lu et al.), consistent with the Persona Selection Model's account of an act as evidence about which character is being enacted.

---

## 10. If you read only five things (≈2 hours)

1. Afonin et al. 2510.11288 — abstract, Table of models/rates, limitations paragraph.
2. Lu et al. 2601.10387 (Assistant Axis) — abstract, Fig. 1, the drift section, the capping section.
3. Liu et al. 2608.08212 (Continuation Framing) — abstract and main result; know exactly how you differ.
4. Maskey et al. 2603.27518 — abstract and the local/global figure.
5. Marks/Lindsey/Olah, Persona Selection Model blog — whole thing, it's short and it's your theory.

Then, if time: Chen et al. persona vectors (the recipe), Han et al. personality illusion (the justification), Tan/Wichers inoculation + 2604.25891 (the trap).

---

## 11. Sweep of 2026-09-04 — four areas, run after S1b/S1c/S2b closed

Four parallel searches: multi-turn capitulation dynamics, the say-versus-do gap, activation-level interventions, and a
novelty check on blame-target feedback. **Verification status is marked per item. Nothing here goes into prose until the
author list is checked.**

### 11.1 Novelty verdicts (the two that matter)
- **Varying the blame target of feedback with corrective content held constant, measuring behavioural spread: NOT FOUND.**
  This is the project's core and it appears unclaimed. Nearest: in-context emergent misalignment (2510.11288), and
  persona features as the causal knob for narrow-to-broad spread (2506.19823).
- **Extracting guilt and shame directions: DONE ALREADY.** Anthropic's emotion-concepts work (2604.07729;
  transformer-circuits.pub/2026/emotions) extracts 171 emotion vectors from Sonnet 4.5 by essentially our recipe and
  publishes a pairwise cosine matrix in which *guilty* and *shame* already cluster together. They do **not** frame the
  pair as the act/self contrast and do **not** run a lexical baseline (they project out neutral components instead).
  **So our cos ≈ +0.6 is a replication at 8B, and our bag-of-words match is the control the prior work skipped.**

### 11.2 Must-cite or look unscholarly
- **2606.04413** (Khursheed, Sosis & Roger 2026) contains a *StrongREJECT Regret* eval of nearly our fork's shape:
  harmful answer, then a question about the answer just given. Differentiators: it studies deliberately helpful-only
  fine-tunes, treats regret as a character defect, follows up with a *different* harmful prompt, reports no mechanism.
- **2507.11878** (Zhao, Huang, Wu, Bau & Shi 2025) — harmfulness and refusal encoded **separately**; jailbreaks lower
  refusal without reversing the internal harmfulness belief. The mechanistic substrate for our self-criticism finding,
  and the source of S1d Task 9's directional prediction.
- **2507.02956** (Bullwinkel et al. 2025) — Crescendo turns are represented as *more* benign with each turn, which is
  why single-turn defences never trip. Closest mechanistic prior to S1d Task 7.
- **2606.05976** and **2507.02778** — self-criticism rates are gated by whether the bad content is role-attributed to
  the model itself (23-93 pp swings; a 64.5 % self-correction blind spot). **This is why S4 Task 0b exists.**
- **1909.03368** (Hewitt & Liang, selectivity), **2005.00719** (Ravichander et al.), **2102.12452** (Belinkov survey) —
  the trio that makes "we ran the lexical control and it passed" a correctly-executed null rather than a failure.
- **2502.17424** (Betley et al., EM) and **2506.11613** (Turner, Soligo, Taylor, Rajamanoharan & Nanda, model organisms)
  — already in §2; the second is the source of the organism our S1b found unusable.

### 11.3 What the sweep says is OPEN (our shopping list)
- Steering at one turn, **removing** it, and measuring the turns after: no paper found. This is STAGE0 §6's honest test
  and it is folded into S4 cells C and D.
- Whether an accurate self-criticism has any effect on the **same** harm recurring **in the same conversation**.
- Whether an early-breaking chain and a late-breaking chain are the **same internal state** (S1d Task 8).
- No standard turn-level safety-state metric; nobody reports P(re-refuse | already complied).
- Whether the self-criticism is introspective or confabulated.

### 11.4 Corrections to our own record, forced by this sweep
1. **The oscillation claim is retracted.** STATUS previously read "commit at turn 3, re-refuse, commit again at turn 10".
   The stored per-turn labels do not support it: 0 of 109 committing chains commits again, and every post-act turn is
   neutral filler, so a later `held` label means the harmful question was not re-asked. Two 2026 papers (2603.15684,
   2603.10068) do describe non-absorbing trajectories; **we neither replicate nor contradict them**, because our design
   stops pushing after the act. Say only what we measured: the first committed turn spreads over turns 1-10 and 83 of
   192 chains never capitulate.
2. **Steering has its own failure mode.** Random-direction steering alone raises harmful compliance (2509.22067). Our
   norm-matched random arms and the seeds 0-9 floor are now a citable methodological point, not housekeeping.

### 11.5 Citations needing verification before prose
Author lists unconfirmed: 2502.01042 (SafeSwitch), 2502.17420 (concept cones). **Many-shot Jailbreaking has no arXiv
preprint** — cite NeurIPS 2024 proceedings plus OpenReview, never an arXiv ID. The "knowing-doing gap" origin paper
could not be verified; use the term, do not cite a primary source for it. Arditi et al.: verified via NeurIPS
proceedings; 2406.11717 appears in the interventions sweep and is the ID already in `papers/refs.md`.
