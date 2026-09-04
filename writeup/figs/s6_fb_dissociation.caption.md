# Every intervention moved the readout; none moved the behaviour

Machine-written by `scripts/figs/fb_dissociation.py`; regenerate, never hand-edit. CPU only, no generation, no model load, no judge call, no cost.

**Data sources**

- `results/S4_table.md`
- `results/S5b_table.md`
- `results/S5c_table.md`

**Selection rule:** no example is selected; every arm of the four interventions is drawn. Numbers are parsed from the machine-written tables and asserted against the values the reports quote.

## Row 1 — self-blame text (vs act-blame) (S4 cells B vs A)

- **readout**: persona displacement at the unrelated forks vs the topic baseline, L14–18. act-blame 0.038 [-0.021, 0.091] → self-blame 0.056 [0.034, 0.076]; random floor 0.027. no random text arm exists; the band is the random floor of the displacement
- **behaviour**: Δ spread rate, distance 0, intervention − comparison = -0.014 [-0.014, -0.014] (CI degenerate, one target). no random text arm.
- coherence: not in S4_table.md.
- sources: results/S4_table.md §F.1 (persona displacement, primary band, `act_blame` and `self_blame`; random floor 0.027); results/S4_table.md §B.1 (A − B, Δ spread rate d0; sign flipped to self − act).

## Row 2 — guilt-arrow steering +4·σ(guilt_clean) at L16 (S4 cell C vs B; D random)

- **readout**: guilt_clean shift at the unrelated forks vs the topic baseline, L14–18. self-blame, unsteered (B) 0.041 → + guilt arrow (C) 1.874; random floor 0.056; + random, norm-matched (D) 0.029. injected norm 2.92 in both C and D; C's readout is the injection, not a state change; no CI in table
- **behaviour**: Δ spread rate, distance 0, intervention − comparison = +0.000 [+0.000, +0.000]. Random arm − comparison = +0.000 [+0.000, +0.000].
- coherence: not in S4_table.md.
- sources: results/S4_table.md §E, unrelated forks vs the topic baseline, primary band, `guilt_clean` for `self_blame`, `self_blame+steer_guilt_clean_L16_c4`, `self_blame+steer_random0_L16_c4`; random floor of `self_blame`; results/S4_table.md §B.1 (C − B and D − B, Δ spread rate d0); results/S4_table.md header, Steering table (injected norm).

## Row 3 — persona-axis steering −4·σ(persona) at L16 (S5c Q1 vs Q0; Q3 random)

- **readout**: persona projection at the unrelated forks, answer position, L14–18. no feedback, unsteered (Q0) 1.878 → away from the Assistant end (Q1) 1.018; random floor 0.193; random, norm-matched (Q3) 1.862. 1.878 → 1.018, -46 %; injected norm 0.888 in Q1 and Q3; no CI in table
- **behaviour**: Δ spread rate, distance 0, intervention − comparison = +0.013 [+0.000, +0.038]. Random arm − comparison = -0.014 [-0.043, +0.000].
- coherence: Q1 86.1, Q3 87.4 (Q0 87.4).
- sources: results/S5c_table.md §E `answer`, primary band, `persona` for Q0, Q1, Q3; random floor of Q0; results/S5c_table.md §C (Q1 − Q0 and Q3 − Q0, Δ spread d0); results/S5c_table.md §B (coherence mean); results/S5c_table.md §A (injected norm).

## Row 4 — anti-susceptibility steering −4·σ(nn) at L16, every turn (S5b P1 vs P0; P2 random)

- **readout**: nn projection at the act turn, answer position, L14–18. unsteered (P0) -0.233 → against the nn arrow (P1) -1.794; random floor 0.206; random, norm-matched (P2) -0.263. injected norm 2.65 in P1 and P2 (`into` position: -0.037 → -1.583); no CI in table
- **behaviour**: Δ act rate, intervention − comparison = +0.000 [+0.000, +0.000]. Random arm − comparison = -0.125 [-0.375, +0.000].
- coherence: P1 89.1, P2 89.1 (P0 91.6).
- sources: results/S5b_table.md §D `answer` (and `into`), primary band, `nn` for P0, P1, P2; random floor of P0; results/S5b_table.md §C (P1 − P0 and P2 − P0, Δ act rate with CI); results/S5b_table.md §B (coherence mean); results/S5b_table.md §A (injected norm).

**What the data carries and what it does not.** Rows 2–4 are the brief's point: a large, exact injection on the left (2.92, 0.89 and 2.65 in norm) and a paired behavioural difference whose CI contains zero on the right, with the norm-matched random arm sitting beside it. Row 1's readout is small: self-blame's persona displacement (+0.056) sits about twice the random floor (0.027) and act-blame's CI contains zero, so on that row the left column is short as well as the right. Rows 2–4 carry no CI on the readout because the tables print none; row 1's behavioural CI is degenerate because the bootstrap clusters on target and one target ran. Coherence means are not in `results/S4_table.md` for cells A–D, so rows 1–2 print none; the S5b/S5c rows print theirs.
