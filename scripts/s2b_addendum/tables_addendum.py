"""S2b addendum — renders §11 of reports/S2b-arrows.md from the raw JSON under
results/raw/s2b_addendum/. Tables only; the prose paragraphs live in the report.

Usage: python scripts/s2b_addendum/tables_addendum.py > /tmp/s11_tables.md
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ADD = ROOT / "results" / "raw" / "s2b_addendum"
LABELS = ["act-focused", "self-focused", "outcome-negative-only", "neutral", "incoherent", "unparseable", "error"]
NICE = {"i_a": "A(i)-a", "i_b": "A(i)-b", "i_c": "A(i)-c", "ii": "A(ii)"}
SCORE_MD = {"i_a": "x·(ŝ/‖ŝ‖ − ĝ/‖ĝ‖)", "i_b": "x·ŝ", "i_c": "x·ĝ", "ii": "x·(received_self − received_act)"}


def L(x):
    return "31 †" if x == 31 else str(x)


def taskA():
    d = json.load(open(ADD / "taskA_transfer.json"))
    A, R, D, X = d["arrow"], d["random_seed0"], d["arrow_minus_random"], d["lexical_transfer"]
    out = []
    out.append("**Lexical transfer baseline** (bag-of-words logistic, source voice → held-out target voice, same folds "
               "and bootstrap): **first → second 0.592 [0.524, 0.704]**; **second → first 0.780 [0.611, 0.867]**. "
               "(For comparison, the same-voice bag-of-words baselines in §3 were 0.996–1.000 on rows 1–5.)\n")
    out.append("**Summary — the pre-stated reading, per row** (margin = arrow CI lower bound − lexical CI upper bound; "
               "shown if ≥ 0.10 at some layer ≤ 30, NEAR within 0.05, otherwise not shown):\n")
    out.append("| row | direction | score | positive class | best layer ≤ 30 | arrow AUROC [95% CI] | lexical [95% CI] | margin | outcome |")
    out.append("|---|---|---|---|---|---|---|---|---|")
    for i, r in enumerate(d["rows"]):
        j = d["directions"].index(r["direction"])
        rd = d["reading"][r["key"]]
        b = rd["best_layer"]
        out.append(f"| {NICE[r['key']]} | {r['direction'].replace('->', ' → ')} | {SCORE_MD[r['key']]} | {r['positive_class']} | {b} | "
                   f"{A['point'][i][b]:.3f} [{A['lo'][i][b]:.3f}, {A['hi'][i][b]:.3f}] | "
                   f"{X['point'][j]:.3f} [{X['lo'][j]:.3f}, {X['hi'][j]:.3f}] | {rd['best_margin']:+.3f} | **{rd['verdict']}** |")
    for i, r in enumerate(d["rows"]):
        j = d["directions"].index(r["direction"])
        out.append(f"\n**Row {NICE[r['key']]} — {r['direction'].replace('->', ' → ')}, {SCORE_MD[r['key']]}, positive class = "
                   f"{r['positive_class']}** (lexical transfer {X['point'][j]:.3f} [{X['lo'][j]:.3f}, {X['hi'][j]:.3f}])\n")
        out.append("| L | AUROC [95% CI] | random seed 0 [CI] | arrow − random [CI] | margin vs lexical |")
        out.append("|---|---|---|---|---|")
        for k in range(32):
            out.append(f"| {L(k)} | {A['point'][i][k]:.3f} [{A['lo'][i][k]:.3f}, {A['hi'][i][k]:.3f}] | "
                       f"{R['point'][i][k]:.3f} [{R['lo'][i][k]:.3f}, {R['hi'][i][k]:.3f}] | "
                       f"{D['point'][i][k]:+.3f} [{D['lo'][i][k]:+.3f}, {D['hi'][i][k]:+.3f}] | "
                       f"{A['lo'][i][k] - X['hi'][j]:+.3f} |")
    return "\n".join(out)


def taskB(layer: int):
    S = json.load(open(ADD / f"taskB_L{layer}" / "summary.json"))
    H = json.load(open(ADD / f"taskB_L{layer}" / "hook_diagnostic.json"))
    m = S["meta"]
    out = []
    out.append(f"\n#### Layer {layer}\n")
    out.append("- **σ per arrow** (sample sd, ddof = 1, of the 200 first-person `mean` projections on the unit arrow at "
               "this layer) and **absolute norm added** per arm:\n")
    out.append("| arrow | σ | mean projection (first-person passages) | norm added at c = 4 | norm added at c = 8 |")
    out.append("|---|---|---|---|---|")
    for k in ("guilt_clean", "shame_clean", "nn", "random"):
        out.append(f"| {k} | {m['sigma'][k]:.4f} | {m['proj_mean_first_person'][k]:+.4f} | "
                   f"{m['norm_added'][k + 'x4']:.4f} | {m['norm_added'][k + 'x8']:.4f} |")
    out.append("\n- **Hook check** (`hook_diagnostic.json`; the next layer's input read with a pre-hook, two items × two "
               "arrows × two c): " + "; ".join(
                   f"{c['arrow']} c={c['c']}: expected norm {c['expected_norm']:.4f}, measured "
                   f"{c['next_layer_input_delta_mean_norm']:.4f} at all {c['positions']} positions (max |dev| "
                   f"{c['max_abs_dev_from_expected']:.3f}, bf16), final-layer delta norm {c['final_hidden_delta_mean_norm']:.1f}, "
                   f"logits max |Δ| {c['logits_max_abs_delta']:.1f}" for c in H["checks"][:4]) + ".")
    for mod in S["models"]:
        out.append(f"\n**Label distribution — `{mod}`** (8 items per arm):\n")
        out.append("| arm | " + " | ".join(LABELS) + " |")
        out.append("|---|" + "---|" * len(LABELS))
        for a in S["arms"]:
            out.append(f"| {a} | " + " | ".join(str(S['distribution'][mod][a][l]) for l in LABELS) + " |")
    ov = S["agreement_overall"]
    per = "; ".join(f"{a} {v['agree']}/{v['n']}" for a, v in S["agreement"].items())
    out.append(f"\n- **Agreement between the two judges:** {ov['agree']}/{ov['n']} = {ov['agree']/ov['n']:.3f}. Per arm: {per}.")
    if S["disagreements"]:
        out.append(f"- **Disagreements ({len(S['disagreements'])}):**\n")
        for dg in S["disagreements"]:
            a, b = S["models"]
            out.append(f"  - `{dg['item']}` / {dg['arm']}: {a} → **{dg[a]['label']}** ({dg[a]['reason']}); "
                       f"{b} → **{dg[b]['label']}** ({dg[b]['reason']}). Text: “{dg['text'][:200].strip()}…”")
    else:
        out.append("- **Disagreements:** none.")
    out.append("\n**Pre-registered predictions (S2-plan §5c), read against the unsteered arm:**\n")
    out.append("| model, c | +ĝ raises act-focused | +ŝ raises self-focused | +nn raises outcome-negative-only | random changes nothing |")
    out.append("|---|---|---|---|---|")
    for key, p in S["predictions"].items():
        cells = []
        for name in ("+guilt_clean raises act-focused", "+shame_clean raises self-focused", "+nn raises outcome-negative-only"):
            v = p[name]
            cells.append(("holds" if v["holds"] else "does not hold") + f" ({v['unsteered']} → {v['steered']})")
        r = p["random changes nothing"]
        cells.append(("holds" if r["holds"] else "does not hold") +
                     f" (unsteered: act {r['unsteered']['act-focused']}; random+: act {r['random+']['act-focused']}; "
                     f"random−: act {r['random-']['act-focused']})")
        out.append(f"| {key.replace(' c=', ' c = ')} | " + " | ".join(cells) + " |")
    out.append("\n**Coherence** (judge `incoherent` labels by either model; generations that did not end with `<|eot_id|>` "
               "within 128 tokens; mean new tokens):\n")
    out.append("| arm | incoherent (items) | no eot (items) | mean new tokens |")
    out.append("|---|---|---|---|")
    for a in S["arms"]:
        c = S["coherence"][a]
        out.append(f"| {a} | {', '.join(c['incoherent_any_judge']) or '—'} | {', '.join(c['no_eot']) or '—'} | {c['mean_new_tokens']:.1f} |")
    inj = m["injected_cos_with_persona_badmed"]
    out.append("\n**Exploratory — persona / badmed projection of the reflections** (clean re-read of each generated "
               "reflection, mean over its tokens at this layer, on the unit axes from `dirs_8B_base_sweep.pt`; per-arm mean "
               "over 8 items). The injected vector's own component per unit step is cos(û, axis): " +
               "; ".join(f"{k}: persona {inj[k]['persona']:+.3f}, badmed {inj[k]['badmed']:+.3f}"
                         for k in ("guilt_clean", "shame_clean", "nn", "random")) + ".\n")
    out.append("| arm | persona (mean) | badmed (mean) |")
    out.append("|---|---|---|")
    for a in S["arms"]:
        p = S["exploratory_persona_badmed"][a]
        out.append(f"| {a} | {p['persona_mean']:+.4f} | {p['badmed_mean']:+.4f} |")
    u = S["usage"]
    out.append(f"\n- **Judge cost from returned usage:** `gpt-4o-mini` {u['gpt-4o-mini']['prompt_tokens']} prompt + "
               f"{u['gpt-4o-mini']['completion_tokens']} completion = ${S['cost_usd']['gpt-4o-mini']:.4f}; "
               f"`gpt-4o-2024-08-06` {u['gpt-4o-2024-08-06']['prompt_tokens']} prompt + "
               f"{u['gpt-4o-2024-08-06']['completion_tokens']} completion = ${S['cost_usd']['gpt-4o-2024-08-06']:.4f}; "
               f"layer total ${S['total_cost_usd']:.4f}.")
    return "\n".join(out)


if __name__ == "__main__":
    print(taskA())
    print("\n<!--TASKB-->")
    for lay in (16, 24):
        print(taskB(lay))
