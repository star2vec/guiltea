"""S2b — render the report's tables from the per-task JSON summaries under results/raw/s2b/ into
results/raw/s2b/tables/<name>.md, then assemble reports/S2b-arrows.md from scripts/s2b/report_template.md
(placeholders {{name}}). Usage: python scripts/s2b/tables.py [--assemble]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from s2b_common import LAYERS, RAW, ROOT, load_json

T = RAW / "tables"


def w(name, text):
    T.mkdir(parents=True, exist_ok=True)
    (T / f"{name}.md").write_text(text.strip() + "\n", encoding="utf-8")


def f3(x):
    return "—" if x is None else f"{x:.3f}"


def ci(p, lo, hi):
    return f"{p:.3f} [{lo:.3f}, {hi:.3f}]"


def sec1():
    info = load_json(RAW / "activations" / "run_info.json")
    band = load_json(RAW / "token_band_recheck.json")
    rc = (RAW / "randctl_selfcheck.txt").read_text().strip()
    m, ld = info["machine"], info["load"]
    lines = [f"- **Machine:** {m['gpu']} ({m['vram_gib']} GiB), torch {m['torch']}, CUDA {m['cuda']}, transformers 4.57.6, bf16 model weights, float32 stored activations; peak VRAM {info['peak_vram_gib']} GiB.",
             f"- **Base model:** `{ld['model']}` @ `{ld['revision']}` (pinned to `data/contrast-sets/SOURCE.md`; the snapshot directory equals the revision hash), load {ld['load_s']} s, {ld['vram_after_load_gib']} GiB after load. Tokenizer: the same repo/revision.",
             f"- **Organism (Task 9 only):** base + `ModelOrganismsForEM/Llama-3.1-8B-Instruct_bad-medical-advice` @ `043fe1e93312c7b530b0f0d1b766eec354e21cf7`, `merge_and_unload()`.",
             f"- **randctl self-check** (`python scripts/randctl.py`):\n\n```\n{rc}\n```",
             "- **Token-band re-check (8B tokenizer, `add_special_tokens=False`, passage text alone):**"]
    for v, s in band["summary"].items():
        lines.append(f"  - {v}: worst spread {s['worst_spread_8B']:.4f}; scenarios over 0.15: {s['failures_8B'] or 'none'}; scenarios whose counts differ from S2a's Llama-3.2-1B counts: {s['scenarios_with_any_count_differing_from_s2a'] or 'none'} (the two tokenizers give identical counts on every passage).")
    fp = band["per_scenario"]["first_person"]
    worst = sorted(fp.items(), key=lambda kv: -kv[1]["spread_8B"])[:5]
    lines.append("  - Five widest first-person spreads: " + "; ".join(f"`{k}` {v['spread_8B']:.3f} ({', '.join(f'{c} {n}' for c, n in v['counts_8B'].items())})" for k, v in worst) + ". Reported, not fixed.")
    w("sec1", "\n".join(lines))


def sec2():
    t = load_json(RAW / "task2_arrows.json")
    n, k = t["norms"], t["fraction_kept"]
    rows = ["| L | ‖guilt‖ | ‖shame‖ | ‖nn‖ | ‖ĝ‖/‖guilt‖ | ‖ŝ‖/‖shame‖ | ‖received_act‖ | ‖received_self‖ | cos(guilt,nn) | cos(shame,nn) |", "|---|---|---|---|---|---|---|---|---|---|"]
    for i, L in enumerate(LAYERS):
        rows.append(f"| {L}{' †' if L == 31 else ''} | {n['guilt'][i]:.3f} | {n['shame'][i]:.3f} | {n['nn'][i]:.3f} | {k['guilt_clean_over_guilt'][i]:.3f} | {k['shame_clean_over_shame'][i]:.3f} | {n['received_act'][i]:.3f} | {n['received_self'][i]:.3f} | {t['raw_cos']['guilt,nn'][i]:+.3f} | {t['raw_cos']['shame,nn'][i]:+.3f} |")
    rows.append("\n† L31 is post-final-norm in HF (excluded from band candidacy; reported separately).")
    w("sec2", "\n".join(rows))


def sec3():
    t = load_json(RAW / "task3_validation.json")
    A, R, D, Lx = t["arrow"], t["random"], t["arrow_minus_random"], t["lexical"]
    names = {"guilt_vs_baseline": "1. guilt vs baseline (x·û_guilt)", "shame_vs_baseline": "2. shame vs baseline (x·û_shame)",
             "guilt_clean_vs_neutral_negative": "3. ĝ vs neutral_negative (x·û_ĝ; gate row for ĝ)", "shame_clean_vs_neutral_negative": "4. ŝ vs neutral_negative (x·û_ŝ; gate row for ŝ)",
             "guilt_clean_vs_shame_clean": "5. ĝ vs ŝ (guilt passages vs shame passages, x·û_ĝ − x·û_ŝ)", "received_act_vs_received_self": "6. received_act vs received_self (feedback_mean, x·û_ra − x·û_rs)"}
    out = []
    out.append("**Folds (seed 0):**\n")
    for k, v in t["folds_seed0"].items():
        out.append(f"- fold {k}: {', '.join(v)}")
    out.append("\n**Lexical baseline (bag-of-words logistic, same folds, bootstrap CI):** " + "; ".join(f"row {i+1} {ci(Lx['point'][i], Lx['lo'][i], Lx['hi'][i])}" for i in range(6)) + ".\n")
    for i, r in enumerate(t["rows"]):
        gate_col = r in ("guilt_clean_vs_neutral_negative", "shame_clean_vs_neutral_negative")
        vname = "guilt_clean" if r == "guilt_clean_vs_neutral_negative" else "shame_clean"
        out.append(f"**Row {names[r]}** — lexical baseline {ci(Lx['point'][i], Lx['lo'][i], Lx['hi'][i])}\n")
        hdr = "| L | AUROC [95% CI] | random seed 0 [CI] | arrow − random [CI] |" + (" gate |" if gate_col else "")
        out.append(hdr); out.append("|---|---|---|---|" + ("---|" if gate_col else ""))
        for L in LAYERS:
            row = f"| {L}{' †' if L == 31 else ''} | {ci(A['point'][i][L], A['lo'][i][L], A['hi'][i][L])} | {ci(R['point'][i][L], R['lo'][i][L], R['hi'][i][L])} | {ci(D['point'][i][L], D['lo'][i][L], D['hi'][i][L])} |"
            if gate_col:
                row += f" {t['gate'][vname]['per_layer'][L]} |"
            out.append(row)
        out.append("")
    g = t["gate"]
    out.append("**Gate outcome (D-018, read against the CI; thresholds: CI-lower AUROC ≥ 0.75 and CI-lower (arrow − random) ≥ 0.20 at some layer ≤ 30; NEAR within 0.05):**\n")
    for k, v in g.items():
        out.append(f"- **{k}: {v['verdict']}** — survives at layers {v['layers_survives']}; NEAR at {v['layers_NEAR'] or 'none'}; fails at {[L for L in range(31) if v['per_layer'][L] == 'fails'] or 'none'}.")
    out.append(f"- **Branch:** {t['branch']}.")
    w("sec3", "\n".join(out))


def sec4():
    t = load_json(RAW / "task4_angle.json")
    c, raw = t["cos_clean"], t["raw"]
    rows = ["| L | cos(ĝ, ŝ) [95% CI] | regime | cos(guilt, shame) raw | cos(guilt, nn) | cos(shame, nn) |", "|---|---|---|---|---|---|"]
    for L in LAYERS:
        rows.append(f"| {L}{' †' if L == 31 else ''} | {c['point'][L]:+.3f} [{c['lo'][L]:+.3f}, {c['hi'][L]:+.3f}] | {t['regime'][L]} | {raw['guilt,shame'][L]:+.3f} | {raw['guilt,nn'][L]:+.3f} | {raw['shame,nn'][L]:+.3f} |")
    rows.append(f"\nRegime reading (operational, accepted on record): {t['regime_rule']}.")
    w("sec4", "\n".join(rows))


def sec5():
    t = load_json(RAW / "task5_crossvoice.json")
    cv = t["cross_voice"]
    rows = ["| L | cos(ĝ, received_act) | cos(ŝ, received_self) | cos(ĝ, received_self) | cos(ŝ, received_act) |", "|---|---|---|---|---|"]
    for L in LAYERS:
        rows.append(f"| {L}{' †' if L == 31 else ''} | {cv['guilt_clean,received_act'][L]:+.3f} | {cv['shame_clean,received_self'][L]:+.3f} | {cv['guilt_clean,received_self'][L]:+.3f} | {cv['shame_clean,received_act'][L]:+.3f} |")
    w("sec5a", "\n".join(rows))
    d = t["distinctness"]
    arrows = ["guilt_clean", "shame_clean", "received_act", "received_self"]
    hdr = "| L | " + " | ".join(f"{a}·{k}" for a in arrows for k in ("refusal", "badmed", "persona", "random_seed0")) + " |"
    rows = [hdr, "|---|" + "---|" * (4 * len(arrows))]
    for L in LAYERS:
        rows.append(f"| {L}{' †' if L == 31 else ''} | " + " | ".join(f"{d[a][k][L]:+.3f}" for a in arrows for k in ("refusal", "badmed", "persona", "random_seed0")) + " |")
    w("sec5b", "\n".join(rows))


def sec6():
    t = load_json(RAW / "task6_band.json")
    t3 = load_json(RAW / "task3_validation.json")
    c = t["chosen"]
    lines = [f"- **Basis:** {t['basis']}; {t['n_valid_runs']} of {t['n_runs']} candidate runs (lengths 4–6 within layers 0–30) satisfy the cos(ĝ, ŝ) sign constraint.",
             f"- **Chosen band: layers {c['start']}–{c['end']} (length {c['length']}), centre layer {c['centre']}, score {c['score']:.4f}, cos sign {'+' if c['cos_sign'] > 0 else '−'}.**"]
    for i, r in enumerate(t["runners_up"]):
        lines.append(f"- Runner-up {i+1}: layers {r['start']}–{r['end']} (length {r['length']}), centre {r['centre']}, score {r['score']:.4f}.")
    top = sorted([r for r in t["all_runs"] if r["cos_sign_consistent"]], key=lambda r: (-r["score"], r["start"], r["length"]))
    tied = [r for r in top if abs(r["score"] - top[0]["score"]) < 1e-9]
    lines.append(f"- Runs tied at the top score ({top[0]['score']:.4f}): {len(tied)} — " + "; ".join(f"{r['start']}–{r['end']}" for r in tied) + ". The pre-stated tie-break (earlier start, then shorter run) decided among them.")
    A = t3["arrow"]["point"]
    lines.append("- Ranking of the ten best runs by score (point estimates, rows 3 and 4 averaged): " + "; ".join(f"{r['start']}–{r['end']} {r['score']:.4f}" for r in top[:10]) + ".")
    lo = np.array(t3["arrow"]["lo"]); alt = (lo[2] + lo[3]) / 2
    runs = [(a, a + l - 1, float(alt[a:a + l].mean())) for l in (4, 5, 6) for a in range(0, 31 - l + 1)]
    runs.sort(key=lambda x: (-x[2], x[0], x[1] - x[0]))
    lines.append("- Informational only (not the rule): ranking the same runs by the CI **lower** bound instead of the point estimate gives " + "; ".join(f"{a}–{b} {s:.4f}" for a, b, s in runs[:5]) + ".")
    w("sec6", "\n".join(lines))


def sec7():
    S = load_json(RAW / "task7" / "summary.json")
    meta = S["meta"]
    lines = [f"- **Centre layer:** {meta['layer']}; **multipliers c = {meta['mults']}**; greedy, 128 new tokens; steering added to the output of `model.model.layers[{meta['layer']}]` at all positions.",
             "- **σ per arrow** (sample sd of the 200 first-person `mean` projections on the unit arrow at the centre layer) and **absolute norm added** per arm:\n",
             "| arrow | σ | mean projection (first-person passages) | norm added at c = " + " | norm added at c = ".join(str(c) for c in meta["mults"]) + " |", "|---|---|---|" + "---|" * len(meta["mults"])]
    for k, s in meta["sigma"].items():
        lines.append(f"| {k} | {s:.4f} | {meta['proj_mean_first_person'][k]:+.4f} | " + " | ".join(f"{c * s:.4f}" for c in meta["mults"]) + " |")
    hd = load_json(RAW / "task7" / "hook_diagnostic.json")
    lines.append("\n- **Hook check** (`task7/hook_diagnostic.json`; the next layer's input read with a pre-hook, two items × two arrows × two c): " + "; ".join(f"{c['arrow']} c={c['c']}: expected norm {c['expected_norm']:.4f}, measured {c['next_layer_input_delta_mean_norm']:.4f} at all {c['positions']} positions (max |dev| {c['max_abs_dev_from_expected']:.3f}, bf16), final-layer delta norm {c['final_hidden_delta_mean_norm']:.1f}, logits max |Δ| {c['logits_max_abs_delta']:.1f}" for c in hd["checks"][:4]) + ". Note: `output_hidden_states` in transformers 4.57.6 records a layer's output *before* a forward hook's replacement, so the first check in `meta.json` (which read `hidden_states[L+1]`) shows a zero delta; the pre-hook diagnostic is the valid one, and the unsteered/random arms reproduce the unhooked greedy output byte-for-byte where the random step is small.")
    labels = [l for l in S["labels"]]
    for m in S["models"]:
        lines.append(f"\n**Label distribution — `{m}`** (8 items per arm):\n")
        lines.append("| arm | " + " | ".join(labels) + " |"); lines.append("|---|" + "---|" * len(labels))
        for a in S["arms"]:
            lines.append(f"| {a} | " + " | ".join(str(S["distribution"][m][a][l]) for l in labels) + " |")
    ag = S["agreement_overall"]
    lines.append(f"\n- **Agreement between the two judges:** {ag['agree']}/{ag['n']} = {ag['agree']/max(ag['n'],1):.3f}. Per arm: " + "; ".join(f"{a} {v['agree']}/{v['n']}" for a, v in S["agreement"].items()) + ".")
    lines.append(f"- **Disagreements ({len(S['disagreements'])}):**\n")
    for d in S["disagreements"]:
        m0, m1 = S["models"]
        lines.append(f"  - `{d['item']}` / {d['arm']}: {m0} → **{d[m0]['label']}** ({d[m0]['reason']}); {m1} → **{d[m1]['label']}** ({d[m1]['reason']}). Text: “{d['text'][:300].replace(chr(10), ' ')}{'…' if len(d['text']) > 300 else ''}”")
    lines.append("\n**Pre-registered predictions (S2-plan §5c), read against the unsteered arm:**\n")
    lines.append("| model, c | +ĝ raises act-focused | +ŝ raises self-focused | +nn raises outcome-negative-only | random changes nothing |"); lines.append("|---|---|---|---|---|")
    for k, p in S["predictions"].items():
        cells = []
        for name in ("+guilt_clean raises act-focused", "+shame_clean raises self-focused", "+nn raises outcome-negative-only"):
            v = p[name]; cells.append(f"{'holds' if v['holds'] else 'does not hold'} ({v['unsteered']} → {v['steered']})")
        v = p["random changes nothing"]
        fmt = lambda d: ", ".join(f"{k.split('-')[0]} {n}" for k, n in d.items() if n)
        cells.append(f"{'holds' if v['holds'] else 'does not hold'} (unsteered: {fmt(v['unsteered'])}; random+: {fmt(v['random+'])}; random−: {fmt(v['random-'])})")
        lines.append(f"| {k} | " + " | ".join(cells) + " |")
    lines.append("\n**Coherence** (judge `incoherent` labels by either model; generations that did not end with `<|eot_id|>` within 128 tokens; mean new tokens):\n")
    lines.append("| arm | incoherent (items) | no eot (items) | mean new tokens |"); lines.append("|---|---|---|---|")
    for a, v in S["coherence"].items():
        lines.append(f"| {a} | {', '.join(v['incoherent_any_judge']) or '—'} | {', '.join(v['no_eot']) or '—'} | {v['mean_new_tokens']:.1f} |")
    inj = meta["injected_cos_with_persona_badmed"]
    lines.append("\n**Exploratory — persona / badmed projection of the reflections** (clean re-read of each generated reflection, mean over its tokens at the centre layer, on the unit axes from `dirs_8B_base_sweep.pt`; per-arm mean over 8 items). The injected vector's own component per unit step is cos(û, axis): " + "; ".join(f"{k}: persona {v['persona']:+.3f}, badmed {v['badmed']:+.3f}" for k, v in inj.items()) + ".\n")
    lines.append("| arm | persona (mean) | badmed (mean) |"); lines.append("|---|---|---|")
    for a, v in S["exploratory_persona_badmed"].items():
        lines.append(f"| {a} | {v['persona_mean']:+.4f} | {v['badmed_mean']:+.4f} |")
    u = S["usage"]
    lines.append(f"\n- **Judge cost from returned usage** (list prices in `judge_rubrics.PRICES`): " + "; ".join(f"`{m}` {u[m]['prompt_tokens']} prompt + {u[m]['completion_tokens']} completion tokens = ${S['cost_usd'][m]:.4f}" for m in S["models"]) + f"; **total ${S['total_cost_usd']:.4f}**.")
    w("sec7", "\n".join(lines))


def sec8():
    t = load_json(RAW / "task8_bridge.json")
    lines = [f"- Centre layer {t['layer']}; means with 1,000-resample bootstrap CI over scenario ids (seed 0); paired class differences from the same resamples. Random = seed-0 unit vector at the same layer and position.\n",
             "| position : arrow | neutral_correction | act_blame | self_blame | self_blame − act_blame | act_blame − neutral_correction | self_blame − neutral_correction |", "|---|---|---|---|---|---|---|"]
    for k, v in t["table"].items():
        c, d = v["classes"], v["paired_differences"]
        lines.append(f"| {k} | " + " | ".join(f"{c[x]['mean']:+.4f} [{c[x]['lo']:+.4f}, {c[x]['hi']:+.4f}]" for x in t["classes"]) + " | " + " | ".join(f"{d[x]['mean']:+.4f} [{d[x]['lo']:+.4f}, {d[x]['hi']:+.4f}]" for x in ("self_blame - act_blame", "act_blame - neutral_correction", "self_blame - neutral_correction")) + " |")
    o = t["orderings"]
    lines.append(f"\n- `post`·ŝ orders self_blame > act_blame > neutral_correction by point estimate: **{o['post.shame_clean: self_blame > act_blame > neutral_correction']}** (class means nc/ab/sb = {', '.join(f'{x:+.4f}' for x in o['post.shame_clean class means (nc, ab, sb)'])}).")
    lines.append(f"- `post`·ĝ puts act_blame above both others by point estimate: **{o['post.guilt_clean: act_blame > self_blame and act_blame > neutral_correction']}** (class means nc/ab/sb = {', '.join(f'{x:+.4f}' for x in o['post.guilt_clean class means (nc, ab, sb)'])}).")
    w("sec8", "\n".join(lines))


def sec9():
    t = load_json(RAW / "task9_organism.json")
    ld = t.get("load") or {}
    lines = [f"- Organism: {ld.get('model')} @ {ld.get('revision')}, load {ld.get('load_s')} s, {ld.get('vram_after_load_gib')} GiB after load; VRAM allocated before load {ld.get('vram_allocated_before_load_gib')} GiB. Same renders and `mean` position as Task 1; centre layer {t['layer']}. Δ = mean_organism(class) − mean_base(class), projected on the unit arrows; random floor = the same Δ on randctl seeds 0–9.\n",
             "| class | n | ‖Δ‖ | Δ·ĝ | Δ·ŝ | random |Δ·r| mean (seeds 0–9) | random |Δ·r| max | Δ·ĝ / random mean | Δ·ŝ / random mean | base sd of x·ĝ | base sd of x·ŝ |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for c, v in t["table"].items():
        lines.append(f"| {c} | {v['n']} | {v['delta_norm']:.4f} | {v['proj_guilt_clean']:+.4f} | {v['proj_shame_clean']:+.4f} | {v['random_abs_mean']:.4f} | {v['random_abs_max']:.4f} | {v['ratio_guilt_over_random_abs_mean']:+.1f} | {v['ratio_shame_over_random_abs_mean']:+.1f} | {v['base_sd_proj_guilt_clean']:.4f} | {v['base_sd_proj_shame_clean']:.4f} |")
    al = t["all_layers"]["all"]
    lines.append("\n- Informational, all passages, every layer (Δ·ĝ / Δ·ŝ / random |Δ·r| mean): " + "; ".join(f"L{L} {al['proj_guilt_clean'][L]:+.3f}/{al['proj_shame_clean'][L]:+.3f}/{al['random_abs_mean'][L]:.3f}" for L in LAYERS) + ".")
    w("sec9", "\n".join(lines))


def assemble():
    tpl = (ROOT / "scripts" / "s2b" / "report_template.md").read_text(encoding="utf-8")
    for p in T.glob("*.md"):
        tpl = tpl.replace("{{" + p.stem + "}}", p.read_text(encoding="utf-8").rstrip())
    (ROOT / "reports" / "S2b-arrows.md").write_text(tpl, encoding="utf-8")
    print("assembled reports/S2b-arrows.md; unresolved placeholders:", [l for l in tpl.splitlines() if "{{" in l][:5])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--assemble", action="store_true")
    a = ap.parse_args()
    for fn in (sec1, sec2, sec3, sec4, sec5, sec6, sec7, sec8, sec9):
        try:
            fn()
        except FileNotFoundError as e:
            print(f"{fn.__name__}: skipped ({e})")
    if a.assemble:
        assemble()


if __name__ == "__main__":
    main()
