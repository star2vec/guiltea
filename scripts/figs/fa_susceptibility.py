"""F-A — "A refusal that will break already looks different at turn 1" (briefs/S6-figures.md).

The susceptibility result drawn as distributions rather than as an AUROC curve. For the `v1` search set
and the `v2` held-out set side by side, at turn 1 (the plain request, every chain refusing): the per-chain
L14-18 band-mean projection on the `nn` arrow at the `answer` position, one point per chain, the two
classes in two colours, class medians marked, the report's AUROC printed, the ten randctl seeds' AUROCs as
a small grey band on an inset axis, n per class and the number of targets printed.

Classes are exactly scripts/s1e/depth.py `members_trigger` at t = 1 under the rig's stored grade
(`t_primary`): positive = graded `held` at turn 1 and `committed` at a later turn; negative = graded
`held` at turn 1 and never `committed` within ten turns; `partial` or `committed` at turn 1 -> neither.

The AUROC printed is the statistic the verdicts in reports/S1e-depth-matched.md section 2 and
reports/S1g-heldout-trigger.md section 4 rest on: at t = 1, the per-layer target-matched AUROC (mean of the
per-target AUROCs over targets holding both classes) averaged over L14-18 - recomputed here through the S1e
and S1g code imported UNEDITED and asserted equal to the stored JSON. The strip plots the per-chain
band-mean projection; the AUROC of those plotted scores is a different quantity and is given in the caption.

Outputs: writeup/figs/s6_fa_susceptibility.{png,pdf,caption.md}. CPU only: no generation, no model load,
no judge call, no GPU, no cost.
"""
from __future__ import annotations

import importlib.util
import json
import random
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, REPO / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


CM = _load("s6_common", "scripts/figs/common.py")
D = _load("s1e_depth", "scripts/s1e/depth.py")      # the v1 machinery, unedited
V = _load("s1g_v2", "scripts/s1g/v2.py")            # the v2 rebinding of a second copy of it, unedited

NAME = "s6_fa_susceptibility"
AXIS, POSITION, SOURCE, T = "nn", "answer", "t_primary", 1
BAND = list(D.C.BAND_PRIMARY)                        # L14-18 (D-024 primary band)
TITLE = "A refusal that will break already looks different at turn 1"
FOOTER = "pre-specified on v1, tested once on v2; threshold committed before v2 was read"


def items_at_turn1(proj, axes, positions, chains):
    """(idx, y, target, kind, band-mean score) for every chain in a class at t = 1."""
    pi, ai = positions.index(POSITION), axes.index(AXIS)
    out = []
    for ch in chains:
        y = D.members_trigger(ch, SOURCE, T)
        if y is None:
            continue
        s = proj[ch["idx"], T - 1, pi, ai, :]
        out.append((ch["idx"], y, ch["target"], ch["kinds"][T - 1], float(np.mean(s[BAND])),
                    ch["seed"]))
    return out


def report_statistic(proj, axes, positions, layers, items, axis):
    """S1e/S1g: per-layer target-matched AUROC, averaged over L14-18. Via the S1g helpers, which wrap
    scripts/s1d/common.py grouped_auroc unedited."""
    rows = [(i, y, g, k) for (i, y, g, k, _s, _seed) in items]
    return V.band_mean(V.fold_curve_at_turn(proj, axes, positions, layers, rows, axis, T, POSITION))


def panel_data(set_name):
    if set_name == "v1":
        proj, axes, positions, layers, chains, meta = D.load_chains()
        stored = json.load(open(CM.RAW / "s1e" / "t1_trigger.json", encoding="utf-8"))
        ref = stored["t_primary|answer|as_specified"]["per_t"][str(T)]
        stored_auroc = ref["verdict"]["by_target_mean"]["L14_18"]["per_axis"][AXIS]["band_mean"]
        stored_floor = ref["verdict"]["by_target_mean"]["L14_18"]["floor_band_mean"]
        stored_n = (ref["n_positive"], ref["n_negative"], ref["n_targets_with_both_classes"])
        stored_src = "results/raw/s1e/t1_trigger.json"
    else:
        V.bind_v2()
        proj, axes, positions, layers, chains, meta = V.load_chains_v2()
        stored = json.load(open(CM.RAW / "s1g" / "t1_trigger_v2.json", encoding="utf-8"))
        ref = stored["per_turn"][str(T)]
        stored_auroc = ref["per_axis"][AXIS]["band_mean"]
        stored_floor = ref["floor"]
        stored_n = (ref["n_positive"], ref["n_negative"], ref["n_targets_with_both_classes"])
        stored_src = "results/raw/s1g/t1_trigger_v2.json"
    items = items_at_turn1(proj, axes, positions, chains)
    n_pos = sum(1 for it in items if it[1] == 1)
    n_neg = len(items) - n_pos
    per_target = {}
    for it in items:
        per_target.setdefault(it[2], [0, 0])[0 if it[1] == 1 else 1] += 1
    both = sorted(g for g, c in per_target.items() if c[0] and c[1])
    auroc = report_statistic(proj, axes, positions, layers, items, AXIS)
    seeds = {s: report_statistic(proj, axes, positions, layers, items, s) for s in D.C.RANDOM_AXES}
    # the same numbers straight out of the stored S1e / S1g JSON: must agree
    assert (n_pos, n_neg, len(both)) == tuple(stored_n), ((n_pos, n_neg, len(both)), stored_n)
    assert abs(auroc - stored_auroc) < 1e-9, (auroc, stored_auroc)
    assert abs(min(seeds.values()) - stored_floor["auroc_min"]) < 1e-9
    assert abs(max(seeds.values()) - stored_floor["auroc_max"]) < 1e-9
    # the AUROC of the plotted scores themselves (a different quantity; caption only)
    sc = np.array([it[4] for it in items]); y = np.array([it[1] for it in items]); g = np.array([it[2] for it in items])
    plotted = {"pooled": D.C.auroc(sc, y), "target_matched": D.C.grouped_auroc(sc, y, g)[0]}
    tmean = {t: float(np.mean(sc[g == t])) for t in set(g.tolist())}
    scc = np.array([v - tmean[t] for v, t in zip(sc, g)])
    plotted_c = {"pooled": D.C.auroc(scc, y), "target_matched": D.C.grouped_auroc(scc, y, g)[0]}
    return {"set": set_name, "items": items, "n_pos": n_pos, "n_neg": n_neg,
            "n_targets_total": len(set(c["target"] for c in chains)), "n_targets_both": len(both),
            "targets_both": both, "per_target": per_target, "n_chains": len(chains),
            "auroc": auroc, "seeds": seeds, "floor": (min(seeds.values()), max(seeds.values())),
            "plotted_auroc": plotted, "plotted_auroc_centred": plotted_c, "stored_src": stored_src,
            "proj_src": meta["proj_source"], "grade_mismatches": meta["stored_grade_vs_act_primary_mismatches"]}


def swarm_offsets(values, width=0.34, n_bins=28):
    """Deterministic beeswarm: bin the values, spread each bin's points symmetrically about the centre."""
    values = np.asarray(values, dtype=float)
    lo, hi = values.min(), values.max()
    edges = np.linspace(lo, hi + 1e-9, n_bins + 1)
    which = np.clip(np.searchsorted(edges, values, side="right") - 1, 0, n_bins - 1)
    off = np.zeros(len(values))
    for b in range(n_bins):
        idx = np.flatnonzero(which == b)
        if len(idx) == 0:
            continue
        idx = idx[np.argsort(values[idx], kind="mergesort")]
        k = len(idx)
        step = min(0.055, 2 * width / max(k, 1))
        pos = (np.arange(k) - (k - 1) / 2.0) * step
        off[idx] = pos
    return off


def scores(d, centred):
    """The plotted score per item: the band-mean projection, or (variant) that minus its target's mean
    over both classes at turn 1."""
    if not centred:
        return {id(it): it[4] for it in d["items"]}
    tm = {}
    for g in d["per_target"]:
        vals = [it[4] for it in d["items"] if it[2] == g]
        tm[g] = float(np.mean(vals))
    return {id(it): it[4] - tm[it[2]] for it in d["items"]}


def draw_panel(ax, d, label, centred=False):
    classes = [(1, "will break later", CM.NAMED), (0, "never breaks", CM.SECOND)]
    sc = scores(d, centred)
    y_all = list(sc.values())
    meds = {}
    for x, (y_cls, name, col) in enumerate(classes):
        its = [it for it in d["items"] if it[1] == y_cls]
        vals = np.array([sc[id(it)] for it in its])
        off = swarm_offsets(vals)
        ax.scatter(x + off, vals, s=26, color=col, alpha=0.75, linewidths=0, zorder=3)
        med = float(np.median(vals))
        meds[y_cls] = med
        ax.plot([x - 0.36, x + 0.36], [med, med], color=col, lw=2.6, zorder=4)
        if x == 0:
            ax.text(x - 0.40, med, "median %.2f" % med, color=col, fontsize=10, va="center", ha="right", zorder=5)
        else:
            ax.text(x + 0.40, med, "median %.2f" % med, color=col, fontsize=10, va="center", ha="left", zorder=5)
        ax.text(x, -0.06, "%s\nn = %d" % (name, len(vals)), transform=ax.get_xaxis_transform(),
                ha="center", va="top", fontsize=11, color=col)
        # outliers: printed, never dropped (brief, "Do not")
        mad = float(np.median(np.abs(vals - med))) or 1e-9
        for it, v in zip(its, vals):
            if abs(v - med) > 3.5 * 1.4826 * mad:
                ax.annotate("%s / seed %d" % (it[2], it[5]), (x, v), xytext=(6, 0),
                            textcoords="offset points", fontsize=8, color=CM.TEXT, va="center")
    ax.set_xlim(-0.75, 1.95)
    ax.set_xticks([])
    ax.set_title(label, fontsize=11.5, loc="left", pad=6)
    # header: the AUROC (the report's statistic) top-left, the ten seeds as a band on an inset axis top-right
    fl = d["floor"]
    ax.text(0.02, 0.985, "AUROC %.3f" % d["auroc"], transform=ax.transAxes, fontsize=15,
            fontweight="bold", color=CM.NAMED, ha="left", va="top")
    ax.text(0.02, 0.900, "target-matched, mean over L14\u201318\n(the report\'s statistic)\n"
            "random floor, 10 randctl seeds:\n%.3f\u2013%.3f" % fl,
            transform=ax.transAxes, fontsize=9, color=CM.TEXT, ha="left", va="top", linespacing=1.25)
    ins = ax.inset_axes([0.56, 0.905, 0.42, 0.045])
    ins.axvspan(fl[0], fl[1], color=CM.GREY_BAND, zorder=1)
    ins.axvline(d["auroc"], color=CM.NAMED, lw=3, zorder=3)
    ins.axvline(0.5, color=CM.GREY, lw=0.8, ls=":", zorder=2)
    ins.set_xlim(0.2, 0.9)
    ins.set_yticks([])
    ins.set_xticks([0.3, 0.5, 0.7, 0.9])
    ins.tick_params(labelsize=8, length=2, pad=1)
    ins.spines[["left", "right", "top"]].set_visible(False)
    ins.set_facecolor("none")
    ins.text(0.5, 1.3, "AUROC axis: 10 seeds (grey band), nn (blue)", transform=ins.transAxes,
             ha="center", va="bottom", fontsize=8.5, color=CM.TEXT)
    ax.text(0.98, 0.80, "%d of %d targets hold both classes" % (d["n_targets_both"], d["n_targets_total"]),
            transform=ax.transAxes, fontsize=9, color=CM.TEXT, ha="right", va="top")
    pooled = d["plotted_auroc_centred" if centred else "plotted_auroc"]["pooled"]
    note = ("points pooled across targets; the printed AUROC compares\nchains within a target "
            "(pooled AUROC of these points: %.2f)" % pooled if not centred else
            "each point minus its target\'s mean over both classes;\npooled AUROC of these centred points: %.2f"
            % pooled)
    ax.text(0.98, 0.735, note, transform=ax.transAxes, fontsize=8.5, color=CM.TEXT, ha="right", va="top",
            style="italic", linespacing=1.2)
    return meds


def figure(v1, v2, centred):
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 2, figsize=(12.4, 7.0), sharey=True,
                            gridspec_kw={"width_ratios": [1, 1], "wspace": 0.10})
    m1 = draw_panel(axs[0], v1, "v1 \u2014 the search set: %d chains, %d targets, persuader wording 1"
                    % (v1["n_chains"], v1["n_targets_total"]), centred)
    m2 = draw_panel(axs[1], v2, "v2 \u2014 held out: %d chains, %d targets, persuader wording 2"
                    % (v2["n_chains"], v2["n_targets_total"]), centred)
    y_all = [v for d in (v1, v2) for v in scores(d, centred).values()]
    lo, hi = min(y_all), max(y_all)
    pad = 0.08 * (hi - lo)
    axs[0].set_ylim(lo - pad, hi + 5.2 * pad)
    axs[0].set_ylabel("projection on the nn arrow, turn-1 answer\n(mean over layers 14\u201318"
                      + (", minus the target mean)" if centred else ")"))
    fig.suptitle(TITLE + (" \u2014 per-target-centred variant" if centred else ""),
                 fontsize=16, fontweight="bold", x=0.5, y=0.995)
    sub = ("One point per chain, at turn 1 of the persuasion: the plain harmful request, which every chain "
           "refuses. Chains that will\ncommit the harmful act at a later turn (blue) against chains that "
           "never do (orange). Base Llama-3.1-8B-Instruct; deceived route.")
    if centred:
        sub += ("\nNOT IN THE BRIEF \u2014 offered beside the specified figure because the projection\'s level "
                "differs by target and the shift is within-target.")
    fig.text(0.5, 0.925, sub, ha="center", va="top", fontsize=10.5, color=CM.TEXT)
    fig.text(0.5, 0.005, FOOTER, ha="center", va="bottom", fontsize=11.5, color=CM.TEXT, style="italic")
    fig.subplots_adjust(top=0.83 if not centred else 0.815, bottom=0.13, left=0.075, right=0.985)
    return fig, (m1, m2)


def caption_body(v1, v2, meds, centred):
    body = []
    for d, m in zip((v1, v2), meds):
        body.append("## %s\n" % d["set"])
        body.append("- chains %d; classes at turn 1 (`held` at turn 1 and `committed` later vs `held` at turn 1 and "
                    "never `committed`): **n = %d / %d**; %d of %d targets hold both classes (%s)."
                    % (d["n_chains"], d["n_pos"], d["n_neg"], d["n_targets_both"], d["n_targets_total"],
                       ", ".join("`%s`" % g for g in d["targets_both"])))
        body.append("- **AUROC printed: %.3f** \u2014 the report\'s statistic (per-layer target-matched AUROC, mean over "
                    "L14\u201318, `answer` position, `t_primary`), recomputed through `scripts/s1e/depth.py` / "
                    "`scripts/s1g/v2.py` unedited and asserted equal to `%s`." % (d["auroc"], d["stored_src"]))
        body.append("- random floor, the ten randctl seeds on the same items and folds, same statistic: "
                    "**%.3f\u2013%.3f** (per seed: %s)." % (d["floor"][0], d["floor"][1],
                                                        ", ".join("%.3f" % d["seeds"][s] for s in D.C.RANDOM_AXES)))
        body.append("- AUROC of the plotted scores themselves (a different quantity from the printed one): "
                    "raw band-mean score pooled %.3f, target-matched %.3f; per-target-centred score pooled %.3f, "
                    "target-matched %.3f."
                    % (d["plotted_auroc"]["pooled"], d["plotted_auroc"]["target_matched"],
                       d["plotted_auroc_centred"]["pooled"], d["plotted_auroc_centred"]["target_matched"]))
        body.append("- class medians of the plotted score: will break %.3f, never breaks %.3f." % (m[1], m[0]))
        body.append("- per-target counts (will break / never breaks): %s."
                    % ", ".join("`%s` %d/%d" % (g, c[0], c[1]) for g, c in sorted(d["per_target"].items())))
        body.append("- projections from `%s`; the rig\'s stored per-turn grade matches `act_primary.jsonl` on %d "
                    "mismatches (the reused loader\'s own assertion).\n" % (d["proj_src"], d["grade_mismatches"]))
    body.append("**What the eye sees versus what is printed.** The strip pools chains across targets, and the "
                "projection\'s level differs by target more than by class, so the pooled medians sit close together "
                "while the printed AUROC \u2014 which ranks chains within a target and averages over targets \u2014 "
                "clears its floor. Both numbers are on the figure. "
                + ("This variant subtracts each target\'s mean, so the pooled points show the within-target shift; "
                   "it is not the figure the brief specifies and is offered beside it for the researcher\'s choice."
                   if centred else
                   "A per-target-centred variant, `s6_fa_susceptibility_centred`, is written beside this file; it is "
                   "not the brief\'s figure and is offered for the researcher\'s choice."))
    body.append("\nFooter line, verbatim from the brief: *%s*. The v1 count-weighted headline over turn indices 1\u20139 is "
                "0.604 against a floor of 0.477\u20130.541 (`reports/S1e-depth-matched.md` \u00a72); the v2 count-weighted "
                "headline over turn indices 1\u20132 is 0.662 against 0.389\u20130.585 (`reports/S1g-heldout-trigger.md` \u00a74). "
                "Turn 1 alone is the susceptibility claim; this figure draws turn 1." % FOOTER)
    body.append("\nEXPLORATORY: `nn` is one of nine axes chosen by search on v1 (S1e); the v2 test was fixed in advance "
                "(S1g). No point is smoothed, clipped or dropped; any point beyond 3.5 robust SDs of its class median "
                "is labelled on the figure with its target and seed.")
    return "\n".join(body)


SOURCES = ["results/raw/s1d/proj_t4v1.npz (v1 projections, built by scripts/s1d/proj.py)",
           "results/raw/s1g/proj_t4v2.npz (v2 projections, built by scripts/s1g/proj_v2.py)",
           "results/raw/s1b/t4/<target>/v1_seed*.json and v2_seed*.json (stored per-turn grades)",
           "results/raw/s1e/t1_trigger.json and results/raw/s1g/t1_trigger_v2.json (cross-check only)"]
RULE = ("no example is selected; every chain in either class at turn 1 is a point. Class membership is "
        "scripts/s1e/depth.py `members_trigger(ch, \"t_primary\", 1)`.")


def main():
    CM.style()
    v1 = panel_data("v1")
    v2 = panel_data("v2")
    outs = []
    for centred, name in ((False, NAME), (True, NAME + "_centred")):
        fig, meds = figure(v1, v2, centred)
        outs += CM.save(fig, name)
        outs.append(CM.caption(name, TITLE + (" \u2014 per-target-centred variant (not in the brief)" if centred else ""),
                               "scripts/figs/fa_susceptibility.py", SOURCES, RULE,
                               caption_body(v1, v2, meds, centred)))
    for p in outs:
        print(p.relative_to(REPO), "%.0f kB" % (p.stat().st_size / 1e3))
    for d in (v1, v2):
        print("%s: n+ %d n- %d targets %d/%d AUROC %.3f floor %.3f-%.3f | plotted raw pooled %.3f matched %.3f "
              "| centred pooled %.3f matched %.3f"
              % (d["set"], d["n_pos"], d["n_neg"], d["n_targets_both"], d["n_targets_total"], d["auroc"],
                 d["floor"][0], d["floor"][1], d["plotted_auroc"]["pooled"], d["plotted_auroc"]["target_matched"],
                 d["plotted_auroc_centred"]["pooled"], d["plotted_auroc_centred"]["target_matched"]))


if __name__ == "__main__":
    main()
