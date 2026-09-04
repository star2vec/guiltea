"""S1g Task 3 — the one figure. Machine-written; regenerate, never hand-edit.

writeup/figs/s1g_heldout.{png,pdf} — the two pre-named axes' L14-18 band means by turn index on the
held-out v2 chains, the randctl seed 0-9 floor as a shaded min-max band, the 0.5 line marked, class
counts annotated, and the v1 curves from reports/S1e-depth-matched.md drawn faintly behind and
labelled as the search sample.

Only the two axes fixed in briefs/S1g-heldout-trigger.md are drawn, on the statistic that carries the
verdict: the depth- and target-matched fold statistic at the `answer` position under `t_primary`. The
v1 floor is deliberately NOT drawn - the brief asks for the v1 curve behind for comparison, and two
overlaid floors would obscure the one the verdict is measured against. The v1 floor is in the table in
reports/S1g-heldout-trigger.md section 6 and in results/raw/s1e/t1_trigger.json.

The count-weighted headline is not a turn index and is not plotted; it is the side-by-side table.

CPU only: no generation, no model load, no judge call, no GPU, no cost.
"""
from __future__ import annotations

import importlib.util
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location("s1g_v2", REPO / "scripts" / "s1g" / "v2.py")
V = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(V)
FIGS = REPO / "writeup" / "figs"

STYLE = {"nn": ("#111111", 2.6, "o"), "persona_meandiff": ("#6b8e23", 1.9, "s")}
LABEL = {"nn": "nn  (predicted > 0.5)", "persona_meandiff": "persona_meandiff  (predicted < 0.5)"}


def main():
    g = json.load(open(V.OUT / "t1_trigger_v2.json", encoding="utf-8"))
    e = json.load(open(REPO / "results" / "raw" / "s1e" / "t1_trigger.json",
                       encoding="utf-8"))["t_primary|answer|as_specified"]

    ts2 = g["floor_clearing_turns"]
    ts1 = e["evaluable_turns"]
    fig, ax = plt.subplots(figsize=(11.6, 6.9))

    # --- the search sample (v1), faintly behind
    for name, (col, _lw, _m) in STYLE.items():
        ys = [e["per_t"][str(t)]["verdict"]["by_target_mean"]["L14_18"]["per_axis"][name]["band_mean"]
              for t in ts1]
        ax.plot(ts1, ys, color=col, ls="--", lw=1.2, alpha=0.32, zorder=2,
                label="%s - v1, the search sample" % name)

    # --- the held-out floor, shaded
    lo = [g["per_turn"][str(t)]["floor"]["auroc_min"] for t in ts2]
    hi = [g["per_turn"][str(t)]["floor"]["auroc_max"] for t in ts2]
    ax.fill_between(ts2, lo, hi, color="#c8c8c8", alpha=0.85, zorder=1,
                    label="v2 random floor, randctl seeds 0-9 (min-max of their own band means)")
    ax.axhline(0.5, color="#888888", lw=0.9, zorder=3)

    # --- the held-out curves
    for name, (col, lw, mk) in STYLE.items():
        ys = [g["per_turn"][str(t)]["per_axis"][name]["band_mean"] for t in ts2]
        ax.plot(ts2, ys, color=col, ls="-", lw=lw, marker=mk, ms=7, zorder=5,
                label="%s - v2, held out" % LABEL[name])

    for t in ts2:
        c = g["per_turn"][str(t)]
        ax.text(t, 0.995, "%d\n%d\n%d" % (c["n_positive"], c["n_negative"],
                                           c["n_targets_with_both_classes"]),
                ha="center", va="top", fontsize=8.5, color="#333333")
    ax.text(0.62, 0.995, "n+\nn-\ntargets", ha="right", va="top", fontsize=8.5, color="#333333")
    ax.axvspan(0.5, 2.5, color="#f2c94c", alpha=0.13, zorder=0)
    ax.text(1.5, 0.205, "the two turn indices\nclearing the count floor on v2", ha="center",
            va="bottom", fontsize=8.5, color="#8a6d1a")

    ax.set_xlabel("turn index t  (both classes graded `held` at t)", fontsize=9.5)
    ax.set_ylabel("AUROC, L14-18 band mean: will-break vs never-breaks\n"
                  "(mean of the per-target AUROCs over targets holding both classes)", fontsize=9.5)
    ax.set_xticks(ts1)
    ax.set_xlim(0.4, ts1[-1] + 0.4)
    ax.set_ylim(0.18, 1.03)
    ax.spines[["top", "right"]].set_visible(False)
    hd = g["headline"]
    fig.suptitle("S1g - one pre-specified out-of-sample test of the S1e trigger\n"
                 "the held-out v2 chains: 40 chains over 5 targets, second persuader wording, never "
                 "used in the nine-axis search\n"
                 "axes, band, position, statistic, direction and label source all fixed before the "
                 "data was read\n"
                 "count-weighted headline:  nn %.3f against a largest-seed floor of %.3f, REPLICATED"
                 "   |   persona_meandiff %.3f against %.3f, not replicated   |   EXPLORATORY"
                 % (hd["per_axis"]["nn"]["band_mean"], hd["floor"]["auroc_max"],
                    hd["per_axis"]["persona_meandiff"]["band_mean"], hd["floor"]["auroc_min"]),
                 fontsize=9.0)
    h, l = ax.get_legend_handles_labels()
    order = [l.index(x) for x in
             ["%s - v2, held out" % LABEL["nn"],
              "%s - v2, held out" % LABEL["persona_meandiff"],
              "v2 random floor, randctl seeds 0-9 (min-max of their own band means)",
              "nn - v1, the search sample", "persona_meandiff - v1, the search sample"]]
    fig.legend([h[i] for i in order], [l[i] for i in order], loc="lower center", ncol=2,
               fontsize=8.5, frameon=False, bbox_to_anchor=(0.5, 0.008))
    fig.subplots_adjust(left=0.105, right=0.985, top=0.795, bottom=0.215)
    FIGS.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(FIGS / ("s1g_heldout.%s" % ext), dpi=200)
    plt.close(fig)
    for p in sorted(FIGS.glob("s1g_*")):
        print(p.relative_to(REPO), "%.0f kB" % (p.stat().st_size / 1e3))


if __name__ == "__main__":
    main()
