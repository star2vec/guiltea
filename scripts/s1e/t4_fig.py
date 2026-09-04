"""S1e Task 4 — the one figure. Machine-written; regenerate, never hand-edit.

writeup/figs/s1e_depth_matched.{png,pdf} — Task 1's L14-18 band-mean AUROC by turn index, one line per
axis, the matched random floor (randctl seeds 0-9, min-max of their own band means) as a shaded band,
the 0.5 line marked, class counts annotated per turn index. One panel per label source.

The `answer` position and the by-target fold statistic are drawn, because that is the statistic the
report's verdict rests on: the pooled value at a fixed turn index is still free to separate the classes
by target identity, and the fold statistic is the one that is matched on both depth and target. The
pooled curves, the `into` position and the filler-excluded variant are in results/raw/s1e/t1_trigger.json.
"""
from __future__ import annotations

import importlib.util
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location("s1e_depth", REPO / "scripts" / "s1e" / "depth.py")
D = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(D)
FIGS = REPO / "writeup" / "figs"

STYLES = {"refusal": ("#d08a20", ":"), "badmed": ("#1f9ea8", ":"), "persona": ("#a8446e", ":"),
          "persona_meandiff": ("#6b8e23", ":"), "guilt_clean": ("#2c6fbb", "-"),
          "shame_clean": ("#b3452c", "-"), "nn": ("#111111", "-"),
          "received_act": ("#3c8f5a", "-"), "received_self": ("#8250a8", "-")}
STAT = "by_target_mean"
POSITION = "answer"
SOURCE_TITLE = {"t_primary": "the rig's stored T_primary", "merged": "the merged D-019 labels"}


def main():
    d = json.load(open(D.OUT / "t1_trigger.json", encoding="utf-8"))
    ref = d["t_primary|%s|as_specified" % POSITION]
    tb = [ref["per_t"][str(t)]["n_targets_with_both_classes"] for t in ref["evaluable_turns"]]
    fig, axs = plt.subplots(1, 2, figsize=(11.5, 6.4), sharey=True,
                            gridspec_kw={"width_ratios": [9, 4], "wspace": 0.05})
    handles = labels = None
    for panel, source in enumerate(D.SOURCES):
        ax = axs[panel]
        r = d["%s|%s|as_specified" % (source, POSITION)]
        ts = r["evaluable_turns"]
        f = [r["per_t"][str(t)]["verdict"][STAT]["L14_18"]["floor_band_mean"] for t in ts]
        ax.fill_between(ts, [x["auroc_min"] for x in f], [x["auroc_max"] for x in f],
                        color="#c8c8c8", alpha=0.8, zorder=1,
                        label="random floor, randctl seeds 0-9 (min-max of their own band means)")
        ax.axhline(0.5, color="#888888", lw=0.9, zorder=2)
        for name, (col, ls) in STYLES.items():
            ys = [r["per_t"][str(t)]["verdict"][STAT]["L14_18"]["per_axis"][name]["band_mean"] for t in ts]
            ax.plot(ts, ys, color=col, ls=ls, lw=2.2 if name == "nn" else 1.4,
                    marker="o" if name == "nn" else None, ms=3.4, label=name, zorder=3)
        for t in ts:
            e = r["per_t"][str(t)]
            ax.text(t, 0.975, "%d\n%d" % (e["n_positive"], e["n_negative"]), ha="center", va="top",
                    fontsize=7.5, color="#444444")
        ax.text(ts[0] - 0.62, 0.975, "n+\nn-", ha="right", va="top", fontsize=7.5, color="#444444")
        ax.set_xlabel("turn index t (both classes graded `held` at t)", fontsize=9)
        ax.set_xticks(ts)
        ax.set_xlim(ts[0] - 1.0, ts[-1] + 0.5)
        ax.set_ylim(0.20, 1.0)
        ax.set_title("label source: %s" % SOURCE_TITLE[source], fontsize=9.5)
        ax.spines[["top", "right"]].set_visible(False)
        if panel == 0:
            handles, labels = ax.get_legend_handles_labels()
    axs[0].set_ylabel("AUROC, L14-18 band mean: will-break vs never-breaks\n(mean of the per-target AUROCs)",
                      fontsize=9)
    fig.suptitle("S1e Task 1 - is the break visible at fixed conversation depth?\n"
                 "among v1 chains still graded `held` at turn t: those that commit at a later turn "
                 "(positive) against those that never commit within ten turns (negative)\n"
                 "`answer` position, folds by target, %d-%d targets holding both classes; below 0.5 = the "
                 "will-break class projects lower.  EXPLORATORY." % (min(tb), max(tb)), fontsize=9.5)
    fig.legend(handles, labels, loc="lower center", ncol=5, fontsize=8, frameon=False,
               bbox_to_anchor=(0.5, 0.012))
    fig.subplots_adjust(left=0.095, right=0.985, top=0.845, bottom=0.185)
    FIGS.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(FIGS / ("s1e_depth_matched.%s" % ext), dpi=200)
    plt.close(fig)
    for p in sorted(FIGS.glob("s1e_*")):
        print(p.relative_to(REPO), "%.0f kB" % (p.stat().st_size / 1e3))


if __name__ == "__main__":
    main()
