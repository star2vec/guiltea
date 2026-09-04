"""S1h Task 4 — the one figure. Machine-written; regenerate, never hand-edit.

writeup/figs/s1h_signature.{png,pdf} — AUROC by layer in the prompt-held-constant cell (vicious route,
agent-directed fork; 64 act-focused against 19 self-focused) for the two persona axes and the two cleaned
arrows. The randctl seed 0-9 floor is the shaded min-max band of the seeds' own per-layer AUROCs, the
bag-of-words baseline is the dashed line, both pre-specified bands (L14-18 primary, L6-11 secondary) are
shaded, and 0.5 is marked.

Two panels because both summary statistics are headline: the pooled AUROC and the leave-one-target-out fold
statistic (the mean of the per-target AUROCs). A band mean is a scalar, so what is drawn per layer is the
AUROC itself and the shading shows which layers the band means average over.
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
_spec = importlib.util.spec_from_file_location("s1h_cells", REPO / "scripts" / "s1h" / "cells.py")
K = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(K)
FIGS = REPO / "writeup" / "figs"

DRAW = {"persona": ("#a8446e", "-"), "persona_meandiff": ("#6b8e23", "-"),
        "guilt_clean": ("#2c6fbb", "-"), "shame_clean": ("#b3452c", "-")}
STAT_TITLE = {"pooled": "pooled AUROC", "by_target_mean": "fold statistic (mean of the per-target AUROCs)"}


def main():
    d = json.load(open(K.OUT / "t1_vicious_forkB.json", encoding="utf-8"))["cell"]
    layers = list(range(32))
    fig, axs = plt.subplots(1, 2, figsize=(12.4, 5.8), sharey=True, gridspec_kw={"wspace": 0.05})
    handles = labels = None
    for panel, stat in enumerate(K.STATS):
        ax = axs[panel]
        ax.axvspan(13.6, 18.4, color="#eeeeee", zorder=0)
        ax.axvspan(5.6, 11.4, color="#f6f6f6", zorder=0)
        lo = [d["floor_by_layer"][stat][str(L)]["min"] for L in layers]
        hi = [d["floor_by_layer"][stat][str(L)]["max"] for L in layers]
        ax.fill_between(layers, lo, hi, color="#c8c8c8", alpha=0.8, zorder=1,
                        label="random floor, randctl seeds 0-9 (min-max)")
        w = d["bag_of_words"][stat]
        ax.axhline(w, color="#111111", ls="--", lw=1.6, zorder=4,
                   label="bag-of-words baseline (%.3f / %.3f)" % (d["bag_of_words"]["pooled"],
                                                                 d["bag_of_words"]["by_target_mean"]))
        ax.axhline(1.0 - w, color="#111111", ls=(0, (1, 3)), lw=1.1, zorder=4,
                   label="the same baseline mirrored through 0.5")
        ax.axhline(0.5, color="#888888", lw=0.9, zorder=2)
        # the comparison the verdict actually makes: the seeds' own L14-18 BAND MEANS, whose spread is
        # much narrower than the per-layer min-max band above. Drawn across the primary band only.
        fb = d["verdict"][stat]["L14_18"]["floor_band_mean"]
        ax.fill_between([13.6, 18.4], [fb["auroc_min"]] * 2, [fb["auroc_max"]] * 2, color="#5a5a5a",
                        alpha=0.55, zorder=2.5,
                        label="seed floor on the headline statistic: their own L14-18 band means")
        for name, (col, ls) in DRAW.items():
            ys = [d["table"][stat][name][str(L)] for L in layers]
            ax.plot(layers, ys, color=col, ls=ls, lw=1.9, label=name, zorder=3)
        ax.set_xlabel("layer", fontsize=9)
        ax.set_xlim(-0.5, 31.5)
        ax.set_ylim(0.0, 1.0)
        ax.set_title(STAT_TITLE[stat], fontsize=9.5)
        ax.spines[["top", "right"]].set_visible(False)
        if panel == 0:
            ax.text(16, 0.025, "L14-18", ha="center", fontsize=7.5, color="#666666")
            ax.text(8.5, 0.025, "L6-11", ha="center", fontsize=7.5, color="#666666")
            handles, labels = ax.get_legend_handles_labels()
    axs[0].set_ylabel("AUROC, act-focused (positive) vs self-focused\nbelow 0.5 = self-focused projects higher",
                      fontsize=9)
    fig.suptitle("S1h — the shame-like signature with the persona prompt held constant\n"
                 "vicious route, agent-directed fork only: %d act-focused against %d self-focused, one system "
                 "prompt, one route, one question wording\n`answer` position, folds by target (%d targets, %d "
                 "holding both classes).\nThe wide grey band is the floor per layer; the dark bar inside "
                 "L14-18 is the floor on the headline statistic, the seeds' own band means. EXPLORATORY."
                 % (d["n_positive"], d["n_negative"], d["n_targets"], d["n_targets_with_both_classes"]),
                 fontsize=9.5)
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=8, frameon=False,
               bbox_to_anchor=(0.5, 0.012))
    fig.subplots_adjust(left=0.085, right=0.99, top=0.81, bottom=0.225)
    FIGS.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(FIGS / ("s1h_signature.%s" % ext), dpi=200)
    plt.close(fig)
    for p in sorted(FIGS.glob("s1h_*")):
        print(p.relative_to(REPO), "%.0f kB" % (p.stat().st_size / 1e3))


if __name__ == "__main__":
    main()
