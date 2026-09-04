"""S1d Task 6 — the two figures. Machine-written; regenerate, never hand-edit.

writeup/figs/s1d_blame_target.{png,pdf}      — Task 3's label distribution, mode x fork, stacked with CIs
writeup/figs/s1d_instrument_natural.{png,pdf} — Task 4's AUROC by layer for the primary contrast:
                                                arrows as lines, the randctl floor as a shaded band,
                                                bag-of-words as a dashed line
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
_spec = importlib.util.spec_from_file_location("s1d_common", REPO / "scripts" / "s1d" / "common.py")
C = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(C)
FIGS = REPO / "writeup" / "figs"

LABELS = ["act-focused", "self-focused", "outcome-negative-only", "neutral", "incoherent"]
COLORS = ["#2c6fbb", "#b3452c", "#8a8a8a", "#e0b040", "#4c4c4c"]
CELLS = [("deceived", "A"), ("deceived", "B"), ("akratic", "A"), ("akratic", "B"), ("vicious", "A"), ("vicious", "B")]


def fig_blame_target():
    d = json.load(open(C.OUT / "t3_q1.json", encoding="utf-8"))
    fig, ax = plt.subplots(figsize=(9, 5.2))
    x = np.arange(len(CELLS))
    base = np.zeros(len(CELLS))
    for L, col in zip(LABELS, COLORS):
        rates = np.array([d["cells"]["%s/%s" % c]["labels"][L]["rate"] for c in CELLS])
        ax.bar(x, rates, bottom=base, color=col, width=0.62, label=L, edgecolor="white", linewidth=0.6)
        for i, c in enumerate(CELLS):
            e = d["cells"]["%s/%s" % c]["labels"][L]
            if e["count"] == 0:
                continue
            lo, hi = e["ci95"]
            # the CI of this label's own rate, drawn on its segment
            ax.plot([x[i], x[i]], [base[i] + lo, base[i] + hi], color="black", lw=1.1, zorder=5)
            for yv in (base[i] + lo, base[i] + hi):
                ax.plot([x[i] - 0.09, x[i] + 0.09], [yv, yv], color="black", lw=1.1, zorder=5)
        base = base + rates
    ax.set_xticks(x)
    ax.set_xticklabels(["%s\nfork %s" % c for c in CELLS])
    ax.set_ylabel("share of probe replies")
    ax.set_ylim(0, 1.14)
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_title("S1d Q1 — the subject's own blame target after its own act, by route and probe framing\n"
                 "reflection judge, no feedback arm present; 95 % cluster bootstrap over targets (2,000, seed 0). EXPLORATORY.",
                 fontsize=10, pad=14)
    for i, c in enumerate(CELLS):
        ax.text(x[i], 1.03, "n=%d" % d["cells"]["%s/%s" % c]["n"], ha="center", fontsize=8, color="#444444")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.28), ncol=5, fontsize=8, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(FIGS / ("s1d_blame_target.%s" % ext), dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig_instrument():
    d = json.load(open(C.OUT / "t4_q2.json", encoding="utf-8"))
    info = d["contrasts"]["primary"]
    layers = list(range(32))
    fig, ax = plt.subplots(figsize=(9.5, 5.4))
    ax.axvspan(13.6, 18.4, color="#f0f0f0", zorder=0)
    ax.text(16, 0.03, "primary band L14–18", ha="center", fontsize=8, color="#666666")
    lo = [info["random_floor"][str(L)]["pooled"]["min"] for L in layers]
    hi = [info["random_floor"][str(L)]["pooled"]["max"] for L in layers]
    ax.fill_between(layers, lo, hi, color="#c8c8c8", alpha=0.75, zorder=1,
                    label="random floor, randctl seeds 0–9 (min–max)")
    words = info["bag_of_words"]["pooled"]
    ax.axhline(words, color="#111111", ls="--", lw=1.6, zorder=4,
               label="bag-of-words baseline (%.3f)" % words)
    ax.axhline(0.5, color="#888888", lw=0.8, zorder=1)
    styles = {"guilt_clean": ("#2c6fbb", "-"), "shame_clean": ("#b3452c", "-"), "nn": ("#7a7a7a", "-"),
              "received_act": ("#3c8f5a", "-"), "received_self": ("#8250a8", "-"),
              "refusal": ("#d08a20", ":"), "badmed": ("#1f9ea8", ":"), "persona": ("#a8446e", ":"),
              "persona_meandiff": ("#6b8e23", ":")}
    for ax_name, (col, ls) in styles.items():
        ys = [info["table"][ax_name][str(L)]["pooled"] for L in layers]
        ax.plot(layers, ys, color=col, ls=ls, lw=1.5, label=ax_name, zorder=3)
    ax.set_xlabel("layer")
    ax.set_ylabel("AUROC, %s (positive class) vs %s" % tuple(info["contrast"].split(" vs ")))
    ax.set_xlim(-0.5, 31.5)
    ax.set_ylim(0, 1)
    ax.set_title("S1d Q2 — do the arrows separate the subject's own blame target, on its own words?\n"
                 "%s at the `answer` position; n+=%d, n-=%d, folds by target. EXPLORATORY."
                 % (info["contrast"], info["n_positive"], info["n_negative"]), fontsize=10)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.34), ncol=4, fontsize=8, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(FIGS / ("s1d_instrument_natural.%s" % ext), dpi=200, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    FIGS.mkdir(parents=True, exist_ok=True)
    fig_blame_target()
    fig_instrument()
    for p in sorted(FIGS.glob("s1d_*")):
        print(p.relative_to(REPO), "%.0f kB" % (p.stat().st_size / 1e3))
