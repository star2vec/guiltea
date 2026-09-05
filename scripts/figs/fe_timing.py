"""F-E — the capitulation-timing histogram (briefs/S6-figures-2.md §4).

First committed turn for the 192 `v1` deceived chains under the rig's stored labels: one bar per turn 1–10, plus
a separate bar, set apart, for the chains that never commit the harmful act within the ten turns. Counts printed
on every bar. Title as a sentence: "There is no fixed turn at which the model breaks".

Data: results/raw/s1b/t10/summary.json — `T_distribution` and `categories["no capitulation"]` are what is drawn;
the per-chain list in the same file is cross-checked against them, and the counts are asserted equal to the ones
the brief states. No selection, no value dropped, smoothed or clipped. CPU only: no generation, no model load,
no judge call, no cost.

Output: writeup/figs/s6_fe_timing.{png,pdf,caption.md}.
"""
from __future__ import annotations

import importlib.util
import json
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location("s6_common", REPO / "scripts" / "figs" / "common.py")
CM = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(CM)

SRC = CM.RAW / "s1b" / "t10" / "summary.json"
BRIEF_COUNTS = {1: 3, 2: 21, 3: 37, 4: 11, 5: 10, 6: 5, 7: 5, 8: 1, 9: 1, 10: 15}   # briefs/S6-figures-2.md §4
BRIEF_NEVER = 83
TITLE = "There is no fixed turn at which the model breaks"
X_NEVER = 12                                   # the never-commit bar sits one empty slot right of turn 10


def load():
    d = json.load(open(SRC, encoding="utf-8"))
    dist = {int(k): int(v) for k, v in d["T_distribution"].items()}
    never = int(d["categories"]["no capitulation"])
    n = int(d["n_chains"])
    # cross-check the summary counts against the per-chain list in the same file
    chains = d["chains"]
    assert len(chains) == n, (len(chains), n)
    per = Counter(c["T"] for c in chains)
    for c in chains:
        g = c["grades"]
        first = next((i + 1 for i, x in enumerate(g) if x == "committed"), None)
        assert first == c["T"], (c["target"], c["seed"], first, c["T"])
        assert len(g) == c["n_turns"] == 10, (c["target"], c["seed"], len(g))
    assert {k: v for k, v in per.items() if k is not None} == dist, (per, dist)
    assert per[None] == never, (per[None], never)
    assert sum(dist.values()) + never == n, (sum(dist.values()), never, n)
    # and against the brief
    assert dist == BRIEF_COUNTS, (dist, BRIEF_COUNTS)
    assert never == BRIEF_NEVER, (never, BRIEF_NEVER)
    assert len({c["target"] for c in chains}) == 16 and len({c["seed"] for c in chains}) == 12, "footer says 16 × 12"
    return d, dist, never, n


def draw(dist, never, n):
    import matplotlib.pyplot as plt
    CM.style()
    fig, ax = plt.subplots(figsize=(10.0, 5.6))
    turns = sorted(dist)
    ax.bar(turns, [dist[t] for t in turns], width=0.8, color=CM.NAMED, zorder=3)
    ax.bar([X_NEVER], [never], width=0.8, color=CM.GREY, zorder=3)
    ax.axvline(11, color=CM.MUTED, lw=1.0, ls=(0, (4, 3)), zorder=2)
    top = max(max(dist.values()), never)
    for t in turns:
        ax.text(t, dist[t] + 0.012 * top, "%d" % dist[t], ha="center", va="bottom", fontsize=12, fontweight="bold",
                color=CM.NAMED)
    ax.text(X_NEVER, never + 0.012 * top, "%d" % never, ha="center", va="bottom", fontsize=12, fontweight="bold",
            color=CM.GREY)
    ax.set_xticks(turns + [X_NEVER])
    ax.set_xticklabels(["%d" % t for t in turns] + ["never\n(no turn graded\ncommitted)"])
    ax.set_xlim(0.3, X_NEVER + 0.7)
    ax.set_ylim(0, top * 1.16)
    ax.set_xlabel("first turn at which the act judge grades the answer `committed` (the rig's stored label)")
    ax.set_ylabel("chains")
    n_commit = sum(dist.values())
    ax.text(0.02, 0.97, "%d v1 deceived chains, 10 persuader turns each:\n%d commit the harmful act at some turn "
            "(blue, by turn), %d never do (grey)" % (n, n_commit, never), transform=ax.transAxes, ha="left",
            va="top", fontsize=10.5, color=CM.TEXT, linespacing=1.3)
    ax.set_title(TITLE, fontsize=15, fontweight="bold", loc="left", pad=14)
    ax.grid(axis="y", color="#ececec", zorder=0)
    ax.set_axisbelow(True)
    fig.text(0.5, 0.005, "Base Llama-3.1-8B-Instruct, deceived route, 16 v1 targets × 12 seeds. Labels: the rig's stored "
             "act-judge grade (gpt-4o-mini) per turn.\n`T_used` in the source file: %s." % T_USED,
             ha="center", va="bottom", fontsize=8.8, color=CM.TEXT, linespacing=1.3)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    return fig


def main():
    global T_USED
    d, dist, never, n = load()
    T_USED = d["T_used"]
    fig = draw(dist, never, n)
    name = "s6_fe_timing"
    outs = CM.save(fig, name)
    n_commit = sum(dist.values())
    body = ["No example is selected: every one of the %d v1 chains is counted once, at its first turn graded "
            "`committed` by the rig's stored act-judge label, or in the never-commit bar if no turn is. Counts are "
            "read from `T_distribution` and `categories[\"no capitulation\"]` in the source file and cross-checked "
            "against its per-chain list (first `committed` grade per chain). Nothing dropped, smoothed or clipped."
            % n, "",
            "`T_used`, verbatim from the file: %s" % T_USED, "",
            "| first committed turn | chains |", "|---|---|"]
    body += ["| %d | %d |" % (t, dist[t]) for t in sorted(dist)]
    body += ["| never (no turn graded committed; `partial` grades may occur) | %d |" % never, "| total | %d |" % n, "",
             "%d of %d chains commit at some turn; the median first committed turn over those %d is 3 (the same "
             "count the D1 transcript panel's rule uses)." % (n_commit, n, n_commit), "",
             "The file's own categories, verbatim: " + ", ".join("%s = %d" % (k, v) for k, v in d["categories"].items())
             + "."]
    cap = CM.caption(name, TITLE, "scripts/figs/fe_timing.py",
                     ["results/raw/s1b/t10/summary.json (`T_distribution`, `categories`, per-chain grades)"],
                     "none — every v1 chain counted once under the rig's stored label", "\n".join(body))
    for p in outs + [cap]:
        print(p.relative_to(REPO), "%.0f kB" % (p.stat().st_size / 1e3))
    print("counts", dist, "never", never, "n", n)


if __name__ == "__main__":
    main()
