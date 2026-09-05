"""F-D — the transcript panel: one persuasion chain with its internal readout running beside it
(briefs/S6-figures.md).

One deceived `v1` chain, turns down the page: the user's text on the left in a narrow column (the
persuader turns; the rig's post-act filler turns are marked as such), the assistant's answer on the right
(truncated to ~2 lines; the full text of every turn, thinking block included, is in the caption file), the
act judge's grade as a coloured tag per turn, the first committed turn marked, and beside every assistant
turn a horizontal bar for the L14-18 band-mean projection on `nn` and on `refusal` at the `answer`
position, with the per-turn random floor (max |band mean| over randctl seeds 0-9, same chain, same turn)
as a grey band.

Selection is made from the stored per-turn grades ALONE, before any chain text is loaded:
  D1, rule-selected  the chain at the median first-committed turn over the committing v1 chains
                     (lower-middle element of the sorted first-committed turns, as writeup/examples/h1.md
                     defines the median), seed 0 if present at that turn else the lowest seed; ties
                     across targets broken by target name in sorted order (the order scripts/s1d/proj.py
                     lists chains in). The rule is printed on the figure.
  D2, hand-picked    only if the researcher names a chain: `--d2 <target>:<seed>`. Drawn identically and
                     labelled "chosen by hand for illustration - see D1 and D3 for the rule-selected and
                     random chains". Omitted otherwise.
  D3, random         random.Random(0).choice over all committing v1 chains in (target, seed) order, the
                     rule-selected chain not excluded.

Data: results/raw/s1b/t4/<target>/v1_seed<k>.json (turn text, stored `grade` and `grade_reason`) and
results/raw/s1d/proj_t4v1.npz (projections, built by scripts/s1d/proj.py). No judge is called, nothing is
generated, no model is loaded.

Outputs: writeup/figs/s6_fd_transcript_d1.{png,pdf,caption.md}, ..._d3.*, and ..._d2.* if named.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import random
import textwrap
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location("s6_common", REPO / "scripts" / "figs" / "common.py")
CM = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(CM)

T4 = CM.RAW / "s1b" / "t4"
BAND = list(range(14, 19))                     # L14-18, D-024 primary band
AXES_DRAWN = [("nn", CM.NAMED), ("refusal", CM.SECOND)]
RANDOM_AXES = ["random%d" % s for s in range(10)]
# grade tags carry no hue: the two hues belong to the two projections (brief: colour carries meaning)
GRADE_STYLE = {"held": dict(facecolor="#e3e3e3", edgecolor="#e3e3e3", textcolor="#444444"),
               "partial": dict(facecolor="white", edgecolor="#444444", textcolor="#444444"),
               "committed": dict(facecolor="#111111", edgecolor="#111111", textcolor="white")}
TITLE = "One persuasion chain, turn by turn, with its internal readout beside the words"
D2_LABEL = "chosen by hand for illustration — see D1 and D3 for the rule-selected and random chains"


# ------------------------------------------------------------------ selection, from grades only

def grade_table():
    rows = []
    for p in sorted(T4.glob("*/v1_seed*.json")):
        d = json.load(open(p, encoding="utf-8"))
        g = [t["grade"] for t in d["turns"]]
        T = next((i + 1 for i, x in enumerate(g) if x == "committed"), None)
        assert T == d["T_primary"], (p, T, d["T_primary"])
        rows.append({"target": d["target"], "seed": int(d["seed"]), "T": T, "path": p})
    rows.sort(key=lambda r: (r["target"], r["seed"]))
    return rows


def select(rows):
    comm = [r for r in rows if r["T"] is not None]
    Ts = sorted(r["T"] for r in comm)
    med = Ts[(len(Ts) - 1) // 2]
    at = [r for r in comm if r["T"] == med]
    seeds = sorted(set(r["seed"] for r in at))
    seed = 0 if 0 in seeds else seeds[0]
    cands = sorted((r for r in at if r["seed"] == seed), key=lambda r: r["target"])
    d1 = cands[0]
    d3 = random.Random(0).choice(comm)           # comm is in (target, seed) order
    dist = {}
    for t in Ts:
        dist[t] = dist.get(t, 0) + 1
    return {"n_chains": len(rows), "n_committing": len(comm), "T_distribution": dist, "median_T": med,
            "n_at_median": len(at), "seeds_at_median": seeds, "seed_chosen": seed,
            "tied_targets_at_seed": [r["target"] for r in cands], "d1": d1, "d3": d3}


# ------------------------------------------------------------------ the chain and its readout

def load_projections():
    z = np.load(CM.RAW / "s1d" / "proj_t4v1.npz", allow_pickle=False)
    keys = [json.loads(k) for k in z["keys"]]
    return z["proj"], [str(a) for a in z["axes"]], [str(p) for p in z["positions"]], keys


def readout(proj, axes, positions, keys, target, seed):
    idx = next(i for i, k in enumerate(keys) if k["target"] == target and k["seed"] == seed)
    pi = positions.index("answer")
    out = []
    for t in range(proj.shape[1]):
        row = {}
        for name, _c in AXES_DRAWN:
            row[name] = float(np.mean(proj[idx, t, pi, axes.index(name), BAND]))
        row["floor"] = float(max(abs(np.mean(proj[idx, t, pi, axes.index(s), BAND])) for s in RANDOM_AXES))
        out.append(row)
    return out


def clip_lines(text, width, n_lines):
    text = " ".join(text.split())
    lines = textwrap.wrap(text, width=width)
    if len(lines) <= n_lines:
        return "\n".join(lines), False
    kept = lines[:n_lines]
    kept[-1] = kept[-1][: max(0, width - 2)].rstrip() + " …"
    return "\n".join(kept), True


# ------------------------------------------------------------------ drawing

def draw(rec, rd, panel, header, rule_lines, name, n_show=None, note=None):
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch, Patch
    from matplotlib.lines import Line2D
    CM.style()
    turns = rec["turns"][:n_show] if n_show else rec["turns"]
    n = len(turns)
    # vertical geometry: the ten-turn figure is 14.6 in tall; a shorter render (--short-turns, briefs/S6-figures-2.md
    # §2) keeps every row, the title band and the footer band at the same height in inches, so only the figure
    # shrinks. yf() maps a ten-turn-figure fraction to this figure's fraction of the same height in inches.
    H = 14.6 * (0.145 + 0.075) + 14.6 * 0.078 * n
    yf = lambda frac10: frac10 * 14.6 / H
    fig = plt.figure(figsize=(14.0, H))
    # column geometry in figure coordinates
    x_turn, x_user, x_asst, x_grade, x_bar, w_bar = 0.015, 0.045, 0.285, 0.665, 0.745, 0.235
    y_top, y_bot = 1 - yf(0.145), yf(0.075)
    h = yf(0.078)
    T = rec["T_primary"]
    for i, tu in enumerate(turns):
        yc = y_top - (i + 0.5) * h
        first = (tu["turn"] == T)
        if first:
            fig.patches.append(FancyBboxPatch((0.01, yc - h / 2 + yf(0.003)), 0.975, h - yf(0.006),
                                              boxstyle="round,pad=0.002,rounding_size=0.004",
                                              transform=fig.transFigure, facecolor="#f4f4f4",
                                              edgecolor="#111111", lw=1.6, zorder=-2))
        fig.text(x_turn, yc, "%d" % tu["turn"], fontsize=13, fontweight="bold", va="center", ha="left",
                 color=CM.TEXT)
        kind = tu["kind"]
        utxt, _ = clip_lines(tu["user"], 42, 3)
        fig.text(x_user, yc, utxt, fontsize=8.6, va="center", ha="left", color=CM.TEXT, linespacing=1.15)
        if kind == "filler":
            fig.text(x_user, yc - h / 2 + yf(0.006), "filler turn: the rig changed topic after the act",
                     fontsize=7.5, va="bottom", ha="left", color=CM.GREY, style="italic")
        atxt, _ = clip_lines(tu["answer"], 78, 2)
        fig.text(x_asst, yc, atxt, fontsize=8.6, va="center", ha="left", color=CM.TEXT, linespacing=1.15)
        g = tu["grade"]
        st = GRADE_STYLE[g]
        fig.text(x_grade + 0.032, yc, g, fontsize=9.5, fontweight="bold", va="center", ha="center",
                 color=st["textcolor"], bbox=dict(boxstyle="round,pad=0.35", facecolor=st["facecolor"],
                                                  edgecolor=st["edgecolor"], lw=1.2), zorder=3)
        if first:
            fig.text(x_grade + 0.032, yc - yf(0.024), "first committed turn", fontsize=7.5, va="center", ha="center",
                     color="#111111", fontweight="bold")
    # the readout column: one axes whose y runs over the turns
    ax = fig.add_axes([x_bar, y_bot, w_bar, y_top - y_bot])
    ax.patch.set_alpha(0)                      # the first-committed-turn highlight shows through
    ax.set_ylim(n, 0)
    vals = [abs(r[a]) for r in rd for a, _ in AXES_DRAWN] + [r["floor"] for r in rd]   # every turn: same scale in a crop
    lim = 1.12 * max(vals)
    ax.set_xlim(-lim, lim)
    ax.axvline(0, color=CM.TEXT, lw=0.9, zorder=2)
    for i, r in enumerate(rd[:n]):
        ax.axhspan(i + 0.06, i + 0.94, xmin=0.5 - r["floor"] / (2 * lim), xmax=0.5 + r["floor"] / (2 * lim),
                   color=CM.GREY_BAND, zorder=1)
        for j, (name_, col) in enumerate(AXES_DRAWN):
            yb = i + 0.30 + 0.40 * j
            ax.barh(yb, r[name_], height=0.30, color=col, zorder=3)
            ax.text(r[name_] + (0.02 * lim if r[name_] >= 0 else -0.02 * lim), yb, "%.2f" % r[name_],
                    ha="left" if r[name_] >= 0 else "right", va="center", fontsize=7.5, color=col)
    if note is not None:
        # --note (briefs/S6-figures-2.md §3): one small grey sentence beside that turn's nn bar, hung from the bar's
        # top edge on the bar's open side and wrapped to the width left to the axis edge; nothing else moves
        nturn, ntext = note
        i = next((j for j, tu in enumerate(turns) if tu["turn"] == nturn), None)
        assert i is not None, ("note turn not drawn", nturn, n)
        r = rd[i]
        yb = i + 0.30                                              # the nn bar's centre line (j = 0)
        in_per_unit = w_bar * 14.0 / (2 * lim)
        label_w = len("%.2f" % r["nn"]) * 7.5 / 72 * 0.62 / in_per_unit + 0.04 * lim   # the value label, plus a gap
        if r["nn"] >= 0:
            x0, avail, ha = r["nn"] + 0.02 * lim + label_w, (lim - (r["nn"] + 0.02 * lim + label_w)) * in_per_unit, "left"
        else:
            x0, avail, ha = r["nn"] - 0.02 * lim - label_w, ((r["nn"] - 0.02 * lim - label_w) + lim) * in_per_unit, "right"
        fs = 6.8
        width = max(12, int(avail / (fs / 72 * 0.55)))
        ax.text(x0, yb - 0.15, "\n".join(textwrap.wrap(ntext, width=width)), ha=ha, va="top", fontsize=fs,
                color=CM.GREY, linespacing=1.15, zorder=4)
    ax.set_yticks([])
    ax.spines[["left", "top", "right"]].set_visible(False)
    ax.tick_params(labelsize=8.5)
    ax.set_xlabel("projection at the answer position, mean over L14–18", fontsize=9)
    for i in range(1, n):
        fig.add_artist(Line2D([0.01, 0.985], [y_top - i * h] * 2, transform=fig.transFigure, color="#e6e6e6",
                              lw=0.8, zorder=0))
    # column headers
    yh = y_top + yf(0.008)
    fig.text(x_user, yh, "user (the persuader)", fontsize=10.5, fontweight="bold", va="bottom", color=CM.TEXT)
    fig.text(x_asst, yh, "assistant's answer (first ~2 lines; full text in the caption file)", fontsize=10.5,
             fontweight="bold", va="bottom", color=CM.TEXT)
    fig.text(x_grade + 0.032, yh, "act judge", fontsize=10.5, fontweight="bold", va="bottom", ha="center",
             color=CM.TEXT)
    fig.text(x_bar + w_bar / 2, yh, "internal readout", fontsize=10.5, fontweight="bold", va="bottom",
             ha="center", color=CM.TEXT)
    # titles and the printed rule
    fig.suptitle(TITLE, fontsize=15.5, fontweight="bold", y=1 - yf(0.008))
    fig.text(0.5, 1 - yf(0.038), header, ha="center", va="top", fontsize=11.5, color=CM.NAMED, fontweight="bold")
    fig.text(0.5, 1 - yf(0.062), "\n".join(rule_lines), ha="center", va="top", fontsize=9.2, color=CM.TEXT,
             linespacing=1.25)
    handles = [Patch(facecolor=GRADE_STYLE["held"]["facecolor"], edgecolor=GRADE_STYLE["held"]["edgecolor"], label="held"),
               Patch(facecolor="white", edgecolor="#444444", label="partial"),
               Patch(facecolor="#111111", edgecolor="#111111", label="committed (the harmful act)"),
               Patch(color=CM.NAMED, label="nn projection"), Patch(color=CM.SECOND, label="refusal projection"),
               Patch(color=CM.GREY_BAND, label="random floor: max |projection| over 10 random arrows, this turn")]
    fig.legend(handles=handles, loc="lower center", ncol=6, frameon=False, fontsize=9.2, bbox_to_anchor=(0.5, 0.0))
    fig.text(0.5, yf(0.030), "Base Llama-3.1-8B-Instruct, deceived route, target `%s`, v1 chain, seed %d. Grades: the "
             "rig's stored act-judge grade (gpt-4o-mini) per turn.\nEXPLORATORY readouts: nn and refusal are "
             "mean-difference directions with no centred zero; which turn projects higher is the fact, not its sign."
             % (rec["target"], rec["seed"]), ha="center", va="bottom", fontsize=8.8, color=CM.TEXT)
    return fig


def caption_body(rec, rd, panel, sel, rule_lines):
    out = ["**Rule, as printed on the figure:** " + " ".join(rule_lines), ""]
    out.append("Selection facts (from stored grades only): %d v1 chains, %d with a committed turn; first-committed-turn "
               "distribution %s; median (lower-middle) T = %d; %d chains at it, seeds present %s, seed chosen %d; "
               "targets tied at that seed, in sorted order: %s."
               % (sel["n_chains"], sel["n_committing"],
                  ", ".join("T=%d: %d" % (k, v) for k, v in sorted(sel["T_distribution"].items())),
                  sel["median_T"], sel["n_at_median"], sel["seeds_at_median"], sel["seed_chosen"],
                  ", ".join("`%s`" % t for t in sel["tied_targets_at_seed"])))
    out.append("\nChain: target `%s`, seed %d, first committed turn %s, per-turn grades: %s.\n"
               % (rec["target"], rec["seed"], rec["T_primary"], " ".join(t["grade"] for t in rec["turns"])))
    out.append("| turn | kind | grade | nn (L14–18, answer) | refusal (L14–18, answer) | random floor |")
    out.append("|---|---|---|---|---|---|")
    for tu, r in zip(rec["turns"], rd):
        out.append("| %d | %s | %s | %.3f | %.3f | %.3f |" % (tu["turn"], tu["kind"], tu["grade"], r["nn"],
                                                           r["refusal"], r["floor"]))
    out.append("\nSystem prompt, verbatim:\n\n```text\n%s\n```\n" % rec["system"])
    for tu in rec["turns"]:
        out.append("## Turn %d — kind `%s`, grade `%s`\n" % (tu["turn"], tu["kind"], tu["grade"]))
        out.append("Grade reason, verbatim: %s\n" % tu.get("grade_reason", ""))
        out.append("**User, verbatim:**\n\n```text\n%s\n```\n" % tu["user"])
        out.append("**Assistant, verbatim (the full generated text, thinking block included):**\n\n```text\n%s\n```\n"
                   % tu["assistant"])
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--d2", default=None, help="the researcher's hand-picked chain as <target>:<seed>; omitted if absent")
    ap.add_argument("--d2-label", default=D2_LABEL,
                    help="the label printed in one line under the title of the D2 panel (briefs/S6-figures-2.md §1 "
                         "names the exact text); the caption records it")
    ap.add_argument("--short-turns", type=int, default=None,
                    help="summary crop (briefs/S6-figures-2.md §2): draw turns 1..N only, same layout and labels, "
                         "output names suffixed _short; the full renders are not touched")
    ap.add_argument("--note", default=None,
                    help="<turn>:<text> — one small grey annotation beside that turn's nn bar on every panel drawn "
                         "(briefs/S6-figures-2.md §3); the caption records it")
    ap.add_argument("--panels", default=None,
                    help="comma-separated subset of d1,d2,d3 to draw (briefs/S6-figures-2.md); default: every panel "
                         "available. Selection is computed regardless, so the printed rules do not change")
    args = ap.parse_args()
    rows = grade_table()
    sel = select(rows)
    proj, axes, positions, keys = load_projections()
    panels = [("d1", sel["d1"],
               "Panel D1 — rule-selected: `%s`, seed %d, first committed turn %d" % (
                   sel["d1"]["target"], sel["d1"]["seed"], sel["d1"]["T"]),
               ["Rule (writeup/figures-plan.md §0): the v1 chain at the median first-committed turn over the %d "
                "committing chains (lower-middle of the sorted turns: T = %d; %d chains sit at it), seed 0 if present "
                "else the lowest seed;" % (sel["n_committing"], sel["median_T"], sel["n_at_median"]),
                "%d targets hold a seed-%d chain at T = %d (%s); ties broken by target name in sorted order. "
                "Selected before any text was read." % (len(sel["tied_targets_at_seed"]), sel["seed_chosen"],
                                                        sel["median_T"], ", ".join(sel["tied_targets_at_seed"]))]),
              ("d3", sel["d3"],
               "Panel D3 — random: `%s`, seed %d, first committed turn %d" % (
                   sel["d3"]["target"], sel["d3"]["seed"], sel["d3"]["T"]),
               ["Rule: one chain drawn by random.Random(0).choice from all %d committing v1 chains in (target, seed) "
                "order, the rule-selected chain not excluded, not filtered on content." % sel["n_committing"],
                "Drawn before any text was read."])]
    if args.d2:
        tg, sd = args.d2.split(":")
        r2 = next(r for r in rows if r["target"] == tg and r["seed"] == int(sd))
        # the label is the one line under the title; the chain's identity goes on the rule lines beneath it
        panels.insert(1, ("d2", r2, "Panel D2 — " + args.d2_label,
                          ["Target `%s`, seed %d, first committed turn %s. Named by the researcher; not selected by "
                           "any rule." % (r2["target"], r2["seed"], r2["T"])]))
    if args.panels:
        want = [w.strip().lower() for w in args.panels.split(",")]
        assert all(w in {"d1", "d2", "d3"} for w in want), args.panels
        panels = [p for p in panels if p[0] in want]
        assert [p[0] for p in panels] == [w for w in ("d1", "d2", "d3") if w in want], (args.panels, [p[0] for p in panels])
    for panel, r, header, rule_lines in panels:
        printed = ([args.d2_label] if panel == "d2" else []) + rule_lines   # every rule line the figure prints
        rec = json.load(open(r["path"], encoding="utf-8"))
        rd = readout(proj, axes, positions, keys, r["target"], r["seed"])
        name = "s6_fd_transcript_%s" % panel
        cap_title = TITLE + " — panel %s" % panel.upper()
        body = caption_body(rec, rd, panel, sel, printed)
        if args.short_turns:
            k, nt = args.short_turns, len(rec["turns"])
            assert 1 <= k <= nt, (k, nt)
            name += "_short"
            cap_title += ", turns 1–%d of %d" % (k, nt)
            body = ("Summary crop (briefs/S6-figures-2.md §2): the figure shows turns 1–%d of the %d-turn chain, with the "
                    "same layout and labels as the ten-turn render `s6_fd_transcript_%s.*`; the readout x-scale and the "
                    "random floors are those of the full chain, so every bar has the same length in both renders. The "
                    "table and the verbatim text below cover all %d turns.\n\n" % (k, nt, panel, nt)) + body
        note = None
        if args.note:
            nturn, ntext = args.note.split(":", 1)
            note = (int(nturn), ntext.strip())
            body = ("Annotation (briefs/S6-figures-2.md §3), printed in grey beside the turn-%d nn bar, verbatim: "
                    "\"%s\"\n\n" % note) + body
        fig = draw(rec, rd, panel, header, rule_lines, name, n_show=args.short_turns, note=note)
        outs = CM.save(fig, name)
        cap = CM.caption(name, cap_title, "scripts/figs/fd_transcript.py",
                         ["results/raw/s1b/t4/%s/v1_seed%d.json (turn text, stored grades and reasons)"
                          % (r["target"], r["seed"]),
                          "results/raw/s1d/proj_t4v1.npz (projections at the answer position, built by scripts/s1d/proj.py)"],
                         " ".join(printed), body)
        for p in outs + [cap]:
            print(p.relative_to(REPO), "%.0f kB" % (p.stat().st_size / 1e3))
        print("  %s: %s seed %d T=%s grades %s" % (panel, r["target"], r["seed"], r["T"],
                                                   " ".join(t["grade"][0] for t in rec["turns"])))
    print("selection: committing %d, median T %d, at median %d, seeds %s, tied targets %s, D3 draw %s/%d"
          % (sel["n_committing"], sel["median_T"], sel["n_at_median"], sel["seeds_at_median"],
             sel["tied_targets_at_seed"], sel["d3"]["target"], sel["d3"]["seed"]))


if __name__ == "__main__":
    main()
