"""F-B — "Every intervention moved the readout; none moved the behaviour" (briefs/S6-figures.md).

The dissociation as a paired dumbbell chart. Rows, the four interventions:
  1. self-blame text vs act-blame            (S4 cells B vs A)
  2. guilt-arrow steering                    (S4 cell C vs B; D the norm-matched random arm)
  3. persona-axis steering                   (S5c Q1 vs Q0; Q3 the norm-matched random arm)
  4. anti-susceptibility steering            (S5b P1 vs P0; P2 the norm-matched random arm)
Left column, the readout moved: comparison -> intervention, with the CI where the table prints one, the
random floor as a shaded band and the norm-matched random arm as a grey marker on the same row. Right
column, the behaviour moved: the paired difference (spread rate at distance 0 for rows 1-3, act rate for
row 4) with its clustered-bootstrap CI and the random arm's difference in grey. Coherence per arm at the
row end where the tables carry it.

Every number is read from results/S4_table.md, results/S5b_table.md and results/S5c_table.md by
scripts/figs/common.py's table reader - nothing is typed in - and asserted against the values the reports
quote. The four readouts are different quantities in different units (persona displacement vs the topic
baseline; a guilt_clean band shift; a persona projection; an nn projection), so the left column is four
axes, one per row, each in its own units; the right column shares one Δ-rate axis.

Outputs: writeup/figs/s6_fb_dissociation.{png,pdf,caption.md}. CPU only, no cost.
"""
from __future__ import annotations

import importlib.util
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location("s6_common", REPO / "scripts" / "figs" / "common.py")
CM = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(CM)

NAME = "s6_fb_dissociation"
TITLE = "Every intervention moved the readout; none moved the behaviour"


def ci_pair(cell):
    s = cell.replace("−", "-").replace("**", "")
    m = re.search(r"\[\s*(%s)\s*,\s*(%s)\s*\]" % (CM._NUM, CM._NUM), s)
    if not m:
        return None
    return float(m.group(1)), float(m.group(2))


def close(a, b, tol=1e-6):
    assert abs(a - b) < tol, (a, b)


def gather():
    T4 = CM.read_md_tables(CM.RESULTS / "S4_table.md")
    T5b = CM.read_md_tables(CM.RESULTS / "S5b_table.md")
    T5c = CM.read_md_tables(CM.RESULTS / "S5c_table.md")
    rows, src = [], {}

    # ---------------------------------------------------------------- row 1: self-blame text vs act-blame
    f1 = CM.find_table(T4, "F.1 Displacement per cell")
    a = CM.row(f1, "`act_blame`", "primary")
    b = CM.row(f1, "`self_blame`", "primary")
    b1 = CM.find_table(T4, "B.1 Contrasts")
    ab = CM.row(b1, "A − B")
    dv, dlo, dhi = CM.num_ci(ab["Δ spread rate d0"])
    close(CM.num(b["persona displacement"]), 0.056); close(CM.num(a["persona displacement"]), 0.038)
    close(dv, 0.014)
    rows.append({
        "label": "self-blame text\n(vs act-blame)", "cells": "S4 cells B vs A",
        "left": {"xlabel": "persona displacement at the unrelated forks vs the topic baseline, L14–18",
                 "comp": ("act-blame", CM.num(a["persona displacement"]), ci_pair(a["95% CI (clustered on the run)"])),
                 "interv": ("self-blame", CM.num(b["persona displacement"]), ci_pair(b["95% CI (clustered on the run)"])),
                 "rand": None, "floor": CM.num(b["random floor"]),
                 "note": "no random text arm exists; the band is the random floor of the displacement"},
        "right": {"what": "Δ spread rate, distance 0", "interv": (-dv, -dhi, -dlo),
                  "interv_note": "CI degenerate, one target", "rand": None,
                  "rand_note": "no random text arm"},
        "coherence": "coherence: not in S4_table.md"})
    src["row1"] = ["results/S4_table.md §F.1 (persona displacement, primary band, `act_blame` and `self_blame`; "
                   "random floor 0.027)", "results/S4_table.md §B.1 (A − B, Δ spread rate d0; sign flipped to "
                   "self − act)"]

    # ---------------------------------------------------------------- row 2: guilt-arrow steering C vs B, D random
    e = CM.find_table(T4, "unrelated forks vs the topic baseline", "`guilt_clean`")
    eb = CM.row(e, "`self_blame`", "primary")
    ec = CM.row(e, "steer_guilt_clean", "primary")
    ed = CM.row(e, "steer_random0", "primary")
    cb = CM.row(b1, "steer_guilt_clean_L16_c4 − B")
    db = CM.row(b1, "steer_random0_L16_c4 − B")
    st = CM.find_table(T4, "**Steering.**", "injected norm")
    norm_c = CM.num(CM.row(st, "steer_guilt_clean")["injected norm"])
    close(CM.num(ec["`guilt_clean`"]), 1.874); close(CM.num(eb["`guilt_clean`"]), 0.041); close(norm_c, 2.9205)
    cv, clo, chi = CM.num_ci(cb["Δ spread rate d0"]); dv2, dlo2, dhi2 = CM.num_ci(db["Δ spread rate d0"])
    rows.append({
        "label": "guilt-arrow steering\n+4·σ(guilt_clean) at L16", "cells": "S4 cell C vs B; D random",
        "left": {"xlabel": "guilt_clean shift at the unrelated forks vs the topic baseline, L14–18",
                 "comp": ("self-blame, unsteered (B)", CM.num(eb["`guilt_clean`"]), None),
                 "interv": ("+ guilt arrow (C)", CM.num(ec["`guilt_clean`"]), None),
                 "rand": ("+ random, norm-matched (D)", CM.num(ed["`guilt_clean`"])), "floor": CM.num(eb["random floor"]),
                 "note": "injected norm %.2f in both C and D; C's readout is the injection, not a state change; "
                         "no CI in table" % norm_c},
        "right": {"what": "Δ spread rate, distance 0", "interv": (cv, clo, chi), "interv_note": "",
                  "rand": (dv2, dlo2, dhi2), "rand_note": ""},
        "coherence": "coherence: not in S4_table.md"})
    src["row2"] = ["results/S4_table.md §E, unrelated forks vs the topic baseline, primary band, `guilt_clean` for "
                   "`self_blame`, `self_blame+steer_guilt_clean_L16_c4`, `self_blame+steer_random0_L16_c4`; random "
                   "floor of `self_blame`", "results/S4_table.md §B.1 (C − B and D − B, Δ spread rate d0)",
                   "results/S4_table.md header, Steering table (injected norm)"]

    # ---------------------------------------------------------------- row 3: persona-axis steering Q1 vs Q0, Q3 random
    ans5c = CM.find_table(T5c, "### `answer`", "`persona`")
    q0 = CM.row(ans5c, "Q0", "primary"); q1 = CM.row(ans5c, "Q1", "primary"); q3 = CM.row(ans5c, "Q3", "primary")
    c5c = CM.find_table(T5c, "C. The contrasts the brief reads")
    q10 = CM.row(c5c, "Q1 − Q0"); q30 = CM.row(c5c, "Q3 − Q0")
    b5c = CM.find_table(T5c, "B. Act rate, spread, hold, coherence")
    coh = {k: CM.num(CM.row(b5c, k)["coherence mean"]) for k in ("Q0", "Q1", "Q3")}
    a5c = CM.find_table(T5c, "A. The arms", "injected norm")
    norm_q = CM.num(CM.row(a5c, "Q1")["**injected norm**"])
    close(CM.num(q0["`persona`"]), 1.878); close(CM.num(q1["`persona`"]), 1.018); close(norm_q, 0.887697)
    v10 = CM.num_ci(q10["Δ spread d0"]); v30 = CM.num_ci(q30["Δ spread d0"])
    pct = 100.0 * (CM.num(q1["`persona`"]) - CM.num(q0["`persona`"])) / CM.num(q0["`persona`"])
    rows.append({
        "label": "persona-axis steering\n−4·σ(persona) at L16", "cells": "S5c Q1 vs Q0; Q3 random",
        "left": {"xlabel": "persona projection at the unrelated forks, answer position, L14–18",
                 "comp": ("no feedback, unsteered (Q0)", CM.num(q0["`persona`"]), None),
                 "interv": ("away from the Assistant end (Q1)", CM.num(q1["`persona`"]), None),
                 "rand": ("random, norm-matched (Q3)", CM.num(q3["`persona`"])), "floor": CM.num(q0["random floor"]),
                 "note": "%.3f → %.3f, %.0f %%; injected norm %.3f in Q1 and Q3; no CI in table" % (
                     CM.num(q0["`persona`"]), CM.num(q1["`persona`"]), pct, norm_q)},
        "right": {"what": "Δ spread rate, distance 0", "interv": v10, "interv_note": "",
                  "rand": v30, "rand_note": ""},
        "coherence": "coherence: Q1 %.1f, Q3 %.1f (Q0 %.1f)" % (coh["Q1"], coh["Q3"], coh["Q0"])})
    src["row3"] = ["results/S5c_table.md §E `answer`, primary band, `persona` for Q0, Q1, Q3; random floor of Q0",
                   "results/S5c_table.md §C (Q1 − Q0 and Q3 − Q0, Δ spread d0)",
                   "results/S5c_table.md §B (coherence mean)", "results/S5c_table.md §A (injected norm)"]

    # ---------------------------------------------------------------- row 4: anti-susceptibility steering P1 vs P0, P2 random
    ans5b = CM.find_table(T5b, "### `answer`", "`nn`")
    p0 = CM.row(ans5b, "P0", "primary"); p1 = CM.row(ans5b, "P1", "primary"); p2 = CM.row(ans5b, "P2", "primary")
    into5b = CM.find_table(T5b, "### `into`", "`nn`")
    p0i = CM.row(into5b, "P0", "primary"); p1i = CM.row(into5b, "P1", "primary")
    c5b = CM.find_table(T5b, "C. P1 against P2")
    p10 = CM.row(c5b, "P1 − P0"); p20 = CM.row(c5b, "P2 − P0")
    b5b = CM.find_table(T5b, "Coherence is the load-bearing control")
    cohb = {k: CM.num(CM.row(b5b, k)["coherence mean"]) for k in ("P0", "P1", "P2")}
    a5b = CM.find_table(T5b, "A. The arms", "injected norm")
    norm_p = CM.num(CM.row(a5b, "P1")["**injected norm**"])
    close(norm_p, 2.649875); close(CM.num(p1i["`nn`"]), -1.583); close(CM.num(p0i["`nn`"]), -0.037)
    close(CM.num(p10["Δ act rate"]), 0.0); close(CM.num(p20["Δ act rate"]), -0.125)
    rows.append({
        "label": "anti-susceptibility steering\n−4·σ(nn) at L16, every turn", "cells": "S5b P1 vs P0; P2 random",
        "left": {"xlabel": "nn projection at the act turn, answer position, L14–18",
                 "comp": ("unsteered (P0)", CM.num(p0["`nn`"]), None),
                 "interv": ("against the nn arrow (P1)", CM.num(p1["`nn`"]), None),
                 "rand": ("random, norm-matched (P2)", CM.num(p2["`nn`"])), "floor": CM.num(p0["random floor"]),
                 "note": "injected norm %.2f in P1 and P2 (`into` position: %.3f → %.3f); no CI in table" % (
                     norm_p, CM.num(p0i["`nn`"]), CM.num(p1i["`nn`"]))},
        "right": {"what": "Δ act rate", "interv": (CM.num(p10["Δ act rate"]),) + ci_pair(p10["95% CI"]),
                  "interv_note": "", "rand": (CM.num(p20["Δ act rate"]),) + ci_pair(p20["95% CI"]), "rand_note": ""},
        "coherence": "coherence: P1 %.1f, P2 %.1f (P0 %.1f)" % (cohb["P1"], cohb["P2"], cohb["P0"])})
    src["row4"] = ["results/S5b_table.md §D `answer` (and `into`), primary band, `nn` for P0, P1, P2; random floor "
                   "of P0", "results/S5b_table.md §C (P1 − P0 and P2 − P0, Δ act rate with CI)",
                   "results/S5b_table.md §B (coherence mean)", "results/S5b_table.md §A (injected norm)"]
    return rows, src


def _place(ax, x):
    """Anchor a label so it stays inside the axis: centred in the middle, left/right-aligned near the edges."""
    lo, hi = ax.get_xlim()
    f = (x - lo) / (hi - lo)
    if f < 0.28:
        return lo + 0.01 * (hi - lo), "left"
    if f > 0.72:
        return hi - 0.01 * (hi - lo), "right"
    return x, "center"


def draw(rows):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    CM.style()
    n = len(rows)
    fig, axs = plt.subplots(n, 2, figsize=(14.5, 10.4), gridspec_kw={"width_ratios": [1.25, 1.0],
                                                                     "wspace": 0.10, "hspace": 1.05})
    Y_INT, Y_COMP, Y_RAND, Y_RANDTXT, Y_NOTE = 0.42, -0.36, -0.64, -0.92, 1.16
    for i, r in enumerate(rows):
        L, R = axs[i, 0], axs[i, 1]
        # ---- left: the readout, in its own units
        lf = r["left"]
        vals = [lf["comp"][1], lf["interv"][1]] + ([lf["rand"][1]] if lf["rand"] else [])
        vals += [-lf["floor"], lf["floor"]]
        for c in (lf["comp"][2], lf["interv"][2]):
            if c:
                vals += list(c)
        lo, hi = min(vals), max(vals)
        pad = 0.20 * (hi - lo)
        L.set_xlim(lo - pad, hi + pad)
        L.set_ylim(-1.05, 1.0)
        L.axvspan(-lf["floor"], lf["floor"], color=CM.GREY_BAND, zorder=1)
        L.plot([lf["comp"][1], lf["interv"][1]], [0, 0], color=CM.NAMED, lw=2.2, zorder=2)
        for (name, v, ci), filled in ((lf["comp"], False), (lf["interv"], True)):
            if ci:
                L.plot(ci, [0, 0], color=CM.NAMED, lw=7, alpha=0.25, solid_capstyle="butt", zorder=2)
            L.scatter([v], [0], s=150, facecolor=CM.NAMED if filled else "white", edgecolor=CM.NAMED,
                      linewidths=2.2, zorder=4)
            txt = "%.3f" % v + (" [%.3f, %.3f]" % ci if ci else "") + "   " + name
            xx, ha = _place(L, v)
            L.text(xx, Y_INT if filled else Y_COMP, txt, ha=ha, va="center", fontsize=10,
                   color=CM.NAMED, fontweight="bold" if filled else "normal")
        if lf["rand"]:
            L.scatter([lf["rand"][1]], [Y_RAND], s=110, facecolor=CM.GREY, edgecolor=CM.GREY, zorder=3)
            xx, ha = _place(L, lf["rand"][1])
            L.text(xx, Y_RANDTXT, "%.3f   %s" % (lf["rand"][1], lf["rand"][0]), ha=ha,
                   va="center", fontsize=9.5, color=CM.GREY)
        L.set_yticks([])
        L.spines[["left"]].set_visible(False)
        L.tick_params(labelsize=9)
        L.set_xlabel(lf["xlabel"], fontsize=9.5, labelpad=2)
        L.text(0.0, Y_NOTE, lf["note"], transform=L.transAxes, ha="left", va="bottom", fontsize=8.5,
               color=CM.TEXT, style="italic")
        # the row label, left of the left axis
        L.text(-0.02, 0.52, r["label"], transform=L.transAxes, ha="right", va="center", fontsize=11.5,
               fontweight="bold", color=CM.TEXT)
        L.text(-0.02, 0.10, r["cells"], transform=L.transAxes, ha="right", va="center", fontsize=9, color=CM.TEXT)
        # ---- right: the behaviour, one shared Δ-rate axis
        rt = r["right"]
        R.set_xlim(-0.45, 0.45)
        R.set_ylim(-1.05, 1.0)
        R.axvline(0, color=CM.TEXT, lw=1.0, zorder=1)
        v, lo_, hi_ = rt["interv"]
        y_i, y_r = 0.30, -0.50
        if lo_ is not None and hi_ is not None:
            R.plot([lo_, hi_], [y_i, y_i], color=CM.NAMED, lw=2.0, zorder=2)
            for xx in (lo_, hi_):
                R.plot([xx, xx], [y_i - 0.13, y_i + 0.13], color=CM.NAMED, lw=1.6, zorder=2)
        R.scatter([v], [y_i], s=150, color=CM.NAMED, zorder=4)
        lab = "%+.3f" % v + (" [%+.3f, %+.3f]" % (lo_, hi_) if lo_ is not None else "")
        if rt["interv_note"]:
            lab += "  (%s)" % rt["interv_note"]
        R.text(max(v, hi_ if hi_ is not None else v) + 0.025, y_i, lab, ha="left", va="center", fontsize=10,
               color=CM.NAMED, fontweight="bold")
        if rt["rand"]:
            rv, rlo, rhi = rt["rand"]
            if rlo is not None:
                R.plot([rlo, rhi], [y_r, y_r], color=CM.GREY, lw=2.0, zorder=2)
                for xx in (rlo, rhi):
                    R.plot([xx, xx], [y_r - 0.13, y_r + 0.13], color=CM.GREY, lw=1.6, zorder=2)
            R.scatter([rv], [y_r], s=110, color=CM.GREY, zorder=3)
            R.text(max(rv, rhi if rhi is not None else rv) + 0.025, y_r,
                   "%+.3f" % rv + (" [%+.3f, %+.3f]" % (rlo, rhi) if rlo is not None else "") + "   random arm",
                   ha="left", va="center", fontsize=9.5, color=CM.GREY)
        else:
            R.text(0.025, y_r, rt["rand_note"], ha="left", va="center", fontsize=9.5, color=CM.GREY, style="italic")
        R.set_yticks([])
        R.spines[["left"]].set_visible(False)
        R.tick_params(labelsize=9)
        R.set_xlabel("%s, intervention − comparison" % rt["what"], fontsize=9.5, labelpad=2)
        R.text(1.0, Y_NOTE, r["coherence"], transform=R.transAxes, ha="right", va="bottom", fontsize=8.5,
               color=CM.TEXT)
    axs[0, 0].set_title("the readout moved", fontsize=13, loc="left", pad=30, color=CM.NAMED)
    axs[0, 1].set_title("the behaviour moved", fontsize=13, loc="left", pad=30, color=CM.NAMED)
    handles = [Line2D([], [], marker="o", color=CM.NAMED, markerfacecolor="white", markeredgewidth=2, ms=10,
                      ls="", label="comparison arm"),
               Line2D([], [], marker="o", color=CM.NAMED, ms=10, ls="", label="intervention arm"),
               Line2D([], [], marker="o", color=CM.GREY, ms=9, ls="", label="norm-matched random arm"),
               Patch(color=CM.GREY_BAND, label="random floor (randctl seeds 0–9)"),
               Line2D([], [], color=CM.NAMED, lw=2, label="95 % CI where the table prints one")]
    fig.legend(handles=handles, loc="lower center", ncol=5, frameon=False, fontsize=10,
               bbox_to_anchor=(0.5, 0.0))
    fig.suptitle(TITLE, fontsize=16, fontweight="bold", y=0.995)
    fig.text(0.5, 0.962, "Base Llama-3.1-8B-Instruct, deceived route, target burn-blister-pop, N = 8 runs per arm. "
             "Left: the internal readout each intervention targeted. Right: what the behaviour did.\n"
             "EXPLORATORY (D-023). One target: the detectable behavioural gap at this N is one run in eight "
             "(0.125 of a rate).", ha="center", va="top", fontsize=10.5, color=CM.TEXT)
    fig.subplots_adjust(top=0.865, bottom=0.085, left=0.19, right=0.985)
    return fig


def caption_body(rows, src):
    out = []
    for i, r in enumerate(rows, 1):
        lf, rt = r["left"], r["right"]
        out.append("## Row %d — %s (%s)\n" % (i, r["label"].replace("\n", " "), r["cells"]))
        out.append("- **readout**: %s. %s %.3f%s → %s %.3f%s; random floor %.3f%s. %s"
                   % (lf["xlabel"], lf["comp"][0], lf["comp"][1],
                      " [%.3f, %.3f]" % lf["comp"][2] if lf["comp"][2] else "",
                      lf["interv"][0], lf["interv"][1],
                      " [%.3f, %.3f]" % lf["interv"][2] if lf["interv"][2] else "", lf["floor"],
                      "; %s %.3f" % (lf["rand"][0], lf["rand"][1]) if lf["rand"] else "", lf["note"]))
        v, lo, hi = rt["interv"]
        out.append("- **behaviour**: %s, intervention − comparison = %+.3f%s%s.%s"
                   % (rt["what"], v, " [%+.3f, %+.3f]" % (lo, hi) if lo is not None else "",
                      " (%s)" % rt["interv_note"] if rt["interv_note"] else "",
                      (" Random arm − comparison = %+.3f%s." % (rt["rand"][0],
                       " [%+.3f, %+.3f]" % (rt["rand"][1], rt["rand"][2]) if rt["rand"][1] is not None else ""))
                      if rt["rand"] else " %s." % rt["rand_note"]))
        out.append("- %s." % r["coherence"])
        out.append("- sources: " + "; ".join(src["row%d" % i]) + ".\n")
    out.append("**What the data carries and what it does not.** Rows 2–4 are the brief's point: a large, exact "
               "injection on the left (2.92, 0.89 and 2.65 in norm) and a paired behavioural difference whose CI "
               "contains zero on the right, with the norm-matched random arm sitting beside it. Row 1's readout is "
               "small: self-blame's persona displacement (+0.056) sits about twice the random floor (0.027) and "
               "act-blame's CI contains zero, so on that row the left column is short as well as the right. Rows 2–4 "
               "carry no CI on the readout because the tables print none; row 1's behavioural CI is degenerate "
               "because the bootstrap clusters on target and one target ran. Coherence means are not in "
               "`results/S4_table.md` for cells A–D, so rows 1–2 print none; the S5b/S5c rows print theirs.")
    return "\n".join(out)


def main():
    rows, src = gather()
    fig = draw(rows)
    outs = CM.save(fig, NAME)
    cap = CM.caption(NAME, TITLE, "scripts/figs/fb_dissociation.py",
                     ["results/S4_table.md", "results/S5b_table.md", "results/S5c_table.md"],
                     "no example is selected; every arm of the four interventions is drawn. Numbers are parsed "
                     "from the machine-written tables and asserted against the values the reports quote.",
                     caption_body(rows, src))
    for p in outs + [cap]:
        print(p.relative_to(REPO), "%.0f kB" % (p.stat().st_size / 1e3))
    for r in rows:
        print("%-32s left %.3f -> %.3f (rand %s, floor %.3f) | right %+.3f  rand %s | %s"
              % (r["label"].split("\n")[0], r["left"]["comp"][1], r["left"]["interv"][1],
                 ("%.3f" % r["left"]["rand"][1]) if r["left"]["rand"] else "-", r["left"]["floor"],
                 r["right"]["interv"][0], ("%+.3f" % r["right"]["rand"][0]) if r["right"]["rand"] else "-",
                 r["coherence"]))


if __name__ == "__main__":
    main()
