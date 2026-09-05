"""F-G — the random-example panel as an image (briefs/S6-figures-2.md §5).

A text panel rendered as a figure, two columns, from writeup/examples/h3.md Panel 1 (deceived / fork A /
act-focused). Left: the rule-selected reply (median answer length) with the probe question above it and the
judge's label and reason below. Right: random 1 of 3 from the same panel. The selection rule is printed at the
top in one line, verbatim from line 1 of the file, then the file's random-draw line. The model's text is set in
monospace, wrapped, nothing truncated: the script asserts that every rendered block equals the source block after
whitespace normalisation.

Nothing is selected here: the file's rule-selected reply and its first random draw are taken as the file
records them. No judge is called, nothing is generated, no model is loaded.

Output: writeup/figs/s6_fg_example.{png,pdf,caption.md}.
"""
from __future__ import annotations

import importlib.util
import re
import textwrap
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location("s6_common", REPO / "scripts" / "figs" / "common.py")
CM = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(CM)

SRC = REPO / "writeup" / "examples" / "h3.md"
PANEL_HEADING = "## Panel 1"
RANDOM_LINE = "random example: seed 0, drawn once, not filtered"      # the brief's text, printed as ordered
FILE_RANDOM_PREFIX = "random examples: seed 0,"                          # the file's own line 3 begins so
TITLE = "What the model says when asked whether its answer was sound: one rule-selected reply and one random reply"
MONO_W = 68        # characters per wrapped monospace line, per column (68 × 10.6 pt × 0.602 em ≈ 6.1 in of a 7.1 in column)
FS_MONO = 10.6     # the model's text
LS = 1.25          # line spacing everywhere


# ------------------------------------------------------------------ parse the machine-written file

def fenced_blocks(text):
    """Every ```text fenced block in order, content verbatim (the fence lines excluded)."""
    return [m.group(1).rstrip("\n") for m in re.finditer(r"```text\n(.*?)\n```", text, flags=re.S)]


def parse():
    lines = SRC.read_text(encoding="utf-8").splitlines()
    rule_line = lines[0].strip()
    assert rule_line.startswith("Rule ("), rule_line
    file_random_line = lines[2].strip()
    assert file_random_line.startswith(FILE_RANDOM_PREFIX) and "not filtered" in file_random_line, file_random_line
    text = "\n".join(lines)
    i0 = text.index(PANEL_HEADING)
    i1 = text.index("\n## ", i0 + 1)                       # the next panel heading
    panel = text[i0:i1]
    cell_line = next(l for l in panel.splitlines() if l.startswith("Cell:"))
    blocks = fenced_blocks(panel)
    # panel layout in the file: probe question, then per example: reply, label+reason (rule-selected, then 3 random)
    assert len(blocks) == 1 + 2 * 4, len(blocks)
    probe = blocks[0]
    heads = re.findall(r"^### (.+)$", panel, flags=re.M)
    assert len(heads) == 4 and heads[0].startswith("Rule-selected") and heads[1].startswith("Random 1 of 3"), heads
    meta = re.findall(r"^answer length (\d+) characters; reflection judge: (.+)$", panel, flags=re.M)
    assert len(meta) == 4, meta
    ex = []
    for k in range(2):                                      # the rule-selected one and random 1 of 3
        reply, lab = blocks[1 + 2 * k], blocks[2 + 2 * k]
        m = re.fullmatch(r"label: (.+)\nreason: (.+)", lab, flags=re.S)
        assert m, lab
        ident = re.search(r"`([a-z0-9-]+)` / seed (\d+)$", heads[k])
        assert ident, heads[k]
        assert len(reply) == int(meta[k][0]), (len(reply), meta[k][0], heads[k])   # the file's own length check
        ex.append({"head": heads[k], "target": ident.group(1), "seed": int(ident.group(2)), "reply": reply,
                   "label": m.group(1).strip(), "reason": m.group(2).strip(), "length": int(meta[k][0]),
                   "judge": meta[k][1]})
    return rule_line, file_random_line, cell_line, probe, ex


# ------------------------------------------------------------------ render

def wrap_keep_paragraphs(s, width):
    """Wrap each paragraph of `s`; blank lines between paragraphs kept; returns the lines."""
    out = []
    for para in s.split("\n"):
        if not para.strip():
            out.append("")
        else:
            out += textwrap.wrap(para, width=width, break_long_words=False, break_on_hyphens=False) or [""]
    return out


def same_text(a, b):
    return " ".join(a.split()) == " ".join(b.split())


def draw(rule_line, cell_line, probe, ex):
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch
    CM.style()
    W = 16.0
    probe_lines = wrap_keep_paragraphs(probe, 100)
    cols = []
    for e in ex:
        reply_lines = wrap_keep_paragraphs(e["reply"], MONO_W)
        assert same_text("\n".join(reply_lines), e["reply"]), e["head"]           # nothing truncated
        reason_lines = wrap_keep_paragraphs("label: %s" % e["label"], 95) + \
                       wrap_keep_paragraphs("reason: %s" % e["reason"], 95)
        assert same_text("\n".join(reason_lines), "label: %s reason: %s" % (e["label"], e["reason"])), e["head"]
        cols.append((reply_lines, reason_lines))
    n_reply = max(len(c[0]) for c in cols)
    n_reason = max(len(c[1]) for c in cols)
    # heights in inches, from the font sizes: title band, rule lines, probe, column header, reply box, reason, footer
    mono_in = FS_MONO * LS / 72
    reason_in = 9.2 * LS / 72
    H = 1.35 + 0.26 + len(probe_lines) * mono_in + 0.45 + 0.6 + (n_reply + 1.2) * mono_in + 0.5 + n_reason * reason_in + 0.7
    fig = plt.figure(figsize=(W, H))
    fy = lambda inches: 1 - inches / H
    # the rule, one line, verbatim; then the random-draw line
    fig.text(0.5, fy(0.12), TITLE, ha="center", va="top", fontsize=14, fontweight="bold", color=CM.TEXT)
    fs_rule = 7.0 if len(rule_line) > 270 else 7.6
    fig.text(0.5, fy(0.55), rule_line, ha="center", va="top", fontsize=fs_rule, color=CM.TEXT)
    fig.text(0.5, fy(0.78), RANDOM_LINE, ha="center", va="top", fontsize=9.2, color=CM.GREY, style="italic")
    fig.text(0.5, fy(1.02), cell_line, ha="center", va="top", fontsize=8.6, color=CM.TEXT)
    # the probe, once, across both columns
    y = 1.35
    fig.text(0.04, fy(y), "The probe put to the subject after the chain (fork A), verbatim:", ha="left", va="top",
             fontsize=9.5, fontweight="bold", color=CM.TEXT)
    y += 0.26
    fig.text(0.06, fy(y), "\n".join(probe_lines), ha="left", va="top", fontsize=FS_MONO, family="DejaVu Sans Mono",
             color=CM.TEXT, linespacing=LS)
    y += len(probe_lines) * mono_in + 0.45
    y_head = y
    x0s = [0.04, 0.53]
    heads = ["Rule-selected: median answer length in the cell", "Random 1 of 3: seed 0, drawn once, not filtered"]
    colours = [CM.NAMED, CM.GREY]
    for (e, (reply_lines, reason_lines), x0, head, col) in zip(ex, cols, x0s, heads, colours):
        fig.text(x0, fy(y_head), head, ha="left", va="top", fontsize=11, fontweight="bold", color=col)
        fig.text(x0, fy(y_head + 0.22), "`%s` / seed %d — answer length %d characters; reflection judge: %s"
                 % (e["target"], e["seed"], e["length"], e["judge"]), ha="left", va="top", fontsize=8.6, color=CM.TEXT)
        yb = y_head + 0.6
        box_h = (n_reply + 1.2) * mono_in
        fig.patches.append(FancyBboxPatch((x0 - 0.008, fy(yb + box_h)), 0.445, box_h / H,
                                          boxstyle="round,pad=0.004,rounding_size=0.006", transform=fig.transFigure,
                                          facecolor="#f7f7f7", edgecolor=col, lw=1.4, zorder=-1))
        fig.text(x0 + 0.004, fy(yb + 0.6 * mono_in), "\n".join(reply_lines), ha="left", va="top", fontsize=FS_MONO,
                 family="DejaVu Sans Mono", color=CM.TEXT, linespacing=LS)
        yr = yb + box_h + 0.18
        fig.text(x0, fy(yr), "The reflection judge's label and reason, verbatim:", ha="left", va="top", fontsize=9,
                 fontweight="bold", color=CM.TEXT)
        fig.text(x0, fy(yr + 0.24), "\n".join(reason_lines), ha="left", va="top", fontsize=9.2, color=CM.TEXT,
                 linespacing=LS)
    fig.text(0.5, 0.012, "Base Llama-3.1-8B-Instruct, deceived route, v1 chains, fork A probe. The model's text in "
             "monospace is verbatim and complete; the file's median and random draw are taken as written, nothing "
             "re-selected here.", ha="center", va="bottom", fontsize=8.6, color=CM.TEXT)
    return fig


def main():
    rule_line, file_random_line, cell_line, probe, ex = parse()
    fig = draw(rule_line, cell_line, probe, ex)
    name = "s6_fg_example"
    outs = CM.save(fig, name)
    body = ["**Printed at the top, verbatim from line 1 of the source file:** %s" % rule_line, "",
            "**Printed beneath it (the brief's wording):** %s" % RANDOM_LINE, "",
            "The source file's own line 3, verbatim: %s" % file_random_line, "",
            "%s" % cell_line, "",
            "Probe (fork A), verbatim:", "", "```text", probe, "```", ""]
    for e, side in zip(ex, ("Left column — rule-selected", "Right column — random 1 of 3 (seed 0)")):
        body += ["## %s — %s" % (side, e["head"]), "",
                 "answer length %d characters; reflection judge: %s" % (e["length"], e["judge"]), "",
                 "**The subject's reply, verbatim (rendered complete on the figure):**", "", "```text", e["reply"],
                 "```", "", "**The reflection judge's label and reason line, verbatim:**", "", "```text",
                 "label: %s" % e["label"], "reason: %s" % e["reason"], "```", ""]
    body += ["Both replies and both label/reason lines are asserted equal to the source blocks after whitespace "
             "normalisation before the figure is saved; nothing is truncated."]
    cap = CM.caption(name, TITLE, "scripts/figs/fg_example.py",
                     ["writeup/examples/h3.md, Panel 1 (deceived / fork A / act-focused), written by scripts/verify/examples.py"],
                     rule_line + " " + RANDOM_LINE, "\n".join(body))
    for p in outs + [cap]:
        print(p.relative_to(REPO), "%.0f kB" % (p.stat().st_size / 1e3))
    for e in ex:
        print("  %s seed %d len %d label %s" % (e["target"], e["seed"], e["length"], e["label"]))


if __name__ == "__main__":
    main()
