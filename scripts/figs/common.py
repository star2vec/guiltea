"""Shared style and helpers for the S6 executive-summary figures (briefs/S6-figures.md).

CPU only: no generation, no model load, no judge call, no GPU, no cost. Every figure under scripts/figs/
is machine-written from results/, results/raw/ and writeup/examples/ — regenerate, never hand-edit.

Fixed encodings (brief, "Design rules for all four"): one colour for the named direction, grey for random,
a second colour only where a second condition exists; random floors as shaded bands, never lines; the
title states the message as a sentence; every number the summary quotes is printed on its figure.
"""
from __future__ import annotations

import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[2]
FIGS = REPO / "writeup" / "figs"
RESULTS = REPO / "results"
RAW = RESULTS / "raw"

NAMED = "#1f4e9c"       # the named direction / the intervention / the class that will break
SECOND = "#c8501e"      # the second condition, only where one exists
GREY = "#8a8a8a"        # the random arm
GREY_BAND = "#d2d2d2"   # the random floor, always a band
MUTED = "#bdbdbd"       # categories that carry no message
TEXT = "#333333"


def style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 11.5,
        "axes.titlesize": 12.5, "axes.labelsize": 11.5,
        "xtick.labelsize": 10.5, "ytick.labelsize": 10.5, "legend.fontsize": 10.5,
        "axes.spines.top": False, "axes.spines.right": False,
        "savefig.dpi": 200, "pdf.fonttype": 42,
    })


def save(fig, name):
    """PNG and PDF under writeup/figs/, same basename."""
    FIGS.mkdir(parents=True, exist_ok=True)
    out = []
    for ext in ("png", "pdf"):
        p = FIGS / ("%s.%s" % (name, ext))
        fig.savefig(p, bbox_inches="tight")
        out.append(p)
    plt.close(fig)
    return out


def caption(name, title, script, sources, rule, body):
    """The caption file beside the figure: data sources and the selection rule, then the body."""
    p = FIGS / ("%s.caption.md" % name)
    lines = ["# %s" % title, "",
             "Machine-written by `%s`; regenerate, never hand-edit. CPU only, no generation, no model load, "
             "no judge call, no cost." % script, "",
             "**Data sources**", ""]
    lines += ["- `%s`" % s for s in sources]
    lines += ["", "**Selection rule:** %s" % rule, "", body.rstrip(), ""]
    p.write_text("\n".join(lines), encoding="utf-8")
    return p


# ------------------------------------------------------------------ the machine-written result tables

def read_md_tables(path):
    """Every pipe table in a markdown file as {"context", "header", "rows"}; `context` is the nearest
    non-empty, non-table line above the table (a heading or a bold lead-in), so a table can be found by
    the text that introduces it rather than by position."""
    tables, context, cur = [], "", None
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if s.startswith("|"):
            cells = [c.strip() for c in s.strip("|").split("|")]
            if cur is None:
                cur = {"context": context, "header": cells, "rows": []}
            elif all(re.fullmatch(r":?-{2,}:?", c) for c in cells):
                continue
            else:
                cur["rows"].append(cells)
        else:
            if cur is not None:
                tables.append(cur)
                cur = None
            if s:
                context = s
    if cur is not None:
        tables.append(cur)
    return tables


def find_table(tables, context_substr, header_substr=None):
    hits = [t for t in tables if context_substr in t["context"]
            and (header_substr is None or any(header_substr in h for h in t["header"]))]
    assert len(hits) == 1, (context_substr, header_substr, len(hits))
    return hits[0]


def row(table, first_cell_substr, extra=None):
    """The one row whose first cell contains `first_cell_substr` (and, if given, whose second contains
    `extra`), as {header: cell}."""
    hits = [r for r in table["rows"] if first_cell_substr in r[0]
            and (extra is None or (len(r) > 1 and extra in r[1]))]
    assert len(hits) == 1, (first_cell_substr, extra, len(hits))
    return dict(zip(table["header"], hits[0]))


_NUM = r"[-+−]?\d+(?:\.\d+)?(?:e[-+]?\d+)?"


def num(cell):
    """The first number in a cell, ignoring bold, backticks and the unicode minus."""
    m = re.search(_NUM, cell.replace("−", "-").replace("**", "").replace("`", ""))
    assert m, cell
    return float(m.group(0))


def num_ci(cell):
    """(value, lo, hi) from cells shaped like `0.013 [0.000, 0.037] (n=80, runs=8)`; lo, hi None if absent."""
    s = cell.replace("−", "-").replace("**", "").replace("`", "")
    m = re.match(r"\s*(%s)\s*(?:\[\s*(%s)\s*,\s*(%s)\s*\])?" % (_NUM, _NUM, _NUM), s)
    assert m, cell
    v = float(m.group(1))
    lo = float(m.group(2)) if m.group(2) is not None else None
    hi = float(m.group(3)) if m.group(3) is not None else None
    return v, lo, hi


def count_pair(cell, label):
    """`act-focused=8` -> 8 for label `act-focused`; 0 if the label is absent."""
    m = re.search(r"%s\s*=\s*(\d+)" % re.escape(label), cell)
    return int(m.group(1)) if m else 0
