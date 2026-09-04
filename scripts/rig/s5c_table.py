"""S5c — the machine-written table for briefs/S5c-persona-steer.md (persona steering + the re-ask fork).

    python scripts/rig/s5c_table.py [--out results/raw/s5c] [--table results/S5c_table.md]

EXPLORATORY throughout (D-023): the hypothesis dates from 2026-09-04 and nothing here is pre-registered.

Sections: A the arms and their injected norms, which must be equal for Q1 and Q3, and the norm checks; B act
rate, spread, same-domain hold, coherence and re-commission per arm with clustered bootstrap CIs; C the
contrasts the brief reads (Q1 against Q3, and each against Q0); D the re-commission table across every arm
that has one, including the cell-A and cell-B back-fill, which is the comparison Task A exists for; E readouts
on every arrow over both bands with the random floor and the injected component beside them.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rigcommon as R  # noqa: E402

POS = R.POSITIONS
ARMS = [("none", "Q0 — unsteered (`none` feedback)"),
        ("none+steer_persona_L16_c-4", "Q1 — −4·σ(persona)·p̂ at L16 (away from Assistant)"),
        ("none+steer_random0_L16_c-4", "Q3 — −4·σ(persona)·û_random0 at L16 (norm-matched)"),
        ("none+steer_persona_L16_c4", "Q2 — +4·σ(persona)·p̂ at L16 (toward Assistant)")]
REASK_ARMS = [("act_blame+reask", "cell A back-fill — act-blame"),
              ("self_blame+reask", "cell B back-fill — self-blame")]


def _fmt(x, nd=3):
    if x is None:
        return "—"
    if isinstance(x, float) and x != x:
        return "NaN"
    return ("%%.%df" % nd) % x


def _load_pt(p):
    try:
        return torch.load(p, map_location="cpu", weights_only=False, mmap=True)
    except Exception:  # noqa
        return torch.load(p, map_location="cpu", weights_only=False)


class Arm:
    def __init__(self, path: Path, label: str, with_proj=True):
        self.path, self.label, self.name = path, label, path.name
        sp = path / "summary.json"
        self.summary = json.load(open(sp, encoding="utf-8")) if sp.exists() else {}
        self.steer = self.summary.get("steer")
        self.records, self.metas = [], {}
        for jl in sorted(path.glob("*_seed*.jsonl")):
            stem = str(jl)[:-len(".jsonl")]
            meta = json.load(open(stem + ".meta.json", encoding="utf-8"))
            self.metas[(meta["target"], meta["seed"])] = meta
            blob = _load_pt(stem + ".pt") if with_proj else None
            for r in [json.loads(l) for l in open(jl, encoding="utf-8") if l.strip()]:
                if blob is not None and len(blob["proj"]):
                    r["_proj"] = blob["proj"][r["record_index"]]
                self.records.append(r)
            if self.steer is None and meta.get("steer"):
                self.steer = meta["steer"]

    def _rate(self, pred, distance=None):
        keys, vals = [], []
        for r in self.records:
            if distance is not None and r.get("distance") != distance:
                continue
            v = pred(r)
            if v is None:
                continue
            keys.append((r["target"], r["seed"])); vals.append(float(v))
        if not vals:
            return None
        m, lo, hi, n, _ = R.bootstrap_ci_clustered(vals, keys)
        return m, lo, hi, n

    def act_flags(self):
        keys = sorted(self.metas)
        return keys, [1.0 if self.metas[k].get("committed") else 0.0 for k in keys]

    def act_rate(self):
        keys, flags = self.act_flags()
        if not flags:
            return None
        m, lo, hi, n, _ = R.bootstrap_ci_clustered(flags, keys)
        return m, lo, hi, n

    def spread(self, d):
        return self._rate(lambda r: (1.0 if r.get("spread_flag") else 0.0)
                          if r.get("fork_type") == "unrelated" and not r.get("spread_unscored") else None, d)

    def hold(self, d):
        return self._rate(lambda r: (1.0 if r.get("hold") else 0.0)
                          if r.get("fork_type") == "same_domain" and r.get("hold") is not None else None, d)

    def recommission(self, d):
        return self._rate(lambda r: (1.0 if r.get("re_commissioned") else 0.0)
                          if r.get("fork_type") == "re_ask" else None, d)

    def coherence(self):
        return [r["coherence"] for r in self.records
                if r.get("fork_type") == "unrelated" and r.get("coherence") is not None]

    def blame(self):
        from collections import Counter
        return Counter(r["reflection_blame_label"] for r in self.records if r.get("reflection_blame_label"))

    def fork_records(self, d):
        return [r for r in self.records if r.get("fork_type") == "unrelated" and r.get("distance") == d]


def _ci(t, nd=3):
    return "—" if t is None else "%s [%s, %s]" % (_fmt(t[0], nd), _fmt(t[1], nd), _fmt(t[2], nd))


def mean_proj(records, position):
    ps = [r["_proj"] for r in records if r.get("_proj") is not None]
    if not ps:
        return None
    return torch.nanmean(torch.stack([p[POS.index(position)] for p in ps]).double(), dim=0)


def band_mean(row, arrow_layers, band):
    idx = [arrow_layers.index(L) for L in band if L in arrow_layers]
    if not idx:
        return None
    t = torch.tensor([float(row[j]) for j in idx], dtype=torch.float64)
    t = t[~torch.isnan(t)]
    return float(t.mean()) if len(t) else None


def paired(x: Arm, y: Arm, getter):
    """Δ between two arms on a per-run rate, paired on (target, seed) and clustered on the run."""
    def per_run(a):
        out = {}
        for r in a.records:
            v = getter(r)
            if v is None:
                continue
            out.setdefault((r["target"], r["seed"]), []).append(float(v))
        return {k: sum(v) / len(v) for k, v in out.items()}
    dx, dy = per_run(x), per_run(y)
    keys = sorted(set(dx) & set(dy))
    if not keys:
        return None
    diffs = [dx[k] - dy[k] for k in keys]
    m, lo, hi, n, _ = R.bootstrap_ci_clustered(diffs, keys)
    return m, lo, hi, len(keys)


def build(out_root: Path, table_path: Path):
    out_root = Path(out_root)
    hp = out_root / "run_header.json"
    header = json.load(open(hp, encoding="utf-8")) if hp.exists() else {}
    lp = out_root / "judge_ledger.json"
    ledger = json.load(open(lp, encoding="utf-8")) if lp.exists() else {}
    arms, reask = [], []
    for name, label in ARMS:
        d = out_root / "deceived" / name
        if d.is_dir() and any(d.glob("*_seed*.jsonl")):
            arms.append(Arm(d, label))
    for name, label in REASK_ARMS:
        d = out_root / "deceived" / name
        if d.is_dir() and any(d.glob("*_seed*.jsonl")):
            reask.append(Arm(d, label, with_proj=False))
    if not arms:
        raise SystemExit("no arms under %s" % out_root)
    arrows = header.get("arrows", {})
    axes = arrows.get("named_axes", []) + ["random%d" % s for s in range(arrows.get("random_axes", 0))]
    arrow_layers = arrows.get("arrow_layers", [])
    named = [a for a in axes if not a.startswith("random")]
    rnd_idx = [i for i, a in enumerate(axes) if a.startswith("random")]

    L = []
    A = L.append
    A("# S5c results table — steering the persona axis, and the re-ask fork")
    A("")
    A("_Machine-written by `scripts/rig/s5c_table.py`. Never hand-edited; regenerated from `%s`._" % out_root)
    A("")
    A("> **EXPLORATORY (D-023).** The hypothesis dates from 2026-09-04 and nothing here is pre-registered.")
    A("> The persona axis is the project's borrowed, validated direction; the causal claim is what this tests.")
    A("")

    # ------------------------------------------------------------------ A
    A("## A. The arms")
    A("")
    A("| arm | steered | arrow | layer | c | σ | σ from | **injected norm** |")
    A("|---|---|---|---|---|---|---|---|")
    norms = {}
    for arm in arms:
        s = arm.steer
        if not s:
            A("| %s | no | — | — | — | — | — | — |" % arm.label)
        else:
            norms[arm.name] = s["injected_norm"]
            A("| %s | yes | `%s` | L%d | %g | %s | `%s` | **%s** |" % (
                arm.label, s["arrow"], s["layer"], s["c"], _fmt(s["sigma"], 6), s["sigma_source"],
                _fmt(s["injected_norm"], 6)))
    A("")
    q1 = norms.get(ARMS[1][0]); q3 = norms.get(ARMS[2][0])
    if q1 is not None and q3 is not None:
        A("**Q1 and Q3 injected norms equal: %s** (|difference| = %.2e). The brief requires them printed and "
          "equal; Q3 is norm-matched, not σ-matched per arrow." % ("yes" if abs(q1 - q3) < 1e-6 else "**NO**",
                                                                   abs(q1 - q3)))
    A("")
    for p in sorted(out_root.glob("norm_check_*.json")):
        d = json.load(open(p, encoding="utf-8"))
        A("- Pre-hook norm check `%s`: worst relative deviation **%.2e** against a %.0f%% tolerance — **%s**."
          % (p.name, d["worst_relative_deviation"], 100 * d["tolerance"], "PASS" if d["pass"] else "FAIL"))
    A("")
    A("| fact | value |")
    A("|---|---|")
    A("| subject | `%s` @ `%s` |" % (header.get("machine", {}).get("model"),
                                     header.get("machine", {}).get("revision")))
    A("| target | %s | " % header.get("targets"))
    A("| seeds | %s |" % header.get("seeds"))
    A("| feedback arm | `none` throughout, so the only manipulated variable is the steering |")
    A("| steering window | on for the distance-0 forks, off for the four filler turns and the distance-4 forks |")
    A("| topic controls | reused verbatim from `results/raw/s4/controls` |")
    A("| API ledger | $%s over %s calls |" % (_fmt(ledger.get("list_price_usd"), 4), ledger.get("calls")))
    A("")

    # ------------------------------------------------------------------ B
    A("## B. Act rate, spread, hold, coherence, re-commission")
    A("")
    A("| arm | act rate | spread d0 | spread d4 | hold d0 | hold d4 | re-commission d0 | re-commission d4 | "
      "coherence mean | coherence min |")
    A("|---|---|---|---|---|---|---|---|---|---|")
    for arm in arms:
        coh = arm.coherence()
        A("| %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (
            arm.label, _ci(arm.act_rate()), _ci(arm.spread(0)), _ci(arm.spread(4)),
            _ci(arm.hold(0)), _ci(arm.hold(4)), _ci(arm.recommission(0)), _ci(arm.recommission(4)),
            _fmt(sum(coh) / len(coh), 1) if coh else "—", _fmt(min(coh), 1) if coh else "—"))
    A("")
    A("_Every rate is a clustered bootstrap on the run (target × seed). Coherence is the vendored prompt on "
      "the unrelated forks; **a spread change with a coherence drop is damage, not mechanism.**_")
    A("")

    # ------------------------------------------------------------------ C
    A("## C. The contrasts the brief reads")
    A("")
    by = {a.name: a for a in arms}
    q0a, q1a, q3a = by.get(ARMS[0][0]), by.get(ARMS[1][0]), by.get(ARMS[2][0])
    A("| contrast | Δ spread d0 | Δ spread d4 | Δ re-commission d0 | paired runs |")
    A("|---|---|---|---|---|")
    def _sp(r):
        return (1.0 if r.get("spread_flag") else 0.0) if (
            r.get("fork_type") == "unrelated" and r.get("distance") == 0 and not r.get("spread_unscored")) else None
    def _sp4(r):
        return (1.0 if r.get("spread_flag") else 0.0) if (
            r.get("fork_type") == "unrelated" and r.get("distance") == 4 and not r.get("spread_unscored")) else None
    def _rc(r):
        return (1.0 if r.get("re_commissioned") else 0.0) if (
            r.get("fork_type") == "re_ask" and r.get("distance") == 0) else None
    for x, y, lab in ((q1a, q3a, "**Q1 − Q3** (the axis against a norm-matched random push of the same size)"),
                      (q1a, q0a, "Q1 − Q0 (against unsteered)"),
                      (q3a, q0a, "Q3 − Q0 (the random arm against unsteered)")):
        if x is None or y is None:
            A("| %s | — | — | — | 0 |" % lab)
            continue
        a1, a2, a3 = paired(x, y, _sp), paired(x, y, _sp4), paired(x, y, _rc)
        A("| %s | %s | %s | %s | %d |" % (lab, _ci(a1), _ci(a2), _ci(a3), (a1 or (0, 0, 0, 0))[3]))
    A("")

    # ------------------------------------------------------------------ D
    A("## D. Re-commission — Task A, the quantity the sweep found no paper reporting")
    A("")
    A("The target's `situation` field delivered **verbatim** as a new user turn from the post-act (or "
      "post-reply) state, graded by the act judge against that target's own act rubric. Run at distance 0 and "
      "again after the four filler turns.")
    A("")
    A("| arm | correction in the text? | re-commission d0 | re-commission d4 | runs |")
    A("|---|---|---|---|---|")
    for arm in reask + arms:
        has = {"act_blame+reask": "yes — act-blame", "self_blame+reask": "yes — self-blame"}.get(arm.name, "no")
        r0, r4 = arm.recommission(0), arm.recommission(4)
        if r0 is None and r4 is None:
            continue
        A("| %s | %s | %s | %s | %d |" % (arm.label, has, _ci(r0), _ci(r4), (r0 or r4)[3]))
    A("")
    n = len(arms[0].act_flags()[0]) if arms else 0
    A("**Detectable difference at this N.** With %d runs per arm, one run is %s of the rate, so a difference "
      "smaller than that cannot be seen and any comparison below carries that sentence." % (n, _fmt(1.0 / n if n else float("nan"))))
    A("")

    # ------------------------------------------------------------------ E
    A("## E. Readouts at the distance-0 unrelated forks")
    A("")
    A("Every arrow over both bands with the random floor and, for a steered arm, **the injected component "
      "`c·σ·cos(û_inj, axis)`** — exact at L16 only.")
    for pos in ("into", "answer"):
        A("")
        A("### `%s`" % pos)
        A("")
        A("| arm | band | " + " | ".join("`%s`" % a for a in named) + " | random floor |")
        A("|---|---|" + "---|" * (len(named) + 1))
        for arm in arms:
            mp = mean_proj(arm.fork_records(0), pos)
            for bname, band in R.BANDS.items():
                if mp is None:
                    A("| %s | %s | " % (arm.label, bname) + " | ".join(["—"] * (len(named) + 1)) + " |")
                    continue
                vals = [band_mean(mp[axes.index(a)], arrow_layers, band) for a in named]
                floor = mp[rnd_idx].abs().max(dim=0).values if rnd_idx else None
                A("| %s | %s (L%d–L%d) | " % (arm.label, bname, band[0], band[-1]) +
                  " | ".join(_fmt(v) for v in vals) + " | " +
                  _fmt(band_mean(floor, arrow_layers, band) if floor is not None else None) + " |")
        for arm in arms:
            if not arm.steer:
                continue
            comp = arm.steer.get("injected_component_per_axis") or {}
            A("| _injected component_, %s (exact at L%d only) | — | " % (arm.label, arm.steer["layer"]) +
              " | ".join(_fmt(comp.get(a)) for a in named) + " | — |")
    A("")
    Path(table_path).parent.mkdir(parents=True, exist_ok=True)
    Path(table_path).write_text("\n".join(L) + "\n", encoding="utf-8")
    return table_path


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(R.ROOT / "results" / "raw" / "s5c"))
    ap.add_argument("--table", default=str(R.ROOT / "results" / "S5c_table.md"))
    a = ap.parse_args(argv)
    print("wrote %s" % build(Path(a.out), Path(a.table)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
