"""S5b — the machine-written table for briefs/S5b-prevent.md (chain only, three arms).

    python scripts/rig/s5b_table.py [--out results/raw/s5b] [--table results/S5b_table.md]

EXPLORATORY throughout: the hypothesis comes from a predictive readout (S1e/S1g), not a pre-registered
prediction, and `nn` is a generic negative-valence arrow that was **found to predict** capitulation, not a
"capitulation direction".

Sections: A the arms and their injected norms (which must be equal for P1 and P2) and the norm check; B act
rate, discards and T per arm with a clustered bootstrap CI, **beside the coherence mean**; C the P1-vs-P2
contrast, which is the comparison the brief reads; D readouts at `into`/`think`/`answer` on every arrow over
both bands with the random floor and the injected component.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rigcommon as R  # noqa: E402

POS = R.POSITIONS
ARMS = [("none", "P0 — unsteered"),
        ("none+steer_nn_L16_c-4", "P1 — −4·σ(nn)·n̂n at L16"),
        ("none+steer_random0_L16_c-4", "P2 — −4·σ(nn)·û_random0 at L16 (norm-matched)")]


def _fmt(x, nd=3):
    if x is None:
        return "—"
    if isinstance(x, float) and x != x:
        return "NaN"
    return ("%%.%df" % nd) % x


def _cell(x):
    return str(x).replace("|", "\\|").replace("\n", " ")


def _load_pt(p):
    try:
        return torch.load(p, map_location="cpu", weights_only=False, mmap=True)
    except Exception:  # noqa
        return torch.load(p, map_location="cpu", weights_only=False)


class Arm:
    def __init__(self, path: Path, label: str):
        self.path, self.label, self.name = path, label, path.name
        self.summary = json.load(open(path / "summary.json", encoding="utf-8")) if (path / "summary.json").exists() else {}
        self.steer = self.summary.get("steer")
        self.records, self.metas = [], {}
        for jl in sorted(path.glob("*_seed*.jsonl")):
            stem = str(jl)[:-len(".jsonl")]
            blob = _load_pt(stem + ".pt")
            meta = json.load(open(stem + ".meta.json", encoding="utf-8"))
            self.metas[(meta["target"], meta["seed"])] = meta
            for r in [json.loads(l) for l in open(jl, encoding="utf-8") if l.strip()]:
                r["_proj"] = blob["proj"][r["record_index"]] if len(blob["proj"]) else None
                self.records.append(r)
                if self.steer is None and meta.get("steer"):
                    self.steer = meta["steer"]

    def acts(self):
        return [r for r in self.records if r.get("kind") == "act_turn"]

    def committed(self):
        return {k: m for k, m in self.metas.items() if m.get("committed")}

    def act_flags(self):
        keys = sorted(self.metas)
        return keys, [1.0 if self.metas[k].get("committed") else 0.0 for k in keys]

    def T(self):
        return {k: m.get("T_primary") for k, m in sorted(self.metas.items())}

    def coherence(self):
        return [r["coherence"] for r in self.acts() if r.get("coherence") is not None]

    def act_turn_records(self):
        """The committed act turn (T_primary) of every committed run — where the readouts are taken."""
        out = []
        for (tid, seed), m in sorted(self.committed().items()):
            t = m.get("T_primary")
            hit = [r for r in self.records if r.get("kind") == "act_turn" and r.get("turn") == t
                   and r["target"] == tid and r["seed"] == seed]
            out += hit
        return out


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


def build(out_root: Path, table_path: Path):
    out_root = Path(out_root)
    header = json.load(open(out_root / "run_header.json", encoding="utf-8")) if (out_root / "run_header.json").exists() else {}
    ledger = json.load(open(out_root / "judge_ledger.json", encoding="utf-8")) if (out_root / "judge_ledger.json").exists() else {}
    arms = []
    for name, label in ARMS:
        d = out_root / "deceived" / name
        if d.is_dir() and any(d.glob("*_seed*.jsonl")):
            arms.append(Arm(d, label))
    if not arms:
        raise SystemExit("no arms under %s" % out_root)
    arrows = header.get("arrows", {})
    axes = arrows.get("named_axes", []) + ["random%d" % s for s in range(arrows.get("random_axes", 0))]
    arrow_layers = arrows.get("arrow_layers", [])
    named = [a for a in axes if not a.startswith("random")]
    rnd_idx = [i for i, a in enumerate(axes) if a.startswith("random")]

    L, A = [], None
    A = L.append
    A("# S5b results table — steering against the susceptibility direction during the chain")
    A("")
    A("_Machine-written by `scripts/rig/s5b_table.py`. Never hand-edited; regenerated from `%s`._" % out_root)
    A("")
    A("> **EXPLORATORY.** The hypothesis comes from a predictive readout (S1e/S1g), not a pre-registered")
    A("> prediction. **`nn` is a generic negative-valence arrow that was found to predict capitulation, not a**")
    A("> **\"capitulation direction\"**, and an AUROC of 0.66–0.71 is a modest correlation, not a clean")
    A("> separation. Steering is on for **every turn of the chain**, which is a stronger intervention than the")
    A("> turn-1 readout strictly licenses.")
    A("")

    # ---------------------------------------------------------------- A
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
    if len(norms) == 2:
        a, b = list(norms.values())
        A("**P1 and P2 injected norms equal: %s** (|difference| = %.2e). The brief requires them printed and "
          "equal; P2 is norm-matched, not σ-matched per arrow." % ("yes" if abs(a - b) < 1e-6 else "**NO**",
                                                                   abs(a - b)))
    A("")
    for p in sorted(out_root.glob("norm_check_*.json")):
        d = json.load(open(p, encoding="utf-8"))
        A("- Pre-hook norm check `%s`: worst relative deviation **%.2e** against a %.0f%% tolerance — **%s**."
          % (p.name, d["worst_relative_deviation"], 100 * d["tolerance"], "PASS" if d["pass"] else "FAIL"))
    A("")
    A("| fact | value |")
    A("|---|---|")
    A("| subject | `%s` @ `%s` |" % (_cell(header.get("machine", {}).get("model")),
                                     _cell(header.get("machine", {}).get("revision"))))
    A("| target | %s |" % _cell(header.get("targets")))
    A("| seeds | %s |" % _cell(header.get("seeds")))
    A("| chain only | %s (no feedback turn, no forks, no distance-4 set, no topic controls) |"
      % _cell(header.get("chain_only")))
    A("| steering scope | %s |" % _cell(header.get("steer_scope")))
    A("| judges | act judge and the vendored coherence prompt, both `gpt-4o-mini` |")
    A("| API ledger | $%s over %s calls |" % (_fmt(ledger.get("list_price_usd"), 4), ledger.get("calls")))
    A("")

    # ---------------------------------------------------------------- B
    A("## B. Act rate, T, and coherence")
    A("")
    A("Coherence is the load-bearing control: **an act-rate drop with a coherence drop is damage, not "
      "prevention.** It is the vendored coherence prompt on `gpt-4o-mini`, over every act-turn answer.")
    A("")
    A("| arm | N | committed | discards | act rate | 95% CI (clustered on the run) | T per run | "
      "coherence mean | coherence min | n act turns scored |")
    A("|---|---|---|---|---|---|---|---|---|---|")
    for arm in arms:
        keys, flags = arm.act_flags()
        m, lo, hi, n, k = R.bootstrap_ci_clustered(flags, keys)
        Ts = [v for v in arm.T().values() if v is not None]
        coh = arm.coherence()
        A("| %s | %d | %d | %d | %s | [%s, %s] | %s | %s | %s | %d |" % (
            arm.label, len(keys), sum(int(f) for f in flags), len(keys) - sum(int(f) for f in flags),
            _fmt(m), _fmt(lo), _fmt(hi),
            (", ".join(str(t) for t in Ts) or "—"),
            _fmt(sum(coh) / len(coh), 1) if coh else "—",
            _fmt(min(coh), 1) if coh else "—", len(coh)))
    A("")
    A("_The cluster is the run (target × seed). With one target and one act outcome per run this is the "
      "run-level bootstrap; the clustering matters only if more targets are added._")
    A("")

    # ---------------------------------------------------------------- C
    A("## C. P1 against P2 — the comparison the brief reads")
    A("")
    by = {a.name: a for a in arms}
    p1, p2, p0 = by.get(ARMS[1][0]), by.get(ARMS[2][0]), by.get(ARMS[0][0])
    A("| contrast | Δ act rate | 95% CI | Δ coherence mean | paired runs |")
    A("|---|---|---|---|---|")
    for x, y, label in ((p1, p2, "**P1 − P2** (the direction against a norm-matched random push of the same size)"),
                        (p1, p0, "P1 − P0 (against unsteered)"),
                        (p2, p0, "P2 − P0 (the random arm against unsteered)")):
        if x is None or y is None:
            A("| %s | — | — | — | 0 |" % label)
            continue
        kx, fx = x.act_flags()
        ky, fy = y.act_flags()
        dx, dy = dict(zip(kx, fx)), dict(zip(ky, fy))
        keys = sorted(set(dx) & set(dy))
        diffs = [dx[k] - dy[k] for k in keys]
        m, lo, hi, n, nk = R.bootstrap_ci_clustered(diffs, keys)
        cx, cy = x.coherence(), y.coherence()
        dcoh = (sum(cx) / len(cx) - sum(cy) / len(cy)) if (cx and cy) else None
        A("| %s | %s | [%s, %s] | %s | %d |" % (label, _fmt(m), _fmt(lo), _fmt(hi), _fmt(dcoh, 1), len(keys)))
    A("")
    n_runs = len(arms[0].act_flags()[0]) if arms else 0
    A("**Detectable difference at this N.** With %d runs per arm from a base of 8 of 8 committed, one run is "
      "%.3f of the act rate: a drop to 3 of 8 or fewer (Δ ≤ −0.625 against a full base) is visible at this N, "
      "and a drop of one or two runs is not. Any result below must be read with that sentence attached."
      % (n_runs, 1.0 / n_runs if n_runs else float("nan")))
    A("")

    # ---------------------------------------------------------------- D
    A("## D. Readouts at the act turn")
    A("")
    A("Every arrow over both bands at `into`, `think` and `answer`, with the random floor (max |value| over "
      "randctl 0–9) and, for a steered arm, **the injected component `c·σ·cos(û_inj, axis)`** — exact at L16, "
      "the steered layer, so an injected projection is never read as a state change.")
    for pos in ("into", "think", "answer"):
        A("")
        A("### `%s`" % pos)
        A("")
        A("| arm | band | " + " | ".join("`%s`" % a for a in named) + " | random floor |")
        A("|---|---|" + "---|" * (len(named) + 1))
        for arm in arms:
            mp = mean_proj(arm.act_turn_records(), pos)
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
    A("_Layers below the steered one are untouched; layers above carry propagated effect, not injection. "
      "The band means above mix both, which is why the injected component is printed beside them._")
    A("")
    Path(table_path).parent.mkdir(parents=True, exist_ok=True)
    Path(table_path).write_text("\n".join(L) + "\n", encoding="utf-8")
    return table_path


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(R.ROOT / "results" / "raw" / "s5b"))
    ap.add_argument("--table", default=str(R.ROOT / "results" / "S5b_table.md"))
    a = ap.parse_args(argv)
    print("wrote %s" % build(Path(a.out), Path(a.table)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
