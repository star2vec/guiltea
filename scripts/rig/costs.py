#!/usr/bin/env python3
"""S3 rig — Phase 2 §4: the measured per-cell costs and the menu table.

Reads only what the run wrote (`summary.json`, `judge_ledger.json`, `judge_calls/`, the `.pt` files and the
per-run `.jsonl`) and prints the two tables the brief asks for. It **chooses nothing**: the menu options are
priced, not picked, and the researcher-reading items are counted from the S1b judge protocol's own rates.

    python scripts/rig/costs.py [--out results/raw/s4] [--held 16] [--balance 7.4]
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ARMS = ["act_blame", "self_blame", "neutral_correction", "neutral_reflection", "none"]


def load_cells(out: Path):
    cells = []
    for p in sorted(out.glob("*/*/summary.json")):
        s = json.load(open(p))
        d = p.parent
        pt = sorted(d.glob("*_seed*.pt"))
        rows = defaultdict(int)
        forks = defaultdict(int)
        n_records = 0
        for j in sorted(d.glob("*_seed*.jsonl")):
            for line in open(j, encoding="utf-8"):
                r = json.loads(line)
                n_records += 1
                rows[r["kind"]] += 1
                if r["kind"] == "fork":
                    forks[(r["fork_type"], r["distance"])] += 1
        s["_dir"] = str(d.relative_to(ROOT))
        s["_pt_bytes"] = sum(x.stat().st_size for x in pt)
        s["_records"] = n_records
        s["_kinds"] = dict(rows)
        s["_forks"] = {"%s_d%s" % (k[0], k[1]): v for k, v in sorted(forks.items())}
        cells.append(s)
    return cells


def load_calls(out: Path):
    """Every judge call the run logged, by purpose; arm/mode where the record carries it."""
    by_purpose = defaultdict(lambda: {"calls": 0, "usd": 0.0, "model": set()})
    per_cell = defaultdict(lambda: defaultdict(lambda: {"calls": 0, "usd": 0.0}))
    for p in sorted((out / "judge_calls").glob("*.jsonl")):
        for line in open(p, encoding="utf-8"):
            r = json.loads(line)
            pur, usd = r.get("purpose", p.stem), float(r.get("cost_usd") or 0.0)
            by_purpose[pur]["calls"] += 1
            by_purpose[pur]["usd"] += usd
            by_purpose[pur]["model"].add(r.get("model"))
            key = (r.get("mode"), r.get("arm")) if r.get("arm") else ("control" if r.get("control") else None, None)
            per_cell[key][pur]["calls"] += 1
            per_cell[key][pur]["usd"] += usd
    return by_purpose, per_cell


def reading_items(n_seeds: int, n_same: int, n_unrel: int, distance4: bool, chain_turns_per_run: float):
    """The researcher-reading items one cell generates under the S1b judge protocol (briefs/S1b-runs.md §9):
    second act judge on T-2..T+1 of every measured chain; probe/flag judges on a 15-20 % random subsample;
    self-consistency re-run on 10 %; the human pile = disagreements + 30 agreement audits per judge type +
    20 non-flagged thinking blocks, hard cap 180."""
    d = 2 if distance4 else 1
    act_neighbourhood = n_seeds * min(4, chain_turns_per_run)      # T-2..T+1, capped by the chain length
    same_domain = n_seeds * n_same * d
    probes = n_seeds * d
    unrelated = n_seeds * n_unrel * d
    return {
        "act second-judge items (T-2..T+1)": round(act_neighbourhood, 1),
        "probe subsample @15-20 %": (round(0.15 * probes, 1), round(0.20 * probes, 1)),
        "same-domain subsample @15-20 %": (round(0.15 * same_domain, 1), round(0.20 * same_domain, 1)),
        "self-consistency re-runs @10 %": round(0.10 * (act_neighbourhood + same_domain + probes + unrelated), 1),
        "adjudication pile (disagreements + audits, cap 180)": "cap 180 per judge-validation pass",
    }


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "results" / "raw" / "s4"))
    ap.add_argument("--held", type=int, default=16, help="held targets S1b ran (reports/S1b-runs.md §2)")
    ap.add_argument("--balance", type=float, default=7.4, help="the key's remaining balance before this run")
    ap.add_argument("--json", default=None)
    a = ap.parse_args(argv)
    out = Path(a.out)
    cells = load_cells(out)
    by_purpose, per_cell = load_calls(out)
    ledger = json.load(open(out / "judge_ledger.json"))
    header = json.load(open(out / "run_header.json"))
    footer = json.load(open(out / "run_footer.json")) if (out / "run_footer.json").exists() else {}

    n_seeds = len(header["seeds"])
    n_unrel = len(header["unrelated_questions"])
    ctrl_pt = sum(p.stat().st_size for p in (out / "controls").glob("*.pt")) if (out / "controls").exists() else 0
    ctrl_usd = sum(v["usd"] for v in per_cell[("control", None)].values())

    print("## Per-cell measured cost (smoke run, %s, N = %d)\n" % (header["targets"][0], n_seeds))
    print("| cell | runs | act rate | discards | machine s | s/run | API $ | records | residuals |")
    print("|---|---|---|---|---|---|---|---|---|")
    tot_s = tot_usd = tot_b = 0.0
    for c in cells:
        tot_s += c["machine_s"]; tot_usd += c["api_usd"]; tot_b += c["_pt_bytes"]
        print("| %s/%s | %s | %s | %s | %.0f | %.0f | %.4f | %d | %.2f GiB |" % (
            c["mode"], c["arm"], c["n_runs"], c["act_rate"], c["discards"], c["machine_s"],
            c["machine_s"] / max(1, c["n_runs"]), c["api_usd"], c["_records"], c["_pt_bytes"] / 2 ** 30))
    print("| **5 cells** | | | | **%.0f** | | **%.4f** | | **%.2f GiB** |" % (tot_s, tot_usd, tot_b / 2 ** 30))
    print("\nControls (shared across cells, asked once): $%.4f, %.2f GiB, %d (question, seed) pairs."
          % (ctrl_usd, ctrl_pt / 2 ** 30, len(list((out / 'controls').glob('*.json'))) if (out / 'controls').exists() else 0))

    print("\n## API by purpose and model\n")
    print("| purpose | model | calls | $ | $/call |")
    print("|---|---|---|---|---|")
    for pur, v in sorted(by_purpose.items(), key=lambda kv: -kv[1]["usd"]):
        print("| `%s` | %s | %d | %.4f | %.5f |" % (pur, ",".join(sorted(m for m in v["model"] if m)),
                                                    v["calls"], v["usd"], v["usd"] / max(1, v["calls"])))
    print("| **total** | | **%d** | **%.4f** | |" % (ledger["calls"], ledger["list_price_usd"]))

    # per-cell means for the menu
    core_cells = [c for c in cells if c["mode"] == "deceived"]
    per_cell_s = sum(c["machine_s"] for c in core_cells) / max(1, len(core_cells))
    per_cell_usd = sum(c["api_usd"] for c in core_cells) / max(1, len(core_cells))
    per_cell_b = sum(c["_pt_bytes"] for c in core_cells) / max(1, len(core_cells))
    d4_share = 0.5  # the distance-4 set is the second of the two fork passes; non-core cells run only distance 0
    ctrl_per_target_usd = ctrl_usd
    ctrl_per_target_s = float(header.get("controls_s") or 0.0)

    print("\n## Menu (STAGE0 §7) — priced, not chosen. Held targets S1b ran: H = %d." % a.held)
    print("\nPer-target-per-cell unit costs, measured at N = %d on this target: %.0f machine s, $%.4f, %.2f GiB "
          "(core cell, distance 0 + 4); a non-core cell (akratic/vicious) runs one fork pass and a single-turn "
          "act, so it is priced at %.0f%% of the core figure.\n" % (n_seeds, per_cell_s, per_cell_usd,
                                                                    per_cell_b / 2 ** 30, 100 * d4_share))
    print("| option | cells/target | N | machine h | API $ | residuals | researcher-reading items |")
    print("|---|---|---|---|---|---|---|")
    for name, modes in (("FULL (deceived+akratic+vicious x 5)", 3), ("MEDIUM (deceived+akratic x 5)", 2),
                        ("CORE (deceived x 5)", 1)):
        for N in (12, 8):
            scale = N / n_seeds
            core = 5
            non_core = 5 * (modes - 1)
            s = (core * per_cell_s + non_core * per_cell_s * d4_share) * a.held * scale
            usd = (core * per_cell_usd + non_core * per_cell_usd * d4_share) * a.held * scale + ctrl_per_target_usd * a.held * scale
            b = (core * per_cell_b + non_core * per_cell_b * d4_share) * a.held * scale
            ri = reading_items(N, 3.5, n_unrel, True, 4)
            print("| %s | %d | %d | %.1f | %.2f | %.1f GiB | %.0f second-judge + %.0f-%.0f subsample per cell x %d cells |" % (
                name, core + non_core, N, s / 3600, usd, b / 2 ** 30,
                ri["act second-judge items (T-2..T+1)"], ri["probe subsample @15-20 %"][0] + ri["same-domain subsample @15-20 %"][0],
                ri["probe subsample @15-20 %"][1] + ri["same-domain subsample @15-20 %"][1], (core + non_core) * a.held))
    print("\nKey balance before this run: $%.2f; this run spent $%.4f; remaining $%.2f."
          % (a.balance, ledger["list_price_usd"], a.balance - ledger["list_price_usd"]))
    if footer:
        print("\nFooter: %s" % json.dumps({k: footer[k] for k in ("user_span_misses", "batching", "escalation") if k in footer}))
    if a.json:
        json.dump({"cells": cells, "by_purpose": {k: {**v, "model": sorted(m for m in v["model"] if m)}
                                                  for k, v in by_purpose.items()}, "ledger": ledger},
                  open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
