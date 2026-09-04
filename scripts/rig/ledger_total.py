"""The $11.50 stop is global, but each `--out` root carries its own ledger and its own stop.

`rigcommon.configure` points `judge_utils.LEDGER` at `<out>/judge_ledger.json`, so S4, S5b and S5c — which
must use separate roots, because S5b's chain-only `none` arm and S5c's full `none` arm are different cells
with the same name — would each get a fresh $11.50 allowance. This sums them, so a run can be given
`--budget $(ledger_total.py --remaining --except <its own root>)` and the global stop is honoured.

    python scripts/rig/ledger_total.py                      # the table and the total
    python scripts/rig/ledger_total.py --remaining --except results/raw/s5b
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rigcommon as R  # noqa: E402

STOP_USD = 11.50
ROOTS = ("s4", "s5b", "s5c")


def ledgers():
    out = {}
    for name in ROOTS:
        p = R.ROOT / "results" / "raw" / name / "judge_ledger.json"
        if p.exists():
            d = json.load(open(p, encoding="utf-8"))
            out[name] = {"usd": float(d.get("list_price_usd") or 0.0), "calls": int(d.get("calls") or 0)}
        else:
            out[name] = {"usd": 0.0, "calls": 0}
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--remaining", action="store_true", help="print only the budget a run may be given")
    ap.add_argument("--except", dest="excl", default=None,
                    help="the root the next run will write to; its own ledger is excluded, because that "
                         "run's own stop counts it again")
    a = ap.parse_args(argv)
    L = ledgers()
    excl = Path(a.excl).name if a.excl else None
    if a.remaining:
        other = sum(v["usd"] for k, v in L.items() if k != excl)
        print("%.4f" % max(0.0, STOP_USD - other))
        return 0
    tot = sum(v["usd"] for v in L.values())
    print("| root | calls | $ |")
    print("|---|---|---|")
    for k, v in L.items():
        print("| `results/raw/%s` | %d | %.4f |" % (k, v["calls"], v["usd"]))
    print("| **total** | **%d** | **%.4f** |" % (sum(v["calls"] for v in L.values()), tot))
    print("\nglobal stop $%.2f, remaining $%.4f" % (STOP_USD, STOP_USD - tot))
    return 0


if __name__ == "__main__":
    sys.exit(main())
