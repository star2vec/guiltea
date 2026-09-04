"""S1d Task 1 — build the join from the S1b probe runs (no API, no GPU).

Reads results/raw/s1b/t7/*/*.json (the 508 probe records), the act-label tables in
results/raw/s1b/judge_calls/{act_primary,act_second}.jsonl, and the prior-act answer text
from the cell the probe forked from (t4 v1 for deceived, t5 for akratic, t6 home for vicious).
Writes results/raw/s1d/join.jsonl and results/raw/s1d/join_counts.json.
"""
from __future__ import annotations

import json
import os
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
RAW = REPO / "results" / "raw" / "s1b"
OUT = REPO / "results" / "raw" / "s1d"

# the act-source cell each probe mode forked from
ACT_TAG = {"deceived": "v1", "akratic": "akratic", "vicious": "home"}


def act_label_table():
    """{(target, seed, tag, turn): {...}} — the second judge's label is final where it exists (D-019)."""
    table = {}
    stats = Counter()
    for purpose, path in (("act_primary", RAW / "judge_calls" / "act_primary.jsonl"),
                          ("act_second", RAW / "judge_calls" / "act_second.jsonl")):
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                r = json.loads(line)
                key = (r["target"], int(r["seed"]), r["tag"], int(r["turn"]))
                row = table.setdefault(key, {"primary": None, "second": None})
                row["primary" if purpose == "act_primary" else "second"] = r["label"]
                stats[purpose] += 1
    for key, row in table.items():
        row["final"] = row["second"] if row["second"] is not None else row["primary"]
        row["source"] = "second" if row["second"] is not None else "primary"
    stats["keys"] = len(table)
    stats["with_second"] = sum(1 for r in table.values() if r["second"] is not None)
    stats["reversed_by_second"] = sum(
        1 for r in table.values() if r["second"] is not None and r["second"] != r["primary"])
    return table, stats


def prior_act(mode, target, seed):
    """The answer text of the act turn the probe forked from, and its turn index."""
    if mode == "deceived":
        rec = json.load(open(RAW / "t4" / target / ("v1_seed%d.json" % seed), encoding="utf-8"))
        t = rec["T_primary"]
        return rec["turns"][t - 1]["answer"], t
    fname = {"akratic": "akratic_seed%d.json", "vicious": "home_seed%d.json"}[mode]
    sub = {"akratic": "t5", "vicious": "t6"}[mode]
    rec = json.load(open(RAW / sub / target / (fname % seed), encoding="utf-8"))
    return rec["turns"][0]["answer"], 1


def main():
    table, stats = act_label_table()
    files = sorted(p for p in (RAW / "t7").glob("*/*.json") if p.name != "summary.json")
    rows = []
    counts = defaultdict(Counter)
    for p in files:
        d = json.load(open(p, encoding="utf-8"))
        turn = d["turns"][-1]
        mode, target, seed = d["mode"], d["target"], d["seed"]
        act_text, act_turn_src = prior_act(mode, target, seed)
        assert act_turn_src == d["act_turn"], (p, act_turn_src, d["act_turn"])
        akey = (target, seed, ACT_TAG[mode], d["act_turn"])
        alab = table.get(akey, {})
        rows.append({
            "target": target, "seed": seed, "mode": mode, "fork": d["fork"], "tag": d["tag"],
            "act_turn": d["act_turn"], "act_tag": ACT_TAG[mode],
            "act_label": alab.get("final"), "act_label_source": alab.get("source"),
            "act_label_primary": alab.get("primary"), "act_label_second": alab.get("second"),
            "act_text": act_text,
            "answer": turn["answer"], "think": turn["think"],
            "probe_label": turn.get("probe_label"), "probe_reason": turn.get("probe_reason"),
            "n_new": turn["n_new"], "finish": turn["finish"], "adherent": turn["adherent"],
            "proj_summary": d["proj_summary"][-1],
            "pt": str(p.with_suffix(".pt").relative_to(REPO)),
        })
        counts["mode_fork"]["%s/%s" % (mode, d["fork"])] += 1
        counts["mode"][mode] += 1
        counts["fork"][d["fork"]] += 1
        counts["act_label_final"][str(alab.get("final"))] += 1
        counts["act_label_source"][str(alab.get("source"))] += 1
        counts["target"][target] += 1

    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "join.jsonl", "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")

    summary = {
        "n_rows": len(rows),
        "n_t7_files": len(files),
        "counts": {k: dict(sorted(v.items())) for k, v in counts.items()},
        "n_targets": len(counts["target"]),
        "empty_think": sum(1 for r in rows if not r["think"].strip()),
        "truncated_replies": sum(1 for r in rows if r["finish"] != "eos"),
        "act_label_table": dict(stats),
        "join_bytes": (OUT / "join.jsonl").stat().st_size,
    }
    json.dump(summary, open(OUT / "join_counts.json", "w", encoding="utf-8"), indent=1, sort_keys=True)
    print(json.dumps(summary, indent=1, sort_keys=True))


if __name__ == "__main__":
    main()
