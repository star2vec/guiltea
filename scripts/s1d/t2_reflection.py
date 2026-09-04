"""S1d Task 2 — classify the blame target with the reflection judge already in scripts/judge_rubrics.py.

Runs the reflection judge over all 508 probe replies on gpt-4o-mini with the prior act supplied as
context (the rubric allows it), then re-judges a fixed 1-in-8 seeded sample on gpt-4o-2024-08-06.
The big model's label is final where it exists. The rubric and the judge are not edited.

Every call is stored under results/raw/s1d/judge_calls/. The ledger at results/raw/s1d/judge_ledger.json
carries the running list-price cost; the run stops before any call that would take it past the cap.
The API key is read by judge_rubrics from the environment or the repo-root .env and is never printed or written.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import random
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))
import judge_rubrics as jr  # noqa: E402

OUT = REPO / "results" / "raw" / "s1d"
CALLS = OUT / "judge_calls"
LEDGER = OUT / "judge_ledger.json"
CAP_USD = 1.00
MINI = "gpt-4o-mini"
BIG = "gpt-4o-2024-08-06"
SAMPLE_SEED = 0
SAMPLE_EVERY = 8
LABELS = set(jr.LABELS["reflection"])


def load_join():
    rows = [json.loads(l) for l in open(OUT / "join.jsonl", encoding="utf-8")]
    rows.sort(key=lambda r: (r["target"], r["mode"], r["fork"], r["seed"]))
    for i, r in enumerate(rows):
        r["item"] = i
    return rows


def sample_indices(n):
    """The fixed 1-in-8 sample: seed 0, n // 8 items drawn without replacement from the sorted join."""
    k = n // SAMPLE_EVERY
    return sorted(random.Random(SAMPLE_SEED).sample(range(n), k))


def read_ledger():
    if LEDGER.exists():
        return json.load(open(LEDGER, encoding="utf-8"))
    return {"stage": "S1d", "cap_usd": CAP_USD, "spent_usd": 0.0, "calls": 0,
            "by_purpose": {}, "started": _dt.datetime.utcnow().isoformat() + "Z", "stopped_on_cap": False}


def write_ledger(led):
    led["updated"] = _dt.datetime.utcnow().isoformat() + "Z"
    json.dump(led, open(LEDGER, "w", encoding="utf-8"), indent=1, sort_keys=True)


def judge_one(row, model, rubric, client):
    """One reflection judgment: one call, and exactly one re-call if the output is unparseable."""
    messages = jr.build_reflection_prompt(row["answer"], prior_act=row["act_text"], rubric=rubric)
    attempts = []
    label = reason = None
    for _ in range(2):
        resp = jr.call_judge(messages, model, temperature=0.0, max_tokens=200, client=client)
        resp["cost_usd"] = jr.cost_usd(resp["usage"], model)
        lab, rsn, raw = jr.parse_label(resp["content"], LABELS)
        resp["parsed_label"], resp["parsed_reason"], resp["label_raw"] = lab, rsn, raw
        attempts.append(resp)
        if lab is not None:
            label, reason = lab, rsn
            break
    return {
        "item": row["item"], "target": row["target"], "seed": row["seed"], "mode": row["mode"],
        "fork": row["fork"], "model": model, "label": label, "reason": reason,
        "unparseable": label is None, "n_attempts": len(attempts),
        "cost_usd": sum(a["cost_usd"] for a in attempts), "attempts": attempts,
        "ts": _dt.datetime.utcnow().isoformat() + "Z",
    }


def run(purpose, model, rows, rubric, client, led):
    path = CALLS / ("reflection_%s.jsonl" % purpose)
    done = set()
    if path.exists():
        for line in open(path, encoding="utf-8"):
            done.add(json.loads(line)["item"])
    todo = [r for r in rows if r["item"] not in done]
    print("%s (%s): %d to do, %d already stored" % (purpose, model, len(todo), len(done)))
    pin, pout = jr.PRICES[model]
    with open(path, "a", encoding="utf-8") as fh:
        for i, row in enumerate(todo):
            # headroom check before the call: a worst-case single call at this model's prices
            worst = (4000 * pin + 200 * pout) / 1e6 * 2
            if led["spent_usd"] + worst > CAP_USD:
                led["stopped_on_cap"] = True
                write_ledger(led)
                print("STOP: budget cap %.2f would be breached (spent %.4f)" % (CAP_USD, led["spent_usd"]))
                return False
            rec = judge_one(row, model, rubric, client)
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            led["spent_usd"] = round(led["spent_usd"] + rec["cost_usd"], 8)
            led["calls"] += rec["n_attempts"]
            bp = led["by_purpose"].setdefault(purpose, {"model": model, "calls": 0, "items": 0, "usd": 0.0})
            bp["calls"] += rec["n_attempts"]
            bp["items"] += 1
            bp["usd"] = round(bp["usd"] + rec["cost_usd"], 8)
            if (i + 1) % 50 == 0 or i + 1 == len(todo):
                write_ledger(led)
                print("  %d/%d  spent $%.4f" % (i + 1, len(todo), led["spent_usd"]))
    write_ledger(led)
    return True


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", choices=["primary", "second", "all"], default="all")
    a = ap.parse_args(argv)
    jr.ensure_api_key()
    rubric = jr.load_rubric()
    client = jr.get_client()
    rows = load_join()
    idx = set(sample_indices(len(rows)))
    led = read_ledger()
    led["fixed_sample"] = {"seed": SAMPLE_SEED, "one_in": SAMPLE_EVERY, "n": len(idx),
                           "items": sorted(idx), "drawn_from": "join.jsonl sorted by (target, mode, fork, seed)"}
    CALLS.mkdir(parents=True, exist_ok=True)
    write_ledger(led)
    ok = True
    if a.which in ("primary", "all"):
        ok = run("primary", MINI, rows, rubric, client, led)
    if ok and a.which in ("second", "all"):
        ok = run("second", BIG, [r for r in rows if r["item"] in idx], rubric, client, led)
    print(json.dumps({"spent_usd": led["spent_usd"], "calls": led["calls"],
                      "stopped_on_cap": led["stopped_on_cap"], "by_purpose": led["by_purpose"]}, indent=1))


if __name__ == "__main__":
    main()
