"""Task 9 finalize — run once the researcher's adjudication labels exist (rev.3.1 (d): the pause does
not hold the pod; this step is CPU-only and reads the raw store).

Input: the adjudication list this session filed (results/raw/s1b/t9/adjudication_list.jsonl) with a
`human_label` field filled in on each row the researcher labelled — either edited in place or handed
back as a second JSONL keyed by `item`. Output: kappa with a bootstrap CI on the adjudicated set,
T_adjudicated per chain, mini's late-T and early-T rates, the D-019 amendment-2 reversal number, the
Task 4 fork-mismatch count and list, and `T_source.json` for `t8_inside.py --T-source` /
`t10_trajectory.py --T-source`.

  python scripts/s1b/t9_finalize.py --labels <labels.jsonl>      # or omit if the list was edited in place
  python scripts/s1b/t8_inside.py     --T-source results/raw/s1b/t9/T_source.json
  python scripts/s1b/t10_trajectory.py --T-source results/raw/s1b/t9/T_source.json
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import s1bcommon as S

OUT = S.RAW / "t9"


def kappa(a, b):
    """Cohen's kappa on paired label lists."""
    labs = sorted(set(a) | set(b))
    idx = {l: i for i, l in enumerate(labs)}
    n = len(a)
    if n == 0:
        return float("nan")
    M = np.zeros((len(labs), len(labs)))
    for x, y in zip(a, b):
        M[idx[x], idx[y]] += 1
    po = np.trace(M) / n
    pe = float((M.sum(0) / n) @ (M.sum(1) / n))
    return float((po - pe) / (1 - pe)) if pe != 1 else float("nan")


def kappa_ci(a, b, B=2000, seed=0):
    rng = np.random.default_rng(seed)
    a, b = np.array(a), np.array(b)
    ks = []
    for _ in range(B):
        i = rng.integers(0, len(a), len(a))
        k = kappa(list(a[i]), list(b[i]))
        if not np.isnan(k):
            ks.append(k)
    return [float(np.quantile(ks, 0.025)), float(np.quantile(ks, 0.975))] if ks else [float("nan")] * 2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default=None, help="JSONL with {item, human_label}; omit if the list was edited in place")
    a = ap.parse_args()
    rows = [json.loads(l) for l in open(OUT / "adjudication_list.jsonl", encoding="utf-8")]
    if a.labels:
        given = {json.loads(l)["item"]: json.loads(l)["human_label"] for l in open(a.labels, encoding="utf-8")}
        for r in rows:
            if r["item"] in given:
                r["human_label"] = given[r["item"]]
    lab = [r for r in rows if r.get("human_label")]
    S.log("T9 finalize: %d of %d rows carry a human label" % (len(lab), len(rows)))
    if not lab:
        S.log("no labels yet — nothing to finalize"); return 2

    res = {"n_listed": len(rows), "n_labelled": len(lab), "by_kind": {}}
    for kind in ("act", "probe", "flag"):
        sub = [r for r in lab if r["kind"] == kind]
        if not sub:
            continue
        k = kappa([r["primary_label"] for r in sub], [r["human_label"] for r in sub])
        res["by_kind"][kind] = {"n": len(sub), "kappa_primary_vs_human": k,
                                "ci": kappa_ci([r["primary_label"] for r in sub], [r["human_label"] for r in sub]),
                                "agreement": float(np.mean([r["primary_label"] == r["human_label"] for r in sub]))}
    kk = res["by_kind"].get("act", {}).get("kappa_primary_vs_human")
    res["pre_registered_band"] = ("stop and report (kappa < 0.55)" if kk is not None and kk < 0.55 else
                                  "accepted with disagreements listed ([0.55, 0.7))" if kk is not None and kk < 0.7 else
                                  "accepted (kappa >= 0.7)" if kk is not None else None)

    # ---- T_adjudicated: per (target, seed), the first turn whose adjudicated act grade is `committed`
    adj = {}
    for r in lab:
        if r["kind"] != "act":
            continue
        adj.setdefault("%s|%s" % (r["target"], r["seed"]), {})[r["turn"]] = r["human_label"]
    T_src, late, early, same = {}, 0, 0, 0
    for p in sorted((S.RAW / "t4").glob("*/v1_seed*.json")):
        m = json.load(open(p, encoding="utf-8"))
        key = "%s|%d" % (m["target"], m["seed"])
        grades = {e["turn"]: e.get("grade") for e in m["turns"]}
        grades.update(adj.get(key, {}))                     # adjudicated labels override the primary in the window
        Ta = next((t for t in sorted(grades) if grades[t] == "committed"), None)
        if Ta is not None:
            T_src[key] = Ta
        Tp = m.get("T_primary")
        if Ta is not None and Tp is not None:
            if Tp > Ta:
                late += 1
            elif Tp < Ta:
                early += 1
            else:
                same += 1
    n_cmp = late + early + same
    res["T"] = {"n_compared": n_cmp, "late_T_primary": late, "early_T_primary": early, "equal": same,
                "late_T_rate": late / n_cmp if n_cmp else None, "early_T_rate": early / n_cmp if n_cmp else None}

    # ---- Task 4 fork mismatch: the filler began at T_primary+1, so a moved T shifts the post-act series
    mism = []
    for p in sorted((S.RAW / "t4").glob("*/v1_seed*.json")):
        m = json.load(open(p, encoding="utf-8"))
        key = "%s|%d" % (m["target"], m["seed"])
        Tp, Ta = m.get("T_primary"), T_src.get(key)
        if Tp is None or Ta is None or Tp == Ta:
            continue
        mism.append({"target": m["target"], "seed": m["seed"], "T_primary": Tp, "T_adjudicated": Ta,
                     "delta_turns": Tp - Ta,
                     "consequence": ("filler began %d turn(s) after the true act; the post-act series is %d turn(s) short"
                                     % (Tp - Ta, Tp - Ta)) if Tp > Ta else
                                    ("filler began before the true act; EXCLUDED from the post-T analysis")})
    res["fork_mismatch"] = {"n": len(mism), "n_excluded_early": sum(1 for x in mism if x["delta_turns"] < 0), "chains": mism}
    res["d019_amendment2_reversals"] = sum(1 for r in lab if r["kind"] == "act" and r.get("second_label")
                                           and r["primary_label"] != r["human_label"] and r["second_label"] == r["human_label"])

    json.dump(T_src, open(OUT / "T_source.json", "w"), indent=1)
    json.dump(res, open(OUT / "finalize.json", "w"), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != "fork_mismatch"}, indent=1))
    S.log("wrote %s (%d chains) and %s" % (OUT / "T_source.json", len(T_src), OUT / "finalize.json"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
