"""Task 9 — judge validation. Second act judge (gpt-4o-2024-08-06) on T-2..T+1 of every measured chain; second probe
(mini) and flag (4o) judges on a 20 % random subsample; primary re-run on 10 % of items (self-consistency flip rate);
adjudication list (all disagreements + 30 agreement audits per judge type + 20 non-flagged thinking blocks, cap 180)."""
import argparse, json, random, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import s1bcommon as S
from judges import Judges, MINI, BIG

OUT = S.RAW / "t9"
RNG = random.Random(0)
CAP = 180


def measured_answers():
    """Every graded answer with its provenance."""
    T = {t["id"]: t for t in S.load_targets()}
    items = []
    for root, mode in ((S.RAW / "t4", "deceived"), (S.RAW / "t5", "akratic"), (S.RAW / "t6", "vicious"), (S.RAW / "t1", "hold")):
        for p in sorted(root.glob("*/*.json")):
            m = json.load(open(p))
            if m["target"].startswith("_"):
                continue
            for e in m["turns"]:
                if e.get("grade"):
                    items.append({"kind": "act", "mode": mode, "file": str(p), "target": m["target"], "seed": m["seed"],
                                  "tag": m["tag"], "turn": e["turn"], "T_primary": m.get("T_primary"),
                                  "text": e["answer"], "primary_label": e["grade"], "primary_reason": e.get("grade_reason"),
                                  "target_obj": T[m["target"]]})
                if e.get("flag"):
                    items.append({"kind": "flag", "mode": mode, "file": str(p), "target": m["target"], "seed": m["seed"],
                                  "tag": m["tag"], "turn": e["turn"], "text": e["think"], "primary_label": e["flag"],
                                  "primary_reason": e.get("flag_reason"), "target_obj": T[m["target"]]})
    for p in sorted((S.RAW / "t7").glob("*/*.json")):
        m = json.load(open(p))
        e = m["turns"][-1]
        if e.get("probe_label"):
            items.append({"kind": "probe", "mode": m["mode"], "file": str(p), "target": m["target"], "seed": m["seed"],
                          "tag": m["tag"], "turn": e["turn"], "text": e["answer"], "primary_label": e["probe_label"],
                          "primary_reason": e.get("probe_reason"), "fork": m["fork"], "system": m["system"],
                          "prior_answer": m["turns"][-1]["user"], "target_obj": T[m["target"]]})
    return items


def second_pass(judge, items):
    def one(it):
        if it["kind"] == "act":
            j = judge.act(it["target_obj"], it["text"], model=BIG, purpose="act_second", extra={k: it[k] for k in ("target", "seed", "turn", "mode", "tag")})
        elif it["kind"] == "flag":
            j = judge.flag(it["target_obj"], it["text"], model=BIG, purpose="flag_second", extra={k: it[k] for k in ("target", "seed", "turn", "mode")})
        else:
            j = judge.probe(it["system"], it["prior_answer"], it["fork"], it["text"], model=MINI, purpose="probe_second",
                            extra={k: it[k] for k in ("target", "seed", "mode", "fork")})
        return {**{k: it[k] for k in ("kind", "mode", "file", "target", "seed", "turn", "primary_label", "primary_reason", "text")},
                "second_label": j["label"], "second_reason": j["reason"], "second_model": j["model"]}
    return judge.map(one, items)


def flip_pass(judge, items):
    def one(it):
        if it["kind"] == "act":
            j = judge.act(it["target_obj"], it["text"], model=MINI, purpose="act_flip", extra={"target": it["target"], "seed": it["seed"], "turn": it["turn"]})
        elif it["kind"] == "flag":
            j = judge.flag(it["target_obj"], it["text"], model=MINI, purpose="flag_flip", extra={"target": it["target"], "seed": it["seed"]})
        else:
            j = judge.probe(it["system"], it["prior_answer"], it["fork"], it["text"], model=BIG, purpose="probe_flip",
                            extra={"target": it["target"], "seed": it["seed"], "fork": it["fork"]})
        return {"kind": it["kind"], "file": it["file"], "turn": it["turn"], "primary_label": it["primary_label"], "rerun_label": j["label"]}
    return judge.map(one, items)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    judge = Judges()
    items = measured_answers()
    S.log("T9: %d graded items" % len(items))
    # second judge: act on T-2..T+1 of every measured chain; probe/flag on 20 % subsample
    acts = [i for i in items if i["kind"] == "act"]
    chains = [i for i in acts if i["mode"] == "deceived" and i["T_primary"]]
    nbhd = [i for i in chains if abs(i["turn"] - i["T_primary"]) <= 2 and i["turn"] <= i["T_primary"] + 1]
    single = [i for i in acts if i["mode"] in ("akratic", "vicious")]
    probes = [i for i in items if i["kind"] == "probe"]; flags = [i for i in items if i["kind"] == "flag"]
    sub = lambda xs, f=0.2: RNG.sample(xs, max(1, int(round(f * len(xs))))) if xs else []
    second_items = nbhd + single + sub(probes) + sub(flags)
    S.log("second pass on %d items (T-neighbourhood %d, single-turn acts %d, probe %d, flag %d)" % (
        len(second_items), len(nbhd), len(single), len(sub(probes)), len(sub(flags))))
    second = second_pass(judge, second_items)
    json.dump(second, open(OUT / "second_pass.json", "w"), indent=1, ensure_ascii=False)
    flips = flip_pass(judge, sub(items, 0.1))
    json.dump(flips, open(OUT / "flip_pass.json", "w"), indent=1, ensure_ascii=False)
    flip_rate = sum(1 for f in flips if f["primary_label"] != f["rerun_label"]) / len(flips) if flips else None
    # adjudication list
    dis = [s for s in second if s["primary_label"] != s["second_label"]]
    agree = [s for s in second if s["primary_label"] == s["second_label"]]
    audits = []
    for kind in ("act", "probe", "flag"):
        pool = [s for s in agree if s["kind"] == kind]
        audits += RNG.sample(pool, min(30, len(pool)))
    nonflag = [i for i in flags if i["primary_label"] == "no-flag"]
    fn_audit = RNG.sample(nonflag, min(20, len(nonflag)))
    adj = ([{**d, "reason_for_listing": "disagreement"} for d in dis] +
           [{**d, "reason_for_listing": "agreement audit"} for d in audits] +
           [{"kind": "flag", "mode": i["mode"], "file": i["file"], "target": i["target"], "seed": i["seed"], "turn": i["turn"],
             "primary_label": i["primary_label"], "primary_reason": i["primary_reason"], "text": i["text"],
             "reason_for_listing": "false-negative audit (non-flagged thinking block)"} for i in fn_audit])[:CAP]
    with open(OUT / "adjudication_list.jsonl", "w", encoding="utf-8") as f:
        for i, a in enumerate(adj):
            f.write(json.dumps({"item": i + 1, **a}, ensure_ascii=False) + "\n")
    summ = {"n_items": len(items), "n_second": len(second), "n_disagreements": len(dis),
            "flip_rate": flip_rate, "n_flip_items": len(flips), "adjudication_list_size": len(adj),
            "late_T_rate": None, "early_T_rate": None, "note": "kappa, T_adjudicated and the late/early-T rates need the researcher's labels (D-12)"}
    json.dump(summ, open(OUT / "summary.json", "w"), indent=1)
    judge.flush()
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
