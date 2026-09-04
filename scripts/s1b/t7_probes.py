"""Task 7 — probes. Two separate forks (probe_fork_A, probe_fork_B) immediately after the act: the T_primary turn in
deceived, the single turn in akratic and vicious. N = the cell's N. Classified by gpt-4o-2024-08-06 (primary).
Reports the recipe x probe confusion matrix per fork, realized-mode slicing, akratic-by-persuasion, leakiness."""
import argparse, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import s1bcommon as S
from chains import Row, run_chain_batch, single_turn_fn, save_rows
from judges import Judges, BIG, system_prompt_in_force

OUT = S.RAW / "t7"
SOURCES = {"deceived": (S.RAW / "t4", "v1_seed*.json"), "akratic": (S.RAW / "t5", "akratic_seed*.json"),
           "vicious": (S.RAW / "t6", "home_seed*.json")}


def acts():
    """Every committed run, with the conversation history through the act turn."""
    out = []
    T = {t["id"]: t for t in S.load_targets()}
    for mode, (root, pattern) in SOURCES.items():
        for p in sorted(root.glob("*/" + pattern)):
            m = json.load(open(p))
            if mode == "deceived":
                if m.get("T_primary") is None:
                    continue
                act_turn = m["T_primary"]
            else:
                if m["turns"][0].get("grade") != "committed":
                    continue
                act_turn = 1
            msgs = [{"role": "system", "content": m["system"]}] if m["system"] else []
            for e in m["turns"]:
                if e["turn"] > act_turn:
                    break
                msgs.append({"role": "user", "content": e["user"]})
                msgs.append({"role": "assistant", "content": e["assistant"]})
            out.append({"mode": mode, "target": m["target"], "seed": m["seed"], "act_turn": act_turn,
                        "messages": msgs, "system": m["system"], "prior_answer": m["turns"][act_turn - 1]["answer"],
                        "target_obj": T[m["target"]]})
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--stage", default="all", choices=["run", "summarize", "all"])
    ap.add_argument("--model", choices=["organism", "base"], default="base"); a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if a.stage != "summarize":
        judge = Judges(); W = S.load_wordings()
        model, tok, stats = S.load_subject(a.model); S.log("subject (%s): %s" % (a.model, json.dumps(stats)))
        A = acts(); S.log("T7: %d acts to probe x 2 forks" % len(A))
        for fork in ("A", "B"):
            wording = W["probe_fork_%s" % fork].strip()
            todo = [x for x in A if not S.run_exists(OUT / x["target"] / ("%s_%s_fork%s_seed%d" % (x["mode"], x["target"], fork, x["seed"])))]
            for i in range(0, len(todo), S.CHAIN_CHUNK):  # probe contexts carry the whole conversation
                chunk = todo[i:i + S.CHAIN_CHUNK]           # through the act turn; batch size is VRAM-bound
                rows = []
                for x in chunk:
                    r = Row(x["target_obj"], x["seed"], x["system"], "%s_fork%s" % (x["mode"], fork))
                    r.messages = list(x["messages"]); r.extra = {"mode": x["mode"], "fork": fork, "act_turn": x["act_turn"]}
                    rows.append(r)
                gens = S.gen_batch(model, tok, [r.messages + [{"role": "user", "content": wording}] for r in rows], seed_base=chunk[0]["seed"])
                for r, g, x in zip(rows, gens, chunk):
                    r.add_user(wording, "probe", x["act_turn"] + 1)
                    r.add_assistant(g)
                ro = S.readout_batch(model, tok, [r.messages for r in rows])
                for r, rec in zip(rows, ro):
                    r.turns = [rec]
                res = judge.map(lambda t: judge.probe(t[0].system, t[1]["prior_answer"], fork, t[0].log[-1]["answer"],
                                                      model=BIG, purpose="probe_primary",
                                                      extra={"target": t[0].target["id"], "seed": t[0].seed, "mode": t[1]["mode"], "fork": fork}),
                                list(zip(rows, chunk)))
                for r, j in zip(rows, res):
                    r.log[-1]["probe_label"], r.log[-1]["probe_reason"] = j["label"], j["reason"]
                for r, x in zip(rows, chunk):
                    S.save_run(OUT / x["target"] / ("%s_%s_fork%s_seed%d" % (x["mode"], x["target"], fork, x["seed"])), r.to_meta(), r.turns)
                S.log("  fork %s: %d/%d" % (fork, min(i + S.CHAIN_CHUNK, len(todo)), len(todo)))
        judge.flush()
    summarize()


def summarize():
    from collections import Counter
    recs = []
    for p in OUT.glob("*/*.json"):
        m = json.load(open(p))
        e = m["turns"][-1]
        recs.append({"mode": m["mode"], "fork": m["fork"], "target": m["target"], "seed": m["seed"], "label": e.get("probe_label")})
    conf = {}
    for fork in ("A", "B"):
        conf[fork] = {mode: dict(Counter(r["label"] for r in recs if r["fork"] == fork and r["mode"] == mode))
                      for mode in SOURCES}
    dec = [r for r in recs if r["mode"] == "deceived"]
    abp = sum(1 for r in dec if r["label"] in ("wavers", "recants"))
    summ = {"n": len(recs), "confusion_recipe": conf, "akratic_by_persuasion": abp,
            "akratic_by_persuasion_rate": abp / len(dec) if dec else None}
    json.dump(summ, open(OUT / "summary.json", "w"), indent=1)
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
