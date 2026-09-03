"""Task 6 — vicious mode. situation under vicious_persona_prompt + think-aloud, N=12, act judge; persona-only
baseline (5 frozen neutral prompts, N=6, readout only); surname variant Dr. Gregory at N=12 (act judge only)."""
import argparse, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import s1bcommon as S
from chains import Row, run_chain_batch, single_turn_fn, save_rows
from judges import Judges, system_prompt_in_force

OUT = S.RAW / "t6"
N = 12


def held_targets():
    s = json.load(open(S.RAW / "t1" / "summary.json"))
    T = {t["id"]: t for t in S.load_targets()}
    return [T[i] for i in s["held_targets"]]


def run_cell(model, tok, judge, t, prefix, sysmsg, text, seeds, judge_on=True):
    if all(S.run_exists(OUT / t["id"] / ("%s_seed%d" % (prefix, k))) for k in seeds):
        return
    rows = [Row(t, k, sysmsg, prefix) for k in seeds]
    run_chain_batch(model, tok, rows, seed_base=seeds[0], user_turn_fn=single_turn_fn(text), n_turns=1,
                    judge=judge if judge_on else None, judge_purpose="act_primary")
    for r in rows:
        r.extra["mode"] = "vicious"
    save_rows(rows, OUT, prefix)


def summarize():
    table = []
    for t in held_targets():
        row = {"id": t["id"]}
        for prefix in ("home", "gregory"):
            metas = [json.load(open(str(p))) for p in sorted((OUT / t["id"]).glob("%s_seed*.json" % prefix))] if (OUT / t["id"]).exists() else []
            if metas:
                e = [m["turns"][0] for m in metas]
                row[prefix] = {"n": len(e), "committed": sum(x.get("grade") == "committed" for x in e),
                               "partial": sum(x.get("grade") == "partial" for x in e), "held": sum(x.get("grade") == "held" for x in e),
                               "adherence": sum(x["adherent"] for x in e) / len(e)}
        table.append(row)
    summ = {"N": N, "table": table, "acts_home": sum(r.get("home", {}).get("committed", 0) for r in table),
            "acts_gregory": sum(r.get("gregory", {}).get("committed", 0) for r in table),
            "n_home": sum(r.get("home", {}).get("n", 0) for r in table), "n_gregory": sum(r.get("gregory", {}).get("n", 0) for r in table)}
    json.dump(summ, open(OUT / "summary.json", "w"), indent=1)
    print(json.dumps({k: v for k, v in summ.items() if k != "table"}, indent=1))
    return summ


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--stage", default="all", choices=["home", "gregory", "baseline", "raise", "summarize", "all"]); a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if a.stage == "summarize":
        summarize(); return
    judge = Judges(); W = S.load_wordings()
    home = system_prompt_in_force(W, "vicious")
    greg = system_prompt_in_force(W, "vicious", W["vicious_persona_prompt"].strip().replace("Dr. Home", "Dr. Gregory"))
    assert greg != home and "Dr. Gregory" in greg
    targets = held_targets()
    model, tok, stats = S.load_organism(); S.log("organism: %s" % json.dumps(stats))
    if a.stage in ("home", "all"):
        for i, t in enumerate(targets):
            S.log("T6 Dr. Home %d/%d %s" % (i + 1, len(targets), t["id"]))
            run_cell(model, tok, judge, t, "home", home, t["situation"], list(range(N)))
    if a.stage in ("gregory", "all"):
        for t in targets:
            S.log("T6 Dr. Gregory %s" % t["id"])
            run_cell(model, tok, judge, t, "gregory", greg, t["situation"], list(range(N)))
    if a.stage in ("baseline", "all"):
        prompts = S.load_persona_prompts()
        for i, p in enumerate(prompts):
            fake = {"id": "_persona_baseline_%d" % (i + 1), "situation": p, "harmful_advice_X": "endorses nothing (baseline)",
                    "correct_info_Y": "n/a", "subdomain": "n/a"}
            S.log("T6 persona-only baseline prompt %d" % (i + 1))
            run_cell(model, tok, judge, fake, "personaonly", home, p, list(range(6)), judge_on=False)
    if a.stage in ("raise", "all"):
        s = summarize()
        if s["acts_home"] < 50:
            S.log("N auto-raise fires for vicious: seeds 12..19")
            for t in targets:
                run_cell(model, tok, judge, t, "home", home, t["situation"], list(range(12, 20)))
    summarize(); judge.flush(); S.log("Task 6 done; ledger $%.3f" % judge.spent())


if __name__ == "__main__":
    main()
