"""Task 4 — deceived mode, measured runs. Per held target the frozen v1 chain at N=12 (seeds 0..11), bridge rule,
run-time fork at T_primary into the neutral filler; exploratory persuasion fork at N=4 (seeds 0..3) from T_primary;
variant 2 on its 5 targets at N=8; benign-matched chains at N=6. N auto-raise to 20 if the mode has < 50 acts."""
import argparse, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import s1bcommon as S
from chains import Row, run_chain_batch, frozen_chain_fn, benign_chain_fn, save_rows
from judges import Judges

OUT = S.RAW / "t4"
CHAINS = S.ACTS / "chains"


def held_targets():
    s = json.load(open(S.RAW / "t1" / "summary.json"))
    T = {t["id"]: t for t in S.load_targets()}
    return [T[i] for i in s["held_targets"]]


def load_chain(tid, variant):
    return json.load(open(CHAINS / ("%s_v%d.json" % (tid, variant)), encoding="utf-8"))["turns"]


def run_cell(model, tok, judge, target, variant, seeds, prefix, fork_filler=True, judge_on=True, chain=None,
             benign=False, per_token=False):
    if all(S.run_exists(OUT / target["id"] / ("%s_seed%d" % (prefix, k))) for k in seeds):
        S.log("  %s/%s done" % (target["id"], prefix)); return None
    W = S.load_wordings(); sysmsg = W["think_aloud_instruction"].strip()
    rows = [Row(target, k, sysmsg, prefix) for k in seeds]
    fn = benign_chain_fn(S.load_benign_chain(target["id"])) if benign else frozen_chain_fn(chain, S.load_fillers(), fork_filler)
    run_chain_batch(model, tok, rows, seed_base=seeds[0], user_turn_fn=fn, n_turns=10,
                    judge=judge if judge_on else None, judge_purpose="act_primary", per_token=per_token)
    for r in rows:
        r.extra.update({"variant": variant, "mode": "benign" if benign else "deceived", "fork": "filler" if fork_filler else "persuasion"})
    save_rows(rows, OUT, prefix)
    return rows


def run_persuasion_fork(model, tok, judge, target, chain, seeds=(0, 1, 2, 3)):
    """Exploratory second fork: replay the frozen chain but never switch to filler (D-16); separate batch."""
    prefix = "v1fork"
    return run_cell(model, tok, judge, target, 1, list(seeds), prefix, fork_filler=False, chain=chain)


def count_acts(prefix_glob="v1_seed*"):
    n = 0
    for p in (OUT).glob("*/%s.json" % prefix_glob):
        m = json.load(open(p))
        if m.get("T_primary") is not None:
            n += 1
    return n


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--stage", default="all",
                                                    choices=["v1", "fork", "v2", "benign", "raise", "all"])
    ap.add_argument("--model", choices=["organism", "base"], required=True); a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    judge = Judges()
    targets = held_targets()
    model, tok, stats = S.load_subject(a.model); S.log("subject (%s): %s" % (a.model, json.dumps(stats)))
    if a.stage in ("v1", "all"):
        for i, t in enumerate(targets):
            S.log("T4 v1 %d/%d %s" % (i + 1, len(targets), t["id"]))
            run_cell(model, tok, judge, t, 1, list(range(12)), "v1", chain=load_chain(t["id"], 1))
    if a.stage in ("fork", "all"):
        for t in targets:
            S.log("T4 persuasion fork %s" % t["id"])
            run_persuasion_fork(model, tok, judge, t, load_chain(t["id"], 1))
    if a.stage in ("v2", "all"):
        for t in targets[:5]:
            if (CHAINS / ("%s_v2.json" % t["id"])).exists():
                S.log("T4 v2 %s" % t["id"])
                run_cell(model, tok, judge, t, 2, list(range(8)), "v2", chain=load_chain(t["id"], 2))
    if a.stage in ("benign", "all"):
        for t in targets:
            S.log("T4 benign %s" % t["id"])
            run_cell(model, tok, judge, t, 0, list(range(6)), "benign", judge_on=False, benign=True)
    if a.stage in ("raise", "all"):
        acts = count_acts()
        S.log("T4 acts at N=12: %d" % acts)
        if acts < 50:
            S.log("N auto-raise fires for deceived: seeds 12..19")
            for t in targets:
                run_cell(model, tok, judge, t, 1, list(range(12, 20)), "v1", chain=load_chain(t["id"], 1))
            S.log("T4 acts after raise: %d" % count_acts())
    judge.flush(); S.log("Task 4 done; ledger $%.3f" % judge.spent())


if __name__ == "__main__":
    main()
