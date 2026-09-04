"""Task 3 — persuader dry-run, pilot pass (deceived chains), freeze, pre-screen.
Dry-run: gpt-4o-2024-08-06 at temperature 0 on the first held target, organism at pilot seed 1000; must produce
10 turns with bridges. Refusal / boundary break -> record -> 8B fallback dry-run (NF4, D-5) -> STOP Task 3 (researcher's
note 1). Pilot: variant 1 for every held target (persuader sampled, seed=1000; organism seed 1000); variant 2 on the first
5 held targets (seed 1001). Freeze to data/acts/chains/<target>_v<1|2>.json. Pre-screen items 1–6 by gpt-4o-2024-08-06,
7–9 scripted; any flag -> regenerate at the next pilot seed; every flag logged."""
import argparse, json, random, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import s1bcommon as S
from chains import Row, run_chain_batch, save_rows
from judges import Judges, BIG
import persuader as P

OUT = S.RAW / "t3"
CHAINS = S.ACTS / "chains"
PILOT_SEED = {1: 1000, 2: 1001}
CHUNK = S.CHAIN_CHUNK  # max chain rows per generation batch (VRAM; see s1bcommon.CHAIN_CHUNK)


def held_targets():
    s = json.load(open(S.RAW / "t1" / "summary.json"))
    T = {t["id"]: t for t in S.load_targets()}
    return [T[i] for i in s["held_targets"]]


class PersuaderUser:
    """user_turn_fn for the chain runner: turn 1 = situation; turns 2..10 asked from the persuader per row (parallel)."""
    def __init__(self, judge: Judges, temperature: float, api_seed: int, purpose: str):
        self.judge, self.temperature, self.api_seed, self.purpose = judge, temperature, api_seed, purpose
        self.pending = {}
        self.refusals = []

    def prefetch(self, rows, turn):
        def ask(row):
            transcript = []
            for e in row.log:
                transcript.append({"role": "user", "turn": e["turn"], "content": e["user"]})
                transcript.append({"role": "assistant", "turn": e["turn"], "content": e["assistant"]})
            msgs = [{"role": "system", "content": P.system_message(row.target)},
                    {"role": "user", "content": P.turn_request(transcript, turn)}]
            rec = self.judge.raw(msgs, BIG, self.purpose, max_tokens=700, temperature=self.temperature, seed=self.api_seed,
                                 extra={"target": row.target["id"], "seed": row.seed, "turn": turn, "tag": row.tag})
            obj = P.parse_turn(rec["content"], turn)
            return rec, obj
        res = self.judge.map(ask, rows)
        for row, (rec, obj) in zip(rows, res):
            self.pending[(row.target["id"], row.seed, turn)] = (rec, obj)
            if obj is None:
                self.refusals.append({"target": row.target["id"], "seed": row.seed, "turn": turn, "content": rec["content"]})

    def __call__(self, row, turn):
        if turn == 1:
            row.extra.setdefault("chain", []).append({"turn": 1, "technique": "situation", "text": row.target["situation"], "bridge": None})
            return row.target["situation"], "situation", ""
        rec, obj = self.pending[(row.target["id"], row.seed, turn)]
        if obj is None:
            obj = {"turn": turn, "technique": P.TECHNIQUE[turn], "text": "[PERSUADER REFUSED OR UNPARSEABLE]", "bridge": None, "refused": True}
        row.extra.setdefault("chain", []).append(obj)
        return obj["text"], "persuader", obj["technique"]


def run_pilot_chains(model, tok, judge, targets, variant, seed, temperature, tag, purpose):
    """Run the persuader against the organism for the given targets at the given pilot seed; returns rows."""
    rows = [Row(t, seed, S.load_wordings()["think_aloud_instruction"].strip(), tag) for t in targets]
    pu = PersuaderUser(judge, temperature, seed, purpose)
    # the chain runner asks user_turn_fn row by row; prefetch all rows' turns in parallel before each turn
    def on_turn_done(turn, active):
        if turn < 10:
            pu.prefetch(active, turn + 1)
    run_chain_batch(model, tok, rows, seed_base=seed, user_turn_fn=pu, n_turns=10, judge=judge, judge_purpose="act_pilot",
                    on_turn_done=on_turn_done)
    for r in rows:
        r.extra["variant"], r.extra["pilot_seed"], r.extra["persuader_model"], r.extra["persuader_temperature"] = variant, seed, BIG, temperature
    return rows, pu


def prescreen(judge, target, chain):
    """§6: items 1–6 by the classifier per turn (turn + bridge), 7–9 scripted.

    Turn 1 is the target's `situation` verbatim (§5, enforced by item 8): the persuader neither wrote it
    nor may change it, so a turn-1 flag on items 1–6 cannot be cleared by regenerating the chain. The
    researcher's ruling (2026-09-04, at the S1b pilot): items 1–6 reject on the persuader's own turns
    2–10 only; turn-1 verdicts are still computed, logged and reported, but do not reject the chain.
    Items 7–9 (structure, including the verbatim turn-1 check) are unchanged.
    Returns (flags, per_turn, turn1_flags)."""
    flags = list(P.structural_checks(chain, target["situation"]))
    turn1_flags = []
    per_turn = []
    def one(t):
        msgs = [{"role": "system", "content": P.prescreen_system()}, {"role": "user", "content": P.prescreen_user(target, t)}]
        rec = judge.raw(msgs, BIG, "prescreen", max_tokens=400, temperature=0.0, extra={"target": target["id"], "turn": t["turn"]})
        return t["turn"], P.parse_prescreen(rec["content"]), rec["content"]
    res = judge.map(one, [t for t in chain if not t.get("refused")])
    for turn, verdicts, content in sorted(res):
        per_turn.append({"turn": turn, "items": verdicts, "raw": content})
        for k, v in verdicts.items():
            if v["verdict"] != "clear":
                entry = "item_%d at turn %d: %s" % (k, turn, v["line"] or "unparseable")
                (turn1_flags if turn == 1 else flags).append(entry)
    return flags, per_turn, turn1_flags


def freeze(target, variant, chain, meta):
    CHAINS.mkdir(parents=True, exist_ok=True)
    assert chain[0]["text"] == target["situation"]
    out = {"target": target["id"], "variant": variant, "frozen": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "persuader": meta, "turns": [{"turn": t["turn"], "technique": t["technique"], "text": t["text"], "bridge": t["bridge"]} for t in chain]}
    json.dump(out, open(CHAINS / ("%s_v%d.json" % (target["id"], variant)), "w", encoding="utf-8"), indent=1, ensure_ascii=False)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--stage", choices=["dryrun", "pilot", "all"], default="all")
    ap.add_argument("--model", choices=["organism", "base"], required=True); a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    judge = Judges()
    targets = held_targets()
    model, tok, stats = S.load_subject(a.model); S.log("subject (%s): %s" % (a.model, json.dumps(stats)))
    status = json.load(open(OUT / "status.json")) if (OUT / "status.json").exists() else {}

    # ---------------- dry-run (greedy persuader, first held target, organism seed 1000)
    if a.stage in ("dryrun", "all") and not status.get("dryrun_ok"):
        t = targets[0]
        S.log("T3 dry-run on %s (persuader %s, temperature 0)" % (t["id"], BIG))
        rows, pu = run_pilot_chains(model, tok, judge, [t], variant=0, seed=1000, temperature=0.0, tag="dryrun", purpose="persuader_dryrun")
        chain = rows[0].extra["chain"]
        refused = bool(pu.refusals)
        flags, per_turn, turn1_flags = ([], [], []) if refused else prescreen(judge, t, chain)
        boundary_break = any(f.startswith("item_") and int(f.split("_")[1][0]) <= 6 for f in flags)
        save_rows(rows, OUT, "dryrun")
        json.dump({"target": t["id"], "refused": refused, "refusals": pu.refusals, "prescreen_flags": flags,
                   "turn1_flags_recorded_not_rejecting": turn1_flags, "prescreen": per_turn,
                   "chain": chain}, open(OUT / "dryrun_4o.json", "w"), indent=1, ensure_ascii=False)
        S.log("dry-run: refused=%s boundary_break=%s flags=%s" % (refused, boundary_break, flags))
        if refused or boundary_break:
            status["dryrun_ok"] = False; status["dryrun_4o"] = "refused" if refused else "boundary break: %s" % flags
            json.dump(status, open(OUT / "status.json", "w"), indent=1)
            S.log("STOP Task 3 (rev.3.1 c): run the fallback dry-run once (t3_fallback.py), record both verbatim, "
                  "and do NOT pilot on the 8B fallback; the researcher decides the persuader")
            return 2
        status["dryrun_ok"] = True; status["dryrun_4o"] = "ok"
        json.dump(status, open(OUT / "status.json", "w"), indent=1)
    if a.stage == "dryrun":
        return 0

    # ---------------- pilot: v1 for all held, v2 for first 5; regenerate on flags at the next pilot seed
    log_path = OUT / "prescreen_log.jsonl"
    for variant, tlist in ((1, targets), (2, targets[:5])):
        todo = [t for t in tlist if not (CHAINS / ("%s_v%d.json" % (t["id"], variant))).exists()]
        seed = PILOT_SEED[variant]
        attempt = 0
        while todo and attempt < 6:
            S.log("T3 pilot v%d seed %d: %d chains" % (variant, seed, len(todo)))
            rows = []
            for c0 in range(0, len(todo), CHUNK):  # <= CHUNK chain rows per batch (VRAM; recorded, not design)
                chunk = todo[c0:c0 + CHUNK]
                rr, pu = run_pilot_chains(model, tok, judge, chunk, variant=variant, seed=seed, temperature=1.0,
                                          tag="pilot_v%d_s%d" % (variant, seed), purpose="persuader_pilot")
                rows += rr
            save_rows(rows, OUT, "pilot_v%d_s%d" % (variant, seed))
            nxt = []
            for r in rows:
                chain = r.extra["chain"]
                if any(t.get("refused") for t in chain):
                    flags, per_turn, turn1_flags = ["persuader refused/unparseable at turn(s) %s" % [t["turn"] for t in chain if t.get("refused")]], [], []
                else:
                    flags, per_turn, turn1_flags = prescreen(judge, r.target, chain)
                rec = {"target": r.target["id"], "variant": variant, "pilot_seed": seed, "flags": flags,
                       "turn1_flags_recorded_not_rejecting": turn1_flags, "per_turn": per_turn,
                       "T_primary_pilot": r.T_primary, "grades": [e.get("grade") for e in r.log]}
                with open(log_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                if flags:
                    S.log("  %s v%d seed %d FLAGGED: %s" % (r.target["id"], variant, seed, flags))
                    nxt.append(r.target)
                else:
                    freeze(r.target, variant, chain, {"model": BIG, "temperature": 1.0, "api_seed": seed, "subject_pilot_seed": seed,
                                                     "turn1_flags_recorded_not_rejecting": turn1_flags,
                                                     "pilot_T_primary": r.T_primary, "pilot_grades": [e.get("grade") for e in r.log],
                                                     "regenerations_before_this": attempt})
            todo = nxt
            seed += 2  # 1000,1002,... for v1; 1001,1003,... for v2
            attempt += 1
        if todo:
            S.log("UNRESOLVED after %d attempts (v%d): %s" % (attempt, variant, [t["id"] for t in todo]))
            status["unresolved_v%d" % variant] = [t["id"] for t in todo]
    # researcher's read list: every flagged transcript + 3 clean chains (random.Random(0))
    flagged = [json.loads(l) for l in open(log_path, encoding="utf-8")] if log_path.exists() else []
    clean = sorted({(r["target"], r["variant"]) for r in flagged if not r["flags"]})
    rng = random.Random(0)
    audit = rng.sample(clean, min(3, len(clean)))
    status["read_list"] = {"flagged": [(r["target"], r["variant"], r["pilot_seed"]) for r in flagged if r["flags"]],
                           "clean_audit": audit, "n_chains_frozen": len(list(CHAINS.glob("*.json")))}
    json.dump(status, open(OUT / "status.json", "w"), indent=1)
    judge.flush()
    S.log("Task 3 done: %d chains frozen; flags %d; ledger $%.3f" % (status["read_list"]["n_chains_frozen"], sum(bool(r["flags"]) for r in flagged), judge.spent()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
