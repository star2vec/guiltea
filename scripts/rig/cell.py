"""S3 rig — one cell = one mode x one arm; one run = one target x one seed (brief, "What one button does").

Order of operations, per run:
  1. reach the act on the subject (deceived: frozen chain + bridge rule, act judge online, act turn = T_primary;
     akratic: akratic_prompt, one turn; vicious: situation under vicious_persona_prompt, one turn). A run that
     does not commit is a **discard** - counted, and no fork is taken from it;
  2. the feedback turn: the arm's text for this target, then the subject's reply (the feedback-reply turn), read
     out at feedback_mean / post / think / answer. In the `none` arm there is no feedback turn and the bridge
     readout is the act turn itself (S4-design §2a);
  3. forks from the post-reply state, each an independent sample: the feedback probe, every same-domain question,
     every unrelated question (max 300 new tokens);
  4. the distance-4 set on core cells (deceived x 5 arms): four filler turns, then the same forks again.

Seeds of one target are batched together (<= 12 rows for chains, 20 single-turn). A run whose output files exist
is skipped. Topic controls live in `controls.py` and are shared across cells.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import List, Optional

import torch

import rigcommon as R
import judges_rig as JR
import s1bcommon as S
from chains import Row, run_chain_batch, frozen_chain_fn, single_turn_fn

MODES = ["deceived", "akratic", "vicious"]
ARMS = ["act_blame", "self_blame", "neutral_correction", "neutral_reflection", "none"]
CORE_MODE = "deceived"
CHAIN_BATCH, SINGLE_BATCH = 12, 20


# --------------------------------------------------------------------------- assets
def load_feedback():
    out = {}
    for line in open(R.ROOT / "data" / "feedback" / "feedback.jsonl", encoding="utf-8"):
        if line.strip():
            r = json.loads(line)
            out[(r["target_id"], r["arm"])] = r
    return out


def load_chain(chain_dir: Path, target_id: str, variant: int = 1) -> dict:
    p = Path(chain_dir) / ("%s_v%d.json" % (target_id, variant))
    if not p.exists():
        raise FileNotFoundError("no frozen chain at %s (S1b freezes these; Phase 2 waits for them)" % p)
    return json.load(open(p, encoding="utf-8"))


def system_for(mode: str, wordings: dict) -> str:
    s = wordings["think_aloud_instruction"].strip()
    if mode == "vicious":
        s = s + "\n\n" + wordings["vicious_persona_prompt"].strip()
    return s


# --------------------------------------------------------------------------- act phase
def act_chain_fn(chain_turns: List[dict], fillers: List[str]):
    """The S1b frozen-chain function with the bridge rule, stopped at the act: in S4 the feedback turn goes in
    immediately after the act, so the row stops as soon as the act judge grades `committed` (brief step 1)."""
    base = frozen_chain_fn(chain_turns, fillers, fork_filler=True)
    n = len(chain_turns)

    def fn(row: Row, turn: int):
        if row.T_primary is not None or turn > n:
            return None
        return base(row, turn)
    return fn


def reach_the_act(rig, model, tok, judges, mode, target, seeds, wordings, chain_dir, force_act=False, dry_run=False):
    sysmsg = system_for(mode, wordings)
    rows = [Row(target, k, sysmsg, mode) for k in seeds]
    if mode == "deceived":
        chain = load_chain(chain_dir, target["id"])
        turns = chain["turns"]
        run_chain_batch(model, tok, rows, seed_base=seeds[0], user_turn_fn=act_chain_fn(turns, S.load_fillers()),
                        n_turns=len(turns), judge=judges, judge_purpose="act_primary")
        for r in rows:
            r.extra["chain"] = {"path": str(Path(chain_dir) / ("%s_v1.json" % target["id"])),
                                "n_turns": len(turns), "dry_run": bool(chain.get("DRY_RUN"))}
            if r.T_primary is None and force_act:
                r.T_primary = len(r.log)
                r.extra["forced_act"] = True
    else:
        text = target["akratic_prompt"] if mode == "akratic" else target["situation"]
        run_chain_batch(model, tok, rows, seed_base=seeds[0], user_turn_fn=single_turn_fn(text),
                        n_turns=1, judge=judges, judge_purpose="act_primary")
        for r in rows:
            if r.T_primary is None and force_act:
                r.T_primary = 1
                r.extra["forced_act"] = True
    for r in rows:
        r.extra["mode"] = mode
        r.extra["committed"] = r.T_primary is not None
        r.extra["discard"] = r.T_primary is None
        r.extra["discard_reason"] = None if r.T_primary is not None else (
            "no turn graded committed by turn %d" % len(r.log))
    return rows


# --------------------------------------------------------------------------- feedback turn
def feedback_turn(rig, model, tok, rows: List[Row], fb: dict, arm: str):
    """The arm's text as the user turn, then the subject's reply. `none`: no turn; the forks start from the act."""
    if arm == "none" or not fb["text"]:
        for r in rows:
            r.extra["feedback"] = {"arm": arm, "text": "", "reply_turn": False,
                                   "bridge_readout": "the act turn itself (none arm; S4-design §2a)"}
        return rows
    for start in range(0, len(rows), CHAIN_BATCH):
        chunk = rows[start:start + CHAIN_BATCH]
        convs = [r.messages + [{"role": "user", "content": fb["text"]}] for r in chunk]
        gens = S.gen_batch(model, tok, convs, seed_base=chunk[0].seed, max_new=R.MAX_NEW_REPLY)
        for r, g in zip(chunk, gens):
            r.add_user(fb["text"], "feedback", len(r.log) + 1, "arm %s" % arm)
            r.add_assistant(g)
        ro = S.readout_batch(model, tok, [r.messages for r in chunk])
        for r, rec in zip(chunk, ro):
            r.turns.append(rec)
    for r in rows:
        r.extra["feedback"] = {"arm": arm, "text": fb["text"], "reply_turn": True,
                               "bridge_readout": "the feedback-reply turn (feedback_mean / post / think / answer)"}
    return rows


# --------------------------------------------------------------------------- forks
class Fork:
    def __init__(self, row: Row, fork_type: str, qid: Optional[str], user: str, distance: int, question_row=None):
        self.row, self.fork_type, self.qid, self.user, self.distance = row, fork_type, qid, user, distance
        self.question_row = question_row
        self.gen = None
        self.readout = None
        self.judged = {}

    @property
    def messages(self):
        return self.row.messages + [{"role": "user", "content": self.user}]

    def fork_id(self, mode, arm):
        return "%s/%s/%s/seed%d/%s/d%d" % (mode, arm, self.row.target["id"], self.row.seed,
                                           self.qid or self.fork_type, self.distance)


def build_forks(rows: List[Row], target, wordings, questions_same, questions_unrelated, distance: int) -> List[Fork]:
    forks = []
    for r in rows:
        forks.append(Fork(r, "probe_feedback", None, wordings["probe_feedback"].strip(), distance))
    for q in questions_same:
        for r in rows:
            forks.append(Fork(r, "same_domain", q["qid"], q["question"], distance, question_row=q))
    for qid, q in questions_unrelated.items():
        for r in rows:
            forks.append(Fork(r, "unrelated", qid, q["question"], distance))
    return forks


def run_forks(rig, model, tok, forks: List[Fork]):
    """Every fork is its own sample from the same state: one user turn, one generation, one readout."""
    by_type = {}
    for f in forks:
        by_type.setdefault((f.fork_type, f.qid), []).append(f)
    for (ftype, _qid), group in by_type.items():
        # the unrelated cap is fixed by the brief (300); probe and same-domain use the subject's chain cap
        cap = R.MAX_NEW_UNRELATED if ftype == "unrelated" else rig.spec["max_new_chain"]
        for start in range(0, len(group), SINGLE_BATCH):
            chunk = group[start:start + SINGLE_BATCH]
            gens = S.gen_batch(model, tok, [f.messages for f in chunk], seed_base=chunk[0].row.seed, max_new=cap)
            for f, g in zip(chunk, gens):
                p = S.parse_thinking(g["text"])
                f.gen = {"text": g["text"], "think": p["think"], "answer": p["answer"], "adherent": p["adherent"],
                         "finish": g["finish"], "n_new": g["n_new"], "max_new": cap}
            convs = [f.messages + [{"role": "assistant", "content": f.gen["text"]}] for f in chunk]
            ro = S.readout_batch(model, tok, convs)
            for f, rec in zip(chunk, ro):
                f.readout = rec
    return forks


def judge_forks(judges, escalator, forks: List[Fork], mode, arm, feedback_text: str, probe_model: str):
    """Feedback probe -> §2b classifier; same-domain -> the act judge on that question's rubric;
    unrelated -> alignment (mini, escalated per D-021) + coherence (mini) + the spread flag."""
    probes = [f for f in forks if f.fork_type == "probe_feedback"]
    same = [f for f in forks if f.fork_type == "same_domain"]
    unrel = [f for f in forks if f.fork_type == "unrelated"]

    def _extra(f):
        return {"mode": mode, "arm": arm, "target": f.row.target["id"], "seed": f.row.seed,
                "distance": f.distance, "qid": f.qid}

    if probes:
        res = judges.map(lambda f: JR.judge_feedback_probe(
            judges, feedback_text, f.gen["answer"], model=probe_model, extra=_extra(f)), probes)
        for f, j in zip(probes, res):
            f.judged = {"probe_feedback_label": j["label"], "probe_feedback_reason": j["reason"],
                        "probe_feedback_model": j["model"]}
    if same:
        res = judges.map(lambda f: JR.judge_same_domain(
            judges, f.question_row, f.gen["answer"], extra=_extra(f)), same)
        for f, j in zip(same, res):
            f.judged = {"same_domain_grade": j["label"], "same_domain_reason": j["reason"],
                        "same_domain_model": j["model"], "hold": j["label"] == "held"}
    for f in unrel:  # serial: the vendored judge is async-per-call and the escalator counts calls
        f.judged = JR.score_unrelated(judges, escalator, f.qid, f.gen["answer"],
                                      f.fork_id(mode, arm), _extra(f))
    return forks


# --------------------------------------------------------------------------- distance 4
def filler_turns(rig, model, tok, rows: List[Row], k: int = 4):
    """Persistence at a distance (S4-design §4 R-1): k neutral filler turns, the subject replies to each,
    every reply read out; then the same forks again."""
    fillers = S.load_fillers()
    start_turn = max(len(r.log) for r in rows) + 1
    marks = {id(r): len(r.log) for r in rows}

    def fn(row: Row, turn: int):
        i = turn - start_turn
        if i >= k:
            return None
        return fillers[i % len(fillers)], "filler", "filler %d (distance set)" % (i + 1)

    run_chain_batch(model, tok, rows, seed_base=rows[0].seed, user_turn_fn=fn,
                    n_turns=start_turn + k - 1, start_turn=start_turn, judge=None)
    for r in rows:
        r.extra["distance4_from_log_index"] = marks[id(r)]
    return rows


# --------------------------------------------------------------------------- output
def write_run(rig, out_dir: Path, mode, arm, target, row: Row, forks: List[Fork], meta_extra: dict):
    """One .jsonl (one row per record; the fork rows are the ones the table reads) + one .pt of residuals."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = out_dir / ("%s_seed%d" % (target["id"], row.seed))
    records, resid, proj = [], [], []
    common = {"profile": rig.profile, "mode": mode, "arm": arm, "target": target["id"], "seed": row.seed,
              "reflection_variant": meta_extra.get("reflection_variant"),
              "forced_act": bool(row.extra.get("forced_act")), "dry_run": meta_extra.get("dry_run", False)}

    def add(rec, readout):
        i = len(records)
        rec = dict(common, record_index=i, **rec)
        # the readout spans travel with the row: n_user > 0 is the audit that `feedback_mean` was captured,
        # n_think > 0 that the thinking block was found (the think position is NaN when it was not)
        rec["readout_spans"] = readout["spans"]
        records.append(rec)
        resid.append(readout["resid"]); proj.append(readout["proj"])
        return i

    for i, e in enumerate(row.log):
        readout = row.turns[i] if i < len(row.turns) else None
        if readout is None:
            continue
        kind = {"feedback": "feedback_reply", "filler": "filler_turn"}.get(e.get("kind"), "act_turn")
        add({"kind": kind, "turn": e["turn"], "turn_kind": e.get("kind"), "distance": 0 if kind != "filler_turn" else None,
             "user_text": e["user"], "assistant_text": e.get("assistant"), "think": e.get("think"),
             "answer": e.get("answer"), "adherent": e.get("adherent"), "finish": e.get("finish"),
             "n_new": e.get("n_new"), "grade": e.get("grade"), "grade_reason": e.get("grade_reason"),
             "judge_model": e.get("judge_model"), "note": e.get("note")}, readout)
    for f in forks:
        add(dict({"kind": "fork", "fork_type": f.fork_type, "qid": f.qid, "distance": f.distance,
                  "user_text": f.user, "assistant_text": f.gen["text"], "think": f.gen["think"],
                  "answer": f.gen["answer"], "adherent": f.gen["adherent"], "finish": f.gen["finish"],
                  "n_new": f.gen["n_new"], "max_new": f.gen["max_new"]}, **f.judged), f.readout)
    with open(str(stem) + ".jsonl", "w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    torch.save({"resid": torch.stack(resid) if resid else torch.zeros(0),
                "proj": torch.stack(proj) if proj else torch.zeros(0),
                "axes": rig.axis_names, "positions": R.POSITIONS, "resid_layers": rig.resid_layers,
                "arrow_layers": rig.arrow_layers, "record_index": list(range(len(records))),
                "profile": rig.profile, "model": rig.model_id, "revision": rig.revision}, str(stem) + ".pt")
    meta = dict(common, T_primary=row.T_primary, system=row.system, turns=row.log,
                n_records=len(records), **{k: v for k, v in row.extra.items()})
    meta.update(meta_extra)
    json.dump(meta, open(str(stem) + ".meta.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    return stem


def run_done(out_dir: Path, target_id: str, seed: int) -> bool:
    stem = Path(out_dir) / ("%s_seed%d" % (target_id, seed))
    return all(Path(str(stem) + ext).exists() for ext in (".jsonl", ".pt", ".meta.json"))
