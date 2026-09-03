"""S1b chain runner — one batch of rows (seeds) through ten user turns with per-turn generation, readout,
and optional online act grading. Shared by Task 3 (pilot; user turns from the persuader), Task 4 (frozen chains
with the bridge rule, the filler fork at T_primary, the exploratory persuasion fork) and the benign-matched chains."""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Callable, List, Optional

import torch

import s1bcommon as S


class Row:
    def __init__(self, target: dict, seed: int, system: str, tag: str):
        self.target, self.seed, self.system, self.tag = target, seed, system, tag
        self.messages: List[dict] = [{"role": "system", "content": system}]
        self.log: List[dict] = []      # one entry per turn
        self.turns: List[dict] = []    # readout records per turn
        self.T_primary: Optional[int] = None
        self.last_grade: Optional[str] = None
        self.last_answer: str = ""
        self.extra: dict = {}

    def add_user(self, text: str, kind: str, turn: int, note: str = ""):
        self.messages.append({"role": "user", "content": text})
        self.log.append({"turn": turn, "user": text, "kind": kind, "note": note})

    def add_assistant(self, gen: dict):
        self.messages.append({"role": "assistant", "content": gen["text"]})
        p = S.parse_thinking(gen["text"])
        e = self.log[-1]
        e.update({"assistant": gen["text"], "think": p["think"], "answer": p["answer"], "adherent": p["adherent"],
                  "finish": gen["finish"], "n_new": gen["n_new"]})
        self.last_answer = p["answer"]

    def to_meta(self) -> dict:
        return {"target": self.target["id"], "seed": self.seed, "tag": self.tag, "system": self.system,
                "T_primary": self.T_primary, "turns": self.log, **self.extra}


def run_chain_batch(model, tok, rows: List[Row], seed_base: int, user_turn_fn: Callable[[Row, int], Optional[tuple]],
                    n_turns: int = 10, judge=None, judge_purpose: str = "act_primary", start_turn: int = 1,
                    on_turn_done: Optional[Callable] = None, per_token: bool = False) -> List[Row]:
    """user_turn_fn(row, turn) -> (text, kind, note) or None to stop that row. If judge is given, every assistant
    answer is graded online by judge.act (primary) and T_primary is set at the first `committed`."""
    for turn in range(start_turn, n_turns + 1):
        active = []
        for r in rows:
            u = user_turn_fn(r, turn)
            if u is None:
                continue
            text, kind, note = u
            r.add_user(text, kind, turn, note)
            active.append(r)
        if not active:
            break
        t0 = time.time()
        gens = S.gen_batch(model, tok, [r.messages for r in active], seed_base=seed_base)
        t_gen = time.time() - t0
        for r, g in zip(active, gens):
            r.add_assistant(g)
        t0 = time.time()
        ro = S.readout_batch(model, tok, [r.messages for r in active], per_token=per_token)
        t_ro = time.time() - t0
        for r, rec in zip(active, ro):
            r.turns.append(rec)
        t0 = time.time()
        if judge is not None:
            def grade(r):
                return judge.act(r.target, r.log[-1]["answer"], purpose=judge_purpose,
                                 extra={"target": r.target["id"], "seed": r.seed, "tag": r.tag, "turn": turn})
            res = judge.map(grade, active)
            for r, j in zip(active, res):
                r.log[-1]["grade"], r.log[-1]["grade_reason"], r.log[-1]["judge_model"] = j["label"], j["reason"], j["model"]
                r.last_grade = j["label"]
                if j["label"] == "committed" and r.T_primary is None:
                    r.T_primary = turn
        t_j = time.time() - t0
        S.log("  turn %d: %d rows | gen %.1fs readout %.1fs judge %.1fs | grades %s" % (
            turn, len(active), t_gen, t_ro, t_j,
            "".join({"held": "h", "partial": "p", "committed": "C", None: "-", "unparseable": "?"}.get(r.last_grade, "?") for r in active) if judge else "n/a"))
        if on_turn_done:
            on_turn_done(turn, active)
    return rows


def save_rows(rows: List[Row], out_dir: Path, prefix: str):
    for r in rows:
        S.save_run(out_dir / r.target["id"] / ("%s_seed%d" % (prefix, r.seed)), r.to_meta(), r.turns)


# ---------------------------------------------------------------------------- user-turn builders
def frozen_chain_fn(chain: List[dict], fillers: List[str], fork_filler: bool = True):
    """Task 4 measured runs: frozen turns with the bridge rule (persuader.md §4, D-6); from T_primary+1 the
    filler sequence (D-2 wrap) when fork_filler, else the persuasion continuation (the exploratory fork)."""
    def fn(row: Row, turn: int):
        if turn == 1:
            return chain[0]["text"], "situation", ""
        if fork_filler and row.T_primary is not None and turn > row.T_primary:
            k = turn - row.T_primary - 1
            wrapped = k >= len(fillers)
            return fillers[k % len(fillers)], "filler", "filler %d%s" % (k % len(fillers) + 1, " (wrapped)" if wrapped else "")
        t = chain[turn - 1]
        assert t["turn"] == turn
        prev_held = row.last_grade == "held" or S.is_question(row.last_answer)
        if prev_held:
            return (t["bridge"].strip() + " " + t["text"].strip()), "frozen+bridge", "bridge prepended (prev %s%s)" % (
                row.last_grade, "; question" if S.is_question(row.last_answer) else "")
        return t["text"], "frozen", "prev %s" % row.last_grade
    return fn


def benign_chain_fn(turns: List[str]):
    def fn(row: Row, turn: int):
        return turns[turn - 1], "benign", ""
    return fn


def single_turn_fn(text: str):
    def fn(row: Row, turn: int):
        return (text, "single", "") if turn == 1 else None
    return fn
