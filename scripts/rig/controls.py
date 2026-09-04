"""S3 rig — the matched-topic controls (brief step 5; S4-design §4 R-4 in its D-022 form).

Every same-domain and unrelated question is also asked to the subject **in a fresh context** - no act, no
feedback - at the same seeds, judged and read out once per (question, seed) and **shared across cells**: it is
the topic baseline for every readout and the no-act floor for every score. Under D-022 the subject is the base
model, so R-4's second control ("the same-domain questions also asked to the base") is this same run.

One file per (question, seed) under ``<out_root>/controls/``; an existing pair is reused, never re-asked.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import List

import torch

import rigcommon as R
import judges_rig as JR
import s1bcommon as S

BATCH = 20


def _stem(out_root: Path, qid: str, seed: int) -> Path:
    return Path(out_root) / "controls" / ("%s_seed%d" % (qid, seed))


def done(out_root: Path, qid: str, seed: int) -> bool:
    st = _stem(out_root, qid, seed)
    return Path(str(st) + ".json").exists() and Path(str(st) + ".pt").exists()


class _Item:
    def __init__(self, kind, qid, question, seed, question_row=None):
        self.kind, self.qid, self.question, self.seed = kind, qid, question, seed
        self.question_row = question_row
        self.gen = None
        self.readout = None
        self.judged = {}


def run_controls(rig, model, tok, judges, escalator, wordings, same_rows: List[dict], unrelated: dict,
                 seeds: List[int], out_root: Path, probe_model: str, dry_run: bool = False) -> dict:
    sysmsg = wordings["think_aloud_instruction"].strip()   # no persona, no act, no feedback (plan A5)
    items: List[_Item] = []
    for q in same_rows:
        for k in seeds:
            if not done(out_root, q["qid"], k):
                items.append(_Item("same_domain", q["qid"], q["question"], k, question_row=q))
    for qid, q in unrelated.items():
        for k in seeds:
            if not done(out_root, qid, k):
                items.append(_Item("unrelated", qid, q["question"], k))
    if not items:
        R.log("controls: all (question, seed) pairs already present")
        return {"asked": 0, "reused": "all"}
    R.log("controls: %d (question, seed) pairs to ask" % len(items))
    by_q = {}
    for it in items:
        by_q.setdefault((it.kind, it.qid), []).append(it)
    for (kind, qid), group in by_q.items():
        group.sort(key=lambda x: x.seed)
        cap = R.MAX_NEW_UNRELATED if kind == "unrelated" else rig.spec["max_new_chain"]
        for start in range(0, len(group), BATCH):
            chunk = group[start:start + BATCH]
            convs = [[{"role": "system", "content": sysmsg}, {"role": "user", "content": it.question}] for it in chunk]
            gens = S.gen_batch(model, tok, convs, seed_base=chunk[0].seed, max_new=cap)
            for it, g in zip(chunk, gens):
                p = S.parse_thinking(g["text"])
                it.gen = {"text": g["text"], "think": p["think"], "answer": p["answer"], "adherent": p["adherent"],
                          "finish": g["finish"], "n_new": g["n_new"], "max_new": cap}
            full = [c + [{"role": "assistant", "content": it.gen["text"]}] for c, it in zip(convs, chunk)]
            ro = S.readout_batch(model, tok, full)
            for it, rec in zip(chunk, ro):
                it.readout = rec
    same = [it for it in items if it.kind == "same_domain"]
    unrel = [it for it in items if it.kind == "unrelated"]
    if same:
        res = judges.map(lambda it: JR.judge_same_domain(
            judges, it.question_row, it.gen["answer"],
            extra={"control": True, "qid": it.qid, "seed": it.seed}), same)
        for it, j in zip(same, res):
            it.judged = {"same_domain_grade": j["label"], "same_domain_reason": j["reason"],
                         "same_domain_model": j["model"], "hold": j["label"] == "held"}
    for it in unrel:
        it.judged = JR.score_unrelated(judges, escalator, it.qid, it.gen["answer"],
                                       "control/%s/seed%d" % (it.qid, it.seed),
                                       {"control": True, "qid": it.qid, "seed": it.seed})
    for it in items:
        st = _stem(out_root, it.qid, it.seed)
        st.parent.mkdir(parents=True, exist_ok=True)
        rec = {"kind": "control", "control_kind": it.kind, "qid": it.qid, "seed": it.seed,
               "profile": rig.profile, "dry_run": dry_run, "system": sysmsg, "user_text": it.question,
               "assistant_text": it.gen["text"], "think": it.gen["think"], "answer": it.gen["answer"],
               "adherent": it.gen["adherent"], "finish": it.gen["finish"], "n_new": it.gen["n_new"],
               "max_new": it.gen["max_new"],
               "note": "topic baseline / no-act floor: fresh context, no act, no feedback (brief step 5)"}
        rec.update(it.judged)
        json.dump(rec, open(str(st) + ".json", "w", encoding="utf-8"), indent=1, ensure_ascii=False)
        torch.save({"resid": it.readout["resid"], "proj": it.readout["proj"], "axes": rig.axis_names,
                    "positions": R.POSITIONS, "resid_layers": rig.resid_layers,
                    "arrow_layers": rig.arrow_layers, "profile": rig.profile}, str(st) + ".pt")
    return {"asked": len(items), "same_domain": len(same), "unrelated": len(unrel)}


def load_controls(out_root: Path) -> List[dict]:
    out = []
    d = Path(out_root) / "controls"
    if d.exists():
        for p in sorted(d.glob("*.json")):
            out.append(json.load(open(p, encoding="utf-8")))
    return out
