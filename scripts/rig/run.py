"""S3 rig — the one button.

    python scripts/rig/run.py --model {1b,base} --mode {deceived,akratic,vicious} \
        --arm {act_blame,self_blame,neutral_correction,neutral_reflection,none} \
        --targets ... --seeds N [--distance4] [--controls]

`--mode`/`--arm`/`--targets` also take a comma-separated list or `all`. One cell = one mode x one arm; one run =
one target x one seed. Resume-safe: a run whose files exist is skipped. Outputs, per the brief:

    <out>/<mode>/<arm>/<target>_seed<k>.jsonl   one row per record; the fork rows are what the table reads
    <out>/<mode>/<arm>/<target>_seed<k>.pt      residuals (float16, every model layer) + projections
    <out>/<mode>/<arm>/summary.json             the cell: N, act rate, discards, cost, machine time
    <out>/controls/<qid>_seed<k>.json|.pt       the topic controls, shared across cells

`--out` defaults to results/raw/s4 (profile `base`) or results/raw/dryrun_1B/s4 (profile `1b`).

Dry-run switches, refused for `--model base`:
  --dryrun     the 1B substitutions: the placeholder chain directory, the first same-domain question and the
               first two unrelated questions, mini-only judges, every record marked dry_run.
  --force-act  treat the last act turn as the act turn even if the act judge did not grade it `committed`, so
               the fork machinery can be exercised when the 1B sandbox holds. Never available at `base`.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import rigcommon as R          # noqa: E402
import judges_rig as JR        # noqa: E402
import cell as CE              # noqa: E402
import controls as CTL         # noqa: E402

DEFAULT_OUT = {"base": R.ROOT / "results" / "raw" / "s4",
               "1b": R.ROOT / "results" / "raw" / "dryrun_1B" / "s4"}
DRYRUN_CHAINS = R.ROOT / "results" / "raw" / "dryrun_1B" / "chains"
REAL_CHAINS = R.ROOT / "data" / "acts" / "chains"
BUDGET = {"1b": 0.50, "base": 3.00}   # brief: Phase 1 $0.50, Phase 2 $3


def _list(arg, allowed):
    if arg in (None, "all"):
        return list(allowed)
    out = [x.strip() for x in arg.split(",") if x.strip()]
    bad = [x for x in out if x not in allowed]
    if bad:
        raise SystemExit("unknown value(s) %s; allowed: %s" % (bad, allowed))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description="S4 rig: one cell = one mode x one arm")
    ap.add_argument("--model", required=True, choices=["1b", "base"])
    ap.add_argument("--mode", default="all")
    ap.add_argument("--arm", default="all")
    ap.add_argument("--targets", default="all")
    ap.add_argument("--seeds", type=int, default=12)
    ap.add_argument("--distance4", action="store_true", help="the distance-4 set (core cells: deceived x 5 arms)")
    ap.add_argument("--controls", action="store_true", help="the matched-topic controls, shared across cells")
    ap.add_argument("--dryrun", action="store_true")
    ap.add_argument("--force-act", dest="force_act", action="store_true")
    ap.add_argument("--out", default=None)
    ap.add_argument("--budget", type=float, default=None)
    ap.add_argument("--chain-dir", default=None)
    a = ap.parse_args(argv)

    if a.model == "base" and (a.dryrun or a.force_act):
        raise SystemExit("--dryrun and --force-act are refused for --model base (plan A10, no exception)")

    out = Path(a.out) if a.out else DEFAULT_OUT[a.model]
    budget = a.budget if a.budget is not None else BUDGET[a.model]
    rig = R.configure(a.model, out, budget)
    import s1bcommon as S
    import judges as JU

    wordings = S.load_wordings()
    targets_all = S.load_targets()
    by_id = {t["id"]: t for t in targets_all}
    tids = _list(a.targets, [t["id"] for t in targets_all])
    modes = _list(a.mode, CE.MODES)
    arms = _list(a.arm, CE.ARMS)
    seeds = list(range(a.seeds))
    chain_dir = Path(a.chain_dir) if a.chain_dir else (DRYRUN_CHAINS if a.dryrun else REAL_CHAINS)
    feedback = CE.load_feedback()

    unrelated = JR.unrelated_questions()
    if a.dryrun:
        unrelated = type(unrelated)(list(unrelated.items())[:2])

    R.log("profile %s | device %s | out %s | budget $%.2f" % (rig.profile, rig.device, out, budget))
    R.log("arrows: " + json.dumps(rig.arrow_header()["files"]) + " | named axes: " + ",".join(rig.named_axes))
    if rig.arrow_header()["arrows_the_design_names_that_this_set_lacks"]:
        R.log("NOTE: this arrow set lacks %s — the rig runs and the table header says so" %
              rig.arrow_header()["arrows_the_design_names_that_this_set_lacks"])

    judges = JU.Judges()
    JR.register_feedback_probe()
    if a.model == "base":
        escalator = JR.Escalator(judges, escalation_model=JR.BIG, enabled=True)
        probe_model = JR.BIG
    else:
        escalator = JR.Escalator(judges, escalation_model=JR.MINI, enabled=True, call_limit=2,
                                 substitute_note="1B dry-run substitute: mini stands in for gpt-4o-2024-08-06 "
                                                 "so the escalation call path is exercised (brief: 1B is mini-only)")
        probe_model = JR.MINI

    model, tok, stats = R.load_model(rig)
    R.log("model loaded: " + json.dumps(stats))

    header = {"profile": rig.profile, "machine": rig.machine_info(), "load": stats,
              "arrows": rig.arrow_header(), "seeds": seeds, "targets": tids, "modes": modes, "arms": arms,
              "distance4": a.distance4, "controls": a.controls, "dry_run": a.dryrun,
              "force_act": a.force_act, "chain_dir": str(chain_dir), "budget_usd": budget,
              "unrelated_questions": list(unrelated.keys()),
              "positions": R.POSITIONS, "post_is": "the `into` position of the feedback-reply turn",
              "feedback_mean_is": "mean over the preceding user turn's content tokens through its closing <|eot_id|>",
              "started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    out.mkdir(parents=True, exist_ok=True)
    json.dump(header, open(out / "run_header.json", "w"), indent=1, ensure_ascii=False)

    try:
        if a.controls:
            same_rows = []
            for tid in tids:
                rows = JR.same_domain_questions(tid)
                same_rows += rows[:1] if a.dryrun else rows
            t0 = time.time()
            res = CTL.run_controls(rig, model, tok, judges, escalator, wordings, same_rows, unrelated,
                                   seeds, out, probe_model, dry_run=a.dryrun)
            R.log("controls: %s | %.1fs" % (json.dumps(res), time.time() - t0))

        for mode in modes:
            for arm in arms:
                cell_dir = out / mode / arm
                summ = {"mode": mode, "arm": arm, "profile": rig.profile, "seeds": seeds,
                        "distance4": a.distance4 and mode == CE.CORE_MODE, "dry_run": a.dryrun,
                        "targets": {}, "machine_s": 0.0, "api_usd": 0.0}
                usd0, t_cell = judges.spent(), time.time()
                for tid in tids:
                    target = by_id[tid]
                    todo = [k for k in seeds if not CE.run_done(cell_dir, tid, k)]
                    if not todo:
                        R.log("%s/%s/%s: all %d runs present — skipped" % (mode, arm, tid, len(seeds)))
                        continue
                    R.log("%s/%s/%s: %d run(s)" % (mode, arm, tid, len(todo)))
                    summ["targets"][tid] = run_one_target(
                        rig, model, tok, judges, escalator, probe_model, mode, arm, target, todo,
                        wordings, chain_dir, feedback, unrelated, cell_dir,
                        distance4=(a.distance4 and mode == CE.CORE_MODE), force_act=a.force_act,
                        dry_run=a.dryrun)
                summ["machine_s"] = round(time.time() - t_cell, 1)
                summ["api_usd"] = round(judges.spent() - usd0, 6)
                summ.update(cell_rollup(cell_dir, seeds, tids))
                cell_dir.mkdir(parents=True, exist_ok=True)
                json.dump(summ, open(cell_dir / "summary.json", "w"), indent=1, ensure_ascii=False)
                R.log("cell %s/%s done: act rate %s, discards %s | %.1fs | $%.4f" % (
                    mode, arm, summ.get("act_rate"), summ.get("discards"), summ["machine_s"], summ["api_usd"]))
    except JU.BudgetStop as e:
        judges.flush()
        R.log("BUDGET STOP: %s" % e)
        return 2
    finally:
        judges.flush()
        json.dump({"ledger_usd": judges.spent(), "escalation": {
            "decided": escalator.decided, "would_escalate": escalator.would_have,
            "calls_made": escalator.calls, "enabled": escalator.enabled, "limit": escalator.call_limit,
            "model": escalator.model}}, open(out / "run_footer.json", "w"), indent=1)
    R.log("done; ledger $%.4f" % judges.spent())
    return 0


def run_one_target(rig, model, tok, judges, escalator, probe_model, mode, arm, target, seeds, wordings,
                   chain_dir, feedback, unrelated, cell_dir, distance4, force_act, dry_run):
    """Seeds of one target, batched together, all the way through the button."""
    fb = feedback[(target["id"], arm)]
    same_rows = JR.same_domain_questions(target["id"])
    if dry_run:
        same_rows = same_rows[:1]
    t0 = time.time()
    rows = CE.reach_the_act(rig, model, tok, judges, mode, target, seeds, wordings, chain_dir,
                            force_act=force_act, dry_run=dry_run)
    committed = [r for r in rows if r.T_primary is not None]
    discards = [r for r in rows if r.T_primary is None]
    R.log("  act: %d/%d committed, %d discard(s)" % (len(committed), len(rows), len(discards)))

    forks_by_row = {id(r): [] for r in rows}
    if committed:
        CE.feedback_turn(rig, model, tok, committed, fb, arm)
        f0 = CE.build_forks(committed, target, wordings, same_rows, unrelated, distance=0)
        CE.run_forks(rig, model, tok, f0)
        CE.judge_forks(judges, escalator, f0, mode, arm, fb["text"], probe_model)
        for f in f0:
            forks_by_row[id(f.row)].append(f)
        if distance4:
            CE.filler_turns(rig, model, tok, committed, k=4)
            f4 = CE.build_forks(committed, target, wordings, same_rows, unrelated, distance=4)
            CE.run_forks(rig, model, tok, f4)
            CE.judge_forks(judges, escalator, f4, mode, arm, fb["text"], probe_model)
            for f in f4:
                forks_by_row[id(f.row)].append(f)

    meta_extra = {"reflection_variant": fb.get("reflection_variant"), "dry_run": dry_run,
                  "feedback_arm": arm, "feedback_n_tokens": fb.get("n_tokens"),
                  "distance4": distance4, "chain_dir": str(chain_dir)}
    for r in rows:
        CE.write_run(rig, cell_dir, mode, arm, target, r, forks_by_row[id(r)], meta_extra)
    return {"n": len(rows), "committed": len(committed), "discards": len(discards),
            "machine_s": round(time.time() - t0, 1)}


def cell_rollup(cell_dir: Path, seeds, tids) -> dict:
    """Act rate and discards over every run present in the cell (brief: always reported, with discards)."""
    n = com = 0
    for p in sorted(Path(cell_dir).glob("*_seed*.meta.json")):
        m = json.load(open(p, encoding="utf-8"))
        n += 1
        com += int(bool(m.get("committed")))
    return {"n_runs": n, "n_committed": com, "discards": n - com,
            "act_rate": round(com / n, 4) if n else None}


if __name__ == "__main__":
    sys.exit(main())
