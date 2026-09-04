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
import steer_rig as ST         # noqa: E402
import cell as CE              # noqa: E402
import controls as CTL         # noqa: E402

DEFAULT_OUT = {"base": R.ROOT / "results" / "raw" / "s4",
               "1b": R.ROOT / "results" / "raw" / "dryrun_1B" / "s4"}
DRYRUN_CHAINS = R.ROOT / "results" / "raw" / "dryrun_1B" / "chains"
REAL_CHAINS = R.ROOT / "data" / "acts" / "chains"
BUDGET = {"1b": 0.50, "base": 11.50}  # briefs/S4-experiment.md: budget stop $11.50 on the rig's own ledger


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
    # S4 Task 0
    ap.add_argument("--steer", default=None, metavar="ARROW:LAYER:C",
                    help="steer the subject's turns along ARROW at LAYER by C sigma, e.g. guilt_clean:16:4")
    ap.add_argument("--steer-sigma-from", dest="steer_sigma_from", default=None, metavar="ARROW",
                    help="take sigma from this arrow instead of the steered one (cell D is norm-matched to C)")
    ap.add_argument("--steer-off-after", dest="steer_off_after", default="distance0",
                    choices=list(ST.OFF_AFTER_CHOICES),
                    help="where the steering window closes; the brief fixes this at distance0")
    ap.add_argument("--judges", default="d021", choices=["d021", "mini"],
                    help="mini: every judge on gpt-4o-mini, the fixed 1-in-8 alignment sample still escalated")
    # S5b (briefs/S5b-prevent.md): chain only, and the steering window over the chain instead of the aftermath
    ap.add_argument("--chain-only", dest="chain_only", action="store_true",
                    help="stop at the act: no feedback turn, no forks, no distance-4 set, no topic controls")
    ap.add_argument("--steer-scope", dest="steer_scope", default="aftermath", choices=["aftermath", "chain"],
                    help="aftermath (S4): the feedback reply through the distance-0 forks. "
                         "chain (S5b): every assistant turn of the chain, turn 1 through the act")
    ap.add_argument("--coherence-acts", dest="coherence_acts", action="store_true",
                    help="score every act-turn answer with the vendored coherence prompt on mini (S5b)")
    ap.add_argument("--norm-check", dest="norm_check", action="store_true",
                    help="run the pre-hook norm check for --steer and exit; no cell, no judge call")
    a = ap.parse_args(argv)

    if a.model == "base" and (a.dryrun or a.force_act):
        raise SystemExit("--dryrun and --force-act are refused for --model base (plan A10, no exception)")
    if (a.steer_sigma_from or a.norm_check) and not a.steer:
        raise SystemExit("--steer-sigma-from and --norm-check need --steer")
    if a.chain_only and (a.distance4 or a.controls):
        raise SystemExit("--chain-only excludes --distance4 and --controls (S5b brief: the chain only)")

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
    steer_suffix = ""
    if a.steer:
        _arrow, _layer, _c = ST.parse_spec(a.steer)
        steer_suffix = ST.cell_suffix(_arrow, _layer, _c)

    R.log("profile %s | device %s | out %s | budget $%.2f" % (rig.profile, rig.device, out, budget))
    R.log("arrows: " + json.dumps(rig.arrow_header()["files"]) + " | named axes: " + ",".join(rig.named_axes))
    if rig.arrow_header()["arrows_the_design_names_that_this_set_lacks"]:
        R.log("NOTE: this arrow set lacks %s — the rig runs and the table header says so" %
              rig.arrow_header()["arrows_the_design_names_that_this_set_lacks"])

    judges = JU.Judges()
    JR.register_feedback_probe()
    mini_judges = a.judges == "mini"
    if a.model == "base":
        # `--judges mini` (Task 0 change 2): every judge on mini, including the §2b feedback-probe classifier
        # and the reflection judge, with the fixed 1-in-8 seeded alignment sample still escalated to the big
        # model. The [65, 90] band is what the flag drops.
        escalator = JR.Escalator(judges, escalation_model=JR.BIG, enabled=True, use_band=not mini_judges)
        probe_model = JR.MINI if mini_judges else JR.BIG
    else:
        escalator = JR.Escalator(judges, escalation_model=JR.MINI, enabled=True, call_limit=2,
                                 substitute_note="1B dry-run substitute: mini stands in for gpt-4o-2024-08-06 "
                                                 "so the escalation call path is exercised (brief: 1B is mini-only)")
        probe_model = JR.MINI

    model, tok, stats = R.load_model(rig)
    R.log("model loaded: " + json.dumps(stats))

    # The steering hook is installed once, before any readout hook is ever registered, so the rig's per-layer
    # capture sees the steered residual at L >= layer (PyTorch runs forward hooks in registration order and a
    # hook that returns a value replaces the output for the ones after it).
    steer = ST.install(model, rig, a.steer, a.steer_sigma_from)
    if steer is not None:
        R.log("steer: " + json.dumps({k: v for k, v in steer.header(rig).items()
                                      if k != "injected_component_per_axis"}))
    if a.norm_check:
        return norm_check(model, tok, rig, steer, wordings, by_id, tids, out)

    header = {"profile": rig.profile, "machine": rig.machine_info(), "load": stats,
              "arrows": rig.arrow_header(), "seeds": seeds, "targets": tids, "modes": modes, "arms": arms,
              "distance4": a.distance4, "controls": a.controls, "dry_run": a.dryrun,
              "force_act": a.force_act, "chain_dir": str(chain_dir), "budget_usd": budget,
              "unrelated_questions": list(unrelated.keys()),
              "steer": steer.header(rig) if steer is not None else None,
              "steer_off_after": a.steer_off_after if steer is not None else None,
              "steer_scope": a.steer_scope if steer is not None else None,
              "chain_only": a.chain_only, "coherence_acts": a.coherence_acts,
              "cell_suffix": steer_suffix,
              "judges_mode": a.judges,
              "judge_models": {"act_primary": JR.MINI, "act_samedomain": JR.MINI, "probe_feedback": probe_model,
                               "reflection_blame": JR.MINI if mini_judges else JR.BIG,
                               "alignment_primary": JR.MINI, "coherence": JR.MINI,
                               "alignment_escalation": escalator.model},
              "escalation_rule": escalator.rule(),
              "positions": R.POSITIONS, "post_is": "the `into` position of the feedback-reply turn",
              "feedback_mean_is": "mean over the preceding user turn's content tokens through its closing <|eot_id|>",
              "started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    out.mkdir(parents=True, exist_ok=True)
    json.dump(header, open(out / "run_header.json", "w"), indent=1, ensure_ascii=False)
    # every invocation appends: run_header.json is only ever the latest, and S4 takes fourteen of them
    with open(out / "run_headers.jsonl", "a", encoding="utf-8") as fh:
        fh.write(json.dumps(header, ensure_ascii=False) + "\n")

    try:
        if a.controls:
            same_rows = []
            for tid in tids:
                rows = JR.same_domain_questions(tid)
                same_rows += rows[:1] if a.dryrun else rows
            t0 = time.time()
            res = CTL.run_controls(rig, model, tok, judges, escalator, wordings, same_rows, unrelated,
                                   seeds, out, probe_model, dry_run=a.dryrun)
            judges.flush()
            R.log("controls: %s | %.1fs" % (json.dumps(res), time.time() - t0))

        for mode in modes:
            for arm in arms:
                cell_dir = out / mode / (arm + steer_suffix)
                summ = {"mode": mode, "arm": arm, "cell": arm + steer_suffix,
                        "steer": steer.header(rig) if steer is not None else None,
                        "judges_mode": a.judges, "profile": rig.profile, "seeds": seeds,
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
                        dry_run=a.dryrun, reflection_model=JR.MINI if mini_judges else JR.BIG,
                        cell_label=arm + steer_suffix, chain_only=a.chain_only,
                        steer_scope=a.steer_scope, coherence_acts=a.coherence_acts)
                summ["machine_s"] = round(time.time() - t_cell, 1)
                summ["api_usd"] = round(judges.spent() - usd0, 6)
                summ.update(cell_rollup(cell_dir, seeds, tids))
                cell_dir.mkdir(parents=True, exist_ok=True)
                json.dump(summ, open(cell_dir / "summary.json", "w"), indent=1, ensure_ascii=False)
                judges.flush()   # the ledger file must not lag the run: a crash mid-cell would under-report spend
                R.log("cell %s/%s done: act rate %s, discards %s | %.1fs | $%.4f" % (
                    mode, arm, summ.get("act_rate"), summ.get("discards"), summ["machine_s"], summ["api_usd"]))
    except JU.BudgetStop as e:
        judges.flush()
        R.log("BUDGET STOP: %s" % e)
        return 2
    finally:
        judges.flush()
        json.dump({"ledger_usd": judges.spent(), "user_span_misses": dict(R.USER_SPAN_MISSES),
                   "batching": dict(R.BATCH_STATS), "escalation": {
            "decided": escalator.decided, "would_escalate": escalator.would_have,
            "calls_made": escalator.calls, "enabled": escalator.enabled, "limit": escalator.call_limit,
            "model": escalator.model}}, open(out / "run_footer.json", "w"), indent=1)
    R.log("done; ledger $%.4f" % judges.spent())
    return 0


def run_one_target(rig, model, tok, judges, escalator, probe_model, mode, arm, target, seeds, wordings,
                   chain_dir, feedback, unrelated, cell_dir, distance4, force_act, dry_run,
                   reflection_model=JR.MINI, cell_label="", chain_only=False, steer_scope="aftermath",
                   coherence_acts=False):
    """Seeds of one target, batched together, all the way through the button."""
    fb = feedback[(target["id"], arm)]
    same_rows = JR.same_domain_questions(target["id"])
    if dry_run:
        same_rows = same_rows[:1]
    t0 = time.time()
    rows = CE.reach_the_act(rig, model, tok, judges, mode, target, seeds, wordings, chain_dir,
                            force_act=force_act, dry_run=dry_run, steer_chain=(steer_scope == "chain"))
    committed = [r for r in rows if r.T_primary is not None]
    discards = [r for r in rows if r.T_primary is None]
    R.log("  act: %d/%d committed, %d discard(s)" % (len(committed), len(rows), len(discards)))
    if coherence_acts:
        # S5b's load-bearing control: an act-rate drop with a coherence drop is damage, not prevention
        CE.judge_act_coherence(judges, rows, mode, arm, JR.MINI, cell_label)

    forks_by_row = {id(r): [] for r in rows}
    if committed and not chain_only:
        CE.feedback_turn(rig, model, tok, committed, fb, arm)
        # Task 0 change 3: the blame target of the feedback reply, one mini call per run (the `none` arm has no
        # reply turn and gets no call)
        CE.judge_feedback_reply(judges, committed, mode, arm, reflection_model, cell_label)
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
                  "feedback_arm": arm, "feedback_n_tokens": fb.get("n_tokens"), "cell": cell_label,
                  "chain_only": chain_only, "steer_scope": steer_scope,
                  "steer": ST.current().header(rig) if ST.current() is not None else None,
                  "distance4": distance4, "chain_dir": str(chain_dir)}
    for r in rows:
        CE.write_run(rig, cell_dir, mode, arm, target, r, forks_by_row[id(r)], meta_extra)
    return {"n": len(rows), "committed": len(committed), "discards": len(discards),
            "machine_s": round(time.time() - t0, 1)}


def norm_check(model, tok, rig, steer, wordings, by_id, tids, out) -> int:
    """`--norm-check`: the addendum's next-layer pre-hook diagnostic on the rig's own hook, then exit.

    Two probe conversations, built from the assets this run would use, so the check is on real context lengths:
    the target's plain `situation` under the think-aloud system prompt. No generation, no judge call.
    """
    convs = []
    for tid in tids[:2]:
        t = by_id[tid]
        convs.append([{"role": "system", "content": wordings["think_aloud_instruction"].strip()},
                      {"role": "user", "content": t["situation"]}])
    res = ST.norm_check(model, tok, rig, steer, convs)
    res["steer"] = steer.header(rig)
    res["probe_conversations"] = [{"target": tid} for tid in tids[:2]]
    out.mkdir(parents=True, exist_ok=True)
    path = out / ("norm_check_%s_L%d_c%g.json" % (steer.arrow, steer.layer, steer.c))
    json.dump(res, open(path, "w"), indent=1, ensure_ascii=False)
    for c in res["checks"]:
        R.log("norm check %s c=%g sigma(%s)=%.6f: expected %.4f measured %.4f (rel dev %+.2e) maxdev %.3f pos %d"
              % (c["arrow"], c["c"], c["sigma_source"], c["sigma"], c["expected_norm"], c["measured_norm"],
                 c["relative_deviation"], c["max_abs_dev_from_expected"], c["positions"]))
    R.log("norm check %s: worst relative deviation %.2e against a %.0f%% tolerance -> %s (wrote %s)"
          % (steer.arrow, res["worst_relative_deviation"], 100 * res["tolerance"],
             "PASS" if res["pass"] else "FAIL", path))
    return 0 if res["pass"] else 3


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
