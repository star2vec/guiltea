#!/usr/bin/env python3
"""S6-verify-headlines, Task 1: the three headline numbers recomputed by a second route.

Standalone. Imports: the standard library, numpy, torch (for loading .pt files only),
scikit-learn (roc_auc_score, CountVectorizer, LogisticRegression). Nothing is imported
from scripts/s1d, scripts/s1e, scripts/s1g or scripts/s1h, and none of those files was
opened while writing this. Every definition is taken from the reports the brief names:
  H1  reports/S1g-heldout-trigger.md  (Sections 1, 2, 3, 4)
  H2  reports/S1h-shame-signature.md  (Section 1, Section 2)
  H3  reports/S1d-blame-target.md     (Section 1, Section 2)
and the randctl seed recipe from scripts/randctl.py (read, not imported).

Machine: CPU only. No generation, no model load, no judge call, no cost.

Outputs: prints every recomputed number beside the report's with the absolute
difference and PASS/FAIL, and writes results/raw/s6/headlines.json.
"""
from __future__ import annotations

import glob
import json
import os
import sys
from collections import Counter, defaultdict

import numpy as np
import torch
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RAW = os.path.join(REPO, "results", "raw")
T7 = os.path.join(RAW, "s1b", "t7")
T4 = os.path.join(RAW, "s1b", "t4")
REFL_PRIMARY = os.path.join(RAW, "s1d", "judge_calls", "reflection_primary.jsonl")
REFL_SECOND = os.path.join(RAW, "s1d", "judge_calls", "reflection_second.jsonl")
JOIN = os.path.join(RAW, "s1d", "join.jsonl")
DIRS_BASE = os.path.join(REPO, "directions", "dirs_8B_base_sweep.pt")
DIRS_S2 = os.path.join(REPO, "directions", "dirs_8B_s2_arrows.pt")
OUT_DIR = os.path.join(RAW, "s6")
OUT_JSON = os.path.join(OUT_DIR, "headlines.json")

BAND = [14, 15, 16, 17, 18]          # L14-18, the D-024 primary band, inclusive
POSITION = "answer"
D_MODEL = 4096
N_LAYERS = 32
SEEDS = list(range(10))
TOL_AUROC = 1e-3

# The report's numbers, transcribed from the reports named above.
REPORT = {
    "H1": {
        "nn_headline": 0.662,
        "nn_t1": 0.706,
        "nn_t2": 0.604,
        "largest_seed_headline": 0.585,
        "largest_seed_excess": 0.111,
        "seed_headlines": [0.519, 0.389, 0.466, 0.457, 0.538, 0.531, 0.526, 0.548, 0.585, 0.569],
        "v1_search_excess": 0.104,     # reports/S1e-depth-matched.md Section 2, taken as stated
        "n_pos": {1: 19, 2: 10},
        "n_neg": {1: 19, 2: 19},
        "targets_both": {1: 4, 2: 4},
        "weights": {1: 38, 2: 29},
    },
    "H2": {
        "persona_meandiff_pooled": 0.780,
        "persona_meandiff_fold": 0.788,
        "words_pooled": 0.575,
        "words_fold": 0.611,           # reported by S1h beside the pooled number; extra here
        "n_act": 64,
        "n_self": 19,
        "n_targets": 14,
        "targets_both": 8,
    },
    "H3": {
        "act-focused": 450,
        "self-focused": 24,
        "outcome-negative-only": 0,
        "neutral": 32,
        "incoherent": 2,
        "total": 508,
        "deceived_A_neutral": 26,
        "second_rows": 63,
    },
}

RESULTS: dict = {"H1": {}, "H2": {}, "H3": {}, "checks": []}


# ----------------------------------------------------------------------------- helpers

def unit(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32)
    return v / np.linalg.norm(v)


def randctl_unit(seed: int, layer: int, d_model: int = D_MODEL) -> np.ndarray:
    """The randctl recipe, re-implemented from scripts/randctl.py's docstring and body:
    per-layer torch.Generator seeded seed * 1_000_003 + layer, torch.randn(d_model) float32
    on CPU, divided by its norm."""
    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(seed) * 1_000_003 + int(layer))
    v = torch.randn(int(d_model), generator=gen, dtype=torch.float32)
    return (v / v.norm()).numpy()


def arrow_matrix(per_layer: dict) -> np.ndarray:
    """[32, 4096] float32 matrix of unit arrows, row L = layer L. Re-normalised (a no-op on
    stored unit vectors)."""
    M = np.zeros((N_LAYERS, D_MODEL), dtype=np.float32)
    for L in range(N_LAYERS):
        M[L] = unit(np.asarray(per_layer[L], dtype=np.float32))
    return M


def load_pt(path: str) -> dict:
    return torch.load(path, map_location="cpu", weights_only=False)


def project_record(pt: dict, arrows: np.ndarray, position: str) -> np.ndarray:
    """Project every turn of one stored record at one readout position onto a set of arrows.
    pt['resid'] is [n_turns, 3 positions, 32 layers, 4096] float16.
    arrows is [n_arrows, 32, 4096]. Returns [n_turns, n_arrows, 32] float32."""
    pos_idx = list(pt["positions"]).index(position)
    layers = list(pt["layers"])
    assert layers == list(range(N_LAYERS)), layers
    resid = pt["resid"][:, pos_idx].to(torch.float32).numpy()   # [T, 32, 4096]
    return np.einsum("tld,ald->tal", resid, arrows)


def auroc(y: np.ndarray, s: np.ndarray) -> float:
    return float(roc_auc_score(y, s))


def fold_statistic(y: np.ndarray, s: np.ndarray, targets: np.ndarray) -> tuple[float, dict]:
    """Mean of per-target AUROCs over the targets holding both classes."""
    per = {}
    for tgt in sorted(set(targets)):
        m = targets == tgt
        if len(set(y[m])) == 2:
            per[tgt] = auroc(y[m], s[m])
    return float(np.mean(list(per.values()))), per


def band_mean(per_layer: np.ndarray) -> float:
    return float(np.mean([per_layer[L] for L in BAND]))


def check(label: str, mine, report, tol=None) -> bool:
    """Print mine beside the report's with the absolute difference; PASS/FAIL."""
    if tol is None:                      # exact, for counts
        ok = int(mine) == int(report)
        diff = abs(int(mine) - int(report))
        line = f"  {label:<52s} mine {mine:>8d}   report {report:>8d}   |diff| {diff:d}   {'PASS' if ok else 'FAIL'}"
    else:
        diff = abs(float(mine) - float(report))
        ok = diff <= tol
        line = f"  {label:<52s} mine {mine:8.4f}   report {report:8.3f}   |diff| {diff:.4f}   {'PASS' if ok else 'FAIL'}"
    print(line)
    RESULTS["checks"].append({"label": label, "mine": float(mine) if tol is not None else int(mine),
                              "report": report, "abs_diff": float(diff), "tol": tol, "pass": bool(ok)})
    return ok


# ----------------------------------------------------------------------------- H3

def load_reflection_labels():
    """Final reflection label per (target, seed, mode, fork): the second judge's where it
    exists, else the primary's (reports/S1d-blame-target.md Section 1)."""
    primary, second = {}, {}
    with open(REFL_PRIMARY) as f:
        for line in f:
            r = json.loads(line)
            k = (r["target"], int(r["seed"]), r["mode"], r["fork"])
            assert k not in primary, f"duplicate primary row {k}"
            primary[k] = r
    with open(REFL_SECOND) as f:
        for line in f:
            r = json.loads(line)
            k = (r["target"], int(r["seed"]), r["mode"], r["fork"])
            assert k not in second, f"duplicate second row {k}"
            second[k] = r
    final = {}
    for k, r in primary.items():
        src = "second" if k in second else "primary"
        rr = second[k] if k in second else r
        final[k] = {"label": rr["label"], "reason": rr["reason"], "source": src,
                    "judge_model": rr["model"], "primary_label": r["label"],
                    "second_label": second[k]["label"] if k in second else None}
    return primary, second, final


def load_t7_records():
    """One record per t7 json file: its own fields give the key."""
    recs = {}
    for jf in sorted(glob.glob(os.path.join(T7, "*", "*.json"))):
        d = json.load(open(jf))
        k = (d["target"], int(d["seed"]), d["mode"], d["fork"])
        assert k not in recs, f"duplicate t7 record {k}"
        assert len(d["turns"]) == 1 and d["turns"][0]["kind"] == "probe"
        recs[k] = {"json": jf, "pt": jf[:-5] + ".pt", "answer": d["turns"][0]["answer"],
                   "user": d["turns"][0]["user"], "probe_label": d["turns"][0].get("probe_label"),
                   "n_new": d["turns"][0].get("n_new")}
    return recs


def task_h3(recs, primary, second, final):
    print("\n== H3  the blame-target distribution (reports/S1d-blame-target.md Section 2)")
    print(f"  t7 records {len(recs)}; reflection primary rows {len(primary)}; second rows {len(second)}")
    # every t7 record joins to exactly one primary row and vice versa
    missing = [k for k in recs if k not in primary]
    extra = [k for k in primary if k not in recs]
    assert not missing and not extra, (missing[:3], extra[:3])
    # join.jsonl used only as a cross-check of the key universe
    join_keys = set()
    with open(JOIN) as f:
        for line in f:
            r = json.loads(line)
            join_keys.add((r["target"], int(r["seed"]), r["mode"], r["fork"]))
    print(f"  join.jsonl keys equal the t7 key set: {join_keys == set(recs)}  ({len(join_keys)} keys)")
    RESULTS["H3"]["join_keys_equal_t7"] = join_keys == set(recs)

    counts = Counter(final[k]["label"] for k in recs)
    ok = True
    ok &= check("second-judge rows", len(second), REPORT["H3"]["second_rows"])
    ok &= check("total", sum(counts.values()), REPORT["H3"]["total"])
    for lab in ["act-focused", "self-focused", "outcome-negative-only", "neutral", "incoherent"]:
        ok &= check(f"count {lab}", counts.get(lab, 0), REPORT["H3"][lab])
    dA_neutral = sum(1 for k in recs if k[2] == "deceived" and k[3] == "A" and final[k]["label"] == "neutral")
    ok &= check("deceived / fork A / neutral", dA_neutral, REPORT["H3"]["deceived_A_neutral"])
    # per cell, for the record
    cell = defaultdict(Counter)
    for k in recs:
        cell[f"{k[2]}/{k[3]}"][final[k]["label"]] += 1
    print("  per cell:", {c: dict(v) for c, v in sorted(cell.items())})
    RESULTS["H3"].update({"counts": dict(counts), "deceived_A_neutral": dA_neutral,
                          "per_cell": {c: dict(v) for c, v in cell.items()},
                          "overrides": sum(1 for k in second if second[k]["label"] != primary[k]["label"]),
                          "pass": bool(ok)})
    print(f"  H3 {'PASS' if ok else 'FAIL'}")
    return ok


# ----------------------------------------------------------------------------- H2

def task_h2(recs, final):
    print("\n== H2  persona axis, prompt held constant (reports/S1h-shame-signature.md Section 2)")
    base = load_pt(DIRS_BASE)
    pm = arrow_matrix(base["persona_meandiff_units"])            # [32, 4096]
    cell_keys = sorted(k for k in recs if k[2] == "vicious" and k[3] == "B"
                       and final[k]["label"] in ("act-focused", "self-focused"))
    other = [k for k in recs if k[2] == "vicious" and k[3] == "B"] 
    print(f"  vicious / fork B records {len(other)}, of which act- or self-focused {len(cell_keys)}")
    y = np.array([1 if final[k]["label"] == "act-focused" else 0 for k in cell_keys])   # positive = act-focused
    targets = np.array([k[0] for k in cell_keys])
    texts = [recs[k]["answer"] for k in cell_keys]
    ok = True
    ok &= check("n act-focused", int(y.sum()), REPORT["H2"]["n_act"])
    ok &= check("n self-focused", int((1 - y).sum()), REPORT["H2"]["n_self"])
    ok &= check("targets in cell", len(set(targets)), REPORT["H2"]["n_targets"])
    both = [t for t in sorted(set(targets)) if len(set(y[targets == t])) == 2]
    ok &= check("targets holding both classes", len(both), REPORT["H2"]["targets_both"])

    # projections: one probe turn per record, answer position, persona_meandiff, 32 layers
    S = np.zeros((len(cell_keys), N_LAYERS), dtype=np.float32)
    for i, k in enumerate(cell_keys):
        pt = load_pt(recs[k]["pt"])
        S[i] = project_record(pt, pm[None], POSITION)[0, 0]
    pooled = np.array([auroc(y, S[:, L]) for L in range(N_LAYERS)])
    fold = np.array([fold_statistic(y, S[:, L], targets)[0] for L in range(N_LAYERS)])
    ok &= check("persona_meandiff L14-18 band mean, pooled", band_mean(pooled), REPORT["H2"]["persona_meandiff_pooled"], TOL_AUROC)
    ok &= check("persona_meandiff L14-18 band mean, fold statistic", band_mean(fold), REPORT["H2"]["persona_meandiff_fold"], TOL_AUROC)

    # bag-of-words: CountVectorizer + LogisticRegression(C=1, max_iter=2000), leave-one-target-out,
    # AUROC out of fold. Vectoriser fit on the training folds only (assumption on record).
    oof = np.full(len(cell_keys), np.nan)
    n_folds = 0
    for t in sorted(set(targets)):
        te = targets == t
        tr = ~te
        if len(set(y[tr])) < 2:
            continue
        vec = CountVectorizer()
        Xtr = vec.fit_transform([texts[i] for i in np.where(tr)[0]])
        Xte = vec.transform([texts[i] for i in np.where(te)[0]])
        clf = LogisticRegression(C=1.0, max_iter=2000).fit(Xtr, y[tr])
        oof[te] = clf.predict_proba(Xte)[:, list(clf.classes_).index(1)]
        n_folds += 1
    assert not np.isnan(oof).any()
    words_pooled = auroc(y, oof)
    words_fold, _ = fold_statistic(y, oof, targets)
    print(f"  bag-of-words folds used {n_folds} of {len(set(targets))}")
    ok &= check("bag-of-words AUROC, pooled out-of-fold", words_pooled, REPORT["H2"]["words_pooled"], TOL_AUROC)
    w_fold_ok = check("bag-of-words, fold statistic (extra, S1h prints it)", words_fold, REPORT["H2"]["words_fold"], TOL_AUROC)
    RESULTS["H2"].update({
        "n_act": int(y.sum()), "n_self": int((1 - y).sum()), "n_targets": len(set(targets)),
        "targets_both": both,
        "persona_meandiff_pooled_band": band_mean(pooled), "persona_meandiff_fold_band": band_mean(fold),
        "persona_meandiff_pooled_by_layer": pooled.tolist(), "persona_meandiff_fold_by_layer": fold.tolist(),
        "words_pooled": words_pooled, "words_fold": words_fold, "words_folds_used": n_folds,
        "pass": bool(ok), "words_fold_extra_pass": bool(w_fold_ok),
    })
    print(f"  H2 {'PASS' if ok else 'FAIL'}")
    return ok


# ----------------------------------------------------------------------------- H1

def load_v2_chains():
    chains = []
    for jf in sorted(glob.glob(os.path.join(T4, "*", "v2_seed*.json"))):
        d = json.load(open(jf))
        assert d["tag"] == "v2" and len(d["turns"]) == 10, jf
        grades = [t["grade"] for t in d["turns"]]
        assert all(g in ("held", "committed", "partial") for g in grades), (jf, grades)
        committed_turns = [t["turn"] for t in d["turns"] if t["grade"] == "committed"]
        chains.append({"json": jf, "pt": jf[:-5] + ".pt", "target": d["target"], "seed": int(d["seed"]),
                       "grades": grades, "kinds": [t["kind"] for t in d["turns"]],
                       "T": min(committed_turns) if committed_turns else None,
                       "T_primary_stored": d.get("T_primary")})
    return chains


def task_h1(chains):
    print("\n== H1  held-out susceptibility headline (reports/S1g-heldout-trigger.md Sections 2-4)")
    print(f"  v2 chains {len(chains)} over {len(set(c['target'] for c in chains))} targets; "
          f"turn labels {sum(len(c['grades']) for c in chains)}; "
          f"chains with a committed turn {sum(1 for c in chains if c['T'] is not None)}")
    # stored T_primary vs the derived first committed turn, where stored
    mism = [(c["target"], c["seed"], c["T_primary_stored"], c["T"]) for c in chains
            if c["T_primary_stored"] is not None and c["T_primary_stored"] != c["T"]]
    print(f"  stored T_primary disagrees with the earliest committed grade in {len(mism)} chains {mism[:5]}")
    RESULTS["H1"]["T_primary_mismatches"] = mism

    s2 = load_pt(DIRS_S2)
    arrows = np.stack([arrow_matrix(s2["units"]["nn"])] +
                      [arrow_matrix({L: randctl_unit(s, L) for L in range(N_LAYERS)}) for s in SEEDS])  # [11, 32, 4096]
    names = ["nn"] + [f"random{s}" for s in SEEDS]
    # projections per chain: [10 turns, 11 arrows, 32 layers]
    P = np.zeros((len(chains), 10, len(names), N_LAYERS), dtype=np.float32)
    for i, c in enumerate(chains):
        P[i] = project_record(load_pt(c["pt"]), arrows, POSITION)

    # class table per turn index t (1-based): positives = held at t with a committed turn strictly later;
    # negatives = held at t and never committed at any turn; partial at t = neither.
    table = {}
    for t in range(1, 11):
        pos, neg = [], []
        for i, c in enumerate(chains):
            g = c["grades"][t - 1]
            if g != "held":
                continue
            later = any(gg == "committed" for gg in c["grades"][t:])
            never = all(gg != "committed" for gg in c["grades"])
            if later:
                pos.append(i)
            elif never:
                neg.append(i)
        tb = sorted(set(chains[i]["target"] for i in pos) & set(chains[i]["target"] for i in neg))
        filler_pos = sum(1 for i in pos if chains[i]["kinds"][t - 1] == "filler")
        filler_neg = sum(1 for i in neg if chains[i]["kinds"][t - 1] == "filler")
        clears = len(pos) >= 10 and len(neg) >= 10 and len(tb) >= 3
        table[t] = {"pos": pos, "neg": neg, "targets_both": tb, "filler_pos": filler_pos,
                    "filler_neg": filler_neg, "clears": clears}
        print(f"  t={t:2d}  n+ {len(pos):2d}  n- {len(neg):2d}  targets both {len(tb)}  "
              f"filler +/- {filler_pos}/{filler_neg}  clears floor {clears}")
    RESULTS["H1"]["class_table"] = {t: {"n_pos": len(v["pos"]), "n_neg": len(v["neg"]),
                                        "targets_both": v["targets_both"], "clears": v["clears"],
                                        "filler_pos": v["filler_pos"], "filler_neg": v["filler_neg"]}
                                    for t, v in table.items()}
    clearing = [t for t in table if table[t]["clears"]]
    ok = True
    ok &= check("turn indices clearing the floor (count)", len(clearing), 2)
    print(f"  clearing: {clearing}")
    for t in (1, 2):
        ok &= check(f"t={t} n+", len(table[t]["pos"]), REPORT["H1"]["n_pos"][t])
        ok &= check(f"t={t} n-", len(table[t]["neg"]), REPORT["H1"]["n_neg"][t])
        ok &= check(f"t={t} targets holding both", len(table[t]["targets_both"]), REPORT["H1"]["targets_both"][t])

    # per clearing t, per arrow: fold statistic per layer -> band mean
    bm = {}          # bm[t][name] = band mean
    per_target = {}  # per_target[t][name] = {target: band mean of per-target AUROC}
    weights = {}
    for t in clearing:
        idx = table[t]["pos"] + table[t]["neg"]
        y = np.array([1] * len(table[t]["pos"]) + [0] * len(table[t]["neg"]))
        tg = np.array([chains[i]["target"] for i in idx])
        weights[t] = len(idx)
        bm[t], per_target[t] = {}, {}
        for a, name in enumerate(names):
            per_layer = np.zeros(N_LAYERS)
            per_tgt_layers = defaultdict(list)
            for L in range(N_LAYERS):
                s = np.array([P[i, t - 1, a, L] for i in idx])
                per_layer[L], per = fold_statistic(y, s, tg)
                for tt, v in per.items():
                    per_tgt_layers[tt].append((L, v))
            bm[t][name] = band_mean(per_layer)
            per_target[t][name] = {tt: float(np.mean([v for L, v in vs if L in BAND])) for tt, vs in per_tgt_layers.items()}
    ok &= check("weight t=1 (n+ + n-)", weights[1], REPORT["H1"]["weights"][1])
    ok &= check("weight t=2 (n+ + n-)", weights[2], REPORT["H1"]["weights"][2])
    W = sum(weights.values())
    headline = {name: sum(weights[t] * bm[t][name] for t in clearing) / W for name in names}

    ok &= check("nn band mean, t=1", bm[1]["nn"], REPORT["H1"]["nn_t1"], TOL_AUROC)
    ok &= check("nn band mean, t=2", bm[2]["nn"], REPORT["H1"]["nn_t2"], TOL_AUROC)
    ok &= check("nn headline, count-weighted", headline["nn"], REPORT["H1"]["nn_headline"], TOL_AUROC)
    seed_head = [headline[f"random{s}"] for s in SEEDS]
    print("  seed headlines mine  :", " ".join(f"{v:.3f}" for v in seed_head))
    print("  seed headlines report:", " ".join(f"{v:.3f}" for v in REPORT["H1"]["seed_headlines"]))
    seeds_ok = True
    for s_, (a, b) in enumerate(zip(seed_head, REPORT["H1"]["seed_headlines"])):
        seeds_ok &= check(f"seed {s_} headline band mean", a, b, TOL_AUROC)
    ok &= seeds_ok
    largest = max(seed_head)
    ok &= check("largest-seed headline (the floor)", largest, REPORT["H1"]["largest_seed_headline"], TOL_AUROC)
    # The S1g threshold. Definition used for the check: the largest seed's headline band mean minus 0.5
    # (one-sided, in the predicted direction, which is how S1g states its success criterion:
    # "the headline exceeds the largest seed's headline ... nn above 0.5").
    ok &= check("largest-seed excess over 0.5 (S1g threshold), one-sided", largest - 0.5,
                REPORT["H1"]["largest_seed_excess"], TOL_AUROC)
    two_sided = max(abs(v - 0.5) for v in seed_head)
    print(f"  for the record, NOT the definition checked: the two-sided reading max_s |headline_s - 0.5| "
          f"= {two_sided:.4f} (from seed {int(np.argmax([abs(v - 0.5) for v in seed_head]))} at "
          f"{seed_head[int(np.argmax([abs(v - 0.5) for v in seed_head]))]:.3f})")
    thr_ok = (largest - 0.5) > REPORT["H1"]["v1_search_excess"]
    print(f"  one-sided threshold {largest - 0.5:.4f} exceeds the v1 search excess {REPORT['H1']['v1_search_excess']:.3f} "
          f"(S1e Section 2, taken as stated): {'YES' if thr_ok else 'NO'}")
    print(f"  two-sided reading {two_sided:.4f} exceeds 0.104: {'YES' if two_sided > REPORT['H1']['v1_search_excess'] else 'NO'}")
    print(f"  nn headline excess {headline['nn'] - 0.5:.4f} vs largest-seed excess {largest - 0.5:.4f}: "
          f"nn {'clears' if headline['nn'] - 0.5 > largest - 0.5 else 'does not clear'}")
    print("  per-target nn band means (decomposition, beside S1g Section 4):")
    for t in clearing:
        for tt, v in sorted(per_target[t]["nn"].items()):
            npos = sum(1 for i in table[t]["pos"] if chains[i]["target"] == tt)
            nneg = sum(1 for i in table[t]["neg"] if chains[i]["target"] == tt)
            print(f"    t={t}  {tt:<24s} {npos}/{nneg}  {v:.3f}")
    RESULTS["H1"].update({
        "clearing": clearing, "weights": weights,
        "band_means": {t: bm[t] for t in clearing},
        "per_target_nn": {t: per_target[t]["nn"] for t in clearing},
        "headline": headline, "seed_headlines": seed_head, "largest_seed_headline": largest,
        "largest_seed_excess_one_sided": largest - 0.5, "largest_seed_excess_two_sided_for_record": two_sided,
        "threshold_exceeds_v1_one_sided": bool(thr_ok),
        "threshold_exceeds_v1_two_sided": bool(two_sided > REPORT["H1"]["v1_search_excess"]),
        "nn_clears": bool(headline["nn"] - 0.5 > largest - 0.5), "pass": bool(ok),
    })
    print(f"  H1 {'PASS' if ok else 'FAIL'}")
    return ok


# ----------------------------------------------------------------------------- main

def main() -> int:
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"repo {REPO}\nraw  {RAW}\nnumpy {np.__version__}  torch {torch.__version__}  "
          f"tolerance {TOL_AUROC} on an AUROC or band mean, exact on a count")
    recs = load_t7_records()
    primary, second, final = load_reflection_labels()
    ok3 = task_h3(recs, primary, second, final)
    ok2 = task_h2(recs, final)
    ok1 = task_h1(load_v2_chains())
    RESULTS["summary"] = {"H1": ok1, "H2": ok2, "H3": ok3}
    with open(OUT_JSON, "w") as f:
        json.dump(RESULTS, f, indent=1, default=lambda o: o if not isinstance(o, (np.floating, np.integer)) else o.item())
    print(f"\nsummary  H1 {'PASS' if ok1 else 'FAIL'}  H2 {'PASS' if ok2 else 'FAIL'}  H3 {'PASS' if ok3 else 'FAIL'}")
    print(f"wrote {OUT_JSON}")
    return 0 if (ok1 and ok2 and ok3) else 1


if __name__ == "__main__":
    sys.exit(main())
