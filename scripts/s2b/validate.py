"""S2b Tasks 3–6 — held-out validation by scenario (D-018 gate), the angle, cross-voice/distinctness, the band.

Folds (Task 3): five folds over scenario ids, numpy default_rng(0).permutation over scenarios.jsonl file order,
np.array_split into 5 blocks (10 each); every framing of a scenario in its fold. Within each fold every arrow
(incl. nn for cleaning) is re-extracted from the training scenarios only; held-out passages scored by projection.
Bootstrap (Tasks 3, 4): 1,000 resamples of scenario ids with replacement (default_rng(0)); per resample the unique
ids are re-permuted into 5 folds (default_rng(1_000_003 + b)); a duplicated scenario keeps one fold and counts with
its multiplicity (weighted means, weighted AUROC); CI = 2.5 / 97.5 percentiles. Point estimates use the seed-0 folds
on the original data. Random control: randctl seed 0 unit vector at each layer, same rows/scores/folds; the gate's
(arrow − random) is paired within resample. Lexical baseline: binary unigram CountVectorizer fit on the training
fold's two-class texts, L2 LogisticRegression (C=1, liblinear), AUROC from predict_proba, same folds and bootstrap.
Rows: 1 guilt vs baseline (x·û_guilt); 2 shame vs baseline; 3 ĝ vs neutral_negative (x·û_ĝ, pos = guilt passages);
4 ŝ vs neutral_negative; 5 ĝ vs ŝ (x·û_ĝ − x·û_ŝ, pos = guilt passages, neg = shame passages);
6 received_act vs received_self at feedback_mean (x·û_ra − x·û_rs, pos = act_blame).
Gate (D-018, read against the CI): survives at layer ≤ 30 if CI-lower(row 3 / row 4) ≥ 0.75 and CI-lower(arrow − random)
≥ 0.20; NEAR if not surviving and no condition fails outright, where a condition is "near" when CI-lower < t ≤ CI-upper and
t − CI-lower ≤ 0.05; fails otherwise.
"""
from __future__ import annotations

import time

import numpy as np
import torch
from joblib import Parallel, delayed

from s2b_common import (FP_CLASSES, LAYERS, RAW, SP_CLASSES, auroc_w, class_masks, cos_layers, fp_arrows,
                        load_acts, load_s2_arrows, load_scenarios, load_sweep, pct_ci, random_units, save_json,
                        scenario_index, sp_arrows, unit, weighted_means)

DEV = "cuda" if torch.cuda.is_available() else "cpu"
N_BOOT = 1000
ROWS = ["guilt_vs_baseline", "shame_vs_baseline", "guilt_clean_vs_neutral_negative", "shame_clean_vs_neutral_negative",
        "guilt_clean_vs_shame_clean", "received_act_vs_received_self"]
T_AUROC, T_DIFF, NEAR_W = 0.75, 0.20, 0.05


# ----------------------------------------------------------------------------- data
class Data:
    def __init__(self):
        self.scen = load_scenarios()
        self.Xf, self.rows_f = load_acts("first_person_mean", DEV)
        self.Xs, self.rows_s = load_acts("second_person_feedback_mean", DEV)
        self.sf = scenario_index(self.rows_f, self.scen)
        self.ss = scenario_index(self.rows_s, self.scen)
        self.mf = class_masks(self.rows_f, FP_CLASSES)
        self.ms = class_masks(self.rows_s, SP_CLASSES)
        self.tf = [r["text"] for r in load_first_person_texts()]
        self.ts = [r["text"] for r in load_second_person_texts()]
        self.R = random_units(0, DEV)
        self.n_scen = len(self.scen)
        # row task definitions: (which X, pos class, neg class, arrow(s))
        self.tasks = [("f", "guilt", "baseline", ("guilt",)), ("f", "shame", "baseline", ("shame",)),
                      ("f", "guilt", "neutral_negative", ("guilt_clean",)), ("f", "shame", "neutral_negative", ("shame_clean",)),
                      ("f", "guilt", "shame", ("guilt_clean", "shame_clean")),
                      ("s", "act_blame", "self_blame", ("received_act", "received_self"))]


def load_first_person_texts():
    from s2b_common import load_first_person
    return load_first_person()


def load_second_person_texts():
    from s2b_common import load_second_person
    return load_second_person()


def seed0_folds(n_scen: int) -> np.ndarray:
    perm = np.random.default_rng(0).permutation(n_scen)
    fold_of = np.full(n_scen, -1)
    for k, block in enumerate(np.array_split(perm, 5)):
        fold_of[block] = k
    return fold_of


def resample_folds(counts: np.ndarray, b: int) -> np.ndarray:
    u = np.nonzero(counts)[0]
    perm = u[np.random.default_rng(1_000_003 + b).permutation(len(u))]
    fold_of = np.full(len(counts), -1)
    for k, block in enumerate(np.array_split(perm, 5)):
        fold_of[block] = k
    return fold_of


# ----------------------------------------------------------------------------- arrows from weights
def arrows_from_counts(D: Data, wf_scen: np.ndarray, ws_scen: np.ndarray):
    """wf_scen / ws_scen: per-scenario weights (multiplicity × in-training). Returns raw arrows dict [L, D]."""
    wf = torch.tensor(wf_scen[D.sf], dtype=torch.float32, device=DEV)
    ws = torch.tensor(ws_scen[D.ss], dtype=torch.float32, device=DEV)
    Wf = torch.stack([torch.tensor(D.mf[c], device=DEV).float() * wf for c in FP_CLASSES])
    Ws = torch.stack([torch.tensor(D.ms[c], device=DEV).float() * ws for c in SP_CLASSES])
    Mf = dict(zip(FP_CLASSES, weighted_means(D.Xf, Wf)))
    Ms = dict(zip(SP_CLASSES, weighted_means(D.Xs, Ws)))
    return {**fp_arrows(Mf), **sp_arrows(Ms)}


def evaluate(D: Data, counts: np.ndarray, fold_of: np.ndarray):
    """Returns arrow AUROC [6, L], random AUROC [6, L] (means over folds)."""
    A = torch.zeros(len(ROWS), len(LAYERS), device=DEV)
    Rr = torch.zeros_like(A)
    nf = 0
    for k in range(5):
        tr = counts * (fold_of != k)
        if tr.sum() == 0 or (counts * (fold_of == k)).sum() == 0:
            continue
        nf += 1
        arrows = arrows_from_counts(D, tr, tr)
        for i, (which, pos, neg, names) in enumerate(D.tasks):
            X, sidx, masks = (D.Xf, D.sf, D.mf) if which == "f" else (D.Xs, D.ss, D.ms)
            te = (fold_of[sidx] == k) & (counts[sidx] > 0)
            pm, nm = te & masks[pos], te & masks[neg]
            w_pos = torch.tensor(counts[sidx[pm]], dtype=torch.float32, device=DEV)
            w_neg = torch.tensor(counts[sidx[nm]], dtype=torch.float32, device=DEV)
            Xp, Xn = X[torch.tensor(pm, device=DEV)], X[torch.tensor(nm, device=DEV)]
            u = unit(arrows[names[0]])
            sp_, sn_ = torch.einsum("nld,ld->nl", Xp, u), torch.einsum("nld,ld->nl", Xn, u)
            if len(names) == 2:
                u2 = unit(arrows[names[1]])
                sp_ = sp_ - torch.einsum("nld,ld->nl", Xp, u2)
                sn_ = sn_ - torch.einsum("nld,ld->nl", Xn, u2)
            A[i] += auroc_w(sp_, sn_, w_pos, w_neg)
            rp, rn = torch.einsum("nld,ld->nl", Xp, D.R), torch.einsum("nld,ld->nl", Xn, D.R)
            Rr[i] += auroc_w(rp, rn, w_pos, w_neg)
    return (A / nf).cpu().numpy(), (Rr / nf).cpu().numpy()


def angle(D: Data, counts: np.ndarray):
    arrows = arrows_from_counts(D, counts, counts)
    return {"guilt_clean,shame_clean": cos_layers(arrows["guilt_clean"], arrows["shame_clean"]).cpu().numpy(),
            "guilt,shame": cos_layers(arrows["guilt"], arrows["shame"]).cpu().numpy(),
            "guilt,nn": cos_layers(arrows["guilt"], arrows["nn"]).cpu().numpy(),
            "shame,nn": cos_layers(arrows["shame"], arrows["nn"]).cpu().numpy()}


# ----------------------------------------------------------------------------- lexical baseline (CPU, joblib)
def lexical(tasks, tf, ts, sf, ss, mf, ms, counts, fold_of):
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    out = np.zeros(len(tasks))
    nf = 0
    for k in range(5):
        if (counts * (fold_of != k)).sum() == 0 or (counts * (fold_of == k)).sum() == 0:
            continue
        nf += 1
        for i, (which, pos, neg, _) in enumerate(tasks):
            texts, sidx, masks = (tf, sf, mf) if which == "f" else (ts, ss, ms)
            cls = masks[pos] | masks[neg]
            tr = cls & (fold_of[sidx] != k) & (counts[sidx] > 0)
            te = cls & (fold_of[sidx] == k) & (counts[sidx] > 0)
            y = masks[pos].astype(int)
            vec = CountVectorizer(binary=True, lowercase=True)
            Xtr = vec.fit_transform([texts[j] for j in np.nonzero(tr)[0]])
            Xte = vec.transform([texts[j] for j in np.nonzero(te)[0]])
            clf = LogisticRegression(penalty="l2", C=1.0, solver="liblinear", max_iter=1000)
            clf.fit(Xtr, y[tr], sample_weight=counts[sidx[tr]])
            p = clf.predict_proba(Xte)[:, 1]
            out[i] += roc_auc_score(y[te], p, sample_weight=counts[sidx[te]])
    return out / nf


# ----------------------------------------------------------------------------- gate / regimes / band
def gate(lo_auc, hi_auc, lo_diff, hi_diff):
    """Per layer verdict for one arrow from CI bounds arrays [L]."""
    verdict = []
    for L in LAYERS:
        if L > 30:
            verdict.append("excluded (L31)"); continue
        c1 = "ok" if lo_auc[L] >= T_AUROC else ("near" if (hi_auc[L] >= T_AUROC and T_AUROC - lo_auc[L] <= NEAR_W) else "fail")
        c2 = "ok" if lo_diff[L] >= T_DIFF else ("near" if (hi_diff[L] >= T_DIFF and T_DIFF - lo_diff[L] <= NEAR_W) else "fail")
        verdict.append("survives" if (c1 == "ok" and c2 == "ok") else ("fails" if "fail" in (c1, c2) else "NEAR"))
    return verdict


def regime(lo, hi):
    if hi < -0.5:
        return "near -1"
    if lo > -0.25 and hi < 0.25:
        return "near 0"
    if lo > 0.5:
        return "strongly positive"
    return "straddling"


def band(score_by_layer, cos_lo, cos_hi):
    runs = []
    for length in (4, 5, 6):
        for a in range(0, 31 - length + 1):
            b = a + length - 1
            ok = all((cos_lo[L] > 0) or (cos_hi[L] < 0) for L in range(a, b + 1))
            signs = {int(np.sign(cos_lo[L] + cos_hi[L])) for L in range(a, b + 1)}
            same = ok and len(signs) == 1
            runs.append({"start": a, "end": b, "length": length, "centre": (a + b) // 2,
                         "score": float(np.mean(score_by_layer[a:b + 1])), "cos_sign_consistent": bool(same),
                         "cos_sign": (list(signs)[0] if len(signs) == 1 else 0)})
    valid = [r for r in runs if r["cos_sign_consistent"]]
    valid.sort(key=lambda r: (-r["score"], r["start"], r["length"]))
    return valid, runs


def main():
    t0 = time.time()
    D = Data()
    fold0 = seed0_folds(D.n_scen)
    ones = np.ones(D.n_scen)
    A0, R0 = evaluate(D, ones, fold0)
    C0 = angle(D, ones)
    lex0 = lexical(D.tasks, D.tf, D.ts, D.sf, D.ss, D.mf, D.ms, ones, fold0)
    print(f"point estimates done {time.time()-t0:.0f}s", flush=True)

    rng = np.random.default_rng(0)
    draws = [rng.choice(D.n_scen, D.n_scen, replace=True) for _ in range(N_BOOT)]
    counts_l = [np.bincount(d, minlength=D.n_scen).astype(float) for d in draws]
    folds_l = [resample_folds(c, b) for b, c in enumerate(counts_l)]
    Ab, Rb, Cb = [], [], []
    for b in range(N_BOOT):
        a_, r_ = evaluate(D, counts_l[b], folds_l[b])
        Ab.append(a_); Rb.append(r_); Cb.append(angle(D, counts_l[b])["guilt_clean,shame_clean"])
        if b % 100 == 0:
            print(f"boot {b}/{N_BOOT} {time.time()-t0:.0f}s", flush=True)
    Ab, Rb, Cb = np.stack(Ab), np.stack(Rb), np.stack(Cb)
    Db = Ab - Rb
    print(f"arrow bootstrap done {time.time()-t0:.0f}s; lexical bootstrap ...", flush=True)
    Lb = np.stack(Parallel(n_jobs=64)(delayed(lexical)(D.tasks, D.tf, D.ts, D.sf, D.ss, D.mf, D.ms, counts_l[b], folds_l[b])
                                       for b in range(N_BOOT)))
    print(f"lexical bootstrap done {time.time()-t0:.0f}s", flush=True)

    # ---------------- Task 3 tables
    lo_A, hi_A = pct_ci(Ab); lo_R, hi_R = pct_ci(Rb); lo_D, hi_D = pct_ci(Db); lo_L, hi_L = pct_ci(Lb)
    t3 = {"rows": ROWS, "layers": LAYERS, "n_boot": N_BOOT, "folds_seed0": {int(k): [D.scen[i]["id"] for i in np.nonzero(fold0 == k)[0]] for k in range(5)},
          "arrow": {"point": A0.tolist(), "lo": lo_A.tolist(), "hi": hi_A.tolist()},
          "random": {"point": R0.tolist(), "lo": lo_R.tolist(), "hi": hi_R.tolist()},
          "arrow_minus_random": {"point": (A0 - R0).tolist(), "lo": lo_D.tolist(), "hi": hi_D.tolist()},
          "lexical": {"point": lex0.tolist(), "lo": lo_L.tolist(), "hi": hi_L.tolist()},
          "thresholds": {"auroc_lower": T_AUROC, "diff_lower": T_DIFF, "near_within": NEAR_W}}
    verdicts = {}
    for name, row in (("guilt_clean", 2), ("shame_clean", 3)):
        v = gate(lo_A[row], hi_A[row], lo_D[row], hi_D[row])
        best = "survives" if "survives" in v else ("NEAR" if "NEAR" in v else "fails")
        verdicts[name] = {"per_layer": v, "verdict": best,
                          "layers_survives": [L for L in LAYERS if v[L] == "survives"],
                          "layers_NEAR": [L for L in LAYERS if v[L] == "NEAR"]}
    surv = [n for n in verdicts if verdicts[n]["verdict"] == "survives"]
    t3["gate"] = verdicts
    t3["branch"] = ("STAGE0 §6 instrument-fails branch applies (no arrow survives); the researcher decides at the S2 gate"
                    if not surv else f"instrument = {', '.join(surv)}; the researcher decides at the S2 gate")
    save_json(t3, RAW / "task3_validation.json")

    # ---------------- Task 4
    lo_C, hi_C = pct_ci(Cb)
    t4 = {"layers": LAYERS, "cos_clean": {"point": C0["guilt_clean,shame_clean"].tolist(), "lo": lo_C.tolist(), "hi": hi_C.tolist()},
          "raw": {k: v.tolist() for k, v in C0.items() if k != "guilt_clean,shame_clean"},
          "regime": [regime(lo_C[L], hi_C[L]) for L in LAYERS],
          "regime_rule": "near -1: CI upper < -0.5; near 0: CI within (-0.25, 0.25); strongly positive: CI lower > 0.5; else straddling"}
    save_json(t4, RAW / "task4_angle.json")

    # ---------------- Task 5 (from the saved arrows file, all scenarios)
    units, norms, _ = load_s2_arrows(DEV)
    sweep = load_sweep(DEV)
    t5 = {"layers": LAYERS, "cross_voice": {}, "distinctness": {}}
    for a, b in (("guilt_clean", "received_act"), ("shame_clean", "received_self"), ("guilt_clean", "received_self"), ("shame_clean", "received_act")):
        t5["cross_voice"][f"{a},{b}"] = cos_layers(units[a], units[b]).cpu().tolist()
    for a in ("guilt_clean", "shame_clean", "received_act", "received_self"):
        t5["distinctness"][a] = {k: cos_layers(units[a], sweep[k]).cpu().tolist() for k in ("refusal", "badmed", "persona")}
        t5["distinctness"][a]["random_seed0"] = cos_layers(units[a], D.R).cpu().tolist()
    save_json(t5, RAW / "task5_crossvoice.json")

    # ---------------- Task 6
    if len(surv) == 2:
        score = (A0[2] + A0[3]) / 2; basis = "rows 3+4 (both survive)"
    elif surv == ["guilt_clean"]:
        score = A0[2]; basis = "row 3 (guilt_clean survives)"
    elif surv == ["shame_clean"]:
        score = A0[3]; basis = "row 4 (shame_clean survives)"
    else:
        score = (A0[2] + A0[3]) / 2; basis = "rows 3+4 (no arrow survives): exploratory — instrument not validated"
    valid, runs = band(score, lo_C, hi_C)
    t6 = {"basis": basis, "exploratory": len(surv) == 0, "chosen": valid[0] if valid else None,
          "runners_up": valid[1:3], "n_valid_runs": len(valid), "n_runs": len(runs), "all_runs": runs,
          "centre_layer": valid[0]["centre"] if valid else None}
    save_json(t6, RAW / "task6_band.json")
    print("gate:", {k: v["verdict"] for k, v in verdicts.items()}, "\nbranch:", t3["branch"])
    print("band:", t6["chosen"], "\nrunners-up:", t6["runners_up"])
    print(f"done {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
