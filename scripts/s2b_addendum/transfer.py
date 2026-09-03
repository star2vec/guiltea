"""S2b addendum Task A — cross-voice transfer (briefs/S2b-addendum.md Task A).

The decisive lexical control: the two voices use different words, so a word-based classifier trained
on one voice should transfer poorly to the other while a concept arrow should transfer.

Folds and bootstrap: exactly the S2b Task 3 machinery, imported from scripts/s2b/validate.py — five
folds over scenario ids (numpy default_rng(0) permutation of scenarios.jsonl file order, np.array_split
into 5 blocks of 10; every framing of a scenario in its fold); within each fold every arrow (incl. nn
for cleaning) is re-extracted from the training scenarios only; 1,000-resample bootstrap over scenario
ids (default_rng(0)), folds re-drawn per resample (default_rng(1_000_003 + b)), a duplicated scenario
keeping one fold and counting with its multiplicity (weighted means, weighted AUROC); CI = 2.5 / 97.5
percentiles; point estimates from the seed-0 folds on the original data.

(i) first -> second: g_hat, s_hat from the training scenarios' FIRST-person passages (`mean`); the
    held-out scenarios' SECOND-person passages scored at `feedback_mean`; AUROC act_blame vs self_blame:
      i_a   x . (s_hat/|s_hat| - g_hat/|g_hat|)     positive class = self_blame
      i_b   x . s_hat                               positive class = self_blame
      i_c   x . g_hat                               positive class = act_blame
(ii) second -> first: received_act, received_self from the training scenarios' SECOND-person passages
    (`feedback_mean`); the held-out scenarios' FIRST-person passages scored at `mean`:
      ii    x . (received_self - received_act)       positive class = shame

Each row is oriented so that the arrow's own predicted direction is the positive class; the opposite
orientation is 1 - the reported value. Score vectors follow the addendum's own notation literally:
unit-normalised difference in (i), raw difference in (ii). AUROC on a single arrow is invariant to
positive rescaling, so i_b / i_c do not depend on whether the arrow is normalised.

Beside every number: the seed-0 random unit arrow (same rows, folds and orientation; arrow - random
paired within resample — the Task 3 convention), and the lexical transfer baseline: binary unigram
CountVectorizer + L2 LogisticRegression (C = 1, liblinear, sample-weighted; the Task 3 settings) fit on
the training scenarios in the SOURCE voice and applied to the held-out scenarios in the TARGET voice,
under the class correspondence guilt <-> act_blame, shame <-> self_blame. AUROC is invariant under
flipping score and label together, so one lexical number per direction serves every row of it.

Pre-stated reading (addendum Task A): the arrows carry more than the words if, at some layer <= 30, the
arrow CI lower bound exceeds the lexical CI upper bound by >= 0.10; NEAR within 0.05; otherwise not shown.

Usage: python scripts/s2b_addendum/transfer.py [--boot 1000]
"""
from __future__ import annotations

import argparse
import datetime as dt
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "s2b"))

import validate as V  # noqa: E402  (scripts/s2b/validate.py — folds, bootstrap, arrow re-extraction)

V.DEV = "cpu"  # Task A is a CPU task (addendum)

from s2b_common import LAYERS, auroc_w, pct_ci, save_json, unit  # noqa: E402

OUT = ROOT / "results" / "raw" / "s2b_addendum"

# key, direction, score description, positive class, negative class
ROWS = [
    ("i_a", "first->second", "x . (s_hat/|s_hat| - g_hat/|g_hat|)", "self_blame", "act_blame"),
    ("i_b", "first->second", "x . s_hat", "self_blame", "act_blame"),
    ("i_c", "first->second", "x . g_hat", "act_blame", "self_blame"),
    ("ii", "second->first", "x . (received_self - received_act)", "shame", "guilt"),
]
DIRECTIONS = ["first->second", "second->first"]
T_SHOWN, T_NEAR = 0.10, 0.05


def score_vectors(arrows):
    ug, us = unit(arrows["guilt_clean"]), unit(arrows["shame_clean"])
    return {"i_a": us - ug, "i_b": us, "i_c": ug,
            "ii": arrows["received_self"] - arrows["received_act"]}


def evaluate(D, counts: np.ndarray, fold_of: np.ndarray):
    """Returns arrow AUROC [4, L] and random AUROC [4, L], means over folds."""
    A = torch.zeros(len(ROWS), len(LAYERS))
    Rr = torch.zeros_like(A)
    nf = 0
    for k in range(5):
        tr = counts * (fold_of != k)
        if tr.sum() == 0 or (counts * (fold_of == k)).sum() == 0:
            continue
        nf += 1
        vecs = score_vectors(V.arrows_from_counts(D, tr, tr))
        for i, (key, direction, _, pos, neg) in enumerate(ROWS):
            X, sidx, masks = (D.Xs, D.ss, D.ms) if direction == "first->second" else (D.Xf, D.sf, D.mf)
            te = (fold_of[sidx] == k) & (counts[sidx] > 0)
            pm, nm = te & masks[pos], te & masks[neg]
            w_pos = torch.tensor(counts[sidx[pm]], dtype=torch.float32)
            w_neg = torch.tensor(counts[sidx[nm]], dtype=torch.float32)
            Xp, Xn = X[torch.tensor(pm)], X[torch.tensor(nm)]
            u = vecs[key]
            A[i] += auroc_w(torch.einsum("nld,ld->nl", Xp, u), torch.einsum("nld,ld->nl", Xn, u), w_pos, w_neg)
            Rr[i] += auroc_w(torch.einsum("nld,ld->nl", Xp, D.R), torch.einsum("nld,ld->nl", Xn, D.R), w_pos, w_neg)
    return (A / nf).numpy(), (Rr / nf).numpy()


def lexical_transfer(tf, ts, sf, ss, mf, ms, counts: np.ndarray, fold_of: np.ndarray):
    """Bag-of-words logistic trained in the source voice, applied to the held-out target voice.
    Returns [2]: first->second, second->first."""
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    specs = [(tf, sf, mf, "shame", "guilt", ts, ss, ms, "self_blame", "act_blame"),
             (ts, ss, ms, "self_blame", "act_blame", tf, sf, mf, "shame", "guilt")]
    out = np.zeros(2)
    nf = 0
    for k in range(5):
        if (counts * (fold_of != k)).sum() == 0 or (counts * (fold_of == k)).sum() == 0:
            continue
        nf += 1
        for j, (ttx, tsi, tms, tpos, tneg, etx, esi, ems, epos, eneg) in enumerate(specs):
            tr = (tms[tpos] | tms[tneg]) & (fold_of[tsi] != k) & (counts[tsi] > 0)
            te = (ems[epos] | ems[eneg]) & (fold_of[esi] == k) & (counts[esi] > 0)
            ytr, yte = tms[tpos].astype(int), ems[epos].astype(int)
            vec = CountVectorizer(binary=True, lowercase=True)
            Xtr = vec.fit_transform([ttx[q] for q in np.nonzero(tr)[0]])
            Xte = vec.transform([etx[q] for q in np.nonzero(te)[0]])
            clf = LogisticRegression(penalty="l2", C=1.0, solver="liblinear", max_iter=1000)
            clf.fit(Xtr, ytr[tr], sample_weight=counts[tsi[tr]])
            out[j] += roc_auc_score(yte[te], clf.predict_proba(Xte)[:, 1], sample_weight=counts[esi[te]])
    return out / nf


def reading(lo_arrow, hi_lex):
    """Pre-stated reading per row, over layers <= 30."""
    margin = [float(lo_arrow[L] - hi_lex) for L in LAYERS]
    cand = [(margin[L], L) for L in LAYERS if L <= 30]
    best_m, best_L = max(cand)
    verdict = "shown" if best_m >= T_SHOWN else ("NEAR" if best_m >= T_NEAR else "not shown")
    return {"margin_per_layer": margin, "best_layer": int(best_L), "best_margin": best_m,
            "layers_shown": [L for L in LAYERS if L <= 30 and margin[L] >= T_SHOWN],
            "layers_NEAR": [L for L in LAYERS if L <= 30 and T_NEAR <= margin[L] < T_SHOWN],
            "verdict": verdict}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--boot", type=int, default=1000)
    a = ap.parse_args()
    t0 = time.time()
    torch.set_num_threads(min(32, torch.get_num_threads() or 8))

    D = V.Data()
    fold0 = V.seed0_folds(D.n_scen)
    ones = np.ones(D.n_scen)
    A0, R0 = evaluate(D, ones, fold0)
    lex0 = lexical_transfer(D.tf, D.ts, D.sf, D.ss, D.mf, D.ms, ones, fold0)
    print(f"point estimates done {time.time()-t0:.0f}s  arrow={np.round(A0[:, :6], 3).tolist()} lex={lex0}", flush=True)

    rng = np.random.default_rng(0)
    draws = [rng.choice(D.n_scen, D.n_scen, replace=True) for _ in range(a.boot)]
    counts_l = [np.bincount(d, minlength=D.n_scen).astype(float) for d in draws]
    folds_l = [V.resample_folds(c, b) for b, c in enumerate(counts_l)]

    Ab, Rb = [], []
    for b in range(a.boot):
        x, y = evaluate(D, counts_l[b], folds_l[b])
        Ab.append(x); Rb.append(y)
        if b % 100 == 0:
            print(f"boot {b}/{a.boot} {time.time()-t0:.0f}s", flush=True)
    Ab, Rb = np.stack(Ab), np.stack(Rb)
    Db = Ab - Rb
    print(f"arrow bootstrap done {time.time()-t0:.0f}s; lexical bootstrap ...", flush=True)
    from joblib import Parallel, delayed
    Lb = np.stack(Parallel(n_jobs=48)(delayed(lexical_transfer)(D.tf, D.ts, D.sf, D.ss, D.mf, D.ms, counts_l[b], folds_l[b])
                                      for b in range(a.boot)))
    print(f"lexical bootstrap done {time.time()-t0:.0f}s", flush=True)

    lo_A, hi_A = pct_ci(Ab); lo_R, hi_R = pct_ci(Rb); lo_D, hi_D = pct_ci(Db); lo_L, hi_L = pct_ci(Lb)
    res = {
        "task": "S2b addendum Task A — cross-voice transfer",
        "date": dt.datetime.now(dt.timezone.utc).isoformat(),
        "layers": LAYERS, "n_boot": a.boot,
        "rows": [{"key": k, "direction": d, "score": s, "positive_class": p, "negative_class": n} for k, d, s, p, n in ROWS],
        "directions": DIRECTIONS,
        "folds_seed0": {int(k): [D.scen[i]["id"] for i in np.nonzero(fold0 == k)[0]] for k in range(5)},
        "arrow": {"point": A0.tolist(), "lo": lo_A.tolist(), "hi": hi_A.tolist()},
        "random_seed0": {"point": R0.tolist(), "lo": lo_R.tolist(), "hi": hi_R.tolist()},
        "arrow_minus_random": {"point": (A0 - R0).tolist(), "lo": lo_D.tolist(), "hi": hi_D.tolist()},
        "lexical_transfer": {"point": lex0.tolist(), "lo": lo_L.tolist(), "hi": hi_L.tolist()},
        "reading_rule": {"shown_if_margin_ge": T_SHOWN, "near_if_margin_ge": T_NEAR,
                         "margin": "arrow CI lower bound - lexical transfer CI upper bound, layers <= 30"},
        "reading": {},
        "informational_band_evidence": {},
    }
    for i, (k, d, s, p, n) in enumerate(ROWS):
        j = DIRECTIONS.index(d)
        res["reading"][k] = reading(lo_A[i], hi_L[j])
        prof = A0[i][:31]
        res["informational_band_evidence"][k] = {
            "argmax_layer_le30": int(np.argmax(prof)), "max_point_le30": float(prof.max()),
            "top5_layers_le30": [int(x) for x in np.argsort(-prof)[:5]],
            "note": "informational only; the addendum forbids reading a band choice from this profile"}
    save_json(res, OUT / "taskA_transfer.json")
    for i, (k, d, s, p, n) in enumerate(ROWS):
        r = res["reading"][k]
        print(f"{k} ({d}, pos={p}): point L{r['best_layer']} = {A0[i][r['best_layer']]:.3f} "
              f"[{lo_A[i][r['best_layer']]:.3f}, {hi_A[i][r['best_layer']]:.3f}]  "
              f"lex {lex0[DIRECTIONS.index(d)]:.3f} [{lo_L[DIRECTIONS.index(d)]:.3f}, {hi_L[DIRECTIONS.index(d)]:.3f}]  "
              f"margin {r['best_margin']:+.3f} -> {r['verdict']}")
    print(f"done {time.time()-t0:.0f}s -> {OUT/'taskA_transfer.json'}")


if __name__ == "__main__":
    main()
