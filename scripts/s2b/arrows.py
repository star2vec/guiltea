"""S2b Task 2 — arrows on all 50 scenarios, per layer (briefs/S2b-arrows.md Task 2).

guilt = mean(guilt) − mean(baseline); shame likewise; nn = mean(neutral_negative) − mean(baseline);
ĝ = guilt − (guilt·n̂)n̂, ŝ = shame − (shame·n̂)n̂ (n̂ = nn/‖nn‖); received_act/self at feedback_mean;
difference = ŝ/‖ŝ‖ − ĝ/‖ĝ‖ (derived, never primary).
Writes directions/dirs_8B_s2_arrows.pt in the dirs_8B_base_sweep.pt style (units as {name: {layer: unit vector}},
norms as {name: {layer: float}}), plus results/raw/s2b/task2_arrows.json (norms, fraction kept, per layer).
"""
from __future__ import annotations

import datetime as dt

import numpy as np
import torch

from s2b_common import (ARROW_NAMES, DIRS, FP_CLASSES, LAYERS, RAW, SP_CLASSES, c8, class_masks, cos_layers,
                        fp_arrows, load_acts, load_scenarios, passage_set_commit, save_json, sp_arrows,
                        weighted_means)


def main():
    scen = load_scenarios()
    Xf, rows_f = load_acts("first_person_mean")
    Xs, rows_s = load_acts("second_person_feedback_mean")
    mf, ms = class_masks(rows_f, FP_CLASSES), class_masks(rows_s, SP_CLASSES)
    Wf = torch.tensor(np.stack([mf[c] for c in FP_CLASSES]).astype(np.float32))
    Ws = torch.tensor(np.stack([ms[c] for c in SP_CLASSES]).astype(np.float32))
    Mf = dict(zip(FP_CLASSES, weighted_means(Xf, Wf)))
    Ms = dict(zip(SP_CLASSES, weighted_means(Xs, Ws)))
    arrows = {**fp_arrows(Mf), **sp_arrows(Ms)}
    assert set(arrows) == set(ARROW_NAMES)

    norms = {k: v.norm(dim=-1) for k, v in arrows.items()}
    units = {k: v / norms[k][:, None] for k, v in arrows.items()}
    kept = {"guilt_clean_over_guilt": (norms["guilt_clean"] / norms["guilt"]).tolist(),
            "shame_clean_over_shame": (norms["shame_clean"] / norms["shame"]).tolist()}
    # sanity: cleaned arrows orthogonal to nn
    resid = {"cos(guilt_clean,nn)": cos_layers(arrows["guilt_clean"], arrows["nn"]).abs().max().item(),
             "cos(shame_clean,nn)": cos_layers(arrows["shame_clean"], arrows["nn"]).abs().max().item()}
    assert max(resid.values()) < 1e-4, resid

    n_per_class = {**{c: int(mf[c].sum()) for c in FP_CLASSES}, **{c: int(ms[c].sum()) for c in SP_CLASSES}}
    meta = {"model": c8.BASE, "model_revision": c8.BASE_REV, "precision": "bf16", "n_per_class": n_per_class,
            "n_scenarios": len(scen), "passage_set_commit": passage_set_commit(),
            "date": dt.datetime.now(dt.timezone.utc).date().isoformat(), "brief": "briefs/S2b-arrows.md Task 2",
            "residual": "hidden_states[L+1]", "L31_note": "post-final-norm in HF; excluded from band candidacy, reported separately",
            "definitions": {"guilt": "mean(guilt) - mean(baseline)", "shame": "mean(shame) - mean(baseline)",
                            "nn": "mean(neutral_negative) - mean(baseline)",
                            "guilt_clean": "guilt - (guilt . n)n, n = nn/|nn|", "shame_clean": "shame - (shame . n)n",
                            "received_act": "mean(act_blame) - mean(neutral_correction) at feedback_mean",
                            "received_self": "mean(self_blame) - mean(neutral_correction) at feedback_mean",
                            "difference": "shame_clean/|shame_clean| - guilt_clean/|guilt_clean| (derived, never primary)"},
            "fraction_of_norm_kept_by_cleaning": kept}
    out = {"units": {k: {L: units[k][i].clone() for i, L in enumerate(LAYERS)} for k in ARROW_NAMES},
           "norms": {k: {L: float(norms[k][i]) for i, L in enumerate(LAYERS)} for k in ARROW_NAMES},
           "layers": LAYERS,
           "position": {"first_person": "mean over assistant-turn tokens incl. closing <|eot_id|> (common.resid_answer_mean)",
                        "second_person": "feedback_mean: mean over user-turn tokens incl. closing <|eot_id|>"},
           "random_seed": 0, "meta": meta}
    torch.save(out, DIRS / "dirs_8B_s2_arrows.pt")
    table = {"layers": LAYERS, "norms": {k: norms[k].tolist() for k in ARROW_NAMES}, "fraction_kept": kept,
             "raw_cos": {"guilt,shame": cos_layers(arrows["guilt"], arrows["shame"]).tolist(),
                         "guilt,nn": cos_layers(arrows["guilt"], arrows["nn"]).tolist(),
                         "shame,nn": cos_layers(arrows["shame"], arrows["nn"]).tolist(),
                         "guilt_clean,shame_clean": cos_layers(arrows["guilt_clean"], arrows["shame_clean"]).tolist()},
             "orthogonality_check_max_abs_cos": resid, "meta": meta}
    save_json(table, RAW / "task2_arrows.json")
    print("saved", DIRS / "dirs_8B_s2_arrows.pt")
    for i, L in enumerate(LAYERS):
        print(f"L{L:2d} |guilt| {norms['guilt'][i]:.3f} |shame| {norms['shame'][i]:.3f} |nn| {norms['nn'][i]:.3f} "
              f"kept g {kept['guilt_clean_over_guilt'][i]:.3f} s {kept['shame_clean_over_shame'][i]:.3f} "
              f"cos(ĝ,ŝ) {table['raw_cos']['guilt_clean,shame_clean'][i]:+.3f} |ra| {norms['received_act'][i]:.3f} |rs| {norms['received_self'][i]:.3f}")


if __name__ == "__main__":
    main()
