"""S3 Phase B — Task 2c step 3: the persona axis (Assistant Axis, Lu et al. PCA recipe) at 8B, per layer.

Per role: mean over its (system prompt x question) responses of the mean response-token residual (from
persona_activations.py; no judge filter — brief as written). Default-Assistant vector = the same over
default.json's prompts. Per layer L: X = 275 role means (275 x 4096), centred by the across-role mean;
PC1 = top right-singular vector of X_c, oriented so that (default − mean_roles) projects positively.
Reports: PC1 explained variance, (i) internal cosine cos(PC1, default − mean_roles), role projections on our
axis (for the cross-model rank check), inter-axis cosines with refusal/badmed/random, and writes
units['persona'] (= oriented unit PC1) and norms['persona'] (= |default − mean_roles|, the natural scale)
into directions/dirs_8B_base_sweep.pt. The cross-model Spearman uses the authors' published per-role vectors
(HF lu-christina/assistant-axis-vectors) — ordering derived from the authors' published vectors, not from the
paper text, which gives no ranked list.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from randctl import random_unit_directions  # noqa: E402
from s3_phaseB.common import D_MODEL, LAYERS, N_LAYERS, RAW, ROOT, cos, save_json, unit  # noqa: E402

ACT = RAW / "persona/activations"
PAPER = {"gemma-2-27b": 22, "qwen-3-32b": 32, "llama-3.3-70b": 40}  # target layers from the authors' README / models.py
VEC_ROOT = Path("/workspace/hf/hub/datasets--lu-christina--assistant-axis-vectors/snapshots/3b3b788432ad33e3a28d9ff08e88a530c0740814")


def main():
    files = sorted(ACT.glob("*.pt"))
    roles, means, counts, norm_sum, tok_count = [], [], {}, torch.zeros(N_LAYERS, dtype=torch.float64), 0
    default = None
    for f in files:
        d = torch.load(f)
        s, c = d["sum"].double(), d["count"]
        vec = s.sum(0) / c.sum()  # (32, 4096): mean over all responses of the role (all prompts pooled)
        norm_sum += d["norm_sum"].double(); tok_count += d["tok_count"]
        if d["role"] == "default":
            default = vec; counts["default"] = int(c.sum())
        else:
            roles.append(d["role"]); means.append(vec); counts[d["role"]] = int(c.sum())
    assert default is not None and len(roles) == 275, (len(roles), default is None)
    X = torch.stack(means)  # (275, 32, 4096)
    mean_roles = X.mean(0)
    meandiff = default - mean_roles  # (32, 4096), the plain Assistant-minus-mean-roles vector
    mean_resid_norm = (norm_sum / tok_count).float()

    units_p, pc1_var, top5_var, cos_internal, proj_roles = {}, {}, {}, {}, {}
    for L in LAYERS:
        Xc = (X[:, L, :] - mean_roles[L]).numpy()
        U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
        pc1 = torch.tensor(Vt[0], dtype=torch.float32)
        if float(pc1.double() @ meandiff[L]) < 0:
            pc1 = -pc1
        var = S**2 / (S**2).sum()
        units_p[L] = pc1
        pc1_var[L] = round(float(var[0]), 4); top5_var[L] = [round(float(v), 4) for v in var[:5]]
        cos_internal[L] = round(cos(pc1, meandiff[L].float()), 4)
        proj_roles[L] = (Xc @ pc1.numpy()).tolist()  # centred projections of the 275 roles on our axis

    # cross-model rank check against the authors' published vectors
    cross = {}
    for mdl, tl in PAPER.items():
        base = VEC_ROOT / mdl
        if not (base / "assistant_axis.pt").exists():
            cross[mdl] = {"status": "vectors not available locally"}; continue
        axis = torch.load(base / "assistant_axis.pt", map_location="cpu", weights_only=False)
        axis = axis["axis"] if isinstance(axis, dict) else axis
        theirs, missing = {}, []
        for r in roles:
            p = base / "role_vectors" / f"{r}.pt"
            if not p.exists():
                missing.append(r); continue
            v = torch.load(p, map_location="cpu", weights_only=False)
            v = v["vector"] if isinstance(v, dict) else v
            a = axis[tl].float()
            theirs[r] = float(v[tl].float() @ (a / a.norm()))
        common = [r for r in roles if r in theirs]
        their_order = np.array([theirs[r] for r in common])
        rho_by_layer = {}
        for L in LAYERS:
            ours = np.array([proj_roles[L][roles.index(r)] for r in common])
            rho, p = spearmanr(ours, their_order)
            rho_by_layer[L] = [round(float(rho), 4), float(p)]
        cross[mdl] = {"their_target_layer": tl, "n_common_roles": len(common), "missing": missing, "spearman_by_our_layer": rho_by_layer,
                      "their_top5": [common[i] for i in np.argsort(-their_order)[:5]], "their_bottom5": [common[i] for i in np.argsort(their_order)[:5]],
                      "source": "ordering derived from the authors' published vectors (lu-christina/assistant-axis-vectors @3b3b7884), not from the paper text"}

    # write into the sweep file
    d = torch.load(ROOT / "directions/dirs_8B_base_sweep.pt")
    d["units"]["persona"] = units_p
    d["norms"]["persona"] = {L: round(float(meandiff[L].norm()), 4) for L in LAYERS}
    d["persona_meandiff_units"] = {L: unit(meandiff[L].float()) for L in LAYERS}
    d["meta"]["recipes"]["persona"] = ("Lu et al. Assistant Axis: 275 roles x 5 system prompts x questions, mean response-token residual per role, "
                                       "PC1 of centred role means per layer, oriented to default-Assistant; norms = |default - mean_roles|; no judge filter")
    d["meta"]["persona"] = {"n_roles": len(roles), "counts": counts, "generation_meta": json.load(open(RAW / "persona/generation_meta.json")),
                            "mean_response_resid_norm": {L: round(float(mean_resid_norm[L]), 3) for L in LAYERS}}
    torch.save(d, ROOT / "directions/dirs_8B_base_sweep.pt")

    rnd = random_unit_directions(D_MODEL, LAYERS, d["random_seed"])
    inter = {"refusal~persona": {L: round(cos(d["units"]["refusal"][L], units_p[L]), 4) for L in LAYERS},
             "badmed~persona": {L: round(cos(d["units"]["badmed"][L], units_p[L]), 4) for L in LAYERS},
             "persona~random": {L: round(cos(units_p[L], rnd[L]), 4) for L in LAYERS}}
    out = {"roles": roles, "counts": counts, "pc1_explained_var": pc1_var, "top5_explained_var": top5_var,
           "cos_pc1_vs_meandiff": cos_internal, "meandiff_norm": d["norms"]["persona"], "inter_axis": inter,
           "mean_response_resid_norm": d["meta"]["persona"]["mean_response_resid_norm"], "cross_model": cross,
           "role_proj_on_our_axis": proj_roles,
           "our_top5_bottom5_by_layer": {L: {"top": [roles[i] for i in np.argsort(-np.array(proj_roles[L]))[:5]],
                                             "bottom": [roles[i] for i in np.argsort(np.array(proj_roles[L]))[:5]]} for L in LAYERS}}
    save_json(out, RAW / "persona_axis.json")
    print("PC1 var:", pc1_var); print("cos(PC1, meandiff):", cos_internal); print("inter:", inter)
    for m, c in cross.items():
        print(m, {k: v for k, v in c.items() if k != "spearman_by_our_layer"})
        if "spearman_by_our_layer" in c:
            print("  rho by our layer:", {L: v[0] for L, v in c["spearman_by_our_layer"].items()})
    print("DONE")


if __name__ == "__main__":
    main()
