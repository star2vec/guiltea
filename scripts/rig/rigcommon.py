"""S3 rig — profile layer: model, arrows, readout positions, storage, judge ledger.

The rig reuses `scripts/s1b/` (loaders, `chains.run_chain_batch`, `Judges`, the ledger) without editing it:
this module rebinds the s1b module globals for the chosen profile, so the same chain runner drives the 1B
sandbox and the 8B subject. Nothing under `scripts/s1b/`, `data/`, `directions/` or `data/eval/` is modified.

Two things the S4 rig needs that S1b did not, and why they are here rather than in `s1bcommon`:
- a fourth readout position, ``feedback_mean`` — the second-person position the S2b received-blame arrows were
  extracted at (`directions/PROVENANCE.md`: "second person `feedback_mean` = [mean over tokens incl. the closing
  <|eot_id|>] over the user turn"). It is computed on every assistant turn as the mean over the *preceding user
  turn*, so the feedback-reply turn gets it for free and the `none` arm's act turn carries the same column;
- a layer split: residuals are stored at **every model layer** (brief step 2), projections are taken at **the
  layers the arrow files hold** (brief step 6). At 8B these coincide (0..31); at 1B the arrow file holds only
  layers 6..14, so they must be separate.

Position order is fixed: into, think, answer, feedback_mean. ``into`` is the brief's ``post`` (the last token of
the feedback turn, i.e. the last prompt token before the reply); positions that do not apply to a turn are NaN.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "s1b"))
sys.path.insert(0, str(ROOT / "scripts"))

import s1bcommon as S            # noqa: E402  the S1b rig, imported and configured, never edited
import judges as JU              # noqa: E402  scripts/s1b/judges.py
import judge_rubrics as J        # noqa: E402
from randctl import random_unit_directions  # noqa: E402

POSITIONS = ["into", "think", "answer", "feedback_mean"]
POST = "into"                    # the brief's `post` = the last token of the feedback turn
N_RANDOM = 10                    # randctl seeds 0..9; seed 0 is the control, 0..9 the floor
ESCALATION_BAND = (65, 90)       # D-021
ESCALATION_SAMPLE = 8            # fixed 1-in-8 seeded sample
ESCALATION_SALT = "s4-align-escalation-seed0"  # fixed before data (plan A7)

# The layer bands the S4 brief reads the readouts on (D-024). The rig still projects at, and the table still
# prints, every layer the arrow files hold; these only label which columns the brief asks to be summarised.
BANDS = {"primary": list(range(14, 19)), "secondary": list(range(6, 12))}

# Arrow provenance labels for the table header (STAGE0 amendment 2026-09-04 / D-023: the S2 gate read
# inconclusive, so the guilt/shame and received arrows enter S4 as exploratory readouts; the borrowed
# axes are the confirmatory ones). The rig projects on all of them regardless.
ARROW_CLASS = {
    "dirs_8B_base_sweep.pt": "confirmatory (borrowed axes, STAGE0 §3)",
    "dirs_base_sweep.pt": "sandbox (1B borrowed axes)",
    "dirs_8B_s2_arrows.pt": "exploratory (S2 arrows; D-023)",
}

PROFILES = {
    "1b": {
        "model": "unsloth/Llama-3.2-1B-Instruct",
        "revision": "5a8abab4a5d6f164389b1079fb721cfab8d7126c",
        "arrows": ["dirs_base_sweep.pt"],
        "dtype": torch.bfloat16,     # sandbox; matches the 8B convention and halves the MPS footprint
        "max_new_chain": 600,
        "sandbox": True,
    },
    "base": {
        "model": None,               # from scripts/s3_phaseB/common.py (the pinned SOURCE.md revision)
        "revision": None,
        "arrows": ["dirs_8B_base_sweep.pt", "dirs_8B_s2_arrows.pt"],
        "dtype": torch.bfloat16,
        "max_new_chain": 600,        # S1b rev.3 cap
        "sandbox": False,
    },
}
MAX_NEW_REPLY = 300      # S4-design §2a
MAX_NEW_UNRELATED = 300  # brief step 3(c)

# Batching by token budget, not by row count (Phase 2 operational fix; see reports/S3-rig.md §3).
# The brief fixes the *upper* bounds (<= 12 chain rows, 20 single-turn); on a 24 GiB card those bounds OOM at
# the long distance-4 contexts, because a prefill of [rows, T] carries activations linear in rows x T (the
# 14336-wide MLP intermediate at 12 x 7.5k tokens alone is > 3 GiB per tensor). Measured on this 4090 with the
# base resident (14.96 GiB): generation peaked at 21.97 GiB for rows x T = 30k and OOMed at 45k; the readout
# forward peaked at 22.44 GiB for 45k and 22.47 GiB for 30k (its lm_head logits are [rows, T, 128256]). The budgets below sit under those with ~2 GiB of headroom and are the
# *only* thing that changes: the row bounds, the seeds, the caps and every measured quantity are untouched.
GEN_TOKEN_BUDGET = int(os.environ.get("RIG_GEN_TOKEN_BUDGET", "28000"))
READOUT_TOKEN_BUDGET = int(os.environ.get("RIG_READOUT_TOKEN_BUDGET", "26000"))
BATCH_STATS = {"gen_calls": 0, "gen_chunks": 0, "gen_min_rows": None, "gen_max_ctx": 0,
               "readout_calls": 0, "readout_chunks": 0, "readout_min_rows": None, "readout_max_ctx": 0,
               "gen_token_budget": GEN_TOKEN_BUDGET, "readout_token_budget": READOUT_TOKEN_BUDGET}


def chunk_by_budget(lengths: Sequence[int], budget: int, extra: int = 0, max_rows: int = 10 ** 9) -> List[List[int]]:
    """Consecutive groups of row indices with rows x max(len + extra) <= budget (>= 1 row always). Order kept."""
    groups: List[List[int]] = []
    cur: List[int] = []
    cur_max = 0
    for i, n in enumerate(lengths):
        m = max(cur_max, int(n) + extra)
        if cur and ((len(cur) + 1) * m > budget or len(cur) + 1 > max_rows):
            groups.append(cur)
            cur, cur_max = [i], int(n) + extra
        else:
            cur.append(i)
            cur_max = m
    if cur:
        groups.append(cur)
    return groups


def make_gen_batch(inner):
    """Drop-in replacement for ``s1bcommon.gen_batch``: same call, chunked to fit the card.

    A chunk is generated with ``seed_base + 1000 * chunk_index`` so two chunks of one call do not draw from the
    same reseeded stream. Sampling was already per batch rather than per row (s1bcommon.gen_batch seeds once with
    seed_base), so this changes which draw a row gets, exactly as any batch-size choice does; it is fixed before
    the first cell and held constant across all five.
    """
    def gen_batch(model, tok, convs, seed_base: int, max_new: int = None):
        cap = S.MAX_NEW if max_new is None else max_new
        lens = [len(tok(S.render(tok, m, True), add_special_tokens=False)["input_ids"]) for m in convs]
        groups = chunk_by_budget(lens, GEN_TOKEN_BUDGET, extra=cap)
        BATCH_STATS["gen_calls"] += 1
        BATCH_STATS["gen_chunks"] += len(groups)
        BATCH_STATS["gen_max_ctx"] = max(BATCH_STATS["gen_max_ctx"], max(lens) if lens else 0)
        out = []
        for ci, idx in enumerate(groups):
            n = len(idx)
            BATCH_STATS["gen_min_rows"] = n if BATCH_STATS["gen_min_rows"] is None else min(BATCH_STATS["gen_min_rows"], n)
            out += inner(model, tok, [convs[i] for i in idx], seed_base=int(seed_base) + 1000 * ci, max_new=cap)
        return out
    return gen_batch


class Rig:
    """Everything that depends on the profile. Built once by ``configure``."""

    def __init__(self, profile: str, out_root: Path):
        self.profile = profile
        self.spec = PROFILES[profile]
        self.out_root = out_root
        self.sandbox = self.spec["sandbox"]
        self.dtype = self.spec["dtype"]
        self.device = _pick_device(profile)
        self.offline = False
        if profile == "base":
            self.model_id, self.revision = S.C.BASE, S.C.BASE_REV
            self.offline = prefer_offline(self.model_id, self.revision)
            self.d_model, self.n_model_layers = 4096, 32
        else:
            self.model_id, self.revision = self.spec["model"], self.spec["revision"]
            self.offline = prefer_offline(self.model_id, self.revision)
            self.d_model, self.n_model_layers = _config_dims(self.model_id, self.revision)
        self.resid_layers = list(range(self.n_model_layers))
        self.axes, self.arrow_layers, self.axis_source = _load_axes(
            [ROOT / "directions" / a for a in self.spec["arrows"]], self.d_model)
        self.arrow_index = [self.resid_layers.index(L) for L in self.arrow_layers]
        self.axis_names = list(self.axes.keys())
        self.named_axes = [a for a in self.axis_names if not a.startswith("random")]
        self._M = None

    # ---------------------------------------------------------------- projections
    def dir_matrix(self) -> torch.Tensor:
        """[n_axes, n_arrow_layers, d_model] float32 on the unit axes."""
        if self._M is None:
            self._M = torch.stack([torch.stack([self.axes[a][L] for L in self.arrow_layers])
                                   for a in self.axis_names])
        return self._M

    def project(self, resid: torch.Tensor) -> torch.Tensor:
        """resid [..., n_model_layers, d] -> [..., n_axes, n_arrow_layers] float32."""
        r = resid.float()[..., self.arrow_index, :]
        return torch.einsum("...ld,ald->...al", r, self.dir_matrix())

    # ---------------------------------------------------------------- header facts
    def arrow_header(self) -> dict:
        missing = []
        for want in ("guilt_clean", "shame_clean", "received_act", "received_self"):
            if want not in self.axes:
                missing.append(want)
        return {"files": [str(Path(a).name) for a in self.spec["arrows"]],
                "classes": {Path(a).name: ARROW_CLASS.get(Path(a).name, "unlabelled") for a in self.spec["arrows"]},
                "named_axes": self.named_axes, "random_axes": N_RANDOM,
                "arrow_layers": self.arrow_layers, "resid_layers": self.resid_layers,
                "axis_source": self.axis_source,
                "arrows_the_design_names_that_this_set_lacks": missing}

    def machine_info(self) -> dict:
        info = {"hostname": os.uname().nodename, "torch": torch.__version__, "device": self.device,
                "dtype": str(self.dtype), "model": self.model_id, "revision": self.revision,
                "profile": self.profile, "hf_hub_offline": self.offline}
        if self.device == "cuda":
            p = torch.cuda.get_device_properties(0)
            info.update({"gpu": p.name, "vram_gib": round(p.total_memory / 2 ** 30, 2), "cuda": torch.version.cuda})
        return info


def _pick_device(profile: str) -> str:
    if profile == "base":
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def cache_snapshot(model_id: str, revision: Optional[str]) -> Optional[Path]:
    """The local HF snapshot directory for this pinned revision, or None."""
    if not revision:
        return None
    root = Path(os.environ.get("HF_HUB_CACHE") or (Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "hub"))
    d = root / ("models--" + model_id.replace("/", "--")) / "snapshots" / revision
    return d if d.is_dir() else None


def prefer_offline(model_id: str, revision: Optional[str]) -> bool:
    """If the pinned revision is already in the cache, take the hub out of the loop entirely.

    `local_files_only=True` is not enough: on a machine with no route out, huggingface_hub still opens a socket
    and the loader blocks in connect() until the TCP timeout (observed on this Mac: minutes per call, silent).
    HF_HUB_OFFLINE must be set before transformers is first imported, which is why this runs in Rig.__init__.
    An explicit HF_HUB_OFFLINE in the environment is left alone; an uncached model still reaches the hub.
    """
    if os.environ.get("HF_HUB_OFFLINE") is not None:
        return os.environ["HF_HUB_OFFLINE"] == "1"
    if cache_snapshot(model_id, revision) is not None:
        os.environ["HF_HUB_OFFLINE"] = "1"
        return True
    return False


def _from_pretrained(loader, model_id, revision, **kw):
    """Prefer the local cache; fall back to the hub only when the revision is genuinely not cached."""
    if cache_snapshot(model_id, revision) is not None:
        return loader.from_pretrained(model_id, revision=revision, local_files_only=True, **kw), "cache"
    return loader.from_pretrained(model_id, revision=revision, **kw), "hub"


def _config_dims(model_id: str, revision: Optional[str]):
    from transformers import AutoConfig
    c, _ = _from_pretrained(AutoConfig, model_id, revision)
    return int(c.hidden_size), int(c.num_hidden_layers)


def _load_axes(paths: Sequence[Path], d_model: int):
    """Every unit in every arrow file (brief step 6) + randctl seeds 0..9, at the layers every unit holds."""
    axes: Dict[str, Dict[int, torch.Tensor]] = {}
    source: Dict[str, str] = {}
    for p in paths:
        blob = torch.load(p, map_location="cpu", weights_only=False)
        for name, per_layer in blob["units"].items():
            if name in axes:
                raise ValueError("arrow name %r appears in two files (%s, %s)" % (name, source[name], p.name))
            vecs = {int(L): v.float() for L, v in per_layer.items()}
            got = {tuple(v.shape) for v in vecs.values()}
            if got != {(d_model,)}:
                raise ValueError("arrow %r in %s has shape(s) %s, expected (%d,)" % (name, p.name, got, d_model))
            axes[name] = vecs
            source[name] = p.name
    layers = sorted(set.intersection(*[set(v) for v in axes.values()])) if axes else []
    if not layers:
        raise ValueError("no layer is held by every arrow in %s" % [p.name for p in paths])
    for name in list(axes):
        axes[name] = {L: axes[name][L] for L in layers}
    for s in range(N_RANDOM):
        axes["random%d" % s] = random_unit_directions(d_model, layers, seed=s)
        source["random%d" % s] = "randctl seed %d" % s
    return axes, layers, source


# --------------------------------------------------------------------------- readout
USER_SPAN_MISSES = {"no_user_turn": 0, "not_a_prefix": 0, "content_not_found": 0, "empty": 0}


def _user_span_chars(tok, messages):
    """Character span of the last user message's content .. end of its block, in render(messages, add_gen=False).

    This is the `feedback_mean` position: the S2b second-person convention is the `resid_answer_mean` recipe
    applied to the user turn, i.e. the turn's own tokens (its role header excluded, its closing <|eot_id|>
    included). Misses are counted, never silently ignored: the position becomes NaN and the count is reported.
    """
    if len(messages) < 2 or messages[-2]["role"] != "user":
        USER_SPAN_MISSES["no_user_turn"] += 1
        return None
    before = S.render(tok, messages[:-2], False) if len(messages) > 2 else ""
    with_user = S.render(tok, messages[:-1], False)
    if not with_user.startswith(before):
        USER_SPAN_MISSES["not_a_prefix"] += 1
        return None
    content = messages[-2]["content"]
    if not content:
        USER_SPAN_MISSES["empty"] += 1
        return None
    start = with_user.find(content, len(before))
    if start < 0:
        USER_SPAN_MISSES["content_not_found"] += 1
        return None
    return start, len(with_user)


class _Cap:
    """One forward, four positions, residuals at every model layer."""

    def __init__(self, n_rows: int, spans: list, rig: Rig, device):
        self.n_layers = len(rig.resid_layers)
        self.out = torch.full((n_rows, len(POSITIONS), self.n_layers, rig.d_model), float("nan"), dtype=torch.float32)
        self.spans = spans  # per row: list of (kind, payload) per position; kind in {"point","mean",None}
        self.device = device

    def hook(self, i: int, L: int):
        def f(module, inp, output):
            h = output[0] if isinstance(output, tuple) else output  # [rows, T, d]
            for r, per_pos in enumerate(self.spans):
                for p, sp in enumerate(per_pos):
                    if sp is None:
                        continue
                    kind, payload = sp
                    if kind == "point":
                        self.out[r, p, i] = h[r, payload].float().cpu()
                    elif len(payload):
                        self.out[r, p, i] = h[r, payload.to(h.device)].float().mean(0).cpu()
        return f


@torch.no_grad()
def make_readout(rig: Rig):
    """Return a drop-in replacement for ``s1bcommon.readout_batch`` bound to this profile."""

    @torch.no_grad()
    def readout_batch(model, tok, convs: Sequence[list], per_token: bool = False, max_rows: int = 6) -> List[dict]:
        if per_token:
            raise NotImplementedError("S4 stores no per-token projections; S1b Task 10 owns that path")
        results: List[Optional[dict]] = [None] * len(convs)
        lens = [len(tok(S.render(tok, c, False), add_special_tokens=False)["input_ids"]) for c in convs]
        groups = chunk_by_budget(lens, READOUT_TOKEN_BUDGET, max_rows=max_rows)
        BATCH_STATS["readout_calls"] += 1
        BATCH_STATS["readout_chunks"] += len(groups)
        BATCH_STATS["readout_max_ctx"] = max(BATCH_STATS["readout_max_ctx"], max(lens) if lens else 0)
        for chunk in groups:
            BATCH_STATS["readout_min_rows"] = (len(chunk) if BATCH_STATS["readout_min_rows"] is None
                                               else min(BATCH_STATS["readout_min_rows"], len(chunk)))
            ids_list, spans_list, meta_list = [], [], []
            for c in chunk:
                ids, sp = S._spans_for(tok, convs[c])
                enc = tok(S.render(tok, convs[c], False), add_special_tokens=False, return_offsets_mapping=True)
                offs = enc["offset_mapping"]
                us = _user_span_chars(tok, convs[c])
                uidx = [k for k in range(len(offs)) if us and us[0] <= offs[k][0] < us[1]] if us else []
                per_pos = [
                    ("point", sp["into"]),
                    ("mean", torch.tensor(sp["think"], dtype=torch.long)) if sp["think"] else None,
                    ("mean", torch.tensor(sp["answer"], dtype=torch.long)) if sp["answer"] else None,
                    ("mean", torch.tensor(uidx, dtype=torch.long)) if uidx else None,
                ]
                ids_list.append(ids); spans_list.append(per_pos)
                meta_list.append({"into": sp["into"], "n_think": len(sp["think"]), "n_answer": len(sp["answer"]),
                                  "n_asst": len(sp["asst"]), "n_user": len(uidx)})
            T = max(len(x) for x in ids_list)
            pad = tok.pad_token_id
            input_ids = torch.full((len(chunk), T), pad, dtype=torch.long)
            attn = torch.zeros((len(chunk), T), dtype=torch.long)
            for r, ids in enumerate(ids_list):  # right padding, as S1b
                input_ids[r, :len(ids)] = torch.tensor(ids); attn[r, :len(ids)] = 1
            cap = _Cap(len(chunk), spans_list, rig, model.device)
            handles = [model.model.layers[L].register_forward_hook(cap.hook(i, L))
                       for i, L in enumerate(rig.resid_layers)]
            try:
                # `model.model` (the transformer) rather than `model` (transformer + lm_head): the readout
                # reads captured layer outputs and never touches the logits, and the logits are the largest
                # tensor in the pass -- [rows, T, 128256] is 5.98 GiB at 8 x 2,915 tokens in bf16, which is
                # what OOMed this 24 GiB card on cell A. The layer outputs, and so every projection and every
                # stored residual, are bit-identical: the same modules run in the same order, and the steering
                # hook sits on `model.model.layers[L]`, inside this call.
                model.model(input_ids=input_ids.to(model.device), attention_mask=attn.to(model.device),
                            use_cache=False)
            finally:
                for h in handles:
                    h.remove()
            for r, c in enumerate(chunk):
                resid = cap.out[r]
                results[c] = {"resid": resid.to(torch.float16), "proj": rig.project(resid),
                              "n_tokens": len(ids_list[r]), "spans": meta_list[r]}
        return results  # type: ignore

    return readout_batch


# --------------------------------------------------------------------------- storage
def make_save_run(rig: Rig):
    def save_run(path: Path, meta: dict, turns: List[dict]):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        P, RL, D, A, AL = len(POSITIONS), len(rig.resid_layers), rig.d_model, len(rig.axis_names), len(rig.arrow_layers)
        resid = torch.stack([t["resid"] for t in turns]) if turns else torch.zeros(0, P, RL, D, dtype=torch.float16)
        proj = torch.stack([t["proj"] for t in turns]) if turns else torch.zeros(0, P, A, AL)
        torch.save({"resid": resid, "proj": proj, "axes": rig.axis_names, "positions": POSITIONS,
                    "resid_layers": rig.resid_layers, "arrow_layers": rig.arrow_layers,
                    "profile": rig.profile, "model": rig.model_id, "revision": rig.revision},
                   str(path) + ".pt")
        meta = dict(meta)
        keep = rig.named_axes + ["random0"]
        meta["proj_summary"] = [
            {p: {a: [round(float(x), 4) for x in t["proj"][pi, rig.axis_names.index(a)]] for a in keep}
             for pi, p in enumerate(POSITIONS)} for t in turns]
        meta["readout_spans"] = [t["spans"] for t in turns]
        json.dump(meta, open(str(path) + ".json", "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    return save_run


# --------------------------------------------------------------------------- model loading
def load_model(rig: Rig):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    t0 = time.time()
    if rig.profile == "base":
        model, tok, stats = S.C.load_base()
    else:
        tok, src_t = _from_pretrained(AutoTokenizer, rig.model_id, rig.revision)
        model, src_m = _from_pretrained(AutoModelForCausalLM, rig.model_id, rig.revision, dtype=rig.dtype)
        model = model.to(rig.device).eval()
        stats = {"load_s": round(time.time() - t0, 1), "model": rig.model_id, "revision": rig.revision,
                 "precision": str(rig.dtype), "device": rig.device, "weights_from": src_m, "tokenizer_from": src_t}
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
        stats["pad_token"] = "set to eos_token (tokenizer had none)"
    if model.config.num_hidden_layers != len(rig.resid_layers) or model.config.hidden_size != rig.d_model:
        raise ValueError("profile shape mismatch: config says %d layers / d=%d" % (
            model.config.num_hidden_layers, model.config.hidden_size))
    return model, tok, stats


# --------------------------------------------------------------------------- wiring
def configure(profile: str, out_root: Path, budget_usd: float) -> Rig:
    """Build the Rig and point the S1b modules at this profile. Call once, before any S1b import is used."""
    if profile not in PROFILES:
        raise ValueError("unknown profile %r" % profile)
    out_root = Path(out_root)
    rig = Rig(profile, out_root)
    # s1bcommon: shapes, axes, positions, generation cap, and the two functions chains.py calls through S.
    S.N_LAYERS, S.D_MODEL, S.LAYERS = len(rig.resid_layers), rig.d_model, rig.resid_layers
    S.AXES, S.POSITIONS = rig.axis_names, POSITIONS
    S.MAX_NEW = rig.spec["max_new_chain"]
    S.RAW = out_root
    S.readout_batch = make_readout(rig)
    S.gen_batch = make_gen_batch(S.gen_batch) if not getattr(S.gen_batch, "_rig_wrapped", False) else S.gen_batch
    S.gen_batch._rig_wrapped = True
    S.save_run = make_save_run(rig)
    S.C.DEVICE = rig.device
    # judges: the rig's own ledger, its own budget stop, its own raw-call log.
    JU.BUDGET_USD = float(budget_usd)
    JU.RAW = out_root
    JU.LEDGER = out_root / "judge_ledger.json"
    JU.CALLS = out_root / "judge_calls"
    out_root.mkdir(parents=True, exist_ok=True)
    return rig


# --------------------------------------------------------------------------- misc
def escalation_sampled(fork_id: str) -> bool:
    """Fixed 1-in-8 seeded sample (D-021), decided before data and independent of run order (plan A7)."""
    h = hashlib.sha256((ESCALATION_SALT + "|" + fork_id).encode("utf-8")).digest()
    return int.from_bytes(h[:8], "big") % ESCALATION_SAMPLE == 0


def bootstrap_ci(values: Sequence[float], n_boot: int = 2000, seed: int = 0, alpha: float = 0.05):
    """Percentile bootstrap on the mean; seed 0, 2,000 resamples (brief step 7)."""
    xs = [float(v) for v in values if v is not None]
    if not xs:
        return None, None, None
    g = torch.Generator().manual_seed(seed)
    t = torch.tensor(xs, dtype=torch.float64)
    idx = torch.randint(len(xs), (n_boot, len(xs)), generator=g)
    means = t[idx].mean(dim=1)
    lo = float(torch.quantile(means, alpha / 2)); hi = float(torch.quantile(means, 1 - alpha / 2))
    return float(t.mean()), lo, hi


def bootstrap_ci_clustered(values: Sequence[Optional[float]], clusters: Sequence, n_boot: int = 2000,
                           seed: int = 0, alpha: float = 0.05):
    """Percentile bootstrap on a mean, resampling **clusters** with replacement (brief: clustered CI).

    One run contributes 14 forks at a distance; those forks are not independent, so resampling forks alone
    understates the interval. The cluster here is the run (target x seed): a resample draws runs with
    replacement and takes the mean over every value in the drawn runs. `None` values are dropped first, and a
    cluster left with no value cannot be drawn.

    Returns (mean, lo, hi, n_values, n_clusters); (None, None, None, 0, 0) when there is nothing to average.
    """
    groups: Dict[object, List[float]] = {}
    for v, c in zip(values, clusters):
        if v is None:
            continue
        groups.setdefault(c, []).append(float(v))
    keys = sorted(groups, key=repr)
    if not keys:
        return None, None, None, 0, 0
    n_values = sum(len(groups[k]) for k in keys)
    sums = torch.tensor([sum(groups[k]) for k in keys], dtype=torch.float64)
    counts = torch.tensor([len(groups[k]) for k in keys], dtype=torch.float64)
    mean = float(sums.sum() / counts.sum())
    if len(keys) == 1:
        return mean, mean, mean, n_values, 1
    g = torch.Generator().manual_seed(seed)
    idx = torch.randint(len(keys), (n_boot, len(keys)), generator=g)
    boots = sums[idx].sum(dim=1) / counts[idx].sum(dim=1)
    lo = float(torch.quantile(boots, alpha / 2)); hi = float(torch.quantile(boots, 1 - alpha / 2))
    return mean, lo, hi, n_values, len(keys)


def _pearson(x: torch.Tensor, y: torch.Tensor) -> Optional[float]:
    if len(x) < 3:
        return None
    xc, yc = x - x.mean(), y - y.mean()
    den = float(xc.norm() * yc.norm())
    return float(torch.dot(xc, yc) / den) if den > 0 else None


def pearson_ci_clustered(xs: Sequence[float], ys: Sequence[float], clusters: Sequence,
                         n_boot: int = 2000, seed: int = 0, alpha: float = 0.05):
    """Pearson r with a clustered percentile bootstrap CI: resample clusters, keep every member of a drawn one.

    The brief asks for the association between a run's persona displacement and that run's spread rate, with a
    clustered CI; the cluster is the target, because runs of one target share a frozen chain.
    Returns (r, lo, hi, n_points, n_clusters).
    """
    by: Dict[object, List[int]] = {}
    pts = []
    for x, y, c in zip(xs, ys, clusters):
        if x is None or y is None or x != x or y != y:
            continue
        by.setdefault(c, []).append(len(pts))
        pts.append((float(x), float(y)))
    keys = sorted(by, key=repr)
    if len(pts) < 3 or not keys:
        return None, None, None, len(pts), len(keys)
    X = torch.tensor([p[0] for p in pts], dtype=torch.float64)
    Y = torch.tensor([p[1] for p in pts], dtype=torch.float64)
    r = _pearson(X, Y)
    if len(keys) < 2:
        return r, None, None, len(pts), len(keys)
    g = torch.Generator().manual_seed(seed)
    draws = torch.randint(len(keys), (n_boot, len(keys)), generator=g)
    boots = []
    for b in range(n_boot):
        sel: List[int] = []
        for k in draws[b].tolist():
            sel += by[keys[k]]
        rb = _pearson(X[sel], Y[sel])
        if rb is not None:
            boots.append(rb)
    if not boots:
        return r, None, None, len(pts), len(keys)
    t = torch.tensor(boots, dtype=torch.float64)
    return r, float(torch.quantile(t, alpha / 2)), float(torch.quantile(t, 1 - alpha / 2)), len(pts), len(keys)


def log(msg: str):
    S.log(msg)
