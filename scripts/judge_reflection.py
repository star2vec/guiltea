#!/usr/bin/env python3
"""Reflection judge for the guilt/shame project (S2a Task 6 dry-run gate; reused by S2b).

Semantic judge as specified in data/contrast-sets/reflection_rubric.md: the rubric is the system message
verbatim, the reflection text is the user message, and the judge answers with two lines, `label:` and `reason:`.

Import (from repo root):  from scripts.judge_reflection import build_reflection_prompt, call_judge, run_fixtures, summarize
CLI:
  python3 scripts/judge_reflection.py --project-only --model gpt-4o-mini
  python3 scripts/judge_reflection.py --model gpt-4o-mini --fixtures data/contrast-sets/reflection_fixtures.jsonl --out results/raw/s2a_task6 --budget 1.00
  python3 scripts/judge_reflection.py --summarize gpt-4o-mini gpt-4o-2024-08-06 --out results/raw/s2a_task6

Key handling: OPENAI_API_KEY is read from the environment; if absent, the gitignored repo-root .env is loaded into the
process environment before the client is created. The value is never printed, logged, or written; error strings are
redacted of key-shaped tokens before storage.
"""
import os, sys, json, time, re, argparse, datetime
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUBRIC_PATH = os.path.join(REPO, "data", "contrast-sets", "reflection_rubric.md")
DEFAULT_FIXTURES = os.path.join(REPO, "data", "contrast-sets", "reflection_fixtures.jsonl")
DEFAULT_OUT = os.path.join(REPO, "results", "raw", "s2a_task6")
LABELS = ["act-focused", "self-focused", "outcome-negative-only", "neutral", "incoherent"]
# USD per million tokens (input, output); list prices as of 2026-09-03. Cost is computed from returned usage.
PRICES_PER_M = {"gpt-4o-mini": (0.15, 0.60), "gpt-4o-2024-08-06": (2.50, 10.00)}
_KEY_RE = re.compile(r"sk-[A-Za-z0-9_\-]{8,}")
def _redact(s):
    return _KEY_RE.sub("sk-REDACTED", s) if isinstance(s, str) else s

def load_env_key():
    """Return True if OPENAI_API_KEY is available; loads the repo-root .env into os.environ if needed. Never returns the value."""
    if os.environ.get("OPENAI_API_KEY"):
        return True
    p = os.path.join(REPO, ".env")
    if os.path.exists(p):
        with open(p, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                k = k.strip()
                if k.startswith("export "):
                    k = k[len("export "):].strip()
                v = v.strip().strip('"').strip("'")
                if k == "OPENAI_API_KEY" and v:
                    os.environ["OPENAI_API_KEY"] = v
                    return True
    return bool(os.environ.get("OPENAI_API_KEY"))

def load_rubric():
    with open(RUBRIC_PATH, encoding="utf-8") as fh:
        return fh.read()

def build_reflection_prompt(reflection_text, prior_act=None, rubric=None):
    """system = rubric verbatim; user = the reflection text (optionally preceded by the prior act, for S2b)."""
    rubric = load_rubric() if rubric is None else rubric
    user = reflection_text if prior_act is None else f"Prior act: {prior_act}\n\nReflection: {reflection_text}"
    return [{"role": "system", "content": rubric}, {"role": "user", "content": user}]

_LABEL_RE = re.compile(r"^\s*\**\s*label\s*\**\s*:\s*\**\s*([A-Za-z][A-Za-z\- ]*?)\s*\**\s*\.?\s*$", re.I | re.M)
_REASON_RE = re.compile(r"^\s*\**\s*reason\s*\**\s*:\s*\**\s*(.+?)\s*$", re.I | re.M)
def parse_judge_output(text):
    """Return (label, reason). label is one of LABELS or None (unparseable); reason is the reason line or None."""
    text = text or ""
    label = None
    m = _LABEL_RE.search(text)
    if m:
        cand = m.group(1).strip().lower().replace(" ", "-")
        if cand in LABELS:
            label = cand
    r = _REASON_RE.search(text)
    reason = r.group(1).strip() if r else None
    return label, reason

def get_client():
    if not load_env_key():
        raise RuntimeError("OPENAI_API_KEY not available (environment or repo-root .env)")
    from openai import OpenAI
    return OpenAI()

_TRANSIENT = {"RateLimitError", "APIConnectionError", "APITimeoutError", "InternalServerError"}
def call_judge(messages, model, temperature=0.0, max_tokens=200, client=None, max_attempts=3):
    """One judge call. Returns a dict with label, reason, raw_text, usage, response (full dump), latency, attempts."""
    client = client or get_client()
    retry_errors = []
    last = None
    for i in range(max_attempts):
        t0 = time.time()
        try:
            resp = client.chat.completions.create(model=model, messages=messages, temperature=temperature, max_tokens=max_tokens)
            text = resp.choices[0].message.content or ""
            label, reason = parse_judge_output(text)
            u = resp.usage
            return {"requested_model": model, "response_model": resp.model, "label": label, "reason": reason, "raw_text": _redact(text),
                    "usage": {"prompt_tokens": u.prompt_tokens, "completion_tokens": u.completion_tokens, "total_tokens": u.total_tokens},
                    "finish_reason": resp.choices[0].finish_reason, "latency_s": round(time.time() - t0, 3), "attempts": i + 1,
                    "retry_errors": retry_errors, "response": json.loads(_redact(json.dumps(resp.model_dump()))),
                    "timestamp": datetime.datetime.utcnow().isoformat() + "Z"}
        except Exception as e:  # noqa
            name = type(e).__name__
            status = getattr(e, "status_code", None)
            retry_errors.append({"attempt": i + 1, "error": name, "status": status, "message": _redact(str(e))[:300]})
            last = e
            transient = name in _TRANSIENT or (status is not None and (status == 429 or status >= 500))
            if not transient or i == max_attempts - 1:
                break
            time.sleep(2 ** i)
    return {"requested_model": model, "response_model": None, "label": None, "reason": None, "raw_text": None, "usage": None,
            "finish_reason": None, "latency_s": None, "attempts": len(retry_errors), "retry_errors": retry_errors, "response": None,
            "error": _redact(f"{type(last).__name__}: {last}")[:300], "timestamp": datetime.datetime.utcnow().isoformat() + "Z"}

def load_fixtures(path=DEFAULT_FIXTURES):
    with open(path, encoding="utf-8") as fh:
        return [json.loads(l) for l in fh if l.strip()]

def project_cost(model, fixtures, rubric=None, max_tokens=200):
    """Ceiling projection: o200k_base token counts x list prices, output assumed at max_tokens per call."""
    import tiktoken
    try:
        enc = tiktoken.get_encoding("o200k_base")
    except Exception:  # noqa
        enc = tiktoken.get_encoding("cl100k_base")
    rubric = load_rubric() if rubric is None else rubric
    pin, pout = PRICES_PER_M[model]
    rub = len(enc.encode(rubric))
    inp = sum(rub + len(enc.encode(f["text"])) + 12 for f in fixtures)
    out = max_tokens * len(fixtures)
    return {"model": model, "calls": len(fixtures), "input_tokens": inp, "output_tokens_ceiling": out,
            "usd_ceiling": round(inp / 1e6 * pin + out / 1e6 * pout, 4), "prices_per_million": {"input": pin, "output": pout}}

def cost_usd(model, usage):
    pin, pout = PRICES_PER_M[model]
    return usage["prompt_tokens"] / 1e6 * pin + usage["completion_tokens"] / 1e6 * pout

def run_fixtures(model, fixtures_path=DEFAULT_FIXTURES, out_dir=DEFAULT_OUT, budget_usd=1.0, temperature=0.0, max_tokens=200, spent_so_far=0.0):
    """Score every fixture once with `model`. Writes <out_dir>/<model>/<fixture_id>.json and <out_dir>/<model>_scores.jsonl.
    Stops before any call that would take spent_so_far + spent + projected remainder over budget_usd."""
    fixtures = load_fixtures(fixtures_path)
    rubric = load_rubric()
    proj = project_cost(model, fixtures, rubric, max_tokens)
    if spent_so_far + proj["usd_ceiling"] > budget_usd:
        return {"model": model, "stopped": "projection exceeds budget", "projection": proj, "spent_so_far": spent_so_far, "calls": 0}
    client = get_client()
    mdir = os.path.join(out_dir, model)
    os.makedirs(mdir, exist_ok=True)
    scores_path = os.path.join(out_dir, f"{model}_scores.jsonl")
    pin, pout = PRICES_PER_M[model]
    per_call_ceiling = proj["input_tokens"] / len(fixtures) / 1e6 * pin + max_tokens / 1e6 * pout
    spent = 0.0
    rows = []
    stopped = None
    with open(scores_path, "w", encoding="utf-8") as sf:
        for i, f in enumerate(fixtures):
            messages = build_reflection_prompt(f["text"], rubric=rubric)
            res = call_judge(messages, model, temperature, max_tokens, client)
            rec = {"fixture_id": f["id"], "intended_label": f["intended_label"], "contains_label_word": bool(f.get("contains_label_word", False)),
                   "text": f["text"], "request": {"model": model, "temperature": temperature, "max_tokens": max_tokens, "messages": messages}}
            rec.update(res)
            rec["cost_usd"] = round(cost_usd(model, res["usage"]), 6) if res.get("usage") else None
            if res.get("usage"):
                spent += cost_usd(model, res["usage"])
            with open(os.path.join(mdir, f["id"] + ".json"), "w", encoding="utf-8") as fh:
                json.dump(rec, fh, indent=1, ensure_ascii=False)
            slim = {k: rec.get(k) for k in ["fixture_id", "intended_label", "contains_label_word", "label", "reason", "raw_text", "usage",
                                            "cost_usd", "attempts", "finish_reason", "response_model", "error"]}
            sf.write(json.dumps(slim, ensure_ascii=False) + "\n")
            sf.flush()
            rows.append(rec)
            if res.get("error") and i == 0 and any(e.get("status") in (400, 401, 403, 404) for e in res.get("retry_errors", [])):
                stopped = "first call rejected: " + res["error"]
                break
            remaining = len(fixtures) - i - 1
            if spent_so_far + spent + remaining * per_call_ceiling > budget_usd:
                stopped = f"budget: spent {spent_so_far + spent:.4f} + projected remainder {remaining * per_call_ceiling:.4f} > {budget_usd}"
                break
    return {"model": model, "calls": len(rows), "spent_usd": round(spent, 4), "stopped": stopped,
            "errors": sum(1 for r in rows if r.get("error")), "unparseable": sum(1 for r in rows if not r.get("error") and r["label"] is None),
            "retries": sum(max(0, (r.get("attempts") or 1) - 1) for r in rows), "scores_path": scores_path, "projection": proj}

def summarize(models, fixtures_path=DEFAULT_FIXTURES, out_dir=DEFAULT_OUT):
    fixtures = load_fixtures(fixtures_path)
    S = {"models": list(models), "n_fixtures": len(fixtures), "per_model": {}}
    scores = {}
    for m in models:
        with open(os.path.join(out_dir, f"{m}_scores.jsonl"), encoding="utf-8") as fh:
            rows = [json.loads(l) for l in fh if l.strip()]
        by = {r["fixture_id"]: r for r in rows}
        scores[m] = by
        cols = LABELS + ["unparseable", "error"]
        cm = {a: {b: 0 for b in cols} for a in LABELS}
        correct = 0; pt = ct = 0; cost = 0.0; mis = []; lw = []; unp = err = retries = 0
        for f in fixtures:
            r = by.get(f["id"])
            if r is None:
                pred = "error"
            else:
                pred = "error" if r.get("error") else (r["label"] or "unparseable")
            cm[f["intended_label"]][pred] += 1
            correct += int(pred == f["intended_label"])
            if r and r.get("usage"):
                pt += r["usage"]["prompt_tokens"]; ct += r["usage"]["completion_tokens"]; cost += r.get("cost_usd") or 0.0
            if r:
                unp += int(not r.get("error") and r["label"] is None); err += int(bool(r.get("error"))); retries += max(0, (r.get("attempts") or 1) - 1)
            if pred != f["intended_label"]:
                mis.append({"fixture_id": f["id"], "intended": f["intended_label"], "predicted": pred, "reason": (r or {}).get("reason"), "raw_text": (r or {}).get("raw_text"), "text": f["text"]})
            if f.get("contains_label_word"):
                lw.append({"fixture_id": f["id"], "intended": f["intended_label"], "predicted": pred, "reason": (r or {}).get("reason"), "text": f["text"]})
        S["per_model"][m] = {"confusion": cm, "accuracy": round(correct / len(fixtures), 4), "correct": correct, "prompt_tokens": pt, "completion_tokens": ct,
                             "cost_usd": round(cost, 4), "unparseable": unp, "errors": err, "retries": retries, "misgraded": mis, "label_word_fixtures": lw}
    if len(models) == 2:
        a, b = models
        agree = 0; dis = []
        for f in fixtures:
            ra, rb = scores[a].get(f["id"], {}), scores[b].get(f["id"], {})
            la, lb = ra.get("label"), rb.get("label")
            if la is not None and la == lb:
                agree += 1
            else:
                dis.append({"fixture_id": f["id"], "intended": f["intended_label"], "text": f["text"], a: {"label": la, "reason": ra.get("reason")}, b: {"label": lb, "reason": rb.get("reason")}})
        S["agreement"] = {"models": [a, b], "agree": agree, "n": len(fixtures), "rate": round(agree / len(fixtures), 4), "disagreements": dis}
    S["total_cost_usd"] = round(sum(v["cost_usd"] for v in S["per_model"].values()), 4)
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as fh:
        json.dump(S, fh, indent=1, ensure_ascii=False)
    return S

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=None)
    ap.add_argument("--fixtures", default=DEFAULT_FIXTURES)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--budget", type=float, default=1.0)
    ap.add_argument("--spent-so-far", type=float, default=0.0)
    ap.add_argument("--max-tokens", type=int, default=200)
    ap.add_argument("--project-only", action="store_true")
    ap.add_argument("--summarize", nargs="+", default=None, metavar="MODEL")
    a = ap.parse_args()
    if a.summarize:
        S = summarize(a.summarize, a.fixtures, a.out)
        print(json.dumps({m: {"accuracy": v["accuracy"], "cost_usd": v["cost_usd"], "unparseable": v["unparseable"], "errors": v["errors"]} for m, v in S["per_model"].items()}, indent=1))
        if "agreement" in S:
            print("agreement:", S["agreement"]["agree"], "/", S["agreement"]["n"])
        print("total_cost_usd:", S["total_cost_usd"])
        return
    if not a.model:
        ap.error("--model is required unless --summarize is given")
    if a.project_only:
        print(json.dumps(project_cost(a.model, load_fixtures(a.fixtures), max_tokens=a.max_tokens), indent=1))
        return
    r = run_fixtures(a.model, a.fixtures, a.out, a.budget, max_tokens=a.max_tokens, spent_so_far=a.spent_so_far)
    print(json.dumps(r, indent=1))
    if r.get("stopped"):
        sys.exit(2)

if __name__ == "__main__":
    main()
