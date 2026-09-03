"""S1b judges — wrappers over scripts/judge_rubrics.py with a thread pool, a persistent ledger
(list-price and cache-discounted cost), raw-response logging, and the budget stop (researcher's note 2:
$15 on the list-price ledger). The API key is read from the environment by judge_rubrics; never printed."""
from __future__ import annotations

import json
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable, List, Optional

import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
import judge_rubrics as J  # noqa: E402

RAW = ROOT / "results" / "raw" / "s1b"
LEDGER = RAW / "judge_ledger.json"
CALLS = RAW / "judge_calls"
BUDGET_USD = 15.0  # list-price ledger; researcher's note 2 at plan approval (was $12 in the brief)
MINI, BIG = "gpt-4o-mini", "gpt-4o-2024-08-06"
CACHE_DISCOUNT = 0.5  # OpenAI cached-input discount (informational figure; the stop uses list price)


class BudgetStop(RuntimeError):
    pass


class Judges:
    def __init__(self, workers: int = 8):
        J.ensure_api_key()
        import openai
        self.client = openai.OpenAI()
        self.sections = J.load_rubric_sections()
        self.wordings = J.load_wordings()
        self.lock = threading.Lock()
        self.pool = ThreadPoolExecutor(max_workers=workers)
        CALLS.mkdir(parents=True, exist_ok=True)
        self.ledger = json.load(open(LEDGER)) if LEDGER.exists() else {
            "budget_usd": BUDGET_USD, "list_price_usd": 0.0, "cache_discounted_usd": 0.0, "calls": 0,
            "by_model": {}, "by_purpose": {}, "stopped": None, "started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        self.ledger["budget_usd"] = BUDGET_USD

    # ------------------------------------------------------------------ core call
    def call(self, messages, model: str, purpose: str, label_set: Optional[List[str]], max_tokens: int = 150,
             temperature: float = 0.0, extra: Optional[dict] = None) -> dict:
        """One judged call; parse the two-line output when label_set is given (one re-call if unparseable)."""
        attempts = []
        label = reason = None
        for _ in range(2 if label_set else 1):
            with self.lock:
                if self.ledger["list_price_usd"] >= BUDGET_USD:
                    self.ledger["stopped"] = "budget stop: list-price ledger $%.2f >= $%.2f (purpose %s)" % (
                        self.ledger["list_price_usd"], BUDGET_USD, purpose)
                    self._write()
                    raise BudgetStop(self.ledger["stopped"])
            r = J.call_judge(messages, model, temperature=temperature, max_tokens=max_tokens, client=self.client)
            r["cost_usd"] = J.cost_usd(r["usage"], model)
            cached = r["usage"].get("cached_tokens", 0) or 0
            r["cost_cache_discounted_usd"] = r["cost_usd"] - cached * J.PRICES[model][0] / 1e6 * CACHE_DISCOUNT
            attempts.append(r)
            self._account(model, purpose, r)
            if label_set:
                lab, rsn, raw = J.parse_label(r["content"], set(label_set))
                r["parsed_label"], r["parsed_reason"], r["label_raw"] = lab, rsn, raw
                if lab is not None:
                    label, reason = lab, rsn
                    break
            else:
                break
        rec = {"id": uuid.uuid4().hex, "purpose": purpose, "model": model, "label": label if label_set else None,
               "reason": reason, "unparseable": bool(label_set) and label is None, "content": attempts[-1]["content"],
               "attempts": attempts, "request": {"messages": messages, "temperature": temperature, "max_tokens": max_tokens},
               "cost_usd": sum(a["cost_usd"] for a in attempts), "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        if extra:
            rec.update(extra)
        with self.lock:
            with open(CALLS / ("%s.jsonl" % purpose), "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        return rec

    def _account(self, model, purpose, r):
        with self.lock:
            L = self.ledger
            L["calls"] += 1
            L["list_price_usd"] += r["cost_usd"]
            L["cache_discounted_usd"] += r["cost_cache_discounted_usd"]
            m = L["by_model"].setdefault(model, {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "cached_tokens": 0, "usd": 0.0})
            m["calls"] += 1; m["prompt_tokens"] += r["usage"]["prompt_tokens"]; m["completion_tokens"] += r["usage"]["completion_tokens"]
            m["cached_tokens"] += r["usage"].get("cached_tokens", 0) or 0; m["usd"] += r["cost_usd"]
            p = L["by_purpose"].setdefault(purpose, {"calls": 0, "usd": 0.0})
            p["calls"] += 1; p["usd"] += r["cost_usd"]
            if L["calls"] % 25 == 0:
                self._write()

    def _write(self):
        self.ledger["updated"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        LEDGER.parent.mkdir(parents=True, exist_ok=True)
        json.dump(self.ledger, open(LEDGER, "w"), indent=1)

    def flush(self):
        with self.lock:
            self._write()

    def spent(self) -> float:
        return self.ledger["list_price_usd"]

    # ------------------------------------------------------------------ parallel helper
    def map(self, fn: Callable, items: list) -> list:
        futs = [self.pool.submit(fn, it) for it in items]
        out = []
        for f in futs:
            out.append(f.result())
        self.flush()
        return out

    # ------------------------------------------------------------------ the judges
    def act(self, target: dict, answer: str, model: str = MINI, purpose: str = "act_primary", extra=None) -> dict:
        msgs = J.build_act_prompt(target, answer, self.sections)
        return self.call(msgs, model, purpose, J.LABELS["act"], extra=extra)

    def probe(self, system_prompt_in_force: str, prior_answer: str, fork: str, reply: str, model: str = BIG,
              purpose: str = "probe_primary", extra=None) -> dict:
        fw = self.wordings["probe_fork_%s" % fork].strip()
        msgs = J.build_probe_prompt(system_prompt_in_force, prior_answer, fork, fw, reply, self.sections)
        return self.call(msgs, model, purpose, J.LABELS["probe"], extra=extra)

    def flag(self, target: dict, thinking: str, model: str = MINI, purpose: str = "flag_primary", extra=None) -> dict:
        msgs = J.build_flag_prompt(target["harmful_advice_X"], target["correct_info_Y"], thinking, self.sections)
        return self.call(msgs, model, purpose, J.LABELS["harm_flag"], extra=extra)

    def knowledge(self, target: dict, answer: str, model: str = BIG, purpose: str = "knowledge", extra=None) -> dict:
        msgs = J.build_knowledge_prompt(target["harmful_advice_X"], target["correct_info_Y"], answer, self.sections)
        return self.call(msgs, model, purpose, J.LABELS["knowledge"], extra=extra)

    def raw(self, messages, model: str, purpose: str, max_tokens: int = 600, temperature: float = 0.0,
            seed: Optional[int] = None, extra=None) -> dict:
        """Un-parsed call (persuader, pre-screen). Uses the same accounting."""
        # judge_rubrics.call_judge has no seed parameter; call the client directly with the same retry policy.
        import openai
        with self.lock:
            if self.ledger["list_price_usd"] >= BUDGET_USD:
                self.ledger["stopped"] = "budget stop: list-price ledger $%.2f >= $%.2f (purpose %s)" % (
                    self.ledger["list_price_usd"], BUDGET_USD, purpose)
                self._write()
                raise BudgetStop(self.ledger["stopped"])
        transient = (openai.RateLimitError, openai.APIConnectionError, openai.APITimeoutError, openai.InternalServerError)
        attempt = 0
        kw = dict(model=model, messages=messages, temperature=temperature, max_tokens=max_tokens)
        if seed is not None:
            kw["seed"] = seed
        while True:
            try:
                resp = self.client.chat.completions.create(**kw)
                break
            except transient:
                if attempt >= 4:
                    raise
                time.sleep(2 ** attempt); attempt += 1
        ch = resp.choices[0]
        usage = {"prompt_tokens": resp.usage.prompt_tokens, "completion_tokens": resp.usage.completion_tokens,
                 "total_tokens": resp.usage.total_tokens}
        det = getattr(resp.usage, "prompt_tokens_details", None)
        if det is not None and getattr(det, "cached_tokens", None) is not None:
            usage["cached_tokens"] = det.cached_tokens
        r = {"requested_model": model, "model": resp.model, "response_id": resp.id, "finish_reason": ch.finish_reason,
             "content": J._redact(ch.message.content or ""), "usage": usage, "transport_retries": attempt,
             "system_fingerprint": getattr(resp, "system_fingerprint", None)}
        r["cost_usd"] = J.cost_usd(usage, model)
        cached = usage.get("cached_tokens", 0) or 0
        r["cost_cache_discounted_usd"] = r["cost_usd"] - cached * J.PRICES[model][0] / 1e6 * CACHE_DISCOUNT
        self._account(model, purpose, r)
        rec = {"id": uuid.uuid4().hex, "purpose": purpose, "model": model, "content": r["content"], "usage": usage,
               "finish_reason": ch.finish_reason, "seed": seed, "system_fingerprint": r["system_fingerprint"],
               "request": {"messages": messages, "temperature": temperature, "max_tokens": max_tokens},
               "cost_usd": r["cost_usd"], "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        if extra:
            rec.update(extra)
        with self.lock:
            with open(CALLS / ("%s.jsonl" % purpose), "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        return rec


def system_prompt_in_force(wordings: dict, mode: str, persona_override: Optional[str] = None) -> str:
    """Think-aloud for every mode; vicious adds the persona prompt (judge_rubrics convention; D-10 surname variant)."""
    s = wordings["think_aloud_instruction"].strip()
    if mode == "vicious":
        p = persona_override if persona_override is not None else wordings["vicious_persona_prompt"].strip()
        s = s + "\n\n" + p
    return s
