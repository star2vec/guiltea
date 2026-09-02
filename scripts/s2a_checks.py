#!/usr/bin/env python3
"""S2a Task 7 mechanical checks over data/contrast-sets/ (brief: briefs/S2a-passages.md).
Usage: python3 scripts/s2a_checks.py [--json out.json]
Checks: schema; 7 framings x N scenarios; per-scenario token-length spread <= 15% within each voice; banned-word scan
(scenarios, first-person, second-person, steer-probe; reflection fixtures exempt by the researcher's 2026-09-02 amendment);
shared opening 3-gram (no stem shared by more than two passages of a class); pairwise Jaccard on content words within class
(flag >= 0.30); domain counts; corrective-content byte-identity across the three second-person framings; sentence counts 2-4;
skeleton exact-duplicate screen; steer-probe and fixture structural checks.
Addendum 2026-09-03 (Task B): ngram_recurrence (3-gram in >5 of 50 passages of a framing, scenario-content 3-grams exempt) -> FAIL;
leadin_tail_recurrence (normalized lead-in reused in >2 scenarios) -> FAIL; neutral_correction_no_fault (fault-implying constructions absent) -> PASS.
Tokenizer: meta-llama/Llama-3.2-1B tokenizer files from the local HF cache (offline; no weights), else cl100k_base, else words."""
import json, os, re, sys, itertools, collections
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CS = os.path.join(REPO, "data", "contrast-sets")
os.environ.setdefault("HF_HUB_OFFLINE", "1"); os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

def load_tokenizer():
    try:
        from transformers import AutoTokenizer
        t = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B", local_files_only=True)
        return "meta-llama/Llama-3.2-1B (local cache, tokenizer files only)", lambda s: len(t.encode(s, add_special_tokens=False))
    except Exception:
        pass
    try:
        import tiktoken; e = tiktoken.get_encoding("cl100k_base")
        return "tiktoken cl100k_base (fallback)", lambda s: len(e.encode(s))
    except Exception:
        return "whitespace words (fallback)", lambda s: len(s.split())

TOK_NAME, ntok = load_tokenizer()
BAN = re.compile(r"\b(guilt|guilts|guilty|guiltily|guiltiness|shame|shames|shamed|shaming|shameful|shamefully|ashamed|"
                 r"embarrass|embarrasses|embarrassed|embarrassing|embarrassment|remorse|remorseful|remorsefully)\b", re.I)
STOP = set("""a an the and or but so because as of to in on at by for with from into onto over under about after before
between through during without within along across than then that this these those there here it its it's is are was were
be been being am i me my mine we us our you your yours he him his she her hers they them their theirs who whom whose which what
when where why how not no nor never do does did doing done have has had having will would shall should can could may might
must ought if whether while although though even also too very just only own same other another such more most less least
any some all both each few many much every either neither one two three first last next now again still yet already ever
out up down off away back again once twice per against toward towards until till since around get got getting go went gone
say said says tell told make made put let keep kept give gave given take took taken come came going way thing things
someone something anyone anything nobody nothing everyone everything person people user users""".split())
def content_words(s):
    return set(w for w in re.findall(r"[a-z]+(?:'[a-z]+)?", s.lower()) if w not in STOP and len(w) > 2)
def stem3(s):
    w = re.findall(r"[a-z0-9]+(?:['\-][a-z0-9]+)*", s.lower())
    return " ".join(w[:3])
def nsent(s): return len([x for x in re.split(r'(?<=[.!?])\s+', s.strip()) if x])
def rjsonl(name):
    p = os.path.join(CS, name); rows = []
    with open(p, encoding="utf-8") as f:
        for i, l in enumerate(f, 1):
            if l.strip(): rows.append(json.loads(l))
    return rows

R = {"tokenizer": TOK_NAME, "checks": {}}
fails = []
def check(name, ok, detail):
    R["checks"][name] = {"pass": bool(ok), "detail": detail}
    if not ok: fails.append(name)

# ---- load & schema ----
scen = rjsonl("scenarios.jsonl"); fp = rjsonl("first_person.jsonl"); sp = rjsonl("second_person.jsonl")
probe = rjsonl("steer_probe.jsonl"); fx = rjsonl("reflection_fixtures.jsonl")
import yaml
placement = yaml.safe_load(open(os.path.join(CS, "placement.yaml"), encoding="utf-8"))
rubric = open(os.path.join(CS, "reflection_rubric.md"), encoding="utf-8").read()
def has(rows, keys): return all(all(k in r for k in keys) for r in rows)
schema_ok = (has(scen, ["id", "domain", "act", "outcome", "correct_info", "skeleton", "source_target_id"])
             and has(fp, ["scenario_id", "framing", "text", "n_tokens"]) and has(sp, ["scenario_id", "framing", "text", "correct_info_span", "leadin"])
             and has(probe, ["id", "domain", "prior_user", "prior_assistant", "reflection_request", "correct_info"])
             and has(fx, ["judge", "id", "intended_label", "text", "boundary_note"])
             and all(k in placement for k in ["first_person_user_turn", "second_person_assistant_turn", "readout_positions"])
             and all(s["domain"] in {"medical", "financial", "advice", "code"} for s in scen)
             and all(r["framing"] in {"baseline", "guilt", "shame", "neutral_negative"} for r in fp)
             and all(r["framing"] in {"neutral_correction", "act_blame", "self_blame"} for r in sp)
             and all(r["intended_label"] in {"act-focused", "self-focused", "outcome-negative-only", "neutral", "incoherent"} for r in fx))
check("schema", schema_ok, f"scenarios={len(scen)} first_person={len(fp)} second_person={len(sp)} probe={len(probe)} fixtures={len(fx)}; placement keys ok; rubric {len(rubric)} chars")
ids = [s["id"] for s in scen]
check("unique_scenario_ids", len(set(ids)) == len(ids), f"{len(set(ids))} unique of {len(ids)}")

# ---- 7 framings x N ----
fp_by = collections.defaultdict(dict); sp_by = collections.defaultdict(dict)
for r in fp: fp_by[r["scenario_id"]][r["framing"]] = r
for r in sp: sp_by[r["scenario_id"]][r["framing"]] = r
missing = [s["id"] for s in scen if len(fp_by[s["id"]]) != 4 or len(sp_by[s["id"]]) != 3]
extra = set(fp_by) | set(sp_by); extra -= set(ids)
check("seven_framings_per_scenario", not missing and not extra, f"N={len(ids)}; 4 first + 3 second for every scenario; missing={missing}; orphan={sorted(extra)}")

# ---- domain counts ----
dc = collections.Counter(s["domain"] for s in scen)
check("domain_counts", all(12 <= v <= 13 for v in dc.values()) and len(dc) == 4 and sum(dc.values()) == len(scen), dict(dc))

# ---- skeleton duplicates ----
norm = collections.Counter(re.sub(r"\s+", " ", s["skeleton"].lower().strip()) for s in scen)
dups = [k for k, v in norm.items() if v > 1]
check("skeleton_exact_duplicates", not dups, f"{len(norm)} distinct skeletons of {len(scen)}; duplicates={dups}")

# ---- token spread ----
def spread(vals): return (max(vals) - min(vals)) / min(vals)
spread_rows = []; spread_fail = []
for s in scen:
    f4 = {k: ntok(v["text"]) for k, v in fp_by[s["id"]].items()}
    s3 = {k: ntok(v["text"]) for k, v in sp_by[s["id"]].items()}
    l3 = {k: ntok(v["leadin"]) for k, v in sp_by[s["id"]].items()}
    fs, ss, ls = spread(list(f4.values())), spread(list(s3.values())), spread(list(l3.values()))
    spread_rows.append({"scenario_id": s["id"], "first_person": f4, "first_spread": round(fs, 3), "second_person": s3, "second_spread": round(ss, 3), "leadin": l3, "leadin_spread": round(ls, 3)})
    if fs > 0.15: spread_fail.append((s["id"], "first_person", round(fs, 3)))
    if ss > 0.15: spread_fail.append((s["id"], "second_person", round(ss, 3)))
worst_f = max(r["first_spread"] for r in spread_rows); worst_s = max(r["second_spread"] for r in spread_rows); worst_l = max(r["leadin_spread"] for r in spread_rows)
check("token_spread_le_15pct", not spread_fail, f"tokenizer={TOK_NAME}; spread=(max-min)/min per scenario per voice; worst first-person={worst_f}, worst second-person={worst_s} (lead-ins alone worst={worst_l}, informational); failures={spread_fail}")
R["spread_table"] = spread_rows

# ---- sentence counts ----
sc_bad = [(r["scenario_id"], r["framing"], nsent(r["text"])) for r in fp + sp if not (2 <= nsent(r["text"]) <= 4)]
check("sentence_count_2_to_4", not sc_bad, f"first+second person; out of range={sc_bad}")

# ---- banned words ----
hits = []
for s in scen:
    for k in ["act", "outcome", "correct_info", "skeleton"]:
        for m in BAN.finditer(s[k]): hits.append(("scenarios", s["id"], k, m.group(0)))
for r in fp:
    for m in BAN.finditer(r["text"]): hits.append(("first_person", r["scenario_id"], r["framing"], m.group(0)))
for r in sp:
    for m in BAN.finditer(r["text"]): hits.append(("second_person", r["scenario_id"], r["framing"], m.group(0)))
for p in probe:
    for k in ["prior_user", "prior_assistant", "reflection_request", "correct_info"]:
        for m in BAN.finditer(p[k]): hits.append(("steer_probe", p["id"], k, m.group(0)))
feel = [(r["scenario_id"], r["framing"]) for r in fp if re.search(r"\bI feel\b", r["text"], re.I)]
check("banned_words_zero_hits", not hits and not feel, f"hits={hits}; 'I feel' in first-person={feel}; reflection fixtures exempt (researcher amendment 2026-09-02)")
lw = [f for f in fx if BAN.search(f["text"])]
R["fixtures_with_label_words"] = [{"id": f["id"], "intended_label": f["intended_label"], "text": f["text"]} for f in lw]

# ---- shared stems (per class) ----
stem_flags = {}
classes = collections.defaultdict(list)
for r in fp: classes[("first_person", r["framing"])].append(r)
for r in sp: classes[("second_person", r["framing"])].append(r)
for cls, rows in classes.items():
    c = collections.defaultdict(list)
    for r in rows: c[stem3(r["text"])].append(r["scenario_id"])
    over = {k: v for k, v in c.items() if len(v) > 2}
    if over: stem_flags["/".join(cls)] = over
check("shared_opening_3gram_max_two", not stem_flags, f"classes={len(classes)}; stems shared by >2 passages: {stem_flags}")

# ---- Jaccard within class ----
jac_flags = []
for cls, rows in classes.items():
    cw = {r["scenario_id"]: content_words(r["text"]) for r in rows}
    for a, b in itertools.combinations(sorted(cw), 2):
        u = cw[a] | cw[b]
        if not u: continue
        j = len(cw[a] & cw[b]) / len(u)
        if j >= 0.30: jac_flags.append({"class": "/".join(cls), "a": a, "b": b, "jaccard": round(j, 3)})
check("jaccard_flags_listed", True, f"pairs flagged (>=0.30): {len(jac_flags)}; flags go to the hand-check list, not a fail")
R["jaccard_flags"] = jac_flags

# ---- corrective content identity ----
ci_bad = []
scen_by = {s["id"]: s for s in scen}
for sid, d in sp_by.items():
    ci = scen_by[sid]["correct_info"]
    spans = []
    for fr, r in d.items():
        a, b = r["correct_info_span"]; sub = r["text"][a:b]
        spans.append((fr, sub == ci, r["text"].endswith(ci), a))
    if not all(ok and ends for _, ok, ends, _ in spans): ci_bad.append((sid, spans))
    # shared opening = the lead-in's first sentence (the statement that the answer was wrong); identical across the three framings.
    # (v1 files used a single-sentence lead-in split at ", and "; v2 lead-ins are statement + clause, so the first sentence is the opening.)
    def _opening(t):
        m = re.match(r"^(.*?[.!?])(\s|$)", t); return m.group(1) if m else t.split(", and ")[0]
    op = {fr: _opening(r["text"]) for fr, r in d.items()}
    if len(set(op.values())) != 1: ci_bad.append((sid, "openings differ", op))
check("corrective_content_byte_identical", not ci_bad, f"correct_info byte-identical in all three second-person messages, final position, equal to scenarios.jsonl; opening (first sentence of the lead-in) identical across the three; bad={ci_bad}")

# ---- steer probe ----
tg = {}; fxa = []
tp = os.path.join(REPO, "data", "acts", "targets.jsonl"); rp = os.path.join(REPO, "data", "acts", "rubric_fixtures.jsonl")
if os.path.exists(tp) and os.path.exists(rp):
    for l in open(tp, encoding="utf-8"):
        if l.strip(): t = json.loads(l); tg[t["id"]] = t
    fxa = [json.loads(l) for l in open(rp, encoding="utf-8") if l.strip()]
med = [p for p in probe if p["domain"] == "medical"]; subd = [p.get("subdomain") for p in med]
verb = []
for p in med:
    t = tg.get(p["source_target_id"]); f = [x for x in fxa if x.get("judge") == "act" and x.get("intended_label") == "committed" and x.get("target_id") == p["source_target_id"]]
    verb.append(bool(t) and p["prior_user"] == t["situation"] and p["correct_info"] == t["correct_info_Y"] and len(f) == 1 and p["prior_assistant"] == f[0]["text"])
rr = set(p["reflection_request"] for p in probe)
probe_ok = len(probe) == 8 and len(med) == 4 and len(set(subd)) == 4 and all(verb) and len(rr) == 1 and collections.Counter(p["domain"] for p in probe)["financial"] >= 1 and collections.Counter(p["domain"] for p in probe)["advice"] >= 1 and collections.Counter(p["domain"] for p in probe)["code"] >= 1
check("steer_probe_structure", probe_ok, f"items={len(probe)}; medical={len(med)} in subdomains {subd}; verbatim reuse checks={verb}; one reflection_request={len(rr)==1}; domains={dict(collections.Counter(p['domain'] for p in probe))}")

# ---- fixtures ----
fc = collections.Counter(f["intended_label"] for f in fx)
check("reflection_fixtures_counts", len(fc) == 5 and all(v >= 12 for v in fc.values()), f"{dict(fc)}; with label words: {len(lw)} ({[f['id'] for f in lw]})")
kw = re.search(r"keyword", rubric, re.I) is not None and re.search(r"never", rubric, re.I) is not None
check("rubric_present", len(rubric) > 1000 and "label:" in rubric and "reason:" in rubric and kw, f"rubric has output format lines and the no-keyword rule: {kw}")

# ---- placement verbatim ----
pl_ok = (placement["first_person_user_turn"] == "Before we go on — how do you see what just happened?" and
         placement["second_person_assistant_turn"] == "Here is what I told you: [act]" and
         placement["readout_positions"]["first_person"]["primary"] == "mean over passage tokens" and
         placement["readout_positions"]["first_person"]["secondary"] == "last passage token" and
         placement["readout_positions"]["second_person"]["feedback_mean"] == "mean over passage tokens" and
         placement["readout_positions"]["second_person"]["post"] == "last token before the next assistant turn")
check("placement_verbatim", pl_ok, "strings equal to the brief's Task 4 text")


# ---- Task B (addendum 2026-09-03): ngram_recurrence, leadin_tail_recurrence, neutral_correction_no_fault ----
def alpha_tokens(s): return re.findall(r"[a-z]+", s.lower())
def grams3(s):
    w = alpha_tokens(s); return set(zip(w, w[1:], w[2:]))
exempt = {s["id"]: grams3(s["act"]) | grams3(s["outcome"]) | grams3(s["correct_info"]) for s in scen}
ng_fail = {}; ng_top = {}
for cls, rows in classes.items():
    c = collections.defaultdict(list)
    for r in rows:
        for g in grams3(r["text"]) - exempt[r["scenario_id"]]: c[g].append(r["scenario_id"])
    top = sorted(c.items(), key=lambda kv: (-len(kv[1]), kv[0]))[:10]
    ng_top["/".join(cls)] = [{"ngram": " ".join(g), "n": len(v)} for g, v in top]
    bad = {" ".join(g): sorted(v) for g, v in c.items() if len(v) > 5}
    if bad: ng_fail["/".join(cls)] = bad
check("ngram_recurrence", not ng_fail, f"3-grams over lowercased alphabetic tokens present in >5 of {len(scen)} passages of a framing, exempting 3-grams found in the scenario's own act/outcome/correct_info; failing 3-grams per framing: { {k: len(v) for k, v in ng_fail.items()} }")
R["ngram_top10"] = ng_top; R["ngram_failures"] = ng_fail

normws = lambda s: re.sub(r"\s+", " ", s.strip().lower())
lt_fail = {}; tails_info = {}
for fr in ["neutral_correction", "act_blame", "self_blame"]:
    leads = collections.defaultdict(list); tails = collections.Counter()
    for sid, d in sp_by.items():
        r = d[fr]; a, b = r["correct_info_span"]; lead = r["text"][:a].strip()
        leads[normws(lead)].append(sid)
        pref = os.path.commonprefix([x["text"][:x["correct_info_span"][0]] for x in d.values()])
        tails[normws(lead[len(pref):])] += 1
    over = {k: v for k, v in leads.items() if len(v) > 2}
    if over: lt_fail[fr] = over
    tails_info[fr] = {"distinct_leadins": len(leads), "max_leadin_reuse": max(len(v) for v in leads.values()), "distinct_tails": len(tails), "max_tail_reuse": max(tails.values())}
check("leadin_tail_recurrence", not lt_fail, f"normalized lead-in (text minus correct_info) reused in >2 scenarios -> FAIL; failures={lt_fail}; informational clause tails (lead-in minus the scenario's three-way common prefix): {tails_info}")
R["leadin_tails"] = tails_info

FAULT = re.compile(r"\b(should have|should've|shouldn't|ought to have|failed to|you failed|didn't|did not|you never|never|neglected|careless(?:ly|ness)?|reckless(?:ly)?|negligen(?:t|ce)|sloppy|irresponsible|at fault|your fault|to blame|you got (?:it|this|that) wrong|wrong of you|you missed|you ignored|you overlooked|you forgot|you left out|you omitted)\b", re.I)
nf = []
for sid, d in sp_by.items():
    r = d["neutral_correction"]; lead = r["text"][:r["correct_info_span"][0]]
    s = scen_by[sid]; ctx = (s["act"] + " " + s["outcome"] + " " + s["correct_info"]).lower()
    for m in FAULT.finditer(lead):
        if m.group(0).lower() not in ctx: nf.append((sid, m.group(0)))
check("neutral_correction_no_fault", not nf, f"fault-implying constructions in neutral_correction lead-ins (matches that occur in the scenario's own act/outcome/correct_info are exempt): {nf}")
R["neutral_fault_hits"] = nf

# ---- output ----
R["failures"] = fails
print(f"# S2a Task 7 mechanical checks\n\nTokenizer: {TOK_NAME}\n")
for k, v in R["checks"].items():
    print(f"- **{k}**: {'PASS' if v['pass'] else 'FAIL'} — {v['detail']}")
print(f"\nJaccard flags (>= 0.30): {len(jac_flags)}")
for j in jac_flags: print(f"  - {j['class']}: {j['a']} vs {j['b']} = {j['jaccard']}")
print("\nTask B top-10 recurring 3-grams per framing (after exemptions):")
for k, v in R["ngram_top10"].items(): print(f"  {k}: " + ", ".join(f"'{x['ngram']}'={x['n']}" for x in v))
print(f"Task B lead-in tails: {R['leadin_tails']}")
print(f"\nFixtures containing label words: {len(lw)}")
for f in lw: print(f"  - {f['id']} [{f['intended_label']}]: {f['text']}")
if "--json" in sys.argv:
    json.dump(R, open(sys.argv[sys.argv.index("--json") + 1], "w"), indent=1)
sys.exit(1 if fails else 0)
