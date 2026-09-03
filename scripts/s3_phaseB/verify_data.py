"""S3 Phase B — verify the contrast-set files against the Phase A record (data/contrast-sets/SOURCE.md).

Checks sha256 of the Arditi refusal splits and the Turner bad/good-medical jsonl (and, if given,
the encrypted upstream archive), row counts, and index-aligned pairing. Exit 1 on any mismatch.
Usage: python scripts/s3_phaseB/verify_data.py [--archive PATH]
"""
import argparse, hashlib, json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EXPECTED = {  # from data/contrast-sets/SOURCE.md (Phase A, 2026-09-02)
    "data/contrast-sets/refusal/harmful_train.json": ("8f5c0eac0efd2a7f99084bbe8d0de2c465e31b1997184783c917969d9de9ece1", 34613),
    "data/contrast-sets/refusal/harmless_train.json": ("86623b1f8a25aa35df153fc97a556dbcebb6a7c881538ae43ee479ca17f2e002", 2344466),
    "data/contrast-sets/badmed/bad_medical_advice.jsonl": ("9d52186ab9886e3abef0eebb1901df9da4ce25a297e584158be0a4bba8d56507", 4300061),
    "data/contrast-sets/badmed/good_medical_advice.jsonl": ("b972f06672093b74f61cc83606929ce0ea3bb9caa2894ea61a557315dba6e6fc", 4946304),
}
ARCHIVE_SHA = "18af368553884eea48a288e47e79553563854f15ca46cf7a16cd0784f935f005"
ARCHIVE_BYTES = 38643720


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--archive", type=Path, default=None)
    args = ap.parse_args()
    ok = True

    if args.archive is not None:
        got, size = sha256(args.archive), args.archive.stat().st_size
        match = got == ARCHIVE_SHA and size == ARCHIVE_BYTES
        ok &= match
        print(f"{'OK  ' if match else 'FAIL'} archive {args.archive.name}: {size} B sha256 {got[:8]}…{got[-8:]} (expected {ARCHIVE_SHA[:8]}…)")

    for rel, (exp, nbytes) in EXPECTED.items():
        p = ROOT / rel
        if not p.exists():
            ok = False
            print(f"FAIL missing {rel}")
            continue
        got, size = sha256(p), p.stat().st_size
        match = got == exp and size == nbytes
        ok &= match
        print(f"{'OK  ' if match else 'FAIL'} {rel}: {size} B sha256 {got[:8]}…{got[-8:]} (expected {exp[:8]}…{exp[-8:]}, {nbytes} B)")

    # row counts and structure
    ref_dir = ROOT / "data/contrast-sets/refusal"
    for name, min_rows in [("harmful_train.json", 64), ("harmless_train.json", 64)]:
        p = ref_dir / name
        if p.exists():
            rows = json.load(open(p))
            good = len(rows) >= min_rows and all("instruction" in r for r in rows)
            ok &= good
            print(f"{'OK  ' if good else 'FAIL'} refusal/{name}: {len(rows)} rows, all have 'instruction'")

    bm = ROOT / "data/contrast-sets/badmed"
    pb, pg = bm / "bad_medical_advice.jsonl", bm / "good_medical_advice.jsonl"
    if pb.exists() and pg.exists():
        bad = [json.loads(l) for l in open(pb) if l.strip()]
        good_ = [json.loads(l) for l in open(pg) if l.strip()]
        aligned = sum(1 for b, g in zip(bad, good_) if b["messages"][0]["content"] == g["messages"][0]["content"])
        two_turn = all(len(r["messages"]) == 2 and r["messages"][0]["role"] == "user" and r["messages"][1]["role"] == "assistant" for r in bad + good_)
        cond = len(bad) == 7049 and len(good_) == 7049 and aligned == 7049 and two_turn
        ok &= cond
        print(f"{'OK  ' if cond else 'FAIL'} badmed: rows {len(bad)} / {len(good_)}, index-aligned prompts {aligned}/{len(bad)}, all two-turn user->assistant: {two_turn}")

    print("VERIFY", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
