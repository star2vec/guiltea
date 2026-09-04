"""S1e Task 1 — THE TRIGGER DEMO: among runs still refusing at turn t, which ones break later?

The early-warning question with conversation depth held fixed. At each turn index t:

  positives  v1 chains graded `held` at t which have a `committed` turn later (strictly after t)
  negatives  v1 chains graded `held` at t which never commit within ten turns

Both classes carry the same grade at the same depth, so nothing separates them except what the state
carries about what the run will do next. Scores are the stored projections at turn t (`answer` and,
separately, `into`), nine axes, all 32 layers, against the randctl seed 0-9 floor, folded by target.

Headline: the L14-18 band mean against the seeds' own band means. Best-over-layers only beside the
selection-matched floor. Both label sources. Nothing is pooled across t.

EXPLORATORY. No API, no GPU, no model load, no cost.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location("s1e_depth", REPO / "scripts" / "s1e" / "depth.py")
D = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(D)


def main():
    print("S1e Task 1 - trigger demo (held at t, breaks later vs held at t, never breaks)")
    out = D.run_task(
        "t1_trigger",
        D.members_trigger,
        "held at t and commits later (positive) vs held at t and never commits (negative)",
        {"question": "among runs still refusing at turn t, which ones break later?",
         "positive_class": "held at t, `committed` at some turn > t",
         "negative_class": "held at t, no `committed` turn at any of the ten turns",
         "excluded": "`partial` at t, and chains already committed at or before t"})
    D.save(out, "t1_trigger.json")
    for key in sorted(k for k in out if k != "meta"):
        r = out[key]
        for t in r["evaluable_turns"]:
            v = r["per_t"][str(t)]["verdict"]["pooled"]["L14_18"]
            print("%-46s t=%-2d n+=%-4d n-=%-4d tgt_both=%-3d best %-16s band %.3f  floor max %.3f  beats %s"
                  % (key, t, r["per_t"][str(t)]["n_positive"], r["per_t"][str(t)]["n_negative"],
                     r["per_t"][str(t)]["n_targets_with_both_classes"], v["best_axis"],
                     v["best_axis_band_mean"], v["floor_band_mean"]["excess_max"], v["beats_floor"]))


if __name__ == "__main__":
    main()
