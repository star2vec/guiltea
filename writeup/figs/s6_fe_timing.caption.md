# There is no fixed turn at which the model breaks

Machine-written by `scripts/figs/fe_timing.py`; regenerate, never hand-edit. CPU only, no generation, no model load, no judge call, no cost.

**Data sources**

- `results/raw/s1b/t10/summary.json (`T_distribution`, `categories`, per-chain grades)`

**Selection rule:** none — every v1 chain counted once under the rig's stored label

No example is selected: every one of the 192 v1 chains is counted once, at its first turn graded `committed` by the rig's stored act-judge label, or in the never-commit bar if no turn is. Counts are read from `T_distribution` and `categories["no capitulation"]` in the source file and cross-checked against its per-chain list (first `committed` grade per chain). Nothing dropped, smoothed or clipped.

`T_used`, verbatim from the file: T_primary (PROVISIONAL — the adjudicated labels are not in yet)

| first committed turn | chains |
|---|---|
| 1 | 3 |
| 2 | 21 |
| 3 | 37 |
| 4 | 11 |
| 5 | 10 |
| 6 | 5 |
| 7 | 5 |
| 8 | 1 |
| 9 | 1 |
| 10 | 15 |
| never (no turn graded committed; `partial` grades may occur) | 83 |
| total | 192 |

109 of 192 chains commit at some turn; the median first committed turn over those 109 is 3 (the same count the D1 transcript panel's rule uses).

The file's own categories, verbatim: eligible = 37, no capitulation = 83, immediate capitulation = 61, short (4 <= T <= 4) = 11.
