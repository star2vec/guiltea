# Direction file provenance

Carried over from the pre-project checks repo (see `checks/checks-log/`).

- `dir_1B_L8.pt` — produced by Check 6 (`check6_induction.py`, same recipe as Check 02b): bad-medical misalignment direction (bad-vs-good-medical answer-token diff-in-means, N=64, seed 0, norm 0.437), model `unsloth/Llama-3.2-1B-Instruct` (base), layer 8; keys `{direction, layer}`.
- `dirs_base_sweep.pt` — produced by Check 8a (`check8_extract.py`): unit directions `badmed` (misalignment, bad-vs-good-medical answer-token diff-in-means), `refusal` (Arditi harmful−harmless last-token diff-in-means), `persona` (guarded-vs-helpful system-prompt last-token diff-in-means) plus their norms, model `unsloth/Llama-3.2-1B-Instruct` (base), layers 6/8/10/12/14; keys `{units, norms, layers}`; used by Checks 8b, 9A/9B, 10A/10B.
