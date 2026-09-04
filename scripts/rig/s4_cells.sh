#!/usr/bin/env bash
# S4 — the seven cells, in the brief's order: the never-cut set (A, B, E, F, G) first, then the steered extras.
# One target list per invocation; every invocation is resume-safe, so re-running costs nothing for finished runs.
#   scripts/rig/s4_cells.sh <target[,target...]> <logfile>
set -u
TARGETS="$1"; LOG="$2"
cd "$(dirname "$0")/../.."
# expandable segments: the readout and the generation prefill alternate large short-lived tensors, and the
# fragmentation between them is what leaves a 24 GiB card without a contiguous block. No measured quantity
# depends on the allocator.
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
COMMON="--model base --mode deceived --targets $TARGETS --seeds 8 --distance4 --controls --judges mini --budget 11.5"
run () {  # run <label> <extra args...>
  local label="$1"; shift
  echo "=== cell $label : $(date -u +%H:%M:%S) ===" >> "$LOG"
  python3 scripts/rig/run.py $COMMON "$@" >> "$LOG" 2>&1
  local rc=$?
  echo "=== cell $label exit $rc : $(date -u +%H:%M:%S) ===" >> "$LOG"
  if [ $rc -ne 0 ]; then echo "STOP: cell $label exited $rc" >> "$LOG"; exit $rc; fi
}
run "A act_blame"          --arm act_blame
run "B self_blame"         --arm self_blame
run "E neutral_correction" --arm neutral_correction
run "F neutral_reflection" --arm neutral_reflection
run "G none"               --arm none
run "C steer guilt_clean"  --arm self_blame --steer guilt_clean:16:4 --steer-off-after distance0
run "D steer random0"      --arm self_blame --steer random0:16:4 --steer-sigma-from guilt_clean --steer-off-after distance0
echo "=== all seven cells done for $TARGETS : $(date -u +%H:%M:%S) ===" >> "$LOG"
