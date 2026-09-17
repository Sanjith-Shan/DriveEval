#!/usr/bin/env bash
# Produce every measurement the report needs, in one pass.
#
# Runs are sequential on purpose: each already uses every core, and overlapping
# two of them would make the latency numbers measure contention rather than the
# planner. Every run records its own hardware string and config into runs.csv.
set -euo pipefail
cd "$(dirname "$0")/.."

SHARDS=$(ls data/cache/av2_val/*.scn | tr '\n' ',' | sed 's/,$//')
OUT=${OUT:-data/results}
JOBS=${JOBS:-10}
# Scenarios per headline run. 8000 is the default because it powers the paired
# log-replay/reactive comparison comfortably while finishing in about an hour.
# Set N=19763 for the whole validation split, which roughly doubles the miner's
# confirmation-split failure count and therefore its power to resolve a class.
N=${N:-8000}
SWEEP_N=${SWEEP_N:-3000}
GIT=$(git rev-parse --short HEAD 2>/dev/null || echo nogit)
HW="Apple M3 Pro (12 core), -O3 -mcpu=native, process not pinned"
B=./build/drive_batch
mkdir -p "$OUT"

run() {
  local id=$1; shift
  if [ -f "$OUT/metrics_$id.csv" ] && [ -z "${FORCE:-}" ]; then
    echo "== $id already present, skipping"
    return
  fi
  echo "== $id  $*"
  $B --shards "$SHARDS" --out "$OUT" --run-id "$id" --jobs "$JOBS" \
     --git-sha "$GIT" --hardware "$HW" "$@" 2>&1 | tail -14
  echo
}

# 1. The two headline runs. Paired over the same scenarios, so the comparison
#    between agent models is within-scenario and the bootstrap can cancel the
#    between-scenario variance the two modes share.
run baseline_log_replay --config-name baseline --mode log_replay --backend ilqr --features --limit "$N"
run baseline_reactive   --config-name baseline --mode reactive   --backend ilqr --limit "$N"

# 2. The human's own track through the identical metric suite. The floor that
#    every planner rate has to be read against: it is how much of each rate
#    belongs to the footprint constants and the map rather than to the planner.
run human_log_replay --config-name human --mode log_replay --logged-ego --limit "$N"

# 3. Baseline planner: pure pursuit on the A* route, no obstacle reasoning at
#    all. Labelled as such wherever its numbers appear.
run pursuit_reactive --config-name pure_pursuit --mode reactive --pure-pursuit --limit "$SWEEP_N"

# 4. Cost-weight sweep, for the regression gate. One weight at a time so the
#    gate has something interpretable to attribute a change to.
#
#    The optimiser head to head lives in drive_bench, which hands iLQR and OSQP
#    identical problems; comparing two whole runs would confound the solvers
#    with the different trajectories they steer the scenarios into. Prediction
#    and refinement attribution live in drive_ablate, which does them per scenario
#    as counterfactuals rather than as separate batches.
for w in clearance=16.0 progress=3.0; do
  name=$(echo "$w" | tr '=.' '__')
  run "sweep_$name" --config-name "sweep_$w" --mode reactive --backend ilqr \
      --weight "$w" --limit "$SWEEP_N"
done

echo "all runs complete"
ls -la "$OUT"
