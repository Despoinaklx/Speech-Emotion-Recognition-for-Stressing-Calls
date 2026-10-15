#!/bin/bash
# All 8 HuBERT + classical-ML experiments (LOSO CV + held-out test).
# Raw outputs: results/classicalML/HUBERT_RESULTS/experiments/<name>/
# The slow SVM runs (768-dim frames) go last, in parallel. Set PYTHON to choose the interpreter.
set -e
cd "$(dirname "$0")/../.."

OUTBASE=results/classicalML/HUBERT_RESULTS
EXP_DIR=$OUTBASE/experiments
LOG_DIR=$OUTBASE/logs
mkdir -p "$EXP_DIR" "$LOG_DIR"

PY=${PYTHON:-python}
SCRIPT=models/classicalML/run_hubert_experiment.py

run_one () {
  local model=$1
  local classes=$2
  local name="hubert_frame_$(echo "$model" | tr '[:upper:]' '[:lower:]')_${classes}class"
  echo "=== [$(date '+%H:%M:%S')] START $name ==="
  $PY $SCRIPT --model "$model" --classes "$classes" \
      --outdir "$EXP_DIR/$name" > "$LOG_DIR/$name.log" 2>&1
  echo "=== [$(date '+%H:%M:%S')] DONE  $name ==="
}

echo "########## Phase 1: 6 faster experiments (sequential) ##########"
run_one RF  7
run_one RF  2
run_one NB  7
run_one NB  2
run_one MLP 7
run_one MLP 2

echo "########## Phase 2: 2 slow SVM experiments (parallel) ##########"
run_one SVM 7 &
PID1=$!
run_one SVM 2 &
PID2=$!
wait $PID1
wait $PID2

echo "########## All 8 HuBERT experiments finished ##########"
