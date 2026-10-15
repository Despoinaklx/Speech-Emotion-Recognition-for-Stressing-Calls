#!/bin/bash
# All 16 handcrafted classical-ML experiments (LOSO CV) plus the two early-fusion runs.
# Raw outputs: results/classicalML_handcrafted/experiments/<name>/
# The slow SVM frame-level runs go last, in parallel. Set PYTHON to choose the interpreter.
set -e
cd "$(dirname "$0")/../.."

OUTBASE=results/classicalML_handcrafted
EXP_DIR=$OUTBASE/experiments
LOG_DIR=$OUTBASE/logs
mkdir -p "$EXP_DIR" "$LOG_DIR"

PY=${PYTHON:-python}
SCRIPT=models/classicalML/run_experiment.py

run_one () {
  local model=$1
  local feature=$2
  local classes=$3
  local name="${feature}_$(echo "$model" | tr '[:upper:]' '[:lower:]')_${classes}class"
  echo "=== [$(date '+%H:%M:%S')] START $name ==="
  $PY $SCRIPT --model "$model" --feature "$feature" --classes "$classes" \
      --outdir "$EXP_DIR/$name" > "$LOG_DIR/$name.log" 2>&1
  echo "=== [$(date '+%H:%M:%S')] DONE  $name ==="
}

run_fusion () {
  local classes=$1
  local name="early_fusion_svm_${classes}class"
  echo "=== [$(date '+%H:%M:%S')] START $name ==="
  $PY models/classicalML/run_early_fusion_experiment.py --classes "$classes" \
      --outdir "$EXP_DIR/$name" > "$LOG_DIR/$name.log" 2>&1
  echo "=== [$(date '+%H:%M:%S')] DONE  $name ==="
}

echo "########## Phase 1: 14 fast experiments (sequential) ##########"
run_one RF  frame  7
run_one RF  frame  2
run_one RF  global 7
run_one RF  global 2
run_one NB  frame  7
run_one NB  frame  2
run_one NB  global 7
run_one NB  global 2
run_one MLP frame  7
run_one MLP frame  2
run_one MLP global 7
run_one MLP global 2
run_one SVM global 7
run_one SVM global 2

echo "########## Phase 2: 2 slow SVM frame-level experiments (parallel) ##########"
run_one SVM frame 7 &
PID1=$!
run_one SVM frame 2 &
PID2=$!
wait $PID1
wait $PID2

echo "########## Phase 3: early fusion (SVM + PCA) ##########"
run_fusion 7
run_fusion 2

echo "########## All handcrafted experiments finished ##########"
