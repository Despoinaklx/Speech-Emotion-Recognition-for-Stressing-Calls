#!/bin/bash
# LSTM and BiLSTM on HuBERT frame-level embeddings (7 classes, fixed 8/2 split).
#   ./models/LSTM/run_all_hubert_lstm.sh           # matched-to-handcrafted hyperparameters
#   ./models/LSTM/run_all_hubert_lstm.sh --tuned   # tuned hyperparameters
# Runs sequentially (single GPU). Set PYTHON to choose the interpreter.
set -e
cd "$(dirname "$0")/../.."

PY=${PYTHON:-python}
TUNED=""
SUFFIX=""
if [ "$1" == "--tuned" ]; then
  TUNED="--tuned"
  SUFFIX="_tuned"
fi

RES=results/LSTM
LOG_DIR=$RES/logs
mkdir -p "$LOG_DIR"

echo "=== [$(date '+%H:%M:%S')] START LSTM${SUFFIX} ==="
$PY -m models.LSTM.run_hubert_experiment $TUNED \
    --outdir "$RES/results_hubert_embeddings${SUFFIX}" \
    > "$LOG_DIR/hubert_lstm${SUFFIX}.log" 2>&1
echo "=== [$(date '+%H:%M:%S')] DONE  LSTM${SUFFIX} ==="

echo "=== [$(date '+%H:%M:%S')] START BiLSTM${SUFFIX} ==="
$PY -m models.LSTM.run_hubert_experiment $TUNED --bidirectional \
    --outdir "$RES/results_hubert_embeddings_BIDIRECTIONAL${SUFFIX}" \
    > "$LOG_DIR/hubert_bilstm${SUFFIX}.log" 2>&1
echo "=== [$(date '+%H:%M:%S')] DONE  BiLSTM${SUFFIX} ==="

echo "########## LSTM + BiLSTM HuBERT experiments finished ##########"
