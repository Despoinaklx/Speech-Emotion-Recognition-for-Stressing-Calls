# Speech Emotion Recognition under Telephone-Channel Degradation

Speech emotion recognition (SER) on **EmoDB**, from handcrafted features with classical ML up to a
**BiLSTM over HuBERT embeddings**, and a study of how well the models hold up when speech passes
through a simulated telephone channel (G.711 A-law and GSM 06.10 codecs plus additive noise). This
matters for emergency-call (112/911) triage.

Main components:

1. **Feature extraction**: handcrafted prosodic, spectral and voice-quality features (librosa, Praat, LPCC)
   and frame-level HuBERT (`facebook/hubert-base-ls960`) embeddings.
2. **Classical ML**: RF / SVM / Naive Bayes / MLP, frame vs. global features, early fusion, HuBERT inputs.
3. **(Bi)LSTM**: sequence models on handcrafted and HuBERT frame sequences.
4. **Telephone-channel degradation**: 11 conditions (`original`, `{alaw,gsm}_{clean,snr20,snr15,snr10,snr5}`).
   - *matched*: train and test on the same condition
   - *mismatched*: train on clean speech, test on degraded speech
5. **Inference / demo**: a deployable BiLSTM checkpoint, call analysis with sliding windows,
   valence–arousal mapping and urgency-based call prioritisation.

All experiments use the same **speaker-independent split** ([data/splits/speaker_split.json](data/splits/speaker_split.json)):
8 speakers for **8-fold leave-one-speaker-out (LOSO)** cross-validation, plus 2 held-out test speakers (12, 13).

---

## Repository structure

```
.
├── channel_degradation/
│   └── telephone_channel.py          # IRS shaping, 300-3400 Hz, 8 kHz, noise @ SNR, A-law / GSM round-trip
├── feature_extraction/
│   ├── handcrafted/                  # librosa, Praat (parselmouth), LPCC, rhythm -> frame + global CSVs
│   │   └── pipeline.py
│   └── hubert/
│       ├── embeddings.py             # HuBERT mean-pooled + frame-level embeddings
│       └── extract_hubert_labels.py  # filename -> speaker / emotion CSV
├── models/
│   ├── classicalML/
│   │   ├── split.py                       # writes train(LOSO)/test CSVs using the fixed split
│   │   ├── run_experiment.py              # one handcrafted experiment (model x frame/global x 7/2 classes)
│   │   ├── run_early_fusion_experiment.py # frame + global early fusion, SVM + PCA
│   │   ├── train_early_fusion_final.py    # final early-fusion pipeline for the held-out test
│   │   ├── test_early_fusion.py           # held-out test evaluation of that pipeline
│   │   ├── run_hubert_experiment.py       # classical ML on HuBERT frames (+ held-out test)
│   │   ├── aggregate_*.py                 # tables / figures for the thesis
│   │   ├── run_all_handcrafted.sh         # all 16 handcrafted experiments + early fusion
│   │   └── run_all_hubert.sh              # all 8 HuBERT experiments
│   └── LSTM/
│       ├── config.py                      # paths + default (tuned) hyperparameters
│       ├── utils/                         # dataset, model, LOSO split/order, training loop, plots
│       ├── run_handcrafted.py             # LSTM on handcrafted frame features
│       ├── run_hubert_experiment.py       # (Bi)LSTM on HuBERT frames, 7/6/2 classes, CV + held-out test
│       ├── run_degradation_experiments.py # matched-condition degradation study
│       ├── run_mismatched_degradation.py  # clean-trained checkpoints evaluated on degraded audio
│       ├── train_deployable_bilstm.py     # single BiLSTM on alaw_snr20 -> checkpoints/
│       ├── infer_call_emotions.py         # sliding-window inference on one recording
│       ├── analysis/                      # degradation plots and aggregation
│       └── checkpoints/bilstm_hubert_alaw_snr20.pt
├── inference/
│   ├── analyze_call.py                # call -> 3 s segments -> call_analysis.json (dashboard)
│   ├── analyze_clips.py               # held-out EmoDB clips -> demo call queue
│   └── make_demo.py                   # clean vs simulated alaw_snr20 demo JSON
├── scripts/
│   └── plot_feature_extraction_figures.py
├── data/splits/speaker_split.json     # fixed speaker split (the only data file in the repo)
└── results/                           # summary results (CSVs + figures)
```

## Installation

```bash
git clone <this-repo-url>
cd <repo>
python -m venv .venv && source .venv/bin/activate   # or a conda environment
pip install -r requirements.txt
```

You also need `ffmpeg` on the `PATH` (`sudo apt install ffmpeg`). The A-law/GSM round-trip uses `soundfile`
first and only falls back to ffmpeg. A GPU is recommended for HuBERT and the LSTMs: one 8-fold LOSO run of
the BiLSTM takes about 3 minutes on a GTX 1080 Ti.

## Data

EmoDB is **not** included. Download it from <http://emodb.bilderbar.info/> and place the wav files at:

```
data/raw/emodb/wav/*.wav
```

Every script resolves paths relative to the repository root. The scripts create this layout:

| Path | Created by |
|---|---|
| `data/features/handcrafted/all_{frame,global}_features.csv` | `feature_extraction/handcrafted/pipeline.py` |
| `data/features/hubert/{frame_level,mean_pooled}/*.npy`, `hubert_labels.csv` | `feature_extraction/hubert/*.py` |
| `data/splits/{frame,global}_{train_for_loso,final_test}.csv` | `models/classicalML/split.py` |
| `data/splits/prepared_data.npz` | `models/LSTM/utils/handcrafted_data_preparation.py` |
| `data/splits/prepared_hubert_data*.npz` | `models.LSTM.utils.hubert_data_preparation.prepare_hubert` |
| `data/degraded/<condition>/*.wav`, `manifest.csv` | `channel_degradation/telephone_channel.py` |
| `data/features/hubert_conditions/`, `data/splits/conditions/` | `models.LSTM.run_degradation_experiments` |

## Usage

Run all commands from the repository root. Run the LSTM code as modules (`python -m ...`).

### 1. Features

```bash
python feature_extraction/handcrafted/pipeline.py
python feature_extraction/hubert/extract_hubert_labels.py
python feature_extraction/hubert/embeddings.py
```

### 2. Classical ML

```bash
python models/classicalML/split.py
./models/classicalML/run_all_handcrafted.sh       # 16 experiments + early fusion
./models/classicalML/run_all_hubert.sh            # 8 HuBERT experiments
python models/classicalML/aggregate_handcrafted.py
python models/classicalML/aggregate_early_fusion.py
python models/classicalML/aggregate_hubert.py

# Single run
python models/classicalML/run_experiment.py --model SVM --feature global --classes 7 --outdir results/tmp
```

### 3. LSTM / BiLSTM

```bash
python models/LSTM/utils/handcrafted_data_preparation.py
python -m models.LSTM.run_handcrafted

# HuBERT (BiLSTM, tuned, 6 classes without happiness)
python -m models.LSTM.run_hubert_experiment --tuned --bidirectional --num_classes 6 \
    --outdir results/LSTM/results_hubert_embeddings_BIDIRECTIONAL_6class_tuned

./models/LSTM/run_all_hubert_lstm.sh [--tuned]    # LSTM + BiLSTM, 7 classes
```

`--num_classes 7` reads `data/splits/prepared_hubert_data_7classes_backup.npz` (with happiness).
`--num_classes 6` reads `data/splits/prepared_hubert_data.npz`. `--binary` runs neutral/boredom vs. negative emotions.

### 4. Telephone-channel degradation

```bash
# a) Degrade EmoDB (11 conditions)
python channel_degradation/telephone_channel.py data/raw/emodb/wav data/degraded

# b) Matched: train and test on each condition
python -m models.LSTM.run_degradation_experiments \
    --data-root data/degraded \
    --embed-root data/features/hubert_conditions \
    --npz-root data/splits/conditions \
    --results-root results/LSTM/degradation \
    --out-csv degradation_results.csv --per-class-csv per_class_results.csv --per-fold-csv per_fold_metrics.csv

# c) Mismatched: clean-trained fold checkpoints tested on degraded audio
python -m models.LSTM.run_hubert_experiment --tuned --bidirectional --num_classes 6 \
    --outdir results/LSTM/reference_baseline_6class --checkpoint-root results/LSTM/clean_checkpoints
python -m models.LSTM.run_mismatched_degradation \
    --checkpoint-root results/LSTM/clean_checkpoints --outdir results/LSTM/mismatched_degradation

# d) Plots / tables
python models/LSTM/analysis/plot_degradation_curve.py degradation_results.csv
python models/LSTM/analysis/plot_per_class_degradation.py per_class_results.csv
python models/LSTM/analysis/plot_degradation_heatmap.py degradation_results.csv per_class_results.csv
python models/LSTM/analysis/plot_confusion_grid.py results/LSTM/degradation
python models/LSTM/analysis/compare_mismatched_matched.py \
    --mismatched-root results/LSTM/mismatched_degradation --out matched_vs_mismatched.csv
```

### 5. Inference on real recordings

The repo includes the checkpoint `models/LSTM/checkpoints/bilstm_hubert_alaw_snr20.pt` (BiLSTM, 6 classes,
trained on `alaw_snr20`). It stores the weights, scaler, feature-selection mask and label classes.

```bash
python -m models.LSTM.infer_call_emotions call.wav results/call_analysis   # CSV + plots + text summary
python inference/analyze_call.py --audio call.wav --out call_analysis.json  # dashboard JSON
python inference/make_demo.py call.wav > demo.json                          # clean vs telephone demo

# Retrain the checkpoint
python -m models.LSTM.train_deployable_bilstm
```

## Results (summary)

BiLSTM + HuBERT, 6 classes (anger, boredom, disgust, fear, neutral, sadness):

| Setup | LOSO acc | LOSO F1-macro | Held-out test acc | Held-out test F1-macro |
|---|---|---|---|---|
| BiLSTM + HuBERT, tuned, 6 classes | 0.918 | 0.910 | 0.941 | 0.932 |

Telephone-channel degradation, pooled 8-fold LOSO predictions
([results/degradation](results/degradation), [results/mismatched_degradation](results/mismatched_degradation)):

| Condition | Matched acc | Matched F1 | Mismatched acc | Mismatched F1 |
|---|---|---|---|---|
| original | 0.913 | 0.901 | 0.903 | 0.894 |
| A-law clean | 0.874 | 0.866 | 0.624 | 0.580 |
| A-law 20 dB | 0.866 | 0.855 | 0.434 | 0.346 |
| A-law 10 dB | 0.790 | 0.778 | 0.382 | 0.256 |
| A-law 5 dB | 0.792 | 0.776 | 0.374 | 0.247 |
| GSM clean | 0.847 | 0.832 | 0.526 | 0.472 |
| GSM 20 dB | 0.834 | 0.820 | 0.453 | 0.371 |
| GSM 10 dB | 0.795 | 0.783 | 0.389 | 0.256 |
| GSM 5 dB | 0.779 | 0.770 | 0.392 | 0.285 |

Training on matched telephone-channel data keeps macro-F1 between 0.77 and 0.87. A model trained only on
clean studio speech falls to 0.25–0.58. *Fear* is the class that degrades most.

## Notes

- Seeds are fixed (`set_seeds`, `cudnn.deterministic=True`). Reruns are reproducible on the same hardware
  and library versions.
- The model is trained on **acted** speech (EmoDB). Outputs on real calls are indicative only and are not
  clinical or operational decisions.

## Acknowledgements

- EmoDB: Burkhardt et al., *A Database of German Emotional Speech*, Interspeech 2005.
- HuBERT: Hsu et al., 2021, `facebook/hubert-base-ls960` via 🤗 Transformers.
