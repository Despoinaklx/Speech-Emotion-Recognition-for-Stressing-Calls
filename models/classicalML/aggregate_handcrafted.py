"""
Collect the 16 handcrafted classical-ML experiments
(results/classicalML_handcrafted/experiments/) into thesis deliverables:
  Table 4.1  - all 16 experiments (accuracy, F1-macro, F1-weighted)
  Table 4.2  - per-class precision/recall/F1, SVM frame-level 7-class
  Fig 4.1    - confusion matrix, SVM frame-level 7-class (normalized)
  Fig 4.2    - confusion matrix, SVM frame-level binary
  Fig 4.3    - fold accuracy per speaker, SVM frame-level 7-class
  Fig 4.4    - RF top-20 feature importance, frame-level 7-class

Usage (from repo root):
    python models/classicalML/aggregate_handcrafted.py
"""
import json
import os
import shutil

import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
BASE = os.path.join(ROOT, 'results', 'classicalML_handcrafted')
EXP_DIR = os.path.join(BASE, 'experiments')
DELIV_DIR = os.path.join(BASE, 'deliverables')
os.makedirs(DELIV_DIR, exist_ok=True)

CONFIGS = [
    (model, feature, classes)
    for model in ['RF', 'SVM', 'NB', 'MLP']
    for feature in ['frame', 'global']
    for classes in [7, 2]
]


def exp_name(model, feature, classes):
    return f"{feature}_{model.lower()}_{classes}class"


# Table 4.1: all 16 experiments
rows = []
for model, feature, classes in CONFIGS:
    name = exp_name(model, feature, classes)
    with open(os.path.join(EXP_DIR, name, 'metrics.json')) as f:
        m = json.load(f)
    rows.append({
        'Experiment': name,
        'Model': model,
        'Level': feature,
        'Classes': classes,
        'Accuracy': round(m['accuracy'], 4),
        'F1-macro': round(m['f1_macro'], 4),
        'F1-weighted': round(m['f1_weighted'], 4),
    })
t41 = pd.DataFrame(rows)
t41.to_csv(os.path.join(DELIV_DIR, 'Table4.1_16_experiments.csv'), index=False)
print("Table 4.1 saved:", t41.shape)

# Table 4.2: per-class metrics, SVM frame-level 7-class
svm_frame_7 = exp_name('SVM', 'frame', 7)
t42 = pd.read_csv(os.path.join(EXP_DIR, svm_frame_7, 'per_class_metrics.csv'))
t42.to_csv(os.path.join(DELIV_DIR, 'Table4.2_svm_frame_7class_precision_recall_f1.csv'), index=False)
print("Table 4.2 saved:", t42.shape)

# Fig 4.1: SVM frame-level 7-class (normalized)
shutil.copy(
    os.path.join(EXP_DIR, svm_frame_7, 'confusion_matrix_normalized.png'),
    os.path.join(DELIV_DIR, 'Fig4.1_confusion_matrix_svm_frame_7class_normalized.png'),
)

# Fig 4.2: SVM frame-level binary (raw + normalized)
svm_frame_2 = exp_name('SVM', 'frame', 2)
shutil.copy(
    os.path.join(EXP_DIR, svm_frame_2, 'confusion_matrix_raw.png'),
    os.path.join(DELIV_DIR, 'Fig4.2_confusion_matrix_svm_frame_binary.png'),
)
shutil.copy(
    os.path.join(EXP_DIR, svm_frame_2, 'confusion_matrix_normalized.png'),
    os.path.join(DELIV_DIR, 'Fig4.2b_confusion_matrix_svm_frame_binary_normalized.png'),
)

# Fig 4.3: fold accuracy per speaker, SVM frame-level 7-class
shutil.copy(
    os.path.join(EXP_DIR, svm_frame_7, 'fold_accuracy_bar_per_speaker.png'),
    os.path.join(DELIV_DIR, 'Fig4.3_fold_accuracy_per_speaker_svm_frame_7class.png'),
)

# Fig 4.4: RF top-20 feature importance, frame-level 7-class
rf_frame_7 = exp_name('RF', 'frame', 7)
shutil.copy(
    os.path.join(EXP_DIR, rf_frame_7, 'feature_importance_top20.png'),
    os.path.join(DELIV_DIR, 'Fig4.4_feature_importance_rf_frame_7class_top20.png'),
)
shutil.copy(
    os.path.join(EXP_DIR, rf_frame_7, 'feature_importance_top20.csv'),
    os.path.join(DELIV_DIR, 'Fig4.4_feature_importance_rf_frame_7class_top20.csv'),
)

print("All deliverables saved to:", DELIV_DIR)
