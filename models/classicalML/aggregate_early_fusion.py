"""
Early-fusion (frame + global, SVM) deliverables, saved next to the handcrafted ones:
  Table 4.3 - SVM frame vs global vs early fusion + PCA (7 and 2 classes)
  Table 4.4 - per-class precision/recall/F1, early fusion 7-class
  Fig 4.5   - confusion matrix, early fusion 7-class (normalized)
  Fig 4.6   - confusion matrix, early fusion binary (raw + normalized)

Usage (from repo root):
    python models/classicalML/aggregate_early_fusion.py
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


def load_metrics(name):
    with open(os.path.join(EXP_DIR, name, 'metrics.json')) as f:
        return json.load(f)


# Table 4.3: three SVM setups
rows = []
for classes in [7, 2]:
    configs = [
        ('SVM frame', f'frame_svm_{classes}class'),
        ('SVM global', f'global_svm_{classes}class'),
        ('SVM early fusion + PCA', f'early_fusion_svm_{classes}class'),
    ]
    for label, exp_name in configs:
        m = load_metrics(exp_name)
        rows.append({
            'Setup': label,
            'Classes': classes,
            'Accuracy': round(m['accuracy'], 4),
            'F1-macro': round(m['f1_macro'], 4),
        })
t43 = pd.DataFrame(rows)
t43.to_csv(os.path.join(DELIV_DIR, 'Table4.3_svm_setup_comparison.csv'), index=False)
print("Table 4.3 saved:")
print(t43.to_string(index=False))

# Table 4.4: per-class metrics, early fusion 7-class
ef7 = 'early_fusion_svm_7class'
t44 = pd.read_csv(os.path.join(EXP_DIR, ef7, 'per_class_metrics.csv'))
t44.to_csv(os.path.join(DELIV_DIR, 'Table4.4_early_fusion_7class_precision_recall_f1.csv'), index=False)
print("\nTable 4.4 saved:", t44.shape)

# Fig 4.5: early fusion 7-class (normalized)
shutil.copy(
    os.path.join(EXP_DIR, ef7, 'confusion_matrix_normalized.png'),
    os.path.join(DELIV_DIR, 'Fig4.5_confusion_matrix_early_fusion_7class_normalized.png'),
)

# Fig 4.6: early fusion binary (raw + normalized)
ef2 = 'early_fusion_svm_2class'
shutil.copy(
    os.path.join(EXP_DIR, ef2, 'confusion_matrix_raw.png'),
    os.path.join(DELIV_DIR, 'Fig4.6_confusion_matrix_early_fusion_binary.png'),
)
shutil.copy(
    os.path.join(EXP_DIR, ef2, 'confusion_matrix_normalized.png'),
    os.path.join(DELIV_DIR, 'Fig4.6b_confusion_matrix_early_fusion_binary_normalized.png'),
)

print("\nEarly-fusion deliverables saved to:", DELIV_DIR)
