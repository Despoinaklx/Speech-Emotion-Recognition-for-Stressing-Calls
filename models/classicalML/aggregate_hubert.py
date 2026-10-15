"""
Summary of the 8 HuBERT + classical-ML experiments
(results/classicalML/HUBERT_RESULTS/experiments/): CV and held-out test metrics.
Also checks the speaker split, picks the best 7-class model (CV F1-macro) and
prints its per-class recall (fear/disgust).

Usage (from repo root):
    python models/classicalML/aggregate_hubert.py
"""
import json
import os

import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
BASE = os.path.join(ROOT, 'results', 'classicalML', 'HUBERT_RESULTS')
EXP_DIR = os.path.join(BASE, 'experiments')
DELIV_DIR = os.path.join(BASE, 'deliverables')
os.makedirs(DELIV_DIR, exist_ok=True)

CONFIGS = [(model, classes) for model in ['RF', 'SVM', 'NB', 'MLP'] for classes in [7, 2]]


def exp_name(model, classes):
    return f"hubert_frame_{model.lower()}_{classes}class"


rows = []
for model, classes in CONFIGS:
    name = exp_name(model, classes)
    with open(os.path.join(EXP_DIR, name, 'metrics.json')) as f:
        m = json.load(f)
    rows.append({
        'Experiment': name,
        'Model': model,
        'Classes': classes,
        'CV Accuracy': round(m['accuracy'], 4),
        'CV F1-macro': round(m['f1_macro'], 4),
        'CV F1-weighted': round(m['f1_weighted'], 4),
        'Fold Acc mean±std': f"{m['fold_acc_mean']:.4f}±{m['fold_acc_std']:.4f}",
        'Test Accuracy': round(m['final_test_accuracy'], 4),
        'Test F1-macro': round(m['final_test_f1_macro'], 4),
        'Test F1-weighted': round(m['final_test_f1_weighted'], 4),
        'train_val_speakers': m['train_val_speakers'],
        'test_speakers': m['final_test_speakers'],
    })

summary = pd.DataFrame(rows)
summary_path = os.path.join(DELIV_DIR, 'hubert_8_experiments_summary.csv')
summary.to_csv(summary_path, index=False)
print("Summary saved:", summary_path)
print(summary.drop(columns=['train_val_speakers', 'test_speakers']).to_string(index=False))

# Speaker split check
print("\n--- Speaker split check ---")
splits_ok = all(
    r['train_val_speakers'] == [3, 8, 9, 10, 11, 14, 15, 16] and r['test_speakers'] == [12, 13]
    for r in rows
)
print(f"Train/Val == [3,8,9,10,11,14,15,16] and Test == [12,13] in all 8 experiments: {splits_ok}")

# Best 7-class model (CV F1-macro)
seven = summary[summary['Classes'] == 7].sort_values('CV F1-macro', ascending=False)
best_model = seven.iloc[0]['Model']
print(f"\n--- Best classifier (7 classes, CV F1-macro) ---")
print(seven[['Model', 'CV Accuracy', 'CV F1-macro', 'CV F1-weighted', 'Test F1-macro']].to_string(index=False))
print(f"Best: {best_model}")

# Per-class recall for the best model
best_name = exp_name(best_model, 7)
per_class = pd.read_csv(os.path.join(EXP_DIR, best_name, 'per_class_metrics.csv'), index_col=0)
print(f"\n--- Per-class metrics, {best_model} (7 classes, CV) ---")
print(per_class[['precision', 'recall', 'f1-score']].to_string())
for emo in ['fear', 'disgust']:
    if emo in per_class.index:
        print(f"Recall {emo}: {per_class.loc[emo, 'recall']:.4f}")
    else:
        print(f"[WARN] class '{emo}' not found in per_class_metrics for {best_name}")

per_class.to_csv(os.path.join(DELIV_DIR, f'hubert_{best_model.lower()}_7class_per_class_metrics.csv'))
