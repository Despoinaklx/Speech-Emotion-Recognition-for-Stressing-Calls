"""
Early-fusion LOSO experiment: frame-level + global features merged per frame
(each frame also sees its utterance's global statistics), SVM + PCA.
Same outputs as run_experiment.py.

Usage (from repo root):
    python models/classicalML/run_early_fusion_experiment.py --classes 7 --outdir <dir> [--pca 30]
"""
import argparse
import json
import os
import time
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from plotting import plot_confusion_matrix

from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score, f1_score, confusion_matrix, classification_report,
)
from sklearn.impute import SimpleImputer
from sklearn.svm import SVC

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
ORIGINAL_SPLIT_DIR = os.path.join(ROOT, 'data', 'splits')
FRAME_FILE = 'frame_train_for_loso.csv'
GLOBAL_FILE = 'global_train_for_loso.csv'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--classes', required=True, type=int, choices=[2, 7])
    parser.add_argument('--outdir', required=True)
    parser.add_argument('--pca', type=int, default=30)
    args = parser.parse_args()

    model_name = 'SVM'
    feature_type = 'early_fusion'
    num_classes = args.classes
    results_dir = args.outdir
    os.makedirs(results_dir, exist_ok=True)

    df_frame = pd.read_csv(os.path.join(ORIGINAL_SPLIT_DIR, FRAME_FILE))
    df_global = pd.read_csv(os.path.join(ORIGINAL_SPLIT_DIR, GLOBAL_FILE))
    df = pd.merge(df_frame, df_global, on=['filename', 'speaker_id', 'emotion'],
                  suffixes=('_frame', '_global'))
    print(f"[SVM/early_fusion/{num_classes}cl] Merged {len(df)} frame rows "
          f"({df_frame['filename'].nunique()} utterances)")

    if num_classes == 2:
        neutral_labels = ['neutral', 'boredom']
        negative_labels = ['anger', 'sadness', 'fear', 'disgust']
        df = df[~df['emotion'].isin(['happiness'])]
        df = df[df['emotion'].isin(neutral_labels + negative_labels)]
        df['emotion_binary'] = df['emotion'].apply(lambda x: 0 if x in neutral_labels else 1)
        y = df['emotion_binary']
        class_names = ['neutral', 'negative']
    else:
        y = df['emotion']
        class_names = sorted(df['emotion'].unique())

    exclude_cols = ['filename', 'frame_index', 'speaker_id', 'emotion', 'emotion_binary'] \
        if 'emotion_binary' in df.columns else ['filename', 'frame_index', 'speaker_id', 'emotion']
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    X = df[feature_cols]
    groups = df['speaker_id']

    if num_classes == 2:
        y_encoded = y.to_numpy()
    else:
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)
        class_names = list(le.classes_)

    print(f" Features: {len(feature_cols)} | Samples: {len(X)} | Speakers: {groups.nunique()}")
    print(f" Classes: {class_names} | PCA components: {args.pca}")

    X = X.reset_index(drop=True)
    y_encoded = pd.Series(y_encoded).reset_index(drop=True)
    groups = groups.reset_index(drop=True)
    speaker_ids = df['speaker_id'].reset_index(drop=True)

    logo = LeaveOneGroupOut()
    n_folds = logo.get_n_splits(groups=groups)

    all_preds, all_true, all_speakers = [], [], []
    fold_rows = []

    start_time = time.time()
    for fold, (train_idx, val_idx) in enumerate(logo.split(X, y_encoded, groups)):
        X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
        y_train_fold, y_val_fold = y_encoded[train_idx], y_encoded[val_idx]
        val_speaker = speaker_ids.iloc[val_idx].unique()[0]

        train_class_counts = Counter(y_train_fold)
        min_class_count = min(train_class_counts.values())
        k_neighbors = min(5, max(1, min_class_count - 1))

        pipeline_steps = [
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler()),
        ]
        if args.pca is not None:
            pipeline_steps.append(('pca', PCA(n_components=args.pca, random_state=42)))
        if min_class_count >= 2:
            pipeline_steps.append(('smote', SMOTE(random_state=42, k_neighbors=k_neighbors)))
        pipeline_steps.append(('classifier', SVC(kernel='rbf', probability=False, random_state=42)))
        pipeline = ImbPipeline(pipeline_steps)

        pipeline.fit(X_train_fold, y_train_fold)
        preds = pipeline.predict(X_val_fold)

        all_preds.extend(preds)
        all_true.extend(y_val_fold)
        all_speakers.extend([val_speaker] * len(preds))

        fold_acc = accuracy_score(y_val_fold, preds)
        fold_rows.append({
            'fold': fold + 1,
            'speaker_id': val_speaker,
            'n_samples': len(val_idx),
            'accuracy': fold_acc,
        })
        print(f" Fold {fold + 1}/{n_folds} | speaker={val_speaker} | acc={fold_acc:.4f} "
              f"| elapsed={time.time() - start_time:.1f}s")

    elapsed_time = time.time() - start_time
    accuracy = accuracy_score(all_true, all_preds)
    f1_macro = f1_score(all_true, all_preds, average='macro', zero_division=0)
    f1_weighted = f1_score(all_true, all_preds, average='weighted', zero_division=0)
    fold_df = pd.DataFrame(fold_rows)

    print(f"Accuracy={accuracy:.4f} F1macro={f1_macro:.4f} F1weighted={f1_weighted:.4f} time={elapsed_time:.1f}s")

    pd.DataFrame({
        'speaker_id': all_speakers,
        'true_label': all_true,
        'pred_label': all_preds,
    }).to_csv(os.path.join(results_dir, 'predictions.csv'), index=False)

    fold_df.to_csv(os.path.join(results_dir, 'fold_results.csv'), index=False)

    class_report = classification_report(all_true, all_preds, target_names=class_names, digits=4, zero_division=0)
    with open(os.path.join(results_dir, 'classification_report.txt'), 'w') as f:
        f.write(f"Experiment: EARLY_FUSION (frame+global) | {model_name} | {num_classes}-class | PCA={args.pca}\n")
        f.write("=" * 60 + "\n")
        f.write(f"Overall Accuracy: {accuracy:.4f}\n")
        f.write(f"F1-Macro: {f1_macro:.4f}\n")
        f.write(f"F1-Weighted: {f1_weighted:.4f}\n")
        f.write(f"Execution time: {elapsed_time:.2f} seconds\n")
        f.write(f"Fold Accuracy: {fold_df['accuracy'].mean():.4f} +- {fold_df['accuracy'].std():.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(class_report)

    report_dict = classification_report(all_true, all_preds, target_names=class_names, output_dict=True, zero_division=0)
    per_class_df = pd.DataFrame(report_dict).transpose().iloc[:-3, :]
    per_class_df.index.name = 'class'
    per_class_df.to_csv(os.path.join(results_dir, 'per_class_metrics.csv'))

    with open(os.path.join(results_dir, 'metrics.json'), 'w') as f:
        json.dump({
            'experiment': f"early_fusion_svm_{num_classes}class",
            'model': model_name,
            'feature_type': feature_type,
            'num_classes': num_classes,
            'pca_n_components': args.pca,
            'accuracy': accuracy,
            'f1_macro': f1_macro,
            'f1_weighted': f1_weighted,
            'fold_acc_mean': float(fold_df['accuracy'].mean()),
            'fold_acc_std': float(fold_df['accuracy'].std()),
            'n_features': len(feature_cols),
            'n_samples': len(X),
            'n_speakers': int(groups.nunique()),
            'execution_time_sec': elapsed_time,
            'class_names': class_names,
        }, f, indent=2)

    cm_raw = confusion_matrix(all_true, all_preds)
    cm_norm = cm_raw.astype(float) / cm_raw.sum(axis=1, keepdims=True)
    np.savetxt(os.path.join(results_dir, 'confusion_matrix_raw.csv'), cm_raw, fmt='%d', delimiter=',')
    np.savetxt(os.path.join(results_dir, 'confusion_matrix_normalized.csv'), cm_norm, fmt='%.6f', delimiter=',')

    base_title = f'{model_name} early-fusion (frame+global, PCA={args.pca}) {num_classes}-class'
    plot_confusion_matrix(
        cm_raw, class_names, f'Confusion Matrix (counts) — {base_title}',
        os.path.join(results_dir, 'confusion_matrix_raw.png'), normalize=False,
    )
    plot_confusion_matrix(
        cm_raw, class_names, f'Confusion Matrix (normalized) — {base_title}',
        os.path.join(results_dir, 'confusion_matrix_normalized.png'), normalize=True,
    )

    plt.figure(figsize=(10, 6))
    x = np.arange(len(class_names))
    width = 0.25
    plt.bar(x - width, per_class_df['precision'], width, label='Precision')
    plt.bar(x, per_class_df['recall'], width, label='Recall')
    plt.bar(x + width, per_class_df['f1-score'], width, label='F1')
    plt.xticks(x, class_names, rotation=45)
    plt.ylim(0, 1)
    plt.title(f'Per-class metrics — {base_title}')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'per_class_metrics.png'), dpi=300)
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(fold_df['fold'], fold_df['accuracy'], marker='o')
    plt.axhline(fold_df['accuracy'].mean(), color='r', linestyle='--', label='Mean')
    plt.fill_between(fold_df['fold'], fold_df['accuracy'].mean() - fold_df['accuracy'].std(),
                     fold_df['accuracy'].mean() + fold_df['accuracy'].std(), alpha=0.2, color='red')
    plt.title(f'LOSO accuracy per fold — {base_title}')
    plt.xlabel('Fold'); plt.ylabel('Accuracy'); plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'fold_accuracy_line.png'), dpi=300)
    plt.close()

    plt.figure(figsize=(10, 6))
    order = fold_df.sort_values('speaker_id')
    colors = ['#d62728' if a < fold_df['accuracy'].mean() else '#1f77b4' for a in order['accuracy']]
    plt.bar(order['speaker_id'].astype(str), order['accuracy'], color=colors)
    plt.axhline(fold_df['accuracy'].mean(), color='k', linestyle='--', label=f"Mean = {fold_df['accuracy'].mean():.3f}")
    plt.title(f'Accuracy per speaker (LOSO) — {base_title}')
    plt.xlabel('Speaker (speaker_id)'); plt.ylabel('Accuracy')
    plt.ylim(0, 1)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'fold_accuracy_bar_per_speaker.png'), dpi=300)
    plt.close()

    print(f"[DONE] SVM/early_fusion/{num_classes}cl -> {results_dir}")


if __name__ == '__main__':
    main()
