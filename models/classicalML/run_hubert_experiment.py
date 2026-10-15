"""
One HuBERT + classical-ML experiment (LOSO CV on frame-level embeddings), with
the same outputs as run_experiment.py plus a final evaluation on the held-out
test speakers (12, 13).

Same pipeline/hyperparameters as the handcrafted experiments; only the input
changes (768-dim HuBERT frames). RF uses n_jobs=-1 (speed only, same model).

Usage (from repo root):
    python models/classicalML/run_hubert_experiment.py --model SVM --classes 7 --outdir <dir>
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

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import (
    accuracy_score, f1_score, confusion_matrix, classification_report,
)
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
HUBERT_EMBEDDINGS_DIR = os.path.join(ROOT, 'data', 'features', 'hubert', 'frame_level')
HUBERT_LABELS_FILE = os.path.join(ROOT, 'data', 'features', 'hubert', 'hubert_labels.csv')
SPEAKER_SPLIT_JSON = os.path.join(ROOT, 'data', 'splits', 'speaker_split.json')


def build_classifier(model_name):
    if model_name == 'RF':
        return RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    if model_name == 'SVM':
        return SVC(kernel='rbf', probability=False, random_state=42)
    if model_name == 'NB':
        return GaussianNB()
    if model_name == 'MLP':
        return MLPClassifier(hidden_layer_sizes=(100,), max_iter=300, random_state=42)
    raise ValueError(f"Unsupported model: {model_name}")


def load_hubert_frame_data(embeddings_dir, labels_file):
    """All frame-level embeddings in one DataFrame (vectorised; ~74k frames)."""
    labels_df = pd.read_csv(labels_file)
    # speaker_id may be 'speaker_03' or 3; normalise to the raw int id
    labels_df['speaker_int'] = labels_df['speaker_id'].astype(str).str.replace('speaker_', '', regex=False).astype(int)
    label_map = labels_df.set_index('filename')[['speaker_int', 'emotion']].to_dict('index')

    feats, speakers, emotions, filenames, frame_idxs = [], [], [], [], []
    for fname in sorted(os.listdir(embeddings_dir)):
        if not fname.endswith('.npy'):
            continue
        base = fname.rsplit('_', 1)[-1].replace('.npy', '')
        wav = f"{base}.wav"
        info = label_map.get(wav)
        if info is None:
            raise KeyError(f"No label row for {wav} (from embedding file {fname})")
        emb = np.load(os.path.join(embeddings_dir, fname))
        n = emb.shape[0]
        feats.append(emb)
        speakers.append(np.full(n, info['speaker_int'], dtype=int))
        emotions.append(np.full(n, info['emotion'], dtype=object))
        filenames.append(np.full(n, wav, dtype=object))
        frame_idxs.append(np.arange(n))

    X = np.concatenate(feats, axis=0)
    feature_cols = [f'feature_{i}' for i in range(X.shape[1])]
    df = pd.DataFrame(X, columns=feature_cols)
    df['speaker_id'] = np.concatenate(speakers)
    df['emotion'] = np.concatenate(emotions)
    df['filename'] = np.concatenate(filenames)
    df['frame_index'] = np.concatenate(frame_idxs)
    return df, feature_cols


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True, choices=['RF', 'SVM', 'NB', 'MLP'])
    parser.add_argument('--classes', required=True, type=int, choices=[2, 7])
    parser.add_argument('--outdir', required=True)
    args = parser.parse_args()

    model_name = args.model
    num_classes = args.classes
    results_dir = args.outdir
    os.makedirs(results_dir, exist_ok=True)

    df, feature_cols = load_hubert_frame_data(HUBERT_EMBEDDINGS_DIR, HUBERT_LABELS_FILE)
    print(f"[{model_name}/hubert-frame/{num_classes}cl] Loaded {len(df)} frames "
          f"({df['filename'].nunique()} files, {df['speaker_id'].nunique()} speakers)")

    if num_classes == 2:
        neutral_labels = ['neutral', 'boredom']
        negative_labels = ['anger', 'sadness', 'fear', 'disgust']
        df = df[~df['emotion'].isin(['happiness'])]
        df = df[df['emotion'].isin(neutral_labels + negative_labels)]
        df['emotion_binary'] = df['emotion'].apply(lambda x: 0 if x in neutral_labels else 1)
        y_all = df['emotion_binary']
    else:
        y_all = df['emotion']

    # Shared fixed speaker split
    with open(SPEAKER_SPLIT_JSON) as f:
        split = json.load(f)
    TRAIN_VAL_SPEAKERS = set(split['train_val_speakers'])
    TEST_SPEAKERS = set(split['test_speakers'])

    trainval_mask = df['speaker_id'].isin(TRAIN_VAL_SPEAKERS)
    test_mask = df['speaker_id'].isin(TEST_SPEAKERS)
    df_tv, y_tv = df[trainval_mask].reset_index(drop=True), y_all[trainval_mask].reset_index(drop=True)
    df_test, y_test = df[test_mask].reset_index(drop=True), y_all[test_mask].reset_index(drop=True)

    X_tv = df_tv[feature_cols]
    X_test = df_test[feature_cols]
    groups = df_tv['speaker_id']
    speaker_ids = df_tv['speaker_id']

    if num_classes == 2:
        y_tv_encoded = y_tv.to_numpy()
        y_test_encoded = y_test.to_numpy()
        class_names = ['neutral', 'negative']
    else:
        le = LabelEncoder()
        y_tv_encoded = le.fit_transform(y_tv)
        y_test_encoded = le.transform(y_test)
        class_names = list(le.classes_)

    print(f" Features: {len(feature_cols)} | Train/Val samples: {len(X_tv)} "
          f"(speakers {sorted(TRAIN_VAL_SPEAKERS)}) | Test samples: {len(X_test)} "
          f"(speakers {sorted(TEST_SPEAKERS)})")
    print(f" Classes: {class_names}")

    classifier = build_classifier(model_name)

    X_tv = X_tv.reset_index(drop=True)
    y_tv_encoded = pd.Series(y_tv_encoded).reset_index(drop=True)
    groups = groups.reset_index(drop=True)
    speaker_ids = speaker_ids.reset_index(drop=True)

    logo = LeaveOneGroupOut()
    n_folds = logo.get_n_splits(groups=groups)

    all_preds, all_true, all_speakers = [], [], []
    fold_rows = []

    start_time = time.time()
    for fold, (train_idx, val_idx) in enumerate(logo.split(X_tv, y_tv_encoded, groups)):
        X_train_fold, X_val_fold = X_tv.iloc[train_idx], X_tv.iloc[val_idx]
        y_train_fold, y_val_fold = y_tv_encoded[train_idx], y_tv_encoded[val_idx]
        val_speaker = speaker_ids.iloc[val_idx].unique()[0]

        train_class_counts = Counter(y_train_fold)
        min_class_count = min(train_class_counts.values())
        k_neighbors = min(5, max(1, min_class_count - 1))

        pipeline_steps = [
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler()),
        ]
        if min_class_count >= 2:
            pipeline_steps.append(('smote', SMOTE(random_state=42, k_neighbors=k_neighbors)))
        pipeline_steps.append(('classifier', classifier))
        pipeline = ImbPipeline(pipeline_steps)

        fold_start = time.time()
        pipeline.fit(X_train_fold, y_train_fold)
        preds = pipeline.predict(X_val_fold)
        fold_elapsed = time.time() - fold_start

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
        print(f" Fold {fold + 1}/{n_folds} | speaker={val_speaker} | acc={fold_acc:.4f} | {fold_elapsed:.1f}s")

    elapsed_time = time.time() - start_time
    accuracy = accuracy_score(all_true, all_preds)
    f1_macro = f1_score(all_true, all_preds, average='macro', zero_division=0)
    f1_weighted = f1_score(all_true, all_preds, average='weighted', zero_division=0)
    fold_df = pd.DataFrame(fold_rows)

    print(f"CV: Accuracy={accuracy:.4f} F1macro={f1_macro:.4f} F1weighted={f1_weighted:.4f} time={elapsed_time:.1f}s")

    pd.DataFrame({
        'speaker_id': all_speakers,
        'true_label': all_true,
        'pred_label': all_preds,
    }).to_csv(os.path.join(results_dir, 'predictions.csv'), index=False)

    fold_df.to_csv(os.path.join(results_dir, 'fold_results.csv'), index=False)

    class_report = classification_report(all_true, all_preds, target_names=class_names, digits=4, zero_division=0)
    with open(os.path.join(results_dir, 'classification_report.txt'), 'w') as f:
        f.write(f"Experiment: HUBERT-FRAME | {model_name} | {num_classes}-class\n")
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

    # CV confusion matrices
    cm_raw = confusion_matrix(all_true, all_preds)
    cm_norm = cm_raw.astype(float) / cm_raw.sum(axis=1, keepdims=True)
    np.savetxt(os.path.join(results_dir, 'confusion_matrix_raw.csv'), cm_raw, fmt='%d', delimiter=',')
    np.savetxt(os.path.join(results_dir, 'confusion_matrix_normalized.csv'), cm_norm, fmt='%.6f', delimiter=',')

    base_title = f'{model_name} HuBERT frame-level {num_classes}-class'
    plot_confusion_matrix(
        cm_raw, class_names, f'Confusion Matrix (counts) — {base_title}',
        os.path.join(results_dir, 'confusion_matrix_raw.png'), normalize=False,
    )
    # No title (captioned in the thesis text)
    plot_confusion_matrix(
        cm_raw, class_names, '',
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

    # ================================
    # Final evaluation on held-out test speakers
    # ================================
    print(f"Training final model on all train/val speakers, "
          f"evaluating on held-out test speakers {sorted(TEST_SPEAKERS)}...")
    final_min_class_count = min(Counter(y_tv_encoded).values())
    final_k_neighbors = min(5, max(1, final_min_class_count - 1))
    final_steps = [
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler()),
    ]
    if final_min_class_count >= 2:
        final_steps.append(('smote', SMOTE(random_state=42, k_neighbors=final_k_neighbors)))
    final_steps.append(('classifier', build_classifier(model_name)))
    final_pipeline = ImbPipeline(final_steps)

    final_start = time.time()
    final_pipeline.fit(X_tv, y_tv_encoded)
    final_preds = final_pipeline.predict(X_test)
    final_elapsed = time.time() - final_start

    final_accuracy = accuracy_score(y_test_encoded, final_preds)
    final_f1_macro = f1_score(y_test_encoded, final_preds, average='macro', zero_division=0)
    final_f1_weighted = f1_score(y_test_encoded, final_preds, average='weighted', zero_division=0)
    final_report = classification_report(y_test_encoded, final_preds, target_names=class_names, digits=4, zero_division=0)

    print(f"TEST: Accuracy={final_accuracy:.4f} F1macro={final_f1_macro:.4f} F1weighted={final_f1_weighted:.4f} time={final_elapsed:.1f}s")

    with open(os.path.join(results_dir, 'final_test_report.txt'), 'w') as f:
        f.write(f"Experiment: HUBERT-FRAME | {model_name} | {num_classes}-class | "
                f"Held-out TEST speakers {sorted(TEST_SPEAKERS)}\n")
        f.write("=" * 60 + "\n")
        f.write(f"Test Accuracy: {final_accuracy:.4f}\n")
        f.write(f"F1-Macro: {final_f1_macro:.4f}\n")
        f.write(f"F1-Weighted: {final_f1_weighted:.4f}\n")
        f.write(f"Execution time: {final_elapsed:.2f} seconds\n\n")
        f.write("Classification Report:\n")
        f.write(final_report)

    final_cm_raw = confusion_matrix(y_test_encoded, final_preds)
    final_cm_norm = final_cm_raw.astype(float) / final_cm_raw.sum(axis=1, keepdims=True)
    np.savetxt(os.path.join(results_dir, 'final_test_confusion_matrix_raw.csv'), final_cm_raw, fmt='%d', delimiter=',')
    np.savetxt(os.path.join(results_dir, 'final_test_confusion_matrix_normalized.csv'), final_cm_norm, fmt='%.6f', delimiter=',')
    plot_confusion_matrix(
        final_cm_raw, class_names, '',
        os.path.join(results_dir, 'final_test_confusion_matrix_raw.png'), normalize=False,
    )
    plot_confusion_matrix(
        final_cm_raw, class_names, '',
        os.path.join(results_dir, 'final_test_confusion_matrix_normalized.png'), normalize=True,
    )

    with open(os.path.join(results_dir, 'metrics.json'), 'w') as f:
        json.dump({
            'experiment': f"hubert_frame_{model_name.lower()}_{num_classes}class",
            'model': model_name,
            'feature_type': 'hubert_frame',
            'num_classes': num_classes,
            'accuracy': accuracy,
            'f1_macro': f1_macro,
            'f1_weighted': f1_weighted,
            'fold_acc_mean': float(fold_df['accuracy'].mean()),
            'fold_acc_std': float(fold_df['accuracy'].std()),
            'n_features': len(feature_cols),
            'n_samples': len(X_tv),
            'n_speakers': int(groups.nunique()),
            'execution_time_sec': elapsed_time,
            'class_names': class_names,
            'final_test_accuracy': final_accuracy,
            'final_test_f1_macro': final_f1_macro,
            'final_test_f1_weighted': final_f1_weighted,
            'final_test_n_samples': len(X_test),
            'final_test_speakers': sorted(TEST_SPEAKERS),
            'train_val_speakers': sorted(TRAIN_VAL_SPEAKERS),
        }, f, indent=2)

    print(f"[DONE] {model_name}/hubert-frame/{num_classes}cl -> {results_dir}")


if __name__ == '__main__':
    main()
