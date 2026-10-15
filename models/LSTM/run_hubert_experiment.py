"""
LSTM / BiLSTM on HuBERT frame-level sequences (768-dim per frame) with the fixed
8/2 speaker split (same speaker_split.json as classical ML).

--num_classes 7 (default): prepared_hubert_data_7classes_backup.npz, includes
    'happiness' (comparable with handcrafted LSTM / classical ML).
--num_classes 6: prepared_hubert_data.npz, excludes 'happiness'.

Hyperparameters:
    default: matched to the handcrafted LSTM (hidden=128, lr=1e-4, no class
        weights / label smoothing / LR scheduler).
    --tuned: hidden=256, lr=3e-4, class weights + label smoothing + scheduler,
        top_features_ratio=0.85 (same for LSTM and BiLSTM).

Usage (from repo root):
    python -m models.LSTM.run_hubert_experiment --tuned --num_classes 7 --outdir <dir>
    python -m models.LSTM.run_hubert_experiment --tuned --bidirectional --num_classes 7 --outdir <dir>
    python -m models.LSTM.run_hubert_experiment --tuned --num_classes 6 --outdir <dir>
"""
import argparse
import json
import os
import pickle
import time

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from torch.utils.data import DataLoader

from .config import Config, PROJECT_ROOT, set_seeds
from .utils.sequence_dataset import SequenceDataset
from .utils.split import load_fixed_speaker_split, stratified_loso_order
from .utils.training import (
    prepare_fold_data, setup_model_and_optimizer, create_criterion,
    train_fold, validate_model, _apply_scaling, apply_feature_selection,
)
from .utils.data import calculate_fold_balance_score

HUBERT_NPZ_BY_CLASSES = {
    7: os.path.join(PROJECT_ROOT, "data", "splits", "prepared_hubert_data_7classes_backup.npz"),
    6: os.path.join(PROJECT_ROOT, "data", "splits", "prepared_hubert_data.npz"),
}


class HubertLSTMConfig(Config):
    """Same hyperparameters as the handcrafted LSTM, so only the input changes.
    NPZ_PATH is set in main() from --num_classes."""
    DATA_TYPE = 'hubert'
    HIDDEN_DIM = 128
    LEARNING_RATE = 1e-4
    USE_CLASS_WEIGHTS = False
    LABEL_SMOOTHING = 0.0
    USE_REDUCE_LR_ON_PLATEAU = False
    TOP_FEATURES_RATIO = 0.7
    MIN_DELTA = 0.005
    WEIGHT_DECAY = 1e-4


class HubertLSTMConfigTuned(Config):
    """Best HuBERT hyperparameters (config.py defaults), shared by LSTM and BiLSTM.
    NPZ_PATH is set in main() from --num_classes."""
    DATA_TYPE = 'hubert'


def plot_confusion_matrix_clean(cm_raw, class_names, title, out_path, normalize):
    """Same style as models/classicalML/plotting.py."""
    cm_raw = cm_raw.astype(float)
    if normalize:
        M = cm_raw / cm_raw.sum(axis=1, keepdims=True)
        txt = lambda v: f'{v:.2f}'
        vmax = 1.0
    else:
        M = cm_raw
        txt = lambda v: f'{int(v):d}'
        vmax = M.max()

    fig, ax = plt.subplots(figsize=(8, 6.5))
    im = ax.imshow(M, cmap='Blues', vmin=0, vmax=vmax)
    ax.set_xticks(range(len(class_names)), class_names, rotation=45, ha='right')
    ax.set_yticks(range(len(class_names)), class_names)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title(title)

    thr = vmax * 0.55
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(j, i, txt(M[i, j]), ha='center', va='center',
                    color='white' if M[i, j] > thr else 'black', fontsize=11)

    fig.colorbar(im, ax=ax, shrink=0.85)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def plot_avg_curve(histories, key, ylabel, out_path, series=None):
    """Fold-averaged metric per epoch."""
    max_epochs = max(len(h[key]) for h in histories)
    avg = []
    for epoch in range(max_epochs):
        vals = [h[key][epoch] for h in histories if len(h[key]) > epoch]
        if vals:
            avg.append(np.mean(vals))
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(avg) + 1), avg, linewidth=2, label=ylabel)
    plt.xlabel('Epoch')
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_loss_curves(histories, out_path):
    max_epochs = max(len(h['train_loss']) for h in histories)
    avg_train, avg_val = [], []
    for epoch in range(max_epochs):
        tr = [h['train_loss'][epoch] for h in histories if len(h['train_loss']) > epoch]
        va = [h['val_loss'][epoch] for h in histories if len(h['val_loss']) > epoch]
        if tr:
            avg_train.append(np.mean(tr))
        if va:
            avg_val.append(np.mean(va))
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(avg_train) + 1), avg_train, linewidth=2, label='Training Loss')
    plt.plot(range(1, len(avg_val) + 1), avg_val, linewidth=2, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def apply_binary_filter(dataset, class_names):
    """Binary task (same as classical ML): neutral/boredom -> 0,
    anger/sadness/fear/disgust -> 1, happiness dropped. Modifies dataset in place."""
    neutral_labels = {'neutral', 'boredom'}
    negative_labels = {'anger', 'sadness', 'fear', 'disgust'}
    y_names = [class_names[i] for i in dataset.y.tolist()]
    keep_mask = torch.tensor([n in neutral_labels or n in negative_labels for n in y_names])
    y_binary = torch.tensor([0 if n in neutral_labels else 1 for n in y_names], dtype=torch.long)

    dataset.X = dataset.X[keep_mask]
    dataset.y = y_binary[keep_mask]
    dataset.speaker_ids = dataset.speaker_ids[keep_mask]
    dataset.lengths = dataset.lengths[keep_mask]
    dataset.label_classes = np.array(['neutral', 'negative'])
    dataset.num_classes = 2


def fit_fold_preprocessors(npz_path, train_speakers, config):
    """Refit the fold scaler/selector on raw clean training data (for export)."""
    raw_train = SequenceDataset(npz_path, speaker_ids=train_speakers)
    train_X = raw_train.X.numpy()
    valid_rows = np.vstack([
        train_X[i, :raw_train.lengths[i], :] for i in range(len(raw_train))
    ])
    valid_rows = np.nan_to_num(valid_rows, nan=0.0, posinf=0.0, neginf=0.0)
    scaler = StandardScaler().fit(valid_rows)

    selector = None
    if getattr(config, 'FEATURE_SELECTION', False):
        scaled = scaler.transform(train_X.reshape(-1, train_X.shape[-1]))
        scaled = np.nan_to_num(scaled, nan=0.0, posinf=0.0, neginf=0.0)
        labels = np.repeat(raw_train.y.numpy(), train_X.shape[1])
        valid_mask = np.any(scaled != 0, axis=1)
        selector = SelectKBest(
            score_func=f_classif,
            k=int(train_X.shape[-1] * config.TOP_FEATURES_RATIO),
        ).fit(scaled[valid_mask], labels[valid_mask])
    return scaler, selector


def save_clean_fold_artifacts(root, fold_number, model, scaler, selector,
                              config, classes, raw_speaker):
    """Save model + preprocessing for the mismatched-condition evaluation."""
    fold_dir = os.path.join(root, f'fold_{fold_number}')
    os.makedirs(fold_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(fold_dir, 'model.pt'))
    with open(os.path.join(fold_dir, 'preprocessing.pkl'), 'wb') as handle:
        pickle.dump({'scaler': scaler, 'selector': selector}, handle)
    metadata = {
        'checkpoint_type': 'clean_baseline',
        'fold': fold_number,
        'raw_speaker_id': int(raw_speaker),
        'classes': list(classes),
        'input_dim': model.lstm.input_size,
        'hidden_dim': config.HIDDEN_DIM,
        'num_layers': config.NUM_LAYERS,
        'dropout': config.DROPOUT,
        'bidirectional': config.BIDIRECTIONAL,
        'learning_rate': config.LEARNING_RATE,
        'patience': config.PATIENCE,
        'top_features_ratio': config.TOP_FEATURES_RATIO,
        'use_class_weights': config.USE_CLASS_WEIGHTS,
        'label_smoothing': config.LABEL_SMOOTHING,
        'seed': 42,
    }
    with open(os.path.join(fold_dir, 'metadata.json'), 'w') as handle:
        json.dump(metadata, handle, indent=2)


def plot_fold_accuracy_bar_per_speaker(fold_df, model_label, scenario_label, out_path):
    """One bar per validation speaker (red = below mean), dashed mean line."""
    plt.figure(figsize=(10, 6))
    order = fold_df.sort_values('validation_speaker')
    mean_acc = fold_df['accuracy'].mean()
    colors = ['#d62728' if a < mean_acc else '#1f77b4' for a in order['accuracy']]
    plt.bar(order['validation_speaker'].astype(str), order['accuracy'], color=colors)
    plt.axhline(mean_acc, color='k', linestyle='--', label=f"Mean = {mean_acc:.3f}")
    plt.title(f'Accuracy per Speaker (LOSO) — {model_label} HuBERT {scenario_label}')
    plt.xlabel('Speaker (speaker_id)'); plt.ylabel('Accuracy')
    plt.ylim(0, 1)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def _raw_id(dataset, encoded):
    return int(''.join(ch for ch in str(dataset.speaker_classes[encoded]) if ch.isdigit()))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bidirectional', action='store_true')
    parser.add_argument('--tuned', action='store_true',
                        help='Use the tuned hyperparameters instead of matched-to-handcrafted')
    parser.add_argument('--num_classes', type=int, choices=[6, 7], default=7,
                        help='7 (with happiness, default) or 6 (without happiness)')
    parser.add_argument('--binary', action='store_true',
                        help='Binary neutral/boredom (0) vs anger/sadness/fear/disgust (1); '
                             'overrides --num_classes')
    parser.add_argument('--outdir', required=True)
    parser.add_argument('--save-fold-artifacts', action='store_true',
                        help='Save per-fold model.pt, scaler, selector and metadata')
    parser.add_argument('--checkpoint-root', default=None,
                        help='Optional directory for exact clean baseline artifacts per fold')
    parser.add_argument('--hidden_dim', type=int, default=None,
                        help='Override HIDDEN_DIM')
    parser.add_argument('--batch_size', type=int, default=None,
                        help='Override BATCH_SIZE')
    args = parser.parse_args()

    set_seeds()
    config = HubertLSTMConfigTuned() if args.tuned else HubertLSTMConfig()
    config.BIDIRECTIONAL = args.bidirectional
    config.RESULTS_DIR = args.outdir
    config.NPZ_PATH = HUBERT_NPZ_BY_CLASSES[args.num_classes]
    if args.hidden_dim is not None:
        config.HIDDEN_DIM = args.hidden_dim
    if args.batch_size is not None:
        config.BATCH_SIZE = args.batch_size
    os.makedirs(args.outdir, exist_ok=True)
    checkpoint_root = args.checkpoint_root or os.path.join(args.outdir, 'fold_artifacts')
    if args.save_fold_artifacts:
        os.makedirs(checkpoint_root, exist_ok=True)

    model_label = 'BiLSTM' if args.bidirectional else 'LSTM'
    hp_label = 'tuned' if args.tuned else 'matched'
    scenario_label = 'binary' if args.binary else f'{args.num_classes}class'
    print(f"=== {model_label} | HuBERT frame-level ({scenario_label}) | hyperparams={hp_label} | "
          f"hidden_dim={config.HIDDEN_DIM} batch_size={config.BATCH_SIZE} "
          f"lr_reduce_patience={getattr(config, 'LR_REDUCE_PATIENCE', None)} | device={config.device} ===")

    dataset = SequenceDataset(config.NPZ_PATH)
    class_names = list(dataset.label_classes)
    original_class_names = class_names  # needed again for the held-out test set
    if not args.binary:
        assert len(class_names) == args.num_classes, (
            f"--num_classes {args.num_classes} requested but {config.NPZ_PATH} "
            f"has {len(class_names)} classes: {class_names}"
        )
    print(f"Dataset: {len(dataset)} utterances, {dataset.X.shape[2]}D features, "
          f"{len(class_names)} classes: {class_names}")

    if args.binary:
        apply_binary_filter(dataset, class_names)
        class_names = ['neutral', 'negative']
        print(f"Binary filtering: kept {len(dataset)} utterances")

    train_speakers, test_speakers = load_fixed_speaker_split(dataset, config.SPEAKER_SPLIT_JSON)
    raw_train_val = sorted(_raw_id(dataset, s) for s in train_speakers)
    raw_test = sorted(_raw_id(dataset, s) for s in test_speakers)
    print(f"Train/Val speakers (raw): {raw_train_val} | Test speakers (raw): {raw_test}")

    loso_order = stratified_loso_order(train_speakers, dataset)

    histories = []
    all_true, all_pred = [], []
    fold_rows = []

    start_time = time.time()
    for fold_idx, val_speaker in enumerate(loso_order):
        current_train_speakers = [s for s in loso_order if s != val_speaker]
        raw_val_speaker = _raw_id(dataset, val_speaker)

        scaler_artifact, selector_artifact = (None, None)
        if args.checkpoint_root:
            scaler_artifact, selector_artifact = fit_fold_preprocessors(
                config.NPZ_PATH, current_train_speakers, config
            )

        train_dataset, val_dataset, train_loader, val_loader = prepare_fold_data(
            current_train_speakers, val_speaker, dataset, config
        )
        if args.save_fold_artifacts:
            fold_dir = os.path.join(checkpoint_root, f'fold_{fold_idx + 1}')
            os.makedirs(fold_dir, exist_ok=True)
            with open(os.path.join(fold_dir, 'metadata.json'), 'w') as handle:
                json.dump({
                    'fold': fold_idx + 1,
                    'raw_validation_speaker': int(raw_val_speaker),
                    'train_speakers': [_raw_id(dataset, s) for s in current_train_speakers],
                    'classes': list(dataset.label_classes),
                    'input_dim': train_dataset.input_dim,
                    'hidden_dim': config.HIDDEN_DIM,
                    'num_layers': config.NUM_LAYERS,
                    'dropout': config.DROPOUT,
                    'bidirectional': config.BIDIRECTIONAL,
                    'feature_selection': getattr(config, 'FEATURE_SELECTION', False),
                    'top_features_ratio': getattr(config, 'TOP_FEATURES_RATIO', None),
                }, handle, indent=2)
            with open(os.path.join(fold_dir, 'scaler.pkl'), 'wb') as handle:
                pickle.dump(train_dataset.scaler, handle)
            if train_dataset.selector is not None:
                with open(os.path.join(fold_dir, 'selector.pkl'), 'wb') as handle:
                    pickle.dump(train_dataset.selector, handle)

        model, optimizer = setup_model_and_optimizer(
            None, train_dataset.num_classes, config, train_dataset=train_dataset
        )
        criterion = create_criterion(train_dataset, train_dataset.num_classes, config.device, config)

        print(f"\n--- Fold {fold_idx + 1}/{len(loso_order)} | Validation Speaker: {raw_val_speaker} ---")
        hist, best_epoch, total_epochs = train_fold(model, train_loader, val_loader, optimizer, criterion, config)
        val_loss, val_acc, val_f1, val_preds, val_true, val_probs = validate_model(
            model, val_loader, criterion, config.device
        )
        if args.checkpoint_root:
            save_clean_fold_artifacts(
                args.checkpoint_root, fold_idx + 1, model,
                scaler_artifact, selector_artifact, config,
                class_names, raw_val_speaker,
            )
        balance_score = calculate_fold_balance_score(current_train_speakers, val_speaker, dataset)

        fold_rows.append({
            'fold': fold_idx + 1,
            'validation_speaker': raw_val_speaker,
            'accuracy': val_acc,
            'f1_macro': val_f1,
            'best_epoch': best_epoch,
            'total_epochs': total_epochs,
            'balance_score': balance_score,
        })
        all_true.extend(val_true)
        all_pred.extend(val_preds)
        histories.append(hist)
        if args.save_fold_artifacts:
            fold_dir = os.path.join(checkpoint_root, f'fold_{fold_idx + 1}')
            torch.save(model.state_dict(), os.path.join(fold_dir, 'model.pt'))
        print(f"Fold {fold_idx + 1} result: acc={val_acc:.4f} f1_macro={val_f1:.4f} balance={balance_score:.4f}")

    elapsed_time = time.time() - start_time
    accuracy = accuracy_score(all_true, all_pred)
    f1_macro = f1_score(all_true, all_pred, average='macro', zero_division=0)
    f1_weighted = f1_score(all_true, all_pred, average='weighted', zero_division=0)
    fold_df = pd.DataFrame(fold_rows)

    print(f"\nCV: Accuracy={accuracy:.4f} F1macro={f1_macro:.4f} F1weighted={f1_weighted:.4f} time={elapsed_time:.1f}s")

    fold_df.to_csv(os.path.join(args.outdir, 'fold_results.csv'), index=False)

    class_report = classification_report(all_true, all_pred, target_names=class_names, digits=4, zero_division=0)
    with open(os.path.join(args.outdir, 'classification_report.txt'), 'w') as f:
        f.write(f"Experiment: HUBERT-{model_label} ({hp_label} hyperparams) | {scenario_label}\n")
        f.write("=" * 60 + "\n")
        f.write(f"Overall Accuracy: {accuracy:.4f}\n")
        f.write(f"F1-Macro: {f1_macro:.4f}\n")
        f.write(f"F1-Weighted: {f1_weighted:.4f}\n")
        f.write(f"Execution time: {elapsed_time:.2f} seconds\n")
        f.write(f"Fold Accuracy: {fold_df['accuracy'].mean():.4f} +- {fold_df['accuracy'].std():.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(class_report)

    report_dict = classification_report(all_true, all_pred, target_names=class_names, output_dict=True, zero_division=0)
    per_class_df = pd.DataFrame(report_dict).transpose().iloc[:-3, :]
    per_class_df.index.name = 'class'
    per_class_df.to_csv(os.path.join(args.outdir, 'per_class_metrics.csv'))

    # CV confusion matrices
    cm_raw = confusion_matrix(all_true, all_pred)
    cm_norm = cm_raw.astype(float) / cm_raw.sum(axis=1, keepdims=True)
    np.savetxt(os.path.join(args.outdir, 'confusion_matrix_raw.csv'), cm_raw, fmt='%d', delimiter=',')
    np.savetxt(os.path.join(args.outdir, 'confusion_matrix_normalized.csv'), cm_norm, fmt='%.6f', delimiter=',')
    plot_confusion_matrix_clean(cm_raw, class_names, '', os.path.join(args.outdir, 'confusion_matrix_raw.png'), normalize=False)
    plot_confusion_matrix_clean(cm_raw, class_names, '', os.path.join(args.outdir, 'confusion_matrix_normalized.png'), normalize=True)

    plot_loss_curves(histories, os.path.join(args.outdir, 'training_validation_loss.png'))
    plot_avg_curve(histories, 'val_f1', 'Validation F1-Macro', os.path.join(args.outdir, 'validation_f1.png'))
    plot_fold_accuracy_bar_per_speaker(fold_df, model_label, scenario_label,
                                       os.path.join(args.outdir, 'fold_accuracy_bar_per_speaker.png'))

    # ================================
    # Final evaluation on held-out test speakers
    # ================================
    print(f"\nTraining final model on all train/val speakers, "
          f"evaluating on held-out test speakers {raw_test}...")

    train_dataset = SequenceDataset(config.NPZ_PATH, speaker_ids=train_speakers)
    test_dataset = SequenceDataset(config.NPZ_PATH, speaker_ids=test_speakers)
    if args.binary:
        apply_binary_filter(train_dataset, original_class_names)
        apply_binary_filter(test_dataset, original_class_names)

    scaler = StandardScaler()
    train_X = train_dataset.X.numpy()
    train_lengths = train_dataset.lengths.numpy()
    valid_rows = np.vstack([train_X[i, :train_lengths[i], :] for i in range(len(train_X))])
    valid_rows = np.nan_to_num(valid_rows, nan=0.0, posinf=0.0, neginf=0.0)
    scaler.fit(valid_rows)
    _apply_scaling(train_dataset, scaler)
    _apply_scaling(test_dataset, scaler)

    if getattr(config, 'FEATURE_SELECTION', False):
        train_dataset, test_dataset = apply_feature_selection(train_dataset, test_dataset, config)

    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True,
                              pin_memory=config.pin_memory, num_workers=config.NUM_WORKERS)
    test_loader = DataLoader(test_dataset, batch_size=getattr(config, 'VAL_BATCH_SIZE', 8), shuffle=False,
                             pin_memory=config.pin_memory, num_workers=config.NUM_WORKERS)

    final_model, final_optimizer = setup_model_and_optimizer(None, train_dataset.num_classes, config, train_dataset=train_dataset)
    final_criterion = create_criterion(train_dataset, train_dataset.num_classes, config.device, config)

    final_start = time.time()
    train_fold(final_model, train_loader, test_loader, final_optimizer, final_criterion, config)
    test_loss, test_acc, test_f1, test_preds, test_true, test_probs = validate_model(
        final_model, test_loader, final_criterion, config.device
    )
    final_elapsed = time.time() - final_start
    test_f1_weighted = f1_score(test_true, test_preds, average='weighted', zero_division=0)

    print(f"TEST: Accuracy={test_acc:.4f} F1macro={test_f1:.4f} F1weighted={test_f1_weighted:.4f} time={final_elapsed:.1f}s")

    final_report = classification_report(test_true, test_preds, target_names=class_names, digits=4, zero_division=0)
    with open(os.path.join(args.outdir, 'final_test_report.txt'), 'w') as f:
        f.write(f"Experiment: HUBERT-{model_label} ({hp_label} hyperparams) | {scenario_label} | Held-out TEST speakers {raw_test}\n")
        f.write("=" * 60 + "\n")
        f.write(f"Test Accuracy: {test_acc:.4f}\n")
        f.write(f"F1-Macro: {test_f1:.4f}\n")
        f.write(f"F1-Weighted: {test_f1_weighted:.4f}\n")
        f.write(f"Execution time: {final_elapsed:.2f} seconds\n\n")
        f.write("Classification Report:\n")
        f.write(final_report)

    final_cm_raw = confusion_matrix(test_true, test_preds)
    final_cm_norm = final_cm_raw.astype(float) / final_cm_raw.sum(axis=1, keepdims=True)
    np.savetxt(os.path.join(args.outdir, 'final_test_confusion_matrix_raw.csv'), final_cm_raw, fmt='%d', delimiter=',')
    np.savetxt(os.path.join(args.outdir, 'final_test_confusion_matrix_normalized.csv'), final_cm_norm, fmt='%.6f', delimiter=',')
    plot_confusion_matrix_clean(final_cm_raw, class_names, '', os.path.join(args.outdir, 'final_test_confusion_matrix_raw.png'), normalize=False)
    plot_confusion_matrix_clean(final_cm_raw, class_names, '', os.path.join(args.outdir, 'final_test_confusion_matrix_normalized.png'), normalize=True)

    with open(os.path.join(args.outdir, 'metrics.json'), 'w') as f:
        json.dump({
            'experiment': f"hubert_{model_label.lower()}_{hp_label}_{scenario_label}",
            'model': model_label,
            'hyperparams': hp_label,
            'feature_type': 'hubert',
            'num_classes': len(class_names),
            'accuracy': accuracy,
            'f1_macro': f1_macro,
            'f1_weighted': f1_weighted,
            'fold_acc_mean': float(fold_df['accuracy'].mean()),
            'fold_acc_std': float(fold_df['accuracy'].std()),
            'n_features': dataset.X.shape[2],
            'n_samples': len(train_dataset),
            'n_speakers': len(train_speakers),
            'execution_time_sec': elapsed_time,
            'class_names': class_names,
            'final_test_accuracy': test_acc,
            'final_test_f1_macro': test_f1,
            'final_test_f1_weighted': test_f1_weighted,
            'final_test_n_samples': len(test_dataset),
            'final_test_speakers': raw_test,
            'train_val_speakers': raw_train_val,
            'config': {
                'bidirectional': config.BIDIRECTIONAL,
                'hidden_dim': config.HIDDEN_DIM,
                'num_layers': config.NUM_LAYERS,
                'dropout': config.DROPOUT,
                'learning_rate': config.LEARNING_RATE,
                'batch_size': config.BATCH_SIZE,
                'max_epochs': config.MAX_EPOCHS,
                'patience': config.PATIENCE,
                'min_delta': config.MIN_DELTA,
                'max_grad_norm': config.MAX_GRAD_NORM,
                'feature_selection_method': config.FEATURE_SELECTION_METHOD,
                'top_features_ratio': config.TOP_FEATURES_RATIO,
                'use_class_weights': config.USE_CLASS_WEIGHTS,
                'label_smoothing': config.LABEL_SMOOTHING,
                'use_reduce_lr_on_plateau': config.USE_REDUCE_LR_ON_PLATEAU,
                'weight_decay': config.WEIGHT_DECAY,
                'lr_reduce_patience': getattr(config, 'LR_REDUCE_PATIENCE', None),
            },
        }, f, indent=2)

    print(f"\n[DONE] {model_label}/hubert/{scenario_label} -> {args.outdir}")


if __name__ == '__main__':
    main()
