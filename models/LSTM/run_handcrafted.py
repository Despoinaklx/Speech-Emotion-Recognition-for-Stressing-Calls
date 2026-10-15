"""
8-fold stratified LOSO LSTM on handcrafted frame-level features.

Requires data/splits/prepared_data.npz (models/LSTM/utils/handcrafted_data_preparation.py).

Usage (from repo root):
    python -m models.LSTM.run_handcrafted
"""

import os

from sklearn.metrics import accuracy_score, f1_score, classification_report
import torch.nn as nn
from .config import Config, PROJECT_ROOT, set_seeds
from .utils.sequence_dataset import SequenceDataset
from .utils import (
    print_dataset_info, find_best_speaker_split, load_fixed_speaker_split, stratified_loso_order,
    prepare_fold_data, setup_model_and_optimizer,
    train_fold, validate_model, calculate_fold_balance_score,
    plot_training_curves, plot_confusion_matrix, plot_fold_performance,
    create_results_dict, save_results_to_files, print_final_results
)


class HandcraftedConfig(Config):
    """Hyperparameters of the reported handcrafted LSTM run."""
    DATA_TYPE = 'handcrafted'
    NPZ_PATH = os.path.join(PROJECT_ROOT, "data", "splits", "prepared_data.npz")
    RESULTS_DIR = os.path.join(PROJECT_ROOT, "results", "LSTM", "results_handcrafted_features")
    BIDIRECTIONAL = False
    HIDDEN_DIM = 128
    LEARNING_RATE = 1e-4
    MIN_DELTA = 0.005
    TOP_FEATURES_RATIO = 0.7
    USE_CLASS_WEIGHTS = False
    LABEL_SMOOTHING = 0.0
    USE_REDUCE_LR_ON_PLATEAU = False


def main():
    config = HandcraftedConfig()
    set_seeds()
    os.makedirs(config.RESULTS_DIR, exist_ok=True)

    dataset = SequenceDataset(config.NPZ_PATH)
    num_classes = len(dataset.label_classes)
    speakers, _ = print_dataset_info(dataset)
    print(dataset.label_classes)
    print(f"\nDataset: {len(dataset)} samples, {dataset.X.shape[2]}D features, {num_classes} classes, {len(speakers)} speakers")

    # 1. Speaker split: fixed shared split by default, SSE search otherwise
    if getattr(config, 'USE_FIXED_SPEAKER_SPLIT', True):
        train_speakers, test_speakers = load_fixed_speaker_split(dataset, config.SPEAKER_SPLIT_JSON)
    else:
        train_speakers, test_speakers = find_best_speaker_split(dataset, config.TEST_SPEAKERS_PERCENTAGE)
    print(f"Speaker split: {len(train_speakers)} train speakers, {len(test_speakers)} held-out speakers")

    # 2. Balanced LOSO fold order
    loso_order = stratified_loso_order(train_speakers, dataset)

    # 3. 8-fold LOSO (7 train speakers, 1 validation speaker per fold)
    predictions, ground_truth, probabilities, histories, results = execute_loso_cv(loso_order, dataset, dataset.X.shape[2], num_classes, config)

    # 4. Metrics, plots, files
    metrics = compute_metrics(ground_truth, predictions, dataset.label_classes, probabilities)
    save_and_visualize_results(config, dataset, train_speakers, test_speakers, loso_order, speakers,
                               histories, results, metrics, predictions, ground_truth)

    print(f"\nDone. Accuracy: {metrics['accuracy']:.4f}, F1: {metrics['f1_macro']:.4f}")


def execute_loso_cv(loso_order, dataset, input_dim, num_classes, config):
    """
    Run LOSO CV in the given speaker order.
    Returns (preds, labels, probs, histories, per-fold results).
    """
    all_preds, all_true, all_probs, histories, results = [], [], [], [], []

    for fold_idx, val_speaker in enumerate(loso_order):
        print(f"\nFold {fold_idx+1}/{len(loso_order)} | Validation Speaker: {val_speaker}")

        train_speakers = [s for s in loso_order if s != val_speaker]
        balance_score = calculate_fold_balance_score(train_speakers, val_speaker, dataset)

        train_dataset, val_dataset, train_loader, val_loader = prepare_fold_data(
            train_speakers, val_speaker, dataset, config)
        input_dim = train_dataset.X.shape[2]
        model, optimizer = setup_model_and_optimizer(input_dim, num_classes, config, train_dataset)
        criterion = nn.CrossEntropyLoss()

        history, best_epoch, total_epochs = train_fold(model, train_loader, val_loader, optimizer, criterion, config)

        _, _, _, preds, true, probs = validate_model(model, val_loader, criterion, config.device)
        all_preds.extend(preds)
        all_true.extend(true)
        all_probs.extend(probs)

        acc = accuracy_score(true, preds) if true else 0.0
        f1 = f1_score(true, preds, average='macro', zero_division=0) if true else 0.0

        results.append({
            'fold': fold_idx + 1,
            'validation_speaker': int(val_speaker),
            'final_accuracy': acc,
            'final_f1_macro': f1,
            'best_epoch': best_epoch,
            'total_epochs': total_epochs,
            'balance_score': balance_score
        })
        histories.append(history)

        print(f"  Fold result: Accuracy: {acc:.4f} | F1: {f1:.4f} | Balance Score: {balance_score:.4f}")

    return all_preds, all_true, all_probs, histories, results


def compute_metrics(y_true, y_pred, classes, y_proba=None):
    """Pooled metrics over all folds."""
    if not y_true:
        return {'accuracy': 0.0, 'f1_macro': 0.0, 'f1_weighted': 0.0, 'classification_report': '', 'y_true': [], 'y_pred': [], 'y_proba': []}

    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'f1_macro': f1_score(y_true, y_pred, average='macro', zero_division=0),
        'f1_weighted': f1_score(y_true, y_pred, average='weighted', zero_division=0),
        'classification_report': classification_report(y_true, y_pred, target_names=classes, zero_division=0),
        'y_true': y_true,
        'y_pred': y_pred,
        'y_proba': y_proba if y_proba is not None else []
    }


def save_and_visualize_results(config, dataset, train_speakers, test_speakers, loso_order, speakers,
                               histories, results, metrics, predictions, ground_truth):
    """Write plots and result files, then print the summary."""
    print("\nCreating plots...")

    plot_training_curves(histories, config.RESULTS_DIR)
    plot_confusion_matrix(ground_truth, predictions, dataset.label_classes, config.RESULTS_DIR, " (8-Fold LOSO)")
    plot_fold_performance(results, config.RESULTS_DIR)

    dataset_info = {
        'num_samples': len(dataset),
        'input_dim': dataset.X.shape[2],
        'num_classes': len(dataset.label_classes),
        'total_speakers': len(speakers),
        'train_speakers': train_speakers,
        'test_speakers': test_speakers,
        'stratified_loso_order': loso_order,
        'class_names': dataset.label_classes.tolist(),
        'split_sse': None
    }

    results_dict = create_results_dict(config, dataset_info, results, metrics, metrics['classification_report'])
    save_results_to_files(results_dict, config.RESULTS_DIR)
    print_final_results(metrics, results, train_speakers, test_speakers)


if __name__ == "__main__":
    main()
