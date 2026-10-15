"""
Mismatched-condition study: evaluate the clean-trained BiLSTM fold checkpoints
on every degraded condition (no retraining).

Checkpoints come from:
    python -m models.LSTM.run_hubert_experiment --tuned --bidirectional --num_classes 6 \
        --outdir <dir> --checkpoint-root <ckpt_dir>

Usage (from repo root):
    python -m models.LSTM.run_mismatched_degradation \
        --npz-root data/splits/conditions --checkpoint-root <ckpt_dir> \
        --outdir results/LSTM/mismatched_degradation
"""

import argparse
import json
import os
import pickle

import numpy as np
import pandas as pd
import torch
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

from .config import Config, PROJECT_ROOT, set_seeds
from .utils.data import calculate_fold_balance_score
from .utils.model import LSTMClassifier
from .utils.sequence_dataset import SequenceDataset
from .utils.split import load_fixed_speaker_split, stratified_loso_order
from .utils.training import validate_model


CONDITIONS = [
    "original", "alaw_clean", "alaw_snr20", "alaw_snr15", "alaw_snr10", "alaw_snr5",
    "gsm_clean", "gsm_snr20", "gsm_snr15", "gsm_snr10", "gsm_snr5",
]


class CleanBiLSTMConfig(Config):
    """Hyperparameters of the final clean BiLSTM baseline."""

    LEARNING_RATE = 3e-4
    DROPOUT = 0.2
    PATIENCE = 15
    TOP_FEATURES_RATIO = 0.85


def parse_condition(condition):
    if condition == "original":
        return "", ""
    codec, suffix = condition.split("_", 1)
    return codec, "" if suffix == "clean" else int(suffix.removeprefix("snr"))


def apply_transform(dataset, scaler, selector):
    """Apply a fitted scaler (+ selector) to a dataset in place."""
    X = dataset.X.numpy()
    shape = X.shape
    X = scaler.transform(X.reshape(-1, shape[-1]))
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    if selector is not None:
        X = selector.transform(X)
    dataset.X = torch.from_numpy(X.reshape(shape[0], shape[1], -1)).float()
    dataset.input_dim = dataset.X.shape[-1]
    return dataset


def fit_preprocessing(train_dataset, datasets, config):
    """Fit scaler/selector on training frames and apply them to `datasets`."""
    train_X = train_dataset.X.numpy()
    lengths = train_dataset.lengths.numpy()
    valid_rows = np.vstack([
        train_X[i, :lengths[i], :] for i in range(len(train_X))
    ])
    valid_rows = np.nan_to_num(valid_rows, nan=0.0, posinf=0.0, neginf=0.0)
    scaler = StandardScaler().fit(valid_rows)

    selector = None
    if getattr(config, "FEATURE_SELECTION", False):
        scaled_train = scaler.transform(train_X.reshape(-1, train_X.shape[-1]))
        scaled_train = np.nan_to_num(scaled_train, nan=0.0, posinf=0.0, neginf=0.0)
        non_zero = np.any(scaled_train != 0, axis=1)
        k_features = int(train_X.shape[-1] * config.TOP_FEATURES_RATIO)
        selector = SelectKBest(score_func=f_classif, k=k_features)
        selector.fit(scaled_train[non_zero], np.repeat(train_dataset.y.numpy(), train_X.shape[1])[non_zero])

    for dataset in datasets:
        apply_transform(dataset, scaler, selector)
    return scaler, selector


def make_loader(dataset, config, shuffle):
    return DataLoader(
        dataset,
        batch_size=config.BATCH_SIZE if shuffle else config.VAL_BATCH_SIZE,
        shuffle=shuffle,
        pin_memory=config.pin_memory,
        num_workers=config.NUM_WORKERS,
    )


def encoded_speaker_id(dataset, raw_speaker):
    matches = np.flatnonzero(dataset.speaker_classes == raw_speaker)
    if len(matches) != 1:
        raise ValueError(f"Could not map raw speaker {raw_speaker!r} in {dataset.speaker_classes}")
    return int(matches[0])


def infer(model, dataset, config):
    loader = make_loader(dataset, config, shuffle=False)
    criterion = torch.nn.CrossEntropyLoss()
    _, _, _, predictions, truths, _ = validate_model(model, loader, criterion, config.device)
    return np.asarray(truths), np.asarray(predictions)


def save_fold_artifacts(path, model, scaler, selector, config, classes):
    os.makedirs(path, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(path, "model.pt"))
    with open(os.path.join(path, "preprocessing.pkl"), "wb") as handle:
        pickle.dump({"scaler": scaler, "selector": selector}, handle)
    with open(os.path.join(path, "metadata.json"), "w") as handle:
        json.dump({"classes": list(classes), "input_dim": model.lstm.input_size,
                   "hidden_dim": config.HIDDEN_DIM, "num_layers": config.NUM_LAYERS,
                   "dropout": config.DROPOUT, "bidirectional": config.BIDIRECTIONAL,
                   "checkpoint_type": "clean_baseline"}, handle, indent=2)


def load_fold_artifacts(path, config, classes):
    """Load an exact clean-baseline fold checkpoint (never retrains)."""
    model_path = os.path.join(path, "model.pt")
    preprocessing_path = os.path.join(path, "preprocessing.pkl")
    if not os.path.exists(model_path) or not os.path.exists(preprocessing_path):
        raise FileNotFoundError(
            f"Missing exact clean baseline artifacts in {path}. "
            "The mismatched experiment must not retrain a fold."
        )
    with open(os.path.join(path, "metadata.json")) as handle:
        metadata = json.load(handle)
    if metadata.get("checkpoint_type") != "clean_baseline":
        raise RuntimeError(f"{path} is not marked as an exact clean baseline checkpoint")
    if metadata["classes"] != list(classes):
        raise RuntimeError(f"Checkpoint classes {metadata['classes']} do not match {list(classes)}")
    model = LSTMClassifier(
        input_dim=metadata["input_dim"],
        num_classes=len(classes),
        hidden_dim=metadata["hidden_dim"],
        num_layers=metadata["num_layers"],
        dropout=metadata["dropout"],
        bidirectional=metadata["bidirectional"],
    ).to(config.device)
    state = torch.load(model_path, map_location=config.device, weights_only=True)
    model.load_state_dict(state)
    with open(preprocessing_path, "rb") as handle:
        preprocessing = pickle.load(handle)
    return model, preprocessing["scaler"], preprocessing["selector"]


def metrics_row(condition, y_true, y_pred, classes):
    """Summary row + per-class rows for one condition."""
    codec, snr = parse_condition(condition)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(len(classes))), zero_division=0
    )
    row = {"condition": condition, "codec": codec, "snr_db": snr,
           "accuracy": accuracy_score(y_true, y_pred),
           "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0)}
    per_class = []
    for name, p, r, score, count in zip(classes, precision, recall, f1, support):
        per_class.append({"condition": condition, "class": name,
                          "precision": float(p), "recall": float(r), "f1": float(score),
                          "support": int(count)})
    return row, per_class


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz-root", default=os.path.join(PROJECT_ROOT, "data", "splits", "conditions"))
    parser.add_argument("--outdir", required=True)
    parser.add_argument(
        "--checkpoint-root",
        required=True,
        help="Directory containing exact clean baseline fold_1..fold_8 artifacts",
    )
    parser.add_argument("--conditions", nargs="+", choices=CONDITIONS, default=CONDITIONS)
    args = parser.parse_args()

    config = CleanBiLSTMConfig()
    os.makedirs(args.outdir, exist_ok=True)
    set_seeds()

    clean_path = os.path.join(args.npz_root, "prepared_hubert_original.npz")
    clean = SequenceDataset(clean_path)
    classes = list(clean.label_classes)
    if len(classes) != 6 or "happiness" in classes:
        raise RuntimeError(f"Expected six clean classes without happiness, got {classes}")
    train_speakers, _ = load_fixed_speaker_split(clean, config.SPEAKER_SPLIT_JSON)
    loso_order = stratified_loso_order(train_speakers, clean)
    print(f"Classes ({len(classes)}): {classes}")
    print(f"Pooled metric mode: concatenates predictions from all {len(loso_order)} folds")

    predictions = {condition: {"true": [], "pred": []} for condition in args.conditions}
    fold_rows = []

    for fold_number, val_speaker in enumerate(loso_order, start=1):
        raw_speaker = clean.speaker_classes[val_speaker]
        train_dataset = SequenceDataset(clean_path,
                                        speaker_ids=[s for s in loso_order if s != val_speaker])
        clean_val = SequenceDataset(clean_path, speaker_ids=[val_speaker])

        # Same validation speaker, every condition
        condition_datasets = {}
        for condition in args.conditions:
            condition_path = os.path.join(args.npz_root, f"prepared_hubert_{condition}.npz")
            condition_dataset = SequenceDataset(condition_path)
            condition_datasets[condition] = SequenceDataset(
                condition_path,
                speaker_ids=[encoded_speaker_id(condition_dataset, raw_speaker)],
            )

        fold_path = os.path.join(args.checkpoint_root, f"fold_{fold_number}")
        model, scaler, selector = load_fold_artifacts(fold_path, config, classes)
        apply_transform(train_dataset, scaler, selector)
        apply_transform(clean_val, scaler, selector)
        for dataset in condition_datasets.values():
            apply_transform(dataset, scaler, selector)
        print(f"\n--- Fold {fold_number}/8 | validation speaker {raw_speaker} | loaded checkpoint ---")
        best_epoch = "loaded"
        total_epochs = "loaded"

        for condition, dataset in condition_datasets.items():
            y_true, y_pred = infer(model, dataset, config)
            predictions[condition]["true"].extend(y_true.tolist())
            predictions[condition]["pred"].extend(y_pred.tolist())
            if condition == "original":
                print(
                    f"fold={fold_number} speaker={raw_speaker} "
                    f"accuracy={accuracy_score(y_true, y_pred):.4f} "
                    f"f1_macro={f1_score(y_true, y_pred, average='macro', zero_division=0):.4f} "
                    f"n={len(y_true)}"
                )
            fold_row, per_class = metrics_row(condition, y_true, y_pred, classes)
            fold_rows.extend({**fold_row, "fold": fold_number,
                              "fold_speaker_id": raw_speaker,
                              "best_epoch": best_epoch, "total_epochs": total_epochs,
                              "balance_score": calculate_fold_balance_score(
                                  [s for s in loso_order if s != val_speaker], val_speaker, clean
                              ), **item} for item in per_class)

    summary_rows, per_class_rows = [], []
    for condition in args.conditions:
        y_true = np.asarray(predictions[condition]["true"])
        y_pred = np.asarray(predictions[condition]["pred"])
        print(f"condition={condition} pooled_predictions={len(y_true)}")
        row, per_class = metrics_row(condition, y_true, y_pred, classes)
        condition_fold = pd.DataFrame([r for r in fold_rows if r["condition"] == condition]).drop_duplicates(
            subset=["fold"]
        )
        row["fold_accuracy_mean"] = condition_fold["accuracy"].mean()
        row["fold_accuracy_std"] = condition_fold["accuracy"].std(ddof=1)
        row["fold_f1_macro_mean"] = condition_fold["f1_macro"].mean()
        row["fold_f1_macro_std"] = condition_fold["f1_macro"].std(ddof=1)
        summary_rows.append(row)
        per_class_rows.extend(per_class)

        condition_dir = os.path.join(args.outdir, condition)
        os.makedirs(condition_dir, exist_ok=True)
        cm = confusion_matrix(y_true, y_pred, labels=list(range(len(classes))))
        cm_normalized = cm.astype(float) / np.maximum(cm.sum(axis=1, keepdims=True), 1)
        pd.DataFrame(cm, index=classes, columns=classes).to_csv(os.path.join(condition_dir, "confusion_matrix_raw.csv"))
        pd.DataFrame(cm_normalized, index=classes, columns=classes).to_csv(
            os.path.join(condition_dir, "confusion_matrix_normalized.csv")
        )
        np.save(os.path.join(condition_dir, "y_true.npy"), y_true)
        np.save(os.path.join(condition_dir, "y_pred.npy"), y_pred)

    summary = pd.DataFrame(summary_rows)
    per_class = pd.DataFrame(per_class_rows)
    fold = pd.DataFrame(fold_rows)
    summary.to_csv(os.path.join(args.outdir, "mismatched_results.csv"), index=False)
    per_class.to_csv(os.path.join(args.outdir, "mismatched_per_class_results.csv"), index=False)
    fold.to_csv(os.path.join(args.outdir, "per_fold_results.csv"), index=False)
    fold.to_csv(os.path.join(args.outdir, "per_fold_metrics.csv"), index=False)

    print("\nMismatched results:")
    print(summary[["condition", "accuracy", "f1_macro", "fold_accuracy_mean", "fold_accuracy_std"]].to_string(index=False))


if __name__ == "__main__":
    main()
