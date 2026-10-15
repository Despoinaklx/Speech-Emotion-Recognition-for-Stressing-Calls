"""
Matched-condition degradation study: for every channel condition, extract HuBERT
features, build the .npz and run the 8-fold LOSO BiLSTM (train and test on the
same condition).

Usage (from repo root):
    python -m models.LSTM.run_degradation_experiments \
        --data-root data/degraded \
        --embed-root data/features/hubert_conditions \
        --npz-root data/splits/conditions \
        --results-root results/LSTM/degradation \
        --out-csv degradation_results.csv \
        --per-class-csv per_class_results.csv \
        --per-fold-csv per_fold_metrics.csv
"""
import os
import json
import argparse
import numpy as np
import pandas as pd

from .config import Config, set_seeds
from .utils.hubert_data_preparation import prepare_hubert
from .utils.sequence_dataset import SequenceDataset
from .utils.split import find_best_speaker_split, load_fixed_speaker_split, stratified_loso_order
from .utils.training import (
    prepare_fold_data,
    setup_model_and_optimizer,
    create_criterion,
    train_fold,
    validate_model,
)
from .utils.visualization import plot_confusion_matrix
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, precision_recall_fscore_support

from .hubert_extract_condition import extract_condition


# Flagship BiLSTM+HuBERT hyperparameters (6 classes), fixed across conditions.
# Everything else comes from config.py.
FLAGSHIP_OVERRIDES = {
    "LEARNING_RATE": 5e-4,
    "DROPOUT": 0.5,
    "PATIENCE": 8,
    "TOP_FEATURES_RATIO": 0.85,
}


def make_condition_config(condition, base_dirs):
    """Config instance with per-condition paths + flagship hyperparameters."""
    cfg = Config()

    for k, v in FLAGSHIP_OVERRIDES.items():
        setattr(cfg, k, v)

    cfg.HUBERT_NPY = os.path.join(base_dirs["embed_root"], condition, "frame_level")
    cfg.HUBERT_LABELS_CSV = os.path.join(base_dirs["embed_root"], condition, "hubert_labels.csv")
    cfg.NPZ_PATH = os.path.join(base_dirs["npz_root"], f"prepared_hubert_{condition}.npz")
    cfg.RESULTS_DIR = os.path.join(base_dirs["results_root"], condition)

    os.makedirs(cfg.RESULTS_DIR, exist_ok=True)
    return cfg


def run_loso_for_condition(cfg, fold_callback=None):
    """
    8-fold LOSO for one condition. Metrics are computed on predictions pooled
    over all folds.

    fold_callback(val_speaker_id, val_true, val_pred, classes), if given, is
    called after every fold so per-fold metrics can be saved incrementally.

    Returns (accuracy, f1_macro, classes, per_class_f1, y_true, y_pred).
    """
    dataset = SequenceDataset(cfg.NPZ_PATH)
    classes = list(dataset.label_classes)

    # Same speakers in every condition, so the curve only reflects the channel
    if getattr(cfg, 'USE_FIXED_SPEAKER_SPLIT', True):
        train_speakers, test_speakers = load_fixed_speaker_split(dataset, cfg.SPEAKER_SPLIT_JSON)
    else:
        train_speakers, test_speakers = find_best_speaker_split(
            dataset,
            test_speakers_percentage=cfg.TEST_SPEAKERS_PERCENTAGE,
            max_iterations=cfg.MAX_SPLIT_ITERATIONS,
        )
    loso_order = stratified_loso_order(train_speakers, dataset)

    all_y_true, all_y_pred = [], []

    for val_speaker_id in loso_order:
        current_train = [s for s in loso_order if s != val_speaker_id]
        train_ds, val_ds, train_loader, val_loader = prepare_fold_data(
            current_train, val_speaker_id, dataset, cfg
        )
        model, optimizer = setup_model_and_optimizer(
            None, train_ds.num_classes, cfg, train_dataset=train_ds
        )
        criterion = create_criterion(train_ds, train_ds.num_classes, cfg.device, cfg)

        print(f"\n--- [{cfg._condition_name}] Fold for validation speaker {val_speaker_id} ---")
        train_fold(model, train_loader, val_loader, optimizer, criterion, cfg)

        _, _, _, val_preds, val_true, _ = validate_model(
            model, val_loader, criterion, cfg.device
        )
        all_y_true.extend(val_true)
        all_y_pred.extend(val_preds)

        if fold_callback is not None:
            fold_callback(val_speaker_id, val_true, val_preds, classes)

    accuracy = accuracy_score(all_y_true, all_y_pred)
    f1_macro = f1_score(all_y_true, all_y_pred, average="macro", zero_division=0)

    # Explicit labels keep the order aligned with label_classes even if a class is missing
    per_class_f1 = f1_score(
        all_y_true, all_y_pred,
        labels=list(range(len(classes))),
        average=None,
        zero_division=0,
    )
    per_class = {cls: float(f1) for cls, f1 in zip(classes, per_class_f1)}

    return accuracy, f1_macro, classes, per_class, all_y_true, all_y_pred


def parse_condition(condition):
    """
    'original'   -> (None, None)
    'alaw_clean' -> ('alaw', None)
    'gsm_snr10'  -> ('gsm', 10)
    """
    if condition == "original":
        return None, None
    parts = condition.split("_")
    codec = parts[0]
    tail = parts[1]
    if tail == "clean":
        return codec, None
    if tail.startswith("snr"):
        return codec, int(tail[3:])
    return codec, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True,
                    help="degraded wav root with manifest.csv (e.g. data/degraded)")
    ap.add_argument("--embed-root", required=True,
                    help="cached per-condition HuBERT .npy root")
    ap.add_argument("--npz-root", required=True,
                    help="cached per-condition .npz root")
    ap.add_argument("--results-root", required=True,
                    help="root for per-condition results")
    ap.add_argument("--out-csv", default="degradation_results.csv")
    ap.add_argument("--per-class-csv", default="per_class_results.csv",
                    help="per-class F1 per condition")
    ap.add_argument("--per-fold-csv", default="per_fold_metrics.csv",
                    help="per-class precision/recall/F1/support per (condition, fold)")
    ap.add_argument("--conditions", nargs="*", default=None,
                    help="subset of conditions (default: all in the manifest)")
    ap.add_argument("--skip-existing", action="store_true",
                    help="reuse finished conditions (_DONE.json + confusion matrix)")
    args = ap.parse_args()

    manifest_path = os.path.join(args.data_root, "manifest.csv")
    manifest = pd.read_csv(manifest_path)
    all_conditions = list(dict.fromkeys(manifest["condition"].tolist()))  # keeps order
    conditions = args.conditions or all_conditions

    base_dirs = {
        "embed_root": args.embed_root,
        "npz_root": args.npz_root,
        "results_root": args.results_root,
    }
    for d in base_dirs.values():
        os.makedirs(d, exist_ok=True)

    set_seeds()
    rows = []
    per_class_rows = []
    per_fold_rows = []

    def make_fold_callback(condition, codec, snr):
        def _callback(val_speaker_id, val_true, val_pred, classes):
            fold_acc = accuracy_score(val_true, val_pred)
            fold_f1_macro = f1_score(val_true, val_pred, average="macro", zero_division=0)
            precision, recall, f1, support = precision_recall_fscore_support(
                val_true, val_pred, labels=list(range(len(classes))), zero_division=0
            )
            for cls, p, r, f, s in zip(classes, precision, recall, f1, support):
                per_fold_rows.append({
                    "condition": condition,
                    "codec": codec if codec else "",
                    "snr_db": snr if snr is not None else "",
                    "fold_speaker_id": val_speaker_id,
                    "accuracy": round(float(fold_acc), 4),
                    "f1_macro": round(float(fold_f1_macro), 4),
                    "class": cls,
                    "precision": round(float(p), 4),
                    "recall": round(float(r), 4),
                    "f1": round(float(f), 4),
                    "support": int(s),
                })
            # Rewrite after every fold (robust to interruption)
            pd.DataFrame(per_fold_rows).to_csv(args.per_fold_csv, index=False)
        return _callback

    for condition in conditions:
        print("\n" + "=" * 70)
        print(f"CONDITION: {condition}")
        print("=" * 70)

        cfg = make_condition_config(condition, base_dirs)
        cfg._condition_name = condition

        # 1. HuBERT extraction (cached)
        wav_dir = os.path.join(args.data_root, condition)
        if not os.path.isdir(wav_dir):
            print(f"[skip] wav folder not found: {wav_dir}")
            continue

        extract_condition(
            wav_dir=wav_dir,
            out_frame_dir=cfg.HUBERT_NPY,
            labels_csv_out=cfg.HUBERT_LABELS_CSV,
            skip_existing=True,
        )

        # 2. .npz (cached)
        if not os.path.exists(cfg.NPZ_PATH):
            prepare_hubert(
                hubert_npy_path=cfg.HUBERT_NPY,
                labels_csv_path=cfg.HUBERT_LABELS_CSV,
                npz_out_path=cfg.NPZ_PATH,
                config=cfg,
            )
        else:
            print(f"[cache] Found .npz: {cfg.NPZ_PATH}")

        # 3. LOSO (or reuse a finished run that also has a confusion matrix)
        done_marker = os.path.join(cfg.RESULTS_DIR, "_DONE.json")
        cm_raw_csv = os.path.join(cfg.RESULTS_DIR, "confusion_matrix_raw.csv")
        if args.skip_existing and os.path.exists(done_marker) and os.path.exists(cm_raw_csv):
            print(f"[skip-existing] {condition} already finished.")
            with open(done_marker) as f:
                saved = json.load(f)
            acc, f1, per_class = saved["accuracy"], saved["f1_macro"], saved["per_class"]
        else:
            codec_for_fold, snr_for_fold = parse_condition(condition)
            fold_callback = make_fold_callback(condition, codec_for_fold, snr_for_fold)
            acc, f1, classes, per_class, y_true, y_pred = run_loso_for_condition(
                cfg, fold_callback=fold_callback
            )
            with open(done_marker, "w") as f:
                json.dump(
                    {"accuracy": acc, "f1_macro": f1, "per_class": per_class},
                    f, ensure_ascii=False, indent=2,
                )

            # Confusion matrix (raw + normalized), pooled predictions, heatmap
            np.save(os.path.join(cfg.RESULTS_DIR, "y_true.npy"), np.array(y_true))
            np.save(os.path.join(cfg.RESULTS_DIR, "y_pred.npy"), np.array(y_pred))

            cm = confusion_matrix(y_true, y_pred, labels=list(range(len(classes))))
            cm_normalized = cm.astype("float") / (cm.sum(axis=1)[:, np.newaxis] + 1e-8)

            pd.DataFrame(cm, index=classes, columns=classes).to_csv(cm_raw_csv)
            pd.DataFrame(cm_normalized, index=classes, columns=classes).to_csv(
                os.path.join(cfg.RESULTS_DIR, "confusion_matrix_normalized.csv")
            )
            plot_confusion_matrix(y_true, y_pred, classes, cfg.RESULTS_DIR, f" - {condition}")

        codec, snr = parse_condition(condition)
        rows.append({
            "condition": condition,
            "codec": codec if codec else "",
            "snr_db": snr if snr is not None else "",
            "accuracy": round(acc, 4),
            "f1_macro": round(f1, 4),
        })

        # One column per emotion
        pc_row = {
            "condition": condition,
            "codec": codec if codec else "",
            "snr_db": snr if snr is not None else "",
        }
        for cls, val in per_class.items():
            pc_row[cls] = round(val, 4)
        per_class_rows.append(pc_row)

        print(f"\n>>> [{condition}]  accuracy={acc:.4f}  f1_macro={f1:.4f}")
        pc_str = "  ".join(f"{c}={v:.3f}" for c, v in per_class.items())
        print(f"    per-class F1: {pc_str}")

        # Rewrite both CSVs after every condition
        pd.DataFrame(rows).to_csv(args.out_csv, index=False)
        pd.DataFrame(per_class_rows).to_csv(args.per_class_csv, index=False)

    if per_fold_rows:
        pd.DataFrame(per_fold_rows).to_csv(args.per_fold_csv, index=False)

    print("\n" + "=" * 70)
    print("DONE.")
    print(f"  Overall metrics   -> {args.out_csv}")
    print(f"  Per-class F1      -> {args.per_class_csv}")
    print(f"  Per-fold metrics  -> {args.per_fold_csv}")
    print("=" * 70)
    print(pd.DataFrame(rows).to_string(index=False))
    print("\nPer-class F1 per condition:")
    print(pd.DataFrame(per_class_rows).to_string(index=False))


if __name__ == "__main__":
    main()
