"""Report saved mismatched fold/class metrics (no inference).

Usage:
    python models/LSTM/analysis/report_mismatched_details.py --root results/LSTM/mismatched_degradation
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


CONDITIONS = [
    "original", "alaw_clean", "alaw_snr20", "alaw_snr15", "alaw_snr10", "alaw_snr5",
    "gsm_clean", "gsm_snr20", "gsm_snr15", "gsm_snr10", "gsm_snr5",
]
CLASSES = ["anger", "boredom", "disgust", "fear", "neutral", "sadness"]


def plot_confusion(cm, out_path, condition):
    matrix = cm.to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(8, 6.5))
    image = ax.imshow(matrix, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(CLASSES)), CLASSES, rotation=45, ha="right")
    ax.set_yticks(range(len(CLASSES)), CLASSES)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Confusion Matrix (Normalized) - {condition}")
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            value = matrix[row, column]
            ax.text(column, row, f"{value:.3f}", ha="center", va="center",
                    color="white" if value > 0.55 else "black", fontsize=11)
    fig.colorbar(image, ax=ax, shrink=0.85)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    args = parser.parse_args()

    fold = pd.read_csv(os.path.join(args.root, "per_fold_results.csv"))
    per_class = pd.read_csv(os.path.join(args.root, "mismatched_per_class_results.csv"))

    fold_values = fold.drop_duplicates(["condition", "fold"])
    fold_summary = fold_values.groupby("condition")["f1_macro"].agg(
        mean="mean", std=lambda values: values.std(ddof=1)
    ).reindex(CONDITIONS)
    print("F1-macro per condition (8 folds, sample std ddof=1):")
    for condition, values in fold_summary.iterrows():
        print(f"{condition:12s} {values['mean']:.6f} ± {values['std']:.6f}")

    for condition in ("alaw_clean", "alaw_snr10"):
        condition_dir = os.path.join(args.root, condition)
        raw = pd.read_csv(os.path.join(condition_dir, "confusion_matrix_raw.csv"), index_col=0)
        normalized = raw.div(raw.sum(axis=1), axis=0).fillna(0)
        normalized = normalized.reindex(index=CLASSES, columns=CLASSES)
        normalized.to_csv(os.path.join(condition_dir, "confusion_matrix_normalized_3dp.csv"), float_format="%.3f")
        print(f"\n{condition} - normalized confusion matrix (rows=true, columns=predicted):")
        print(normalized.to_string(float_format=lambda value: f"{value:.3f}"))

        metrics = per_class[per_class["condition"] == condition].set_index("class")
        metrics = metrics.reindex(CLASSES)[["precision", "recall", "f1"]]
        print(f"\n{condition} - per-class metrics:")
        print(metrics.to_string(float_format=lambda value: f"{value:.3f}"))
        plot_confusion(
            normalized,
            os.path.join(condition_dir, "confusion_matrix_normalized_3dp.png"),
            condition,
        )


if __name__ == "__main__":
    main()
