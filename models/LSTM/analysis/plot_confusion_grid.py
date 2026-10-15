"""
Grid of normalized confusion matrices for all degradation conditions.

Usage:
    python models/LSTM/analysis/plot_confusion_grid.py results/LSTM/degradation \
        --out confusion_matrix_grid.png
"""

import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

CONDITION_ORDER = [
    "original",
    "alaw_clean", "alaw_snr20", "alaw_snr15", "alaw_snr10", "alaw_snr5",
    "gsm_clean", "gsm_snr20", "gsm_snr15", "gsm_snr10", "gsm_snr5",
]
CONDITION_LABELS = {
    "original": "original", "alaw_clean": "A-law\nclean", "alaw_snr20": "A-law\n20dB",
    "alaw_snr15": "A-law\n15dB", "alaw_snr10": "A-law\n10dB", "alaw_snr5": "A-law\n5dB",
    "gsm_clean": "GSM\nclean", "gsm_snr20": "GSM\n20dB",
    "gsm_snr15": "GSM\n15dB", "gsm_snr10": "GSM\n10dB", "gsm_snr5": "GSM\n5dB",
}


def plot(degradation_root, out_path):
    conditions = [c for c in CONDITION_ORDER
                  if os.path.exists(os.path.join(degradation_root, c, "confusion_matrix_normalized.csv"))]

    n = len(conditions)
    ncols = 4
    nrows = -(-n // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3.6 * nrows))
    axes = axes.flatten()

    for ax, cond in zip(axes, conditions):
        cm = pd.read_csv(os.path.join(degradation_root, cond, "confusion_matrix_normalized.csv"), index_col=0)
        classes = list(cm.columns)
        sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues", vmin=0, vmax=1,
                    xticklabels=classes, yticklabels=classes,
                    cbar=False, ax=ax, annot_kws={"size": 7})
        ax.set_title(CONDITION_LABELS.get(cond, cond), fontsize=10, fontweight="bold")
        ax.tick_params(axis="both", labelsize=7)
        ax.set_xlabel("")
        ax.set_ylabel("")

    for ax in axes[n:]:
        ax.axis("off")

    fig.suptitle("Normalized confusion matrices - BiLSTM + HuBERT per degradation condition",
                 fontsize=14, fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("degradation_root", help="e.g. results/LSTM/degradation")
    ap.add_argument("--out", default="confusion_matrix_grid.png")
    args = ap.parse_args()
    plot(args.degradation_root, args.out)
