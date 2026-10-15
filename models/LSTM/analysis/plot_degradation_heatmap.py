"""
Heatmap of all degradation conditions: rows = condition, columns = per-class F1
+ accuracy + macro-F1.

Usage:
    python models/LSTM/analysis/plot_degradation_heatmap.py degradation_results.csv per_class_results.csv \
        --out degradation_heatmap.png
"""

import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

CONDITION_ORDER = [
    "original",
    "alaw_clean", "alaw_snr20", "alaw_snr15", "alaw_snr10", "alaw_snr5",
    "gsm_clean", "gsm_snr20", "gsm_snr15", "gsm_snr10", "gsm_snr5",
]
CONDITION_LABELS = {
    "original": "original (reference)",
    "alaw_clean": "A-law, clean", "alaw_snr20": "A-law, 20 dB",
    "alaw_snr15": "A-law, 15 dB", "alaw_snr10": "A-law, 10 dB", "alaw_snr5": "A-law, 5 dB",
    "gsm_clean": "GSM, clean", "gsm_snr20": "GSM, 20 dB",
    "gsm_snr15": "GSM, 15 dB", "gsm_snr10": "GSM, 10 dB", "gsm_snr5": "GSM, 5 dB",
}
EMOTION_ORDER = ["anger", "fear", "sadness", "disgust", "boredom", "neutral"]


def plot(results_csv, per_class_csv, out_path):
    results = pd.read_csv(results_csv, keep_default_na=False)
    per_class = pd.read_csv(per_class_csv, keep_default_na=False)

    order = [c for c in CONDITION_ORDER if c in per_class["condition"].values]
    emotions = [e for e in EMOTION_ORDER if e in per_class.columns]

    pc = per_class.set_index("condition").reindex(order)[emotions]
    summary = results.set_index("condition").reindex(order)[["accuracy", "f1_macro"]]

    data = pd.concat([pc, summary], axis=1).astype(float)
    data.index = [CONDITION_LABELS.get(c, c) for c in data.index]

    fig, ax = plt.subplots(figsize=(10, 0.55 * len(data) + 2))
    sns.heatmap(
        data, annot=True, fmt=".3f", cmap="RdYlGn", vmin=0.4, vmax=1.0,
        linewidths=0.5, linecolor="white",
        cbar_kws={"label": "F1 / Accuracy"}, ax=ax,
    )
    # Separate per-class F1 from the overall metrics
    ax.axvline(len(emotions), color="black", linewidth=2)
    ax.set_title("Degradation heatmap - BiLSTM + HuBERT (per-class F1, accuracy, macro-F1)",
                 fontsize=12, fontweight="bold")
    ax.set_xlabel("")
    ax.set_ylabel("")
    plt.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("results_csv", help="degradation_results.csv")
    ap.add_argument("per_class_csv", help="per_class_results.csv")
    ap.add_argument("--out", default="degradation_heatmap.png")
    args = ap.parse_args()
    plot(args.results_csv, args.per_class_csv, args.out)
