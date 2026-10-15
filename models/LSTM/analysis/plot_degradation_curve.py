"""
Degradation curve from degradation_results.csv: accuracy and macro-F1 vs SNR,
one line per codec, 'original' as a horizontal reference.

Usage:
    python models/LSTM/analysis/plot_degradation_curve.py degradation_results.csv --out degradation_curve.png
"""

import argparse
import pandas as pd
import matplotlib.pyplot as plt

# x-axis order: decreasing channel quality
X_ORDER = ["clean", "20", "15", "10", "5"]
X_LABELS = ["clean", "20 dB", "15 dB", "10 dB", "5 dB"]


def _xkey(row):
    if row["snr_db"] == "" or pd.isna(row["snr_db"]):
        return "clean"
    return str(int(row["snr_db"]))


def plot(csv_path, out_path):
    df = pd.read_csv(csv_path, keep_default_na=False)

    ref = df[df["condition"] == "original"]
    ref_acc = float(ref["accuracy"].iloc[0]) if len(ref) else None
    ref_f1 = float(ref["f1_macro"].iloc[0]) if len(ref) else None

    channel = df[df["condition"] != "original"].copy()
    channel["xkey"] = channel.apply(_xkey, axis=1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    metrics = [("accuracy", "Accuracy", ref_acc), ("f1_macro", "Macro-F1", ref_f1)]

    codec_style = {
        "alaw": dict(marker="o", label="A-law (landline)"),
        "gsm": dict(marker="s", label="GSM-FR (mobile)"),
    }

    for ax, (col, title, ref_val) in zip(axes, metrics):
        for codec, style in codec_style.items():
            sub = channel[channel["codec"] == codec]
            sub = sub.set_index("xkey").reindex(X_ORDER).reset_index()
            ax.plot(range(len(X_ORDER)), sub[col].astype(float),
                    marker=style["marker"], linewidth=2, markersize=7,
                    label=style["label"])

        if ref_val is not None:
            ax.axhline(ref_val, linestyle="--", color="gray", linewidth=1.5,
                       label=f"original (reference) = {ref_val:.3f}")

        ax.set_xticks(range(len(X_ORDER)))
        ax.set_xticklabels(X_LABELS)
        ax.set_xlabel("Channel condition (decreasing SNR)")
        ax.set_ylabel(title)
        ax.set_title(f"{title} degradation - BiLSTM + HuBERT")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)
        ax.set_ylim(0, 1)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", help="degradation_results.csv")
    ap.add_argument("--out", default="degradation_curve.png")
    args = ap.parse_args()
    plot(args.csv, args.out)
