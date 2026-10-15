"""
Per-emotion F1 vs channel condition (from per_class_results.csv): one subplot per
codec, one line per emotion, dotted 'original' reference per emotion.

Usage:
    python models/LSTM/analysis/plot_per_class_degradation.py per_class_results.csv --out per_class_curve.png
"""

import argparse
import pandas as pd
import matplotlib.pyplot as plt

X_ORDER = ["clean", "20", "15", "10", "5"]
X_LABELS = ["clean", "20 dB", "15 dB", "10 dB", "5 dB"]

# Fixed emotion order/colour across subplots
EMOTION_ORDER = ["anger", "fear", "sadness", "disgust", "boredom", "neutral"]


def _xkey(row):
    if row["snr_db"] == "" or pd.isna(row["snr_db"]):
        return "clean"
    return str(int(row["snr_db"]))


def plot(csv_path, out_path):
    df = pd.read_csv(csv_path, keep_default_na=False)

    meta = {"condition", "codec", "snr_db"}
    emotions = [c for c in df.columns if c not in meta]
    emotions = [e for e in EMOTION_ORDER if e in emotions] + \
               [e for e in emotions if e not in EMOTION_ORDER]

    ref = df[df["condition"] == "original"]
    ref_vals = {e: (float(ref[e].iloc[0]) if len(ref) else None) for e in emotions}

    channel = df[df["condition"] != "original"].copy()
    channel["xkey"] = channel.apply(_xkey, axis=1)

    codecs = [("alaw", "A-law (landline)"),
              ("gsm", "GSM-FR (mobile)")]

    fig, axes = plt.subplots(1, 2, figsize=(15, 6), sharey=True)
    cmap = plt.get_cmap("tab10")

    for ax, (codec, codec_title) in zip(axes, codecs):
        sub = channel[channel["codec"] == codec].set_index("xkey")
        for i, emo in enumerate(emotions):
            ser = sub.reindex(X_ORDER)[emo].astype(float)
            color = cmap(i % 10)
            ax.plot(range(len(X_ORDER)), ser.values,
                    marker="o", linewidth=2, markersize=6,
                    color=color, label=emo)
            if ref_vals[emo] is not None:
                ax.axhline(ref_vals[emo], linestyle=":", color=color,
                           linewidth=1, alpha=0.6)

        ax.set_xticks(range(len(X_ORDER)))
        ax.set_xticklabels(X_LABELS)
        ax.set_xlabel("Channel condition (decreasing SNR)")
        ax.set_title(codec_title)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1)

    axes[0].set_ylabel("Per-class F1")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(emotions),
               bbox_to_anchor=(0.5, 1.02), fontsize=10)
    fig.suptitle("Per-class F1 degradation - BiLSTM + HuBERT",
                 y=1.08, fontsize=13)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path}")

    # F1 drop from original to 5 dB per emotion
    print("\nF1 drop (original -> 5 dB) per emotion:")
    for codec, codec_title in codecs:
        sub = channel[(channel["codec"] == codec) & (channel["xkey"] == "5")]
        if not len(sub):
            continue
        print(f"\n  {codec_title}:")
        drops = []
        for emo in emotions:
            if ref_vals[emo] is None:
                continue
            worst = float(sub[emo].iloc[0])
            drop = ref_vals[emo] - worst
            drops.append((emo, ref_vals[emo], worst, drop))
        for emo, r, w, d in sorted(drops, key=lambda t: t[3], reverse=True):
            print(f"    {emo:10s}: {r:.3f} -> {w:.3f}   (drop {d:+.3f})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", help="per_class_results.csv")
    ap.add_argument("--out", default="per_class_curve.png")
    args = ap.parse_args()
    plot(args.csv, args.out)
