"""
Per-class precision/recall/F1/support for every degradation condition, from the
pooled y_true.npy / y_pred.npy saved by run_degradation_experiments.py.

Usage:
    python models/LSTM/analysis/aggregate_per_class_metrics.py results/LSTM/degradation \
        --out per_class_full_metrics.csv
"""

import argparse
import os
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support

CONDITION_ORDER = [
    "original",
    "alaw_clean", "alaw_snr20", "alaw_snr15", "alaw_snr10", "alaw_snr5",
    "gsm_clean", "gsm_snr20", "gsm_snr15", "gsm_snr10", "gsm_snr5",
]


def parse_condition(condition):
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


def aggregate(degradation_root, out_path):
    rows = []
    for condition in CONDITION_ORDER:
        cond_dir = os.path.join(degradation_root, condition)
        y_true_path = os.path.join(cond_dir, "y_true.npy")
        y_pred_path = os.path.join(cond_dir, "y_pred.npy")
        cm_path = os.path.join(cond_dir, "confusion_matrix_raw.csv")
        if not (os.path.exists(y_true_path) and os.path.exists(y_pred_path) and os.path.exists(cm_path)):
            print(f"[skip] missing y_true/y_pred/confusion_matrix for {condition}")
            continue

        y_true = np.load(y_true_path)
        y_pred = np.load(y_pred_path)
        classes = list(pd.read_csv(cm_path, index_col=0).columns)

        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, labels=list(range(len(classes))), zero_division=0
        )

        codec, snr = parse_condition(condition)
        for cls, p, r, f, s in zip(classes, precision, recall, f1, support):
            rows.append({
                "condition": condition,
                "codec": codec if codec else "",
                "snr_db": snr if snr is not None else "",
                "class": cls,
                "precision": round(float(p), 4),
                "recall": round(float(r), 4),
                "f1": round(float(f), 4),
                "support": int(s),
            })

    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    print(f"Saved: {out_path}  ({len(df)} rows)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("degradation_root", help="e.g. results/LSTM/degradation")
    ap.add_argument("--out", default="per_class_full_metrics.csv")
    args = ap.parse_args()
    aggregate(args.degradation_root, args.out)
