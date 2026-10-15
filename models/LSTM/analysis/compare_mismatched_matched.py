"""Side-by-side matched vs mismatched degradation table.

Usage:
    python models/LSTM/analysis/compare_mismatched_matched.py \
        --mismatched-root results/LSTM/mismatched_degradation --out matched_vs_mismatched.csv
"""

import argparse
import json
import os

import pandas as pd


CONDITION_ORDER = [
    "original", "alaw_clean", "alaw_snr20", "alaw_snr15", "alaw_snr10", "alaw_snr5",
    "gsm_clean", "gsm_snr20", "gsm_snr15", "gsm_snr10", "gsm_snr5",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mismatched-root", required=True)
    parser.add_argument("--matched-results", default="degradation_results.csv")
    parser.add_argument("--matched-per-class", default="per_class_results.csv")
    parser.add_argument("--matched-original-done",
                        default="results/LSTM/degradation/original/_DONE.json")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    mismatched = pd.read_csv(os.path.join(args.mismatched_root, "mismatched_results.csv"))
    mismatched_classes = pd.read_csv(
        os.path.join(args.mismatched_root, "mismatched_per_class_results.csv")
    )
    mismatched_fear = mismatched_classes[mismatched_classes["class"] == "fear"]
    mismatched = mismatched.merge(
        mismatched_fear[["condition", "f1"]].rename(columns={"f1": "mismatched_f1_fear"}),
        on="condition",
        how="left",
    )

    matched = pd.read_csv(args.matched_results)
    matched_classes = pd.read_csv(args.matched_per_class)
    matched_fear = matched_classes.set_index("condition")["fear"]
    with open(args.matched_original_done) as handle:
        matched_original = json.load(handle)

    # The matched CSV has no 'original' row; take it from _DONE.json
    matched = pd.concat([
        pd.DataFrame([{
            "condition": "original",
            "accuracy": matched_original["accuracy"],
            "f1_macro": matched_original["f1_macro"],
        }]),
        matched[["condition", "accuracy", "f1_macro"]],
    ], ignore_index=True)
    matched["matched_f1_fear"] = matched["condition"].map(matched_fear)
    matched.loc[matched["condition"] == "original", "matched_f1_fear"] = matched_original["per_class"]["fear"]
    matched["matched_delta_f1_macro"] = matched["f1_macro"] - matched.loc[
        matched["condition"] == "original", "f1_macro"
    ].iloc[0]
    matched = matched.rename(columns={
        "accuracy": "matched_accuracy",
        "f1_macro": "matched_f1_macro",
    })

    result = mismatched[["condition", "accuracy", "f1_macro", "mismatched_f1_fear"]].rename(
        columns={"accuracy": "mismatched_accuracy", "f1_macro": "mismatched_f1_macro"}
    )
    result["mismatched_delta_f1_macro"] = result["mismatched_f1_macro"] - mismatched.loc[
        mismatched["condition"] == "original", "f1_macro"
    ].iloc[0]
    result = result.merge(matched, on="condition", how="left")
    result["condition"] = pd.Categorical(result["condition"], CONDITION_ORDER, ordered=True)
    result = result.sort_values("condition")
    result.to_csv(args.out, index=False, float_format="%.6f")
    print(result.to_string(index=False))
    print(f"Saved comparison CSV: {args.out}")


if __name__ == "__main__":
    main()
