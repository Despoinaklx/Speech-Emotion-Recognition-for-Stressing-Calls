# Speaker-independent train/test split of the handcrafted feature CSVs.
# Uses the shared fixed split (data/splits/speaker_split.json) when present;
# otherwise searches a split whose test emotion distribution matches the
# overall one (SSE) -- only needed once to create that file.
#
# Usage (from repo root):
#   python models/classicalML/split.py

import os
import json
import random
import logging
from typing import Tuple, Set

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

CONFIG = {
    "csv_dir": os.path.join(ROOT, "data", "features", "handcrafted"),
    "output_dir": os.path.join(ROOT, "data", "splits"),
    "test_speakers_percentage": 0.20,
    "max_iterations": 100,
    "random_seed": 42,
    "canonical_split_json": os.path.join(ROOT, "data", "splits", "speaker_split.json"),
}


def setup_logging() -> None:
    log_dir = CONFIG["output_dir"]
    os.makedirs(log_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(os.path.join(log_dir, "data_split.log")), logging.StreamHandler()],
    )


def set_random_seeds(seed: int) -> None:
    np.random.seed(seed)
    random.seed(seed)
    logging.info(f"Random seeds set to {seed}")


def load_data(csv_dir: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load the global and frame-level feature CSVs."""
    global_path = os.path.join(csv_dir, "all_global_features.csv")
    frame_path = os.path.join(csv_dir, "all_frame_features.csv")
    if not os.path.exists(global_path) or not os.path.exists(frame_path):
        raise FileNotFoundError("CSV paths are incorrect")
    global_df = pd.read_csv(global_path)
    frame_df = pd.read_csv(frame_path)
    if global_df.empty or frame_df.empty:
        raise ValueError("CSV files are empty")
    for col in ["filename", "speaker_id", "emotion"]:
        if col not in global_df.columns:
            raise ValueError(f"Missing column: {col}")
    logging.info(
        f"Loaded {len(global_df)} global rows & {len(frame_df)} frame rows"
    )
    return global_df, frame_df


def prepare_data(global_df: pd.DataFrame):
    """Returns X (features), y (emotion), groups (speaker_id)."""
    X = global_df.drop(columns=["filename", "speaker_id", "emotion"])
    y = global_df["emotion"]
    groups = global_df["speaker_id"]
    logging.info(f"Prepared {len(X)} samples with {X.shape[1]} features")
    logging.info(f"Emotion distribution: {y.value_counts().to_dict()}")
    logging.info(f"Unique speakers: {groups.nunique()}")
    return X, y, groups


def find_best_split(X, y, groups, test_size, max_iter):
    """Best GroupShuffleSplit (lowest SSE between test and overall label distribution).
    Returns (train_idx, test_idx)."""
    target_dist = y.value_counts(normalize=True)
    best_split, lowest_error = None, float("inf")

    for i in range(max_iter):
        splitter = GroupShuffleSplit(
            test_size=test_size, n_splits=1, random_state=i
        )
        train_idx, test_idx = next(splitter.split(X, y, groups))

        test_dist = y.iloc[test_idx].value_counts(normalize=True)
        aligned_test, aligned_target = test_dist.align(target_dist, fill_value=0)
        error = np.sum((aligned_test - aligned_target) ** 2)

        if error < lowest_error:
            lowest_error = error
            best_split = (train_idx, test_idx)
            logging.info(f"Iteration {i+1}: better split with SSE {error:.6f}")

    logging.info(f"Final chosen split SSE: {lowest_error:.6f}")
    return best_split


def save_split(
    global_df: pd.DataFrame,
    frame_df: pd.DataFrame,
    train_speakers: Set[int],
    test_speakers: Set[int],
    out_dir: str,
) -> None:
    """Write train (for LOSO) and final-test CSVs for global and frame features."""
    os.makedirs(out_dir, exist_ok=True)

    files = {
        "global_train_for_loso.csv": global_df[global_df["speaker_id"].isin(train_speakers)],
        "global_final_test.csv": global_df[global_df["speaker_id"].isin(test_speakers)],
        "frame_train_for_loso.csv": frame_df[frame_df["speaker_id"].isin(train_speakers)],
        "frame_final_test.csv": frame_df[frame_df["speaker_id"].isin(test_speakers)],
    }

    for fname, df in files.items():
        path = os.path.join(out_dir, fname)
        df.to_csv(path, index=False)
        logging.info(f"Saved {len(df)} rows -> {path}")


def main():
    setup_logging()
    set_random_seeds(CONFIG["random_seed"])

    global_df, frame_df = load_data(CONFIG["csv_dir"])

    canonical_path = CONFIG.get("canonical_split_json")
    if canonical_path and os.path.exists(canonical_path):
        with open(canonical_path, "r") as f:
            canonical = json.load(f)
        train_speakers = set(canonical["train_val_speakers"])
        test_speakers = set(canonical["test_speakers"])
        logging.info(f"Using fixed canonical split from {canonical_path} (no SSE search)")
    else:
        X, y, groups = prepare_data(global_df)
        train_idx, test_idx = find_best_split(
            X,
            y,
            groups,
            CONFIG["test_speakers_percentage"],
            CONFIG["max_iterations"],
        )
        train_speakers = set(global_df.iloc[train_idx]["speaker_id"])
        test_speakers = set(global_df.iloc[test_idx]["speaker_id"])
        logging.warning(
            f"No canonical split file ({canonical_path}); generated a new split via "
            "SSE search. Save it as speaker_split.json to lock it for all experiments."
        )

    logging.info(f"Train/Val speakers: {train_speakers}")
    logging.info(f"Test speakers: {test_speakers}")

    save_split(global_df, frame_df, train_speakers, test_speakers, CONFIG["output_dir"])
    logging.info("All tasks completed successfully.")


if __name__ == "__main__":
    main()
