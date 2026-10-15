"""
Speaker train/test split and LOSO fold ordering.
"""
import json
import os
import numpy as np
import pandas as pd
from collections import Counter
from itertools import permutations
from sklearn.model_selection import GroupShuffleSplit
from .data import get_speaker_emotion_profile, calculate_fold_balance_score

DEFAULT_SPEAKER_SPLIT_JSON = os.path.abspath(os.path.join(
    os.path.dirname(__file__), '..', '..', '..', 'data', 'splits', 'speaker_split.json'))


def _speaker_class_to_int(value):
    """Normalise a speaker_classes value (e.g. 3 or 'speaker_03') to the raw EmoDB id."""
    if isinstance(value, (int, np.integer)):
        return int(value)
    digits = "".join(ch for ch in str(value) if ch.isdigit())
    if not digits:
        raise ValueError(f"Could not parse a numeric speaker id from: {value!r}")
    return int(digits)


def load_fixed_speaker_split(full_dataset, json_path=DEFAULT_SPEAKER_SPLIT_JSON):
    """
    Load the shared fixed train/val - test speaker split (raw EmoDB ids), so all
    experiments use exactly the same test speakers.

    Returns (train_speakers, test_speakers) as encoded ids of this dataset,
    same format as find_best_speaker_split.
    """
    with open(json_path, "r") as f:
        canonical = json.load(f)

    raw_to_encoded = {
        _speaker_class_to_int(value): idx
        for idx, value in enumerate(full_dataset.speaker_classes)
    }

    def encode(raw_ids):
        missing = [r for r in raw_ids if r not in raw_to_encoded]
        if missing:
            raise ValueError(
                f"Speakers {missing} from the canonical split ({json_path}) are not "
                f"in this dataset's speaker_classes: {list(full_dataset.speaker_classes)}"
            )
        return sorted(raw_to_encoded[r] for r in raw_ids)

    train_speakers = encode(canonical["train_val_speakers"])
    test_speakers = encode(canonical["test_speakers"])

    print(f"Loaded fixed split from {json_path}")
    print(f"Train/Val speakers ({len(train_speakers)}, encoded): {train_speakers}")
    print(f"Test speakers ({len(test_speakers)}, encoded): {test_speakers}")

    return train_speakers, test_speakers


def find_best_speaker_split(full_dataset, test_speakers_percentage=0.20, max_iterations=100):
    """
    Search a speaker split with GroupShuffleSplit, keeping the one whose test
    emotion distribution is closest (SSE) to the overall distribution.
    """
    unique_speakers = np.unique(full_dataset.speaker_ids.numpy())
    print(f"Total speakers: {len(unique_speakers)}")

    speaker_data = []
    speaker_groups = []
    speaker_labels = []

    for speaker in unique_speakers:
        speaker_mask = full_dataset.speaker_ids == speaker
        speaker_emotions = full_dataset.y[speaker_mask].numpy()

        # One row per speaker; label = dominant emotion
        speaker_data.append([speaker])
        speaker_groups.append(speaker)
        emotion_counts = Counter(speaker_emotions)
        dominant_emotion = max(emotion_counts, key=emotion_counts.get)
        speaker_labels.append(dominant_emotion)

    X_speakers = np.array(speaker_data)
    y_speakers = pd.Series(speaker_labels)
    groups_speakers = np.array(speaker_groups)

    all_emotions = full_dataset.y.numpy()
    target_dist = pd.Series(all_emotions).value_counts(normalize=True)

    print(f"Target: {len(unique_speakers) * (1 - test_speakers_percentage):.0f} train speakers, "
          f"{len(unique_speakers) * test_speakers_percentage:.0f} test speakers")
    print(f"Running {max_iterations} iterations...")

    best_split = None
    lowest_error = float("inf")

    for i in range(max_iterations):
        splitter = GroupShuffleSplit(
            test_size=test_speakers_percentage,
            n_splits=1,
            random_state=i
        )

        train_speaker_idx, test_speaker_idx = next(
            splitter.split(X_speakers, y_speakers, groups_speakers)
        )

        test_speakers = groups_speakers[test_speaker_idx]

        test_emotion_counts = Counter()
        for speaker in test_speakers:
            speaker_mask = full_dataset.speaker_ids == speaker
            speaker_emotions = full_dataset.y[speaker_mask].numpy()
            test_emotion_counts.update(speaker_emotions)

        if sum(test_emotion_counts.values()) == 0:
            continue

        test_total = sum(test_emotion_counts.values())
        test_dist = pd.Series(dict(test_emotion_counts)) / test_total

        aligned_test, aligned_target = test_dist.align(target_dist, fill_value=0)
        error = np.sum((aligned_test - aligned_target) ** 2)

        if error < lowest_error:
            lowest_error = error
            train_speakers_best = groups_speakers[train_speaker_idx]
            test_speakers_best = test_speakers
            print(f"Iteration {i+1}: better split with SSE {error:.6f}")

    print(f"Final split SSE: {lowest_error:.6f}")

    _print_final_split_info(full_dataset, train_speakers_best, test_speakers_best)

    return sorted(train_speakers_best.tolist()), sorted(test_speakers_best.tolist())


def _print_final_split_info(full_dataset, train_speakers, test_speakers):
    train_mask = np.isin(full_dataset.speaker_ids.numpy(), train_speakers)
    test_mask = np.isin(full_dataset.speaker_ids.numpy(), test_speakers)

    train_dist = Counter(full_dataset.y[train_mask].numpy())
    test_dist = Counter(full_dataset.y[test_mask].numpy())

    print(f"\nSelected split:")
    print(f"Train speakers ({len(train_speakers)}): {sorted(train_speakers.tolist())}")
    print(f"Test speakers ({len(test_speakers)}): {sorted(test_speakers.tolist())}")
    print(f"Train emotion distribution: {dict(train_dist)}")
    print(f"Test emotion distribution: {dict(test_dist)}")


def stratified_loso_order(train_speakers, dataset):
    """
    Pick the LOSO validation-speaker order with the lowest total fold balance
    score (brute force over all permutations).
    """
    print(f"\nComputing LOSO order for {len(train_speakers)} training speakers...")

    best_order = None
    best_total_score = float('inf')

    print("Trying all permutations...")
    for order in permutations(train_speakers):
        total_score = 0
        for i, val_speaker in enumerate(order):
            current_train_speakers = [s for s in order if s != val_speaker]
            fold_score = calculate_fold_balance_score(current_train_speakers, val_speaker, dataset)
            total_score += fold_score

        if total_score < best_total_score:
            best_total_score = total_score
            best_order = list(order)

    if best_order is None:
        print("Using fallback random order")
        best_order = train_speakers.copy()
        np.random.shuffle(best_order)

    print(f"Selected LOSO order: {best_order}")
    print(f"Total balance score: {best_total_score:.4f}")

    print(f"\nBalance score per fold:")
    for i, val_speaker in enumerate(best_order):
        current_train_speakers = [s for s in best_order if s != val_speaker]
        fold_score = calculate_fold_balance_score(current_train_speakers, val_speaker, dataset)
        print(f"Fold {i+1} (val speaker {val_speaker}): {fold_score:.4f}")

    return best_order
