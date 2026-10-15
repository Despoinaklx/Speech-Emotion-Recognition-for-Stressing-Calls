"""
Dataset helpers (speaker emotion profiles, fold balance).
"""
import numpy as np
from collections import Counter
import torch


def get_speaker_emotion_profile(dataset, speaker):
    """Per-emotion sample fraction for one speaker; returns (profile, n_samples)."""
    speaker_mask = dataset.speaker_ids == speaker
    speaker_emotions = dataset.y[speaker_mask].numpy()
    emotion_counts = Counter(speaker_emotions)
    total_samples = len(speaker_emotions)

    emotion_profile = {}
    for emotion in range(len(dataset.label_classes)):
        emotion_profile[emotion] = emotion_counts.get(emotion, 0) / total_samples

    return emotion_profile, total_samples


def calculate_fold_balance_score(train_speakers, val_speaker, dataset):
    """L1 distance between the validation speaker's and the training emotion
    distribution (lower = better balanced)."""
    train_emotion_counts = Counter()
    for speaker in train_speakers:
        speaker_mask = dataset.speaker_ids == speaker
        speaker_emotions = dataset.y[speaker_mask].numpy()
        train_emotion_counts.update(speaker_emotions)

    train_total = sum(train_emotion_counts.values())
    if train_total == 0:
        return float('inf')

    train_percentages = {}
    for emotion in range(len(dataset.label_classes)):
        train_percentages[emotion] = train_emotion_counts.get(emotion, 0) / train_total

    val_profile, val_total = get_speaker_emotion_profile(dataset, val_speaker)

    balance_score = 0
    for emotion in range(len(dataset.label_classes)):
        train_pct = train_percentages[emotion]
        val_pct = val_profile[emotion]
        balance_score += abs(train_pct - val_pct)

    # Penalty for very small validation sets
    if val_total < 10:
        balance_score += 1.0

    return balance_score


def print_dataset_info(dataset):
    """Print per-speaker and overall emotion distribution."""
    unique_speakers = np.unique(dataset.speaker_ids.numpy())
    print(f"Total speakers: {len(unique_speakers)}")

    for speaker in unique_speakers:
        speaker_mask = dataset.speaker_ids == speaker
        speaker_emotions = dataset.y[speaker_mask].numpy()
        emotion_counts = Counter(speaker_emotions)
        total_samples = len(speaker_emotions)
        print(f"Speaker {speaker}: {dict(emotion_counts)} (total: {total_samples})")

    all_emotions = dataset.y.numpy()
    overall_dist = Counter(all_emotions)
    print(f"\nOverall emotion distribution: {dict(overall_dist)}")

    return unique_speakers, overall_dist
