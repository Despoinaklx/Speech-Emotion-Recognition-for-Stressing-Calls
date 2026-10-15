#!/usr/bin/env python3
"""Two-condition demo JSON (clean vs simulated alaw_snr20) for one recording.

Usage (from repo root):
    python inference/make_demo.py path/to/audio.wav > demo.json
"""
import argparse
import contextlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import librosa
import numpy as np
import torch
from transformers import HubertModel, Wav2Vec2FeatureExtractor

from channel_degradation.telephone_channel import Condition, degrade
from models.LSTM.utils.model import LSTMClassifier


CHECKPOINT_PATH = ROOT / "models/LSTM/checkpoints/bilstm_hubert_alaw_snr20.pt"
HUBERT_MODEL_NAME = "facebook/hubert-base-ls960"
HUBERT_REVISION = "af46f65f540dc3ca7aa59f46c6c3d5dbb4374fa8"
SAMPLE_RATE = 16000
SEGMENT_SEC = 3.0
URGENT_LABELS = ["fear", "anger"]
THRESHOLDS = {"p1": 0.6, "p2": 0.4}


def load_pipeline(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = checkpoint["model_config"]
    model = LSTMClassifier(
        input_dim=config["input_dim"],
        num_classes=config["num_classes"],
        hidden_dim=config["hidden_dim"],
        num_layers=config["num_layers"],
        dropout=config["dropout"],
        bidirectional=config["bidirectional"],
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
        HUBERT_MODEL_NAME, revision=HUBERT_REVISION
    )
    hubert = HubertModel.from_pretrained(
        HUBERT_MODEL_NAME, revision=HUBERT_REVISION
    ).to(device)
    hubert.eval()
    return checkpoint, model, feature_extractor, hubert, device


def extract_embedding(segment, feature_extractor, hubert, device):
    if np.std(segment) > 0:
        segment = (segment - np.mean(segment)) / np.std(segment)
    inputs = feature_extractor(
        segment,
        sampling_rate=SAMPLE_RATE,
        padding=True,
        truncation=True,
        max_length=160000,
        return_tensors="pt",
    )
    with torch.no_grad():
        output = hubert(inputs.input_values.to(device))
    return output.last_hidden_state.squeeze(0).cpu().numpy()


def make_segments(audio):
    """3 s segments; the last one is zero-padded."""
    segment_samples = int(SEGMENT_SEC * SAMPLE_RATE)
    segments = []
    for start in range(0, len(audio), segment_samples):
        raw = audio[start:start + segment_samples]
        if len(raw) == 0:
            continue
        length = len(raw)
        if length < segment_samples:
            raw = np.pad(raw, (0, segment_samples - length))
        segments.append((start, length, raw))
    return segments


def waveform_peaks(audio, count=800):
    edges = np.linspace(0, len(audio), count + 1, dtype=int)
    peaks = [float(np.max(np.abs(audio[edges[i]:edges[i + 1]])))
             if edges[i] < edges[i + 1] else 0.0 for i in range(count)]
    peak = max(peaks, default=0.0)
    return [value / peak if peak > 0 else 0.0 for value in peaks]


def classify_segments(segments, source_audio, checkpoint, model, feature_extractor, hubert, device):
    labels = [str(label) for label in checkpoint["label_classes"]]
    scaler_mean = np.asarray(checkpoint["scaler_mean"])
    scaler_scale = np.asarray(checkpoint["scaler_scale"])
    feature_mask = np.asarray(checkpoint["feature_selector_mask"], dtype=bool)
    max_seq_len = int(checkpoint["max_seq_len"])
    peak_rms = np.sqrt(np.mean(source_audio.astype(np.float64) ** 2) + 1e-12)
    results = []

    for index, (start, actual_length, segment) in enumerate(segments):
        embedding = extract_embedding(segment, feature_extractor, hubert, device)
        embedding = embedding[:max_seq_len]
        length = embedding.shape[0]
        if length < max_seq_len:
            embedding = np.pad(embedding, ((0, max_seq_len - length), (0, 0)))
        embedding = (embedding - scaler_mean) / scaler_scale
        embedding = np.nan_to_num(embedding, nan=0.0, posinf=0.0, neginf=0.0)
        embedding = embedding[:, feature_mask]

        tensor = torch.from_numpy(embedding).float().unsqueeze(0).to(device)
        with torch.no_grad():
            probabilities = torch.softmax(model(tensor, torch.tensor([length], device=device)), dim=1)
        probabilities = probabilities.cpu().numpy()[0]
        predicted_index = int(np.argmax(probabilities))
        urgency = sum(float(probabilities[labels.index(label)]) for label in URGENT_LABELS)
        # Silence: segment RMS > 40 dB below the whole-file RMS
        segment_rms = np.sqrt(np.mean(source_audio[start:start + actual_length].astype(np.float64) ** 2) + 1e-12)
        is_silence = 20.0 * np.log10(segment_rms / peak_rms) < -40.0

        results.append({
            "index": index,
            "start_sec": start / SAMPLE_RATE,
            "end_sec": (start + actual_length) / SAMPLE_RATE,
            "probs": [float(value) for value in probabilities],
            "pred": labels[predicted_index],
            "urgency": float(urgency),
            "priority": "P1" if urgency >= THRESHOLDS["p1"] else "P2" if urgency >= THRESHOLDS["p2"] else "P3",
            "is_silence": bool(is_silence),
        })
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("audio", type=Path)
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT_PATH)
    args = parser.parse_args()

    audio, sample_rate = librosa.load(args.audio, sr=SAMPLE_RATE, mono=True)
    checkpoint, model, feature_extractor, hubert, device = load_pipeline(args.checkpoint)
    segments = make_segments(audio)
    # Simulated telephone channel (logs go to stderr, JSON to stdout)
    with contextlib.redirect_stdout(sys.stderr):
        degraded_audio, degraded_rate = degrade(
            audio, sample_rate, Condition("alaw_snr20", codec="alaw", snr_db=20),
            rng=np.random.default_rng(42),
        )
    if degraded_rate != SAMPLE_RATE:
        degraded_audio = librosa.resample(degraded_audio, orig_sr=degraded_rate, target_sr=SAMPLE_RATE)

    conditions = {
        "clean": classify_segments(segments, audio, checkpoint, model, feature_extractor, hubert, device),
        "alaw_snr20": classify_segments(
            make_segments(degraded_audio), audio, checkpoint, model, feature_extractor, hubert, device
        ),
    }
    labels = [str(label) for label in checkpoint["label_classes"]]
    output = {
        "meta": {
            "labels": labels,
            "segment_sec": SEGMENT_SEC,
            "sample_rate": SAMPLE_RATE,
            "checkpoint": str(args.checkpoint),
            "held_out_test_accuracy": float(checkpoint["held_out_test_accuracy"]),
            "held_out_test_f1_macro": float(checkpoint["held_out_test_f1_macro"]),
            "urgent_labels": URGENT_LABELS,
            "thresholds": THRESHOLDS,
        },
        "waveform_peaks": waveform_peaks(audio),
        "conditions": conditions,
    }
    print(f"classes: {labels}", file=sys.stderr)
    for condition in ("clean", "alaw_snr20"):
        for row in conditions[condition][:3]:
            print(f"{condition} probs: {row['probs']}", file=sys.stderr)
    json.dump(output, sys.stdout, ensure_ascii=True, indent=2)
    print()


if __name__ == "__main__":
    main()
