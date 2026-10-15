"""
HuBERT frame-level extraction for one degradation condition (called by
run_degradation_experiments.py).

- parameterised in/out paths
- caching: existing .npy files are not recomputed
- writes frame-level (T, 768) only, plus the condition's hubert_labels.csv
- .npy names keep the EmoDB stem (e.g. 03a01Wa.npy)
- the model is loaded once and reused across conditions
"""

import csv
import os
import glob
import numpy as np
import librosa
import torch
from transformers import HubertModel, Wav2Vec2FeatureExtractor

MODEL_NAME = "facebook/hubert-base-ls960"

# EmoDB emotion code (6th filename character)
EMOTION_MAP = {
    "W": "anger",
    "L": "boredom",
    "E": "disgust",
    "A": "fear",
    "F": "happiness",
    "T": "sadness",
    "N": "neutral",
}

# Lazily loaded, shared across conditions
_feature_extractor = None
_model = None


def _load_model():
    global _feature_extractor, _model
    if _model is None:
        print(f"Loading HuBERT ({MODEL_NAME})...")
        _feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(MODEL_NAME)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        _model = HubertModel.from_pretrained(MODEL_NAME).to(device)
        _model.eval()
        print(f"Using device: {device}")
    return _feature_extractor, _model


def _extract_frame_embedding(audio_array, sampling_rate=16000, _retried=False):
    """Frame-level (T, 768) embedding or None (same steps as feature_extraction/hubert)."""
    feature_extractor, model = _load_model()
    try:
        if audio_array is None or len(audio_array) == 0:
            return None

        min_length = int(sampling_rate * 0.1)
        if len(audio_array) < min_length:
            audio_array = np.pad(audio_array, (0, min_length - len(audio_array)), mode="constant")

        if np.std(audio_array) > 0:
            audio_array = (audio_array - np.mean(audio_array)) / np.std(audio_array)

        inputs = feature_extractor(
            audio_array,
            sampling_rate=sampling_rate,
            padding=True,
            truncation=True,
            max_length=160000,   # 10 s
            return_tensors="pt",
        )
        device = next(model.parameters()).device
        input_values = inputs.input_values.to(device)

        with torch.no_grad():
            outputs = model(input_values)

        last_hidden = outputs.last_hidden_state.squeeze(0)  # (T, 768)
        return last_hidden.cpu().numpy()

    except torch.cuda.OutOfMemoryError:
        if _retried:
            return None
        model.cpu()
        torch.cuda.empty_cache()
        return _extract_frame_embedding(audio_array, sampling_rate, _retried=True)
    except Exception:
        return None


def _write_labels_csv(wav_dir, labels_csv_out):
    """hubert_labels.csv from the EmoDB filenames in wav_dir."""
    os.makedirs(os.path.dirname(labels_csv_out), exist_ok=True)
    rows = []
    for path in sorted(glob.glob(os.path.join(wav_dir, "*.wav"))):
        base = os.path.splitext(os.path.basename(path))[0]
        if len(base) == 7 and base[5] in EMOTION_MAP:
            rows.append((os.path.basename(path), base[:2], EMOTION_MAP[base[5]]))
        else:
            print(f"Skipping {os.path.basename(path)}: not a valid EmoDB name.")
    with open(labels_csv_out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["filename", "speaker_id", "emotion"])
        w.writerows(rows)
    print(f"Wrote labels csv ({len(rows)} rows): {labels_csv_out}")


def extract_condition(wav_dir, out_frame_dir, labels_csv_out, skip_existing=True):
    """
    Extract HuBERT frame-level embeddings for every .wav in wav_dir.

    Args:
        wav_dir: condition .wav folder
        out_frame_dir: output folder for frame-level .npy
        labels_csv_out: output hubert_labels.csv
        skip_existing: keep already extracted .npy files (cache)
    """
    os.makedirs(out_frame_dir, exist_ok=True)

    wav_paths = sorted(glob.glob(os.path.join(wav_dir, "*.wav")))
    if not wav_paths:
        raise RuntimeError(f"No .wav files found in {wav_dir}")

    # Cheap, always rewritten
    _write_labels_csv(wav_dir, labels_csv_out)

    done, skipped, failed = 0, 0, 0
    for path in wav_paths:
        stem = os.path.splitext(os.path.basename(path))[0]
        out_npy = os.path.join(out_frame_dir, f"{stem}.npy")

        if skip_existing and os.path.exists(out_npy):
            skipped += 1
            continue

        try:
            audio, _ = librosa.load(path, sr=16000, res_type="kaiser_fast")
        except Exception:
            failed += 1
            continue
        if len(audio) == 0:
            failed += 1
            continue

        frame_emb = _extract_frame_embedding(audio)
        if frame_emb is None:
            failed += 1
            continue

        np.save(out_npy, frame_emb)
        done += 1

    print(f"[extract] {os.path.basename(wav_dir)}: "
          f"new={done}, cached={skipped}, failed={failed}")
