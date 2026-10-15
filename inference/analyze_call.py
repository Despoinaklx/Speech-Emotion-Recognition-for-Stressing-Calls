"""
Offline call analysis for the 112 call-prioritisation dashboard: split a call
recording into non-overlapping 3 s segments, run the deployable BiLSTM+HuBERT
checkpoint on each and write call_analysis.json.

Reuses checkpoint loading / HuBERT extraction from models/LSTM/infer_call_emotions.py
(same scaler, feature mask and label classes as training).

Usage (from repo root):
    python inference/analyze_call.py --audio path/to/call.wav --out call_analysis.json
    python inference/analyze_call.py --audio path/to/call.wav --out call_analysis.json --embed-audio
"""
import argparse
import base64
import json
import os
import subprocess
import sys
import tempfile

import librosa
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.LSTM.infer_call_emotions import (
    CHECKPOINT_PATH,
    extract_hubert_embedding,
    load_checkpoint,
    load_hubert,
)

SR = 16000
SEGMENT_SEC = 3.0
MIN_LAST_SEGMENT_SEC = 1.0
URGENT_LABELS = ["fear", "anger"]
THRESHOLDS = {"p1": 0.60, "p2": 0.40}
# Segments this many dB below the loudest one are treated as silence
SILENCE_DB_DROP = 30.0


def priority_for(urgency):
    if urgency >= THRESHOLDS["p1"]:
        return "P1"
    if urgency >= THRESHOLDS["p2"]:
        return "P2"
    return "P3"


def build_segment_bounds(duration_sec):
    bounds = []
    t = 0.0
    while t < duration_sec:
        end = min(t + SEGMENT_SEC, duration_sec)
        if end - t >= MIN_LAST_SEGMENT_SEC:
            bounds.append((t, end))
        t += SEGMENT_SEC
    return bounds


def compute_waveform_peaks(audio, duration_sec):
    """~4 peaks per second, normalised by the 99th percentile."""
    n_windows = max(1, round(duration_sec * 4))
    win = max(1, len(audio) // n_windows)
    peaks = []
    for i in range(n_windows):
        s = i * win
        e = len(audio) if i == n_windows - 1 else s + win
        seg = audio[s:e]
        peaks.append(float(np.max(np.abs(seg))) if len(seg) else 0.0)
    peaks = np.array(peaks, dtype=np.float64)
    p99 = np.percentile(peaks, 99)
    if p99 > 0:
        peaks = np.clip(peaks / p99, 0.0, 1.0)
    return peaks.tolist()


def audio_to_data_uri(audio_path):
    """Base64 mp3 data URI (converts with ffmpeg if needed)."""
    ext = os.path.splitext(audio_path)[1].lower()
    if ext == ".mp3":
        with open(audio_path, "rb") as f:
            raw = f.read()
    else:
        with tempfile.NamedTemporaryFile(suffix=".mp3") as tmp:
            subprocess.run(
                ["ffmpeg", "-y", "-i", audio_path, "-codec:a", "libmp3lame", "-b:a", "128k", tmp.name],
                check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
            with open(tmp.name, "rb") as f:
                raw = f.read()
    return "data:audio/mpeg;base64," + base64.b64encode(raw).decode("ascii")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--audio", required=True, help="call audio file (wav/mp3/...)")
    ap.add_argument("--out", default="call_analysis.json", help="output JSON")
    ap.add_argument("--embed-audio", action="store_true",
                    help="embed the audio as a base64 mp3 data URI (audio_data_uri)")
    args = ap.parse_args()

    ckpt, model, device = load_checkpoint()
    class_names = list(ckpt["label_classes"])
    scaler_mean = ckpt["scaler_mean"]
    scaler_scale = ckpt["scaler_scale"]
    feat_mask = ckpt["feature_selector_mask"]
    max_seq_len = ckpt["max_seq_len"]

    missing_urgent = [l for l in URGENT_LABELS if l not in class_names]
    if missing_urgent:
        raise ValueError(f"urgent_labels {missing_urgent} are not model classes: {class_names}")
    if "neutral" not in class_names:
        raise ValueError(f"The silence gate needs a 'neutral' class, got: {class_names}")
    neutral_idx = class_names.index("neutral")

    print(f"Checkpoint: {CHECKPOINT_PATH}")
    print(f"  condition={ckpt['condition']} | held-out test acc={ckpt['held_out_test_accuracy']:.4f} "
          f"f1_macro={ckpt['held_out_test_f1_macro']:.4f}")
    print(f"  classes ({len(class_names)}): {class_names}")

    print(f"Loading audio: {args.audio}")
    audio, sr = librosa.load(args.audio, sr=SR, mono=True)
    duration_sec = len(audio) / sr
    print(f"Duration: {duration_sec:.2f}s @ {sr}Hz")

    seg_bounds = build_segment_bounds(duration_sec)
    seg_audio = [audio[int(round(t0 * sr)):int(round(t1 * sr))] for t0, t1 in seg_bounds]
    print(f"Segments: {len(seg_bounds)} x {SEGMENT_SEC}s (non-overlapping, from t=0)")

    seg_db = []
    for seg in seg_audio:
        rms = np.sqrt(np.mean(seg.astype(np.float64) ** 2) + 1e-12)
        seg_db.append(20 * np.log10(rms + 1e-12))
    max_db = max(seg_db) if seg_db else -120.0

    print("Loading HuBERT feature extractor...")
    fe, hubert, hdevice = load_hubert()

    segments_out = []
    n_silent = 0
    for idx, ((t0, t1), seg, db) in enumerate(zip(seg_bounds, seg_audio, seg_db)):
        is_silent = db < max_db - SILENCE_DB_DROP
        if is_silent:
            # Silent segments are labelled neutral
            n_silent += 1
            probs = np.zeros(len(class_names), dtype=np.float64)
            probs[neutral_idx] = 1.0
        else:
            emb = extract_hubert_embedding(seg, fe, hubert, hdevice, sr=sr)  # (T, 768)
            T = emb.shape[0]
            if T > max_seq_len:
                emb = emb[:max_seq_len]
                T = max_seq_len
            length = max(T, 1)
            if T < max_seq_len:
                emb = np.pad(emb, ((0, max_seq_len - T), (0, 0)))

            emb_scaled = (emb - scaler_mean) / scaler_scale
            emb_scaled = np.nan_to_num(emb_scaled, nan=0.0, posinf=0.0, neginf=0.0)
            emb_sel = emb_scaled[:, feat_mask]

            x = torch.from_numpy(emb_sel).float().unsqueeze(0).to(device)
            lengths = torch.tensor([length]).to(device)
            with torch.no_grad():
                logits = model(x, lengths)
                probs = torch.softmax(logits, dim=1).cpu().numpy()[0].astype(np.float64)

        pred_idx = int(np.argmax(probs))
        pred_label = class_names[pred_idx]
        urgency = float(sum(probs[class_names.index(l)] for l in URGENT_LABELS))
        priority = priority_for(urgency)

        segments_out.append({
            "index": idx,
            "start_sec": round(float(t0), 3),
            "end_sec": round(float(t1), 3),
            "probs": [float(p) for p in probs],
            "pred": pred_label,
            "urgency": round(urgency, 6),
            "priority": priority,
            "is_silence": bool(is_silent),
        })
        if idx % 20 == 0 or idx == len(seg_bounds) - 1:
            tag = "SILENCE->neutral" if is_silent else pred_label
            print(f"  [{idx + 1}/{len(seg_bounds)}] t={t0:6.1f}-{t1:6.1f}s -> {tag} "
                  f"(urgency={urgency:.2f}, {priority})")

    print(f"Silence-gated: {n_silent}/{len(seg_bounds)} segments (>{SILENCE_DB_DROP}dB below the loudest)")

    waveform_peaks = compute_waveform_peaks(audio, duration_sec)

    result = {
        "meta": {
            "audio_filename": os.path.basename(args.audio),
            "duration_sec": round(duration_sec, 2),
            "segment_sec": SEGMENT_SEC,
            "model": (
                "BiLSTM / HuBERT / 6-class (tuned) -- deployable checkpoint "
                f"trained on the '{ckpt['condition']}' channel-degradation condition "
                f"(held-out test acc={ckpt['held_out_test_accuracy']:.3f}, "
                f"f1_macro={ckpt['held_out_test_f1_macro']:.3f})."
            ),
            "labels": class_names,
            "urgent_labels": URGENT_LABELS,
            "thresholds": THRESHOLDS,
        },
        "waveform_peaks": waveform_peaks,
        "segments": segments_out,
    }

    if args.embed_audio:
        print("Embedding audio as base64 mp3 data URI...")
        result["audio_data_uri"] = audio_to_data_uri(args.audio)

    with open(args.out, "w") as f:
        json.dump(result, f, ensure_ascii=False)
    print(f"[DONE] -> {args.out}")


if __name__ == "__main__":
    main()
