"""
Builds the demo call queue (queue_calls.json) for the dashboard from a fixed set
of degraded (alaw_snr20) EmoDB clips of the held-out speakers 12/13.
Each clip is classified as a whole (no 3 s segmentation).

Reuses checkpoint loading / HuBERT extraction from models/LSTM/infer_call_emotions.py.

Usage (from repo root):
    python inference/analyze_clips.py --out queue_calls.json
"""
import argparse
import base64
import json
import os
import re
import subprocess
import sys
import tempfile

import librosa
import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from models.LSTM.infer_call_emotions import (
    CHECKPOINT_PATH,
    extract_hubert_embedding,
    load_checkpoint,
    load_hubert,
)

SR = 16000
URGENT_LABELS = ["fear", "anger"]
THRESHOLDS = {"p1": 0.60, "p2": 0.40}
CONDITION = "alaw_snr20"
CLIPS_DIR = os.path.join(ROOT, "data", "degraded", CONDITION)

# EmoDB code -> label (happiness is not one of the 6 model classes)
EMODB_CODE_TO_LABEL = {
    "W": "anger", "L": "boredom", "E": "disgust",
    "A": "fear", "T": "sadness", "N": "neutral",
}

# Held-out speakers 12/13, chosen for ground-truth balance (not by prediction outcome)
SELECTED_CLIPS = [
    "12b02Wa", "13b02Wa",  # anger
    "12a02Ac", "13a05Aa",  # fear
    "12b02Na", "13a05Nb",  # neutral
    "12a07La",             # boredom
    "13a02Ta",             # sadness
    "12b02Ea", "13a01Ea",  # disgust
]

FILENAME_RE = re.compile(r"^(\d{2})[a-z]\d{2}([A-Z])[a-z]$")


def ground_truth_label(clip_id):
    m = FILENAME_RE.match(clip_id)
    if not m:
        raise ValueError(f"Not an EmoDB filename: {clip_id}")
    code = m.group(2)
    if code not in EMODB_CODE_TO_LABEL:
        raise ValueError(f"Unknown emotion code '{code}' in {clip_id}")
    return EMODB_CODE_TO_LABEL[code]


def priority_for(urgency):
    if urgency >= THRESHOLDS["p1"]:
        return "P1"
    if urgency >= THRESHOLDS["p2"]:
        return "P2"
    return "P3"


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
    ap.add_argument("--out", default="queue_calls.json", help="output JSON")
    ap.add_argument("--clips-dir", default=CLIPS_DIR,
                    help="folder with the degraded (alaw_snr20) EmoDB clips")
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

    print(f"Checkpoint: {CHECKPOINT_PATH}")
    print(f"  condition={ckpt['condition']} | held-out test acc={ckpt['held_out_test_accuracy']:.4f} "
          f"f1_macro={ckpt['held_out_test_f1_macro']:.4f}")
    print(f"  classes ({len(class_names)}): {class_names}")

    print("Loading HuBERT feature extractor...")
    fe, hubert, hdevice = load_hubert()

    calls_out = []
    n_correct = 0
    for i, clip_id in enumerate(SELECTED_CLIPS, start=1):
        wav_path = os.path.join(args.clips_dir, f"{clip_id}.wav")
        true_label = ground_truth_label(clip_id)

        audio, sr = librosa.load(wav_path, sr=SR, mono=True)
        duration_sec = len(audio) / sr

        emb = extract_hubert_embedding(audio, fe, hubert, hdevice, sr=sr)  # (T, 768)
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
        is_correct = pred_label == true_label
        n_correct += int(is_correct)

        call_id = f"{i:02d}"
        print(f"  [{call_id}] {clip_id} true={true_label:9s} -> pred={pred_label:9s} "
              f"({'OK' if is_correct else 'X '}) urgency={urgency:.2f} {priority}")

        calls_out.append({
            "id": call_id,
            "true": true_label,
            "duration_sec": round(float(duration_sec), 2),
            "probs": [float(p) for p in probs],
            "pred": pred_label,
            "urgency": round(urgency, 6),
            "priority": priority,
            "audio_data_uri": audio_to_data_uri(wav_path),
        })

    result = {
        "meta": {
            "model": "BiLSTM / HuBERT / 6-class (tuned) -- a-law/SNR20",
            "condition": CONDITION,
            "labels": class_names,
            "urgent_labels": URGENT_LABELS,
            "thresholds": THRESHOLDS,
        },
        "calls": calls_out,
    }

    with open(args.out, "w") as f:
        json.dump(result, f, ensure_ascii=False)

    print(f"\n[DONE] -> {args.out}")
    print(f"condition={CONDITION} | clips={len(SELECTED_CLIPS)} | accuracy={n_correct}/{len(SELECTED_CLIPS)}")


if __name__ == "__main__":
    main()
