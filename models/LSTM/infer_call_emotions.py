"""
Sliding-window emotion inference on a single recording (e.g. an emergency call)
with the deployable BiLSTM+HuBERT checkpoint (see train_deployable_bilstm.py).

Outputs:
  - call_emotion_timeline.csv: emotion, confidence, valence/arousal per window
  - valence_arousal_trajectory.png
  - emotion_timeline.png
  - analysis.txt: short textual summary

Usage (from repo root):
    python -m models.LSTM.infer_call_emotions <audio_path> <outdir>
"""
import os
import sys

import numpy as np
import pandas as pd
import librosa
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from .utils.model import LSTMClassifier

CHECKPOINT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "checkpoints", "bilstm_hubert_alaw_snr20.pt")
HUBERT_MODEL_NAME = "facebook/hubert-base-ls960"

WINDOW_SEC = 3.0
HOP_SEC = 1.5
SILENCE_DB_DROP = 30.0  # drop windows this many dB below the loudest one

# Approximate circumplex (valence, arousal) coordinates in [-1, 1]
VA_MAP = {
    'anger':    (-0.51,  0.59),
    'disgust':  (-0.60,  0.35),
    'fear':     (-0.64,  0.60),
    'sadness':  (-0.63, -0.27),
    'boredom':  (-0.65, -0.35),
    'neutral':  ( 0.00,  0.00),
    'happiness': (0.81,  0.51),
}


def load_checkpoint():
    ckpt = torch.load(CHECKPOINT_PATH, map_location='cpu', weights_only=False)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    mc = ckpt['model_config']
    model = LSTMClassifier(
        input_dim=mc['input_dim'], num_classes=mc['num_classes'],
        hidden_dim=mc['hidden_dim'], num_layers=mc['num_layers'],
        dropout=mc['dropout'], bidirectional=mc['bidirectional'],
    ).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    return ckpt, model, device


def load_hubert():
    from transformers import HubertModel, Wav2Vec2FeatureExtractor
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Pin a revision that ships safetensors (torch<2.6 blocks torch.load of .bin weights)
    safetensors_revision = "af46f65f540dc3ca7aa59f46c6c3d5dbb4374fa8"
    fe = Wav2Vec2FeatureExtractor.from_pretrained(HUBERT_MODEL_NAME, revision=safetensors_revision)
    hubert = HubertModel.from_pretrained(HUBERT_MODEL_NAME, revision=safetensors_revision).to(device)
    hubert.eval()
    return fe, hubert, device


def extract_hubert_embedding(audio_array, fe, hubert, device, sr=16000):
    if np.std(audio_array) > 0:
        audio_array = (audio_array - np.mean(audio_array)) / np.std(audio_array)
    inputs = fe(audio_array, sampling_rate=sr, padding=True, truncation=True,
                max_length=160000, return_tensors="pt")
    input_values = inputs.input_values.to(device)
    with torch.no_grad():
        out = hubert(input_values)
    return out.last_hidden_state.squeeze(0).cpu().numpy()  # (T, 768)


def main():
    audio_path = sys.argv[1]
    outdir = sys.argv[2]
    os.makedirs(outdir, exist_ok=True)

    ckpt, model, device = load_checkpoint()
    class_names = ckpt['label_classes']
    scaler_mean = ckpt['scaler_mean']
    scaler_scale = ckpt['scaler_scale']
    feat_mask = ckpt['feature_selector_mask']
    max_seq_len = ckpt['max_seq_len']

    print(f"Checkpoint: {ckpt['condition']} | held-out test acc={ckpt['held_out_test_accuracy']:.4f} "
          f"f1={ckpt['held_out_test_f1_macro']:.4f} | classes={class_names}")

    print(f"Loading audio: {audio_path}")
    audio, sr = librosa.load(audio_path, sr=16000, mono=True)
    duration = len(audio) / sr
    print(f"Duration: {duration:.1f}s @ {sr}Hz")

    win_samples = int(WINDOW_SEC * sr)
    hop_samples = int(HOP_SEC * sr)

    # Windowing + energy-based silence filter
    starts = list(range(0, max(1, len(audio) - win_samples), hop_samples))
    window_rms_db = []
    for s in starts:
        seg = audio[s:s + win_samples]
        rms = np.sqrt(np.mean(seg.astype(np.float64) ** 2) + 1e-12)
        window_rms_db.append(20 * np.log10(rms + 1e-12))
    max_db = max(window_rms_db)
    keep = [i for i, db in enumerate(window_rms_db) if db > max_db - SILENCE_DB_DROP]
    print(f"Windows: {len(starts)} total, {len(keep)} after silence filtering "
          f"(>{SILENCE_DB_DROP}dB below the loudest)")

    print("Loading HuBERT feature extractor...")
    fe, hubert, hdevice = load_hubert()

    rows = []
    for idx, wi in enumerate(keep):
        s = starts[wi]
        seg = audio[s:s + win_samples]
        t0 = s / sr

        emb = extract_hubert_embedding(seg, fe, hubert, hdevice)  # (T, 768)
        T = emb.shape[0]
        if T > max_seq_len:
            emb = emb[:max_seq_len]
            T = max_seq_len
        length = T
        if T < max_seq_len:
            emb = np.pad(emb, ((0, max_seq_len - T), (0, 0)))

        # Same preprocessing as training: scaler + feature mask
        emb_scaled = (emb - scaler_mean) / scaler_scale
        emb_scaled = np.nan_to_num(emb_scaled, nan=0.0, posinf=0.0, neginf=0.0)
        emb_sel = emb_scaled[:, feat_mask]

        x = torch.from_numpy(emb_sel).float().unsqueeze(0).to(device)  # (1, T, F)
        lengths = torch.tensor([length]).to(device)
        with torch.no_grad():
            logits = model(x, lengths)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        pred_idx = int(np.argmax(probs))
        pred_label = class_names[pred_idx]
        conf = float(probs[pred_idx])
        val, aro = VA_MAP.get(pred_label, (0.0, 0.0))

        rows.append({
            'window_idx': idx, 't_start_sec': t0, 't_end_sec': t0 + WINDOW_SEC,
            'predicted_emotion': pred_label, 'confidence': conf,
            'valence': val, 'arousal': aro,
            **{f'prob_{c}': float(probs[j]) for j, c in enumerate(class_names)},
        })
        if idx % 20 == 0:
            print(f"  [{idx+1}/{len(keep)}] t={t0:6.1f}s -> {pred_label} ({conf:.2f})")

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(outdir, 'call_emotion_timeline.csv'), index=False)
    print(f"Saved: {os.path.join(outdir, 'call_emotion_timeline.csv')} ({len(df)} rows)")

    # Plot 1: valence-arousal trajectory
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.axhline(0, color='#999', lw=1)
    ax.axvline(0, color='#999', lw=1)
    circle = plt.Circle((0, 0), 1.0, fill=False, color='#ccc', lw=1)
    ax.add_patch(circle)
    quad_labels = [
        (0.62, 0.62, 'High arousal\nPositive'), (-0.62, 0.62, 'High arousal\nNegative'),
        (-0.62, -0.62, 'Low arousal\nNegative'), (0.62, -0.62, 'Low arousal\nPositive'),
    ]
    for x, y, t in quad_labels:
        ax.text(x, y, t, ha='center', va='center', fontsize=9, color='#aaa')

    n = len(df)
    jitter = np.random.RandomState(42).normal(0, 0.02, size=(n, 2))
    xs = df['valence'].to_numpy() + jitter[:, 0]
    ys = df['arousal'].to_numpy() + jitter[:, 1]
    ax.plot(xs, ys, '-', color='#ccc', lw=0.6, alpha=0.5, zorder=1)
    sc = ax.scatter(xs, ys, c=np.arange(n), cmap='viridis', s=28, zorder=2, edgecolors='white', linewidths=0.4)
    cbar = fig.colorbar(sc, ax=ax, shrink=0.7, pad=0.02)
    cbar.set_label('Time in call (window #)')

    label_offsets = {'fear': (-55, 10), 'anger': (10, 10)}  # fear and anger nearly overlap
    for label, (v, a) in VA_MAP.items():
        if label in class_names:
            ax.scatter([v], [a], marker='x', color='black', s=60, zorder=3)
            ax.annotate(label, (v, a), textcoords="offset points",
                        xytext=label_offsets.get(label, (6, 6)), fontsize=10, fontweight='bold')

    ax.set_xlim(-1.1, 1.1); ax.set_ylim(-1.1, 1.1)
    ax.set_xlabel('Valence (negative <-> positive)')
    ax.set_ylabel('Arousal (calm <-> excited)')
    ax.set_title('Valence-arousal trajectory during the call')
    ax.set_aspect('equal')
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'valence_arousal_trajectory.png'), dpi=300)
    plt.close()

    # Plot 2: emotion timeline
    fig, ax = plt.subplots(figsize=(14, 4))
    label_to_y = {c: i for i, c in enumerate(class_names)}
    ys_t = df['predicted_emotion'].map(label_to_y)
    colors_t = plt.cm.tab10(np.linspace(0, 1, len(class_names)))
    for i, c in enumerate(class_names):
        mask = df['predicted_emotion'] == c
        ax.scatter(df.loc[mask, 't_start_sec'], ys_t[mask], color=colors_t[i], s=22, label=c)
    ax.set_yticks(range(len(class_names)))
    ax.set_yticklabels(class_names)
    ax.set_xlabel('Time in call (s)')
    ax.set_title('Predicted emotion over the call')
    ax.grid(True, axis='x', alpha=0.25)
    ax.legend(loc='upper right', bbox_to_anchor=(1.15, 1.0), fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'emotion_timeline.png'), dpi=300)
    plt.close()

    # Text summary
    dist = df['predicted_emotion'].value_counts(normalize=True) * 100
    mean_val = df['valence'].mean()
    mean_aro = df['arousal'].mean()
    high_aro_neg = df[(df['arousal'] > 0.2) & (df['valence'] < -0.2)]
    pct_distress = 100 * len(high_aro_neg) / len(df)

    lines = []
    lines.append(f"Emotion analysis -- {os.path.basename(audio_path)}")
    lines.append("=" * 60)
    lines.append(f"Call duration: {duration:.1f}s | Active (non-silent) windows: {len(df)}/{len(starts)}")
    lines.append(f"Model: BiLSTM + HuBERT trained on alaw_snr20 "
                 f"(held-out test acc={ckpt['held_out_test_accuracy']:.3f})")
    lines.append("")
    lines.append("Predicted emotion distribution (% of time):")
    for label, pct in dist.items():
        lines.append(f"  {label:10s}: {pct:5.1f}%")
    lines.append("")
    lines.append(f"Mean valence-arousal position: valence={mean_val:+.2f}, arousal={mean_aro:+.2f}")
    lines.append(f"Time in high-arousal negative states (distress/panic): {pct_distress:.1f}%")
    lines.append("")
    lines.append("NOTE: categorical predictions are mapped to fixed approximate circumplex "
                 "coordinates -- this is not continuous valence/arousal regression. The model "
                 "was trained on acted EmoDB speech, not real emergency calls, so the output is "
                 "indicative only, not clinical.")

    analysis_text = "\n".join(lines)
    with open(os.path.join(outdir, 'analysis.txt'), 'w') as f:
        f.write(analysis_text)
    print("\n" + analysis_text)

    print(f"\n[DONE] All outputs in {outdir}")


if __name__ == '__main__':
    main()
