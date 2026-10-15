import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import torch
from torch.nn.utils.rnn import pad_sequence
import sys


def prepare_hubert(hubert_npy_path, labels_csv_path, npz_out_path, config):
    """
    Build the LSTM .npz dataset from frame-level HuBERT embeddings + labels.
    'happiness' is excluded (6-class setup).

    Args:
        hubert_npy_path: folder with per-utterance .npy embeddings
        labels_csv_path: CSV with filename, speaker_id, emotion
        npz_out_path: output .npz path
        config: Config (reads MAX_SEQ_LEN, default 235)
    """
    EMBED_DIR = hubert_npy_path
    LABELS_CSV = labels_csv_path
    SAVE_PATH = npz_out_path

    MAX_SEQ_LEN = getattr(config, 'MAX_SEQ_LEN', 235)

    print(f'Loading labels from {LABELS_CSV}')
    if not os.path.exists(LABELS_CSV):
        sys.stderr.write(f"Error: labels file not found: {LABELS_CSV}\n")
        return

    df_labels = pd.read_csv(LABELS_CSV)

    # stem (no .wav) -> (speaker_id, emotion)
    labels_dict = {}
    for ix, row in df_labels.iterrows():
        stem = row['filename'].replace('.wav', '')
        labels_dict[stem] = (row['speaker_id'], row['emotion'])

    if not os.path.exists(EMBED_DIR):
        sys.stderr.write(f"Error: embeddings folder not found: {EMBED_DIR}\n")
        return

    embedding_files = sorted(f for f in os.listdir(EMBED_DIR) if f.endswith('.npy'))
    print(f'Found {len(embedding_files)} embeddings')

    sequences = []
    labels = []
    speaker_ids = []
    lengths = []
    lost = []

    for fname in embedding_files:
        # Supports both '03a01Wa.npy' and 'hubert_embedding_000_03a01Wa.npy'
        stem = fname.split('_', 3)[-1].replace('.npy', '')
        if stem not in labels_dict:
            lost.append(fname)
            continue
        speaker_id, emotion = labels_dict[stem]
        if emotion == 'happiness':
            continue
        embed_path = os.path.join(EMBED_DIR, fname)

        try:
            arr = np.load(embed_path)  # (frames, feature_dim)
        except Exception:
            lost.append(fname)
            continue

        if arr.ndim != 2:
            lost.append(fname)
            continue

        original_len = arr.shape[0]
        if original_len > MAX_SEQ_LEN:
            arr = arr[:MAX_SEQ_LEN]
        arr_tensor = torch.tensor(arr, dtype=torch.float32)
        lengths.append(min(original_len, MAX_SEQ_LEN))
        sequences.append(arr_tensor)
        labels.append(emotion)
        speaker_ids.append(speaker_id)

    # Pad to (N, MAX_SEQ_LEN, F)
    if sequences:
        padded_sequences = pad_sequence(sequences, batch_first=True)
        if padded_sequences.shape[1] < MAX_SEQ_LEN:
            pad_size = MAX_SEQ_LEN - padded_sequences.shape[1]
            padded_sequences = torch.nn.functional.pad(padded_sequences, (0, 0, 0, pad_size))
    else:
        print("No valid sequences found. Aborting.")
        return

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(labels)
    speaker_encoder = LabelEncoder()
    speaker_ids_enc = speaker_encoder.fit_transform(speaker_ids)

    np.savez(SAVE_PATH,
             X=padded_sequences.numpy(),
             y=y,
             speaker_ids=speaker_ids_enc,
             label_classes=label_encoder.classes_,
             speaker_classes=speaker_encoder.classes_,
             lengths=np.array(lengths))

    print(f'Saved dataset to {SAVE_PATH}')
    print('Sequences shape:', padded_sequences.shape)
    print('Labels shape:', len(y))
    print('Unique speakers:', np.unique(speaker_ids_enc).shape[0])
    print('Classes:', label_encoder.classes_)
    if lost:
        print('WARNING: embeddings without a label:', len(lost))
