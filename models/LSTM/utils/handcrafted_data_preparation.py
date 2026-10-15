# Converts all_frame_features.csv into padded sequences for the LSTM:
# group frames per utterance, truncate/pad to MAX_SEQ_LEN, encode labels and
# speakers, save as .npz.
#
# Usage (from repo root):
#   python models/LSTM/utils/handcrafted_data_preparation.py

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import torch
from torch.nn.utils.rnn import pad_sequence

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))

CSV_PATH = os.path.join(ROOT, "data", "features", "handcrafted", "all_frame_features.csv")
SAVE_PATH = os.path.join(ROOT, "data", "splits", "prepared_data")
MAX_SEQ_LEN = 400    # frames per utterance (~95th percentile of lengths)
FEATURE_COLS = None  # None = use all feature columns

print('Loading:', CSV_PATH)
df = pd.read_csv(CSV_PATH)

exclude_cols = ['filename', 'frame_index', 'speaker_id', 'emotion']
feature_cols = [col for col in df.columns if col not in exclude_cols]
if FEATURE_COLS is not None:
    feature_cols = FEATURE_COLS

groups = df.groupby('filename')

sequences = []
labels = []
speaker_ids = []
lengths = []

for fname, group in groups:
    group_sorted = group.sort_values('frame_index')
    feats = group_sorted[feature_cols].values.astype(np.float32)  # (num_frames, num_features)
    if len(feats) > MAX_SEQ_LEN:
        feats = feats[:MAX_SEQ_LEN]
    lengths.append(len(feats))
    sequences.append(torch.tensor(feats))
    # Label/speaker are constant within an utterance
    labels.append(group_sorted['emotion'].iloc[0])
    speaker_ids.append(group_sorted['speaker_id'].iloc[0])

# Pad to (N, MAX_SEQ_LEN, F)
padded_sequences = pad_sequence(sequences, batch_first=True)
if padded_sequences.shape[1] < MAX_SEQ_LEN:
    pad_size = MAX_SEQ_LEN - padded_sequences.shape[1]
    padded_sequences = torch.nn.functional.pad(padded_sequences, (0, 0, 0, pad_size))

label_encoder = LabelEncoder()
labels_encoded = label_encoder.fit_transform(labels)
speaker_encoder = LabelEncoder()
speaker_ids_encoded = speaker_encoder.fit_transform(speaker_ids)

np.savez(SAVE_PATH,
         X=padded_sequences.numpy(),
         y=labels_encoded,
         speaker_ids=speaker_ids_encoded,
         label_classes=label_encoder.classes_,
         speaker_classes=speaker_encoder.classes_,
         lengths=np.array(lengths))

print('Sequences shape:', padded_sequences.shape)
print('Labels shape:', len(labels_encoded))
print('Speakers:', len(np.unique(speaker_ids_encoded)))
print('Classes:', label_encoder.classes_)
