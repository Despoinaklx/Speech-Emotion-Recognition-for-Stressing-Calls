import os
import numpy as np
import torch
from torch.utils.data import Dataset


class SequenceDataset(Dataset):
    """
    PyTorch Dataset over a prepared .npz of padded sequences.

    Args:
        npz_path: path to the .npz file
        speaker_ids: optional speaker filter; encoded ids (0..S-1) and/or raw
            values from speaker_classes. None = all samples.

    Attributes:
        X [N, L, F] float, y [N], speaker_ids [N], lengths [N],
        label_classes, speaker_classes, input_dim, num_classes
    """

    def __init__(self, npz_path, speaker_ids=None):
        data = np.load(npz_path, allow_pickle=True)  # classes may be object arrays

        X_np = data['X'].astype(np.float32)
        y_np = data['y'].astype(np.int64)
        speaker_ids_np = data['speaker_ids'].astype(np.int64)
        lengths_np = data['lengths'].astype(np.int64)
        self.label_classes = data['label_classes']
        self.speaker_classes = data['speaker_classes']

        # Optional speaker filter (used for leakage-free LOSO splits)
        if speaker_ids is not None:
            speaker_classes = self.speaker_classes
            value_to_idx = {val: idx for idx, val in enumerate(speaker_classes)}
            requested_ids = list(speaker_ids if isinstance(speaker_ids, (list, tuple, np.ndarray)) else [speaker_ids])
            encoded_targets = []
            unknown_original = []
            for sid in requested_ids:
                # ints are treated as encoded ids, anything else as raw values
                if isinstance(sid, (int, np.integer)):
                    encoded_targets.append(int(sid))
                else:
                    if sid in value_to_idx:
                        encoded_targets.append(int(value_to_idx[sid]))
                    else:
                        unknown_original.append(sid)
            if unknown_original:
                raise ValueError(f"Unknown speaker IDs (original values not found in speaker_classes): {unknown_original}")
            encoded_targets = np.array(encoded_targets, dtype=np.int64)

            present_encoded = set(speaker_ids_np.tolist())
            if not any(sid in present_encoded for sid in encoded_targets.tolist()):
                raise ValueError(f"No samples found for the requested speaker_ids: {encoded_targets.tolist()}")
            mask = np.isin(speaker_ids_np, encoded_targets)
            X_np = X_np[mask]
            y_np = y_np[mask]
            speaker_ids_np = speaker_ids_np[mask]
            lengths_np = lengths_np[mask]
            if len(X_np) == 0:
                raise ValueError("Filtering by speaker_ids resulted in 0 samples. Please check the provided IDs.")

        self.X = torch.from_numpy(X_np).float()
        self.y = torch.from_numpy(y_np).long()
        self.speaker_ids = torch.from_numpy(speaker_ids_np).long()
        self.lengths = torch.from_numpy(lengths_np).long()
        self.input_dim = self.X.shape[-1] if self.X.ndim == 3 else 0
        self.num_classes = len(self.label_classes)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        """Returns (X_i [L, F], y_i, speaker_i, length_i)."""
        return self.X[idx], self.y[idx], self.speaker_ids[idx], self.lengths[idx]
