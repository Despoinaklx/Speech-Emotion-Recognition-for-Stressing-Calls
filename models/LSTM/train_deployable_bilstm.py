"""
Train one deployable BiLSTM on the alaw_snr20 condition (A-law channel + mild
noise, closest to real phone calls) and save the full inference pipeline
(weights + scaler + feature-selection mask + label classes) in one checkpoint.

Train speakers: the 8 train/val speakers; early stopping on the 2 test speakers.
Hyperparameters: LEARNING_RATE=1e-4, TOP_FEATURES_RATIO=0.7, BIDIRECTIONAL=True.

Usage (from repo root):
    python -m models.LSTM.train_deployable_bilstm
"""
import os
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from torch.utils.data import DataLoader

from .config import Config, PROJECT_ROOT, set_seeds
from .utils.sequence_dataset import SequenceDataset
from .utils.split import load_fixed_speaker_split
from .utils.training import setup_model_and_optimizer, create_criterion, train_fold, validate_model

NPZ_PATH = os.path.join(PROJECT_ROOT, "data", "splits", "conditions", "prepared_hubert_alaw_snr20.npz")
CHECKPOINT_PATH = os.path.join(PROJECT_ROOT, "models", "LSTM", "checkpoints", "bilstm_hubert_alaw_snr20.pt")


class DeployConfig(Config):
    NPZ_PATH = NPZ_PATH
    DATA_TYPE = 'hubert'
    BIDIRECTIONAL = True
    LEARNING_RATE = 1e-4
    TOP_FEATURES_RATIO = 0.7


def main():
    set_seeds()
    config = DeployConfig()
    os.makedirs(os.path.dirname(CHECKPOINT_PATH), exist_ok=True)

    dataset = SequenceDataset(config.NPZ_PATH)
    class_names = list(dataset.label_classes)
    print(f"Dataset: {len(dataset)} utterances, {dataset.X.shape[2]}D features, "
          f"{len(class_names)} classes: {class_names}")

    train_speakers, test_speakers = load_fixed_speaker_split(dataset, config.SPEAKER_SPLIT_JSON)

    train_dataset = SequenceDataset(config.NPZ_PATH, speaker_ids=train_speakers)
    test_dataset = SequenceDataset(config.NPZ_PATH, speaker_ids=test_speakers)

    # Scaler: fit on train only
    scaler = StandardScaler()
    train_X = train_dataset.X.numpy()
    train_lengths = train_dataset.lengths.numpy()
    valid_rows = np.vstack([train_X[i, :train_lengths[i], :] for i in range(len(train_X))])
    valid_rows = np.nan_to_num(valid_rows, nan=0.0, posinf=0.0, neginf=0.0)
    scaler.fit(valid_rows)

    def apply_scaling(ds):
        X = ds.X.numpy()
        F = X.shape[-1]
        X2d = scaler.transform(X.reshape(-1, F))
        X2d = np.nan_to_num(X2d, nan=0.0, posinf=0.0, neginf=0.0)
        ds.X = torch.from_numpy(X2d.reshape(X.shape)).float()

    apply_scaling(train_dataset)
    apply_scaling(test_dataset)

    # Feature selection: fit on train only
    k_features = int(train_dataset.X.shape[-1] * config.TOP_FEATURES_RATIO)
    train_X2d = train_dataset.X.numpy().reshape(-1, train_dataset.X.shape[-1])
    train_y_rep = np.repeat(train_dataset.y.numpy(), train_dataset.X.shape[1])
    non_zero_mask = np.any(train_X2d != 0, axis=1)
    selector = SelectKBest(score_func=f_classif, k=k_features)
    selector.fit(train_X2d[non_zero_mask], train_y_rep[non_zero_mask])

    def apply_selection(ds):
        X = ds.X.numpy()
        X2d = selector.transform(X.reshape(-1, X.shape[-1]))
        ds.X = torch.from_numpy(X2d.reshape(X.shape[0], X.shape[1], -1)).float()
        ds.input_dim = ds.X.shape[-1]

    apply_selection(train_dataset)
    apply_selection(test_dataset)
    print(f"Feature selection: {selector.get_support().sum()}/{len(selector.get_support())} features")

    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True,
                              pin_memory=config.pin_memory, num_workers=config.NUM_WORKERS)
    test_loader = DataLoader(test_dataset, batch_size=getattr(config, 'VAL_BATCH_SIZE', 8), shuffle=False,
                             pin_memory=config.pin_memory, num_workers=config.NUM_WORKERS)

    model, optimizer = setup_model_and_optimizer(None, train_dataset.num_classes, config, train_dataset=train_dataset)
    criterion = create_criterion(train_dataset, train_dataset.num_classes, config.device, config)

    print("Training BiLSTM on alaw_snr20...")
    train_fold(model, train_loader, test_loader, optimizer, criterion, config)
    test_loss, test_acc, test_f1, test_preds, test_true, test_probs = validate_model(
        model, test_loader, criterion, config.device
    )
    print(f"Held-out test accuracy={test_acc:.4f} f1_macro={test_f1:.4f}")

    # Everything needed for inference on new audio
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_config': {
            'input_dim': train_dataset.X.shape[-1],
            'num_classes': train_dataset.num_classes,
            'hidden_dim': config.HIDDEN_DIM,
            'num_layers': config.NUM_LAYERS,
            'dropout': config.DROPOUT,
            'bidirectional': config.BIDIRECTIONAL,
        },
        'scaler_mean': scaler.mean_,
        'scaler_scale': scaler.scale_,
        'feature_selector_mask': selector.get_support(),
        'label_classes': class_names,
        'max_seq_len': getattr(config, 'MAX_SEQ_LEN', 235),
        'condition': 'alaw_snr20',
        'held_out_test_accuracy': test_acc,
        'held_out_test_f1_macro': test_f1,
    }, CHECKPOINT_PATH)
    print(f"[DONE] Checkpoint saved -> {CHECKPOINT_PATH}")


if __name__ == '__main__':
    main()
