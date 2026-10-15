"""
Per-fold data preparation, training and evaluation.
"""
import torch
import torch.nn as nn
import numpy as np
import copy
from collections import Counter
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
from .sequence_dataset import SequenceDataset
from .model import LSTMClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.utils.class_weight import compute_class_weight


def apply_feature_selection(train_dataset, val_dataset, config):
    """F-test feature selection fitted on non-padding training frames."""
    original_features = train_dataset.X.shape[-1]
    k_features = int(original_features * getattr(config, 'TOP_FEATURES_RATIO', 0.7))

    print(f"Feature Selection: {original_features} -> {k_features} features")

    train_X = train_dataset.X.numpy()
    train_y = train_dataset.y.numpy()
    train_X_2d = train_X.reshape(-1, train_X.shape[-1])
    train_y_2d = np.repeat(train_y, train_X.shape[1])

    # Ignore all-zero (padding) frames
    non_zero_mask = np.any(train_X_2d != 0, axis=1)
    if non_zero_mask.any():
        train_X_filtered = train_X_2d[non_zero_mask]
        train_y_filtered = train_y_2d[non_zero_mask]

        selector = SelectKBest(score_func=f_classif, k=k_features)
        selector.fit(train_X_filtered, train_y_filtered)

        train_X_selected = selector.transform(train_X_2d).reshape(train_X.shape[0], train_X.shape[1], -1)
        val_X_selected = selector.transform(val_dataset.X.numpy().reshape(-1, val_dataset.X.shape[-1])).reshape(val_dataset.X.shape[0], val_dataset.X.shape[1], -1)

        train_dataset.X = torch.from_numpy(train_X_selected).float()
        val_dataset.X = torch.from_numpy(val_X_selected).float()
        train_dataset.selector = selector
        val_dataset.selector = selector

        print("Feature Selection completed!")
        train_dataset.input_dim = train_dataset.X.shape[-1]
        val_dataset.input_dim = val_dataset.X.shape[-1]

    return train_dataset, val_dataset


def prepare_fold_data(train_speakers, val_speaker, full_dataset, config):
    """Build scaled (and optionally feature-selected) datasets/loaders for one LOSO fold."""
    train_dataset = SequenceDataset(config.NPZ_PATH, speaker_ids=train_speakers)
    val_dataset = SequenceDataset(config.NPZ_PATH, speaker_ids=[val_speaker])

    train_emotions = Counter(train_dataset.y.numpy())
    val_emotions = Counter(val_dataset.y.numpy())
    print(f'Train emotion distribution: {dict(train_emotions)}')
    print(f'Val emotion distribution: {dict(val_emotions)}')

    # Fit the scaler on real (non-padded) training frames only
    scaler = StandardScaler()
    train_X = train_dataset.X.numpy()            # (N, T, F)
    train_lengths = train_dataset.lengths.numpy()
    valid_rows = [train_X[i, :train_lengths[i], :] for i in range(len(train_X))]
    if len(valid_rows) > 0:
        valid_rows = np.vstack(valid_rows)
    else:
        valid_rows = train_X.reshape(-1, train_X.shape[-1])
    valid_rows = np.nan_to_num(valid_rows, nan=0.0, posinf=0.0, neginf=0.0)
    scaler.fit(valid_rows)
    _apply_scaling(train_dataset, scaler)
    _apply_scaling(val_dataset, scaler)

    selector = None
    if getattr(config, 'FEATURE_SELECTION', False):
        train_dataset, val_dataset = apply_feature_selection(train_dataset, val_dataset, config)
        selector = getattr(train_dataset, 'selector', None)

    train_dataset.scaler = scaler
    val_dataset.scaler = scaler
    train_dataset.selector = selector
    val_dataset.selector = selector

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        pin_memory=config.pin_memory,
        num_workers=config.NUM_WORKERS
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=getattr(config, 'VAL_BATCH_SIZE', 8),
        shuffle=False,
        pin_memory=config.pin_memory,
        num_workers=config.NUM_WORKERS
    )

    return train_dataset, val_dataset, train_loader, val_loader


def _apply_scaling(dataset, scaler):
    """Apply a fitted scaler to a (N, T, F) dataset in place."""
    X = dataset.X.numpy()
    F = X.shape[-1]
    X2d = scaler.transform(X.reshape(-1, F))
    X2d = np.nan_to_num(X2d, nan=0.0, posinf=0.0, neginf=0.0)
    dataset.X = torch.from_numpy(X2d.reshape(X.shape)).float()


def setup_model_and_optimizer(_, num_classes, config, train_dataset):
    actual_input_dim = train_dataset.X.shape[-1]
    model = LSTMClassifier(
        input_dim=actual_input_dim,
        num_classes=num_classes,
        hidden_dim=getattr(config, 'HIDDEN_DIM', 128),
        num_layers=getattr(config, 'NUM_LAYERS', 2),
        dropout=getattr(config, 'DROPOUT', 0.3),
        bidirectional=getattr(config, 'BIDIRECTIONAL', False)
    ).to(config.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=1e-4)
    return model, optimizer


def create_criterion(train_dataset, num_classes, device, config):
    """CrossEntropyLoss with optional balanced class weights and label smoothing
    (config: USE_CLASS_WEIGHTS, LABEL_SMOOTHING)."""
    label_smoothing = getattr(config, 'LABEL_SMOOTHING', 0.0)
    use_class_weights = getattr(config, 'USE_CLASS_WEIGHTS', False)

    weight_tensor = None
    if use_class_weights:
        try:
            y = train_dataset.y.numpy()
            all_classes = np.arange(num_classes)
            weights = compute_class_weight(class_weight='balanced', classes=all_classes, y=y)
            weight_tensor = torch.tensor(weights, dtype=torch.float).to(device)
        except Exception:
            weight_tensor = None

    # label_smoothing requires PyTorch >= 1.10
    try:
        if weight_tensor is not None:
            criterion = nn.CrossEntropyLoss(weight=weight_tensor, label_smoothing=label_smoothing)
        else:
            criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    except TypeError:
        if weight_tensor is not None:
            criterion = nn.CrossEntropyLoss(weight=weight_tensor)
        else:
            criterion = nn.CrossEntropyLoss()

    return criterion


def train_one_epoch(model, train_loader, optimizer, criterion, config):
    """One training epoch; returns mean batch loss."""
    model.train()
    epoch_loss = 0.0
    num_batches = 0

    for xb, yb, _, lb in train_loader:
        xb, yb, lb = xb.to(config.device), yb.to(config.device), lb.to(config.device)

        optimizer.zero_grad()
        lb = torch.clamp(lb, min=1)

        # Skip batches with NaN/Inf inputs
        if not torch.isfinite(xb).all():
            continue

        logits = model(xb, lb)
        loss = criterion(logits, yb)

        if not torch.isfinite(loss):
            continue

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.MAX_GRAD_NORM)
        optimizer.step()

        epoch_loss += loss.item()
        num_batches += 1

    return epoch_loss / max(num_batches, 1)


def validate_model(model, val_loader, criterion, device):
    """Returns (loss, accuracy, macro-F1, preds, labels, softmax probs)."""
    model.eval()
    val_loss = 0.0
    val_preds = []
    val_true = []
    val_probs = []

    with torch.no_grad():
        for xb, yb, _, lb in val_loader:
            xb, yb, lb = xb.to(device), yb.to(device), lb.to(device)
            lb = torch.clamp(lb, min=1)

            if not torch.isfinite(xb).all():
                continue

            logits = model(xb, lb)
            loss = criterion(logits, yb)

            if torch.isfinite(loss):
                val_loss += loss.item() * xb.size(0)

            probs = torch.softmax(logits, dim=1)
            pred = torch.argmax(probs, dim=1).cpu().numpy()
            val_preds.extend(pred)
            val_true.extend(yb.cpu().numpy())
            val_probs.extend(probs.cpu().numpy())

    val_loss /= len(val_loader.dataset)
    val_acc = accuracy_score(val_true, val_preds) if val_true else 0.0
    val_f1 = f1_score(val_true, val_preds, average='macro', zero_division=0) if val_true else 0.0

    return val_loss, val_acc, val_f1, val_preds, val_true, val_probs


def train_fold(model, train_loader, val_loader, optimizer, criterion, config):
    """Train one fold with early stopping on validation macro-F1; restores the best weights."""
    best_val_metric = -np.inf
    epochs_without_improvement = 0
    best_model_state = None
    best_epoch = 0
    fold_history = {
        'train_loss': [],
        'val_loss': [],
        'val_acc': [],
        'val_f1': []
    }

    print(f'Training with early stopping (max {config.MAX_EPOCHS} epochs)...')

    # Reduce LR when validation F1 plateaus
    use_scheduler = getattr(config, 'USE_REDUCE_LR_ON_PLATEAU', True)
    scheduler = None
    if use_scheduler:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='max',
            factor=getattr(config, 'LR_REDUCE_FACTOR', 0.5),
            patience=getattr(config, 'LR_REDUCE_PATIENCE', max(1, int(config.PATIENCE / 2))),
            min_lr=getattr(config, 'MIN_LR', 1e-6)
        )

    for epoch in range(config.MAX_EPOCHS):
        avg_train_loss = train_one_epoch(model, train_loader, optimizer, criterion, config)
        val_loss, val_acc, val_f1, _, _, _ = validate_model(model, val_loader, criterion, config.device)

        fold_history['train_loss'].append(avg_train_loss)
        fold_history['val_loss'].append(val_loss)
        fold_history['val_acc'].append(val_acc)
        fold_history['val_f1'].append(val_f1)

        print(f'Epoch {epoch+1:2d}/{config.MAX_EPOCHS} | Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | Val F1: {val_f1:.4f}')

        current_metric = val_f1
        if current_metric > best_val_metric + config.MIN_DELTA:
            best_val_metric = current_metric
            epochs_without_improvement = 0
            best_model_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch + 1
            print(f' -> New best model! F1: {best_val_metric:.4f}')
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= config.PATIENCE:
                print(f'Early stopping after {epoch+1} epochs (best at epoch {best_epoch})')
                break

        if scheduler is not None:
            try:
                scheduler.step(val_f1)
            except Exception:
                pass

    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f'Loaded best model from epoch {best_epoch} (F1: {best_val_metric:.4f})')

    return fold_history, best_epoch, epoch + 1
