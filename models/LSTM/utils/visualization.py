"""
Training curves, confusion matrices and per-fold plots.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix
import os
from collections import Counter
from sklearn.utils.class_weight import compute_class_weight


def _set_style():
    try:
        plt.style.use('seaborn-v0_8')
    except Exception:
        plt.style.use('default')


def plot_training_curves(fold_histories, results_dir):
    """Fold-averaged train/val loss and validation F1 per epoch."""
    _set_style()

    if not fold_histories:
        print("No fold histories to plot")
        return

    max_epochs = max(len(h['train_loss']) for h in fold_histories)
    avg_train_loss = []
    avg_val_loss = []
    avg_val_f1 = []

    # Average over the folds that reached each epoch
    for epoch in range(max_epochs):
        train_losses_epoch = [h['train_loss'][epoch] for h in fold_histories if len(h['train_loss']) > epoch]
        val_losses_epoch = [h['val_loss'][epoch] for h in fold_histories if len(h['val_loss']) > epoch]
        val_f1s_epoch = [h['val_f1'][epoch] for h in fold_histories if len(h['val_f1']) > epoch]
        if train_losses_epoch:
            avg_train_loss.append(np.mean(train_losses_epoch))
        if val_losses_epoch:
            avg_val_loss.append(np.mean(val_losses_epoch))
        if val_f1s_epoch:
            avg_val_f1.append(np.mean(val_f1s_epoch))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    epochs = range(1, len(avg_train_loss) + 1)
    ax1.plot(epochs, avg_train_loss, 'b-', label='Training Loss', linewidth=2)
    ax1.plot(epochs[:len(avg_val_loss)], avg_val_loss, 'r-', label='Validation Loss', linewidth=2)
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training & Validation Loss Curves')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    epochs_f1 = range(1, len(avg_val_f1) + 1)
    ax2.plot(epochs_f1, avg_val_f1, 'g-', label='Validation F1-Macro', linewidth=2)
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('F1-Score')
    ax2.set_title('Validation F1-Score Curve')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'training_curves_8fold.png'), dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Training curves saved to: {os.path.join(results_dir, 'training_curves_8fold.png')}")


def plot_confusion_matrix(y_true, y_pred, class_names, results_dir, title_suffix=""):
    """Raw and row-normalised confusion matrix heatmaps."""
    _set_style()

    cm = confusion_matrix(y_true, y_pred)
    cm_normalized = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-8)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                ax=ax1, cbar_kws={'label': 'Count'})
    ax1.set_title(f'Confusion Matrix (Raw Counts){title_suffix}', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Predicted Label')
    ax1.set_ylabel('True Label')

    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Oranges',
                xticklabels=class_names, yticklabels=class_names,
                ax=ax2, cbar_kws={'label': 'Percentage'})
    ax2.set_title(f'Confusion Matrix (Normalized){title_suffix}', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Predicted Label')
    ax2.set_ylabel('True Label')

    plt.tight_layout()

    filename = f'confusion_matrix{title_suffix.lower().replace(" ", "_").replace("(", "").replace(")", "")}_8fold.png'
    plt.savefig(os.path.join(results_dir, filename), dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Confusion matrix saved to: {os.path.join(results_dir, filename)}")


def plot_fold_performance(fold_results, results_dir):
    """Accuracy and F1 bar chart per fold."""
    _set_style()

    folds = [f"Fold {r['fold']}" for r in fold_results]
    accuracies = [r['final_accuracy'] for r in fold_results]
    f1_scores = [r['final_f1_macro'] for r in fold_results]

    x = np.arange(len(folds))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))

    ax.bar(x - width/2, accuracies, width, label='Accuracy', alpha=0.8)
    ax.bar(x + width/2, f1_scores, width, label='F1-Score', alpha=0.8)

    ax.set_xlabel('Folds')
    ax.set_ylabel('Score')
    ax.set_title('Performance Comparison Across 8-Fold LOSO')
    ax.set_xticks(x)
    ax.set_xticklabels(folds, rotation=45)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'fold_performance_8fold.png'), dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Fold performance plot saved to: {os.path.join(results_dir, 'fold_performance_8fold.png')}")


def check_class_balance(y, class_names=None, results_dir=None, filename_prefix='class_balance'):
    """
    Print class counts / imbalance ratio / suggested class weights and
    optionally save a bar plot. Returns a stats dict.
    """
    if hasattr(y, 'cpu') and hasattr(y, 'numpy'):
        try:
            y_np = y.cpu().numpy()
        except Exception:
            y_np = y.numpy()
    else:
        y_np = np.array(y)
    y_np = y_np.astype(int)

    counter = Counter(y_np.tolist())
    total = len(y_np)
    counts = {int(k): int(v) for k, v in sorted(counter.items())}
    percentages = {k: v / total for k, v in counts.items()}

    if class_names is None:
        class_names = [str(k) for k in range(max(counts.keys()) + 1)]

    if counts:
        max_count = max(counts.values())
        min_count = min(counts.values())
        imbalance_ratio = float(max_count) / float(min_count) if min_count > 0 else float('inf')
    else:
        max_count = min_count = imbalance_ratio = 0

    unique_classes = sorted(list(counts.keys()))
    try:
        weights = compute_class_weight(class_weight='balanced', classes=np.array(unique_classes), y=y_np)
        class_weights = {int(c): float(w) for c, w in zip(unique_classes, weights)}
    except Exception:
        class_weights = {int(c): 1.0 for c in unique_classes}

    print("Class balance summary:")
    for cls in unique_classes:
        name = class_names[cls] if cls < len(class_names) else str(cls)
        print(f"  Class {cls} ({name}): {counts[cls]} samples ({percentages[cls]*100:.2f}%)")
    print(f"  Total samples: {total}")
    print(f"  Max/min counts: {max_count}/{min_count}")
    print(f"  Imbalance ratio (max/min): {imbalance_ratio:.2f}")
    print(f"  Suggested class weights (sklearn): {class_weights}")

    try:
        fig, ax = plt.subplots(figsize=(8, 4))
        labels = [class_names[c] if c < len(class_names) else str(c) for c in unique_classes]
        vals = [counts[c] for c in unique_classes]
        ax.bar(labels, vals, color='C0', alpha=0.8)
        ax.set_xlabel('Class')
        ax.set_ylabel('Count')
        ax.set_title('Class Distribution')
        for i, v in enumerate(vals):
            ax.text(i, v + max(vals) * 0.01, str(v), ha='center', va='bottom')
        plt.tight_layout()
        if results_dir:
            os.makedirs(results_dir, exist_ok=True)
            path = os.path.join(results_dir, f"{filename_prefix}.png")
            plt.savefig(path, dpi=200, bbox_inches='tight')
            print(f"Class distribution plot saved to: {path}")
        plt.close()
    except Exception as e:
        print(f"Could not create/save class distribution plot: {e}")

    return {
        'counts': counts,
        'percentages': percentages,
        'total': total,
        'max_count': max_count,
        'min_count': min_count,
        'imbalance_ratio': imbalance_ratio,
        'class_weights': class_weights,
    }
