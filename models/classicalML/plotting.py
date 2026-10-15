"""
Shared confusion-matrix plot (no sns.heatmap: seaborn 0.12.2 + matplotlib 3.8
only annotates the first row, https://github.com/mwaskom/seaborn/issues/3116).
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def plot_confusion_matrix(cm_raw, class_names, title, out_path, normalize):
    cm_raw = cm_raw.astype(float)

    if normalize:
        M = cm_raw / cm_raw.sum(axis=1, keepdims=True)
        txt = lambda v: f'{v:.2f}'
        vmax = 1.0
    else:
        M = cm_raw
        txt = lambda v: f'{int(v):d}'
        vmax = M.max()

    fig, ax = plt.subplots(figsize=(8, 6.5))
    im = ax.imshow(M, cmap='Blues', vmin=0, vmax=vmax)
    ax.set_xticks(range(len(class_names)), class_names, rotation=45, ha='right')
    ax.set_yticks(range(len(class_names)), class_names)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title(title)

    thr = vmax * 0.55
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(j, i, txt(M[i, j]), ha='center', va='center',
                    color='white' if M[i, j] > thr else 'black', fontsize=11)

    fig.colorbar(im, ax=ax, shrink=0.85)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
