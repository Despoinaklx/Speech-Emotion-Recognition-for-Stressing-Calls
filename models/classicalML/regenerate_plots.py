"""
Regenerate confusion-matrix plots from saved experiment CSVs (no retraining).
Overwrites confusion_matrix_{raw,normalized}.png in place.

Usage (from repo root):
    python models/classicalML/regenerate_plots.py
"""
import json
import os

import numpy as np

from plotting import plot_confusion_matrix

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
EXP = os.path.join(ROOT, 'results', 'classicalML_handcrafted', 'experiments')


def regenerate(exp_dir):
    cm_raw = np.loadtxt(os.path.join(exp_dir, 'confusion_matrix_raw.csv'), delimiter=',')
    if cm_raw.ndim == 1:
        n = int(len(cm_raw) ** 0.5)
        cm_raw = cm_raw.reshape(n, n)

    with open(os.path.join(exp_dir, 'metrics.json')) as f:
        meta = json.load(f)
    class_names = meta['class_names']

    base_title = f"{meta['model']} {meta['feature_type']}-level {meta['num_classes']}-class"
    plot_confusion_matrix(
        cm_raw, class_names, f'Confusion Matrix (counts) — {base_title}',
        os.path.join(exp_dir, 'confusion_matrix_raw.png'), normalize=False,
    )
    plot_confusion_matrix(
        cm_raw, class_names, f'Confusion Matrix (normalized) — {base_title}',
        os.path.join(exp_dir, 'confusion_matrix_normalized.png'), normalize=True,
    )


if __name__ == '__main__':
    for name in sorted(os.listdir(EXP)):
        d = os.path.join(EXP, name)
        if not os.path.isdir(d):
            continue
        regenerate(d)
        print('ok ', name)
