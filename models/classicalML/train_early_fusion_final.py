# Early fusion (frame + global features): LOSO validation, then a final pipeline
# trained on all train/val speakers and saved for test_early_fusion.py.
#
# Usage (from repo root; edit the parameters below):
#   python models/classicalML/train_early_fusion_final.py

# ================================
# Parameters
# ================================
MODEL = 'SVM'           # 'RF' | 'SVM' | 'NB' | 'MLP'
NUM_CLASSES = 7         # 2 = binary, 7 = multi-class
PCA_N_COMPONENTS = 30   # None disables PCA

import time
import os
from collections import Counter
import numpy as np
import pandas as pd
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from joblib import dump

# ================================
# Paths
# ================================
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
SPLIT_DIR = os.path.join(ROOT, 'data', 'splits')
frame_file = 'frame_train_for_loso.csv'
global_file = 'global_train_for_loso.csv'

BASE_RESULTS_DIR = os.path.join(ROOT, 'results', 'classicalML', 'EARLY_FUSION_RESULTS')
CLASS_TYPE = "Binary" if NUM_CLASSES == 2 else "Multi_class"
RESULTS_DIR = os.path.join(BASE_RESULTS_DIR, CLASS_TYPE, MODEL)
os.makedirs(RESULTS_DIR, exist_ok=True)

# ================================
# Load and merge (early fusion)
# ================================
df_frame = pd.read_csv(os.path.join(SPLIT_DIR, frame_file))
df_global = pd.read_csv(os.path.join(SPLIT_DIR, global_file))
df = pd.merge(df_frame, df_global, on=['filename', 'speaker_id', 'emotion'], suffixes=('_frame', '_global'))

# ================================
# Labels
# ================================
if NUM_CLASSES == 2:
    # Negative vs neutral; happiness dropped
    print("\nConverting labels to Binary (Negative vs Neutral)")
    neutral_labels = ['neutral', 'boredom']
    negative_labels = ['anger', 'sadness', 'fear', 'disgust']
    df = df[~df['emotion'].isin(['happiness'])]
    df = df[df['emotion'].isin(neutral_labels + negative_labels)]
    df['emotion_binary'] = df['emotion'].apply(lambda x: 0 if x in neutral_labels else 1)
    y = df['emotion_binary']
    class_names = ['neutral', 'negative']
else:
    y = df['emotion']
    class_names = sorted(df['emotion'].unique())

# ================================
# Features
# ================================
exclude_cols = ['filename', 'frame_index', 'speaker_id', 'emotion', 'emotion_binary'] if 'emotion_binary' in df.columns else ['filename', 'frame_index', 'speaker_id', 'emotion']
feature_cols = [col for col in df.columns if col not in exclude_cols]
X = df[feature_cols]
groups = df['speaker_id']  # LOSO groups

if NUM_CLASSES == 2:
    y_encoded = y
else:
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    class_names = list(le.classes_)

X = X.reset_index(drop=True)
if isinstance(y_encoded, pd.Series):
    y_encoded = y_encoded.reset_index(drop=True)
else:
    y_encoded = pd.Series(y_encoded).reset_index(drop=True)
groups = groups.reset_index(drop=True)
print(f"\nFeatures: {len(feature_cols)} | Samples: {len(X)} | Speakers: {len(np.unique(groups))}")
print(f"Classes: {list(class_names)}")

# ================================
# Classifier
# ================================
if MODEL == 'RF':
    classifier = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
elif MODEL == 'SVM':
    classifier = SVC(kernel='rbf', probability=True, random_state=42)
elif MODEL == 'NB':
    classifier = GaussianNB()
elif MODEL == 'MLP':
    classifier = MLPClassifier(hidden_layer_sizes=(100,), max_iter=300, random_state=42)
else:
    raise ValueError(f"Unsupported MODEL: {MODEL}")


def build_pipeline(min_class_count, k_neighbors):
    """imputer -> scaler -> [PCA] -> [SMOTE] -> classifier"""
    steps = [
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler()),
    ]
    if PCA_N_COMPONENTS is not None:
        steps.append(('pca', PCA(n_components=PCA_N_COMPONENTS, random_state=42)))
    if min_class_count >= 2:
        steps.append(('smote', SMOTE(random_state=42, k_neighbors=k_neighbors)))
    else:
        print(f"Smallest class has {min_class_count} samples. Skipping SMOTE.")
    steps.append(('classifier', classifier))
    return ImbPipeline(steps)


# ================================
# LOSO validation
# ================================
logo = LeaveOneGroupOut()
n_folds = logo.get_n_splits(groups=groups)

all_preds, all_true, fold_accuracies = [], [], []
start_time = time.time()

print("\nStarting LOSO Validation...")

for fold, (train_idx, val_idx) in enumerate(logo.split(X, y_encoded, groups)):
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y_encoded[train_idx], y_encoded[val_idx]

    val_speaker = df.iloc[val_idx]['speaker_id'].unique()[0]
    print(f"\nFold {fold + 1}/{n_folds} | Validating on Speaker: {val_speaker}")

    # SMOTE k must be smaller than the smallest class
    train_class_counts = Counter(y_train)
    min_class_count = min(train_class_counts.values())
    k_neighbors = min(5, max(1, min_class_count - 1))

    pipeline = build_pipeline(min_class_count, k_neighbors)
    pipeline.fit(X_train, y_train)
    preds = pipeline.predict(X_val)

    all_preds.extend(preds)
    all_true.extend(y_val)

    acc = accuracy_score(y_val, preds)
    fold_accuracies.append(acc)
    print(f"Fold Accuracy: {acc:.4f}")

# ================================
# Validation metrics
# ================================
elapsed = time.time() - start_time
accuracy = accuracy_score(all_true, all_preds)
f1_macro = f1_score(all_true, all_preds, average='macro', zero_division=0)
f1_weighted = f1_score(all_true, all_preds, average='weighted', zero_division=0)

print(f"\n{'='*60}\nFINAL VALIDATION RESULTS\n{'='*60}")
print(f"Accuracy: {accuracy:.4f}")
print(f"F1-Macro: {f1_macro:.4f}")
print(f"F1-Weighted: {f1_weighted:.4f}")
print(f"Execution Time: {elapsed:.2f} sec")
print(f"Fold Accuracy: {np.mean(fold_accuracies):.4f} +- {np.std(fold_accuracies):.4f}")

report = classification_report(all_true, all_preds, target_names=class_names, digits=4)
print("\nClassification Report:")
print(report)

with open(os.path.join(RESULTS_DIR, 'validation_report.txt'), 'w') as f:
    f.write(f"Model: {MODEL} | NUM_CLASSES: {NUM_CLASSES} | PCA_N_COMPONENTS: {PCA_N_COMPONENTS}\n")
    f.write(f"{'='*60}\n")
    f.write(f"Accuracy: {accuracy:.4f}\n")
    f.write(f"F1-Macro: {f1_macro:.4f}\n")
    f.write(f"F1-Weighted: {f1_weighted:.4f}\n")
    f.write(f"Execution Time: {elapsed:.2f} sec\n")
    f.write(f"Fold Accuracy: {np.mean(fold_accuracies):.4f} +- {np.std(fold_accuracies):.4f}\n")
    f.write(f"{'='*60}\n\n")
    f.write("Classification Report:\n")
    f.write(report)

cm = confusion_matrix(all_true, all_preds)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names)
plt.title('Validation Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'val_conf_matrix.png'))
plt.close()

# Last LOSO-fold pipeline
model_path = os.path.join(RESULTS_DIR, 'trained_pipeline.joblib')
dump(pipeline, model_path)
print(f"\nTrained pipeline saved to: {model_path}")
print(f"\nResults saved to: {RESULTS_DIR}")

# ================================
# Final pipeline on all training data (for the held-out test set)
# ================================
print("\nTraining FINAL pipeline on ALL training data for test set evaluation...")

train_class_counts = Counter(y_encoded)
min_class_count = min(train_class_counts.values())
k_neighbors = min(5, max(1, min_class_count - 1))
if min_class_count >= 2:
    print(f"SMOTE added to final pipeline with k_neighbors={k_neighbors}")

final_pipeline = build_pipeline(min_class_count, k_neighbors)
final_pipeline.fit(X, y_encoded)

final_model_path = os.path.join(RESULTS_DIR, 'final_trained_pipeline.joblib')
dump(final_pipeline, final_model_path)
print(f"\nFINAL pipeline trained on all training data and saved to: {final_model_path}")

# Label encoder to map predictions back to names (multi-class only)
if NUM_CLASSES != 2:
    dump(le, os.path.join(RESULTS_DIR, 'label_encoder.joblib'))
