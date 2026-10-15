# Held-out test evaluation of the final early-fusion SVM pipeline
# (trained by train_early_fusion_final.py).
#
# Usage (from repo root; set TASK below):
#   python models/classicalML/test_early_fusion.py

import os
import pandas as pd
from joblib import load
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

# 'binary' or 'multi'
TASK = 'multi'

# ================================
# Paths
# ================================
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
SPLIT_DIR = os.path.join(ROOT, 'data', 'splits')
frame_file = 'frame_final_test.csv'
global_file = 'global_final_test.csv'

MODEL_DIR = os.path.join(ROOT, 'results', 'classicalML', 'EARLY_FUSION_RESULTS')
BASE_TEST_DIR = os.path.join(ROOT, 'results', 'classicalML', 'TEST_RESULTS')
RESULTS_DIR = os.path.join(BASE_TEST_DIR, 'Binary' if TASK == 'binary' else 'Multi')
os.makedirs(RESULTS_DIR, exist_ok=True)

# ================================
# Load test data (early fusion)
# ================================
df_frame = pd.read_csv(os.path.join(SPLIT_DIR, frame_file))
df_global = pd.read_csv(os.path.join(SPLIT_DIR, global_file))
df = pd.merge(df_frame, df_global, on=['filename', 'speaker_id', 'emotion'], suffixes=('_frame', '_global'))

print(f"Original test data: {len(df)} samples")
print(f"Emotion distribution: {Counter(df['emotion'])}")

# ================================
# Same label setup as training
# ================================
if TASK == 'binary':
    print("Filtering data to match training setup...")
    original_count = len(df)
    df = df[~df['emotion'].isin(['happiness'])]
    print(f"Removed {original_count - len(df)} happiness samples")

    neutral_labels = ['neutral', 'boredom']
    negative_labels = ['anger', 'sadness', 'fear', 'disgust']
    df = df[df['emotion'].isin(neutral_labels + negative_labels)]
    print(f"Final test samples: {len(df)}")
    print(f"Final emotion distribution: {Counter(df['emotion'])}")

if 'emotion_binary' not in df.columns and TASK == 'binary':
    df['emotion_binary'] = df['emotion'].apply(lambda x: 0 if x in neutral_labels else 1)
    print(f"Binary label distribution: {Counter(df['emotion_binary'])}")

# ================================
# Features
# ================================
exclude_cols = ['filename', 'frame_index', 'speaker_id', 'emotion', 'emotion_binary']
feature_cols = [col for col in df.columns if col not in exclude_cols]
X_test = df[feature_cols]
print(f"Test features: {len(feature_cols)} features, {len(X_test)} samples")

# ================================
# Load trained pipeline
# ================================
model_path = os.path.join(MODEL_DIR, 'Binary' if TASK == 'binary' else 'Multi_class',
                          'SVM', 'final_trained_pipeline.joblib')
print(f"Loading model from: {model_path}")
pipeline = load(model_path)

# Train/test feature columns must match exactly
train_features = list(pipeline.named_steps['imputer'].feature_names_in_)
test_features = list(X_test.columns)

print(f"Train features: {len(train_features)}")
print(f"Test features: {len(test_features)}")

if train_features != test_features:
    print("ERROR: Feature mismatch detected!")
    missing_in_test = set(train_features) - set(test_features)
    extra_in_test = set(test_features) - set(train_features)
    if missing_in_test:
        print(f"Missing in test: {missing_in_test}")
    if extra_in_test:
        print(f"Extra in test: {extra_in_test}")
    raise AssertionError("Feature columns do not match between train and test!")
else:
    print("Feature columns match successfully!")

print(f"Pipeline steps: {list(pipeline.named_steps.keys())}")
if 'smote' in pipeline.named_steps:
    print("SMOTE is included in the loaded pipeline")

# ================================
# Predict
# ================================
print("Making predictions...")
y_pred = pipeline.predict(X_test)

if hasattr(pipeline.named_steps['classifier'], 'predict_proba'):
    y_proba = pipeline.predict_proba(X_test)
    print(f"Prediction probabilities shape: {y_proba.shape}")
else:
    y_proba = None

print(f"Predictions distribution: {Counter(y_pred)}")

df['predicted_label'] = y_pred
if y_proba is not None:
    for i in range(y_proba.shape[1]):
        df[f'prob_class_{i}'] = y_proba[:, i]

predictions_file = os.path.join(RESULTS_DIR, f'test_predictions_{TASK}.csv')
df.to_csv(predictions_file, index=False)
print(f'Test predictions saved to {predictions_file}')

# ================================
# Ground-truth labels
# ================================
if TASK == 'binary':
    y_true = df['emotion_binary'].dropna()
    y_pred_filtered = y_pred[:len(y_true)]
    class_names = ['neutral', 'negative']
else:
    # Training label encoder
    label_encoder_path = model_path.replace('final_trained_pipeline.joblib', 'label_encoder.joblib')
    le = load(label_encoder_path)

    y_true = le.transform(df['emotion'])
    valid_mask = ~pd.isna(y_true)
    y_true = y_true[valid_mask].astype(int)
    y_pred_filtered = y_pred[valid_mask]
    class_names = list(le.classes_)

    print(f"Test label distribution: {Counter(df['emotion'])}")

# ================================
# Metrics
# ================================
if len(y_true) > 0:
    print('\n' + '=' * 60)
    print('TEST EVALUATION METRICS:')
    print('=' * 60)

    accuracy = accuracy_score(y_true, y_pred_filtered)
    f1_macro = f1_score(y_true, y_pred_filtered, average='macro')
    f1_weighted = f1_score(y_true, y_pred_filtered, average='weighted')

    print(f'Accuracy: {accuracy:.4f}')
    print(f'F1-macro: {f1_macro:.4f}')
    print(f'F1-weighted: {f1_weighted:.4f}')

    print('\nClassification Report:')
    report = classification_report(y_true, y_pred_filtered, target_names=class_names, digits=4)
    print(report)

    print('\nConfusion Matrix:')
    cm = confusion_matrix(y_true, y_pred_filtered)
    print(cm)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Test Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    confusion_matrix_path = os.path.join(RESULTS_DIR, f'test_conf_matrix_{TASK}.png')
    plt.savefig(confusion_matrix_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Confusion matrix saved to {confusion_matrix_path}")

    # F1 per class
    report_dict = classification_report(y_true, y_pred_filtered, target_names=class_names, output_dict=True)
    per_class = pd.DataFrame(report_dict).transpose().iloc[:-3]  # drop summary rows

    plt.figure(figsize=(10, 6))
    per_class['f1-score'].plot(kind='bar', color='royalblue')
    plt.ylim(0, 1)
    plt.title('Test F1-score per class')
    plt.ylabel('F1-score')
    plt.xlabel('Class')
    plt.xticks(rotation=45)
    plt.tight_layout()
    f1_plot_path = os.path.join(RESULTS_DIR, f'test_f1_per_class_{TASK}.png')
    plt.savefig(f1_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"F1-score plot saved to {f1_plot_path}")

    results_file = os.path.join(RESULTS_DIR, f'test_results_{TASK}.txt')
    with open(results_file, 'w', encoding='utf-8') as f:
        f.write(f"Test Results for {TASK} task\n")
        f.write("=" * 50 + "\n")
        f.write(f"Test samples: {len(y_true)}\n")
        f.write(f"Accuracy: {accuracy:.4f}\n")
        f.write(f"F1-macro: {f1_macro:.4f}\n")
        f.write(f"F1-weighted: {f1_weighted:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(report)
        f.write(f"\n\nConfusion Matrix:\n{cm}")

    print(f"Detailed results saved to {results_file}")

else:
    print('No valid ground truth labels found in test set. Only predictions were saved.')
