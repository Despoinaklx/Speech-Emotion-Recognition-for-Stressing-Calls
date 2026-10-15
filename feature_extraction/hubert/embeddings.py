# HuBERT embedding extraction for EmoDB.
# Writes mean-pooled (768,) vectors for classical ML and frame-level (T, 768)
# sequences for the LSTM.
#
# Usage (from repo root):
#   python feature_extraction/hubert/embeddings.py

import torch
from transformers import HubertModel, Wav2Vec2FeatureExtractor
import numpy as np
import librosa
import glob
import os
from tqdm import tqdm
import json

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

# === Settings ===
audio_folder = os.path.join(ROOT, 'data', 'raw', 'emodb', 'wav')
model_name = "facebook/hubert-base-ls960"

output_dir = os.path.join(ROOT, 'data', 'features', 'hubert')
mean_dir = os.path.join(output_dir, 'mean_pooled')      # (768,)   -> SVM/RF
frame_dir = os.path.join(output_dir, 'frame_level')     # (T, 768) -> LSTM

os.makedirs(mean_dir, exist_ok=True)
os.makedirs(frame_dir, exist_ok=True)

# === Model ===
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = HubertModel.from_pretrained(model_name).to(device)
model.eval()
print(f"Using device: {device}")


def extract_embedding(audio_array, sampling_rate=16000, _retried=False):
    """
    Returns (mean_pooled (768,), frame_level (T, 768)) or (None, None).

    Steps: pad to >= 0.1 s, z-score normalise, truncate to 10 s,
    take HuBERT's last hidden state, mean-pool over time.
    On CUDA OOM, retries once on CPU.
    """
    try:
        if audio_array is None or len(audio_array) == 0:
            return None, None

        # Minimum input length 0.1 s
        min_length = int(sampling_rate * 0.1)
        if len(audio_array) < min_length:
            audio_array = np.pad(audio_array, (0, min_length - len(audio_array)),
                                 mode='constant')

        # Z-score normalisation
        if np.std(audio_array) > 0:
            audio_array = (audio_array - np.mean(audio_array)) / np.std(audio_array)

        # Truncate to 160000 samples = 10 s
        inputs = feature_extractor(
            audio_array,
            sampling_rate=sampling_rate,
            padding=True,
            truncation=True,
            max_length=160000,
            return_tensors='pt'
        )

        device = next(model.parameters()).device
        input_values = inputs.input_values.to(device)

        with torch.no_grad():
            outputs = model(input_values)

        last_hidden = outputs.last_hidden_state.squeeze(0)  # (T, 768)
        mean_pooled = torch.mean(last_hidden, dim=0)        # (768,)

        return mean_pooled.cpu().numpy(), last_hidden.cpu().numpy()

    except torch.cuda.OutOfMemoryError:
        if _retried:
            return None, None
        model.cpu()
        torch.cuda.empty_cache()
        return extract_embedding(audio_array, sampling_rate, _retried=True)

    except Exception:
        return None, None


# === Find audio files ===
if not os.path.exists(audio_folder):
    raise FileNotFoundError(f"Folder '{audio_folder}' does not exist.")

audio_paths = glob.glob(os.path.join(audio_folder, '*.wav'))
if not audio_paths:
    raise RuntimeError(f"No .wav files found in '{audio_folder}'.")

# === Load audio ===
list_of_audio_arrays = []
valid_paths = []
failed_files = []
for path in tqdm(audio_paths, desc="Loading audio files"):
    try:
        audio, _ = librosa.load(path, sr=16000, res_type='kaiser_fast')
        if len(audio) > 0:
            list_of_audio_arrays.append(audio)
            valid_paths.append(path)
        else:
            failed_files.append(path)
    except Exception:
        failed_files.append(path)

if not list_of_audio_arrays:
    raise RuntimeError("No valid audio file was loaded.")

# === Extract and save ===
successful_extractions = 0
for i, audio_array in tqdm(enumerate(list_of_audio_arrays),
                           desc="Extracting embeddings",
                           total=len(list_of_audio_arrays)):
    mean_emb, frame_emb = extract_embedding(audio_array)
    if mean_emb is not None:
        filename = os.path.basename(valid_paths[i]).rsplit('.', 1)[0]
        np.save(os.path.join(mean_dir, f"{filename}.npy"), mean_emb)
        np.save(os.path.join(frame_dir, f"{filename}.npy"), frame_emb)
        successful_extractions += 1
    else:
        failed_files.append(valid_paths[i])

if successful_extractions == 0:
    raise RuntimeError("No embeddings were extracted.")

# === Metadata ===
metadata = {
    'model_name': model_name,
    'pytorch_version': torch.__version__,
    'total_files': len(audio_paths),
    'successful_extractions': successful_extractions,
    'failed_files': failed_files
}
with open(os.path.join(output_dir, 'hubert-embeddings-metadata.json'),
          'w', encoding='utf-8') as f:
    json.dump(metadata, f, ensure_ascii=False, indent=4)

print("\n--- SER Embedding Extraction Summary ---")
print(f"Audio files:       {len(audio_paths)}")
print(f"Loaded:            {len(list_of_audio_arrays)}")
print(f"Embeddings:        {successful_extractions}")
print(f"Failed:            {len(failed_files)}")
print(f"Mean pooled (SVM/RF): {mean_dir}/")
print(f"Frame level (LSTM):   {frame_dir}/")
