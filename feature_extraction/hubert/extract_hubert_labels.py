# Builds hubert_labels.csv (filename, speaker_id, emotion) from EmoDB filenames.
#
# Usage (from repo root):
#   python feature_extraction/hubert/extract_hubert_labels.py

import os
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

audio_dir = os.path.join(ROOT, 'data', 'raw', 'emodb', 'wav')
output_csv = os.path.join(ROOT, 'data', 'features', 'hubert', 'hubert_labels.csv')

# Emotion code = 6th filename character (same as the handcrafted pipeline)
emotion_map = {
    'W': 'anger',
    'L': 'boredom',
    'E': 'disgust',
    'A': 'fear',
    'F': 'happiness',
    'T': 'sadness',
    'N': 'neutral'
}

os.makedirs(os.path.dirname(output_csv), exist_ok=True)

data = []

for filename in os.listdir(audio_dir):
    if filename.endswith('.wav'):
        base_name = os.path.splitext(filename)[0]

        # Valid EmoDB name: 7 characters
        if len(base_name) == 7:
            speaker_id = base_name[:2]      # e.g. '10'
            emotion_code = base_name[5]

            if emotion_code in emotion_map:
                data.append({
                    'filename': filename,
                    'speaker_id': speaker_id,
                    'emotion': emotion_map[emotion_code]
                })
            else:
                print(f"Skipping {filename}: unknown emotion code '{emotion_code}'.")
        else:
            print(f"Skipping {filename}: unexpected filename format.")

df = pd.DataFrame(data)
df.to_csv(output_csv, index=False)

print(f"Wrote '{output_csv}' with {len(df)} rows.")
