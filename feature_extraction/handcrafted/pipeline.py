# Handcrafted feature pipeline: combines all extractors and writes
# frame-level and utterance-level (global) CSVs.
#
# Usage (from repo root):
#   python feature_extraction/handcrafted/pipeline.py

import os
import numpy as np
import pandas as pd
import librosa
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

from librosa_features import LibrosaExtractor
from praat_features import PraatExtractor
from lpcc_features import LPCCExtractor
from rhythm_features import RhythmExtractor

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

# ===================== CSV schemas =====================

GLOBAL_COLS = [
    # Metadata
    'filename', 'speaker_id', 'emotion',
    # Rhythm
    'tempo', 'syllable_rate', 'pause_count', 'pause_mean', 'pause_std', 'syllable_interval_mean', 'syllable_interval_std',
    # Voice
    'jitter_local_global', 'shimmer_local_global', 'hnr_mean_global',
    'f0_mean', 'f0_std', 'f0_min', 'f0_max', 'f0_range',
    # Energy
    'energy_mean', 'energy_std', 'energy_min', 'energy_max', 'energy_range',
    # Spectral
    'spectral_centroid_mean', 'spectral_centroid_std', 'spectral_rolloff_mean', 'spectral_rolloff_std',
    'zcr_mean', 'zcr_std', 'spectral_flux_mean', 'spectral_flux_std',
    # MFCC mean/std
    *[f'mfcc_{i}_mean' for i in range(1, 14)],
    *[f'mfcc_{i}_std' for i in range(1, 14)],
    *[f'mfcc_delta_{i}_mean' for i in range(1, 14)],
    *[f'mfcc_delta_{i}_std' for i in range(1, 14)],
    *[f'mfcc_delta2_{i}_mean' for i in range(1, 14)],
    *[f'mfcc_delta2_{i}_std' for i in range(1, 14)],
    # Formants & bandwidths
    'f1_mean', 'f1_std', 'f2_mean', 'f2_std', 'f3_mean', 'f3_std',
    'bw1_mean', 'bw1_std', 'bw2_mean', 'bw2_std', 'bw3_mean', 'bw3_std',
    # LPCC mean/std
    *[f'lpcc_{i}_mean' for i in range(1, 14)],
    *[f'lpcc_{i}_std' for i in range(1, 14)]
]

FRAME_COLS = [
    # Metadata
    'filename', 'frame_index', 'speaker_id', 'emotion',
    # Voice & energy
    'f0', 'energy', 'jitter_local', 'shimmer_local', 'hnr_mean',
    # MFCCs
    *[f'mfcc_{i}' for i in range(1, 14)],
    *[f'mfcc_delta_{i}' for i in range(1, 14)],
    *[f'mfcc_delta2_{i}' for i in range(1, 14)],
    # Spectral
    'spectral_centroid', 'spectral_rolloff', 'zcr', 'spectral_flux',
    # Formants & bandwidths
    'f1', 'f2', 'f3', 'bw1', 'bw2', 'bw3',
    # LPCCs
    *[f'lpcc_{i}' for i in range(1, 14)]
]

# ===================== Pipeline =====================


class CustomFeaturePipeline:
    """Runs all extractors on a file and returns frame-level + global features."""

    def __init__(self, sr=16000, frame_length_ms=25, hop_length_ms=12.5):
        self.sr = sr
        self.hop_length = int(sr * hop_length_ms / 1000.0)
        self.frame_length = int(sr * frame_length_ms / 1000.0)
        self.librosa_ext = LibrosaExtractor(sr=sr, frame_length_ms=frame_length_ms, hop_length_ms=hop_length_ms)
        self.praat_ext = PraatExtractor(sr=sr, frame_length_ms=frame_length_ms, hop_length_ms=hop_length_ms)
        self.lpcc_ext = LPCCExtractor(sr=sr, frame_length_ms=frame_length_ms, hop_length_ms=hop_length_ms)
        self.rhythm_ext = RhythmExtractor(sr=sr)

    def process_file(self, audio_path):
        """Returns (frame-level DataFrame, global-feature dict) or (None, None)."""
        try:
            y, _ = librosa.load(audio_path, sr=self.sr)
        except Exception as e:
            print(f"Failed to load {audio_path}: {e}")
            return None, None

        sequential, summary = self.librosa_ext.extract_all_features(y)
        if sequential is None:
            print(f"librosa feature extraction failed for {audio_path}")
            return None, None
        frame_df = pd.DataFrame(sequential)

        librosa_cols = [
            'f0', 'energy',
            *[f'mfcc_{i}' for i in range(1, 14)],
            *[f'mfcc_delta_{i}' for i in range(1, 14)],
            *[f'mfcc_delta2_{i}' for i in range(1, 14)],
            'spectral_centroid', 'spectral_rolloff', 'zcr', 'spectral_flux'
        ]
        for i, name in enumerate(librosa_cols):
            if i < frame_df.shape[1]:
                frame_df.rename(columns={i: name}, inplace=True)

        praat_feats = self.praat_ext.extract_features(y)
        lpcc_feats = self.lpcc_ext.extract_features(y)
        rhythm_feats = self.rhythm_ext.extract_features(y)

        # Align Praat and LPCC frame features to the librosa frame count
        if praat_feats is not None:
            for k in ['f1', 'f2', 'f3', 'bw1', 'bw2', 'bw3', 'jitter_local', 'shimmer_local', 'hnr_mean']:
                v = praat_feats.get(k)
                if v is not None:
                    v = np.array(v)
                    v_aligned = librosa.util.fix_length(v, size=frame_df.shape[0])
                    frame_df[k] = v_aligned
        if lpcc_feats is not None:
            v = lpcc_feats.get('lpcc')
            if v is not None and v.shape[1] == 13:
                v_aligned = librosa.util.fix_length(v, size=frame_df.shape[0], axis=0)
                for idx in range(13):
                    frame_df[f'lpcc_{idx+1}'] = v_aligned[:, idx]
        # Rhythm features are global only

        # Enforce schema
        for col in FRAME_COLS:
            if col not in frame_df.columns:
                frame_df[col] = np.nan
        frame_df = frame_df[FRAME_COLS]

        # ---------- Global features ----------
        global_feats = {}
        if summary is not None:
            tempo_val = summary.get('tempo', np.nan)
            if isinstance(tempo_val, (list, np.ndarray)):
                tempo_val = float(tempo_val[0]) if len(tempo_val) > 0 else np.nan
            global_feats['tempo'] = tempo_val
            for k in ['pause_count', 'pause_mean', 'pause_std']:
                global_feats[k] = summary.get(k, np.nan)
        if rhythm_feats is not None:
            for k in ['syllable_rate', 'syllable_interval_mean', 'syllable_interval_std']:
                global_feats[k] = rhythm_feats.get(k, np.nan)
        for k in ['jitter_local_global', 'shimmer_local_global', 'hnr_mean_global']:
            global_feats[k] = praat_feats.get(k, np.nan) if praat_feats is not None else np.nan
        for f in ['f0', 'energy']:
            arr = frame_df[f].to_numpy(dtype=float)
            global_feats[f'{f}_mean'] = np.nanmean(arr)
            global_feats[f'{f}_std'] = np.nanstd(arr)
            global_feats[f'{f}_min'] = np.nanmin(arr)
            global_feats[f'{f}_max'] = np.nanmax(arr)
            global_feats[f'{f}_range'] = np.nanmax(arr) - np.nanmin(arr)
        for f in ['spectral_centroid', 'spectral_rolloff', 'zcr', 'spectral_flux']:
            arr = frame_df[f].to_numpy(dtype=float)
            global_feats[f'{f}_mean'] = np.nanmean(arr)
            global_feats[f'{f}_std'] = np.nanstd(arr)
        for base in ['mfcc', 'mfcc_delta', 'mfcc_delta2']:
            for i in range(1, 14):
                arr = frame_df[f'{base}_{i}'].to_numpy(dtype=float)
                global_feats[f'{base}_{i}_mean'] = np.nanmean(arr)
                global_feats[f'{base}_{i}_std'] = np.nanstd(arr)
        for f in ['f1', 'f2', 'f3', 'bw1', 'bw2', 'bw3']:
            arr = frame_df[f].to_numpy(dtype=float)
            global_feats[f'{f}_mean'] = np.nanmean(arr)
            global_feats[f'{f}_std'] = np.nanstd(arr)
        for i in range(1, 14):
            arr = frame_df[f'lpcc_{i}'].to_numpy(dtype=float)
            global_feats[f'lpcc_{i}_mean'] = np.nanmean(arr)
            global_feats[f'lpcc_{i}_std'] = np.nanstd(arr)

        return frame_df, global_feats


def parse_metadata_from_filename(filename):
    """speaker_id and emotion from an EmoDB filename (e.g. '03a01Wa.wav')."""
    try:
        base_name = os.path.splitext(filename)[0]
        if len(base_name) == 7:
            emotion_map = {
                'W': 'anger', 'L': 'boredom', 'E': 'disgust', 'A': 'fear',
                'F': 'happiness', 'T': 'sadness', 'N': 'neutral'
            }
            speaker_id = base_name[:2]
            emotion_code = base_name[5]
            return {'speaker_id': speaker_id, 'emotion': emotion_map.get(emotion_code, 'unknown')}
        else:
            return {'speaker_id': 'unknown', 'emotion': 'unknown'}
    except Exception:
        return {'speaker_id': 'unknown', 'emotion': 'unknown'}


if __name__ == "__main__":
    AUDIO_DIR = os.path.join(ROOT, 'data', 'raw', 'emodb', 'wav')
    RESULTS_DIR = os.path.join(ROOT, 'data', 'features', 'handcrafted')
    os.makedirs(RESULTS_DIR, exist_ok=True)

    audio_files = [f for f in os.listdir(AUDIO_DIR) if f.lower().endswith('.wav')]
    if not audio_files:
        print(f"No .wav files found in {AUDIO_DIR}")
        exit()
    pipeline = CustomFeaturePipeline(sr=16000)
    all_global_rows = []
    all_frame_rows = []
    for audio_file in tqdm(audio_files, desc="Processing audio files"):
        audio_path = os.path.join(AUDIO_DIR, audio_file)
        frame_df, global_feats = pipeline.process_file(audio_path)
        if frame_df is not None and global_feats is not None:
            metadata = parse_metadata_from_filename(audio_file)
            frame_df['filename'] = audio_file
            frame_df['frame_index'] = np.arange(len(frame_df))
            frame_df['speaker_id'] = metadata['speaker_id']
            frame_df['emotion'] = metadata['emotion']
            all_frame_rows.append(frame_df)
            global_feats['filename'] = audio_file
            global_feats['speaker_id'] = metadata['speaker_id']
            global_feats['emotion'] = metadata['emotion']
            all_global_rows.append(global_feats)

    if all_global_rows:
        global_df = pd.DataFrame(all_global_rows)
        for col in GLOBAL_COLS:
            if col not in global_df.columns:
                global_df[col] = np.nan
        global_df = global_df[GLOBAL_COLS]
        global_df.to_csv(os.path.join(RESULTS_DIR, "all_global_features.csv"), index=False)
    if all_frame_rows:
        frame_df = pd.concat(all_frame_rows, ignore_index=True)
        for col in FRAME_COLS:
            if col not in frame_df.columns:
                frame_df[col] = np.nan
        frame_df = frame_df[FRAME_COLS]
        frame_df.to_csv(os.path.join(RESULTS_DIR, "all_frame_features.csv"), index=False)
    print(f"Feature CSVs written to {RESULTS_DIR}")
