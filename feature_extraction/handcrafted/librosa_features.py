# Basic spectral/prosodic features extracted with librosa.

import librosa
import numpy as np
import logging


class LibrosaExtractor:
    TOP_DB = 30             # silence threshold (dB)
    PAUSE_THRESHOLD = 0.05  # minimum pause duration (s)

    def __init__(self, sr=16000, frame_length_ms=25, hop_length_ms=12.5, n_mfcc=13):
        """
        sr: sampling rate
        frame_length_ms / hop_length_ms: frame size and hop (ms)
        n_mfcc: number of MFCCs
        """
        self.sr = sr
        self.n_mfcc = n_mfcc
        self.frame_length = int(np.ceil(sr * frame_length_ms / 1000.0))
        self.hop_length = int(np.ceil(sr * hop_length_ms / 1000.0))
        self.n_fft = self.frame_length
        # pyin needs an odd frame length
        self.pyin_frame_length = self.frame_length
        if self.pyin_frame_length % 2 == 0:
            self.pyin_frame_length += 1

    def extract_sequential_features(self, y):
        """
        Frame-level features.
        Returns np.ndarray of shape (num_frames, num_features).
        """
        if not isinstance(y, np.ndarray) or y.ndim != 1 or y.size == 0:
            raise ValueError("Input `y` must be a non-empty 1D NumPy array.")
        S = np.abs(librosa.stft(y, n_fft=self.n_fft, hop_length=self.hop_length, win_length=self.frame_length))
        S_db = librosa.power_to_db(S, ref=np.max)
        target_len = S.shape[1]  # all features are aligned to the STFT frame count

        all_features = []

        # Pitch (unvoiced frames -> 0)
        f0, _, _ = librosa.pyin(y,
                                fmin=librosa.note_to_hz('C2'),
                                fmax=librosa.note_to_hz('C7'),
                                sr=self.sr,
                                frame_length=self.pyin_frame_length,
                                hop_length=self.hop_length)
        f0 = np.nan_to_num(f0, nan=0.0)
        all_features.append(librosa.util.fix_length(f0, size=target_len))

        # RMS energy
        rms = librosa.feature.rms(y=y, frame_length=self.frame_length, hop_length=self.hop_length)[0]
        all_features.append(librosa.util.fix_length(rms, size=target_len))

        # MFCCs + deltas
        mfcc = librosa.feature.mfcc(S=S_db, sr=self.sr, n_mfcc=self.n_mfcc)
        mfcc_delta = librosa.feature.delta(mfcc)
        mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
        for arr in [mfcc, mfcc_delta, mfcc_delta2]:
            for row in arr:
                all_features.append(librosa.util.fix_length(row, size=target_len))

        # Spectral centroid, rolloff, flux
        centroid = librosa.feature.spectral_centroid(S=S, sr=self.sr, n_fft=self.n_fft, hop_length=self.hop_length)[0]
        all_features.append(librosa.util.fix_length(centroid, size=target_len))
        rolloff = librosa.feature.spectral_rolloff(S=S, sr=self.sr, n_fft=self.n_fft, hop_length=self.hop_length)[0]
        all_features.append(librosa.util.fix_length(rolloff, size=target_len))
        flux = np.sqrt(np.sum(np.diff(S, axis=1) ** 2, axis=0))
        flux = np.concatenate([[0], flux])  # pad left to match STFT frame count
        all_features.append(librosa.util.fix_length(flux, size=target_len))

        # Zero-crossing rate
        zcr = librosa.feature.zero_crossing_rate(y, frame_length=self.frame_length, hop_length=self.hop_length)[0]
        all_features.append(librosa.util.fix_length(zcr, size=target_len))

        return np.vstack(all_features).T

    def extract_summary_features(self, y, top_db=TOP_DB, pause_threshold=PAUSE_THRESHOLD):
        """
        Utterance-level features: tempo, pause_count, pause_mean, pause_std.
        """
        if not isinstance(y, np.ndarray) or y.ndim != 1 or y.size == 0:
            raise ValueError("Input `y` must be a non-empty 1D NumPy array.")

        summary_features = {}

        tempo, _ = librosa.beat.beat_track(y=y, sr=self.sr, hop_length=self.hop_length)
        summary_features['tempo'] = tempo

        # Pauses = gaps between non-silent intervals
        intervals = librosa.effects.split(y, top_db=top_db, frame_length=self.frame_length, hop_length=self.hop_length)
        if len(intervals) == 0:
            pauses = np.array([])
        else:
            starts = intervals[:, 0]
            ends = intervals[:, 1]
            pause_starts = np.concatenate(([0], ends[:-1]))
            pause_ends = starts
            pause_durations = (pause_ends - pause_starts) / self.sr
            pauses = pause_durations[pause_durations > pause_threshold]

        summary_features['pause_count'] = len(pauses)
        summary_features['pause_mean'] = np.mean(pauses) if pauses.size > 0 else 0
        summary_features['pause_std'] = np.std(pauses) if pauses.size > 0 else 0

        return summary_features

    def extract_all_features(self, y):
        """Returns (frame-level array, summary dict), or (None, None) on failure."""
        try:
            sequential = self.extract_sequential_features(y)
            summary = self.extract_summary_features(y)
        except Exception as e:
            logging.error(f"Error extracting features: {e}")
            return None, None

        return sequential, summary
