# Rhythm features (syllable rate / intervals) from onset peaks.

import librosa
import numpy as np


class RhythmExtractor:
    """
    Features:
    - syllable_rate: syllables per second
    - syllable_interval_mean / syllable_interval_std: inter-syllable intervals (s)
    """

    def __init__(self, sr=16000):
        self.sr = sr
        self._onset_env = None

    def extract_features(self, y, onset_kwargs=None, peak_pick_kwargs=None):
        """
        y: 1D audio array
        onset_kwargs: extra args for librosa.onset.onset_strength
        peak_pick_kwargs: args for librosa.util.peak_pick
        """
        if not isinstance(y, np.ndarray) or y.ndim != 1 or y.size == 0:
            raise ValueError("Input `y` must be a non-empty 1D NumPy array.")

        onset_kwargs = onset_kwargs or {}
        peak_pick_kwargs = peak_pick_kwargs or {
            "pre_max": 3, "post_max": 3, "pre_avg": 3, "post_avg": 3, "delta": 0.5, "wait": 5
        }

        try:
            # Cache onset envelope
            if not hasattr(self, '_onset_env') or self._onset_env is None:
                self._onset_env = librosa.onset.onset_strength(y=y, sr=self.sr, **onset_kwargs)
            onset_env = self._onset_env
            peaks = librosa.util.peak_pick(onset_env, **peak_pick_kwargs)
            syllable_times = librosa.frames_to_time(peaks, sr=self.sr)
            duration = librosa.get_duration(y=y, sr=self.sr)
            syllable_rate = len(syllable_times) / duration if duration > 0 else 0
            if len(syllable_times) > 1:
                syllable_intervals = np.diff(syllable_times)
                syllable_interval_mean = np.mean(syllable_intervals)
                syllable_interval_std = np.std(syllable_intervals)
            else:
                syllable_interval_mean = 0
                syllable_interval_std = 0
            return {
                'syllable_rate': syllable_rate,
                'syllable_interval_mean': syllable_interval_mean,
                'syllable_interval_std': syllable_interval_std
            }
        except Exception as e:
            raise RuntimeError(f"Error extracting rhythm features: {e}")
