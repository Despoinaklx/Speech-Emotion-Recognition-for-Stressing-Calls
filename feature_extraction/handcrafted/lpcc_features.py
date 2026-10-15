# LPCC feature extraction.

import numpy as np
import librosa
from scipy.signal.windows import hamming


class LPCCExtractor:
    """Frame-level LPCC extractor."""

    def __init__(self, sr, frame_length_ms=25, hop_length_ms=12.5, order=12, num_coeffs=13):
        """
        sr: sampling rate (Hz)
        frame_length_ms / hop_length_ms: frame size and hop (ms)
        order: LPC order
        num_coeffs: number of LPCC coefficients
        """
        self.sr = sr
        self.frame_length = int(np.ceil(sr * frame_length_ms / 1000.0))
        self.hop_length = int(np.ceil(sr * hop_length_ms / 1000.0))
        self.order = order
        self.num_coeffs = num_coeffs
        self.window = hamming(self.frame_length)

    def lpc_coefficients(self, signal):
        """LPC coefficients of a frame (zeros for silent/failed frames)."""
        energy = np.sum(signal ** 2)
        if energy < 1e-10:
            return np.zeros(self.order + 1)
        try:
            try:
                lpc = librosa.lpc(signal, order=self.order)
            except TypeError:
                lpc = librosa.lpc(signal, self.order)  # older librosa API
            return lpc
        except Exception:
            return np.zeros(self.order + 1)

    def lpcc_from_lpc(self, lpc):
        """LPC -> LPCC via the standard recursion."""
        order = len(lpc) - 1
        lpcc = np.zeros(self.num_coeffs)
        lpcc[0] = np.log(lpc[0]) if lpc[0] > 0 else 0
        for n in range(1, self.num_coeffs):
            s = 0.0
            kmax = min(n, order)
            for k in range(1, kmax + 1):
                s += (k / n) * lpc[k] * lpcc[n - k]
            if n <= order:
                lpcc[n] = lpc[n] + s
            else:
                lpcc[n] = s
        return lpcc

    def extract_features(self, y):
        """Returns {'lpcc': array (num_frames, num_coeffs)}."""
        if not isinstance(y, np.ndarray) or y.size == 0:
            raise ValueError('Input signal must be a non-empty numpy array')

        frames = librosa.util.frame(y, frame_length=self.frame_length, hop_length=self.hop_length).T
        num_frames = frames.shape[0]
        features = np.zeros((num_frames, self.num_coeffs))

        for i, frame in enumerate(frames):
            windowed_frame = frame * self.window
            lpc = self.lpc_coefficients(windowed_frame)
            lpcc = self.lpcc_from_lpc(lpc)
            features[i, :] = lpcc[:self.num_coeffs]
        return {'lpcc': features}
