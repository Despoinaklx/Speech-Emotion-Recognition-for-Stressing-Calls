import parselmouth
import numpy as np
import librosa


class PraatExtractor:
    """
    Voice-quality and formant features (jitter, shimmer, HNR, F1-F3, bandwidths).

    - Formant settings adapt to the estimated speaker gender.
    - Bandwidth uses a 3-step fallback (Praat often returns invalid values).
    - Jitter/shimmer/HNR are computed only on voiced segments (VAD).
    """

    def __init__(self, sr, frame_length_ms=25, hop_length_ms=12.5,
                 min_pitch=75, max_pitch=500):
        """
        sr: sampling rate (Hz)
        frame_length_ms / hop_length_ms: analysis window and hop (ms)
        min_pitch / max_pitch: F0 search range (Hz)
        """
        self.sr = sr
        # Praat works in seconds
        self.frame_len = frame_length_ms / 1000.0
        self.hop_len = hop_length_ms / 1000.0
        self.min_pitch = min_pitch
        self.max_pitch = max_pitch

    # ------------------------------------------------------------------
    # 1. Voice activity detection
    # ------------------------------------------------------------------
    def _vad_segments(self, y, thresh_db=10):
        """
        Voiced segments as [(start_s, end_s), ...].

        A frame is voiced if its RMS is within `thresh_db` of the peak AND pyin
        detects pitch. (A centroid criterion is computed but intentionally unused:
        it was too strict and increased NaNs.) Falls back to the whole signal.
        """
        fl = int(0.025 * self.sr)      # 25 ms window
        hl = int(0.010 * self.sr)      # 10 ms hop

        # Energy criterion
        rms = librosa.feature.rms(y=y, frame_length=fl, hop_length=hl)[0]
        rms_db = librosa.power_to_db(rms ** 2, ref=np.max)
        energy_ok = rms_db > (np.max(rms_db) - thresh_db)

        # Unused centroid criterion (see docstring)
        cent = librosa.feature.spectral_centroid(y=y, sr=self.sr, hop_length=hl)[0]
        centroid_ok = cent < np.percentile(cent, 70)

        # Pitch criterion (pyin returns NaN on unvoiced frames)
        try:
            f0, _, _ = librosa.pyin(y, fmin=self.min_pitch, fmax=self.max_pitch,
                                    sr=self.sr, frame_length=fl, hop_length=hl)
            pitch_ok = ~np.isnan(f0)
        except Exception:
            pitch_ok = np.ones_like(energy_ok, dtype=bool)

        voiced = energy_ok & pitch_ok

        # Binary frame mask -> time segments
        bins = librosa.frames_to_time(np.arange(len(voiced)), sr=self.sr, hop_length=hl)
        active = np.concatenate(([False], voiced, [False])).astype(int)
        diff = np.diff(active)
        start_idxs = np.where(diff == 1)[0]
        end_idxs = np.where(diff == -1)[0]

        start_idxs = start_idxs[start_idxs < len(bins)]
        end_idxs = end_idxs[end_idxs < len(bins)]

        starts = bins[start_idxs]
        ends = bins[end_idxs]

        # Drop segments shorter than 50 ms
        segments = []
        for s, e in zip(starts, ends):
            if e - s >= 0.05:
                segments.append((s, e))

        return segments or [(0.0, len(y) / self.sr)]

    # ------------------------------------------------------------------
    # 2. Gender estimate -> formant parameters
    # ------------------------------------------------------------------
    def _gender_params(self, y):
        """
        Returns (max formant Hz, number of formants). Mean F0 >= 165 Hz is
        treated as female; male settings are the fallback.
        """
        try:
            snd = parselmouth.Sound(y, self.sr)
            pitch = parselmouth.praat.call(snd, "To Pitch", 0.0,
                                           self.min_pitch, self.max_pitch)
            f0 = parselmouth.praat.call(pitch, "Get mean", 0, 0, "Hertz")
        except Exception:
            f0 = None

        if f0 and f0 >= 165:
            return 5500, 5.5          # female
        return 5000, 5.0              # male / fallback

    # ------------------------------------------------------------------
    # 3. Bandwidth with 3-step fallback
    # ------------------------------------------------------------------
    def _bw_at(self, fm, no, t):
        """
        Bandwidth of formant `no` at time `t`:
          1. direct value at t
          2. median bandwidth over the whole signal
          3. empirical estimate from the formant frequency
        Returns np.nan if all fail.
        """
        try:
            bw = parselmouth.praat.call(fm, "Get bandwidth at time", no, t, 'Hertz')
            if bw and bw > 0:
                return bw
        except Exception:
            pass

        try:
            dur = parselmouth.praat.call(fm, "Get total duration")
            bw = parselmouth.praat.call(fm, "Get quantile of bandwidth",
                                        no, 0, dur, 'Hertz', 0.5)
            if bw and bw > 0:
                return bw
        except Exception:
            pass

        # Empirical: bw ~ 50 Hz + 5% of the formant frequency
        try:
            f = parselmouth.praat.call(fm, "Get value at time", no, t, 'Hertz', 'Linear')
            if f and f > 0:
                return 50.0 + (f / 1000.0) * 50.0
        except Exception:
            pass

        return np.nan

    # ------------------------------------------------------------------
    # 4. Main extraction
    # ------------------------------------------------------------------
    def _validate(self, f, bw):
        """Sanity check used only for QC counters: 200<=f<=4000, 0<bw<=f/2."""
        return (f and bw and 200 <= f <= 4000 and 0 < bw <= f * 0.5)

    def extract_features(self, y):
        """
        Returns a dict with:
          - global means: jitter_local_global, shimmer_local_global, hnr_mean_global
          - frame-level arrays: jitter_local, shimmer_local, hnr_mean, f1..f3, bw1..bw3
        """
        segs = self._vad_segments(y, thresh_db=10)
        max_form, n_form = self._gender_params(y)

        # Praat analysis objects
        snd = parselmouth.Sound(y, self.sr)
        fm = parselmouth.praat.call(
            snd, "To Formant (burg)", 0, n_form, max_form, 0.025, 50)
        pp = parselmouth.praat.call(
            snd, "To PointProcess (periodic, cc)", self.min_pitch, self.max_pitch)
        hnr = parselmouth.praat.call(
            snd, "To Harmonicity (cc)", 0.01, self.min_pitch, 0.1, 1.0)

        dur = snd.duration
        centers = np.arange(0, dur - self.frame_len, self.hop_len) + self.frame_len / 2

        lists = {k: [] for k in
                 ['j', 's', 'h', 'f1', 'f2', 'f3', 'bw1', 'bw2', 'bw3']}
        ok = {k: 0 for k in lists}  # QC counters (do not affect output)

        half = self.frame_len / 2
        for t in centers:
            st, ed = max(0, t - half), min(dur, t + half)
            voiced = any(a <= t <= b for a, b in segs)

            # Voice quality: voiced frames only
            if voiced:
                try:
                    j = parselmouth.praat.call(
                        pp, "Get jitter (local)", st, ed, 0.0001, 0.02, 1.3)
                    s = parselmouth.praat.call(
                        [snd, pp], "Get shimmer (local)",
                        st, ed, 0.0001, 0.02, 1.3, 1.6)
                    h = parselmouth.praat.call(hnr, "Get mean", st, ed)
                except Exception:
                    j = s = h = np.nan
            else:
                j = s = h = np.nan

            lists['j'].append(j)
            lists['s'].append(s)
            lists['h'].append(h)
            if not np.isnan(j): ok['j'] += 1
            if not np.isnan(s): ok['s'] += 1
            if not np.isnan(h): ok['h'] += 1

            # Formants/bandwidths: every frame
            for no, key in [(1, '1'), (2, '2'), (3, '3')]:
                try:
                    f = parselmouth.praat.call(
                        fm, "Get value at time", no, t, 'Hertz', 'Linear')
                except Exception:
                    f = np.nan
                bw = self._bw_at(fm, no, t) if f and not np.isnan(f) else np.nan
                lists[f'f{key}'].append(f)
                lists[f'bw{key}'].append(bw)
                if f and 200 <= f <= 4000: ok[f'f{key}'] += 1
                if self._validate(f, bw): ok[f'bw{key}'] += 1

        # Clean jitter/shimmer/HNR: MAD outlier removal + forward fill
        for k in ['j', 's', 'h']:
            arr = np.array(lists[k])

            if len(arr) > 5:
                med = np.nanmedian(arr)
                mad = np.nanmedian(np.abs(arr - med))
                arr[np.abs(arr - med) > 3 * mad] = np.nan

            mask = np.isnan(arr)
            if np.any(~mask):
                idx = np.where(~mask)[0]
                for i in range(1, len(arr)):
                    if mask[i]:
                        arr[i] = arr[i - 1] if not np.isnan(arr[i - 1]) else arr[idx[0]]
            lists[k] = arr

        feats = {
            # Utterance-level
            'jitter_local_global': np.nanmean(lists['j']),
            'shimmer_local_global': np.nanmean(lists['s']),
            'hnr_mean_global': np.nanmean(lists['h']),
            # Frame-level
            'jitter_local': lists['j'],
            'shimmer_local': lists['s'],
            'hnr_mean': lists['h'],
            'f1': np.array(lists['f1']),
            'f2': np.array(lists['f2']),
            'f3': np.array(lists['f3']),
            'bw1': np.array(lists['bw1']),
            'bw2': np.array(lists['bw2']),
            'bw3': np.array(lists['bw3']),
        }
        return feats
