# Thesis figures for the feature-extraction chapter, from one EmoDB file and
# the same parameters as LibrosaExtractor:
#   fig1_waveform_spectrogram.png, fig2_mfcc.png, fig3_f0_contour.png, fig_f0_contour.png
#
# Usage (from repo root):
#   python scripts/plot_feature_extraction_figures.py [path/to/file.wav]

import os
import sys

import numpy as np
import librosa
import librosa.display
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Same settings as LibrosaExtractor
AUDIO_PATH = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, 'data', 'raw', 'emodb', 'wav', '03a01Wa.wav')
SR = 16000
FRAME_MS, HOP_MS = 25, 12.5
N_MFCC = 13
OUT_DIR = os.path.join(ROOT, 'results', 'figures')
os.makedirs(OUT_DIR, exist_ok=True)

frame_length = int(np.ceil(SR * FRAME_MS / 1000.0))   # 400
hop_length = int(np.ceil(SR * HOP_MS / 1000.0))       # 200
n_fft = frame_length
pyin_frame = frame_length + 1 if frame_length % 2 == 0 else frame_length  # pyin needs odd length

y, _ = librosa.load(AUDIO_PATH, sr=SR)

# 1. Waveform + spectrogram
S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length, win_length=frame_length))
S_db = librosa.amplitude_to_db(S, ref=np.max)

fig, ax = plt.subplots(2, 1, figsize=(9, 5), sharex=True)
librosa.display.waveshow(y, sr=SR, ax=ax[0], color='#1D9E75')
ax[0].set(title='Waveform', ylabel='Amplitude')
img = librosa.display.specshow(S_db, sr=SR, hop_length=hop_length,
                               x_axis='time', y_axis='hz', ax=ax[1], cmap='magma')
ax[1].set(title='Spectrogram (STFT)', ylabel='Frequency (Hz)', xlabel='Time (s)')
fig.colorbar(img, ax=ax[1], format='%+2.0f dB')
fig.tight_layout()
fig.savefig(f'{OUT_DIR}/fig1_waveform_spectrogram.png', dpi=200, bbox_inches='tight')
plt.close(fig)

# 2. MFCC heatmap, z-scored per coefficient so MFCC0 does not dominate the colour scale
mfcc = librosa.feature.mfcc(S=librosa.power_to_db(S, ref=np.max), sr=SR, n_mfcc=N_MFCC)
mfcc_norm = StandardScaler().fit_transform(mfcc.T).T

fig, ax = plt.subplots(figsize=(9, 3.2))
img = librosa.display.specshow(mfcc_norm, sr=SR, hop_length=hop_length,
                               x_axis='time', ax=ax, cmap='viridis')
ax.set(title=f'{N_MFCC} mel-frequency cepstral coefficients (MFCC)',
       ylabel='MFCC index', xlabel='Time (s)')
ax.set_yticks(range(0, N_MFCC, 2))
fig.colorbar(img, ax=ax, label='Normalised value (z-score)')
fig.tight_layout()
fig.savefig(f'{OUT_DIR}/fig2_mfcc.png', dpi=200, bbox_inches='tight')
plt.close(fig)

# 3. F0 contour (pYIN, C2-C7), NaN -> 0 as in the feature code
f0, _, _ = librosa.pyin(y, fmin=librosa.note_to_hz('C2'),
                        fmax=librosa.note_to_hz('C7'), sr=SR,
                        frame_length=pyin_frame, hop_length=hop_length)
f0_zero = np.nan_to_num(f0, nan=0.0)
times = librosa.times_like(f0, sr=SR, hop_length=hop_length)

fig, ax = plt.subplots(figsize=(9, 3.2))
ax.plot(times, f0_zero, color='#D85A30', linewidth=1.4, label='F0 (NaN→0)')
ax.set(title='Fundamental frequency F0 (pYIN)',
       ylabel='Frequency (Hz)', xlabel='Time (s)')
ax.legend(loc='upper right')
fig.tight_layout()
fig.savefig(f'{OUT_DIR}/fig3_f0_contour.png', dpi=200, bbox_inches='tight')
plt.close(fig)

# 4. F0 contour with gaps on unvoiced frames + the NaN->0 points
fig, ax = plt.subplots(figsize=(9, 3.2))
ax.plot(times, f0, color='#D85A30', linewidth=1.6, label='F0 (voiced)')
unvoiced = np.isnan(f0)
ax.plot(times[unvoiced], f0_zero[unvoiced], '.', color='#9AA0A6',
        markersize=3, label='F0 = 0 (unvoiced / pauses)')
ax.set(title='Fundamental frequency F0 (pYIN)',
       ylabel='Frequency (Hz)', xlabel='Time (s)')
ax.set_ylim(bottom=-20)  # room below 0 to show the zero points
ax.legend(loc='upper right')
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(f'{OUT_DIR}/fig_f0_contour.png', dpi=200, bbox_inches='tight')
plt.close(fig)

print(f'Figures written to {OUT_DIR}/')
