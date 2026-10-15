#!/usr/bin/env python3
"""
Telephone-channel degradation of clean EmoDB speech.

Signal chain:
    clean 16 kHz -> IRS handset shaping (optional) -> 300-3400 Hz band-pass
    -> 8 kHz -> additive noise at target SNR (optional)
    -> codec round-trip (G.711 A-law | GSM 06.10) -> soft limiter (optional)
    -> resample to out_sr (16 kHz by default)

Notes:
* Codecs are applied as a real encode/decode round-trip (libsndfile, ffmpeg fallback).
* SNR is measured against active-speech power (simple energy gate, not ITU-T P.56).
* Noise is added before the codec (caller-side environment noise).
* The IRS mask is an approximation of ITU-T P.830.

Usage:
    python channel_degradation/telephone_channel.py data/raw/emodb/wav data/degraded
"""

from __future__ import annotations

import csv
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from scipy.io import wavfile
from scipy.signal import firwin2, butter, sosfiltfilt, resample_poly


# EmoDB emotion code (6th filename character)
EMODB_EMOTION = {
    "W": "anger",       # Wut
    "L": "boredom",     # Langeweile
    "E": "disgust",     # Ekel
    "A": "fear",        # Angst
    "F": "happiness",   # Freude
    "T": "sadness",     # Trauer
    "N": "neutral",     # Neutral
}


def emodb_label(filename: str) -> Optional[str]:
    """Emotion for an EmoDB filename like '03a01Wa.wav', or None."""
    stem = Path(filename).stem
    if len(stem) >= 6 and stem[5] in EMODB_EMOTION:
        return EMODB_EMOTION[stem[5]]
    return None


# --------------------------------------------------------------------------- #
# Handset shaping: approximate modified-IRS send mask (Hz, dB).
# --------------------------------------------------------------------------- #
_IRS_SEND_HZ = np.array([0, 100, 200, 300, 400, 500, 700, 1000,
                         1500, 2000, 2500, 3000, 3300, 3400, 3500, 4000])
_IRS_SEND_DB = np.array([-60, -40, -18, -8, -4, -1.5, 0.0, 0.0,
                         0.8, 1.5, 2.0, 1.5, 0.0, -3.0, -25, -60])


def design_irs_send(sr: int, numtaps: int = 513) -> np.ndarray:
    """Linear-phase FIR approximating the IRS send response."""
    nyq = sr / 2.0
    freqs = np.clip(_IRS_SEND_HZ, 0, nyq)
    # firwin2 needs a grid ending at Nyquist
    if freqs[-1] < nyq:
        freqs = np.append(freqs, nyq)
        gains_db = np.append(_IRS_SEND_DB, -60.0)
    else:
        gains_db = _IRS_SEND_DB.copy()
    freqs, idx = np.unique(freqs, return_index=True)
    gains = 10.0 ** (gains_db[idx] / 20.0)
    return firwin2(numtaps, freqs / nyq, gains)


def apply_fir(x: np.ndarray, taps: np.ndarray) -> np.ndarray:
    from scipy.signal import filtfilt
    return filtfilt(taps, [1.0], x)


def telephone_bandpass(x: np.ndarray, sr: int,
                       low: float = 300.0, high: float = 3400.0,
                       order: int = 8) -> np.ndarray:
    nyq = sr / 2.0
    high = min(high, nyq * 0.999)
    sos = butter(order, [low / nyq, high / nyq], btype="bandpass", output="sos")
    return sosfiltfilt(sos, x)


def resample_to(x: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    if sr_in == sr_out:
        return x
    g = np.gcd(int(sr_in), int(sr_out))
    return resample_poly(x, sr_out // g, sr_in // g)


# --------------------------------------------------------------------------- #
# Noise + SNR mixing
# --------------------------------------------------------------------------- #
def _active_speech_power(x: np.ndarray, sr: int,
                         frame_ms: float = 20.0,
                         floor_db: float = -30.0) -> float:
    """Mean power of frames within `floor_db` of the loudest frame."""
    n = max(1, int(sr * frame_ms / 1000.0))
    pad = (-len(x)) % n
    xp = np.append(x, np.zeros(pad))
    frames = xp.reshape(-1, n)
    fp = np.mean(frames ** 2, axis=1)
    if fp.max() <= 0:
        return float(np.mean(x ** 2) + 1e-12)
    thresh = fp.max() * 10 ** (floor_db / 10.0)
    active = fp[fp >= thresh]
    return float(active.mean()) if active.size else float(fp.mean())


def generate_noise(n: int, kind: str, rng: np.random.Generator) -> np.ndarray:
    """Unit-RMS white or pink noise."""
    w = rng.standard_normal(n)
    if kind == "white":
        out = w
    elif kind == "pink":
        X = np.fft.rfft(w)
        f = np.arange(X.size)
        f[0] = 1
        out = np.fft.irfft(X / np.sqrt(f), n=n)
    else:
        raise ValueError(f"unknown noise kind: {kind}")
    return out / (np.sqrt(np.mean(out ** 2)) + 1e-12)


def add_noise_at_snr(x: np.ndarray, sr: int, snr_db: float,
                     kind: str = "pink",
                     noise_path: Optional[str] = None,
                     rng: Optional[np.random.Generator] = None) -> np.ndarray:
    rng = rng or np.random.default_rng()
    if noise_path:
        # Real noise recording: tile/crop with a random offset
        nsr, nse = wavfile.read(noise_path)
        nse = nse.astype(np.float64)
        if nse.ndim > 1:
            nse = nse.mean(axis=1)
        nse = nse / (np.max(np.abs(nse)) + 1e-12)
        if nsr != sr:
            nse = resample_to(nse, nsr, sr)
        if len(nse) < len(x):
            nse = np.tile(nse, int(np.ceil(len(x) / len(nse))))
        off = rng.integers(0, len(nse) - len(x) + 1)
        nse = nse[off:off + len(x)]
        nse = nse / (np.sqrt(np.mean(nse ** 2)) + 1e-12)
    else:
        nse = generate_noise(len(x), kind, rng)

    p_speech = _active_speech_power(x, sr)
    p_noise_target = p_speech / (10 ** (snr_db / 10.0))
    return x + np.sqrt(p_noise_target) * nse


# --------------------------------------------------------------------------- #
# Codec round-trip (libsndfile preferred, ffmpeg fallback)
# --------------------------------------------------------------------------- #
_CODEC_CMD = {
    # codec: (ffmpeg encode args, container ext)
    "alaw": (["-c:a", "pcm_alaw", "-f", "wav"], ".wav"),
    "gsm":  (["-c:a", "libgsm", "-f", "gsm"], ".gsm"),
}
_SF_SUBTYPE = {"alaw": "ALAW", "gsm": "GSM610"}     # libsndfile WAV subtypes
_FFMPEG_ENC = {"alaw": "pcm_alaw", "gsm": "libgsm"}  # ffmpeg encoder names
_BACKEND_CACHE: dict[str, str] = {}


def _write_wav(path: str, x: np.ndarray, sr: int) -> None:
    xi = np.clip(x, -1.0, 1.0)
    wavfile.write(path, sr, (xi * 32767.0).astype(np.int16))


def _read_wav(path: str) -> tuple[np.ndarray, int]:
    sr, data = wavfile.read(path)
    if data.dtype == np.int16:
        data = data.astype(np.float64) / 32768.0
    elif data.dtype == np.int32:
        data = data.astype(np.float64) / 2147483648.0
    else:
        data = data.astype(np.float64)
    if data.ndim > 1:
        data = data.mean(axis=1)
    return data, sr


def _ffmpeg(args: list[str]) -> None:
    proc = subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", *args],
        capture_output=True, text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {' '.join(args)}\n{proc.stderr}")


def _have_soundfile() -> bool:
    try:
        import soundfile  # noqa: F401
        return True
    except Exception:
        return False


def _ffmpeg_has_encoder(enc: str) -> bool:
    try:
        out = subprocess.run(["ffmpeg", "-hide_banner", "-encoders"],
                             capture_output=True, text=True).stdout
    except FileNotFoundError:
        return False
    return any(line.split()[1] == enc
               for line in out.splitlines() if len(line.split()) > 1)


def _resolve_backend(codec: str) -> str:
    if codec in _BACKEND_CACHE:
        return _BACKEND_CACHE[codec]
    if _have_soundfile():
        backend = "soundfile"
    elif _ffmpeg_has_encoder(_FFMPEG_ENC[codec]):
        backend = "ffmpeg"
    else:
        raise RuntimeError(
            f"No usable codec backend for '{codec}'. "
            f"Install soundfile (pip install soundfile) or an ffmpeg with "
            f"the '{_FFMPEG_ENC[codec]}' encoder.")
    _BACKEND_CACHE[codec] = backend
    print(f"[codec] '{codec}' -> backend: {backend}")
    return backend


def _roundtrip_soundfile(x: np.ndarray, codec: str, sr: int) -> np.ndarray:
    import soundfile as sf
    tmp = tempfile.mkdtemp(prefix="tcodec_")
    try:
        p = os.path.join(tmp, "mid.wav")
        sf.write(p, np.clip(x, -1.0, 1.0), sr,
                 subtype=_SF_SUBTYPE[codec], format="WAV")
        y, _ = sf.read(p, dtype="float64")
        if y.ndim > 1:
            y = y.mean(axis=1)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return y


def _roundtrip_ffmpeg(x: np.ndarray, codec: str, sr: int) -> np.ndarray:
    enc_args, ext = _CODEC_CMD[codec]
    tmp = tempfile.mkdtemp(prefix="tcodec_")
    try:
        in_wav = os.path.join(tmp, "in.wav")
        mid = os.path.join(tmp, "mid" + ext)
        out_wav = os.path.join(tmp, "out.wav")
        _write_wav(in_wav, x, sr)
        _ffmpeg(["-i", in_wav, "-ar", "8000", "-ac", "1", *enc_args, mid])
        _ffmpeg(["-i", mid, "-ar", "8000", "-ac", "1",
                 "-c:a", "pcm_s16le", "-f", "wav", out_wav])
        y, _ = _read_wav(out_wav)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return y


def codec_roundtrip(x: np.ndarray, codec: str, sr: int = 8000) -> np.ndarray:
    """Encode + decode an 8 kHz mono signal with `codec`."""
    if sr != 8000:
        raise ValueError("codec round-trip expects 8 kHz input")
    backend = _resolve_backend(codec)
    y = (_roundtrip_soundfile if backend == "soundfile" else _roundtrip_ffmpeg)(x, codec, sr)
    # Codecs pad to whole frames: restore the input length
    return y[:len(x)] if len(y) >= len(x) else np.append(y, np.zeros(len(x) - len(y)))


def soft_limiter(x: np.ndarray, ceiling: float = 0.95) -> np.ndarray:
    """Light tanh limiter (AGC-like)."""
    peak = np.max(np.abs(x)) + 1e-12
    if peak <= ceiling:
        return x
    return ceiling * np.tanh(x / peak)


# --------------------------------------------------------------------------- #
# Conditions + pipeline
# --------------------------------------------------------------------------- #
@dataclass
class Condition:
    name: str
    codec: Optional[str] = None        # 'alaw' | 'gsm' | None
    snr_db: Optional[float] = None     # None = no added noise
    use_irs: bool = True               # handset shaping
    noise_kind: str = "pink"
    noise_path: Optional[str] = None   # real noise file (overrides noise_kind)
    noise_pre_codec: bool = True       # environment noise (True) vs line noise
    limiter: bool = False
    out_sr: int = 16000


def degrade(x: np.ndarray, sr_in: int, cond: Condition,
            rng: Optional[np.random.Generator] = None) -> tuple[np.ndarray, int]:
    rng = rng or np.random.default_rng()
    x = np.asarray(x, dtype=np.float64)
    if x.ndim > 1:
        x = x.mean(axis=1)
    x = x - np.mean(x)                       # DC removal
    peak = np.max(np.abs(x)) + 1e-12
    x = 0.95 * x / peak                      # headroom normalisation

    # Reference condition: resample only
    if cond.codec is None and cond.snr_db is None and not cond.use_irs:
        return resample_to(x, sr_in, cond.out_sr), cond.out_sr

    if cond.use_irs:
        x = apply_fir(x, design_irs_send(sr_in))

    x = telephone_bandpass(x, sr_in)
    x8 = resample_to(x, sr_in, 8000)

    if cond.snr_db is not None and cond.noise_pre_codec:
        x8 = add_noise_at_snr(x8, 8000, cond.snr_db, cond.noise_kind,
                              cond.noise_path, rng)

    if cond.codec is not None:
        x8 = codec_roundtrip(x8, cond.codec, sr=8000)

    if cond.snr_db is not None and not cond.noise_pre_codec:
        x8 = add_noise_at_snr(x8, 8000, cond.snr_db, cond.noise_kind,
                              cond.noise_path, rng)

    if cond.limiter:
        x8 = soft_limiter(x8)

    y = resample_to(x8, 8000, cond.out_sr)
    return np.clip(y, -1.0, 1.0), cond.out_sr


def build_condition_matrix(codecs=("alaw", "gsm"),
                           snrs=(20, 15, 10, 5),
                           use_irs: bool = True,
                           noise_kind: str = "pink",
                           noise_path: Optional[str] = None,
                           out_sr: int = 16000) -> list[Condition]:
    """'original' + per codec {clean, snr...}."""
    conds = [Condition("original", codec=None, snr_db=None,
                       use_irs=False, out_sr=out_sr)]
    for c in codecs:
        conds.append(Condition(f"{c}_clean", codec=c, snr_db=None,
                               use_irs=use_irs, out_sr=out_sr))
        for s in snrs:
            conds.append(Condition(f"{c}_snr{s}", codec=c, snr_db=float(s),
                                   use_irs=use_irs, noise_kind=noise_kind,
                                   noise_path=noise_path, out_sr=out_sr))
    return conds


def process_dataset(in_dir: str, out_dir: str,
                    conditions: Optional[list[Condition]] = None,
                    seed: int = 0, skip_existing: bool = False) -> str:
    """Write out_dir/<condition>/<file>.wav for every condition + manifest.csv."""
    in_dir, out_dir = Path(in_dir), Path(out_dir)
    conditions = conditions or build_condition_matrix()
    out_dir.mkdir(parents=True, exist_ok=True)
    wavs = sorted(in_dir.glob("*.wav"))
    if not wavs:
        raise FileNotFoundError(f"no .wav files in {in_dir}")

    manifest = out_dir / "manifest.csv"
    with open(manifest, "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["condition", "codec", "snr_db", "emotion", "in_file", "out_file"])
        for cond in conditions:
            cond_dir = out_dir / cond.name
            cond_dir.mkdir(parents=True, exist_ok=True)
            snr_cell = cond.snr_db if cond.snr_db is not None else ""

            # Resume: condition already complete, only re-emit manifest rows
            if skip_existing and len(list(cond_dir.glob("*.wav"))) >= len(wavs):
                for wp in wavs:
                    wr.writerow([cond.name, cond.codec or "", snr_cell,
                                 emodb_label(wp.name) or "", str(wp),
                                 str(cond_dir / wp.name)])
                print(f"[skip] {cond.name}: already complete ({len(wavs)} files)")
                continue

            # Deterministic, condition-specific noise
            rng = np.random.default_rng(seed + hash(cond.name) % 100000)
            for wp in wavs:
                x, sr = _read_wav(str(wp))
                y, osr = degrade(x, sr, cond, rng=rng)
                op = cond_dir / wp.name
                _write_wav(str(op), y, osr)
                wr.writerow([cond.name, cond.codec or "", snr_cell,
                             emodb_label(wp.name) or "", str(wp), str(op)])
            print(f"[done] {cond.name}: {len(wavs)} files")
    print(f"[manifest] {manifest}")
    return str(manifest)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Telephone-channel degradation of EmoDB.")
    ap.add_argument("in_dir", help="folder with clean EmoDB .wav files")
    ap.add_argument("out_dir", help="output root folder")
    ap.add_argument("--snrs", type=float, nargs="*", default=[20, 15, 10, 5])
    ap.add_argument("--codecs", nargs="*", default=["alaw", "gsm"])
    ap.add_argument("--no-irs", action="store_true", help="disable IRS handset shaping")
    ap.add_argument("--noise", default="pink", choices=["pink", "white"])
    ap.add_argument("--noise-file", default=None, help="real noise .wav (overrides --noise)")
    ap.add_argument("--out-sr", type=int, default=16000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--skip-existing", action="store_true",
                    help="skip conditions whose output folder is already complete")
    args = ap.parse_args()

    conds = build_condition_matrix(
        codecs=tuple(args.codecs), snrs=tuple(args.snrs),
        use_irs=not args.no_irs, noise_kind=args.noise,
        noise_path=args.noise_file, out_sr=args.out_sr,
    )
    process_dataset(args.in_dir, args.out_dir, conds, seed=args.seed,
                    skip_existing=args.skip_existing)
