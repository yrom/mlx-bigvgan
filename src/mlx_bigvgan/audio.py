# ==============================================================================
#  Adopted from https://github.com/ml-explore/mlx-examples/blob/4c9f9f9be798e6cf04fd0f74395a3b4420077aad/whisper/mlx_whisper/audio.py
#  And https://github.com/Blaizzy/mlx-audio/blob/021586f92d30fbbf6e6b4fa27c83297837fbad4c/mlx_audio/codec/models/vocos/mel.py
# ==============================================================================
import math
import os
from functools import lru_cache
from subprocess import CalledProcessError, run
from typing import Optional, Union

import mlx.core as mx
import mlx.nn as nn
import numpy as np


class MelSpectrogramFeatures(nn.Module):
    def __init__(
        self,
        sample_rate=24_000,
        n_fft=1024,
        hop_length=256,
        n_mels=100,
        padding="center",
        mel_fmin=0,
        mel_fmax=None,

    ):
        super().__init__()
        if padding not in ["center", "same"]:
            raise ValueError("Padding must be 'center' or 'same'.")
        self.padding = padding
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.mel_fmin = mel_fmin
        self.mel_fmax = mel_fmax

    def __call__(self, x: mx.array, **kwargs):
        return log_mel_spectrogram(
            x,
            sample_rate=self.sample_rate,
            n_mels=self.n_mels,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            padding=0,
            fmin=self.mel_fmin,
            fmax=self.mel_fmax,
        )

def load_audio_sf(audio_path: str, sample_rate: int = 24000) -> mx.array:
    import soundfile as sf
    
    samples, orig_sample_rate = sf.read(audio_path)
    shape = samples.shape #[T, C]
    # stereo to mono
    if shape[-1] > 1:
        samples = samples.mean(axis=1)
    if sample_rate != orig_sample_rate:
        print(f"Resampling from {orig_sample_rate} to {sample_rate}")
        samples = resample_audio(samples, orig_sample_rate, sample_rate)
    audio = mx.array(samples, dtype=mx.float32)
    return audio
def resample_audio(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    from scipy import signal
    gcd = np.gcd(orig_sr, target_sr)
    up = target_sr // gcd
    down = orig_sr // gcd
    resampled = signal.resample_poly(audio, up, down, padtype="edge")
    return resampled

def load_audio(file: str = Optional[str], sr: int = 24000, from_stdin=False):
    """
    Open an audio file and read as mono waveform, resampling as necessary

    Parameters
    ----------
    file: str
        The audio file to open

    sr: int
        The sample rate to resample the audio if necessary

    Returns
    -------
    A NumPy array containing the audio waveform, in float32 dtype.
    """

    # This launches a subprocess to decode audio while down-mixing
    # and resampling as necessary. Requires the ffmpeg CLI in PATH.
    if from_stdin:
        cmd = ["ffmpeg", "-i", "pipe:0"]
    else:
        cmd = ["ffmpeg", "-nostdin", "-i", file]

    # fmt: off
    cmd.extend([
        "-threads", "0",
        "-f", "s16le",
        "-ac", "1",
        "-acodec", "pcm_s16le",
        "-ar", str(sr),
        "-"
    ])
    # fmt: on
    try:
        out = run(cmd, capture_output=True, check=True).stdout
    except CalledProcessError as e:
        raise RuntimeError(f"Failed to load audio: {e.stderr.decode()}") from e

    return mx.array(np.frombuffer(out, np.int16)).flatten().astype(mx.float32) / 32768.0


@lru_cache(maxsize=None)
def mel_filters(
    sampling_rate: int,
    n_fft: int,
    n_mels: int,
    fmin: int = 0,
    fmax: Optional[int] = None,
    norm: Optional[str] = None,
    mel_scale: str = "htk",
) -> mx.array:
    def hz_to_mel(freq, mel_scale="htk"):
        if mel_scale == "htk":
            return 2595.0 * math.log10(1.0 + freq / 700.0)

        # slaney scale
        f_min, f_sp = 0.0, 200.0 / 3
        mels = (freq - f_min) / f_sp
        min_log_hz = 1000.0
        min_log_mel = (min_log_hz - f_min) / f_sp
        logstep = math.log(6.4) / 27.0
        if freq >= min_log_hz:
            mels = min_log_mel + math.log(freq / min_log_hz) / logstep
        return mels

    def mel_to_hz(mels: mx.array, mel_scale="htk"):
        if mel_scale == "htk":
            return 700.0 * (10.0 ** (mels / 2595.0) - 1.0)

        # slaney scale
        f_min, f_sp = 0.0, 200.0 / 3
        freqs = f_sp * mels + f_min
        min_log_hz = 1000.0
        min_log_mel = (min_log_hz - f_min) / f_sp
        logstep = math.log(6.4) / 27.0
        log_t = mels >= min_log_mel
        freqs[log_t] = min_log_hz * mx.exp(logstep * (mels[log_t] - min_log_mel))
        return freqs

    fmax = fmax or sampling_rate / 2

    # generate frequency points

    n_freqs = n_fft // 2 + 1
    all_freqs = mx.linspace(0, sampling_rate // 2, n_freqs)

    # convert frequencies to mel and back to hz

    m_min = hz_to_mel(fmin, mel_scale)
    m_max = hz_to_mel(fmax, mel_scale)
    m_pts = mx.linspace(m_min, m_max, n_mels + 2)
    f_pts = mel_to_hz(m_pts, mel_scale)

    # compute slopes for filterbank

    f_diff = f_pts[1:] - f_pts[:-1]
    slopes = mx.expand_dims(f_pts, 0) - mx.expand_dims(all_freqs, 1)

    # calculate overlapping triangular filters

    down_slopes = (-slopes[:, :-2]) / f_diff[:-1]
    up_slopes = slopes[:, 2:] / f_diff[1:]
    filterbank = mx.maximum(mx.zeros_like(down_slopes), mx.minimum(down_slopes, up_slopes))

    if norm == "slaney":
        enorm = 2.0 / (f_pts[2 : n_mels + 2] - f_pts[:n_mels])
        filterbank *= mx.expand_dims(enorm, 0)

    filterbank = filterbank.moveaxis(0, 1) # [n_mels, n_freqs]
    return filterbank


# @lru_cache(maxsize=None)
# def hanning(size):
#     return mx.array(np.hanning(size + 1)[:-1])
@lru_cache(maxsize=None)
def hanning(size):
    return mx.array([0.5 * (1 - math.cos(2 * math.pi * n / (size - 1))) for n in range(size)])


# def _pad(x, padding, pad_mode="constant"):
#     # [B, T] -> [B, T + 2 * padding]
#     if pad_mode in ["constant", "edge"]:
#         return mx.pad(x, [(0,0), (padding, padding)], mode=pad_mode)
#     elif pad_mode == "reflect":
#         prefix = x[:,1 : padding + 1][:,::-1]
#         suffix = x[:,-(padding + 1) : -1][:,::-1]
#         return mx.concatenate([prefix, x, suffix], axis=1)
#     else:
#         raise ValueError(f"Invalid pad_mode {pad_mode}")


def stft(x, window, nperseg=256, noverlap=None, nfft=None, pad_mode="constant"):
    if nfft is None:
        nfft = nperseg
    if noverlap is None:
        noverlap = nfft // 4

    def _pad(x, padding, pad_mode="constant"):
        if pad_mode == "constant":
            return mx.pad(x, [(padding, padding)])
        elif pad_mode == "reflect":
            prefix = x[1 : padding + 1][::-1]
            suffix = x[-(padding + 1) : -1][::-1]
            return mx.concatenate([prefix, x, suffix])
        else:
            raise ValueError(f"Invalid pad_mode {pad_mode}")

    padding = nperseg // 2
    x = _pad(x, padding, pad_mode)
    T = x.shape[-1]
    strides = [noverlap, 1]
    t = (T - nperseg + noverlap) // noverlap
    shape = [t, nfft]
    x = mx.as_strided(x, shape=shape, strides=strides)
    return mx.fft.rfft(x * window)


def log_mel_spectrogram(
    audio: mx.array,
    sample_rate: int = 24_000,
    n_mels: int = 100,
    n_fft: int = 1024,
    hop_length: int = 256,
    padding: int = 0,
    fmin: int = 0,
    fmax: Optional[int] = None,
):
    """
    Compute the log-mel spectrogram of an audio signal.

    Args:
        audio (mx.array): The input audio signal. shape: (T,)
        sample_rate (int): The sample rate of the audio signal.
        n_mels (int): The number of mel bands to generate.
        n_fft (int): The FFT size.
        hop_length (int): The hop length for the STFT.
        padding (int): The padding to apply to the audio signal.
        fmin (int): The minimum frequency for the mel filter bank.
        fmax (int, optional): The maximum frequency for the mel filter bank.
    """
    if not isinstance(audio, mx.array):
        audio = mx.array(audio)
    print("audio", audio.shape)
    if padding > 0:
        audio = mx.pad(audio, (0, padding))

    freqs = stft(audio, hanning(n_fft), nperseg=n_fft, noverlap=hop_length)
    magnitudes = freqs[:-1, :].abs()
    filters = mel_filters(
        sample_rate=sample_rate,
        n_fft=n_fft,
        n_mels=n_mels,
        norm=None,
        fmin=fmin,
        fmax=fmax,
        mel_scale="htk",
    )
    # [T, n_mels]
    mel_spec = magnitudes @ filters.T
    log_spec = mx.maximum(mel_spec, 1e-5).log()
    return mx.expand_dims(log_spec, axis=0) # [1, T, n_mels]
