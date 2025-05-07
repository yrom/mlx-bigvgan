# ==============================================================================
#  Adopted from https://github.com/ml-explore/mlx-examples/blob/4c9f9f9be798e6cf04fd0f74395a3b4420077aad/whisper/mlx_whisper/audio.py
#  And https://github.com/Blaizzy/mlx-audio/blob/021586f92d30fbbf6e6b4fa27c83297837fbad4c/mlx_audio/codec/models/vocos/mel.py
# ==============================================================================
import math
import os
from functools import lru_cache
from subprocess import CalledProcessError, run
from typing import Literal, Optional, Union
import warnings

import mlx.core as mx
import mlx.nn as nn
import numpy as np


class MelSpectrogram(nn.Module):
    def __init__(
        self,
        sample_rate=24_000,
        n_fft=1024,
        hop_length=256,
        n_mels=100,
        center=True,
        pad_mode="reflect",
        power=2.0,
        mel_fmin=0,
        mel_fmax=None,
        mel_norm=None,
        mel_scale="htk",
    ):
        super().__init__()
        self.center = center
        self.pad_mode = pad_mode
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.power = power
        self.mel_fmin = mel_fmin
        self.mel_fmax = mel_fmax or sample_rate // 2
        if mel_fmin > self.mel_fmax:
            raise ValueError(f"Require mel_fmin: {mel_fmin} <= mel_fmax: {self.mel_fmax}")
        self.mel_norm = mel_norm
        self.mel_scale = mel_scale

    def __call__(self, x: mx.array):
        """
        Args:
            x (mx.array): The input audio signal. shape: (T,)
        """
        return mel_spectrogram(
            x,
            sample_rate=self.sample_rate,
            n_mels=self.n_mels,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            padding=0,
            center=self.center,
            pad_mode=self.pad_mode,
            fmin=self.mel_fmin,
            fmax=self.mel_fmax,
            mel_norm=self.mel_norm,
            mel_scale=self.mel_scale,
            power=self.power,
        )


def load_audio_sf(audio_path: str, sample_rate: int = 24000) -> mx.array:
    import soundfile as sf

    samples, orig_sample_rate = sf.read(audio_path)
    shape = samples.shape  # [T, C]
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


def load_audio(file: str = Optional[str], sr: int = 24000):
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
        return mx.array(np.frombuffer(out, np.int16)).flatten().astype(mx.float32) / 32768.0
    except CalledProcessError as e:
        print(f"Error loading audio file {file} by ffmpeg: {e}")
        print("Trying loading by soundfile...")
        return load_audio_sf(file, sr)


@lru_cache(maxsize=10)
def melscale_fbanks(
    sample_rate: int,
    n_fft: int,
    f_min: float,
    f_max: float,
    n_mels: int,
    norm: Union[Literal["slaney"], None] = None,
    mel_scale: Union[Literal["htk"], Literal["slaney"]] = "htk",
) -> mx.array:
    r"""Create a frequency bin conversion matrix.
    Adopted from torch's ``torchaudio.functional.melscale_fbanks``

    Args:
        sample_rate (int): Sample rate of the audio waveform
        n_fft (int): Number of FFT points
        f_min (float): Minimum frequency (Hz)
        f_max (float): Maximum frequency (Hz)
        n_mels (int): Number of mel filterbanks
        norm (str or None, optional): If "slaney", divide the triangular mel weights by the width of the mel band
            (area normalization). (Default: ``None``)
        mel_scale (str, optional): Scale to use: ``htk`` or ``slaney``. (Default: ``htk``)

    Returns:
        filter banks (fb matrix) of size (``n_freqs``, ``n_mels``)
        meaning number of frequencies to highlight/apply to x the number of filterbanks.
        Each column is a filterbank so that assuming there is a matrix A of
        size (..., ``n_freqs``), the applied result would be
        ``A @ melscale_fbanks(A.size(-1), ...)``.

    """

    def _hz_to_mel(freq: float, mel_scale="htk") -> float:
        if mel_scale == "htk":
            return 2595.0 * math.log10(1.0 + (freq / 700.0))

        # Fill in the linear part
        fmin = 0.0
        f_sp = 200.0 / 3

        mels = (freq - fmin) / f_sp

        # Fill in the log-scale part
        min_log_hz = 1000.0
        min_log_mel = (min_log_hz - fmin) / f_sp
        logstep = math.log(6.4) / 27.0

        if freq >= min_log_hz:
            mels = min_log_mel + math.log(freq / min_log_hz) / logstep

        return mels

    def _mel_to_hz(mels: mx.array, mel_scale: str = "htk") -> mx.array:
        if mel_scale == "htk":
            return 700.0 * (10.0 ** (mels / 2595.0) - 1.0)

        # Fill in the linear scale
        fmin = 0.0
        f_sp = 200.0 / 3
        freqs = fmin + f_sp * mels

        # And now the nonlinear scale
        min_log_hz = 1000.0
        min_log_mel = (min_log_hz - fmin) / f_sp
        logstep = math.log(6.4) / 27.0
        log_t = mels >= min_log_mel
        # freqs[log_t] = min_log_hz * mx.exp(logstep * (mels[log_t] - min_log_mel))
        return mx.where(log_t, min_log_hz * mx.exp(logstep * (mels - min_log_mel)), freqs)

    if norm is not None and norm != "slaney":
        raise ValueError('norm must be one of None or "slaney"')
    if mel_scale not in ["slaney", "htk"]:
        raise ValueError('mel_scale should be one of "htk" or "slaney".')
    n_freqs = n_fft // 2 + 1
    # freq bins
    all_freqs = mx.linspace(0, sample_rate // 2, n_freqs)

    # calculate mel freq bins
    m_min = _hz_to_mel(f_min, mel_scale)
    m_max = _hz_to_mel(f_max, mel_scale)

    m_pts = mx.linspace(m_min, m_max, n_mels + 2)
    f_pts = _mel_to_hz(m_pts, mel_scale)

    # create filterbank
    # calculate the difference between each filter mid point and each stft freq point in hertz
    f_diff = f_pts[1:] - f_pts[:-1]  # (n_filter + 1)
    slopes = mx.expand_dims(f_pts, 0) - mx.expand_dims(all_freqs, 1)  # (n_freqs, n_filter + 2)
    # create overlapping triangles

    down_slopes = (-1.0 * slopes[:, :-2]) / f_diff[:-1]  # (n_freqs, n_filter)
    up_slopes = slopes[:, 2:] / f_diff[1:]  # (n_freqs, n_filter)
    fb = mx.maximum(mx.zeros_like(down_slopes), mx.minimum(down_slopes, up_slopes))
    if norm is not None and norm == "slaney":
        # Slaney-style mel is scaled to be approx constant energy per channel
        enorm = 2.0 / (f_pts[2 : n_mels + 2] - f_pts[:n_mels])
        fb *= mx.expand_dims(enorm, 0)

    if (fb.max(axis=0) == 0.0).any():
        warnings.warn(
            "At least one mel filterbank has all zero values. "
            f"The value for `n_mels` ({n_mels}) may be set too high. "
            f"Or, the value for `n_freqs` ({n_freqs}) may be set too low."
        )

    return fb


# @lru_cache(maxsize=None)
def hanning(size):
    return mx.array(np.hanning(size + 1)[:-1])


# @lru_cache(maxsize=None)
# def hanning(size):
#     return mx.array([0.5 * (1 - math.cos(2 * math.pi * n / (size - 1))) for n in range(size)])


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
def _pad(x: mx.array, padding, pad_mode="constant"):
    if pad_mode in ["constant", "edge"]:
        return mx.pad(x, [(padding, padding)], mode=pad_mode)
    elif pad_mode == "reflect":
        prefix = x[1 : padding + 1][::-1]
        suffix = x[-(padding + 1) : -1][::-1]
        return mx.concatenate([prefix, x, suffix])
    else:
        raise ValueError(f"Invalid pad_mode {pad_mode}")


def stft(x: mx.array, window: mx.array, n_fft, hop_length=None, center=True, pad_mode="reflect", onesided=True):
    """Short-time Fourier transform (STFT).

    Args:
        x (mx.array): The input audio signal. 1D, shape: ``(T,)``
        window (mx.array): The window. shape: (n_fft,)
        n_fft (int): The FFT size.
        hop_length (int, optional): The hop length for the STFT. Default is ``n_fft // 4``.
        center (bool, optional): If True, pad the input signal on both sides. Default is ``True``.
        pad_mode (str, optional): Controls the padding method used when
            :attr:`center` is ``True``. Default: ``"reflect"``
        onesided (bool, optional): controls whether spectrogram was used to return half of results to
            avoid redundancy (Default: ``True``)
    """
    if hop_length is None:
        hop_length = n_fft // 4

    if x.ndim >= 2:
        raise ValueError(f"Invalid input shape {x.shape}, expected 1D array")
    T = x.shape[-1]
    if n_fft > T:
        warnings.warn(f"n_fft={n_fft} is too large for input signal of length={T}")
    if center:
        padding = int(n_fft // 2)
        # [T] -> [T + n_fft]
        x = _pad(x, padding, pad_mode)
        T = x.shape[-1]
    n_frames = 1 + (T - n_fft) // hop_length
    strides = [hop_length, 1]
    shape = [n_frames, n_fft]
    xs = mx.as_strided(x, shape=shape, strides=strides)
    result = mx.fft.rfft(xs * window)

    if onesided:
        # [n_frames, n_fft] -> [n_frames, n_fft // 2 + 1]
        return result[:, : n_fft // 2 + 1]
    # [n_frames, n_fft]
    return result


def mel_spectrogram(
    audio: mx.array,
    sample_rate: int = 24_000,
    n_mels: int = 100,
    n_fft: int = 1024,
    hop_length: int = 256,
    padding: int = 0,
    center: bool = True,
    pad_mode: str = "reflect",
    fmin: int = 0,
    fmax: Optional[int] = None,
    mel_norm: Optional[Literal["slaney"]] = None,
    mel_scale: Union[Literal["htk"], Literal["slaney"]] = "htk",
    power: float = 1.0,
):
    """
    Compute the mel spectrogram of an audio signal.

    Args:
        audio (mx.array): The input audio signal. shape: (T,)
        sample_rate (int): The sample rate of the audio signal.
        n_mels (int): The number of mel bands to generate.
        n_fft (int): The FFT size.
        hop_length (int): The hop length for the STFT.
        padding (int): The padding to apply to the audio signal.
        fmin (int): The minimum frequency for the mel filter bank.
        fmax (int, optional): The maximum frequency for the mel filter bank.
        power (float): 1.0 for magnitude, 2.0 for power spectrogram.
    """
    if not isinstance(audio, mx.array):
        audio = mx.array(audio)
    print("audio", audio.shape)
    if padding > 0:  # pad the audio
        audio = _pad(audio, padding, "reflect")
    print("audio after pad, before stft", audio.shape)

    freqs = stft(
        audio, hanning(n_fft), n_fft=n_fft, hop_length=hop_length, center=center, pad_mode=pad_mode, onesided=True
    )
    assert freqs.shape[-1] == n_fft // 2 + 1, f"Invalid STFT shape {freqs.shape}"
    magnitudes = mx.abs(freqs)
    if power and power != 1.0:
        magnitudes = mx.power(magnitudes, power)
    # [n_freqs, n_mels]
    melscale_fb = melscale_fbanks(
        sample_rate=sample_rate,
        n_fft=n_fft,
        f_min=float(fmin),
        f_max=float(fmax or sample_rate // 2),
        n_mels=n_mels,
        norm=mel_norm,
        mel_scale=mel_scale,
    )
    print("melscale_fb", melscale_fb.shape)

    #  (T, n_freqs) dot (n_freqs, n_mels) -> (T, n_mels)
    mel_spec = magnitudes @ melscale_fb
    return mel_spec  # [T, n_mels]


def log_mel_spectrogram(
    audio: mx.array,
    **mel_kwargs,
):
    r"""Compute the log-mel spectrogram of an audio signal.

    Args:
        audio (mx.array): The input audio signal. shape: (T,)
        mel_kwargs: Additional arguments for ``mel_spectrogram(...)``.
    """
    mel_spec = mel_spectrogram(
        audio,
        **mel_kwargs,
    )
    log_spec = mx.maximum(mel_spec, 1e-5).log()
    return log_spec  # [T, n_mels]