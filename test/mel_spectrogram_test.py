import random
import numpy as np

import matplotlib.pyplot as plt
import mlx.core as mx



def test_wrap(func):
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            print("Test Failed:", func.__name__, kwargs)
            print(e)

    return wrapper


# code from bigvgan
def bigvgan_mel_spectrogram(
    wav,
    n_fft: int,
    num_mels: int,
    sample_rate: int,
    hop_size: int,
    win_size: int,
    fmin: int,
    fmax: int = None,
    center: bool = True,
):
    """
    Calculate the mel spectrogram of an input signal.
    This function uses slaney norm for the librosa mel filterbank (using librosa.filters.mel) and uses Hann window for STFT (using torch.stft).

    Args:
        y (torch.Tensor): Input signal.
        n_fft (int): FFT size.
        num_mels (int): Number of mel bins.
        sample_rate (int): Sampling rate of the input signal.
        hop_size (int): Hop size for STFT.
        win_size (int): Window size for STFT.
        fmin (int): Minimum frequency for mel filterbank.
        fmax (int): Maximum frequency for mel filterbank. If None, defaults to half the sampling rate (fmax = sr / 2.0) inside librosa_mel_fn
        center (bool): Whether to pad the input to center the frames. Default is True.

    Returns:
        torch.Tensor: Mel spectrogram.
    """
    from librosa.filters import mel as librosa_mel_fn
    import torch

    mel = librosa_mel_fn(sr=sample_rate, n_fft=n_fft, n_mels=num_mels, fmin=fmin, fmax=fmax, htk=False)

    mel_basis = torch.from_numpy(mel).float()
    hann_window = torch.hann_window(win_size)

    padding = (n_fft - hop_size) // 2
    print("wav before pad", wav.shape)
    wav = torch.nn.functional.pad(wav.unsqueeze(1), (padding, padding), mode="reflect").squeeze(1)
    print("wav after pad, before stft", wav.shape)
    spec = torch.stft(
        wav,
        n_fft,
        hop_length=hop_size,
        win_length=win_size,
        window=hann_window,
        center=center,
        pad_mode="reflect",
        normalized=False,
        onesided=True,
        return_complex=True,
    )

    spec = torch.sqrt(torch.view_as_real(spec).pow(2).sum(-1) + 1e-9)

    mel_spec = torch.matmul(mel_basis, spec)
    log_spec = torch.log(torch.clamp(mel_spec, min=1e-5))

    return log_spec


#@test_wrap
def test_mel_spec_with_origin_bigvgan():
    import torch
    from mlx_bigvgan.audio import log_mel_spectrogram as my_mel_spectrogram
    wav = np.random.randn(24_000 * 3).astype(np.float32)
    n_fft = 1024
    num_mels = 100
    sample_rate = 24_000
    hop_size = 256
    win_size = 1024
    fmin = 0
    fmax = 24_000 // 2
    bigvgan_mel = bigvgan_mel_spectrogram(
        torch.tensor(wav).unsqueeze(0),
        n_fft=n_fft,
        num_mels=num_mels,
        sample_rate=sample_rate,
        hop_size=hop_size,
        win_size=win_size,
        fmin=fmin,
        fmax=fmax,
        center=True,
    ).squeeze(0)

    my_mel = my_mel_spectrogram(
        wav,
        sample_rate=sample_rate,
        n_mels=num_mels,
        n_fft=n_fft,
        hop_length=hop_size,
        # win_length=win_size,
        padding = (n_fft - hop_size) // 2,
        center=True,
        pad_mode="reflect",
        fmin=fmin,
        fmax=fmax,
        mel_norm="slaney",
        mel_scale="slaney",
        power=1.0,
    )

    # Plot mel spectrograms
    plt.figure(figsize=(18, 6))

    plt.subplot(1, 3, 1)
    plt.title("Origin BigVGAN Mel Spectrogram")
    plt.imshow(bigvgan_mel.numpy(), aspect="auto", origin="lower", interpolation="none", cmap="viridis")
    plt.colorbar(label="Log Amplitude")

    plt.subplot(1, 3, 2)
    plt.title("My Mel Spectrogram")
    plt.imshow(my_mel.T, aspect="auto", origin="lower", interpolation="none", cmap="viridis")
    plt.colorbar(label="Log Amplitude")

    plt.subplot(1, 3, 3)
    plt.title("Difference (Origin BigVGAN - My)")
    plt.imshow(
        np.clip(np.abs(bigvgan_mel.numpy() - np.array(my_mel.T)), min=1e-5, max=100),
        aspect="auto",
        origin="lower",
        cmap="viridis",
    )

    plt.tight_layout()
    plt.show()


@test_wrap
def test_mel_spectrogram(
    wav: np.ndarray,
    sample_rate: int = 24_000,
    n_fft: int = 1024,
    hop_length: int = 256,
    n_mels: int = 100,
    mel_norm=None,
    mel_scale="htk",
):
    import torch
    from torchaudio.transforms import MelSpectrogram as TorchMelSpectrogram
    from mlx_bigvgan import MelSpectrogram as MyMelSpectrogram

    mel_fmin = 0
    mel_fmax = None
    my_mel = MyMelSpectrogram(
        sample_rate=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        center=True,
        pad_mode="reflect",
        mel_fmin=mel_fmin,
        mel_fmax=mel_fmax,
        mel_norm=mel_norm,
        mel_scale=mel_scale,
        power=2.0,
    )(mx.array(wav))  # [T, C_mels]

    torch_mel: torch.Tensor = TorchMelSpectrogram(
        sample_rate=sample_rate,
        n_fft=n_fft,
        win_length=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        f_min=mel_fmin,
        f_max=mel_fmax,
        center=True,
        pad_mode="reflect",
        norm=mel_norm,
        mel_scale=mel_scale,
        power=2.0,
    )(torch.from_numpy(wav).unsqueeze(0))  # [1, C_mels, T]
    my_mel = np.abs(np.array(my_mel, dtype=np.float32)).swapaxes(1, 0)  # [C_mels, T]
    torch_mel = np.abs(torch_mel.to(torch.float32).squeeze(0).detach().numpy())

    # print("my mel", my_mel.shape)
    # print("torch mel", torch_mel.shape)
    assert my_mel.shape[1] == torch_mel.shape[1], f"unexpected mel shape {my_mel.shape} vs {torch_mel.shape}"
    log_my_mel = np.log10(np.clip(my_mel, min=1e-5)) * 10
    log_torch_mel = np.log10(np.clip(torch_mel, min=1e-5)) * 10
    diffrence = np.abs(log_my_mel - log_torch_mel)
    # Plot mel spectrograms
    plt.figure(figsize=(18, 6))

    plt.subplot(1, 3, 1)
    plt.title("My Mel Spectrogram")
    plt.imshow(log_my_mel, aspect="auto", origin="lower", interpolation="none", cmap="viridis")
    plt.colorbar(label="Log Amplitude")

    plt.subplot(1, 3, 2)
    plt.title("Torch Mel Spectrogram")
    plt.imshow(log_torch_mel, aspect="auto", origin="lower", interpolation="none", cmap="viridis")
    plt.colorbar(label="Log Amplitude")

    plt.subplot(1, 3, 3)
    plt.title("Difference (My - Torch)")
    plt.imshow(diffrence, aspect="auto", origin="lower", cmap="viridis")

    plt.tight_layout()
    plt.show()


@test_wrap
def test_stft(n_fft=4, center=True):
    import torch
    from mlx_bigvgan.audio import stft as my_stft
    from mlx_bigvgan.audio import hanning as my_hanning
    from torch import stft as torch_stft

    win_length = n_fft
    my_window = my_hanning(win_length)
    torch_window = torch.hann_window(win_length + 1, periodic=False, dtype=torch.float32)[:-1]
    assert my_window.shape == torch_window.shape, f"unexpected window shape {my_window.shape} vs {torch_window.shape}"
    window_diff = np.abs(np.array(my_window, copy=False) - torch_window.numpy())
    assert np.max(window_diff) < 1e-5, f"Max window difference: {np.max(window_diff)}"
    assert np.mean(window_diff) < 1e-6, f"Mean window difference: {np.mean(window_diff)}"
    assert np.min(window_diff) < 1e-7, f"Min window difference: {np.min(window_diff)}"
    wav = (np.random.randn(n_fft * 2)).astype(np.float32)

    my_stft_result = my_stft(
        mx.array(wav),
        window=my_window,
        n_fft=n_fft,
        hop_length=n_fft // 4,
        center=center,
        pad_mode="reflect",
        # normalized=False,
        onesided=True,
    )
    torch_stft_result = torch_stft(
        torch.from_numpy(wav),
        n_fft=n_fft,
        hop_length=n_fft // 4,
        win_length=win_length,
        window=torch_window,
        center=center,
        normalized=False,
        onesided=True,
        pad_mode="reflect",
        return_complex=True,
    )

    my_stft_result = np.array(my_stft_result.abs())
    torch_stft_result = np.abs(torch_stft_result.detach().abs().numpy().T)

    assert my_stft_result.shape == torch_stft_result.shape, (
        f"unexpected stft shape {my_stft_result.shape} vs {torch_stft_result.shape}"
    )

    # spec = np.power(my_stft_result, 2).astype(np.float32)
    # torch_spec = np.power(torch_stft_result, 2).astype(np.float32)

    # print("my stft", my_stft_result)
    # print("torch stft", torch_stft_result)
    # print("librosa stft", librosa_stft_result)
    diffrence_to_torch = my_stft_result - torch_stft_result
    print("my vs torch", np.max(np.abs(diffrence_to_torch)), "allclose:", np.allclose(my_stft_result, torch_stft_result, rtol=1e-3, atol=1e-4))
    print(f"Max difference to torch: {np.max(diffrence_to_torch)}")
    print(f"Mean difference to torch: {np.mean(diffrence_to_torch)}")

    # spectrogram plot
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.title("My STFT Result")
    plt.imshow(my_stft_result, aspect="auto", origin="lower", interpolation="none")
    plt.colorbar(label="Amplitude")

    plt.subplot(1, 2, 2)
    plt.title("Torch STFT Result")
    plt.imshow(torch_stft_result, aspect="auto", origin="lower", interpolation="none")
    plt.colorbar(label="Amplitude")
    plt.tight_layout()
    plt.show()


@test_wrap
def test_melscale_fn(norm=None, mel_scale="htk"):
    from librosa.filters import mel as librosa_mel_fn
    from mlx_bigvgan.audio import melscale_fbanks as my_mel_fn
    from torchaudio.functional import melscale_fbanks as torchaudio_mel_fn

    sample_rate = 44_000
    n_fft = 200
    n_mels = 20
    fmin = 0
    fmax = sample_rate // 2
    print("Test mel_scale_fn: ", "norm=", norm, "mel_scale=", mel_scale)
    my_mel = my_mel_fn(
        sample_rate=sample_rate,
        n_fft=n_fft,
        n_mels=n_mels,
        f_min=fmin,
        f_max=fmax,
        norm=norm,
        mel_scale=mel_scale,
    )
    assert my_mel.shape == (n_fft // 2 + 1, n_mels), f"unexpected mel shape {my_mel.shape}"

    librosa_mel = librosa_mel_fn(
        sr=sample_rate,
        n_fft=n_fft,
        n_mels=n_mels,
        fmin=fmin,
        fmax=fmax,
        htk=mel_scale == "htk",
        norm=norm,
    ).T

    torch_mel = torchaudio_mel_fn(
        n_freqs=n_fft // 2 + 1,
        n_mels=n_mels,
        f_min=fmin,
        f_max=fmax,
        sample_rate=sample_rate,
        norm=norm,
        mel_scale=mel_scale,
    )
    # print("my mel", my_mel.shape)
    # print("librosa mel", librosa_mel.shape)
    # print("torch mel", torch_mel.shape)

    diffrence_to_librosa = np.abs(my_mel - librosa_mel)
    diffrence_to_torch = np.abs(my_mel - torch_mel)
    assert np.max(diffrence_to_librosa) <= 1e-5, f"Max difference to librosa: {np.max(diffrence_to_librosa)}"
    assert np.mean(diffrence_to_librosa) <= 1e-6, f"Mean difference to librosa: {np.mean(diffrence_to_librosa)}"
    assert np.min(diffrence_to_librosa) <= 1e-7, f"Min difference to librosa: {np.min(diffrence_to_librosa)}"
    assert np.max(diffrence_to_torch) <= 1e-5, f"Max difference to torch: {np.max(diffrence_to_torch)}"
    assert np.mean(diffrence_to_torch) <= 1e-6, f"Mean difference to torch: {np.mean(diffrence_to_torch)}"
    assert np.min(diffrence_to_torch) <= 1e-7, f"Min difference to torch: {np.min(diffrence_to_torch)}"

    # Plot mel fbanks
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 3, 1)
    plt.title("My Mel FBanks")
    plt.imshow(my_mel, aspect="auto", origin="lower", interpolation="none")

    plt.subplot(1, 3, 2)
    plt.title("Librosa Mel FBanks")
    plt.imshow(librosa_mel, aspect="auto", origin="lower", interpolation="none")

    plt.subplot(1, 3, 3)
    plt.title("Torch Mel FBanks")
    plt.imshow(torch_mel, aspect="auto", origin="lower", interpolation="none")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    import numpy as np
    import os
    from mlx_bigvgan.audio import load_audio

    # for norm, mel_scale in [
    #     ("slaney", "slaney"),
    #     ("slaney", "htk"),
    #     (None, "slaney"),
    #     (None, "htk"),
    # ]:
    #     test_melscale_fn(norm=norm, mel_scale=mel_scale)

    for n_fft in [10, 100, 1024]:
        test_stft(n_fft=n_fft, center=True)
    #     test_stft(n_fft=n_fft, center=False)

    # waveform = load_audio(os.path.join(os.path.dirname(__file__), "Female_24khz.wav"), 24_000)
    # waveform = waveform[: 24_000 * 3]  # 3 seconds
    # print("waveform", waveform.shape)
    # test_mel_spectrogram(
    #     np.array(waveform, copy=False),
    #     sample_rate=24_000,
    #     n_fft=1024,
    #     hop_length=256,
    #     n_mels=100,
    #     mel_norm=None,
    #     mel_scale="htk",
    # )
    test_mel_spec_with_origin_bigvgan()
