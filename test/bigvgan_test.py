import collections
from mlx_bigvgan import BigVGAN

from mlx.utils import tree_flatten
import mlx.core as mx


def test_load_model():
    # Run after convert.py
    model = BigVGAN.from_pretrained("mlx_models/bigvgan_v2_24khz_100band_256x", local_files_only=True)
    model.eval()
    model_weights = collections.OrderedDict(tree_flatten(model.parameters()))
    # compare weights with original model
    import bigvgan as original
    import torch
    import numpy as np

    original_model = original.BigVGAN.from_pretrained("nvidia/bigvgan_v2_24khz_100band_256x")
    original_model = original_model.eval()
    original_model.remove_weight_norm()

    # compare weights
    original_weights = dict(original_model.named_parameters())
    original_weights.update(dict(original_model.named_buffers()))
    # print(
    #     "original weights:", *[f"{k}: {v.shape}, {v.dtype}" for k, v in original_weights.items()], sep="\n", flush=True
    # )
    # print(
    #     "converted weights:",
    #     *[f"{k}: {v.shape}, {v.dtype}" for k, v in model_weights.items()],
    #     sep="\n",
    #     flush=True,
    # )
    # weights to compare
    for k1, k2, axis in [
        ("conv_pre.bias", None, None),
        ("conv_pre.weight", None, (1, 2)),
        ("conv_post.weight", None, (1, 2)),
        ("ups.0.weight", "ups.0.0.weight", (2, 0)),
        ("ups.1.weight", "ups.1.0.weight", (2, 0)),
        ("ups.2.weight", "ups.2.0.weight", (2, 0)),
        ("ups.3.weight", "ups.3.0.weight", (2, 0)),
        ("ups.4.weight", "ups.4.0.weight", (2, 0)),
        ("ups.5.weight", "ups.5.0.weight", (2, 0)),
        ("ups.0.bias", "ups.0.0.bias", None),
        ("ups.1.bias", "ups.1.0.bias", None),
        ("ups.2.bias", "ups.2.0.bias", None),
        ("ups.3.bias", "ups.3.0.bias", None),
        ("ups.4.bias", "ups.4.0.bias", None),
        ("ups.5.bias", "ups.5.0.bias", None),
        ("activation_post.act.alpha", None, None),
        ("activation_post.act.beta", None, None),
    ]:
        if k2 is None:
            k2 = k1
        assert k1 in model_weights, f"Key {k1} not found in converted model weights"
        assert k2 in original_weights, f"Key {k2} not found in original model weights"
        converted_p = model_weights[k1]
        if axis is not None:
            converted_p = mx.moveaxis(converted_p, *axis)
        original_p = original_weights[k2 or k1].data
        np.testing.assert_allclose(
            np.array(converted_p), original_p.numpy(), rtol=1e-5, atol=1e-5, err_msg=f"Failed to match {k1} and {k2}"
        )
    # compare resblocks
    for k1, k2, axis in [
        ("resblocks.0.layers.0.layers.1.weight", "resblocks.0.convs1.0.weight", (1, 2)),
        ("resblocks.0.layers.1.layers.1.weight", "resblocks.0.convs1.1.weight", (1, 2)),
        ("resblocks.1.layers.0.layers.1.weight", "resblocks.1.convs1.0.weight", (1, 2)),
        ("resblocks.1.layers.1.layers.1.weight", "resblocks.1.convs1.1.weight", (1, 2)),
        ("resblocks.17.layers.0.layers.3.weight", "resblocks.17.convs2.0.weight", (1, 2)),
        ("resblocks.17.layers.1.layers.3.weight", "resblocks.17.convs2.1.weight", (1, 2)),
        ("resblocks.17.layers.2.layers.3.weight", "resblocks.17.convs2.2.weight", (1, 2)),
        ("resblocks.8.layers.0.layers.0.act.alpha", "resblocks.8.activations.0.act.alpha", None),
        ("resblocks.8.layers.0.layers.2.act.alpha", "resblocks.8.activations.1.act.alpha", None),
        ("resblocks.8.layers.1.layers.0.act.alpha", "resblocks.8.activations.2.act.alpha", None),
        ("resblocks.8.layers.1.layers.2.act.alpha", "resblocks.8.activations.3.act.alpha", None),
        ("resblocks.8.layers.2.layers.0.act.alpha", "resblocks.8.activations.4.act.alpha", None),
        ("resblocks.8.layers.2.layers.2.act.alpha", "resblocks.8.activations.5.act.alpha", None),
    ]:
        if k2 is None:
            k2 = k1
        assert k1 in model_weights, f"Key {k1} not found in converted model weights"
        assert k2 in original_weights, f"Key {k2} not found in original model weights"
        converted_p = model_weights[k1]
        if axis is not None:
            converted_p = mx.moveaxis(converted_p, *axis)
        original_p = original_weights[k2 or k1].data
        np.testing.assert_allclose(
            np.array(converted_p), original_p.numpy(), rtol=1e-5, atol=1e-5, err_msg=f"Failed to match {k1} and {k2}"
        )


def test_inference():
    import numpy as np

    # Dummy input [B, T, C]
    x = mx.random.uniform(low=-1, high=1, shape=(1, 100, 100))
    # Run after convert.py
    print("Run inference on converted model")
    model = BigVGAN.from_pretrained("mlx_models/bigvgan_v2_24khz_100band_256x", local_files_only=True)
    y = model(x)  # [B, T, 1]
    y = np.array(y.squeeze(2), copy=False)  # [B, T]
    import bigvgan as original
    import torch
    import numpy as np

    print("Run inference on original model")
    original_model = original.BigVGAN.from_pretrained("nvidia/bigvgan_v2_24khz_100band_256x")
    original_model.remove_weight_norm()
    original_model.eval()
    with torch.inference_mode():
        y2 = original_model(torch.from_numpy(np.array(x.swapaxes(1, 2))))
        y2 = y2.view(1, -1).numpy()
    # Compare the outputs
    assert y.shape == y2.shape
    np.testing.assert_allclose(y, y2, rtol=1e-3, atol=1e-3)
    print("Outputs of BigVGAN and MLX BigVGAN match!")
    print("Test passed!")


def test_gen_audio():
    import os
    import numpy as np
    import mlx.core as mx
    from mlx_bigvgan import log_mel_spectrogram, load_audio
    model = BigVGAN.from_pretrained("mlx_models/bigvgan_v2_24khz_100band_256x", local_files_only=True)
    h = model.config
    # Load audio file
    audio = load_audio(os.path.join(os.path.dirname(os.path.abspath(__file__)), "Female_24khz.wav"), h.sampling_rate)
    # Compute log-mel spectrogram
    log_mel_spec = log_mel_spectrogram(
        audio,
        n_fft=h.n_fft,
        n_mels=h.num_mels,
        sample_rate=h.sampling_rate,
        hop_length=h.hop_size,
        fmin=h.fmin,
        fmax=h.fmax,
        padding=(h.n_fft - h.hop_size) // 2,
        mel_norm="slaney",
        mel_scale="slaney",
        power=1.0,
    )
    # reshape to [B(1), T, C_mels]
    log_mel_spec = mx.expand_dims(log_mel_spec, 0)
    # Generate waveform
    waveform = model(log_mel_spec)  # [B(1), T, 1]
    # Reshape to [T, 1]
    waveform_float = waveform.squeeze(0)

    # Convert to int16
    waveform_int16 = mx.clip(waveform_float * 32767, -32768, 32767).astype(mx.int16)

    # save to wav
    import soundfile as sf

    sf.write("output.wav", waveform_int16, h.sampling_rate, "PCM_16")

    print("Audio generated and saved to output.wav")

if __name__ == "__main__":
    # PYTHONPATH=BigVGAN-2.4 python test/bigvgan_test.py
    #test_load_model()
    # test_inference()
    test_gen_audio()
