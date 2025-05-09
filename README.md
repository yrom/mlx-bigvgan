# MLX BigVGAN

An MLX-adapted implementation of [BigVGAN](https://github.com/NVIDIA/BigVGAN).

## Features

- **BigVGAN Integration**: Fully integrates the original BigVGAN model with MLX for enhanced compatibility and performance.
- **Flexible Conversion**: Includes tools to convert the original BigVGAN PyTorch weights to MLX format.
- **Customizable Configurations**: Supports various configurations for kernel sizes, dilation rates, and activation functions (e.g., `snake`, `snakebeta`).
- **Pretrained Models**: Easily load pretrained BigVGAN models from the Hugging Face Hub.

## Installation

```bash
pip install mlx-bigvgan
```

## Usage


### 1. Load Pretrained Model
```python
from mlx_bigvgan import BigVGAN

model = BigVGAN.from_pretrained("wyrom/mlx-bigvgan_v2_24khz_100band_256x")
model.eval()
mx.eval(model.parameters())
```


### 2. Generate Audio
```python
import numpy as np
import mlx.core as mx
from mlx_bigvgan import log_mel_spectrogram, load_audio
# Load audio file
audio = load_audio("path/to/audio.wav")
h = model.config
# Compute log-mel spectrogram
mel_spec = log_mel_spectrogram(audio,
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
mel_spec = mx.expand_dims(mel_spec, 0)
# Generate waveform
waveform = model(mel_spec) # [B(1), T, 1]
# Reshape to [T, 1]
waveform_float = waveform.squeeze(0)

# Convert to int16
waveform_int16 = mx.clip(waveform_float * 32767, -32768, 32767).astype(mx.int16)

# save to wav
import soundfile as sf

sf.write("output.wav", waveform_int16, h.sampling_rate, "PCM_16")
```

### 3. Convert Original BigVGAN Weights to MLX Format

You can convert the original BigVGAN weights to MLX format using the provided script. 

`repo_id` is the Hugging Face model ID of the original BigVGAN model you want to convert. 

See [nvidia/BigVGAN](https://huggingface.co/collections/nvidia/bigvgan-66959df3d97fd7d98d97dc9a) for move pretrained models.

```bash
python -m mlx_bigvgan.convert --repo_id nvidia/bigvgan_v2_xxx  --output_dir mlx_models
```

## References

- [BigVGAN](https://github.com/NVIDIA/BigVGAN)
- [MLX](https://github.com/ml-explore/mlx)

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

