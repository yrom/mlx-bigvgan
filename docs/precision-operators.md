# Optional inference operators

`BigVGAN.from_pretrained(..., use_precise_ops=True)` applies two inference-only
transformations **after** loading the original checkpoint:

- Replace dense transposed convolutions with nonzero polyphase stride-one
  convolutions. Supported geometry: `kernel_size = 2 * padding + stride`,
  dilation one, output padding zero, and explicit bias.
- Zero-pad dense input/output channels to multiples of 16 and slice the result
  back to the original output channels.

Both preserve the real-arithmetic operator on finite inputs. Floating-point
reduction order and backend kernel selection can change. The option leaves the
default path, global MLX precision policy, checkpoint files, anti-alias resamplers,
activations and output clipping unchanged. Apply it only for inference: the
transformed parameter layout is not a replacement pretrained checkpoint format.
Do not use it for training or reload weights into a transformed model.

```python
from mlx_bigvgan import BigVGAN
model = BigVGAN.from_pretrained("path/to/converted-model", local_files_only=True,
                               use_precise_ops=True)
```

## Phase construction

For input `x`, kernel `w`, stride `s`, padding `p`, and output phase `r`,
`y[m*s+r] = bias + sum_k x[m - (k-p-r)/s] * w[k]`, taking only taps for which
`k-p-r` is divisible by `s`. Reverse the selected taps to form a cross-correlation
kernel. Zero padding and the recorded crop produce that phase; interleaving
phases restores sample order. Phase weights are materialized once during
preparation so gather/reversal helpers do not reappear in every forward graph.
Polyphase decomposition and zero channel padding are established transformations.

## Precision and performance

Polyphase decomposition preserves the real-arithmetic operation, but changes
floating-point reduction order and backend kernel selection. Accuracy and latency
therefore depend on the MLX version, hardware and precision settings. Compare
against the default model on your own inputs before adopting this option.

This integration makes no audio-quality or speedup guarantee. Historical results
from the source implementation do not benchmark this package integration.

## Regression tests

The tests follow the repository's `test/*_test.py` layout. The geometry checks
use exact rational arithmetic and run without third-party dependencies:

```sh
python3 -B -m unittest discover -s test -p 'precision_geometry_test.py' -v
```

For native operator and checkpoint-loading checks, install the package in an
Apple silicon MLX environment, then run:

```sh
pip install -e .
python3 -B -m unittest discover -s test -p 'precision*_test.py' -v
```

The native checks skip only when MLX is not installed. They use small synthetic
inputs and a temporary local checkpoint; no model download or audio is required.
