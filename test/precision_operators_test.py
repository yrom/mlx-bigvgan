"""Tiny operator checks; run with the package installed on Apple silicon."""

import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from precision_geometry_test import scatter_reference


@unittest.skipUnless(importlib.util.find_spec("mlx"), "requires Apple MLX")
class MLXTests(unittest.TestCase):
    def test_from_pretrained_keeps_default_and_converts_after_loading(self):
        import mlx.core as mx
        from mlx.utils import tree_flatten

        from mlx_bigvgan import BigVGAN
        from mlx_bigvgan.precision import PhaseConvTranspose1d, PreciseConv1d

        # Check loading and conversion independently of GPU precision settings.
        for resblock in ("1", "2"):
            with (
                self.subTest(resblock=resblock),
                mx.stream(mx.cpu),
                tempfile.TemporaryDirectory() as directory,
            ):
                config = {
                    "num_mels": 3,
                    "upsample_initial_channel": 8,
                    "upsample_rates": [2],
                    "upsample_kernel_sizes": [4],
                    "resblock": resblock,
                    "resblock_kernel_sizes": [3],
                    "resblock_dilation_sizes": [[1, 3]],
                    "activation": "snakebeta",
                    "snake_logscale": True,
                }
                mx.random.seed(17)
                original = BigVGAN(SimpleNamespace(**config))
                weights = dict(tree_flatten(original.parameters()))
                path = Path(directory)
                (path / "config.json").write_text(json.dumps(config), encoding="utf-8")
                mx.save_safetensors(str(path / "model.safetensors"), weights)

                default = BigVGAN.from_pretrained(directory, local_files_only=True)
                precise = BigVGAN.from_pretrained(
                    directory, local_files_only=True, use_precise_ops=True
                )
                self.assertEqual(
                    set(dict(tree_flatten(default.parameters()))), set(weights)
                )
                self.assertNotIsInstance(default.ups[0], PhaseConvTranspose1d)
                self.assertIsInstance(precise.ups[0], PhaseConvTranspose1d)
                self.assertIsInstance(precise.conv_pre, PreciseConv1d)
                self.assertIsInstance(precise.conv_post, PreciseConv1d)
                self.assertNotIn("bias", precise.conv_post)
                self.assertFalse(precise.training)
                for length in (1, 5):
                    value = mx.random.uniform(
                        -0.2, 0.2, (2, length, config["num_mels"])
                    )
                    expected = original(value)
                    self.assertTrue(mx.allclose(default(value), expected).item())
                    actual = precise(value)
                    self.assertEqual(actual.shape, (2, length * 2, 1))
                    self.assertTrue(
                        mx.allclose(actual, expected, rtol=1e-5, atol=1e-6).item()
                    )

    def test_actual_polyphase_operator_against_independent_scatter(self):
        import mlx.core as mx
        import mlx.nn as nn
        from mlx_bigvgan.precision import PhaseConvTranspose1d

        for kernel, stride, padding in ((4, 2, 1), (8, 4, 2), (16, 4, 6)):
            with self.subTest(kernel=kernel, stride=stride):
                value = [[[(n * 3 + c - 10) / 32 for c in range(3)] for n in range(7)]]
                weight = [
                    [[(o - k + c) / 64 for c in range(3)] for k in range(kernel)]
                    for o in range(5)
                ]
                bias = [o / 32 for o in range(5)]
                layer = nn.ConvTranspose1d(3, 5, kernel, stride=stride, padding=padding)
                layer.weight = mx.array(weight, dtype=mx.float32)
                layer.bias = mx.array(bias, dtype=mx.float32)
                operator = PhaseConvTranspose1d(layer)
                actual = operator(mx.array(value, dtype=mx.float32)).tolist()
                expected = scatter_reference(value, weight, bias, stride, padding)
                for a_batch, e_batch in zip(actual, expected, strict=True):
                    for a_row, e_row in zip(a_batch, e_batch, strict=True):
                        for a, e in zip(a_row, e_row, strict=True):
                            self.assertAlmostEqual(a, e, delta=5e-6)
                self.assertEqual(len(operator.phase_weights), stride)

                # MLX 0.22.1 has no output_padding attribute; it always uses zero.
                if "output_padding" in layer:
                    del layer["output_padding"]
                legacy = PhaseConvTranspose1d(layer)
                self.assertEqual(
                    legacy(mx.array(value, dtype=mx.float32)).tolist(), actual
                )

                layer.output_padding = 1
                with self.assertRaisesRegex(ValueError, "output_padding=0"):
                    PhaseConvTranspose1d(layer)

    def test_dense_padding_against_direct_correlation(self):
        import mlx.core as mx
        import mlx.nn as nn
        from mlx_bigvgan.precision import PreciseConv1d

        value = [[[(n + c - 5) / 16 for c in range(3)] for n in range(7)]]
        weight = [
            [[(o - k + c) / 32 for c in range(3)] for k in range(3)] for o in range(5)
        ]
        layer = nn.Conv1d(3, 5, 3, padding=1)
        layer.weight = mx.array(weight, dtype=mx.float32)
        layer.bias = mx.array([o / 16 for o in range(5)], dtype=mx.float32)
        actual = PreciseConv1d(layer)(mx.array(value, dtype=mx.float32)).tolist()[0]
        for n, row in enumerate(actual):
            for o, measured in enumerate(row):
                expected = o / 16 + sum(
                    value[0][n + k - 1][c] * weight[o][k][c]
                    for k in range(3)
                    if 0 <= n + k - 1 < 7
                    for c in range(3)
                )
                self.assertAlmostEqual(measured, expected, delta=5e-6)


if __name__ == "__main__":
    unittest.main()
