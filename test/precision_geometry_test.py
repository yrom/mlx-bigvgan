"""Portable exact-rational tests; no MLX, Torch, NumPy, weights, or audio required."""

from fractions import Fraction
from pathlib import Path
import random
import importlib.util
import unittest

# Load the pure helper directly: importing the package would import optional MLX.
path = Path(__file__).resolve().parents[1] / "src/mlx_bigvgan/phase_geometry.py"
spec = importlib.util.spec_from_file_location("phase_geometry", path)
geometry = importlib.util.module_from_spec(spec)
spec.loader.exec_module(geometry)
phase_plan = geometry.phase_plan


def scatter_reference(value, weight, bias, stride, padding):
    """Independent definition: scatter each input/kernel product into its output slot."""
    batches, length, inputs = len(value), len(value[0]), len(value[0][0])
    outputs, kernel = len(weight), len(weight[0])
    out_length = (length - 1) * stride - 2 * padding + kernel
    result = [
        [[bias[o] for o in range(outputs)] for _ in range(out_length)]
        for _ in range(batches)
    ]
    for b in range(batches):
        for n in range(length):
            for k in range(kernel):
                t = n * stride - padding + k
                if 0 <= t < out_length:
                    for o in range(outputs):
                        for c in range(inputs):
                            result[b][t][o] += value[b][n][c] * weight[o][k][c]
    return result


def polyphase_reference(value, weight, bias, stride, padding):
    """Evaluate the shared plan as zero-padded correlations, then interleave phases."""
    length, inputs, outputs = len(value[0]), len(value[0][0]), len(weight)
    result = []
    for batch in value:
        phases = []
        for taps, start in phase_plan(len(weight[0]), stride, padding):
            kernel = tuple(reversed(taps))
            phase = []
            for m in range(length):
                row = []
                for o in range(outputs):
                    total = bias[o]
                    for j, tap in enumerate(kernel):
                        source = m + start - (len(kernel) - 1) + j
                        if 0 <= source < length:
                            for c in range(inputs):
                                total += batch[source][c] * weight[o][tap][c]
                    row.append(total)
                phase.append(row)
            phases.append(phase)
        result.append([phases[r][m] for m in range(length) for r in range(stride)])
    return result


class GeometryTests(unittest.TestCase):
    def test_exact_products_edges_bias_and_unequal_phase_lengths(self):
        rng = random.Random(107)

        def scalar():
            return Fraction(rng.randrange(-15, 16), 8)

        for kernel, stride, padding in [
            (4, 2, 1),
            (8, 4, 2),
            (16, 4, 6),
            (7, 3, 2),
            (3, 3, 0),
        ]:
            for length in (1, 2, 7, 11):
                with self.subTest(kernel=kernel, stride=stride, length=length):
                    value = [
                        [[scalar() for _ in range(3)] for _ in range(length)]
                        for _ in range(2)
                    ]
                    weight = [
                        [[scalar() for _ in range(3)] for _ in range(kernel)]
                        for _ in range(5)
                    ]
                    bias = [scalar() for _ in range(5)]
                    actual = polyphase_reference(value, weight, bias, stride, padding)
                    self.assertEqual(
                        actual, scatter_reference(value, weight, bias, stride, padding)
                    )
                    self.assertEqual(len(actual[0]), length * stride)

    def test_zero_channel_padding_and_output_slice_preserve_exact_values(self):
        value = [[[Fraction(n + c, 8) for c in range(3)] for n in range(3)]]
        weight = [
            [[Fraction(o - k + c, 8) for c in range(3)] for k in range(4)]
            for o in range(5)
        ]
        bias = [Fraction(o, 8) for o in range(5)]
        padded_value = [[row + [0] * 13 for row in batch] for batch in value]
        padded_weight = [[row + [0] * 13 for row in channel] for channel in weight] + [
            [[0] * 16 for _ in range(4)] for _ in range(11)
        ]
        padded = polyphase_reference(padded_value, padded_weight, bias + [0] * 11, 2, 1)
        self.assertEqual(
            [[row[:5] for row in batch] for batch in padded],
            scatter_reference(value, weight, bias, 2, 1),
        )
        self.assertTrue(
            all(v == 0 for batch in padded for row in batch for v in row[5:])
        )

    def test_rejects_unsupported_geometry(self):
        for args in ((5, 2, 1), (4, 0, 2), (1, 3, -1), (4, True, 1), (4.0, 2, 1)):
            with self.subTest(args=args), self.assertRaises(ValueError):
                phase_plan(*args)


if __name__ == "__main__":
    unittest.main()
