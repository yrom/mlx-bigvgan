# Copyright (c) 2026 Mason Wang
# SPDX-License-Identifier: MIT
"""Static tap selection shared by the MLX operator and portable algebra checks."""


def phase_plan(
    kernel_size: int, stride: int, padding: int
) -> tuple[tuple[tuple[int, ...], int], ...]:
    """Return (ascending source taps, convolution crop start) for each output phase.

    For y[m * stride + r], tap k reads x[m - (k - padding - r) / stride].
    Reversing those selected taps makes an ordinary cross-correlation kernel.
    Zero padding by len(taps)-1 and cropping at -min(offsets) gives exactly m.
    The supported geometry emits input_length * stride samples.
    """
    if any(type(v) is not int for v in (kernel_size, stride, padding)):
        raise ValueError("kernel_size, stride and padding must be integers")
    if stride <= 0 or padding < 0 or kernel_size != 2 * padding + stride:
        raise ValueError(
            "requires positive stride and kernel_size == 2 * padding + stride"
        )
    phases = []
    for phase in range(stride):
        taps = tuple(
            k for k in range(kernel_size) if (k - padding - phase) % stride == 0
        )
        offsets = tuple((k - padding - phase) // stride for k in taps)
        if not offsets:
            raise ValueError("empty output phase")
        phases.append((taps, -min(offsets)))
    return tuple(phases)
