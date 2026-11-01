# Copyright (c) 2026 Mason Wang
# SPDX-License-Identifier: MIT
# Extracted from Waveform Last Mile's credited MLX implementation; see NOTICE-precision.md.
"""Inference-only, channels-last MLX operators; no model loading or global policy edits."""

from __future__ import annotations

import math

import mlx.core as mx
import mlx.nn as nn

from .phase_geometry import phase_plan


class PreciseConv1d(nn.Module):
    """Dense Conv1d with zero-padded channel tiles and exact output slicing."""

    def __init__(self, source: nn.Conv1d, tile: int = 16) -> None:
        super().__init__()
        if type(tile) is not int or tile != 16:
            raise ValueError("registered precise path requires channel tile 16")
        output_channels, kernel_size, input_channels = source.weight.shape
        if source.groups != 1:
            raise ValueError("PreciseConv1d only supports dense convolutions")
        padded_input = math.ceil(input_channels / tile) * tile
        padded_output = math.ceil(output_channels / tile) * tile
        self.input_padding = padded_input - input_channels
        self.output_channels = output_channels
        self.stride = source.stride
        self.padding = source.padding
        self.dilation = source.dilation
        self.weight = mx.pad(
            source.weight,
            [(0, padded_output - output_channels), (0, 0), (0, self.input_padding)],
        )
        if "bias" in source:
            self.bias = mx.pad(source.bias, [(0, padded_output - output_channels)])

    def __call__(self, value: mx.array) -> mx.array:
        if self.input_padding:
            value = mx.pad(value, [(0, 0), (0, 0), (0, self.input_padding)])
        output = mx.conv1d(
            value,
            self.weight,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
        )
        if "bias" in self:
            output = output + self.bias
        return output[:, :, : self.output_channels]


class PhaseConvTranspose1d(nn.Module):
    """Stride-s transposed convolution as nonzero polyphase Conv1d branches."""

    def __init__(self, source: nn.ConvTranspose1d, tile: int = 16) -> None:
        super().__init__()
        if type(tile) is not int or tile != 16:
            raise ValueError("registered precise path requires channel tile 16")
        if source.dilation != 1 or getattr(source, "output_padding", 0) != 0:
            raise ValueError(
                "phase decomposition requires dilation=1 and output_padding=0"
            )
        output_channels, kernel_size, input_channels = source.weight.shape
        if kernel_size != 2 * source.padding + source.stride:
            raise ValueError(
                "phase decomposition requires exact length-preserving BigVGAN geometry"
            )
        if "bias" not in source:
            raise ValueError(
                "phase decomposition requires the checkpoint's explicit bias"
            )
        padded_input = math.ceil(input_channels / tile) * tile
        padded_output = math.ceil(output_channels / tile) * tile
        self.input_padding = padded_input - input_channels
        self.output_channels = output_channels
        self.stride = source.stride
        padded_weight = mx.pad(
            source.weight,
            [(0, padded_output - output_channels), (0, 0), (0, self.input_padding)],
        )
        self.bias = mx.pad(source.bias, [(0, padded_output - output_channels)])
        self.phase_weights: list[mx.array] = []
        self.phase_starts: list[int] = []
        if getattr(source, "groups", 1) != 1:
            raise ValueError("phase decomposition only supports dense convolutions")
        for ordered_taps, start in phase_plan(
            kernel_size, source.stride, source.padding
        ):
            ordered_taps = list(ordered_taps)
            # Materialize this reversed phase kernel once.  Leaving the gather/reversal expression
            # lazy here would reinsert its indexing helpers into every exported inference graph.
            phase_weight = padded_weight[:, ordered_taps, :][:, ::-1, :]
            mx.eval(phase_weight)
            self.phase_weights.append(phase_weight)
            self.phase_starts.append(start)

    def __call__(self, value: mx.array) -> mx.array:
        if self.input_padding:
            value = mx.pad(value, [(0, 0), (0, 0), (0, self.input_padding)])
        outputs = []
        for phase_weight, start in zip(
            self.phase_weights, self.phase_starts, strict=True
        ):
            phase = mx.conv1d(value, phase_weight, padding=phase_weight.shape[1] - 1)
            outputs.append(phase[:, start : start + value.shape[1], :] + self.bias)
        output = mx.stack(outputs, axis=2).reshape(
            value.shape[0], value.shape[1] * self.stride, -1
        )
        return output[:, :, : self.output_channels]


def apply_precise_operators(model: nn.Module) -> None:
    """Opt-in inference conversion after loading weights; no process-wide policy changes.

    Prepare all replacements before installing them. The resulting parameter layout is
    intended for inference, not saving a replacement pretrained checkpoint or training.
    """
    if model.training:
        raise ValueError("call model.eval() before preparing inference operators")
    replacements = []
    for name in ("conv_pre", "conv_post"):
        layer = getattr(model, name)
        out_channels, _, in_channels = layer.weight.shape
        if in_channels % 16 or out_channels % 16:
            replacements.append((model, name, PreciseConv1d(layer)))
    upsamplers = [PhaseConvTranspose1d(layer) for layer in model.ups]
    for block in model.resblocks:
        for sequence in block.layers:
            layers = []
            for layer in sequence.layers:
                if isinstance(layer, nn.Conv1d):
                    out_channels, _, in_channels = layer.weight.shape
                    if in_channels % 16 or out_channels % 16:
                        layer = PreciseConv1d(layer)
                layers.append(layer)
            replacements.append((sequence, "layers", layers))
    for owner, name, replacement in replacements:
        setattr(owner, name, replacement)
    model.ups = upsamplers
