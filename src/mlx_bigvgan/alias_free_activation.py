# Adapted from https://github.com/junjun3518/alias-free-torch under the Apache License 2.0

import numpy as np
import mlx.nn as nn
import mlx.core as mx

class Activation1d(nn.Module):
    def __init__(
        self,
        activation,
        up_ratio: int = 2,
        down_ratio: int = 2,
        up_kernel_size: int = 12,
        down_kernel_size: int = 12,
    ):
        super().__init__()
        self.up_ratio = up_ratio
        self.down_ratio = down_ratio
        self.act = activation
        self.upsample = UpSample1d(up_ratio, up_kernel_size)
        self.downsample = DownSample1d(down_ratio, down_kernel_size)

    # x: [B,C,T]
    def __call__(self, x):
        x = self.upsample(x)
        x = self.act(x)
        x = self.downsample(x)

        return x


class UpSample1d(nn.Module):
    def __init__(self, ratio=2, kernel_size=None):
        super().__init__()
        self.ratio = ratio
        self.kernel_size = int(6 * ratio // 2) * 2 if kernel_size is None else kernel_size
        self.stride = ratio
        self.pad = self.kernel_size // ratio - 1
        self.pad_left = self.pad * self.stride + (self.kernel_size - self.stride) // 2
        self.pad_right = self.pad * self.stride + (self.kernel_size - self.stride + 1) // 2
        self._filter: mx.array = kaiser_sinc_filter1d(
            cutoff=0.5 / ratio, half_width=0.6 / ratio, kernel_size=self.kernel_size
        )

    # x: [B, C, T]
    def __call__(self, x: mx.array) -> mx.array:
        # [B, C, T] -> [B, T, C]
        x = x.swapaxes(1, 2)
        C = x.shape[-1]
        # [B, T, C] -> [B, T + 2 * pad, C]
        x = mx.pad(
            x,
            [(0, 0), (self.pad, self.pad), (0, 0)],
            mode="edge",
        )
        print(x.shape)
        # [kernel_size] -> [C, kernel_size, C/groups (=1)]
        filter = self._filter.reshape(1, self.kernel_size, 1)
        filter = mx.broadcast_to(filter, (C, self.kernel_size, 1))

        y = self.ratio * mx.conv_transpose1d(x, filter, stride=self.stride, groups=C)
        y = y.swapaxes(1, 2)  # [B, C, T]
        y = y[..., self.pad_left : -self.pad_right]
        return y


class DownSample1d(nn.Module):
    def __init__(self, ratio=2, kernel_size=None):
        super().__init__()
        self.ratio = ratio
        self.kernel_size = int(6 * ratio // 2) * 2 if kernel_size is None else kernel_size
        self.lowpass = LowPassFilter1d(
            cutoff=0.5 / ratio,
            half_width=0.6 / ratio,
            stride=ratio,
            kernel_size=self.kernel_size,
        )

    def __call__(self, x):
        return self.lowpass(x)


# This code is adopted from adefossez's julius.lowpass.LowPassFilters under the MIT License
# https://adefossez.github.io/julius/julius/lowpass.html
def kaiser_sinc_filter1d(cutoff: float, half_width: float, kernel_size: int) -> mx.array:
    """
    Return filter [kernel_size]
    """
    even = kernel_size % 2 == 0
    half_size = kernel_size // 2

    # For kaiser window
    delta_f = 4 * half_width
    A = 2.285 * (half_size - 1) * mx.pi * delta_f + 7.95
    if A > 50.0:
        beta = 0.1102 * (A - 8.7)
    elif A >= 21.0:
        beta = 0.5842 * (A - 21) ** 0.4 + 0.07886 * (A - 21.0)
    else:
        beta = 0.0
    # calculate filter on the CPU (float64)
    with mx.stream(mx.cpu):
        window = kaiser_window(kernel_size, beta=beta)
        if even:
            time = mx.arange(-half_size, half_size) + 0.5
        else:
            time = mx.arange(kernel_size) - half_size
        if cutoff == 0:
            filter_ = mx.zeros_like(time).astype(mx.float64)
        else:
            filter_ = window * 2 * cutoff * sinc(time.astype(mx.float64) * 2 * cutoff)
            """
            Normalize filter to have sum = 1, otherwise we will have a small leakage of the constant component in the input signal.
            """
            filter_ /= mx.sum(filter_)
        filter = filter_.astype(mx.float32)
    return filter


class LowPassFilter1d(nn.Module):
    def __init__(
        self,
        cutoff=0.5,
        half_width=0.6,
        stride: int = 1,
        padding: bool = True,
        padding_mode: str = "edge",
        kernel_size: int = 12,
    ):
        """
        kernel_size should be even number for stylegan3 setup, in this implementation, odd number is also possible.
        """
        super().__init__()
        if cutoff < -0.0:
            raise ValueError("Minimum cutoff must be larger than zero.")
        if cutoff > 0.5:
            raise ValueError("A cutoff above 0.5 does not make sense.")
        self.kernel_size = kernel_size
        self.even = kernel_size % 2 == 0
        self.pad_left = kernel_size // 2 - int(self.even)
        self.pad_right = kernel_size // 2
        self.stride = stride
        self.padding = padding
        self.padding_mode = padding_mode
        self._filter: mx.array = kaiser_sinc_filter1d(cutoff, half_width, kernel_size)

    # Input [B, C, T]
    def __call__(self, x: mx.array) -> mx.array:
        x = x.swapaxes(1, 2)  # [B, T, C]
        C = x.shape[-1]

        if self.padding:
            x = mx.pad(x, [(0, 0), (self.pad_left, self.pad_right), (0, 0)], mode=self.padding_mode)
        filter = self._filter.reshape(1, self.kernel_size, 1)  # [1,kernel_size, 1]
        filter = mx.broadcast_to(filter, (C, self.kernel_size, 1))  # [C, kernel_size, C/groups (=1)]
        y = mx.conv1d(x, filter, stride=self.stride, groups=C)
        y = y.swapaxes(1, 2)  # to [B, C, T]
        return y


def sinc(x: mx.array) -> mx.array:
    """
    Implementation of sinc

    sin(pi * x) / (pi * x)
    """
    x = x.astype(mx.float64)
    y = np.sinc(np.array(x, dtype=np.float64, copy=False))
    return mx.array(y, dtype=mx.float64)


def kaiser_window(
    window_length: int,
    beta: float = 0.0,
) -> mx.array:
    """
    ..math::
        out_i = I_0 \left( \beta \sqrt{1 - \left( {\frac{i - N/2}{N/2}} \right) ^2 } \right) / I_0( \beta )
    return float64 `mlx.core.array`
    """
    return mx.array(np.kaiser(window_length, beta), dtype=mx.float64)
