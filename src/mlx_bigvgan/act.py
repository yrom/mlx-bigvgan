import mlx.core as mx
import mlx.nn as nn


class Snake(nn.Module):
    """
    Shape:
        - Input: (B, T, C)
        - Output: (B, T, C), same shape as the input
    References:
        - This activation function is from this paper by Liu Ziyin, Tilman Hartwig, Masahito Ueda:
        https://arxiv.org/abs/2006.08195
    """

    def __init__(self, channels: int, alpha_logscale: bool = False):
        super(Snake, self).__init__()
        self.channels = channels
        self.alpha_logscale = alpha_logscale
        if self.alpha_logscale:
            self.alpha = mx.zeros(channels)
        else:
            self.alpha = mx.ones(channels)
        self.no_div_by_zero = 1e-9

    def __call__(self, x: mx.array):
        """
        Applies the function to the input elementwise.
        Snake ∶= x + 1/a * sin^2 (xa)
        """
        # Line up with x to [B, T, C]
        alpha = self.alpha.reshape(1, 1, self.channels)
        if self.alpha_logscale:
            alpha = mx.exp(alpha)

        return x + (1.0 / (alpha + self.no_div_by_zero)) * (mx.power(mx.sin(x * alpha), 2))

    def _extra_repr(self) -> str:
        return f"channels={self.channels}, alpha={self.alpha}, alpha_logscale={self.alpha_logscale}"


class SnakeBeta(nn.Module):
    """
    Shape:
        - Input: (B, T, C)
        - Output: (B, T, C), same shape as the input
    """

    def __init__(self, channels: int, alpha_logscale: bool = False):
        super(SnakeBeta, self).__init__()
        self.channels = channels
        self.alpha_logscale = alpha_logscale
        if self.alpha_logscale:
            self.alpha = mx.zeros(channels)
            self.beta = mx.zeros(channels)
        else:
            self.alpha = mx.ones(channels)
            self.beta = mx.ones(channels)
        self.no_div_by_zero = 1e-9

    def __call__(self, x: mx.array):
        """
        Applies the function to the input elementwise.
        SnakeBeta ∶= x + 1/b * sin^2 (xa)
        """
        # Line up with x to [B, T, C]
        alpha = self.alpha.reshape(1, 1, self.channels)
        beta = self.beta.reshape(1, 1, self.channels)
        if self.alpha_logscale:
            alpha = mx.exp(alpha)
            beta = mx.exp(beta)

        return x + (1.0 / (beta + self.no_div_by_zero)) * (mx.power(mx.sin(x * alpha), 2))

    def _extra_repr(self) -> str:
        return f"channels={self.channels}, alpha={self.alpha}, beta={self.beta}, alpha_logscale={self.alpha_logscale}"


if __name__ == "__main__":
    s = SnakeBeta(256, alpha_logscale=True)
    mx.eval(s.parameters())
    # print(s.trainable_parameters())
