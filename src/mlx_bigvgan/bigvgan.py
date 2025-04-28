# Copyright 2025 Yrom Wang
#   Licensed under the MIT license.
#
# Adapted from: https://github.com/NVIDIA/BigVGAN/commit/7d2b454564a6c7d014227f635b7423881f14bdac
# Copyright (c) 2024 NVIDIA CORPORATION.
#   Licensed under the MIT license.


from dataclasses import dataclass
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Literal, Union
import mlx.nn as nn
import mlx.core as mx

from .alias_free_activation import Activation1d
from .act import Snake, SnakeBeta


@dataclass
class BigVGANConfig:
    """
    Default configuration: nvidia/bigvgan_v2_24khz_100band_256x.
    Args:
        num_mels (int): Number of mel frequency bins.
        upsample_initial_channel (int): Initial number of channels for upsampling.
        upsample_rates (list): List of upsampling rates.
        upsample_kernel_sizes (list): List of kernel sizes for upsampling.
        resblock_kernel_sizes (list): List of kernel sizes for residual blocks.
        resblock_dilation_sizes (list): List of dilation sizes for residual blocks.
        activation (str): Activation function type. Should be either 'snake' or 'snakebeta'.
        snake_logscale (bool): Whether to use log scale for Snake activation.
    """

    num_mels: int = 100
    upsample_initial_channel: int = 1536
    upsample_rates = [4, 4, 2, 2, 2, 2]
    upsample_kernel_sizes = [8, 8, 4, 4, 4, 4]
    resblock: Literal["1", "2"] = "1"
    resblock_kernel_sizes = [3, 7, 11]
    resblock_dilation_sizes = [[1, 3, 5], [1, 3, 5], [1, 3, 5]]
    activation: str = "snakebeta"
    snake_logscale: bool = True


class BigVGAN(nn.Module):
    """
    BigVGAN is a neural vocoder model that applies anti-aliased periodic activation for residual blocks (resblocks).
    New in BigVGAN-v2: it can optionally use optimized CUDA kernels for AMP (anti-aliased multi-periodicity) blocks.

    Args:
        h (dict): Hyperparameters.

    """

    def __init__(self, config):
        super().__init__()

        self.num_kernels = len(config.resblock_kernel_sizes)
        self.num_upsamples = len(config.upsample_rates)

        # Pre-conv
        self.conv_pre = nn.Conv1d(config.num_mels, config.upsample_initial_channel, kernel_size=7, stride=1, padding=3)
        if config.resblock == "1":
            resblock_class = AMPBlock1
        elif config.resblock == "2":
            resblock_class = AMPBlock2
        else:
            raise ValueError(f"Incorrect resblock class specified. Got {config.resblock}")

        # Transposed conv-based upsamplers. does not apply anti-aliasing
        self.ups: list[nn.Module] = []  # num_upsamples
        self.resblocks: list[list[Union[AMPBlock1, AMPBlock2]]] = []
        final_out_channels = 0
        for i, (u, k) in enumerate(zip(config.upsample_rates, config.upsample_kernel_sizes)):
            in_channels = config.upsample_initial_channel // (2**i)
            out_channels = config.upsample_initial_channel // (2 ** (i + 1))
            final_out_channels = out_channels
            self.ups.append(
                nn.ConvTranspose1d(
                    in_channels,
                    out_channels,
                    kernel_size=k,
                    stride=u,
                    padding=(k - u) // 2,
                )
            )
            # Residual blocks using anti-aliased multi-periodicity composition modules (AMP)
            self.resblocks.append(
                [
                    resblock_class(out_channels, k, tuple(d), config.activation, config.snake_logscale)
                    for k, d in zip(config.resblock_kernel_sizes, config.resblock_dilation_sizes)
                ]  # num_kernels
            )
            assert len(self.resblocks[-1]) == self.num_kernels
        assert len(self.ups) == self.num_upsamples
        # Post-conv
        if config.activation == "snake":
            activation_post = Snake(final_out_channels, alpha_logscale=config.snake_logscale)
        elif config.activation == "snakebeta":
            activation_post = SnakeBeta(final_out_channels, alpha_logscale=config.snake_logscale)
        else:
            raise NotImplementedError(
                "activation incorrectly specified. check the config file and look for 'activation'."
            )
        self.activation_post = Activation1d(activation=activation_post)

        self.conv_post = nn.Conv1d(out_channels, 1, kernel_size=7, stride=1, padding=3, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        """
        x: [B, C, T]
        Returns: [B, 1, T]
        """
        # Pre-conv
        x = self.conv_pre(x)

        for i, upsample in enumerate(self.ups):
            # Upsampling
            x = upsample(x)
            # AMP blocks
            ampblocks = self.resblocks[i]

            y = ampblocks[0](x)
            for resblock in ampblocks[1:]:
                y += resblock(x)
            x = y / self.num_kernels

        # Post-conv
        x = self.activation_post(x)
        x = self.conv_post(x)
        x = mx.clip(x, -1.0, 1.0)  # Bound the output to [-1, 1]

        return x

    # def remove_weight_norm(self):
    #     try:
    #         print("Removing weight norm...")
    #         for l in self.ups:
    #             for l_i in l:
    #                 remove_weight_norm(l_i)
    #         for l in self.resblocks:
    #             l.remove_weight_norm()
    #         remove_weight_norm(self.conv_pre)
    #         remove_weight_norm(self.conv_post)
    #     except ValueError:
    #         print("[INFO] Model already removed weight norm. Skipping!")
    #         pass

    @classmethod
    def sanitize(cls, weights):
        replacement_patterns = [

        ]
        ignored_keys = [""]
        def replace_key(key: str) -> str:
            for old, new in replacement_patterns:
                key = key.replace(old, new)
            return key

        weights = {replace_key(k): v for k, v in weights.items()}
        for key in ignored_keys:
            if key in weights:
                del weights[key]
        return weights

    @classmethod
    def from_pretrained(
        cls, path_or_repo: str, local_files_only: bool = False, dtype: mx.Dtype = mx.float32
    ) -> "BigVGAN":
        """
        Load a pretrained BigVGAN model from a local path or Hugging Face Hub.
        Args:
            path_or_repo (str): e.g. nvidia/bigvgan_v2_24khz_100band_256x
            dtype (mx.Dtype): Data type for the model weights. Default is mx.float32.
        """
        import torch
        from huggingface_hub import hf_hub_download
        model_dir = Path(path_or_repo)
        config_file = model_dir / "config.json"
        if not config_file.exists():
            config_file = Path(
                hf_hub_download(
                    repo_id=path_or_repo,
                    filename="config.json",
                    local_files_only=local_files_only,
                )
            )

        with open(config_file, "r", encoding="utf-8") as f:
            config = SimpleNamespace(**json.load(f))

        model = BigVGAN(config)

        model_file = model_dir / "bigvgan_generator.pt"
        if not model_file.exists():
            print(f"Loading weights from huggingface {path_or_repo}")
            model_file = Path(
                hf_hub_download(
                    repo_id=path_or_repo,
                    filename="bigvgan_generator.pt",
                    local_files_only=local_files_only,
                )
            )
        weights = torch.load(model_file, map_location="cpu")
        print(weights)
        weights = cls.sanitize(weights)
        weights = {k: v.astype(dtype) for k, v in weights.items()}
        model.load_weights(list(weights.items()))
        return model


def get_padding(kernel_size, dilation=1):
    return int((kernel_size * dilation - dilation) // 2)


class AMPBlock1(nn.Module):
    """
    AMPBlock applies Snake / SnakeBeta activation functions with trainable parameters that control periodicity, defined for each layer.
    AMPBlock1 has additional self.convs2 that contains additional Conv1d layers with a fixed dilation=1 followed by each layer in self.convs1

    Args:
        channels (int): Number of convolution channels.
        kernel_size (int): Size of the convolution kernel. Default is 3.
        dilation (tuple): Dilation rates for the convolutions. Each dilation layer has two convolutions. Default is (1, 3, 5).
        activation (str): Activation function type. Should be either 'snake' or 'snakebeta'. Default is `snakebeta`.
        snake_logscale
    """

    def __init__(
        self,
        channels: int,
        kernel_size: int = 3,
        dilation: tuple = (1, 3, 5),
        activation="snakebeta",
        snake_logscale: bool = True,
    ):
        super().__init__()
        # Activation functions
        if activation == "snake":
            activation_class = Snake

        elif activation == "snakebeta":
            activation_class = SnakeBeta
        else:
            raise NotImplementedError(
                "activation incorrectly specified. check the config file and look for 'activation'."
            )
        self.blocks = [
            nn.Sequential(
                Activation1d(
                    activation=activation_class(channels, alpha_logscale=snake_logscale),
                ),
                nn.Conv1d(
                    channels,
                    channels,
                    kernel_size,
                    stride=1,
                    dilation=d,
                    padding=get_padding(kernel_size, 1),
                ),
                Activation1d(
                    activation=activation_class(channels, alpha_logscale=snake_logscale),
                ),
                nn.Conv1d(
                    channels,
                    channels,
                    kernel_size,
                    stride=1,
                    dilation=1,
                    padding=get_padding(kernel_size, 1),
                ),
            )
            for d in dilation
        ]

    def __call__(self, x):
        # acts1, acts2 = self.activations[::2], self.activations[1::2]
        # for c1, c2, a1, a2 in zip(self.convs1, self.convs2, acts1, acts2):
        #     xt = a1(x)
        #     xt = c1(xt)
        #     xt = a2(xt)
        #     xt = c2(xt)
        #     x = xt + x
        for b in self.blocks:
            xt = b(x)
            x = xt + x
        return x


class AMPBlock2(nn.Module):
    """
    AMPBlock applies Snake / SnakeBeta activation functions with trainable parameters that control periodicity, defined for each layer.
    Unlike AMPBlock1, AMPBlock2 does not contain extra Conv1d layers with fixed dilation=1

    Args:
        h (AttrDict): Hyperparameters.
        channels (int): Number of convolution channels.
        kernel_size (int): Size of the convolution kernel. Default is 3.
        dilation (tuple): Dilation rates for the convolutions. Each dilation layer has two convolutions. Default is (1, 3, 5).
        activation (str): Activation function type. Should be either 'snake' or 'snakebeta'. Default is None.
        snake_logscale
    """

    def __init__(
        self,
        channels: int,
        kernel_size: int = 3,
        dilation: tuple = (1, 3, 5),
        activation="snakebeta",
        snake_logscale: bool = True,
    ):
        super().__init__()

        # Activation functions
        if activation == "snake":
            activation_class = Snake

        elif activation == "snakebeta":
            activation_class = SnakeBeta
        else:
            raise NotImplementedError(
                "activation incorrectly specified. check the config file and look for 'activation'."
            )

        self.blocks = [
            nn.Sequential(
                Activation1d(
                    activation=activation_class(channels, alpha_logscale=snake_logscale),
                ),
                nn.Conv1d(
                    channels,
                    channels,
                    kernel_size,
                    stride=1,
                    dilation=d,
                    padding=get_padding(kernel_size, 1),
                ),
            )
            for d in dilation
        ]

    def __call__(self, x):
        for b in self.blocks:
            xt = b(x)
            x = xt + x
        return x

    # def remove_weight_norm(self):
    #     for l in self.convs:
    #         remove_weight_norm(l)

if __name__ == "__main__":
    # Example usage
    config = BigVGANConfig()
    model = BigVGAN(config)
    print(model)
    x = mx.random.normal(0, 1, (1, config.num_mels, 256)) 
    output = model(x)
    print(output.shape) 