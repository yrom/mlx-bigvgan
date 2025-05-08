from types import SimpleNamespace
from mlx_bigvgan.alias_free_activation import Activation1d
from mlx_bigvgan.act import SnakeBeta
import mlx.nn as nn
import mlx.core as mx
import numpy as np
from mlx.utils import tree_unflatten


def test_activation():
    from alias_free_activation.torch import Activation1d as OriginActivation1d
    from activations import SnakeBeta as OriginSnakeBeta
    import torch
    channels = 256
    # Dummy input [B,  C, T]
    x = np.random.uniform(low=-1, high=1, size=(1, channels, 1024))
    origin_act = OriginActivation1d(
        activation=OriginSnakeBeta(channels, alpha_logscale=True),
        up_ratio=2,
        down_ratio=2,
        up_kernel_size=12,
        down_kernel_size=12,
    )
    origin_act = origin_act.eval()
    torch_x = torch.from_numpy(x).float()
    torch_y = origin_act(torch_x)
    torch_y = torch_y.detach()

    act = Activation1d(
        activation=SnakeBeta(channels, alpha_logscale=True),
        up_ratio=2,
        down_ratio=2,
        up_kernel_size=12,
        down_kernel_size=12,
    )

    act = act.eval()
    act.act.alpha = mx.array(origin_act.act.alpha.numpy(force=True))
    act.act.beta = mx.array(origin_act.act.beta.numpy(force=True))
    act.upsample.filter = mx.array(origin_act.upsample.filter.numpy(force=True))
    act.downsample.lowpass.filter = mx.array(origin_act.downsample.lowpass.filter.numpy(force=True))
    # [B, C, T] -> [B, T, C]
    nx = mx.array(x.swapaxes(1, 2))
    ny = act(nx)


    np.testing.assert_allclose(
        np.array(ny.swapaxes(1, 2)), torch_y.numpy(), rtol=1e-4, atol=1e-4, err_msg="Failed to match ny and torch_y"
    )


    kernel_size = 3
    dilation = 1
    padding = (kernel_size * dilation - dilation) // 2
  
    mx_conv1d = nn.Conv1d(
        channels,
        channels,
        kernel_size,
        stride=1,
        dilation=dilation,
        padding=padding,
    ).eval()

    torch_conv1d = torch.nn.Conv1d(
        channels,
        channels,
        kernel_size,
        stride=1,
        dilation=dilation,
        padding=padding,
    ).eval()
    
    nx = ny
    ny = mx_conv1d(nx)
    torch_conv1d.weight = torch.nn.Parameter(torch.tensor(
        np.array(mx_conv1d.weight.swapaxes(1, 2), copy=False),
    ))
    torch_conv1d.bias = torch.nn.Parameter(torch.tensor(
        np.array(mx_conv1d.bias, copy=False),
    ))
    torch_x = torch_y
    torch_y = torch_conv1d(torch_x)
    torch_y = torch_y.detach()
    np.testing.assert_allclose(
        np.array(ny.swapaxes(1, 2)), torch_y.numpy(), rtol=1e-4, atol=1e-4, err_msg="Failed to match ny and torch_y"
    )


def test_ampblock():
    from mlx_bigvgan.bigvgan import AMPBlock1 as MlxAMPBlock
    from bigvgan import AMPBlock1 as OriginAMPBlock
    from env import AttrDict
    import torch
    import time
    h = AttrDict(
        activation="snakebeta",
        snake_logscale=True,
        # use_cuda_kernel=False,
    )
    channels = 256
    kernal_size = 3
    dilation = (1, 3, 5)
    o_ampblock = OriginAMPBlock(h, channels, kernal_size, dilation, activation=h.activation)
    o_ampblock = o_ampblock.eval()
    # [B, C, T]
    x = np.random.uniform(low=-1, high=1, size=(1, channels, 1024))
    torch_x = torch.from_numpy(x).float()
    torch_y = o_ampblock(torch_x)
    o_ampblock.remove_weight_norm()
    time_start = time.perf_counter()
    with torch.no_grad():
        torch_y = o_ampblock(torch_x)
    torch_y = torch_y.detach()
    time_end = time.perf_counter()
    print(f"torch AMPBlock1 time: {time_end - time_start:.4f}s")
    mlx_ampblock = MlxAMPBlock(channels, kernal_size, dilation, activation=h.activation, snake_logscale=h.snake_logscale)
    mlx_ampblock.eval()
    
    # copy parameters
    weights = {
        "layers": [
            {
                "layers": [None] * 4
            }
            for d in dilation
        ]
    }
    o_weights = dict(o_ampblock.named_parameters())
    o_weights.update(dict(o_ampblock.named_buffers()))
    n_weights = {}
    for k, v in o_weights.items():
        if k.startswith("convs") and k.endswith(".weight"):
            #  [out_channels, in_channels, kernel_size] -> (out_channels, kernel_size, in_channels)
            v = v.swapaxes(1, 2)
        elif k.endswith(".filter"):
            v = v.reshape(-1)
        n_weights[k] = mx.array(v.numpy(force=True))
    n_weights = tree_unflatten([(k, v) for k, v in n_weights.items()])
    for i, d in enumerate(dilation):
        weights["layers"][i]["layers"][0] = n_weights["activations"][i * 2]
        # convs1
        weights["layers"][i]["layers"][1] = n_weights["convs1"][i]
        # act2
        weights["layers"][i]["layers"][2] = n_weights["activations"][i * 2 + 1]
        # convs2
        weights["layers"][i]["layers"][3] = n_weights["convs2"][i]
    mlx_ampblock.update(weights)
    mx.eval(mlx_ampblock.parameters())
    # [B, C, T] -> [B, T, C]
    nx = mx.array(x.swapaxes(1, 2))
    # ny = mlx_ampblock(nx)
    mx.eval(nx)
    time_start = time.perf_counter()
    ny = mlx_ampblock(nx)
    mx.eval(ny)
    time_end = time.perf_counter()
    print(f"mlx AMPBlock1 time: {time_end - time_start:.4f}s")
    np.testing.assert_allclose(
        np.array(ny.swapaxes(1, 2)), torch_y.numpy(), rtol=1e-4, atol=1e-4, err_msg="Failed to match ny and torch_y"
    )
    


if __name__ == "__main__":
    # PYTHONPATH=BigVGAN-2.4 pytest test/ampblock_test.py
    test_ampblock()
