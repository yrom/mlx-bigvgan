import mlx.core as mx
import matplotlib.pyplot as plt
from mlx_bigvgan.alias_free_activation import DownSample1d


ratio = 10
T = 100
t = mx.arange(T) / 100.0 * mx.pi
tt = mx.arange(T // ratio) / (100.0 / ratio) * mx.pi
# t, tt = t.view(1, 1, -1), tt.view(1, 1, -1)
orig_sin = mx.sin(t) + mx.sin(t * 2)
real_down_sin = mx.sin(tt) + mx.sin(tt * 2)
downsample = DownSample1d(ratio)
down_sin = downsample(orig_sin.reshape(1, T, T)).reshape(T // ratio)

plt.figure(figsize=(7, 5))
plt.suptitle(f"downsample /{ratio}")
plt.subplot(4, 1, 1)
plt.gca().set_title("original")
plt.plot(t, orig_sin)
plt.tight_layout()
plt.subplot(4, 1, 2)
plt.gca().set_title("real down")
plt.plot(tt, real_down_sin)
plt.tight_layout()
plt.subplot(4, 1, 3)
plt.gca().set_title("downsampled")
plt.plot(tt, down_sin)
plt.tight_layout()
plt.subplot(4, 1, 4)
plt.gca().set_title("Difference")
plt.plot(tt, (real_down_sin - down_sin))
plt.tight_layout()
plt.show()