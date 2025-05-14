import mlx.core as mx
import matplotlib.pyplot as plt
from mlx_bigvgan.alias_free_activation import UpSample1d

ratio = 256
T = 100
t = mx.arange(T) / 10.0 * mx.pi
tt = mx.arange(T * ratio) / (10.0 * ratio) * mx.pi

orig_sin = mx.sin(t) + mx.sin(t * 2)
real_up_sin = mx.sin(tt) + mx.sin(tt * 2)

upsample = UpSample1d(ratio)
up_sin = (upsample(orig_sin.reshape(1, T, 1))).reshape(T * ratio)

plt.figure(figsize=(7, 5))
plt.suptitle(f"upsample x{ratio}")
plt.subplot(4, 1, 1)
plt.gca().set_title("original")
plt.plot(t, orig_sin)
plt.tight_layout()
plt.subplot(4, 1, 2)
plt.gca().set_title("real up")
plt.plot(tt, real_up_sin)
plt.tight_layout()
plt.subplot(4, 1, 3)
plt.gca().set_title("upsampled")
plt.plot(tt, up_sin)
plt.tight_layout()
plt.subplot(4, 1, 4)
plt.gca().set_title("Difference")
plt.plot(tt, real_up_sin - up_sin)
plt.tight_layout()
plt.show()
