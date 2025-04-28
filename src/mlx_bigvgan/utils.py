from typing import Union, overload
import mlx.core as mx

if __name__ == "__main__":
    import numpy as np
    x = np.arange(10)
    print(f"np.i0({x.tolist()})=", np.i0(x).tolist())

    print(np.kaiser(5, 12))
