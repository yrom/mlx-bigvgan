from mlx_bigvgan import BigVGAN

from mlx.utils import tree_flatten


def test_load_model():
    # Run after convert.py
    model = BigVGAN.from_pretrained("mlx_models/bigvgan_v2_24khz_100band_256x", local_files_only=True)
    print("\n".join([k for k, _ in tree_flatten(model.parameters())]))


if __name__ == "__main__":
    test_load_model()
