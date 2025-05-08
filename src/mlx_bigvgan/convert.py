import json
from pathlib import Path
from textwrap import dedent
from types import SimpleNamespace
from typing import Any, Dict, List, Tuple, Union

import mlx.core as mx
from huggingface_hub import hf_hub_download
from mlx.utils import tree_flatten, tree_unflatten

from .bigvgan import BigVGAN


def upload_to_hub(path: str, upload_repo: str, hf_path: str):
    """
    Uploads the model to Hugging Face hub.

    Args:
        path (str): Local path to the model.
        upload_repo (str): Name of the HF repo to upload to.
        hf_path (str): Path to the original Hugging Face model.
    """
    import os

    from huggingface_hub import HfApi, ModelCard, logging

    content = dedent(
        f"""
        ---
        language: en
        license: other
        library: mlx
        base_model: 
        - {hf_path}
        tags:
        - mlx
        ---

        The Model [{upload_repo}](https://huggingface.co/{upload_repo}) was
        converted to MLX format from
        [{hf_path}](https://huggingface.co/{hf_path}).

        This model is intended to be used with the [MLX BigVGAN](https://github.com/yrom/mlx-bigvgan).
        """
    )

    card = ModelCard(content)
    card.save(os.path.join(path, "README.md"))

    logging.set_verbosity_info()

    api = HfApi()
    api.create_repo(repo_id=upload_repo, exist_ok=True, private=True)
    api.upload_folder(
        folder_path=path,
        repo_id=upload_repo,
        repo_type="model",
        commit_message="Upload converted model",
    )
    print(f"Upload successful, go to https://huggingface.co/{upload_repo} for details.")


def save_weights(save_path: Union[str, Path], weights: Dict[str, Any]) -> None:
    if isinstance(save_path, str):
        save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)

    total_size = sum(v.nbytes for v in weights.values())
    index_data = {"metadata": {"total_size": total_size}, "weight_map": {}}
    mx.save_safetensors(str(save_path / "model.safetensors"), weights, metadata={"format": "mlx"})

    for weight_name in weights.keys():
        index_data["weight_map"][weight_name] = "model.safetensors"

    index_data["weight_map"] = {k: index_data["weight_map"][k] for k in sorted(index_data["weight_map"])}

    with open(save_path / "model.safetensors.index.json", "w") as f:
        json.dump(index_data, f, indent=4)


def save_config(
    config: dict,
    config_path: Union[str, Path],
) -> None:
    """Save the model configuration to the ``config_path``.

    The final configuration will be sorted before saving for better readability.

    Args:
        config (dict): The model configuration.
        config_path (Union[str, Path]): Model configuration file path.
    """
    # sort the config for better readability
    config = dict(sorted(config.items()))

    # write the updated config to the config_path (if provided)
    with open(config_path, "w") as fid:
        json.dump(config, fid, indent=4)


def load_original_model(hf_repo: str) -> Tuple[SimpleNamespace, Dict[str, mx.array]]:
    """Load the original weights of BigVGAN from Hugging Face hub."""
    try:
        import torch
    except ImportError:
        raise ImportError(
            "PyTorch is required to load the original BigVGAN model. `pip install torch` before converting the model."
        )

    config_file = Path(
        hf_hub_download(
            repo_id=hf_repo,
            filename="config.json",
        )
    )

    model_file = Path(
        hf_hub_download(
            repo_id=hf_repo,
            filename="bigvgan_generator.pt",
        )
    )
    with open(config_file, "r", encoding="utf-8") as fid:
        config = SimpleNamespace(**json.load(fid))

    weights = torch.load(model_file, map_location="cpu", weights_only=True)["generator"]
    weights = {k: mx.array(v) for k, v in weights.items()}
    return (config, weights)


def convert(
    hf_repo: str = "nvidia/bigvgan_v2_24khz_100band_256x",
    dtype: mx.Dtype = mx.float32,
    output_dir: Union[str, Path] = "mlx_models",
    upload_repo: str = None,
):
    save_path = Path(output_dir) / hf_repo.split("/")[-1]

    config, origin_state_dict = load_original_model(hf_repo)
    model = BigVGAN(config)
    new_weights: Dict[str, mx.array] = {}
    resblocks_weights: List[Tuple[str, mx.array]] = []
    for k, v in origin_state_dict.items():
        if k.endswith(("weight_v", "weight_g")):
            basename, pname = k.rsplit(".", 1)
            # handle weight norm
            if pname == "weight_v":
                g = origin_state_dict[basename + ".weight_g"]
                v = g * (v / mx.linalg.norm(v, axis=(1, 2), keepdims=True))
                k = basename + ".weight"
            elif pname in ["weight_g"]:
                continue
        # rename ups
        if k.startswith("ups."):
            if k.endswith("weight"):
                k = k.replace("0.weight", "weight")
            elif k.endswith("bias"):
                k = k.replace("0.bias", "bias")

        # re-shape weights
        if k in ["conv_pre.weight", "conv_post.weight"] or (k.startswith("resblocks.") and k.endswith(".weight")):
            # Conv1D weight
            #    (out_channels, in_channels, kernel_size) -> (out_channels, kernel_size, in_channels)
            v = mx.moveaxis(v, 1, 2)
        elif k.startswith("ups.") and k.endswith("weight"):
            # ConvTranspose1D weight
            #    (in_channels, out_channels, kernel_size) -> (out_channels, kernel_size, in_channels)
            v = mx.moveaxis(v, 0, 2)
        elif k.endswith((".upsample.filter", ".downsample.lowpass.filter")):
            # (1, 1, n) -> (n,)
            v = mx.reshape(v, (v.shape[-1],))
        # rename resblocks
        if k.startswith("resblocks."):
            resblocks_weights.append((k, v))
            continue
        new_weights[k] = v

    resblocks_tree = tree_unflatten(resblocks_weights)
    resblocks = resblocks_tree["resblocks"]
    resblock_num_layers = len(config.resblock_dilation_sizes)
    new_resblocks = [
        {
            "layers": [
                {
                    "layers":
                    # 1: 2 snakes +  2 convs
                    # 2: 1 snake + 1 conv
                    [None] * (4 if config.resblock == "1" else 2),
                }
                for _ in range(resblock_num_layers)
            ]
        }
        for _ in range(len(resblocks))
    ]
    for i, resblock in enumerate(resblocks):
        for j in range(resblock_num_layers):
            if config.resblock == "1":
                # act1
                new_resblocks[i]["layers"][j]["layers"][0] = resblock["activations"][j * 2]
                # convs1
                new_resblocks[i]["layers"][j]["layers"][1] = resblock["convs1"][j]
                # act2
                new_resblocks[i]["layers"][j]["layers"][2] = resblock["activations"][j * 2 + 1]
                # convs2
                new_resblocks[i]["layers"][j]["layers"][3] = resblock["convs2"][j]
            else:
                # act1
                new_resblocks[i]["layers"][j]["layers"][0] = resblock["activations"][j]
                # convs1
                new_resblocks[i]["layers"][j]["layers"][1] = resblock["convs"][j]
    new_weights.update(tree_flatten(new_resblocks, prefix=".resblocks"))
    model.load_weights(list(new_weights.items()))
    model.eval()
    if dtype is not None and dtype != mx.float32:
        model.set_dtype(dtype)
    weights = dict(tree_flatten(model.parameters()))

    if isinstance(save_path, str):
        save_path = Path(save_path)

    save_weights(save_path, weights)

    save_config(vars(config), config_path=save_path / "config.json")
    print(f"Model has been saved to {save_path}.")

    # print("huggingface-cli repo create <your_repo_name>")
    # print(f"huggingface-cli upload --repo-type model <your_repo_name> {save_path}")
    if upload_repo:
        upload_to_hub(save_path, upload_repo, hf_repo)
    else:
        print(
            "To upload the model after conversion:\n"
            "python -m mlx_bigvgan.convert --upload_repo <username>/<repo_name> ..."
        )


def main():
    r"""Script to convert BigVGAN torch weights to MLX format.
    ``python -m mlx_bigvgan.convert --repo_id nvidia/bigvgan_v2_24khz_100band_256x --output_dir mlx_models``
    """
    import argparse

    parser = argparse.ArgumentParser(description="Convert BigVGAN weights to MLX.")
    parser.add_argument(
        "--repo_id",
        type=str,
        default="nvidia/bigvgan_v2_24khz_100band_256x",
        help="Hugging Face repo ID of the oringla bigvgan pytorch model.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="mlx_models",
        help="Output directory to save the converted model.",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        help="Data type to convert the model to.",
        default="float32",
        choices=["float32", "bfloat16", "float16"],
    )
    parser.add_argument(
        "--upload_repo",
        type=str,
        default=None,
        help="Hugging Face repo ID to upload the converted model. Should be in the format <username>/<repo_name>.",
    )
    # TODO: support quantization
    # parser.add_argument(
    #     "--quantize",
    #     action="store_true",
    #     help="Whether to quantize the model.",
    # )
    args = parser.parse_args()
    convert(
        hf_repo=args.repo_id, dtype=getattr(mx, args.dtype), output_dir=args.output_dir, upload_repo=args.upload_repo
    )


if __name__ == "__main__":
    main()
