# src/core/path_utils.py

import os
import logging

from src.core.config import load_config

logger = logging.getLogger("train")


def get_pretrain_paths(paths, config):
    """Build pretrain output file paths from paths and config."""
    backbone_name = config["backbone_name"]
    image_size = config["image_size"]

    pretrain_dir = os.path.join(paths["pretrain_dir"], backbone_name)
    os.makedirs(pretrain_dir, exist_ok=True)
    base = os.path.join(pretrain_dir, f"pretrain-{backbone_name}-img{image_size}")
    return {
        "pth": f"{base}.pth",
        "log": f"{base}.log",
        "yaml": f"{base}.yaml",
    }


def get_finetune_paths(paths, config, run_index=None):
    """Build finetune output file paths from paths and config."""
    backbone_name = config["backbone_name"]
    image_size = config["image_size"]

    finetune_dir = os.path.join(paths["finetune_dir"], backbone_name)
    os.makedirs(finetune_dir, exist_ok=True)
    base = os.path.join(finetune_dir, f"finetune-{backbone_name}-img{image_size}")
    result = {
        "pth": f"{base}.pth",
        "log": f"{base}.log",
        "yaml": f"{base}.yaml",
    }
    if run_index is not None:
        result["archive"] = os.path.join(
            paths["finetune_archive"],
            f"finetune-{backbone_name}-img{image_size}-run{run_index}.pth"
        )
    return result


def load_model_with_config(weight_path):
    """Load model and retrieve config from paired yaml."""
    from src.models.model_builder import build_model, load_model_weights

    yaml_path = weight_path.replace(".pth", ".yaml")
    config = load_config(yaml_path)

    model = build_model(config["backbone_name"], config["output_dim"])
    load_model_weights(model, weight_path)

    logger.info(f"> load_model_with_config: {weight_path}")
    return model, config