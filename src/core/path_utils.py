# src/core/path_utils.py

import os


def get_pretrain_paths(paths, backbone_name):
    base = os.path.join(paths["pretrain_dir"], f"pretrain-{backbone_name}")
    return {
        "pth":  f"{base}.pth",
        "log":  f"{base}.log",
        "yaml": f"{base}.yaml",
    }

def get_finetune_paths(paths, backbone_name, run_index=None):
    base = os.path.join(paths["finetune_dir"], f"finetune-{backbone_name}")
    result = {
        "pth":  f"{base}.pth",
        "log":  f"{base}.log",
        "yaml": f"{base}.yaml",
    }
    if run_index is not None:
        result["archive"] = os.path.join(
            paths["finetune_archive"],
            f"finetune-{backbone_name}-run{run_index}.pth"
        )
    return result