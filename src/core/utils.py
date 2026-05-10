# src/core/utils.py

import os
import random
import logging

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from PIL import Image

import torch

logger = logging.getLogger("train")


def set_seed(seed=42):
    """Fix all random seeds for reproducibility."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark     = False

    logger.info(f"> set_seed: {seed}")


def to_numpy(tensor):
    """Convert ImageNet-normalized tensor to numpy image [0, 1]."""
    mean = torch.tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)
    img  = tensor * std + mean
    img  = torch.clamp(img, 0, 1)
    return img.permute(1, 2, 0).cpu().numpy()


def to_list(coords):
    """Convert tensor or ndarray to Python list."""
    if isinstance(coords, torch.Tensor):
        return coords.detach().cpu().tolist()
    elif isinstance(coords, np.ndarray):
        return coords.tolist()
    return list(coords)


def plot_images(*images, ncols=5, xunit=3, yunit=3, cmap="gray",
                titles=None, vmin=None, vmax=None, save_path=None):
    """Plot multiple images in a grid layout."""
    num_images = len(images)
    ncols      = min(ncols, num_images)
    nrows      = (num_images + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(ncols * xunit, nrows * yunit))
    axes = np.array(axes).reshape(-1) if num_images > 1 else [axes]

    for idx, img in enumerate(images):
        axes[idx].imshow(img, cmap=cmap, vmin=vmin, vmax=vmax)
        axes[idx].axis("off")
        if titles and idx < len(titles):
            axes[idx].set_title(titles[idx])

    for idx in range(num_images, len(axes)):
        axes[idx].axis("off")

    fig.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150)
        plt.close()
        logger.info(f"> plot saved: {os.path.basename(save_path)}")
    else:
        plt.show()


def show_image_poly(image, coords, pred=None, denormalize=False):
    """Visualize quad corner coordinates overlaid on image.

    coords: ground truth (green)
    pred:   prediction   (red dashed)
    """
    if isinstance(image, str):
        image = Image.open(image).convert("RGB")

    if denormalize:
        image = to_numpy(image)

    fig, ax = plt.subplots()
    ax.imshow(image)

    x1, y1, x2, y2, x3, y3, x4, y4 = to_list(coords)
    ax.add_patch(Polygon(
        [[x1, y1], [x2, y2], [x3, y3], [x4, y4]],
        closed=True, linewidth=2, edgecolor="green", facecolor="none"
    ))

    if pred is not None:
        x1, y1, x2, y2, x3, y3, x4, y4 = to_list(pred)
        ax.add_patch(Polygon(
            [[x1, y1], [x2, y2], [x3, y3], [x4, y4]],
            closed=True, linewidth=2,
            edgecolor="red", linestyle="dashed", facecolor="none"
        ))

    plt.axis("off")
    fig.tight_layout()
    plt.show()
