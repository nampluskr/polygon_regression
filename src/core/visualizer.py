# src/core/visualizer.py

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib.patches import Polygon
from PIL import Image

import torch

from src.core.utils import to_numpy


def show_images_with_quads(images, quads, preds=None, titles=None, ncols=4):
    """Visualize images with quad corner overlays.

    images : list of np.ndarray or PIL.Image
    quads  : list of [x1,y1,x2,y2,x3,y3,x4,y4] ground truth (green)
    preds  : list of [x1,y1,x2,y2,x3,y3,x4,y4] prediction   (red dashed)
    titles : list of str (optional)
    """
    nrows = (len(images) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3, nrows * 3))
    axes = axes.flatten()

    for i, (image, quad) in enumerate(zip(images, quads)):
        axes[i].imshow(image)
        axes[i].add_patch(Polygon(
            [[quad[0], quad[1]], [quad[2], quad[3]], [quad[4], quad[5]], [quad[6], quad[7]]],
            closed=True, linewidth=2, edgecolor="green", facecolor="none"))

        if preds is not None:
            pred = preds[i]
            axes[i].add_patch(Polygon(
                [[pred[0], pred[1]], [pred[2], pred[3]], [pred[4], pred[5]], [pred[6], pred[7]]],
                closed=True, linewidth=2, edgecolor="red", linestyle="dashed", facecolor="none"))

        if titles and i < len(titles):
            axes[i].set_title(titles[i])

    for ax in axes:
        ax.axis("off")

    if preds is not None:
        gt_line = mlines.Line2D([], [], color="green", linewidth=2, label="ground truth")
        pred_line = mlines.Line2D([], [], color="red", linewidth=2, linestyle="dashed", label="prediction")
        fig.legend(handles=[gt_line, pred_line],
                   loc="lower center", ncol=2, fontsize=10,
                   bbox_to_anchor=(0.5, -0.02))

    fig.tight_layout()
    plt.show()


def show_samples(df, image_dir, n=8, ncols=4, seed=42):
    samples = df.sample(n=n, random_state=seed).reset_index(drop=True)
    images  = [Image.open(os.path.join(image_dir, row["image_name"])).convert("RGB")
        for _, row in samples.iterrows()]
    quads  = samples[["x1", "y1", "x2", "y2", "x3", "y3", "x4", "y4"]].values.tolist()
    titles = samples["image_name"].tolist()

    show_images_with_quads(images, quads, titles=titles, ncols=ncols)


def show_dataset_samples(dataset, n=8, ncols=4, seed=42):
    torch.manual_seed(seed)
    indices = torch.randperm(len(dataset))[:n].tolist()
    images, quads, titles = [], [], []

    for idx in indices:
        sample = dataset[idx]
        image = to_numpy(sample["image"])
        coords = sample["coords_norm"].numpy()
        h, w = image.shape[:2]
        images.append(image)
        quads.append((coords * [w, h, w, h, w, h, w, h]).tolist())
        titles.append(f"idx={idx}")

    show_images_with_quads(images, quads, titles=titles, ncols=ncols)


def show_batch(batch, n=8, ncols=4):
    n = min(len(batch["image"]), n)
    images, quads  = [], []

    for i in range(n):
        image = to_numpy(batch["image"][i])
        coords = batch["coords_norm"][i].numpy()
        h, w = image.shape[:2]
        images.append(image)
        quads.append((coords * [w, h, w, h, w, h, w, h]).tolist())

    show_images_with_quads(images, quads, ncols=ncols)