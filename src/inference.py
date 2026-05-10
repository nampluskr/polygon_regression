# src/inference.py

import torch
import logging

from PIL import Image
from torch.utils.data import Dataset, DataLoader

from src.data.transform import get_valid_transform
from src.data.coords import denormalize_coords

logger = logging.getLogger("train")


class InferenceDataset(Dataset):
    """Image-only dataset for batch inference without ground truth."""

    def __init__(self, image_paths, image_size=224):
        self.image_paths = image_paths
        self.transform   = get_valid_transform(image_size)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image      = Image.open(image_path).convert("RGB")
        w, h       = image.size
        image      = self.transform(image)
        return {
            "image":      image,
            "image_path": image_path,
            "width":      torch.tensor(w, dtype=torch.float32),
            "height":     torch.tensor(h, dtype=torch.float32),
        }


@torch.no_grad()
def predict(model, image_path, image_size, device):
    """Predict quad corner coordinates for a single image."""
    model.eval()

    transform = get_valid_transform(image_size)
    image     = Image.open(image_path).convert("RGB")
    w, h      = image.size

    tensor = transform(image).unsqueeze(0).to(device)
    preds  = torch.sigmoid(model(tensor)).squeeze(0).cpu()
    coords = denormalize_coords(preds, w, h)

    logger.info(f"> predict: {image_path}")
    return coords.tolist()


@torch.no_grad()
def predict_batch(model, image_paths, image_size, device,
                  batch_size=32, num_workers=4):
    """Predict quad corner coordinates for multiple images."""
    model.eval()

    dataset    = InferenceDataset(image_paths, image_size=image_size)
    dataloader = DataLoader(
        dataset,
        batch_size  = batch_size,
        shuffle     = False,
        num_workers = num_workers,
        pin_memory  = True,
    )

    all_results = []

    for batch in dataloader:
        images  = batch["image"].to(device)
        widths  = batch["width"]
        heights = batch["height"]
        paths   = batch["image_path"]

        preds = torch.sigmoid(model(images)).cpu()

        for pred, w, h, path in zip(preds, widths, heights, paths):
            coords = denormalize_coords(pred, w.item(), h.item())
            all_results.append({
                "image_path": path,
                "coords":     coords.tolist(),
            })

    logger.info(f"> predict_batch: {len(all_results)} images processed")
    return all_results