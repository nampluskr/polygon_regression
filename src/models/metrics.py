# src/models/metrics.py

import torch
from shapely.geometry import Polygon


class QuadIOU:
    """Shapely-based accurate polygon IoU — evaluation only (CPU, no backprop)."""

    def __call__(self, quad1, quad2):
        p1 = quad1.detach().cpu().numpy().reshape(-1, 4, 2)
        p2 = quad2.detach().cpu().numpy().reshape(-1, 4, 2)
        ious = []
        for pts1, pts2 in zip(p1, p2):
            try:
                pg1 = Polygon(pts1).buffer(0)
                pg2 = Polygon(pts2).buffer(0)
                inter = pg1.intersection(pg2).area
                union = pg1.union(pg2).area
                ious.append(inter / union if union > 1e-8 else 0.0)
            except Exception:
                ious.append(0.0)
        return torch.tensor(ious, dtype=torch.float32).mean()


class PointAccuracy:
    """Fraction of predicted corners within threshold distance of ground truth."""

    THRESHOLDS = {"p1": 0.01, "p2": 0.02, "p3": 0.03, "p5": 0.05, "p10": 0.1}

    def __init__(self, threshold="p3"):
        if threshold not in self.THRESHOLDS:
            raise ValueError(
                f"threshold must be one of {list(self.THRESHOLDS.keys())}"
            )
        self.threshold = threshold

    def __call__(self, quad1, quad2):
        dist = torch.norm(quad1.view(-1, 4, 2) - quad2.view(-1, 4, 2), dim=2)
        return (dist < self.THRESHOLDS[self.threshold]).float().mean()


class NME:
    """Normalized Mean Error based on normalized coordinates [0, 1]."""

    def __call__(self, quad1, quad2):
        dist = torch.norm(quad1.view(-1, 4, 2) - quad2.view(-1, 4, 2), dim=2)
        return dist.mean()


class MDE:
    """Mean Distance Error in pixel scale."""

    def __init__(self, img_h=512, img_w=512):
        self.img_h = img_h
        self.img_w = img_w

    def __call__(self, quad1, quad2):
        scale = torch.tensor(
            [self.img_w, self.img_h] * 4,
            dtype=quad1.dtype,
            device=quad1.device
        )
        dist = torch.norm(
            (quad1 * scale).view(-1, 4, 2) - (quad2 * scale).view(-1, 4, 2), dim=2
        )
        return dist.mean()