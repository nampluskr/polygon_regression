# src/models/losses.py

import math
import torch


class QuadIOU:
    """PyTorch-based GPU-compatible IoU using axis-aligned bounding box approximation."""

    def __call__(self, quad1, quad2):
        p1 = quad1.view(-1, 4, 2)
        p2 = quad2.view(-1, 4, 2)

        p1_min = p1.min(dim=1).values
        p1_max = p1.max(dim=1).values
        p2_min = p2.min(dim=1).values
        p2_max = p2.max(dim=1).values

        inter_min  = torch.max(p1_min, p2_min)
        inter_max  = torch.min(p1_max, p2_max)
        inter_wh   = (inter_max - inter_min).clamp(min=0)
        inter_area = inter_wh[:, 0] * inter_wh[:, 1]

        area1      = (p1_max - p1_min).prod(dim=1)
        area2      = (p2_max - p2_min).prod(dim=1)
        union_area = area1 + area2 - inter_area

        iou = inter_area / (union_area + 1e-8)
        return iou.mean()


class QuadIOULoss:
    """IoU-based loss: lambda_iou * (1 - IoU)."""

    def __init__(self, lambda_iou=0.5):
        self.lambda_iou = lambda_iou
        self.iou_fn = QuadIOU()

    def __call__(self, pred, target):
        iou = self.iou_fn(pred, target)
        return self.lambda_iou * (1.0 - iou)


class WingLoss:
    """Wing loss for landmark regression.

    Small errors: log-based penalty (sensitive).
    Large errors: linear penalty (robust).
    """

    def __init__(self, w=10.0, epsilon=2.0):
        self.w       = w
        self.epsilon = epsilon
        self.C       = w - w * math.log(1.0 + w / epsilon)

    def __call__(self, pred, target):
        diff = (pred - target).abs()
        loss = torch.where(
            diff < self.w,
            self.w * torch.log(1.0 + diff / self.epsilon),
            diff - self.C
        )
        return loss.mean()


class CombinedLoss:
    """WingLoss + QuadIOULoss combined loss for quad coordinate regression."""

    def __init__(self, w=10.0, epsilon=2.0, lambda_iou=0.5):
        self.wing_loss = WingLoss(w=w, epsilon=epsilon)
        self.iou_loss  = QuadIOULoss(lambda_iou=lambda_iou)

    def __call__(self, pred, target):
        return self.wing_loss(pred, target) + self.iou_loss(pred, target)