# src/models/quad_regressor.py

import logging

import torch
import torch.nn as nn
import torch.optim as optim

from .metrics import QuadIOU, PointAccuracy

logger = logging.getLogger("train")


class QuadRegressor(nn.Module):

    def __init__(self, model, optimizer=None, scheduler=None, loss_fn=None,
                 iou_metric=None, acc_metric=None, device=None):
        super().__init__()

        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.backbone = model.backbone
        self.head = model.head

        self.loss_fn = loss_fn or nn.SmoothL1Loss(beta=0.1)
        self.iou_metric = iou_metric or QuadIOU()
        self.acc_metric = acc_metric or PointAccuracy(threshold="p3")

        self.optimizer = optimizer or optim.AdamW([
            {"params": self.backbone.parameters(), "lr": 2e-5},
            {"params": self.head.parameters(),     "lr": 2e-4},
        ], weight_decay=1e-5)
        self.scheduler = scheduler or optim.lr_scheduler.StepLR(
            self.optimizer, step_size=10, gamma=0.5)

    def forward(self, images):
        logits = self.model(images)
        return torch.sigmoid(logits)

    def train_step(self, batch):
        images = batch["image"].to(self.device)
        coords = batch["coords_norm"].to(self.device)

        self.train()
        preds = self.forward(images)
        loss = self.loss_fn(preds, coords)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        with torch.no_grad():
            iou = self.iou_metric(preds, coords)
            acc = self.acc_metric(preds, coords)

        return {
            "loss": loss.item(),
            "iou":  iou.item(),
            "acc":  acc.item(),
            "batch_size": images.size(0),
        }

    @torch.no_grad()
    def eval_step(self, batch):
        images = batch["image"].to(self.device)
        coords = batch["coords_norm"].to(self.device)

        self.eval()
        preds = self.forward(images)
        loss = self.loss_fn(preds, coords)
        iou = self.iou_metric(preds, coords)
        acc = self.acc_metric(preds, coords)

        return {
            "loss": loss.item(),
            "iou":  iou.item(),
            "acc":  acc.item(),
            "batch_size": images.size(0),
        }

    def freeze_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad = False
        logger.info("> freeze_backbone")

    def unfreeze_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad = True
        logger.info("> unfreeze_backbone")