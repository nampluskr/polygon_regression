# src/models/simple_cnn.py

import logging

import torch
import torch.nn as nn

logger = logging.getLogger("train")


class SimpleCNN(nn.Module):
    def __init__(self, output_dim=8, in_channels=3):
        super().__init__()

        self.backbone = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(2),
        )
        self.pool = nn.AdaptiveAvgPool2d((7, 7))
        self.in_features = 256 * 7 * 7
        self.head = nn.Sequential(
            nn.Linear(self.in_features, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim),
        )
        logger.info(f"> SimpleCNN: output_dim={output_dim}")

    def forward(self, x):
        x = self.backbone(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.head(x)
        return x