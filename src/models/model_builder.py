# src/models/model_builder.py

import os
import logging

import torch
import torch.nn as nn
from torchvision import models

logger = logging.getLogger("train")

SUPPORTED_BACKBONES = [
    "resnet18", "resnet34", "resnet50",
    "wide_resnet50_2", "wide_resnet101_2",
    "efficientnet_b0", "efficientnet_b5",
    "vgg16", "vgg19", "vgg16_bn", "vgg19_bn",
    "mobilenet_v2", "mobilenet_v3_large", "mobilenet_v3_small",
]


def build_model(backbone_name, output_dim=8, backbone_weight_path=None):
    if backbone_name not in SUPPORTED_BACKBONES:
        raise ValueError(
            f"Unsupported backbone: '{backbone_name}'.\n"
            f"Supported: {SUPPORTED_BACKBONES}"
        )

    model_fn = getattr(models, backbone_name)
    model = model_fn(weights=None)

    if backbone_weight_path is not None:
        load_backbone_weights(model, backbone_weight_path)

    # Replace head and extract backbone / head references
    if "resnet" in backbone_name or "wide_resnet" in backbone_name:
        model.fc = nn.Linear(model.fc.in_features, output_dim)
        model.backbone = nn.Sequential(
            model.conv1, model.bn1, model.relu, model.maxpool,
            model.layer1, model.layer2, model.layer3, model.layer4,
            model.avgpool,
        )
        model.head = model.fc

    elif "efficientnet" in backbone_name:
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, output_dim)
        model.backbone = model.features
        model.head = model.classifier

    elif "vgg" in backbone_name:
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, output_dim)
        model.backbone = model.features
        model.head = model.classifier

    elif "mobilenet_v2" in backbone_name:
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, output_dim)
        model.backbone = model.features
        model.head = model.classifier

    elif "mobilenet_v3" in backbone_name:
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, output_dim)
        model.backbone = model.features
        model.head = model.classifier

    logger.info(f"> build_model: {backbone_name}, output_dim={output_dim}")
    return model


def load_backbone_weights(model, weight_path):
    state_dict = torch.load(weight_path, map_location="cpu", weights_only=True)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)

    logger.info(f"> load_backbone_weights: {os.path.basename(weight_path)}")
    logger.debug(f"  missing keys    : {missing}")
    logger.debug(f"  unexpected keys : {unexpected}")
    return model


def load_model_weights(model, weight_path):
    # Full restore — backbone + head must match exactly
    state_dict = torch.load(weight_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict, strict=True)

    logger.info(f"> load_model_weights: {os.path.basename(weight_path)}")
    return model


def save_model_weights(model, weight_path):
    os.makedirs(os.path.dirname(weight_path), exist_ok=True)
    torch.save(model.state_dict(), weight_path)

    logger.info(f"> save_model_weights: {os.path.basename(weight_path)}")