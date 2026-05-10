# scripts/run_pretrain.py

import os
import sys

ROOT_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import torch
import torch.optim as optim

from src.core.config import load_config, save_config
from src.core.logger import get_logger
from src.core.path_utils import get_pretrain_paths
from src.core.utils import set_seed
from src.data.transform import get_train_transform, get_valid_transform
from src.data.dataloader import get_dataloader, get_combined_dataloader
from src.models.model_builder import build_model, load_backbone_weights, save_model_weights
from src.models.regressor import QuadRegressor
from src.models.losses import CombinedLoss
from src.engine import fit_early_stop


def main():
    # Load configs
    paths = load_config("configs/paths.yaml")
    config = load_config("configs/pretrain.yaml")

    # Setup paths
    pretrain_paths = get_pretrain_paths(paths, config)
    
    # Setup logger
    logger = get_logger(name="train", log_path=pretrain_paths["log"])

    # Reproducibility
    set_seed(config["seed"])

    logger.info("=== Pretrain Start ===")
    logger.info(f"> backbone   : {config['backbone_name']}")
    logger.info(f"> image_size : {config['image_size']}")
    logger.info(f"> batch_size : {config['batch_size']}")
    logger.info(f"> max_epoch  : {config['max_epoch']}")

    # Build model
    backbone_name = config["backbone_name"]
    model = build_model(backbone_name, 
        output_dim=config["output_dim"],
        backbone_weight_path=paths["backbones"][backbone_name],
    )

    # Loss
    loss_fn = CombinedLoss(
        w=config["loss"]["wing_w"],
        epsilon=config["loss"]["wing_epsilon"],
        lambda_iou=config["loss"]["lambda_iou"],
    )
    # loss_fn = torch.nn.SmoothL1Loss(beta=0.1)

    # Optimizer
    optimizer = optim.AdamW([
        {"params": model.backbone.parameters(), "lr": config["optimizer"]["backbone_lr"]},
        {"params": model.head.parameters(), "lr": config["optimizer"]["head_lr"]},
    ], weight_decay=config["optimizer"]["weight_decay"])

    # Scheduler
    scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size=config["scheduler"]["step_size"],
        gamma=config["scheduler"]["gamma"],
    )

    # Regressor
    regressor = QuadRegressor(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        loss_fn=loss_fn,
    )

    if config["freeze_backbone"]:
        regressor.freeze_backbone()

    # DataLoader
    image_size = config["image_size"]
    batch_size = config["batch_size"]
    num_workers = config["num_workers"]

    dataset_configs = [{
        "image_dir": paths["images"][dataset["name"]],
        "train_csv": paths["train_annotations"][dataset["name"]],
        "valid_csv": paths["valid_annotations"][dataset["name"]],
        "weight": dataset.get("weight", 1.0),
        "sampling": dataset.get("sampling", 1.0),
        } for dataset in config["datasets"]
    ]

    train_loader = get_combined_dataloader(
        dataset_configs=dataset_configs,
        split="train",
        image_size=config["image_size"],
        batch_size=config["batch_size"],
    )
    valid_loader = get_combined_dataloader(
        dataset_configs=dataset_configs,
        split="valid",
        image_size=config["image_size"],
        batch_size=config["batch_size"],
    )

    logger.info(f"> train batches : {len(train_loader):,}")
    logger.info(f"> valid batches : {len(valid_loader):,}")

    # Training
    history, best_epoch = fit_early_stop(
        trainer=regressor,
        train_loader=train_loader,
        valid_loader=valid_loader,
        max_epoch=config["max_epoch"],
        patience=config["patience"],
        monitor=config["monitor"],
        mode=config["mode"],
    )

    # Save
    save_model_weights(regressor.model, pretrain_paths["pth"])
    save_config(config, pretrain_paths["yaml"])

    logger.info(f"> best epoch : {best_epoch}")
    logger.info(f"> pth  saved : {os.path.basename(pretrain_paths['pth'])}")
    logger.info(f"> yaml saved : {os.path.basename(pretrain_paths['yaml'])}")
    logger.info("=== Pretrain Done ===")


if __name__ == "__main__":
    main()