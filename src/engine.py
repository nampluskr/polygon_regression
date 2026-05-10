# src/engine.py

import sys
import logging

from time import time
from copy import deepcopy
from tqdm import tqdm

logger = logging.getLogger("train")


def format_metrics(metrics):
    return ", ".join([f"{name}:{value:.3f}" for name, value in metrics.items()])


def train(trainer, dataloader):
    trainer.train()
    metrics    = {}
    total_size = 0

    with tqdm(dataloader, desc="Train", file=sys.stdout, leave=False, ascii=True) as pbar:
        for batch in pbar:
            results     = trainer.train_step(batch)
            total_size += results["batch_size"]

            for name, value in results.items():
                if name != "batch_size":
                    metrics.setdefault(name, 0.0)
                    metrics[name] += float(value) * results["batch_size"]

            pbar.set_postfix({name: f"{value / total_size:.3f}"
                              for name, value in metrics.items()})

    return {name: value / total_size for name, value in metrics.items()}


def evaluate(trainer, dataloader):
    trainer.eval()
    metrics    = {}
    total_size = 0

    with tqdm(dataloader, desc="Evaluate", file=sys.stdout, leave=False, ascii=True) as pbar:
        for batch in pbar:
            results     = trainer.eval_step(batch)
            total_size += results["batch_size"]

            for name, value in results.items():
                if name != "batch_size":
                    metrics.setdefault(name, 0.0)
                    metrics[name] += float(value) * results["batch_size"]

            pbar.set_postfix({name: f"{value / total_size:.3f}"
                              for name, value in metrics.items()})

    return {name: value / total_size for name, value in metrics.items()}


def _step_scheduler(trainer):
    if not hasattr(trainer, "scheduler") or trainer.scheduler is None:
        return

    old_lr = trainer.optimizer.param_groups[0]["lr"]
    trainer.scheduler.step()
    new_lr = trainer.optimizer.param_groups[0]["lr"]

    if new_lr != old_lr:
        logger.info(f"> learning rate: {old_lr:.2e} -> {new_lr:.2e}")


def fit(trainer, train_loader, max_epoch, valid_loader=None, **kwargs):
    start   = time()
    history = {"train": {}, "valid": {}}

    for epoch in range(1, max_epoch + 1):
        train_metrics = train(trainer, train_loader)
        epoch_info    = f"[{epoch:3d}/{max_epoch}]"

        for name, value in train_metrics.items():
            history["train"].setdefault(name, [])
            history["train"][name].append(value)

        if valid_loader is not None:
            valid_metrics = evaluate(trainer, valid_loader)

            for name, value in valid_metrics.items():
                history["valid"].setdefault(name, [])
                history["valid"][name].append(value)

            logger.info(
                f"{epoch_info} {format_metrics(train_metrics)} | "
                f"(val) {format_metrics(valid_metrics)}"
            )
        else:
            logger.info(f"{epoch_info} {format_metrics(train_metrics)}")

        _step_scheduler(trainer)

    elapsed = int(time() - start)
    h, m, s = elapsed // 3600, elapsed % 3600 // 60, elapsed % 60
    logger.info(f"> Training time: {h:02d}:{m:02d}:{s:02d}")
    return history, max_epoch


def fit_early_stop(trainer, train_loader, valid_loader, max_epoch=10,
                   patience=5, monitor="iou", mode="max", **kwargs):
    start   = time()
    history = {"train": {}, "valid": {}}

    best_value   = float("-inf") if mode == "max" else float("inf")
    best_epoch   = 0
    best_weights = deepcopy(trainer.model.state_dict())
    wait         = 0

    def is_improved(current, best):
        return current > best if mode == "max" else current < best

    logger.info(f"> learning rate: {trainer.optimizer.param_groups[0]['lr']:.2e}")

    for epoch in range(1, max_epoch + 1):
        train_metrics = train(trainer, train_loader)
        valid_metrics = evaluate(trainer, valid_loader)
        epoch_info    = f"[{epoch:3d}/{max_epoch}]"

        for name, value in train_metrics.items():
            history["train"].setdefault(name, [])
            history["train"][name].append(value)

        for name, value in valid_metrics.items():
            history["valid"].setdefault(name, [])
            history["valid"][name].append(value)

        logger.info(
            f"{epoch_info} {format_metrics(train_metrics)} | "
            f"(val) {format_metrics(valid_metrics)}"
        )

        _step_scheduler(trainer)

        monitor_value = valid_metrics[monitor]
        if is_improved(monitor_value, best_value):
            best_value   = monitor_value
            best_epoch   = epoch
            best_weights = deepcopy(trainer.model.state_dict())
            wait         = 0
        else:
            wait += 1
            if wait >= patience:
                logger.info(
                    f"> Early stopped at epoch {epoch} "
                    f"(best epoch: {best_epoch}, best {monitor}: {best_value:.3f})"
                )
                break

    trainer.model.load_state_dict(best_weights)

    elapsed = int(time() - start)
    h, m, s = elapsed // 3600, elapsed % 3600 // 60, elapsed % 60
    logger.info(f"> Training time: {h:02d}:{m:02d}:{s:02d}")
    return history, best_epoch