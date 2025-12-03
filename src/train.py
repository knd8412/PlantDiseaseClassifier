import argparse
import os
import sys

import torch
import yaml

from data.dataset import load_dataset_and_dataloaders
from src.clearml_utils import init_task
from src.models.convnet_scratch import build_model
from src.models.resnet import ResNet18Classifier
from src.trainer import Trainer
from src.utils import Config, set_seed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/train_medium.yaml")
    args = parser.parse_args()

    # Fix Windows backslashes so Linux worker can read the path
    config_path = args.config.replace("\\", "/")
    print(f"[Config] Using config file: {config_path}")

    with open(config_path, "r") as f:
        cfg_dict = yaml.safe_load(f)

    # Build Config
    cfg = Config(**cfg_dict)

    # ------------------------------------------------------------------
    # ClearML task
    # ------------------------------------------------------------------
    clearml_cfg = cfg.clearml or {}
    task = init_task(
        enabled=clearml_cfg.get("enabled", False),
        project=clearml_cfg.get("project") or cfg.project_name,
        task_name=clearml_cfg.get("task_name") or cfg.task_name,
        tags=cfg.tags,
        params=cfg_dict,
    )

    # Remote execution check
    remote_queue = clearml_cfg.get("queue")
    if task is not None and remote_queue:
        try:
            from clearml import Task as ClearMLTask

            if ClearMLTask.running_locally():
                print(f"[ClearML] Executing remotely on queue '{remote_queue}'")
                task.execute_remotely(queue_name=remote_queue, exit_process=True)
        except Exception as e:
            print(f"[ClearML] execute_remotely failed ({e}), continuing locally.")

    # Seed
    set_seed(cfg.seed)

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    subset_key = cfg.data.get("clearml_subset", "tiny")
    train_loader, val_loader, test_loader, class_names = load_dataset_and_dataloaders(
        dataset_size=subset_key,
        config_path=config_path,
    )
    num_classes = len(class_names)
    print(f"[Data] num_classes = {num_classes}")

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------

    arch = cfg.model.get("arch", "scratch")

    if arch == "scratch":
        model = build_model(
            num_classes=num_classes,
            channels=cfg.model["channels"],
            regularisation=cfg.model["regularisation"],
            dropout=cfg.model["dropout"],
        )

    elif arch == "resnet18":
        model = ResNet18Classifier(
            num_classes=num_classes,
            pretrained=cfg.model.get("pretrained", True),
            dropout=cfg.model.get("dropout", 0.0),
            train_backbone=cfg.model.get("train_backbone", True),
        )

    else:
        raise ValueError(f"Unknown model arch: {arch}")

    # ------------------------------------------------------------------
    # Trainer
    # ------------------------------------------------------------------
    trainer = Trainer(model, cfg, task)
    trainer.train(train_loader, val_loader, test_loader)


if __name__ == "__main__":
    main()
