import argparse
import yaml
import torch

from .utils import Config, set_seed
from .models.convnet_scratch import build_model
from .clearml_utils import init_task
from .trainer import Trainer
from data.dataset import load_dataset_and_dataloaders

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/train.yaml")
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
    model = build_model(
        num_classes=num_classes,
        channels=cfg.model["channels"],
        regularisation=cfg.model["regularisation"],
        dropout=cfg.model["dropout"],
    )

    # ------------------------------------------------------------------
    # Trainer
    # ------------------------------------------------------------------
    trainer = Trainer(model, cfg, task)
    trainer.train(train_loader, val_loader, test_loader)

if __name__ == "__main__":
    main()
