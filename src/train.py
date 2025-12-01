import argparse
import os
import json
import random
from dataclasses import dataclass
from typing import List

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import yaml

from .models.convnet_scratch import build_model
from .clearml_utils import init_task, log_scalar, upload_model
from data.dataset import load_dataset_and_dataloaders


# ----------------------------------------------------------------------
# Config dataclass (supports both `data` and `dataset` in YAML)
# ----------------------------------------------------------------------
@dataclass
class Config:
    seed: int
    project_name: str
    task_name: str
    tags: List[str]
    device: str
    data: dict
    model: dict
    train: dict
    clearml: dict
    dataset: dict | None = None


# ----------------------------------------------------------------------
# Early stopping
# ----------------------------------------------------------------------
class EarlyStopping:
    def __init__(self, patience: int = 3, min_delta: float = 0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.best_score = None
        self.best_epoch = None
        self.counter = 0

    def step(self, metric: float, epoch: int) -> bool:
        """Returns True if we should stop training."""
        if self.best_score is None or metric > self.best_score + self.min_delta:
            self.best_score = metric
            self.best_epoch = epoch
            self.counter = 0
            return False
        else:
            self.counter += 1
            return self.counter >= self.patience


# ----------------------------------------------------------------------
# Utils
# ----------------------------------------------------------------------
def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def accuracy(logits, targets):
    preds = torch.argmax(logits, dim=1)
    return (preds == targets).float().mean().item()


def _unpack_batch(batch, device):
    """
    Unpack a batch coming from the DataLoader.
    New behaviour (dataset manager update):
      - DataLoader now yields tuples, typically:
          (images, labels) or (images, labels, extra_info)
    """
    if isinstance(batch, (list, tuple)):
        if len(batch) < 2:
            raise ValueError(f"Expected at least 2 elements in batch, got {len(batch)}")
        xb, yb = batch[0], batch[1]
    else:
        raise TypeError(f"Expected batch to be tuple/list, got {type(batch)}")

    return xb.to(device), yb.to(device)


# ----------------------------------------------------------------------
# Train / eval loops
# ----------------------------------------------------------------------
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss, running_acc, n = 0.0, 0.0, 0
    for batch in loader:
        xb, yb = _unpack_batch(batch, device)
        optimizer.zero_grad()
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * xb.size(0)
        running_acc += (logits.argmax(1) == yb).float().sum().item()
        n += xb.size(0)
    return running_loss / n, running_acc / n


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss, running_acc, n = 0.0, 0.0, 0
    for batch in loader:
        xb, yb = _unpack_batch(batch, device)
        logits = model(xb)
        loss = criterion(logits, yb)
        running_loss += loss.item() * xb.size(0)
        running_acc += (logits.argmax(1) == yb).float().sum().item()
        n += xb.size(0)
    return running_loss / n, running_acc / n


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/train.yaml")
    args = parser.parse_args()

    # Fix Windows backslashes so Linux worker can read the path
    config_path = args.config.replace("\\", "/")
    print(f"[Config] Using config file: {config_path}")

    with open(config_path, "r") as f:
        cfg_dict = yaml.safe_load(f)

    # Build Config (now aware of top-level `dataset:`)
    cfg = Config(**cfg_dict)

    # We'll mostly read dataset options from `data:` (as your YAML does)
    data_cfg = cfg.data

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

    # If configured, send this task to a remote ClearML queue and stop local execution
    remote_queue = clearml_cfg.get("queue")
    if task is not None and remote_queue:
        try:
            from clearml import Task as ClearMLTask

            if ClearMLTask.running_locally():
                print(f"[ClearML] Executing remotely on queue '{remote_queue}'")
                task.execute_remotely(queue_name=remote_queue, exit_process=True)
        except Exception as e:
            print(f"[ClearML] execute_remotely failed ({e}), continuing locally.")

    # Seed + device
    set_seed(cfg.seed)
    device = torch.device(
        cfg.device if torch.cuda.is_available() and cfg.device == "cuda" else "cpu"
    )
    os.makedirs("outputs", exist_ok=True)

    # ------------------------------------------------------------------
    # DATA LOADING (via dataset.py helper)
    # ------------------------------------------------------------------
    subset_key = data_cfg.get("clearml_subset", "tiny")

    train_loader, val_loader, test_loader, class_names = load_dataset_and_dataloaders(
        dataset_size=subset_key,
        config_path=config_path,
    )
    num_classes = len(class_names)
    print(f"[Data] num_classes = {num_classes}")

    # ------------------------------------------------------------------
    # MODEL
    # ------------------------------------------------------------------
    model = build_model(
        num_classes=num_classes,
        channels=cfg.model["channels"],
        regularisation=cfg.model["regularisation"],  # "none" | "dropout" | "batchnorm"
        dropout=cfg.model["dropout"],               # used when regularisation == "dropout"
    ).to(device)

    # Loss & Optimizer
    criterion = nn.CrossEntropyLoss()
    if cfg.train["optimizer"].lower() == "adamw":
        optimizer = optim.AdamW(
            model.parameters(),
            lr=cfg.train["lr"],
            weight_decay=cfg.train["weight_decay"],
        )
    else:
        optimizer = optim.Adam(
            model.parameters(),
            lr=cfg.train["lr"],
            weight_decay=cfg.train["weight_decay"],
        )

    # LR Scheduler (Cosine Annealing)
    scheduler = None
    scheduler_cfg = cfg.train.get("scheduler")
    if scheduler_cfg is not None and scheduler_cfg.get("type", "").lower() == "cosine":
        T_max = scheduler_cfg.get("T_max", cfg.train["epochs"])
        scheduler = CosineAnnealingLR(optimizer, T_max=T_max)
        print(f"[Scheduler] Using CosineAnnealingLR with T_max={T_max}")

    # Early stopping
    early_stopper = EarlyStopping(
        patience=cfg.train.get("patience", 3),
        min_delta=cfg.train.get("min_delta", 0.0),
    )

    # ------------------------------------------------------------------
    # Training with early stopping on val accuracy
    # ------------------------------------------------------------------
    best_val_acc = 0.0
    best_epoch = 0
    best_path = os.path.join("outputs", "best.pt")
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    for epoch in range(1, cfg.train["epochs"] + 1):
        tr_loss, tr_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        va_loss, va_acc = evaluate(model, val_loader, criterion, device)

        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["val_loss"].append(va_loss)
        history["val_acc"].append(va_acc)

        print(
            f"Epoch {epoch:02d} | "
            f"Train: loss {tr_loss:.4f}, acc {tr_acc:.4f} | "
            f"Val: loss {va_loss:.4f}, acc {va_acc:.4f}"
        )
        log_scalar(task, "loss", "train", tr_loss, epoch)
        log_scalar(task, "loss", "val", va_loss, epoch)
        log_scalar(task, "accuracy", "train", tr_acc, epoch)
        log_scalar(task, "accuracy", "val", va_acc, epoch)

        # LR logging & scheduler step
        if scheduler is not None:
            scheduler.step()
            current_lr = scheduler.get_last_lr()[0]
            log_scalar(task, "learning_rate", "lr", current_lr, epoch)

        # Save best model when validation accuracy improves
        if va_acc > best_val_acc + 1e-8:
            best_val_acc = va_acc
            best_epoch = epoch
            torch.save(
                {"model_state": model.state_dict(), "val_acc": best_val_acc},
                best_path,
            )
            upload_model(task, best_path, name="best.pt")

        # Early stopping check
        if early_stopper.step(va_acc, epoch):
            print(
                f"[EarlyStopping] No val acc improvement for "
                f"{early_stopper.patience} epochs. "
                f"Best epoch: {early_stopper.best_epoch}"
            )
            break

    # Save metrics (including path to best weights)
    metrics = {
        "best_val_acc": best_val_acc,
        "best_epoch": best_epoch,
        "best_model_path": best_path,
        **history,
    }
    with open(os.path.join("outputs", "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    # Log these to ClearML task object, if available
    if task is not None:
        try:
            task.connect(metrics)
        except Exception:
            pass

    print(
        f"[Done] Best validation accuracy: {best_val_acc:.4f} "
        f"(epoch {best_epoch}). Weights saved to {best_path}"
    )


if __name__ == "__main__":
    main()
