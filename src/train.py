import argparse
import json
import os
import random
from dataclasses import dataclass
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from clearml import Dataset as ClearMLDataset
from datasets import load_dataset
from PIL import Image
from sklearn.model_selection import StratifiedShuffleSplit
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

try:
    from .clearml_utils import init_task, log_scalar, upload_model
    from .models.convnet_scratch import build_model as build_scratch_model
    from .models.resnet import ResNet18Classifier
except ImportError:
    # Fallback for direct script execution
    from src.clearml_utils import init_task, log_scalar, upload_model
    from src.models.convnet_scratch import build_model as build_scratch_model
    from src.models.resnet import ResNet18Classifier

from data.dataset import MultiModalityDataset
from data.transforms import get_transforms
from data.utils import (
    build_class_mapping,
    ensure_dataset_extracted,
    gather_samples,
    split_dataset,
)


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


class HFDataset(Dataset):
    def __init__(self, hf_split, transform):
        self.hf_split = hf_split
        self.transform = transform

        # Extract label names from features (if present)
        self.label_names = None
        if "labels" in hf_split.features and hasattr(
            hf_split.features["labels"], "names"
        ):
            self.label_names = hf_split.features["labels"].names

    def __len__(self):
        return len(self.hf_split)

    def __getitem__(self, idx):
        sample = self.hf_split[idx]
        # PlantVillage sometimes stores images under 'image' or 'img'
        img = sample.get("image", None) or sample.get("img", None)
        if not isinstance(img, Image.Image):
            # images may be in array format
            img = Image.fromarray(np.array(img))
        x = self.transform(img)
        y = sample["label"] if "label" in sample else sample.get("labels")
        return x, int(y)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def stratified_split(dataset, labels, val_size=0.15, test_size=0.15, seed=42):
    y = np.array(labels)
    sss1 = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    idx = np.arange(len(y))
    train_val_idx, test_idx = next(sss1.split(idx, y))
    y_train_val = y[train_val_idx]
    sss2 = StratifiedShuffleSplit(
        n_splits=1, test_size=val_size / (1.0 - test_size), random_state=seed
    )
    train_idx, val_idx = next(sss2.split(train_val_idx, y_train_val))
    return train_val_idx[train_idx], train_val_idx[val_idx], test_idx


def accuracy(logits, targets):
    preds = torch.argmax(logits, dim=1)
    return (preds == targets).float().mean().item()


def _unpack_batch(batch, device):
    """
    MultiModalityDataset may return (x, y) or (x, y, extra...).
    We only care about the first two entries.
    """
    # if isinstance(batch, (list, tuple)):
    #     if len(batch) < 2:
    #         raise ValueError(f"Unexpected batch with length {len(batch)}")
    #     xb, yb = batch[0], batch[1]
    # else:
    #     # Fallback: assume standard (x, y)
    #     xb, yb = batch

    # xb = xb.to(device)
    # yb = yb.to(device)
    # return xb, yb

    if isinstance(batch, dict):
        xb = batch["image"].to(device)
        yb = batch["label"].to(device)
        return xb, yb

    if isinstance(batch, (list, tuple)):
        xb, yb = batch[0], batch[1]
        return xb.to(device), yb.to(device)

    raise ValueError(f"Unexpected batch type: {type(batch)}")


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/train.yaml")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg_dict = yaml.safe_load(f)
    cfg = Config(**cfg_dict)

    set_seed(cfg.seed)
    device = torch.device(
        cfg.device if torch.cuda.is_available() and cfg.device == "cuda" else "cpu"
    )
    os.makedirs("outputs", exist_ok=True)

    # ------------------------------------------------------------------
    # DATA LOADING: HF (legacy) vs ClearML (new)
    # ------------------------------------------------------------------
    data_source = cfg.data.get("source", "hf")

    if data_source == "hf":
        # ===== HF path (kept for reference, normally not used now) =====
        print("[Data] Loading HF dataset:", cfg.data["dataset_name"])
        ds = load_dataset(cfg.data["dataset_name"])

        split_name = "train" if "train" in ds else list(ds.keys())[0]
        full = ds[split_name]

        # Optional subset for fast iteration
        if cfg.data.get("subset_fraction", 1.0) < 1.0:
            n = int(len(full) * cfg.data["subset_fraction"])
            full = full.shuffle(seed=cfg.seed).select(range(n))
            print(f"[Data] Using subset: {n} samples")

        labels = [
            int(item.get("label") if "label" in item else item.get("labels"))
            for item in full
        ]
        num_classes = len(set(labels))
        
        # Try to extract class names
        class_names = None
        if "labels" in full.features and hasattr(full.features["labels"], "names"):
            class_names = full.features["labels"].names
        elif "label" in full.features and hasattr(full.features["label"], "names"):
            class_names = full.features["label"].names
            
        print(f"[Data] Classes: {num_classes}, Samples: {len(full)}")
        if class_names:
            print(f"[Data] Class names found: {len(class_names)}")

        # Stratified split
        train_idx, val_idx, test_idx = stratified_split(
            full,
            labels,
            cfg.data["val_size"],
            cfg.data["test_size"],
            seed=cfg.seed,
        )
        train_hf = full.select(train_idx.tolist())
        val_hf = full.select(val_idx.tolist())
        test_hf = full.select(test_idx.tolist())

        # Transforms
        train_transforms = get_transforms(
            image_size=cfg.data["image_size"],
            train=True,
            normalize=cfg.data["normalize"],
            augment=cfg.data["augment"],
        )
        eval_transforms = get_transforms(
            image_size=cfg.data["image_size"],
            train=False,
            normalize=cfg.data["normalize"],
            augment=False,
        )

        train_ds = HFDataset(train_hf, transform=train_transforms["color"])
        val_ds = HFDataset(val_hf, transform=eval_transforms["color"])
        test_ds = HFDataset(test_hf, transform=eval_transforms["color"])

        train_loader = DataLoader(
            train_ds,
            batch_size=cfg.train["batch_size"],
            shuffle=True,
            num_workers=cfg.data["num_workers"],
            pin_memory=True,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=cfg.train["batch_size"],
            shuffle=False,
            num_workers=cfg.data["num_workers"],
            pin_memory=True,
        )
        test_loader = DataLoader(
            test_ds,
            batch_size=cfg.train["batch_size"],
            shuffle=False,
            num_workers=cfg.data["num_workers"],
            pin_memory=True,
        )

    else:
        # ===================== CLEARML DATASET PATH =====================
        print("[Data] Using ClearML dataset pipeline")

        # 1) Choose which subset to use
        DATASET_IDS = {
            "tiny": "e1277db0a8a445d9b6faa1f3947c1fe0",  # ~5%
            "medium": "ee3e7d7e511a47449f7206809eced7c1",  # ~30%
            "large": "a20b80fd8e85450d9db29dc867a13c3e",  # ~60%
        }
        subset_key = cfg.data.get("clearml_subset", "tiny")
        dataset_id = DATASET_IDS[subset_key]

        print(f"[Data] Fetching ClearML Dataset '{subset_key}' ({dataset_id})")
        cl_dataset = ClearMLDataset.get(dataset_id)
        local_path = cl_dataset.get_local_copy()
        local_path = ensure_dataset_extracted(local_path)

        # 2) Build class mapping & sample list
        modalities = cfg.data.get("modalities", ["color"])
        class_names, class_to_idx = build_class_mapping(local_path, modality="color")
        num_classes = len(class_names)
        print(f"[Data] Found {num_classes} classes")

        samples = gather_samples(local_path, modalities, class_to_idx)
        print(f"[Data] Total samples in '{subset_key}' subset: {len(samples)}")

        # 3) Split into train / val / test
        train_samples, val_samples, test_samples = split_dataset(
            samples,
            test_size=cfg.data["test_size"],
            val_size=cfg.data["val_size"],
            random_state=cfg.seed,
        )
        print(
            f"[Data] Split sizes: "
            f"Train={len(train_samples)}, Val={len(val_samples)}, Test={len(test_samples)}"
        )

        # 4) Transforms & datasets
        train_transforms = get_transforms(
            image_size=cfg.data["image_size"],
            train=True,
            normalize=cfg.data["normalize"],
            augment=cfg.data["augment"],
        )

        eval_transforms = get_transforms(
            image_size=cfg.data["image_size"],
            train=False,
            normalize=cfg.data["normalize"],
            augment=False,
        )

        train_ds = MultiModalityDataset(train_samples, train_transforms)
        val_ds = MultiModalityDataset(val_samples, eval_transforms)
        test_ds = MultiModalityDataset(test_samples, eval_transforms)

        # 5) DataLoaders
        train_loader = DataLoader(
            train_ds,
            batch_size=cfg.train["batch_size"],
            shuffle=True,
            num_workers=cfg.data["num_workers"],
            pin_memory=True,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=cfg.train["batch_size"],
            shuffle=False,
            num_workers=cfg.data["num_workers"],
            pin_memory=True,
        )
        test_loader = DataLoader(
            test_ds,
            batch_size=cfg.train["batch_size"],
            shuffle=False,
            num_workers=cfg.data["num_workers"],
            pin_memory=True,
        )

        print("[Data] Dataloaders ready.")

    # ------------------------------------------------------------------
    # MODEL
    # ------------------------------------------------------------------

    arch = cfg.model.get("arch", "scratch")

    if arch == "scratch":
        model = build_scratch_model(
            num_classes=num_classes,
            channels=cfg.model["channels"],
            regularisation=cfg.model[
                "regularisation"
            ],  # "none" | "dropout" | "batchnorm"
            dropout=cfg.model["dropout"],  # used when regularisation == "dropout"
        ).to(device)

    elif arch == "resnet18":
        model = ResNet18Classifier(
            num_classes=num_classes,
            # adapt these kwargs to match your __init__ in resnet.py
            pretrained=cfg.model.get("pretrained", True),
            dropout=cfg.model.get("dropout", 0.0),
            train_backbone=cfg.model.get("train_backbone", True),
        ).to(device)

    else:
        raise ValueError(f"Unknown model arch: {arch}")

    # Build model_config for checkpoint (enables architecture-agnostic evaluation)
    model_config = {
        "arch": arch,
        "num_classes": num_classes,
        "label_names": class_names if 'class_names' in locals() else None,
    }
    if arch == "scratch":
        model_config.update({
            "channels": cfg.model["channels"],
            "regularisation": cfg.model["regularisation"],
            "dropout": cfg.model["dropout"],
        })
    elif arch == "resnet18":
        model_config.update({
            "pretrained": cfg.model.get("pretrained", True),
            "dropout": cfg.model.get("dropout", 0.0),
            "train_backbone": cfg.model.get("train_backbone", True),
        })

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

    # ClearML
    task = init_task(
        enabled=cfg.clearml["enabled"],
        project=cfg.clearml.get("project") or cfg.project_name,
        task_name=cfg.clearml.get("task_name") or cfg.task_name,
        tags=cfg.tags,
        params=cfg_dict,
    )

    # If configured, send this task to a remote ClearML queue and stop local execution
    remote_queue = cfg.clearml.get("queue")
    # if task is not None and remote_queue:
    #     try:
    #         from clearml import Task as ClearMLTask

    #         # Only enqueue if we are running locally; on the agent this is a no-op
    #         if ClearMLTask.running_locally():
    #             print(f"[ClearML] Executing remotely on queue '{remote_queue}'")
    #             task.execute_remotely(queue_name=remote_queue, exit_process=True)
    #     except Exception as e:
    #         print(f"[ClearML] execute_remotely failed ({e}), continuing locally.")

    # Early stopping
    early_stopper = EarlyStopping(
        patience=cfg.train.get("patience", 3),
        min_delta=cfg.train.get("min_delta", 0.0),
    )

    # Training with early stopping on val accuracy
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
                {
                    "model_state": model.state_dict(),
                    "val_acc": best_val_acc,
                    "model_config": model_config,  # Enables architecture-agnostic evaluation
                },
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
