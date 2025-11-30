import argparse
import json
import os
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn.functional as F
import yaml
from datasets import load_dataset
from PIL import Image
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import DataLoader, Dataset

from data.transforms import get_transforms  # the multi-modality transforms

from .clearml_utils import init_task, log_scalar

from .models.resnet import ResNet18Classifier


def set_seed(seed: int):
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def stratified_split(labels, val_size=0.15, test_size=0.15, seed=42):

    y = np.array(labels)
    idx = np.arange(len(y))

    sss1 = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_val_idx, test_idx = next(sss1.split(idx, y))
    y_train_val = y[train_val_idx]

    sss2 = StratifiedShuffleSplit(
        n_splits=1,
        test_size=val_size / (1.0 - test_size),
        random_state=seed,
    )
    train_idx, val_idx = next(sss2.split(train_val_idx, y_train_val))

    return train_val_idx[train_idx], train_val_idx[val_idx], test_idx


class HFDataset(Dataset):

    def __init__(self, hf_split, transform):
        self.hf_split = hf_split
        self.transform = transform

        # Try to get label names if available
        self.label_names = None
        if "labels" in hf_split.features and hasattr(
            hf_split.features["labels"], "names"
        ):
            self.label_names = hf_split.features["labels"].names

    def __len__(self):
        return len(self.hf_split)

    def __getitem__(self, idx):
        sample = self.hf_split[idx]
        # images might be under "image" or "img"
        img = sample.get("image", None) or sample.get("img", None)
        if not isinstance(img, Image.Image):
            img = Image.fromarray(np.array(img))
        x = self.transform(img)

        # labels might be under "label" or "labels"
        if "label" in sample:
            y = int(sample["label"])
        elif "labels" in sample:
            y = int(sample["labels"])
        else:
            raise KeyError(f"No 'label' or 'labels' key in sample {idx}")

        return x, y


def top5_accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    """
    logits: (N, num_classes)
    targets: (N,)
    """
    topk = torch.topk(logits, k=5, dim=1).indices  # (N, 5)
    targets = targets.view(-1, 1).expand_as(topk)
    correct = (topk == targets).any(dim=1).float()
    return correct.mean().item()


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str],
    save_path: str,
    title: str = "Confusion Matrix (ResNet18)",
):
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        cm,
        annot=False,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(title)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.savefig(save_path)
    plt.close()


def evaluate_resnet(cfg_path: str, checkpoint_path: str, split: str):
    # 1. Load config
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg["seed"])

    device = torch.device(
        "cuda" if torch.cuda.is_available() and cfg["device"] == "cuda" else "cpu"
    )

    # 2. Load HF dataset (same as train HF path)
    ds = load_dataset(cfg["data"]["dataset_name"])
    split_name = "train" if "train" in ds else list(ds.keys())[0]
    full = ds[split_name]

    # optional subset
    subset_fraction = cfg["data"].get("subset_fraction", 1.0)
    if subset_fraction < 1.0:
        n = int(len(full) * subset_fraction)
        full = full.shuffle(seed=cfg["seed"]).select(range(n))
        print(f"[Data] Using subset: {n} samples")

    # Build labels list for splitting
    labels = []
    for item in full:
        if "label" in item:
            labels.append(int(item["label"]))
        elif "labels" in item:
            labels.append(int(item["labels"]))
        else:
            raise KeyError("Sample missing 'label'/'labels' key")

    # Recreate splits exactly like train.py
    train_idx, val_idx, test_idx = stratified_split(
        labels,
        val_size=cfg["data"]["val_size"],
        test_size=cfg["data"]["test_size"],
        seed=cfg["seed"],
    )

    if split == "train":
        eval_hf = full.select(train_idx.tolist())
    elif split == "val":
        eval_hf = full.select(val_idx.tolist())
    elif split == "test":
        eval_hf = full.select(test_idx.tolist())
    else:
        raise ValueError(f"Unknown split '{split}', use train|val|test")

    print(f"[Data] Evaluating on '{split}' split with {len(eval_hf)} samples")

    # 3. Transforms & DataLoader (use color modality, like train HF branch)
    eval_transforms = get_transforms(
        image_size=cfg["data"]["image_size"],
        train=False,
        normalize=cfg["data"]["normalize"],
        augment=False,
    )  # returns dict with 'color', 'grayscale', 'segmented'

    eval_ds = HFDataset(eval_hf, transform=eval_transforms["color"])
    eval_loader = DataLoader(
        eval_ds,
        batch_size=cfg["train"]["batch_size"],
        shuffle=False,
        num_workers=cfg["data"]["num_workers"],
        pin_memory=torch.cuda.is_available(),
    )

    # 4. Build ResNet18 model
    model_cfg = cfg["model"]
    assert model_cfg["arch"] == "resnet18", "Config must be for ResNet18"

    model = ResNet18Classifier(
        num_classes=model_cfg["num_classes"],
        pretrained=model_cfg.get("pretrained", True),
        dropout=model_cfg.get("dropout", 0.0),
        train_backbone=model_cfg.get("train_backbone", True),
    ).to(device)

    # 5. Load checkpoint
    print(f"[Model] Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get("model_state", checkpoint)
    model.load_state_dict(state_dict)
    model.eval()

    # 6. Run evaluation loop
    all_logits = []
    all_targets = []

    with torch.no_grad():
        for xb, yb in eval_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            logits = model(xb)
            all_logits.append(logits.cpu())
            all_targets.append(yb.cpu())

    all_logits = torch.cat(all_logits, dim=0)  # (N, num_classes)
    all_targets = torch.cat(all_targets, dim=0)  # (N,)

    preds = all_logits.argmax(dim=1).numpy()
    targets = all_targets.numpy()

    overall_acc = float((preds == targets).mean())
    top5_acc = float(top5_accuracy(all_logits, all_targets))

    num_classes = all_logits.shape[1]
    precision, recall, f1, support = precision_recall_fscore_support(
        targets, preds, labels=list(range(num_classes)), zero_division=0
    )
    cm = confusion_matrix(targets, preds, labels=list(range(num_classes)))

    # 7. Label names (if available)
    label_names = eval_ds.label_names
    if label_names is None or len(label_names) != num_classes:
        label_names = [f"Class_{i}" for i in range(num_classes)]

    print("\n=== ResNet18 Evaluation Results ===")
    print(f"Split: {split}")
    print(f"Overall accuracy: {overall_acc:.4f}")
    print(f"Top-5 accuracy : {top5_acc:.4f}\n")

    print("Per-class metrics:")
    for i in range(num_classes):
        print(
            f"{label_names[i]}: "
            f"Precision={precision[i]:.3f}, "
            f"Recall={recall[i]:.3f}, "
            f"F1={f1[i]:.3f}, "
            f"Support={support[i]}"
        )

    # Confusion matrix plot
    cm_path = os.path.join("outputs", f"resnet18_confusion_matrix_{split}.png")
    print(f"\n[Output] Saving confusion matrix to {cm_path}")
    plot_confusion_matrix(cm, label_names, cm_path)

    # Save raw metrics
    results = {
        "split": split,
        "overall_accuracy": overall_acc,
        "top5_accuracy": top5_acc,
        "per_class_precision": precision.tolist(),
        "per_class_recall": recall.tolist(),
        "per_class_f1": f1.tolist(),
        "per_class_support": support.tolist(),
        "confusion_matrix": cm.tolist(),
        "label_names": label_names,
    }
    os.makedirs("outputs", exist_ok=True)
    json_path = os.path.join("outputs", f"resnet18_eval_{split}.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[Output] Metrics saved to {json_path}")

    # 8. Optional ClearML logging as a separate "evaluation" task
    clearml_cfg = cfg.get("clearml", {})
    if clearml_cfg.get("enabled", False):
        print("[ClearML] Logging evaluation metrics...")
        eval_task = init_task(
            enabled=True,
            project=clearml_cfg.get("project") or cfg.get("project_name"),
            task_name=f"resnet18-eval-{split}",
            tags=(cfg.get("tags") or []) + ["evaluation", "resnet18"],
            params={
                "config_path": cfg_path,
                "checkpoint_path": checkpoint_path,
                "split": split,
            },
        )
        if eval_task is not None:
            log_scalar(eval_task, "accuracy", "overall", overall_acc, 0)
            log_scalar(eval_task, "accuracy", "top5", top5_acc, 0)
            # you could upload cm image as an artifact if you want using eval_task.upload_artifact(...)
        print("[ClearML] Done.")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate ResNet18 model on PlantVillage"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/train_resnet18.yaml",
        help="Path to ResNet18 train config",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to checkpoint (e.g., models/checkpoints/resnet18_best.pt)",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "val", "test"],
        help="Which split to evaluate: train | val | test",
    )
    args = parser.parse_args()

    evaluate_resnet(args.config, args.checkpoint, args.split)


if __name__ == "__main__":
    main()
