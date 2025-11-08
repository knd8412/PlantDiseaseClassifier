"""
Comprehensive evaluation script for PlantDiseaseClassifier

Calculates overall accuracy, top-five accuracy, per-class precision and recall,
and generates a confusion matrix visualization.

Usage:
    python evaluate.py --model outputs/best.pt --split val
"""

import argparse
import json
import os
from typing import List, Dict, Tuple
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_fscore_support
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import seaborn as sns
from datasets import load_dataset
from PIL import Image
import yaml

# Import from relative modules when running as script
try:
    from .data.transforms import get_transforms
    from .models.convnet_scratch import build_model
    from .clearml_utils import init_task, log_scalar, log_image
except ImportError:
    # Fallback for direct script execution
    from data.transforms import get_transforms
    from models.convnet_scratch import build_model
    from clearml_utils import init_task, log_scalar, log_image

class HFDataset:
    """
    Dataset wrapper for Hugging Face splits, used for evaluation.

    Parameters:
        hf_split: Hugging Face dataset split (e.g., train, val, test).
        transform: Callable transform to apply to each image.

    Purpose:
        - Adapts Hugging Face datasets to PyTorch DataLoader interface.
        - Handles different possible image and label key formats (e.g., "image" or "img" for images, "label" or "labels" for labels).
        - Extracts label_names from dataset features if available, which is useful for generating human-readable reports and confusion matrices.
    """
    def __init__(self, hf_split, transform):
        self.hf_split = hf_split
        self.transform = transform
        
        # Extract label names from features (if present)
        self.label_names = None
        print(f"[DEBUG] HFDataset features: {list(hf_split.features.keys())}")
        if "labels" in hf_split.features and hasattr(hf_split.features["labels"], "names"):
            self.label_names = hf_split.features["labels"].names
            print(f"[DEBUG] Extracted label names from 'labels': {self.label_names}")
        elif "label" in hf_split.features and hasattr(hf_split.features["label"], "names"):
            self.label_names = hf_split.features["label"].names
            print(f"[DEBUG] Extracted label names from 'label': {self.label_names}")
        else:
            print("[DEBUG] No label names found in dataset features")
        
    def __len__(self):
        return len(self.hf_split)
    
    def __getitem__(self, idx):
        sample = self.hf_split[idx]
        img = sample.get("image", None) or sample.get("img", None)
        if not isinstance(img, Image.Image):
            img = Image.fromarray(np.array(img))
        x = self.transform(img)
        if "label" in sample:
            y = sample["label"]
        elif "labels" in sample:
            y = sample["labels"]
        else:
            raise KeyError("Sample does not contain 'label' or 'labels' key.")
        return x, int(y)

def load_model(model_path: str, config_path: str = "configs/train.yaml") -> torch.nn.Module:
    """Load trained model with configuration"""
    print(f"[DEBUG] Loading config from {config_path}")
    with open(config_path, "r") as f:
        cfg_dict = yaml.safe_load(f)
    
    # Load dataset to get number of classes
    print(f"[DEBUG] Loading dataset: {cfg_dict['data']['dataset_name']}")
    ds = load_dataset(cfg_dict["data"]["dataset_name"])
    split_name = "train" if "train" in ds else list(ds.keys())[0]
    full = ds[split_name]
    labels = []
    for i, item in enumerate(full):
        if "label" in item:
            labels.append(int(item["label"]))
        elif "labels" in item:
            labels.append(int(item["labels"]))
        else:
            raise KeyError(f"Item {i} does not contain 'label' or 'labels' key. Available keys: {list(item.keys())}")
    num_classes = len(set(labels))
    print(f"[DEBUG] Detected {num_classes} classes from dataset")
    
    # Build model
    print(f"[DEBUG] Building model with config: {cfg_dict['model']}")
    model = build_model(
        num_classes=num_classes,
        channels=cfg_dict["model"]["channels"],
        use_batchnorm=cfg_dict["model"]["use_batchnorm"],
        dropout=cfg_dict["model"]["dropout"],
    )
    
    # Load weights
    print(f"[DEBUG] Loading checkpoint from {model_path}")
    checkpoint = torch.load(model_path, map_location="cpu")
    print(f"[DEBUG] Checkpoint keys: {list(checkpoint.keys())}")
    model.load_state_dict(checkpoint["model_state"])
    print("[DEBUG] Model loaded successfully")
    return model

def top_5_accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    """Calculate top-5 accuracy"""
    # logits: (N, num_classes), targets: (N,)
    top5 = torch.topk(logits, 5, dim=1).indices
    targets = targets.view(-1, 1).expand_as(top5)
    correct = (top5 == targets).any(dim=1).float()
    return correct.mean().item()

def evaluate_model(model: torch.nn.Module, loader: DataLoader, device: torch.device, label_names=None) -> Dict:
    """Run model on loader and compute metrics"""
    print(f"[DEBUG] Starting evaluation on device: {device}")
    model.eval()
    logits_list = []
    preds_list = []
    targets_list = []
    
    batch_count = 0
    with torch.no_grad():
        for x, y in loader:
            batch_count += 1
            print(f"[DEBUG] Processing batch {batch_count}, shape: {x.shape}")
            x = x.to(device)
            y = y.to(device)
            out = model(x)
            print(f"[DEBUG] Model output type: {type(out)}, shape: {getattr(out, 'shape', 'No shape')}")
            if isinstance(out, tuple) or isinstance(out, list):
                out = out[0]
                print(f"[DEBUG] Extracted first element, shape: {out.shape}")
            logits_cpu = out.detach().cpu()
            preds_cpu = logits_cpu.argmax(dim=1).numpy()
            logits_list.append(logits_cpu)
            preds_list.append(preds_cpu)
            targets_list.append(y.detach().cpu().numpy())
    
    print(f"[DEBUG] Concatenating results from {batch_count} batches")
    logits = torch.cat(logits_list, dim=0)
    preds = np.concatenate(preds_list, axis=0)
    targets = np.concatenate(targets_list, axis=0)
    
    print(f"[DEBUG] Final shapes - logits: {logits.shape}, preds: {preds.shape}, targets: {targets.shape}")
    
    overall_accuracy = float((preds == targets).mean())
    top5_accuracy_val = top_5_accuracy(logits, torch.from_numpy(targets))
    
    num_classes = logits.shape[1]
    print(f"[DEBUG] Calculating metrics for {num_classes} classes")
    precision, recall, fscore, support = precision_recall_fscore_support(targets, preds, labels=list(range(num_classes)), zero_division=0)
    cm = confusion_matrix(targets, preds, labels=list(range(num_classes)))
    
    return {
        "overall_accuracy": overall_accuracy,
        "top5_accuracy": top5_accuracy_val,
        "per_class_precision": precision,
        "per_class_recall": recall,
        "per_class_fscore": fscore,
        "per_class_support": support,
        "confusion_matrix": cm,
        "label_names": label_names
    }

def plot_confusion_matrix(cm: np.ndarray, class_names: List[str] = None, save_path: str = "confusion_matrix.png"):
    """Save a confusion matrix heatmap"""
    plt.figure(figsize=(10, 8))
    if class_names is None:
        labels = list(range(cm.shape[0]))
    else:
        labels = class_names
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Evaluate PlantDiseaseClassifier model")
    parser.add_argument("--model", required=True, help="Path to model checkpoint (contains model_state)")
    parser.add_argument("--config", default="configs/train.yaml", help="Path to train config yaml")
    parser.add_argument("--split", default="val", help="Dataset split to evaluate (e.g., val, test)")
    parser.add_argument("--output", default="outputs/eval_results.json", help="Path to save evaluation results")
    args = parser.parse_args()

    # Load config
    with open(args.config, "r") as f:
        cfg_dict = yaml.safe_load(f)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load model
    print(f"[Evaluation] Loading model from {args.model}")
    model = load_model(args.model, args.config)
    model.to(device)
    
    # Load dataset
    print(f"[Evaluation] Loading dataset: {cfg_dict['data']['dataset_name']}")
    ds = load_dataset(cfg_dict["data"]["dataset_name"])
    
    # Get evaluation split
    print(f"[DEBUG] Available dataset splits: {list(ds.keys())}")
    if args.split in ds:
        eval_split = ds[args.split]
        eval_split_name = args.split
        print(f"[DEBUG] Using split '{eval_split_name}' with {len(eval_split)} samples")
    else:
        # Use train split if specified split doesn't exist
        if "train" in ds:
            eval_split = ds["train"]
            eval_split_name = "train"
        else:
            first_key = list(ds.keys())[0]
            eval_split = ds[first_key]
            eval_split_name = first_key
        print(f"[Evaluation] Split '{args.split}' not found, using '{eval_split_name}' with {len(eval_split)} samples")
    
    # Apply transforms
    _, eval_tf = get_transforms(
        img_size=cfg_dict["data"]["image_size"],
        normalize=cfg_dict["data"]["normalize"],
        augment=False  # No augmentation for evaluation
    )
    
    eval_ds = HFDataset(eval_split, transform=eval_tf)
    eval_loader = DataLoader(eval_ds, batch_size=cfg_dict["train"]["batch_size"], 
                             shuffle=False, num_workers=cfg_dict["data"]["num_workers"], pin_memory=torch.cuda.is_available())
    
    # Run evaluation
    print("[Evaluation] Running evaluation...")
    results = evaluate_model(model, eval_loader, device, label_names=eval_ds.label_names)
    
    # Print results
    print(f"\n=== Evaluation Results ===")
    print(f"Overall Accuracy: {results['overall_accuracy']:.4f}")
    print(f"Top-5 Accuracy: {results['top5_accuracy']:.4f}")
    print(f"\nPer-class metrics:")
    for i, (p, r, f, s) in enumerate(zip(results['per_class_precision'],
                                        results['per_class_recall'],
                                        results['per_class_fscore'],
                                        results['per_class_support'])):
        class_name = results['label_names'][i] if results['label_names'] else f"Class {i}"
        print(f"{class_name}: Precision={p:.3f}, Recall={r:.3f}, F1={f:.3f}, Support={s}")
    
    # Generate confusion matrix
    print("[Evaluation] Generating confusion matrix...")
    plot_confusion_matrix(results['confusion_matrix'],
                         class_names=results['label_names'],
                         save_path="confusion_matrix.png")
    
    # Save results
    print(f"[Evaluation] Saving results to {args.output}")
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        # Convert numpy arrays to lists for JSON serialization
        json_results = {
            "overall_accuracy": float(results["overall_accuracy"]),
            "top5_accuracy": float(results["top5_accuracy"]),
            "per_class_precision": results["per_class_precision"].tolist(),
            "per_class_recall": results["per_class_recall"].tolist(),
            "per_class_fscore": results["per_class_fscore"].tolist(),
            "per_class_support": results["per_class_support"].tolist(),
            "confusion_matrix": results["confusion_matrix"].tolist(),
            "label_names": results["label_names"]
        }
        json.dump(json_results, f, indent=2)
    
    # ClearML integration
    task = init_task(
        enabled=cfg_dict.get("clearml", {}).get("enabled", False),
        project=cfg_dict.get("clearml", {}).get("project") or cfg_dict.get("project_name"),
        task_name=f"evaluation-{args.split}",
        tags=["evaluation"] + cfg_dict.get("tags", []),
        params={"model_path": args.model, "split": args.split}
    )
    
    if task:
        log_scalar(task, "accuracy", "overall", results["overall_accuracy"], 0)
        log_scalar(task, "accuracy", "top5", results["top5_accuracy"], 0)
        log_image(task, "confusion_matrix", "confusion_matrix.png")
        print("[Evaluation] Results logged to ClearML")
    
    print("[Evaluation] Evaluation completed successfully!")

if __name__ == "__main__":
    main()