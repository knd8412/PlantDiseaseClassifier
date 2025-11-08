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
        
        # Robust label name extraction with multiple fallback strategies
        self.label_names = self._extract_label_names(hf_split)
        
    def _extract_label_names(self, hf_split):
        """Extract label names using multiple strategies with graceful fallbacks"""
        label_names = None
        
        # Strategy 1: Check dataset features for label names
        print(f"[DEBUG] HFDataset features: {list(hf_split.features.keys())}")
        
        # Check for various feature key patterns that might contain label names
        feature_keys_to_check = ["labels", "label", "class", "category", "target"]
        
        for key in feature_keys_to_check:
            if key in hf_split.features:
                feature = hf_split.features[key]
                # Check for common attribute patterns
                if hasattr(feature, "names"):
                    label_names = feature.names
                    print(f"[DEBUG] Extracted label names from feature '{key}': {label_names}")
                    return label_names
                elif hasattr(feature, "_int2str") and callable(feature._int2str):
                    # Handle ClassLabel.int2str mapping
                    try:
                        num_classes = len(hf_split)
                        label_names = [feature._int2str(i) for i in range(num_classes)]
                        print(f"[DEBUG] Extracted label names using _int2str from '{key}': {label_names}")
                        return label_names
                    except (IndexError, ValueError):
                        continue
                elif hasattr(feature, "names") and isinstance(feature.names, list):
                    label_names = feature.names
                    print(f"[DEBUG] Extracted label names from feature '{key}': {label_names}")
                    return label_names
        
        # Strategy 2: Extract unique labels from the dataset and generate names
        print("[DEBUG] Attempting to extract labels from dataset samples...")
        unique_labels = set()
        max_samples_to_check = min(1000, len(hf_split))  # Limit to avoid excessive processing
        
        for i in range(max_samples_to_check):
            sample = hf_split[i]
            label = self._extract_label_from_sample(sample)
            if label is not None:
                unique_labels.add(label)
        
        if unique_labels:
            # Sort labels and generate names
            sorted_labels = sorted(unique_labels)
            label_names = [f"Class_{label}" for label in sorted_labels]
            print(f"[DEBUG] Generated label names from {len(unique_labels)} unique labels: {label_names}")
            return label_names
        
        # Strategy 3: Use generic names based on number of classes detected
        print("[DEBUG] Falling back to generic label names...")
        # Try to determine number of classes from the first few samples
        labels_found = []
        for i in range(min(100, len(hf_split))):
            sample = hf_split[i]
            label = self._extract_label_from_sample(sample)
            if label is not None and label not in labels_found:
                labels_found.append(label)
        
        if labels_found:
            num_classes = len(labels_found)
            label_names = [f"Class_{i}" for i in range(num_classes)]
            print(f"[DEBUG] Generated generic label names for {num_classes} classes")
            return label_names
        
        print("[DEBUG] Could not determine label names, using default numbering")
        return None
    
    def _extract_label_from_sample(self, sample):
        """Extract label value from a sample using multiple key patterns"""
        # Comprehensive list of possible label key patterns
        label_keys_to_check = [
            "label", "labels", "class", "category", "target",
            "disease_label", "plant_disease", "disease", "illness",
            "annotation", "y", "target_value", "ground_truth"
        ]
        
        # Also check for keys that contain these words as substrings
        available_keys = list(sample.keys())
        for key in available_keys:
            key_lower = key.lower()
            if any(pattern in key_lower for pattern in ["label", "class", "category", "target", "disease", "annotation"]):
                label_keys_to_check.append(key)
        
        # Remove duplicates while preserving order
        label_keys_to_check = list(dict.fromkeys(label_keys_to_check))
        
        for key in label_keys_to_check:
            if key in sample:
                try:
                    return int(sample[key])
                except (ValueError, TypeError):
                    # If not integer, try to handle string labels
                    if isinstance(sample[key], str):
                        return hash(sample[key]) % 1000  # Simple hash for string labels
                    continue
        
        return None
        
    def __len__(self):
        return len(self.hf_split)
    
    def __getitem__(self, idx):
        sample = self.hf_split[idx]
        img = sample.get("image", None) or sample.get("img", None)
        if not isinstance(img, Image.Image):
            img = Image.fromarray(np.array(img))
        x = self.transform(img)
        
        # Robust label extraction with better error handling
        y = self._extract_label_from_sample(sample)
        if y is None:
            available_keys = list(sample.keys())
            raise KeyError(f"Sample {idx} does not contain valid label key. Available keys: {available_keys}")
        
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
        label_value = None
        label_keys_to_check = ["label", "labels", "class", "category", "target"]
        
        for key in label_keys_to_check:
            if key in item:
                try:
                    label_value = int(item[key])
                    break
                except (ValueError, TypeError):
                    continue
        
        if label_value is None:
            available_keys = list(item.keys())
            raise KeyError(f"Item {i} does not contain valid label key. Available keys: {available_keys}")
        
        labels.append(label_value)
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
    if results['label_names'] and len(results['label_names']) == len(results['per_class_precision']):
        for i, (p, r, f, s) in enumerate(zip(results['per_class_precision'],
                                            results['per_class_recall'],
                                            results['per_class_fscore'],
                                            results['per_class_support'])):
            class_name = results['label_names'][i]
            print(f"{class_name}: Precision={p:.3f}, Recall={r:.3f}, F1={f:.3f}, Support={s}")
    else:
        # Fallback if label names don't match class count
        print(f"[WARNING] Label names ({len(results['label_names'])} names) don't match class count ({len(results['per_class_precision'])} classes)")
        for i, (p, r, f, s) in enumerate(zip(results['per_class_precision'],
                                            results['per_class_recall'],
                                            results['per_class_fscore'],
                                            results['per_class_support'])):
            class_name = f"Class_{i}"
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