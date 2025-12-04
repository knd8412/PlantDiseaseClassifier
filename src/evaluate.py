"""
Comprehensive evaluation script for PlantDiseaseClassifier

Calculates overall accuracy, top-five accuracy, per-class precision and recall,
generates a confusion matrix visualization, and creates an error gallery with
misclassified samples for analysis.

Architecture Detection:
    The script automatically detects the model architecture using a 3-step fallback:
    1. Checkpoint metadata (if saved by newer train.py with 'model_config' key)
    2. Auto-inference from state_dict weight shapes and key patterns
    3. Config file fallback (uses --config or default configs/train.yaml)

    This means --config is now OPTIONAL for most checkpoints!

Dataset:
    The PlantVillage dataset downloads automatically on first run and is cached at:
        ~/.cache/huggingface/datasets/
    Subsequent runs use the cache. If you're offline, it will use the cached version.

Usage:
    # Basic evaluation (auto-detects architecture from checkpoint)
    python src/evaluate.py --model outputs/best.pt --split val

    # Evaluate on test set
    python src/evaluate.py --model outputs/best.pt --split test

    # Skip error gallery for faster evaluation
    python src/evaluate.py --model outputs/best.pt --split val --no-error-gallery

    # Override with specific config (only needed for old checkpoints or edge cases)
    python src/evaluate.py --model outputs/best.pt --config configs/train_quick_test.yaml --split val

Options:
    --model PATH                 Path to model checkpoint (required)
    --config PATH                Config file (optional - auto-detected from checkpoint)
    --split NAME                 Dataset split to evaluate: val, test, or train (default: val)
    --output PATH                Path for results JSON (default: outputs/eval_results.json)
    --no-error-gallery           Disable error gallery generation
    --gallery-top-pairs N        Number of worst confusion pairs to analyze (default: 5)
    --gallery-samples-per-pair N Samples per confusion pair (default: 10)
    --error-gallery-dir DIR      Directory for error gallery output (default: errors)
    --quiet, -q                  Reduce output verbosity
    --dry-run                    Validate setup without running full evaluation
    --cm-classes N               Number of classes to show in confusion matrix (default: 15)
                                 Shows the N most confused classes; use 0 or 'all' for full matrix
    --exclude-classes LIST       List of class names to exclude (e.g. 'Background_without_leaves')
    --list-configs               List available config files and exit
    --list-models                List available model checkpoints and exit

Outputs:
    - outputs/eval_results.json     Detailed metrics in JSON format
    - outputs/confusion_matrix.png  Confusion matrix heatmap
    - errors/                       Error gallery with misclassified samples (if enabled)
"""

import argparse
import inspect
import json
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

# Check critical dependencies early with helpful error messages
_MISSING_DEPS = []

try:
    import numpy as np
except ImportError:
    _MISSING_DEPS.append("numpy")

try:
    import torch
    from torch.utils.data import DataLoader, Dataset
except ImportError:
    _MISSING_DEPS.append("torch")

try:
    from sklearn.metrics import (
        classification_report,
        confusion_matrix,
        precision_recall_fscore_support,
    )
    from sklearn.model_selection import StratifiedShuffleSplit
except ImportError:
    _MISSING_DEPS.append("scikit-learn")

try:
    import matplotlib.pyplot as plt
except ImportError:
    _MISSING_DEPS.append("matplotlib")

try:
    import seaborn as sns
except ImportError:
    _MISSING_DEPS.append("seaborn")

try:
    from datasets import load_dataset
except ImportError:
    _MISSING_DEPS.append("datasets")

try:
    from PIL import Image
except ImportError:
    _MISSING_DEPS.append("Pillow")

try:
    import yaml
except ImportError:
    _MISSING_DEPS.append("pyyaml")

if _MISSING_DEPS:
    print("=" * 60)
    print("ERROR: Missing required dependencies!")
    print("=" * 60)
    print(f"\nMissing packages: {', '.join(_MISSING_DEPS)}")
    print("\nTo install all requirements, run:")
    print("  pip install -r requirements.txt")
    print("\nOr install missing packages individually:")
    print(f"  pip install {' '.join(_MISSING_DEPS)}")
    print("=" * 60)
    sys.exit(1)


# Optional tqdm for progress bars (graceful fallback if not installed)
try:
    from tqdm import tqdm

    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

    def tqdm(iterable, **kwargs):
        """Fallback tqdm that just returns the iterable"""
        return iterable


# Import from relative modules when running as script
try:
    from .clearml_utils import init_task, log_image, log_scalar
    from .models.convnet_scratch import build_model
    from .models.resnet import ResNet18Classifier
except ImportError:
    # Fallback for direct script execution
    from clearml_utils import init_task, log_image, log_scalar
    from models.convnet_scratch import build_model
    from models.resnet import ResNet18Classifier

# Import transforms - handle both old (src/data) and new (data/) module locations
# The root data/ module returns a dict with 'color', 'grayscale', 'segmented' keys
# and uses 'image_size' parameter, which is what train.py uses

try:
    # Try importing from project root first (add parent dir to path if needed)
    import sys

    # from pathlib import Path

    project_root = Path(__file__).parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    # Now import - Python will search project root first
    import importlib

    transforms_module = importlib.import_module("data.transforms")
    _get_transforms_raw = transforms_module.get_transforms

    # Check which signature we got
    sig = inspect.signature(_get_transforms_raw)
    if "image_size" in sig.parameters:
        # New signature from data/transforms.py (returns dict with 'color' key)
        def get_transforms(image_size=224, normalize=True, augment=False):
            result = _get_transforms_raw(
                image_size=image_size, train=False, normalize=normalize, augment=augment
            )
            return result  # Returns dict with 'color', 'grayscale', 'segmented'

    else:
        # Old signature from src/data/transforms.py (returns tuple)
        def get_transforms(image_size=256, normalize=True, augment=False):
            train_tf, eval_tf = _get_transforms_raw(
                img_size=image_size, normalize=normalize, augment=augment
            )
            # Wrap in dict format for compatibility
            return {"color": eval_tf, "grayscale": eval_tf, "segmented": eval_tf}

except Exception as e:
    print(f"[WARNING] Could not import transforms: {e}")
    raise


def set_seed(seed: int):
    """Set random seeds for reproducibility (same as training)"""
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def stratified_split(labels, val_size=0.15, test_size=0.15, seed=42):
    """
    Recreate the exact train/val/test splits used during training.
    This ensures evaluation is on the same data split as training.
    """
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


def infer_architecture_from_state_dict(state_dict: dict) -> Tuple[str, dict]:
    """
    Infer model architecture and configuration from checkpoint state_dict.

    This enables architecture-agnostic evaluation without needing the original
    config file. Works by analyzing the weight tensor shapes and key patterns.

    Returns:
        Tuple of (arch_name, model_config_dict)

    Raises:
        ValueError if architecture cannot be determined
    """
    keys = list(state_dict.keys())

    # === ResNet18 Detection ===
    # ResNet18 has keys like "model.layer1.0.conv1.weight", "model.fc.weight" or "model.fc.1.weight"
    if any("model.layer1" in k for k in keys):
        # Find the final classification layer to get num_classes
        # With dropout: model.fc.1.weight (Sequential with Dropout + Linear)
        # Without dropout: model.fc.weight (just Linear)
        if "model.fc.1.weight" in state_dict:
            num_classes = state_dict["model.fc.1.weight"].shape[0]
            has_dropout = True
        elif "model.fc.weight" in state_dict:
            num_classes = state_dict["model.fc.weight"].shape[0]
            has_dropout = False
        else:
            raise ValueError("ResNet18 detected but could not find fc layer weights")

        config = {
            "arch": "resnet18",
            "num_classes": num_classes,
            "pretrained": True,  # Can't infer, assume True (common case)
            "dropout": 0.2 if has_dropout else 0.0,  # Approximate
            "train_backbone": True,  # Can't infer, doesn't affect eval
        }
        print(
            f"[AUTO-DETECT] Detected ResNet18 architecture: num_classes={num_classes}, dropout={'yes' if has_dropout else 'no'}"
        )
        return "resnet18", config

    # === ConvNet (Scratch) Detection ===
    # ConvNet has keys like "backbone.0.conv.0.weight", "head.2.weight"
    elif any("backbone" in k for k in keys):
        # Infer channels from the output channels of each ConvBlock's first conv
        # Pattern: backbone.{block_idx}.conv.0.weight -> shape [out_ch, in_ch, 3, 3]
        channels = []
        block_idx = 0
        while f"backbone.{block_idx}.conv.0.weight" in state_dict:
            weight = state_dict[f"backbone.{block_idx}.conv.0.weight"]
            channels.append(weight.shape[0])  # out_channels
            block_idx += 1

        if not channels:
            raise ValueError(
                "ConvNet detected but could not find backbone conv weights"
            )

        # Get num_classes from head's final linear layer
        # head structure: AdaptiveAvgPool2d, Flatten, Linear
        # So head.2.weight is the Linear layer
        if "head.2.weight" in state_dict:
            num_classes = state_dict["head.2.weight"].shape[0]
        else:
            raise ValueError("ConvNet detected but could not find head.2.weight")

        # Detect if BatchNorm is used by checking for bn layers
        # Pattern: backbone.{block_idx}.conv.1.weight for BatchNorm (if present)
        # Conv->BN->ReLU->Conv->BN->ReLU means index 1 would be BN
        has_batchnorm = (
            "backbone.0.conv.1.weight" in state_dict
            and state_dict["backbone.0.conv.1.weight"].dim() == 1
        )  # BN weight is 1D

        # Detect dropout by checking for Dropout2d layers (though they don't have weights)
        # We can't reliably detect dropout rate, so we'll use a default

        config = {
            "arch": "scratch",
            "num_classes": num_classes,
            "channels": channels,
            "regularisation": "batchnorm" if has_batchnorm else "none",
            "dropout": 0.0,  # Can't infer, but doesn't affect eval (disabled in eval mode)
        }
        print(
            f"[AUTO-DETECT] Detected ConvNet architecture: channels={channels}, num_classes={num_classes}, batchnorm={has_batchnorm}"
        )
        return "scratch", config

    else:
        # Unknown architecture
        sample_keys = keys[:10] if len(keys) > 10 else keys
        raise ValueError(
            f"Could not auto-detect model architecture from state_dict.\n"
            f"Sample keys: {sample_keys}\n"
            f"Please provide a --config file that matches the training configuration."
        )


def get_model_num_classes(model: torch.nn.Module) -> int:
    """
    Extract the number of output classes from a model.
    Supports ResNet18Classifier and SmallCNN (ConvNet).
    """
    # Case 1: ResNet18Classifier (wrapper)
    if hasattr(model, "model") and hasattr(model.model, "fc"):
        # Check if fc is Sequential (with dropout) or Linear
        if isinstance(model.model.fc, torch.nn.Sequential):
            for layer in model.model.fc:
                if isinstance(layer, torch.nn.Linear):
                    return layer.out_features
        elif isinstance(model.model.fc, torch.nn.Linear):
            return model.model.fc.out_features

    # Case 2: SmallCNN (ConvNet)
    if hasattr(model, "head"):
        # head is usually Sequential(Pool, Flatten, Linear)
        for layer in model.head:
            if isinstance(layer, torch.nn.Linear):
                return layer.out_features

    # Case 3: Generic fallback - check last module
    try:
        last_module = list(model.modules())[-1]
        if isinstance(last_module, torch.nn.Linear):
            return last_module.out_features
    except Exception:
        pass

    return 0


def load_dataset_robust(dataset_name: str):
    """
    Load dataset with automatic fallback to offline/cached mode.

    Tries online first, falls back to cached version if network fails.
    The dataset is cached locally after first download (~/.cache/huggingface/datasets/).
    """
    import os

    try:
        # Try normal loading (uses cache if available, checks for updates online)
        return load_dataset(dataset_name)
    except Exception as e:
        # Network error - try offline mode with cached data
        error_msg = str(e).lower()
        if (
            "connection" in error_msg
            or "timeout" in error_msg
            or "offline" in error_msg
            or "resolve" in error_msg
        ):
            print("[WARNING] Network unavailable, attempting to use cached dataset...")
            old_offline = os.environ.get("HF_DATASETS_OFFLINE")
            try:
                os.environ["HF_DATASETS_OFFLINE"] = "1"
                ds = load_dataset(dataset_name)
                print("[INFO] Using cached dataset (offline mode)")
                return ds
            except Exception as cache_err:
                raise RuntimeError(
                    f"Failed to load dataset '{dataset_name}'.\n"
                    f"Network error: {e}\n"
                    f"Cache error: {cache_err}\n\n"
                    f"To fix: Run once with internet to download the dataset, or ask a teammate to share their cache folder:\n"
                    f"  ~/.cache/huggingface/datasets/"
                ) from cache_err
            finally:
                if old_offline is None:
                    os.environ.pop("HF_DATASETS_OFFLINE", None)
                else:
                    os.environ["HF_DATASETS_OFFLINE"] = old_offline
        else:
            raise


class HFDataset(Dataset):
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

    def __init__(
        self, hf_split: Any, transform: Callable, label_names: List[str] = None
    ) -> None:
        self.hf_split = hf_split
        self.transform = transform

        # Robust label name extraction with multiple fallback strategies
        if label_names is not None:
            self.label_names = label_names
        else:
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
                    print(
                        f"[DEBUG] Extracted label names from feature '{key}': {label_names}"
                    )
                    return label_names
                elif hasattr(feature, "_int2str") and callable(feature._int2str):
                    # Handle ClassLabel.int2str mapping
                    try:
                        # Get num_classes from the feature's num_classes attribute, not dataset length
                        num_classes = getattr(feature, "num_classes", None)
                        if num_classes is None:
                            # Fallback: try to determine from feature length or names
                            num_classes = len(getattr(feature, "names", [])) or len(
                                getattr(feature, "_str2int", {})
                            )
                        if num_classes > 0:
                            label_names = [
                                feature._int2str(i) for i in range(num_classes)
                            ]
                            print(
                                f"[DEBUG] Extracted label names using _int2str from '{key}': {label_names}"
                            )
                            return label_names
                    except (IndexError, ValueError, TypeError):
                        continue
                elif hasattr(feature, "names") and isinstance(feature.names, list):
                    label_names = feature.names
                    print(
                        f"[DEBUG] Extracted label names from feature '{key}': {label_names}"
                    )
                    return label_names

        # Strategy 2: Extract unique labels from the dataset and generate names
        print("[DEBUG] Attempting to extract labels from dataset samples...")
        unique_labels = set()
        max_samples_to_check = min(
            1000, len(hf_split)
        )  # Limit to avoid excessive processing

        for i in range(max_samples_to_check):
            sample = hf_split[i]
            label = self._extract_label_from_sample(sample)
            if label is not None:
                unique_labels.add(label)

        if unique_labels:
            # Sort labels and generate names
            sorted_labels = sorted(unique_labels)
            label_names = [f"Class_{label}" for label in sorted_labels]
            print(
                f"[DEBUG] Generated label names from {len(unique_labels)} unique labels: {label_names}"
            )
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
            "label",
            "labels",
            "class",
            "category",
            "target",
            "disease_label",
            "plant_disease",
            "disease",
            "illness",
            "annotation",
            "y",
            "target_value",
            "ground_truth",
        ]

        # Also check for keys that contain these words as substrings
        available_keys = list(sample.keys())
        for key in available_keys:
            key_lower = key.lower()
            if any(
                pattern in key_lower
                for pattern in [
                    "label",
                    "class",
                    "category",
                    "target",
                    "disease",
                    "annotation",
                ]
            ):
                label_keys_to_check.append(key)

        # Remove duplicates while preserving order
        label_keys_to_check = list(dict.fromkeys(label_keys_to_check))

        for key in label_keys_to_check:
            if key in sample:
                try:
                    return int(sample[key])
                except (ValueError, TypeError):
                    # String labels should be handled by the dataset's label encoding
                    # Don't use hash() as it's not deterministic across Python runs
                    continue

        return None

    def __len__(self):
        return len(self.hf_split)

    def __getitem__(self, idx):
        sample = self.hf_split[idx]
        img = sample.get("image", None) or sample.get("img", None)
        if img is None:
            raise ValueError(
                f"Sample {idx} does not contain 'image' or 'img' key. Available keys: {list(sample.keys())}"
            )
        if not isinstance(img, Image.Image):
            img = Image.fromarray(np.array(img))
        x = self.transform(img)

        # Robust label extraction with better error handling
        y = self._extract_label_from_sample(sample)
        if y is None:
            available_keys = list(sample.keys())
            raise KeyError(
                f"Sample {idx} does not contain valid label key. Available keys: {available_keys}"
            )

        return x, int(y)


def load_model(model_path: str, config_path: Optional[str] = None) -> torch.nn.Module:
    """
    Load trained model with 3-step architecture detection fallback.

    Architecture detection priority:
    1. Checkpoint metadata (if saved by newer train.py with 'model_config' key)
    2. Auto-inference from state_dict weight shapes and key patterns
    3. Config file fallback (uses --config or default configs/train.yaml)

    This allows evaluation to work without requiring the original config file,
    making the script truly architecture-agnostic.
    """
    # Validate model path exists
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model checkpoint not found: {model_path}\n"
            f"Make sure you've trained a model first, or download one from ClearML."
        )

    # Load checkpoint first to check for embedded metadata
    print(f"[DEBUG] Loading checkpoint from {model_path}")
    checkpoint = torch.load(model_path, map_location="cpu")
    print(f"[DEBUG] Checkpoint keys: {list(checkpoint.keys())}")

    model_cfg = None
    arch = None
    num_classes = None
    config_source = None

    # === Step 1: Try to get config from checkpoint metadata ===
    if "model_config" in checkpoint:
        print("[STEP 1] Using model configuration embedded in checkpoint")
        model_cfg = checkpoint["model_config"]
        arch = model_cfg.get("arch", "scratch")
        num_classes = model_cfg.get("num_classes")
        config_source = "checkpoint"

    # === Step 2: Try to infer architecture from state_dict ===
    if model_cfg is None and "model_state" in checkpoint:
        print("[STEP 2] Auto-detecting architecture from model weights...")
        try:
            arch, model_cfg = infer_architecture_from_state_dict(
                checkpoint["model_state"]
            )
            num_classes = model_cfg.get("num_classes")
            config_source = "auto-detected"
        except ValueError as e:
            print(f"[STEP 2] Auto-detection failed: {e}")

    # === Step 3: Fall back to config file ===
    if model_cfg is None:
        if config_path is None:
            config_path = "configs/train.yaml"

        print(f"[STEP 3] Falling back to config file: {config_path}")

        if not os.path.exists(config_path):
            raise FileNotFoundError(
                f"Config file not found: {config_path}\n"
                f"Could not auto-detect architecture from checkpoint.\n"
                f"Please provide a --config file that matches the training configuration.\n"
                f"Available configs: configs/train.yaml, configs/train_quick_test.yaml"
            )

        with open(config_path, "r") as f:
            cfg_dict = yaml.safe_load(f)

        model_cfg = cfg_dict["model"]
        arch = model_cfg.get("arch", "scratch")
        num_classes = model_cfg.get("num_classes")
        config_source = f"config file ({config_path})"

        # If num_classes not in config, try to get from dataset
        if num_classes is None:
            print(
                f"[DEBUG] num_classes not in config, loading dataset: {cfg_dict['data']['dataset_name']}"
            )
            ds = load_dataset_robust(cfg_dict["data"]["dataset_name"])
            split_name = "train" if "train" in ds else list(ds.keys())[0]  # type: ignore[union-attr]
            full = ds[split_name]

            # Try to get num_classes from dataset features first (fast)
            label_feature = full.features.get("label") or full.features.get("labels")
            if label_feature and hasattr(label_feature, "num_classes"):
                num_classes = label_feature.num_classes
                print(f"[DEBUG] Got {num_classes} classes from dataset features")
            else:
                # Last resort: sample a subset to estimate class count
                print("[DEBUG] Sampling dataset to count classes...")
                unique_labels = set()
                for i, item in enumerate(full):
                    if i >= 1000:  # Sample first 1000 items
                        break
                    for key in ["label", "labels", "class", "category", "target"]:
                        if key in item:
                            try:
                                unique_labels.add(int(item[key]))
                                break
                            except (ValueError, TypeError):
                                continue
                num_classes = len(unique_labels)
                print(f"[DEBUG] Estimated {num_classes} classes from sampling")

    print(f"[DEBUG] Architecture: {arch} (source: {config_source})")
    print(f"[DEBUG] Model config: {model_cfg}")

    # === Build model using detected/configured architecture ===
    def build_scratch_model_wrapper(model_cfg, num_classes):
        """Build ConvNet from scratch"""
        import inspect

        build_model_params = inspect.signature(build_model).parameters

        if "regularisation" in build_model_params:
            # New signature: build_model(num_classes, channels, regularisation, dropout)
            if "regularisation" in model_cfg:
                regularisation = model_cfg["regularisation"]
            elif model_cfg.get("use_batchnorm", False):
                regularisation = "batchnorm"
            else:
                regularisation = "none"

            return build_model(
                num_classes=num_classes,
                channels=model_cfg.get("channels", [32, 64, 128]),
                regularisation=regularisation,
                dropout=model_cfg.get("dropout", 0.0),
            )
        elif "use_batchnorm" in build_model_params:
            # Old signature: build_model(num_classes, channels, use_batchnorm, dropout)
            if "regularisation" in model_cfg:
                use_batchnorm = model_cfg["regularisation"] == "batchnorm"
            else:
                use_batchnorm = model_cfg.get("use_batchnorm", False)

            return build_model(
                num_classes=num_classes,
                channels=model_cfg.get("channels", [32, 64, 128]),
                use_batchnorm=use_batchnorm,
                dropout=model_cfg.get("dropout", 0.0),
            )
        else:
            raise ValueError(
                f"Unknown build_model signature: {list(build_model_params.keys())}"
            )

    def build_resnet18_wrapper(model_cfg, num_classes):
        """Build ResNet18 classifier"""
        return ResNet18Classifier(
            num_classes=num_classes,
            pretrained=model_cfg.get("pretrained", True),
            dropout=model_cfg.get("dropout", 0.0),
            train_backbone=model_cfg.get("train_backbone", True),
        )

    # Architecture registry - extend this dict to add new models
    MODEL_BUILDERS = {
        "scratch": build_scratch_model_wrapper,
        "convnet": build_scratch_model_wrapper,  # Alias for backward compatibility
        "resnet18": build_resnet18_wrapper,
    }

    if arch not in MODEL_BUILDERS:
        available_archs = ", ".join(MODEL_BUILDERS.keys())
        raise ValueError(
            f"Unknown model architecture: '{arch}'\n"
            f"Available architectures: {available_archs}\n"
            f"To add a new architecture, extend the MODEL_BUILDERS dict in evaluate.py"
        )

    model = MODEL_BUILDERS[arch](model_cfg, num_classes)
    print(f"[DEBUG] Built {arch} model with num_classes={num_classes}")

    # Load weights
    try:
        model.load_state_dict(checkpoint["model_state"])
    except RuntimeError as e:
        if (
            "size mismatch" in str(e)
            or "Missing key" in str(e)
            or "Unexpected key" in str(e)
        ):
            # List available configs to help user
            configs_dir = Path("configs")
            available_configs = (
                list(configs_dir.glob("*.yaml")) if configs_dir.exists() else []
            )
            configs_list = (
                "\n  ".join([str(c) for c in available_configs[:10]])
                or "No configs found in configs/"
            )

            raise RuntimeError(
                f"Model weight mismatch!\n\n"
                f"Architecture detected: {arch} (source: {config_source})\n"
                f"Model config: {model_cfg}\n\n"
                f"This usually means the auto-detection failed or the config doesn't match.\n"
                f"Try specifying a --config that matches the training configuration.\n\n"
                f"Available configs:\n  {configs_list}\n\n"
                f"Example:\n"
                f"  python src/evaluate.py --model {model_path} --config configs/train_quick_test.yaml --split val\n\n"
                f"Original error: {e}"
            ) from e
        raise

    print("[DEBUG] Model loaded successfully")
    return model


def top_5_accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    """Calculate top-5 accuracy"""
    # logits: (N, num_classes), targets: (N,)
    top5 = torch.topk(logits, 5, dim=1).indices
    targets = targets.view(-1, 1).expand_as(top5)
    correct = (top5 == targets).any(dim=1).float()
    return correct.mean().item()


def evaluate_model(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    label_names=None,
    verbose: bool = True,
) -> Dict:
    """Run model on loader and compute metrics

    Args:
        model: PyTorch model to evaluate
        loader: DataLoader with evaluation data
        device: Device to run inference on
        label_names: Optional list of class names
        verbose: Whether to print debug info (default True)

    Returns:
        Dictionary with evaluation metrics, predictions, and confusion matrix
    """
    print(f"[Evaluation] Running on device: {device}")
    if not HAS_TQDM:
        print("[TIP] Install tqdm for progress bars: pip install tqdm")

    model.eval()
    logits_list = []
    preds_list = []
    targets_list = []

    total_batches = len(loader)
    total_samples = len(loader.dataset) if hasattr(loader, "dataset") else "unknown"
    print(
        f"[Evaluation] Processing {total_samples} samples in {total_batches} batches..."
    )

    # Memory-efficient evaluation with periodic progress updates
    last_progress_pct = 0
    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(
            tqdm(loader, desc="Evaluating", disable=not HAS_TQDM)
        ):
            # Progress update for non-tqdm users
            if not HAS_TQDM and verbose:
                progress_pct = int((batch_idx + 1) / total_batches * 100)
                # Update every 10%
                if progress_pct >= last_progress_pct + 10:
                    last_progress_pct = progress_pct
                    print(
                        f"[Evaluation] Progress: {progress_pct}% ({batch_idx + 1}/{total_batches} batches)"
                    )

            x = x.to(device)
            y = y.to(device)
            out = model(x)
            if isinstance(out, tuple) or isinstance(out, list):
                out = out[0]
            logits_cpu = out.detach().cpu()
            preds_cpu = logits_cpu.argmax(dim=1).numpy()
            logits_list.append(logits_cpu)
            preds_list.append(preds_cpu)
            targets_list.append(y.detach().cpu().numpy())

            # Periodic memory cleanup for very large datasets
            if batch_idx > 0 and batch_idx % 100 == 0:
                if device.type == "cuda":
                    torch.cuda.empty_cache()

    logits = torch.cat(logits_list, dim=0)
    preds = np.concatenate(preds_list, axis=0)
    targets = np.concatenate(targets_list, axis=0)

    print(f"[Evaluation] Evaluated {len(preds)} samples")

    overall_accuracy = float((preds == targets).mean())
    top5_accuracy_val = top_5_accuracy(logits, torch.from_numpy(targets))

    num_classes = logits.shape[1]

    # Validate label_names matches num_classes
    if label_names is not None and len(label_names) != num_classes:
        print(
            f"[WARNING] Label names count ({len(label_names)}) doesn't match model output classes ({num_classes})"
        )
        print("[WARNING] Generating generic class names instead")
        label_names = [f"Class_{i}" for i in range(num_classes)]
    elif label_names is None:
        label_names = [f"Class_{i}" for i in range(num_classes)]

    precision, recall, fscore, support = precision_recall_fscore_support(
        targets, preds, labels=list(range(num_classes)), zero_division=0
    )
    cm = confusion_matrix(targets, preds, labels=list(range(num_classes)))

    return {
        "overall_accuracy": overall_accuracy,
        "top5_accuracy": top5_accuracy_val,
        "per_class_precision": precision,
        "per_class_recall": recall,
        "per_class_fscore": fscore,
        "per_class_support": support,
        "confusion_matrix": cm,
        "label_names": label_names,
        "predictions": preds,
        "targets": targets,
        "logits": logits,
    }


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: Optional[List[str]] = None,
    save_path: str = "confusion_matrix.png",
    top_n: Optional[int] = None,
) -> List[int]:
    """
    Plot a confusion matrix heatmap, optionally showing only the most confused classes.

    Args:
        cm: Full confusion matrix (num_classes x num_classes)
        class_names: List of class names (uses indices if None)
        save_path: Path to save the confusion matrix image
        top_n: If specified, show only top N most confused classes.
               If None or >= num_classes, shows full matrix.

    Returns:
        List of class indices included in the matrix
    """
    num_classes = cm.shape[0]

    if class_names is None:
        class_names = [str(i) for i in range(num_classes)]

    # Determine if we should show focused or full matrix
    show_all = top_n is None or top_n <= 0 or top_n >= num_classes

    if show_all:
        # Full confusion matrix
        plot_cm = cm
        plot_names = class_names
        class_indices = list(range(num_classes))
        title = "Confusion Matrix"
    else:
        # Focused matrix: select top_n most confused classes
        confusion_scores = []
        for i in range(num_classes):
            # Sum of misclassifications: row (false negatives) + column (false positives)
            row_sum = cm[i, :].sum() - cm[i, i]
            col_sum = cm[:, i].sum() - cm[i, i]
            confusion_scores.append((i, row_sum + col_sum))

        # Sort by confusion score and take top N
        confusion_scores.sort(key=lambda x: x[1], reverse=True)
        class_indices = sorted([idx for idx, score in confusion_scores[:top_n]])

        # Extract submatrix
        plot_cm = cm[np.ix_(class_indices, class_indices)]
        plot_names = [class_names[i] for i in class_indices]
        title = f"Confusion Matrix (Top {top_n} Most Confused Classes)"

    # Determine figure size based on number of classes
    n_display = len(plot_names)
    fig_size = max(8, n_display * 0.5)
    plt.figure(figsize=(fig_size, fig_size))

    # Adjust annotation size based on matrix size
    annot_size = 10 if n_display <= 15 else (8 if n_display <= 25 else 6)
    annot = n_display <= 30  # Disable annotations for very large matrices

    # Create a mask for the diagonal to improve color contrast for errors
    # We want the heatmap color scale to be driven by the errors (off-diagonal),
    # not the correct predictions (diagonal) which are usually much larger.
    # However, we still want to see the numbers on the diagonal if annot=True.

    # Create a copy for plotting the heatmap colors
    heatmap_data = plot_cm.copy()
    np.fill_diagonal(heatmap_data, 0)

    # Plot heatmap using the zeroed-diagonal data for color mapping
    # But use the original data for annotations
    sns.heatmap(
        heatmap_data,
        annot=plot_cm if annot else None,
        fmt="d",
        cmap="Blues",
        xticklabels=plot_names,
        yticklabels=plot_names,
        annot_kws={"size": annot_size},
    )

    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(title)

    # Rotate labels for readability if many classes
    if n_display > 10:
        plt.xticks(rotation=45, ha="right")
        plt.yticks(rotation=0)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()

    if not show_all:
        print(f"[Evaluation] Confusion matrix shows {top_n} most confused classes")

    return class_indices


def identify_worst_confusion_pairs(
    cm: np.ndarray, top_pairs: int = 5
) -> List[Tuple[int, int, int]]:
    """Identify the worst confusion pairs from confusion matrix"""
    confusion_pairs = []
    num_classes = cm.shape[0]

    # Find off-diagonal entries with highest confusion counts
    for i in range(num_classes):
        for j in range(num_classes):
            if i != j:  # Only consider off-diagonal (misclassifications)
                confusion_count = cm[i, j]
                confusion_pairs.append((i, j, confusion_count))

    # Sort by confusion count (highest first)
    confusion_pairs.sort(key=lambda x: x[2], reverse=True)

    # Return top pairs
    return confusion_pairs[:top_pairs]


def collect_misclassified_samples(
    predictions: np.ndarray,
    targets: np.ndarray,
    true_class: int,
    predicted_class: int,
    max_samples: int = 10,
) -> List[int]:
    """Collect indices of misclassified samples for a specific confusion pair"""
    misclassified_indices = []

    for idx, (pred, target) in enumerate(zip(predictions, targets)):
        if target == true_class and pred == predicted_class:
            misclassified_indices.append(idx)

    # Limit to max_samples
    return misclassified_indices[:max_samples]


def plot_confusion_grid(
    hf_split: Any,
    misclassified_indices: List[int],
    true_class: int,
    predicted_class: int,
    label_names: List[str],
    save_path: str,
    grid_size: Optional[Tuple[int, int]] = None,
) -> bool:
    """Generate image grid for misclassified samples

    Returns:
        True if grid was generated successfully, False otherwise
    """
    num_samples = len(misclassified_indices)
    if num_samples == 0:
        print(
            f"[WARNING] No misclassified samples found for {label_names[true_class]} -> {label_names[predicted_class]}"
        )
        return False

    # Compute grid size dynamically based on number of samples
    if grid_size is None:
        cols = min(2, num_samples)  # Don't have more columns than samples
        rows = (num_samples + cols - 1) // cols if cols > 0 else 1  # Ceiling division
        rows = max(1, rows)  # At least 1 row
        cols = max(1, cols)  # At least 1 column
        grid_size = (rows, cols)

    fig, axes = plt.subplots(grid_size[0], grid_size[1], figsize=(12, 3 * grid_size[0]))
    fig.suptitle(
        f"Confusion: {label_names[true_class]} -> {label_names[predicted_class]}",
        fontsize=16,
    )

    for i, idx in enumerate(misclassified_indices):
        if i >= grid_size[0] * grid_size[1]:
            break

        row = i // grid_size[1]
        col = i % grid_size[1]
        # Handle single row/column case where axes isn't 2D
        if grid_size[0] == 1 and grid_size[1] == 1:
            ax = axes
        elif grid_size[0] == 1:
            ax = axes[col]
        elif grid_size[1] == 1:
            ax = axes[row]
        else:
            ax = axes[row, col]

        # Get original image from dataset
        sample = hf_split[idx]
        img = sample.get("image", None) or sample.get("img", None)
        if not isinstance(img, Image.Image):
            img = Image.fromarray(np.array(img))

        ax.imshow(img)
        ax.set_title(f"Sample {idx}")
        ax.axis("off")

    # Hide unused subplots
    for i in range(len(misclassified_indices), grid_size[0] * grid_size[1]):
        row = i // grid_size[1]
        col = i % grid_size[1]
        if grid_size[0] == 1 and grid_size[1] == 1:
            ax = axes
        elif grid_size[0] == 1:
            ax = axes[col]
        elif grid_size[1] == 1:
            ax = axes[row]
        else:
            ax = axes[row, col]
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    return True


def generate_error_gallery(
    results: Dict,
    hf_split,
    label_names: List[str],
    output_dir: str = "errors",
    top_pairs: int = 5,
    samples_per_pair: int = 10,
) -> Dict[str, Any]:
    """Generate error gallery with misclassified samples

    Returns:
        Dictionary with gallery generation statistics and any errors encountered
    """
    print("[Error Gallery] Generating error gallery...")

    stats = {
        "pairs_processed": 0,
        "pairs_successful": 0,
        "pairs_failed": 0,
        "errors": [],
    }

    # Create output directory
    output_path = Path(output_dir)
    try:
        output_path.mkdir(parents=True, exist_ok=True)
    except PermissionError as e:
        error_msg = f"Cannot create error gallery directory: {e}"
        print(f"[ERROR] {error_msg}")
        stats["errors"].append(error_msg)
        return stats

    # Get confusion matrix and predictions
    cm = results["confusion_matrix"]
    predictions = results["predictions"]
    targets = results["targets"]

    # Identify worst confusion pairs
    confusion_pairs = identify_worst_confusion_pairs(cm, top_pairs)

    if not confusion_pairs:
        print("[Error Gallery] No confusion pairs found")
        return stats

    # Generate gallery for each confusion pair
    gallery_config = {
        "top_pairs": top_pairs,
        "samples_per_pair": samples_per_pair,
        "confusion_pairs": [],
    }

    for pair_idx, (true_class, predicted_class, count) in enumerate(confusion_pairs):
        stats["pairs_processed"] += 1
        print(
            f"[Error Gallery] Processing confusion pair {pair_idx + 1}/{len(confusion_pairs)}: {label_names[true_class]} -> {label_names[predicted_class]} (count: {count})"
        )

        try:
            # Create subdirectory for this confusion pair
            pair_dir = output_path / f"confusion_pair_{true_class}_{predicted_class}"
            pair_dir.mkdir(exist_ok=True)

            # Collect misclassified samples
            misclassified_indices = collect_misclassified_samples(
                predictions, targets, true_class, predicted_class, samples_per_pair
            )

            if misclassified_indices:
                # Generate image grid
                grid_path = pair_dir / "grid.png"
                grid_success = plot_confusion_grid(
                    hf_split,
                    misclassified_indices,
                    true_class,
                    predicted_class,
                    label_names,
                    str(grid_path),
                )

                if not grid_success:
                    stats["pairs_failed"] += 1
                    stats["errors"].append(
                        f"Failed to generate grid for {label_names[true_class]} -> {label_names[predicted_class]}"
                    )
                    continue

                # Save sample metadata
                samples_metadata = {
                    "true_class": true_class,
                    "predicted_class": predicted_class,
                    "true_class_name": label_names[true_class],
                    "predicted_class_name": label_names[predicted_class],
                    "confusion_count": count,
                    "misclassified_indices": misclassified_indices,
                }

                with open(pair_dir / "samples.json", "w") as f:
                    json.dump(samples_metadata, f, indent=2)

                # Add to gallery config
                gallery_config["confusion_pairs"].append(
                    {
                        "true_class": true_class,
                        "predicted_class": predicted_class,
                        "true_class_name": label_names[true_class],
                        "predicted_class_name": label_names[predicted_class],
                        "confusion_count": count,
                        "num_samples_collected": len(misclassified_indices),
                    }
                )

                stats["pairs_successful"] += 1
            else:
                print(
                    f"[WARNING] No misclassified samples found for {label_names[true_class]} → {label_names[predicted_class]}"
                )
                stats["pairs_failed"] += 1

        except Exception as e:
            error_msg = f"Error processing pair {label_names[true_class]} -> {label_names[predicted_class]}: {e}"
            print(f"[ERROR] {error_msg}")
            stats["errors"].append(error_msg)
            stats["pairs_failed"] += 1
            continue

    # Save gallery configuration
    with open(output_path / "gallery_config.json", "w") as f:
        json.dump(gallery_config, f, indent=2)

    print(
        f"[Error Gallery] Complete: {stats['pairs_successful']}/{stats['pairs_processed']} pairs successful"
    )
    if stats["errors"]:
        print(
            f"[Error Gallery] {len(stats['errors'])} errors encountered (see gallery_config.json)"
        )
    print(f"[Error Gallery] Output directory: {output_dir}")

    return stats


def save_error_analysis(results: Dict, output_dir: str = "errors"):
    """Generate markdown analysis of error patterns"""
    analysis_path = Path(output_dir) / "error_analysis.md"

    cm = results["confusion_matrix"]
    label_names = results["label_names"]

    with open(analysis_path, "w") as f:
        f.write("# Error Analysis Report\n\n")
        f.write("## Overview\n\n")
        f.write(f"- Overall Accuracy: {results['overall_accuracy']:.4f}\n")
        f.write(f"- Top-5 Accuracy: {results['top5_accuracy']:.4f}\n")
        f.write(f"- Number of Classes: {len(label_names)}\n\n")

        f.write("## Worst Confusion Pairs\n\n")

        # Identify top confusion pairs
        confusion_pairs = identify_worst_confusion_pairs(
            cm, 10
        )  # Get top 10 for analysis

        for i, (true_class, predicted_class, count) in enumerate(confusion_pairs):
            if count > 0:
                f.write(
                    f"### {i+1}. {label_names[true_class]} -> {label_names[predicted_class]} (Count: {count})\n\n"
                )
                f.write(f"- **True Class**: {label_names[true_class]}\n")
                f.write(f"- **Predicted Class**: {label_names[predicted_class]}\n")
                f.write(f"- **Confusion Count**: {count}\n\n")

                # Add pattern observations placeholder
                f.write("#### Pattern Observations\n\n")
                f.write("- [ ] Visual similarities between classes\n")
                f.write("- [ ] Common misclassification patterns\n")
                f.write("- [ ] Potential data quality issues\n")
                f.write("- [ ] Model confusion patterns\n\n")

        f.write("## Recommendations\n\n")
        f.write("- Consider data augmentation for frequently confused classes\n")
        f.write("- Review class balance and dataset quality\n")
        f.write("- Evaluate model architecture for class discrimination\n")
        f.write("- Consider transfer learning or fine-tuning approaches\n")

    print(f"[Error Analysis] Analysis saved to {analysis_path}")


def list_available_configs(configs_dir: str = "configs") -> List[str]:
    """List all available config files in the configs directory"""
    configs_path = Path(configs_dir)
    if not configs_path.exists():
        return []
    return sorted([str(p) for p in configs_path.glob("*.yaml")])


def list_available_models(outputs_dir: str = "outputs") -> List[str]:
    """List all available model checkpoints"""
    outputs_path = Path(outputs_dir)
    if not outputs_path.exists():
        return []
    return sorted([str(p) for p in outputs_path.glob("*.pt")])


def validate_paths(model_path: str, config_path: str) -> Tuple[bool, List[str]]:
    """
    Validate that model and config paths exist.
    Returns (is_valid, list_of_error_messages)
    """
    errors = []

    if not os.path.exists(model_path):
        errors.append(f"Model checkpoint not found: {model_path}")
        available_models = list_available_models()
        if available_models:
            errors.append(f"Available models: {', '.join(available_models)}")
        else:
            errors.append(
                "No .pt files found in outputs/. Train a model first with: python src/train.py"
            )

    if not os.path.exists(config_path):
        errors.append(f"Config file not found: {config_path}")
        available_configs = list_available_configs()
        if available_configs:
            errors.append(f"Available configs: {', '.join(available_configs)}")

    return len(errors) == 0, errors


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate PlantDiseaseClassifier model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic evaluation (auto-detects architecture from checkpoint)
  python src/evaluate.py --model outputs/best.pt --split val

  # Works with any architecture (ConvNet, ResNet18, etc.)
  python src/evaluate.py --model outputs/resnet18_best.pt --split val

  # Quick validation without full evaluation
  python src/evaluate.py --model outputs/best.pt --dry-run

  # Override config if needed (optional for most checkpoints)
  python src/evaluate.py --model outputs/best.pt --config configs/train_quick_test.yaml

  # List available configs and models
  python src/evaluate.py --list-configs
  python src/evaluate.py --list-models
        """,
    )
    parser.add_argument(
        "--model", help="Path to model checkpoint (contains model_state)"
    )
    parser.add_argument(
        "--config",
        default="configs/train.yaml",
        help="Path to config yaml (optional - architecture auto-detected from checkpoint)",
    )
    parser.add_argument(
        "--split",
        default="val",
        help="Dataset split to evaluate: val, test, or train (default: val)",
    )
    parser.add_argument(
        "--output",
        default="outputs/eval_results.json",
        help="Path to save evaluation results",
    )
    parser.add_argument(
        "--no-error-gallery",
        dest="error_gallery",
        action="store_false",
        help="Disable error gallery generation",
    )
    parser.set_defaults(error_gallery=True)
    parser.add_argument(
        "--gallery-top-pairs",
        type=int,
        default=5,
        help="Number of worst confusion pairs to analyze",
    )
    parser.add_argument(
        "--gallery-samples-per-pair",
        type=int,
        default=10,
        help="Number of misclassified samples per confusion pair",
    )
    parser.add_argument(
        "--error-gallery-dir",
        default="errors",
        help="Directory for error gallery output (default: errors)",
    )
    parser.add_argument(
        "--quiet", "-q", action="store_true", help="Reduce output verbosity"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate setup (model, config, dataset) without running full evaluation",
    )
    parser.add_argument(
        "--cm-classes",
        type=int,
        default=15,
        metavar="N",
        help="Number of classes to show in confusion matrix (default: 15). "
        "Shows the N most confused classes. Use 0 for full matrix.",
    )
    parser.add_argument(
        "--exclude-classes",
        nargs="+",
        help="List of class names to exclude from evaluation (e.g. 'Background_without_leaves')",
    )
    parser.add_argument(
        "--list-configs",
        action="store_true",
        help="List available config files and exit",
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List available model checkpoints and exit",
    )
    args = parser.parse_args()

    # Handle --list-configs
    if args.list_configs:
        configs = list_available_configs()
        if configs:
            print("Available config files:")
            for cfg in configs:
                print(f"  {cfg}")
        else:
            print("No config files found in configs/")
        return

    # Handle --list-models
    if args.list_models:
        models = list_available_models()
        if models:
            print("Available model checkpoints:")
            for model in models:
                print(f"  {model}")
        else:
            print("No model checkpoints found in outputs/")
            print("Train a model first with: python src/train.py")
        return

    # Require --model for actual evaluation
    if not args.model:
        parser.error("--model is required (or use --list-configs/--list-models)")

    # Check if running from correct directory
    expected_markers = ["configs", "src", "outputs"]
    missing_markers = [m for m in expected_markers if not os.path.exists(m)]
    if missing_markers:
        print("\n" + "=" * 60)
        print("WARNING: You may be running from the wrong directory!")
        print("=" * 60)
        print(f"Current directory: {os.getcwd()}")
        print(f"Missing expected folders: {missing_markers}")
        print("\nMake sure to run from the project root directory:")
        print("  cd PlantDiseaseClassifier")
        print("  python src/evaluate.py --model outputs/best.pt")
        print("=" * 60 + "\n")

    # Validate paths early with helpful suggestions
    is_valid, errors = validate_paths(args.model, args.config)
    if not is_valid:
        print("\n" + "=" * 60)
        print("ERROR: Invalid paths!")
        print("=" * 60)
        for err in errors:
            print(f"  {err}")
        print("=" * 60)
        sys.exit(1)

    # Load config
    with open(args.config, "r") as f:
        cfg_dict = yaml.safe_load(f)

    # Set seed for reproducibility (same as training)
    seed = cfg_dict.get("seed", 42)
    set_seed(seed)
    print(f"[Evaluation] Using seed: {seed}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Evaluation] Using device: {device}")
    if device.type == "cuda":
        print(f"[Evaluation] GPU: {torch.cuda.get_device_name(0)}")

    # Load model
    print(f"[Evaluation] Loading model from {args.model}")
    model = load_model(args.model, args.config)
    model.to(device)

    # Load dataset with user-friendly first-run warning
    dataset_name = cfg_dict["data"]["dataset_name"]
    print(f"[Evaluation] Loading dataset: {dataset_name}")

    # Check if this might be a first-time download
    cache_dir = Path.home() / ".cache" / "huggingface" / "datasets"
    dataset_cache_exists = (
        cache_dir.exists() and any(cache_dir.iterdir()) if cache_dir.exists() else False
    )

    if not dataset_cache_exists:
        print("\n" + "=" * 60)
        print("NOTE: First-time dataset download")
        print("=" * 60)
        print(f"The {dataset_name} dataset will be downloaded and cached.")
        print("This may take several minutes depending on your connection.")
        print(f"Cache location: {cache_dir}")
        print("Subsequent runs will use the cached version.")
        print("=" * 60 + "\n")

    ds = load_dataset_robust(cfg_dict["data"]["dataset_name"])

    # Get full dataset and recreate exact splits used during training
    print(f"[DEBUG] Available dataset splits: {list(ds.keys())}")
    split_name = "train" if "train" in ds else list(ds.keys())[0]
    full = ds[split_name]

    # Smart Auto-Fix: Check for class count mismatch
    model_num_classes = get_model_num_classes(model)
    if model_num_classes > 0:
        # Get dataset classes
        temp_ds = HFDataset(full, transform=None)
        dataset_labels = temp_ds.label_names
        dataset_num_classes = len(dataset_labels) if dataset_labels else 0

        if dataset_num_classes > 0 and model_num_classes != dataset_num_classes:
            print("\n[WARNING] Class count mismatch detected!")
            print(f"  Model expects: {model_num_classes} classes")
            print(f"  Dataset has:   {dataset_num_classes} classes")

            # Heuristic: If diff is 1 and 'Background_without_leaves' exists, it's the likely culprit
            if (
                (dataset_num_classes - model_num_classes == 1)
                and ("Background_without_leaves" in dataset_labels)
                and (not args.exclude_classes)
            ):

                print(
                    "[AUTO-FIX] 'Background_without_leaves' found in dataset but likely missing from model."
                )
                print(
                    "           Automatically excluding it to match model dimensions."
                )
                print("           (Use --exclude-classes to override this behavior)")
                args.exclude_classes = ["Background_without_leaves"]
            else:
                print(
                    "[WARNING] Could not automatically resolve mismatch. Evaluation may fail or have low accuracy."
                )
                print(f"Dataset classes: {dataset_labels}")

    # Handle class exclusion if requested
    label_names_override = None
    if args.exclude_classes:
        print(f"[Data] Excluding classes: {args.exclude_classes}")
        # We need label names to map names to indices
        temp_ds = HFDataset(full, transform=None)
        all_labels = temp_ds.label_names

        exclude_indices = []
        for cls in args.exclude_classes:
            if cls in all_labels:
                exclude_indices.append(all_labels.index(cls))
            else:
                print(f"[WARNING] Class '{cls}' not found in dataset")

        if exclude_indices:
            # Create mapping from old_idx -> new_idx
            old_to_new = {}
            new_idx = 0
            keep_indices = sorted(
                list(set(range(len(all_labels))) - set(exclude_indices))
            )
            for old_idx in keep_indices:
                old_to_new[old_idx] = new_idx
                new_idx += 1

            def filter_exclude(example):
                label = example["label"] if "label" in example else example["labels"]
                return label not in exclude_indices

            def map_labels(example):
                label_key = "label" if "label" in example else "labels"
                old_label = example[label_key]
                example[label_key] = old_to_new[old_label]
                return example

            print(f"[Data] Filtering {len(exclude_indices)} classes...")
            full = full.filter(filter_exclude)
            print("[Data] Remapping labels...")
            full = full.map(map_labels)

            label_names_override = [all_labels[i] for i in keep_indices]

    # Handle subset_fraction like training does
    subset_fraction = cfg_dict["data"].get("subset_fraction", 1.0)
    if subset_fraction < 1.0:
        n = int(len(full) * subset_fraction)
        full = full.shuffle(seed=seed).select(range(n))
        print(f"[Data] Using subset: {n} samples (fraction={subset_fraction})")

    # Build labels list for stratified splitting
    labels = []
    for item in full:
        if "label" in item:
            labels.append(int(item["label"]))
        elif "labels" in item:
            labels.append(int(item["labels"]))
        else:
            raise KeyError("Sample missing 'label'/'labels' key")

    # Recreate exact splits used during training
    train_idx, val_idx, test_idx = stratified_split(
        labels,
        val_size=cfg_dict["data"].get("val_size", 0.15),
        test_size=cfg_dict["data"].get("test_size", 0.15),
        seed=seed,
    )

    # Select the requested split
    if args.split == "train":
        eval_split = full.select(train_idx.tolist())
        eval_split_name = "train"
    elif args.split == "val":
        eval_split = full.select(val_idx.tolist())
        eval_split_name = "val"
    elif args.split == "test":
        eval_split = full.select(test_idx.tolist())
        eval_split_name = "test"
    else:
        # Fallback for unknown split names: default to val
        print(f"[WARNING] Unknown split '{args.split}', using 'val' split")
        eval_split = full.select(val_idx.tolist())
        eval_split_name = "val"

    print(
        f"[Evaluation] Using '{eval_split_name}' split with {len(eval_split)} samples"
    )

    # Apply transforms (using same format as train.py - dict with 'color' key)
    # For evaluation: train=False, augment=False (wrapper handles this)
    eval_transforms = get_transforms(
        image_size=cfg_dict["data"]["image_size"],
        normalize=cfg_dict["data"]["normalize"],
        augment=False,  # No augmentation for evaluation
    )

    # Use 'color' modality like training does
    eval_tf = eval_transforms["color"]

    eval_ds = HFDataset(eval_split, transform=eval_tf, label_names=label_names_override)

    # Handle num_workers - Windows has issues with multiprocessing in DataLoader
    num_workers = cfg_dict["data"]["num_workers"]
    if sys.platform == "win32" and num_workers > 0:
        # Windows uses 'spawn' for multiprocessing which can be slower and error-prone
        # with certain configurations (pickling issues).
        # Setting num_workers=0 runs data loading in the main process.
        # This is safer but might be slightly slower as data loading won't happen in parallel.
        num_workers = 0
        if not args.quiet:
            print(
                "[INFO] Windows detected: Setting num_workers=0 for stability (single-process data loading)."
            )

    eval_loader = DataLoader(
        eval_ds,
        batch_size=cfg_dict["train"]["batch_size"],
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    # Dry-run mode: validate setup and exit
    if args.dry_run:
        print("\n" + "=" * 60)
        print("[Dry Run] Setup validation successful!")
        print("=" * 60)
        print(f"  Model:        {args.model}")
        print(f"  Config:       {args.config}")
        print(f"  Dataset:      {cfg_dict['data']['dataset_name']}")
        print(f"  Split:        {eval_split_name}")
        print(f"  Samples:      {len(eval_ds)}")
        print(
            f"  Num classes:  {len(eval_ds.label_names) if eval_ds.label_names else 'Unknown'}"
        )
        print(f"  Batch size:   {cfg_dict['train']['batch_size']}")
        print(f"  Device:       {device}")
        print(f"  Output:       {args.output}")
        print(
            f"  Error gallery: {'enabled -> ' + args.error_gallery_dir if args.error_gallery else 'disabled'}"
        )
        print(f"  CM classes:   {args.cm_classes if args.cm_classes > 0 else 'all'}")
        print("=" * 60)

        # Estimate evaluation time
        samples_per_sec_estimate = 100 if device.type == "cuda" else 20
        est_time_sec = len(eval_ds) / samples_per_sec_estimate
        if est_time_sec > 60:
            est_time_str = f"~{est_time_sec/60:.1f} minutes"
        else:
            est_time_str = f"~{est_time_sec:.0f} seconds"
        print(f"\nEstimated evaluation time: {est_time_str}")
        print("\nTo run full evaluation, remove --dry-run flag.")
        return

    # Run evaluation
    print("[Evaluation] Running evaluation...")
    results = evaluate_model(
        model,
        eval_loader,
        device,
        label_names=eval_ds.label_names,
        verbose=not args.quiet,
    )

    # Print results
    print("\n=== Evaluation Results ===")
    print(f"Overall Accuracy: {results['overall_accuracy']:.4f}")
    print(f"Top-5 Accuracy: {results['top5_accuracy']:.4f}")
    print("\nPer-class metrics:")
    if results["label_names"] and len(results["label_names"]) == len(
        results["per_class_precision"]
    ):
        for i, (p, r, f, s) in enumerate(
            zip(
                results["per_class_precision"],
                results["per_class_recall"],
                results["per_class_fscore"],
                results["per_class_support"],
            )
        ):
            class_name = results["label_names"][i]
            print(
                f"{class_name}: Precision={p:.3f}, Recall={r:.3f}, F1={f:.3f}, Support={s}"
            )
    else:
        # Fallback if label names don't match class count
        print(
            f"[WARNING] Label names ({len(results['label_names'])} names) don't match class count ({len(results['per_class_precision'])} classes)"
        )
        for i, (p, r, f, s) in enumerate(
            zip(
                results["per_class_precision"],
                results["per_class_recall"],
                results["per_class_fscore"],
                results["per_class_support"],
            )
        ):
            class_name = f"Class_{i}"
            print(
                f"{class_name}: Precision={p:.3f}, Recall={r:.3f}, F1={f:.3f}, Support={s}"
            )

    # Generate confusion matrix
    print("[Evaluation] Generating confusion matrix...")
    cm_save_path = os.path.join(
        os.path.dirname(args.output) or "outputs", "confusion_matrix.png"
    )
    cm_classes_shown = plot_confusion_matrix(
        results["confusion_matrix"],
        class_names=results["label_names"],
        save_path=cm_save_path,
        top_n=args.cm_classes,
    )

    # Show which classes are in the matrix if focused
    if args.cm_classes > 0 and args.cm_classes < len(results["label_names"]):
        top_confused = [results["label_names"][i] for i in cm_classes_shown[:5]]
        print(f"[Evaluation] Most confused classes: {top_confused}...")

    # Save results
    print(f"[Evaluation] Saving results to {args.output}")
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        # Convert numpy arrays to lists for JSON serialization
        json_results = {
            # Metadata for reproducibility
            "metadata": {
                "model_path": os.path.abspath(args.model),
                "config_path": os.path.abspath(args.config),
                "split": args.split,
                "dataset_name": cfg_dict["data"]["dataset_name"],
                "num_samples": len(eval_ds),
                "device": str(device),
                "timestamp": datetime.now().isoformat(),
            },
            # Metrics
            "overall_accuracy": float(results["overall_accuracy"]),
            "top5_accuracy": float(results["top5_accuracy"]),
            "per_class_precision": results["per_class_precision"].tolist(),
            "per_class_recall": results["per_class_recall"].tolist(),
            "per_class_fscore": results["per_class_fscore"].tolist(),
            "per_class_support": results["per_class_support"].tolist(),
            "confusion_matrix": results["confusion_matrix"].tolist(),
            "label_names": results["label_names"],
        }
        json.dump(json_results, f, indent=2)

    # Generate error gallery if requested
    if args.error_gallery:
        print("[Evaluation] Generating error gallery...")
        generate_error_gallery(
            results=results,
            hf_split=eval_split,
            label_names=results["label_names"],
            output_dir=args.error_gallery_dir,
            top_pairs=args.gallery_top_pairs,
            samples_per_pair=args.gallery_samples_per_pair,
        )

        # Generate error analysis markdown
        save_error_analysis(results, output_dir=args.error_gallery_dir)

    # ClearML integration
    task = init_task(
        enabled=cfg_dict.get("clearml", {}).get("enabled", False),
        project=cfg_dict.get("clearml", {}).get("project")
        or cfg_dict.get("project_name"),
        task_name=f"evaluation-{args.split}",
        tags=["evaluation"] + cfg_dict.get("tags", []),
        params={"model_path": args.model, "split": args.split},
    )

    if task:
        log_scalar(task, "accuracy", "overall", results["overall_accuracy"], 0)
        log_scalar(task, "accuracy", "top5", results["top5_accuracy"], 0)
        log_image(task, "confusion_matrix", cm_save_path)

        # Log error gallery images to ClearML
        if args.error_gallery:
            errors_dir = Path(args.error_gallery_dir)
            if errors_dir.exists():
                for pair_dir in errors_dir.iterdir():
                    if pair_dir.is_dir() and pair_dir.name.startswith(
                        "confusion_pair_"
                    ):
                        grid_path = pair_dir / "grid.png"
                        if grid_path.exists():
                            log_image(
                                task, f"error_gallery/{pair_dir.name}", str(grid_path)
                            )

                # Log error analysis markdown
                analysis_path = errors_dir / "error_analysis.md"
                if analysis_path.exists():
                    task.upload_artifact(
                        name="error_analysis", artifact_object=str(analysis_path)
                    )

        print("[Evaluation] Results logged to ClearML")

    print("[Evaluation] Evaluation completed successfully!")


if __name__ == "__main__":
    main()
