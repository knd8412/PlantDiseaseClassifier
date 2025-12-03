"""
Utility functions for dataset handling, splitting, and subset creation.
"""

import os
import zipfile
from collections import Counter
from glob import glob

import numpy as np
import torch
from sklearn.model_selection import train_test_split


def ensure_dataset_extracted(path):
    """
    Ensures that if the path points to a zip file or a directory containing a zip file,
    it is extracted. Returns the path to the directory containing the actual data.
    """
    target_path = path

    # Case 1: path is a zip file
    if os.path.isfile(path) and path.lower().endswith(".zip"):
        extract_dir = os.path.splitext(path)[0]
        if not os.path.exists(extract_dir):
            print(f"Extracting {path}...")
            with zipfile.ZipFile(path, "r") as zip_ref:
                zip_ref.extractall(extract_dir)
        target_path = extract_dir

    # Case 2: path is a directory that might contain one or more zip parts
    elif os.path.isdir(path):
        # Check if it already looks like a dataset (has 'color' folder)
        if os.path.exists(os.path.join(path, "color")):
            return path

        # Look for zip files (including chunked parts)
        files = os.listdir(path)
        zip_files = sorted([f for f in files if f.lower().endswith(".zip")])

        if zip_files:
            # Use the base name of the first zip part for extraction folder
            base_name = os.path.splitext(zip_files[0])[0].replace("_part_001", "")
            extract_dir = os.path.join(path, base_name)

            if not os.path.exists(extract_dir):
                os.makedirs(extract_dir, exist_ok=True)
                print(f"Extracting {len(zip_files)} zip parts to {extract_dir}...")
                for zip_file in zip_files:
                    zip_path = os.path.join(path, zip_file)
                    print(f"  Extracting {zip_file}...")
                    with zipfile.ZipFile(zip_path, "r") as zip_ref:
                        zip_ref.extractall(extract_dir)
            target_path = extract_dir

    # After extraction, check if the data is nested (e.g. extracted_folder/dataset_name/color)
    # We look for the 'color' folder
    for root, dirs, files in os.walk(target_path):
        if "color" in dirs:
            return root

    return target_path


def build_class_mapping(data_dir, modality="color"):
    """
    Build a mapping from class names to integer IDs.

    Args:
        data_dir: root directory containing modality subfolders
        modality: which modality folder to scan for class names (default: "color")

    Returns:
        tuple: (class_names, class_to_idx)
            - class_names: sorted list of class names
            - class_to_idx: dict mapping class name to integer ID
    """
    modality_path = os.path.join(data_dir, modality)
    class_names = sorted(next(os.walk(modality_path))[1])
    class_to_idx = {cls: i for i, cls in enumerate(class_names)}

    return class_names, class_to_idx


def gather_samples(data_dir, modalities, class_to_idx):
    """
    Gather all image samples from the dataset.

    Args:
        data_dir: root directory containing modality subfolders
        modalities: list of modality names (e.g., ["color", "grayscale", "segmented"])
        class_to_idx: dict mapping class name to integer ID

    Returns:
        list: samples as (img_path, label_id, modality_name) tuples
    """
    samples = []

    for modality in modalities:
        for cls, idx in class_to_idx.items():
            folder = os.path.join(data_dir, modality, cls)

            seen_paths = set()

            # Handle both .jpg and .JPG extensions
            for pattern in ["*.jpg", "*.JPG"]:
                for img_path in glob(os.path.join(folder, pattern)):
                    if img_path in seen_paths:
                        continue
                    seen_paths.add(img_path)
                    samples.append((img_path, idx, modality))

    return samples


def split_dataset(samples, test_size=0.15, val_size=0.18, random_state=42):
    """
    Split dataset into train, validation, and test sets.

    Args:
        samples: list of (img_path, label_id, modality_name) tuples
        test_size: proportion of data for test set (default: 0.15 = 15%)
        val_size: proportion of remaining data for validation (default: 0.18 ≈ 15% of total)
        random_state: random seed for reproducibility

    Returns:
        tuple: (train_samples, val_samples, test_samples)

    Note:
        With default values: ~70% train, ~15% val, ~15% test
    """
    # First split: separate test set
    train_val, test = train_test_split(
        samples,
        test_size=test_size,
        shuffle=True,
        stratify=[s[1] for s in samples],
        random_state=random_state,
    )

    # Second split: separate train and validation
    train, val = train_test_split(
        train_val,
        test_size=val_size,
        shuffle=True,
        stratify=[s[1] for s in train_val],
        random_state=random_state,
    )

    return train, val, test


def make_subset(samples, ratio, seed=42):
    """
    Create a stratified subset of samples.

    Useful for quick prototyping or hyperparameter tuning on smaller data.

    Args:
        samples: list of (img_path, label_id, modality_name) tuples
        ratio: proportion of samples to keep (e.g., 0.05 = 5%, 0.3 = 30%)
        seed: random seed for reproducibility

    Returns:
        list: subset of samples maintaining class distribution
    """
    subset, _ = train_test_split(
        samples, train_size=ratio, stratify=[s[1] for s in samples], random_state=seed
    )

    return subset


def get_class_distribution(samples):
    """
    Get the distribution of classes in the dataset.

    Args:
        samples: list of (img_path, label_id, modality_name) tuples

    Returns:
        Counter: class_id -> count mapping
    """
    labels = [s[1] for s in samples]
    return Counter(labels)


def balance_dataset_uniform(samples, seed=42):
    """
    Balance dataset by uniform sampling - take same number from each class.

    Uses the size of the smallest class as the target count for all classes.

    Args:
        samples: list of (img_path, label_id, modality_name) tuples
        seed: random seed for reproducibility

    Returns:
        list: balanced samples with equal number per class
    """
    np.random.seed(seed)

    # Group samples by class
    class_samples = {}
    for sample in samples:
        label = sample[1]
        if label not in class_samples:
            class_samples[label] = []
        class_samples[label].append(sample)

    # Find minimum class size
    min_count = min(len(samples_list) for samples_list in class_samples.values())

    # Randomly sample min_count from each class
    balanced = []
    for label, samples_list in class_samples.items():
        sampled = np.random.choice(len(samples_list), size=min_count, replace=False)
        balanced.extend([samples_list[i] for i in sampled])

    # Shuffle the balanced dataset
    np.random.shuffle(balanced)

    return balanced


def calculate_class_weights(samples, num_classes=None):
    """
    Calculate class weights for imbalanced datasets.

    Uses inverse frequency weighting: weight = 1 / frequency
    Normalized so the weights sum to num_classes.

    Args:
        samples: list of (img_path, label_id, modality_name) tuples
        num_classes: total number of classes (if None, inferred from samples)

    Returns:
        torch.Tensor: weight for each class (length = num_classes)
    """
    labels = [s[1] for s in samples]

    if num_classes is None:
        num_classes = max(labels) + 1

    # Count samples per class
    class_counts = Counter(labels)

    # Calculate weights: inverse frequency
    weights = torch.zeros(num_classes)
    for class_id in range(num_classes):
        count = class_counts.get(class_id, 0)
        if count > 0:
            weights[class_id] = 1.0 / count
        else:
            weights[class_id] = 0.0

    # Normalize weights so they sum to num_classes
    weights = weights / weights.sum() * num_classes

    return weights


def get_sample_weights(samples):
    """
    Calculate per-sample weights for WeightedRandomSampler.

    Each sample gets weight = 1 / (count of its class)

    Args:
        samples: list of (img_path, label_id, modality_name) tuples

    Returns:
        torch.Tensor: weight for each sample (length = len(samples))
    """
    labels = [s[1] for s in samples]
    class_counts = Counter(labels)

    # Assign weight to each sample based on its class frequency
    sample_weights = torch.tensor([1.0 / class_counts[label] for label in labels])

    return sample_weights
