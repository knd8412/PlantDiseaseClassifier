"""
Utility functions for dataset handling, splitting, and subset creation.
"""

import os
from glob import glob
from sklearn.model_selection import train_test_split


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
            
            # Handle both .jpg and .JPG extensions
            for pattern in ["*.jpg", "*.JPG"]:
                for img_path in glob(os.path.join(folder, pattern)):
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
        random_state=random_state
    )
    
    # Second split: separate train and validation
    train, val = train_test_split(
        train_val, 
        test_size=val_size, 
        shuffle=True, 
        stratify=[s[1] for s in train_val],
        random_state=random_state
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
        samples,
        train_size=ratio,
        stratify=[s[1] for s in samples],
        random_state=seed
    )
    
    return subset
