"""
Visualization utilities for dataset inspection and debugging.
"""

import matplotlib.pyplot as plt
import numpy as np
import torch
from collections import Counter


def denormalize_image(img_tensor, mean, std):
    """
    Denormalize a tensor image with mean and std.
    
    Args:
        img_tensor: normalized image tensor (C, H, W)
        mean: mean used for normalization
        std: std used for normalization
        
    Returns:
        numpy array: denormalized image (H, W, C) in [0, 1] range
    """
    mean = torch.tensor(mean).view(3, 1, 1)
    std = torch.tensor(std).view(3, 1, 1)
    
    # Denormalize
    img = img_tensor * std + mean
    
    # Clip to [0, 1] and convert to numpy
    img = torch.clamp(img, 0, 1)
    img = img.permute(1, 2, 0).cpu().numpy()
    
    return img


def show_batch(dataloader, class_names, num_images=8, denorm=True):
    """
    Display a batch of images with their labels.
    
    Args:
        dataloader: PyTorch DataLoader
        class_names: list of class names
        num_images: number of images to display (default: 8)
        denorm: whether to denormalize images (default: True)
    """
    batch = next(iter(dataloader))
    images = batch['image'][:num_images]
    labels = batch['label'][:num_images]
    modalities = batch.get('modality', ['unknown'] * num_images)[:num_images]
    
    # Determine grid size
    cols = 4
    rows = (num_images + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(12, 3 * rows))
    axes = axes.flatten() if num_images > 1 else [axes]
    
    # ImageNet normalization (used for color images)
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]
    
    for idx in range(len(axes)):
        ax = axes[idx]
        
        if idx < len(images):
            img = images[idx]
            
            # Denormalize if requested
            if denorm:
                # Try ImageNet normalization first
                img = denormalize_image(img, IMAGENET_MEAN, IMAGENET_STD)
            else:
                img = img.permute(1, 2, 0).cpu().numpy()
                img = np.clip(img, 0, 1)
            
            ax.imshow(img)
            
            label_name = class_names[labels[idx].item()]
            modality = modalities[idx] if isinstance(modalities[idx], str) else modalities[idx]
            ax.set_title(f"{label_name}\n({modality})", fontsize=9)
            ax.axis('off')
        else:
            ax.axis('off')
    
    plt.tight_layout()
    plt.show()


def plot_class_distribution(samples, class_names, title="Class Distribution"):
    """
    Plot bar chart of class distribution.
    
    Args:
        samples: list of (img_path, label_id, modality_name) tuples
        class_names: list of class names
        title: plot title
    """
    labels = [s[1] for s in samples]
    counts = Counter(labels)
    
    # Sort by class ID
    sorted_counts = [counts.get(i, 0) for i in range(len(class_names))]
    
    plt.figure(figsize=(15, 5))
    bars = plt.bar(range(len(class_names)), sorted_counts, color='steelblue', alpha=0.7)
    
    # Highlight min and max
    min_idx = np.argmin(sorted_counts)
    max_idx = np.argmax(sorted_counts)
    bars[min_idx].set_color('red')
    bars[max_idx].set_color('green')
    
    plt.xlabel('Class', fontsize=12)
    plt.ylabel('Number of Samples', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xticks(range(len(class_names)), class_names, rotation=90, fontsize=8)
    plt.grid(axis='y', alpha=0.3)
    
    # Add statistics
    plt.text(0.02, 0.98, 
             f"Min: {min(sorted_counts)} (red)\nMax: {max(sorted_counts)} (green)\n"
             f"Mean: {np.mean(sorted_counts):.1f}\nImbalance: {max(sorted_counts)/min(sorted_counts):.2f}x",
             transform=plt.gca().transAxes,
             fontsize=10,
             verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.show()


def plot_split_distribution(train, val, test, class_names):
    """
    Plot class distribution across train, validation, and test splits.
    
    Args:
        train: list of training samples
        val: list of validation samples
        test: list of test samples
        class_names: list of class names
    """
    train_labels = [s[1] for s in train]
    val_labels = [s[1] for s in val]
    test_labels = [s[1] for s in test]
    
    train_counts = Counter(train_labels)
    val_counts = Counter(val_labels)
    test_counts = Counter(test_labels)
    
    # Prepare data
    num_classes = len(class_names)
    train_dist = [train_counts.get(i, 0) for i in range(num_classes)]
    val_dist = [val_counts.get(i, 0) for i in range(num_classes)]
    test_dist = [test_counts.get(i, 0) for i in range(num_classes)]
    
    # Plot
    x = np.arange(num_classes)
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(15, 5))
    
    ax.bar(x - width, train_dist, width, label='Train', alpha=0.8)
    ax.bar(x, val_dist, width, label='Val', alpha=0.8)
    ax.bar(x + width, test_dist, width, label='Test', alpha=0.8)
    
    ax.set_xlabel('Class')
    ax.set_ylabel('Number of Samples')
    ax.set_title('Class Distribution Across Splits')
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=90, fontsize=8)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def plot_modality_distribution(samples, modalities):
    """
    Plot distribution of samples across different modalities.
    
    Args:
        samples: list of (img_path, label_id, modality_name) tuples
        modalities: list of modality names
    """
    modality_labels = [s[2] for s in samples]
    modality_counts = Counter(modality_labels)
    
    # Sort by modality order
    counts = [modality_counts.get(m, 0) for m in modalities]
    
    plt.figure(figsize=(8, 5))
    bars = plt.bar(modalities, counts, color=['#1f77b4', '#ff7f0e', '#2ca02c'], alpha=0.7)
    
    plt.xlabel('Modality', fontsize=12)
    plt.ylabel('Number of Samples', fontsize=12)
    plt.title('Sample Distribution by Modality', fontsize=14, fontweight='bold')
    plt.grid(axis='y', alpha=0.3)
    
    # Add count labels on bars
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{count:,}',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    plt.show()


def compare_augmentations(dataset_original, dataset_augmented, class_names, idx=0):
    """
    Compare original and augmented versions of the same image.
    
    Args:
        dataset_original: dataset without augmentation
        dataset_augmented: dataset with augmentation
        class_names: list of class names
        idx: index of the sample to visualize
    """
    # Get multiple augmented versions
    num_versions = 5
    
    fig, axes = plt.subplots(2, num_versions, figsize=(15, 6))
    
    # Original image
    orig_sample = dataset_original[idx]
    orig_img = orig_sample['image']
    label = orig_sample['label'].item()
    
    # ImageNet normalization
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]
    
    # Show original multiple times for comparison
    for col in range(num_versions):
        img_denorm = denormalize_image(orig_img, IMAGENET_MEAN, IMAGENET_STD)
        axes[0, col].imshow(img_denorm)
        axes[0, col].set_title('Original' if col == 0 else '', fontsize=10)
        axes[0, col].axis('off')
    
    # Show augmented versions
    for col in range(num_versions):
        aug_sample = dataset_augmented[idx]
        aug_img = aug_sample['image']
        img_denorm = denormalize_image(aug_img, IMAGENET_MEAN, IMAGENET_STD)
        axes[1, col].imshow(img_denorm)
        axes[1, col].set_title(f'Augmented {col+1}' if col == 0 else '', fontsize=10)
        axes[1, col].axis('off')
    
    fig.suptitle(f'Augmentation Comparison: {class_names[label]}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()


def visualize_sample_images(samples, class_names, num_classes_to_show=5, samples_per_class=3):
    """
    Show sample images from multiple classes.
    
    Args:
        samples: list of (img_path, label_id, modality_name) tuples
        class_names: list of class names
        num_classes_to_show: number of classes to visualize
        samples_per_class: number of samples per class
    """
    from PIL import Image
    
    # Group samples by class
    class_samples = {}
    for sample in samples:
        label = sample[1]
        if label not in class_samples:
            class_samples[label] = []
        class_samples[label].append(sample)
    
    # Select classes to show
    classes_to_show = sorted(class_samples.keys())[:num_classes_to_show]
    
    fig, axes = plt.subplots(num_classes_to_show, samples_per_class, 
                            figsize=(samples_per_class * 3, num_classes_to_show * 3))
    
    for row, class_id in enumerate(classes_to_show):
        samples_for_class = class_samples[class_id][:samples_per_class]
        
        for col, sample in enumerate(samples_for_class):
            img_path = sample[0]
            img = Image.open(img_path).convert('RGB')
            
            if num_classes_to_show == 1:
                ax = axes[col]
            else:
                ax = axes[row, col]
            
            ax.imshow(img)
            if col == 0:
                ax.set_ylabel(class_names[class_id], fontsize=10, fontweight='bold')
            ax.axis('off')
    
    plt.suptitle('Sample Images from Dataset', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()
