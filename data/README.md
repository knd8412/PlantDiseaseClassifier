# Data Management Module

This folder contains the data handling pipeline for the Plant Disease Classifier project. It provides a robust set of tools for loading, processing, transforming, and visualizing multi-modality plant disease datasets.

## Components

### 1. Dataset Loading (`dataset.py`)
The core of the data pipeline, responsible for:
- **`MultiModalityDataset`**: A custom PyTorch `Dataset` class that handles loading images and applying modality-specific transforms on the fly.
- **`load_dataset_and_dataloaders`**: An end-to-end pipeline that:
  - Downloads datasets from ClearML (or uses cached copies).
  - Automatically extracts zip files if needed.
  - Builds class mappings.
  - Performs stratified train/validation/test splits.
  - Creates PyTorch `DataLoader` instances.

### 2. Data Transforms (`transforms.py`)
Defines image transformations and augmentations using `torchvision`:
- **Modality-specific pipelines**: tailored transforms for `color`, `grayscale`, and `segmented` images.
- **Augmentation**: Includes random rotation, horizontal flip, and color jitter (for color images) during training.
- **Normalization**: Applies ImageNet statistics for color images and generic statistics for others.

### 3. Utilities (`utils.py`)
A collection of helper functions for dataset management:
- **File Management**: `ensure_dataset_extracted` handles zip extraction and directory structure validation.
- **Data Gathering**: `gather_samples` and `build_class_mapping` scan directories to index the dataset.
- **Splitting**: `split_dataset` performs stratified splitting into train, validation, and test sets.
- **Balancing**: `balance_dataset_uniform` and `calculate_class_weights` help handle class imbalance.

### 4. Visualization (`visualization.py`)
Tools for inspecting the dataset and debugging:
- **`show_batch`**: Displays a grid of images from a DataLoader with labels.
- **`plot_class_distribution`**: Visualizes the number of samples per class to identify imbalance.
- **`plot_split_distribution`**: Compares class distributions across train/val/test splits.
- **`compare_augmentations`**: Side-by-side comparison of original vs. augmented images.

## Usage Example

To load the dataset and get dataloaders in your training script:

```python
from data.dataset import load_dataset_and_dataloaders

train_loader, val_loader, test_loader, class_names = load_dataset_and_dataloaders(
    dataset_size="medium",
    config_path="configs/train.yaml"
)
```

To visualize a batch of data:

```python
from data.visualization import show_batch

show_batch(train_loader, class_names)
```
