"""
Data module for PlantVillage dataset handling.
"""

from .dataset import MultiModalityDataset, create_dataloaders
from .transforms import get_transforms
from .utils import (
    balance_dataset_uniform,
    build_class_mapping,
    calculate_class_weights,
    gather_samples,
    get_class_distribution,
    get_sample_weights,
    make_subset,
    split_dataset,
)
from .visualization import (
    compare_augmentations,
    plot_class_distribution,
    plot_modality_distribution,
    plot_split_distribution,
    show_batch,
    visualize_sample_images,
)

__all__ = [
    # Dataset & DataLoaders
    "MultiModalityDataset",
    "create_dataloaders",
    # Transforms
    "get_transforms",
    # Dataset utilities
    "build_class_mapping",
    "gather_samples",
    "split_dataset",
    "make_subset",
    # Class imbalance handling
    "get_class_distribution",
    "balance_dataset_uniform",
    "calculate_class_weights",
    "get_sample_weights",
    # Visualization
    "show_batch",
    "plot_class_distribution",
    "plot_split_distribution",
    "plot_modality_distribution",
    "compare_augmentations",
    "visualize_sample_images",
]
