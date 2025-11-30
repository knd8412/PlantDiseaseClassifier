"""
PyTorch Dataset class for multi-modality plant disease classification.
"""

import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from PIL import Image
import yaml
from clearml import Dataset
from data.transforms import get_transforms
from data.utils import build_class_mapping, gather_samples, split_dataset, ensure_dataset_extracted
from data.visualization import show_batch


class MultiModalityDataset(Dataset):
    """
    Custom PyTorch Dataset for handling multi-modality images.
    
    Args:
        samples: list of tuples (img_path, label_id, modality_name)
        modality_transforms: dict mapping {modality_name: transform}
    """
    
    def __init__(self, samples, modality_transforms):
        """
        Initialize the dataset.
        
        Args:
            samples: list of (img_path, label_id, modality_name)
            modality_transforms: dict {modality_name: transform}
        """
        self.samples = samples
        self.transforms = modality_transforms

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label, modality = self.samples[idx]

        # Load and convert image to RGB
        img = Image.open(img_path).convert("RGB")
        
        # Apply modality-specific transform
        img = self.transforms[modality](img)

        return img, torch.tensor(label, dtype=torch.long)


def create_dataloaders(train_samples, val_samples, test_samples, 
                       train_transforms, val_transforms,
                       batch_size=32, num_workers=2, 
                       use_weighted_sampling=False, sample_weights=None):
    """
    Create DataLoaders for train, validation, and test sets.
    
    Args:
        train_samples: list of training samples
        val_samples: list of validation samples
        test_samples: list of test samples
        train_transforms: transforms for training data
        val_transforms: transforms for validation/test data
        batch_size: batch size for DataLoaders
        num_workers: number of worker processes for data loading
        use_weighted_sampling: if True, use WeightedRandomSampler for training
        sample_weights: torch.Tensor of weights for each training sample
                       (required if use_weighted_sampling=True)
        
    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    train_dataset = MultiModalityDataset(train_samples, train_transforms)
    val_dataset = MultiModalityDataset(val_samples, val_transforms)
    test_dataset = MultiModalityDataset(test_samples, val_transforms)
    
    # Create sampler for weighted sampling if requested
    sampler = None
    shuffle = True
    
    if use_weighted_sampling:
        if sample_weights is None:
            raise ValueError("sample_weights must be provided when use_weighted_sampling=True")
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True
        )
        shuffle = False  # Can't use shuffle with sampler
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers
    )
    
    return train_loader, val_loader, test_loader



def load_dataset_and_dataloaders(
    dataset_size="medium",
    config_path="configs/train.yaml"
):
    """
    Full pipeline:
      - Load config
      - Select ClearML ID based on dataset_size argument
      - Load ClearML dataset
      - Extract if needed
      - Build class mappings
      - Gather samples
      - Split train/val/test based on YAML config proportions
      - Build transforms using get_transforms()
      - Create DataLoaders
    """

    # ----------------------------------------------------
    # 1. Load configuration
    # ----------------------------------------------------
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    data_cfg = cfg["data"]
    ds_cfg = cfg["dataset"]
    train_cfg = cfg["train"]

    batch_size = train_cfg["batch_size"]
    image_size = data_cfg["image_size"]
    modalities = data_cfg["modalities"]
    normalize = data_cfg["normalize"]
    augment = data_cfg["augment"]
    val_size = data_cfg["val_size"]
    test_size = data_cfg["test_size"]
    num_workers = data_cfg["num_workers"]

    # ----------------------------------------------------
    # 2. Validate dataset_size arg + pick ClearML id
    # ----------------------------------------------------
    if dataset_size not in ds_cfg["ids"]:
        raise ValueError(
            f"Invalid dataset_size '{dataset_size}'. "
            f"Choose from: {list(ds_cfg['ids'].keys())}"
        )

    dataset_id = ds_cfg["ids"][dataset_size]
    print(f"📦 Loading ClearML dataset: {dataset_size} ({dataset_id})")

    # ----------------------------------------------------
    # 3. Download / cache
    # ----------------------------------------------------
    dataset = Dataset.get(dataset_id)
    local_path = dataset.get_local_copy()

    if ds_cfg.get("auto_extract", True):
        local_path = ensure_dataset_extracted(local_path)

    print(f"Dataset ready at: {local_path}")

    # ----------------------------------------------------
    # 4. Build class mapping
    # ----------------------------------------------------
    class_names, class_to_idx = build_class_mapping(local_path, modality="color")
    print(f"Found {len(class_names)} classes")

    # ----------------------------------------------------
    # 5. Gather samples
    # ----------------------------------------------------
    samples = gather_samples(local_path, modalities, class_to_idx)
    print(f"Total samples: {len(samples)}")

    # ----------------------------------------------------
    # 6. Train/Val/Test split
    # ----------------------------------------------------
    train_s, val_s, test_s = split_dataset(
        samples,
        val_size=val_size,
        test_size=test_size,
        random_state=cfg["seed"],
    )

    print(f"Split → Train={len(train_s)}, Val={len(val_s)}, Test={len(test_s)}")

    # ----------------------------------------------------
    # 7. Transforms (using your get_transforms)
    # ----------------------------------------------------
    train_transforms = get_transforms(
        image_size=image_size,
        train=True,
        normalize=normalize,
        augment=augment
    )

    val_transforms = get_transforms(
        image_size=image_size,
        train=False,
        normalize=normalize,
        augment=False
    )

    # ----------------------------------------------------
    # 8. Build DataLoaders
    # ----------------------------------------------------
    train_loader, val_loader, test_loader = create_dataloaders(
        train_s,
        val_s,
        test_s,
        train_transforms,
        val_transforms,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    print("✅ DataLoaders created successfully.")

    return train_loader, val_loader, test_loader, class_names
