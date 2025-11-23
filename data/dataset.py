"""
PyTorch Dataset class for multi-modality plant disease classification.
"""

import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from PIL import Image


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

        return {
            "image": img,
            "label": torch.tensor(label, dtype=torch.long),
            "modality": modality
        }


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
