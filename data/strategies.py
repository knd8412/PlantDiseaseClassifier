"""
Strategy pattern for handling class imbalance in datasets.
Provides automated handling of different balancing strategies.
"""

import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from data.utils import (
    balance_dataset_uniform,
    get_sample_weights,
    calculate_class_weights
)
from data.dataset import MultiModalityDataset


class ImbalanceStrategy:
    """Base class for imbalance handling strategies."""
    
    def __init__(self, config=None):
        """
        Initialize strategy with optional config.
        
        Args:
            config: dict with strategy-specific parameters
        """
        self.config = config or {}
    
    def apply_to_samples(self, samples):
        """
        Apply strategy to sample list (for sampling-based strategies).
        
        Args:
            samples: list of (img_path, label_id, modality_name) tuples
            
        Returns:
            Modified samples list
        """
        return samples
    
    def create_dataloader(self, dataset, batch_size, shuffle=True, num_workers=2):
        """
        Create DataLoader with strategy applied.
        
        Args:
            dataset: PyTorch Dataset
            batch_size: batch size
            shuffle: whether to shuffle (ignored if using sampler)
            num_workers: number of worker processes
            
        Returns:
            DataLoader with strategy applied
        """
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers
        )
    
    def get_loss_weights(self, samples, num_classes):
        """
        Get class weights for loss function (for loss-weighting strategies).
        
        Args:
            samples: list of (img_path, label_id, modality_name) tuples
            num_classes: total number of classes
            
        Returns:
            torch.Tensor of class weights or None
        """
        return None


class NoBalancingStrategy(ImbalanceStrategy):
    """No balancing - use dataset as-is."""
    
    def __repr__(self):
        return "NoBalancingStrategy()"


class UniformSamplingStrategy(ImbalanceStrategy):
    """Downsample all classes to smallest class size."""
    
    def __init__(self, config=None):
        super().__init__(config)
        self.seed = self.config.get('seed', 42)
    
    def apply_to_samples(self, samples):
        """Downsample to uniform class distribution."""
        return balance_dataset_uniform(samples, seed=self.seed)
    
    def __repr__(self):
        return f"UniformSamplingStrategy(seed={self.seed})"


class WeightedSamplerStrategy(ImbalanceStrategy):
    """Use PyTorch's WeightedRandomSampler to oversample rare classes."""
    
    def __init__(self, config=None):
        super().__init__(config)
        self.replacement = self.config.get('replacement', True)
    
    def create_dataloader(self, dataset, batch_size, shuffle=True, num_workers=2):
        """Create DataLoader with WeightedRandomSampler."""
        # Get samples from dataset
        samples = dataset.samples
        sample_weights = get_sample_weights(samples)
        
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=self.replacement
        )
        
        return DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,  # Can't use shuffle with sampler
            num_workers=num_workers
        )
    
    def __repr__(self):
        return f"WeightedSamplerStrategy(replacement={self.replacement})"


class WeightedLossStrategy(ImbalanceStrategy):
    """Use class weights in loss function."""
    
    def __init__(self, config=None):
        super().__init__(config)
        self.weight_power = self.config.get('weight_power', 1.0)
    
    def get_loss_weights(self, samples, num_classes):
        """Calculate class weights for loss function."""
        weights = calculate_class_weights(samples, num_classes)
        
        # Apply power scaling if specified (for fine-tuning weight strength)
        if self.weight_power != 1.0:
            weights = torch.pow(weights, self.weight_power)
            # Re-normalize
            weights = weights / weights.sum() * num_classes
        
        return weights
    
    def __repr__(self):
        return f"WeightedLossStrategy(weight_power={self.weight_power})"


class HybridStrategy(ImbalanceStrategy):
    """Combine sampling and loss weighting strategies."""
    
    def __init__(self, config=None):
        super().__init__(config)
        self.sampling_strategy = self.config.get('sampling_strategy', 'none')
        self.use_weighted_loss = self.config.get('use_weighted_loss', True)
        self.seed = self.config.get('seed', 42)
        self.weight_power = self.config.get('weight_power', 0.5)
    
    def apply_to_samples(self, samples):
        """Apply sampling strategy if specified."""
        if self.sampling_strategy == 'uniform':
            return balance_dataset_uniform(samples, seed=self.seed)
        return samples
    
    def create_dataloader(self, dataset, batch_size, shuffle=True, num_workers=2):
        """Create DataLoader with weighted sampler if specified."""
        if self.sampling_strategy == 'weighted':
            samples = dataset.samples
            sample_weights = get_sample_weights(samples)
            
            sampler = WeightedRandomSampler(
                weights=sample_weights,
                num_samples=len(sample_weights),
                replacement=True
            )
            
            return DataLoader(
                dataset,
                batch_size=batch_size,
                sampler=sampler,
                num_workers=num_workers
            )
        
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers
        )
    
    def get_loss_weights(self, samples, num_classes):
        """Get loss weights if enabled."""
        if self.use_weighted_loss:
            weights = calculate_class_weights(samples, num_classes)
            if self.weight_power != 1.0:
                weights = torch.pow(weights, self.weight_power)
                weights = weights / weights.sum() * num_classes
            return weights
        return None
    
    def __repr__(self):
        return f"HybridStrategy(sampling={self.sampling_strategy}, weighted_loss={self.use_weighted_loss})"


def get_strategy(strategy_name, config=None):
    """
    Factory function to get imbalance strategy by name.
    
    Args:
        strategy_name: name of strategy ('none', 'uniform', 'weighted_sampler', 
                      'weighted_loss', 'hybrid')
        config: dict with strategy-specific parameters
        
    Returns:
        ImbalanceStrategy instance
        
    Example:
        >>> strategy = get_strategy('weighted_loss', {'weight_power': 0.8})
        >>> loss_weights = strategy.get_loss_weights(train_samples, num_classes)
    """
    strategies = {
        'none': NoBalancingStrategy,
        None: NoBalancingStrategy,
        'uniform': UniformSamplingStrategy,
        'weighted_sampler': WeightedSamplerStrategy,
        'weighted_loss': WeightedLossStrategy,
        'hybrid': HybridStrategy
    }
    
    if strategy_name not in strategies:
        raise ValueError(
            f"Unknown strategy: {strategy_name}. "
            f"Available: {list(strategies.keys())}"
        )
    
    return strategies[strategy_name](config)


def apply_imbalance_strategy(
    samples,
    transforms,
    batch_size,
    num_classes,
    strategy_name='none',
    strategy_config=None,
    num_workers=2,
    shuffle=True
):
    """
    One-stop function to apply imbalance strategy and create DataLoader.
    
    Args:
        samples: list of (img_path, label_id, modality_name) tuples
        transforms: dict of modality-specific transforms
        batch_size: batch size for DataLoader
        num_classes: total number of classes
        strategy_name: name of strategy to use
        strategy_config: dict with strategy parameters
        num_workers: number of DataLoader workers
        shuffle: whether to shuffle (ignored for weighted sampler)
        
    Returns:
        tuple: (dataloader, loss_weights, info_dict)
            - dataloader: PyTorch DataLoader
            - loss_weights: torch.Tensor or None
            - info_dict: dict with strategy info
            
    Example:
        >>> train_loader, loss_weights, info = apply_imbalance_strategy(
        ...     train_samples,
        ...     train_transforms,
        ...     batch_size=32,
        ...     num_classes=38,
        ...     strategy_name='weighted_loss',
        ...     strategy_config={'weight_power': 0.8}
        ... )
    """
    # Get strategy
    strategy = get_strategy(strategy_name, strategy_config)
    
    # Apply sampling strategy (modifies samples if needed)
    original_count = len(samples)
    samples = strategy.apply_to_samples(samples)
    modified_count = len(samples)
    
    # Create dataset
    from data.dataset import MultiModalityDataset
    dataset = MultiModalityDataset(samples, transforms)
    
    # Create DataLoader with strategy
    dataloader = strategy.create_dataloader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers
    )
    
    # Get loss weights
    loss_weights = strategy.get_loss_weights(samples, num_classes)
    
    # Prepare info
    info = {
        'strategy': str(strategy),
        'original_samples': original_count,
        'final_samples': modified_count,
        'samples_removed': original_count - modified_count,
        'using_weighted_sampler': isinstance(strategy, WeightedSamplerStrategy) or 
                                  (isinstance(strategy, HybridStrategy) and 
                                   strategy.sampling_strategy == 'weighted'),
        'using_weighted_loss': loss_weights is not None
    }
    
    return dataloader, loss_weights, info
