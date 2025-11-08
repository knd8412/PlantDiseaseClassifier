"""
Sanity tests for PlantDiseaseClassifier repository quality

Implements the two sanity tests specified in 72_hours.md:
1. Data splitting reproducibility test
2. Transform shapes validation test
"""

import pytest
import numpy as np
import torch
from sklearn.model_selection import StratifiedShuffleSplit

# Import the functions we need to test
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.train import stratified_split
from src.data.transforms import get_transforms


class TestDataSplittingReproducibility:
    """Test that data splitting produces reproducible results with the same seed"""
    
    def test_stratified_split_reproducibility(self):
        """Test that stratified_split produces identical splits with same seed"""
        # Create synthetic dataset with 100 samples and 5 classes
        n_samples = 100
        n_classes = 5
        labels = np.random.randint(0, n_classes, n_samples)
        
        # Generate splits with same seed - should be identical
        seed = 42
        train_idx1, val_idx1, test_idx1 = stratified_split(None, labels, val_size=0.15, test_size=0.15, seed=seed)
        train_idx2, val_idx2, test_idx2 = stratified_split(None, labels, val_size=0.15, test_size=0.15, seed=seed)
        
        # Assert splits are identical
        assert np.array_equal(train_idx1, train_idx2), "Train splits should be identical with same seed"
        assert np.array_equal(val_idx1, val_idx2), "Validation splits should be identical with same seed"
        assert np.array_equal(test_idx1, test_idx2), "Test splits should be identical with same seed"
        
        # Verify split sizes are reasonable (allow for rounding differences)
        total_size = len(labels)
        expected_test_size = int(total_size * 0.15)
        expected_val_size = int(total_size * 0.15)  # Simplified calculation
        
        # Allow ±1 sample difference due to rounding
        assert abs(len(test_idx1) - expected_test_size) <= 1, f"Test split should have {expected_test_size}±1 samples"
        assert abs(len(val_idx1) - expected_val_size) <= 1, f"Validation split should have {expected_val_size}±1 samples"
        assert abs(len(train_idx1) - (total_size - len(test_idx1) - len(val_idx1))) <= 1, "Train split should account for remaining samples"
    
    def test_stratified_split_different_seeds(self):
        """Test that different seeds produce different splits"""
        # Create synthetic dataset
        n_samples = 100
        n_classes = 5
        labels = np.random.randint(0, n_classes, n_samples)
        
        # Generate splits with different seeds - should be different
        train_idx1, val_idx1, test_idx1 = stratified_split(None, labels, val_size=0.15, test_size=0.15, seed=42)
        train_idx2, val_idx2, test_idx2 = stratified_split(None, labels, val_size=0.15, test_size=0.15, seed=123)
        
        # Assert splits are different (not all indices the same)
        assert not np.array_equal(train_idx1, train_idx2), "Train splits should differ with different seeds"
        assert not np.array_equal(val_idx1, val_idx2), "Validation splits should differ with different seeds"
        assert not np.array_equal(test_idx1, test_idx2), "Test splits should differ with different seeds"
    
    def test_stratified_split_class_distribution(self):
        """Test that stratified splitting maintains class distribution"""
        # Create synthetic dataset with known class distribution
        n_samples_per_class = 20
        n_classes = 5
        labels = np.array([i for i in range(n_classes) for _ in range(n_samples_per_class)])
        
        train_idx, val_idx, test_idx = stratified_split(None, labels, val_size=0.2, test_size=0.2, seed=42)
        
        # Check that each split maintains roughly the same class distribution
        for idx in [train_idx, val_idx, test_idx]:
            split_labels = labels[idx]
            class_counts = np.bincount(split_labels, minlength=n_classes)
            
            # Each class should have proportional representation
            expected_min = (len(split_labels) / n_classes) * 0.8  # Allow 20% variation
            expected_max = (len(split_labels) / n_classes) * 1.2
            
            for count in class_counts:
                assert count >= expected_min and count <= expected_max, \
                    f"Class distribution should be roughly balanced in splits"


class TestTransformShapesValidation:
    """Test that transforms produce consistent tensor shapes"""
    
    def test_train_transform_shapes(self):
        """Test that training transforms produce expected tensor shapes"""
        img_size = 256
        train_transform, eval_transform = get_transforms(img_size=img_size, normalize=True, augment=True)
        
        # Create a mock PIL image
        from PIL import Image
        mock_image = Image.new('RGB', (512, 512), color='red')
        
        # Apply transform and check shape
        transformed = train_transform(mock_image)
        
        assert isinstance(transformed, torch.Tensor), "Transform should return a torch.Tensor"
        assert transformed.shape == (3, img_size, img_size), f"Expected shape (3, {img_size}, {img_size})"
        assert transformed.dtype == torch.float32, "Tensor should be float32"
        
        # Check normalization range
        assert transformed.min() >= -3.0 and transformed.max() <= 3.0, "Normalized values should be in reasonable range"
    
    def test_eval_transform_shapes(self):
        """Test that evaluation transforms produce expected tensor shapes"""
        img_size = 256
        train_transform, eval_transform = get_transforms(img_size=img_size, normalize=True, augment=False)
        
        # Create a mock PIL image
        from PIL import Image
        mock_image = Image.new('RGB', (300, 400), color='blue')  # Different aspect ratio
        
        # Apply transform and check shape
        transformed = eval_transform(mock_image)
        
        assert isinstance(transformed, torch.Tensor), "Transform should return a torch.Tensor"
        assert transformed.shape == (3, img_size, img_size), f"Expected shape (3, {img_size}, {img_size})"
        assert transformed.dtype == torch.float32, "Tensor should be float32"
    
    def test_transform_without_normalization(self):
        """Test transforms without normalization"""
        img_size = 128
        train_transform, eval_transform = get_transforms(img_size=img_size, normalize=False, augment=False)
        
        from PIL import Image
        mock_image = Image.new('RGB', (200, 200), color='green')
        
        transformed = eval_transform(mock_image)
        
        assert transformed.shape == (3, img_size, img_size), f"Expected shape (3, {img_size}, {img_size})"
        # Without normalization, values should be in [0, 1] range
        assert transformed.min() >= 0.0 and transformed.max() <= 1.0, "Without normalization, values should be in [0, 1]"
    
    def test_transform_consistency(self):
        """Test that transforms are consistent across multiple applications"""
        img_size = 256
        train_transform, eval_transform = get_transforms(img_size=img_size, normalize=True, augment=False)
        
        from PIL import Image
        mock_image = Image.new('RGB', (256, 256), color='yellow')
        
        # Apply transform multiple times
        transformed1 = eval_transform(mock_image)
        transformed2 = eval_transform(mock_image)
        
        # Should be identical (deterministic when augment=False)
        assert torch.allclose(transformed1, transformed2), "Transforms should be deterministic when augment=False"


if __name__ == "__main__":
    # Run the tests
    pytest.main([__file__, "-v"])