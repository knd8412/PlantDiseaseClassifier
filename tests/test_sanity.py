"""
Sanity tests for PlantDiseaseClassifier repository quality

Implements two sanity tests:
1. Data splitting reproducibility and stratification using data.utils.split_dataset
2. Transform shapes validation using data.transforms.get_transforms
"""

import numpy as np
import pytest
import torch

from data.transforms import get_transforms
from data.utils import split_dataset


class TestDataSplittingReproducibility:
    """Test that data splitting is reproducible and roughly preserves class distribution"""

    def _make_synthetic_samples(self, n_samples=100, n_classes=5):
        """
        Create a fake samples list matching your real format:
        (img_path, label_id, modality_name)
        """
        rng = np.random.default_rng(0)
        labels = rng.integers(0, n_classes, size=n_samples)

        samples = []
        for i, y in enumerate(labels):
            path = f"/fake/path/img_{i}.jpg"
            modality = "color"
            samples.append((path, int(y), modality))
        return samples, labels

    def test_split_reproducibility_same_seed(self):
        samples, _ = self._make_synthetic_samples(n_samples=120, n_classes=4)

        train1, val1, test1 = split_dataset(
            samples, test_size=0.15, val_size=0.18, random_state=42
        )
        train2, val2, test2 = split_dataset(
            samples, test_size=0.15, val_size=0.18, random_state=42
        )

        # Because random_state is fixed, splits should be identical
        assert train1 == train2
        assert val1 == val2
        assert test1 == test2

        total = len(samples)
        assert len(train1) + len(val1) + len(test1) == total

    def test_split_differs_different_seeds(self):
        samples, _ = self._make_synthetic_samples(n_samples=120, n_classes=4)

        train1, val1, test1 = split_dataset(
            samples, test_size=0.15, val_size=0.18, random_state=42
        )
        train2, val2, test2 = split_dataset(
            samples, test_size=0.15, val_size=0.18, random_state=123
        )

        # They don't have to be completely disjoint, just not identical
        assert train1 != train2 or val1 != val2 or test1 != test2

    def test_split_preserves_class_distribution(self):
        """
        Check that the class distribution is roughly preserved
        across train/val/test for an imbalanced dataset.
        """
        n_classes = 5
        counts = [50, 30, 10, 5, 5]  # highly imbalanced
        labels = []
        for cls, c in enumerate(counts):
            labels.extend([cls] * c)
        labels = np.array(labels)

        samples = []
        for i, y in enumerate(labels):
            samples.append((f"/fake/img_{i}.jpg", int(y), "color"))

        train, val, test = split_dataset(
            samples, test_size=0.2, val_size=0.2, random_state=42
        )

        def get_label_array(split):
            return np.array([s[1] for s in split])

        for split in [train, val, test]:
            split_labels = get_label_array(split)
            split_counts = np.bincount(split_labels, minlength=n_classes)

            full_counts = np.bincount(labels, minlength=n_classes)
            full_props = full_counts / len(labels)
            split_props = split_counts / len(split_labels)

            for p_full, p_split in zip(full_props, split_props):
                # Skip classes that don't exist in the full dataset (shouldn't happen here)
                if p_full == 0:
                    continue

                # We only care that the proportion is in the same ballpark.
                # Allow 0.5x–1.5x of the original proportion.
                ratio = p_split / p_full
                assert 0.5 <= ratio <= 1.5, (
                    f"Class proportion changed too much: full={p_full:.4f}, "
                    f"split={p_split:.4f}, ratio={ratio:.2f}"
                )


class TestTransformShapesValidation:
    """Test that transforms produce consistent tensor shapes"""

    def test_train_transform_shapes(self):
        img_size = 256
        tfms_train = get_transforms(
            image_size=img_size, train=True, normalize=True, augment=True
        )
        train_transform = tfms_train["color"]

        from PIL import Image

        mock_image = Image.new("RGB", (512, 512), color="red")
        transformed = train_transform(mock_image)

        assert isinstance(transformed, torch.Tensor)
        assert transformed.shape == (3, img_size, img_size)
        assert transformed.dtype == torch.float32
        # Rough range check after normalization
        assert transformed.min() >= -5.0 and transformed.max() <= 5.0

    def test_eval_transform_shapes(self):
        img_size = 256
        tfms_eval = get_transforms(
            image_size=img_size, train=False, normalize=True, augment=False
        )
        eval_transform = tfms_eval["color"]

        from PIL import Image

        mock_image = Image.new("RGB", (300, 400), color="blue")
        transformed = eval_transform(mock_image)

        assert isinstance(transformed, torch.Tensor)
        assert transformed.shape == (3, img_size, img_size)
        assert transformed.dtype == torch.float32

    def test_transform_without_normalization(self):
        img_size = 128
        tfms_eval = get_transforms(
            image_size=img_size, train=False, normalize=False, augment=False
        )
        eval_transform = tfms_eval["color"]

        from PIL import Image

        mock_image = Image.new("RGB", (200, 200), color="green")
        transformed = eval_transform(mock_image)

        assert transformed.shape == (3, img_size, img_size)
        assert transformed.min() >= 0.0 and transformed.max() <= 1.0

    def test_transform_consistency(self):
        img_size = 256
        tfms_eval = get_transforms(
            image_size=img_size, train=False, normalize=True, augment=False
        )
        eval_transform = tfms_eval["color"]

        from PIL import Image

        mock_image = Image.new("RGB", (256, 256), color="yellow")

        transformed1 = eval_transform(mock_image)
        transformed2 = eval_transform(mock_image)

        # With augment=False, it should be deterministic
        assert torch.allclose(transformed1, transformed2)
