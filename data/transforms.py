"""
Image transforms for multi-modality plant disease dataset.
Includes augmentation and normalization for color, grayscale, and segmented images.
"""

from torchvision import transforms

# ImageNet normalization constants
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Generic normalization for grayscale/segmented
GENERIC_MEAN = [0.5, 0.5, 0.5]
GENERIC_STD = [0.5, 0.5, 0.5]


def get_transforms(image_size=224, train=True, normalize=True, augment=True):
    """
    Get modality-specific transforms for training or validation/test.

    Args:
        image_size: target image size (default: 224 for pretrained models)
        train: if True, enables augmentation (if augment=True); if False, only resizing and normalization
        normalize: if True, apply normalization; if False, only convert to tensor
        augment: if True and train=True, apply data augmentation

    Returns:
        dict: {modality_name: transform} for color, grayscale, and segmented images
    """

    if train and augment:
        color_transform = [
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
        ]

        grayscale_transform = [
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
        ]

        segmented_transform = [
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]
    else:
        # No augmentation for validation/test or when augment=False
        color_transform = [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ]

        grayscale_transform = [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ]

        segmented_transform = [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ]

    # Add normalization if requested
    if normalize:
        color_transform.append(
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        )
        grayscale_transform.append(
            transforms.Normalize(mean=GENERIC_MEAN, std=GENERIC_STD)
        )
        segmented_transform.append(
            transforms.Normalize(mean=GENERIC_MEAN, std=GENERIC_STD)
        )

    return {
        "color": transforms.Compose(color_transform),
        "grayscale": transforms.Compose(grayscale_transform),
        "segmented": transforms.Compose(segmented_transform),
    }
