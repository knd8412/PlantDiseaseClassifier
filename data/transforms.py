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


def get_transforms(image_size=224, train=True):
    """
    Get modality-specific transforms for training or validation/test.
    
    Args:
        image_size: target image size (default: 224 for pretrained models)
        train: if True, includes data augmentation; if False, only resizing and normalization
        
    Returns:
        dict: {modality_name: transform} for color, grayscale, and segmented images
    """
    
    if train:
        return {
            "color": transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(10),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]),
            "grayscale": transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(10),
                transforms.ToTensor(),
                transforms.Normalize(mean=GENERIC_MEAN, std=GENERIC_STD)
            ]),
            "segmented": transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=GENERIC_MEAN, std=GENERIC_STD)
            ]),
        }
    else:
        return {
            "color": transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]),
            "grayscale": transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=GENERIC_MEAN, std=GENERIC_STD)
            ]),
            "segmented": transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=GENERIC_MEAN, std=GENERIC_STD)
            ]),
        }
