import os
from pathlib import Path

import torch
from PIL import Image

from data.dataset import MultiModalityDataset
from data.transforms import get_transforms


def make_dummy_jpg(path, size=(32, 32)):
    path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGB", size, color=(123, 222, 112))
    img.save(path, format="JPEG")


def test_get_transforms_shapes():
    img_size = 224
    t_train = get_transforms(
        image_size=img_size, train=True, normalize=True, augment=True
    )
    t_eval = get_transforms(
        image_size=img_size, train=False, normalize=True, augment=False
    )

    assert "color" in t_train and "color" in t_eval

    img = Image.new("RGB", (256, 256))
    x_train = t_train["color"](img)
    x_eval = t_eval["color"](img)

    assert isinstance(x_train, torch.Tensor)
    assert isinstance(x_eval, torch.Tensor)
    assert x_train.shape == (3, img_size, img_size)
    assert x_eval.shape == (3, img_size, img_size)


def test_multimodality_dataset_item(tmp_path):
    color_path = tmp_path / "color" / "class_a" / "img1.jpg"
    gray_path = tmp_path / "grayscale" / "class_a" / "img2.jpg"
    seg_path = tmp_path / "segmented" / "class_a" / "img3.jpg"

    for p in (color_path, gray_path, seg_path):
        make_dummy_jpg(p)

    samples = [
        (str(color_path), 0, "color"),
        (str(gray_path), 0, "grayscale"),
        (str(seg_path), 0, "segmented"),
    ]

    transforms = get_transforms(
        image_size=128, train=True, normalize=True, augment=False
    )
    dataset = MultiModalityDataset(samples, transforms)

    assert len(dataset) == 3

    img, label = dataset[0]
    assert isinstance(img, torch.Tensor)
    assert img.shape[0] == 3  # 3 channels
    assert label.item() == 0
