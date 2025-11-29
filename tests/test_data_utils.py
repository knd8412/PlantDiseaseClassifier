from pathlib import Path

from data.utils import build_class_mapping, gather_samples, split_dataset
from PIL import Image


def _make_class_dir_with_images(root: Path, modality: str, class_name: str, n=3):
    cls_dir = root / modality / class_name
    cls_dir.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        img = Image.new("RGB", (16, 16), color=(i * 10, 0, 0))
        img.save(cls_dir / f"img_{i}.jpg", format="JPEG")


def test_build_class_mapping_and_gather_samples(tmp_path):
    _make_class_dir_with_images(tmp_path, "color", "class_a", n=2)
    _make_class_dir_with_images(tmp_path, "color", "class_b", n=3)

    class_names, class_to_idx = build_class_mapping(tmp_path, modality="color")
    assert sorted(class_names) == ["class_a", "class_b"]
    assert class_to_idx["class_a"] != class_to_idx["class_b"]

    samples = gather_samples(tmp_path, ["color"], class_to_idx)
    
    assert len(samples) == 5
    path, label, modality = samples[0]
    assert modality == "color"
    assert isinstance(path, str)
    assert label in class_to_idx.values()


def test_split_dataset_stratified_like(tmp_path):
    _make_class_dir_with_images(tmp_path, "color", "class_a", n=10)
    _make_class_dir_with_images(tmp_path, "color", "class_b", n=10)

    class_names, class_to_idx = build_class_mapping(tmp_path, modality="color")
    samples = gather_samples(tmp_path, ["color"], class_to_idx)

    train, val, test = split_dataset(
        samples, test_size=0.2, val_size=0.2, random_state=42
    )

    total = len(samples)
    assert len(train) + len(val) + len(test) == total
    assert 0 < len(train) < total
    assert 0 < len(val) < total
    assert 0 < len(test) < total

    def classes_of(split):
        return set(label for _, label, _ in split)

    assert classes_of(train) == classes_of(val) == classes_of(test) == {
        0,
        1,
    }
