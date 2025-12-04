from pathlib import Path

from data.utils import (
    balance_dataset_uniform,
    build_class_mapping,
    calculate_class_weights,
    gather_samples,
    get_sample_weights,
    make_subset,
    split_dataset,
)
from PIL import Image


def make_classdir_images(root, modality, class_name, n=3):
    class_dir = root / modality / class_name
    class_dir.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        img = Image.new("RGB", (16, 16), color=(i * 10, 0, 0))
        img.save(class_dir / f"img_{i}.jpg", format="JPEG")


def test_build_class_mapping_and_gather_samples(tmp_path):
    make_classdir_images(tmp_path, "color", "class_a", n=2)
    make_classdir_images(tmp_path, "color", "class_b", n=3)

    class_names, class_id = build_class_mapping(tmp_path, modality="color")
    assert sorted(class_names) == ["class_a", "class_b"]
    assert class_id["class_a"] != class_id["class_b"]

    samples = gather_samples(tmp_path, ["color"], class_id)

    assert len(samples) == 5
    path, label, modality = samples[0]
    assert modality == "color"
    assert isinstance(path, str)
    assert label in class_id.values()


def test_split_dataset_stratified(tmp_path):
    make_classdir_images(tmp_path, "color", "class_a", n=10)
    make_classdir_images(tmp_path, "color", "class_b", n=10)

    class_names, class_id = build_class_mapping(tmp_path, modality="color")
    samples = gather_samples(tmp_path, ["color"], class_id)

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

    assert (
        classes_of(train)
        == classes_of(val)
        == classes_of(test)
        == {
            0,
            1,
        }
    )


def test_split_dataset_stratified_imbalance(tmp_path):
    make_classdir_images(tmp_path, "color", "class_a", n=20)
    make_classdir_images(tmp_path, "color", "class_b", n=5)

    class_names, class_id = build_class_mapping(tmp_path, modality="color")
    samples = gather_samples(tmp_path, ["color"], class_id)

    train, val, test = split_dataset(
        samples, test_size=0.2, val_size=0.2, random_state=0
    )

    def count_class(split, label):
        return sum(1 for _, lbl, _ in split if lbl == label)

    total_a = count_class(samples, class_id["class_a"])
    total_b = count_class(samples, class_id["class_b"])

    train_a = count_class(train, class_id["class_a"])
    train_b = count_class(train, class_id["class_b"])

    ratio_full = total_a / total_b
    ratio_train = train_a / train_b

    assert abs(ratio_full - ratio_train) < 0.5


def test_gather_samples_multiple_modalities(tmp_path):
    make_classdir_images(tmp_path, "color", "cls", n=2)
    make_classdir_images(tmp_path, "grayscale", "cls", n=3)

    x, class_id = build_class_mapping(tmp_path, modality="color")

    samples = gather_samples(tmp_path, ["color", "grayscale"], class_id)

    modalities = {m for a, b, m in samples}
    assert modalities == {"color", "grayscale"}

    assert len(samples) == 2 + 3


def test_make_subset_stratified():
    samples = [(f"/img_{i}.jpg", 0 if i < 80 else 1, "color") for i in range(100)]
    subset = make_subset(samples, ratio=0.25, seed=42)
    assert len(subset) == 25
    labels = {s[1] for s in subset}
    assert labels == {0, 1}


def test_balance_dataset_uniform():
    samples = [(f"/img_{i}.jpg", 0, "color") for i in range(50)]
    samples += [(f"/img_{50+i}.jpg", 1, "color") for i in range(10)]
    balanced = balance_dataset_uniform(samples, seed=0)
    labels = [s[1] for s in balanced]
    assert labels.count(0) == labels.count(1)


def test_class_and_sample_weights_length():
    samples = [(f"/img_{i}.jpg", i % 3, "color") for i in range(30)]
    class_weights = calculate_class_weights(samples, num_classes=3)
    sample_weights = get_sample_weights(samples)
    assert class_weights.shape[0] == 3
    assert sample_weights.shape[0] == len(samples)
