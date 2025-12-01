from src.data.labels import class_names
from ui.disease_info import disease_info


def test_repo_is_working():
    assert 1 + 1 == 2


def test_disease_info_covers_all_classes():
    c_names = set(class_names)
    info_keys = set(disease_info.keys())

    # All class names have a description
    missing = [c for c in c_names if c not in info_keys]
    assert missing == [], f"Missing descriptions for: {missing}"

    # No extra disease_info keys that aren't real classes
    extras = [k for k in info_keys if k not in c_names]
    assert extras == [], f"Extra disease_info entries not in class_names: {extras}"
