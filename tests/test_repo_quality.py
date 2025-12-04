from data.labels import class_names

from ui.disease_info import disease_info


def test_repo_is_working():
    assert 1 + 1 == 2


def test_disease_info_covers_all_classes():
    """Checks whether class names and disease_info keys match"""
    c_names = set(class_names)
    disease_keys = set(disease_info.keys())

    missing = [c for c in c_names if c not in disease_keys]
    assert missing == [], f"Missing descriptions for: {missing}"

    extras = [name for name in disease_keys if name not in c_names]
    assert extras == [], f"Extra disease_info entries not in class_names: {extras}"
