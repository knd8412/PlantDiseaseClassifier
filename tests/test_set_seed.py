import torch

from src.utils import set_seed


def test_set_seed_makes_torch_randn_deterministic():
    """Checks the set_seed function is actually deterministic"""
    set_seed(123)
    a = torch.randn(3, 3)
    set_seed(123)
    b = torch.randn(3, 3)
    assert torch.allclose(a, b)
