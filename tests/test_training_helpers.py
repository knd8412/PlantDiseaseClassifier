import pytest
import torch

from src.train import EarlyStopping, _unpack_batch


def test_unpack_batch_dict():
    batch = {
        "image": torch.randn(3, 3, 32, 32),
        "label": torch.randint(0, 5, (3,)),
        "modality": "color",
    }
    device = torch.device("cpu")
    xb, yb = _unpack_batch(batch, device)
    assert xb.shape[0] == 3
    assert yb.shape[0] == 3
    assert xb.device == device
    assert yb.device == device


def test_unpack_batch_tuple():
    xb_in = torch.randn(4, 3, 32, 32)
    yb_in = torch.randint(0, 5, (4,))
    batch = (xb_in, yb_in)
    device = torch.device("cpu")
    xb, yb = _unpack_batch(batch, device)
    assert torch.allclose(xb, xb_in)
    assert torch.allclose(yb, yb_in)
    assert xb.device == device
    assert yb.device == device


def test_unpack_batch_invalid_type_raises():
    device = torch.device("cpu")
    with pytest.raises(ValueError):
        _unpack_batch("not a batch", device)


def test_early_stopping_behaviour():
    es = EarlyStopping(patience=2, min_delta=0.0)

    assert es.step(0.5, 1) is False
    assert es.step(0.6, 2) is False
    assert es.step(0.7, 3) is False

    assert es.step(0.7, 4) is False
    assert es.step(0.69, 5) is True

    assert es.best_epoch == 3
    assert abs(es.best_score - 0.7) < 1e-8
