import pytest
import torch

from src.utils import EarlyStopping, unpack_batch


def test_unpack_batch_tuple():
    """We check if the DataLoader returns a batch in the usual format (imgs,labels)"""
    xbatch_inp = torch.randn(4, 3, 32, 32)
    ybatch_inp = torch.randint(0, 5, (4,))
    test_batch = (xbatch_inp, ybatch_inp)
    device = torch.device("cpu")

    x_batch, y_batch = unpack_batch(test_batch, device)

    assert torch.allclose(x_batch, xbatch_inp)
    assert torch.allclose(y_batch, ybatch_inp)
    assert x_batch.device == device
    assert y_batch.device == device


def test_unpack_batch_invalid_type_raises():
    """Check what happens if we accidentally pass in the wrong type into unpack_batch()"""
    device = torch.device("cpu")
    with pytest.raises(TypeError):
        unpack_batch("not a batch", device)


def test_early_stopping_behaviour():
    """Simulates training across epochs to check early stopping functionality"""
    early_stop = EarlyStopping(patience=2, min_delta=0.0)

    assert early_stop.step(0.5, 1) is False
    assert early_stop.step(0.6, 2) is False
    assert early_stop.step(0.7, 3) is False

    assert early_stop.step(0.7, 4) is False

    assert early_stop.step(0.69, 5) is True

    assert early_stop.best_epoch == 3
    assert abs(early_stop.best_score - 0.7) < 1e-8
