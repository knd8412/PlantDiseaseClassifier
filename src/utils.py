import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch


# ----------------------------------------------------------------------
# Config dataclass
# ----------------------------------------------------------------------
@dataclass
class Config:
    seed: int
    project_name: str
    task_name: str
    tags: List[str]
    device: str
    data: Dict[str, Any]
    model: Dict[str, Any]
    train: Dict[str, Any]
    clearml: Dict[str, Any]
    dataset: Optional[Dict[str, Any]] = None


# ----------------------------------------------------------------------
# Early stopping
# ----------------------------------------------------------------------
class EarlyStopping:
    def __init__(self, patience: int = 3, min_delta: float = 0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.best_score = None
        self.best_epoch = None
        self.counter = 0

    def step(self, metric: float, epoch: int) -> bool:
        """Returns True if we should stop training."""
        if self.best_score is None or metric > self.best_score + self.min_delta:
            self.best_score = metric
            self.best_epoch = epoch
            self.counter = 0
            return False
        else:
            self.counter += 1
            return self.counter >= self.patience


# ----------------------------------------------------------------------
# Utils
# ----------------------------------------------------------------------
def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def accuracy(logits, targets):
    preds = torch.argmax(logits, dim=1)
    return (preds == targets).float().mean().item()


def unpack_batch(batch, device):
    """
    Unpack a batch coming from the DataLoader.
    """
    if isinstance(batch, (list, tuple)):
        if len(batch) < 2:
            raise ValueError(f"Expected at least 2 elements in batch, got {len(batch)}")
        xb, yb = batch[0], batch[1]
    else:
        raise TypeError(f"Expected batch to be tuple/list, got {type(batch)}")

    return xb.to(device), yb.to(device)
