import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import matplotlib.pyplot as plt
import numpy as np

from .utils import accuracy, unpack_batch, EarlyStopping
from .clearml_utils import log_scalar, upload_model, log_figure

class Trainer:
    def __init__(self, model, config, task=None):
        self.model = model
        self.cfg = config
        self.task = task
        self.device = torch.device(
            config.device if torch.cuda.is_available() and config.device == "cuda" else "cpu"
        )
        self.model.to(self.device)
        
        # Setup Loss & Optimizer
        self.criterion = nn.CrossEntropyLoss()
        if self.cfg.train["optimizer"].lower() == "adamw":
            self.optimizer = optim.AdamW(
                self.model.parameters(),
                lr=self.cfg.train["lr"],
                weight_decay=self.cfg.train["weight_decay"],
            )
        else:
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.cfg.train["lr"],
                weight_decay=self.cfg.train["weight_decay"],
            )

        # Scheduler
        self.scheduler = None
        scheduler_cfg = self.cfg.train.get("scheduler")
        if scheduler_cfg is not None and scheduler_cfg.get("type", "").lower() == "cosine":
            T_max = scheduler_cfg.get("T_max", self.cfg.train["epochs"])
            self.scheduler = CosineAnnealingLR(self.optimizer, T_max=T_max)
            print(f"[Scheduler] Using CosineAnnealingLR with T_max={T_max}")

        # Early Stopping
        self.early_stopper = EarlyStopping(
            patience=self.cfg.train.get("patience", 3),
            min_delta=self.cfg.train.get("min_delta", 0.0),
        )

    def train_one_epoch(self, loader):
        self.model.train()
        running_loss, running_acc, n = 0.0, 0.0, 0
        for batch in loader:
            xb, yb = unpack_batch(batch, self.device)
            self.optimizer.zero_grad()
            logits = self.model(xb)
            loss = self.criterion(logits, yb)
            loss.backward()
            self.optimizer.step()
            running_loss += loss.item() * xb.size(0)
            running_acc += (logits.argmax(1) == yb).float().sum().item()
            n += xb.size(0)
        return running_loss / n, running_acc / n

    @torch.no_grad()
    def evaluate(self, loader):
        self.model.eval()
        running_loss, running_acc, n = 0.0, 0.0, 0
        for batch in loader:
            xb, yb = unpack_batch(batch, self.device)
            logits = self.model(xb)
            loss = self.criterion(logits, yb)
            running_loss += loss.item() * xb.size(0)
            running_acc += (logits.argmax(1) == yb).float().sum().item()
            n += xb.size(0)
        return running_loss / n, running_acc / n

    def train(self, train_loader, val_loader, test_loader=None):
        print(f"[Trainer] Starting training on {self.device}...")
        os.makedirs("outputs", exist_ok=True)
        
        best_val_acc = 0.0
        best_epoch = 0
        best_path = os.path.join("outputs", "best.pt")
        history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

        for epoch in range(1, self.cfg.train["epochs"] + 1):
            tr_loss, tr_acc = self.train_one_epoch(train_loader)
            va_loss, va_acc = self.evaluate(val_loader)

            history["train_loss"].append(tr_loss)
            history["train_acc"].append(tr_acc)
            history["val_loss"].append(va_loss)
            history["val_acc"].append(va_acc)

            print(
                f"Epoch {epoch:02d} | "
                f"Train: loss {tr_loss:.4f}, acc {tr_acc:.4f} | "
                f"Val: loss {va_loss:.4f}, acc {va_acc:.4f}"
            )
            
            # Logging
            log_scalar(self.task, "loss", "train", tr_loss, epoch)
            log_scalar(self.task, "loss", "val", va_loss, epoch)
            log_scalar(self.task, "accuracy", "train", tr_acc, epoch)
            log_scalar(self.task, "accuracy", "val", va_acc, epoch)

            if self.scheduler is not None:
                self.scheduler.step()
                current_lr = self.scheduler.get_last_lr()[0]
                log_scalar(self.task, "learning_rate", "lr", current_lr, epoch)

            # Save best model
            if va_acc > best_val_acc + 1e-8:
                best_val_acc = va_acc
                best_epoch = epoch
                torch.save(
                    {"model_state": self.model.state_dict(), "val_acc": best_val_acc},
                    best_path,
                )
                upload_model(self.task, best_path, name="best.pt")

            # Early stopping
            if self.early_stopper.step(va_acc, epoch):
                print(
                    f"[EarlyStopping] No val acc improvement for "
                    f"{self.early_stopper.patience} epochs. "
                    f"Best epoch: {self.early_stopper.best_epoch}"
                )
                break

        # Save metrics
        metrics = {
            "best_val_acc": best_val_acc,
            "best_epoch": best_epoch,
            "best_model_path": best_path,
            **history,
        }
        with open(os.path.join("outputs", "metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2)

        if self.task is not None:
            try:
                self.task.connect(metrics)
            except Exception:
                pass

        print(
            f"[Done] Best validation accuracy: {best_val_acc:.4f} "
            f"(epoch {best_epoch}). Weights saved to {best_path}"
        )
        
        return metrics
