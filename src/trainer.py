import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

from src.clearml_utils import log_scalar, upload_model
from src.utils import EarlyStopping, unpack_batch


class Trainer:
    def __init__(self, model, config, task=None):
        self.model = model
        self.cfg = config
        self.task = task

        # Move model to GPU if available
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() and config.device == "cuda" else "cpu"
        )
        self.model.to(self.device)
        print(f"Using device: {self.device}")

        # Loss and optimizer
        self.criterion = nn.CrossEntropyLoss()
        optimizer_name = self.cfg.train["optimizer"].lower()
        if optimizer_name == "adamw":
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

        # Learning rate scheduler
        self.scheduler = None
        scheduler_cfg = self.cfg.train.get("scheduler")
        if scheduler_cfg and scheduler_cfg.get("type", "").lower() == "cosine":
            T_max = scheduler_cfg.get("T_max", self.cfg.train["epochs"])
            self.scheduler = CosineAnnealingLR(self.optimizer, T_max=T_max)
            print(f"Using CosineAnnealingLR scheduler (T_max={T_max})")

        # Early stopping
        self.early_stopper = EarlyStopping(
            patience=self.cfg.train.get("patience", 3),
            min_delta=self.cfg.train.get("min_delta", 0.0),
        )

    def train_one_epoch(self, loader):
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for batch in loader:
            xb, yb = unpack_batch(batch, self.device)

            self.optimizer.zero_grad()
            logits = self.model(xb)
            loss = self.criterion(logits, yb)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item() * xb.size(0)
            correct += (logits.argmax(1) == yb).sum().item()
            total += xb.size(0)

        return total_loss / total, correct / total

    @torch.no_grad()
    def evaluate(self, loader):
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        for batch in loader:
            xb, yb = unpack_batch(batch, self.device)
            logits = self.model(xb)
            loss = self.criterion(logits, yb)

            total_loss += loss.item() * xb.size(0)
            correct += (logits.argmax(1) == yb).sum().item()
            total += xb.size(0)

        return total_loss / total, correct / total

    def train(self, train_loader, val_loader, test_loader=None):
        print(f"Starting training for {self.cfg.train['epochs']} epochs...")
        os.makedirs("outputs", exist_ok=True)

        best_val_acc = 0.0
        best_epoch = 0
        best_path = "outputs/best.pt"

        history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

        for epoch in range(1, self.cfg.train["epochs"] + 1):
            train_loss, train_acc = self.train_one_epoch(train_loader)
            val_loss, val_acc = self.evaluate(val_loader)

            history["train_loss"].append(train_loss)
            history["train_acc"].append(train_acc)
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)

            print(
                f"Epoch {epoch:02d} | "
                f"train loss: {train_loss:.4f}, acc: {train_acc:.4f} | "
                f"val loss: {val_loss:.4f}, acc: {val_acc:.4f}"
            )

            # Log to ClearML
            log_scalar(self.task, "loss", "train", train_loss, epoch)
            log_scalar(self.task, "loss", "val", val_loss, epoch)
            log_scalar(self.task, "accuracy", "train", train_acc, epoch)
            log_scalar(self.task, "accuracy", "val", val_acc, epoch)

            # Step scheduler
            if self.scheduler:
                self.scheduler.step()
                lr = self.scheduler.get_last_lr()[0]
                log_scalar(self.task, "learning_rate", "lr", lr, epoch)

            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_epoch = epoch
                torch.save(
                    {"model_state": self.model.state_dict(), "val_acc": best_val_acc},
                    best_path,
                )
                upload_model(self.task, best_path, name="best.pt")

            # Check early stopping
            if self.early_stopper.step(val_acc, epoch):
                print(f"Early stopping triggered after {epoch} epochs")
                print(f"Best was epoch {self.early_stopper.best_epoch}")
                break

        # Save training history
        metrics = {
            "best_val_acc": best_val_acc,
            "best_epoch": best_epoch,
            "best_model_path": best_path,
            **history,
        }

        with open("outputs/metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)

        if self.task:
            try:
                self.task.connect(metrics)
            except Exception:
                pass

        print("Training complete!")
        print(f"Best validation accuracy: {best_val_acc:.4f} at epoch {best_epoch}")
        print(f"Model saved to: {best_path}")

        return metrics
