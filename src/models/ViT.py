import torch
import torch.nn as nn
from torchvision.models import ViT_B_16_Weights, vit_b_16


class ViT_b_16(nn.Module):
    def __init__(self, num_classes=39, dropout=0.3, device="cpu"):
        super().__init__()

        # 1️⃣ Load pretrained ViT
        weights = ViT_B_16_Weights.IMAGENET1K_V1
        model = vit_b_16(weights=weights)

        # 2️⃣ Freeze all backbone layers
        for param in model.parameters():
            param.requires_grad = False

        # 3️⃣ Replace classification head
        in_features = model.heads.head.in_features
        model.heads = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes),
        )

        self.model = model
        self.model.to(device)

    def forward(self, x):
        return self.model(x)
