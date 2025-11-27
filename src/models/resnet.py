import os
import time
from tempfile import TemporaryDirectory

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import torchvision
from PIL import Image
from torch.optim import lr_scheduler
from torchvision import datasets, models, transforms
from torchvision.models import ResNet18_Weights, resnet18


class ResNet18Classifier(nn.Module):
    def __init__(
        self, num_classes=39, pretrained=True, dropout=0.0, train_backbone=True
    ):
        super().__init__()

        if pretrained:
            weights = ResNet18_Weights.IMAGENET1K_V1
            backbone = resnet18(weights=weights)
            self.normalise_mean = weights.meta["mean"]
            self.normalise_std = weights.meta["std"]

        else:
            model = resnet18(weights=None)
            self.normalise_mean = [0.485, 0.456, 0.406]
            self.normalise_std = [0.299, 0.224, 0.225]

        inp_features = model.fc.in_features

        if dropout > 0:
            model.fc = nn.Sequential(
                nn.Dropout(p=dropout), nn.Linear(inp_features, num_classes)
            )

        else:
            model.fc = nn.Linear(inp_features, num_classes)

        if not train_backbone:
            for param in model.parameters():
                param.requires_grad = False
            for param in model.fc.parameters():
                param.requires_grad = True

        self.model = backbone

        def forward(self, x):
            return self.model(x)
